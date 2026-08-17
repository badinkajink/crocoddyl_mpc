#!/usr/bin/env python3
"""Stability metrics for a braced pose, all over the SAME feasible set the
contact-selection QP already uses.

Three metrics.  They were picked because they are LPs over machinery
contact_select.py already builds, they plot in 2-D, and they are mutually
consistent -- the same friction cones, the same torque basis, the same contact
Jacobians.  Anything requiring a dynamic rollout is deliberately excluded.

1. SLIP MARGIN (per contact).  u = ||lam_t|| / (mu lam_n).  u = 0 is a pure
   normal push, u = 1 sits exactly on the friction cone and is about to slide.
   This is the metric for the "the bracing arm slides along the table" problem:
   a pose whose brace contact runs at u ~ 1 is holding station by friction it
   barely has.

2. STATIC-EQUILIBRIUM REGION.  The set of horizontal CoM positions for which
   SOME admissible contact-force distribution exists.  This is the multi-contact
   generalisation of the support polygon (Bretl & Lall 2008; Caron et al. 2015),
   and it is the honest version of "the brace extends the support area" -- with
   a brace the region grows forward, and how much it grows IS the benefit.

   Equilibrium about the world origin, for CoM c and total mass M:
       sum_i lam_i          = (0, 0, Mg)
       sum_i p_i x lam_i    = (-Mg c_y, +Mg c_x, 0)
   The moment is independent of c_z, so the region is genuinely a 2-D set and
   needs no height assumption.

   Computed in two versions, and the gap between them is the point:
     contact-only  6 equilibrium rows + friction cones.  Exact.  What the
                   contact SET can do, ignoring whether the robot can hold it.
     actuated      the same, plus |tau| <= tau_max with tau = g(q) - J' lam.
                   APPROXIMATE: g(q) is frozen at the solved pose while the CoM
                   is swept, so this is a linearisation about that pose, not a
                   re-solved posture.  Reported separately for that reason.

3. ROBUSTNESS MARGIN.  Distance from the actual CoM to the boundary of the
   region, in metres.  One scalar per pose, comparable across contact sets,
   and the natural thing to rank poses by.

Also reported: MAX RESISTIBLE PUSH per direction -- the largest horizontal force
applicable at the CoM that the contact set can still balance.  Same LP, different
objective, and it answers "how hard can something shove this pose" in newtons.

usage:  stability.py [x y z] [site,site,...]
        TAU_BASIS=model|urdf|clamp   LEAN_MODEL=handless|magpie
"""
import itertools
import json
import os
import sys

import numpy as np
import mujoco
from scipy.optimize import linprog

import contact_select as cs

G = 9.81
NDIR = 36           # rays for the region boundary; 10 deg resolution
RAY_MAX = 1.5       # [m] longest CoM excursion ever searched
BISECT = 18         # bisection steps per ray -> ~6 um resolution


# --------------------------------------------------------------------------- #
# shared assembly
# --------------------------------------------------------------------------- #
def blocks(m, d, subset):
    """Contact points, their world positions, the stacked Jacobian transpose and
    the gravity bias -- everything both the QP and these LPs need."""
    d.qvel[:] = 0
    d.qacc[:] = 0
    mujoco.mj_forward(m, d)
    # Same resolved contacts and same contact-frame force variables as the QP --
    # if these two disagree the metrics stop describing the pose the QP scored.
    contacts = cs.resolved_contacts(m, d, subset)
    P = np.array([p for _, p, _ in contacts])
    Rs = [R for _, _, R in contacts]
    J = []
    for b, p, R in contacts:
        jacp = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jacp, None, p, b)
        J.append(R @ jacp[:, :cs.N_ROBOT_DOF])
    A = np.hstack([Ji.T for Ji in J])              # (33, 3nc)
    g = d.qfrc_bias[:cs.N_ROBOT_DOF].copy()
    # ROBOT subtree only.  mj_getTotalmass / subtree_com[0] include the table and
    # the object -- 153.7 kg against the robot's 67.7 -- which silently doubles
    # the weight the region has to balance and puts the CoM inside the table.
    root = cs.bid(m, "pelvis")
    mass = float(m.body_subtreemass[root])
    com = np.array(d.subtree_com[root])
    return contacts, P, A, g, mass, com, Rs


def friction_rows(nc, mu=None):
    """Linearised Coulomb cone as a 4-facet pyramid, per contact.

    Two rows per tangential axis:  +lam_t - mu lam_n / sqrt2 <= 0
                                   -lam_t - mu lam_n / sqrt2 <= 0
    Negating the whole row instead (a bug that cost a session, plan S10c.1) pins
    |lam_t| TO the boundary rather than bounding it, and every LP comes back
    infeasible."""
    mu = cs.MU if mu is None else mu
    rows = []
    k = mu / np.sqrt(2.0)
    for i in range(nc):
        for ax in (1, 2):                       # tangents; component 0 is normal
            for sgn in (+1.0, -1.0):
                r = np.zeros(3 * nc)
                r[3 * i + ax] = sgn
                r[3 * i] = -k
                rows.append(r)
    return np.array(rows), np.zeros(len(rows))


def wrench_rows(P, Rs):
    """(6, 3nc) map from stacked CONTACT-FRAME forces to the total wrench about
    the world origin.  lam_i is (n, t1, t2) in contact frame i, so the world
    force is R_i' lam_i and both blocks carry that rotation."""
    nc = len(P)
    W = np.zeros((6, 3 * nc))
    for i, (p, R) in enumerate(zip(P, Rs)):
        Rw = R.T                                   # contact frame -> world
        W[0:3, 3 * i:3 * i + 3] = Rw
        skew = np.array([[0, -p[2], p[1]],
                         [p[2], 0, -p[0]],
                         [-p[1], p[0], 0]])
        W[3:6, 3 * i:3 * i + 3] = skew @ Rw
    return W


def feasible_at(W, Fr, br, Aj, gj, tau_max, mass, cx, cy, actuated, fext=None):
    """Is static equilibrium possible with the CoM at (cx, cy)?

    fext, if given, is an extra horizontal force applied AT the CoM -- the
    contacts must then balance weight + fext and its moment."""
    # Newton-Euler about the world origin, all external wrenches summing to zero:
    #   sum lam + (0,0,-Mg) + f          = 0
    #   sum p_i x lam_i + c x (0,0,-Mg) + c x f = 0
    # with c x (0,0,-Mg) = (-Mg c_y, +Mg c_x, 0), so the contacts must SUPPLY the
    # negative of that.  (Getting this sign backwards makes every LP infeasible,
    # which is indistinguishable from "the pose is unstable" -- so it is checked
    # against the QP solution in self_test().)
    Mg = mass * G
    wrench = np.array([0.0, 0.0, Mg, Mg * cy, -Mg * cx, 0.0])
    if fext is not None:
        fx, fy, cz = fext
        wrench[0] -= fx
        wrench[1] -= fy
        # -(c x f) with f = (fx, fy, 0):  c x f = (-cz fy, cz fx, cx fy - cy fx)
        wrench[3] += cz * fy
        wrench[4] -= cz * fx
        wrench[5] -= (cx * fy - cy * fx)
    nv = W.shape[1]
    Aub, bub = [Fr], [br]
    if actuated:
        # tau = gj - Aj lam,  |tau| <= tau_max
        Aub += [-Aj, Aj]
        bub += [tau_max - gj, tau_max + gj]
    res = linprog(np.zeros(nv), A_ub=np.vstack(Aub), b_ub=np.concatenate(bub),
                  A_eq=W, b_eq=wrench,
                  bounds=[(0, None) if i % 3 == 0 else (None, None)
                          for i in range(nv)],
                  method="highs")
    return bool(res.status == 0)


def equilibrium_region(m, d, subset, actuated=True, ndir=NDIR):
    """Ray-shoot the boundary of the static-equilibrium CoM region.

    Returns (boundary points (ndir,2), the pose's own CoM (2,), margin [m]).
    margin < 0 means the pose is NOT in static equilibrium at all."""
    contacts, P, A, g, mass, com, Rs = blocks(m, d, subset)
    W = wrench_rows(P, Rs)
    Fr, br = friction_rows(len(P))
    Aj, gj = A[6:], g[6:]
    tau_max = cs.torque_limits(m)

    def ok(cx, cy):
        return feasible_at(W, Fr, br, Aj, gj, tau_max, mass, cx, cy, actuated)

    c0 = com[:2].copy()
    inside0 = ok(c0[0], c0[1])
    # If the pose's own CoM is outside, the region may still be non-empty; seed
    # the ray origin from the centroid of the contact points instead, which is
    # inside by construction for any set that can hold ANY weight.
    origin = c0 if inside0 else P[:, :2].mean(axis=0)
    if not ok(origin[0], origin[1]):
        return np.zeros((0, 2)), c0, float("nan")

    pts = []
    for th in np.linspace(0, 2 * np.pi, ndir, endpoint=False):
        u = np.array([np.cos(th), np.sin(th)])
        lo, hi = 0.0, RAY_MAX
        if ok(*(origin + hi * u)):
            pts.append(origin + hi * u)
            continue
        for _ in range(BISECT):
            mid = 0.5 * (lo + hi)
            if ok(*(origin + mid * u)):
                lo = mid
            else:
                hi = mid
        pts.append(origin + lo * u)
    pts = np.array(pts)

    # signed distance from the pose's CoM to the boundary polygon
    margin = _poly_margin(pts, c0)
    if not inside0:
        margin = -abs(margin)
    return pts, c0, float(margin)


def _poly_margin(poly, p):
    """Distance from p to the closest edge of a closed polygon."""
    if len(poly) < 3:
        return float("nan")
    best = np.inf
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
        best = min(best, np.linalg.norm(p - (a + t * ab)))
    return best


def max_push(m, d, subset, ndir=16, actuated=True):
    """Largest horizontal force at the CoM the contact set can still balance,
    per direction.  Returns (angles, newtons)."""
    contacts, P, A, g, mass, com, Rs = blocks(m, d, subset)
    W = wrench_rows(P, Rs)
    Fr, br = friction_rows(len(P))
    Aj, gj = A[6:], g[6:]
    tau_max = cs.torque_limits(m)
    cx, cy, cz = com
    out = []
    angs = np.linspace(0, 2 * np.pi, ndir, endpoint=False)
    for th in angs:
        u = np.array([np.cos(th), np.sin(th)])
        lo, hi = 0.0, 600.0
        if feasible_at(W, Fr, br, Aj, gj, tau_max, mass, cx, cy, actuated,
                       fext=(hi * u[0], hi * u[1], cz)):
            out.append(hi)
            continue
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            if feasible_at(W, Fr, br, Aj, gj, tau_max, mass, cx, cy, actuated,
                           fext=(mid * u[0], mid * u[1], cz)):
                lo = mid
            else:
                hi = mid
        out.append(lo)
    return angs, np.array(out)


def slip_margins(m, d, subset, lam):
    """Friction-cone utilisation per contact from a solved force distribution."""
    contacts, P, A, g, mass, com, Rs = blocks(m, d, subset)
    nfoot = len(P) - len(subset)
    out = {}
    for i in range(len(P)):
        n = lam[3 * i]
        t = np.linalg.norm(lam[3 * i + 1:3 * i + 3])
        cap = cs.MU * max(n, 0.0)
        u = float(t / cap) if cap > 1e-9 else (0.0 if t < 1e-9 else float("inf"))
        name = ("foot%d" % i) if i < nfoot else subset[i - nfoot]
        if i < nfoot:
            continue                      # feet are reported as an aggregate
        out[name] = dict(normal=float(n), tangential=float(t), utilisation=u)
    # feet aggregated: the worst corner is what slides first
    worst, wn, wt = 0.0, 0.0, 0.0
    for i in range(nfoot):
        n = lam[3 * i]
        t = np.linalg.norm(lam[3 * i + 1:3 * i + 3])
        cap = cs.MU * max(n, 0.0)
        u = t / cap if cap > 1e-9 else (0.0 if t < 1e-9 else np.inf)
        if u > worst:
            worst, wn, wt = u, n, t
    out["feet_worst"] = dict(normal=float(wn), tangential=float(wt),
                             utilisation=float(worst))
    return out


def self_test(m, d, subset, lam):
    """Cross-check the world-origin wrench map against the QP's own solution.

    The QP balances MuJoCo's floating-base generalised forces (qfrc_bias[:6]);
    this file balances a world-origin wrench built by hand.  They are different
    parameterisations of the same equilibrium, so a force distribution that
    satisfies one must satisfy the other -- and if it does not, the hand-built
    one has a sign or frame error.  Cheap, and it catches exactly the class of
    bug that makes every region come back empty.
    """
    contacts, P, A, g, mass, com, Rs = blocks(m, d, subset)
    W = wrench_rows(P, Rs)
    Mg = mass * G
    want = np.array([0.0, 0.0, Mg, Mg * com[1], -Mg * com[0], 0.0])
    got = W @ lam
    return dict(force_residual=float(np.linalg.norm(got[:3] - want[:3])),
                moment_residual=float(np.linalg.norm(got[3:] - want[3:])),
                weight=float(Mg))


# --------------------------------------------------------------------------- #
def report(target, subset, verbose=True):
    m, d = cs.load()
    ik = cs.solve_ik(m, d, np.array(target), subset)
    qp = cs.equilibrium_qp(m, d, subset)
    slips = slip_margins(m, d, subset, qp["lam"])
    st = self_test(m, d, subset, qp["lam"])
    pts_a, c0, marg_a = equilibrium_region(m, d, subset, actuated=True)
    pts_c, _, marg_c = equilibrium_region(m, d, subset, actuated=False)
    angs, push = max_push(m, d, subset)

    res = dict(target=list(target), subset=list(subset),
               tau_basis=cs.TAU_BASIS, model=os.path.basename(cs.MODEL),
               ik=dict(reach=ik["reach"], foot=ik["foot"],
                       site=ik.get("site", {}), ok=bool(ik.get("ok"))),
               feasible=qp["feasible"], max_ratio=qp["max_ratio"],
               effort=qp["effort"], slip=slips, self_test=st,
               com=[float(c0[0]), float(c0[1])],
               margin_actuated=marg_a, margin_contact=marg_c,
               region_actuated=pts_a.tolist(), region_contact=pts_c.tolist(),
               push_dirs=angs.tolist(), push_newtons=push.tolist())
    if verbose:
        print("target %s  subset %s  basis=%s  model=%s"
              % (target, "+".join(subset) or "legs-only", cs.TAU_BASIS,
                 os.path.basename(cs.MODEL)))
        print("  IK reach %.4f m  foot %.4f m  feasible=%s  peak tau %.3f"
              % (ik["reach"], ik["foot"], qp["feasible"], qp["max_ratio"]))
        print("  slip utilisation (1.0 = on the friction cone):")
        for k, v in slips.items():
            flag = "  <-- SLIDING" if v["utilisation"] > 0.95 else ""
            print("    %-12s n=%7.1f N  t=%6.1f N  u=%.3f%s"
                  % (k, v["normal"], v["tangential"], v["utilisation"], flag))
        print("  wrench self-test (QP solution vs world-origin map): "
              "force %.2f N, moment %.2f Nm, weight %.0f N"
              % (st["force_residual"], st["moment_residual"], st["weight"]))
        print("  equilibrium-region margin:  contact-only %.4f m   actuated %.4f m"
              % (marg_c, marg_a))
        print("  max resistible push:  fwd %.0f N  back %.0f N  lat %.0f N"
              % (push[0], push[len(push) // 2], push[len(push) // 4]))
    return res


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 3:
        tgt = [float(a) for a in args[:3]]
        sub = tuple(args[3].split(",")) if len(args) > 3 else ("elbow", "forearm")
    else:
        tgt, sub = [1.00, 0.15, 1.05], ("elbow", "forearm")
    r = report(tgt, sub)
    out = "stability_%s.json" % ("+".join(sub) or "legs")
    json.dump(r, open(out, "w"), indent=1)
    print("wrote", out)
