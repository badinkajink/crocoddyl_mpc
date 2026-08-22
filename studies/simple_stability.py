#!/usr/bin/env python3
"""Static-stability metrics for a Lean Simple frame, computed from the contacts
the frame ACTUALLY has.

This is stability.py's machinery (Bretl & Lall 2008; Caron et al. 2015) pointed
at a different question. stability.py scores a pose the offline QP *solved*, so
it takes the contact set as an argument and builds the contacts itself. Here the
pose comes off an MJPC rollout, where nobody declared anything: the only honest
contact set is the one MuJoCo resolved at that frame. That difference is not
cosmetic -- the study has twice been burned by trusting a mode's NAME over its
load path (a declared `elbow+palm` whose palm carried 0.00 N; a brace actually
held by an uncommanded wrist pad), and reading d.contact is what makes those
visible instead of assumed.

Two other things are deliberately different from stability.py:

  * NOTHING IS HARDCODED TO A MODEL. stability.py slices `[:cs.N_ROBOT_DOF]`
    with N_ROBOT_DOF = 33. Lean Simple has nv = 39 -- the extra 6 are the
    object_anchor free joint -- so that slice is right here only by accident,
    and would be silently wrong on the next model. The robot's DOFs are derived
    from the pelvis subtree instead.
  * FRICTION COMES FROM THE MODEL, not from a MU environment variable. These
    frames are scored against the same coefficients the rollout was simulated
    with; a margin computed at a different mu than the physics used is a number
    about a robot that does not exist.

Metrics:

  equilibrium_margin  signed distance [m] from the CoM's ground projection to
                      the boundary of the static-equilibrium region -- the set
                      of horizontal CoM positions some admissible contact-force
                      distribution can hold. This is the multi-contact
                      generalisation of "distance to the edge of the support
                      polygon", and with a brace the region grows forward. < 0
                      means the frame is not in static equilibrium at all.
  max_push_at         largest horizontal force applicable AT A GIVEN POINT (the
                      reaching hand) that the contact set can still balance,
                      per direction [N]. The static half of "how robust is this
                      to an end-effector disturbance".

Both come in `actuated=False` (contacts + friction only -- what the contact SET
can do) and `actuated=True` (also |tau| <= tau_max, so what THIS robot can hold)
variants. The actuated version is a linearisation: g(q) is frozen at the frame
while the CoM is swept, so it describes this posture, not a re-solved one.
"""
import numpy as np
import mujoco
from scipy.optimize import linprog

G = 9.81
NDIR = 36           # rays for the region boundary; 10 deg resolution
RAY_MAX = 1.5       # [m] longest CoM excursion ever searched
BISECT = 18         # bisection steps per ray -> ~6 um resolution
PUSH_MAX = 800.0    # [N] ceiling for the push search


# --------------------------------------------------------------------------- #
# model structure, derived rather than assumed
# --------------------------------------------------------------------------- #
def robot_dofs(m, root="pelvis"):
    """DOF indices belonging to the robot's subtree, and the root body id.

    Everything else in the model (the table, the object anchor) is environment:
    its DOFs must not appear in the equilibrium rows, and its mass must not be
    added to the weight the contacts have to carry. Getting that wrong doubles
    the modelled weight and puts the CoM inside the table."""
    rb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, root)
    if rb < 0:
        raise ValueError("no body named %r" % root)
    # a body is in the subtree if rb is on its parent chain
    inside = np.zeros(m.nbody, dtype=bool)
    for b in range(m.nbody):
        p = b
        while p > 0:
            if p == rb:
                inside[b] = True
                break
            p = m.body_parentid[p]
    inside[rb] = True
    dofs = [v for v in range(m.nv) if inside[m.dof_bodyid[v]]]
    return np.array(dofs, dtype=int), rb, inside


def tau_limits(m, dofs):
    """Per-DOF torque bound from the actuator force ranges.

    Only joint-transmission actuators map to a single DOF; anything else is left
    unbounded rather than guessed. An unactuated robot DOF (none here beyond the
    floating base, which is excluded separately) gets 0."""
    lim = np.zeros(len(dofs))
    index = {d: i for i, d in enumerate(dofs)}
    for a in range(m.nu):
        if m.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        j = m.actuator_trnid[a, 0]
        v = m.jnt_dofadr[j]
        if v in index:
            fr = m.actuator_forcerange[a]
            lim[index[v]] = abs(fr[1]) if fr[1] != 0 else np.inf
    return lim


# --------------------------------------------------------------------------- #
# contacts, as resolved
# --------------------------------------------------------------------------- #
def resolved_contacts(m, d, inside):
    """Every active robot<->environment contact at this frame.

    Returns a list of dicts with the contact point, a frame whose FIRST ROW is
    the normal pointing into the robot, the friction coefficient MuJoCo used,
    and the robot body carrying it.

    The normal is re-oriented per contact because MuJoCo's frame points from
    geom1 to geom2 and either one may be the robot. Skipping that flip makes
    half the contacts able to *pull*, and the LP then certifies poses that fall
    over."""
    out = []
    for i in range(d.ncon):
        c = d.contact[i]
        if c.dist > 0:
            continue
        b1 = m.geom_bodyid[c.geom1]
        b2 = m.geom_bodyid[c.geom2]
        r1, r2 = inside[b1], inside[b2]
        if r1 == r2:
            continue                      # robot-robot or env-env: not support
        F = np.array(c.frame).reshape(3, 3).copy()
        body = b2 if r2 else b1
        if r1:
            # normal points geom1 -> geom2, i.e. out of the robot; flip it so
            # lam[0] >= 0 means "the environment pushes the robot".
            F[0] = -F[0]
            F[2] = -F[2]                  # keep the frame right-handed
        out.append(dict(p=np.array(c.pos), R=F, mu=float(c.friction[0]),
                        body=int(body)))
    return out


# --------------------------------------------------------------------------- #
# the LP
# --------------------------------------------------------------------------- #
def friction_rows(cons):
    """Linearised Coulomb cone, 4 facets per contact, using each contact's own mu.

    Two rows per tangential axis:  +lam_t - mu lam_n / sqrt2 <= 0
                                   -lam_t - mu lam_n / sqrt2 <= 0
    Negating the whole row instead pins |lam_t| TO the cone rather than bounding
    it, and every LP comes back infeasible (stability.py, plan S10c.1)."""
    nc = len(cons)
    rows = []
    for i, c in enumerate(cons):
        k = c["mu"] / np.sqrt(2.0)
        for ax in (1, 2):
            for sgn in (+1.0, -1.0):
                r = np.zeros(3 * nc)
                r[3 * i + ax] = sgn
                r[3 * i] = -k
                rows.append(r)
    return np.array(rows), np.zeros(len(rows))


def wrench_rows(cons):
    """(6, 3nc) map from stacked contact-frame forces to the wrench about the
    world origin."""
    nc = len(cons)
    W = np.zeros((6, 3 * nc))
    for i, c in enumerate(cons):
        Rw = c["R"].T                                  # contact frame -> world
        p = c["p"]
        W[0:3, 3 * i:3 * i + 3] = Rw
        skew = np.array([[0, -p[2], p[1]],
                         [p[2], 0, -p[0]],
                         [-p[1], p[0], 0]])
        W[3:6, 3 * i:3 * i + 3] = skew @ Rw
    return W


def jac_rows(m, d, cons, dofs):
    """Stacked J^T over the robot DOFs: maps contact forces to joint torques."""
    J = []
    for c in cons:
        jacp = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jacp, None, c["p"], c["body"])
        J.append(c["R"] @ jacp[:, dofs])
    return np.hstack([Ji.T for Ji in J])               # (ndof, 3nc)


def _feasible(W, Fr, br, Aj, gj, tau_max, mass, cx, cy, actuated,
              fext=None, fpoint=None):
    """Static equilibrium with the CoM at (cx, cy)?

    fext is an extra force (fx, fy, 0) applied at world point `fpoint` -- the
    disturbance the contacts must also balance. Newton-Euler about the origin,
    every external wrench summing to zero:
        sum lam + (0,0,-Mg) + f            = 0
        sum p_i x lam_i + c x (0,0,-Mg) + q x f = 0
    with c x (0,0,-Mg) = (-Mg c_y, +Mg c_x, 0), so the contacts must supply the
    negative of it. This sign has been gotten backwards before, and the symptom
    -- every LP infeasible -- is indistinguishable from a genuinely unstable
    pose, so self_test() checks it against a real force distribution."""
    Mg = mass * G
    wrench = np.array([0.0, 0.0, Mg, Mg * cy, -Mg * cx, 0.0])
    if fext is not None:
        f = np.array([fext[0], fext[1], 0.0])
        q = np.asarray(fpoint, dtype=float)
        wrench[0:3] -= f
        wrench[3:6] -= np.cross(q, f)
    Aub, bub = [Fr], [br]
    if actuated:
        keep = np.isfinite(tau_max)
        Aub += [-Aj[keep], Aj[keep]]
        bub += [tau_max[keep] - gj[keep], tau_max[keep] + gj[keep]]
    res = linprog(np.zeros(W.shape[1]),
                  A_ub=np.vstack(Aub), b_ub=np.concatenate(bub),
                  A_eq=W, b_eq=wrench,
                  bounds=[(0, None) if i % 3 == 0 else (None, None)
                          for i in range(W.shape[1])],
                  method="highs")
    return bool(res.status == 0)


class Frame:
    """Everything the LPs need, assembled once per frame."""

    def __init__(self, m, d, root="pelvis"):
        self.dofs, self.rb, self.inside = robot_dofs(m, root)
        self.cons = resolved_contacts(m, d, self.inside)
        self.n = len(self.cons)
        self.mass = float(m.body_subtreemass[self.rb])
        self.com = np.array(d.subtree_com[self.rb])
        self.ok = self.n > 0
        if not self.ok:
            return
        self.W = wrench_rows(self.cons)
        self.Fr, self.br = friction_rows(self.cons)
        A = jac_rows(m, d, self.cons, self.dofs)
        g = d.qfrc_bias[self.dofs].copy()
        # rows 0:6 are the floating base -- unactuated, already enforced as the
        # equilibrium equality. The torque bound applies to the rest.
        self.Aj, self.gj = A[6:], g[6:]
        self.tau_max = tau_limits(m, self.dofs)[6:]

    def feasible(self, cx, cy, actuated=True, fext=None, fpoint=None):
        return _feasible(self.W, self.Fr, self.br, self.Aj, self.gj,
                         self.tau_max, self.mass, cx, cy, actuated,
                         fext, fpoint)

    # ----------------------------------------------------------------- #
    def equilibrium_region(self, actuated=True, ndir=NDIR):
        """Ray-shoot the region boundary. Returns (pts (ndir,2), com_xy, margin)."""
        if not self.ok:
            return np.zeros((0, 2)), self.com[:2], float("nan")
        c0 = self.com[:2].copy()
        inside0 = self.feasible(c0[0], c0[1], actuated)
        # If the CoM itself is outside, the region can still be non-empty; seed
        # from the contact centroid, which is inside for any set holding weight.
        P = np.array([c["p"] for c in self.cons])
        origin = c0 if inside0 else P[:, :2].mean(axis=0)
        if not self.feasible(origin[0], origin[1], actuated):
            return np.zeros((0, 2)), c0, float("nan")
        pts = []
        for th in np.linspace(0, 2 * np.pi, ndir, endpoint=False):
            u = np.array([np.cos(th), np.sin(th)])
            lo, hi = 0.0, RAY_MAX
            if self.feasible(*(origin + hi * u), actuated=actuated):
                pts.append(origin + hi * u)
                continue
            for _ in range(BISECT):
                mid = 0.5 * (lo + hi)
                if self.feasible(*(origin + mid * u), actuated=actuated):
                    lo = mid
                else:
                    hi = mid
            pts.append(origin + lo * u)
        pts = np.array(pts)
        margin = _poly_margin(pts, c0)
        if not inside0:
            margin = -abs(margin)
        return pts, c0, float(margin)

    def max_push_at(self, point, ndir=16, actuated=True, fmax=PUSH_MAX):
        """Largest horizontal force at `point` the contacts can still balance."""
        if not self.ok:
            return np.zeros(ndir), np.full(ndir, np.nan)
        cx, cy = self.com[0], self.com[1]
        angs = np.linspace(0, 2 * np.pi, ndir, endpoint=False)
        out = []
        for th in angs:
            u = np.array([np.cos(th), np.sin(th)])
            if not self.feasible(cx, cy, actuated):
                out.append(0.0)             # cannot even hold itself
                continue
            lo, hi = 0.0, fmax
            if self.feasible(cx, cy, actuated, fext=hi * u, fpoint=point):
                out.append(hi)
                continue
            for _ in range(14):
                mid = 0.5 * (lo + hi)
                if self.feasible(cx, cy, actuated, fext=mid * u, fpoint=point):
                    lo = mid
                else:
                    hi = mid
            out.append(lo)
        return angs, np.array(out)


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
