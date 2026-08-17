#!/usr/bin/env python3
"""Is a contact CONTROLLABLE and RECOVERABLE, not merely optimal?

Static effort says a contact is cheap. It says nothing about whether the robot can
modulate it precisely, or let go of it without falling. Two metrics:

MODULATION AUTHORITY
    How far can the normal force at contact i be varied while still satisfying
    equilibrium, friction cones and actuator torque limits? Solved as two LPs
    (max and min of lambda_i^n over the feasible set). A wide admissible band means
    the contact force is a controllable degree of freedom; a band pinned to a point
    means the contact is whatever the geometry makes it and cannot be commanded.
    Reported normalised: (lam_max - lam_min) / lam_nominal.

RELEASE MARGIN
    Is the SAME pose still in static equilibrium with contact i deleted? If yes the
    robot can break that contact in place and remains standing -- recovery is a local
    move. If no, the pose is committed to the contact: releasing it means falling,
    and recovery requires a whole-body manoeuvre back out of the lean.
    Reported as the residual unbalanced base wrench without the contact (0 = free to
    release) and the resulting peak torque ratio.
"""
import itertools
import sys

import numpy as np
from scipy.optimize import linprog

import contact_select as cs


def _blocks(m, d, subset):
    contacts = []
    for f in cs.FEET:
        contacts += cs.foot_corners(m, d, f)
    for s in subset:
        contacts.append(cs.SITES[s])
    J = [cs.point_jac(m, d, b, o)[:, :cs.N_ROBOT_DOF] for b, o in contacts]
    A = np.hstack([Ji.T for Ji in J])
    d.qvel[:] = 0; d.qacc[:] = 0
    import mujoco; mujoco.mj_forward(m, d)
    g = d.qfrc_bias[:cs.N_ROBOT_DOF].copy()
    return A, g, len(contacts)


def modulation_band(m, d, subset, idx):
    """LP max/min of the normal force at contact `idx` over the feasible set."""
    A, g, nc = _blocks(m, d, subset)
    tau_max = cs.torque_limits(m)                 # TAU_BASIS, see contact_select
    nv = 3 * nc
    Aeq, beq = A[:6], g[:6]                       # base equilibrium, hard
    Aj, gj = A[6:], g[6:]
    # |tau| <= tau_max  ->  |gj - Aj lam| <= tau_max
    Aub = np.vstack([Aj, -Aj]); bub = np.concatenate([tau_max + gj, tau_max - gj])
    # friction pyramid + unilateral
    for i in range(nc):
        for t in (0, 1):
            # want |lam_t| <= mu*lam_n/sqrt(2), i.e. BOTH
            #    +lam_t - mu*lam_n/sqrt(2) <= 0
            #    -lam_t - mu*lam_n/sqrt(2) <= 0
            # (negating the whole row instead pins lam_t TO the cone boundary)
            rp = np.zeros(nv); rp[3*i+t] = 1.0;  rp[3*i+2] = -cs.MU/np.sqrt(2)
            rm = np.zeros(nv); rm[3*i+t] = -1.0; rm[3*i+2] = -cs.MU/np.sqrt(2)
            Aub = np.vstack([Aub, rp, rm]); bub = np.concatenate([bub, [0.0, 0.0]])
    bounds = [(None, None)] * nv
    for i in range(nc):
        bounds[3*i+2] = (0.0, None)
    c = np.zeros(nv); c[3*idx+2] = 1.0
    lo = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bounds)
    hi = linprog(-c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bounds)
    if not (lo.success and hi.success):
        return None
    return float(lo.x[3*idx+2]), float(-hi.fun)


def release_margin(m, d, subset, site):
    """Equilibrium at the SAME pose with `site` deleted."""
    reduced = tuple(s for s in subset if s != site)
    return cs.equilibrium_qp(m, d, reduced)


if __name__ == "__main__":
    tgt = [float(x) for x in sys.argv[1:4]]
    subset = tuple(sys.argv[4].split(",")) if len(sys.argv) > 4 else ("elbow", "forearm", "hip")
    m, d = cs.load()
    P = cs.solve_ik(m, d, np.array(tgt), subset)
    base = cs.equilibrium_qp(m, d, subset)
    print(f"target {tgt}  subset {subset}")
    print(f"  pose: reach {P['reach']:.4f}  pen {P['penetration']*1000:.1f}mm  "
          f"peak |tau|/lim {base['max_ratio']:.3f}")
    n_foot = 8
    for k, s in enumerate(subset):
        band = modulation_band(m, d, subset, n_foot + k)
        rel = release_margin(m, d, subset, s)
        nom = float(np.linalg.norm(base["lam"][3*(n_foot+k):3*(n_foot+k)+3]))
        if band is None:
            print(f"  {s:8s} modulation: LP infeasible")
        else:
            lo, hi = band
            span = hi - lo
            print(f"  {s:8s} normal force {nom:6.1f} N | admissible band "
                  f"[{lo:6.1f}, {hi:6.1f}] N  span {span:6.1f} N "
                  f"({'CONTROLLABLE' if span > 20 else 'PINNED'})")
        print(f"           release: base residual {rel['base_residual']:6.2f} N, "
              f"peak {rel['max_ratio']:.2f} -> "
              f"{'CAN RELEASE IN PLACE' if rel['base_residual'] < 1.0 and rel['max_ratio'] <= 1.0 else 'COMMITTED (releasing falls)'}")
