#!/usr/bin/env python3
"""How precisely can each contact POINT be placed?

Force controllability (controllability.py) is not the same as being able to put the
contact where you meant to. A contact driven through a long chain with a large moment
arm slides a long way for a small joint error. Metric: the largest singular value of
the contact point's position Jacobian restricted to the ACTUATED joints --
    sigma_max(J_p)  [m per rad]
i.e. worst-case contact-point displacement per unit joint error. Multiply by a
realistic joint tracking error to get millimetres of contact-placement uncertainty.

Also reports how many joints carry most of that sensitivity: a contact whose placement
depends on 2 joints is servo-able; one that depends on 10 is not.
"""
import sys
import numpy as np
import mujoco
import contact_select as cs

JOINT_ERR = np.deg2rad(1.0)   # ~1 deg tracking error, generous for a good servo


def precision(m, d, site):
    body, off = cs.SITES[site]
    J = cs.point_jac(m, d, body, off)[:, 6:cs.N_ROBOT_DOF]   # actuated only
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    # which joints dominate the worst-case direction
    v = np.abs(Vt[0])
    order = np.argsort(-v)
    names = []
    for k in order[:3]:
        dof = 6 + k
        j = int(np.argmax(m.jnt_dofadr == dof))
        names.append((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j), round(float(v[k]), 2)))
    # effective number of joints carrying the sensitivity (participation ratio)
    p = v**2 / (v**2).sum()
    neff = 1.0 / (p**2).sum()
    return float(S[0]), neff, names


if __name__ == "__main__":
    tgt = [float(x) for x in sys.argv[1:4]]
    subset = tuple(sys.argv[4].split(",")) if len(sys.argv) > 4 else ("elbow", "forearm", "hip")
    m, d = cs.load()
    cs.solve_ik(m, d, np.array(tgt), subset)
    print(f"target {tgt}   (contact-placement error for {np.rad2deg(JOINT_ERR):.0f} deg joint error)")
    for s in cs.SITES:
        smax, neff, names = precision(m, d, s)
        print(f"  {s:8s} sigma_max {smax:5.2f} m/rad -> {smax*JOINT_ERR*1000:5.1f} mm "
              f"| effective joints {neff:4.1f} | top: {names}")
