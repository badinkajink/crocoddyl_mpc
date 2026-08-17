#!/usr/bin/env python3
"""Is the braced pose actually falling to the right, and how skewed is the arm?

User observation (2026-08-04): "the bracing arm is skewed to the left, and overall
the body looks like it is precariously leaning down to the right, about to
fall/collapse, and probably placing large torques on the right leg."

That is three separate measurable claims, and they are not the same claim:

  SKEW      the angle between the bracing forearm's long axis and the table's
            long axis (world +x).  0 deg = the forearm lies along the table's
            deep direction, which is the 1.18 m one.
  LEAN      lateral CoM offset from the midpoint of the two feet, and base roll.
  LOAD      left-vs-right leg torque asymmetry, per joint and as a whole-leg
            normalised effort, so "large torques on the right leg" is a number.

Reported per contact set at a given target, for whatever stance is configured, so
the same script answers "is it skewed" before a fix and "did the fix work" after.

usage: asymmetry.py [x y z] [set,set,...]
       STANCE_DY=... FOREARM_ALIGN=0|1 SITE_SET=v1|v2
"""
import json
import os
import sys

import numpy as np
import mujoco

import contact_select as cs
import stability as st

SETS = [("elbow", "forearm"), ("elbow", "forearm", "palm")]
_LEG_JOINTS = ["hip_yaw", "hip_pitch", "hip_roll", "knee", "ankle_pitch", "ankle_roll"]


def forearm_axis(m, d):
    """World direction of the bracing forearm, elbow -> wrist."""
    a = cs.point_world(m, d, "%s_elbow_link" % cs.BRACE_ARM, np.zeros(3))
    b = cs.point_world(m, d, "%s_wrist_roll_link" % cs.BRACE_ARM, np.zeros(3))
    v = b - a
    return v / max(np.linalg.norm(v), 1e-9)


def leg_indices(m):
    """actuator indices for the left and right legs, in _LEG_JOINTS order."""
    L, R = [], []
    for i in range(m.nu):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        base = cs._base_joint(n)
        if base in _LEG_JOINTS:
            (L if n.startswith("left_") else R).append(i)
    return L, R


def measure(target, subset):
    m, d = cs.load()
    ik = cs.solve_ik(m, d, np.asarray(target), subset)
    qp = cs.equilibrium_qp(m, d, subset)
    tau, tmax = qp["tau"], cs.torque_limits(m)

    # --- SKEW -------------------------------------------------------------
    v = forearm_axis(m, d)
    vh = v[:2] / max(np.linalg.norm(v[:2]), 1e-9)
    skew = float(np.degrees(np.arctan2(vh[1], vh[0])))   # 0 = along table +x

    # --- LEAN -------------------------------------------------------------
    pel = cs.bid(m, "pelvis")
    com = np.array(d.subtree_com[pel])
    feet = np.array([d.xpos[cs.bid(m, f)] for f in cs.FEET])
    mid = feet.mean(axis=0)
    lat = float(com[1] - mid[1])                          # + = toward left foot
    Rb = d.xmat[pel].reshape(3, 3)
    roll = float(np.degrees(np.arctan2(Rb[2, 1], Rb[2, 2])))

    # --- LOAD -------------------------------------------------------------
    L, R = leg_indices(m)
    rl = np.abs(tau[L]) / tmax[L]
    rr = np.abs(tau[R]) / tmax[R]
    # per-leg effort: RMS of the normalised joint torques, the same currency the
    # QP minimises, so this is comparable with `peak` elsewhere in the study
    eff_l, eff_r = float(np.sqrt(np.mean(rl ** 2))), float(np.sqrt(np.mean(rr ** 2)))
    imb = (eff_r - eff_l) / max(eff_r + eff_l, 1e-9)      # +1 = all on the right

    # foot normal loads, from the QP's own contact forces (4 corners per foot)
    lam = qp["lam"]
    fn_l = float(sum(lam[3 * i] for i in range(4)))
    fn_r = float(sum(lam[3 * i] for i in range(4, 8)))

    per_joint = {}
    for k, (il, ir) in enumerate(zip(L, R)):
        per_joint[_LEG_JOINTS[k]] = dict(left=float(tau[il]), right=float(tau[ir]),
                                         left_r=float(rl[k]), right_r=float(rr[k]))
    return dict(subset=list(subset), target=list(target),
                stance_dy=cs.STANCE_DY, reach=float(ik["reach"]),
                placed=bool(ik.get("all_placed", True)),
                skew_deg=skew, lateral_com=lat, base_roll_deg=roll,
                eff_left=eff_l, eff_right=eff_r, imbalance=imb,
                foot_fn_left=fn_l, foot_fn_right=fn_r,
                peak=float(qp["max_ratio"]), per_joint=per_joint)


def main():
    tgt = ([float(a) for a in sys.argv[1:4]] if len(sys.argv) >= 4
           else [1.35, -0.2348, 1.0982])
    sets = ([tuple(s.split("+")) for s in sys.argv[4].split(",")]
            if len(sys.argv) > 4 else SETS)

    print("target %s  stance_dy %.2f  site_set %s  forearm_align %s"
          % (tgt, cs.STANCE_DY, cs.SITE_SET, os.environ.get("FOREARM_ALIGN", "1")))
    print("%-24s %6s %8s %7s %7s %7s %7s %8s %8s"
          % ("set", "skew", "lat CoM", "roll", "eff L", "eff R", "imbal",
             "footN L", "footN R"))
    rows = []
    for s in sets:
        r = measure(tgt, s)
        rows.append(r)
        print("%-24s %6.1f %8.3f %7.1f %7.3f %7.3f %+7.3f %8.0f %8.0f"
              % ("+".join(s), r["skew_deg"], r["lateral_com"], r["base_roll_deg"],
                 r["eff_left"], r["eff_right"], r["imbalance"],
                 r["foot_fn_left"], r["foot_fn_right"]), flush=True)

    print("\nper-joint leg torque [Nm] and fraction of limit, %s:" % "+".join(sets[-1]))
    for j, v in rows[-1]["per_joint"].items():
        print("   %-12s L %7.1f (%.2f)   R %7.1f (%.2f)"
              % (j, v["left"], v["left_r"], v["right"], v["right_r"]))

    dst = os.environ.get("OUT", "asymmetry.json")
    json.dump(dict(target=tgt, stance_dy=cs.STANCE_DY, site_set=cs.SITE_SET,
                   rows=rows), open(dst, "w"), indent=1)
    print("wrote", dst)


if __name__ == "__main__":
    main()
