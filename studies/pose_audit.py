#!/usr/bin/env python3
"""Numeric substitute for eyeballing a render.

Checks the specific failure modes the filmstrip catches: body inside the table,
feet off the ground or slid, joints pinned at limits, and whether the claimed
contact sites are the ones actually touching.
"""
import sys
import numpy as np
import mujoco
import contact_select as cs


def audit(target, subset):
    m, d = cs.load()
    P = cs.solve_ik(m, d, np.array(target), subset)
    tbl = cs.bid(m, "table")
    z_tab = cs.table_top_z(m, d)

    # 1. torso / pelvis / head clear of the tabletop
    clear = {}
    for b in ("torso_link", "pelvis", "left_hip_yaw_link", "right_hip_yaw_link"):
        i = cs.bid(m, b)
        if i >= 0:
            clear[b] = round(float(d.xpos[i][2] - z_tab), 3)

    # 2. feet planted: sole height above floor, and drift from seed
    feet = {}
    for f in cs.FEET:
        i = cs.bid(m, f)
        feet[f] = round(float(d.xpos[i][2]), 3)

    # 3. joints pinned at their limits (a classic IK-gone-wrong signature)
    pinned = []
    for j in range(1, m.njnt):
        if m.jnt_limited[j] and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            a = m.jnt_qposadr[j]
            lo, hi = m.jnt_range[j]
            if d.qpos[a] <= lo + 1e-3 or d.qpos[a] >= hi - 1e-3:
                pinned.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j))

    # 4. what is ACTUALLY touching the table
    touch = set()
    for c in range(d.ncon):
        con = d.contact[c]
        if con.dist > cs.IK_MARGIN * 0.5:
            continue
        b1, b2 = m.geom_bodyid[con.geom[0]], m.geom_bodyid[con.geom[1]]
        if b1 == tbl:
            touch.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b2))
        elif b2 == tbl:
            touch.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b1))

    return dict(reach=round(P["reach"], 4), foot=round(P["foot"], 4),
                pen_mm=round(P["penetration"] * 1000, 1),
                clear=clear, feet_z=feet, pinned=pinned,
                touching=sorted(touch), placed=P["sites_placed"])


if __name__ == "__main__":
    tgt = [float(x) for x in sys.argv[1:4]]
    for sub in [(), ("palm",), ("elbow", "forearm"), ("elbow", "forearm", "palm")]:
        r = audit(tgt, sub)
        ok = (all(v > -0.01 for v in r["clear"].values())
              and r["pen_mm"] < 10 and r["foot"] < 0.02 and not r["pinned"])
        print(f"  {str(sub):30s} {'PLAUSIBLE' if ok else 'SUSPECT  '} "
              f"reach={r['reach']:.4f} foot={r['foot']:.4f} pen={r['pen_mm']:5.1f}mm")
        print(f"      clearance above tabletop: {r['clear']}")
        print(f"      touching table: {r['touching']}")
        if r["pinned"]:
            print(f"      JOINTS PINNED AT LIMIT: {r['pinned']}")
