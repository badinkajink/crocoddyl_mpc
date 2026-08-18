#!/usr/bin/env python3
"""Static-equilibrium CoM regions and margins for the poses MJPC actually reaches.

The S11 handoff page (docs/lean/2026-08-04_mjpc_handoff.html §5) drew these
regions for poses produced by the offline IK: "here is the region a braced pose
would have". That answers whether a mode is worth wanting. It does not answer
whether the controller ever gets there -- and S12 then measured MJPC arriving at
the right contact SET and ~1.5 rad away from the certified pose, so the two
questions had visibly come apart.

This computes the same quantities at the ROLLOUT's own settled pose, with the
contact set MuJoCo's narrowphase actually reports, so the margin describes the
robot the controller produced. Everything below is stability.py's LP -- the
multi-contact support region of Bretl & Lall (2008) / Caron et al. (2015), ray-
shot at 10 deg and bisected to ~6 um, solved twice (contact-only, and with the
actuator torque envelope). Only the pose it is asked about is new.

Each panel also draws the LEGS-ONLY region at the same pose, which is the
comparison that matters here: it is the region the robot would have if its arm
contacts carried nothing, so the gap between the two is what the brace bought,
measured on the pose that exists rather than on the pose that was designed.

usage: simple_region.py --run DIR [--cells a,b,c] [--out DIR] [--times 4,9,14]
"""
import argparse
import json
import os

import numpy as np
import mujoco

import simple_lean as S
import contact_select as cs
import stability as st

# Candidate contacts, in stability.py's naming. `hip`/`torso` are the trunk
# contacts the keepout is supposed to prevent; they are included so that a
# rollout that cheats on the table edge shows up here as a load-bearing contact
# rather than as a silently better margin.
ARM = ("elbow", "forearm", "palm")
TRUNK = ("hip", "torso")
BODY = {"elbow": "left_shoulder_yaw_link", "forearm": "left_elbow_link",
        "palm": "left_magpie_gripper", "hip": "torso_link", "torso": "torso_link"}


def achieved_subset(m, d):
    """Which candidate links MuJoCo reports on the slab at this pose."""
    table = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top_collision")
    out = []
    for name in ARM:
        gs = S.body_geoms(m, BODY[name])
        if S.touching(m, d, gs, table):
            out.append(name)
    # The trunk is one body carrying both the `hip` capsule and the `torso` box,
    # so ask per GEOM which of the two is down -- crediting the whole body to
    # either site would put a contact force at a point nothing is touching.
    for name in TRUNK:
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                              "hip" if name == "hip" else "torso")
        if g >= 0 and S.touching(m, d, {g}, table):
            out.append(name)
    return out


def pose_at(m, d, col, rows, k):
    d.qpos[:] = rows[k, [col["qpos%d" % i] for i in range(m.nq)]]
    d.qvel[:] = 0
    d.ctrl[:] = rows[k, [col["ctrl%d" % i] for i in range(m.nu)]]
    mujoco.mj_forward(m, d)


def region_at(m, d, subset):
    pa, c0, ma = st.equilibrium_region(m, d, subset, actuated=True)
    pc, _, mc = st.equilibrium_region(m, d, subset, actuated=False)
    qp = cs.equilibrium_qp(m, d, subset)
    contacts, P, A, g, mass, com, Rs = st.blocks(m, d, subset)
    labels = ["foot"] * (len(P) - len(subset)) + list(subset)
    pts = [dict(label=lab, p=[float(v) for v in p],
                fn=float(qp["lam"][3 * i]),
                ft=float(np.linalg.norm(qp["lam"][3 * i + 1:3 * i + 3])))
           for i, (lab, p) in enumerate(zip(labels, P))]
    slips = st.slip_margins(m, d, subset, qp["lam"])
    return dict(actuated=pa.tolist(), contact=pc.tolist(),
                com=[float(c0[0]), float(c0[1])],
                margin_a=float(ma), margin_c=float(mc), pts=pts,
                feasible=bool(qp["feasible"]), peak=float(qp["max_ratio"]),
                slip={k: float(v["utilisation"]) for k, v in slips.items()},
                table_x=[float(v) for v in cs.table_x_range(m, d)],
                table_y=[float(v) for v in cs.table_y_range(m, d)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cells", default="far_none_s0,far_brace_s0,far_palm_s0,"
                                       "vfar_brace_s0")
    ap.add_argument("--out", default="")
    ap.add_argument("--times", default="",
                    help="comma-separated sim times for a margin-vs-time series")
    a = ap.parse_args()
    out = a.out or a.run

    # ik_margin=0: the regions must be built on contacts that EXIST, not on the
    # inflated look-ahead margin the IK uses to keep its trust region feasible.
    m, d = cs.load(ik_margin=0)

    R = {}
    for cell in a.cells.split(","):
        p = os.path.join(a.run, cell + ".csv")
        if not os.path.exists(p):
            print("  missing", p)
            continue
        col, rows, meta = S.load_traj(p)
        k = len(rows) - 3                      # settled: the last full frame
        pose_at(m, d, col, rows, k)
        sub = achieved_subset(m, d)
        r = region_at(m, d, sub)
        r["subset"] = sub
        r["t"] = float(rows[k, col["time"]])
        R[cell] = r
        # legs-only at the SAME pose: the region if the arm carried nothing
        pose_at(m, d, col, rows, k)
        R[cell + "__legs"] = region_at(m, d, [])
        print("%-18s t=%.1f contacts=%-24s margin %+6.0f mm actuated "
              "(%+6.0f contact-only)   legs-only %+6.0f mm   peak tau %.2f"
              % (cell, r["t"], "+".join(sub) or "none", 1000 * r["margin_a"],
                 1000 * r["margin_c"], 1000 * R[cell + "__legs"]["margin_a"],
                 r["peak"]))

        if a.times:
            ser = []
            for tt in [float(x) for x in a.times.split(",")]:
                kk = int(np.argmin(np.abs(rows[:, col["time"]] - tt)))
                pose_at(m, d, col, rows, kk)
                s = achieved_subset(m, d)
                _, _, ma = st.equilibrium_region(m, d, s, actuated=True)
                _, _, ml = st.equilibrium_region(m, d, [], actuated=True)
                ser.append(dict(t=float(rows[kk, col["time"]]), subset=s,
                                margin_a=float(ma), margin_legs=float(ml)))
                print("      t=%5.1f  %-22s margin %+6.0f mm  (legs-only %+6.0f)"
                      % (ser[-1]["t"], "+".join(s) or "none",
                         1000 * ma, 1000 * ml))
            R[cell]["series"] = ser

    dst = os.path.join(out, "regions_simple.json")
    with open(dst, "w") as f:
        json.dump(R, f, indent=1)
    print("wrote", dst)


if __name__ == "__main__":
    main()
