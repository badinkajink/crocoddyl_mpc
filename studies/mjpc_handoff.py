#!/usr/bin/env python3
"""Can MJPC's sampling planner get to the pose the QP says exists?

The offline study produces a braced pose by solving IK + a static-equilibrium QP.
That pose is a CERTIFICATE: it says an admissible configuration exists.  It says
nothing about whether a sampling MPC can FIND it online, which is the question
that decides whether the handoff is MJPC or something like crocoddyl.

This runs `testspeed --dump_traj` on the lean task and scores the resulting
rollout with EXACTLY the machinery the offline study uses -- same contact
resolution, same torque basis, same equilibrium region -- so the comparison is
pose-to-pose, not cost-to-pose.  An average cost cannot be compared with a
margin in metres; a final configuration can.

Per run it reports, at the best (lowest-reach-residual, upright) frame:
  reach       hand-to-target distance
  contacts    which brace bodies MuJoCo reports touching the table
  peak        max |tau| / tau_max over the 27 joints, from the LOGGED torques
  margin      actuated equilibrium-region margin at that configuration
  upright     pelvis height and base tilt, so a "converged" topple is visible

and compares against the QP solution at the same target.

usage: mjpc_handoff.py [--seeds N] [--time SEC] [--strategy S] [--target x,y,z]
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np
import mujoco

import contact_select as cs
import stability as st

ROOT = "/home/humanoid/Programs/mjpc_icra2026"
BIN = ROOT + "/build/bin/testspeed"
TASK = "Lean H12 Magpie"
BRACE_BODIES = {"elbow": "%s_shoulder_yaw_link" % cs.BRACE_ARM,
                "forearm": "%s_elbow_link" % cs.BRACE_ARM,
                "palm": "%s_magpie_gripper" % cs.BRACE_ARM,
                "wrist": "%s_wrist_yaw_link" % cs.BRACE_ARM,
                "hip": "torso_link"}


def run_mjpc(out_csv, seconds, strategy, start_key, threads):
    cmd = [BIN, "--task=" + TASK, "--total_time=%g" % seconds,
           "--planner_thread=%d" % threads, "--dump_traj=" + out_csv]
    if strategy >= 0:
        cmd.append("--strategy=%d" % strategy)
    if start_key:
        cmd.append("--start_key=" + start_key)
    t = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    wall = [l for l in t.stdout.splitlines() if "Total wall time" in l]
    avg = [l for l in t.stdout.splitlines() if "Average cost" in l]
    return dict(returncode=t.returncode,
                wall=wall[0] if wall else "", avg=avg[0] if avg else "",
                stderr=t.stderr[-2000:] if t.returncode else "")


def load_traj(path):
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#")]
    r = csv.reader(lines)
    hdr = next(r)
    rows = np.array([[float(v) for v in row] for row in r])
    col = {n: i for i, n in enumerate(hdr)}
    return hdr, col, rows


def score_frame(m, d, q, target, tau_log, region=True):
    """Static score of ONE logged configuration, with the study's own metrics.

    `region=False` skips the equilibrium-region LP, which is ~36 rays x 18
    bisections of a linear program and dominates the cost by three orders of
    magnitude.  Scanning a whole rollout needs the cheap fields on every frame
    and the region on only the handful of frames that get reported."""
    d.qpos[:] = q
    d.qvel[:] = 0
    d.qacc[:] = 0
    mujoco.mj_forward(m, d)

    hand = cs.point_world(m, d, cs.REACH_BODY, cs.REACH_OFF)
    reach = float(np.linalg.norm(np.asarray(target) - hand))

    tbl = cs.bid(m, "table")
    touching = set()
    for c in range(d.ncon):
        con = d.contact[c]
        b1, b2 = m.geom_bodyid[con.geom[0]], m.geom_bodyid[con.geom[1]]
        if b1 == tbl:
            touching.add(b2)
        elif b2 == tbl:
            touching.add(b1)
    contacts = sorted(k for k, b in BRACE_BODIES.items()
                      if cs.bid(m, b) in touching)

    tau_max = cs.torque_limits(m)
    peak_log = float(np.max(np.abs(tau_log) / tau_max))

    # equilibrium region over whatever contacts the pose ACTUALLY has -- the
    # planner is not obliged to honour the contact plan, so scoring it against
    # the planned set would credit or blame it for contacts it never made.
    subset = tuple(k for k in ("elbow", "forearm", "palm", "hip") if k in contacts)
    marg = float("nan")
    if region:
        try:
            _, _, marg = st.equilibrium_region(m, d, subset, actuated=True)
        except Exception:
            pass

    pel = cs.bid(m, "pelvis")
    R = d.xmat[pel].reshape(3, 3)
    tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
    return dict(reach=reach, contacts=contacts, peak_logged=peak_log,
                margin=float(marg), pelvis_z=float(d.xpos[pel][2]), tilt_deg=tilt,
                subset=list(subset))


def qp_reference(target, subset):
    m, d = cs.load()
    ik = cs.solve_ik(m, d, np.asarray(target), subset)
    qp = cs.equilibrium_qp(m, d, subset)
    _, _, marg = st.equilibrium_region(m, d, subset, actuated=True)
    return dict(reach=float(ik["reach"]), placed=bool(ik["all_placed"]),
                peak=float(qp["max_ratio"]), margin=float(marg),
                feasible=bool(qp["feasible"]), subset=list(subset))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--time", type=float, default=12.0)
    ap.add_argument("--strategy", type=int, default=22)
    ap.add_argument("--start-key", default="stand")
    ap.add_argument("--threads", type=int, default=15)
    ap.add_argument("--target", default="")
    ap.add_argument("--outdir", default="runs/mjpc_handoff")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    m0, d0 = cs.load(ik_margin=0)
    if a.target:
        target = [float(v) for v in a.target.split(",")]
    else:
        # the task's own reach_target numeric -- the target MJPC is being scored
        # against must be the one it is actually driving at
        nid = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_NUMERIC, "reach_target")
        adr = m0.numeric_adr[nid]
        target = [float(v) for v in m0.numeric_data[adr:adr + 3]]
    print("target %s   task=%s strategy=%d start=%s  %.0f s x %d seeds"
          % (np.round(target, 4), TASK, a.strategy, a.start_key, a.time, a.seeds))

    ref = {}
    for sub in [("elbow", "forearm"), ("elbow", "forearm", "palm")]:
        ref["+".join(sub)] = qp_reference(target, sub)
        r = ref["+".join(sub)]
        print("  QP  %-22s reach %.4f placed=%s peak %.3f margin %.4f"
              % ("+".join(sub), r["reach"], r["placed"], r["peak"], r["margin"]))

    runs = []
    for s in range(a.seeds):
        csv_path = os.path.join(a.outdir, "traj_seed%d.csv" % s)
        print("\n-- seed %d --" % s, flush=True)
        meta = run_mjpc(csv_path, a.time, a.strategy, a.start_key, a.threads)
        if meta["returncode"] != 0:
            print("   FAILED:", meta["stderr"][-400:])
            runs.append(dict(seed=s, failed=True, **meta))
            continue
        print("  ", meta["wall"], "|", meta["avg"])

        hdr, col, rows = load_traj(csv_path)
        nq, nu = m0.nq, m0.nu
        qi = [col["qpos%d" % i] for i in range(nq)]
        ai = [col["afrc%d" % i] for i in range(nu)]

        # score the LAST second (the sustained pose), plus the single best frame
        t = rows[:, col["time"]]
        tail = rows[t >= t[-1] - 1.0]
        scored = []
        for row in tail[::10]:
            scored.append(score_frame(m0, d0, row[qi], target, row[ai]))
        upright = [x for x in scored if x["tilt_deg"] < 60 and x["pelvis_z"] > 0.5]
        best = min(upright or scored, key=lambda x: x["reach"])
        final = scored[-1]
        print("   final : reach %.4f  contacts %-28s peak %.3f margin %.4f "
              "pelvis_z %.3f tilt %.0f deg"
              % (final["reach"], "+".join(final["contacts"]) or "none",
                 final["peak_logged"], final["margin"], final["pelvis_z"],
                 final["tilt_deg"]))
        print("   best  : reach %.4f  contacts %-28s peak %.3f margin %.4f"
              % (best["reach"], "+".join(best["contacts"]) or "none",
                 best["peak_logged"], best["margin"]))
        runs.append(dict(seed=s, failed=False, wall=meta["wall"],
                         avg=meta["avg"], final=final, best=best,
                         n_frames=int(len(rows))))

    out = dict(task=TASK, strategy=a.strategy, start_key=a.start_key,
               seconds=a.time, threads=a.threads, target=target,
               basis=cs.TAU_BASIS, site_set=cs.SITE_SET, mu=cs.MU,
               qp_reference=ref, runs=runs)
    dst = os.path.join(a.outdir, "handoff.json")
    json.dump(out, open(dst, "w"), indent=1)
    print("\nwrote", dst)


if __name__ == "__main__":
    main()
