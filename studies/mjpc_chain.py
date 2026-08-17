#!/usr/bin/env python3
"""Can MJPC ENTER, HOLD and LEAVE the braced pose the offline study certifies?

The previous handoff run (S11) asked only "does sampling find a brace", got no,
and read that as a cost-definition problem.  It was, but not the one it named.
Strategy 22 is EIGHT keyframes -- stand_up, forearm_brace_lean,
forearm_brace_reach, forearm_brace_release, standback_r1..r3, stand_up -- and
every one of them carries `success_sustain_time: 9999` and `time_limit: 9999`.
With the Phase parameter at its -1 default, lean.cc's auto-advance needs
`t - success_start > 9999` to fire.  It never fires.  A headless run therefore
spends its entire duration in keyframe 0, whose weight map sets Brace Pos, Brace
Force, Contact and Reaching Hand Dist to ZERO.  S11 measured the STAND cost and
reported it as the brace task's.

So the phase has to be driven externally (testspeed --phase_schedule), and then
three separate questions become askable, which is the point:

  hold    start ON the QP pose with the brace cost live.  Does it stay?
  enter   start standing, switch to the brace cost.  Does it get there, and does
          it pick the contacts the offline enumeration picked?
  return  from a held brace, walk the release/standback phases.  Does it recover
          a stand, and does it do so without a fall?
  chain   all three back to back, which is the deployable artefact.

`hold_p0` is the control: the same QP start with the STAND cost live, i.e. what
S11 actually ran.  If hold_p1 survives and hold_p0 does not, the S11 "MJPC
rejects the braced pose" result is explained by the phase, not by the planner.

usage: mjpc_chain.py [--episodes hold,hold_p0,enter,return,chain] [--seeds N]
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np
import mujoco

import contact_select as cs
import stability as st

ROOT = "/home/humanoid/Programs/mjpc_icra2026"
BIN = ROOT + "/build/bin/testspeed"
TASK = "Lean H12 Magpie"
QP_POSE = ROOT + "/studies/runs/2026-08-04_session11/qp_brace_qpos.txt"

BRACE_BODIES = {"elbow": "%s_shoulder_yaw_link" % cs.BRACE_ARM,
                "forearm": "%s_elbow_link" % cs.BRACE_ARM,
                "palm": "%s_magpie_gripper" % cs.BRACE_ARM,
                "wrist": "%s_wrist_yaw_link" % cs.BRACE_ARM,
                "hip": "torso_link"}

PHASE_NAMES = ["stand_up", "brace_lean", "brace_reach", "brace_release",
               "standback_r1", "standback_r2", "standback_r3", "stand_up_end"]

# short actuator names, filled from the model on first load
ACT_NAMES = []

# Episode definitions: (phase schedule, start pose, seconds).
# Dwell times are the strategy's own target_ramp_sec plus a settling margin --
# brace_lean ramps over 14 s, so anything shorter measures a partial ramp and
# calls it a failure to brace.
EPISODES = {
    # 1-3. HOLD, three costs, ONE start pose.  The QP pose is braced AND
    # reaching, so phase 2 (brace_reach) is the cost that matches it, phase 1
    # (brace_lean) braces but zeroes Reaching Hand Dist, and phase 0 is the
    # stand -- what S11 unknowingly ran.  Same certificate, three cost stacks:
    # whatever differs is the cost, because nothing else can be.
    "hold_p2": dict(sched="0:2", start_qpos=QP_POSE, seconds=12.0),
    "hold_p1": dict(sched="0:1", start_qpos=QP_POSE, seconds=12.0),
    "hold_p0": dict(sched="0:0", start_qpos=QP_POSE, seconds=12.0),
    # 3b. The discriminator.  hold_p2 falls and `enter` does not, and the two
    # differ in BOTH cost and history, so neither alone says which one matters.
    # This holds the cost fixed against hold_p2 and changes only the history:
    # same QP start, but 6 s under brace_lean (which hold_p1 shows is survivable
    # cold) before brace_reach comes on.  Survives => the reach cost is fine and
    # the cold entry into it is not.  Falls => the cost is the problem.
    "hold_p1p2": dict(sched="0:1,6:2", start_qpos=QP_POSE, seconds=18.0),
    # 4. ENTER: stand for 3 s, then brace, then reach.  brace_lean ramps its
    # targets over 14 s, so a shorter dwell measures a partial ramp and calls it
    # a failure to brace.
    "enter":   dict(sched="0:0,3:1,19:2", start_qpos=None, seconds=27.0),
    # 5. RETURN: hold the certified pose, then release + the three standback
    # rungs + the final stand.  Isolates the exit from the entry.
    "return":  dict(sched="0:2,10:3,24:4,29:5,34:6,39:7", start_qpos=QP_POSE,
                    seconds=49.0),
    # 6. CHAIN: the whole mission from a stand and back to one, no QP seed.
    "chain":   dict(sched="0:0,3:1,19:2,27:3,41:4,46:5,51:6,56:7",
                    start_qpos=None, seconds=66.0),
}


def run(out_csv, ep, seed, threads, strategy=22):
    cmd = [BIN, "--task=" + TASK, "--total_time=%g" % ep["seconds"],
           "--planner_thread=%d" % threads, "--strategy=%d" % strategy,
           "--start_key=stand", "--phase_schedule=" + ep["sched"],
           "--dump_traj=" + out_csv]
    if ep["start_qpos"]:
        cmd.append("--start_qpos=" + ep["start_qpos"])
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    avg = [l for l in p.stdout.splitlines() if "Average cost" in l]
    return dict(returncode=p.returncode, wall=time.time() - t0,
                avg=avg[0].split(":")[-1].strip() if avg else "",
                cmd=" ".join(cmd),
                stderr=p.stderr[-2000:] if p.returncode else "")


def load_traj(path):
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#")]
    r = csv.reader(lines)
    hdr = next(r)
    rows = np.array([[float(v) for v in row] for row in r])
    return {n: i for i, n in enumerate(hdr)}, rows


def cheap_frame(m, d, q, qv, u, target, tau_log):
    """Per-frame scalars, no LP.  The equilibrium region is ~650 linear programs
    and cannot run on 5600 frames; it runs only on the settled frames below.

    qv and u are the LOGGED velocity and control and are not optional.  Every
    actuator on this robot is a <position> servo with kp 40-200, so leaving
    d.ctrl at its default zero does not mean "no command" -- it commands every
    joint to angle 0 and reports the contact forces of a robot being driven to
    the all-zeros pose.  Measured against a correct replay that read the trunk
    load in phase 0 as 434 N when it was 42 N, and the braced arm load as 82 N
    when it was 128 N.  Contact SETS are unaffected (collision detection is a
    function of qpos alone); only the forces were wrong."""
    d.qpos[:] = q
    d.qvel[:] = qv
    d.ctrl[:] = u
    mujoco.mj_forward(m, d)

    hand = cs.point_world(m, d, cs.REACH_BODY, cs.REACH_OFF)
    reach = float(np.linalg.norm(np.asarray(target) - hand))

    tbl = cs.bid(m, "table")
    touching, normal_force = set(), 0.0
    for c in range(d.ncon):
        con = d.contact[c]
        b1, b2 = m.geom_bodyid[con.geom[0]], m.geom_bodyid[con.geom[1]]
        other = b2 if b1 == tbl else (b1 if b2 == tbl else None)
        if other is None:
            continue
        touching.add(other)
        f = np.zeros(6)
        mujoco.mj_contactForce(m, d, c, f)
        normal_force += abs(float(f[0]))
    contacts = sorted(k for k, b in BRACE_BODIES.items()
                      if cs.bid(m, b) in touching)

    tau_max = cs.torque_limits(m)
    ratio = np.abs(tau_log) / tau_max
    peak = float(np.max(ratio))
    # WHICH joint is at the ceiling, not just that one is.  A peak of exactly
    # 1.000 is MuJoCo clipping to forcerange, so the identity of the joint is
    # the whole content of the number: an arm joint at its limit is a brace
    # loaded to spec, a leg joint at its limit is a pose that is falling over.
    hot = int(np.argmax(ratio))

    pel = cs.bid(m, "pelvis")
    R = d.xmat[pel].reshape(3, 3)
    tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
    return dict(reach=reach, contacts=contacts, peak=peak, hot_joint=hot,
                brace_force=normal_force,
                pelvis_z=float(d.xpos[pel][2]), tilt=tilt)


def region_at(m, d, q, contacts):
    """Actuated equilibrium-region margin, over the contacts the pose ACTUALLY
    has.  Scoring against the PLANNED set would credit the planner for contacts
    it never made."""
    d.qpos[:] = q
    d.qvel[:] = 0
    d.qacc[:] = 0
    mujoco.mj_forward(m, d)
    subset = tuple(k for k in ("elbow", "forearm", "palm", "hip")
                   if k in contacts)
    try:
        _, _, marg = st.equilibrium_region(m, d, subset, actuated=True)
        return float(marg), list(subset)
    except Exception:
        return float("nan"), list(subset)


def phase_windows(col, rows):
    """(phase, t_start, t_end) spans from the logged phase column."""
    t, ph = rows[:, col["time"]], rows[:, col["phase"]]
    spans, cur, start = [], ph[0], t[0]
    for i in range(1, len(ph)):
        if ph[i] != cur:
            spans.append((int(cur), float(start), float(t[i - 1])))
            cur, start = ph[i], t[i]
    spans.append((int(cur), float(start), float(t[-1])))
    return spans


def score_episode(m, d, target, col, rows, stride=25):
    """Per-phase summary plus the settled frame of each phase.

    'Settled' = the last 0.5 s of the phase.  A phase is judged on where it
    ENDED, because a ramp that is still moving has not answered anything yet."""
    t = rows[:, col["time"]]
    nq, nv, nu = m.nq, m.nv, m.nu
    qi = [col["qpos%d" % i] for i in range(nq)]
    vi = [col["qvel%d" % i] for i in range(nv)]
    ui = [col["ctrl%d" % i] for i in range(nu)]
    ai = [col["afrc%d" % i] for i in range(nu)]

    series, out_phases = [], []
    for i in range(0, len(rows), stride):
        f = cheap_frame(m, d, rows[i, qi], rows[i, vi], rows[i, ui],
                        target, rows[i, ai])
        f["t"] = float(t[i])
        f["phase"] = int(rows[i, col["phase"]])
        series.append(f)
    reach_t0 = series[0]["reach"]

    for ph, t0, t1 in phase_windows(col, rows):
        win = [f for f in series if t0 <= f["t"] <= t1]
        if not win:
            continue
        settle = [f for f in win if f["t"] >= t1 - 0.5] or win[-1:]
        # the modal contact set over the settled window, so one bouncing frame
        # does not decide what the pose was braced on
        sets = {}
        for f in settle:
            sets["+".join(f["contacts"])] = sets.get("+".join(f["contacts"]), 0) + 1
        modal = max(sets, key=sets.get)
        last = settle[-1]
        idx = int(np.argmin(np.abs(t - last["t"])))
        marg, subset = region_at(m, d, rows[idx, qi], last["contacts"])
        # braced fraction: frames with >= 1 arm contact on the table
        braced = np.mean([1.0 if [c for c in f["contacts"] if c != "hip"] else 0.0
                          for f in win])
        # per-link contact DUTY over the whole phase.  contacts-at-end credits a
        # grazing touch that happened to be present on the last frame; duty says
        # whether the link was actually carrying for the phase.
        duty = {}
        for f in win:
            for c in f["contacts"]:
                duty[c] = duty.get(c, 0) + 1.0 / len(win)
        # Is the command settled or chattering?  Path length of u(t) against its
        # net displacement.  A hold that has converged runs ~1-3x; the S12 chain
        # ran 14-55x, which is the visible tremor in the videos.
        seg = rows[(t >= t0) & (t <= t1)][:, ui]
        path = float(np.abs(np.diff(seg, axis=0)).sum()) if len(seg) > 1 else 0.0
        net = float(np.abs(seg[-1] - seg[0]).sum()) if len(seg) > 1 else 0.0
        out_phases.append(dict(
            phase=ph, name=PHASE_NAMES[ph] if ph < len(PHASE_NAMES) else "?",
            t0=t0, t1=t1,
            contact_duty={k: round(v, 3) for k, v in sorted(duty.items())},
            ctrl_path=path, ctrl_net=net,
            ctrl_churn=path / net if net > 1e-6 else float("nan"),
            # The number that matters for a REACH task is not the residual, it
            # is the residual against where the hand already was at t=0.
            reach_t0=reach_t0, reach_gain=reach_t0 - last["reach"],
            reach_end=last["reach"], reach_min=min(f["reach"] for f in win),
            contacts_end=modal, braced_frac=float(braced),
            brace_force_end=last["brace_force"],
            peak_end=last["peak"], peak_max=max(f["peak"] for f in win),
            hot_joint_end=ACT_NAMES[last["hot_joint"]],
            hot_joint_max=ACT_NAMES[max(win, key=lambda f: f["peak"])["hot_joint"]],
            margin_end=marg, region_subset=subset,
            pelvis_z_end=last["pelvis_z"], pelvis_z_min=min(f["pelvis_z"] for f in win),
            tilt_end=last["tilt"], tilt_max=max(f["tilt"] for f in win),
            fell=bool(min(f["pelvis_z"] for f in win) < 0.55
                      or max(f["tilt"] for f in win) > 45)))
    return series, out_phases


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
    ap.add_argument("--episodes", default="hold,hold_p0,enter,return,chain")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--threads", type=int, default=15)
    ap.add_argument("--outdir", default="runs/mjpc_chain")
    ap.add_argument("--rescore", action="store_true",
                    help="score the CSVs already in --outdir; do not run MJPC")
    ap.add_argument("--strategy", type=int, default=22,
                    help="22 = stock (control), 24 = +Right Foot Lift 0, "
                         "23 = +all three silently-inherited weights filled")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    m, d = cs.load(ik_margin=0)
    ACT_NAMES.extend(
        (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "?")
        .replace("_joint", "").replace("left_", "L_").replace("right_", "R_")
        for i in range(m.nu))
    nid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, "reach_target")
    target = [float(v) for v in
              m.numeric_data[m.numeric_adr[nid]:m.numeric_adr[nid] + 3]]

    print("target %s  basis=%s torso_tau=%.0f  mu=%.2f"
          % (np.round(target, 4), cs.TAU_BASIS,
             cs.CLAMP_RATIO * cs.TAU_ESTOP["torso"], cs.MU))

    ref = {}
    for sub in [("elbow", "forearm"), ("elbow", "forearm", "palm")]:
        ref["+".join(sub)] = qp_reference(target, sub)
        r = ref["+".join(sub)]
        print("  QP %-22s reach %.4f placed=%s peak %.3f margin %.4f"
              % ("+".join(sub), r["reach"], r["placed"], r["peak"], r["margin"]))

    results = {}
    for name in a.episodes.split(","):
        ep = EPISODES[name]
        results[name] = []
        for s in range(a.seeds):
            csv_path = os.path.join(a.outdir, "%s_seed%d.csv" % (name, s))
            print("\n== %s seed %d == %s (%.0f s)"
                  % (name, s, ep["sched"], ep["seconds"]), flush=True)
            if a.rescore:
                # Re-read a CSV that already exists instead of burning another
                # rollout.  The 2026-08-05 runs were scored with a broken
                # cheap_frame (see its docstring); the trajectories themselves
                # are fine, so the fix is a re-score, not a re-run -- which also
                # keeps the numbers comparable to the videos already rendered.
                if not os.path.exists(csv_path):
                    print("   no CSV, skipping")
                    continue
                meta = dict(returncode=0, avg="", wall=0.0)
            else:
                meta = run(csv_path, ep, s, a.threads, a.strategy)
            if meta["returncode"] != 0:
                print("   FAILED:", meta["stderr"][-400:])
                results[name].append(dict(seed=s, failed=True, **meta))
                continue
            col, rows = load_traj(csv_path)
            series, phases = score_episode(m, d, target, col, rows)
            print("   %-13s %-11s %-20s %6s %5s %-12s %7s %6s %5s %4s"
                  % ("phase", "t", "contacts@end", "brace", "peak", "hot joint",
                     "margin", "reach", "pel_z", "fell"))
            for p in phases:
                print("   %-13s %5.1f-%-5.1f %-20s %5.0fN %5.3f %-12s %7.4f "
                      "%6.3f %5.3f %4s"
                      % (p["name"], p["t0"], p["t1"], p["contacts_end"] or "none",
                         p["brace_force_end"], p["peak_end"], p["hot_joint_end"],
                         p["margin_end"], p["reach_end"], p["pelvis_z_end"],
                         "YES" if p["fell"] else "-"))
            results[name].append(dict(seed=s, failed=False, avg=meta["avg"],
                                      wall=meta["wall"], sched=ep["sched"],
                                      phases=phases,
                                      series=series if s == 0 else None))

    out = dict(task=TASK, strategy=a.strategy, target=target, basis=cs.TAU_BASIS,
               torso_tau=cs.CLAMP_RATIO * cs.TAU_ESTOP["torso"],
               site_set=cs.SITE_SET, mu=cs.MU, qp_reference=ref,
               episodes={k: EPISODES[k] for k in results}, runs=results)
    dst = os.path.join(a.outdir, "chain.json")
    json.dump(out, open(dst, "w"), indent=1, default=float)
    print("\nwrote", dst)


if __name__ == "__main__":
    main()
