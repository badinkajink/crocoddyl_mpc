#!/usr/bin/env python3
"""How much reach does the BRACE buy, and what does it cost in stability?

The paper's claim is that bracing a link on the table buys reach the robot does
not otherwise have. That claim needs two halves, and this harness measures both
against the SAME task (`Lean Simple H12 Magpie`), the same planner budget and
the same target -- the only thing that differs between the two arms is the three
brace weights:

    stand   Brace Elbow/Forearm/Palm = 0     the reaching hand goes out from a
                                             free-standing posture; all three
                                             brace links are actively held off
                                             the slab by Table Keepout
    brace   Brace Elbow/Forearm     = 300    the left arm seats on the table
            Brace Palm              = 0      first, then the right hand reaches

Three stages.

  nominal   both arms at the study's reach target, several replicates. Gives the
            headline reach gain and the settled stability metrics.
  sweep     the target is walked outward along +x. Reach is scored as the
            SETTLED ERROR to a target the run was actually given (--numeric
            reach_target), so the answer is an envelope -- the x beyond which an
            arm stops arriving -- not a single number.
  disturb   an unmodelled force pulse train on the reaching wrist. The planner
            cannot see it (see --disturb in testspeed.h), so this measures
            rejection, not tracking.

Everything is scored twice over: `simple_lean.score` for reach/contact/torque,
and `simple_stability.Frame` for the static-equilibrium CoM margin and the
largest push the contact set can absorb at the hand. The stability half is the
part that is not obvious -- a brace is only worth having if it widens the region
the CoM may occupy, and that is a measurement, not a diagram.

usage:
  brace_vs_stand.py run     --out DIR [--stage nominal|sweep|disturb|all]
  brace_vs_stand.py analyze --run DIR [--json OUT]
  brace_vs_stand.py plot    --json IN --out DIR
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simple_lean as S
import simple_stability as ST

ARMS = {
    "stand": "Brace Elbow=0,Brace Forearm=0,Brace Palm=0",
    "brace": "Brace Elbow=300,Brace Forearm=300,Brace Palm=0",
}
ARM_LABEL = {"stand": "Standing reach", "brace": "Braced reach"}

# The reaching hand. `right_hand` is the site the task's Reach term uses; the
# disturbance is applied to the body that carries it.
HAND_SITE = "right_hand"
HAND_BODY = "right_wrist_yaw_link"

# Stage parameters. Durations are the settle time the brace needs: at 8 s the
# forearm has not seated yet (duty 0.00), so anything shorter measures the
# approach, not the hold.
NOMINAL_SECONDS = 20.0
NOMINAL_REPS = 4
SWEEP_X = [0.90, 0.98, 1.06, 1.14, 1.22]
SWEEP_SECONDS = 16.0
SWEEP_REPS = 2
DISTURB_N = [30.0, 60.0, 90.0]      # [N] pulse magnitude, +x (away from robot)
DISTURB_T0, DISTURB_DUR, DISTURB_PERIOD, DISTURB_COUNT = 10.0, 0.3, 4.5, 3
DISTURB_SECONDS = 25.0
DISTURB_REPS = 2

# MAX REACH. Scoring reach against a target both postures can hit measures the
# target, not the posture -- standing arrives at the study target to within 6 mm,
# so the reach columns come out equal and say nothing. Here the target is put at
# the FAR EDGE OF THE TABLE (x = 1.60; the slab ends at 1.63), far outside
# anything the robot can touch, so the Reach residual never saturates and the
# optimiser simply stretches as far as the posture allows. What is scored is then
# not "did it arrive" but "how far did it get", which is the question the brace
# is supposed to answer. y and z are held at the study target so the comparison
# moves along one axis only.
MAXREACH_TARGET = (1.60, -0.2348, 1.0982)
MAXREACH_SECONDS = 20.0
MAXREACH_REPS = 4

THREADS = int(os.environ.get("BVS_THREADS", "18"))
STRIDE = 5                           # 100 Hz rows

# Stability is ~200 LPs per frame, so it runs on a coarse grid. 0.5 s resolves
# the settle transient; the margin does not carry features faster than that.
STAB_DT = 0.5
STAB_NDIR = 16


# --------------------------------------------------------------------------- #
# running
# --------------------------------------------------------------------------- #
def launch(out_csv, arm, seconds, numeric="", disturb="", log=None):
    cmd = [S.BIN, "--task=%s" % S.TASK, "--total_time=%g" % seconds,
           "--planner_thread=%d" % THREADS, "--dump_traj=%s" % out_csv,
           "--dump_stride=%d" % STRIDE, "--weights=%s" % ARMS[arm]]
    if numeric:
        cmd.append("--numeric=%s" % numeric)
    if disturb:
        cmd.append("--disturb=%s" % disturb)
    t = time.time()
    with open(log or os.devnull, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    return rc, time.time() - t, " ".join(cmd)


def stage_jobs(stage):
    """(tag, arm, seconds, numeric, disturb) for every run in a stage."""
    jobs = []
    if stage in ("nominal", "all"):
        for arm in ARMS:
            for r in range(NOMINAL_REPS):
                jobs.append(("nominal_%s_r%d" % (arm, r), arm,
                             NOMINAL_SECONDS, "", ""))
    if stage in ("sweep", "all"):
        for x in SWEEP_X:
            for arm in ARMS:
                for r in range(SWEEP_REPS):
                    num = "reach_target=%g|-0.2348|1.0982" % x
                    jobs.append(("sweep_x%03d_%s_r%d" % (round(x * 100), arm, r),
                                 arm, SWEEP_SECONDS, num, ""))
    if stage in ("maxreach", "all"):
        for arm in ARMS:
            for r in range(MAXREACH_REPS):
                num = "reach_target=%g|%g|%g" % MAXREACH_TARGET
                jobs.append(("maxreach_%s_r%d" % (arm, r), arm,
                             MAXREACH_SECONDS, num, ""))
    # The same push train, but delivered at FULL STRETCH rather than at the
    # study target. The static push capacity already differs most there, so this
    # is where the dynamic rejection difference should be largest -- and it is
    # the condition a braced reach actually exists to survive.
    if stage in ("disturb_max",):
        for f in DISTURB_N:
            for arm in ARMS:
                for r in range(DISTURB_REPS):
                    dis = ("body=%s,force=%g|0|0,t0=%g,dur=%g,period=%g,n=%d"
                           % (HAND_BODY, f, DISTURB_T0, DISTURB_DUR,
                              DISTURB_PERIOD, DISTURB_COUNT))
                    num = "reach_target=%g|%g|%g" % MAXREACH_TARGET
                    jobs.append(("disturbmax_f%03d_%s_r%d" % (round(f), arm, r),
                                 arm, DISTURB_SECONDS, num, dis))
    if stage in ("disturb", "all"):
        for f in DISTURB_N:
            for arm in ARMS:
                for r in range(DISTURB_REPS):
                    dis = ("body=%s,force=%g|0|0,t0=%g,dur=%g,period=%g,n=%d"
                           % (HAND_BODY, f, DISTURB_T0, DISTURB_DUR,
                              DISTURB_PERIOD, DISTURB_COUNT))
                    jobs.append(("disturb_f%03d_%s_r%d" % (round(f), arm, r),
                                 arm, DISTURB_SECONDS, "", dis))
    return jobs


def cmd_run(a):
    os.makedirs(a.out, exist_ok=True)
    jobs = stage_jobs(a.stage)
    print("%d runs, ~%.0f s of sim at %d threads"
          % (len(jobs), sum(j[2] for j in jobs), THREADS), flush=True)
    manifest = []
    for i, (tag, arm, secs, numeric, disturb) in enumerate(jobs):
        csv_path = os.path.join(a.out, tag + ".csv")
        if os.path.exists(csv_path) and not a.force:
            print("[%2d/%d] %-28s cached" % (i + 1, len(jobs), tag), flush=True)
            manifest.append(dict(tag=tag, arm=arm, seconds=secs,
                                 numeric=numeric, disturb=disturb))
            continue
        rc, wall, cmd = launch(csv_path, arm, secs, numeric, disturb,
                               log=os.path.join(a.out, tag + ".log"))
        print("[%2d/%d] %-28s rc=%d  %.0f s wall" % (i + 1, len(jobs), tag, rc,
                                                     wall), flush=True)
        manifest.append(dict(tag=tag, arm=arm, seconds=secs, numeric=numeric,
                             disturb=disturb, rc=rc, wall=wall, cmd=cmd))
    with open(os.path.join(a.out, "manifest_%s.json" % a.stage), "w") as f:
        json.dump(manifest, f, indent=1)
    print("wrote", os.path.join(a.out, "manifest_%s.json" % a.stage))


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def read_csv(path):
    col, rows, meta = S.load_traj(path)
    return col, rows, meta


def analyse_one(path, stab=True):
    """Everything one rollout contributes, reach + stability + disturbance."""
    col, rows, meta = read_csv(path)
    if len(rows) < 10:
        return dict(path=os.path.basename(path), error="too short")
    base = S.score(path)
    if isinstance(base, dict) and "error" in base:
        return dict(path=os.path.basename(path), error=base["error"])
    base = base[0]

    m, d = S.load()
    nq, nv, nu = m.nq, m.nv, m.nu
    qi = [col["qpos%d" % i] for i in range(nq)]
    vi = [col["qvel%d" % i] for i in range(nv)]
    ui = [col["ctrl%d" % i] for i in range(nu)]
    hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, HAND_SITE)
    feet_b = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
              for b in ("left_ankle_roll_link", "right_ankle_roll_link")}
    ankles = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
              for b in ("left_ankle_roll_link", "right_ankle_roll_link")]
    hbody = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
    n = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, "reach_target")
    target = S.target_from_meta(
        meta, m.numeric_data[m.numeric_adr[n]:m.numeric_adr[n] + 3])

    t = rows[:, col["time"]]
    has_d = "dfx" in col
    dfx = rows[:, col["dfx"]] if has_d else np.zeros(len(rows))

    # dense pass: hand position and reach error at every dumped row
    hp = np.zeros((len(rows), 3))
    # FUNCTIONAL REACH: horizontal hand-to-ankle-midpoint distance. World hand x
    # is not reach -- it moves when the base does, so a robot that shuffles
    # forward scores a reach it did not extend. Measured from the feet, the
    # number is the posture's, which is what is being compared.
    fr_ = np.zeros(len(rows))
    for k in range(len(rows)):
        d.qpos[:] = rows[k, qi]
        d.qvel[:] = rows[k, vi]
        mujoco.mj_forward(m, d)
        hp[k] = d.site_xpos[hand]
        mid = 0.5 * (d.xpos[ankles[0]] + d.xpos[ankles[1]])
        fr_[k] = np.linalg.norm(hp[k][:2] - mid[:2])
    err = np.linalg.norm(hp - target, axis=1)

    out = dict(path=os.path.basename(path), target=target.tolist(),
               t_end=float(t[-1]), fell=base["fell"],
               achieved_mode=base["achieved_mode"], duty=base["duty"],
               reach_base=base["reach_base"], reach_gain=base["reach_gain"],
               reach_settled=base["reach_settled"],
               peak_tau_ratio=base["peak_tau_ratio"],
               saturated_frac=base["saturated_frac"],
               churn=base["churn"], cost_settled=base["cost_settled"],
               pelvis_z_settled=base["pelvis_z_settled"])

    # ---- precision, over a quiet window that excludes every pulse ---------- #
    quiet = np.ones(len(rows), dtype=bool)
    if has_d:
        # exclude each pulse and the 2 s of transient after it
        for k in np.where(np.abs(dfx) > 0)[0]:
            quiet &= ~((t >= t[k] - 0.1) & (t <= t[k] + 2.0))
    settle = quiet & (t >= 0.6 * t[-1])
    if settle.sum() < 5:
        settle = t >= 0.75 * t[-1]
    out["precision_rms_mm"] = float(np.linalg.norm(
        hp[settle] - hp[settle].mean(axis=0), axis=1).std() * 1000)
    out["hand_jitter_mm"] = float(np.linalg.norm(
        hp[settle] - hp[settle].mean(axis=0), axis=1).mean() * 1000)
    out["reach_err_settled"] = float(err[settle].mean())
    out["reach_err_std_mm"] = float(err[settle].std() * 1000)
    out["hand_x_settled"] = float(hp[settle, 0].mean())
    out["hand_xyz_settled"] = hp[settle].mean(axis=0).tolist()
    out["func_reach_settled"] = float(fr_[settle].mean())
    out["func_reach_max"] = float(fr_.max())

    # ---- settling time: first time the error stays within 2 cm of final ---- #
    final = err[settle].mean()
    band = 0.02
    ok = np.abs(err - final) <= band
    ts = float("nan")
    for k in range(len(ok)):
        if ok[k:].all():
            ts = float(t[k])
            break
    out["settle_time_s"] = ts

    # ---- disturbance response --------------------------------------------- #
    if has_d and np.abs(dfx).max() > 0:
        out["pulses"] = pulse_response(t, hp, dfx)

    # ---- stability on a coarse grid --------------------------------------- #
    if stab:
        step = max(1, int(round(STAB_DT / (t[1] - t[0]))))
        idx = list(range(0, len(rows), step))
        mc, ma, tt, ncon = [], [], [], []
        bl, bodyload, nframe = [], {}, [0]
        for k in idx:
            d.qpos[:] = rows[k, qi]
            d.qvel[:] = rows[k, vi]
            d.ctrl[:] = rows[k, ui]
            mujoco.mj_forward(m, d)
            fr = ST.Frame(m, d)
            tt.append(float(t[k]))
            ncon.append(fr.n)
            if not fr.ok:
                # every series stays index-aligned with tt, or `sel` reads the
                # wrong frames out of the short one
                mc.append(np.nan); ma.append(np.nan); bl.append(0.0)
                continue
            mc.append(fr.equilibrium_region(actuated=False, ndir=STAB_NDIR)[2])
            ma.append(fr.equilibrium_region(actuated=True, ndir=STAB_NDIR)[2])
            # BRACE LOAD, measured. This task is contact-implicit: the planner
            # may put a link on the table whether or not a brace cost asked it
            # to, and it does -- a zero-brace-cost rollout was found carrying
            # 62 N through the left gripper. So a run is classified by the load
            # its contacts actually carry, never by the weights it was given.
            per = {}
            for ci in range(d.ncon):
                c = d.contact[ci]
                if c.dist > 0:
                    continue
                b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
                if fr.inside[b1] == fr.inside[b2]:
                    continue
                rb = b2 if fr.inside[b2] else b1
                if rb in feet_b:
                    continue
                f6 = np.zeros(6)
                mujoco.mj_contactForce(m, d, ci, f6)
                nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, rb)
                per[nm] = per.get(nm, 0.0) + abs(float(f6[0]))
            bl.append(sum(per.values()))
            for nm, v in per.items():
                bodyload[nm] = bodyload.get(nm, 0.0) + v
            nframe[0] += 1
        out["stab_t"] = tt
        out["margin_contact"] = mc
        out["margin_actuated"] = ma
        out["ncontacts"] = ncon
        sel = [i for i, x in enumerate(tt) if x >= 0.6 * t[-1]]
        out["margin_contact_settled"] = float(np.nanmean([mc[i] for i in sel]))
        out["margin_actuated_settled"] = float(np.nanmean([ma[i] for i in sel]))
        out["brace_load_t"] = bl
        out["brace_load_N"] = float(np.mean([bl[i] for i in sel])) if sel \
            else float("nan")
        out["brace_load_bodies"] = {k: v / max(1, nframe[0])
                                    for k, v in sorted(bodyload.items(),
                                                       key=lambda kv: -kv[1])}
        # "Did this run brace?" is a question about newtons. 15 N is ~2% of body
        # weight -- above incidental grazing, below anything load-bearing.
        out["braced_by_load"] = bool(out["brace_load_N"] >= 15.0)

        # Largest push the contact set absorbs at the hand. Averaged over
        # several settled frames, not read off one: at a single frame this swung
        # 33 -> 97 N between two replicates of the SAME standing condition,
        # because the posture keeps breathing and the LP is exquisitely
        # sensitive to exactly where the feet are. Three frames is enough to
        # stop the metric reporting posture noise as a difference between arms.
        picks = [idx[i] for i in (sel[len(sel) // 4], sel[len(sel) // 2],
                                  sel[3 * len(sel) // 4])] if len(sel) >= 4 \
            else ([idx[sel[len(sel) // 2]]] if sel else [idx[-1]])
        pmin, ppx, last = [], [], None
        for k in picks:
            d.qpos[:] = rows[k, qi]; d.qvel[:] = rows[k, vi]
            d.ctrl[:] = rows[k, ui]
            mujoco.mj_forward(m, d)
            fr = ST.Frame(m, d)
            if not fr.ok:
                continue
            angs, push = fr.max_push_at(d.site_xpos[hand], ndir=16,
                                        actuated=True)
            pmin.append(float(np.nanmin(push)))
            ppx.append(float(push[0]))
            last = (angs, push)
        if pmin:
            out["push_min_N"] = float(np.mean(pmin))
            out["push_plus_x_N"] = float(np.mean(ppx))
            out["push_min_spread_N"] = float(max(pmin) - min(pmin))
            out["push_angles"] = last[0].tolist()
            out["push_newtons"] = last[1].tolist()
        k = picks[len(picks) // 2]
        d.qpos[:] = rows[k, qi]; d.qvel[:] = rows[k, vi]; d.ctrl[:] = rows[k, ui]
        mujoco.mj_forward(m, d)
        fr = ST.Frame(m, d)
        if fr.ok:
            # The region itself, at this frame, for the geometric figure: the
            # polygon IS the claim ("the brace extends the support area"), so it
            # is worth carrying rather than re-deriving at plot time.
            pts, c0, _ = fr.equilibrium_region(actuated=True, ndir=36)
            out["region_xy"] = pts.tolist()
            out["region_com"] = c0.tolist()
            out["region_contacts"] = [
                dict(p=c["p"].tolist(),
                     body=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                            c["body"]))
                for c in fr.cons]
            out["region_hand"] = d.site_xpos[hand].tolist()
            out["region_t"] = float(t[k])
    return out


def pulse_response(t, hp, dfx):
    """Per-pulse peak hand deviation and recovery time.

    Deviation is measured against the mean hand position over the 1 s BEFORE the
    pulse, not against the target: the two arms settle to different places, and
    the question here is how far the push moves the hand from wherever it was."""
    on = np.abs(dfx) > 0
    edges = np.where(np.diff(on.astype(int)) == 1)[0] + 1
    out = []
    for e in edges:
        pre = (t >= t[e] - 1.0) & (t < t[e])
        if pre.sum() < 3:
            continue
        p0 = hp[pre].mean(axis=0)
        band = max(0.005, 2.0 * np.linalg.norm(hp[pre] - p0, axis=1).max())
        end = t[e] + 4.0
        win = (t >= t[e]) & (t <= end)
        if win.sum() < 5:
            continue
        dev = np.linalg.norm(hp[win] - p0, axis=1)
        tw = t[win]
        peak = float(dev.max())
        t_peak = float(tw[int(np.argmax(dev))])
        # recovery: first time after the peak that it stays inside the band
        rec = float("nan")
        back = dev <= band
        for k in range(int(np.argmax(dev)), len(dev)):
            if back[k:].all():
                rec = float(tw[k] - t[e])
                break
        out.append(dict(t0=float(t[e]), peak_dev_mm=peak * 1000,
                        t_peak=t_peak, recover_s=rec, band_mm=band * 1000,
                        residual_mm=float(dev[-1] * 1000)))
    return out


def cmd_analyze(a):
    paths = sorted(glob.glob(os.path.join(a.run, "*.csv")))
    out_path = a.json or os.path.join(a.run, "analysis.json")
    res, done = [], set()
    # Scoring a rollout costs ~25 s, nearly all of it in the equilibrium-region
    # LPs, so re-analysing 48 runs to add 12 is 20 minutes of recomputing
    # identical numbers. --append keeps what is already scored and does only the
    # new files.
    if a.append and os.path.exists(out_path):
        res = json.load(open(out_path))
        done = {r.get("tag") for r in res}
        print("appending to %d existing results" % len(res), flush=True)
    print("%d rollouts (%d to score)"
          % (len(paths), sum(1 for p in paths if os.path.splitext(
              os.path.basename(p))[0] not in done)), flush=True)
    for i, p in enumerate(paths):
        name = os.path.splitext(os.path.basename(p))[0]
        if name in done:
            continue
        try:
            o = analyse_one(p, stab=not a.no_stab)
        except Exception as e:                                  # noqa: BLE001
            print("  [%2d] %-28s FAILED %s" % (i + 1, name, e), flush=True)
            continue
        o["tag"] = name
        parts = name.split("_")
        o["stage"] = parts[0]
        o["arm"] = "brace" if "_brace" in name else "stand"
        res.append(o)
        print("  [%2d] %-28s gain=%+.3f  margin=%.3f/%.3f  jitter=%.1fmm  fell=%s"
              % (i + 1, name, o.get("reach_gain", float("nan")),
                 o.get("margin_contact_settled", float("nan")),
                 o.get("margin_actuated_settled", float("nan")),
                 o.get("hand_jitter_mm", float("nan")), o.get("fell")),
              flush=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote %s (%d results)" % (out_path, len(res)))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--stage", default="all",
                   choices=["nominal", "sweep", "maxreach", "disturb",
                            "disturb_max", "all"])
    r.add_argument("--force", action="store_true")
    an = sub.add_parser("analyze")
    an.add_argument("--run", required=True)
    an.add_argument("--json", default="")
    an.add_argument("--no-stab", action="store_true")
    an.add_argument("--append", action="store_true",
                    help="keep results already in the json; score only new CSVs")
    p = sub.add_parser("plot")
    p.add_argument("--json", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "analyze":
        cmd_analyze(a)
    else:
        import bvs_plots
        bvs_plots.make_all(a.json, a.out)


if __name__ == "__main__":
    main()
