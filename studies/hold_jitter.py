#!/usr/bin/env python3
"""Jitter measured over the HOLD the robot actually reached, not a fixed slice.

`brace_vs_stand.analyse_one` reads its settled numbers over `t >= 0.6*t_end` --
the last 40% of the rollout. That is a heuristic, not a steady state, and it is
the wrong window for a dispersion metric in both directions:

  * TOO EARLY when the reach is slow. The unbraced arm's settling time at max
    reach is 15.4 s of a 20 s rollout, so the 12 s window opens 3 s before the
    hand stops moving and charges the approach transient to "jitter".
  * TOO LATE when the reach is fast. Both arms are parked by ~4 s at the
    hardware's target, so 8 s of a 16 s hold is thrown away and the estimate
    rests on half the samples it could have.

This finds the hold instead, by TIMESCALE: a 1 s boxcar on the position averages
the jitter away and leaves the trend, and the hold begins at the last downward
crossing of TREND_SPEED by that trend, plus a dwell margin. A rollout that never
establishes one is reported as having no hold rather than being papered over
with a fallback slice that would look like a measurement.

WHAT IT FOUND, and it is not a small correction: only 15 of 64 settled rollouts
(23%) reach a steady hold inside 20 s at all. Where one exists the RMS is
2.1-3.4 mm and barely varies by condition -- against 6-20 mm from the last-40%
window, which at max reach reports 19.7 mm for the unbraced arm. So the old
number is mostly SLOW DRIFT, not jitter: the hand is still creeping outward at
t = 12-20 s, and a dispersion taken about the window mean charges that creep to
jitter. The 19.7-vs-18.0 mm "jitter difference" between the arms at max reach
was never a jitter difference.

Residual drift inside the accepted holds is 0.7-10.3 mm (median 5.8), reported
per rollout as `hold_drift_mm`. Where that approaches the RMS, the window is
still drifting and the number should be read as such.

Two dispersion numbers come out of it, because "jitter" has been used for both
in this study and they differ by ~2x:

  MAD  mean_k || p(k) - p_bar ||          the existing `hand_jitter_mm`
  RMS  sqrt(mean_k || p(k) - p_bar ||^2)  the 3-D standard deviation of the pose

RMS is the one to quote as "variance of the pose at the hold" -- it IS the
square root of the trace of the position covariance. MAD is kept so the new
numbers can be lined up against every number already written down.

Runs a hand-kinematics pass only (no equilibrium LPs), so it is ~1 s per
rollout against ~25 s for a full re-analysis, and merges into an existing
analysis.json in place.

usage:
  hold_jitter.py --run DIR [--json PATH] [--write]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simple_lean as S
import brace_vs_stand as BVS

# THE HAND NEVER STOPS. Measured across nominal, max-reach and hardware-target
# rollouts, the reaching hand's instantaneous speed through the "hold" is
# 16-47 mm/s against a 240-310 mm/s approach peak -- a ratio of about 1:7, not
# the 1:100 an absolute stop threshold would need. A sampling MPC re-plans every
# step and the hand is in continuous small motion; thresholding raw speed finds
# no hold in ANY of the 116 rollouts, which is how this was discovered.
#
# So the split is by TIMESCALE, not by magnitude. A 1 s boxcar on the POSITION
# averages the jitter away and leaves the trend; the trend runs at ~190 mm/s on
# the approach and 1-40 mm/s afterwards, which is separable. The hold begins at
# the last downward crossing of TREND_SPEED, plus a dwell margin.
TREND_WIN = 1.0             # [s] boxcar on position; must exceed the jitter period
TREND_SPEED = 10.0          # [mm/s] drift below this is a hold, not an approach
HOLD_DWELL = 0.50           # [s] margin after the last crossing
HOLD_MIN = 2.0              # [s] shorter than this is not a hold


def trend(t, hp):
    """(t_trend, smoothed position, trend speed in mm/s)."""
    dt = float(t[1] - t[0])
    w = max(3, int(round(TREND_WIN / dt)))
    k = np.ones(w) / w
    sm = np.stack([np.convolve(hp[:, j], k, mode="valid") for j in range(3)], 1)
    ts = t[w // 2:w // 2 + len(sm)]
    sp = np.zeros(len(sm))
    sp[1:] = np.linalg.norm(np.diff(sm, axis=0), axis=1) / dt * 1000.0
    return ts, sm, sp


def hold_window(t, hp):
    """(t0, reason). t0 is the time the hold begins, or None if there is none."""
    ts, sm, sp = trend(t, hp)
    above = np.where(sp > TREND_SPEED)[0]
    if len(above) == 0:
        return float(t[0]), "stationary throughout"
    t0 = float(ts[above[-1]]) + HOLD_DWELL
    if t[-1] - t0 < HOLD_MIN:
        return None, "no hold: trend still >%.0f mm/s at t_end-%.1f s" % (
            TREND_SPEED, HOLD_MIN)
    return t0, "ok"


def measure(path):
    col, rows, meta = S.load_traj(path)
    m, d = S.load()
    qi = [col["qpos%d" % i] for i in range(m.nq)]
    vi = [col["qvel%d" % i] for i in range(m.nv)]
    hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, BVS.HAND_SITE)
    t = rows[:, col["time"]]
    hp = np.zeros((len(rows), 3))
    for k in range(len(rows)):
        d.qpos[:] = rows[k, qi]
        d.qvel[:] = rows[k, vi]
        mujoco.mj_forward(m, d)
        hp[k] = d.site_xpos[hand]

    # Pulses are excluded the same way analyse_one does it, so a disturbance
    # rollout's hold is quiet-state and not a rejection transient.
    keep = np.ones(len(t), bool)
    if "dfx" in col:
        dfx = rows[:, col["dfx"]]
        for k in np.where(np.abs(dfx) > 0)[0]:
            keep &= ~((t >= t[k] - 0.1) & (t <= t[k] + 2.0))

    t0, why = hold_window(t, hp)
    out = {"hold_reason": why}
    if t0 is None:
        out.update(hold_t0=None, hold_len_s=0.0, hold_n=0, hold_drift_mm=None,
                   jitter_hold_rms_mm=None, jitter_hold_mad_mm=None)
        return out
    w = (t >= t0) & keep
    if w.sum() < 10:
        out.update(hold_t0=t0, hold_len_s=0.0, hold_n=int(w.sum()),
                   hold_drift_mm=None, jitter_hold_rms_mm=None,
                   jitter_hold_mad_mm=None,
                   hold_reason="hold survives too few samples after pulse cuts")
        return out
    r = np.linalg.norm(hp[w] - hp[w].mean(axis=0), axis=1)
    # RESIDUAL DRIFT, reported so the window is auditable. The detector admits a
    # trend of up to TREND_SPEED, so a long "hold" can still walk several
    # millimetres; if drift approaches the RMS then the number is a drift
    # measurement wearing a jitter label and should be read as one.
    ts, sm, _ = trend(t, hp)
    ins = ts >= t0
    drift = (float(np.linalg.norm(sm[ins][-1] - sm[ins][0]) * 1000)
             if ins.sum() > 1 else 0.0)
    out.update(hold_t0=t0, hold_len_s=float(t[-1] - t0), hold_n=int(w.sum()),
               hold_drift_mm=drift,
               jitter_hold_rms_mm=float(np.sqrt((r ** 2).mean()) * 1000),
               jitter_hold_mad_mm=float(r.mean() * 1000))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    jpath = a.json or os.path.join(a.run, "analysis.json")
    rows = json.load(open(jpath))
    idx = {r.get("tag"): r for r in rows}

    print("%-28s %8s %8s %9s %9s %9s  %s"
          % ("tag", "hold t0", "len [s]", "RMS [mm]", "MAD [mm]", "drift[mm]",
             "old MAD"))
    n = 0
    for p in sorted(glob.glob(os.path.join(a.run, "*.csv"))):
        tag = os.path.splitext(os.path.basename(p))[0]
        r = idx.get(tag)
        if r is None:
            continue
        try:
            o = measure(p)
        except Exception as e:                                  # noqa: BLE001
            print("%-28s FAILED %s" % (tag, e))
            continue
        r.update(o)
        n += 1
        f = lambda k: ("-" if o.get(k) is None else "%.1f" % o[k])
        print("%-28s %8s %8.1f %9s %9s %9s  %8.1f"
              % (tag, f("hold_t0"), o["hold_len_s"], f("jitter_hold_rms_mm"),
                 f("jitter_hold_mad_mm"), f("hold_drift_mm"),
                 r.get("hand_jitter_mm", float("nan"))))
    if a.write:
        with open(jpath, "w") as f:
            json.dump(rows, f, indent=1)
        print("\nwrote %s (%d rollouts updated)" % (jpath, n))
    else:
        print("\n%d rollouts measured; --write to merge into %s" % (n, jpath))


if __name__ == "__main__":
    main()
