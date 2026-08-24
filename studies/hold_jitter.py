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
from scipy.signal import butter, filtfilt

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

# ALLEN'S DEFINITION (received 2026-08-23), implemented verbatim where the sim
# allows it:
#
#   "Hand jitter reports the high-pass RMS of hand motion during the braced
#    hold. The window is ~60 s during the Brace phase, between 3 s after brace
#    starting and 1 s before recovery starting.
#    jitter = sqrt( mean|| x(t) - xbar(t) ||^2 )  [mm]
#    RMS tremor above ~1 Hz is only counted."
#
# `xbar(t)` is a FUNCTION OF t, so `x - xbar` is the high-passed signal and the
# formula is just the RMS of it. Implemented as a 2nd-order zero-phase
# Butterworth at HP_CORNER, run over the WHOLE trace and only then windowed, so
# the filter's edge transient never lands inside the measurement.
#
# This is a strictly better metric for a sim/hardware comparison than either of
# the two above it, because it is DRIFT-IMMUNE BY CONSTRUCTION: the slow outward
# creep that made `hand_jitter_mm` unusable (see the module docstring) lives
# below 1 Hz and is removed, so a rollout no longer has to reach a steady hold
# to be measurable. That is why the window below is phase-based like Allen's
# rather than gated on TREND_SPEED.
#
# TWO DEVIATIONS, both forced by rollout length and both recorded on the figure:
#   * Our rollouts are 20 s against his ~60 s. High-pass RMS is stationary, so a
#     shorter window is a NOISIER estimate of the same quantity, not a biased
#     one -- but the spread is wider than his.
#   * There is no recovery phase in these rollouts and no brace command onset to
#     anchor to, so "brace start" is taken as the hand's ARRIVAL (last downward
#     crossing of ARRIVE_TOL toward its final position). `hp_margin_s` records
#     how much of the 3 s post-brace margin actually fitted.
HP_CORNER = 1.0             # [Hz] Allen's "above ~1 Hz"
HP_ORDER = 2                # zero-phase (filtfilt) => effective 4th order
ARRIVE_TOL = 0.050          # [m] within this of the final pose = arrived
BRACE_MARGIN = 3.0          # [s] Allen: 3 s after brace starting
END_MARGIN = 1.0            # [s] Allen: 1 s before recovery starting
HP_MIN = 2.0                # [s] shorter than this is not reportable


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


def arrival(t, hp):
    """Time the hand reaches its final pose, the sim's stand-in for brace onset.

    Last downward crossing of ARRIVE_TOL, not the first: a rollout that touches
    the tolerance early and then wanders back out has not arrived."""
    final = hp[t >= t[-1] - 1.0].mean(axis=0)
    far = np.where(np.linalg.norm(hp - final, axis=1) > ARRIVE_TOL)[0]
    return float(t[0]) if len(far) == 0 else float(t[far[-1]])


def hp_jitter(t, hp, keep):
    """Allen's metric: RMS of the >HP_CORNER content over the braced hold."""
    dt = float(t[1] - t[0])
    nyq = 0.5 / dt
    if HP_CORNER >= nyq:
        return {"jitter_hp_rms_mm": None, "hp_reason": "corner above Nyquist"}
    b, a = butter(HP_ORDER, HP_CORNER / nyq, btype="highpass")
    # Filter the FULL trace, then window -- filtfilt's edge transient is several
    # cycles long at 1 Hz and would otherwise sit inside a 2 s window.
    xhp = filtfilt(b, a, hp, axis=0)

    t_arr = arrival(t, hp)
    t0 = t_arr + BRACE_MARGIN
    t1 = t[-1] - END_MARGIN
    # Keep Allen's margin if it fits; otherwise take what is left after arrival
    # and say by how much we fell short, rather than silently using a different
    # window from his.
    short = 0.0
    if t1 - t0 < HP_MIN:
        short = t0 - t_arr
        t0 = t_arr
        short -= max(0.0, min(short, t1 - t_arr - HP_MIN))
    w = (t >= t0) & (t <= t1) & keep
    if w.sum() < 10 or t1 - t0 < HP_MIN:
        return {"jitter_hp_rms_mm": None, "hp_t0": t0, "hp_win_s": max(0.0, t1 - t0),
                "hp_arrive_s": t_arr, "hp_margin_s": BRACE_MARGIN - short,
                "hp_reason": "window shorter than %.1f s (arrived %.1f s of %.1f s)"
                             % (HP_MIN, t_arr, t[-1])}
    r = np.linalg.norm(xhp[w], axis=1)
    return {"jitter_hp_rms_mm": float(np.sqrt((r ** 2).mean()) * 1000),
            "hp_t0": t0, "hp_win_s": float(t1 - t0), "hp_arrive_s": t_arr,
            "hp_margin_s": BRACE_MARGIN - short, "hp_n": int(w.sum()),
            "hp_reason": "ok" if short == 0 else
                         "ok, post-brace margin trimmed %.1f s -> %.1f s"
                         % (BRACE_MARGIN, BRACE_MARGIN - short)}


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
    out.update(hp_jitter(t, hp, keep))
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

    print("%-28s %8s %8s %9s %9s %9s %9s %8s  %s"
          % ("tag", "hold t0", "len [s]", "RMS [mm]", "MAD [mm]", "drift[mm]",
             "HP RMS", "HP win", "old MAD"))
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
        print("%-28s %8s %8.1f %9s %9s %9s %9s %8s  %8.1f"
              % (tag, f("hold_t0"), o["hold_len_s"], f("jitter_hold_rms_mm"),
                 f("jitter_hold_mad_mm"), f("hold_drift_mm"),
                 f("jitter_hp_rms_mm"), f("hp_win_s"),
                 r.get("hand_jitter_mm", float("nan"))))
    if a.write:
        with open(jpath, "w") as f:
            json.dump(rows, f, indent=1)
        print("\nwrote %s (%d rollouts updated)" % (jpath, n))
    else:
        print("\n%d rollouts measured; --write to merge into %s" % (n, jpath))


if __name__ == "__main__":
    main()
