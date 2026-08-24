#!/usr/bin/env python3
"""The sim run at the target the HARDWARE was given, beside the hardware's trace.

The hardware runs used a reach target that is none of this study's sim
conditions, so none of the existing figures answer "does the sim land where the
robot landed". This does: both arms are given `brace_vs_stand.REALPOSE_TARGET`
and the reaching hand's position is plotted the way Allen plots it -- one line
per axis, faint per-replicate, thick mean, dashed target -- so the sim panels
and `target_position.png` can sit side by side and be read as one figure.

THREE THINGS THIS COMPARISON CANNOT SETTLE, all of which push the same way
(they make the sim look better than it should) and all of which belong in the
caption:

  1. THE TARGET IS READ OFF A PNG. `REALPOSE_TARGET` was recovered by pixel from
     the published trace, not from Allen's logs. About +-3 mm per axis. See the
     constant's comment for how the dashed lines were separated from the curves.
  2. THE TIP MAY NOT BE THE SAME POINT. This plots the `right_hand` site, which
     sits 0.17 m out along the wrist's local x -- through the gripper, so
     tip-like. Whether it is Allen's tip to within a centimetre is unknown, and
     a constant frame offset would move the sim error by its full magnitude
     without changing the shape of any curve.
  3. THE SIM ROBOT IS NOT CARRYING WHAT THE REAL ONE CARRIES. The hand-mass
     parity gap is a known open item in this study; a lighter hand settles
     tighter.

So read the ERROR STRUCTURE, not the error magnitude: hardware misses almost
entirely in +y (58 of its 62 mm), and whether the sim misses the same way is a
statement about the task, not about the frame.

usage:
  realpose_compare.py --run DIR [--out DIR]
"""
import argparse
import os
import sys

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simple_lean as S
import brace_vs_stand as BVS
from bvs_plots import C, INK, INK2, COL, TEXT, _ms, _save, C_REAL

# Allen's axis colours, reused EXACTLY rather than re-derived from the study
# palette. These panels are meant to be placed next to `target_position.png`,
# and two figures side by side that encode x/y/z in different hues are worse
# than one figure using colours chosen by someone else.
AXC = {"x": "#a87c05", "y": "#1f6b31", "z": "#1a3a8a"}
ARMS = ["stand", "brace"]
# Matches `bvs_plots.REAL_ARM_LABEL`; the two figures share a caption.
ARM_LABEL = {"stand": "No brace cost", "brace": "Brace Encouraged"}


def hand_series(path):
    """(t, hand xyz over time, target) for one rollout."""
    col, rows, meta = S.load_traj(path)
    m, d = S.load()
    qi = [col["qpos%d" % i] for i in range(m.nq)]
    vi = [col["qvel%d" % i] for i in range(m.nv)]
    hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, BVS.HAND_SITE)
    n = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, "reach_target")
    tgt = S.target_from_meta(
        meta, m.numeric_data[m.numeric_adr[n]:m.numeric_adr[n] + 3])
    hp = np.zeros((len(rows), 3))
    for k in range(len(rows)):
        d.qpos[:] = rows[k, qi]
        d.qvel[:] = rows[k, vi]
        mujoco.mj_forward(m, d)
        hp[k] = d.site_xpos[hand]
    return rows[:, col["time"]], hp, np.asarray(tgt, float)


def settled(t, hp):
    """The same window every other number in this study is read over."""
    w = t >= 0.6 * t[-1]
    return hp[w].mean(axis=0)


def load(run):
    """Clean realpose rollouts per arm, in analysis.json's judgement.

    NOT every realpose_*.csv. 12 of the 16 commanded-brace rollouts at this
    target finish with the trunk on the slab, and a precision figure that
    averaged those would be reporting how accurately the robot can lie down.
    The classification lives in analysis.json (contact forces per body), so the
    tag list is taken from there rather than re-derived here -- one definition
    of "clean" for the whole study."""
    import json
    from bvs_plots import by
    jp = os.path.join(run, "analysis.json")
    if not os.path.exists(jp):
        raise SystemExit("need %s -- run `brace_vs_stand.py analyze` first" % jp)
    rows = json.load(open(jp))
    keep = {r["tag"] for r in by(rows, "realpose", clean=True)}
    dropped = sum(1 for r in rows if r.get("stage") == "realpose") - len(keep)
    print("realpose: %d clean rollouts, %d dropped (trunk rest / fell)"
          % (len(keep), dropped))
    out = {}
    for arm in ARMS:
        runs = []
        for tag in sorted(t for t in keep if t.endswith("_r%s" % t.rsplit("_r", 1)[1])
                          and "_%s_" % arm in t):
            p = os.path.join(run, tag + ".csv")
            if os.path.exists(p):
                runs.append(hand_series(p))
        out[arm] = runs
    return out


def fig_realpose(data, out):
    fig = plt.figure(figsize=(TEXT, 2.9), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.92])
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]

    lo, hi = [], []
    for ai, arm in enumerate(ARMS):
        ax = axs[ai]
        runs = data[arm]
        if not runs:
            continue
        tgt = runs[0][2]
        for j, k in enumerate("xyz"):
            for t, hp, _ in runs:                       # faint replicates
                ax.plot(t, hp[:, j], color=AXC[k], lw=0.7, alpha=0.30,
                        zorder=2)
            n = min(len(t) for t, _, _ in runs)
            mu = np.mean([hp[:n, j] for _, hp, _ in runs], axis=0)
            ax.plot(runs[0][0][:n], mu, color=AXC[k], lw=2.0, zorder=4,
                    label=k if ai == 0 else None)
            ax.axhline(tgt[j], color=AXC[k], lw=1.4, ls=(0, (4, 2.5)),
                       zorder=3)
            lo.append(min(mu.min(), tgt[j]))
            hi.append(max(mu.max(), tgt[j]))
        # Mean of the per-replicate misses, NOT the miss of the mean
        # trajectory. The two differ here by a factor of two (8 mm vs 4 mm for
        # the unbraced arm) because the replicates miss in different
        # directions and cancel; the bar panel and the table both report the
        # per-replicate mean, so the title has to as well or the figure
        # disagrees with itself.
        e, ehr = _ms([np.linalg.norm(settled(t, hp) - tgt) * 1000
                      for t, hp, _ in runs])
        ax.set_title("%s\nsettled miss %.0f ± %.0f mm  (n=%d)"
                     % (ARM_LABEL[arm], e, ehr, len(runs)), fontsize=8.5)
        ax.set_xlabel("Time  [s]")
        if ai == 0:
            ax.set_ylabel("Tip position  [m]")
    for ax in axs[:2]:
        ax.set_ylim(min(lo) - 0.10, max(hi) + 0.14)
    h = [plt.Line2D([], [], color=AXC[k], lw=2.0, label="$%s$" % k)
         for k in "xyz"]
    h.append(plt.Line2D([], [], color=INK2, lw=1.4, ls=(0, (4, 2.5)),
                        label="target"))
    # Mid-height is the only clear band: y sits at -0.16, x climbs to 1.0 and
    # z is pinned at 1.17 from the first frame, so anything anchored to a
    # corner lands on a curve.
    axs[0].legend(handles=h, loc="center left", ncol=2, fontsize=7.2,
                  columnspacing=1.0, handlelength=1.6)

    # (c) the number the comparison exists for, per axis and as a norm.
    ax = axs[2]
    hw_mu, hw_sd = BVS.REALPOSE_HW_ERR_MM
    bars = []
    for arm in ARMS:
        runs = data[arm]
        if not runs:
            continue
        tgt = runs[0][2]
        v = [np.linalg.norm(settled(t, hp) - tgt) * 1000 for t, hp, _ in runs]
        mu, hr = _ms(v)
        bars.append((ARM_LABEL[arm], C[arm], mu, hr))
    bars.append(("Hardware", C_REAL, hw_mu, hw_sd))
    for i, (lab, col, mu, hr) in enumerate(bars):
        ax.bar(i, mu, width=0.6, color=col, zorder=3, edgecolor="white",
               linewidth=0.9)
        if hr > 0:
            ax.errorbar(i, mu, yerr=hr, color=INK2, lw=1.0, capsize=3,
                        zorder=4)
        ax.annotate("%.0f" % mu, xy=(i, mu + hr), xytext=(0, 2.5),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=7.4, color=INK, zorder=5)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0].replace(" ", "\n", 1) for b in bars],
                       fontsize=6.6)
    ax.set_ylabel("Settled miss  [mm]")
    ax.set_title("Precision at the\nhardware's target", fontsize=8.5)
    ax.margins(y=0.24)
    ax.grid(axis="x", visible=False)
    _save(fig, out, "fig_realpose_compare")


def table(data, out):
    hw_mu, hw_sd = BVS.REALPOSE_HW_ERR_MM
    tgt = None
    txt = ["target (read from target_position.png): "
           "(%.3f, %.3f, %.3f) m" % BVS.REALPOSE_TARGET, "",
           "%-18s %5s %9s %9s %9s %10s"
           % ("", "n", "dx [mm]", "dy [mm]", "dz [mm]", "|d| [mm]")]
    for arm in ARMS:
        runs = data[arm]
        if not runs:
            continue
        tgt = runs[0][2]
        dv = np.array([settled(t, hp) - tgt for t, hp, _ in runs]) * 1000
        nrm = [_ms([np.linalg.norm(d) for d in dv])]
        txt.append("%-18s %5d %9s %9s %9s %10s"
                   % (ARM_LABEL[arm], len(runs),
                      "%+.0f" % dv[:, 0].mean(), "%+.0f" % dv[:, 1].mean(),
                      "%+.0f" % dv[:, 2].mean(),
                      "%.0f ± %.0f" % nrm[0]))
    # Hardware per-axis comes from the same pixel read as the target, so it is
    # reported as one row and flagged, not merged into the sim rows.
    txt.append("%-18s %5s %9s %9s %9s %10s"
               % ("Hardware", "?", "-6", "+58", "+22",
                  "%.0f ± %.0f" % (hw_mu, hw_sd)))
    txt += ["", "Hardware per-axis is read off the published trace (mean curve",
            "vs dashed target); its |d| is Allen's reported 53 ± 9 mm. The sim",
            "rows are settled means over t >= 0.6*t_end, the study's window."]
    body = "\n".join(txt)
    with open(os.path.join(out, "table_realpose.txt"), "w") as f:
        f.write(body + "\n")
    print("\n" + body + "\n")
    print("  wrote table_realpose.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = load(a.run)
    if not any(data.values()):
        raise SystemExit("no realpose_*.csv in %s -- run the stage first" % a.run)
    fig_realpose(data, a.out)
    table(data, a.out)


if __name__ == "__main__":
    main()
