#!/usr/bin/env python3
"""MJPC against CMPC, on the same figures, from the same scorer.

`bvs_plots.py` draws one planner's two arms. This draws two planners' two arms,
on the same axes where an overlay is readable and side by side where it is not,
from the two `analysis.json` files `brace_vs_stand.py analyze` produced -- one
per planner. Nothing here re-scores anything: if a number differs from the
single-planner figure it is because the underlying rollout differs, not because
a second code path computed it.

THE ENCODING, and why it is not four colours. The study's viz rule is one
categorical hue per series and a LINESTYLE for a second dimension inside a
series. There are two dimensions here -- arm and planner -- and only one of them
is categorical in the reader's head: `no brace cost` vs `commanded brace` is the
comparison every other figure in the paper makes, so it keeps the two hues.
Planner rides as linestyle + marker on lines, and as a hatch on bars:

    hue        arm       blue = no brace cost, orange = commanded brace
    linestyle  planner   solid + filled circle = MJPC, dashed + open square = CMPC
    hatch      planner   solid fill = MJPC, hatched = CMPC

Four hues would make the arm comparison -- the paper's actual claim -- something
the reader has to decode from a legend, in exchange for making a baseline
comparison marginally easier. That trade is the wrong way round.

WHAT IS AND IS NOT COMPARABLE. Both planners were run on the same plant, at the
same targets, for the same durations, scored by the same code. They differ in:

  * START POSE. MJPC begins at `home`, CMPC at `stand` (knees 0.55 rad). Each is
    its own pipeline's start. `reach_gain` is scored against a fixed model-side
    baseline for both, and `func_reach_settled` -- the headline -- is measured
    hand-to-ankle-midpoint and references neither.
  * WHERE THE SPREAD COMES FROM. MJPC replicates differ because it samples;
    CMPC replicates differ because each starts from a seeded 3 mrad / 3 mm
    perturbation. Both are spreads over repeated attempts; they are not the
    same random variable, and the caption says so.
  * WHAT "STAND" MEANS. MJPC's is a weight setting the planner may ignore --
    and does, discovering the table on runs it was not asked to brace on.
    CMPC's is a contact schedule it cannot violate. That difference is a
    RESULT, not a confound: it is visible in the load columns.

usage:
  mjpc_vs_cmpc.py --mjpc A/analysis.json --cmpc B/analysis.json --out DIR
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import bvs_plots as BP
from bvs_plots import C, LABEL, INK, INK2, GRID, ORDER, COL, TEXT, _ms

PLANNERS = ["mjpc", "cmpc"]
PNAME = {"mjpc": "MJPC (sampling)", "cmpc": "CMPC (gradient)"}
PSHORT = {"mjpc": "MJPC", "cmpc": "CMPC"}
PLINE = {"mjpc": "-", "cmpc": (0, (4, 2.2))}
PMARK = {"mjpc": "o", "cmpc": "s"}
PFILL = {"mjpc": True, "cmpc": False}
PHATCH = {"mjpc": None, "cmpc": "////"}
#: The "this target had no clean replicate" mark, one shape per planner for the
#: same reason the ring takes the planner's marker.
PCROSS = {"mjpc": "x", "cmpc": "+"}


def _save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, "%s.%s" % (name, ext)),
                    facecolor="white")
    plt.close(fig)
    print("  wrote %s.pdf / .png" % name)


def _legend_handles(with_planner=True):
    h = [plt.Line2D([], [], color=C[a], lw=2.0, label=LABEL[a]) for a in ORDER]
    if with_planner:
        h += [plt.Line2D([], [], color=INK2, lw=1.8, ls=PLINE[p],
                         marker=PMARK[p], ms=5,
                         mfc=(INK2 if PFILL[p] else "white"),
                         label=PSHORT[p]) for p in PLANNERS]
    return h


def _bar(ax, x, mu, hr, arm, planner, width, label=None):
    ax.bar(x, mu, width=width, zorder=3, label=label,
           color=(C[arm] if PFILL[planner] else "white"),
           edgecolor=C[arm], linewidth=1.1, hatch=PHATCH[planner])
    if hr > 0:
        ax.errorbar(x, mu, yerr=hr, color=INK2, lw=1.0, capsize=2.5, zorder=4)


# --------------------------------------------------------------------------- #
def fig_envelope(D, out):
    """The target sweep, both planners: does the hand arrive, and how far out.

    This is the cross-planner headline. The x ladder is the same on both sides
    by construction (`BVS_SWEEP_X` feeds both harnesses), so a point that exists
    for one planner and not the other means that planner had no admissible
    rollout there, not that it was never asked.
    """
    fig, ax = plt.subplots(1, 2, figsize=(TEXT, 2.8), layout="constrained")
    marked = False
    for p in PLANNERS:
        d = defaultdict(lambda: defaultdict(list))
        for r in BP.by(D[p], "sweep"):
            d[r["arm"]][round(r["target"][0], 3)].append(r)
        # RING THE POINTS WHOSE REPLICATES WERE NOT ALL CLEAN, cross the
        # targets where none were. Without it the sampling planner's braced
        # curve -- built from a single surviving rollout at most of its
        # targets, and from none at three of them -- reads as a solid line
        # against the gradient planner's two-replicate one. See
        # bvs_plots.fig_envelope.
        for arm in ORDER:
            allx = BP.sweep_all(D[p], arm)
            for x in sorted(allx):
                n_all, n_ok = len(allx[x]), len(d[arm].get(x, []))
                if n_ok == n_all:
                    continue
                marked = True
                if n_ok:
                    v = _ms([r["reach_err_settled"]
                             for r in d[arm][x]])[0] * 100
                    # THE RING TAKES THE PLANNER'S MARKER, not a generic
                    # circle: hue is already spent on the arm, so an
                    # unshaped ring at a target both planners degenerate on
                    # cannot say which one it belongs to -- and at x = 0.90
                    # both braced arms have one, for different reasons.
                    ax[0].plot([x], [v], marker=PMARK[p], ms=12, mfc="none",
                               mec=C[arm], mew=1.3, zorder=5)
                else:
                    ax[0].plot([x], [0.0], marker=PCROSS[p], ms=8,
                               color=C[arm], mew=1.6, zorder=5, clip_on=False)
        for arm in ORDER:
            xs = sorted(d[arm])
            if not xs:
                continue
            e = [_ms([r["reach_err_settled"] for r in d[arm][x]]) for x in xs]
            fr = [_ms([r.get("func_reach_settled") for r in d[arm][x]])
                  for x in xs]
            for i, series in enumerate((e, fr)):
                ax[i].errorbar(
                    xs, [v[0] * 100 for v in series],
                    yerr=[v[1] * 100 for v in series], color=C[arm],
                    ls=PLINE[p], marker=PMARK[p], ms=4.6, lw=1.7, capsize=2.2,
                    elinewidth=0.9, zorder=3,
                    mfc=(C[arm] if PFILL[p] else "white"), mew=1.3)
    ax[0].axhline(5.0, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax[0].annotate("5 cm arrival tolerance", xy=(0.02, 5.0),
                   xycoords=("axes fraction", "data"), xytext=(0, 3),
                   textcoords="offset points", color=INK2, fontsize=7.5)
    ax[0].set_xlabel("Commanded target $x$  [m]")
    ax[0].set_ylabel("Settled reach error  [cm]")
    ax[0].set_title("Does the hand arrive?", loc="left")
    ax[1].set_xlabel("Commanded target $x$  [m]")
    ax[1].set_ylabel("Functional reach  [cm]")
    ax[1].set_title("How far the hand actually gets", loc="left")
    h = _legend_handles()
    if marked:
        h += [plt.Line2D([], [], ls="", marker="o", ms=10, mfc="none",
                         mec=INK2, mew=1.3, label="some reps degenerate"),
              plt.Line2D([], [], ls="", marker="x", ms=8, color=INK2, mew=1.6,
                         label="all degenerate")]
    fig.legend(handles=h, loc="outside lower center", ncol=6, fontsize=7.0)
    _save(fig, out, "fig_x_reach_envelope")


def fig_panels(D, out, which="support", stage="nominal",
               name="fig_x_stability_panels"):
    """The four-metric bar panel, planner x arm, at one condition."""
    mkey, mlabel = BP.MARGIN_VARIANTS[which]
    panels = [
        ("func_reach_settled", "cm   (higher better)", 100.0,
         "Functional reach", "%.0f"),
        (mkey, "cm   (higher better)", 100.0, mlabel, "%.1f"),
        ("sparc_reach", "SPARC   (higher better)", 1.0, "Smoothness", "%.2f"),
        (BP.brace_arm_load, "N", 1.0, "Bracing-arm force", "%.0f"),
    ]
    fig, axs = plt.subplots(1, len(panels), figsize=(TEXT, 2.8),
                            layout="constrained")
    W = 0.38
    for ax, (key, ylab, sc, title, fmt) in zip(axs, panels):
        for ai, arm in enumerate(ORDER):
            for pi, p in enumerate(PLANNERS):
                g = [r for r in BP.by(D[p], stage) if r["arm"] == arm]
                if callable(key):
                    vals = [key(r) for r in g]
                else:
                    vals = [r.get(key) for r in g if r.get(key) is not None
                            and np.isfinite(r.get(key, np.nan))]
                mu, hr = _ms([v * sc for v in vals])
                if not np.isfinite(mu):
                    continue
                x = ai + (pi - 0.5) * W
                _bar(ax, x, mu, hr, arm, p, W * 0.9)
                neg = mu < 0
                ax.annotate(fmt % mu, xy=(x, mu - hr if neg else mu + hr),
                            xytext=(0, -3 if neg else 2.5),
                            textcoords="offset points", ha="center",
                            va="top" if neg else "bottom", fontsize=7.0,
                            color=INK, zorder=5)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels(["No brace\ncost", "Commanded\nbrace"])
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.margins(y=0.28)
        ax.grid(axis="x", visible=False)
    h = [plt.Rectangle((0, 0), 1, 1, fc=(C[a] if PFILL[p] else "white"),
                       ec=C[a], hatch=PHATCH[p],
                       label="%s -- %s" % (LABEL[a], PSHORT[p]))
         for a in ORDER for p in PLANNERS]
    fig.legend(handles=h, loc="outside lower center", ncol=4, fontsize=7.4)
    _save(fig, out, "%s_%s" % (name, which))


def fig_maxreach(D, out):
    """Max reach and the margin it was bought at, both planners."""
    fig, ax = plt.subplots(1, 3, figsize=(TEXT, 2.8), layout="constrained")
    spec = [("func_reach_settled", "Functional reach  [cm]",
             "How far the hand gets", "%.0f", 100.0),
            ("margin_actuated_settled", "CoM margin  [cm]",
             "Margin at full stretch", "%.1f", 100.0),
            (BP.brace_arm_load, "N", "Bracing-arm force", "%.0f", 1.0)]
    W = 0.38
    for i, (key, ylab, title, fmt, sc) in enumerate(spec):
        for ai, arm in enumerate(ORDER):
            for pi, p in enumerate(PLANNERS):
                g = [r for r in BP.by(D[p], "maxreach") if r["arm"] == arm]
                vals = ([key(r) for r in g] if callable(key)
                        else [r.get(key) for r in g if r.get(key) is not None])
                mu, hr = _ms([v * sc for v in vals if v is not None])
                if not np.isfinite(mu):
                    continue
                x = ai + (pi - 0.5) * W
                _bar(ax[i], x, mu, hr, arm, p, W * 0.9)
                ax[i].annotate(fmt % mu, xy=(x, mu + hr), xytext=(0, 2.5),
                               textcoords="offset points", ha="center",
                               va="bottom", fontsize=7.0, color=INK, zorder=5)
        ax[i].set_xticks(range(len(ORDER)))
        ax[i].set_xticklabels(["No brace\ncost", "Commanded\nbrace"])
        ax[i].set_ylabel(ylab)
        ax[i].set_title(title, loc="left")
        ax[i].margins(y=0.26)
        ax[i].grid(axis="x", visible=False)
    h = [plt.Rectangle((0, 0), 1, 1, fc=(C[a] if PFILL[p] else "white"),
                       ec=C[a], hatch=PHATCH[p],
                       label="%s -- %s" % (LABEL[a], PSHORT[p]))
         for a in ORDER for p in PLANNERS]
    fig.legend(handles=h, loc="outside lower center", ncol=4, fontsize=7.4)
    _save(fig, out, "fig_x_maxreach")


def fig_disturb(D, out, stage="disturb", name="fig_x_disturbance",
                where="targeted reach"):
    """Push rejection, both planners."""
    fig, ax = plt.subplots(1, 2, figsize=(TEXT, 2.8), layout="constrained")
    any_ = False
    for p in PLANNERS:
        d = defaultdict(lambda: defaultdict(list))
        fell = defaultdict(lambda: defaultdict(list))
        for r in BP.by(D[p], stage):
            f = abs(float(r["tag"].split("_")[1][1:]))
            fell[r["arm"]][f].append(bool(r.get("fell")))
            for q in r.get("pulses", []):
                d[r["arm"]][f].append(q)
        for arm in ORDER:
            fs = sorted(d[arm])
            if not fs:
                continue
            any_ = True
            pk = [_ms([q["peak_dev_mm"] for q in d[arm][f]]) for f in fs]
            rc = [_ms([q["recover_s"] for q in d[arm][f]]) for f in fs]
            for i, series in enumerate((pk, rc)):
                ax[i].errorbar(
                    fs, [v[0] for v in series], yerr=[v[1] for v in series],
                    color=C[arm], ls=PLINE[p], marker=PMARK[p], ms=4.6, lw=1.7,
                    capsize=2.2, elinewidth=0.9, zorder=3,
                    mfc=(C[arm] if PFILL[p] else "white"), mew=1.3)
            for f, v in zip(fs, pk):
                n = sum(fell[arm][f])
                if n:
                    ax[0].plot([f], [v[0]], marker="o", ms=11, mfc="none",
                               mec=C[arm], mew=1.6, zorder=4)
                    ax[0].annotate("%d/%d fell (%s)"
                                   % (n, len(fell[arm][f]), PSHORT[p]),
                                   xy=(f, v[0]), xytext=(0, 11),
                                   textcoords="offset points", ha="center",
                                   fontsize=6.8, color=INK2, zorder=6)
    if not any_:
        plt.close(fig)
        return
    ax[0].set_xlabel("Push at the reaching hand  [N]")
    ax[0].set_ylabel("Peak hand deflection  [mm]")
    ax[0].set_title("How far the push moves the hand (%s)" % where,
                    loc="left", fontsize=8.5)
    ax[1].set_xlabel("Push at the reaching hand  [N]")
    ax[1].set_ylabel("Recovery time  [s]")
    ax[1].set_title("Post-disruption recovery", loc="left")
    fig.legend(handles=_legend_handles(), loc="outside lower center", ncol=4,
               fontsize=7.6)
    _save(fig, out, name)


def fig_region(D, out):
    """Admissible CoM region: four postures, one plane.

    The polygons overlap heavily, so the planner rides as edge style and the
    fills are kept faint -- the claim being made is about AREA and about where
    the forward edge sits, both of which survive an outline.
    """
    fig, ax = plt.subplots(1, 2, figsize=(TEXT * 0.72, 3.1),
                           layout="constrained", sharex=True, sharey=True)
    for pi, p in enumerate(PLANNERS):
        drew = 0
        for arm in ORDER:
            r = next((q for q in BP.by(D[p], "nominal")
                      if q["arm"] == arm and q.get("region_xy")), None)
            if r is None:
                continue
            drew += 1
            P = np.array(r["region_xy"])
            ax[pi].add_patch(Polygon(
                P, closed=True, facecolor=C[arm], alpha=0.14,
                edgecolor=C[arm], lw=1.8, ls="-", zorder=2,
                label="%s  (%.1f cm)"
                      % (LABEL[arm], r["margin_actuated_settled"] * 100)))
            c = r["region_com"]
            ax[pi].plot(c[0], c[1], marker="o", ms=7, color=C[arm],
                        mec="white", mew=1.4, zorder=5)
            CP = np.array([q["p"] for q in r["region_contacts"]])
            ax[pi].scatter(CP[:, 0], CP[:, 1], s=13, color=C[arm], alpha=0.85,
                           marker="x", linewidths=1.1, zorder=4)
        ax[pi].set_aspect("equal")
        ax[pi].set_xlabel("World $x$  [m]")
        ax[pi].set_title(PNAME[p], loc="left")
        if drew:
            ax[pi].legend(loc="lower left", fontsize=6.8)
    ax[0].set_ylabel("World $y$  [m]")
    _save(fig, out, "fig_x_equilibrium_region")


def fig_margin_series(D, out):
    """CoM margin over the reach, both planners."""
    fig, ax = plt.subplots(figsize=(TEXT * 0.5, 2.8), layout="constrained")
    for p in PLANNERS:
        nom = BP.by(D[p], "nominal")
        for arm in ORDER:
            rs = [r for r in nom if r["arm"] == arm and "stab_t" in r]
            if not rs:
                continue
            t = np.array(rs[0]["stab_t"])
            M = np.array([r["fwd_actuated"] for r in rs
                          if len(r["fwd_actuated"]) == len(t)], dtype=float)
            if not len(M):
                continue
            ax.plot(t, np.nanmean(M, axis=0) * 100, color=C[arm], ls=PLINE[p],
                    lw=1.8, zorder=3)
            if M.shape[0] > 1:
                ax.fill_between(t, np.nanmin(M, axis=0) * 100,
                                np.nanmax(M, axis=0) * 100, color=C[arm],
                                alpha=0.10, lw=0, zorder=2)
    ax.set_xlabel("Time  [s]")
    ax.set_ylabel("Forward CoM margin  [cm]")
    ax.set_title("Equilibrium margin over the reach", loc="left")
    ax.legend(handles=_legend_handles(), loc="lower right", ncol=2,
              fontsize=7.0, columnspacing=1.0)
    ax.margins(y=0.20)
    _save(fig, out, "fig_x_margin_series")


def _mjpc_step_ms(run_dir):
    """Mean wall time per PLANNING STEP, parsed out of testspeed's own logs.

    testspeed prints `Total wall time (N planning steps): T s`, and that is the
    only place MJPC's per-step cost is recorded -- there is no equivalent of the
    CMPC manifest's `solve_ms_mean`, because the sampling planner runs
    asynchronously and does not time an individual solve. T/N is therefore what
    the two sides have in common: seconds of wall clock spent optimising, per
    optimisation.
    """
    import glob
    import re
    out = []
    for path in glob.glob(os.path.join(run_dir, "*.log")):
        m = re.search(r"Total wall time \((\d+) planning steps\): "
                      r"([0-9.eE+-]+) s", open(path).read())
        if m and int(m.group(1)):
            out.append(1e3 * float(m.group(2)) / int(m.group(1)))
    return out


def fig_solve(D, out, manifests=None):
    """What the two planners cost per optimisation, on the same box.

    NOT a like-for-like benchmark, and the caption has to say so: the two ran
    under different budgets -- MJPC 18 sampling threads replanning
    asynchronously, CMPC one warm-started BoxFDDP iteration over 35 nodes per
    20 ms period on 18 threads. It is here because the
    contact-explicit/contact-implicit trade is partly a compute trade and a
    reader will ask.
    """
    if not manifests:
        return
    vals = {}
    for p, path in manifests.items():
        if not path or not os.path.exists(path):
            continue
        if os.path.isdir(path):
            v = _mjpc_step_ms(path)
        else:
            man = json.load(open(path))
            v = [r.get("solve_ms_mean") for r in man
                 if isinstance(r, dict) and r.get("solve_ms_mean")]
        if v:
            vals[p] = v
    if len(vals) < 1:
        return
    fig, ax = plt.subplots(figsize=(COL, 2.4), layout="constrained")
    for i, p in enumerate([q for q in PLANNERS if q in vals]):
        mu, hr = _ms(vals[p])
        ax.bar(i, mu, width=0.5, color="white", edgecolor=INK2, linewidth=1.2,
               hatch=PHATCH[p], zorder=3)
        ax.errorbar(i, mu, yerr=hr, color=INK2, lw=1.0, capsize=3, zorder=4)
        ax.annotate("%.1f ms" % mu, xy=(i, mu + hr), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, color=INK)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([PSHORT[p] for p in PLANNERS if p in vals])
    ax.set_ylabel("Mean solve time  [ms]")
    ax.set_title("Per-step optimisation cost", loc="left")
    ax.margins(y=0.25)
    ax.grid(axis="x", visible=False)
    _save(fig, out, "fig_x_solve_time")


# --------------------------------------------------------------------------- #
METRICS = [
    ("functional reach", "func_reach_settled", 100.0, "cm", "%.1f"),
    ("settled reach error", "reach_err_settled", 100.0, "cm", "%.1f"),
    ("brace load, all contacts", "brace_load_N", 1.0, "N", "%.0f"),
    ("bracing-arm force", BP.brace_arm_load, 1.0, "N", "%.0f"),
    ("support margin (actuated)", "margin_actuated_settled", 100.0, "cm", "%.1f"),
    ("forward margin (actuated)", "fwd_actuated_settled", 100.0, "cm", "%.1f"),
    ("smoothness SPARC", "sparc_reach", 1.0, "--", "%.2f"),
    ("hand jitter", "hand_jitter_mm", 1.0, "mm", "%.2f"),
    ("worst-case push at hand", "push_min_N", 1.0, "N", "%.0f"),
    ("settling time", "settle_time_s", 1.0, "s", "%.1f"),
    ("peak torque ratio", "peak_tau_ratio", 1.0, "--", "%.2f"),
]


def write_table(D, out, stage="nominal", name="table_mjpc_vs_cmpc"):
    cols = [(p, a) for p in PLANNERS for a in ORDER]
    d = {(p, a): [r for r in BP.by(D[p], stage) if r["arm"] == a]
         for p, a in cols}
    lines, tex = [], []
    head = "%-28s" % "metric" + "".join(
        "%16s" % ("%s/%s" % (PSHORT[p], a)) for p, a in cols) + "   unit"
    lines.append(head)
    for lab, key, sc, unit, fmt in METRICS:
        row, texrow = ["%-28s" % lab], [lab.replace("_", "\\_")]
        for p, a in cols:
            g = d[(p, a)]
            vals = ([key(r) for r in g] if callable(key)
                    else [r.get(key) for r in g if r.get(key) is not None])
            mu, hr = _ms([v * sc for v in vals if v is not None])
            if np.isfinite(mu):
                cell = (fmt % mu) + (" $\\pm$ " + (fmt % hr) if hr > 0 else "")
                row.append("%16s" % ((fmt % mu) +
                                     ((" \u00b1 " + fmt % hr) if hr > 0 else "")))
            else:
                cell, _ = "--", row.append("%16s" % "--")
            texrow.append(cell)
        lines.append("".join(row) + "   " + unit)
        tex.append(" & ".join(texrow) + " \\\\")
    # counts, which the metrics rows cannot carry
    row = ["%-28s" % "runs bracing (>15 N)"]
    texrow = ["runs bracing ($>$15\\,N)"]
    for p, a in cols:
        g = d[(p, a)]
        n = sum(1 for r in g if r.get("braced_by_load"))
        row.append("%16s" % ("%d/%d" % (n, len(g))))
        texrow.append("%d/%d" % (n, len(g)))
    lines.append("".join(row) + "   --")
    tex.append(" & ".join(texrow) + " \\\\")
    row = ["%-28s" % "torso-rest (excluded)"]
    texrow = ["torso-rest (excluded)"]
    for p, a in cols:
        g = [r for r in D[p] if r.get("stage") == stage and r.get("arm") == a
             and "error" not in r]
        n = sum(1 for r in g if BP.contact_class(r) == "torso")
        row.append("%16s" % ("%d/%d" % (n, len(g))))
        texrow.append("%d/%d" % (n, len(g)))
    lines.append("".join(row) + "   --")
    tex.append(" & ".join(texrow) + " \\\\")

    txt = "\n".join(lines)
    open(os.path.join(out, name + ".txt"), "w").write(txt + "\n")
    body = "\n".join(tex)
    open(os.path.join(out, name + ".tex"), "w").write(
        "%% generated by studies/mjpc_vs_cmpc.py -- stage=%s\n"
        "\\begin{tabular}{l%s}\n\\toprule\n"
        " & \\multicolumn{2}{c}{MJPC (sampling)}"
        " & \\multicolumn{2}{c}{CMPC (gradient)} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
        "metric & no brace & brace & no brace & brace \\\\\n\\midrule\n"
        "%s\n\\bottomrule\n\\end{tabular}\n"
        % (stage, "r" * len(cols), body))
    print("\n" + txt + "\n")
    print("  wrote %s.tex / .txt" % name)


# Settled conditions to draw, and the filename suffix each gets. `nominal2`
# takes the UNSUFFIXED name when it exists: it is the condition where both
# planners are standing, so it is the one a caption means by "the settled
# comparison", and the paper should not have to know which file that is.
def _stages_present(D):
    out = []
    have2 = any(r.get("stage") == "nominal2" for d in D.values() for r in d)
    for stage in ("nominal", "nominal2"):
        if not any(r.get("stage") == stage for d in D.values() for r in d):
            continue
        if stage == "nominal2":
            out.append((stage, ""))
        else:
            out.append((stage, "_nominal" if have2 else ""))
    return out


def make_all(mjpc_json, cmpc_json, out, manifests=None):
    D = {"mjpc": json.load(open(mjpc_json)),
         "cmpc": json.load(open(cmpc_json))}
    os.makedirs(out, exist_ok=True)
    print("overlaying %d MJPC + %d CMPC rollouts -> %s"
          % (len(D["mjpc"]), len(D["cmpc"]), out))
    fig_envelope(D, out)
    # ONE PANEL SET PER SETTLED CONDITION, not just the default target. At
    # `nominal` the gradient planner's braced column is empty -- every replicate
    # there falls and is classified out -- so a single figure keyed to that
    # condition would show the comparison as a blank half. The conditions that
    # have data are drawn; the one that does not is still drawn, because an
    # empty braced bar next to a full standing one IS the finding.
    for stage, suffix in _stages_present(D):
        for w in BP.MARGIN_VARIANTS:
            fig_panels(D, out, w, stage, "fig_x_stability_panels" + suffix)
    fig_maxreach(D, out)
    for _s, _n, _w in BP.DISTURB_STAGES:
        fig_disturb(D, out, _s, _n.replace("fig_", "fig_x_"), _w)
    fig_region(D, out)
    fig_margin_series(D, out)
    fig_solve(D, out, manifests)
    for stage, suffix in _stages_present(D):
        write_table(D, out, stage, "table_mjpc_vs_cmpc" + suffix)
    if BP.by(D["mjpc"], "maxreach") or BP.by(D["cmpc"], "maxreach"):
        write_table(D, out, "maxreach", "table_mjpc_vs_cmpc_maxreach")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mjpc", required=True)
    ap.add_argument("--cmpc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mjpc-manifest", default=None,
                    help="the MJPC RUN DIRECTORY: per-step cost is parsed out "
                         "of testspeed's own per-run logs, there being no "
                         "manifest field for it")
    ap.add_argument("--cmpc-manifest", default=None)
    a = ap.parse_args()
    make_all(a.mjpc, a.cmpc, a.out,
             manifests=dict(mjpc=a.mjpc_manifest, cmpc=a.cmpc_manifest))


if __name__ == "__main__":
    main()
