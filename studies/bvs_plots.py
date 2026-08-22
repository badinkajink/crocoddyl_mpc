#!/usr/bin/env python3
"""Paper figures for the braced-vs-standing reach comparison.

Written for LaTeX, so every figure goes out as PDF (vector, embeds cleanly in
IEEEtran) with a PNG beside it for quick viewing. Design rules followed here,
from the data-viz method:

  * Two series, so two categorical hues in FIXED slot order -- slot 1 blue for
    standing, slot 2 orange for braced. Never cycled, never reassigned by rank.
    That pair is the documented-validated opening of the reference palette
    (worst adjacent CVD dE 9.1 light, normal-vision 19.6), so it survives
    colour-blind readers and greyscale printing.
  * A second dimension inside one series is a LINESTYLE, never a second hue --
    contact-only vs actuated margin are the same arm, so they share its colour.
  * No dual axes anywhere. Metrics with different units become small multiples.
  * Legend whenever two series are on one axes; recessive grid; thin marks.
  * Text is ink-coloured, never series-coloured.
"""
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

C = {"stand": "#2a78d6", "brace": "#eb6834"}         # categorical slots 1, 2
# The arms differ in whether a brace was ASKED FOR, not in whether one occurs.
# Lean Simple is contact-implicit: with all three brace weights at zero the
# planner still discovers the table and leans on it (measured: up to 76 N through
# the left gripper). Calling that arm "standing" would be a claim the force data
# contradicts, so it is named for what was specified, not for what happened.
LABEL = {"stand": "No brace cost", "brace": "Commanded brace"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
ORDER = ["stand", "brace"]

# IEEEtran column and text widths, in inches. Figures are authored AT their final
# size so LaTeX never scales them -- a figure dropped in at 0.8\linewidth carries
# 8.5 pt labels that print at 6.8 pt, which is how a readable figure becomes an
# unreadable one between the plot script and the PDF.
COL, TEXT = 3.45, 7.16

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160,
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "grid.alpha": 0.9, "axes.axisbelow": True,
    "legend.frameon": False, "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
})


def _save(fig, out, name):
    for ext in ("pdf", "png"):
        p = os.path.join(out, "%s.%s" % (name, ext))
        fig.savefig(p, facecolor="white")
    plt.close(fig)
    print("  wrote %s.pdf / .png" % name)


def _ms(v):
    """mean, and half-range as the spread bar -- replicates are few (2-4), so a
    standard error would imply a precision the sample size does not have."""
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return np.nan, 0.0
    return float(np.mean(v)), float((max(v) - min(v)) / 2.0)


# --------------------------------------------------------------------------- #
# Load-path classification.
#
# A "commanded brace" run is not necessarily bracing. At the far targets the
# planner discovers that lying on the table pays: 3 of 4 max-reach and 8 of 10
# swept brace rollouts finish with 300-470 N -- 47-70% of body weight -- through
# `torso_link`. lean_simple.cc has a Trunk Clear term at weight 1500 to forbid
# exactly this, but its residual saturates: once the chest is ON the slab the
# violation stops growing, while the reach error keeps paying, so buying reach
# with the trunk becomes the cheaper trade. The task's own comment calls 398 N
# through the trunk "not something to deploy".
#
# Those runs are not a brace and must not be averaged into one. They are kept,
# labelled, and drawn separately -- deleting them would hide a real failure mode
# of the cost function, and pooling them would credit the brace with a margin
# that a chest-on-table pose earned.
TRUNK_N = 15.0


def trunk_load(r):
    b = r.get("brace_load_bodies") or {}
    return b.get("torso_link", 0.0) + b.get("pelvis", 0.0)


def arm_load(r):
    b = r.get("brace_load_bodies") or {}
    return sum(v for k, v in b.items()
               if "torso" not in k and "pelvis" not in k)


def contact_class(r):
    if trunk_load(r) > TRUNK_N:
        return "torso"
    return "brace" if arm_load(r) >= TRUNK_N else "free"


def by(rows, stage, clean=True):
    """Rollouts of a stage. clean=True drops the torso-rest degenerates."""
    out = [r for r in rows if r.get("stage") == stage and "error" not in r]
    return [r for r in out if contact_class(r) != "torso"] if clean else out


# --------------------------------------------------------------------------- #
def fig_envelope(rows, out):
    """The headline: how far out each arm still ARRIVES."""
    d = defaultdict(lambda: defaultdict(list))
    for r in by(rows, "sweep"):
        d[r["arm"]][round(r["target"][0], 3)].append(r)
    if not d:
        return
    fig, ax = plt.subplots(1, 2, figsize=(TEXT, 2.7), layout="constrained")

    for arm in ORDER:
        xs = sorted(d[arm])
        e = [_ms([r["reach_err_settled"] for r in d[arm][x]]) for x in xs]
        g = [_ms([r["reach_gain"] for r in d[arm][x]]) for x in xs]
        ax[0].errorbar(xs, [v[0] * 100 for v in e], yerr=[v[1] * 100 for v in e],
                       color=C[arm], marker="o", ms=4.5, lw=1.8, capsize=2.5,
                       elinewidth=1.0, label=LABEL[arm], zorder=3)
        fr_ = [_ms([r.get("func_reach_settled") for r in d[arm][x]
                    if r.get("func_reach_settled") is not None]) for x in xs]
        ax[1].errorbar(xs, [v[0] * 100 for v in fr_],
                       yerr=[v[1] * 100 for v in fr_],
                       color=C[arm], marker="o", ms=4.5, lw=1.8, capsize=2.5,
                       elinewidth=1.0, label=LABEL[arm], zorder=3)

    ax[0].axhline(5.0, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax[0].annotate("5 cm arrival tolerance", xy=(0.02, 5.0),
                   xycoords=("axes fraction", "data"), xytext=(0, 3),
                   textcoords="offset points", color=INK2, fontsize=7.5)
    ax[0].set_xlabel("commanded target $x$  [m]")
    ax[0].set_ylabel("settled reach error  [cm]")
    ax[0].set_title("Does the hand arrive?", loc="left")
    ax[1].set_xlabel("commanded target $x$  [m]")
    ax[1].set_ylabel("functional reach  [cm]")
    ax[1].set_title("How far the hand actually gets", loc="left")
    ax[0].legend(loc="upper left")
    _save(fig, out, "fig_reach_envelope")


def fig_stability(rows, out):
    """Small multiples, each metric grouped by CONDITION and coloured by arm.

    Two conditions, because they answer different halves of the question. At the
    study target both postures arrive (standing to within 6 mm), so that group
    isolates the STABILITY difference at equal task success. At max reach the
    target is unreachable and each posture stretches as far as it can, so that
    group is where the reach difference lives -- and its stability numbers are
    the ones that say what the extra reach cost."""
    groups = [("nominal", "targeted\nreach"), ("maxreach", "max\nreach")]
    have = [(g, lab) for g, lab in groups if by(rows, g)]
    if not have:
        return
    d = defaultdict(lambda: defaultdict(list))
    for g, _ in have:
        for r in by(rows, g):
            d[g][r["arm"]].append(r)

    panels = [
        ("func_reach_settled", "m   (higher better)", 100.0,
         "Functional reach", "%.0f", 100.0),
        ("margin_actuated_settled", "cm   (higher better)", 100.0,
         "CoM margin", "%.1f", 1.0),
        ("hand_jitter_mm", "mm   (lower better)", 1.0, "Hand jitter", "%.1f", 1.0),
        ("push_min_N", "N   (higher better)", 1.0, "Push at hand", "%.0f", 1.0),
    ]
    fig, axs = plt.subplots(1, len(panels), figsize=(TEXT, 2.7),
                            layout="constrained")
    W = 0.36
    for ax, (key, ylab, sc, title, fmt, _u) in zip(axs, panels):
        for gi, (g, lab) in enumerate(have):
            for ai, arm in enumerate(ORDER):
                vals = [r.get(key) for r in d[g][arm] if r.get(key) is not None]
                mu, hr = _ms([v * sc for v in vals])
                if not np.isfinite(mu):
                    continue
                x = gi + (ai - 0.5) * W
                ax.bar(x, mu, width=W * 0.92, color=C[arm], zorder=3,
                       edgecolor="white", linewidth=1.0,
                       label=LABEL[arm] if gi == 0 else None)
                if hr > 0:
                    ax.errorbar(x, mu, yerr=hr, color=INK2, lw=1.0, capsize=2.5,
                                zorder=4)
                ax.annotate(fmt % mu, xy=(x, mu + hr), xytext=(0, 2.5),
                            textcoords="offset points", ha="center",
                            va="bottom", fontsize=7.2, color=INK, zorder=5)
        ax.set_xticks(range(len(have)))
        ax.set_xticklabels([lab for _, lab in have])
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left")
        ax.margins(y=0.26)
        ax.grid(axis="x", visible=False)
    # ylabel of panel 0 is cm for functional reach -- relabel it honestly
    axs[0].set_ylabel("cm   (higher better)")
    # One figure-level legend outside the axes: inside panel 0 it lands on top
    # of the bars, and there is no headroom in any panel that is not already
    # carrying a value label.
    h = [plt.Rectangle((0, 0), 1, 1, color=C[a]) for a in ORDER]
    fig.legend(h, [LABEL[a] for a in ORDER], loc="outside lower center",
               ncol=2, fontsize=8)
    _save(fig, out, "fig_stability_panels")


def fig_margin_series(rows, out):
    """CoM margin over the rollout: solid contact-only, dashed actuated."""
    nom = by(rows, "nominal")
    if not nom:
        return
    fig, ax = plt.subplots(figsize=(COL, 2.7), layout="constrained")
    for arm in ORDER:
        rs = [r for r in nom if r["arm"] == arm and "stab_t" in r]
        if not rs:
            continue
        t = np.array(rs[0]["stab_t"])
        for key, ls, lw, a in (("margin_contact", "-", 1.9, 1.0),
                               ("margin_actuated", (0, (4, 2.5)), 1.5, 0.95)):
            M = np.array([r[key] for r in rs if len(r[key]) == len(t)],
                         dtype=float)
            if not len(M):
                continue
            mu = np.nanmean(M, axis=0)
            ax.plot(t, mu * 100, color=C[arm], ls=ls, lw=lw, alpha=a, zorder=3)
            if M.shape[0] > 1:
                lo, hi = np.nanmin(M, axis=0) * 100, np.nanmax(M, axis=0) * 100
                ax.fill_between(t, lo, hi, color=C[arm], alpha=0.13, lw=0,
                                zorder=2)
    ax.set_xlabel("time  [s]")
    ax.set_ylabel("CoM margin  [cm]")
    ax.set_title("Static-equilibrium margin over the reach", loc="left")
    h = [plt.Line2D([], [], color=C[a], lw=1.9, label=LABEL[a]) for a in ORDER]
    h += [plt.Line2D([], [], color=INK2, lw=1.9, label="contact set only"),
          plt.Line2D([], [], color=INK2, lw=1.5, ls=(0, (4, 2.5)),
                     label="with torque limits")]
    ax.legend(handles=h, loc="lower right", ncol=2, columnspacing=1.2)
    ax.margins(y=0.18)
    _save(fig, out, "fig_margin_series")


def fig_maxreach(rows, out):
    """Max reach: how far each posture stretches when the target is unreachable."""
    mr = by(rows, "maxreach")
    if not mr:
        return
    d = defaultdict(list)
    for r in mr:
        d[r["arm"]].append(r)
    fig, ax = plt.subplots(1, 2, figsize=(TEXT, 2.7), layout="constrained")

    # (a) how far it got, and what the margin was there
    for i, (key, ylab, title, fmt, sc) in enumerate([
            ("func_reach_settled", "functional reach  [cm]",
             "How far the hand gets", "%.0f", 100.0),
            ("margin_actuated_settled", "CoM margin  [cm]",
             "Margin at full stretch", "%.1f", 100.0)]):
        for ai, arm in enumerate(ORDER):
            mu, hr = _ms([r.get(key) * sc for r in d[arm]
                          if r.get(key) is not None])
            ax[i].bar(ai, mu, width=0.55, color=C[arm], zorder=3,
                      edgecolor="white", linewidth=1.2, label=LABEL[arm])
            if hr > 0:
                ax[i].errorbar(ai, mu, yerr=hr, color=INK2, lw=1.0, capsize=3,
                               zorder=4)
            ax[i].annotate(fmt % mu, xy=(ai, mu + hr), xytext=(0, 3),
                           textcoords="offset points", ha="center",
                           va="bottom", fontsize=8, color=INK, zorder=5)
        ax[i].set_xticks(range(len(ORDER)))
        ax[i].set_xticklabels(["stand", "brace"])
        ax[i].set_ylabel(ylab)
        ax[i].set_title(title, loc="left")
        ax[i].margins(y=0.24)
        ax[i].grid(axis="x", visible=False)
    _save(fig, out, "fig_maxreach")


def fig_load_vs_margin(rows, out):
    """Margin against MEASURED brace load, pooled over both arms.

    This is the figure the contact-implicit result needs. The arms are labelled
    by what they were asked for, but the planner decides what it leans on, so
    the honest independent variable is newtons through non-foot contacts -- not
    the weights. If bracing is what buys margin, every run should lie on one
    trend regardless of which arm it came from."""
    allp = [r for r in rows if r.get("brace_load_N") is not None
            and np.isfinite(r.get("brace_load_N", np.nan))
            and r.get("margin_actuated_settled") is not None]
    pts = [r for r in allp if contact_class(r) != "torso"]
    degen = [r for r in allp if contact_class(r) == "torso"]
    if len(pts) < 4:
        return
    fig, ax = plt.subplots(figsize=(COL, 2.8), layout="constrained")
    MK = {"nominal": "o", "sweep": "s", "maxreach": "^", "disturb": "D"}
    for arm in ORDER:
        for stage, mk in MK.items():
            g = [r for r in pts if r["arm"] == arm and r.get("stage") == stage]
            if not g:
                continue
            ax.scatter([r["brace_load_N"] for r in g],
                       [r["margin_actuated_settled"] * 100 for r in g],
                       s=26, marker=mk, color=C[arm], alpha=0.85,
                       edgecolor="white", linewidth=0.7, zorder=3)
    if degen:
        ax.scatter([r["brace_load_N"] for r in degen],
                   [r["margin_actuated_settled"] * 100 for r in degen],
                   s=30, marker="x", color=INK2, alpha=0.75, linewidths=1.2,
                   zorder=3, label="torso on table")
    x = np.array([r["brace_load_N"] for r in pts])
    y = np.array([r["margin_actuated_settled"] * 100 for r in pts])
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() >= 3:
        b, a = np.polyfit(x[ok], y[ok], 1)
        xx = np.linspace(x[ok].min(), x[ok].max(), 50)
        ax.plot(xx, a + b * xx, color=INK2, lw=1.3, ls=(0, (5, 3)), zorder=2)
        r = np.corrcoef(x[ok], y[ok])[0, 1]
        ax.text(0.03, 0.95, "slope %.3f cm/N,  $r$ = %.2f" % (b, r),
                transform=ax.transAxes, va="top", fontsize=7.5, color=INK2)
    ax.axvline(15.0, color=GRID, lw=1.0, zorder=1)
    ax.set_xlabel("measured brace load  [N]")
    ax.set_ylabel("CoM margin  [cm]")
    ax.set_title("Margin follows load, not labels", loc="left")
    h = [plt.Line2D([], [], ls="", marker="o", ms=6, color=C[a],
                    label=LABEL[a]) for a in ORDER]
    if degen:
        h.append(plt.Line2D([], [], ls="", marker="x", ms=6, color=INK2,
                            label="torso on table (excluded)"))
    ax.legend(handles=h, loc="lower right", fontsize=6.8)
    _save(fig, out, "fig_load_vs_margin")


def fig_disturb(rows, out):
    """Push rejection: peak deviation and recovery time vs push magnitude."""
    dis = by(rows, "disturb")
    if not dis:
        return
    d = defaultdict(lambda: defaultdict(list))
    fell = defaultdict(lambda: defaultdict(list))
    for r in dis:
        f = abs(float(r["tag"].split("_")[1][1:]))
        fell[r["arm"]][f].append(bool(r.get("fell")))
        for p in r.get("pulses", []):
            d[r["arm"]][f].append(p)
    if not d:
        return
    fig, ax = plt.subplots(1, 2, figsize=(TEXT, 2.7), layout="constrained")
    for arm in ORDER:
        fs = sorted(d[arm])
        pk = [_ms([p["peak_dev_mm"] for p in d[arm][f]]) for f in fs]
        rc = [_ms([p["recover_s"] for p in d[arm][f]]) for f in fs]
        ax[0].errorbar(fs, [v[0] for v in pk], yerr=[v[1] for v in pk],
                       color=C[arm], marker="o", ms=4.5, lw=1.8, capsize=2.5,
                       elinewidth=1.0, label=LABEL[arm], zorder=3)
        ax[1].errorbar(fs, [v[0] for v in rc], yerr=[v[1] for v in rc],
                       color=C[arm], marker="o", ms=4.5, lw=1.8, capsize=2.5,
                       elinewidth=1.0, label=LABEL[arm], zorder=3)
        # A toppled run has no recovery time, so it would silently vanish from
        # the mean and make the arm look BETTER the harder it is pushed. Mark it.
        for f, v in zip(fs, pk):
            n = sum(fell[arm][f])
            if n:
                ax[0].plot([f], [v[0]], marker="o", ms=10, mfc="none",
                           mec=C[arm], mew=1.6, zorder=4)
                ax[0].annotate("%d/%d fell" % (n, len(fell[arm][f])),
                               xy=(f, v[0]), xytext=(0, 11),
                               textcoords="offset points", ha="center",
                               fontsize=7, color=INK2, zorder=6)
    ax[0].set_xlabel("push at the reaching hand  [N]")
    ax[0].set_ylabel("peak hand deflection  [mm]")
    ax[0].set_title("How far the push moves the hand", loc="left")
    ax[1].set_xlabel("push at the reaching hand  [N]")
    ax[1].set_ylabel("recovery time  [s]")
    ax[1].set_title("Post-disruption recovery", loc="left")
    ax[0].legend(loc="upper left")
    _save(fig, out, "fig_disturbance")


def fig_region(rows, out):
    """The geometric claim: the static-equilibrium region, both arms."""
    nom = by(rows, "nominal")
    pick = {}
    for arm in ORDER:
        for r in nom:
            if r["arm"] == arm and r.get("region_xy"):
                pick[arm] = r
                break
    if len(pick) < 2:
        return
    fig, ax = plt.subplots(figsize=(COL, 3.0), layout="constrained")
    for arm in ORDER:
        r = pick[arm]
        P = np.array(r["region_xy"])
        ax.add_patch(Polygon(P, closed=True, facecolor=C[arm], alpha=0.16,
                             edgecolor=C[arm], lw=1.8, zorder=2,
                             label="%s  (margin %.1f cm)"
                                   % (LABEL[arm],
                                      r["margin_actuated_settled"] * 100)))
        c = r["region_com"]
        ax.plot(c[0], c[1], marker="o", ms=7, color=C[arm], mec="white",
                mew=1.4, zorder=5)
        CP = np.array([q["p"] for q in r["region_contacts"]])
        ax.scatter(CP[:, 0], CP[:, 1], s=13, color=C[arm], alpha=0.85,
                   marker="x", linewidths=1.1, zorder=4)
    ax.set_aspect("equal")
    ax.set_xlabel("world $x$  [m]")
    ax.set_ylabel("world $y$  [m]")
    ax.set_title("Admissible CoM region", loc="left")
    ax.legend(loc="lower left", fontsize=7.2)
    _save(fig, out, "fig_equilibrium_region")


def write_table(rows, out):
    """The numbers behind the figures, at BOTH conditions, as LaTeX and text."""
    conds = [(g, lab) for g, lab in (("nominal", "targeted reach"),
                                     ("maxreach", "max reach"))
             if by(rows, g)]
    if not conds:
        return
    d = defaultdict(lambda: defaultdict(list))
    for g, _ in conds:
        for r in by(rows, g):
            d[g][r["arm"]].append(r)
    metrics = [
        ("functional reach", "func_reach_settled", 100.0, "cm", "%.1f"),
        ("settled reach error", "reach_err_settled", 100.0, "cm", "%.1f"),
        ("measured brace load", "brace_load_N", 1.0, "N", "%.0f"),
        ("CoM margin (contact)", "margin_contact_settled", 100.0, "cm", "%.1f"),
        ("CoM margin (actuated)", "margin_actuated_settled", 100.0, "cm", "%.1f"),
        ("hand jitter", "hand_jitter_mm", 1.0, "mm", "%.2f"),
        ("reach error spread", "reach_err_std_mm", 1.0, "mm", "%.2f"),
        ("worst-case push at hand", "push_min_N", 1.0, "N", "%.0f"),
        ("settling time (initial)", "settle_time_s", 1.0, "s", "%.1f"),
        ("peak torque ratio", "peak_tau_ratio", 1.0, "--", "%.2f"),
    ]

    def cell(g, arm, key, sc, fmt):
        mu, hr = _ms([r.get(key) for r in d[g][arm] if r.get(key) is not None])
        if not np.isfinite(mu):
            return "n/a"
        return (fmt % (mu * sc)) + ((" ± " + fmt % (hr * sc)) if hr else "")

    head = ["%-24s" % "metric"]
    for _, lab in conds:
        head += ["%13s" % ("stand/" + lab.split()[0]),
                 "%13s" % ("brace/" + lab.split()[0])]
    head += ["%6s" % "unit"]
    txt = [" ".join(head)]
    ncol = 2 * len(conds)
    tex = [r"\begin{tabular}{l%sl}" % ("r" * ncol), r"\toprule"]
    tex.append("metric & " + " & ".join(
        r"\multicolumn{2}{c}{%s}" % lab for _, lab in conds) + r" & unit \\")
    tex.append("& " + " & ".join(["standing & braced"] * len(conds)) +
               r" & \\")
    tex.append(r"\midrule")
    for name, key, sc, unit, fmt in metrics:
        cells = [cell(g, arm, key, sc, fmt)
                 for g, _ in conds for arm in ORDER]
        txt.append(" ".join(["%-24s" % name] + ["%13s" % c for c in cells]
                            + ["%6s" % unit]))
        tex.append("%s & %s & %s \\\\" % (
            name, " & ".join(c.replace("±", r"$\pm$") for c in cells), unit))
    # How often a brace HAPPENED, as opposed to being asked for. On a
    # contact-implicit task this is a result, not bookkeeping.
    cells = []
    for g, _ in conds:
        for arm in ORDER:
            rs = [r for r in d[g][arm] if r.get("braced_by_load") is not None]
            cells.append("n/a" if not rs else
                         "%d/%d" % (sum(bool(r["braced_by_load"]) for r in rs),
                                    len(rs)))
    txt.append(" ".join(["%-24s" % "runs bracing (>15 N)"]
                        + ["%13s" % c for c in cells] + ["%6s" % "--"]))
    dcells = []
    for g, _ in conds:
        for arm in ORDER:
            allr = [r for r in rows if r.get("stage") == g
                    and r.get("arm") == arm and "error" not in r]
            dcells.append("%d/%d" % (sum(contact_class(r) == "torso"
                                         for r in allr), len(allr)))
    txt.append(" ".join(["%-24s" % "torso-rest (excluded)"]
                        + ["%13s" % c for c in dcells] + ["%6s" % "--"]))
    tex.append("torso-rest (excluded) & %s & -- \\\\" % " & ".join(dcells))
    tex.append("runs bracing ($>$15\\,N) & %s & -- \\\\" % " & ".join(cells))
    tex += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(out, "table_brace_vs_stand.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")
    body = "\n".join(txt)
    with open(os.path.join(out, "table_brace_vs_stand.txt"), "w") as f:
        f.write(body + "\n")
    print("\n" + body + "\n")
    print("  wrote table_brace_vs_stand.tex / .txt")


def write_disturb_table(rows, out):
    """Post-disruption response, per push magnitude.

    Kept separate from the main table on purpose: `settling time (initial)`
    there is how long the posture takes to establish itself from the home
    keyframe, which is a different quantity from how long it takes to come back
    after being shoved. Conflating them was a misreading worth designing out --
    the brace is SLOWER to establish (the arm must seat first) and the claim
    under test is that it is FASTER to recover."""
    dis = by(rows, "disturb")
    if not dis:
        return
    d = defaultdict(lambda: defaultdict(list))
    fell = defaultdict(lambda: defaultdict(list))
    for r in dis:
        f = abs(float(r["tag"].split("_")[1][1:]))
        fell[r["arm"]][f].append(bool(r.get("fell")))
        for p in r.get("pulses", []):
            d[r["arm"]][f].append(p)
    fs = sorted({f for arm in d for f in d[arm]})
    txt = ["%-10s %14s %14s %14s %14s %12s"
           % ("push [N]", "stand peak[mm]", "brace peak[mm]",
              "stand rec[s]", "brace rec[s]", "fell s/b")]
    tex = [r"\begin{tabular}{lrrrrc}", r"\toprule",
           r"push & \multicolumn{2}{c}{peak deflection [mm]} & "
           r"\multicolumn{2}{c}{recovery [s]} & fell \\",
           r"[N] & standing & braced & standing & braced & s/b \\",
           r"\midrule"]
    for f in fs:
        c = []
        for key, fmt in (("peak_dev_mm", "%.0f"), ("recover_s", "%.2f")):
            for arm in ORDER:
                mu, hr = _ms([p[key] for p in d[arm].get(f, [])])
                c.append("n/a" if not np.isfinite(mu) else
                         (fmt % mu) + ((" ± " + fmt % hr) if hr else ""))
        nf = "%d/%d" % (sum(fell["stand"].get(f, [])),
                        sum(fell["brace"].get(f, [])))
        txt.append("%-10.0f %14s %14s %14s %14s %12s"
                   % (f, c[0], c[1], c[2], c[3], nf))
        tex.append("%.0f & %s & %s & %s & %s & %s \\\\"
                   % (f, c[0].replace("±", r"$\pm$"),
                      c[1].replace("±", r"$\pm$"),
                      c[2].replace("±", r"$\pm$"),
                      c[3].replace("±", r"$\pm$"), nf))
    tex += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(out, "table_disturbance.tex"), "w") as f_:
        f_.write("\n".join(tex) + "\n")
    body = "\n".join(txt)
    with open(os.path.join(out, "table_disturbance.txt"), "w") as f_:
        f_.write(body + "\n")
    print("\n" + body + "\n")
    print("  wrote table_disturbance.tex / .txt")


def make_all(json_path, out):
    rows = json.load(open(json_path))
    os.makedirs(out, exist_ok=True)
    print("plotting %d rollouts -> %s" % (len(rows), out))
    fig_envelope(rows, out)
    fig_maxreach(rows, out)
    fig_load_vs_margin(rows, out)
    fig_stability(rows, out)
    fig_margin_series(rows, out)
    fig_disturb(rows, out)
    fig_region(rows, out)
    write_table(rows, out)
    write_disturb_table(rows, out)
