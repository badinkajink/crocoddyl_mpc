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
# HOW THEY ARE COUNTED (revised 2026-08-23).  They used to be dropped from every
# aggregate.  They are now POOLED, because "the trunk is on the table" is a
# statement about which body carries the load and not about whether the rollout
# is valid: the robot is upright (pelvis 0.97-1.03 m), it holds the target to
# 4-16 mm, and nothing in the stack trips on a torso contact.  Excluding them
# left the commanded brace with no clean-arrival band at all -- an artifact of
# the filter, not a property of the planner.
#
# What is NOT hidden by pooling: `trunk load` and `peak contact load` are
# reported as their own rows, the trunk-assisted count stays in the table, and
# every scatter still draws them as rings.  The number that qualifies the whole
# decision is the PEAK, not the settled mean -- the settled trunk load is
# 75-472 N but the transient that puts the chest there reaches 1107 N, 1.6x body
# weight, in the max-reach rollouts.  A reader can see that and disagree.
TRUNK_N = 15.0


def trunk_load(r):
    b = r.get("brace_load_bodies") or {}
    return b.get("torso_link", 0.0) + b.get("pelvis", 0.0)


def arm_load(r):
    b = r.get("brace_load_bodies") or {}
    return sum(v for k, v in b.items()
               if "torso" not in k and "pelvis" not in k)


# lean_simple.cc: brace_arm = 0 means the LEFT arm braces and the right reaches.
BRACE_SIDE = "left_"


def brace_arm_load(r):
    """Newtons through the BRACING arm only.

    Not the same as `brace_load_N`, which is every non-foot contact and so also
    counts the reaching arm resting on the slab (8.9 N right gripper in one
    commanded-brace run) and the trunk in the degenerate ones. This is the
    number that says how hard the brace is actually working."""
    b = r.get("brace_load_bodies") or {}
    return sum(v for k, v in b.items() if k.startswith(BRACE_SIDE))


def peak_load(r):
    """The LARGEST non-foot contact load anywhere in the run, not the settled
    mean. Pooling trunk-rest rollouts is defensible on the settled number and
    only on that: the chest arriving on the slab is a transient that the mean
    over the last two fifths of the episode never sees."""
    bl = r.get("brace_load_t")
    return float(np.max(bl)) if bl else float("nan")


def contact_class(r):
    if trunk_load(r) > TRUNK_N:
        return "torso"
    return "brace" if arm_load(r) >= TRUNK_N else "free"


def by(rows, stage, clean=False, upright=True):
    """Rollouts of a stage.

    `clean` drops the torso-rest rollouts; it defaults OFF since 2026-08-23 --
    see the TRUNK_N block for why they are pooled and what is reported instead.
    `upright` drops rollouts that
    FELL, and it defaults on for the same reason: a settled metric read off a
    robot lying on the floor is not that posture's number. It was implicit
    while nothing fell -- the sampling planner's settled conditions are all
    upright -- and stopped being implicit with the gradient planner, whose
    braced max-reach rollouts fall in 4 of 4 and contributed a `bracing-arm
    force` of 363 N that is the weight of a collapsed robot resting on its own
    arm. The disturbance figures pass `upright=False`: there, whether a run
    fell IS the measurement, and they count and annotate it.

    A FALL AND A TRUNK REST ARE NOT THE SAME EVENT and that is why only one of
    them is still filtered. A fallen robot is not in the posture the metric
    names; a robot resting its chest on the table is -- it is upright, on
    target, and merely bracing with a body the mode did not ask for.
    """
    out = [r for r in rows if r.get("stage") == stage and "error" not in r]
    if upright:
        out = [r for r in out if not r.get("fell")]
    return [r for r in out if contact_class(r) != "torso"] if clean else out


# --------------------------------------------------------------------------- #
def arm_braced(r):
    """Did this rollout brace with the arm it was told to use?

    The ring predicate, split out on 2026-08-23. It used to be implicit: `by()`
    dropped trunk rests, so "was this point dropped" and "did it brace with the
    commanded arm" were the same question and the figures asked the first one.
    Pooling separated them, and the figures want the second -- a trunk rest is
    in the mean now, and the ring is what says it got there another way."""
    return not r.get("fell") and contact_class(r) != "torso"


def sweep_all(rows, arm):
    """Every swept rollout of one arm, UNFILTERED, keyed by target x."""
    d = defaultdict(list)
    for r in rows:
        if (r.get("stage") == "sweep" and r.get("arm") == arm
                and "error" not in r):
            d[round(r["target"][0], 3)].append(r)
    return d


def fig_envelope(rows, out):
    """The headline: how far out each arm still ARRIVES.

    TRUNK RESTS ARE MARKED, NOT DROPPED (revised 2026-08-23). The curves
    average every upright rollout. A chest on the table is a real arrival --
    the robot is upright and on target -- so its reach error is that posture's
    number and belongs in the mean; what it is NOT is the arm the mode asked
    for. So every point whose replicates did not all brace with the commanded
    arm gets a ring, and a target where none did gets a cross on the axis: the
    curve says how well the runs arrived, the rings say what they arrived on.
    Measured here, the sampling planner's braced arm rests its trunk at three
    of seven targets outright and at three more in one replicate of two.
    """
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

    marked = False
    for arm in ORDER:
        allx = sweep_all(rows, arm)
        for x in sorted(allx):
            n_all = len(allx[x])
            n_ok = sum(1 for r in allx[x] if arm_braced(r))
            if n_ok == n_all:
                continue
            marked = True
            if d[arm].get(x):
                v = _ms([r["reach_err_settled"] for r in d[arm][x]])[0] * 100
                ax[0].plot([x], [v], marker="o", ms=11, mfc="none",
                           mec=C[arm], mew=1.4, zorder=5)
                v2 = _ms([r.get("func_reach_settled")
                          for r in d[arm][x]])[0] * 100
                ax[1].plot([x], [v2], marker="o", ms=11, mfc="none",
                           mec=C[arm], mew=1.4, zorder=5)
            else:
                for a_ in ax:
                    a_.plot([x], [a_.get_ylim()[0]], marker="x", ms=7,
                            color=C[arm], mew=1.6, zorder=5, clip_on=False)
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
    h, l = ax[0].get_legend_handles_labels()
    if marked:
        h += [plt.Line2D([], [], ls="", marker="o", ms=9, mfc="none",
                         mec=INK2, mew=1.4),
              plt.Line2D([], [], ls="", marker="x", ms=7, color=INK2, mew=1.6)]
        l += ["some replicates degenerate", "all degenerate"]
    ax[0].legend(h, l, loc="upper left", fontsize=7.2)
    _save(fig, out, "fig_reach_envelope")


# The two margins answer different questions (see simple_stability.margins), so
# they get one panel set each rather than sharing a figure: side by side in one
# row they invite reading a 3 cm gap between them as a result, when the gap is
# just the difference between "any direction" and "toward the target".
# The qualifier rides in the title, not in a subtitle line: at ~1.7 in per panel
# a second line under the title lands on the neighbour's title every time.
MARGIN_VARIANTS = {
    "support": ("margin_actuated_settled", "Support margin (any dir.)"),
    "forward": ("fwd_actuated_settled", "Forward margin ($+x$)"),
}


# The settled conditions, in reach order, filtered at draw time by which ones
# the analysis actually holds. `nominal2` is a later addition (see
# brace_vs_stand.NOMINAL2_X) and every figure degrades to two groups without it,
# which is what makes older analysis.json files still plot.
CONDITIONS = [("nominal", "Targeted\nreach"),
              ("nominal2", "Extended\nreach"),
              ("maxreach", "Max\nreach")]


def fig_stability(rows, out, which="support"):
    """Small multiples, each metric grouped by CONDITION and coloured by arm.

    Two conditions, because they answer different halves of the question. At the
    study target both postures arrive (standing to within 6 mm), so that group
    isolates the STABILITY difference at equal task success. At max reach the
    target is unreachable and each posture stretches as far as it can, so that
    group is where the reach difference lives -- and its stability numbers are
    the ones that say what the extra reach cost."""
    mkey, mlabel = MARGIN_VARIANTS[which]
    groups = CONDITIONS
    have = [(g, lab) for g, lab in groups if by(rows, g)]
    if not have:
        return
    d = defaultdict(lambda: defaultdict(list))
    for g, _ in have:
        for r in by(rows, g):
            d[g][r["arm"]].append(r)

    panels = [
        ("func_reach_settled", "cm   (higher better)", 100.0,
         "Functional reach", "%.0f"),
        (mkey, "cm   (higher better)", 100.0, mlabel, "%.1f"),
        ("sparc_reach", "SPARC   (higher better)", 1.0, "Smoothness", "%.2f"),
        (brace_arm_load, "N", 1.0, "Bracing-arm force", "%.0f"),
    ]
    fig, axs = plt.subplots(1, len(panels), figsize=(TEXT, 2.7),
                            layout="constrained")
    W = 0.36
    for ax, (key, ylab, sc, title, fmt) in zip(axs, panels):
        for gi, (g, lab) in enumerate(have):
            for ai, arm in enumerate(ORDER):
                if callable(key):
                    vals = [key(r) for r in d[g][arm]]
                else:
                    vals = [r.get(key) for r in d[g][arm]
                            if r.get(key) is not None
                            and np.isfinite(r.get(key, np.nan))]
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
                neg = mu < 0
                ax.annotate(fmt % mu, xy=(x, mu - hr if neg else mu + hr),
                            xytext=(0, -3 if neg else 2.5),
                            textcoords="offset points", ha="center",
                            va="top" if neg else "bottom", fontsize=7.2,
                            color=INK, zorder=5)
        ax.set_xticks(range(len(have)))
        # Three two-line condition labels across a quarter of the text width
        # collide; the size steps down rather than the labels being abbreviated,
        # because "Ext." and "Tgt." are not words a reader should have to
        # decode from a caption.
        ax.set_xticklabels([lab for _, lab in have],
                           fontsize=6.8 if len(have) > 2 else 8)
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.margins(y=0.26)
        ax.grid(axis="x", visible=False)
    h = [plt.Rectangle((0, 0), 1, 1, color=C[a]) for a in ORDER]
    fig.legend(h, [LABEL[a] for a in ORDER], loc="outside lower center",
               ncol=2, fontsize=8)
    _save(fig, out, "fig_stability_panels_" + which)


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
        for key, ls, lw, a in (("fwd_actuated", "-", 1.9, 1.0),
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
    ax.set_xlabel("Time  [s]")
    ax.set_ylabel("CoM margin  [cm]")
    ax.set_title("Equilibrium margin over the reach", loc="left")
    h = [plt.Line2D([], [], color=C[a], lw=1.9, label=LABEL[a]) for a in ORDER]
    h += [plt.Line2D([], [], color=INK2, lw=1.9, label="forward margin"),
          plt.Line2D([], [], color=INK2, lw=1.5, ls=(0, (4, 2.5)),
                     label="support margin")]
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
            ("func_reach_settled", "Functional reach  [cm]",
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
        ax[i].set_xticklabels(["No brace\ncost", "Commanded\nbrace"])
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
    # Every point is in the fit since 2026-08-23. The ring still marks a
    # trunk rest, but it marks it -- it no longer removes it. If leaning on the
    # table buys margin, a chest is a lean like any other and belongs on the
    # trend; keeping it off was what made the trend look cleaner than it is.
    pts = allp
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
    ax.set_xlabel("Measured brace load  [N]")
    ax.set_ylabel("CoM margin  [cm]")
    ax.set_title("Margin follows load, not labels", loc="left")
    h = [plt.Line2D([], [], ls="", marker="o", ms=6, color=C[a],
                    label=LABEL[a]) for a in ORDER]
    if degen:
        h.append(plt.Line2D([], [], ls="", marker="x", ms=6, color=INK2,
                            label="torso on table (excluded)"))
    ax.legend(handles=h, loc="lower right", fontsize=6.8)
    _save(fig, out, "fig_load_vs_margin")


# The two push conditions, same ladder at two targets. `disturb2` is a later
# addition (see brace_vs_stand.NOMINAL2_X) and the figure simply does not appear
# when an analysis holds no such rollouts.
DISTURB_STAGES = [("disturb", "fig_disturbance", "targeted reach"),
                  ("disturb2", "fig_disturbance_ext", "extended reach")]


def fig_disturb(rows, out, stage="disturb", name="fig_disturbance",
                where="targeted reach"):
    """Push rejection: peak deviation and recovery time vs push magnitude."""
    dis = by(rows, stage, upright=False)
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
    ax[0].set_xlabel("Push at the reaching hand  [N]")
    ax[0].set_ylabel("Peak hand deflection  [mm]")
    ax[0].set_title("How far the push moves the hand (%s)" % where,
                    loc="left", fontsize=8.5)
    ax[1].set_xlabel("Push at the reaching hand  [N]")
    ax[1].set_ylabel("Recovery time  [s]")
    ax[1].set_title("Post-disruption recovery", loc="left")
    ax[0].legend(loc="upper left")
    _save(fig, out, name)


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
    ax.set_xlabel("World $x$  [m]")
    ax.set_ylabel("World $y$  [m]")
    ax.set_title("Admissible CoM region", loc="left")
    ax.legend(loc="lower left", fontsize=7.2)
    _save(fig, out, "fig_equilibrium_region")


def fig_success(rows, out):
    """Max-reach outcome rate. On hardware the braced max reach worked 11/15, so
    the sim number worth reporting is the same one: of N attempts, how many
    produced a legitimate braced reach rather than a chest on the table.

    TWO BARS PER ARM since 2026-08-23, because "success" got two meanings when
    trunk rests stopped being excluded. The full bar is UPRIGHT attempts: the
    robot is standing and on target, however it got there. The hatched part of
    it is the share that arrived on the trunk rather than on the commanded arm
    -- still an arrival, still not the behaviour that was asked for, and on a
    real table a very different thing. Reporting only the outer bar credits the
    planner with a brace it did not perform; reporting only the inner one hides
    that the robot did the task."""
    mr = [r for r in rows if r.get("stage") == "maxreach" and "error" not in r]
    if not mr:
        return
    fig, ax = plt.subplots(1, 2, figsize=(TEXT * 0.66, 2.6),
                           layout="constrained")
    for ai, arm in enumerate(ORDER):
        g = [r for r in mr if r["arm"] == arm]
        if not g:
            continue
        n = len(g)
        up = [r for r in g if not r.get("fell")]
        clean = sum(1 for r in up if contact_class(r) != "torso")
        ax[0].bar(ai, 100.0 * len(up) / n, width=0.55, color=C[arm], zorder=3,
                  edgecolor="white", linewidth=1.2, alpha=0.35)
        ax[0].bar(ai, 100.0 * clean / n, width=0.55, color=C[arm], zorder=4,
                  edgecolor="white", linewidth=1.2)
        ax[0].annotate("%d/%d" % (clean, n), xy=(ai, 100.0 * clean / n),
                       xytext=(0, 3), textcoords="offset points",
                       ha="center", va="bottom", fontsize=8, color=INK,
                       zorder=6)
        if len(up) > clean:
            ax[0].annotate("%d/%d upright" % (len(up), n),
                           xy=(ai, 100.0 * len(up) / n), xytext=(0, 3),
                           textcoords="offset points", ha="center",
                           va="bottom", fontsize=7, color=INK2, zorder=6)
        # reach achieved, upright attempts -- a trunk rest reached that far
        v = [r["func_reach_settled"] * 100 for r in up
             if r.get("func_reach_settled") is not None]
        mu, hr = _ms(v)
        if np.isfinite(mu):
            ax[1].bar(ai, mu, width=0.55, color=C[arm], zorder=3,
                      edgecolor="white", linewidth=1.2)
            if hr > 0:
                ax[1].errorbar(ai, mu, yerr=hr, color=INK2, lw=1.0, capsize=3,
                               zorder=4)
            ax[1].annotate("%.0f" % mu, xy=(ai, mu + hr), xytext=(0, 3),
                           textcoords="offset points", ha="center",
                           va="bottom", fontsize=8, color=INK)
    for i, (ylab, title) in enumerate([
            ("% of attempts   (higher better)", "Max-reach outcome"),
            ("Functional reach  [cm]", "Reach (upright attempts)")]):
        ax[i].set_xticks(range(len(ORDER)))
        ax[i].set_xticklabels(["No brace\ncost", "Commanded\nbrace"])
        ax[i].set_ylabel(ylab)
        ax[i].set_title(title, loc="left")
        ax[i].margins(y=0.24)
        ax[i].grid(axis="x", visible=False)
    ax[0].set_ylim(0, 105)
    _save(fig, out, "fig_maxreach_success")


def write_table(rows, out):
    """The numbers behind the figures, at BOTH conditions, as LaTeX and text."""
    conds = [(g, lab.replace("\n", " ")) for g, lab in CONDITIONS
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
        ("brace load, all contacts", "brace_load_N", 1.0, "N", "%.0f"),
        ("bracing-arm force", brace_arm_load, 1.0, "N", "%.0f"),
        ("trunk load", trunk_load, 1.0, "N", "%.0f"),
        ("peak contact load", peak_load, 1.0, "N", "%.0f"),
        ("support margin (contact)", "margin_contact_settled", 100.0, "cm", "%.1f"),
        ("support margin (actuated)", "margin_actuated_settled", 100.0, "cm", "%.1f"),
        ("forward margin (contact)", "fwd_contact_settled", 100.0, "cm", "%.1f"),
        ("forward margin (actuated)", "fwd_actuated_settled", 100.0, "cm", "%.1f"),
        ("smoothness SPARC", "sparc_reach", 1.0, "--", "%.2f"),
        ("hand jitter", "hand_jitter_mm", 1.0, "mm", "%.2f"),
        ("reach error spread", "reach_err_std_mm", 1.0, "mm", "%.2f"),
        ("worst-case push at hand", "push_min_N", 1.0, "N", "%.0f"),
        ("settling time (initial)", "settle_time_s", 1.0, "s", "%.1f"),
        ("peak torque ratio", "peak_tau_ratio", 1.0, "--", "%.2f"),
    ]

    def cell(g, arm, key, sc, fmt):
        vals = ([key(r) for r in d[g][arm]] if callable(key)
                else [r.get(key) for r in d[g][arm] if r.get(key) is not None])
        mu, hr = _ms(vals)
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
    # TRUNK-ASSISTED IS COUNTED OVER UPRIGHT ROLLOUTS ONLY. A robot on the
    # floor has its torso in contact by definition, so counting falls here read
    # "4/4 trunk-assisted" for a condition in which nothing braced and
    # everything collapsed. Falls get their own row.
    dcells, fcells = [], []
    for g, _ in conds:
        for arm in ORDER:
            allr = [r for r in rows if r.get("stage") == g
                    and r.get("arm") == arm and "error" not in r]
            up = [r for r in allr if not r.get("fell")]
            dcells.append("%d/%d" % (sum(contact_class(r) == "torso"
                                         for r in up), len(up))
                          if up else "n/a")
            fcells.append("%d/%d" % (len(allr) - len(up), len(allr)))
    txt.append(" ".join(["%-24s" % "trunk-assisted (pooled)"]
                        + ["%13s" % c for c in dcells] + ["%6s" % "--"]))
    txt.append(" ".join(["%-24s" % "fell (excluded)"]
                        + ["%13s" % c for c in fcells] + ["%6s" % "--"]))
    tex.append("trunk-assisted (pooled) & %s & -- \\\\" % " & ".join(dcells))
    tex.append("fell (excluded) & %s & -- \\\\" % " & ".join(fcells))
    tex.append("runs bracing ($>$15\\,N) & %s & -- \\\\" % " & ".join(cells))
    tex += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(out, "table_brace_vs_stand.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")
    body = "\n".join(txt)
    with open(os.path.join(out, "table_brace_vs_stand.txt"), "w") as f:
        f.write(body + "\n")
    print("\n" + body + "\n")
    print("  wrote table_brace_vs_stand.tex / .txt")


def write_disturb_table(rows, out, stage="disturb", name="table_disturbance"):
    """Post-disruption response, per push magnitude.

    Kept separate from the main table on purpose: `settling time (initial)`
    there is how long the posture takes to establish itself from the home
    keyframe, which is a different quantity from how long it takes to come back
    after being shoved. Conflating them was a misreading worth designing out --
    the brace is SLOWER to establish (the arm must seat first) and the claim
    under test is that it is FASTER to recover."""
    dis = by(rows, stage, upright=False)
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
    with open(os.path.join(out, name + ".tex"), "w") as f_:
        f_.write("\n".join(tex) + "\n")
    body = "\n".join(txt)
    with open(os.path.join(out, name + ".txt"), "w") as f_:
        f_.write(body + "\n")
    print("\n" + body + "\n")
    print("  wrote %s.tex / .txt" % name)


# --------------------------------------------------------------------------- #
# Hardware, drawn beside the simulation.
#
# Allen's real-robot runs of the COMMANDED BRACE, transcribed from
# paper/figures/brace_reach/allen_real_experiments/brace_reach_real_{1,2}.png
# (max reach) and target_position_metrics.jpeg (targeted). Values are already in
# DISPLAY units (the sim rows are SI and get scaled at draw time), so the two
# paths through the panel loop differ and are kept visibly separate rather than
# merged into a fake rollout record.
#
# FOUR CAVEATS THAT BELONG IN THE CAPTION, NOT JUST HERE:
#  1. The spread convention may not match. Sim bars are HALF-RANGE, (max-min)/2
#     (see `_ms`: 2-8 replicates cannot support a standard error). The hardware
#     "+-" is transcribed as given; if it is an SD the two bar kinds are not
#     the same statistic and the figure should say so.
#  2. `n` is not recorded in either source table. The hardware max-reach attempt
#     rate quoted elsewhere in this study is 11/15; whether these means are over
#     11 runs or some subset is unknown here.
#  3. Only the SUPPORT margin was reported, at either pose. There is no hardware
#     forward-margin number, so `which="forward"` draws the sim pair alone --
#     deliberately, rather than reusing the support value under another name.
#  4. Hand jitter exists only at max reach, and see the panel's own comment for
#     why that number is not what its name suggests on the sim side.
#
# The two hardware poses are nearly the same reach -- 98.8 cm targeted against
# 99.1 cm at max -- which is not an error: Allen chose the targeted pose near
# the limit "to show that we can reach that far".
# Legend labels for the hardware comparison ONLY. Every other figure in this set
# still reads "Commanded brace"; renaming in `LABEL` would silently retitle nine
# figures whose captions are already written against that wording, so the
# override is scoped here and the rename propagates when it is decided to.
# (Case is as requested and deliberately not "fixed": "Brace Encouraged" beside
# "No brace cost" is a mixed-case legend pair -- worth a look before print.)
REAL_ARM_LABEL = {"stand": "No brace cost", "brace": "Brace Encouraged"}
REAL_LABEL = "Brace Encouraged (real)"
# Categorical slot 3 (aqua) of the documented reference palette, not an
# arbitrary green. Slots 1-3 are the validated all-pairs opening -- worst pair
# CVD dE 9.2, normal-vision 24.0 on a light surface -- which is the gate that
# applies here, because all three bars sit side by side inside one group and so
# every pair is adjacent.
C_REAL = "#1baf7a"
REAL = {
    "realpose": {
        "hand_x_settled": (98.8, 1.7),
        "margin_actuated_settled": (16.1, 1.1),
        "sparc_reach": (-3.44, 0.45),
        "brace_arm_load": (108.0, 23.0),
        # No jitter reported at the targeted pose. The panel simply has no
        # green bar there rather than borrowing the max-reach value.
    },
    "maxreach": {
        "hand_x_settled": (99.1, 2.7),
        "margin_actuated_settled": (14.3, 1.2),
        "sparc_reach": (-3.85, 0.75),
        "hand_jitter_mm": (25.6, 3.9),
        "brace_arm_load": (109.0, 28.0),
    },
}

# REACH IS TIP x, NOT THIS STUDY'S FUNCTIONAL REACH. The two are different
# measurements and mixing them was an error in the first version of this figure.
#
# `func_reach_settled` is the horizontal hand-to-ankle-midpoint distance, chosen
# deliberately (see brace_vs_stand.analyse_one) because world hand x moves when
# the base does. Allen's "functional reach" is the tip's x in the robot base
# frame. Measured against the same rollouts they differ by the ankle offset:
#
#     realpose   81 cm ankle-mid  vs  99.5 cm tip x   -> 18.8 cm
#     maxreach  111 cm ankle-mid  vs  128  cm tip x   -> 16.6 cm
#
# Sim tip x at the hardware target is 99.5-99.9 cm against Allen's 98.8 +- 1.7,
# which is what identified the definition: 8 mm apart, where the ankle-mid
# reading is 18 cm out. Plotting 111/118 beside a hardware 99.1 -- as the first
# version did -- compared two different quantities and made the sim look CLOSER
# to hardware than it is. Same-definition, max reach is sim 128/132 vs real 99.
# Tip x is the only definition available on both sides, so it is what both sides
# are plotted in; the ankle-mid numbers stay in `table_brace_vs_stand`.
# HAND JITTER: mean radial deviation of the reaching-hand site about its own
# settled-window mean position, in mm (brace_vs_stand.analyse_one).
#
#     settle = t >= 0.6 * t_end        final 40% of the rollout; for the 20 s
#                                      settled conditions that is t >= 12.0 s,
#                                      800 samples at the 100 Hz dump rate
#     jitter = mean_k || p(k) - mean(p[settle]) ||   over k in settle
#
# It is a DISPERSION, not an accuracy: taken about the window's own centroid, a
# constant offset from the commanded target costs nothing (that is
# `reach_err_settled`) while a slow drift across the window costs a lot. It is a
# mean absolute deviation, not an RMS and not a peak-to-peak -- `precision_rms_mm`
# is the SD of the same radius if the second moment is wanted. In the disturbance
# stages each pulse and the 2 s after it are cut from the window first, so the
# number is always quiet-state jitter and never a rejection transient.
#
# It is kept ALONGSIDE SPARC rather than replaced by it because the two disagree
# about what "unsteady" means and the disagreement is informative: SPARC scores
# the whole reaching movement's speed profile and is amplitude- and
# duration-invariant, jitter scores only the hold and is neither.
REAL_PANELS = [
    ("hand_x_settled", 100.0, "cm   (higher better)",
     "Reach", "(tip $x$, base frame)", "%.0f"),
    # Title is overridden per margin variant at draw time -- see MARGIN_TITLE.
    ("margin_actuated_settled", 100.0, "cm   (higher better)",
     "Support margin", "(any direction)", "%.1f"),
    ("sparc_reach", 1.0, "SPARC   (higher better)",
     "Smoothness", "", "%.2f"),
    # WINDOW NAMED IN THE TITLE, because this number is not what it sounds
    # like. `hold_jitter.py` shows only 23% of settled rollouts ever reach a
    # steady hold in 20 s, and where one exists the RMS is 2-3 mm regardless of
    # condition. The 6-20 mm here is therefore mostly the hand still creeping
    # outward at t = 12-20 s, charged to jitter by taking a dispersion about
    # the window mean. It stays in the figure because it is the number the
    # hardware table can be lined up against; it is labelled so nobody reads it
    # as high-frequency shake.
    ("hand_jitter_mm", 1.0, "mm   (lower better)",
     "Hand jitter", "(last 40 % of run)", "%.1f"),
    # NOT "higher better". More newtons through the bracing arm is what the
    # posture costs, not what it achieves -- the source figure labelled this
    # panel "higher better" and that reading would have the degenerate
    # chest-on-the-table rollouts winning it outright at 300-470 N.
    ("brace_arm_load", 1.0, "N",
     "Bracing-arm force", "", "%.0f"),
]

# Group centres and slot width. Five panels across one text width leaves ~1.15 in
# of drawing area each, so the bars are thin and the two conditions sit close
# together: at the previous 0.36 slot the three-bar max-reach group ran into its
# neighbour's tick label.
REAL_GAP, REAL_W = 0.72, 0.20

# Two-line forms of MARGIN_VARIANTS' labels. The one-line versions overrun a
# 1.15 in panel, and a margin panel silently mislabelled as the OTHER margin is
# the single most misleading thing this figure could do -- the two differ by a
# factor of ~1.5 and the paper quotes both.
MARGIN_TITLE = {
    "support": ("Support margin", "(any direction)"),
    "forward": ("Forward margin", "(toward target)"),
}


# THE TARGETED GROUP IS THE HARDWARE'S TARGET, not the task XML's.
#
# `nominal` is the study target at x = 0.9047, which is not where the real robot
# reached -- Allen's runs used (0.998, -0.156, 1.163), confirmed to 2 mm in x
# and z (see brace_vs_stand.REALPOSE_TARGET). Putting `nominal` in a figure
# whose whole purpose is a hardware comparison would compare two different
# reaches and call the difference a result. `realpose` runs the same two arms at
# the target hardware was actually given, so every column of this figure is at a
# pose the hardware also attempted, and the green bars drop in when they exist.
#
# Kept separate from the shared CONDITIONS list on purpose: adding a fourth
# group there would silently retitle the nine sim-only figures built on it.
REAL_CONDITIONS = [("realpose", "Targeted\nreach"), ("maxreach", "Max\nreach")]


def _real_value(cond, key, which):
    """The hardware bar for one panel, or None if hardware has no such number."""
    if which != "support" and key == "margin_actuated_settled":
        return None
    return REAL.get(cond, {}).get(key)


def fig_stability_real(rows, out, which="support"):
    """The stability panels with the hardware runs drawn in beside the sim.

    Same measurements as `fig_stability` plus hand jitter, restricted to the two
    conditions hardware can be compared against, and with the commanded brace
    appearing twice -- once as simulated, once as measured. The sim-vs-real pair
    is the point of the figure, so the two conditions are drawn tight and the
    bars thin rather than dropping a panel to make room.
    """
    mkey, _ = MARGIN_VARIANTS[which]
    # clean=True PASSED EXPLICITLY, not inherited. `by()`'s default flipped to
    # pooling trunk rests (ringed rather than dropped) while this figure was
    # being written, which is right for the envelope curves and wrong here:
    # pooling would credit "Brace Encouraged" with a margin earned by lying on
    # the table (23.1 cm pooled vs 18.5 cm clean at max reach, n=16 vs n=6),
    # and the hardware bar it is being compared against is certainly not doing
    # that. Pinned so a future default change cannot move these bars silently.
    groups = [(g, lab) for g, lab in REAL_CONDITIONS
              if by(rows, g, clean=True)]
    if not groups:
        return
    d = defaultdict(lambda: defaultdict(list))
    for g, _ in groups:
        for r in by(rows, g, clean=True):
            d[g][r["arm"]].append(r)

    fig, axs = plt.subplots(1, len(REAL_PANELS), figsize=(TEXT, 3.1),
                            layout="constrained")
    for ax, (key, sc, ylab, t1, t2, fmt) in zip(axs, REAL_PANELS):
        k = key
        if key == "margin_actuated_settled":
            k = mkey
            t1, t2 = MARGIN_TITLE[which]
        for gi, (g, lab) in enumerate(groups):
            # Series present in THIS group. The bars are centred on the group,
            # so the targeted-reach pair is not left with a hole where the
            # hardware bar would go -- there is no hardware run there to imply.
            series = []
            for arm in ORDER:
                if key == "brace_arm_load":
                    vals = [brace_arm_load(r) for r in d[g][arm]]
                else:
                    vals = [r.get(k) for r in d[g][arm]
                            if r.get(k) is not None
                            and np.isfinite(r.get(k, np.nan))]
                mu, hr = _ms([v * sc for v in vals])
                if np.isfinite(mu):
                    series.append((C[arm], REAL_ARM_LABEL[arm], mu, hr))
            rv = _real_value(g, key, which)
            if rv is not None:
                series.append((C_REAL, REAL_LABEL, rv[0], rv[1]))
            n = len(series)
            for si, (col, _lab, mu, hr) in enumerate(series):
                x = gi * REAL_GAP + (si - (n - 1) / 2.0) * REAL_W
                ax.bar(x, mu, width=REAL_W * 0.88, color=col, zorder=3,
                       edgecolor="white", linewidth=0.8)
                if hr > 0:
                    ax.errorbar(x, mu, yerr=hr, color=INK2, lw=0.9,
                                capsize=2.0, zorder=4)
                # Value labels ZIGZAG. At this width a slot is ~9.6 pt across
                # and a three-character label at a legible size is wider than
                # that, so neighbours inside a group WILL overlap horizontally
                # no matter how the bars are spaced. Alternating the vertical
                # offset separates them along the other axis instead of
                # shrinking the type to 5.5 pt, which does not print.
                neg = mu < 0
                dy = 2.0 + 9.0 * (si % 2)
                ax.annotate(fmt % mu, xy=(x, mu - hr if neg else mu + hr),
                            xytext=(0, -dy if neg else dy),
                            textcoords="offset points", ha="center",
                            va="top" if neg else "bottom", fontsize=6.4,
                            color=INK, zorder=5)
        ax.set_xticks([gi * REAL_GAP for gi in range(len(groups))])
        ax.set_xticklabels([lab for _, lab in groups], fontsize=7.4)
        ax.set_xlim(-0.46, (len(groups) - 1) * REAL_GAP + 0.46)
        ax.set_ylabel(ylab, fontsize=7.8)
        ax.tick_params(labelsize=7.2)
        # Every title carries two lines even when the second is blank. Under
        # constrained layout a taller title shrinks its own axes, so a mix of
        # one- and two-line titles leaves the five panels with visibly
        # different plot heights.
        ax.set_title("%s\n%s" % (t1, t2), fontsize=8.0)
        ax.margins(y=0.30)
        ax.grid(axis="x", visible=False)

    h = [plt.Rectangle((0, 0), 1, 1, color=C[a]) for a in ORDER]
    lab = [REAL_ARM_LABEL[a] for a in ORDER]
    if any(_real_value(g, k[0], which) is not None
           for g, _ in groups for k in REAL_PANELS):
        h.append(plt.Rectangle((0, 0), 1, 1, color=C_REAL))
        lab.append(REAL_LABEL)
    fig.legend(h, lab, loc="outside lower center", ncol=len(h), fontsize=8)
    _save(fig, out, "fig_stability_panels_real_" + which)


def make_all(json_path, out):
    rows = json.load(open(json_path))
    os.makedirs(out, exist_ok=True)
    print("plotting %d rollouts -> %s" % (len(rows), out))
    fig_envelope(rows, out)
    fig_maxreach(rows, out)
    fig_load_vs_margin(rows, out)
    fig_success(rows, out)
    for _w in MARGIN_VARIANTS:
        fig_stability(rows, out, _w)
    # The hardware comparison lives in its own subdirectory -- it is the only
    # figure in this set that is not pure simulation, and the transcribed
    # numbers in `REAL` have a provenance the rest of the tree does not share.
    # One canonical copy, written by the normal pipeline, so it cannot drift
    # from the analysis.json the sim bars came from.
    real_out = os.path.join(out, "allen_real_experiments")
    os.makedirs(real_out, exist_ok=True)
    for _w in MARGIN_VARIANTS:
        fig_stability_real(rows, real_out, _w)
    fig_margin_series(rows, out)
    for _s, _n, _w in DISTURB_STAGES:
        fig_disturb(rows, out, _s, _n, _w)
    fig_region(rows, out)
    write_table(rows, out)
    for _s, _n, _w in DISTURB_STAGES:
        write_disturb_table(rows, out, _s, _n.replace("fig_disturbance",
                                                      "table_disturbance"))


# `brace_vs_stand.py plot` is the usual entry point; this one exists so a single
# figure can be redrawn without re-running the whole set (the region and strip
# figures load the model and take minutes).
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="all",
                    help="all | real | stability | table")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.only == "all":
        make_all(a.json, a.out)
        return
    rows = json.load(open(a.json))
    if a.only == "real":
        for w in MARGIN_VARIANTS:
            fig_stability_real(rows, a.out, w)
    elif a.only == "stability":
        for w in MARGIN_VARIANTS:
            fig_stability(rows, a.out, w)
    elif a.only == "table":
        write_table(rows, a.out)
    else:
        raise SystemExit("unknown --only %s" % a.only)


if __name__ == "__main__":
    main()
