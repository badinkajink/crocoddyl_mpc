#!/usr/bin/env python3
"""Three-row film strip: the sim's targeted reach beside the hardware's.

`bvs_strips.py` renders the six-row ablation over every strategy the planner
found. This is the SIDE-BY-SIDE cut of it -- one condition, both plants -- so a
reader can put the rendered robot next to the photographed one and see whether
the posture the study reports is the posture the robot adopts.

WHY THE SIM ROWS ARE `maxreach` AND NOT `realpose`. The hardware's targeted
reach IS its max reach: Allen ran one target and the robot stretched to it.
In sim those are two different conditions, and it is the MAXREACH rollouts that
look like the hardware's trial -- deep torso pitch, arm out across the slab --
while the sim rollouts at the hardware's own target (`realpose`) arrive early
and stand comparatively upright. That is the sim2real gap this study has been
chasing, not a labelling accident, and the strip is honest about it by naming
both sim rows "Targeted Reach" and letting the frames disagree with the number
in `MAXREACH_TARGET`. Do not "fix" this by swapping in the realpose rollouts;
the two robots would then be doing visibly different things under one label.

THE COLUMNS ARE PHASE-ALIGNED, NOT TIME-ALIGNED. The sim rollouts run 20 s and
the hardware clip runs 45+, so there is no shared clock to put along the top.
Both clocks are drawn -- sim above the first row, hardware below the last -- and
a column means "same stage of the maneuver", nothing more.

FORCES ARE BRACING-ARM ONLY, in all three rows. The six-row strip annotates
`brace_load_N` (every non-foot contact, so it also counts the reaching hand
resting on the slab and the trunk in the degenerate runs). The hardware number
is the bracing arm alone, so quoting the two side by side would compare
different quantities: `brace_arm_load` is used throughout instead, which drops
the sim commanded-brace exemplar from 152 N to 117 N against the hardware's
109 +- 28 N.

NO MARKER LEGEND, by request -- the contact dots are hard to read at this size.
They are still DRAWN and still sized by normal force, so whatever caption this
figure gets has to say what they are; nothing on the page does.

usage:
  bvs_strip_real.py --json analysis.json --run DIR --frames DIR --out DIR
"""
import argparse
import json
import os

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import simple_lean as sl
import simple_video as sv
import bvs_plots as BP
import bvs_strips as ST

# Allen's clip, as given. Not derived from anything in this repo.
REAL_TIMES = (0.0, 15.0, 25.0, 35.0, 45.0)
REAL_FILES = ("1.png", "2.png", "3.png", "4.png", "5.png")
STAGE = "maxreach"
# "Targeted Reach" rather than "Max reach": on hardware the two are the same
# experiment, and a row labelled "max reach" beside a row labelled "targeted"
# would read as a second condition that was never run.
STAGE_LABEL = "Targeted Reach"
# Sim wording tracks `bvs_plots.REAL_ARM_LABEL` so the strip and the bar figure
# cannot drift apart; the hardware row tracks `REAL_LABEL`.
ROW_SIM = [("stand", BP.REAL_ARM_LABEL["stand"]),
           ("brace", BP.REAL_ARM_LABEL["brace"])]


def pick_sim(res, run_dir):
    """Median-load exemplar per arm, trunk-rests excluded.

    Same rule as the six-row strip -- chosen by MEASURED load, not by the
    weights the run was given -- so the two figures cannot pick different
    rollouts for the same condition. Trunk rests are dropped here even though
    the bar figures pool them: this row is a picture of a posture, and 12 of 16
    commanded-brace rollouts at long range finish with the chest on the slab
    (see the `Trunk Clear` trap in CLAUDE.md). Drawing one of those and calling
    it "Brace Encouraged" would illustrate the cost function's ceiling, not the
    brace."""
    out = []
    for arm, lab in ROW_SIM:
        g = [r for r in res if r.get("stage") == STAGE and r.get("arm") == arm
             and "error" not in r and not r.get("fell")
             and BP.contact_class(r) != "torso"]
        if not g:
            raise SystemExit("no clean %s/%s rollouts in the json" % (STAGE, arm))
        r = sorted(g, key=lambda z: BP.brace_arm_load(z))[len(g) // 2]
        out.append(("%s\n%s" % (STAGE_LABEL, lab),
                    os.path.join(run_dir, r["tag"] + ".csv"), r))
    return out


def load_real(frames_dir):
    """The hardware frames, and the aspect the sim must be rendered at.

    Matching the sim renderer to the PHOTO's aspect rather than cropping the
    photo to the renderer's: a crop that keeps the aspect has to come off the
    top or the bottom, and the robot fills the frame head to feet in every one
    of these five. Nothing is discarded this way."""
    import matplotlib.image as mpimg
    ims = [mpimg.imread(os.path.join(frames_dir, f)) for f in REAL_FILES]
    h, w = ims[0].shape[:2]
    for im in ims:
        if im.shape[:2] != (h, w):
            raise SystemExit("hardware frames differ in size; expected %dx%d"
                             % (w, h))
    return ims, w / float(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--frames", required=True,
                    help="directory holding %s" % ", ".join(REAL_FILES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--az", type=float, default=120.0)
    # Tighter than the six-row strip's 2.75. Three rows at text width give each
    # tile 1.7x the height the ablation's tiles get, and the wide framing that
    # reads fine at a sixth of the page leaves the robot a thumbnail here.
    ap.add_argument("--dist", type=float, default=2.15)
    ap.add_argument("--fracs", default="0.0,0.15,0.35,0.65,1.0")
    a = ap.parse_args()

    res = json.load(open(a.json))
    picks = pick_sim(res, a.run)
    real, ar = load_real(a.frames)
    fracs = [float(x) for x in a.fracs.split(",")]
    if len(fracs) != len(REAL_TIMES):
        raise SystemExit("need %d sim fracs to face %d hardware frames"
                         % (len(REAL_TIMES), len(REAL_TIMES)))
    os.makedirs(a.out, exist_ok=True)

    # Render the sim AT THE PHOTO'S ASPECT. The six-row strip does not do this
    # and letterboxes as a result -- its figure geometry is computed from
    # bvs_strips.W/H (760x620) while sv.make_renderer uses simple_video's own
    # 900x700, which is where its row gaps come from. Here the two agree.
    sv.W = 900
    sv.H = int(round(sv.W / ar))
    m, d = sl.load()
    r = sv.make_renderer(m)
    cam = sv.cam(a.az, el=-10, dist=a.dist, look=(0.86, 0.0, 0.94))

    grid, sim_t = [], []
    for lab, path, meta in picks:
        col, rows, _ = sl.load_traj(path)
        t = rows[:, col["time"]]
        ks = [min(len(rows) - 1, int(f * (len(rows) - 1))) for f in fracs]
        grid.append([ST.frame_at(m, d, r, rows, col, k, a.az, camera=cam)
                     for k in ks])
        sim_t = [t[k] for k in ks]
    grid.append(real)

    labels = [p[0] for p in picks] + ["%s\n%s" % (STAGE_LABEL, BP.REAL_LABEL)]
    hw_mu, hw_sd = BP.REAL[STAGE]["brace_arm_load"]
    loads = ["%.0f N" % BP.brace_arm_load(p[2]) for p in picks]
    # The hardware cell is a MEAN over Allen's runs, not this clip -- these five
    # frames have no force trace attached to them. The "+-" is what says so.
    loads.append("%.0f ± %.0f N" % (hw_mu, hw_sd))

    nr, nc = len(grid), len(fracs)
    fig, axs = plt.subplots(nr, nc,
                            figsize=(BP.TEXT, BP.TEXT * nr / (nc * ar) * 1.06),
                            layout="constrained")
    axs = np.atleast_2d(axs)
    for i in range(nr):
        for j in range(nc):
            ax = axs[i, j]
            ax.imshow(grid[i][j])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(BP.GRID)
            if i == 0:
                ax.set_title("$t$ = %.1f s" % sim_t[j], fontsize=8,
                             color=BP.INK)
            if i == nr - 1:
                ax.set_xlabel("$t$ = %.0f s" % REAL_TIMES[j], fontsize=8,
                              color=BP.INK, labelpad=3)
        axs[i, 0].set_ylabel(labels[i], fontsize=7.6, color=BP.INK, rotation=0,
                             ha="right", va="center", labelpad=8)
        axs[i, 0].annotate(loads[i], xy=(0.03, 0.94), xycoords="axes fraction",
                           fontsize=7.5, color=BP.INK, va="top",
                           bbox=dict(boxstyle="round,pad=0.22", fc="white",
                                     ec=BP.GRID, lw=0.6))
    # Which clock is which. Without this the two time rows look like one axis
    # read twice, and column 3 reads as "the same instant", which it is not.
    axs[0, 0].annotate("simulation", xy=(-0.03, 1.10), xycoords="axes fraction",
                       ha="right", va="center", fontsize=7.2, color=BP.INK2,
                       style="italic", annotation_clip=False)
    axs[nr - 1, 0].annotate("hardware", xy=(-0.02, -0.14),
                            xycoords="axes fraction", ha="right", va="center",
                            fontsize=7.2, color=BP.INK2, style="italic",
                            annotation_clip=False)
    # SQUEEZE OUT THE ROW GAPS. `imshow` keeps the image aspect by shrinking the
    # AXES BOX, so constrained layout is left holding vertical slack it then
    # spreads as padding -- which is where the six-row strip's wide row gaps come
    # from. How much slack there is depends on the label gutter, which is only
    # known after a layout pass, so the height is corrected from the drawn
    # geometry rather than predicted: shrink until the inter-row gap is PAD.
    PAD = 0.05                                          # [in] wanted row gap
    for _ in range(6):
        fig.canvas.draw()
        figh = fig.get_figheight()
        ps = [axs[i, 0].get_position() for i in range(nr)]
        excess = sum((ps[i].y0 - ps[i + 1].y1) * figh - PAD
                     for i in range(nr - 1))
        if excess <= 0.01:
            break
        fig.set_figheight(figh - excess)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, "fig_strategy_strip_real." + ext),
                    facecolor="white", dpi=190)
    plt.close(fig)
    print("wrote fig_strategy_strip_real.pdf / .png  (%d rows)" % nr)
    for (lab, path, meta), ld in zip(picks, loads):
        print("   %-42s %-24s %8s   brace_load_N %5.0f  margin %.3f"
              % (lab.replace("\n", " / "), os.path.basename(path), ld,
                 meta["brace_load_N"],
                 meta.get("margin_actuated_settled", float("nan"))))
    print("   %-42s %-24s %8s"
          % ((STAGE_LABEL + " / " + BP.REAL_LABEL), a.frames, loads[-1]))
    print("   NOTE: contact dots are drawn but no longer captioned in-figure.")


if __name__ == "__main__":
    main()
