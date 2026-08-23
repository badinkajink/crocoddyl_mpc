#!/usr/bin/env python3
"""The five stages of the maneuver, from one CMPC trajectory.

The paper's Fig. 1 is five photographs of the hardware run: stand, lean, lean &
reach, recover, stand again. This is its simulated CMPC counterpart, and it is
drawn from ONE trajectory -- `cycle_brace_r*.csv`, the braced reach held and
then chained into the recovery -- rather than from five separate runs, because
the interesting instants are the transitions and a strip assembled from
independent runs cannot show one.

Frames are chosen by TIME rather than by phase index on purpose: the recovery is
entered by projection (the plan node nearest the measured pose), so the node
index is not a clock and two replicates would pick visually different instants
from the same nominal "stage". The seam time is read from the manifest and the
columns are placed around it.

Contact markers are bvs_strips' -- drawn at MuJoCo's narrowphase positions and
sized by normal force, so a hovering forearm cannot be mistaken for a seated one.

usage:
  cmpc_stage_strip.py --csv RUN/cycle_brace_r0.csv --out DIR
                      [--manifest RUN/manifest_cycle.json] [--az 120]
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import simple_lean as sl
import simple_video as sv
import bvs_strips as BS
import bvs_plots as BP

# Stage labels, in the order the maneuver visits them. `t_frac` is a fraction of
# the REACH leg for the first three and of the RECOVERY leg for the last two, so
# the same table works whatever the two legs' durations are.
STAGES = [("Stand", "reach", 0.00),
          ("Lean", "reach", 0.22),
          ("Lean \\& reach\n(held)", "reach", 0.85),
          ("Recover", "recover", 0.35),
          ("Stand again", "recover", 1.00)]


def pick_times(t, t_seam):
    """Absolute times for the five stages, from the seam."""
    t0, t1 = float(t[0]), float(t[-1])
    out = []
    for _, leg, frac in STAGES:
        a, b = (t0, t_seam) if leg == "reach" else (t_seam, t1)
        out.append(a + frac * (b - a))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--az", type=float, default=120.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    col, rows, meta = sl.load_traj(a.csv)
    t = rows[:, col["time"]]
    t_seam = None
    if a.manifest and os.path.exists(a.manifest):
        tag = os.path.splitext(os.path.basename(a.csv))[0]
        for r in json.load(open(a.manifest)):
            if r.get("tag") == tag:
                t_seam = r["seam"]["t_seam"]
    if t_seam is None:
        # No manifest: fall back to the midpoint, and say so rather than
        # pretending the column times mean what they would with one.
        t_seam = 0.5 * (t[0] + t[-1])
        print("no manifest seam time -- using the midpoint, %.2f s" % t_seam)

    m, d = sl.load()
    r = sv.make_renderer(m)
    times = pick_times(t, t_seam)
    ks = [int(np.argmin(np.abs(t - tt))) for tt in times]
    # A TIGHTER SHOT THAN THE ABLATION STRIP'S. Five columns across the text
    # width give each tile ~1.4 in, and the six-row strip's wide framing puts
    # the robot inside a fifth of that. Same scene, same contact markers, closer
    # camera -- and the frame is identical across the five columns, so the
    # apparent motion is the robot's and not the camera's.
    cam = sv.cam(a.az, el=-8, dist=2.05, look=(0.86, 0.0, 1.00))
    imgs = [BS.frame_at(m, d, r, rows, col, k, a.az, camera=cam) for k in ks]

    # Height is the tile aspect PLUS fixed room for two title lines and the
    # legend. Scaling the whole figure by the tile aspect alone (what the
    # six-row strip does, correctly, because its titles are one row out of six)
    # leaves a one-row strip with 1.2 in for everything and the image squeezed
    # to a stamp.
    nc = len(imgs)
    tile_w = BP.TEXT / nc
    fig, axs = plt.subplots(1, nc,
                            figsize=(BP.TEXT,
                                     tile_w * BS.H / BS.W + 0.52 + 0.34),
                            layout="constrained")
    for j, (ax, img) in enumerate(zip(axs, imgs)):
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(BP.GRID)
        ax.set_title("%s\n$t$ = %.1f s" % (STAGES[j][0].replace("\\&", "&"),
                                           t[ks[j]]),
                     fontsize=7.6, color=BP.INK)
    h = [plt.Line2D([], [], ls="", marker="o", ms=7,
                    color=tuple(v / 255 for v in c), label=l)
         for c, l in ((BS.C_BRACE, "brace-arm contact"),
                      (BS.C_REACH, "reach-arm contact"),
                      (BS.C_TRUNK, "trunk contact"),
                      (BS.C_FOOT, "foot contact"))]
    fig.legend(handles=h, loc="outside lower center", ncol=4, fontsize=7.2,
               title="marker area $\\propto$ normal force", title_fontsize=7.4)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, "fig_cmpc_stages." + ext),
                    facecolor="white", dpi=190)
    plt.close(fig)
    print("wrote fig_cmpc_stages.pdf / .png  (seam at %.2f s)" % t_seam)


if __name__ == "__main__":
    main()
