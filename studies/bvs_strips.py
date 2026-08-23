#!/usr/bin/env python3
"""Rendered comparison strip: one row per strategy, one column per instant.

The tables say the brace buys margin; this says what the brace IS. It exists
because two of the strategies being compared are visually distinguishable and
numerically confusable -- "no brace cost, free standing" and "no brace cost,
planner found the table" carry the same weights and the same label, and differ
only in where the load goes. Rendered side by side with the contact points
drawn, the difference is a glance instead of a column.

Contact markers are drawn at MuJoCo's narrowphase positions and SIZED BY NORMAL
FORCE, so a grazing touch and a load-bearing brace do not look alike. Drawing a
nominal brace site instead would make a hovering forearm look seated, which is a
mistake this study has already published once.

Rows are chosen from analysis.json by MEASURED LOAD, not by which weights the
run was given -- the same rule the rest of the analysis follows.

usage: bvs_strips.py --json analysis.json --run DIR --out DIR [--az 120]
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

W, H = 760, 620

# Contacts are coloured BY ROLE, not by body name. The first version of this
# figure keyed off a list of left-arm bodies and silently drew nothing for
# `right_magpie_gripper` -- which carries 14 N in the commanded-brace rollout,
# because the REACHING hand rests on the table too. A palette that can only
# express the contacts you expected is a palette that hides the ones you did
# not, so every robot body now resolves to something.
#
# Hues are the two categorical slots the rest of the figures use: the brace arm
# takes slot 2 (orange), the reaching arm slot 1 (blue).
BRACE_ARM, REACH_ARM = "left", "right"
C_BRACE = (235, 104, 52)
C_REACH = (42, 120, 214)
C_TRUNK = (232, 123, 164)
C_FOOT = (120, 120, 120)
FOOT = ("left_ankle_roll_link", "right_ankle_roll_link")


def body_colour(name):
    """(rgb, is_foot) for any robot body that can touch the table."""
    if name in FOOT:
        return C_FOOT, True
    if name.startswith(BRACE_ARM + "_"):
        return C_BRACE, False
    if name.startswith(REACH_ARM + "_"):
        return C_REACH, False
    return C_TRUNK, False


_ROBOT = {}


def _is_robot(m, b):
    """Is body b part of the robot (pelvis subtree)? Cached per model."""
    key = id(m)
    if key not in _ROBOT:
        rb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        inside = np.zeros(m.nbody, dtype=bool)
        for i in range(m.nbody):
            p = i
            while p > 0:
                if p == rb:
                    inside[i] = True
                    break
                p = m.body_parentid[p]
        inside[rb] = True
        _ROBOT[key] = inside
    return bool(_ROBOT[key][b])


def frame_at(m, d, r, rows, col, k, az, draw_contacts=True, camera=None):
    """One rendered frame. `camera` overrides the default framing.

    The default is the six-row ablation strip's, where each tile is a sixth of
    the text height and a wide shot reads fine. A one-row strip gets five times
    the tile height and the same wide shot leaves the robot a thumbnail, so the
    caller is allowed to frame it -- the contact drawing below is unchanged
    either way, which is the part that must not vary between figures."""
    qi = [col["qpos%d" % i] for i in range(m.nq)]
    vi = [col["qvel%d" % i] for i in range(m.nv)]
    ui = [col["ctrl%d" % i] for i in range(m.nu)]
    d.qpos[:] = rows[k, qi]
    d.qvel[:] = rows[k, vi]
    d.ctrl[:] = rows[k, ui]
    mujoco.mj_forward(m, d)
    r.update_scene(d, camera=(camera if camera is not None else
                              sv.cam(az, el=-10, dist=2.75,
                                     look=(0.82, 0.0, 0.90))))
    scn = r.scene
    if draw_contacts:
        env = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist > 0:
                continue
            n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                   m.geom_bodyid[c.geom1]) or ""
            n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                   m.geom_bodyid[c.geom2]) or ""
            # the robot side of a robot<->environment contact
            r1, r2 = _is_robot(m, m.geom_bodyid[c.geom1]), \
                _is_robot(m, m.geom_bodyid[c.geom2])
            if r1 == r2:
                continue
            nm = n1 if r1 else n2
            rgb, is_foot = body_colour(nm)
            f6 = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, f6)
            if is_foot:
                sv.add_marker(scn, c.pos, tuple(v / 255 for v in rgb) + (0.45,),
                              0.014)
                continue
            # size ~ sqrt(force): linear makes a 60 N brace 30x a 2 N graze and
            # the graze vanishes; sqrt keeps both visible and still ordered.
            sz = 0.018 + 0.055 * np.sqrt(min(abs(f6[0]), 120.0) / 120.0)
            sv.add_marker(scn, c.pos, tuple(v / 255 for v in rgb) + (0.95,), sz)
    return r.render()


def pick_rows(res, run_dir):
    """Representative rollouts, chosen by measured load.

    A ROLLOUT THAT FELL IS NOT A STRATEGY. Nothing fell in the sampling
    planner's settled conditions, so this filter was implicit; the gradient
    planner's declared brace does fall at the default target, and a fallen
    robot's contact set is large, trunk-heavy and would be selected as the
    "commanded brace" exemplar by load alone -- drawing the failure and
    labelling it the strategy. Fallen runs are dropped, and if that empties a
    condition the EXTENDED one is used instead, which is why the strip's
    caption has to say which target each row came from.
    """
    def get(stage, arm):
        return [r for r in res if r.get("stage") == stage
                and r.get("arm") == arm and "error" not in r
                and not r.get("fell")
                and r.get("brace_load_N") is not None
                and np.isfinite(r["brace_load_N"])]

    def get2(arm):
        """`nominal`, or `nominal2` when the default target has nothing left."""
        return get("nominal", arm) or get("nominal2", arm)

    out = []
    ns, nb = get2("stand"), get2("brace")
    # the zero-brace-cost arm is TWO behaviours; show both ends of it
    pool = ns + get("sweep", "stand")
    if pool:
        lo = min(pool, key=lambda r: r["brace_load_N"])
        hi = max(pool, key=lambda r: r["brace_load_N"])
        out.append(("No brace cost\nfree standing", lo))
        if hi is not lo and hi["brace_load_N"] >= 15:
            out.append(("No brace cost\nplanner found the table", hi))
    if nb:
        out.append(("Commanded brace", sorted(
            nb, key=lambda r: r["brace_load_N"])[len(nb) // 2]))
    import bvs_plots as _BP
    for arm, lab in (("stand", "Max reach\nno brace cost"),
                     ("brace", "Max reach\ncommanded brace")):
        g = [r for r in get("maxreach", arm)
             if _BP.contact_class(r) != "torso"]
        if g:
            out.append((lab, sorted(
                g, key=lambda r: r["brace_load_N"])[len(g) // 2]))
    # The degenerate the cost function actually prefers at long range. Shown on
    # purpose: 3 of 4 commanded-brace max-reach runs end up here, and a reader
    # comparing only the clean rows would not know that.
    degen = [r for r in res if _BP.contact_class(r) == "torso"
             and r.get("stage") == "maxreach" and not r.get("fell")]
    if degen:
        out.append(("Degenerate\ntorso on table",
                    max(degen, key=lambda r: _BP.trunk_load(r))))
    return [(lab, os.path.join(run_dir, r["tag"] + ".csv"), r) for lab, r in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--az", type=float, default=120.0)
    ap.add_argument("--fracs", default="0.0,0.15,0.35,0.65,1.0")
    a = ap.parse_args()
    res = json.load(open(a.json))
    picks = pick_rows(res, a.run)
    if not picks:
        raise SystemExit("no rollouts to draw")
    fracs = [float(x) for x in a.fracs.split(",")]
    os.makedirs(a.out, exist_ok=True)

    m, d = sl.load()
    r = sv.make_renderer(m)
    grid, times = [], []
    for lab, path, meta in picks:
        col, rows, _ = sl.load_traj(path)
        t = rows[:, col["time"]]
        ks = [min(len(rows) - 1, int(f * (len(rows) - 1))) for f in fracs]
        grid.append([frame_at(m, d, r, rows, col, k, a.az) for k in ks])
        times = [t[k] for k in ks]

    nr, nc = len(grid), len(fracs)
    fig, axs = plt.subplots(nr, nc, figsize=(BP.TEXT, BP.TEXT * nr * H /
                                             (nc * W) * 1.02),
                            layout="constrained")
    axs = np.atleast_2d(axs)
    for i, (lab, _, meta) in enumerate(picks):
        for j in range(nc):
            ax = axs[i, j]
            ax.imshow(grid[i][j])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(BP.GRID)
            if i == 0:
                ax.set_title("$t$ = %.1f s" % times[j], fontsize=8,
                             color=BP.INK)
        # The label goes beside the row, not rotated into it: three rotated
        # lines are taller than the tile and collide with the row above.
        axs[i, 0].set_ylabel(lab, fontsize=7.6, color=BP.INK, rotation=0,
                             ha="right", va="center", labelpad=8)
        axs[i, 0].annotate("%.0f N" % meta["brace_load_N"], xy=(0.03, 0.94),
                           xycoords="axes fraction", fontsize=7.5,
                           color=BP.INK, va="top",
                           bbox=dict(boxstyle="round,pad=0.22", fc="white",
                                     ec=BP.GRID, lw=0.6))
    h = [plt.Line2D([], [], ls="", marker="o", ms=7,
                    color=tuple(v / 255 for v in c), label=l)
         for c, l in ((C_BRACE, "brace-arm contact"),
                      (C_REACH, "reach-arm contact"),
                      (C_TRUNK, "trunk contact"),
                      (C_FOOT, "foot contact"),
                      # the model draws these itself; named here so a reader
                      # does not take them for measurements
                      ((51, 178, 51), "reach target / brace pads (model)"))]
    fig.legend(handles=h, loc="outside lower center", ncol=5, fontsize=7.2,
               title="marker area $\\propto$ normal force",
               title_fontsize=7.5)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, "fig_strategy_strip." + ext),
                    facecolor="white", dpi=190)
    plt.close(fig)
    print("wrote fig_strategy_strip.pdf / .png  (%d strategies)" % nr)
    for lab, path, meta in picks:
        print("   %-42s %-28s %5.0f N  margin %.3f"
              % (lab.replace("\n", " / "), os.path.basename(path),
                 meta["brace_load_N"], meta.get("margin_actuated_settled",
                                                float("nan"))))


if __name__ == "__main__":
    main()
