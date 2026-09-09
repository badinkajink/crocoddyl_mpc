#!/usr/bin/env python3
"""Which (table height, reach target) pairs does the braced lean certify?

WHY THIS EXISTS.  The MJPC braced-lean study on `mujoco_mpc@wxie/table-height`
measures a HEIGHT WINDOW -- the sampling planner completes the ladder at a slab
face of 0.985 and 1.035 m and at no other height tried (0.785, 0.885, 1.050,
1.060, 1.085 all fail, 3-6 seeds each).  Its high-end diagnosis is a vertical
deficit: the left shoulder holds ~1.40 m absolute regardless of the slab, so at
1.085 m the forearm pad never gets above the top face and the brace is on thin
air.  That diagnosis was made from rollouts, so it cannot separate two very
different claims:

    (a) no statically valid braced pose exists at that height, or
    (b) one exists and the sampling planner does not find it.

This module answers that, because certification here is STATIC and does not
involve a planner at all: IK for a pose that puts the reaching hand on target,
both feet pinned and each bracing site on the slab, then an equilibrium QP for
the contact forces and joint torques under real friction cones and real actuator
limits.  If a mode certifies at 1.085 m, the MJPC failure is (b).

It also answers the second and third questions in the same sweep, which is why
it is a GRID and not a height sweep:

    * the reach target has never been varied with height.  Both studies pin it
      at one point and ask "does this height work", when the honest question is
      which targets are still stably reachable once the slab moves.  `--tx/--ty`
      and `--dz` sweep it.
    * the brace posture has never been re-chosen with height.  The shipped
      keyframe is a one-height fit (0.985 m) and `elbow+forearm` is the study's
      default by habit -- `cmpc_mode_select` already found it is never the
      ranked pick at any target.  The mode ladder is re-enumerated per cell, so
      the winning posture is an OUTPUT here, not an input.

WHAT A CELL IS.  One (face, target) pair.  For each contact subset in the
ladder: IK -> press onto the real narrowphase -> equilibrium QP -> admissibility
(reach < 3 cm, feet held, no penetration, every requested site actually
touching, base residual < 1 N, max |tau|/limit <= 1).  Admissible modes are
ranked by least normalized actuator effort, ties toward fewer contacts.  The
cell also records the STATIC-EQUILIBRIUM REGION margin at the winning pose --
"stably" in the user's phrase is that margin, not the IK residual -- and the
geometry the MJPC page bounds the window with (shoulder above face, pad
clearance, base pitch), so the two studies can be read on the same axes.

TARGET CONVENTION.  `--dz` is height ABOVE THE FACE, which is the MJPC rung-2
convention (its target is face + 0.15) and the one that means "the same task at
a different height".  `--tz` holds world z instead, for the case where the thing
being reached does not ride with the table.

usage: height_reach.py --faces 0.785,0.885,0.985,1.035,1.085 --out runs/xx
       height_reach.py --faces 0.985,1.085 --tx 0.75,0.85,0.95,1.05 --out ...
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import contact_select as cs
import croco_modes as cm
import stability as st
import mujoco

# The shipped session-12 target, and the slab it was chosen against.
SHIPPED_TARGET = (0.9047, -0.2348, 1.0982)
SHIPPED_FACE = 0.985
SHIPPED_DZ = SHIPPED_TARGET[2] - SHIPPED_FACE      # 0.1132 m above the face

# The heights the MJPC study has run, so the two are comparable row for row.
MJPC_FACES = (0.785, 0.885, 0.985, 1.035, 1.050, 1.060, 1.085)

# A CURATED ladder, not the power set.  32 subsets per cell is 27 solves of
# nothing (croco_sweep makes the same argument).  These eight are the ones the
# study has a physical reason to ask about: legs alone as the control, each arm
# site alone, the shipped default, and the pairs `croco_brace_load_path` found
# actually carry force -- the wrist pad carries the brace in every mode and no
# named mode included it until 2026-08-21.
LADDER = [
    (),
    ("elbow",),
    ("forearm",),
    ("elbow", "forearm"),          # the study default
    ("elbow", "wrist"),
    ("forearm", "wrist"),
    ("forearm", "palm"),
    ("elbow", "forearm", "palm"),
]


def _git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def geometry(m, d, face):
    """The measurements the MJPC page bounds its window with, at this pose.

    Same definitions, so the numbers can be put side by side:
      sh_abs / sh_gap   left shoulder world z, and its height above the face
      pad_clear         forearm pad's lowest point above the face (negative =
                        below it, which is the MJPC high-end failure)
      pitch             base pitch [deg], positive = bowed toward the table
      com_ahead         CoM ahead of the foot midpoint [mm], the balance axis.
                        ROBOT subtree only -- `subtree_com[0]` carries the 86 kg
                        table and the object and reads 437 mm, which is not a
                        posture measurement (`stability.blocks` makes the same
                        correction at its line 85).
      stand_off         ankle midpoint behind the slab's near edge [mm]
    """
    mujoco.mj_forward(m, d)
    mujoco.mj_comPos(m, d)
    arm = cs.BRACE_ARM
    sh = d.xpos[cs.bid(m, "%s_shoulder_pitch_link" % arm)][2]

    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "%s_forearm_pad" % arm)
    if g >= 0:
        # Capsule: lowest point is the lower endpoint minus the radius.  The pad
        # is declared fromto, so MuJoCo stores half-length in size[1] along the
        # geom's local z.
        R = d.geom_xmat[g].reshape(3, 3)
        half = float(m.geom_size[g][1]) * R[:, 2]
        r = float(m.geom_size[g][0])
        pad = min(d.geom_xpos[g][2] - half[2], d.geom_xpos[g][2] + half[2]) - r
    else:
        pad = float("nan")

    q = d.qpos
    w, x, y, z = q[3], q[4], q[5], q[6]
    pitch = np.degrees(np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))

    la = d.xpos[cs.bid(m, "left_ankle_roll_link")]
    ra = d.xpos[cs.bid(m, "right_ankle_roll_link")]
    mid = 0.5 * (la + ra)
    near = cs.table_x_range(m, d)[0]
    return dict(sh_abs=float(sh), sh_gap=float(sh - face),
                pad_clear=float(pad - face),
                pitch=float(-pitch), base_z=float(q[2]),
                com_ahead=float(1000.0 * (d.subtree_com[cs.bid(m, "pelvis")][0]
                                          - mid[0])),
                stand_off=float(1000.0 * (mid[0] - near)))


def cell(face, target, ladder=LADDER, verbose=False):
    """Certify every mode in the ladder at one (face, target) pair."""
    cs.TABLE_H = float(face)
    recs = cm.enumerate_modes(list(target), sites=cs.SITES,
                              subsets=ladder, verbose=verbose)
    order = cm.rank(recs)
    out = dict(face=float(face), target=[float(v) for v in target],
               dz=float(target[2] - face), n_admissible=len(order),
               modes=[])
    for r in recs:
        out["modes"].append(dict(
            subset=r["subset"], admissible=r["admissible"],
            reach_mm=1000.0 * r["reach"], pen_mm=1000.0 * r["penetration"],
            placed=r["placed"], base_res=r["base_res"],
            max_ratio=r["max_ratio"], effort=r["effort"],
            brace_force=r["brace_force"],
            gap_mm={k: (None if v == cm.NO_CONTACT else v)
                    for k, v in r["contact_gap_mm"].items()}))

    # Geometry and stability at the WINNING pose, plus at legs_only as the
    # control -- a cell where only legs_only certifies is a cell with no brace,
    # which is a different result from a cell that certifies nothing.
    for tag, rec in (("best", order[0] if order else None),
                     ("legs", next((r for r in recs if not r["subset"]), None))):
        if rec is None:
            out[tag] = None
            continue
        m, d = cs.load(ik_margin=0.0)
        d.qpos[:] = rec["qpos"]
        mujoco.mj_forward(m, d)
        g = geometry(m, d, face)
        try:
            _, com, margin = st.equilibrium_region(m, d, tuple(rec["subset"]),
                                                   actuated=True)
            g["margin_mm"] = float(1000.0 * margin)
        except Exception as e:                      # a degenerate region is a
            g["margin_mm"] = None                   # result, not a crash
            g["margin_error"] = str(e)
        g["subset"] = rec["subset"]
        g["effort"] = float(rec["effort"])
        g["max_ratio"] = float(rec["max_ratio"])
        g["brace_force"] = float(rec["brace_force"])
        g["admissible"] = bool(rec["admissible"])
        g["qpos"] = np.asarray(rec["qpos"], float)
        out[tag] = g
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces", default=",".join("%.3f" % v for v in MJPC_FACES),
                    help="table top face heights [m], comma separated")
    ap.add_argument("--tx", default="%.4f" % SHIPPED_TARGET[0],
                    help="reach target x [m], comma separated")
    ap.add_argument("--ty", default="%.4f" % SHIPPED_TARGET[1],
                    help="reach target y [m], comma separated")
    ap.add_argument("--dz", default="%.4f" % SHIPPED_DZ,
                    help="reach target height ABOVE THE FACE [m]; the MJPC "
                         "rung-2 convention. Mutually exclusive with --tz.")
    ap.add_argument("--tz", default=None,
                    help="reach target world z [m], held fixed as the table "
                         "moves. Use when the thing being reached does not "
                         "ride with the table.")
    ap.add_argument("--stance-dx", type=float, default=None,
                    help="rigid x offset of the whole robot [m] before the IK")
    ap.add_argument("--stance-dy", type=float, default=None)
    ap.add_argument("--modes", default=None,
                    help="override the ladder, e.g. "
                         "'legs_only,elbow+forearm,forearm'. Use when the "
                         "question is about ONE posture across heights (the "
                         "brace-keyframe question) rather than which posture "
                         "wins.")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.stance_dx is not None:
        cs.STANCE_DX = args.stance_dx
    if args.stance_dy is not None:
        cs.STANCE_DY = args.stance_dy

    ladder = LADDER
    if args.modes:
        ladder = []
        for tok in args.modes.split(","):
            tok = tok.strip()
            if not tok:
                continue
            sub = () if tok == "legs_only" else tuple(
                v.strip() for v in tok.split("+") if v.strip())
            bad = [v for v in sub if v not in cs.SITES]
            if bad:
                raise SystemExit("--modes: no such site(s): %s. Known: %s"
                                 % (", ".join(bad), ", ".join(cs.SITES)))
            ladder.append(sub)

    faces = [float(v) for v in args.faces.split(",") if v.strip()]
    txs = [float(v) for v in args.tx.split(",") if v.strip()]
    tys = [float(v) for v in args.ty.split(",") if v.strip()]
    if args.tz is not None:
        zs = [("abs", float(v)) for v in args.tz.split(",") if v.strip()]
    else:
        zs = [("rel", float(v)) for v in args.dz.split(",") if v.strip()]

    os.makedirs(args.out, exist_ok=True)
    cells, t0 = [], time.time()
    combos = list(itertools.product(faces, txs, tys, zs))
    for i, (face, tx, ty, (kind, zv)) in enumerate(combos):
        tz = face + zv if kind == "rel" else zv
        c = cell(face, (tx, ty, tz), ladder=ladder, verbose=args.verbose)
        c["z_convention"] = kind
        cells.append(c)
        best = c["best"]
        print("[%3d/%3d] face %.3f  target (%.3f, %.3f, %.3f)  "
              "%d/%d admissible  best=%s%s"
              % (i + 1, len(combos), face, tx, ty, tz,
                 c["n_admissible"], len(ladder),
                 ("+".join(best["subset"]) or "legs_only") if best else "NONE",
                 ("  effort=%.3f margin=%s mm pad=%+.0f mm"
                  % (best["effort"],
                     "%.0f" % best["margin_mm"] if best["margin_mm"] is not None else "--",
                     1000.0 * best["pad_clear"])) if best else ""),
              flush=True)

    man = dict(cells=cells, faces=faces, tx=txs, ty=tys,
               z=dict(kind=zs[0][0], values=[v for _, v in zs]),
               ladder=["+".join(s) or "legs_only" for s in ladder],
               stance_dx=cs.STANCE_DX, stance_dy=cs.STANCE_DY,
               model=os.path.basename(cs.MODEL), brace_arm=cs.BRACE_ARM,
               site_set=cs.SITE_SET, seed_key=cs.SEED_KEY,
               tau_basis=cs.TAU_BASIS, commit=_git_head(),
               elapsed_s=time.time() - t0)
    # q* goes to an npz, not into the JSON.  It is 41 floats per pose and the
    # point of keeping it is that a certified pose is a KEYFRAME CANDIDATE:
    # the MJPC study's open question is what its brace keyframe should become at
    # a different slab, and this is a solved answer to that per height.
    poses = {}
    for i, c in enumerate(cells):
        for tag in ("best", "legs"):
            if c.get(tag) is not None and "qpos" in c[tag]:
                poses["%03d_%s" % (i, tag)] = c[tag].pop("qpos")
    np.savez(os.path.join(args.out, "poses.npz"), **poses)

    path = os.path.join(args.out, "cells.json")
    with open(path, "w") as f:
        json.dump(man, f, indent=1)
    print("\nwrote %s and poses.npz  (%d cells, %d poses, %.0f s)"
          % (path, len(cells), len(poses), man["elapsed_s"]))


if __name__ == "__main__":
    main()
