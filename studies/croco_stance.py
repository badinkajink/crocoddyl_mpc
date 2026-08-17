#!/usr/bin/env python3
"""Is the brace on the table's rail because it has to be, or because of where we stand?

croco_why finds that in 11 of 14 certified cells the bracing arm's contact sits
0-2 mm in from the lateral edge of a slab that is 595 mm wide.  That is a very
particular kind of brace -- the arm is hooked over the rail rather than resting
on the wood -- and it is the common factor behind the cells that hold at 100%
duty on 9 N, the cells whose "brace" the plant replaces with its torso, and the
cells that topple.

Nothing in the pipeline chose it.  The IK's non-penetration rows are written
against the tabletop BOX, and a contact on the side face satisfies them exactly
as well as one on the top; the static QP then scores a rail contact with a normal
that still points mostly up, so it certifies.  The pose is never asked to be
INBOARD of anything.

That leaves an obvious question this script answers directly: the study fixes the
robot's stance at y = 0 and has never varied it.  The reach target is at
y = -0.235, i.e. 60 mm from the far rail, so reaching it twists the torso and
swings the bracing arm out to the near rail.  `STANCE_DY` translates the whole
robot, feet included, before the IK runs.  If the rail contact is kinematically
forced, moving the robot will not help.  If it is a stance artifact, it will.

usage: croco_stance.py [--dys 0,-0.06,-0.12,-0.18] [--targets 1.050,1.150]
                       [--modes elbow,palm,elbow+palm] [--out stance.json]
"""

import argparse
import json

import numpy as np

import contact_select as cs
import croco_modes as cm
import croco_why as cw


def row(pr, dy, tx, subset):
    cs.STANCE_DY = dy
    m, d = cs.load()
    target = np.array([tx, -0.2348, 1.0982])
    P = cs.solve_ik(m, d, target, tuple(subset))
    q = d.qpos.copy()
    gaps = cm.contact_gaps(m, d, list(subset))
    qp = cs.equilibrium_qp(m, d, tuple(subset))
    pr.set_q(q)
    cons = [c for c in pr.contacts(list(subset)) if c["where"] in ("top", "edge")]
    inb = min((c["inboard"] for c in cons), default=None)
    ymax = max((abs(c["pos"][1]) for c in cons), default=None)
    return dict(
        dy=dy, target=tx, mode="+".join(subset),
        reach_mm=round(P["reach"] * 1e3, 2),
        pen_mm=round(P["penetration"] * 1e3, 2),
        gaps={k: round(v, 2) for k, v in gaps.items()},
        brace_y_mm=None if ymax is None else round(ymax * 1e3, 1),
        inboard_mm=inb,
        base_res=round(float(qp["base_residual"]), 4),
        max_tau_ratio=round(float(qp["max_ratio"]), 3),
        effort=round(float(qp["effort"]), 3),
        brace_N=round(float(qp["brace_force"]), 1),
        admissible=bool(qp["feasible"] and P["reach"] < 0.03
                        and all(v < cm.PLACE_TOL for v in gaps.values())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dys", default="0,-0.06,-0.12,-0.18")
    ap.add_argument("--targets", default="1.050,1.150")
    ap.add_argument("--modes", default="elbow,palm,elbow+palm")
    ap.add_argument("--out")
    args = ap.parse_args()

    dys = [float(v) for v in args.dys.split(",")]
    txs = [float(v) for v in args.targets.split(",")]
    modes = [m.split("+") for m in args.modes.split(",")]

    pr = cw.Probe()
    rows = []
    print(f"{'dy [mm]':>8s} {'target':>7s} {'mode':14s} {'reach':>7s} "
          f"{'brace |y|':>10s} {'inboard':>8s} {'effort':>7s} {'tau':>6s} "
          f"{'admissible':>11s}")
    for dy in dys:
        for tx in txs:
            for subset in modes:
                r = row(pr, dy, tx, subset)
                rows.append(r)
                print(f"{dy * 1000:8.0f} {tx:7.3f} {r['mode']:14s} "
                      f"{r['reach_mm']:7.1f} "
                      f"{(r['brace_y_mm'] if r['brace_y_mm'] is not None else float('nan')):10.0f} "
                      f"{(r['inboard_mm'] if r['inboard_mm'] is not None else float('nan')):8.1f} "
                      f"{r['effort']:7.2f} {r['max_tau_ratio']:6.2f} "
                      f"{str(r['admissible']):>11s}")
    if args.out:
        json.dump({"rows": rows}, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
