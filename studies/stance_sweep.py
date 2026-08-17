#!/usr/bin/env python3
"""Where should the robot STAND?

Two findings drive this, and they turn out to be the same finding.

1. reach_diag.py: every braced contact set's reach envelope ends on the REACH
   gate (the hand cannot get to the target), not on strength.  Only legs-only
   ends on strength, at x = 1.10.  So past 1.30 the study is measuring arm length,
   and no contact set can beat any other -- the benefit of bracing is CENSORED by
   a kinematic wall.

2. The bracing arm runs off the SIDE of the table before it runs off the end.
   Allen's table is 0.595 m wide and 1.18 m deep, the robot stands square-on and
   centred on it, so the left arm crosses the near corner diagonally and uses the
   NARROW axis.

Both move with the stance.  This sweeps a rigid lateral (and optional fore-aft)
relocation of the robot and reports, per offset:

  wall      furthest x whose reach residual is still < 3 cm  (the kinematic wall)
  strength  furthest x that is also statically feasible      (the honest envelope)
  on_table  fraction of the bracing arm's length whose (x, y) lies over the table
  y_clear   smallest distance from a placed site to a table side edge [m]
  placed    all selected sites in contact per MuJoCo's narrowphase

`on_table` is sampled along the real kinematic chain (elbow joint -> forearm pad
-> wrist links -> gripper box), not along a straight line, so it counts the arm
the robot actually has.

usage: stance_sweep.py [set] [y] [z]
       e.g. stance_sweep.py elbow+forearm+palm
"""
import json
import os
import sys

import numpy as np

import contact_select as cs

# points along the bracing arm, as (body, body-frame offset), proximal -> distal
def arm_chain(arm, my):
    return [
        ("%s_shoulder_yaw_link" % arm, np.array([0.002, my * 0.007, -0.182])),
        ("%s_elbow_link" % arm, np.array([0.02, my * 0.010, -0.015])),
        ("%s_elbow_link" % arm, np.array([0.065, my * 0.020, -0.015])),
        ("%s_elbow_link" % arm, np.array([0.110, my * 0.030, -0.015])),
        ("%s_wrist_roll_link" % arm, np.array([0.043, 0.0, 0.0])),
        ("%s_wrist_pitch_link" % arm, np.array([0.010, 0.0, 0.0])),
        ("%s_wrist_yaw_link" % arm, np.array([0.065, 0.0, 0.0])),
        ("%s_magpie_gripper" % arm, np.array([0.0965, 0.0, 0.0])),
        ("%s_magpie_gripper" % arm, np.array([0.1795, 0.0, 0.0])),
    ]


def geometry(m, d, subset):
    """How much of the bracing arm is over the table, and how close to an edge."""
    x_lo, x_hi = cs.table_x_range(m, d)
    y_lo, y_hi = cs.table_y_range(m, d)
    pts = [cs.point_world(m, d, b, o) for b, o in arm_chain(cs.BRACE_ARM, cs._MY)]
    inside = [bool(x_lo <= p[0] <= x_hi and y_lo <= p[1] <= y_hi) for p in pts]
    y_clear = np.inf
    for s in subset:
        p = cs.point_world(m, d, *cs.SITES[s])
        y_clear = min(y_clear, p[1] - y_lo, y_hi - p[1])
    return (float(np.mean(inside)),
            float(y_clear) if np.isfinite(y_clear) else float("nan"))


def walls(subset, y, z, xs):
    """(kinematic wall, strength-limited envelope, geometry at the wall)."""
    wall = strength = None
    geo = (float("nan"), float("nan"))
    for x in xs:
        m, d = cs.load()
        ik = cs.solve_ik(m, d, np.array([float(x), y, z]), subset)
        reached = ik["reach"] < 0.03 and ik["foot"] < 0.02
        if reached:
            wall = float(x)
            geo = geometry(m, d, subset)
        if reached and ik.get("all_placed", True) and ik["penetration"] < 0.01:
            qp = cs.equilibrium_qp(m, d, subset)
            if qp["feasible"]:
                strength = float(x)
    return wall, strength, geo


def main():
    sub = tuple(sys.argv[1].split("+")) if len(sys.argv) > 1 \
        else ("elbow", "forearm", "palm")
    y = float(sys.argv[2]) if len(sys.argv) > 2 else -0.2348
    z = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0982
    xs = np.round(np.arange(1.00, 1.61, 0.05), 3)
    dys = [0.0, -0.05, -0.10, -0.15, -0.20, -0.25]
    dxs = [0.0, -0.05, 0.05]

    print("set=%s  target y=%.4f z=%.4f  basis=%s  site_set=%s"
          % ("+".join(sub), y, z, cs.TAU_BASIS, cs.SITE_SET))
    print("%6s %6s | %6s %8s | %8s %8s"
          % ("dx", "dy", "wall", "strength", "on_table", "y_clear"))
    rows = []
    for dx in dxs:
        for dy in dys:
            cs.STANCE_DX, cs.STANCE_DY = dx, dy
            w, s, (frac, yc) = walls(sub, y, z, xs)
            rows.append(dict(dx=dx, dy=dy, wall=w, strength=s,
                             on_table=frac, y_clear=yc, subset=list(sub)))
            print("%6.2f %6.2f | %6s %8s | %8.2f %8.4f"
                  % (dx, dy, ("%.2f" % w) if w else "none",
                     ("%.2f" % s) if s else "none", frac, yc), flush=True)

    best = max((r for r in rows if r["strength"]),
               key=lambda r: (r["strength"], r["on_table"]), default=None)
    print("\nbest by strength-limited envelope then arm-on-table:", best)
    dst = os.environ.get("OUT", "stance_sweep.json")
    json.dump(dict(subset=list(sub), y=y, z=z, basis=cs.TAU_BASIS,
                   site_set=cs.SITE_SET, rows=rows, best=best),
              open(dst, "w"), indent=1)
    print("wrote", dst)


if __name__ == "__main__":
    main()
