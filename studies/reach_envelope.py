#!/usr/bin/env python3
"""How far out can the robot reach, with and without a brace?

The contact-selection map answers "which contacts are cheapest HERE". It cannot
answer the question the paper actually rests on, which is whether the brace buys
REACH -- targets that are unreachable without it. This sweeps the target away
from the robot along x at fixed y,z and reports, per contact set, the last target
that is still statically holdable.

Two failure modes are distinguished, because they mean opposite things:
  unsolved    the IK could not put the brace sites on the table (a GEOMETRY
              limit -- the contact is not available at this target)
  infeasible  the pose exists but no admissible force distribution does (a
              STRENGTH limit -- the contact is available and not enough)

usage: reach_envelope.py [y] [z]
       TAU_BASIS=model|urdf|clamp  LEAN_MODEL=magpie|handless  SEED_KEY=...
"""
import json
import os
import sys

import numpy as np

import contact_select as cs
import stability as st
import compare_sets as cmpr

SETS = [(), ("palm",), ("elbow", "forearm"), ("elbow", "forearm", "palm"),
        ("hip",), ("elbow", "forearm", "hip")]


def main():
    y = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2348
    z = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0982
    xs = np.round(np.arange(0.80, 1.45, 0.05), 3)

    rows = []
    print("y=%.4f z=%.4f  basis=%s  model=%s"
          % (y, z, cs.TAU_BASIS, os.path.basename(cs.MODEL)))
    print("%-6s %-26s %5s %5s %7s %6s %8s %6s" %
          ("x", "set", "slvd", "feas", "drift", "peak", "margin", "push"))
    for x in xs:
        for s in SETS:
            r = cmpr.one([float(x), y, z], s)
            r["x"] = float(x)
            rows.append(r)
            print("%-6.2f %-26s %5s %5s %7.1f %6.3f %8.4f %6.0f"
                  % (x, "+".join(s) or "legs-only",
                     "y" if r["solved"] else "NO", "y" if r["feasible"] else "n",
                     1000 * r["drift"], r["peak"], r["margin"], r["push_min"]),
                  flush=True)

    print("\n== reach envelope: furthest x that is BOTH solved and feasible ==")
    for s in SETS:
        name = "+".join(s) or "legs-only"
        good = [r["x"] for r in rows
                if r["subset"] == list(s) and r["solved"] and r["feasible"]]
        print("  %-26s %s" % (name, ("%.2f m" % max(good)) if good else "none"))

    out = os.environ.get("OUT", "reach_envelope.json")
    json.dump(dict(y=y, z=z, basis=cs.TAU_BASIS,
                   model=os.path.basename(cs.MODEL), rows=rows),
              open(out, "w"), indent=1)
    print("wrote", out, len(rows), "evaluations")


if __name__ == "__main__":
    main()
