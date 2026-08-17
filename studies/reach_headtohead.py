#!/usr/bin/env python3
"""The reach envelope with the kinematic wall pushed out of the way.

reach_diag.py established that at the study's centred stance every BRACED set's
envelope ends on the reach gate at x ~ 1.30 -- the arm runs out, not the robot's
balance -- while legs-only ends at 1.10 on strength.  Comparing those two numbers
compares a strength limit against an arm length, so the "+0.20 m" it yields is a
censored lower bound, not the benefit of bracing.

stance_sweep.py found the fix: shifting the robot -0.25 m in y (toward the
reaching side) puts the whole bracing arm over the table, takes the sites from
0.005 m OUTSIDE the table's side edge to 0.143 m inside it, and moves the
kinematic wall from 1.30 to >= 1.60.  With the wall out of the way the strength
limit is measurable again.

This re-runs every contact set at the SAME stance, on a grid that extends past
the old wall, and separates the two failure modes so the reported envelope is
always labelled with the constraint that produced it.  A set whose envelope still
ends on `reach` is reported as a lower bound (">= x"), never as an equality.

usage: reach_headtohead.py [dy] [y] [z]
       STANCE_DY overrides dy;  TAU_BASIS, SITE_SET, MU as usual
"""
import json
import os
import sys

import numpy as np

import contact_select as cs
import stability as st

SETS = [(), ("palm",), ("elbow", "forearm"), ("elbow", "forearm", "palm"),
        ("elbow", "forearm", "hip")]


def gate(ik, qp):
    if ik["reach"] >= 0.03 or ik["foot"] >= 0.02:
        return "reach"          # kinematic: the arm does not get there
    if ik["penetration"] >= 0.01:
        return "penetrate"
    if not ik.get("all_placed", True):
        return "placed"         # geometric: brace not on the table
    if not qp["feasible"]:
        return "STRENGTH"       # the only gate that is about bracing
    return "ok"


def main():
    dy = float(sys.argv[1]) if len(sys.argv) > 1 else -0.25
    y = float(sys.argv[2]) if len(sys.argv) > 2 else -0.2348
    z = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0982
    cs.STANCE_DY = float(os.environ.get("STANCE_DY", dy))
    xs = np.round(np.arange(1.00, 1.81, 0.05), 3)

    print("stance dy=%.2f  target y=%.4f z=%.4f  basis=%s site_set=%s mu=%.2f"
          % (cs.STANCE_DY, y, z, cs.TAU_BASIS, cs.SITE_SET, cs.MU))
    print("%-24s %5s %7s %6s %6s %8s %7s  %s"
          % ("set", "x", "reach", "plcd", "peak", "margin", "push", "gate"))

    rows = []
    for s in SETS:
        for x in xs:
            m, d = cs.load()
            ik = cs.solve_ik(m, d, np.array([float(x), y, z]), s)
            qp = cs.equilibrium_qp(m, d, s)
            _, _, marg = st.equilibrium_region(m, d, s, actuated=True)
            _, push = st.max_push(m, d, s, ndir=12)
            g = gate(ik, qp)
            r = dict(subset=list(s), x=float(x), gate=g,
                     reach=float(ik["reach"]), foot=float(ik["foot"]),
                     placed=bool(ik.get("all_placed", True)),
                     peak=float(qp["max_ratio"]), feasible=bool(qp["feasible"]),
                     margin=float(marg), push_min=float(np.min(push)))
            rows.append(r)
            print("%-24s %5.2f %7.4f %6s %6.3f %8.4f %7.0f  %s"
                  % ("+".join(s) or "legs-only", x, r["reach"],
                     "y" if r["placed"] else "NO", r["peak"], r["margin"],
                     r["push_min"], g), flush=True)
            if g == "reach" and x > 1.4:
                break            # the arm is done; further x only grows the miss

    print("\n== envelope, WITH the constraint that produced it ==")
    summary = {}
    for s in SETS:
        name = "+".join(s) or "legs-only"
        sub = [r for r in rows if r["subset"] == list(s)]
        good = [r["x"] for r in sub if r["gate"] == "ok"]
        end = max(good) if good else None
        nxt = next((r for r in sub if end is not None and r["x"] > end), None)
        cause = nxt["gate"] if nxt else "grid end"
        censored = cause != "STRENGTH"
        summary[name] = dict(envelope=end, cause=cause, censored=censored)
        print("   %-24s %s%.2f m   limited by: %s%s"
              % (name, ">= " if censored else "   ", end or float("nan"), cause,
                 "   (LOWER BOUND -- not a strength result)" if censored else ""))

    dst = os.environ.get("OUT", "reach_headtohead.json")
    json.dump(dict(dy=cs.STANCE_DY, y=y, z=z, basis=cs.TAU_BASIS,
                   site_set=cs.SITE_SET, mu=cs.MU, rows=rows, summary=summary),
              open(dst, "w"), indent=1)
    print("wrote", dst)


if __name__ == "__main__":
    main()
