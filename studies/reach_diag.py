#!/usr/bin/env python3
"""WHY does the reach envelope stop where it stops?

reach_envelope.py reports one bit per (target, set): solved-and-feasible or not.
That bit is the conjunction of four independent conditions, and collapsing them
loses exactly the information the "does bracing buy reach?" question needs:

    reach       the hand got to the target            KINEMATIC
    foot        the feet held their pinned pose       KINEMATIC
    placed      MuJoCo reports brace site <-> table   GEOMETRIC
    feasible    an admissible force distribution      STRENGTH

Only the last is a statement about bracing.  If a braced set drops out because
`placed` went false, the envelope is reporting an IK failure wearing a stability
result's clothes.

This prints all four per target, plus the cold-start vs continuation comparison:
reach_envelope.py re-loads the seed keyframe for EVERY target, so each solve is
a cold start from a pose whose hand sits at x = 1.207.  If a warm start from the
previous target's solution gets further, the envelope was measuring the IK's
basin of attraction, not the robot.

usage: reach_diag.py [y] [z]
"""
import json
import os
import sys

import numpy as np

import contact_select as cs
import stability as st

SETS = [(), ("palm",), ("elbow", "forearm"), ("elbow", "forearm", "palm"),
        ("hip",), ("elbow", "forearm", "hip")]


def probe(m, d, target, subset, warm=False):
    """Solve at `target` and return every gate separately.  If `warm`, d is
    assumed to already hold the previous solution and is NOT reset."""
    ik = cs.solve_ik(m, d, np.asarray(target), subset)
    qp = cs.equilibrium_qp(m, d, subset)
    _, _, marg = st.equilibrium_region(m, d, subset, actuated=True)
    return dict(
        subset=list(subset), x=float(target[0]),
        reach=float(ik["reach"]), foot=float(ik["foot"]),
        pen=float(ik["penetration"]),
        placed=bool(ik.get("all_placed", True)),
        per_site=ik.get("sites_placed", {}),
        qp_fallback=int(ik.get("qp_fallback", 0)),
        feasible=bool(qp["feasible"]), peak=float(qp["max_ratio"]),
        margin=float(marg),
        # the four gates, named
        g_reach=bool(ik["reach"] < 0.03), g_foot=bool(ik["foot"] < 0.02),
        g_pen=bool(ik["penetration"] < 0.01),
        g_placed=bool(ik.get("all_placed", True)),
        g_feas=bool(qp["feasible"]),
    )


def why(r):
    """The FIRST gate that failed, which is the honest label for the drop-out."""
    if not r["g_reach"]:
        return "reach"          # kinematic: hand cannot get there
    if not r["g_foot"]:
        return "foot"           # kinematic: stance broke
    if not r["g_pen"]:
        return "penetrate"      # geometric: pose is inside the table
    if not r["g_placed"]:
        return "placed"         # geometric: brace is not on the table
    if not r["g_feas"]:
        return "STRENGTH"       # the only one that is about bracing
    return "ok"


def main():
    y = float(sys.argv[1]) if len(sys.argv) > 1 else -0.2348
    z = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0982
    xs = np.round(np.arange(1.00, 1.55, 0.05), 3)

    rows = []
    print("y=%.4f z=%.4f basis=%s model=%s   COLD start (as reach_envelope.py)"
          % (y, z, cs.TAU_BASIS, os.path.basename(cs.MODEL)))
    print("%-24s %5s %7s %6s %6s %6s %6s %5s  %s"
          % ("set", "x", "reach", "foot", "pen", "placed", "peak", "feas", "drop-out"))
    for s in SETS:
        for x in xs:
            m, d = cs.load()
            r = probe(m, d, [float(x), y, z], s)
            r["mode"] = "cold"
            rows.append(r)
            print("%-24s %5.2f %7.4f %6.4f %6.4f %6s %6.3f %5s  %s"
                  % ("+".join(s) or "legs-only", x, r["reach"], r["foot"],
                     r["pen"], "y" if r["placed"] else "NO", r["peak"],
                     "y" if r["feasible"] else "n", why(r)), flush=True)

    print("\n== WARM start: each target continues from the previous solution ==")
    for s in SETS:
        m, d = cs.load()
        for x in xs:
            r = probe(m, d, [float(x), y, z], s, warm=True)
            r["mode"] = "warm"
            rows.append(r)
            print("%-24s %5.2f %7.4f %6.4f %6.4f %6s %6.3f %5s  %s"
                  % ("+".join(s) or "legs-only", x, r["reach"], r["foot"],
                     r["pen"], "y" if r["placed"] else "NO", r["peak"],
                     "y" if r["feasible"] else "n", why(r)), flush=True)

    print("\n== envelope by mode, and the reason it ends ==")
    for mode in ("cold", "warm"):
        print(" %s:" % mode)
        for s in SETS:
            sub = [r for r in rows if r["mode"] == mode and r["subset"] == list(s)]
            good = [r["x"] for r in sub if why(r) == "ok"]
            end = max(good) if good else None
            nxt = next((r for r in sub if r["x"] > (end or -1)), None)
            print("   %-24s %s   ends on: %s"
                  % ("+".join(s) or "legs-only",
                     ("%.2f m" % end) if end else "none",
                     why(nxt) if nxt else "grid end"))

    out = os.environ.get("OUT", "reach_diag.json")
    json.dump(dict(y=y, z=z, basis=cs.TAU_BASIS,
                   model=os.path.basename(cs.MODEL), rows=rows),
              open(out, "w"), indent=1)
    print("wrote", out, len(rows), "evaluations")


if __name__ == "__main__":
    main()
