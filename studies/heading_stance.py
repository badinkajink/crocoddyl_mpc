#!/usr/bin/env python3
"""Does aiming the forearm replace decentring the stance?

The decentred stance (dy = -0.25) bought whole-table access and paid for it in
lateral balance: forearm skew 25 deg, base roll 7.4 deg, CoM 79 mm toward the
right foot, right leg carrying 81 % of body weight with its ankle pitch at 0.49
of the clamped limit.  The user's read -- that the real problem is the forearm's
DIRECTION, not the robot's position -- predicts that aiming the forearm along the
table's long axis should recover most of the benefit at a much smaller (or zero)
stance offset.

This sweeps heading x stance and reports, for each combination, both halves of
the trade at once:

  BENEFIT   on_table (fraction of the arm over the wood), y_clear (site to side
            edge), and the strength-limited reach envelope
  COST      forearm skew, base roll, lateral CoM, and the left/right leg effort
            imbalance -- the numbers that say "precarious"

A combination is only better if it improves the first WITHOUT giving back the
second, so both are printed on the same row rather than in two tables.

usage: heading_stance.py [set] [y] [z]
"""
import json
import os
import sys

import numpy as np

import contact_select as cs
import asymmetry as asym

HEADINGS = [None, 0.0, -10.0, -20.0]     # None = task off (the S11 behaviour)
DYS = [0.0, -0.10, -0.25]


def envelope(sub, y, z, xs):
    """Furthest x that is solved, placed and statically feasible."""
    best = None
    for x in xs:
        m, d = cs.load()
        ik = cs.solve_ik(m, d, np.array([float(x), y, z]), sub)
        if ik["reach"] >= 0.03 or ik["foot"] >= 0.02:
            continue
        if not ik.get("all_placed", True) or ik["penetration"] >= 0.01:
            continue
        if cs.equilibrium_qp(m, d, sub)["feasible"]:
            best = float(x)
    return best


def main():
    sub = tuple(sys.argv[1].split("+")) if len(sys.argv) > 1 \
        else ("elbow", "forearm", "palm")
    y = float(sys.argv[2]) if len(sys.argv) > 2 else -0.2348
    z = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0982
    xs = np.round(np.arange(1.05, 1.61, 0.05), 3)

    print("set=%s  target y=%.4f z=%.4f  site_set=%s" % ("+".join(sub), y, z, cs.SITE_SET))
    print("%8s %6s | %6s %8s %8s | %6s %6s %8s %7s"
          % ("heading", "dy", "env", "on_table", "y_clear",
             "skew", "roll", "lat CoM", "imbal"))

    rows = []
    for h in HEADINGS:
        for dy in DYS:
            cs.FOREARM_ALIGN = h is not None
            if h is not None:
                cs.FOREARM_HEADING = h
            cs.STANCE_DY = dy

            env = envelope(sub, y, z, xs)
            # geometry + asymmetry are measured AT the envelope pose, since that
            # is the pose the number describes; fall back to a mid target if the
            # set has no feasible envelope at all
            xm = env if env else 1.20
            m, d = cs.load()
            cs.solve_ik(m, d, np.array([xm, y, z]), sub)
            import stance_sweep as ss
            frac, yc = ss.geometry(m, d, sub)
            a = asym.measure([xm, y, z], sub)

            r = dict(heading=h, dy=dy, envelope=env, on_table=frac, y_clear=yc,
                     skew=a["skew_deg"], roll=a["base_roll_deg"],
                     lat_com=a["lateral_com"], imbalance=a["imbalance"],
                     eff_left=a["eff_left"], eff_right=a["eff_right"],
                     measured_at=xm)
            rows.append(r)
            print("%8s %6.2f | %6s %8.2f %8.4f | %6.1f %6.1f %8.3f %+7.3f"
                  % ("off" if h is None else "%.0f" % h, dy,
                     ("%.2f" % env) if env else "none", frac, yc,
                     a["skew_deg"], a["base_roll_deg"], a["lateral_com"],
                     a["imbalance"]), flush=True)

    ok = [r for r in rows if r["envelope"]]
    if ok:
        # best = furthest reach among poses that are not lopsided
        balanced = [r for r in ok if abs(r["imbalance"]) < 0.20] or ok
        best = max(balanced, key=lambda r: (r["envelope"], -abs(r["imbalance"])))
        print("\nbest balanced (|imbalance| < 0.20): heading=%s dy=%.2f -> %.2f m, "
              "imbalance %+.3f, roll %.1f deg, on_table %.2f"
              % (best["heading"], best["dy"], best["envelope"],
                 best["imbalance"], best["roll"], best["on_table"]))

    dst = os.environ.get("OUT", "heading_stance.json")
    json.dump(dict(subset=list(sub), y=y, z=z, rows=rows), open(dst, "w"), indent=1)
    print("wrote", dst)


if __name__ == "__main__":
    main()
