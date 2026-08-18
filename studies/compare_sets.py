#!/usr/bin/env python3
"""Rank contact sets at a target by STABILITY rather than by effort.

contact_select.py picks the set that minimises normalised actuator effort. That
is a fine objective and it is not the only one -- a set can be cheap to hold and
still be one nudge from sliding, or hold a CoM that sits 5 mm from the edge of
its own equilibrium region. This prints, per set:

  feas      the equilibrium QP found a solution inside the torque basis
  peak      max |tau| / tau_max over the 27 joints
  effort    sum (tau/tau_max)^2, contact_select's objective
  slip      worst friction-cone utilisation over the BRACE contacts (1.0 = sliding)
  feet      same for the worst foot corner
  margin    distance from the CoM to the edge of the actuated equilibrium region
  gain      that margin minus the legs-only margin at the SAME pose -- i.e. what
            the brace actually buys, which is the number the whole idea rests on
  push      min over directions of the largest horizontal force at the CoM that
            the set can still balance

usage: compare_sets.py X Y Z [subset,subset,...]
       TAU_BASIS=model|urdf|clamp  LEAN_MODEL=handless|magpie
"""
import itertools
import json
import os
import sys

import numpy as np

import contact_select as cs
import stability as st

DEFAULT_SETS = [(), ("palm",), ("elbow", "forearm"), ("elbow", "forearm", "palm"),
                ("hip",), ("elbow", "forearm", "hip"), ("elbow", "forearm", "hip", "torso")]


def one(target, subset):
    m, d = cs.load()
    ik = cs.solve_ik(m, d, np.array(target), subset)
    qp = cs.equilibrium_qp(m, d, subset)
    slips = st.slip_margins(m, d, subset, qp["lam"])
    _, _, marg = st.equilibrium_region(m, d, subset, actuated=True)
    _, push = st.max_push(m, d, subset, ndir=12)
    brace_slip = max([v["utilisation"] for k, v in slips.items()
                      if k != "feet_worst"] or [0.0])
    # Placement is decided by MuJoCo's narrowphase (ik["all_placed"], folded into
    # ik["ok"]), NOT by the site's height above the table. The site sits on the
    # link axis and the link touches on its capsule SURFACE, so a perfectly good
    # brace still shows a 2-5 cm site z-residual; gating on that rejected every
    # real brace pose. What is reported instead is `drift`: how far the nominal
    # site is from the contact MuJoCo actually found. That used to be a silent
    # modelling error (the QP applied its force at the site); resolved_contacts
    # now moves the force to the real point, so drift is diagnostic rather than
    # damage -- large values just mean the site is a poor label for the contact.
    drift = 0.0
    for s in subset:
        b, o = cs.SITES[s]
        nominal = cs.point_world(m, d, b, o)
        for bi, p, _ in cs.resolved_contacts(m, d, (s,))[-1:]:
            drift = max(drift, float(np.linalg.norm(nominal - p)))
    return dict(subset=list(subset), reach=ik["reach"], foot=ik["foot"],
                site=ik.get("site", {}), drift=drift,
                placed=bool(ik.get("all_placed", True)),
                solved=bool(ik.get("ok", False)),
                feasible=bool(qp["feasible"]), peak=qp["max_ratio"],
                effort=qp["effort"], slip=brace_slip,
                feet_slip=slips["feet_worst"]["utilisation"],
                margin=marg, push_min=float(np.min(push)),
                slip_detail={k: v["utilisation"] for k, v in slips.items()})


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__.strip().splitlines()[-2].strip())
    target = [float(a) for a in sys.argv[1:4]]
    sets = ([tuple(s.split("+")) if s != "legs" else () for s in sys.argv[4].split(",")]
            if len(sys.argv) > 4 else DEFAULT_SETS)

    rows = [one(target, s) for s in sets]
    base = next((r["margin"] for r in rows if not r["subset"]), float("nan"))

    print("target %s   basis=%s   model=%s   mu=%.2f"
          % (target, cs.TAU_BASIS, os.path.basename(cs.MODEL), cs.MU))
    print("%-26s %4s %4s %6s %6s %7s %6s %6s %8s %8s %6s"
          % ("set", "slvd", "feas", "drift", "peak", "effort", "slip", "feet",
             "margin", "gain", "push"))
    for r in rows:
        name = "+".join(r["subset"]) or "legs-only"
        gain = r["margin"] - base
        print("%-26s %4s %4s %6.0f %6.3f %7.3f %6.3f %6.3f %8.4f %+8.4f %6.0f"
              % (name, "y" if r["solved"] else "NO", "y" if r["feasible"] else "n",
                 1000 * r["drift"], r["peak"], r["effort"],
                 r["slip"], r["feet_slip"], r["margin"], gain, r["push_min"]))
    print("  slvd=NO: MuJoCo reports no contact between a selected site's body and\n"
          "  the table -- every number on that row describes contacts that do not exist.\n"
          "  drift [mm]: nominal site to the contact actually found (diagnostic only;\n"
          "  the QP applies its force at the real contact, not at the site).")
    out = "compare_%.2f_%.2f_%.2f.json" % tuple(target)
    json.dump(dict(target=target, basis=cs.TAU_BASIS,
                   model=os.path.basename(cs.MODEL), rows=rows),
              open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
