#!/usr/bin/env python3
"""Solve for mu* -- the friction coefficient above which FRICTION stops being the
binding constraint and the joint-torque envelope takes over.

Motivation (user, 2026-08-04): every slip and margin number in S1-S10 rests on
MU = 0.6, which nobody measured.  Rather than defend that number, characterise
the JOINT TORQUE limits, which are known -- and to do that honestly you must
first establish how much friction the pose needs before friction stops mattering.
Above mu* the results describe the actuators.  Below it they describe the tape
on the table.

Method
------
The pose is fixed: the IK is purely kinematic (site placement + non-penetration)
and does not see mu at all, so one solve per (target, subset) is re-scored at
every mu.  That makes the comparison exact -- same pose, same contacts, same
Jacobians, one parameter moving.

Four quantities are tracked as mu rises:

  u_max     worst friction-cone utilisation ||lam_t|| / (mu lam_n) over the brace
            contacts at the min-effort solution.  u = 1 means the contact is ON
            the cone: it cannot supply another newton of shear.
  peak      max |tau| / tau_max over the 27 joints at that same solution.
  margin    radius of the ACTUATED static-equilibrium region about the CoM.
  push      min over directions of the largest horizontal force at the CoM the
            set can still balance.

Three thresholds are reported, because they are three different questions and
only one of them is the user's:

  mu*_tau    smallest mu at which `peak` is within PEAK_TOL of its mu -> infinity
             value.  ABOVE THIS THE POSE'S TORQUE DEMAND IS SET BY GEOMETRY, NOT
             BY FRICTION -- i.e. the joint-torque envelope is what the numbers
             are characterising.  This is the one the user asked for.
  mu*_slip   smallest mu at which no contact sits on its cone.  "Nothing is
             sliding" -- necessary, much weaker than mu*_tau.
  mu*_region smallest mu at which the ACTUATED equilibrium region has come within
             MARGIN_TOL of its mu -> infinity limit.  This is the strictest of
             the three and, measured, it is not reached anywhere in the plausible
             range: the region boundary is where shear demand is highest, so
             extra friction keeps buying region long after it has stopped
             changing what the pose itself costs.

Every mu -> infinity reference is computed directly, by solving the same problem
with the friction cones DELETED (tangential forces free, normals still
unilateral), rather than by extrapolating the tail of a sweep.

usage: friction_sweep.py [x y z] [set,set,...]
       TAU_BASIS=... SITE_SET=v1|v2 STANCE_DY=...
"""
import json
import os
import sys

import numpy as np

import contact_select as cs
import stability as st

MUS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]
MARGIN_TOL = 0.005          # [m] "has stopped growing" band on the region radius
PEAK_TOL = 0.01             # ratio band on max |tau| / tau_max
SETS = [("palm",), ("elbow", "forearm"), ("elbow", "forearm", "palm"),
        ("elbow", "forearm", "hip")]


def score(m, d, subset, mu):
    """Re-score a FIXED pose at friction coefficient mu.  mu=None -> cones off."""
    cs.MU = 1e9 if mu is None else mu       # every call site reads cs.MU live
    qp = cs.equilibrium_qp(m, d, subset)
    slips = st.slip_margins(m, d, subset, qp["lam"])
    _, _, marg = st.equilibrium_region(m, d, subset, actuated=True)
    _, push = st.max_push(m, d, subset, ndir=12)
    brace_u = max([v["utilisation"] for k, v in slips.items()
                   if k != "feet_worst"] or [0.0])
    return dict(mu=mu, feasible=bool(qp["feasible"]), peak=float(qp["max_ratio"]),
                effort=float(qp["effort"]), u_max=float(brace_u),
                u_feet=float(slips["feet_worst"]["utilisation"]),
                margin=float(marg), push_min=float(np.min(push)),
                slip_detail={k: float(v["utilisation"]) for k, v in slips.items()})


def main():
    tgt = ([float(a) for a in sys.argv[1:4]] if len(sys.argv) >= 4
           else [1.10, -0.2348, 1.0982])
    sets = ([tuple(s.split("+")) for s in sys.argv[4].split(",")]
            if len(sys.argv) > 4 else SETS)

    out = {"target": tgt, "basis": cs.TAU_BASIS, "site_set": cs.SITE_SET,
           "model": os.path.basename(cs.MODEL), "sets": {}}
    print("target %s  basis=%s  site_set=%s  stance dy=%.3f"
          % (tgt, cs.TAU_BASIS, cs.SITE_SET, cs.STANCE_DY))

    for subset in sets:
        name = "+".join(subset)
        m, d = cs.load()
        ik = cs.solve_ik(m, d, np.array(tgt), subset)
        if not ik.get("all_placed", True):
            print("\n== %s: NOT PLACED, skipping" % name)
            continue

        # mu -> infinity reference, computed not extrapolated
        inf = score(m, d, subset, None)
        rows = [score(m, d, subset, mu) for mu in MUS]

        # the three thresholds (see the module docstring)
        mu_tau = next((r["mu"] for r in rows
                       if abs(r["peak"] - inf["peak"]) <= PEAK_TOL), None)
        mu_slip = next((r["mu"] for r in rows if r["u_max"] < 0.99), None)
        mu_region = next((r["mu"] for r in rows
                          if inf["margin"] - r["margin"] <= MARGIN_TOL), None)

        print("\n== %s   (mu -> inf: peak %.3f, margin %.4f m, push %.0f N)"
              % (name, inf["peak"], inf["margin"], inf["push_min"]))
        print("   %5s %6s %7s %8s %8s %7s %8s  %s"
              % ("mu", "u_max", "peak", "d(peak)", "margin", "push",
                 "d(marg)", "binding"))
        for r in rows:
            gap = inf["margin"] - r["margin"]
            dpeak = r["peak"] - inf["peak"]
            binding = ("friction" if r["u_max"] > 0.99 else
                       ("torque" if abs(dpeak) <= PEAK_TOL else "mixed"))
            print("   %5.2f %6.3f %7.3f %8.4f %8.4f %7.0f %8.4f  %s"
                  % (r["mu"], r["u_max"], r["peak"], dpeak, r["margin"],
                     r["push_min"], gap, binding))
        print("   mu*_tau    = %-5s  <-- above this the TORQUE ENVELOPE is what "
              "the pose's cost measures" % mu_tau)
        print("   mu*_slip   = %-5s  no contact on its cone" % mu_slip)
        print("   mu*_region = %-5s  region within %.0f mm of its mu->inf limit"
              % (mu_region, 1000 * MARGIN_TOL))

        out["sets"][name] = dict(rows=rows, inf=inf, mu_tau=mu_tau,
                                 mu_slip=mu_slip, mu_region=mu_region,
                                 ik=dict(reach=ik["reach"],
                                         placed=bool(ik["all_placed"])))

    dst = os.environ.get("OUT", "friction_sweep.json")
    json.dump(out, open(dst, "w"), indent=1)
    print("\nwrote", dst)


if __name__ == "__main__":
    main()
