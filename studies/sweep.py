#!/usr/bin/env python3
"""Reach-target sweep: for each target, which contact subset does the static
formulation pick?

Selection rule (the whole point of the exercise):
  * a subset is ADMISSIBLE if IK converges, the pose is non-penetrating, every
    named site is actually in contact with the table, static equilibrium is
    satisfiable (base residual ~ 0), and no actuator exceeds its torque limit;
  * among admissible subsets, pick the one minimising normalized actuator effort
    sum((tau/tau_max)^2), tie-broken toward FEWER contacts.
"""
import itertools
import json
import sys

import numpy as np

import contact_select as cs

SUBSETS = [s for k in range(4)
           for s in itertools.combinations(("elbow", "forearm", "palm"), k)]


def run(xs, zs, y=0.15, lock_torso=False):
    out = []
    for z in zs:
        for x in xs:
            row = {"x": float(x), "y": float(y), "z": float(z), "cands": []}
            for sub in SUBSETS:
                m, d = cs.load()
                P = cs.solve_ik(m, d, np.array([x, y, z]), sub,
                                lock_torso=lock_torso)
                r = cs.equilibrium_qp(m, d, sub)
                adm = bool(P["ok"] and r["base_residual"] < 1.0
                           and r["max_ratio"] <= 1.0)
                row["cands"].append(dict(
                    subset=list(sub), admissible=adm,
                    reach=P["reach"], pen=P["penetration"],
                    placed=P["all_placed"],
                    base_res=r["base_residual"], max_ratio=r["max_ratio"],
                    effort=r["effort"], brace=r["brace_force"],
                    knee_l=r["knee_l"]))
            adm = [c for c in row["cands"] if c["admissible"]]
            if adm:
                best = min(adm, key=lambda c: (round(c["effort"], 4),
                                               len(c["subset"])))
                row["best"] = best["subset"]
                row["best_effort"] = best["effort"]
                row["best_ratio"] = best["max_ratio"]
                row["best_brace"] = best["brace"]
            else:
                row["best"] = None
            out.append(row)
            print(f"x={x:.2f} z={z:.2f} -> {row['best']}", flush=True)
    return out


if __name__ == "__main__":
    xs = np.round(np.arange(0.40, 1.36, 0.12), 2)
    zs = np.round(np.arange(0.88, 1.17, 0.07), 2)
    lock = "--lock-torso" in sys.argv
    res = run(xs, zs, lock_torso=lock)
    name = "sweep_locktorso.json" if lock else "sweep.json"
    json.dump(res, open(name, "w"), indent=1)
    print("wrote", name, len(res), "targets")
