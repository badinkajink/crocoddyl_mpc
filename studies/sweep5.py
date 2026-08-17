#!/usr/bin/env python3
"""Enumeration over the ENLARGED candidate set: elbow / forearm / palm / hip / torso.

2^5 = 32 subsets.  Coarser target grid than sweep.py so the run stays ~20 min:
8 targets x 32 subsets = 256 IK + QP solves.
"""
import itertools, json, sys
import numpy as np
import contact_select as cs

SITES5 = ("elbow", "forearm", "palm", "hip", "torso")
SUBSETS = [s for k in range(6) for s in itertools.combinations(SITES5, k)]


def run(xs, zs, y=0.15):
    out = []
    for z in zs:
        for x in xs:
            row = {"x": float(x), "y": float(y), "z": float(z), "cands": []}
            for sub in SUBSETS:
                m, d = cs.load()
                P = cs.solve_ik(m, d, np.array([x, y, z]), sub)
                r = cs.equilibrium_qp(m, d, sub)
                adm = bool(P["ok"] and r["base_residual"] < 1.0
                           and r["max_ratio"] <= 1.0)
                n_arm = len([t for t in sub if t in cs.ARM_SITES])
                row["cands"].append(dict(
                    subset=list(sub), admissible=adm, reach=P["reach"],
                    pen=P["penetration"], placed=P["all_placed"],
                    base_res=r["base_residual"], max_ratio=r["max_ratio"],
                    effort=r["effort"], brace=r["brace_force"], n_arm=n_arm))
            a = [c for c in row["cands"] if c["admissible"]]
            row["best"] = (min(a, key=lambda c: (round(c["effort"], 4),
                                                 len(c["subset"])))["subset"]
                           if a else None)
            out.append(row)
            print(f"x={x:.2f} z={z:.2f} -> {row['best']}", flush=True)
    return out


if __name__ == "__main__":
    res = run(np.round(np.arange(0.52, 1.25, 0.24), 2),
              np.round(np.arange(0.95, 1.17, 0.14), 2))
    import os
    name = os.environ.get("OUT", "sweep5.json")   # honour OUT (it did not, and a
    json.dump(res, open(name, "w"), indent=1)     # CL run clobbered the stale-limit
    print("wrote", name, len(res), "targets")     # baseline before the rename)
