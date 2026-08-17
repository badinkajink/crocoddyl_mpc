#!/usr/bin/env python3
"""The C++ keep-out must be the SAME function as the Python one, not a faster one.

A speed-up that quietly changes the optimisation problem invalidates every
number the study has already published against the Python activation, so this
checks agreement before anything is allowed to use it:

  * the raw SDF and its gradient on random points, including points placed
    exactly on the box faces, edges and corners -- the branches of `sdf_box`
    that a random sample almost never hits;
  * `a_value` and `Ar` out of a real crocoddyl ActivationData, driven through
    both models;
  * `Arr`, remembering that crocoddyl stores it as a DIAGONAL matrix, so the
    Python model's np.outer(g, g) was only ever contributing its diagonal.

S15 adds the same question one level up.  `CostModelBoxKeepOut` fuses all 86
points into one cost term and skips the inactive ones, which is a much bigger
change than swapping an activation: it bypasses crocoddyl's Gauss-Newton
assembly and writes Lx and Lxx itself.  So the last section builds the real S13
action model both ways -- 86 CostModelResidual terms against one fused term --
and compares `cost`, `Lx` and `Lxx` out of a full calc/calcDiff at states taken
from the planned trajectory, including states where the active set is non-empty.
That is the check that matters: agreeing on an activation proves nothing about
an assembly that no longer calls it.

usage: croco_ext/test_keepout.py [--dir runs/... --tag s13]
"""

import os
import sys

import numpy as np

# Run as `croco_ext/test_keepout.py` from studies/ and sys.path[0] is
# croco_ext/, not studies/ -- so croco_bridge is not importable and the
# script that exists to check the extension cannot import the extension's own
# dispatch module.  Put the study's directory on the path explicitly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import croco_bridge as cb          # noqa: E402  (first: sets RTLD_GLOBAL)
import croco_geom as cg

crocoddyl = cb.import_crocoddyl()
import croco_keepout as ck         # noqa: E402  (must follow the RTLD_GLOBAL import)


HALF = np.array([0.3, 0.6, 0.02])
RMIN = 0.015


def sample_points(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    pts = list(rng.uniform(-1.0, 1.0, (n, 3)) * (HALF * 3 + 0.05))
    # the branch boundaries: faces, edges, corners, centre, and points a hair
    # inside/outside each of them
    for sx in (-1, 0, 1):
        for sy in (-1, 0, 1):
            for sz in (-1, 0, 1):
                p = np.array([sx, sy, sz], float) * HALF
                for e in (0.0, 1e-9, -1e-9, 1e-4, -1e-4):
                    pts.append(p * (1 + e) if np.any(p) else p + e)
    return pts


def fused_vs_per_point(run_dir, tag, verbose=True):
    """The fused cost term against the 86-term stack, on the real action models.

    Both are put in a `DifferentialActionModelContactFwdDynamics` with the SAME
    contacts, actuation and damping and nothing else in the cost sum, so the
    comparison is of the keep-out and not of everything around it.  States come
    from the planned trajectory, subsampled across the approach (where points go
    active) and the braced phase (where the contact bodies sit on their
    thresholds, the boundary case).
    """
    import json
    import croco_plan as cp                                   # noqa: E402
    import croco_replay as cr                                 # noqa: E402

    plan_path = os.path.join(run_dir, f"plan_{tag}.json")
    if not os.path.exists(plan_path):
        print(f"\n(skipping the fused-cost check: no {plan_path})")
        return 0.0
    plan = json.load(open(plan_path))
    ocp, _ = cr.build_ocp(plan, run_dir)
    xs = np.load(os.path.join(run_dir, f"xs_{tag}.npy"))
    us = np.load(os.path.join(run_dir, f"us_{tag}.npy"))

    fused = cg.fused_cost(ocp.state, ocp.nu, ocp.table_half, ocp.table_c,
                          ocp.keepout)
    if fused is None:
        print("\n(skipping the fused-cost check: CROCO_KEEPOUT is not `fused`)")
        return 0.0

    def dam(costs, braced):
        return crocoddyl.DifferentialActionModelContactFwdDynamics(
            ocp.state, ocp.actuation, ocp._contacts(braced), costs,
            cp.INV_DAMPING, True)

    per = crocoddyl.CostModelSum(ocp.state, ocp.nu)
    for p in ocp.keepout:
        per.addCost(f"ko_{p['geom']}_{p['fid']}", crocoddyl.CostModelResidual(
            ocp.state, cg.activation(ocp.table_half, p["thresh"]),
            crocoddyl.ResidualModelFrameTranslation(
                ocp.state, p["fid"], ocp.table_c, ocp.nu)), 1e3)
    one = crocoddyl.CostModelSum(ocp.state, ocp.nu)
    one.addCost("ko_all", fused, 1e3)

    worst = dict(cost=0.0, Lx=0.0, Lxx=0.0)
    n_active_seen = []
    for k in range(0, len(us), max(1, len(us) // 40)):
        braced = k >= plan["n_approach"]
        x, u = xs[k].copy(), us[k].copy()
        vals = []
        for costs in (per, one):
            m = dam(costs, braced)
            d = m.createData()
            m.calc(d, x, u)
            m.calcDiff(d, x, u)
            vals.append((float(d.costs.cost), np.array(d.Lx), np.array(d.Lxx)))
            if costs is one:
                n_active_seen.append(
                    ck.CostModelBoxKeepOut.n_active(d.costs.costs["ko_all"]))
        worst["cost"] = max(worst["cost"], abs(vals[0][0] - vals[1][0]))
        worst["Lx"] = max(worst["Lx"],
                          float(np.max(np.abs(vals[0][1] - vals[1][1]))))
        worst["Lxx"] = max(worst["Lxx"],
                           float(np.max(np.abs(vals[0][2] - vals[1][2]))))
    if verbose:
        print(f"\nfused vs 86-term stack over {len(n_active_seen)} planned "
              f"states ({sum(n > 0 for n in n_active_seen)} with a non-empty "
              f"active set, max {max(n_active_seen)} points active):")
        for k, v in worst.items():
            print(f"  {k:5s} max |per-point - fused| = {v:.3e}")
    return max(worst.values())


def main():
    worst_s = worst_g = 0.0
    for p in sample_points():
        s_py = cg.sdf_box(p, HALF)
        s_cc = ck.sdf_box(p[0], p[1], p[2], *HALF)
        worst_s = max(worst_s, abs(s_py - s_cc))
        g_py = cg.sdf_box_grad(p, HALF)
        g_cc = np.array(ck.sdf_box_grad(p[0], p[1], p[2], *HALF))
        worst_g = max(worst_g, float(np.max(np.abs(g_py - g_cc))))
    print(f"sdf        max |py - c++| = {worst_s:.3e}")
    print(f"sdf grad   max |py - c++| = {worst_g:.3e}")

    a_py = cg.ActivationModelBoxKeepOut(HALF, RMIN)
    a_cc = ck.ActivationModelBoxKeepOut(HALF[0], HALF[1], HALF[2], RMIN)
    d_py, d_cc = a_py.createData(), a_cc.createData()
    worst_v = worst_ar = worst_arr = 0.0
    n_active = 0
    for p in sample_points(2000, seed=1):
        a_py.calc(d_py, p);  a_py.calcDiff(d_py, p)
        a_cc.calc(d_cc, p);  a_cc.calcDiff(d_cc, p)
        worst_v = max(worst_v, abs(d_py.a_value - d_cc.a_value))
        worst_ar = max(worst_ar, float(np.max(np.abs(d_py.Ar - d_cc.Ar))))
        worst_arr = max(worst_arr, float(np.max(np.abs(
            np.diag(d_py.Arr) - np.diag(d_cc.Arr)))))
        n_active += d_py.a_value > 0
    print(f"a_value    max |py - c++| = {worst_v:.3e}   "
          f"({n_active} of 2027 points inside the inflated box)")
    print(f"Ar         max |py - c++| = {worst_ar:.3e}")
    print(f"Arr diag   max |py - c++| = {worst_arr:.3e}")

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "runs", "2026-08-06_session13"))
    ap.add_argument("--tag", default="s13")
    args = ap.parse_args()
    worst_fused = fused_vs_per_point(args.dir, args.tag)

    ok = max(worst_s, worst_g, worst_v, worst_ar, worst_arr) < 1e-12
    # The fused term reassociates the same sums (it accumulates one Lxx instead
    # of 86 weighted ones), so it is held to floating-point agreement rather than
    # to bit equality.  Lxx entries are O(1e3) here, so 1e-9 is ~1e-12 relative.
    ok = ok and worst_fused < 1e-9
    print("\nAGREE" if ok else "\nDISAGREE -- do not use the C++ path")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
