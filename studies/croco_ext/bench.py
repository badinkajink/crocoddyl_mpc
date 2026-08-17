#!/usr/bin/env python3
"""How much of the OCP's cost is the keep-out, and how much does C++ buy?

Times the three things that actually matter, on the real S13 problem rather than
a synthetic one:

  build      constructing the ShootingProblem (one activation object per point
             per node -- 86 x 290 = 24 940 of them)
  calcDiff   one full derivative sweep over the horizon, which is what a DDP
             iteration spends most of its time in
  mpc        one warm-started SolverBoxFDDP step over an H-node window, i.e. the
             number that decides whether the receding-horizon controller is
             anywhere near its 20 ms control period

Run per implementation in its own process, because croco_geom picks the backend
at import time.

usage: croco_ext/bench.py [--impl cpp|python|both] [--dir runs/...] [--tag s13]
"""

import argparse
import json
import os
import subprocess
import sys
import time


def bench(run_dir, tag, horizon, reps):
    import numpy as np
    import croco_bridge as cb              # first: sets RTLD_GLOBAL
    import croco_geom as cg
    import croco_replay as cr

    plan = json.load(open(os.path.join(run_dir, f"plan_{tag}.json")))
    xs = np.load(os.path.join(run_dir, f"xs_{tag}.npy"))
    us = np.load(os.path.join(run_dir, f"us_{tag}.npy"))

    t0 = time.time()
    ocp, _ = cr.build_ocp(plan, run_dir)
    t_ocp = time.time() - t0

    t0 = time.time()
    problem = ocp.build(dt=plan["dt"], n_approach=plan["n_approach"],
                        n_braced=plan["n_braced"],
                        n_return=plan.get("n_return", 0),
                        dwell=plan.get("dwell", 0), cones=plan["cones"])
    t_build = time.time() - t0

    xs_l = [np.array(x) for x in xs]
    us_l = [np.array(u) for u in us]
    problem.calc(xs_l, us_l)               # warm the datas
    t0 = time.time()
    for _ in range(reps):
        problem.calcDiff(xs_l, us_l)
    t_calcdiff = (time.time() - t0) / reps

    models = list(problem.runningModels)
    mpc = cr.MPC(ocp, models, problem.terminalModel, horizon=horizon, iters=1,
                 xs_plan=xs, us_plan=us)
    for k in range(6):                      # first call warm-starts from the plan
        mpc(k, xs[k])
    t0 = time.time()
    for k in range(6, 6 + reps * 5):
        mpc(k, xs[k])
    t_mpc = (time.time() - t0) / (reps * 5)

    return dict(impl=cg.IMPL, keepout_points=len(ocp.keepout),
                nodes=problem.T, ocp_s=t_ocp, build_s=t_build,
                calcdiff_ms=1000 * t_calcdiff, mpc_ms=1000 * t_mpc,
                horizon=mpc.H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="both", choices=["cpp", "python", "both"])
    ap.add_argument("--dir", default="runs/2026-08-06_session13")
    ap.add_argument("--tag", default="s13")
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.impl != "both":
        r = bench(args.dir, args.tag, args.horizon, args.reps)
        print(json.dumps(r))
        return

    rows = []
    for impl in ("python", "cpp"):
        env = dict(os.environ, CROCO_KEEPOUT="python" if impl == "python" else "cpp")
        out = subprocess.check_output(
            [sys.executable, os.path.abspath(__file__), "--impl", impl,
             "--dir", args.dir, "--tag", args.tag,
             "--horizon", str(args.horizon), "--reps", str(args.reps)],
            env=env, text=True, cwd=os.getcwd())
        rows.append(json.loads(out.strip().splitlines()[-1]))

    py, cc = rows
    print(f"{'':10s} {'build':>10s} {'calcDiff':>12s} {'MPC step':>12s}")
    for r in rows:
        print(f"{r['impl']:10s} {r['build_s']:9.2f}s {r['calcdiff_ms']:11.1f}ms "
              f"{r['mpc_ms']:11.1f}ms")
    print(f"{'speed-up':10s} {py['build_s']/cc['build_s']:9.1f}x "
          f"{py['calcdiff_ms']/cc['calcdiff_ms']:10.1f}x "
          f"{py['mpc_ms']/cc['mpc_ms']:10.1f}x")
    print(f"\n{py['keepout_points']} keep-out points over {py['nodes']} nodes, "
          f"MPC horizon {py['horizon']}, control period 20 ms")
    if args.out:
        json.dump({"python": py, "cpp": cc}, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
