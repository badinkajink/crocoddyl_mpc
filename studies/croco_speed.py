#!/usr/bin/env python3
"""Where the crocoddyl MPC step actually goes, term by term.

S14 asked "is the Delassus the bottleneck" and answered no: the contact dynamics
are ~27% of a node and the Delassus 2.3% of it.  It attributed the other
three-quarters to "the cost stack, overwhelmingly the 86 box keep-out residuals"
and stopped there.  That attribution was never measured against the OTHER cost
terms, and it has to be, because the thing that makes a crocoddyl node expensive
is not obviously the maths in any one term:

    CostModelSum::calcDiff  does, PER ACTIVE COST TERM,
        Lx  += w * d_i->Lx        (n x = 66)
        Lxx += w * d_i->Lxx       (66 x 66  = 4356 doubles)
        Lxu += w * d_i->Lxu       (66 x 27  = 1782)
        Luu += w * d_i->Luu       (27 x 27  =  729)     -- 6960 doubles

so every term -- a 3-row keep-out point and a 66-row state regulariser alike --
pays 6960 multiply-adds of dense accumulation plus its own residual work, and
every term owns a private CostDataAbstract carrying those same matrices.  A node
with 98 terms therefore does ~680 000 accumulate FLOPs over 5.4 MB of scattered
data before any residual has been differentiated.  If that is where the time is,
then the lever is the NUMBER OF TERMS and not the cost of any one of them, and
"make the keep-out activation faster" (S13) or "make the Delassus faster" (S14)
were both attacking the wrong axis.

This module measures which of those it is:

  terms    Per-group ablation of the real S13 action models.  Each cost group is
           switched off with `changeCostStatus` and the node is re-timed, so the
           number reported for a group is what removing it saves -- including its
           share of the CostModelSum accumulation, which is the point.
  scaling  The per-term fixed cost, isolated: N copies of the SAME trivial cost
           in one CostModelSum, timed against N.  The slope is what a cost term
           costs before it computes anything.
  step     The MPC step decomposed into problem.calc / problem.calcDiff /
           backwardPass / forwardPass over the real 50-node horizon, so the
           node-level numbers can be checked against the 76 ms they add up to.

usage:
  croco_speed.py all     --dir runs/2026-08-06_session13 --tag s13
  croco_speed.py terms|scaling|step [--out speed.json]
"""

import argparse
import ctypes
import json
import os
import sys
import time

# RTLD_GLOBAL before any native extension loads -- croco_bridge explains why.
sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "croco_ext"))

import croco_bridge as cb                      # noqa: E402  (must be first)

crocoddyl = cb.import_crocoddyl()

RUN_DEFAULT = os.path.join(HERE, "runs", "2026-08-06_session13")
TAG_DEFAULT = "s13"

# Cost-name prefixes -> the group they belong to.  Longest prefix wins, so
# "restReg" does not fall into "reach" and "stateReg" does not fall into "state".
GROUPS = [
    ("ko_", "keepout"),
    ("cone_", "cones"),
    ("land_", "land"),
    ("hold_", "hold"),
    ("above_", "above"),
    ("jointLim", "jointLim"),
    ("stateReg", "stateReg"),
    ("ctrlReg", "ctrlReg"),
    ("restReg", "restReg"),
    ("comSupport", "com"),
    ("reach", "reach"),
]


def group_of(name):
    for pre, g in GROUPS:
        if name.startswith(pre):
            return g
    return "other"


def timeit(fn, reps=200, warm=3):
    """Best-of-reps, in microseconds (see croco_delassus.timeit for why best-of)."""
    for _ in range(warm):
        fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e6


def load_ocp(run_dir, tag):
    """The real S13 OCP, its shooting problem, and the planned trajectory."""
    import croco_replay as cr
    plan = json.load(open(os.path.join(run_dir, f"plan_{tag}.json")))
    ocp, _ = cr.build_ocp(plan, run_dir)
    problem = ocp.build(dt=plan["dt"], n_approach=plan["n_approach"],
                        n_braced=plan["n_braced"],
                        n_return=plan.get("n_return", 0),
                        dwell=plan.get("dwell", 0), cones=plan["cones"])
    xs = np.load(os.path.join(run_dir, f"xs_{tag}.npy"))
    us = np.load(os.path.join(run_dir, f"us_{tag}.npy"))
    return plan, ocp, problem, xs, us


def node_indices(problem, xs):
    """One approach node (feet only) and one braced node, with their states."""
    out = {}
    for idx, name in ((0, "approach"), (problem.T - 1, "braced")):
        dm = problem.runningModels[idx].differential
        out[name] = dict(index=idx, nc=int(dm.contacts.nc),
                         x=xs[idx].copy(), model=dm)
    return out


# ----------------------------------------------------------------- terms --- #
def cmd_terms(args):
    """Per-cost-group ablation on the real action models.

    `changeCostStatus(name, False)` makes CostModelSum skip the term entirely --
    its residual, its activation AND its accumulation into the shared Lxx -- so
    the difference is the whole cost of having that group in the problem, which
    is what a decision to restructure it needs.  The data is still allocated, so
    this measures run time and not memory.
    """
    plan, ocp, problem, xs, us = load_ocp(args.dir, args.tag)
    out = []
    for name, nd in node_indices(problem, xs).items():
        dm, x, u = nd["model"], nd["x"], us[nd["index"]].copy()
        data = dm.createData()
        names = list(dm.costs.costs.todict().keys())
        groups = {}
        for n in names:
            groups.setdefault(group_of(n), []).append(n)

        def run():
            dm.calc(data, x, u)
            dm.calcDiff(data, x, u)

        full = timeit(run, args.reps)
        rows = []
        for g, members in sorted(groups.items()):
            for n in members:
                dm.costs.changeCostStatus(n, False)
            t = timeit(run, args.reps)
            for n in members:
                dm.costs.changeCostStatus(n, True)
            rows.append(dict(group=g, n_terms=len(members), us=full - t,
                             per_term_us=(full - t) / len(members)))
        # Everything off: what is left is the contact dynamics plus the empty
        # CostModelSum's own zeroing.
        for n in names:
            dm.costs.changeCostStatus(n, False)
        bare = timeit(run, args.reps)
        for n in names:
            dm.costs.changeCostStatus(n, True)

        rows.sort(key=lambda r: -r["us"])
        out.append(dict(node=name, nc=nd["nc"], n_terms=len(names),
                        full_us=full, bare_us=bare, costs_us=full - bare,
                        groups=rows))

    for r in out:
        print(f"--- {r['node']} node   nc={r['nc']}   {r['n_terms']} cost terms")
        print(f"    calc+calcDiff, everything on : {r['full_us']:8.1f} us")
        print(f"    contact dynamics alone       : {r['bare_us']:8.1f} us"
              f"   ({100*r['bare_us']/r['full_us']:.0f}%)")
        print(f"    {'group':10s} {'terms':>6s} {'us':>8s} {'us/term':>9s} "
              f"{'share':>7s}")
        for g in r["groups"]:
            print(f"    {g['group']:10s} {g['n_terms']:6d} {g['us']:8.1f} "
                  f"{g['per_term_us']:9.2f} {100*g['us']/r['full_us']:6.1f}%")
    print(f"  (microseconds, best of {args.reps})")
    return dict(terms=out)


# --------------------------------------------------------------- scaling --- #
def cmd_scaling(args):
    """The fixed cost of a cost TERM, isolated from what the term computes.

    N copies of the same 3-row keep-out cost are put in one CostModelSum on the
    real state and timed against N.  Every copy computes the identical residual
    at the identical point, so the slope is the marginal cost of one more term:
    its residual + activation + the dense accumulation CostModelSum does for it.
    A second series with a 1-row control cost separates "the accumulation"
    (identical for both) from "the residual" (much cheaper for the 1-row one).
    """
    plan, ocp, problem, xs, us = load_ocp(args.dir, args.tag)
    import croco_geom as cg
    x = xs[0].copy()
    u = us[0].copy()
    p = ocp.keepout[0]
    series = {}
    for kind in ("keepout", "ctrl"):
        rows = []
        for n in args.counts:
            costs = crocoddyl.CostModelSum(ocp.state, ocp.nu)
            for i in range(n):
                if kind == "keepout":
                    costs.addCost(f"c{i}", crocoddyl.CostModelResidual(
                        ocp.state, cg.activation(ocp.table_half, p["thresh"]),
                        crocoddyl.ResidualModelFrameTranslation(
                            ocp.state, p["fid"], ocp.table_c, ocp.nu)), 1e3)
                else:
                    costs.addCost(f"c{i}", crocoddyl.CostModelResidual(
                        ocp.state, crocoddyl.ResidualModelControl(
                            ocp.state, ocp.nu)), 1e-3)
            dm = crocoddyl.DifferentialActionModelContactFwdDynamics(
                ocp.state, ocp.actuation, ocp._contacts(False), costs,
                1e-4, True)
            data = dm.createData()

            def run():
                dm.calc(data, x, u)
                dm.calcDiff(data, x, u)

            rows.append(dict(n=n, us=timeit(run, args.reps)))
        # least squares slope through the series
        A = np.array([[r["n"], 1.0] for r in rows])
        b = np.array([r["us"] for r in rows])
        slope, icept = np.linalg.lstsq(A, b, rcond=None)[0]
        series[kind] = dict(points=rows, per_term_us=float(slope),
                            intercept_us=float(icept))
        print(f"--- {n} identical `{kind}` cost terms in one CostModelSum")
        for r in rows:
            print(f"      n={r['n']:4d}  {r['us']:9.1f} us")
        print(f"    marginal cost of one term: {slope:.2f} us "
              f"(intercept {icept:.1f} us)")
    print(f"  nv={ocp.rmodel.nv}, nx={2*ocp.rmodel.nv}, nu={ocp.nu}; "
          f"CostModelSum accumulates {2*ocp.rmodel.nv}^2 + "
          f"{2*ocp.rmodel.nv}x{ocp.nu} + {ocp.nu}^2 = "
          f"{(2*ocp.rmodel.nv)**2 + 2*ocp.rmodel.nv*ocp.nu + ocp.nu**2} "
          f"doubles per term")
    return dict(scaling=series)


# ------------------------------------------------------------------ step --- #
def cmd_step(args):
    """The MPC step, decomposed, over the real horizon at the plan's states."""
    plan, ocp, problem, xs, us = load_ocp(args.dir, args.tag)
    H = args.horizon
    k0 = plan["n_approach"] - H // 2         # a window straddling touchdown
    models = list(problem.runningModels)[k0:k0 + H]
    sub = crocoddyl.ShootingProblem(xs[k0], models, problem.terminalModel)
    solver = crocoddyl.SolverBoxFDDP(sub)
    xs_w = [np.array(v) for v in xs[k0:k0 + H + 1]]
    us_w = [np.array(v) for v in us[k0:k0 + H]]
    solver.solve(xs_w, us_w, 1, False, 1e-9)   # allocate + warm

    rows = dict(
        horizon=H, start_node=k0,
        problem_calc=timeit(lambda: sub.calc(xs_w, us_w), args.reps // 4 or 1),
        problem_calcDiff=timeit(lambda: sub.calcDiff(xs_w, us_w),
                                args.reps // 4 or 1),
        backwardPass=timeit(solver.backwardPass, args.reps // 4 or 1),
    )
    rows["solve_1_iter"] = timeit(
        lambda: solver.solve(xs_w, us_w, 1, False, 1e-9), 20)
    rows["solve_2_iter"] = timeit(
        lambda: solver.solve(xs_w, us_w, 2, False, 1e-9), 20)
    print(f"--- {H}-node window from node {k0}, microseconds (best of)")
    for k in ("problem_calc", "problem_calcDiff", "backwardPass",
              "solve_1_iter", "solve_2_iter"):
        print(f"    {k:20s} {rows[k]/1000:9.2f} ms")
    print(f"    per node: calc {rows['problem_calc']/H:7.1f} us   "
          f"calcDiff {rows['problem_calcDiff']/H:7.1f} us")
    return dict(step=rows)


# ---------------------------------------------------------------- pieces --- #
def cmd_pieces(args):
    """The three things left inside a node once the cost stack is one term.

    actuation   The plant's passive joint torques are a PYTHON subclass of
                ActuationModelAbstract (croco_plan._make_actuation).  Its
                calcDiff allocates a 33x66 zero matrix and a np.diag per call,
                per node, per sweep.  Timed against crocoddyl's own
                ActuationModelFloatingBase, whose only difference is that it does
                not carry the damping and friction.
    force       `enable_force` in DifferentialActionModelContactFwdDynamics
                gates the contact-force derivatives (df_dx, df_du and the
                per-contact updateForceDiff).  ONLY the cone costs read them, so
                a horizon without cones can switch them off -- this is what that
                is worth.
    cones       The cone costs themselves, already in `terms`, repeated here at
                the node level so the two savings can be added honestly.
    """
    plan, ocp, problem, xs, us = load_ocp(args.dir, args.tag)
    idx = problem.T - 1
    x, u = xs[idx].copy(), us[idx].copy()
    dm = problem.runningModels[idx].differential

    rows = {}
    # --- the actuation model, on its own.  Both are built explicitly: since S15
    # `_make_actuation` returns the C++ one by default, so `dm.actuation` alone
    # would silently time whichever the environment picked.
    import croco_plan as cp
    for label, mode in (("py", "python"), ("cpp", "cpp")):
        os.environ["CROCO_PASSIVE"] = mode
        a = cp._make_actuation(ocp.state, ocp.damping, ocp.friction)
        ad = a.createData()
        rows[f"actuation_{label}_calc_us"] = timeit(
            lambda: a.calc(ad, x, u), args.reps)
        rows[f"actuation_{label}_calcDiff_us"] = timeit(
            lambda: a.calcDiff(ad, x, u), args.reps)
    os.environ.pop("CROCO_PASSIVE", None)
    stock = crocoddyl.ActuationModelFloatingBase(ocp.state)
    sd = stock.createData()
    rows["stock_actuation_calc_us"] = timeit(
        lambda: stock.calc(sd, x, u), args.reps)
    rows["stock_actuation_calcDiff_us"] = timeit(
        lambda: stock.calcDiff(sd, x, u), args.reps)

    # --- the node, with and without the force derivatives / the cones
    def node_us(costs, enable_force):
        m = crocoddyl.DifferentialActionModelContactFwdDynamics(
            ocp.state, dm.actuation, dm.contacts, costs, 1e-4, enable_force)
        d = m.createData()

        def run():
            m.calc(d, x, u)
            m.calcDiff(d, x, u)
        return timeit(run, args.reps)

    full = dm.costs
    nocone = crocoddyl.CostModelSum(ocp.state, ocp.nu)
    for name, item in full.costs.todict().items():
        if not name.startswith("cone_"):
            nocone.addCost(name, item.cost, item.weight)
    rows["node_full_us"] = node_us(full, True)
    rows["node_nocones_us"] = node_us(nocone, True)
    rows["node_nocones_noforce_us"] = node_us(nocone, False)

    print(f"--- braced node {idx}, microseconds (best of {args.reps})")
    print(f"    actuation (python passive)  calc "
          f"{rows['actuation_py_calc_us']:6.2f}"
          f"  calcDiff {rows['actuation_py_calcDiff_us']:6.2f}")
    print(f"    actuation (c++ passive)     calc "
          f"{rows['actuation_cpp_calc_us']:6.2f}"
          f"  calcDiff {rows['actuation_cpp_calcDiff_us']:6.2f}")
    print(f"    actuation (crocoddyl stock) calc "
          f"{rows['stock_actuation_calc_us']:6.2f}"
          f"  calcDiff {rows['stock_actuation_calcDiff_us']:6.2f}")
    print(f"    node, everything                    {rows['node_full_us']:7.1f}")
    print(f"    node, cones dropped                 "
          f"{rows['node_nocones_us']:7.1f}")
    print(f"    node, cones dropped + enable_force=0 "
          f"{rows['node_nocones_noforce_us']:6.1f}")
    return dict(pieces=rows)


# ---------------------------------------------------------------- solver --- #
def cmd_solver(args):
    """What a warm-started BoxFDDP step spends outside calcDiff.

    `step` accounts for calcDiff and the backward pass and leaves ~45% of the
    solve unexplained.  FDDP's forward pass is a full nonlinear rollout PER
    LINE-SEARCH TRIAL, so the missing time is a count: how many trial steps the
    solver takes before it accepts one.  Measured by timing `tryStep` (one
    rollout) and reading `stepLength` after a real solve.
    """
    plan, ocp, problem, xs, us = load_ocp(args.dir, args.tag)
    H = args.horizon
    k0 = plan["n_approach"] - H // 2
    models = list(problem.runningModels)[k0:k0 + H]
    sub = crocoddyl.ShootingProblem(xs[k0], models, problem.terminalModel)
    solver = crocoddyl.SolverBoxFDDP(sub)
    xs_w = [np.array(v) for v in xs[k0:k0 + H + 1]]
    us_w = [np.array(v) for v in us[k0:k0 + H]]

    steps = []
    for _ in range(12):
        solver.solve(xs_w, us_w, args.iters, False, 1e-9)
        steps.append(float(solver.stepLength))
    rows = dict(
        horizon=H, iters=args.iters,
        step_lengths=steps,
        trials_per_iter=[int(round(np.log2(1.0 / s))) + 1 if s > 0 else len(
            solver.alphas) for s in steps],
        n_alphas=len(solver.alphas),
        tryStep_us=timeit(lambda: solver.tryStep(1.0), args.reps // 4 or 1),
        calcDiff_us=timeit(solver.calcDiff, args.reps // 4 or 1),
        backwardPass_us=timeit(solver.backwardPass, args.reps // 4 or 1),
    )
    for n in (1, 2, 3):
        rows[f"solve_{n}_iter_us"] = timeit(
            lambda n=n: solver.solve(xs_w, us_w, n, False, 1e-9), 20)
    print(f"--- {H}-node window, {args.iters} iters")
    print(f"    solver.calcDiff (calc + calcDiff)  "
          f"{rows['calcDiff_us']/1000:7.2f} ms")
    print(f"    backwardPass                       "
          f"{rows['backwardPass_us']/1000:7.2f} ms")
    print(f"    tryStep(1.0)  = one rollout        "
          f"{rows['tryStep_us']/1000:7.2f} ms")
    print(f"    accepted step lengths over 12 solves: "
          f"{sorted(set(steps))}")
    for n in (1, 2, 3):
        print(f"    solve({n} iter)                     "
              f"{rows[f'solve_{n}_iter_us']/1000:7.2f} ms")
    print(f"    marginal cost of one more iteration  "
          f"{(rows['solve_3_iter_us'] - rows['solve_1_iter_us'])/2000:7.2f} ms")
    return dict(solver=rows)


# ----------------------------------------------------------------- sweep --- #
def cmd_sweep(args):
    """(horizon x iterations x cones) against BOTH speed and what survives.

    Every knob below buys time by giving something up, and the only way that is
    a decision rather than a hope is to run the full MuJoCo replay for each and
    score it the way croco_replay already scores every controller.  So each cell
    is a real 4 s replay, not a solver benchmark: the reported reach error, table
    penetration and support margin are what the robot did, and the reported
    milliseconds are the mean over its 200 control steps.

    The baseline cell (H=50, 2 iters, cones on) is the S13 controller.
    """
    import croco_replay as cr
    rows = []
    for cones in args.cones:
      for na in args.alphas:
        for H in args.horizons:
            for it in args.iters_grid:
                log, plan = cr.replay(
                    args.tag, ctrl_mode="mpc", dt_plan=0.02, run_dir=args.dir,
                    mpc_horizon=H, mpc_iters=it, mpc_cones=bool(cones),
                    mpc_alphas=na, seed=args.seed, q_noise=args.q_noise)
                s = cr.summarise(log, plan, "mpc", verbose=False)
                r = dict(horizon=H, iters=it, cones=bool(cones), alphas=na,
                         solve_ms=s["mpc_solve_ms"]["mean"],
                         p95_ms=s["mpc_solve_ms"]["p95"],
                         trials_median=s["mpc_solve_ms"]["trials_median"],
                         hz=1000.0 / s["mpc_solve_ms"]["mean"],
                         fell=s["fell"],
                         reach_mm=1000 * s["reach_err_at_brace_end"],
                         pen_mm=1000 * s["worst_penetration"],
                         margin_mm=1000 * s["min_support_margin"],
                         tau_ratio=s["max_tau_ratio"],
                         brace_N=s["brace_total_braced_mean"])
                rows.append(r)
                print(f"    H={H:3d} it={it} cones={int(bool(cones))} "
                      f"a={na or 10:2d}  "
                      f"{r['solve_ms']:6.1f} ms ({r['hz']:5.1f} Hz)  "
                      f"reach {r['reach_mm']:6.1f} mm  pen {r['pen_mm']:6.1f} mm  "
                      f"margin {r['margin_mm']:+6.1f} mm  "
                      f"{'FELL' if r['fell'] else 'upright'}")
    print(f"\n    {'H':>3s} {'it':>2s} {'cone':>4s} {'a':>3s} {'ms':>7s} "
          f"{'p95':>6s} {'Hz':>6s} {'reach':>7s} {'pen':>6s} {'margin':>7s} "
          f"{'tau':>5s}")
    for r in sorted(rows, key=lambda z: z["solve_ms"]):
        print(f"    {r['horizon']:3d} {r['iters']:2d} {int(r['cones']):4d} "
              f"{r['alphas'] or 10:3d} "
              f"{r['solve_ms']:7.1f} {r['p95_ms']:6.1f} {r['hz']:6.1f} "
              f"{r['reach_mm']:7.1f} "
              f"{r['pen_mm']:6.1f} {r['margin_mm']:+7.1f} {r['tau_ratio']:5.2f}"
              f"{'  FELL' if r['fell'] else ''}")
    return dict(sweep=rows)


# ---------------------------------------------------------------- ladder --- #
# The speed-up, one change at a time, each rung a full MuJoCo replay.
#
# Every rung is a SUBPROCESS because `CROCO_KEEPOUT` and `CROCO_PASSIVE` are read
# at import time -- croco_geom picks its activation and croco_plan its actuation
# when the module first loads, which is the only point at which the choice is
# cheap.  Flipping them in-process would time whatever was imported first.
LADDER = [
    dict(name="S12: python activation, per-point",
         env=dict(CROCO_KEEPOUT="python", CROCO_PASSIVE="python"),
         args=["--mpc-horizon", "50", "--mpc-iters", "2"]),
    dict(name="S13: C++ activation, per-point",
         env=dict(CROCO_KEEPOUT="cpp", CROCO_PASSIVE="python"),
         args=["--mpc-horizon", "50", "--mpc-iters", "2"]),
    dict(name="S15a: + fused keep-out cost",
         env=dict(CROCO_KEEPOUT="fused", CROCO_PASSIVE="python"),
         args=["--mpc-horizon", "50", "--mpc-iters", "2"]),
    dict(name="S15b: + C++ passive actuation",
         env=dict(CROCO_KEEPOUT="fused", CROCO_PASSIVE="cpp"),
         args=["--mpc-horizon", "50", "--mpc-iters", "2"]),
    dict(name="S15c: + one DDP iteration",
         env=dict(CROCO_KEEPOUT="fused", CROCO_PASSIVE="cpp"),
         args=["--mpc-horizon", "50", "--mpc-iters", "1"]),
    dict(name="S15d: + horizon 50 -> 35 nodes",
         env=dict(CROCO_KEEPOUT="fused", CROCO_PASSIVE="cpp"),
         args=["--mpc-horizon", "35", "--mpc-iters", "1"]),
]


def cmd_ladder(args):
    """Each change in turn, scored by a full replay, on this machine."""
    import subprocess
    rows = []
    for i, rung in enumerate(LADDER):
        env = dict(os.environ)
        env.update(rung["env"])
        cmd = [sys.executable, os.path.join(HERE, "croco_replay.py"),
               "--tag", args.tag, "--ctrl", "mpc", "--dir", args.dir,
               "--suffix", f"_ladder{i}"] + rung["args"]
        subprocess.run(cmd, env=env, check=True, cwd=HERE,
                       stdout=subprocess.DEVNULL)
        res = json.load(open(os.path.join(
            args.dir, f"replay_{args.tag}_mpc_ladder{i}.json")))["summary"]
        s = res["mpc_solve_ms"]
        rows.append(dict(rung=rung["name"], **rung["env"],
                         horizon=s["horizon"], iters=s["iters"],
                         solve_ms=s["mean"], p95_ms=s["p95"],
                         hz=1000.0 / s["mean"],
                         trials_median=s["trials_median"],
                         fell=res["fell"],
                         reach_mm=1000 * res["reach_err_at_brace_end"],
                         pen_mm=1000 * res["worst_penetration"],
                         margin_mm=1000 * res["min_support_margin"],
                         brace_N=res["brace_total_braced_mean"]))
        r = rows[-1]
        print(f"    {r['rung']:34s} {r['solve_ms']:6.1f} ms  p95 {r['p95_ms']:5.1f}"
              f"  {r['hz']:5.1f} Hz   reach {r['reach_mm']:5.1f} mm"
              f"  {'FELL' if r['fell'] else 'upright'}")
    base = rows[0]["solve_ms"]
    print(f"\n    cumulative speed-up over the S12 rung: "
          f"{base / rows[-1]['solve_ms']:.1f}x "
          f"({base:.0f} -> {rows[-1]['solve_ms']:.0f} ms)")
    print(f"    over the S13 rung (the published controller): "
          f"{rows[1]['solve_ms'] / rows[-1]['solve_ms']:.1f}x")
    return dict(ladder=rows)


# --------------------------------------------------------------- threads --- #
def cmd_threads(args):
    """MPC step against ShootingProblem.nthreads, each cell a full replay.

    crocoddyl parallelises exactly two loops -- the per-node loops inside
    `ShootingProblem::calc` and `::calcDiff` -- and nothing else.  The backward
    pass is a Riccati recursion and is sequential by construction.  So the
    ceiling here is Amdahl over the split croco_speed.py solver measures
    (~45% derivative sweep, 25% line-search rollouts, 25% backward pass): the
    rollouts go through `calc` and the sweep through `calcDiff`, so ~70% of the
    step is parallel and the asymptote is ~3.3x however many threads are thrown
    at it.

    `nthreads_effective` is read back off the problem rather than assumed,
    because against a libcrocoddyl built WITHOUT OpenMP `set_nthreads` prints a
    warning and pins it to 1 -- so a table of "requested" would silently report
    a speed-up that never happened.
    """
    import croco_replay as cr

    # ONE THROWAWAY REPLAY BEFORE THE SWEEP, and it is not superstition.
    # The closed loop is CHAOTIC: a 1e-12 rad perturbation of the start moves the
    # reach error from 5.34 mm to 12.39 mm (measured), and building an OCP leaves
    # enough process state -- allocator, and so Eigen's aligned/unaligned kernel
    # choice -- to act as exactly such a perturbation. So the FIRST replay in a
    # process lands on a different branch from every later one, and without this
    # line the first row of the table below differs from the rest for a reason
    # that has nothing to do with the thing being swept. It looked like
    # "1 thread gives a different trajectory", i.e. like a race in the parallel
    # build, which is the one conclusion this sweep must not get wrong.
    #
    # It also means the trajectory columns are NOT an A/B signal here: two builds
    # that differ only in floating-point association are expected to land on
    # different branches. What IS a signal is a matched comparison -- same rep
    # index, stock vs preloaded library -- and that one is bit-identical.
    cr.replay(args.tag, ctrl_mode="mpc", dt_plan=0.02, run_dir=args.dir,
              mpc_horizon=args.horizon, mpc_iters=args.iters, mpc_threads=1)

    rows = []
    for n in args.threads:
        log, plan = cr.replay(
            args.tag, ctrl_mode="mpc", dt_plan=0.02, run_dir=args.dir,
            mpc_horizon=args.horizon, mpc_iters=args.iters,
            mpc_threads=n, seed=args.seed, q_noise=args.q_noise)
        sm = cr.summarise(log, plan, "mpc", verbose=False)
        s_ = sm["mpc_solve_ms"]
        rows.append(dict(requested=n, effective=s_.get("nthreads_effective", 1),
                         solve_ms=s_["mean"], p95_ms=s_["p95"],
                         hz=1000.0 / s_["mean"], fell=sm["fell"],
                         reach_mm=1000 * sm["reach_err_at_brace_end"],
                         pen_mm=1000 * sm["worst_penetration"],
                         margin_mm=1000 * sm["min_support_margin"]))
        r = rows[-1]
        print(f"    threads {r['requested']:3d} (effective {r['effective']:2d})  "
              f"{r['solve_ms']:6.2f} ms  p95 {r['p95_ms']:6.2f}  "
              f"{r['hz']:6.1f} Hz   reach {r['reach_mm']:5.1f} mm  "
              f"{'FELL' if r['fell'] else 'upright'}")
    base = rows[0]["solve_ms"]
    print(f"\n    {'threads':>7s} {'ms':>7s} {'p95':>7s} {'Hz':>7s} "
          f"{'speed-up':>9s} {'efficiency':>11s}")
    for r in rows:
        n = max(r["effective"], 1)
        print(f"    {r['effective']:7d} {r['solve_ms']:7.2f} {r['p95_ms']:7.2f} "
              f"{r['hz']:7.1f} {base / r['solve_ms']:8.2f}x "
              f"{100 * (base / r['solve_ms']) / n:10.0f}%")
    return dict(threads=rows)


# ----------------------------------------------------------------- stage --- #
def cmd_stage(args):
    """Which stage of a step the threads actually reach.

    crocoddyl's OpenMP is in the per-node loops of `ShootingProblem::calc` and
    `::calcDiff`, and a step reaches only ONE of them.  Checked in
    src/core/solvers/ddp.cpp rather than assumed:

      solver.calcDiff  -> problem_->calc + problem_->calcDiff   BOTH parallel
      forwardPass      -> a plain `for (t...) m->calc(d, xs_try_[t], ...)`, no
                          pragma, because the rollout is a dependency chain
                          (x_{t+1} comes from step t's xnext).  It never calls
                          ShootingProblem::calc at all, so the thread count
                          cannot touch it.
      backwardPass     -> Riccati recursion, node t+1 feeds node t.  Sequential
                          by construction.

    So the parallel fraction of a step is the derivative sweep alone, and the
    other two set the floor.  That is the difference between a 3x lever and a
    1.5x one, and it is measured here rather than estimated.
    """
    plan, ocp, problem, xs, us = load_ocp(args.dir, args.tag)
    H = args.horizon
    k0 = plan["n_approach"] - H // 2
    models = list(problem.runningModels)[k0:k0 + H]
    sub = crocoddyl.ShootingProblem(xs[k0], models, problem.terminalModel)
    solver = crocoddyl.SolverBoxFDDP(sub)
    xs_w = [np.array(v) for v in xs[k0:k0 + H + 1]]
    us_w = [np.array(v) for v in us[k0:k0 + H]]
    solver.solve(xs_w, us_w, 1, False, 1e-9)

    n_hi = max(args.threads)
    out = []
    meas = {}
    for n in (1, n_hi):
        sub.nthreads = n
        eff = int(sub.nthreads)
        meas[n] = dict(
            eff=eff,
            calcDiff=timeit(solver.calcDiff, args.reps // 4 or 1),
            backward=timeit(solver.backwardPass, args.reps // 4 or 1),
            rollout=timeit(lambda: solver.tryStep(1.0), args.reps // 4 or 1))
    for key, label, parallel in (
            ("calcDiff", "solver.calcDiff (calc + calcDiff sweep)", True),
            ("rollout", "one line-search rollout (SolverDDP::forwardPass)", False),
            ("backward", "backwardPass (Riccati + BoxQP)", False)):
        out.append(dict(stage=label, parallel=parallel,
                        t1_ms=meas[1][key] / 1000.0,
                        tn_ms=meas[n_hi][key] / 1000.0))
    eff = meas[n_hi]["eff"]
    print(f"--- {H}-node window, 1 thread vs {eff}")
    for r in out:
        print(f"    {r['stage']:42s} {r['t1_ms']:7.2f} -> {r['tn_ms']:6.2f} ms"
              f"   {r['t1_ms']/max(r['tn_ms'],1e-9):5.2f}x"
              f"   {'parallel' if r['parallel'] else 'SEQUENTIAL'}")
    # The floor is everything the threads cannot touch.  The rollout is counted
    # ONCE, so this is a lower bound: FDDP's line search takes ~4 of them per
    # iteration, and each one is on this side of the ledger.
    seq = sum(r["tn_ms"] for r in out if not r["parallel"])
    print(f"    sequential floor for a 1-iteration step: {seq:.2f} ms"
          f"  (backward pass + ONE rollout; the line search takes ~4)")
    return dict(stage=out, sequential_ms=seq, nthreads=eff, horizon=H)


# -------------------------------------------------------------- manifest --- #
def cmd_manifest(args):
    """Provenance, plus the allocation cost the term count also drives.

    The per-term accounting has a memory half that a timing table does not show:
    every cost term owns a CostDataAbstract carrying its own Lxx, Lxu and Luu, so
    the number of terms sets the size of the working set a calcDiff sweep has to
    stream.  Reported as measured allocation time and as the exact byte count
    implied by the layout, since both are what the cache is reacting to.
    """
    import platform
    import subprocess
    import croco_geom as cg
    plan, ocp, problem, xs, us = load_ocp(args.dir, args.tag)
    models = list(problem.runningModels)
    t0 = time.perf_counter()
    datas = [m.createData() for m in models]
    alloc_ms = 1000 * (time.perf_counter() - t0)
    n_terms = [len(m.differential.costs.costs.todict()) for m in models]
    ndx, nu = 2 * ocp.rmodel.nv, ocp.nu
    per_term_bytes = 8 * (ndx * ndx + ndx * nu + nu * nu + ndx + nu)

    import crocoddyl as _c
    import pinocchio as _p
    try:
        # LD_PRELOAD is inherited by children, and preloading libcrocoddyl into
        # `git` makes git fail to start -- which silently turned the recorded
        # commit into "unknown" on every measurement taken under the OpenMP
        # build.  Strip it for the subprocess only.
        env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         text=True, cwd=HERE, env=env).strip()
    except Exception:
        commit = "unknown"

    # Whether multithreading is live is a property of the LOADED LIBRARY, not of
    # the Python bindings -- the pywrap is the stock one either way, so its
    # module constants say nothing.  Ask a real problem: set_nthreads pins the
    # count to 1 and warns when the library lacks OpenMP, and honours it when it
    # does not.
    _probe = crocoddyl.ShootingProblem(
        np.zeros(3), [crocoddyl.ActionModelUnicycle()] * 2,
        crocoddyl.ActionModelUnicycle())
    _probe.nthreads = 2
    mt_live = int(_probe.nthreads) == 2
    out = dict(
        keepout_impl=cg.IMPL,
        passive_impl=type(ocp.actuation).__name__,
        crocoddyl=_c.__version__, pinocchio=_p.__version__,
        multithreading=mt_live,
        multithreading_default=int(_probe.nthreads),
        # Which libcrocoddyl is actually mapped -- the OpenMP rebuild is
        # selected by LD_LIBRARY_PATH, so "which build am I measuring" is a
        # question about the process and not about the environment.
        libcrocoddyl=next((l.split()[-1] for l in open("/proc/self/maps")
                           if "libcrocoddyl.so" in l), "unknown"),
        cpu=platform.processor() or platform.machine(),
        cpu_model=next((l.split(":", 1)[1].strip()
                        for l in open("/proc/cpuinfo")
                        if l.startswith("model name")), "unknown"),
        n_cpu=os.cpu_count(), python=platform.python_version(),
        commit=commit, run_dir=args.dir, tag=args.tag,
        nv=int(ocp.rmodel.nv), nu=int(nu), ndx=int(ndx),
        n_keepout_points=len(ocp.keepout),
        n_models=len(models),
        cost_terms_min=int(min(n_terms)), cost_terms_max=int(max(n_terms)),
        createData_ms=alloc_ms,
        per_term_bytes=int(per_term_bytes),
        costdata_mb_per_node=per_term_bytes * float(np.mean(n_terms)) / 1e6,
        costdata_mb_per_50_horizon=(per_term_bytes * float(np.mean(n_terms))
                                    * 50 / 1e6),
        control_period_ms=1000 * plan["dt"],
    )
    for k, v in out.items():
        print(f"    {k:28s} {v}")
    return dict(meta=out)


# --------------------------------------------------------------- offline --- #
def cmd_offline(args):
    """The OFFLINE solve under each keep-out implementation.

    The MPC is not the only consumer of the cost stack -- croco_run.py's staged
    continuation solves the same models to convergence, and the same restructure
    applies to it.  This is also the regression check that matters most: the
    fused term must produce THE SAME PLAN, and "the same plan" is not a timing
    claim, so the landing errors, the reach error and the distance from q* are
    reported next to the seconds.

    The plans differ in their last digit because the fused term reassociates the
    accumulation (one weighted sum instead of 86), which moves the DDP onto a
    marginally different iterate sequence.  What has to match is the answer.
    """
    import subprocess
    rows = []
    for impl in args.impls:
        env = dict(os.environ)
        env.update(CROCO_KEEPOUT=impl, CROCO_PASSIVE=args.passive)
        tag = f"{args.tag}_off_{impl}"
        subprocess.run(
            [sys.executable, os.path.join(HERE, "croco_run.py"),
             "--mode", args.mode, "--dt", "0.02",
             "--n-approach", "120", "--n-braced", "80",
             "--dir", args.dir, "--tag", tag],
            env=env, check=True, cwd=HERE, stdout=subprocess.DEVNULL)
        p = json.load(open(os.path.join(args.dir, f"plan_{tag}.json")))
        rows.append(dict(
            keepout=impl, seconds=p["solve_seconds"], iters=p["iters"],
            cost=p["cost"], converged=p["converged"],
            reach_mm=1000 * p["reach_err"],
            site_mm={k: 1000 * v for k, v in p["site_err"].items()},
            q_err_rad=p["q_err_vs_qstar_rad"],
            brace_dz_worst_mm=p.get("brace_dz_worst_mm")))
    print(f"    {'keep-out':8s} {'seconds':>8s} {'iters':>6s} {'cost':>9s} "
          f"{'reach mm':>9s} {'|q-q*| rad':>11s} {'worst dz mm':>12s}")
    for r in rows:
        print(f"    {r['keepout']:8s} {r['seconds']:8.1f} {r['iters']:6d} "
              f"{r['cost']:9.4g} {r['reach_mm']:9.2f} {r['q_err_rad']:11.3f} "
              f"{r['brace_dz_worst_mm']:12.2f}")
    return dict(offline=rows)


CMDS = dict(terms=cmd_terms, scaling=cmd_scaling, step=cmd_step,
            pieces=cmd_pieces, solver=cmd_solver, sweep=cmd_sweep,
            ladder=cmd_ladder, manifest=cmd_manifest, offline=cmd_offline,
            threads=cmd_threads, stage=cmd_stage)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=list(CMDS) + ["all"])
    ap.add_argument("--dir", default=RUN_DEFAULT)
    ap.add_argument("--tag", default=TAG_DEFAULT)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--counts", type=int, nargs="+",
                    default=[0, 20, 40, 60, 86])
    ap.add_argument("--horizons", type=int, nargs="+", default=[50, 35, 25, 15])
    ap.add_argument("--iters-grid", type=int, nargs="+", default=[2, 1])
    ap.add_argument("--threads", type=int, nargs="+",
                    default=[1, 2, 4, 6, 8, 12, 16, 20])
    ap.add_argument("--cones", type=int, nargs="+", default=[1, 0])
    ap.add_argument("--impls", nargs="+", default=["python", "cpp", "fused"],
                    help="keep-out implementations for the `offline` command")
    ap.add_argument("--passive", default="cpp")
    ap.add_argument("--mode", default="elbow+forearm",
                    help="contact subset for the `offline` command's re-solve. "
                         "Default is the top-ranked mode the CAD-faithful brace "
                         "geometry certifies at the S13 target; the S13 mode "
                         "elbow+forearm+palm certified against a wrist pad and "
                         "gripper box that claimed hardware the CAD does not "
                         "have (see the S14 brace_surfaces.py work).")
    ap.add_argument("--alphas", type=int, nargs="+", default=[0],
                    help="line-search ladder lengths to try (0 = crocoddyl's 10)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--q-noise", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = {}
    for name in (list(CMDS) if args.cmd == "all" else [args.cmd]):
        print(f"\n================ {name} ================")
        out.update(CMDS[name](args))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
