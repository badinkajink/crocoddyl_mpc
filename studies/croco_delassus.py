#!/usr/bin/env python3
"""Does Sathya et al.'s matrix-free Delassus operator speed up the lean MPC?

The S13 controller re-solves a 50-node crocoddyl problem every control step and
takes 40 ms doing it, against a 20 ms period.  Every node runs
`DifferentialActionModelContactFwdDynamics`, which builds a Delassus matrix
G = J M^-1 J^T twice -- once in `calc` (pinocchio::forwardDynamics) and once,
reusing the factor, in `calcDiff` (getKKTContactDynamicMatrixInverse).  So the
paper's object is genuinely in this pipeline, unlike in MuJoCo's default solver
where it is absent altogether (see /home/correlllab/mujoco/mfdelassus).

Pinocchio 4.1.0 ships the matrix-free operator as
`DelassusOperatorRigidBodySystemsTpl` and does not expose it to Python, so it
could not previously be timed here -- the companion MuJoCo study had to quote
the paper's own numbers for it.  `croco_ext/mfdelassus.cpp` closes that gap: it
instantiates the C++-only operator against this robot's own pinocchio model.
This script drives it and writes the numbers the docpage reports.

Five measurements, in the order the argument needs them:

  profile   How much of a crocoddyl node is contact dynamics at all?  Ablates
            the cost stack out of the real S13 action models.
  stages    Inside that, how much is Delassus?  Per-stage timing of
            forwardDynamics and getKKTContactDynamicMatrixInverse with the real
            Jc, splitting Delassus work from M^-1 work from dense GEMMs.
  operator  What does the matrix-free operator cost here, and is it correct?
            Head to head with the explicit build on the same constraint set.
  dropin    The replacement itself: crocoddyl's materialise-then-GEMM against
            one matrix-free KKT solve per right-hand side, same answer both ways.
  sweep     At what contact count would the verdict flip?

usage:
  croco_delassus.py all      --out runs/2026-08-06_session14/delassus.json
  croco_delassus.py profile|stages|operator|dropin|sweep [--out ...]
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
import pinocchio as pin                        # noqa: E402

try:
    import croco_mfd                           # noqa: E402
except ImportError as exc:                     # pragma: no cover
    raise SystemExit(
        f"croco_mfd not importable ({exc}).\n"
        "Build it:  studies/croco_ext/build.sh mfd") from exc

RUN_DEFAULT = os.path.join(HERE, "runs", "2026-08-06_session13")
TAG_DEFAULT = "s13"

# crocoddyl's DifferentialActionModelContactFwdDynamics is constructed with this
# JMinvJt damping in croco_plan.INV_DAMPING; the matrix-free operator gets the
# same number so the two are solving the same regularized system.
DAMPING = 1e-4

# The two contact sets the S13 horizon actually visits.  `subset` is the S13
# brace mode (elbow + forearm + palm) -- 2 six-dimensional feet plus 3 point
# contacts is nc = 21; before the brace lands it is the feet alone, nc = 12.
FEET = ("sole_left", "sole_right")
BRACE = ("elbow", "forearm", "palm")


# --------------------------------------------------------------------------- #
def timeit(fn, reps=200, warm=3):
    """Best-of-reps, in microseconds.

    Best-of rather than mean: these are single-microsecond operations and the
    mean measures the machine's other tenants.  The MPC-level numbers this
    study reports elsewhere are means, because there the question is what a
    control period costs in practice, not what the code costs.
    """
    for _ in range(warm):
        fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e6


def constraint_set(rmodel, sites, n_feet=2, brace=BRACE, extra_points=0):
    """Pinocchio constraint models mirroring the OCP's crocoddyl contacts.

    crocoddyl's ContactModel6D at a sole frame is a FrameAnchorConstraintModel;
    its ContactModel3D at a brace site is a PointContactConstraintModel.  The
    reference frames differ (crocoddyl asks for LOCAL_WORLD_ALIGNED, pinocchio's
    constraints are LOCAL), which rotates G by a block-diagonal orthogonal
    matrix and changes none of its timing or conditioning.

    `extra_points` stacks additional point contacts on the brace frames, to push
    nc past what this robot physically has -- that is the sweep in §sweep, and
    it is a synthetic constraint count, not a physical claim.
    """
    cms = pin.StdVec_ConstraintModel()
    for s in FEET[:n_feet]:
        fr = rmodel.frames[sites[s]]
        cms.append(pin.ConstraintModel(pin.FrameAnchorConstraintModel(
            rmodel, fr.parentJoint, fr.placement)))
    for s in brace:
        fr = rmodel.frames[sites[s]]
        cms.append(pin.ConstraintModel(pin.PointContactConstraintModel(
            rmodel, fr.parentJoint, fr.placement)))
    for i in range(extra_points):
        s = (brace or FEET)[i % max(1, len(brace or FEET))]
        fr = rmodel.frames[sites[s]]
        # Offset the placement so the extra contacts are distinct constraints
        # rather than exact duplicates (a duplicated row makes G singular, which
        # the damping would hide and the timing would not represent).
        pl = pin.SE3(fr.placement)
        pl.translation = pl.translation + np.array(
            [0.004 * (i % 5 - 2), 0.004 * (i // 5 % 5 - 2), 0.0])
        cms.append(pin.ConstraintModel(pin.PointContactConstraintModel(
            rmodel, fr.parentJoint, pl)))
    return cms


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


# --------------------------------------------------------------- profile --- #
def cmd_profile(args):
    """Contact dynamics vs cost stack, per node, on the real action models."""
    crocoddyl = cb.import_crocoddyl()
    plan, ocp, problem, xs, us = load_ocp(args.dir, args.tag)
    rows = []
    for name, nd in node_indices(problem, xs).items():
        dm, x, u = nd["model"], nd["x"], us[nd["index"]].copy()
        d_full = dm.createData()
        dm.calc(d_full, x, u)
        dm.calcDiff(d_full, x, u)

        # The same dynamics with the costs removed.  Not "the costs are free" --
        # the difference is what the costs add on top of a shared kinematics
        # pass, which is the number that matters for deciding where to optimise.
        empty = crocoddyl.CostModelSum(ocp.state, ocp.nu)
        dm_dyn = crocoddyl.DifferentialActionModelContactFwdDynamics(
            ocp.state, dm.actuation, dm.contacts, empty, DAMPING, True)
        d_dyn = dm_dyn.createData()
        dm_dyn.calc(d_dyn, x, u)
        dm_dyn.calcDiff(d_dyn, x, u)

        r = dict(
            node=name, index=nd["index"], nc=nd["nc"],
            nv=int(ocp.rmodel.nv), nu=int(ocp.nu),
            n_costs=len(dm.costs.costs.todict()), nr=int(dm.costs.nr),
            calc_full=timeit(lambda: dm.calc(d_full, x, u), args.reps),
            calcdiff_full=timeit(lambda: dm.calcDiff(d_full, x, u), args.reps),
            calc_dyn=timeit(lambda: dm_dyn.calc(d_dyn, x, u), args.reps),
            calcdiff_dyn=timeit(lambda: dm_dyn.calcDiff(d_dyn, x, u), args.reps),
        )
        r["calc_costs"] = r["calc_full"] - r["calc_dyn"]
        r["calcdiff_costs"] = r["calcdiff_full"] - r["calcdiff_dyn"]
        r["dyn_share"] = ((r["calc_dyn"] + r["calcdiff_dyn"]) /
                          (r["calc_full"] + r["calcdiff_full"]))
        rows.append(r)

    print(f"{'node':10s} {'nc':>3s} {'calc':>22s} {'calcDiff':>24s}")
    print(f"{'':10s} {'':>3s} {'full   dyn  costs':>22s} "
          f"{'full    dyn   costs':>24s}")
    for r in rows:
        print(f"{r['node']:10s} {r['nc']:3d} "
              f"{r['calc_full']:7.1f}{r['calc_dyn']:6.1f}{r['calc_costs']:7.1f} "
              f"{r['calcdiff_full']:8.1f}{r['calcdiff_dyn']:7.1f}"
              f"{r['calcdiff_costs']:8.1f}   "
              f"dynamics = {100 * r['dyn_share']:.0f}% of the node")
    print("  (microseconds, best of %d)" % args.reps)
    return dict(nodes=rows)


# ---------------------------------------------------------------- stages --- #
def cmd_stages(args):
    """Per-stage split of the two pinocchio calls crocoddyl's node makes."""
    plan, ocp, problem, xs, us = load_ocp(args.dir, args.tag)
    rmodel = ocp.rmodel
    rows = []
    for name, nd in node_indices(problem, xs).items():
        dm, x, u = nd["model"], nd["x"], us[nd["index"]].copy()
        d = dm.createData()
        dm.calc(d, x, u)
        dm.calcDiff(d, x, u)
        nc = nd["nc"]
        q = np.ascontiguousarray(x[:rmodel.nq])
        # Exactly the arguments crocoddyl passes.
        Jc = np.ascontiguousarray(np.array(d.multibody.contacts.Jc[:nc, :]))
        a0 = np.ascontiguousarray(np.array(d.multibody.contacts.a0[:nc]))
        tau = np.ascontiguousarray(np.array(d.multibody.actuation.tau))

        r = croco_mfd.stage_bench(rmodel, d.pinocchio, q, tau, Jc, a0,
                                  DAMPING, args.reps)
        r = {k: (float(v) if isinstance(v, float) else int(v))
             for k, v in r.items()}
        r["node"] = name
        # Everything a matrix-free operator could displace, and nothing else:
        # the Delassus build + its factorization in calc, and the nc-column
        # triangular solve in calcDiff.  crocoddyl's calcDiff does NOT
        # refactorize -- it reuses data.llt_JMinvJt from calc.
        r["delassus_total"] = (r["fwd_delassus_build"] + r["fwd_delassus_llt"] +
                               r["fwd_delassus_solve1"] +
                               r["kkt_delassus_solve_nc"])
        r["stage_total"] = r["fwd_total"] + r["kkt_total"]
        r["delassus_share_of_stages"] = r["delassus_total"] / r["stage_total"]
        rows.append(r)

    for r in rows:
        print(f"--- {r['node']}  nv={r['nv']} nc={r['nc']}")
        print(f"  calc: forwardDynamics            {r['fwd_total']:7.2f} us")
        for k, lbl, dela in (
                ("fwd_cholesky_decompose", "LTDL factor of M", False),
                ("fwd_delassus_build", "build J M^-1 J^T", True),
                ("fwd_delassus_llt", "LLT of it", True),
                ("fwd_delassus_solve1", "one G^-1 solve", True),
                ("fwd_minv_solve1", "one M^-1 solve", False)):
            print(f"      {'*' if dela else ' '} {lbl:24s} {r[k]:7.2f} us")
        print(f"  calcDiff: getKKTContactDynMatInv {r['kkt_total']:7.2f} us")
        for k, lbl, dela in (
                ("kkt_delassus_solve_nc", f"G^-1 on {r['nc']} columns", True),
                ("kkt_minv_dense", "M^-1 as a dense block", False),
                ("kkt_gemms", "three Schur GEMMs", False)):
            print(f"      {'*' if dela else ' '} {lbl:24s} {r[k]:7.2f} us")
        print(f"  * Delassus-attributable          {r['delassus_total']:7.2f} us"
              f"   = {100 * r['delassus_share_of_stages']:.1f}% of these two calls")
    return dict(stages=rows)


# -------------------------------------------------------------- operator --- #
def cmd_operator(args):
    """Matrix-free operator vs explicit build, same constraints, same state."""
    plan, ocp, problem, xs, us = load_ocp(args.dir, args.tag)
    rmodel = ocp.rmodel
    sites = ocp.sites
    rows = []
    for name, nd in node_indices(problem, xs).items():
        q = np.ascontiguousarray(nd["x"][:rmodel.nq])
        brace = BRACE if name == "braced" else ()
        cms = constraint_set(rmodel, sites, brace=brace)
        d_mf, d_ex = rmodel.createData(), rmodel.createData()
        mf = croco_mfd.mf_bench(rmodel, d_mf, q, cms, DAMPING, args.reps)
        ex = croco_mfd.explicit_bench(rmodel, d_ex, q, cms, DAMPING, args.reps)
        Gm, Ge = np.array(mf["G"]), np.array(ex["G"])
        rel = float(np.linalg.norm(Gm - Ge) / np.linalg.norm(Ge))

        # Ready-to-use cost of each route: the explicit one has to stack J,
        # build G and factorize it; the matrix-free one runs compute(). Both
        # then pay per application.
        ex_setup = ex["stack_jacobian"] + ex["build"] + ex["llt"]
        mf_setup = mf["compute_both"]
        # Break-even in G^-1 applications: setup_ex + k*solve_ex = setup_mf +
        # k*solve_mf.  Below k the matrix-free operator is ahead, above it the
        # explicit factorization is.
        dsolve = mf["solve1"] - ex["solve1"]
        k_star = (ex_setup - mf_setup) / dsolve if dsolve > 0 else float("inf")

        r = dict(node=name, nc=int(mf["nc"]), nv=int(rmodel.nv), rel_err=rel,
                 mf={k: float(v) for k, v in mf.items()
                     if isinstance(v, (int, float)) and k != "nc"},
                 explicit={k: float(v) for k, v in ex.items()
                           if isinstance(v, (int, float)) and k != "nc"},
                 mf_setup=float(mf_setup), explicit_setup=float(ex_setup),
                 breakeven_solves=float(k_star))
        # What crocoddyl actually asks for per node: one solve in calc for the
        # contact forces, then nc more in calcDiff for the -G^-1 block.
        n_solves = 1 + r["nc"]
        r["croco_solves_per_node"] = n_solves
        r["explicit_at_croco_pattern"] = ex_setup + n_solves * ex["solve1"]
        r["mf_at_croco_pattern"] = mf_setup + n_solves * mf["solve1"]
        rows.append(r)

    for r in rows:
        print(f"--- {r['node']}  nv={r['nv']} nc={r['nc']}")
        print(f"  |G_matrixfree - G_explicit| / |G_explicit| = {r['rel_err']:.2e}")
        print(f"  {'':22s} {'matrix-free':>12s} {'explicit':>12s}")
        for lbl, a, b in (
                ("setup (per state)", r["mf_setup"], r["explicit_setup"]),
                ("one G x", r["mf"]["apply1"], r["explicit"]["apply1"]),
                ("one G^-1 x", r["mf"]["solve1"], r["explicit"]["solve1"]),
                (f"setup + {r['croco_solves_per_node']} solves",
                 r["mf_at_croco_pattern"], r["explicit_at_croco_pattern"])):
            print(f"  {lbl:22s} {a:11.2f}u {b:11.2f}u")
        print(f"  break-even at {r['breakeven_solves']:.1f} solves; "
              f"crocoddyl asks for {r['croco_solves_per_node']}")
    return dict(operator=rows)


# ---------------------------------------------------------------- dropin --- #
def cmd_dropin(args):
    """The replacement, at the only place in calcDiff it could be dropped in."""
    plan, ocp, problem, xs, us = load_ocp(args.dir, args.tag)
    rmodel = ocp.rmodel
    nv, nu = rmodel.nv, ocp.nu
    rows = []
    for name, nd in node_indices(problem, xs).items():
        q = np.ascontiguousarray(nd["x"][:rmodel.nq])
        brace = BRACE if name == "braced" else ()
        cms = constraint_set(rmodel, ocp.sites, brace=brace)
        for n_rhs, what in ((nu, "Fu only"), (2 * nv, "Fx only"),
                            (2 * nv + nu, "Fx and Fu")):
            d = rmodel.createData()
            r = croco_mfd.kkt_route_bench(rmodel, d, q, cms, n_rhs,
                                          DAMPING, max(20, args.reps // 4))
            r = {k: (float(v) if isinstance(v, float) else int(v))
                 for k, v in r.items()}
            r.update(node=name, what=what,
                     ratio=r["route_b"] / r["route_a"])
            rows.append(r)

    print(f"{'node':10s} {'what':11s} {'n_rhs':>5s} {'croco':>9s} "
          f"{'matrix-free':>12s} {'ratio':>7s} {'agree':>10s}")
    for r in rows:
        print(f"{r['node']:10s} {r['what']:11s} {r['n_rhs']:5d} "
              f"{r['route_a']:8.1f}u {r['route_b']:11.1f}u "
              f"{r['ratio']:6.2f}x {r['rel_diff']:10.1e}")
    return dict(dropin=rows)


# ----------------------------------------------------------------- sweep --- #
def cmd_sweep(args):
    """Where the verdict flips: contact count vs the two routes.

    Past the robot's own 21 rows this is synthetic -- extra point contacts
    stacked on the brace frames.  The point is the scaling of the two
    algorithms, which is a property of nv and nc and not of where the contacts
    physically are.
    """
    plan, ocp, problem, xs, us = load_ocp(args.dir, args.tag)
    rmodel = ocp.rmodel
    q = np.ascontiguousarray(xs[problem.T - 1][:rmodel.nq])
    rows = []
    for extra in (0, 3, 9, 19, 39, 79, 159, 319):
        cms = constraint_set(rmodel, ocp.sites, extra_points=extra)
        d_mf, d_ex = rmodel.createData(), rmodel.createData()
        reps = max(20, args.reps // (1 + extra // 40))
        mf = croco_mfd.mf_bench(rmodel, d_mf, q, cms, DAMPING, reps)
        ex = croco_mfd.explicit_bench(rmodel, d_ex, q, cms, DAMPING, reps)
        nc = int(mf["nc"])
        ex_setup = float(ex["stack_jacobian"] + ex["build"] + ex["llt"])
        mf_setup = float(mf["compute_both"])
        n_solves = 1 + nc                        # crocoddyl's access pattern
        rows.append(dict(
            nc=nc, extra=extra,
            mf_setup=mf_setup, explicit_setup=ex_setup,
            mf_solve=float(mf["solve1"]), explicit_solve=float(ex["solve1"]),
            mf_apply=float(mf["apply1"]), explicit_apply=float(ex["apply1"]),
            mf_total=mf_setup + n_solves * float(mf["solve1"]),
            explicit_total=ex_setup + n_solves * float(ex["solve1"]),
            rel_err=float(np.linalg.norm(np.array(mf["G"]) - np.array(ex["G"])) /
                          np.linalg.norm(np.array(ex["G"]))),
            mf_bytes=int(mf["bytes"]), explicit_bytes=int(ex["bytes"])))
        r = rows[-1]
        r["ratio"] = r["mf_total"] / r["explicit_total"]

    print(f"{'nc':>4s} {'setup mf':>9s} {'setup ex':>9s} {'solve mf':>9s} "
          f"{'solve ex':>9s} {'total mf':>10s} {'total ex':>10s} {'mf/ex':>7s} "
          f"{'bytes mf':>9s} {'bytes ex':>9s} {'err':>9s}")
    for r in rows:
        print(f"{r['nc']:4d} {r['mf_setup']:8.2f}u {r['explicit_setup']:8.2f}u "
              f"{r['mf_solve']:8.2f}u {r['explicit_solve']:8.2f}u "
              f"{r['mf_total']:9.1f}u {r['explicit_total']:9.1f}u "
              f"{r['ratio']:6.2f}x {r['mf_bytes']:9d} {r['explicit_bytes']:9d} "
              f"{r['rel_err']:9.1e}")
    print("  total = setup + (1 + nc) solves, i.e. crocoddyl's per-node pattern")
    return dict(sweep=rows)


# ---------------------------------------------------------------- budget --- #
def cmd_budget(args):
    """Roll the per-node numbers up onto the MPC step the controller runs.

    The step is not a sum of nodes -- it also contains the Riccati backward pass
    and however many line-search rollouts the solver takes, neither of which
    touches a Delassus matrix.  So the per-node share is an UPPER BOUND on the
    step share, and it is computed as one generously: `calc` is counted twice
    per node (the rollout plus one line-search trial) and `calcDiff` once.
    """
    import croco_replay as cr
    plan, ocp, problem, xs, us = load_ocp(args.dir, args.tag)
    models = list(problem.runningModels)
    mpc = cr.MPC(ocp, models, problem.terminalModel, horizon=args.horizon,
                 iters=1, xs_plan=xs, us_plan=us)
    for k in range(6):
        mpc(k, xs[k])
    n = 25
    t0 = time.time()
    for k in range(6, 6 + n):
        mpc(k, xs[k])
    step_ms = 1000 * (time.time() - t0) / n

    stages = cmd_stages(args)["stages"]
    by_node = {r["node"]: r for r in stages}
    prof = {r["node"]: r for r in cmd_profile(args)["nodes"]}

    out = dict(mpc_step_ms=step_ms, horizon=mpc.H, iters=1,
               control_period_ms=1000 * plan["dt"])
    for node in ("approach", "braced"):
        s, p = by_node[node], prof[node]
        node_us = 2 * p["calc_full"] + p["calcdiff_full"]
        dela_us = 2 * (s["fwd_delassus_build"] + s["fwd_delassus_llt"] +
                       s["fwd_delassus_solve1"]) + s["kkt_delassus_solve_nc"]
        share = dela_us / node_us
        out[node] = dict(
            node_us=node_us, delassus_us=dela_us, share_of_node=share,
            # If the Delassus cost went to ZERO -- faster than any operator can
            # be -- this is where the step lands.
            floor_ms=step_ms * (1 - share),
            saving_ms=step_ms * share)
        print(f"{node:10s} node {node_us:7.1f}us  Delassus {dela_us:6.2f}us "
              f"({100 * share:4.1f}%)  step {step_ms:6.1f}ms -> floor "
              f"{out[node]['floor_ms']:6.1f}ms")
    print(f"control period {out['control_period_ms']:.0f} ms; "
          f"measured MPC step {step_ms:.1f} ms over H={mpc.H}")
    return dict(budget=out)


# ------------------------------------------------------------------- all --- #
def cmd_all(args):
    out = {}
    for name, fn in (("profile", cmd_profile), ("stages", cmd_stages),
                     ("operator", cmd_operator), ("dropin", cmd_dropin),
                     ("sweep", cmd_sweep), ("budget", cmd_budget)):
        print(f"\n================ {name} ================")
        out.update(fn(args))
    out["meta"] = dict(
        crocoddyl=__import__("crocoddyl").__version__,
        pinocchio=pin.__version__, damping=DAMPING, reps=args.reps,
        note="microseconds, best-of-reps; see croco_delassus.py docstring")
    return out


COMMANDS = dict(profile=cmd_profile, stages=cmd_stages, operator=cmd_operator,
                dropin=cmd_dropin, sweep=cmd_sweep, budget=cmd_budget,
                all=cmd_all)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=sorted(COMMANDS))
    ap.add_argument("--dir", default=RUN_DEFAULT)
    ap.add_argument("--tag", default=TAG_DEFAULT)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=50,
                    help="MPC window for `budget` (S13's is 50)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = COMMANDS[args.cmd](args)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
