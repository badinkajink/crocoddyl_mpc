#!/usr/bin/env python3
"""Drive croco_plan for one contact mode and save the trajectory + provenance."""

import argparse
import json
import os
import subprocess

import numpy as np

import croco_bridge as cb          # must precede pinocchio/crocoddyl: sets RTLD_GLOBAL
import contact_select as cs
import croco_plan as cp
import mujoco


def git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="elbow+forearm",
                    help="'+'-joined contact subset, must exist in modes.json")
    ap.add_argument("--start", default="stand", help="MJCF keyframe to start from")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--n-approach", type=int, default=120)
    ap.add_argument("--n-braced", type=int, default=80)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--mu", type=float, default=0.6)
    ap.add_argument("--dwell", type=int, default=0,
                    help="braced nodes at the end that hold the reach at rest")
    ap.add_argument("--n-return", type=int, default=0,
                    help="nodes of a brace-release/return-to-stand phase")
    ap.add_argument("--no-cones", action="store_true",
                    help="drop in-solver friction/wrench cone costs")
    ap.add_argument("--impulse", action="store_true",
                    help="insert an impulse node at brace touchdown (see croco_plan)")
    ap.add_argument("--legacy", action="store_true",
                    help="S12 formulation: half-space barrier, no hold/CoM terms")
    ap.add_argument("--no-keepout", action="store_true",
                    help="drop the table box keep-out (leaves the arm free to "
                         "sweep through the slab -- for ablations only)")
    ap.add_argument("--w-land", type=float, default=5e3)
    ap.add_argument("--w-hold", type=float, default=1e3)
    ap.add_argument("--w-com", type=float, default=1e3)
    ap.add_argument("--w-cone", type=float, default=1e1,
                    help="friction/wrench cone barrier weight.  1e1 is the "
                         "S13-S16 value; see the S17 docpage for what it costs")
    ap.add_argument("--w-reach", type=float, default=1e2,
                    help="reach cost on the braced phase (terminal/dwell get 10x)")
    ap.add_argument("--cop-shrink", type=float, default=1.0,
                    help="scale on the sole rectangle the foot WRENCH CONE is "
                         "written against.  <1 demands centre-of-pressure "
                         "margin, i.e. asks the plan not to ride the tipping "
                         "edge")
    ap.add_argument("--min-nforce", type=float, default=1.0,
                    help="minimum contact normal force [N] in the cones")
    ap.add_argument("--w-ctrl", type=float, default=1e-3,
                    help="control regulariser.  With bilateral 6D foot welds a "
                         "subspace of internal preload is free to the model and "
                         "is not free to the plant -- this is what prices it")
    ap.add_argument("--n-hold", type=int, default=0,
                    help="leading nodes of the approach spent holding the start "
                         "pose, so the departure from rest is planned")
    ap.add_argument("--w-hold-state", type=float, default=1e2)
    ap.add_argument("--w-jlim", type=float, default=1e3,
                    help="joint-limit barrier weight.  DEFAULT CHANGED in S17 "
                         "from 1e1, which was too weak to hold: seven of the 25 "
                         "grid plans pushed a hip yaw up to 2.6 deg past its "
                         "+-24.6 deg range, and commanding the gripper "
                         "perpendicular to the table pushed one 14 deg past. "
                         "Pass 1e1 to reproduce S13-S16 exactly")
    ap.add_argument("--w-vel", type=float, default=0.0,
                    help="velocity regulariser, all joints, every phase.  The "
                         "move-to-brace peaks at 2.5-3.2 rad/s unpriced")
    ap.add_argument("--reach-rot", default=None,
                    choices=["flat", "down", "side"],
                    help="hold the REACHING gripper at this world orientation "
                         "through the braced phase")
    ap.add_argument("--w-reach-rot", type=float, default=0.0)
    ap.add_argument("--drop", default="",
                    help="comma-separated cost groups to leave out: stateReg, "
                         "ctrlReg, jointLim, cones, keepout, comSupport, land, "
                         "hold.  For the ablation table")
    ap.add_argument("--w-com-damp", type=float, default=0.0,
                    help="weight on horizontal centroidal linear momentum on "
                         "the feet-only phases: the derivative half of the CoM "
                         "terms, which is where a topple shows up first")
    ap.add_argument("--w-com-track", type=float, default=0.0,
                    help="quadratic pull of the CoM toward the middle of the "
                         "feet on the feet-only phases (0 = off, barrier only)")
    ap.add_argument("--com-margin", type=float, default=0.03)
    ap.add_argument("--tag", default=None,
                    help="override the output file tag (default: the mode name)")
    ap.add_argument("--start-settled", action="store_true",
                    help="take the start pose from a static hold of the "
                         "keyframe rather than from the keyframe, so the plan "
                         "begins where the controller's settle ends")
    ap.add_argument("--stance-dx", type=float, default=None,
                    help="rigid x offset of the whole robot [m] (default: the "
                         "value modes.json was certified at)")
    ap.add_argument("--stance-dy", type=float, default=None,
                    help="rigid y offset of the whole robot [m].  Negative "
                         "moves the bracing shoulder toward the table's "
                         "centreline -- see croco_stance.py and S14")
    ap.add_argument("--dir", default="runs/2026-08-05_session12/croco")
    ap.add_argument("--modes", default=None,
                    help="modes.json to read q* from (default: <dir>/modes.json)")
    args = ap.parse_args()

    modes_path = args.modes or os.path.join(args.dir, "modes.json")
    with open(modes_path) as fh:
        modes = json.load(fh)
    entry = next(m for m in modes["modes"] if m["name"] == args.mode)
    if not entry.get("qpos_file"):
        raise SystemExit(f"mode {args.mode} is not admissible; no pose to aim at")
    q_star = np.loadtxt(os.path.join(os.path.dirname(modes_path),
                                     entry["qpos_file"]))

    if args.stance_dx is not None:
        cs.STANCE_DX = args.stance_dx
    if args.stance_dy is not None:
        cs.STANCE_DY = args.stance_dy
    elif "stance_dy" in modes:
        cs.STANCE_DY = modes["stance_dy"]       # follow the pose we aim at
    if "stance_dx" in modes and args.stance_dx is None:
        cs.STANCE_DX = modes["stance_dx"]

    m_mj, d_mj = cs.load()
    q0 = cs.start_qpos(m_mj, args.start)
    if args.start_settled:
        # Plan from where the plant WILL BE when the controller hands over,
        # not from where the keyframe says it is.  See croco_replay.
        import croco_replay as cr
        m_p, d_p = cs.load(ik_margin=0.0)
        q0 = cr.settled_start(m_p, d_p, q0)
        print(f"start     settled: {np.linalg.norm(q0[7:34] - cs.start_qpos(m_mj, args.start)[7:34]):.4f} rad "
              f"from the {args.start} keyframe")
    d_mj.qpos[:] = q0
    mujoco.mj_forward(m_mj, d_mj)
    table_z = cs.table_top_z(m_mj, d_mj)

    subset = entry["subset"]
    print(f"mode      {args.mode}   (static effort {entry['effort']:.3f}, "
          f"max|tau|/lim {entry['max_ratio']:.2f})")
    print(f"start     {args.start}")
    print(f"target    {modes['target']}")
    print(f"table top z = {table_z:.4f}")

    ocp = cp.LeanOCP(subset, q_star, q0, mu=args.mu, table_z=table_z,
                     reach_target=np.array(modes["target"]),
                     cones=not args.no_cones, legacy=args.legacy,
                     keepout=not args.no_keepout, com_margin=args.com_margin,
                     w_land=args.w_land, w_hold=args.w_hold, w_com=args.w_com,
                     w_cone=args.w_cone, w_reach=args.w_reach,
                     cop_shrink=args.cop_shrink, min_nforce=args.min_nforce,
                     w_com_track=args.w_com_track, w_com_damp=args.w_com_damp,
                     w_ctrl=args.w_ctrl,
                     n_hold=args.n_hold, w_hold_state=args.w_hold_state,
                     w_jlim=args.w_jlim, w_vel=args.w_vel,
                     reach_rot=args.reach_rot, w_reach_rot=args.w_reach_rot,
                     drop=[v for v in args.drop.split(",") if v])
    n_tot = args.n_approach + args.n_braced + args.n_return
    total_t = n_tot * args.dt
    print(f"horizon   {n_tot} nodes, dt={args.dt}, "
          f"{total_t:.2f} s ({args.n_approach} approach + {args.n_braced} braced"
          f"{' + %d return' % args.n_return if args.n_return else ''})"
          f"{' + impulse' if args.impulse else ''}")
    print(f"keep-out  {len(ocp.keepout)} points on "
          f"{len({p['geom'] for p in ocp.keepout})} geoms"
          if ocp.keepout else "keep-out  DISABLED")
    print(f"support   CoM box x[{ocp.support_lo[0]:.3f},{ocp.support_hi[0]:.3f}] "
          f"y[{ocp.support_lo[1]:.3f},{ocp.support_hi[1]:.3f}]\n")

    solver, ok, secs, stage1 = cp.solve_staged(
        ocp, args.dt, args.n_approach, args.n_braced, iters=args.iters,
        impulse=args.impulse, n_return=args.n_return, dwell=args.dwell)
    rep = cp.report(solver, ocp)
    rep.update(mode=args.mode, subset=subset, start=args.start, dt=args.dt,
               impulse=args.impulse, cones=not args.no_cones,
               n_approach=args.n_approach, n_braced=args.n_braced,
               n_return=args.n_return, dwell=args.dwell, mu=args.mu,
               legacy=args.legacy, keepout=len(ocp.keepout),
               w_land=args.w_land, w_hold=args.w_hold, w_com=args.w_com,
               w_cone=args.w_cone, w_reach=args.w_reach,
               cop_shrink=args.cop_shrink, min_nforce=args.min_nforce,
               w_com_track=args.w_com_track, w_com_damp=args.w_com_damp,
               w_ctrl=args.w_ctrl,
               n_hold=args.n_hold, w_hold_state=args.w_hold_state,
               drop=args.drop, w_jlim=args.w_jlim, w_vel=args.w_vel,
               reach_rot=args.reach_rot, w_reach_rot=args.w_reach_rot,
               com_margin=args.com_margin,
               converged=bool(ok), solve_seconds=secs, commit=git_head(),
               stage1=({"converged": stage1[0], "cost": stage1[1]}
                       if stage1 else None),
               stance_dx=cs.STANCE_DX, stance_dy=cs.STANCE_DY,
               start_settled=bool(args.start_settled),
               target=modes["target"], model=os.path.basename(cs.MODEL),
               tau_basis=cs.TAU_BASIS)

    print(f"\nsolved={ok} in {secs:.1f}s, {rep['iters']} iters, "
          f"cost={rep['cost']:.4g}, stop={rep['stop']:.3g}")
    for s, e in rep["site_err"].items():
        print(f"  brace {s:9s} lands {e*1000:7.2f} mm from its certified spot")
    if "reach_err" in rep:
        print(f"  reach hand        {rep['reach_err']*1000:7.2f} mm from target")
    for f, e in rep["foot_err"].items():
        print(f"  foot {f:26s} drift {e*1000:6.3f} mm")
    print(f"  terminal |v| = {rep['terminal_vel_norm']:.4f}")
    print(f"  |q - q*| over 27 joints = {rep['q_err_vs_qstar_rad']:.3f} rad "
          f"(worst joint {rep['q_err_max_joint_rad']:.3f} rad)")
    print(f"  max |tau|/limit = {rep['max_torque_ratio']:.3f} "
          f"({rep['torque_ratio_argmax_joint']})")
    for s, v in rep.get("brace_dz_vs_qstar", {}).items():
        print(f"  brace {s:9s} height vs certified over the braced phase: "
              f"{v['min']*1000:+6.1f} .. {v['max']*1000:+6.1f} mm")

    tag = args.tag or args.mode.replace("+", "_")
    np.save(os.path.join(args.dir, f"xs_{tag}.npy"), np.array(solver.xs))
    np.save(os.path.join(args.dir, f"us_{tag}.npy"),
            np.array([u for u in solver.us if len(u) == ocp.nu]))
    # The Riccati feedback gains.  These are the by-product that turns the plan
    # into a controller -- u = u* - K (x (-) x*) is a time-varying LQR about the
    # optimal trajectory, and it costs nothing beyond saving it, because DDP had
    # to compute it to take its own step.  croco_replay --ctrl riccati uses them.
    K = np.array([k for k in solver.K if k.shape[0] == ocp.nu])
    np.save(os.path.join(args.dir, f"K_{tag}.npy"), K)
    with open(os.path.join(args.dir, f"plan_{tag}.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    print(f"\nwrote {args.dir}/xs_{tag}.npy, us_{tag}.npy, plan_{tag}.json")


if __name__ == "__main__":
    main()
