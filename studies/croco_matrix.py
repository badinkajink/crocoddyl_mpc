#!/usr/bin/env python3
"""Run the S13 experiment matrix and collect one summary file.

Four questions, in order, each a row set in `matrix.json`:

  ladder       does closing the loop help?  Every controller in croco_replay's
               ladder on the same plan, including the `hold` control experiment
               that says how long the robot survives doing nothing.
  mpc_sweep    how much horizon does the MPC need?  A short horizon on a long
               maneuver is myopic and falls; this finds where that stops.
  robust       how much disturbance does each surviving controller absorb?  A
               lateral shove at the base mid-approach, and joint noise on the
               initial state, swept until it falls.
  ablate       what did each S13 fix buy?  The S12 plan (legacy formulation) and
               the S13 plan under the same controller.

usage: croco_matrix.py [--only ladder,mpc_sweep,...] [--dir runs/...]
"""

import argparse
import json
import os
import time

import numpy as np

import croco_bridge as cb          # first: sets RTLD_GLOBAL
import croco_replay as cr


def run(tag, ctrl, run_dir, **kw):
    t0 = time.time()
    log, plan = cr.replay(tag, ctrl_mode=ctrl, run_dir=run_dir,
                          dt_plan=cr.plan_dt(run_dir, tag), **kw)
    res = cr.summarise(log, plan, ctrl, verbose=False)
    res["wall_s"] = time.time() - t0
    res["tag"] = tag
    res.update({k: v for k, v in kw.items()
                if k in ("push", "q_noise", "seed", "mpc_horizon", "mpc_iters")})
    return res, log


def checkpoint(path, key, rows):
    """Write after EVERY row, not at the end of a section.

    This machine has been dropping long-running processes mid-sweep, and a
    matrix that only persists on clean exit loses an hour of solves to that.
    Cheap insurance: the file is a few hundred kB.
    """
    prev = json.load(open(path)) if os.path.exists(path) else {}
    prev[key] = rows
    with open(path, "w") as fh:
        json.dump(prev, fh, indent=1)


def line(res):
    return (f"{res['tag']:>10s} {res['ctrl']:>11s} "
            f"{'FELL' if res['fell'] else ' up ':>5s} "
            f"z={res['final_pelvis_z']:.3f} "
            f"reach={res['reach_err_at_brace_end']*1000:8.1f}mm "
            f"brace={res['brace_total_braced_mean']:6.1f}N "
            f"margin={res['min_support_margin']*1000:+7.1f}mm "
            f"track={res['max_tracking_err']:.3f}rad "
            f"tau={res['max_tau_ratio']:.2f} "
            f"({res['wall_s']:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/2026-08-06_session13")
    ap.add_argument("--tag", default="s13")
    ap.add_argument("--legacy-tag", default="legacy")
    ap.add_argument("--only", default="ladder,mpc_sweep,robust,ablate")
    ap.add_argument("--mpc-horizon", type=int, default=50)
    ap.add_argument("--out", default="matrix.json")
    args = ap.parse_args()
    want = set(args.only.split(","))
    path = os.path.join(args.dir, args.out)
    best = dict(mpc_horizon=args.mpc_horizon, mpc_iters=1)

    if "ladder" in want:
        rows = []
        for ctrl in ["kinematic", "hold", "position", "ff", "torque_meas",
                     "riccati", "mpc"]:
            kw = dict(best) if ctrl == "mpc" else {}
            res, _ = run(args.tag, ctrl, args.dir, **kw)
            rows.append(res)
            print(line(res), flush=True)
            checkpoint(path, "ladder", rows)

    if "mpc_sweep" in want:
        rows = []
        for H in (25, 50, 100, 200):
            for it in (1, 2):
                res, _ = run(args.tag, "mpc", args.dir,
                             mpc_horizon=H, mpc_iters=it)
                rows.append(res)
                print(line(res), flush=True)
                checkpoint(path, "mpc_sweep", rows)

    if "robust" in want:
        rows = []
        for ctrl in ("ff", "riccati", "mpc"):
            for push in (0.0, 40.0, 80.0, 120.0, 160.0):
                kw = dict(best) if ctrl == "mpc" else {}
                res, _ = run(args.tag, ctrl, args.dir, push=push,
                             push_at=0.35, push_dir=(-1, 0, 0), **kw)
                rows.append(res)
                print(line(res), flush=True)
                checkpoint(path, "robust", rows)
            for noise in (0.02, 0.05):
                kw = dict(best) if ctrl == "mpc" else {}
                res, _ = run(args.tag, ctrl, args.dir, q_noise=noise, seed=7,
                             **kw)
                rows.append(res)
                print(line(res), flush=True)
                checkpoint(path, "robust", rows)

    if "ablate" in want:
        rows = []
        for tag in (args.legacy_tag, args.tag):
            for ctrl in ("ff", "riccati", "mpc"):
                kw = dict(best) if ctrl == "mpc" else {}
                try:
                    res, _ = run(tag, ctrl, args.dir, **kw)
                except Exception as e:                 # noqa: BLE001
                    print(f"  {tag}/{ctrl} failed: {e}", flush=True)
                    continue
                rows.append(res)
                print(line(res), flush=True)
                checkpoint(path, "ablate", rows)

    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
