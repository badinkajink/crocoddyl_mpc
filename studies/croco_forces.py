#!/usr/bin/env python3
"""Read the contact forces crocoddyl's own dynamics predicted along a plan.

WHY THIS IS THE DIAGNOSTIC THAT MATTERS.  The feet enter the OCP as 6D contacts
pinned at their start placement.  A 6D pin does not just hold the foot down -- it
welds it, so in crocoddyl's model THE ROBOT CANNOT FALL OVER.  Base motion is a
function of joint motion, tipping is not representable, and the entire content of
"is this trajectory balanced" lives in one place: whether the predicted contact
wrench stays inside the foot's wrench cone.  The cone is in the problem as a
QUADRATIC PENALTY (weight 1e1), not as a constraint, so nothing stops the
solution from buying a cheap violation.

So this pulls the per-node contact wrenches straight out of the differential
action data and reports the two rows that decide whether MuJoCo agrees:

  normal force   f_z < 0 means the plan is PULLING the foot down onto the floor,
                 which a floor cannot do.
  centre of      CoP = (-tau_y/f_z, tau_x/f_z) in the sole frame.  Outside the
  pressure       0.20 x 0.08 m sole, the real foot rolls onto its edge and the
                 robot tips -- and the plan does not know, because its foot is
                 welded.

Reported per foot, plus the bracing contacts' friction utilisation.

usage: croco_forces.py --tag s13 --dir runs/...
"""

import argparse
import json
import os

import numpy as np

import croco_bridge as cb          # first: sets RTLD_GLOBAL
import contact_select as cs
import croco_plan as cp
import croco_replay as cr
import mujoco

crocoddyl = cb.import_crocoddyl()
HALF_X, HALF_Y = cb.FOOT_HALF


def forces(tag, run_dir):
    with open(os.path.join(run_dir, f"plan_{tag}.json")) as fh:
        plan = json.load(fh)
    xs = np.load(os.path.join(run_dir, f"xs_{tag}.npy"))
    us = np.load(os.path.join(run_dir, f"us_{tag}.npy"))
    ocp, _ = cr.build_ocp(plan, run_dir)
    problem = ocp.build(dt=plan["dt"], n_approach=plan["n_approach"],
                        n_braced=plan["n_braced"],
                        n_return=plan.get("n_return", 0),
                        dwell=plan.get("dwell", 0), cones=plan["cones"])
    problem.calc(list(xs), list(us))

    rows = []
    for k, dat in enumerate(problem.runningDatas):
        contacts = dat.differential.multibody.contacts.contacts.todict()
        rec = {"k": k, "t": k * plan["dt"]}
        for name, cd in contacts.items():
            f = np.array(cd.f.vector)              # 6D in LOCAL_WORLD_ALIGNED
            if name.startswith("brace_"):
                fn, ft = f[2], np.linalg.norm(f[:2])
                rec[f"fn_{name[6:]}"] = float(fn)
                rec[f"slip_{name[6:]}"] = float(ft / max(plan["mu"] * fn, 1e-9))
            else:
                fz = float(f[2])
                rec[f"fz_{name}"] = fz
                if abs(fz) > 1e-6:
                    rec[f"copx_{name}"] = float(-f[4] / fz)
                    rec[f"copy_{name}"] = float(f[3] / fz)
                else:
                    rec[f"copx_{name}"] = rec[f"copy_{name}"] = float("nan")
        rows.append(rec)
    return rows, plan


def summarise(rows, plan, verbose=True):
    feet = [f for f in ("sole_left", "sole_right")
            if f"fz_{f}" in rows[0]]
    out = {}
    for f in feet:
        fz = np.array([r[f"fz_{f}"] for r in rows])
        cx = np.array([r[f"copx_{f}"] for r in rows])
        cy = np.array([r[f"copy_{f}"] for r in rows])
        # how far the CoP is outside the sole rectangle, in mm
        ox = np.maximum(np.abs(cx) - HALF_X, 0.0)
        oy = np.maximum(np.abs(cy) - HALF_Y, 0.0)
        out[f] = dict(fz_min=float(np.nanmin(fz)), fz_max=float(np.nanmax(fz)),
                      n_pull=int(np.sum(fz < 0)),
                      cop_out_max_mm=float(1000 * np.nanmax(np.hypot(ox, oy))),
                      n_cop_out=int(np.sum(np.hypot(ox, oy) > 0)),
                      cop_x_range=[float(np.nanmin(cx)), float(np.nanmax(cx))],
                      cop_y_range=[float(np.nanmin(cy)), float(np.nanmax(cy))])
    nb = plan["n_approach"]
    nb_end = len(rows) - plan.get("n_return", 0)
    for s in plan["subset"]:
        if f"fn_{s}" in rows[nb]:
            fn = np.array([r.get(f"fn_{s}", np.nan) for r in rows])[nb:nb_end]
            sl = np.array([r.get(f"slip_{s}", np.nan) for r in rows])[nb:nb_end]
            # Slip utilisation is |f_t| / (mu f_n) and blows up when f_n is a
            # newton, so the p95 is reported alongside the max: a ratio of 1.4 on
            # a 1 N contact is arithmetic, a ratio of 1.1 on a 137 N contact is a
            # cone violation the plan actually bought.
            out[s] = dict(fn_braced_mean=float(np.nanmean(fn)),
                          fn_min=float(np.nanmin(fn)),
                          slip_max=float(np.nanmax(sl)),
                          slip_p95=float(np.nanpercentile(sl, 95)),
                          n_pull=int(np.nansum(fn < 0)))
    if verbose:
        print(f"planned contact forces, {len(rows)} nodes "
              f"(sole is {2*HALF_X:.2f} x {2*HALF_Y:.2f} m)")
        for f in feet:
            o = out[f]
            print(f"  {f:11s} fz {o['fz_min']:8.1f} .. {o['fz_max']:8.1f} N"
                  f"   pulls on {o['n_pull']} nodes")
            print(f"              CoP x [{o['cop_x_range'][0]:+.3f},"
                  f"{o['cop_x_range'][1]:+.3f}] y [{o['cop_y_range'][0]:+.3f},"
                  f"{o['cop_y_range'][1]:+.3f}] m"
                  f"   OUTSIDE the sole on {o['n_cop_out']} nodes, "
                  f"worst {o['cop_out_max_mm']:.0f} mm")
        for s in plan["subset"]:
            if s in out:
                o = out[s]
                print(f"  brace {s:9s} fn braced-mean {o['fn_braced_mean']:7.1f} N "
                      f"(min {o['fn_min']:7.1f}), slip util p95 {o['slip_p95']:.2f} "
                      f"max {o['slip_max']:.2f}, pulls on {o['n_pull']} nodes")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s13")
    ap.add_argument("--dir", default="runs/2026-08-06_session13")
    args = ap.parse_args()
    rows, plan = forces(args.tag, args.dir)
    res = summarise(rows, plan)
    out = os.path.join(args.dir, f"forces_{args.tag}.json")
    with open(out, "w") as fh:
        json.dump({"summary": res, "rows": rows}, fh, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
