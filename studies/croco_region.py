#!/usr/bin/env python3
"""Static-equilibrium CoM regions for a crocoddyl plan, before and after MuJoCo.

WHY.  The S11 docpage's region plot is the study's clearest single picture: a
top view of the support region the contact set buys, with the pose's CoM inside
it.  Everything it has ever been drawn for is a KINEMATIC pose -- the IK's q*,
scored by a static QP -- and this session's whole claim is that crocoddyl plans
are a better way to produce those trajectories.  That claim deserves the same
picture: does the pose the closed-loop MuJoCo run actually ENDS AT have the
region the offline plan promised, and is its CoM anywhere near where the offline
plan put it?

THREE REGIONS PER CELL, and the differences between them are the content:

  certified   region at q*, with the PRESCRIBED contact set.  The offline
              kinematic plan, i.e. what S11 would have drawn.
  achieved    region at the pose MuJoCo reaches at the end of the braced phase,
              with the contact set MuJoCo's NARROWPHASE actually reports there.
              This is the honest one: it can be smaller than certified because
              the pose moved, or because a planned contact never happened.
  prescribed  region at the ACHIEVED pose with the PRESCRIBED contact set.  The
              control that separates those two causes -- if `prescribed` matches
              `certified` but `achieved` does not, the pose is fine and a contact
              is missing; if all three differ, the pose moved.

`legs` is drawn under all of them for scale, exactly as in the S11 plot, because
the whole point of a brace is the difference between it and the legs-only region.

usage: croco_region.py --dir runs/.../sweep [--ctrl mpc] [--out regions.json]
"""

import argparse
import json
import os

import numpy as np

import croco_bridge as cb          # first: sets RTLD_GLOBAL
import contact_select as cs
import stability as st
import mujoco


def achieved_subset(m, d, sites=cs.ARM_SITES):
    """Which of the arm sites' BODIES MuJoCo reports against the table, here.

    Contact identity is decided by the narrowphase and not by the plan: a site
    whose body is not touching is not a contact, however confidently the OCP
    drew force through it.  This is the function that turns "the plan says
    elbow+forearm+palm" into "the robot has elbow+palm".
    """
    tbl = cs.bid(m, "table")
    touching = set()
    for c in range(d.ncon):
        con = d.contact[c]
        b1, b2 = m.geom_bodyid[con.geom[0]], m.geom_bodyid[con.geom[1]]
        if tbl not in (b1, b2):
            continue
        touching.add(b2 if b1 == tbl else b1)
    return tuple(s for s in sites if cs.bid(m, cs.SITES[s][0]) in touching)


def region_at(m, d, subset):
    """Region + CoM + labelled contacts at the CURRENT model state.

    Mirrors make_regions.one so the two can be plotted with the same code, but
    takes the pose as given instead of solving IK for it -- the poses here come
    from a replay, not from a solver.
    """
    contacts, P, A, g, mass, com, Rs = st.blocks(m, d, subset)
    qp = cs.equilibrium_qp(m, d, subset)
    labels = ["foot"] * (len(P) - len(subset)) + list(subset)
    pts = [dict(label=lab, p=[float(v) for v in p],
                fn=float(qp["lam"][3 * i]),
                ft=float(np.linalg.norm(qp["lam"][3 * i + 1:3 * i + 3])))
           for i, (lab, p) in enumerate(zip(labels, P))]
    pa, c0, ma = st.equilibrium_region(m, d, subset, actuated=True)
    # A degenerate region (every ray infeasible at the CoM itself, which is what
    # a toppled pose gives) comes back as NaN.  Passed through it poisons every
    # downstream min/max; recorded as null it reads as what it is -- "this pose has
    # no equilibrium region", which is a result, not a missing measurement.
    ma = None if not np.isfinite(ma) else float(ma)
    return dict(actuated=pa.tolist(), com=[float(c0[0]), float(c0[1])],
                margin=ma, pts=pts, subset=list(subset),
                feasible=bool(qp["feasible"]), peak=float(qp["max_ratio"]),
                com3=[float(v) for v in com])


def cell(run_dir, mode, ctrl="mpc", node=None):
    """certified / achieved / prescribed regions for one (target, mode) cell."""
    tag = mode.replace("+", "_")
    plan = json.load(open(os.path.join(run_dir, f"plan_{tag}.json")))
    subset = tuple(plan["subset"])
    modes = json.load(open(os.path.join(run_dir, "modes.json")))
    entry = next(e for e in modes["modes"] if e["name"] == mode)
    q_star = np.loadtxt(os.path.join(run_dir, entry["qpos_file"]))

    m, d = cs.load(ik_margin=0.0)
    out = {"mode": mode, "subset": list(subset),
           "target": modes["target"], "ctrl": ctrl}

    d.qpos[:len(q_star)] = q_star
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    out["certified"] = region_at(m, d, subset)
    out["certified_achieved_subset"] = list(achieved_subset(m, d))
    out["legs"] = region_at(m, d, ())

    qpath = os.path.join(run_dir, f"replay_{tag}_{ctrl}_q.npy")
    if os.path.exists(qpath):
        Q = np.load(qpath)
        # END OF THE BRACED PHASE, not the last node: with a return phase the
        # last node is back on the feet, where "what region does the brace buy"
        # is not a question about this plan.
        k = (len(Q) - 1 - plan.get("n_return", 0)) if node is None else node
        d.qpos[:] = Q[k]
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        got = achieved_subset(m, d)
        out["node"] = int(k)
        out["achieved_subset"] = list(got)
        out["achieved"] = region_at(m, d, got)
        out["prescribed"] = region_at(m, d, subset)
        out["legs_at_achieved"] = region_at(m, d, ())
        # CONTACT DUTY over the braced phase: the fraction of braced nodes on
        # which each planned site carried more than 5 N.  The end-of-phase
        # snapshot above says whether the brace is there at ONE instant, which a
        # brace that establishes and then lets go passes and fails on the same
        # run.  Duty is the whole phase, and it is free -- the replay log already
        # carries a per-node force for every planned site.
        rpath = os.path.join(run_dir, f"replay_{tag}_{ctrl}.json")
        if os.path.exists(rpath):
            log = json.load(open(rpath))["log"]
            braced = log[plan["n_approach"]:len(log) - plan.get("n_return", 0)]
            out["contact_duty"] = {
                s: (sum(r.get(f"F_{s}", 0.0) > 5.0 for r in braced) / len(braced))
                if braced else 0.0 for s in subset}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/2026-08-06_session13/sweep")
    ap.add_argument("--ctrl", default="mpc")
    ap.add_argument("--targets", default=None)
    ap.add_argument("--modes", default=None)
    ap.add_argument("--out", default="regions.json")
    args = ap.parse_args()

    import croco_sweep as sw
    targets = ([float(v) for v in args.targets.split(",")] if args.targets
               else list(sw.TARGETS))
    modes = args.modes.split(",") if args.modes else list(sw.LADDER)

    result = {"ctrl": args.ctrl, "cells": []}
    for x in targets:
        rd = sw.cell_dir(args.dir, x)
        for mode in modes:
            tag = mode.replace("+", "_")
            if not os.path.exists(os.path.join(rd, f"plan_{tag}.json")):
                continue
            try:
                c = cell(rd, mode, args.ctrl)
            except Exception as e:                          # noqa: BLE001
                print(f"  {sw.tname(x)}/{mode}: {e}")
                continue
            c["target_x"] = x
            result["cells"].append(c)
            got = "+".join(c.get("achieved_subset", [])) or "none"
            fmt = lambda v: "  none " if v is None else f"{1000*v:+7.1f}"
            duty = " ".join(f"{s}:{v:.0%}"
                            for s, v in c.get("contact_duty", {}).items())
            print(f"{sw.tname(x)} {mode:22s} certified {fmt(c['certified']['margin'])}"
                  f"  legs {fmt(c['legs']['margin'])}"
                  + (f"  achieved({got}) {fmt(c['achieved']['margin'])}"
                     if "achieved" in c else "  (no replay)")
                  + (f"  duty {duty}" if duty else ""), flush=True)

    path = os.path.join(args.dir, args.out)
    json.dump(result, open(path, "w"), indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
