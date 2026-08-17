#!/usr/bin/env python3
"""The braced reach over a DIVERSE target set: far, askew, and off-centre stance.

WHY THIS AND NOT croco_sweep.  S13b's matrix sweeps x with y and z pinned at the
study's original values, so every cell asks the same question -- how far forward
can it reach -- and the answer has been the same since S11.  It cannot see the
axis that actually breaks this maneuver, which is LATERAL: the reach target sits
60 mm from the table's far rail while the robot stands square on the centreline,
so reaching it twists the torso and throws the bracing arm out over the near
rail.  croco_stance.py measured that statically in S14 and found it expensive
(effort down 2.9x, one torque-infeasible cell made comfortable, by moving the
robot 120-180 mm toward the target) -- and then nothing in the crocoddyl track
used it, because croco_run read its start pose straight out of `m.key_qpos` and
ignored STANCE_DY entirely.  That is fixed (contact_select.start_qpos) and this
module is what it is fixed for.

WHAT A CELL IS.  A reach target (x, y, z) plus a stance offset dy, carried
consistently through all four stages:

  certify   IK + static-equilibrium QP over the contact ladder  ->  q* per mode
  plan      crocoddyl, on the cheapest admissible mode          ->  xs/us/K
  stress    MuJoCo under the MPC, N seeds of initial-joint noise -> survival
  collect   one row per cell, scored by croco_robust.verdict

The verdict is the user's: upright, hand on target, torques inside the clamp
basis, nothing through the table.  WHICH contacts got there is recorded and not
required -- if a pattern falls out of the load-balancing it is a finding, not a
constraint.

usage: croco_grid.py certify --dir runs/.../grid
       croco_grid.py plan    --dir runs/.../grid
       croco_grid.py stress  --dir runs/.../grid --seeds 1 2 3 4 5 6 7 8
       croco_grid.py collect --dir runs/.../grid --out grid.json
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time

import numpy as np

import contact_select as cs

HERE = os.path.dirname(os.path.abspath(__file__))

# The contact ladder, from croco_sweep: curated, not enumerated, with the
# physical argument for each row in that module's docstring.  `forearm+palm` is
# dropped here (never admissible anywhere in S13b) and `legs_only` is kept as the
# control that says whether a brace was needed at all.
LADDER = ["legs_only", "palm", "elbow", "elbow+forearm", "elbow+palm",
          "elbow+forearm+palm"]

# THE TARGET SET.  Two families, because they fail differently.
#
#   reach   x from inside the legs-only envelope out past where any brace helps,
#           on the study's own y = -0.235 line.  This is S13b's axis, kept so the
#           new results can be read against the old ones.
#   askew   y swept across the full width of the table at two depths.  +y is the
#           BRACING arm's side, where the reaching arm has to cross the body and
#           the brace has nowhere to go; -y is the far rail.  The table is only
#           595 mm wide, so |y| = 0.28 is 18 mm from the edge.
#
# z is 1.098 throughout (113 mm above the tabletop) except for the two `low`
# cells, which put the hand 35 mm off the wood -- the case S7.4 flagged as
# unsolved and never revisited.
TABLE_Z = 0.985
TARGETS = (
    [(x, -0.2348, 1.0982) for x in (0.905, 1.050, 1.150, 1.250)] +
    [(1.050, y, 1.0982) for y in (-0.280, -0.120, 0.000, 0.120)] +
    [(1.150, y, 1.0982) for y in (-0.280, -0.120, 0.000)] +
    [(1.050, -0.2348, 1.020), (1.150, -0.2348, 1.020)]
)

# Stance offsets to certify each target at.  0 is the study's stance and the
# control; -0.12 is the middle of the band croco_stance found useful.
STANCES = (0.0, -0.12)


def cname(t, dy):
    return (f"x{int(round(t[0]*1000)):04d}_y{int(round(t[1]*1000)):+04d}"
            f"_z{int(round(t[2]*1000)):04d}_dy{int(round(dy*1000)):+04d}")


def cells(targets=None, stances=None):
    return [(t, dy) for t in (targets or TARGETS)
            for dy in (stances or STANCES)]


def cell_dir(root, t, dy):
    return os.path.join(root, cname(t, dy))


def _sub(args, log_path, timeout):
    """Run a stage in its own process; keep the log; never raise."""
    t0 = time.time()
    with open(log_path, "w") as fh:
        try:
            p = subprocess.run([sys.executable] + args, stdout=fh,
                               stderr=subprocess.STDOUT, timeout=timeout,
                               cwd=os.getcwd())
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = -9
    return rc, time.time() - t0


# ---------------------------------------------------------------- certify --
def certify(root, cs_cells, modes, force=False):
    import croco_modes as cm

    for t, dy in cs_cells:
        out = cell_dir(root, t, dy)
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, "modes.json")
        if os.path.exists(path) and not force:
            print(f"{cname(t, dy)}  exists")
            continue
        cs.STANCE_DY = dy
        subsets = [tuple() if n == "legs_only" else tuple(n.split("+"))
                   for n in modes]
        print(f"\n=== {cname(t, dy)}  target {t}  stance dy {dy:+.3f} ===",
              flush=True)
        t0 = time.time()
        recs = cm.enumerate_modes(list(t), sites=None, verbose=True,
                                  subsets=subsets)
        man = {"target": list(t), "stance_dx": cs.STANCE_DX, "stance_dy": dy,
               "sites": list(cs.ARM_SITES), "model": os.path.basename(cs.MODEL),
               "brace_arm": cs.BRACE_ARM, "site_set": cs.SITE_SET,
               "seed_key": cs.SEED_KEY, "tau_basis": cs.TAU_BASIS,
               "ladder": list(modes), "seconds": time.time() - t0, "modes": []}
        for r in recs:
            name = "+".join(r["subset"]) or "legs_only"
            e = {k: v for k, v in r.items() if k != "qpos"}
            e["name"] = name
            if r["admissible"]:
                f = f"q_{name}.txt"
                np.savetxt(os.path.join(out, f), r["qpos"])
                e["qpos_file"] = f
            man["modes"].append(e)
        adm = [e for e in man["modes"] if e["admissible"]]
        man["ranked"] = [e["name"] for e in
                         sorted(adm, key=lambda e: (e["effort"], e["n_arm"]))]
        with open(path, "w") as fh:
            json.dump(man, fh, indent=1)
        print(f"  {len(adm)} admissible; best {man['ranked'][:3]} "
              f"({time.time()-t0:.0f}s)", flush=True)


def best_mode(man, prefer_brace=True):
    """Cheapest admissible mode, ties toward fewer contacts.

    `prefer_brace` skips `legs_only` when anything else is admissible: the point
    of the study is the braced reach, and legs_only is carried as the control
    column rather than as a candidate the planner should pick.
    """
    ranked = man.get("ranked") or []
    if prefer_brace:
        braced = [n for n in ranked if n != "legs_only"]
        if braced:
            return braced[0]
    return ranked[0] if ranked else None


# ------------------------------------------------------------------- plan --
def plan_all(root, cs_cells, dt, n_approach, n_braced, iters, weights,
             timeout, force=False, modes=None):
    for t, dy in cs_cells:
        out = cell_dir(root, t, dy)
        mpath = os.path.join(out, "modes.json")
        if not os.path.exists(mpath):
            continue
        man = json.load(open(mpath))
        names = modes or [best_mode(man)]
        for name in names:
            if not name:
                print(f"{cname(t, dy)}  no admissible mode, no plan")
                continue
            tag = name.replace("+", "_")
            if os.path.exists(os.path.join(out, f"plan_{tag}.json")) \
                    and not force:
                print(f"{cname(t, dy)} {name:20s} plan exists")
                continue
            cmd = [os.path.join(HERE, "croco_run.py"), "--mode", name,
                   "--tag", tag, "--dir", out, "--dt", str(dt),
                   "--n-approach", str(n_approach), "--n-braced",
                   str(n_braced), "--iters", str(iters)]
            for k, v in weights.items():
                cmd += [f"--{k.replace('_', '-')}", str(v)]
            rc, secs = _sub(cmd, os.path.join(out, f"plan_{tag}.log"), timeout)
            ok = os.path.exists(os.path.join(out, f"plan_{tag}.json"))
            print(f"{cname(t, dy)} {name:20s} plan rc={rc} {secs:6.1f}s "
                  f"{'ok' if ok else 'FAILED'}", flush=True)


# ----------------------------------------------------------------- stress --
def stress_all(root, cs_cells, seeds, profiles, horizon, iters, threads,
               tau_clamp, settle=0, force=False):
    import croco_robust as crb

    for t, dy in cs_cells:
        out = cell_dir(root, t, dy)
        mpath = os.path.join(out, "modes.json")
        if not os.path.exists(mpath):
            continue
        man = json.load(open(mpath))
        name = best_mode(man)
        if not name:
            continue
        tag = name.replace("+", "_")
        if not os.path.exists(os.path.join(out, f"plan_{tag}.json")):
            continue
        dst = os.path.join(out, f"stress_{tag}.json")
        if os.path.exists(dst) and not force:
            print(f"{cname(t, dy)} {name:20s} stress exists")
            continue
        print(f"\n=== {cname(t, dy)}  {name}", flush=True)
        rows = crb.stress(tag, out, seeds=seeds, profiles=profiles,
                          horizon=horizon, iters=iters, threads=threads,
                          tau_clamp=tau_clamp, settle=settle)
        r, n = crb.rate(rows, profiles[-1])
        json.dump(dict(target=list(t), stance_dy=dy, mode=name, rows=rows,
                       settle=settle, profiles=list(profiles),
                       survive=r, survive_n=n,
                       rate={p: list(crb.rate(rows, p)) for p in profiles}),
                  open(dst, "w"), indent=1)
        print(f"    -> {r*100:.0f}% of {n}", flush=True)


# ----------------------------------------------------------------- videos --
# The cells the page shows.  Chosen to span what the grid is FOR: the far edge
# where it is hardest, both extremes of the lateral axis, the stance-shifted
# version of the same reach, and one hand-near-the-wood target.
VIDEO_CELLS = [
    ((1.250, -0.2348, 1.0982), 0.0, "brace",
     "x = 1.25 m, the far edge of the envelope. The hardest cell in the grid "
     "and the one that still falls under the winch."),
    ((1.150, -0.280, 1.0982), 0.0, "wide",
     "x = 1.15, y = &minus;0.28 &mdash; askew to the far rail, 18 mm from the "
     "edge of the table."),
    ((1.050, 0.120, 1.0982), 0.0, "wide",
     "y = +0.12, askew toward the BRACING arm's side: the reaching arm crosses "
     "the body and the brace has less table to work with."),
    ((1.150, -0.2348, 1.0982), -0.12, "wide",
     "The same far reach with the robot stood 120 mm toward the target. "
     "Survival 75% against 62% square-on."),
    ((1.150, -0.2348, 1.020), 0.0, "brace",
     "Hand 35 mm off the wood, the low target S7.4 left unsolved."),
    ((1.050, -0.2348, 1.0982), 0.0, "wide",
     "The middle of the envelope, for reference: stand, lean, brace, reach, hold."),
]


def make_videos(root, media, seed=None, settle=25, horizon=35, iters=1,
                threads=12, force=False):
    """Render the docpage's videos, one per chosen cell, through the same
    replay every number in the grid came from."""
    import croco_robust as crb
    os.makedirs(media, exist_ok=True)
    out = []
    for t, dy, cam, caption in VIDEO_CELLS:
        cell = cell_dir(root, t, dy)
        mpath = os.path.join(cell, "modes.json")
        if not os.path.exists(mpath):
            print(f"{cname(t, dy)}  not certified, skipping")
            continue
        name = best_mode(json.load(open(mpath)))
        tag = name.replace("+", "_")
        if not os.path.exists(os.path.join(cell, f"plan_{tag}.json")):
            continue
        fn = f"s17_{cname(t, dy)}_{cam}.mp4"
        dst = os.path.join(media, fn)
        if os.path.exists(dst) and not force:
            print(f"{fn}  exists")
        else:
            r, _ = crb.one(tag, cell, seed=seed,
                           profile="nominal" if seed is None else "winch1",
                           horizon=horizon, iters=iters, threads=threads,
                           settle=settle, video=dst, cam=cam)
            print(f"{fn}  {'OK' if r['ok'] else r['why']}  "
                  f"reach {r['reach_mm']:.1f} mm", flush=True)
        out.append(dict(file=fn, cell=cname(t, dy), target=list(t),
                        stance_dy=dy, mode=name, cam=cam,
                        caption=f"<b>{name}</b>. {caption}"))
    return out


# ---------------------------------------------------------------- collect --
def collect(root, cs_cells):
    import croco_sweep as csw
    rows = []
    for t, dy in cs_cells:
        out = cell_dir(root, t, dy)
        mpath = os.path.join(out, "modes.json")
        if not os.path.exists(mpath):
            continue
        man = json.load(open(mpath))
        by = {e["name"]: e for e in man["modes"]}
        name = best_mode(man)
        row = dict(target=list(t), stance_dy=dy, cell=cname(t, dy),
                   n_admissible=sum(e["admissible"] for e in man["modes"]),
                   admissible=[e["name"] for e in man["modes"]
                               if e["admissible"]],
                   reasons={e["name"]: csw.reason(e) for e in man["modes"]},
                   mode=name)
        if name:
            e = by[name]
            row.update(static_effort=e["effort"], static_ratio=e["max_ratio"],
                       static_brace_N=e["brace_force"],
                       static_reach_mm=1000 * e["reach"])
            tag = name.replace("+", "_")
            p = os.path.join(out, f"plan_{tag}.json")
            if os.path.exists(p):
                pl = json.load(open(p))
                row.update(planned=True, plan_converged=pl["converged"],
                           plan_reach_mm=1000 * pl.get("reach_err", np.nan),
                           plan_tau=pl["max_torque_ratio"],
                           plan_seconds=pl["solve_seconds"])
            s = os.path.join(out, f"stress_{tag}.json")
            if os.path.exists(s):
                st = json.load(open(s))
                nom = [r for r in st["rows"] if r.get("profile") == "nominal"]
                row.update(survive=st["survive"], survive_n=st["survive_n"],
                           nominal_ok=bool(nom and nom[0]["ok"]),
                           nominal_why=nom[0]["why"] if nom else None,
                           rate=st.get("rate", {}),
                           why=[r["why"] for r in st["rows"]
                                if r.get("profile") != "nominal"],
                           reach_mm=float(np.median(
                               [r["reach_mm"] for r in st["rows"]])),
                           tau=float(np.max([r["tau"] for r in st["rows"]])),
                           brace_N=float(np.median(
                               [r["brace_N"] for r in st["rows"]])),
                           solve_ms=float(np.median(
                               [r["solve_ms"] for r in st["rows"]
                                if r["solve_ms"]])))
        rows.append(row)
    return rows


def show(rows):
    print(f"\n{'cell':30s} {'mode':20s} {'adm':>4s} {'eff':>6s} {'plan':>8s} "
          f"{'nom':>5s} {'survive':>8s} {'reach':>8s} {'tau':>5s}")
    for r in rows:
        pl = f"{r.get('plan_reach_mm', float('nan')):7.1f}m" \
            if r.get("planned") else "       -"
        sv = f"{r['survive']*100:5.0f}% " if "survive" in r else "       -"
        print(f"{r['cell']:30s} {str(r['mode']):20s} {r['n_admissible']:4d} "
              f"{r.get('static_effort', float('nan')):6.3f} {pl} "
              f"{('OK' if r.get('nominal_ok') else '--'):>5s} {sv} "
              f"{r.get('reach_mm', float('nan')):7.1f} "
              f"{r.get('tau', float('nan')):5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage",
                    choices=["certify", "plan", "stress", "collect", "videos"])
    ap.add_argument("--dir", default="runs/2026-08-14_session17/grid")
    ap.add_argument("--targets", default=None,
                    help="'x,y,z;x,y,z' (default: the grid above)")
    ap.add_argument("--stances", default=None,
                    help="comma-separated dy values (default: 0,-0.12)")
    ap.add_argument("--modes", default=None,
                    help="comma-separated ladder (certify) / plan these modes")
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--n-approach", type=int, default=120)
    ap.add_argument("--n-braced", type=int, default=80)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=35)
    ap.add_argument("--mpc-iters", type=int, default=1)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--no-tau-clamp", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--profiles", nargs="+", default=["nominal", "winch1"])
    ap.add_argument("--settle", type=int, default=50)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--weights", default="",
                    help="k=v,k=v passed to croco_run as --k v for every plan")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    targets = None
    if args.targets:
        targets = [tuple(float(v) for v in c.split(","))
                   for c in args.targets.split(";")]
    stances = None
    if args.stances:
        stances = [float(v) for v in args.stances.split(",")]
    modes = args.modes.split(",") if args.modes else None
    weights = dict(kv.split("=") for kv in args.weights.split(",")) \
        if args.weights else {}
    cl = cells(targets, stances)
    os.makedirs(args.dir, exist_ok=True)
    print(f"{len(cl)} cells in {args.dir}")

    if args.stage == "certify":
        certify(args.dir, cl, modes or LADDER, force=args.force)
    elif args.stage == "plan":
        plan_all(args.dir, cl, args.dt, args.n_approach, args.n_braced,
                 args.iters, weights, args.timeout, force=args.force,
                 modes=modes)
    elif args.stage == "stress":
        stress_all(args.dir, cl, args.seeds, args.profiles, args.horizon,
                   args.mpc_iters, args.threads, not args.no_tau_clamp,
                   settle=args.settle, force=args.force)
    elif args.stage == "videos":
        v = make_videos(args.dir, os.path.join(HERE, "..", "docs", "lean",
                                               "media"),
                        settle=args.settle, horizon=args.horizon,
                        iters=args.mpc_iters, threads=args.threads,
                        force=args.force)
        out = args.out or os.path.join(args.dir, "videos.json")
        json.dump(dict(videos=v), open(out, "w"), indent=1)
        print(f"\nwrote {out}")
    else:
        rows = collect(args.dir, cl)
        show(rows)
        out = args.out or os.path.join(args.dir, "grid.json")
        json.dump(dict(rows=rows, targets=[list(t) for t in (targets or TARGETS)],
                       stances=list(stances or STANCES)), open(out, "w"),
                  indent=1)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
