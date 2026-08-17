#!/usr/bin/env python3
"""The (contact mode) x (reach target) matrix, end to end.

WHY THIS EXISTS.  Everything the crocoddyl track has measured so far sits at one
target -- x = 0.905, which the S11 envelope says the robot can reach on its LEGS
ALONE -- and one contact mode.  So "the plan reaches 2.2 mm and the brace carries
107 N" is a true statement about a task that did not need a brace, and it cannot
distinguish a controller that works from a controller that is along for the ride.
This runs the whole grid: for each target, certify each mode statically, plan it
with crocoddyl, replay it in MuJoCo under MPC, and score all three stages the
same way, so the columns can be read against each other.

THE CONTACT LADDER IS CURATED, NOT ENUMERATED.  There are 2^5 subsets over
{elbow, forearm, palm, hip, torso} and croco_modes.py will happily enumerate all
of them, but most are not things a robot would ever do, and a matrix padded with
them measures the enumerator rather than the maneuver.  The physical argument for
each row that IS here:

  legs_only            the control.  If the legs alone can hold the pose, the
                       brace is decoration and every metric that improves with
                       one is measuring something else.
  palm                 the NEAR-reach brace: a hand planted on the table.  It is
                       the only single contact a person makes without leaning,
                       and it is the cheapest to establish and to release.  Its
                       weakness is structural rather than postural -- the load
                       path runs through the wrist, whose pitch/yaw motors are
                       the weakest in the arm (clamp basis: 5 N.m against the
                       elbow's 18), so it should run out of capacity well before
                       the kinematics do.  Measured: it does, at x = 1.25, with
                       max |tau|/limit = 1.15 on a pose that is still reachable.
  elbow                the FAR-reach brace, and the one that puts load into the
                       strong part of the arm.  On its own because it is the
                       first contact of any progressive brace: you do not arrive
                       at forearm contact except through it.  Cheapest admissible
                       mode at every target in the grid.
  elbow+forearm        elbow first, then roll the forearm down.  Expected to be
                       nearly free once the elbow is on the table -- one joint
                       moving the way gravity is already pulling, spreading the
                       same load over a longer lever.  IT IS NOT: admissible at
                       the nearest target only.  As the reach extends the arm
                       straightens and the forearm LIFTS, and the IK's own answer
                       for this mode converges toward the arch below.
  elbow+palm           the ARCH: upper arm and hand down, forearm bridging clear
                       of the table.  Not a hypothesis -- it is what the far-reach
                       geometry actually produces, and it is admissible at every
                       target.
  forearm+palm         the arch's counterpart, elbow raised.  Admissible
                       throughout too, and the sweep's other surprise.
  elbow+forearm+palm   the whole arm laid down.  The S13 mode, and the most
                       fragile: admissible at exactly one target of the four,
                       because getting all three down at once over-constrains the
                       arm and the press has to bury the forearm to do it.

DELIBERATELY ABSENT.  `forearm` alone, which has no progressive route into it;
and `hip`/`torso`, which belong to a different maneuver (leaning the trunk on the
table rather than the arm) and which the S11 envelope already covers.

STAGES.  Each is idempotent and checkpoints after every cell, because this
machine has been dropping long jobs:

  certify   IK + static-equilibrium QP per (target, mode)  ->  q*, admissible?
  plan      crocoddyl, per admissible cell                 ->  xs/us/K + report
  replay    MuJoCo under MPC, per planned cell             ->  survival + reach

Plan and replay run as SUBPROCESSES.  It costs ~5 s of imports per cell and buys
the thing that matters here: a cell that segfaults inside crocoddyl (this study
has three documented ways to do that) loses that cell and not the sweep.

usage: croco_sweep.py [--only certify,plan,replay] [--dir runs/.../sweep]
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

import contact_select as cs

# Curated ladder, in increasing order of "how much of the arm is on the table".
# `elbow+palm` (the ARCH: upper arm and hand down, forearm bridging clear of the
# table) and `forearm+palm` are here because the certification stage put them
# there: at every target past the S13 one the IK's own answer for
# "elbow+forearm" plants the upper arm and the gripper and leaves the forearm
# 10 mm in the air, so the arch is not a hypothetical mode, it is the mode the
# far reach actually produces.  `forearm+palm` is never admissible and is kept as
# the datum that says so.
LADDER = ["legs_only", "palm", "elbow", "elbow+forearm", "elbow+palm",
          "forearm+palm", "elbow+forearm+palm"]
EXTRA = []

# Reach targets.  y and z are the study's, fixed; x is the axis that makes the
# task hard, and the four values are chosen off the S11 static envelope
# (reach_envelope_clamp.json) so the grid straddles the interesting boundaries
# rather than sampling uniformly:
#   0.905  the S12/S13 target.  Legs-only reaches it.  The easy control.
#   1.050  still inside the legs-only envelope, but at the edge of it.
#   1.150  OUTSIDE legs-only, inside palm and elbow+forearm.  The first target
#          where a brace is not optional.
#   1.250  outside palm too; only the elbow-rooted braces are admissible.
TARGETS = [0.905, 1.050, 1.150, 1.250]
TARGET_YZ = (-0.2348, 1.0982)


def tname(x):
    return f"x{int(round(x * 1000)):04d}"


def cell_dir(root, x):
    return os.path.join(root, tname(x))


# ------------------------------------------------------------------ certify --
def certify(root, targets, modes, force=False):
    """IK + static QP for every (target, mode); writes one modes.json per target.

    Same admissibility test the rest of the study uses (croco_modes): IK
    converged with the sites actually placed and nothing through the table, base
    residual < 1 N, and max |tau| within the clamp basis.  Recorded per cell so
    an inadmissible mode is a DATUM ("no static pose exists here") rather than a
    missing row.
    """
    import croco_modes as cm

    for x in targets:
        out = cell_dir(root, x)
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, "modes.json")
        if os.path.exists(path) and not force:
            print(f"{tname(x)}  modes.json exists, skipping certify")
            continue
        target = [x, TARGET_YZ[0], TARGET_YZ[1]]
        subsets = [tuple() if n == "legs_only" else tuple(n.split("+"))
                   for n in modes]
        print(f"\n=== certify {tname(x)}  target {target} ===", flush=True)
        recs = cm.enumerate_modes(target, sites=None, verbose=True,
                                  subsets=subsets)
        manifest = {"target": target, "sites": list(cs.ARM_SITES),
                    "model": os.path.basename(cs.MODEL),
                    "brace_arm": cs.BRACE_ARM, "site_set": cs.SITE_SET,
                    "seed_key": cs.SEED_KEY, "tau_basis": cs.TAU_BASIS,
                    "ladder": list(modes), "modes": []}
        for r in recs:
            name = "+".join(r["subset"]) or "legs_only"
            entry = {k: v for k, v in r.items() if k != "qpos"}
            entry["name"] = name
            if r["admissible"]:
                f = f"q_{name}.txt"
                np.savetxt(os.path.join(out, f), r["qpos"])
                entry["qpos_file"] = f
            manifest["modes"].append(entry)
        with open(path, "w") as fh:
            json.dump(manifest, fh, indent=1)
        print(f"wrote {path}")


# --------------------------------------------------------------------- run --
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


def plan_all(root, targets, modes, dt, n_approach, n_braced, dwell, iters,
             timeout, force=False):
    here = os.path.dirname(os.path.abspath(__file__))
    for x in targets:
        out = cell_dir(root, x)
        man = json.load(open(os.path.join(out, "modes.json")))
        adm = {e["name"] for e in man["modes"] if e.get("qpos_file")}
        for name in modes:
            if name not in adm:
                print(f"{tname(x)} {name:22s} inadmissible, no plan")
                continue
            tag = name.replace("+", "_")
            if os.path.exists(os.path.join(out, f"plan_{tag}.json")) and not force:
                print(f"{tname(x)} {name:22s} plan exists")
                continue
            rc, secs = _sub(
                [os.path.join(here, "croco_run.py"), "--mode", name,
                 "--tag", tag, "--dir", out, "--dt", str(dt),
                 "--n-approach", str(n_approach), "--n-braced", str(n_braced),
                 "--dwell", str(dwell), "--iters", str(iters)],
                os.path.join(out, f"plan_{tag}.log"), timeout)
            ok = os.path.exists(os.path.join(out, f"plan_{tag}.json"))
            print(f"{tname(x)} {name:22s} plan rc={rc} {secs:6.1f}s "
                  f"{'ok' if ok else 'FAILED'}", flush=True)


def replay_all(root, targets, modes, ctrl, horizon, mpc_iters, timeout,
               force=False):
    here = os.path.dirname(os.path.abspath(__file__))
    for x in targets:
        out = cell_dir(root, x)
        for name in modes:
            tag = name.replace("+", "_")
            if not os.path.exists(os.path.join(out, f"plan_{tag}.json")):
                continue
            dst = os.path.join(out, f"replay_{tag}_{ctrl}.json")
            if os.path.exists(dst) and not force:
                print(f"{tname(x)} {name:22s} replay exists")
                continue
            rc, secs = _sub(
                [os.path.join(here, "croco_replay.py"), "--tag", tag,
                 "--ctrl", ctrl, "--dir", out,
                 "--mpc-horizon", str(horizon), "--mpc-iters", str(mpc_iters)],
                os.path.join(out, f"replay_{tag}_{ctrl}.log"), timeout)
            print(f"{tname(x)} {name:22s} replay/{ctrl} rc={rc} {secs:6.1f}s "
                  f"{'ok' if os.path.exists(dst) else 'FAILED'}", flush=True)


# ----------------------------------------------------------------- collect --
def reason(entry):
    """WHY a cell is inadmissible -- the causes are different claims.

    Order matters: they are tested the way the maneuver would hit them.  A pose
    that cannot be reached is a kinematic statement; one that reaches by putting
    a link through the wood is a geometric one; one that reaches and cannot be
    balanced is about the support region; and one that balances only outside the
    torque basis is about the motors.  Collapsing them all into "inadmissible" is
    what left the earlier single-target study unable to say anything about modes.
    """
    if entry["admissible"]:
        return "ok"
    if entry["reach"] >= 0.03:
        return "out of reach"
    if not entry["placed"]:
        return "no contact"
    if entry["penetration"] >= 0.01:
        return "through table"
    if entry["base_res"] >= 1.0:
        return "no balance"
    if entry["max_ratio"] > 1.0:
        return "torque limit"
    return "inadmissible"


def collect(root, targets, modes, ctrls):
    """One row per (target, mode): the static, planned and simulated views."""
    rows = []
    for x in targets:
        out = cell_dir(root, x)
        mpath = os.path.join(out, "modes.json")
        if not os.path.exists(mpath):
            continue
        man = json.load(open(mpath))
        by = {e["name"]: e for e in man["modes"]}
        for name in modes:
            e = by.get(name)
            if e is None:
                continue
            row = dict(target_x=x, mode=name,
                       static_admissible=bool(e["admissible"]),
                       static_effort=e["effort"], static_ratio=e["max_ratio"],
                       static_reach_mm=1000 * e["reach"],
                       static_brace_N=e["brace_force"],
                       static_placed=bool(e["placed"]),
                       static_pen_mm=1000 * e["penetration"],
                       static_base_res=e["base_res"],
                       static_gap_mm=e.get("contact_gap_mm", {}),
                       static_reason=reason(e))
            tag = name.replace("+", "_")
            p = os.path.join(out, f"plan_{tag}.json")
            if os.path.exists(p):
                pl = json.load(open(p))
                row.update(planned=True, plan_converged=pl["converged"],
                           plan_iters=pl["iters"],
                           plan_seconds=pl["solve_seconds"],
                           plan_reach_mm=1000 * pl.get("reach_err", float("nan")),
                           plan_tau_ratio=pl["max_torque_ratio"],
                           plan_q_err=pl["q_err_vs_qstar_rad"],
                           plan_brace_dz_mm=pl.get("brace_dz_worst_mm"))
            else:
                row["planned"] = False
            for c in ctrls:
                r = os.path.join(out, f"replay_{tag}_{c}.json")
                if not os.path.exists(r):
                    continue
                s = json.load(open(r))["summary"]
                row[f"{c}_fell"] = s["fell"]
                row[f"{c}_reach_mm"] = 1000 * s["reach_err_at_brace_end"]
                row[f"{c}_reach_sd_mm"] = 1000 * (s["reach_err_braced_std"] or 0)
                row[f"{c}_brace_N"] = s["brace_total_braced_mean"]
                row[f"{c}_margin_mm"] = 1000 * s["min_support_margin"]
                row[f"{c}_pen_mm"] = 1000 * s["worst_penetration"]
                row[f"{c}_tau"] = s["max_tau_ratio"]
                if "mpc_solve_ms" in s:
                    row[f"{c}_solve_ms"] = s["mpc_solve_ms"]["mean"]
            rows.append(row)
    return rows


def show(rows, ctrl):
    print(f"\n{'target':>7} {'mode':22} {'static':>7} {'plan':>9} "
          f"{'sim':>9} {'brace':>7} {'margin':>8} {'up?':>5}")
    for r in rows:
        st = "ok" if r["static_admissible"] else "--"
        pl = (f"{r['plan_reach_mm']:7.1f}mm" if r.get("planned")
              else "        -")
        k = f"{ctrl}_reach_mm"
        sim = f"{r[k]:7.1f}mm" if k in r else "        -"
        br = f"{r.get(ctrl + '_brace_N', float('nan')):6.1f}N" \
            if ctrl + "_brace_N" in r else "      -"
        mg = f"{r.get(ctrl + '_margin_mm', float('nan')):+7.1f}" \
            if ctrl + "_margin_mm" in r else "      -"
        up = ("FELL" if r.get(ctrl + "_fell") else " up ") \
            if ctrl + "_fell" in r else "  -  "
        print(f"{r['target_x']:7.3f} {r['mode']:22} {st:>7} {pl} {sim} "
              f"{br} {mg} {up:>5}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/2026-08-06_session13/sweep")
    ap.add_argument("--only", default="certify,plan,replay,collect")
    ap.add_argument("--targets", default=None,
                    help="comma-separated x values (default: the ladder above)")
    ap.add_argument("--modes", default=None,
                    help="comma-separated contact modes (default: LADDER)")
    ap.add_argument("--extra", action="store_true",
                    help="also run the arched elbow+palm mode")
    ap.add_argument("--ctrl", default="mpc")
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--n-approach", type=int, default=120)
    ap.add_argument("--n-braced", type=int, default=80)
    ap.add_argument("--dwell", type=int, default=0)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--mpc-horizon", type=int, default=50)
    ap.add_argument("--mpc-iters", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=1200)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="sweep.json")
    args = ap.parse_args()

    targets = ([float(v) for v in args.targets.split(",")] if args.targets
               else list(TARGETS))
    modes = (args.modes.split(",") if args.modes
             else list(LADDER) + (EXTRA if args.extra else []))
    want = set(args.only.split(","))
    os.makedirs(args.dir, exist_ok=True)

    if "certify" in want:
        certify(args.dir, targets, modes, force=args.force)
    if "plan" in want:
        plan_all(args.dir, targets, modes, args.dt, args.n_approach,
                 args.n_braced, args.dwell, args.iters, args.timeout,
                 force=args.force)
    if "replay" in want:
        replay_all(args.dir, targets, modes, args.ctrl, args.mpc_horizon,
                   args.mpc_iters, args.timeout, force=args.force)
    if "collect" in want:
        ctrls = [args.ctrl] if args.ctrl == "mpc" else [args.ctrl, "mpc"]
        rows = collect(args.dir, targets, modes, sorted(set(ctrls + ["mpc", "ff"])))
        show(rows, args.ctrl)
        path = os.path.join(args.dir, args.out)
        json.dump({"targets": targets, "modes": modes,
                   "target_yz": list(TARGET_YZ), "ctrl": args.ctrl,
                   "rows": rows}, open(path, "w"), indent=1)
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
