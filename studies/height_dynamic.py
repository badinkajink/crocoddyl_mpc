#!/usr/bin/env python3
"""Does the crocoddyl brace SURVIVE at a table height MJPC cannot handle?

`height_reach.py` certifies that a statically valid, torque-feasible,
balance-feasible braced pose exists at every height the MJPC study has run,
including the four it fails at.  That is a claim about a POSE.  It does not say
the robot can get there, and `croco_modes` is a conservative screen rather than
a workspace bound (`cmpc-legs-only-is-real`: its 1.14 m rejection was about the
IK's pinned-foot pose, not the workspace).  This closes that gap: plan the
maneuver with crocoddyl, fly it closed-loop in MuJoCo, and hold.

THE COMPARISON THIS SETS UP.  MJPC and CMPC differ in what a contact mode IS --
MJPC's is three cost weights over a contact-implicit planner, CMPC's is a
contact SCHEDULE baked into the action models, and no weight can add a contact
to it.  So if the sampling planner's window is 0.985-1.035 m and the gradient
planner's is wider on the same robot, same slab and same target offset, the
difference is planner architecture and not the machine.  That is the motivator
the study wants, and it is also the diagnosis MJPC's own rollouts cannot make.

ONE FACE PER PROCESS, deliberately.  `solve_plans` already shells out per plan
because croco_run mutates contact_select's module globals; the same argument
applies across heights, and a crocoddyl OCP built against one slab must not be
reused against another.  Loop over faces in the shell, not here.

usage: TABLE_H is set BY THIS SCRIPT from --face; do not also set it in the
       environment.

  height_dynamic.py --face 1.085 --out runs/2026-09-08_height_dyn --seeds 0,1,2
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# The face has to be in the environment BEFORE contact_select is imported: it
# reads TABLE_H at import time, and croco_run is a subprocess that inherits it.
_FACE = None
for _i, _a in enumerate(sys.argv):
    if _a == "--face" and _i + 1 < len(sys.argv):
        _FACE = float(sys.argv[_i + 1])
    elif _a.startswith("--face="):
        _FACE = float(_a.split("=", 1)[1])
if _FACE is not None:
    os.environ["TABLE_H"] = "%.6f" % _FACE

import contact_select as cs                              # noqa: E402
import cmpc_brace_vs_stand as BVS                        # noqa: E402
import mujoco                                            # noqa: E402

# The shipped target sits 113.2 mm above the compiled 0.985 m face.  Holding
# that offset is the "same task at a different height" convention, and it is
# MJPC's own: its rung-2 target is face + 0.15.
SHIPPED_TARGET = (0.9047, -0.2348, 1.0982)
SHIPPED_FACE = 0.985
DZ = SHIPPED_TARGET[2] - SHIPPED_FACE

# DEFAULT REACH x IS 1.06, NOT THE PAPER'S 0.9047, AND THAT IS NOT A DETAIL OF
# TASTE.  x = 0.9047 is the single target on the whole ladder where the CMPC
# brace does not survive a hold: measured 2026-08-22, elbow+forearm at Kp=50,
# it falls at 10.2 s while legs_only holds indefinitely, and the certified
# brace force there is 51.5 N against 80.9 N at 1.05 -- the declared contact is
# a light touch and taking it is HARMFUL, in every contact mode (mode selection
# only halves the fall rate 2/2 -> 1/2).  A height sweep run at 0.9047 measures
# that near-target pathology at every height and calls it a table-height
# result.  `brace_vs_stand.NOMINAL2_X` exists for exactly this reason and is
# the settled condition the whole comparison carries.
DEFAULT_TX = float(os.environ.get("BVS_NOMINAL2_X", "1.06"))

ARMS = [("brace", "elbow_forearm"), ("stand", "stand")]


def outcome(flight, plant, face, target):
    """What the run ended as: upright, on target, and actually braced.

    Three separate questions, because they fail separately -- MJPC's 1.050 and
    1.060 runs seat the brace and load it and topple anyway, and its 1.085 runs
    never touch the wood at all.
    """
    m, d = flight.m, plant.d
    mujoco.mj_forward(m, d)
    hand = cs.point_world(m, d, cs.REACH_BODY, cs.REACH_OFF)
    tbl = cs.bid(m, "table")
    brace_N, pads = 0.0, []
    f = np.zeros(6)
    for c in range(d.ncon):
        con = d.contact[c]
        b = [m.geom_bodyid[con.geom[0]], m.geom_bodyid[con.geom[1]]]
        if tbl not in b:
            continue
        rb = b[1] if b[0] == tbl else b[0]
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, rb) or ""
        if "object" in nm:
            continue
        mujoco.mj_contactForce(m, d, c, f)
        brace_N += abs(float(f[0]))
        pads.append(nm)
    return dict(
        pelvis_z=float(d.qpos[2]),
        upright=bool(d.qpos[2] > 0.80),
        reach_mm=float(1000.0 * np.linalg.norm(np.asarray(target) - hand)),
        brace_N=float(brace_N),
        contact_bodies=sorted(set(pads)),
        braced=bool(brace_N > 5.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--face", type=float, required=True,
                    help="table top face height [m]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tx", type=float, default=None,
                    help="reach target x [m]; default %.3f, the settled "
                         "condition. Do NOT set 0.9047 for a height study -- "
                         "see DEFAULT_TX." % DEFAULT_TX)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--seconds", type=float, default=25.0,
                    help="episode length; the plan is ~4 s and the rest holds")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    assert abs(cs.TABLE_H - a.face) < 1e-9, (
        "TABLE_H did not reach contact_select: %r vs %r" % (cs.TABLE_H, a.face))

    seeds = [int(v) for v in a.seeds.split(",") if v.strip()]
    tx = DEFAULT_TX if a.tx is None else a.tx
    target = (tx, SHIPPED_TARGET[1], a.face + DZ)
    root = os.path.join(a.out, "h%04d_x%04d"
                        % (round(a.face * 1000), round(tx * 1000)))
    cell = os.path.join(root, "cell")
    os.makedirs(root, exist_ok=True)

    print("face %.3f   target %.4f %.4f %.4f   seeds %s   %.0f s episodes"
          % (a.face, *target, seeds, a.seconds), flush=True)

    man = BVS.build_cell(cell, target, force=a.force)
    # Provenance: a cell certified against a moved slab must say so, or it is
    # indistinguishable from one built at the compiled height.
    man["table_h"] = float(a.face)
    with open(os.path.join(cell, "modes.json"), "w") as fh:
        json.dump(man, fh, indent=1)
    print("  admissible: %s"
          % {mm["name"]: mm["admissible"] for mm in man["modes"]}, flush=True)

    plans = BVS.solve_plans(cell, force=a.force, recover=False)
    if len(plans) < 2:
        print("  SOLVE FAILED (%d of 2 plans); nothing to fly" % len(plans))
        json.dump(dict(face=a.face, target=list(target), plans=list(plans),
                       episodes=[], solve_failed=True),
                  open(os.path.join(root, "summary.json"), "w"), indent=1)
        return

    tau_lim = cs.torque_limits(cs.load(ik_margin=0.0)[0])
    plant = BVS.make_recording_plant(tau_lim)

    eps = []
    for arm, tag in ARMS:
        if tag not in plans:
            continue
        t0 = time.time()
        fl = BVS.Flight(cell, tag)
        print("  built OCP %-14s %.1f s" % (tag, time.time() - t0), flush=True)
        for seed in seeds:
            name = "h%04d_x%04d_%s_r%d" % (round(a.face * 1000),
                                                  round(tx * 1000), arm, seed)
            csv_path = os.path.join(root, name + ".csv")
            if os.path.exists(csv_path) and not a.force:
                print("  %-22s cached" % name, flush=True)
                continue
            rows, summ = BVS.run_episode(fl, plant, a.seconds, seed=seed)
            BVS.write_csv(csv_path, rows, fl, target, arm, seed, a.seconds, "")
            oc = outcome(fl, plant, a.face, target)
            print("  %-22s pelvis %.3f  %-8s reach %6.1f mm  brace %6.1f N  %s"
                  % (name, oc["pelvis_z"],
                     "UPRIGHT" if oc["upright"] else "DOWN",
                     oc["reach_mm"], oc["brace_N"],
                     "+".join(oc["contact_bodies"]) or "no table contact"),
                  flush=True)
            eps.append(dict(tag=name, arm=arm, seed=seed, seconds=a.seconds,
                            wall_s=summ.get("wall_s"),
                            solve_ms_p95=summ.get("solve_ms_p95"),
                            overruns=summ.get("overruns"), **oc))

    out = dict(face=float(a.face), tx=float(tx),
               target=[float(v) for v in target],
               table_h=float(cs.TABLE_H), seeds=seeds, seconds=a.seconds,
               model=os.path.basename(cs.MODEL), seed_key=cs.SEED_KEY,
               contact_kp=BVS.CONTACT_KP, episodes=eps)
    with open(os.path.join(root, "summary.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    ok = sum(1 for e in eps if e["arm"] == "brace" and e["upright"] and e["braced"])
    n = sum(1 for e in eps if e["arm"] == "brace")
    print("\nface %.3f: braced upright-and-loaded %d/%d" % (a.face, ok, n))


if __name__ == "__main__":
    main()
