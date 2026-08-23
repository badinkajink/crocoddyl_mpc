#!/usr/bin/env python3
"""The braced-vs-standing reach comparison, run against CROCODDYL-MPC.

`brace_vs_stand.py` asks one question of the sampling planner (MJPC): how much
reach does a brace buy, and what does it cost in stability? This asks the SAME
question, over the SAME stages, at the SAME targets, of the gradient planner
(CMPC / BoxFDDP) -- so the two answers are comparable and the paper's App. D can
carry a figure rather than a paragraph.

WHY THIS IS A SEPARATE HARNESS AND NOT A FLAG. The two planners do not differ in
their weights, they differ in what a "contact mode" IS:

    MJPC    contact-implicit. The mode is three cost weights. `stand` is
            Brace Elbow/Forearm/Palm = 0 and the planner may still discover the
            table (and does -- up to 76 N through a gripper on a zero-weight run).
    CMPC    contact-explicit. The mode is a CONTACT SCHEDULE baked into the
            action models. `stand` is the `legs_only` plan -- feet-only contacts
            for every node -- and no weight can add a contact to it. Reaching
            the same target with the arm off the table is a DIFFERENT OCP, not
            the same one with the arm costs turned down.

So the arms map like this, and the mapping is the experiment:

    arm       MJPC                                  CMPC
    stand     Brace {Elbow,Forearm,Palm} = 0        plan_stand.json      (legs_only)
    brace     Brace {Elbow,Forearm} = 300, Palm=0   plan_elbow_forearm.json (elbow+forearm)

EVERYTHING ELSE IS HELD. The stage list, the durations, the replicate counts,
the x ladder, the max-reach target and the disturbance train are all imported
from `brace_vs_stand` rather than restated, because a comparison in which the
two sides drifted apart by a literal is not a comparison. The CSVs this writes
are byte-compatible with testspeed's `--dump_traj`, so
`brace_vs_stand.py analyze`, `bvs_plots.py`, `bvs_strips.py` and
`simple_video.py` all run on them unchanged -- which is the point: the CMPC
figures are the MJPC figures, produced by the same code from a different
controller.

THE PLANTS ARE THE SAME PHYSICS. `Lean_H12_Magpie.xml` (what CMPC plans and
replays against) and `Lean_Simple_H12_Magpie.xml` (what the MJPC task and the
scorer use) are the same rigid body, verified field by field: identical nq/nv/nu,
body/site/geom/actuator names and order, masses, gains, force ranges, friction,
solref/solimp, contact pairs and exclusions. They differ only in MJPC task
metadata -- keyframes, numerics, sensors -- none of which a replayed state
touches. So a CMPC trajectory can be scored by the MJPC scorer and the numbers
mean the same thing.

TWO PROTOCOL DIFFERENCES, BOTH REAL AND BOTH REPORTED:

  1. START POSE. MJPC starts at the `home` keyframe (knees straight); a CMPC
     plan starts at `stand` (knees at 0.55 rad). That is the pose each planner's
     own pipeline has always begun from, and forcing either onto the other's
     would be measuring a controller outside the setup it was tuned in. The
     consequence is bounded: `reach_gain` is scored against a FIXED baseline
     (the scorer's own `home` keyframe, model-side, identical for both), and the
     headline metric -- functional reach, hand to ankle-midpoint -- is
     posture-intrinsic and does not reference either start.
  2. REPLICATES. MJPC's replicates differ because its planner samples. CMPC is
     a deterministic descent, so replicates would be copies and their spread
     would be a lie of zero width. Each replicate therefore starts from a
     SEEDED PERTURBATION of the plan's start pose (`--q-jitter`, default 3 mrad
     per joint and 3 mm of base), which is a robustness probe rather than
     solver noise -- a different thing from MJPC's spread, and labelled as one
     in the artifact (`seed`, `q_jitter_rad`, `base_jitter_m` in the header).

usage:
  cmpc_brace_vs_stand.py cells --out DIR [--stage all]   # certify + solve plans
  cmpc_brace_vs_stand.py run   --out DIR [--stage all]   # fly them, dump CSVs

then, unchanged:
  brace_vs_stand.py analyze --run DIR --json DIR/analysis.json
  brace_vs_stand.py plot    --json DIR/analysis.json --out paper/figures/cmpc_brace_reach
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brace_vs_stand as BVS          # the protocol, imported not restated

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# The nominal target is the MJPC task's own XML default, read from the model so
# a change there cannot silently desynchronise the two harnesses.
NOMINAL_TARGET = None                 # filled by `_nominal_target()`

# arm -> (contact mode, plan tag). `stand` is the cell's legs_only plan; the
# tag is fixed by croco_twin's own convention (solve_tasks.sh writes plan_stand).
ARM_MODE = {"stand": "legs_only", "brace": "elbow+forearm"}
ARM_TAG = {"stand": "stand", "brace": "elbow_forearm"}

# Contact Baumgarte position gain. 50, not the study's long-standing 0: at Kp=0
# a contact constrains velocity only, the brace creeps in every mode and a
# 20 s hold never settles -- which would make the settled metrics measure the
# creep rather than the brace. See solve_tasks.sh for the swept numbers.
CONTACT_KP = 50.0

HORIZON = 35            # nodes; the study's, and not a knob to trade away
ITERS = 1               # one warm-started BoxFDDP iteration per period
THREADS = int(os.environ.get("CMPC_THREADS", "18"))

# The dumped CSV grid. MJPC dumps every 5th 2 ms step = 100 Hz; matched here so
# the two sides' derived quantities (hand speed, SPARC, jitter) are computed
# over the same sample rate.
DUMP_STRIDE = 5

# Replicate perturbation. Small enough that it is not a different maneuver,
# large enough that the replicates are not copies.
# The full lean-reach-recover cycle, for the five-stage filmstrip. Two plans
# flown back to back into ONE trajectory: the braced reach, held, then the
# recovery. The seam is projected (join the recovery tape at the node whose pose
# the robot is nearest) rather than entered at node 0 -- entering at node 0 asks
# a braced robot to drive forward INTO the brace before it may leave it, which
# is what the chaining lurch looked like from outside.
CYCLE_REACH_S = 11.0
CYCLE_RECOVER_S = 9.0
# AT THE EXTENDED TARGET, NOT THE NOMINAL ONE. The figure is a picture of the
# maneuver working, and at the task's default x = 0.9047 the declared brace does
# not survive an 11 s hold (measured: pelvis 0.082 m at the seam, i.e. the
# "recover" column would show a robot getting up off the floor). x = 1.06 is
# inside both planners' envelope and is the condition the panel figures use.
CYCLE_X = None            # None -> brace_vs_stand.NOMINAL2_X

Q_JITTER = 3e-3         # rad, per joint, gaussian
BASE_JITTER = 3e-3      # m, base translation, gaussian


def _nominal_target():
    """The MJPC task's default reach target, read out of its XML."""
    global NOMINAL_TARGET
    if NOMINAL_TARGET is None:
        import mujoco
        import simple_lean as S
        m, _ = S.load()
        n = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, "reach_target")
        NOMINAL_TARGET = tuple(
            float(v) for v in
            m.numeric_data[m.numeric_adr[n]:m.numeric_adr[n] + 3])
    return NOMINAL_TARGET


def cell_name(target):
    return "x%04d" % round(target[0] * 1000)


def stage_targets(stage):
    """Every distinct reach target a stage needs, as {cell_name: target}."""
    t = {}
    nom = _nominal_target()
    if stage in ("nominal", "disturb", "all"):
        t[cell_name(nom)] = nom
    if stage in ("cycle", "all"):
        tgt = (CYCLE_X if CYCLE_X is not None else BVS.NOMINAL2_X,
               nom[1], nom[2])
        t[cell_name(tgt)] = tgt
    if stage in ("nominal2", "all"):
        tgt = (BVS.NOMINAL2_X, nom[1], nom[2])
        t[cell_name(tgt)] = tgt
    if stage in ("sweep", "all"):
        for x in BVS.SWEEP_X:
            tgt = (x, nom[1], nom[2])
            t[cell_name(tgt)] = tgt
    if stage in ("maxreach", "disturb_max", "all"):
        t[cell_name(BVS.MAXREACH_TARGET)] = tuple(BVS.MAXREACH_TARGET)
    if stage in ("disturb2", "all"):
        t[cell_name((BVS.NOMINAL2_X, nom[1], nom[2]))] = \
            (BVS.NOMINAL2_X, nom[1], nom[2])
    return t


# --------------------------------------------------------------------------- #
# cells: certify the two contact modes at a target, then solve their plans
# --------------------------------------------------------------------------- #
def build_cell(cell_dir, target, force=False, log=print):
    """modes.json + q_*.txt for the two modes this study compares.

    Only two subsets are enumerated -- the empty one and {elbow, forearm} --
    rather than croco_modes' power set over SITES5. That is a COST decision, not
    a scientific one: per-subset IK and QP depend on the subset alone, so the q*
    written here for `elbow+forearm` is bit-identical to the one a full
    enumeration would write. What is lost is the RANKING across 32 modes, which
    this harness does not use: the mode is chosen by the experiment, not by the
    ranker.

    AN INADMISSIBLE MODE STILL GETS A POSE. croco_modes writes q_*.txt only for
    modes that pass admissibility (reach < 3 cm and the rest), and at the
    max-reach target -- 1.60 m, past the far edge of the table, chosen precisely
    so nothing can arrive -- nothing passes. But "how far does the posture get
    when the target is out of reach" is the max-reach question, so the
    best-effort IK pose is written anyway, with `admissible: false` kept in the
    manifest and the achieved IK error recorded next to it. A reader of that
    cell can see it never certified.
    """
    import croco_modes as CM
    import contact_select as cs

    os.makedirs(cell_dir, exist_ok=True)
    mpath = os.path.join(cell_dir, "modes.json")
    if os.path.exists(mpath) and not force:
        log("  modes.json cached")
        return json.load(open(mpath))

    subsets = [(), ("elbow", "forearm")]
    recs = CM.enumerate_modes(list(target), sites=("elbow", "forearm"),
                              subsets=subsets, verbose=True)
    manifest = {"target": [float(v) for v in target],
                "stance_dx": cs.STANCE_DX, "stance_dy": cs.STANCE_DY,
                "sites": ["elbow", "forearm"],
                "model": os.path.basename(cs.MODEL),
                "brace_arm": cs.BRACE_ARM, "site_set": cs.SITE_SET,
                "seed_key": cs.SEED_KEY, "tau_basis": cs.TAU_BASIS,
                "pressed": True,
                "harness": "cmpc_brace_vs_stand",
                "subsets_enumerated": [list(s) for s in subsets],
                "modes": []}
    for r in recs:
        name = "+".join(r["subset"]) or "legs_only"
        entry = {k: v for k, v in r.items() if k != "qpos"}
        entry["name"] = name
        f = "q_%s.txt" % name
        np.savetxt(os.path.join(cell_dir, f), r["qpos"])
        entry["qpos_file"] = f
        if not r["admissible"]:
            entry["note"] = ("pose written despite failing admissibility -- "
                             "this is a max-reach cell, the target is out of "
                             "reach on purpose and q* is the best-effort IK "
                             "pose (reach error %.1f mm)" % (1e3 * r["reach"]))
        manifest["modes"].append(entry)
    manifest["ranked"] = [("+".join(r["subset"]) or "legs_only")
                          for r in CM.rank(recs)]
    with open(mpath, "w") as fh:
        json.dump(manifest, fh, indent=1)
    log("  wrote %s" % mpath)
    return manifest


def solve_plans(cell_dir, force=False, log=print, recover=True):
    """plan_elbow_forearm / plan_stand (/ plan_recover_elbow_forearm) in a cell.

    croco_run is invoked as a SUBPROCESS, not imported: it is a script whose
    main() parses argv and mutates module state in contact_select (stance
    offsets), and building six plans in one interpreter would have each solve
    inherit the last one's globals.
    """
    py = sys.executable
    jobs = [
        ("brace+reach", "elbow_forearm",
         ["--mode", "elbow+forearm", "--start", "stand",
          "--n-approach", "120", "--n-braced", "80"]),
        ("stand", "stand",
         ["--mode", "legs_only", "--start", "stand",
          "--n-approach", "120", "--n-braced", "80"]),
    ]
    if recover:
        jobs.append(
            ("recover", "recover_elbow_forearm",
             ["--mode", "elbow+forearm", "--start", "forearm_brace_reach",
              "--start-q", "qstar", "--return-start", "stand",
              "--n-approach", "0", "--n-braced", "20", "--n-return", "120"]))
    out = {}
    for what, tag, extra in jobs:
        path = os.path.join(cell_dir, "plan_%s.json" % tag)
        if os.path.exists(path) and not force:
            log("  %-14s cached" % what)
            out[tag] = path
            continue
        cmd = [py, os.path.join(HERE, "croco_run.py"), "--dir", cell_dir,
               "--tag", tag, "--dt", "0.02",
               "--contact-kp", "%g" % CONTACT_KP,
               "--reach-rot", "auto", "--w-reach-rot", "1e-2"] + extra
        t = time.time()
        with open(os.path.join(cell_dir, "solve_%s.log" % tag), "w") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
        log("  %-14s rc=%d  %.1f s" % (what, rc, time.time() - t))
        if rc == 0:
            out[tag] = path
    return out


def cmd_cells(a):
    targets = stage_targets(a.stage)
    cells = os.path.join(a.out, "cells")
    os.makedirs(cells, exist_ok=True)
    index = {}
    for i, (name, tgt) in enumerate(sorted(targets.items())):
        cell = os.path.join(cells, name)
        print("\n[%d/%d] cell %s  target %.4f %.4f %.4f"
              % (i + 1, len(targets), name, *tgt), flush=True)
        man = build_cell(cell, tgt, force=a.force)
        adm = {m["name"]: m["admissible"] for m in man["modes"]}
        print("  admissible: %s" % adm, flush=True)
        plans = solve_plans(cell, force=a.force, recover=not a.no_recover)
        index[name] = dict(dir=cell, target=list(tgt), admissible=adm,
                           plans=sorted(os.path.basename(p) for p in plans.values()))
    with open(os.path.join(a.out, "cells.json"), "w") as fh:
        json.dump(index, fh, indent=1)
    print("\nwrote %s (%d cells)" % (os.path.join(a.out, "cells.json"),
                                     len(index)))


# --------------------------------------------------------------------------- #
# the plant: MuJoCo, plus a 100 Hz recorder and an unmodelled force
# --------------------------------------------------------------------------- #
def make_recording_plant(tau_lim):
    """A MuJoCoPlant that logs at 100 Hz and can be pushed.

    The recorder lives INSIDE `step` rather than on the control loop's on_step
    hook because the control period is 20 ms and the dump grid is 10 ms: a hook
    can only see the state at period boundaries, which would halve the sample
    rate and make SPARC and the jitter metrics incomparable with MJPC's.

    The disturbance is `xfrc_applied` on the reaching wrist, set and cleared
    around each mj_step. It is applied by the PLANT, so nothing the controller
    reads mentions it -- which is the point of the stage: MJPC's `--disturb`
    is likewise invisible to its planner, so both measure rejection rather than
    tracking.
    """
    from croco.plant.mujoco_plant import MuJoCoPlant
    import contact_select as cs
    import croco_replay as cr
    import mujoco

    class RecordingPlant(MuJoCoPlant):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            self.rows = []
            self._n = 0
            self.stride = DUMP_STRIDE
            self.disturb = None        # (body_id, force3, [(t0,t1), ...])
            self.cost = 0.0            # written by the policy each period
            self.phase = 0.0
            self.dfx = np.zeros(3)

        def reset_log(self):
            self.rows, self._n = [], 0

        def _push(self):
            m, d = self.m, self.d
            self.rows.append(np.concatenate((
                [d.time, self.cost], d.qpos[:m.nq], d.qvel[:m.nv],
                d.ctrl[:m.nu], d.actuator_force[:m.nu],
                [self.phase], self.dfx)))

        def step(self, dt):
            m, d = self.m, self.d
            n = max(1, int(round(dt / m.opt.timestep)))
            with self.data_lock, self._guard():
                for _ in range(n):
                    if self.disturb is not None:
                        bid, f, windows = self.disturb
                        on = any(a <= d.time < b for a, b in windows)
                        self.dfx = np.asarray(f, float) if on else np.zeros(3)
                        d.xfrc_applied[bid, :3] = self.dfx
                    if self._n % self.stride == 0:
                        self._push()
                    mujoco.mj_step(m, d)
                    self._n += 1

    m2, d2 = cs.load(ik_margin=0.0)
    cr.show_gripper(m2)          # visual only; the jaws still have no dynamics
    mujoco.mj_forward(m2, d2)
    return RecordingPlant(m2, d2, sense=None, tau_limit=tau_lim, nu=27)


# --------------------------------------------------------------------------- #
# the episode
# --------------------------------------------------------------------------- #
class Flight:
    """One (cell, tag) flown repeatedly. The OCP is built once and reused.

    Building the OCP is 1.2-25 s and there are up to eight episodes per plan, so
    a per-episode build would be most of the wall clock. `MPC.reset()` is what
    makes reuse correct: the sliding window only ever moves forward, so an MPC
    that has reached the end of a plan and is handed k=0 again keeps solving the
    LAST window against a robot standing at the first pose -- measured elsewhere
    in this study as the robot flat on the floor with a solve time that had
    DROPPED, because the problem no longer had anything to do with the state.
    """

    def __init__(self, cell, tag, horizon=HORIZON, iters=ITERS,
                 threads=THREADS):
        import croco_twin as CT
        import croco_replay as cr
        import contact_select as cs

        class _A:                       # croco_twin.build reads attributes
            pass
        a = _A()
        a.dir, a.tag = cell, tag
        a.horizon, a.iters, a.alphas, a.threads = horizon, iters, 0, threads
        a.reach_rot, a.w_reach_rot = None, None
        a.contact_kp, a.contact_kd, a.foot_kp = None, None, None
        self.plan, self.mpc, self.xs, self.us = CT.build(a)
        self.horizon = horizon
        self.dt = self.plan["dt"]
        self.m, _ = cs.load(ik_margin=0.0)
        self.kp, self.kd = cr.servo_gains(self.m)
        self.tau_lim = cs.torque_limits(self.m)
        self.q0 = None                  # set in `start_qpos`

    def start_qpos(self):
        import croco_bridge as cb
        import contact_select as cs
        return cb.pin_to_mj(self.xs[0][:cb.NQ_ROBOT],
                            cs.start_qpos(self.m, self.plan["start"]))


def run_episode(flight, plant, seconds, seed=0, disturb=None,
                q_jitter=Q_JITTER, base_jitter=BASE_JITTER,
                resume=False, k0=0):
    """Fly one episode and return (rows, summary).

    The plan is 4 s and the stages are 16-25 s, so every episode HOLDS: past the
    end of the plan the index is clamped rather than the policy returning None,
    and the MPC keeps re-solving the plan's own final window against the live
    state -- which is what a brace that is still braced looks like to this
    controller. It is croco_twin's `hold` submode with one correction, and the
    correction is not cosmetic: see `k_hold` below.
    """
    import croco_bridge as cb
    import croco_twin as CT
    from croco.runtime.loop import ControlLoop, LoopConfig
    import mujoco

    nq = cb.NQ_ROBOT
    if not resume:
        q0 = flight.start_qpos().copy()
        rng = np.random.default_rng(seed)
        if seed:
            q0[7:34] += rng.normal(0.0, q_jitter, 27)
            q0[0:3] += rng.normal(0.0, base_jitter, 3)
        plant.d.qpos[:] = q0
        plant.d.qvel[:] = 0.0
        plant.d.ctrl[:] = q0[7:34]
        plant.d.xfrc_applied[:] = 0.0
        # THE SIM CLOCK RESTARTS WITH THE EPISODE. mjData.time is not touched
        # by writing qpos, so a second episode in the same process starts at
        # t = 20 s -- and everything downstream reads absolute time: the
        # disturbance train fires at t0 = 10 s and would never arrive, and the
        # scorer's settle window (`t >= 0.6 * t[-1]`) would cover three
        # quarters of the run instead of the last two fifths. Both failures are
        # silent and both corrupt only the episodes after the first.
        plant.d.time = 0.0
        mujoco.mj_forward(plant.m, plant.d)
        plant.reset_log()
    # A CHAINED LEG (`resume`) DOES NOT TOUCH THE PLANT. The robot is braced
    # where the last behaviour left it, and putting it back at the new plan's
    # start pose would teleport it out of its own contact -- the one thing a
    # chain exists to avoid. The log keeps accumulating too, so a cycle comes
    # out as one trajectory rather than two files stitched at a cut.
    plant.disturb = disturb
    plant.dfx = np.zeros(3)
    flight.mpc.reset()

    us, xs, dt_plan = flight.us, flight.xs, flight.dt
    # HOLD AT THE LAST FULL WINDOW, NOT AT THE LAST NODE. `MPC.__call__` slides
    # forward only while `k + H <= len(models)`; past that it takes the tail
    # branch and REBUILDS a shrinking-horizon problem from `models[k:]`. Clamped
    # to len(us)-1 that is a ONE-NODE horizon, and the controller is no longer
    # doing MPC -- measured here as the mean solve collapsing 19 ms -> 2.5 ms
    # and the robot on the floor at t = 15 s of a 20 s hold, in every mode. The
    # 2.5 ms is the diagnostic: a solve that got seven times cheaper did not get
    # faster, it got smaller. Clamping to len(us) - H instead keeps the window
    # full, so the hold is the plan's own final window re-solved against the
    # live state, which is what a brace that is still braced looks like.
    k_hold = max(0, len(us) - flight.horizon)
    stats = dict(steps=0, mpc_none=0)

    def policy(t, st):
        k = min(k0 + int(round(t / dt_plan)), k_hold)
        qpos = np.concatenate([st.base_pos, st.base_quat, st.q])
        R = CT._quat_to_mat(st.base_quat)
        qvel = np.concatenate([st.base_linvel, st.base_angvel, st.v])
        x_meas = np.concatenate([cb.mj_to_pin(qpos), cb.mj_to_pin_v(qvel, R)])
        u0, xs1 = flight.mpc(k, x_meas)
        stats["steps"] += 1
        plant.phase = float(k)
        if u0 is None:
            stats["mpc_none"] += 1
            return xs[k][7:nq], np.zeros(27), us[k]
        try:
            plant.cost = float(flight.mpc.solver.cost)
        except Exception:                                        # noqa: BLE001
            pass
        return (xs1[:nq][7:], xs1[nq:][6:],
                np.clip(u0, -flight.tau_lim, flight.tau_lim))

    cfg = LoopConfig(ctrl_hz=1.0 / dt_plan, stale_s=0.05, realtime=None)
    loop = ControlLoop(plant, policy, stance=None, cfg=cfg)
    t0 = time.monotonic()
    log = loop.run(flight.kp, flight.kd, max_seconds=seconds)
    solves = [r["solve_ms"] for r in log if "solve_ms" in r]
    return np.asarray(plant.rows), dict(
        wall_s=time.monotonic() - t0, periods=len(log),
        mpc_steps=stats["steps"], mpc_none=stats["mpc_none"],
        overruns=loop.overruns, watchdog_trips=loop.watchdog_trips,
        tau_saturated=sum(r.get("tau_sat", 0) for r in log),
        q_clipped=sum(r.get("q_clip", 0) for r in log),
        solve_ms_mean=float(np.mean(solves)) if solves else None,
        solve_ms_p95=float(np.percentile(solves, 95)) if solves else None,
        pelvis_z=float(plant.d.qpos[2]), seed=seed)


def _seam_k0(flight, plant, horizon=HORIZON):
    """Which plan node to join a chained behaviour at.

    croco_twin's `--seam project`, reproduced: the node whose pose the measured
    state is nearest, under joint L2 in radians plus base translation at
    3 rad/m. Capped a horizon short of the end so the projection can skip the
    part of the behaviour already done but never the behaviour itself --
    uncapped it lands on the last node of a plan whose end pose the robot is
    already in, and the episode runs zero periods.
    """
    xs = np.asarray(flight.xs, float)
    qj, base = xs[:, 7:34], xs[:, 0:3]
    q, bp = plant.d.qpos[7:34], plant.d.qpos[0:3]
    d = np.linalg.norm(qj - q, axis=1) + 3.0 * np.linalg.norm(base - bp, axis=1)
    cap = max(0, len(flight.us) - 1 - horizon)
    k_raw = int(np.argmin(d))
    return int(np.clip(k_raw, 0, cap)), float(d.min()), k_raw


def run_cycle(reach, recover, plant, seed=0, t_reach=CYCLE_REACH_S,
              t_recover=CYCLE_RECOVER_S):
    """Brace+reach, held, then recover -- one continuous trajectory.

    The paper's Fig. 1 is five stages of the hardware maneuver. This is its
    simulated CMPC counterpart, and it has to be ONE trajectory rather than two
    concatenated CSVs: the interesting instant is the seam, where a robot that
    is braced has to let go, and two runs stitched at the file level would show
    that transition as a cut rather than as physics.
    """
    rows_a, sa = run_episode(reach, plant, t_reach, seed=seed)
    n_a = len(rows_a)
    k0, dist, k_raw = _seam_k0(recover, plant)
    # The recorder is NOT reset for the chained leg (that is what makes the
    # cycle one trajectory), so its buffer already holds leg A. Stacking the
    # two returns would write leg A twice.
    rows, sb = run_episode(recover, plant, t_recover, resume=True, k0=k0)
    return rows, dict(
        reach=sa, recover=sb,
        seam=dict(k0=k0, k0_dist=dist, k0_raw=k_raw, n_reach_rows=n_a,
                  t_seam=float(rows_a[-1][0])))


def write_csv(path, rows, flight, target, arm, seed, seconds, disturb_str):
    """testspeed's `--dump_traj` format, exactly.

    The header lines are not decoration: `simple_lean.load_traj` parses them and
    `target_from_meta` reads the reach target OUT OF THEM. A CSV whose numerics
    line is missing is scored against the XML default, which for every swept
    target is the wrong target and reports a run that landed 4 mm from its own
    goal as a third of a metre of error.
    """
    m = flight.m
    with open(path, "w") as f:
        f.write("# task=CMPC BoxFDDP %s nq=%d nv=%d nu=%d start_key=%s "
                "dt=%g\n" % (flight.plan.get("mode") or flight.plan.get("subset"),
                             m.nq, m.nv, m.nu, flight.plan["start"],
                             m.opt.timestep))
        f.write("# weights=arm=%s,mode=%s,tag=%s,contact_kp=%g,horizon=%d,"
                "iters=%d,seed=%d,q_jitter_rad=%g,base_jitter_m=%g\n"
                % (arm, flight.plan.get("mode"), flight.plan.get("tag", ""),
                   CONTACT_KP, HORIZON, ITERS, seed, Q_JITTER, BASE_JITTER))
        f.write("# numerics=reach_target=%g|%g|%g%s\n"
                % (target[0], target[1], target[2],
                   ("," + disturb_str) if disturb_str else ""))
        hdr = ["time", "cost"]
        hdr += ["qpos%d" % i for i in range(m.nq)]
        hdr += ["qvel%d" % i for i in range(m.nv)]
        hdr += ["ctrl%d" % i for i in range(m.nu)]
        hdr += ["afrc%d" % i for i in range(m.nu)]
        hdr += ["phase", "dfx", "dfy", "dfz"]
        f.write(",".join(hdr) + "\n")
        np.savetxt(f, rows, delimiter=",", fmt="%.17g")


def jobs_for(stage, cells_index):
    """(tag, arm, cell, seconds, target, disturb) per run -- BVS's stage table.

    The stage list, durations, replicate counts and the disturbance train come
    from brace_vs_stand so the two harnesses cannot drift apart; only the arm ->
    plan mapping is this file's.
    """
    nom = _nominal_target()
    out = []

    def cell_of(target):
        return cells_index[cell_name(target)]["dir"]

    if stage in ("nominal", "all"):
        for arm in ("stand", "brace"):
            for r in range(BVS.NOMINAL_REPS):
                out.append(("nominal_%s_r%d" % (arm, r), arm, cell_of(nom),
                            BVS.NOMINAL_SECONDS, nom, None))
    if stage in ("nominal2", "all"):
        tgt = (BVS.NOMINAL2_X, nom[1], nom[2])
        for arm in ("stand", "brace"):
            for r in range(BVS.NOMINAL2_REPS):
                out.append(("nominal2_%s_r%d" % (arm, r), arm, cell_of(tgt),
                            BVS.NOMINAL2_SECONDS, tgt, None))
    if stage in ("sweep", "all"):
        for x in BVS.SWEEP_X:
            tgt = (x, nom[1], nom[2])
            for arm in ("stand", "brace"):
                for r in range(BVS.SWEEP_REPS):
                    out.append(("sweep_x%03d_%s_r%d" % (round(x * 100), arm, r),
                                arm, cell_of(tgt), BVS.SWEEP_SECONDS, tgt, None))
    if stage in ("maxreach", "all"):
        tgt = tuple(BVS.MAXREACH_TARGET)
        for arm in ("stand", "brace"):
            for r in range(BVS.MAXREACH_REPS):
                out.append(("maxreach_%s_r%d" % (arm, r), arm, cell_of(tgt),
                            BVS.MAXREACH_SECONDS, tgt, None))
    if stage in ("disturb", "all"):
        for fN in BVS.DISTURB_N:
            for arm in ("stand", "brace"):
                for r in range(BVS.DISTURB_REPS):
                    out.append(("disturb_f%03d_%s_r%d" % (round(fN), arm, r),
                                arm, cell_of(nom), BVS.DISTURB_SECONDS, nom,
                                fN))
    if stage in ("disturb2", "all"):
        tgt = (BVS.NOMINAL2_X, nom[1], nom[2])
        for fN in BVS.DISTURB_N:
            for arm in ("stand", "brace"):
                for r in range(BVS.DISTURB_REPS):
                    out.append(("disturb2_f%03d_%s_r%d" % (round(fN), arm, r),
                                arm, cell_of(tgt), BVS.DISTURB_SECONDS, tgt,
                                fN))
    if stage in ("disturb_max",):
        tgt = tuple(BVS.MAXREACH_TARGET)
        for fN in BVS.DISTURB_N:
            for arm in ("stand", "brace"):
                for r in range(BVS.DISTURB_REPS):
                    out.append(("disturbmax_f%03d_%s_r%d"
                                % (round(fN), arm, r), arm, cell_of(tgt),
                                BVS.DISTURB_SECONDS, tgt, fN))
    return out


def disturb_spec(plant, fN):
    """The MJPC pulse train, on the same body, at the same times."""
    import mujoco
    if not fN:
        return None, ""
    bid = mujoco.mj_name2id(plant.m, mujoco.mjtObj.mjOBJ_BODY, BVS.HAND_BODY)
    wins = [(BVS.DISTURB_T0 + i * BVS.DISTURB_PERIOD,
             BVS.DISTURB_T0 + i * BVS.DISTURB_PERIOD + BVS.DISTURB_DUR)
            for i in range(BVS.DISTURB_COUNT)]
    spec = ("disturb=body:%s|force:%g|0|0|t0:%g|dur:%g|period:%g|n:%d"
            % (BVS.HAND_BODY, fN, BVS.DISTURB_T0, BVS.DISTURB_DUR,
               BVS.DISTURB_PERIOD, BVS.DISTURB_COUNT))
    return (bid, (fN, 0.0, 0.0), wins), spec


def cmd_run_cycle(a, cells_index, plant):
    """The five-stage figure's trajectory: stand -> lean -> reach -> recover.

    Written as `cycle_r<seed>.csv`, which `brace_vs_stand.py analyze` will
    happily score and which nothing in `bvs_plots` looks for -- the stage prefix
    `cycle` matches no figure. That is deliberate: this trajectory is a PICTURE,
    not a measurement, and pooling a run that deliberately leaves its brace into
    the settled-hold statistics would corrupt every one of them.
    """
    nom = _nominal_target()
    tgt = (CYCLE_X if CYCLE_X is not None else BVS.NOMINAL2_X, nom[1], nom[2])
    cell = cells_index[cell_name(tgt)]["dir"]
    reach = Flight(cell, "elbow_forearm")
    recover = Flight(cell, "recover_elbow_forearm")
    out = []
    for seed in range(2):
        path = os.path.join(a.out, "cycle_brace_r%d.csv" % seed)
        if os.path.exists(path) and not a.force:
            print("cycle_brace_r%d cached" % seed, flush=True)
            continue
        rows, summ = run_cycle(reach, recover, plant, seed=seed)
        write_csv(path, rows, reach, tgt, "brace", seed,
                  CYCLE_REACH_S + CYCLE_RECOVER_S, "")
        print("cycle_brace_r%d  %d rows  seam k0=%d (dist %.3f, raw %d) at "
              "t=%.2f s  pelvis %.3f"
              % (seed, len(rows), summ["seam"]["k0"], summ["seam"]["k0_dist"],
                 summ["seam"]["k0_raw"], summ["seam"]["t_seam"],
                 summ["recover"]["pelvis_z"]), flush=True)
        out.append(dict(tag="cycle_brace_r%d" % seed, **summ))
    if out:
        with open(os.path.join(a.out, "manifest_cycle.json"), "w") as fh:
            json.dump(out, fh, indent=1)


def cmd_run(a):
    cells_index = json.load(open(os.path.join(a.out, "cells.json")))
    jobs = jobs_for(a.stage, cells_index)
    print("%d episodes, ~%.0f s of sim" % (len(jobs), sum(j[3] for j in jobs)),
          flush=True)
    import contact_select as cs
    tau_lim = cs.torque_limits(cs.load(ik_margin=0.0)[0])
    plant = make_recording_plant(tau_lim)
    if a.stage in ("cycle", "all"):
        cmd_run_cycle(a, cells_index, plant)
        if a.stage == "cycle":
            return
    flights, manifest = {}, []
    for i, (tag, arm, cell, secs, target, fN) in enumerate(jobs):
        csv_path = os.path.join(a.out, tag + ".csv")
        if os.path.exists(csv_path) and not a.force:
            print("[%2d/%d] %-28s cached" % (i + 1, len(jobs), tag), flush=True)
            continue
        key = (cell, ARM_TAG[arm])
        # BOUNDED CACHE. Every Flight holds ~200 crocoddyl action models and
        # their datas; the sweep visits nine cells and would end up holding
        # eighteen of them at once. The job order groups both arms of a cell
        # together, so three is enough to never rebuild inside a cell, and a
        # rebuild across cells is 1-20 s against an out-of-memory kill.
        while len(flights) >= 3 and key not in flights:
            flights.pop(next(iter(flights)))
        if key not in flights:
            t = time.time()
            print("       building OCP %s / %s ..."
                  % (os.path.basename(cell), ARM_TAG[arm]), end="", flush=True)
            try:
                flights[key] = Flight(cell, ARM_TAG[arm])
            except Exception as exc:                             # noqa: BLE001
                print(" FAILED: %s" % exc, flush=True)
                flights[key] = None
            else:
                print(" %.1f s" % (time.time() - t), flush=True)
        fl = flights[key]
        if fl is None:
            continue
        seed = int(tag.rsplit("_r", 1)[1]) if "_r" in tag else 0
        dspec, dstr = disturb_spec(plant, fN)
        rows, summ = run_episode(fl, plant, secs, seed=seed, disturb=dspec)
        write_csv(csv_path, rows, fl, target, arm, seed, secs, dstr)
        print("[%2d/%d] %-28s %5d rows  %.0f s wall  pelvis %.3f  "
              "solve %.1f/%.1f ms  overrun %d"
              % (i + 1, len(jobs), tag, len(rows), summ["wall_s"],
                 summ["pelvis_z"], summ["solve_ms_mean"] or -1,
                 summ["solve_ms_p95"] or -1, summ["overruns"]), flush=True)
        manifest.append(dict(tag=tag, arm=arm, cell=cell, seconds=secs,
                             target=list(target), disturb_N=fN, **summ))
    mpath = os.path.join(a.out, "manifest_%s.json" % a.stage)
    if manifest:
        with open(mpath, "w") as fh:
            json.dump(manifest, fh, indent=1)
        print("wrote %s" % mpath)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    STAGES = ["nominal", "nominal2", "sweep", "maxreach", "disturb",
              "disturb2", "disturb_max", "cycle", "all"]
    c = sub.add_parser("cells")
    c.add_argument("--out", required=True)
    c.add_argument("--stage", default="all", choices=STAGES)
    c.add_argument("--force", action="store_true")
    c.add_argument("--no-recover", action="store_true",
                   help="skip the recovery plan (it is only needed for the "
                        "full lean-reach-recover filmstrip)")
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--stage", default="all", choices=STAGES)
    r.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    (cmd_cells if a.cmd == "cells" else cmd_run)(a)


if __name__ == "__main__":
    main()
