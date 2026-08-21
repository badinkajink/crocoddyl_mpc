#!/usr/bin/env python3
"""The braced lean, run against a plant on the far side of Unitree DDS.

This is `croco_replay.py --ctrl mpc` with ONE thing changed: the plant. Same
plan, same OCP, same MPC, same gains. What is different is everything the
in-process replay could not be wrong about --

    the controller no longer owns the clock          (DDSPlant.OWNS_CLOCK=False)
    the state is a MESSAGE and can be old            (State.age, the watchdog)
    the command is a MESSAGE and can be dropped      (lowcmd timeout -> collapse)
    the 20 ms period is real, and a 12 ms solve eats most of it

-- which is the whole point. A replay proves the plan survives the physics; this
proves the controller survives the deployment. They are different claims and the
first has never implied the second.

WHAT IT DOES NOT PROVE. The plant here is `lean_twin`, i.e. the SAME MJCF the
plan was solved against, served over the wire. Scene parity with
`h1_robocasa`/`h1_mujoco` (which carry a kitchen, not the lean table) is the
next problem and is deliberately not mixed into this one: a failure here is a
deployment failure and cannot be blamed on the scene.

BASE POSE. The OCP needs one and `rt/lowstate` does not carry one -- no robot
has it. For this stage it is read from the twin's `--publish-truth` channel,
which is GROUND TRUTH and is why `--base truth` has to be typed. The real
estimator (h12_deploy_mjpc's estimator_node, FAST-LIO, the tag anchor) plugs
into the same `base_source` hook with nothing else changing, and the gap between
those two numbers is the next thing worth measuring.

usage:
  # terminal 1
  python -m croco.twin.lean_twin --model $LEAN_TASK_DIR/Lean_H12_Magpie.xml \
      --key stand --publish-truth
  # terminal 2
  studies/croco_twin.py --dir runs/.../grid/<cell> --tag elbow_palm --base truth
"""
import argparse
import contextlib
import ctypes
import faulthandler
import glob
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "croco_ext"))
sys.path.insert(0, os.path.join(HERE, ".."))

# BEFORE ANYTHING NATIVE IS LOADED. `ensure_runtime` may replace this process
# with the pinned interpreter, and everything imported above that point is
# work thrown away -- but more importantly, running the wrong crocoddyl to the
# point of building a ShootingProblem is a SIGSEGV with no traceback, which is
# precisely the failure it exists to convert into a sentence. See croco/env.py.
from croco.env import ensure_runtime                             # noqa: E402
ensure_runtime()

sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

import numpy as np                                              # noqa: E402

import croco_bridge as cb                                       # noqa: E402
import contact_select as cs                                     # noqa: E402
import croco_replay as cr                                       # noqa: E402
import croco_plan as cp                                         # noqa: E402

from croco.control.mpc import MPC                               # noqa: E402
from croco.plant.dds_plant import (DDSPlant, PollingReceiver,   # noqa: E402
                                   assert_joint_order)
from croco.runtime.loop import ControlLoop, LoopConfig          # noqa: E402
from croco.plant.dds_plant import TOPIC_SAFETY_LOWCMD_IN        # noqa: E402


# --------------------------------------------------------------- base --- #
class TruthBase:
    """Subscribe to the twin's ground-truth base pose (`rt/sim_state`).

    NAMED `truth` ON THE COMMAND LINE ON PURPOSE. This is the one privileged
    input in the loop, it exists so the deployment plumbing can be tested with
    the estimator held at perfect, and every result taken with it has to say so.
    Swapping in a real estimator means replacing this class and nothing else.

    POLLED, for the reason in dds_plant.py: a Python callback cannot run while
    crocoddyl holds the GIL. This channel was the worse of the two offenders --
    it carries JSON, so the callback path spent a `json.loads` per sample at the
    twin's 500 Hz, all of it contending for the same GIL the solver is sitting
    on. Polling parses ONE document per control period, and parses the newest.
    """

    def __init__(self, recv="poll"):
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
        self._v = None
        self._lock = threading.Lock()
        self._recv = None
        if recv == "poll":
            self._recv = PollingReceiver("rt/sim_state", String_)
        else:
            from unitree_sdk2py.core.channel import ChannelSubscriber
            self._sub = ChannelSubscriber("rt/sim_state", String_)
            self._sub.Init(self._on, 10)

    def _on(self, msg):
        self._decode(msg)

    def _decode(self, msg):
        try:
            d = json.loads(msg.data)
        except Exception:                                       # noqa: BLE001
            return
        with self._lock:
            self._v = (np.array(d["base_pos"]), np.array(d["base_quat"]),
                       np.array(d["base_linvel"]), np.array(d["base_angvel"]),
                       time.monotonic())

    def __call__(self):
        msg = self._recv.latest() if self._recv is not None else None
        if msg is not None:
            self._decode(msg)
        with self._lock:
            v = self._v
        if v is None:
            return None
        p, q, lv, av, stamp = v
        return p, q, lv, av, max(0.0, time.monotonic() - stamp)

    def wait(self, timeout=5.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self() is not None:
                return True
            time.sleep(0.01)
        raise TimeoutError(
            "no rt/sim_state in %.1f s -- start lean_twin with --publish-truth, "
            "or the OCP has no base pose to plan from." % timeout)


class EstimatorBase:
    """The fork's proprioceptive base estimator, as the loop's base source.

    THIS IS THE ONE THAT COUNTS. `TruthBase` exists so the plumbing could be
    tested with the estimator held at perfect; this is the estimator. It is
    `mjpc/deploy/helper_scripts/base_estimator_node_v4.py` -- rw-ekf leg
    odometry over `rt/lowstate` and NOTHING ELSE, run as a separate process
    exactly as it runs on the robot, publishing `SportModeState_`. No ground
    truth, no motion capture, no privileged topic. Base linear velocity is
    never measured on a legged robot; the factory `rt/sportmodestate.velocity`
    is itself an estimator output, and this is the same class of quantity with
    its sources written down.

    IT DOES NOT PUBLISH ATTITUDE, and should not: `position` and `velocity` are
    the two things proprioception has to reconstruct, while orientation and body
    rate are measured by the IMU and arrive on `rt/lowstate` already. Returning
    None for those lets `DDSPlant` keep the measured ones.

    THE OFFSET IS NOT COSMETIC. The estimator publishes the IMU SITE, because
    that is the convention `h12_control_node.cc` consumes; the OCP wants the
    PELVIS, which is the MuJoCo free joint. They differ by 0.278 m in z, so
    skipping the inversion puts the robot a foot above where it is and the
    plan's CoM barrier reasons about a different robot. The constant is
    duplicated from the estimator rather than imported because importing it
    would drag in the estimator's whole module (and its MuJoCo scene load) into
    the controller process; it is asserted against the estimator's value in the
    docstring above and must be changed in both places or in neither.
    """

    IMU_OFFSET = np.array([-0.04452, -0.01891, 0.27756])   # pelvis -> IMU site

    def __init__(self, topic="rt/sportmodestate_est", recv="poll"):
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
        self.topic = topic
        self._v = None
        self._lock = threading.Lock()
        self._recv = None
        if recv == "poll":
            self._recv = PollingReceiver(topic, SportModeState_)
        else:
            from unitree_sdk2py.core.channel import ChannelSubscriber
            self._sub = ChannelSubscriber(topic, SportModeState_)
            self._sub.Init(self._decode, 10)

    def _decode(self, msg):
        with self._lock:
            self._v = (np.array(list(msg.position), float),
                       np.array(list(msg.velocity), float),
                       time.monotonic())

    def attach(self, plant):
        """Bind to the plant, whose IMU supplies the attitude this cannot."""
        self._plant = plant
        return self

    def __call__(self):
        msg = self._recv.latest() if self._recv is not None else None
        if msg is not None:
            self._decode(msg)
        with self._lock:
            v = self._v
        if v is None:
            return None
        site_p, site_v, stamp = v
        # Site -> pelvis, using the attitude the plant just read off the IMU.
        quat = self._plant._imu_quat
        R = _quat_to_mat(quat)
        roff = R @ self.IMU_OFFSET
        base_p = site_p - roff
        base_v = site_v - np.cross(R @ self._plant._imu_gyro, roff)
        return base_p, None, base_v, None, max(0.0, time.monotonic() - stamp)

    def wait(self, timeout=15.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self() is not None:
                return True
            time.sleep(0.01)
        raise TimeoutError(
            "no %s in %.1f s -- start base_estimator_node_v4.py against the "
            "same domain, with --out-topic %s. It needs nothing but "
            "rt/lowstate." % (self.topic, timeout, self.topic))


# ---------------------------------------------------------------- run --- #
def ocp_overrides(args):
    """Plan fields the CLI replaces before the OCP is rebuilt, or {}.

    Deliberately narrow. This is not a general "retune the plan from argv"
    hatch -- the sliders already retune weights on a built model, live and for
    free. It exists for the one thing sliders cannot do: make a cost EXIST.
    `reachRot` is absent from every certified cell because those plans were
    solved with w_reach_rot = 0 and `_reach_orientation` returns early, so
    the panel has no slider and the roll knob is dead. Adding it needs a
    rebuild, and a rebuild needs a flag.
    """
    ov = {}
    if args.reach_rot is not None:
        ov["reach_rot"] = None if args.reach_rot == "none" else args.reach_rot
        ov["w_reach_rot"] = (0.0 if args.reach_rot == "none"
                             else (1e-2 if args.w_reach_rot is None
                                   else args.w_reach_rot))
    elif args.w_reach_rot is not None:
        ov["w_reach_rot"] = args.w_reach_rot
    # CONTACT STABILISATION. Unlike the weights above this changes the
    # DYNAMICS, not a cost: `gains` is a read-only property on a built
    # ContactModel3D (checked -- there is no setter), so a Kp change cannot be
    # a slider and has to be a rebuild. The rebuild is 1.2 s for 200 models,
    # which is why the panel can still offer it as a control.
    if args.contact_kp is not None:
        ov["contact_kp"] = args.contact_kp
    if args.contact_kd is not None:
        ov["contact_kd"] = args.contact_kd
    if args.foot_kp is not None:
        ov["foot_kp"] = args.foot_kp
    return ov


def build(args):
    """The MPC and the reference plan, exactly as croco_replay builds them."""
    plan = json.load(open(os.path.join(args.dir, "plan_%s.json" % args.tag)))
    ocp, _ = cr.build_ocp(plan, args.dir, overrides=ocp_overrides(args))
    problem = ocp.build(dt=plan["dt"], n_approach=plan["n_approach"],
                        n_braced=plan["n_braced"],
                        n_return=plan.get("n_return", 0),
                        dwell=plan.get("dwell", 0), cones=plan["cones"])
    xs = np.load(os.path.join(args.dir, "xs_%s.npy" % args.tag))
    us = np.load(os.path.join(args.dir, "us_%s.npy" % args.tag))
    mpc = MPC(ocp, list(problem.runningModels), problem.terminalModel,
              horizon=args.horizon, iters=args.iters, xs_plan=xs, us_plan=us,
              n_alphas=args.alphas, nthreads=args.threads)
    return plan, mpc, xs, us


# --------------------------------------------------------------- video --- #
def render_run(qtrace, plan, run_dir, path, cam="wide", fps=30, dt_plan=0.02,
               width=960, height=540):
    """Render buffered states to an mp4, AFTER the loop has finished.

    OFF A FRESH MODEL, not the plant's. `show_gripper` mutates visual
    attributes, and doing that to a model while it is being stepped changes
    what the run looks like without changing what it did -- a distinction that
    stops being harmless the moment someone reads the video as evidence.

    The markers are croco_replay's, and mean the same things: the certified
    contact sites as ghosts, the reach target, and the LIVE table contacts
    coloured by which link is making them. A video that draws only the planned
    contacts cannot show the failure this study spends most of its time
    chasing -- a link touching the table while the plan says it is not.
    """
    import mujoco
    m, d = cs.load(ik_margin=0.0)
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, width)
    m.vis.global_.offheight = max(m.vis.global_.offheight, height)
    cr.show_gripper(m)
    renderer = mujoco.Renderer(m, height, width, max_geom=m.ngeom + 256)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    preset = cr.CAMERAS[cam]
    camera.lookat[:] = preset["lookat"]
    camera.distance = preset["distance"]
    camera.azimuth = preset["azimuth"]
    camera.elevation = preset["elevation"]

    tbl = cs.bid(m, "table")
    site_bodies = {s: cs.bid(m, cs.SITES[s][0]) for s in cr.SITE_RGBA
                   if s in cs.SITES}
    feet = [cs.bid(m, f) for f in cs.FEET]
    target = np.array(plan["target"])
    try:
        site_ref = cr.certified_sites(plan, run_dir, m, d)
    except Exception:                                            # noqa: BLE001
        site_ref = {}          # a missing artifact costs the ghosts, not the video

    every = max(1, int(round(1.0 / (fps * dt_plan))))
    frames = []
    for k in range(0, len(qtrace), every):
        d.qpos[:] = qtrace[k]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)          # also collides: draw_contacts needs it
        renderer.update_scene(d, camera=camera)
        cr.draw_refs(renderer.scene, target, site_ref)
        cr.draw_contacts(renderer.scene, m, d, site_bodies, tbl, feet)
        frames.append(renderer.render())
    renderer.close()
    if not frames:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    import imageio.v2 as imageio
    imageio.mimsave(path, frames, fps=fps, quality=8, macro_block_size=1)
    return path


# --------------------------------------------------------------- viewer --- #
def _windowed_gl():
    """Can this process open a WINDOW, as opposed to render offscreen?

    MUJOCO_GL=egl/osmesa are offscreen backends: the video renders fine, a
    viewer cannot exist. run_session.sh exports egl, so a shell that sourced
    it inherits an environment where the viewer button must be disabled rather
    than offered and then failing.
    """
    return os.environ.get("MUJOCO_GL", "").lower() not in ("egl", "osmesa")


def _viewer_thread_of(mjv):
    """The thread `launch_passive` just started, found by its TARGET.

    WHY NOT A `threading.enumerate()` DIFF, which is what this used to do.
    The diff assumes that between the two snapshots the only thread that
    appeared is the viewer's. That holds in the `--viewer` CLI path, where
    launch happens on the main thread before anything else is running. It does
    NOT hold for the panel's viewer button: that arrives on a WebSocket socket
    thread, and the server creates and retires threads as browsers connect and
    reconnect, so the diff can just as easily hand back a socket thread.

    The consequence is not a wrong log line. `close_viewer` joins whatever it
    was given, so a mis-capture means the REAL render thread is never joined
    -- and mujoco keeps it daemonic, so the interpreter then exits while C++
    is still tearing down a GL context. Measured in isolation: unjoined, that
    is a core dump in 2 of 3 runs; joined, 8 of 8 clean. It is the one crash
    in this area that reproduces on demand.

    `launch_passive` builds an unnamed `threading.Thread(target=
    _launch_internal, ...)`, so the target is the identity. Fall back to a
    daemon-thread guess only if mujoco renames it.
    """
    target = getattr(mjv, "_launch_internal", None)
    if target is not None:
        for t in threading.enumerate():
            if getattr(t, "_target", None) is target:
                return t
    return None


# ---------------------------------------------------------------- tasks --- #
# WHAT A "TASK" IS HERE, AND WHY IT IS NOT A SET OF WEIGHTS. The panel's
# sliders mutate `costs[name].weight` on a BUILT model, which is why retuning
# is live and free. A task is not that. Each phase of the maneuver builds a
# different `DifferentialActionModelContactFwdDynamics` -- `_contacts(braced)`
# and `_geometry(feet_only=...)` add and remove CONTACT CONSTRAINTS per node
# (croco_plan.py, the approach/impact/braced/return table at the top of that
# file). No weight can add a contact. So switching task means rebuilding the
# OCP, and rebuilding it means having a solved reference to warm-start from:
# `MPC.__call__` seeds its first solve from xs_plan/us_plan precisely because
# "one DDP iteration from a constant guess is not an MPC, it is noise".
#
# Hence a task is (plan json + xs + us) sitting in the cell directory, and a
# task with no artifacts is OFFERED BUT DISABLED rather than hidden -- a
# dropdown that silently omits `recover` looks like the feature was never
# built, when in fact nobody has ever solved a plan with `n_return > 0`.
_nullcontext = contextlib.nullcontext

# TASK AND MODE ARE TWO DIFFERENT AXES, and collapsing them into one dropdown
# is what made the study run `elbow+palm` for weeks without once comparing it.
#
#   KIND   what the robot is doing:  approach and brace / stand / come back off
#   MODE   which links are on the table: elbow, elbow+forearm, elbow+palm, ...
#
# Both are properties of a SOLVED PLAN -- the phases differ by contact set, so
# no weight can turn one mode into another -- but they are independent choices,
# and a UI that offers only their product makes "hold the same brace, in a
# different mode" unsayable. Split, it is one dropdown change, live, with the
# robot left where it is (see Session.command "mode").
KINDS = [
    ("brace+reach", "the certified maneuver: approach, brace, reach"),
    ("stand",       "legs only, no brace (subset=[])"),
    ("recover",     "brace released, back to the start pose (n_return>0)"),
]
TASKS = KINDS                       # the old name, for anything still using it

#: `stand` has no contact mode -- legs_only is the absence of one. Registry
#: entries for it are keyed with mode None so selecting it does not silently
#: change which mode the braced tasks will come back to.
NO_MODE = None


def mode_tag(mode):
    """The filename form of a contact mode: `elbow+forearm` -> `elbow_forearm`."""
    return mode.replace("+", "_")


def discover_plans(run_dir):
    """Read the cell and report what it can actually do, as (kind, mode) -> tag.

    A CELL IS THE SOURCE OF TRUTH, not this file. Which modes exist is a
    property of what has been solved into the directory; `mycell` certifies 17
    admissible subsets at its target and holds plans for whichever of them
    someone has run croco_run on. So the dropdowns are built by reading the
    plans and asking each one what it is, rather than from a table here that
    would go stale the first time anybody solved a new mode.

    Classification is off the plan's own fields, never off the filename:

      n_return > 0            -> `recover`  (it ends somewhere else)
      empty subset            -> `stand`    (nothing on the table)
      otherwise               -> `brace+reach`

    Filenames are still a CONVENTION worth keeping (`plan_<mode>.json`,
    `plan_recover_<mode>.json`) because that is what solve_tasks.sh writes and
    what a human greps for -- but a plan that disagrees with its own name is
    believed, not the name.
    """
    found, bad = {}, {}
    for path in sorted(glob.glob(os.path.join(run_dir, "plan_*.json"))):
        tag = os.path.basename(path)[len("plan_"):-len(".json")]
        try:
            plan = json.load(open(path))
        except Exception as exc:                                 # noqa: BLE001
            bad[tag] = str(exc)
            continue
        subset = plan.get("subset") or []
        mode = plan.get("mode") or tag
        if not subset:
            key = ("stand", NO_MODE)
        elif plan.get("n_return"):
            key = ("recover", mode)
        else:
            key = ("brace+reach", mode)
        # TWO PLANS CLAIMING THE SAME SLOT IS A REAL STATE and it must be
        # loud. It happens the moment anyone solves a variant alongside the
        # original -- `plan_elbow_palm.json` next to `plan_elbow_palm_kp50.json`
        # -- and the loser is chosen by ALPHABETICAL ORDER, which is not a
        # decision anybody made. Reported rather than resolved: which one is
        # wanted is not knowable from here, and quietly running the other is
        # how a session measures a plan nobody thinks is loaded.
        if key in found:
            bad[tag] = ("%s/%s is already served by plan_%s.json -- both "
                        "claim it, the alphabetically first one wins. Rename "
                        "or delete one." % (key[0], key[1] or "-", found[key]))
            continue
        found[key] = tag
    return found, bad
SUBMODES = ["single-shot", "hold", "automode"]

# THE AUTOMODE RING IS NOT `TASKS`, and the difference is a category error that
# cost a lurch. `stand` is the sweep ladder's `legs_only` row -- the CONTROL
# CONDITION, "reach the same target with no brace at all", which exists to say
# whether the brace is doing anything. It is not a step in a sequence, and
# chaining it after a braced robot is the largest seam in the cell: measured on
# the S20 cell's plan endpoints, brace_end -> stand_start is 55.5 mm of base
# and 1.126 rad of joint, against 24.0 mm / 0.499 rad for brace_end ->
# recover_start. Worse, it left `recover` to be entered from a STANDING pose,
# so a plan that begins braced had to drive forward into the brace before it
# could recover from it -- which is what the lurch looked like from outside.
# The ring is the round trip; `stand` stays selectable, on its own.
AUTO_RING = ["brace+reach", "recover"]


def mujoco_id(m, name):
    import mujoco
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)


def _tilt_deg(quat):
    """Angle between the pelvis z-axis and world up [deg], from a wxyz quat."""
    w, x, y, z = (float(v) for v in quat)
    # third column of R(q): where the body's own +z points in world.
    zz = 1.0 - 2.0 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0))))


# THE BRACING ARM'S SEVEN ACTUATORS, in `assert_joint_order`'s order (27
# joints: 0-5 left leg, 6-11 right leg, 12 torso, 13-19 left arm, 20-26 right).
# Kept as an index range rather than looked up by name because the loop already
# asserts that order every episode and a second, softer lookup could disagree
# with it.
BRACE_ACTUATORS = [(13, "sh_pitch"), (14, "sh_roll"), (15, "sh_yaw"),
                   (16, "elbow"), (17, "wr_roll"), (18, "wr_pitch"),
                   (19, "wr_yaw")]


# BAUMGARTE Kp VALUES THE PANEL OFFERS. Not a slider: the gain lives in the
# contact model's dynamics and `ContactModel3D.gains` is a read-only property
# (checked -- there is no setter), so every value here costs a 1.2 s OCP
# rebuild. A short ladder of measured points is more useful than a continuous
# control that rebuilds on every drag.
#
# `None` means "whatever the plan was solved at", which is the honest default:
# a plan re-solved at Kp=50 should be RUN at 50 without anyone selecting it.
# The measured sweep behind these numbers is in docs/lean/2026-08-21_brace_hold.html -- briefly,
# on a 40 s elbow+forearm hold: Kp=0 sinks 33 mm and is still sinking at
# 0.87 mm/s when the clock runs out; Kp=50 sinks 17 mm and has stopped
# (-0.03 mm/s), with brace drift down from 38 mm to 5 mm. It is NOT monotonic
# -- 10 and 20 are worse than 0 -- so the ladder includes them.
CONTACT_KPS = [None, 0.0, 10.0, 20.0, 30.0, 50.0, 100.0]


def telem_views(subset):
    """Which telemetry series share a y-axis, declared where the units are known.

    The page cannot work this out for itself and should not try. Newtons and
    millimetres obviously do not share an axis, but neither do two series in
    the SAME unit at different scales -- support margin lives at tens of mm and
    a pelvis height at 950, so plotting them together turns the one that moves
    into a flat line. So the grouping is authored here, next to the code that
    produces the numbers, and shipped to the browser as data.

    `brace drift` is the view this was built for: how far each bracing site has
    slid from the spot the static QP certified, live, while the brace is held.
    """
    F = ["F_%s" % s for s in subset] + ["F_other", "F_brace_total", "F_feet"]
    drift = ["drift_%s_mm" % s for s in subset] + ["penetration_mm"]
    return [
        dict(name="brace forces", unit="N", signed=False, keys=F),
        dict(name="brace drift", unit="mm", signed=True, keys=drift),
        dict(name="stability", unit="mm", signed=True,
             keys=["support_mm", "pelvis_drop_mm", "com_x_mm"]),
        # The bracing arm drifts ROTATIONALLY as well as translationally, and
        # the two are separate failures: sliding is friction, turning is the
        # brace pivoting about whichever contact is actually carrying it. A mm
        # trace cannot show the second one.
        dict(name="attitude", unit="deg", signed=True,
             keys=["pelvis_tilt_deg", "brace_rot_deg"]),
        # MOTOR EFFORT, which is a DIFFERENT QUANTITY from the F_ series and
        # is worth having next to them precisely because they are so easy to
        # confuse. F_<site> is the table pushing on a LINK; tau_<joint> is a
        # motor pushing on the robot's own skeleton. A brace can be carrying
        # 170 N through the elbow pad while the shoulder motor sits at 20% of
        # its limit, or the reverse. Normalised by the clamp basis so seven
        # joints with limits from 5 to 18 N.m share one axis; 1.0 is the limit.
        dict(name="brace motors", unit="|tau|/limit", signed=False,
             keys=["tau_%s" % n for _, n in BRACE_ACTUATORS] + ["tau_max"]),
    ]


def _lurch(qtrace, dt, window_s=0.5):
    """Measure the LURCH directly, rather than inferring it from the seam.

    `chain_dq_max_rad` says how far the plan's assumed pose is from the real
    one; it does NOT say what the robot did about it, and those are different
    questions -- a large seam that the MPC absorbs over two seconds is not a
    lurch, and a small one it closes in three periods is. So:

      pelvis_x_lurch_mm  how far FORWARD of its own starting x the pelvis ever
                         went. For `recover`, whose whole job is to come back
                         off the table, any positive number is the robot going
                         the wrong way first -- which is the thing you can see.
      dq_peak_rad_s      peak joint speed over the first `window_s`, where a
                         seam correction lands if there is one.

    Both are read off `qtrace`, which is recorded every period regardless of
    --video, so this costs nothing that was not already being paid.
    """
    if not qtrace:
        return {}
    Q = np.asarray(qtrace, float)
    out = dict(pelvis_x_lurch_mm=float(1e3 * (Q[:, 0].max() - Q[0, 0])))
    # A ONE-PERIOD EPISODE IS A REAL OUTCOME, not an impossible one: a task
    # selected while the robot is already at that plan's end has nothing left
    # to run, and `--seam project` can put it there deliberately. `max(2, ...)`
    # does not save a single-row trace -- `Q[:2]` of one row is one row, whose
    # diff is empty, and `.max()` of an empty array raises. Degenerate episodes
    # report the position metric and omit the velocity one rather than killing
    # the session on the way to the summary.
    if len(Q) >= 2:
        n = min(len(Q), int(round(window_s / dt)) + 1)
        dq = np.diff(Q[:max(2, n), 7:34], axis=0) / dt
        out["dq_peak_rad_s"] = float(np.abs(dq).max())
    return out


class Task:
    """One solved plan, and the MPC built from it. Built LAZILY.

    MEASURED, on the certified cell: 1.2 s for 200 models, and `MPC.reset()`
    to rewind the window is 8.8 ms. Both are far cheaper than this study's
    folklore had them -- twin_grid.sh still warns about "the next cell's ~25 s
    OCP build", which is the OFFLINE SOLVE time (plan_*.json records
    solve_seconds: 14.2) and not this. It matters because it is the number
    that decides whether behaviours can be chained live: at 9 ms a transition,
    they can.

    Lazy anyway, because a session may never select a given task and 1.2 s is
    still worth not spending three times at startup.
    """

    def __init__(self, name, run_dir, tag, note="", mode=None, kp=None):
        self.name, self.run_dir, self.tag, self.note = name, run_dir, tag, note
        # WHAT THIS TASK IS, as opposed to what it is called. `mode` is the
        # contact set; `kp` is the Baumgarte position gain the OCP is built
        # with, which is None for "whatever the plan was solved at". Both are
        # part of the CACHE KEY in Session._key, because both change the built
        # models and neither can be changed on a model that is already built.
        self.mode, self.kp = mode, kp
        self.plan = self.mpc = self.xs = self.us = None
        self.error = None

    @property
    def plan_path(self):
        return os.path.join(self.run_dir, "plan_%s.json" % self.tag)

    @property
    def ready(self):
        return all(os.path.exists(os.path.join(self.run_dir, f % self.tag))
                   for f in ("plan_%s.json", "xs_%s.npy", "us_%s.npy"))

    @property
    def built(self):
        return self.mpc is not None

    def why_unavailable(self):
        """Why it is greyed out AND the one command that ungreys it.

        A disabled dropdown entry with only a diagnosis reads as a feature
        that was never finished. It is a missing artifact, and the artifact
        takes seconds to make, so the note says so.

        TWO DIFFERENT MISSING THINGS, and pointing at the wrong fix wastes a
        round trip. A directory with no `modes.json` is not a cell at all --
        it has no reach target and no certified q* per contact mode, so
        `solve_tasks.sh` would bounce it straight back. Only a real cell that
        is merely missing this task gets sent there.
        """
        if self.ready:
            return None
        if not os.path.exists(os.path.join(self.run_dir, "modes.json")):
            return ("%s is not a cell yet -- no modes.json, so there is no "
                    "reach target and no certified pose per contact mode to "
                    "plan toward. Make one first (~10 s), then solve into "
                    "it:  studies/croco_modes.py --out %s --target 1.05 "
                    "-0.2348 1.0982  &&  studies/solve_tasks.sh %s"
                    % (self.run_dir, self.run_dir, self.run_dir))
        return ("no %s in this cell -- a task is a SOLVED PLAN (its phases "
                "differ by contact set, so no weight can substitute), and "
                "the certified grid only ever solved the braced maneuver. "
                "Solve the missing ones, ~5 s total:  "
                "studies/solve_tasks.sh %s"
                % (os.path.basename(self.plan_path), self.run_dir))

    def build(self, args):
        """Build the OCP + MPC for this task. Slow (~20 s); call off-thread."""
        if self.mpc is not None:
            return self
        if not self.ready:
            raise FileNotFoundError(self.why_unavailable())
        ns = argparse.Namespace(**vars(args))
        ns.dir, ns.tag = self.run_dir, self.tag
        if self.kp is not None:
            ns.contact_kp = self.kp
        self.plan, self.mpc, self.xs, self.us = build(ns)
        return self

    @property
    def contact_kp(self):
        """The Baumgarte position gain this task's OCP was actually built with.

        Not `self.kp`: that is the OVERRIDE, and None there means "the plan's
        own", which is the number a reader actually wants. Available only once
        built, for the same reason -- before that, the plan JSON has not been
        opened.
        """
        if self.kp is not None:
            return float(self.kp)
        if self.plan is not None:
            return float(self.plan.get("contact_kp", cp.CONTACT_GAINS[0]))
        return None


# -------------------------------------------------------------- session --- #
class Session:
    """Episodes on a control thread, driven by the panel.

    THE ONE-SHOT SCRIPT IS THE SPECIAL CASE NOW. `croco_twin` used to build an
    OCP, run one 4 s loop and exit, which is why the panel could never be
    opened in time and why a viewer collapsed the instant the maneuver ended.
    A session runs episodes until told to stop, so reset, pause and the viewer
    all have something to be relative to.

    THREADING. One control thread runs episodes. The browser's commands arrive
    on socket threads and only ever set fields under `self.lock`; the control
    thread reads them between periods (weights already worked this way -- see
    Panel.drain). Nothing here calls into crocoddyl from a socket thread.
    """

    def __init__(self, args, tasks, plant, m, kp, kd, tau_lim, hooks):
        self.args, self.plant = args, plant
        # THE REGISTRY IS KEYED BY (kind, mode, contact_kp), NOT BY NAME. All
        # three change the built models -- the contact SET is structure, and
        # `gains` is a read-only property on a built ContactModel3D -- so all
        # three have to be part of the identity of a built OCP. Keying on them
        # is also what makes switching back FREE: the 1.2 s build is paid once
        # per combination, and a mode you have already visited comes back in
        # the 8.8 ms it takes to rewind the MPC window.
        self.reg = dict(tasks)          # {(kind, mode, kp): Task}
        self._plans = {(k, m): t.tag for (k, m, _), t in self.reg.items()}
        self._notes = {(k, m): t.note for (k, m, _), t in self.reg.items()}
        self.m, self.kp, self.kd, self.tau_lim = m, kp, kd, tau_lim
        self.hooks = hooks              # extra on_step consumers (panel, video)
        self.monitor = None             # croco.safety.SafetyMonitor, or None
        self.lock = threading.Lock()
        self.paused = False
        self.quit = False
        self._reset = False
        self._skip = False              # end this episode, go to the next
        self.realtime = args.realtime
        self.submode = args.submode
        self.task_name = args.task
        self.status = "starting"
        self.episode = 0
        self.qtrace = []                # video states, paused periods excluded
        self.runs = []                  # one summary dict per episode
        self.viewer = None
        self._viewer_thread = None
        self._ov_key, self._ov, self._ov_ids = None, (None, {}), None
        self._ov_m = None               # private model for certified_sites
        # MARKER CLASSES, toggled from the panel. Four independent things get
        # drawn on top of the robot and they answer different questions: where
        # it was TOLD to reach, where the QP said the brace WOULD land, where
        # it is ACTUALLY touching, and how hard. Wanting the contacts without
        # the ghosts (or vice versa) is the normal case once you know which
        # question you are asking, and a scene with all four on is unreadable
        # in the one framing where the brace is legible.
        self.markers = dict(target=True, ghosts=True,
                            contacts=True, forces=True)
        # The magpie's jaws and wrist pad are class="collision", i.e. group 3,
        # which MuJoCo hides. `_make_plant` promotes them to group 2 so the arm
        # does not end in a stub 100 mm short of the hardware that is bracing.
        # That is right for a video and merely in the way once you know where
        # they are, so it is a toggle rather than a decision.
        self.show_gripper_geoms = True
        self._grip_geoms = None
        self._tel_ids = self._tel_z0 = self._tel_R0 = None
        self._tel_wrist = None
        self.telem = []                 # per-period telemetry, THIS episode
        self.panel = None
        self.q0 = None                  # in-process reset pose
        self.target = self.target0 = None      # reach target, live-editable
        self.target_nodes = 0
        self.rot_base = None            # the plan's own gripper orientation
        self.rot_deg, self.rot_axis, self.rot_nodes = 0.0, "x", 0
        self._auto = list(AUTO_RING)
        self.task_order = [n for n, _ in KINDS]
        self.mode = args.mode                  # current contact mode, or None
        self.modes = []                        # filled by run_session
        self.contact_kp = args.contact_kp      # None = each plan's own
        self._building = None                  # key currently being built

    # -- which plan is selected -------------------------------------------
    def _key(self, name=None, mode=None):
        """The registry key for a kind at the current mode and Kp.

        `stand` is keyed at NO_MODE deliberately: legs_only is the ABSENCE of a
        contact mode, so standing must not be able to consume or reset the
        mode the braced tasks will come back to.
        """
        name = self.task_name if name is None else name
        mode = self.mode if mode is None else mode
        return (name, NO_MODE if name == "stand" else mode, self.contact_kp)

    def get(self, name=None, mode=None):
        """The Task for a kind at a mode and the session's Kp, materialised.

        MATERIALISED LAZILY, because the Kp axis is unbounded: the registry is
        seeded with one entry per (kind, mode) at the starting gain, and any
        other gain the panel asks for gets its entries made here. Making a Task
        is free (it is a filename and two properties); BUILDING one is the 1.2 s,
        and that still happens once, in `run`, on the control thread.

        Dict insertion is atomic under the GIL and each key is written with the
        same value by whichever thread gets there first, so this needs no lock
        of its own -- which matters because `state()` calls it from sockets.
        """
        key = self._key(name, mode)
        t = self.reg.get(key)
        if t is None:
            kind, m, kp = key
            tag = self._plans.get((kind, m))
            if tag is not None:
                t = Task(kind, self.args.dir, tag,
                         self._notes.get((kind, m), ""), mode=m, kp=kp)
                self.reg[key] = t
        return t

    @property
    def tasks(self):
        """The three kinds at the CURRENT mode, keyed by name.

        Kept as a mapping because everything downstream -- the automode ring,
        the panel's task list, `run()` -- asks the same question it always did
        ("what is selected, and is it ready"). What changed is that the answer
        now depends on the mode, which is the entire point.
        """
        return {n: t for n, _ in KINDS
                for t in [self.get(n)] if t is not None}

    def modes_state(self):
        """Modes offered in the dropdown, with whether this cell can run them.

        `ready` is about the BRACE plan only. A mode with no `recover` plan is
        still perfectly selectable -- you just cannot recover out of it, which
        the task dropdown says on its own when you get there.
        """
        out = []
        for mode in self.modes:
            t = self.reg.get(("brace+reach", mode, self.contact_kp))
            out.append(dict(name=mode, ready=bool(t and t.ready),
                            recover=bool(self.reg.get(
                                ("recover", mode, self.contact_kp))),
                            built=bool(t and t.built)))
        return out

    # -- state the browser sees -------------------------------------------
    def state(self):
        cur = self.get()
        return dict(
            type="session", paused=self.paused, status=self.status,
            episode=self.episode, submode=self.submode, task=self.task_name,
            realtime=self.realtime, viewer=self.viewer is not None,
            target=self.target, target_nodes=self.target_nodes,
            target0=self.target0,
            rot_deg=self.rot_deg, rot_axis=self.rot_axis,
            rot_nodes=self.rot_nodes, rot_axes=sorted(self.ROT_AXES),
            can_viewer=self.args.plant == "mujoco" and _windowed_gl(),
            can_reset=self.args.plant == "mujoco",
            submodes=SUBMODES,
            # THE VIEWS FOLLOW THE TASK, so they ride on the session state
            # (pushed on every change) rather than the config (sent once): a
            # `stand` plan has an EMPTY contact subset, so its brace-force and
            # drift views have no series in them and the page must be told
            # that rather than keep drawing the previous task's.
            markers=dict(self.markers),
            gripper_geoms=bool(self.show_gripper_geoms),
            telem_views=(telem_views(cur.plan["subset"])
                         if (self.args.telemetry
                             and self.args.plant == "mujoco"
                             and cur is not None and cur.plan is not None)
                         else None),
            # ORDER COMES FROM THE SESSION, not from TASKS: discovered modes
            # are not in that table and would silently vanish from the
            # dropdown, which is the same "looks like the feature was never
            # built" failure the disabled-but-offered entries exist to avoid.
            tasks=[dict(name=n, note=(t.note if t is not None else note),
                        ready=bool(t is not None and t.ready),
                        why=(t.why_unavailable() if t is not None
                             else self._why_no_plan(n)))
                   for n, note in KINDS for t in [self.get(n)]],
            # THE CONTACT MODE, as its own axis. `modes` is what the cell has
            # solved, not what it certifies -- 17 subsets are admissible at
            # mycell's target and any of them can be added with one croco_run.
            mode=self.mode, modes=self.modes_state(),
            # BAUMGARTE Kp. Live, but a REBUILD rather than a slider: the gain
            # lives in the dynamics, and `ContactModel3D.gains` has no setter.
            contact_kp=self.contact_kp,
            contact_kp_plan=(cur.contact_kp if cur is not None else None),
            contact_kps=CONTACT_KPS,
            building=self._building is not None,
            built=bool(cur and cur.built))

    def _why_no_plan(self, kind):
        """Why a kind is missing AT THIS MODE, and the command that solves it.

        Distinct from Task.why_unavailable, which answers the same question for
        a task that at least EXISTS as a registry entry. A kind with no entry
        at all is the normal case for `recover` on a freshly added mode, and
        the fix is one croco_run -- so it says which one, rather than greying
        out an entry with no explanation.
        """
        if kind == "stand":
            return ("no plan_stand.json in this cell -- solve it with:  "
                    "studies/solve_tasks.sh %s" % self.run_dir_hint())
        return ("no %s plan for the %s mode in this cell. A task is a SOLVED "
                "PLAN and its phases differ by CONTACT SET, so this cannot be "
                "reached by retuning: solve it, ~3 s, with  "
                "studies/solve_tasks.sh %s %s %s"
                % (kind, self.mode, self.run_dir_hint(),
                   mode_tag(self.mode or ""), self.mode or ""))

    def run_dir_hint(self):
        return self.args.dir

    def push(self):
        st = self.state()
        if self.panel is not None:
            self.panel.set_session(st)      # so a late browser sees it too
            self.panel.server.broadcast(st)

    # -- the reach target, live -------------------------------------------
    # SAME MECHANISM AS THE WEIGHT SLIDERS, and for the same reason it works:
    # `ResidualModelFrameTranslation.reference` is a settable property, so the
    # target is data on a BUILT model rather than structure baked into it. No
    # rebuild, no re-solve -- 81 of the 201 nodes carry the reach cost (the
    # braced phase plus the terminal) and every one of them is repointed in
    # place.
    #
    # IT PERSISTS ACROSS RESETS FOR FREE. `MPC.reset` rebuilds the
    # ShootingProblem from the SAME model objects, so a moved target survives
    # a reset without being re-applied. What does NOT move is `xs_plan`, the
    # offline warm start, which still descends toward the original target --
    # so a large move is pulled to by the cost while being pulled from by the
    # warm start. Small moves track; big ones are a different plan and should
    # be re-solved offline.
    def apply_target(self, xyz, task=None):
        task = task or self.get()
        if task is None or task.mpc is None:
            return 0
        xyz = np.asarray(xyz, float)
        n = 0
        for mdl in list(task.mpc.models) + [task.mpc.terminal]:
            diff = getattr(mdl, "differential", None)
            costs = None if diff is None else getattr(diff, "costs", None)
            if costs is None or "reach" not in costs.costs.todict():
                continue
            try:
                costs.costs["reach"].cost.residual.reference = xyz
                n += 1
            except Exception:                                    # noqa: BLE001
                pass
        if n:
            self.target = [float(v) for v in xyz]
            self.target_nodes = n
        return n

    # -- gripper orientation, live ----------------------------------------
    # ONLY POSSIBLE BECAUSE THE TERM IS PRESENT. `_reach_orientation` returns
    # early when w_reach_rot is 0, so the certified plans carried no `reachRot`
    # cost at all and there was nothing to point anywhere -- a cost cannot be
    # added to a built model without reallocating its per-cost data. The plans
    # are now solved with reach_rot="auto" at a token weight: the reference is
    # the orientation q* already reaches, so it costs nothing and changes no
    # plan (re-solved brace+reach came back at cost 24.733046 against the
    # certified 24.733003), but it EXISTS, and an existing residual's
    # `.reference` is settable. The weight slider is what gives it authority.
    ROT_AXES = {"x": 0, "y": 1, "z": 2}

    def apply_rot(self, deg, axis="x", task=None):
        """Rotate the commanded gripper orientation about its own local axis."""
        task = task or self.get()
        if task is None or task.mpc is None:
            return 0
        a = self.ROT_AXES.get(axis, 0)
        th = np.deg2rad(float(deg))
        c, s_ = np.cos(th), np.sin(th)
        R = np.eye(3)
        i, j = [k for k in range(3) if k != a]
        R[i, i] = R[j, j] = c
        R[i, j], R[j, i] = -s_, s_
        n = 0
        for mdl in list(task.mpc.models) + [task.mpc.terminal]:
            diff = getattr(mdl, "differential", None)
            costs = None if diff is None else getattr(diff, "costs", None)
            if costs is None or "reachRot" not in costs.costs.todict():
                continue
            res = costs.costs["reachRot"].cost.residual
            if self.rot_base is None:
                self.rot_base = np.array(res.reference, float).copy()
            try:
                res.reference = self.rot_base @ R
                n += 1
            except Exception:                                    # noqa: BLE001
                pass
        if n:
            self.rot_deg, self.rot_axis, self.rot_nodes = float(deg), axis, n
        return n

    def set_status(self, s):
        self.status = s
        self.push()

    # -- commands (socket threads) ----------------------------------------
    def command(self, name, payload):
        with self.lock:
            if name == "pause":
                self.paused = bool(payload.get("on", not self.paused))
            elif name == "reset":
                self._reset = True
                self.paused = False
            elif name == "skip":
                self._skip = True
            elif name == "realtime":
                v = payload.get("value")
                self.realtime = None if v in (None, 0, "free") else float(v)
            elif name == "submode":
                if payload.get("value") in SUBMODES:
                    self.submode = payload["value"]
            elif name == "task":
                v = payload.get("value")
                if self.get(v) is not None:
                    self.task_name = v
                    self._skip = True     # take effect at the episode boundary
            elif name == "mode":
                # SWITCHING CONTACT MODE MID-SESSION, which is the whole point
                # of splitting this axis out. It ends the episode (`_skip`) and
                # does NOT reset the plant, so the next episode is CHAINED:
                # the robot stays exactly where it is, braced, and the new
                # mode's plan is joined at the node nearest the measured pose
                # (`--seam project`). What that cannot do is move a limb onto
                # the table that is not already there -- switching from `elbow`
                # to `elbow+forearm` while braced asks the new plan's contact
                # set to be satisfied from a pose where the forearm is in the
                # air, and the seam metrics in the run artifact are what say
                # whether it was.
                v = payload.get("value")
                if v in self.modes and v != self.mode:
                    self.mode = v
                    self._skip = True
            elif name == "contact_kp":
                # Baumgarte Kp. Rebuilds (1.2 s) rather than retuning, and the
                # rebuild happens on the control thread at the next episode
                # boundary like any other task build.
                v = payload.get("value")
                v = None if v in (None, "", "plan") else float(v)
                if v != self.contact_kp:
                    self.contact_kp = v
                    self._skip = True
            elif name == "viewer":
                self._viewer_req = bool(payload.get("on"))
                return self._toggle_viewer(self._viewer_req)
            elif name == "quit":
                self.quit = True
                self._skip = True
            elif name == "render":
                threading.Thread(target=self.render, daemon=True).start()
            elif name == "dump":
                threading.Thread(target=self.dump_telemetry,
                                 daemon=True).start()
            elif name == "markers":
                k = payload.get("name")
                if k in self.markers:
                    self.markers[k] = bool(payload.get("on"))
            elif name == "gripper_geoms":
                self.set_gripper_geoms(bool(payload.get("on")))
            elif name == "target":
                v = payload.get("value")
                if payload.get("reset") and self.target0:
                    v = list(self.target0)
                if v and len(v) == 3:
                    self.apply_target(v)
            elif name == "rot":
                self.apply_rot(payload.get("deg", 0.0),
                               payload.get("axis", self.rot_axis))
        self.push()

    def dump_telemetry(self, path=None):
        """Write this episode's telemetry to JSON. Returns the path, or None.

        A LIVE PLOT THAT CANNOT BE SAVED IS A DEMO. Every question this section
        was built for -- is the brace sliding, how fast, does it stop -- is
        answered by comparing two runs, and a canvas that scrolls 4000 samples
        and forgets them cannot be compared to anything. The file is the same
        per-period records the panel is drawing, so a plot in the browser and a
        plot made afterwards are of the same numbers.
        """
        cur = self.get()
        if not self.telem or cur is None or cur.plan is None:
            return None
        # NEXT TO `--out`, NOT IN THE CELL, when there is an --out to be next
        # to. A batch that sweeps one knob writes one artifact per setting but
        # every episode is still `brace+reach` episode 1, so a cell-relative
        # name collides on the second run and the first result is gone. It has
        # happened once already, mid-sweep, silently.
        if path is None and self.args.out:
            stem = os.path.splitext(self.args.out)[0]
            path = "%s_telemetry_%s_ep%d.json" % (
                stem, self.task_name.replace("+", "_"), self.episode)
        # THE TAG IS IN THE NAME because the mode is now a dropdown: two holds
        # of `brace+reach` in one session can be two different contact modes,
        # and a name that cannot tell them apart is a name that overwrites the
        # comparison you just made.
        path = path or os.path.join(
            self.args.dir, "telemetry_%s_%s_ep%d.json"
            % (cur.tag, self.task_name.replace("+", "_"), self.episode))
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".",
                        exist_ok=True)
            with open(path, "w") as fh:
                json.dump(dict(task=self.task_name, episode=self.episode,
                               cell=self.args.dir, tag=self.args.tag,
                               dt=cur.plan["dt"], mode=cur.mode,
                               contact_kp=cur.contact_kp,
                               seam_mode=self.args.seam,
                               submode=self.submode,
                               views=telem_views(cur.plan["subset"]),
                               rows=self.telem), fh, indent=1)
        except Exception as exc:                                 # noqa: BLE001
            self.set_status("telemetry dump failed: %s" % exc)
            return None
        self.set_status("wrote %s (%d periods)"
                        % (os.path.basename(path), len(self.telem)))
        return path

    def render(self):
        """Render the last episode to the panel. Off the control thread.

        PAUSED PERIODS ARE ABSENT BY CONSTRUCTION, not by filtering: the
        recorder is an `on_step` hook and `ControlLoop` does not call `on_step`
        for a paused period, so a pause leaves no frames rather than a stretch
        of identical ones. A video of a run someone paused halfway is still a
        video of the trajectory, which is what makes it comparable to a replay.
        """
        q, task = list(self.qtrace), self.get()
        # THE TELEMETRY GOES WITH THE VIDEO. Rendering is the point at which
        # someone decides this episode was worth keeping; the numbers behind
        # the picture are worth keeping at the same moment, and asking for them
        # separately means remembering to. Dumped BEFORE the early return, so a
        # session with --no-video still gets them from the render button.
        self.dump_telemetry()
        if not q or task is None or task.plan is None or not self.args.video:
            return
        # ONE VIDEO PER MODE, not one per session. The default path is derived
        # from `--tag`, which was the only mode a session could run; now that
        # the mode is a dropdown, keeping that name means the second mode you
        # try silently overwrites the first -- and the whole reason to switch
        # modes live is to put the two next to each other. An explicit --video
        # is still taken literally: someone who named the file meant it.
        path = self.args.video
        if getattr(self.args, "video_auto", False):
            path = os.path.join(self.args.dir, "twin_%s.mp4" % task.tag)
        self.set_status("rendering %d states ..." % len(q))
        try:
            got = render_run(q, task.plan, task.run_dir, path,
                             cam=self.args.video_cam, fps=self.args.video_fps,
                             dt_plan=task.plan["dt"])
            if got and self.panel is not None:
                self.panel.set_video(got)
        except Exception as exc:                                 # noqa: BLE001
            self.set_status("render failed: %s" % exc)
            return
        self.set_status("rendered %s" % os.path.basename(path))

    # -- the viewer --------------------------------------------------------
    def _toggle_viewer(self, on):
        """Open/close the passive viewer mid-session.

        Called on a socket thread and NOT under the control thread's step, but
        `launch_passive` only reads the model and data pointers -- it does not
        step them -- and `sync()` is called from the control thread as before.
        """
        if on and self.viewer is None:
            if self.args.plant != "mujoco" or not _windowed_gl():
                return
            import mujoco.viewer as _mjv
            # LAUNCH UNDER THE PLANT'S DATA LOCK. `launch_passive` runs
            # mj_forward on this mjData before it hands back a handle, and
            # this call is on a socket thread while the control thread is in
            # mj_step. There is no viewer lock yet -- it does not exist until
            # the handle does -- so the plant's own lock is the only thing
            # that can close the window. This is the crash the panel's viewer
            # button produced; faulthandler put the control thread in
            # mujoco_plant.step and this thread in viewer.py launch_passive.
            with self.plant.data_lock:
                self.viewer = _mjv.launch_passive(self.plant.m, self.plant.d)
            self._viewer_thread = _viewer_thread_of(_mjv)
            # HAND THE PLANT THE VIEWER'S LOCK. Everything that mutates mjData
            # -- mj_step, d.ctrl, the reset -- must hold it while a render
            # thread is reading, or the process dies intermittently with no
            # traceback. Attaching it here rather than at construction keeps
            # the cost at exactly zero for the runs with no viewer, which is
            # all of the batch grid.
            self.plant.viewer_lock = self.viewer.lock
        elif not on and self.viewer is not None:
            self.close_viewer()
        self.push()

    # -- telemetry ---------------------------------------------------------
    # WHY THIS IS NOT THE COST PLOT. The cost traces say what the OPTIMISER
    # thinks, in units of its own weights; they move when a weight moves and
    # they say nothing at all about newtons. The questions this study keeps
    # asking are physical -- which link is carrying the brace, how far it has
    # slid from the spot the QP certified, whether the CoM is still over the
    # feet -- and every one of them is a MuJoCo query, not a crocoddyl one.
    # `croco_replay` has computed exactly these numbers per period for a long
    # time, offline, after the run. This is the same computation on the live
    # plant, through the same `cr.table_forces` so the attribution cannot
    # diverge between what the panel shows and what the replay scores.
    #
    # IN-PROCESS PLANT ONLY. Over DDS there is no local mjData to query -- the
    # forces are on the far side of the wire and the twin does not publish
    # them -- so the section is absent rather than zero-filled.
    def _telemetry(self, task):
        m, d = self.plant.m, self.plant.d
        _, site_ref = self._overlay_refs(task)
        subset = task.plan["subset"] if task.plan else []
        if self._tel_ids is None:
            self._tel_ids = (
                {s: cs.bid(m, cs.SITES[s][0]) for s in subset},
                [cs.bid(m, f) for f in cs.FEET],
                cs.bid(m, "table"))
        brace_bodies, feet, tbl = self._tel_ids
        out = cr.table_forces(m, d, subset, brace_bodies, feet, tbl)
        # `penetration` is the deepest (most negative) table gap in metres;
        # everything else on this axis is mm, so it converts rather than
        # forcing the plot to span six orders of magnitude.
        out["penetration_mm"] = 1e3 * float(out.pop("penetration", 0.0))
        out["support_mm"] = 1e3 * cr.support_margin(m, d)
        if self._tel_z0 is None:
            self._tel_z0 = float(d.qpos[2])
        out["pelvis_drop_mm"] = 1e3 * (self._tel_z0 - float(d.qpos[2]))
        out["pelvis_tilt_deg"] = _tilt_deg(d.qpos[3:7])
        out["com_x_mm"] = 1e3 * float(d.subtree_com[1][0])
        # DRIFT, the reason this exists: distance from each bracing site to the
        # spot the static QP certified and the OCP's hold_ cost is written
        # against. A brace that is holding reads flat; one that is sliding does
        # not, and the trace says so while it is still happening.
        for st_name, ref in site_ref.items():
            p_now = cs.point_world(m, d, *cs.SITES[st_name])
            out["drift_%s_mm" % st_name] = 1e3 * float(
                np.linalg.norm(p_now - ref))
        # Rotational drift of the bracing arm, as the angle of its wrist frame
        # from the orientation it held at the first telemetry sample of this
        # episode. Referenced to the episode and not to q* on purpose: q*'s
        # wrist orientation is an IK by-product, while "has it turned since it
        # landed" is the question being asked.
        # Motor effort. `d.actuator_force` is the torque the position servo is
        # actually applying, i.e. what the joint is doing about the load --
        # NOT what the table is doing to the link.
        tau = d.actuator_force[:len(self.tau_lim)]
        ratio = np.abs(tau) / self.tau_lim
        for i, nm in BRACE_ACTUATORS:
            if i < len(ratio):
                out["tau_%s" % nm] = float(ratio[i])
        out["tau_max"] = float(ratio.max())
        out["tau_max_joint"] = int(np.argmax(ratio))
        R = d.xmat[self._tel_wrist].reshape(3, 3).copy()
        if self._tel_R0 is None:
            self._tel_R0 = R
        c = 0.5 * (float(np.trace(self._tel_R0.T @ R)) - 1.0)
        out["brace_rot_deg"] = float(np.degrees(np.arccos(np.clip(c, -1, 1))))
        return out

    # -- viewer overlay ----------------------------------------------------
    # THE SAME MARKERS THE VIDEO DRAWS, LIVE. `render_run` has always drawn the
    # reach target, the certified landing spots as ghosts, and the live table
    # contacts coloured by which link is making them -- but only into the mp4,
    # i.e. only AFTER the episode, which is the one time you cannot act on
    # them. The viewer showed a robot leaning on nothing.
    #
    # `user_scn` is the passive viewer's own scene, drawn on top of the model
    # and owned by us: it is refilled from ngeom = 0 every period, so nothing
    # accumulates and no geom outlives the contact it marks. The draw helpers
    # are croco_replay's unchanged, which is what makes a frozen frame of the
    # viewer and a frame of the video mean the same thing.
    def _overlay_refs(self, task):
        """Cache the per-task reference markers. Cheap, but not free: the
        certified sites are read by posing the model at q* and restoring."""
        key = task.name
        if getattr(self, "_ov_key", None) == key:
            return self._ov
        target = np.array(task.plan["target"]) if task.plan else None
        try:
            # NOT THE PLANT'S mjData. `certified_sites` poses the model at q*,
            # runs mj_forward, and restores -- correct in a replay, where it
            # owns the model, and a hazard here, where the control thread is
            # mid-episode on that same mjData and the viewer thread may be
            # reading it. It restores what it wrote, so the damage would be
            # intermittent rather than reproducible, which is the worst kind.
            # A private model costs one `cs.load` per session.
            if self._ov_m is None:
                self._ov_m = cs.load(ik_margin=0.0)
            site_ref = cr.certified_sites(task.plan, task.run_dir,
                                          *self._ov_m)
        except Exception:                                        # noqa: BLE001
            site_ref = {}      # a missing artifact costs the ghosts, not the run
        self._ov_key, self._ov = key, (target, site_ref)
        return self._ov

    def draw_overlay(self, task):
        """Refill `viewer.user_scn` with the reference and contact markers."""
        if self.viewer is None or self.args.plant != "mujoco":
            return
        scn = getattr(self.viewer, "user_scn", None)
        if scn is None:                       # older mujoco: no user scene
            return
        target, site_ref = self._overlay_refs(task)
        m, d = self.plant.m, self.plant.d
        if self._ov_ids is None:
            self._ov_ids = (
                {s: cs.bid(m, cs.SITES[s][0]) for s in cr.SITE_RGBA
                 if s in cs.SITES},
                cs.bid(m, "table"),
                [cs.bid(m, f) for f in cs.FEET])
        site_bodies, tbl, feet = self._ov_ids
        scn.ngeom = 0
        mk = self.markers
        try:
            if target is not None and (mk["target"] or mk["ghosts"]):
                cr.draw_refs(scn, target, site_ref,
                             show_target=mk["target"], show_ghosts=mk["ghosts"])
            if mk["contacts"] or mk["forces"]:
                cr.draw_contacts(scn, m, d, site_bodies, tbl, feet,
                                 show_points=mk["contacts"],
                                 show_forces=mk["forces"])
        except Exception:                                        # noqa: BLE001
            scn.ngeom = 0     # a bad frame must not kill the control thread

    def set_gripper_geoms(self, on):
        """Show or hide the magpie's collision jaws, wrist pad and flange.

        GROUP, NOT COLLISION. This moves `geom_group` between 2 (drawn) and 3
        (the collision group MuJoCo's renderer hides) and touches nothing the
        integrator reads -- not contype, not conaffinity, not mass. Worth being
        explicit about, because "the palm carries 0 N" invites the conclusion
        that its collision was switched off, and it was not: the jaws sit 33 mm
        above the wood and the gripper box 22 mm above it, with contype 1 and
        conaffinity 1 the whole time. They report no force because they are in
        the air, and hiding them changes only whether you can see that.
        """
        if self.args.plant != "mujoco":
            return
        m = self.plant.m
        if self._grip_geoms is None:
            names = []
            for arm in ("left", "right"):
                names += ["%s_%s" % (arm, s) for s in
                          ("gripper_collision", "gripper_jaw_a",
                           "gripper_jaw_b", "gripper_flange", "wrist_pad")]
            self._grip_geoms = [g for g in
                                (mujoco_id(m, n) for n in names) if g >= 0]
        for g in self._grip_geoms:
            m.geom_group[g] = 2 if on else 3
        self.show_gripper_geoms = bool(on)

    def close_viewer(self):
        if self.viewer is None:
            return
        v, th = self.viewer, self._viewer_thread
        self.viewer, self._viewer_thread = None, None
        # Detach BEFORE closing: a lock belonging to a torn-down viewer is not
        # a lock the control thread should still be taking.
        try:
            self.plant.viewer_lock = None
        except Exception:                                        # noqa: BLE001
            pass
        try:
            with self.plant.data_lock:   # teardown touches mjData too
                v.close()
            if th is not None:
                th.join(timeout=5.0)     # see the note at first launch
        except Exception:                                        # noqa: BLE001
            pass

    # -- episodes (control thread) ----------------------------------------
    def _reset_plant(self, task):
        """Put the in-process plant back where the plan begins.

        THE TWIN CANNOT BE RESET FROM HERE and the button says so. `lean_twin`
        owns its physics and has no reset channel -- twin_grid.sh starts one
        twin PER CELL for exactly this reason ("restarting is cheaper than
        making it resettable"). Resetting only the controller against a twin
        that kept its pose would restart the maneuver from wherever the robot
        happened to be, which is the initial-condition failure this study
        already spent a session diagnosing.
        """
        # THE CONTROLLER IS PART OF THE STATE BEING RESET. Putting the plant
        # back without this leaves the MPC's window parked at the end of the
        # plan -- see MPC.reset for the measurement.
        if task.mpc is not None:
            task.mpc.reset()
        if self.args.plant != "mujoco":
            return False
        import mujoco as _mj
        q0 = cb.pin_to_mj(task.xs[0][:cb.NQ_ROBOT],
                          cs.start_qpos(self.m, task.plan["start"]))
        guard = getattr(self.plant, "_guard", None)
        lk = getattr(self.plant, "data_lock", None) or _nullcontext()
        with lk, (guard() if guard else _nullcontext()):
            self.plant.d.qpos[:] = q0
            self.plant.d.qvel[:] = 0.0
            self.plant.d.ctrl[:] = 0.0
            _mj.mj_forward(self.plant.m, self.plant.d)
        if self.viewer is not None:
            try:
                self.viewer.sync()
            except Exception:                                    # noqa: BLE001
                pass
        return True

    def _make_policy(self, task, stats):
        """The MPC as a policy, with `hold` expressed as a clamped plan index.

        HOLD COSTS NOTHING. `MPC.__call__` slides its window forward only
        (`while self.head < k + H`), so feeding it a constant k simply stops
        the slide and it keeps re-solving the same window against the live
        state -- which is a hold, and is why this needed no new artifact. S19
        measured the brace holding for 8 s with the same margin it had at
        1.6 s, so the hold is the plan's own final window, not an extrapolation.
        """
        nq, us, xs = cb.NQ_ROBOT, task.us, task.xs
        dt_plan = task.plan["dt"]
        k_hold = len(us) - 1
        q0_plan = cb.pin_to_mj(xs[0][:nq], cs.start_qpos(self.m,
                                                        task.plan["start"]))
        # The plan as a POSE TABLE, for the seam projection below. Built once
        # per episode, not per period: 201 x 27 is nothing, but it is nothing
        # inside a 20 ms budget only if it is not rebuilt 200 times.
        qj_plan = np.asarray(xs, float)[:, 7:34]
        base_plan = np.asarray(xs, float)[:, 0:3]
        seam = {"k0": 0}

        def policy(t, st):
            if not stats.get("seam"):
                # WHERE THE ROBOT ACTUALLY IS when this behaviour starts,
                # against where its plan assumes it is. On a reset these are
                # equal by construction; on a chain they are not, and the
                # number is the whole question of whether chaining works.
                stats["seam"] = True
                stats["chain_dq_max_rad"] = float(
                    np.max(np.abs(st.q - q0_plan[7:34])))
                stats["chain_dbase_mm"] = float(
                    1e3 * np.linalg.norm(st.base_pos - q0_plan[0:3]))
                # JOIN THE PLAN WHERE THE ROBOT ALREADY IS, rather than at
                # node 0. The MPC has always solved from the measured state
                # (`problem.x0 = x_meas`, every period) -- what was seeded
                # offline is not x0 but the REFERENCE TAPE: the index k
                # selects which node's landing spots, hold costs, contact set
                # and state regulariser the window carries, and `k =
                # round(t/dt)` with t restarting at 0 makes that node 0 of the
                # plan no matter where the robot is standing. The costs then
                # pull it onto node 0 at the tape's speed, which is the lurch.
                # Projecting picks the node whose pose the robot is nearest
                # and runs the tape from there, so a chained behaviour starts
                # from truth without re-solving anything.
                #
                # THE METRIC IS DELIBERATELY CRUDE: joint L2 in rad plus base
                # translation at W_BASE rad/m, no velocity. Velocity at a seam
                # is small and noisy and would only make the choice jitter.
                # It CAN alias on a plan that passes through the same pose
                # twice (an out-and-back), so the residual distance is
                # reported rather than assumed small -- read `chain_k0_dist`
                # before trusting `chain_k0` on a new maneuver.
                if self.args.seam == "project":
                    W_BASE = 3.0        # 0.1 m of base ~ 0.3 rad of joint
                    d = (np.linalg.norm(qj_plan - st.q, axis=1)
                         + W_BASE * np.linalg.norm(base_plan - st.base_pos,
                                                   axis=1))
                    # LEAVE A HORIZON OF RUNWAY. Projecting is meant to skip
                    # the part of a behaviour the robot has effectively already
                    # done -- never to skip the behaviour. Uncapped it can land
                    # on the last node of a plan whose end pose the robot is
                    # already in (select `stand` while standing and every node
                    # of a lean is further away than the last one), which runs
                    # an episode of ZERO periods: the ring advances instantly,
                    # nothing moves, and the summary has no trajectory to
                    # measure. Capped, the worst case is a short episode that
                    # holds the final window, which is a hold and is harmless.
                    k_cap = max(0, len(us) - 1 - self.args.horizon)
                    k_raw = int(np.argmin(d))
                    seam["k0"] = int(np.clip(k_raw, 0, k_cap))
                    stats["chain_k0_dist"] = float(d[seam["k0"]])
                    if k_raw != seam["k0"]:
                        # The cap binding means the projection wanted to start
                        # inside the last window: the robot is at (or past) the
                        # end of this behaviour. Worth seeing, not worth
                        # failing on.
                        stats["chain_k0_capped_from"] = k_raw
                stats["chain_k0"] = seam["k0"]
                k0 = seam["k0"]
                stats["chain_dq_max_rad_at_k0"] = float(
                    np.max(np.abs(st.q - qj_plan[k0])))
                stats["chain_dbase_mm_at_k0"] = float(
                    1e3 * np.linalg.norm(st.base_pos - base_plan[k0]))
            k = seam["k0"] + int(round(t / dt_plan))
            if k >= len(us):
                if self.submode == "hold":
                    k = k_hold
                else:
                    return None
            qpos = np.concatenate([st.base_pos, st.base_quat, st.q])
            R = _quat_to_mat(st.base_quat)
            qvel = np.concatenate([st.base_linvel, st.base_angvel, st.v])
            x_meas = np.concatenate([cb.mj_to_pin(qpos), cb.mj_to_pin_v(qvel, R)])
            u0, xs1 = task.mpc(k, x_meas)
            stats["steps"] += 1
            if u0 is None:
                stats["mpc_none"] += 1
                return xs[k][7:nq], np.zeros(27), us[k]
            return xs1[:nq][7:], xs1[nq:][6:], np.clip(u0, -self.tau_lim,
                                                       self.tau_lim)
        return policy

    def run_episode(self, task):
        """One run of one task. Returns its summary dict."""
        a = self.args
        dt_plan = task.plan["dt"]
        stats = dict(steps=0, mpc_none=0)
        submode0 = self.submode         # the policy still reads it live
        self.qtrace = []                # the video is THIS episode, not a pile
        # PER-EPISODE BASELINES. `pelvis_drop_mm` and `brace_rot_deg` are both
        # measured FROM the start of the episode, so carrying them across one
        # would report the previous behaviour's drift as this one's.
        self.telem = []
        self._tel_ids = self._tel_z0 = self._tel_R0 = None
        if a.plant == "mujoco" and self._tel_wrist is None:
            self._tel_wrist = cs.bid(self.plant.m,
                                     "%s_wrist_yaw_link" % cs.BRACE_ARM)

        def record(row, st, cmd):
            if a.plant == "mujoco":
                self.qtrace.append(self.plant.d.qpos.copy())
            else:
                q = self._qtmpl.copy()
                q[0:3], q[3:7] = st.base_pos, st.base_quat
                q[7:7 + 27] = st.q
                self.qtrace.append(q)

        def on_step(row, st, cmd):
            # ANNOTATE BEFORE THE HOOKS RUN, which is the same order the safety
            # monitor already relies on: the panel serialises `row`, so a field
            # added after it has been broadcast arrives one period late.
            if self.args.telemetry and a.plant == "mujoco":
                try:
                    tel = self._telemetry(task)
                    tel["t"] = row.get("t")
                    tel["k"] = len(self.telem)
                    row["telem"] = tel
                    self.telem.append(tel)
                except Exception:                                # noqa: BLE001
                    pass      # telemetry is never a reason to miss a period
            for h in self.hooks:
                try:
                    h(row, st, cmd)
                except Exception:                                # noqa: BLE001
                    pass
            record(row, st, cmd)
            if self.viewer is not None:
                try:
                    self.draw_overlay(task)
                    self.viewer.sync()
                except Exception:                                # noqa: BLE001
                    pass

        with self.lock:
            self._reset = self._skip = False
            rt = self.realtime
        cfg = LoopConfig(ctrl_hz=1.0 / dt_plan, stale_s=a.stale_ms * 1e-3,
                         realtime=rt)
        stance = (cs.start_qpos(self.m, task.plan["start"])[7:]
                  if a.bringup else None)
        loop = ControlLoop(
            self.plant, self._make_policy(task, stats), stance=stance, cfg=cfg,
            on_step=on_step,
            paused=lambda: self.paused,
            stop=lambda: self.quit or self._reset or self._skip)
        t0 = time.monotonic()
        log = loop.run(self.kp, self.kd, max_seconds=a.max_seconds)
        solves = [r["solve_ms"] for r in log if "solve_ms" in r]
        ages = [1e3 * r["age"] for r in log if "age" in r]
        return dict(
            episode=self.episode, task=task.name, submode=submode0,
            # WHICH PLAN THIS WAS. A session can now change contact mode and
            # Baumgarte gain between episodes, so an artifact that records
            # only the task name no longer identifies what ran.
            mode=task.mode, tag=task.tag, contact_kp=task.contact_kp,
            chained=bool(getattr(self, "_chained", False)),
            chain_dq_max_rad=stats.get("chain_dq_max_rad"),
            chain_dbase_mm=stats.get("chain_dbase_mm"),
            # The seam the CONTROLLER actually faces, which is the one that
            # decides whether the transition is smooth. The two above stay
            # keyed on node 0 so a `--seam project` run is comparable against
            # every `--seam node0` run already on disk.
            seam_mode=self.args.seam,
            chain_k0=stats.get("chain_k0"),
            chain_k0_dist=stats.get("chain_k0_dist"),
            chain_k0_capped_from=stats.get("chain_k0_capped_from"),
            chain_dq_max_rad_at_k0=stats.get("chain_dq_max_rad_at_k0"),
            chain_dbase_mm_at_k0=stats.get("chain_dbase_mm_at_k0"),
            wall_s=time.monotonic() - t0, periods=len(log),
            mpc_steps=stats["steps"], overruns=loop.overruns,
            worst_overrun_ms=1e3 * loop.worst_overrun_s,
            watchdog_trips=loop.watchdog_trips,
            paused_periods=loop.paused_periods,
            tau_saturated=sum(r.get("tau_sat", 0) for r in log),
            q_clipped=sum(r.get("q_clip", 0) for r in log),
            solve_ms_mean=float(np.mean(solves)) if solves else None,
            solve_ms_p95=float(np.percentile(solves, 95)) if solves else None,
            age_ms_p95=float(np.percentile(ages, 95)) if ages else None,
            realtime=rt,
            realtime_note=(None if rt is None or rt >= 1.0 else
                           "SLOWED to %.3gx: overruns are NOT a deployment "
                           "result" % rt),
            # Per EPISODE, unlike the monitor's own session counters: a chain
            # that trips on its third behaviour and not its first is a
            # different finding from one that trips throughout.
            estop_periods=sum(1 for r in log if r.get("estop")),
            estop_first_why=next((r["estop_why"] for r in log
                                  if r.get("estop")), None),
            pelvis_z=(float(self.plant.d.qpos[2]) if a.plant == "mujoco"
                      else None),
            **_lurch(self.qtrace, dt_plan))

    def idle(self):
        """Hold the pose and stay alive, waiting for the panel.

        THE VIEWER MUST OUTLIVE THE TRAJECTORY. A single-shot run that tore
        the window down at the last node made the reset button useless -- by
        the time you reached for it there was nothing left to reset.
        """
        self.set_status("idle -- reset to run again")
        while not self.quit:
            with self.lock:
                if self._reset or self._skip:
                    return
            # WEIGHTS MOVED WHILE IDLE MUST STILL LAND. `Panel.drain` is
            # otherwise only called from `on_step`, so between episodes a
            # slider queued an edit and then showed the old value back --
            # which reads as a dead control on exactly the screen where
            # someone is retuning before hitting reset.
            if self.panel is not None:
                try:
                    self.panel.drain()
                except Exception:                                # noqa: BLE001
                    pass
            try:
                self.plant.write(self.plant.safe_hold(2.0))
            except Exception:                                    # noqa: BLE001
                pass
            if self.viewer is not None:
                try:
                    # THE MARKERS MUST OUTLIVE THE EPISODE, for the same reason
                    # the viewer does: the frame you want to read is the one
                    # after the motion stopped, and a scene that drops its
                    # contact dots at the last node hides exactly the state you
                    # paused to look at.
                    t = self.get()
                    if t is not None and t.plan is not None:
                        self.draw_overlay(t)
                    self.viewer.sync()
                except Exception:                                # noqa: BLE001
                    pass
            time.sleep(0.02)

    def run(self):
        """The supervisor. Runs until the panel (or Ctrl-C) says stop."""
        if self.args.plant != "mujoco":
            self._qtmpl = cs.load(ik_margin=0.0)[1].qpos.copy()
        while not self.quit:
            task = self.get()
            if task is None or not task.ready:
                self.set_status("%s is not available -- %s"
                                % (self.task_name,
                                   task and task.why_unavailable()))
                self.idle()
                continue
            if not task.built:
                # THE BUILD IS ANNOUNCED WITH WHAT IT IS BUILDING. A mode or Kp
                # switch is a rebuild, and 1.2 s of a silent panel after a
                # dropdown change reads as a dead control.
                self._building = self._key()
                self.set_status(
                    "building the %s OCP (%s%s) ..."
                    % (task.name, task.mode or "legs only",
                       "" if self.contact_kp is None
                       else ", Kp=%g" % self.contact_kp))
                try:
                    task.build(self.args)
                except Exception as exc:                         # noqa: BLE001
                    task.error = str(exc)
                    self._building = None
                    self.set_status("%s failed to build: %s"
                                    % (task.name, exc))
                    self.idle()
                    continue
                finally:
                    self._building = None
                if self.panel is not None:
                    self.panel.set_mpc(task.mpc)   # sliders follow the OCP
            if task.plan is not None:
                if self.target0 is None:
                    self.target0 = list(task.plan["target"])
                # A target the operator moved is theirs, not the plan's: carry
                # it onto whatever task is selected next rather than silently
                # reverting to the JSON on a task switch.
                self.apply_target(self.target or task.plan["target"], task)
                self.rot_base = None        # each task carries its own q*
                self.apply_rot(self.rot_deg, self.rot_axis, task)
            with self.lock:
                reset_wanted = self._reset
                self._reset = False
            # CHAINING: rewind the CONTROLLER, leave the ROBOT where it is.
            # This is the whole of continuous behaviour, and it is cheap --
            # MPC.reset() is 8.8 ms, well inside one 20 ms period, so the seam
            # between two behaviours costs less than a control step. What it
            # does NOT do is guarantee the next plan's x0 is where the robot
            # actually is; that mismatch is measured per transition and
            # reported as `chain_dq_max_rad`, because a chain that silently
            # starts a maneuver from the wrong pose is the exact failure this
            # study already diagnosed once on the twin.
            chained = not (reset_wanted or self.episode == 0)
            if chained:
                if task.mpc is not None:
                    task.mpc.reset()
            else:
                self._reset_plant(task)
            self._chained = chained
            self.episode += 1
            self.set_status("running %s / %s / %s (episode %d)"
                            % (task.name, task.mode or "legs only",
                               self.submode, self.episode))
            self.runs.append(self.run_episode(task))
            # A RUN ARTIFACT SHOULD CARRY ITS OWN TELEMETRY. The dump button is
            # for a session someone is watching; a batch run has nobody to
            # press it, and "re-run it and press the button this time" is not a
            # thing you can do to a stochastic failure.
            if self.args.out and self.args.telemetry:
                self.dump_telemetry()
            self.push()
            if self.quit:
                break
            if (self.args.episodes is not None
                    and self.episode >= self.args.episodes):
                # Checked AFTER the episode is recorded, so `--episodes 4`
                # writes four episodes and not three plus a truncated one.
                #
                # AND IT RENDERS ON THE WAY OUT. This used to `break` straight
                # past the render below, so `--episodes N` with video on
                # produced a run artifact, a telemetry dump and NO mp4 -- the
                # one artifact you cannot reconstruct afterwards, because the
                # qpos trace lives only in this process. Silent, because
                # nothing failed: the video was simply never asked for.
                self.set_status("stopping: --episodes %d reached"
                                % self.args.episodes)
                self.render()
                self.quit = True
                break
            with self.lock:
                skipped, reset = self._skip, self._reset
                self._skip = False
            if self.submode == "automode" and not (skipped or reset):
                # THE RING STAYS IN THE CURRENT MODE. Cycling brace ->
                # recover in `elbow+forearm` is a round trip; cycling across
                # modes would silently change the experiment between laps.
                cur_tasks = self.tasks
                nxt = [n for n in self._auto
                       if n in cur_tasks and cur_tasks[n].ready]
                if len(nxt) > 1:
                    i = (nxt.index(self.task_name) + 1) % len(nxt)
                    self.task_name = nxt[i]
                continue            # chained: the robot is not put back
            if not (skipped or reset):
                # Render between episodes, never during one. In automode the
                # next task starts immediately instead -- a 3 s render dropped
                # into every loop iteration turns a continuous demo into a
                # slideshow; the button is there for when you want it.
                self.render()
                self.idle()
        self.set_status("stopped")



def make_monitor(args):
    """The safety monitor, or None. Failure to build one is fatal on purpose.

    A monitor that quietly did not load would be worse than no monitor: the
    panel would show no red rules and you would read that as "the limits were
    never crossed" rather than "nobody looked".
    """
    if not args.safety:
        return None
    from croco.safety import SafetyMonitor
    from croco.plant.dds_plant import JOINT_NAMES
    mon = SafetyMonitor(args.safety, joint_names=JOINT_NAMES)
    print("[croco_twin] safety monitor: %s (%s). Observing only -- it never "
          "touches a command." % (mon.config_path, mon.mode))
    return mon


def _make_plant(args, m, tau_lim):
    """The plant and (for DDS) its base source. Shared by both entry points."""
    if args.plant == "mujoco":
        from croco.plant.mujoco_plant import MuJoCoPlant
        import mujoco as _mj
        m2, d2 = cs.load(ik_margin=0.0)
        cr.show_gripper(m2)      # visual only; the jaws still have no dynamics
        _mj.mj_forward(m2, d2)
        return MuJoCoPlant(m2, d2, sense=None, tau_limit=tau_lim, nu=27), None
    plant = DDSPlant(network_interface=args.iface, domain_id=args.domain,
                     twin_dt=float(m.opt.timestep), base_source=None,
                     cmd_topic=(TOPIC_SAFETY_LOWCMD_IN if args.via_safety
                                else "rt/lowcmd"),
                     tau_limit=tau_lim,
                     q_range=(m.jnt_range[1:28, 0].copy(),
                              m.jnt_range[1:28, 1].copy()),
                     recv=args.recv)
    base = (TruthBase(recv=args.recv) if args.base == "truth"
            else EstimatorBase(args.est_topic, recv=args.recv))
    plant.base_source = base
    print("[croco_twin] waiting for the twin ...")
    plant.wait_for_state(timeout=15.0)
    if args.base == "estimator":
        base.attach(plant)
    base.wait(timeout=15.0)
    return plant, base


def run_session(args):
    """The interactive entry point: a panel, a session, and episodes.

    ONLY `--gui` TAKES THIS PATH. Without it croco_twin is still the one-shot
    script that builds an OCP, runs one loop, prints a JSON summary and exits
    -- which is what twin_grid.sh drives 26 times in a row and what every
    recorded result in this study was produced by. A session that idles
    waiting for a browser would hang all of that, so the batch behaviour is
    not merely preserved, it is the default.
    """
    from croco.gui import Panel

    m, _d = cs.load(ik_margin=0.0)
    assert_joint_order(m, nu=27)
    kp, kd = cr.servo_gains(m)
    tau_lim = cs.torque_limits(m)

    # THE CELL DECIDES WHAT IS ON OFFER. Every (kind, mode) it has a plan for
    # becomes a registry entry; the dropdowns are a view of this, not a table.
    found, bad = discover_plans(args.dir)
    for tag, why in sorted(bad.items()):
        print("[croco_twin] ignoring plan_%s.json: %s" % (tag, why))
    modes = sorted({mode for (kind, mode) in found
                    if kind == "brace+reach" and mode})
    if not modes:
        raise SystemExit(
            "%s holds no braced plan -- there is nothing to run. A task is a "
            "SOLVED PLAN; solve one with:  studies/solve_tasks.sh %s"
            % (args.dir, args.dir))

    # WHICH MODE THE SESSION STARTS ON. --mode wins; otherwise --tag names a
    # plan file and that plan's own mode is the answer, which keeps every
    # existing `--tag elbow_palm` command line meaning exactly what it did.
    mode0 = args.mode
    if mode0 is None:
        by_tag = {t: k for k, t in found.items()}
        key = by_tag.get(args.tag)
        mode0 = key[1] if key and key[1] else modes[0]
    if mode0 not in modes:
        raise SystemExit("--mode %s: this cell has %s"
                         % (mode0, ", ".join(modes)))
    args.mode = mode0

    # ONE Task PER (kind, mode), AT THE SESSION'S Kp. Other Kp values get
    # their own entries lazily, when the panel asks for them -- see
    # Session._ensure.
    tasks = {}
    for (kind, mode), tag in sorted(found.items(), key=lambda kv: str(kv[0])):
        note = dict(KINDS).get(kind, "")
        if mode:
            note = "%s -- %s (plan_%s.json)" % (note, mode, tag)
        tasks[(kind, mode, args.contact_kp)] = Task(
            kind, args.dir, tag, note, mode=mode, kp=args.contact_kp)
    task_order = [n for n, _ in KINDS]

    start = tasks.get((args.task, NO_MODE if args.task == "stand" else mode0,
                       args.contact_kp))
    if start is None or not start.ready:
        raise SystemExit(
            "--task %s --mode %s is not solved in this cell.\nThis cell holds: "
            "%s" % (args.task, mode0,
                    ", ".join("%s/%s" % (k, m or "-")
                              for (k, m) in sorted(found, key=str))))

    # dt comes off the plan JSON, which is cheap to read -- the panel must
    # exist BEFORE the OCP build (measured 1.2 s for 200 models), not after
    # it, or the wait looks exactly like a hung page.
    dt_plan = json.load(open(start.plan_path))["dt"]

    monitor = make_monitor(args)
    plant, _base = _make_plant(args, m, tau_lim)
    ov = ocp_overrides(args)
    panel = Panel(None, port=args.gui, period_ms=1e3 * dt_plan,
                  config=dict(
                      plant=args.plant, base=args.base, cell=args.dir,
                      tag=args.tag, horizon=args.horizon, iters=args.iters,
                      threads=args.threads,
                      dds=(None if args.plant == "mujoco"
                           else "domain %d / %s" % (args.domain, args.iface)),
                      cmd_topic=getattr(plant, "cmd_topic", "(in process)"),
                      safety=(None if monitor is None
                              else os.path.basename(monitor.config_path)),
                      ocp_overrides=(None if not ov else
                                     ", ".join("%s=%s" % kv for kv in
                                               sorted(ov.items()))),
                      video=args.video,
                      gl=os.environ.get("MUJOCO_GL", "(default)")))
    # ORDER: the monitor annotates the row, the panel serialises it. Reversed,
    # the browser would get every row one period before its verdict.
    hooks = ([] if monitor is None else [monitor]) + [panel.on_step]
    session = Session(args, tasks, plant, m, kp, kd, tau_lim, hooks=hooks)
    session.task_order = task_order
    session.modes = modes
    session.monitor = monitor
    session.panel = panel
    panel.on_command = session.command
    session.push()
    print("[croco_twin] panel on %s -- the session is interactive: reset, "
          "pause, task and speed are all in the browser." % panel.url)
    if args.viewer:
        session.command("viewer", dict(on=True))
    try:
        session.run()
    except KeyboardInterrupt:
        session.quit = True
    finally:
        session.close_viewer()
        try:
            plant.close()
        except Exception:                                        # noqa: BLE001
            pass
    if args.out and session.runs:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                    exist_ok=True)
        json.dump(dict(session=session.state(), runs=session.runs,
                       **panel.summary()), open(args.out, "w"), indent=1)
        print("[croco_twin] wrote %s (%d episode(s))"
              % (args.out, len(session.runs)))
    for r in session.runs[-3:]:
        print("[croco_twin] " + json.dumps(r))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="grid cell / run directory")
    ap.add_argument("--tag", default="elbow_forearm",
                    help="which brace plan file the session starts on "
                         "(plan_<tag>.json). elbow+palm was the default for "
                         "most of this study and is no longer: its palm "
                         "carries 0.0 N for an entire hold (the gripper is "
                         "9-33 mm above the wood at q*), and elbow+forearm "
                         "beats it on sink, drift, second-contact force and "
                         "peak torque. Prefer --mode, which names the contact "
                         "set rather than a filename.")
    ap.add_argument("--horizon", type=int, default=35)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--alphas", type=int, default=0)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--domain", type=int, default=1)
    ap.add_argument("--iface", default="lo")
    ap.add_argument("--plant", choices=["dds", "mujoco"], default="dds",
                    help="'mujoco' runs the SAME ControlLoop and the SAME "
                         "policy against in-process physics, i.e. with zero "
                         "latency and no wire. It is the control for this "
                         "experiment: if the maneuver survives there and dies "
                         "over DDS, the deployment is what broke it; if it dies "
                         "in both, the bug is in this file and not on the wire.")
    ap.add_argument("--est-topic", default="rt/sportmodestate_est",
                    help="where base_estimator_node_v4.py publishes")
    ap.add_argument("--est-error", action="store_true",
                    help="with --base estimator: also subscribe the twin's "
                         "ground truth and record the estimator's error per "
                         "period. MEASUREMENT ONLY -- the truth never reaches "
                         "the controller, which is why this is a separate flag "
                         "from --base truth. Off by default because a run that "
                         "touches ground truth has to say so.")
    ap.add_argument("--base", choices=["truth", "estimator", "none"],
                    default="none",
                    help="'truth' reads the TWIN'S GROUND TRUTH base pose. "
                         "There is no estimator in this loop yet, so 'none' "
                         "cannot run the MPC -- it is here to make the "
                         "dependency explicit rather than implicit.")
    ap.add_argument("--bringup", action="store_true",
                    help="run the warmup/ramp/hold/blend phases before the "
                         "maneuver, as the MJPC deploy node does on hardware")
    ap.add_argument("--stale-ms", type=float, default=50.0,
                    help="watchdog threshold. The MJPC deploy node's 50 ms was "
                         "chosen for a 200 Hz loop; this one runs at 50 Hz with "
                         "a ~16 ms solve, so the two are not obviously the same "
                         "setting. Raise it to test whether a fall is the "
                         "watchdog or the latency -- not to make it go away.")
    ap.add_argument("--recv", default="poll", choices=("poll", "callback"),
                    help="how lowstate/sim_state are received. `poll` takes the "
                         "newest sample in the control thread; `callback` is "
                         "unitree_sdk2py's listener+queue threads, which starve "
                         "while the solver holds the GIL. Kept only for the A/B.")
    ap.add_argument("--gui", nargs="?", type=int, const=8770, default=None,
                    metavar="PORT",
                    help="serve the live panel (default port 8770): solve time "
                         "against the period, cost per term, state age, and "
                         "sliders for every cost weight. Weight edits are "
                         "applied BETWEEN periods and recorded in --out, "
                         "because a retuned run that does not say so is not "
                         "reproducible.")
    ap.add_argument("--realtime", type=float, default=None, metavar="FACTOR",
                    help="sim seconds per wall second. Unset keeps each "
                         "plant's own default: free-run in process, real time "
                         "over DDS. 1.0 pins the in-process run to real time; "
                         "0.25 runs it at quarter speed, which is what makes "
                         "--viewer watchable and what hands the solver 80 ms "
                         "of wall clock per 20 ms period. BELOW 1.0 THE "
                         "OVERRUN COUNT IS NO LONGER A DEPLOYMENT RESULT -- "
                         "the robot has no such knob -- so it is recorded in "
                         "--out and printed, like --base truth. Over DDS the "
                         "twin must be started with the MATCHING "
                         "`lean_twin --realtime`; it owns the far clock and "
                         "this flag cannot reach it.")
    ap.add_argument("--viewer", action="store_true",
                    help="open MuJoCo's passive viewer on the in-process "
                         "plant (--plant mujoco only; the DDS plant has no "
                         "local physics to show -- run `lean_twin --viewer` "
                         "on that side instead). Needs a windowing GL "
                         "backend: MUJOCO_GL=egl is offscreen and will not "
                         "open a window. Syncing costs the control thread a "
                         "few ms per period, so a viewer run is recorded as "
                         "one.")
    ap.add_argument("--task", default="brace+reach",
                    # NOT `choices=`: a cell's extra plans are discovered at
                    # startup and their mode names are not knowable from argv.
                    # An unknown task is rejected by run_session with the list
                    # the cell actually holds, which is the more useful error.
                    metavar="NAME",
                    help="which maneuver the session starts on. A task is a "
                         "SOLVED PLAN in the cell directory, not a set of "
                         "weights: the phases differ by contact set, so "
                         "switching rebuilds the OCP. `stand` and `recover` "
                         "need plan_stand/plan_recover artifacts, which the "
                         "certified grid does not carry (every cell has "
                         "n_return=0).")
    ap.add_argument("--mode", default=None, metavar="MODE",
                    help="which CONTACT MODE the session starts on -- "
                         "`elbow+forearm`, `elbow+palm`, `elbow`, ... A mode "
                         "is available when the cell holds a solved plan for "
                         "it; the panel switches between them live, with the "
                         "robot left braced where it is. Default: the mode of "
                         "the plan --tag names, which keeps every existing "
                         "command line meaning what it did.")
    ap.add_argument("--submode", default="single-shot", choices=SUBMODES,
                    help="single-shot: run the plan once, then hold position "
                         "and wait for reset. hold: run it, then FREEZE the "
                         "plan index at the last node and keep solving -- the "
                         "brace stays braced. automode: loop the available "
                         "tasks back to back.")
    ap.add_argument("--no-video", action="store_true",
                    help="do not render an mp4 (rendering is on by default)")
    ap.add_argument("--video", default=None, metavar="PATH",
                    help="render an mp4 of the run to PATH. States are "
                         "buffered during the loop (a qpos memcpy per period) "
                         "and rendered AFTER it finishes, so the render "
                         "cannot cost a control period. With --gui the video "
                         "is served in the panel when it is ready.")
    ap.add_argument("--video-cam", default="wide",  # noqa: E128
                    choices=sorted(cr.CAMERAS), help="croco_replay camera preset")
    ap.add_argument("--video-fps", type=int, default=30)
    # -- the OCP, overridden ------------------------------------------------
    ap.add_argument("--reach-rot", default=None,
                    choices=["auto", "flat", "down", "side", "none"],
                    help="REBUILD the OCP with this gripper-orientation "
                         "reference instead of the plan's. The certified grid "
                         "was solved with w_reach_rot = 0, so those cells have "
                         "no `reachRot` cost at all -- no slider, and the "
                         "roll knob disabled, because a cost cannot be added "
                         "to a model that is already built. `auto` points the "
                         "reference at the orientation q* already reaches, "
                         "which at a token weight changes the plan by ~4e-5 "
                         "in cost while making the term EXIST and therefore "
                         "steerable. The run artifact records the override: "
                         "the warm start still comes from a plan solved "
                         "without it.")
    ap.add_argument("--w-reach-rot", type=float, default=None, metavar="W",
                    help="weight for --reach-rot (default 1e-2, the token "
                         "weight the term is meant to be steered up from)")
    ap.add_argument("--contact-kp", type=float, default=None, metavar="KP",
                    help="REBUILD the OCP with this Baumgarte position gain on "
                         "the brace contacts, instead of the plan's. This is "
                         "the knob that decides whether a hold is indefinite: "
                         "at the study's long-standing Kp=0 a contact "
                         "constrains velocity only, so the site keeps whatever "
                         "position error it accumulated and the brace creeps "
                         "in every contact mode. The warm start still comes "
                         "from a plan solved at the plan's own Kp, so a large "
                         "override is a divergence and is recorded as one in "
                         "--out. Also live in the panel (it rebuilds).")
    ap.add_argument("--contact-kd", type=float, default=None, metavar="KD",
                    help="Baumgarte velocity gain (plan default 50). Kd = "
                         "2*sqrt(Kp) is critical damping for the contact "
                         "error's second-order response.")
    ap.add_argument("--foot-kp", type=float, default=None, metavar="KP",
                    help="the same position gain on the FEET (default 0). "
                         "Opt-in separately -- see croco_run --foot-kp.")

    # -- the safety layer ---------------------------------------------------
    ap.add_argument("--safety", nargs="?", const="default_safety_full",
                    default=None, metavar="CONFIG",
                    help="MONITOR the h12_safety_layer's limits every period "
                         "and mark the periods its e-stop would have tripped "
                         "-- red rules on the panel's period plot, "
                         "`safety_*` fields in --out. Takes a path or a bare "
                         "name from that package's config/ (default "
                         "default_safety_full). This is an OBSERVER: it never "
                         "touches a command. Off by default because the "
                         "limits are the robot's and a study run on the twin "
                         "legitimately explores past them.")
    ap.add_argument("--via-safety", action="store_true",
                    help="publish commands to %s instead of rt/lowcmd, so a "
                         "RUNNING h12_safety_layer clips them and forwards to "
                         "rt/lowcmd. --plant dds only. This one has "
                         "authority: the layer's e-stop LATCHES, so the first "
                         "trip zeroes kp/kd/tau for the rest of the session "
                         "and the robot goes limp. Start with --safety alone "
                         "to find out whether it would trip." % TOPIC_SAFETY_LOWCMD_IN)

    ap.add_argument("--no-telemetry", dest="telemetry", action="store_false",
                    help="do not compute the per-period physics telemetry "
                         "(brace forces, drift, support margin, attitude). It "
                         "is a handful of MuJoCo queries on the control "
                         "thread -- cheap, but it IS on the control thread, "
                         "so it is switchable. In-process plant only; over "
                         "DDS there is no local mjData to query and the "
                         "section is absent either way.")
    ap.add_argument("--seam", default="project", choices=["project", "node0"],
                    help="how a CHAINED behaviour picks its starting plan "
                         "index. `project`: the node whose pose the measured "
                         "state is nearest, so the reference tape is joined "
                         "where the robot already is. `node0`: always node 0, "
                         "which is what every run before this flag did -- "
                         "kept so the A/B is one flag and not one checkout. "
                         "Irrelevant on a reset, where the plant is put at "
                         "node 0 by construction.")
    ap.add_argument("--episodes", type=int, default=None, metavar="N",
                    help="stop the session after N episodes (--gui sessions "
                         "otherwise run until the browser or Ctrl-C says "
                         "stop, which is not a thing a batch measurement can "
                         "do). --out is still written.")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--emit-qpos0", default=None,
                    help="write the plan's start qpos to this file and exit. "
                         "Feed it to `lean_twin --qpos0`: the twin must begin "
                         "where the plan begins, and no keyframe is that pose.")
    args = ap.parse_args()

    # A SEGFAULT SHOULD NAME ITSELF. This process links crocoddyl, pinocchio,
    # MuJoCo and a GL driver; when one of them faults, the default is a bare
    # "Segmentation fault (core dumped)" and a guessing game. faulthandler
    # costs nothing until then and prints the Python stack of every thread,
    # which is what says whether the control thread was in mj_step, the render
    # thread was in the viewer, or the solve was in crocoddyl.
    faulthandler.enable()

    # VALIDATE BEFORE BUILDING. `build` spends ~25 s on the OCP and the DDS
    # plant then waits 15 s for a twin, so a guard placed next to the code it
    # guards told you about an unusable flag combination forty seconds after
    # you typed it. These two cost nothing and are knowable from argv alone.
    if args.viewer:
        if args.plant != "mujoco":
            raise SystemExit(
                "--viewer needs the in-process plant (--plant mujoco). The "
                "DDS plant has no local physics to draw; run the twin with "
                "`python -m croco.twin.lean_twin --viewer` instead.")
        gl = os.environ.get("MUJOCO_GL", "").lower()
        if gl in ("egl", "osmesa"):
            raise SystemExit(
                "MUJOCO_GL=%s is an OFFSCREEN backend and cannot open a "
                "window -- launch_passive would fail or draw nothing. Unset "
                "it (or set MUJOCO_GL=glfw) for --viewer. Note run_session.sh "
                "exports egl, so a shell that sourced it carries this." % gl)

    if args.via_safety and args.plant != "dds":
        raise SystemExit(
            "--via-safety is a DDS topic change (%s instead of rt/lowcmd) and "
            "means nothing to the in-process plant, which has no wire and no "
            "safety layer on it. Use --plant dds, and start the layer:\n"
            "  python -m h12_safety_layer.script.safety_layer_main "
            "--config default_safety_full.yaml\n"
            "For the in-process plant, --safety monitors the same limits "
            "without needing the layer to run." % TOPIC_SAFETY_LOWCMD_IN)

    # VIDEO IS ON BY DEFAULT IN THE SESSION, next to the replay mp4 the cell
    # already carries (`replay_<tag>_mpc.mp4`) -- the two are the same kind of
    # artifact and belong side by side: one is what the plan does, the other
    # is what the loop did. --no-video opts out.
    #
    # NOT in the batch path, deliberately. twin_grid.sh runs this 26 times and
    # a default there would silently add 26 renders and 26 files to a
    # certified grid, changing what a batch run produces as a side effect of a
    # GUI convenience. Batch keeps --video opt-in, as it was.
    args.video_auto = False
    if args.no_video:
        args.video = None
    elif args.video is None and args.gui:
        args.video = os.path.join(args.dir, "twin_%s.mp4" % args.tag)
        args.video_auto = True      # Session.render re-derives it per mode

    if args.emit_qpos0:
        plan = json.load(open(os.path.join(args.dir, "plan_%s.json" % args.tag)))
        xs = np.load(os.path.join(args.dir, "xs_%s.npy" % args.tag))
        m, _ = cs.load(ik_margin=0.0)
        q = cb.pin_to_mj(xs[0][:cb.NQ_ROBOT], cs.start_qpos(m, plan["start"]))
        os.makedirs(os.path.dirname(os.path.abspath(args.emit_qpos0)) or ".",
                    exist_ok=True)
        np.savetxt(args.emit_qpos0, q)
        print("[croco_twin] wrote %s (%d) -- pass it to lean_twin --qpos0"
              % (args.emit_qpos0, q.size))
        return 0

    if args.gui:
        return run_session(args)

    if args.plant == "mujoco":
        args.base = "truth"          # in-process physics IS the truth
    if args.base == "none":
        raise SystemExit(
            "--base none: the lean OCP needs a floating-base pose and "
            "rt/lowstate does not carry one. Pass --base estimator to run "
            "against the fork's proprioceptive estimator (what the robot would "
            "use), or --base truth to hold the estimator at perfect while "
            "testing something else -- and say which in whatever you report.")

    truth_probe = None
    monitor = make_monitor(args)      # before the build: a bad config is argv
    plan, mpc, xs, us = build(args)
    dt_plan = plan["dt"]
    nq = cb.NQ_ROBOT

    # Gains and limits come off the SAME model the plan was solved against.
    m, _d = cs.load(ik_margin=0.0)
    assert_joint_order(m, nu=27)
    kp, kd = cr.servo_gains(m)
    tau_lim = cs.torque_limits(m)

    if args.plant == "mujoco":
        from croco.plant.mujoco_plant import MuJoCoPlant
        m2, d2 = cs.load(ik_margin=0.0)
        q0 = cb.pin_to_mj(xs[0][:nq], cs.start_qpos(m2, plan["start"]))
        d2.qpos[:] = q0
        d2.qvel[:] = 0.0
        import mujoco as _mj
        _mj.mj_forward(m2, d2)
        plant = MuJoCoPlant(m2, d2, sense=None, tau_limit=tau_lim, nu=27)
        base = None
    else:
        pass
    # ORDER MATTERS: DDSPlant is what calls ChannelFactoryInitialize, and a
    # subscriber built before the participant exists fails inside cyclonedds as
    # "'NoneType' object has no attribute '_ref'", which names neither the
    # participant nor the ordering.
    if args.plant == "dds":
        # THE PLAN INDEXES ON THE TWIN'S CLOCK, NOT THE WALL CLOCK. `twin_dt`
        # was None, which takes DDSPlant's wall-clock fallback -- exactly the
        # drift its own docstring warns about ("pacing a sim-coupled
        # controller on the wall clock makes its plan index drift against the
        # plant whenever the sim is not real-time"). It went unnoticed because
        # the twin had only ever run at 1.0x, where the two clocks agree.
        # --realtime is what made it visible: at 0.5x the controller played
        # all 199 plan nodes in 4 s of WALL clock against a world that had
        # advanced 2 s, i.e. the maneuver ran at double speed relative to the
        # physics, and the improved landing that produced is an artifact, not
        # a longer solve budget. lean_twin publishes `tick = d.time/timestep`,
        # so tick * timestep IS the twin's sim time, exactly.
        plant = DDSPlant(network_interface=args.iface, domain_id=args.domain,
                         twin_dt=float(m.opt.timestep),
                         base_source=None, tau_limit=tau_lim,
                         q_range=(m.jnt_range[1:28, 0].copy(),
                                  m.jnt_range[1:28, 1].copy()),
                         cmd_topic=(TOPIC_SAFETY_LOWCMD_IN if args.via_safety
                                    else "rt/lowcmd"),
                         recv=args.recv)
        if args.via_safety:
            print("[croco_twin] commands go to %s -- the h12_safety_layer must "
                  "be RUNNING and republishing to rt/lowcmd, or the robot "
                  "receives nothing at all." % TOPIC_SAFETY_LOWCMD_IN)
        base = (TruthBase(recv=args.recv) if args.base == "truth"
                else EstimatorBase(args.est_topic, recv=args.recv))
        plant.base_source = base
        print("[croco_twin] waiting for the twin ...")
        plant.wait_for_state(timeout=15.0)
        if args.base == "estimator":
            base.attach(plant)
            if args.est_error:
                truth_probe = TruthBase(recv=args.recv)
                truth_probe.wait(timeout=15.0)
        base.wait(timeout=15.0)
        print("[croco_twin] twin is up: lowstate + %s"
              % ("rt/sim_state (GROUND TRUTH base)" if args.base == "truth"
                 else "%s (PROPRIOCEPTIVE estimate; attitude from the IMU)"
                      % args.est_topic))

    stats = dict(steps=0, mpc_none=0)
    est_err = []            # |p_est - p_true| per period, measurement only
    first = {}

    def policy(t, st):
        """(q_des, v_des, tau_ff) for the plant's joints, from the MPC.

        `k` is derived from the plant's clock rather than counted, so a missed
        period advances the plan by a period instead of replaying it -- the
        maneuver is a function of time, not of how many times we managed to
        solve.
        """
        k = int(round(t / dt_plan))
        if k >= len(us):
            return None
        if not first:
            # WHERE IS THE ROBOT WHEN THE MANEUVER STARTS? The plan assumes x0.
            # The twin has been holding a pose for as long as this process took
            # to build its OCP -- tens of seconds -- and a hold is not a freeze:
            # the floating base is not held by anything. If the robot has crept,
            # the maneuver begins from somewhere it was never planned from, and
            # that is an initial-condition failure wearing a controller's
            # clothes.
            q0p = cb.pin_to_mj(xs[0][:nq], cs.start_qpos(m, plan["start"]))
            first["dq_max_rad"] = float(np.max(np.abs(st.q - q0p[7:34])))
            first["dq_rms_rad"] = float(np.sqrt(np.mean((st.q - q0p[7:34]) ** 2)))
            first["dbase_mm"] = float(1e3 * np.linalg.norm(st.base_pos - q0p[0:3]))
            first["dquat"] = float(np.linalg.norm(st.base_quat - q0p[3:7]))
            first["v_max"] = float(np.max(np.abs(st.v)))
            print("[croco_twin] at first command: dq_max %.4f rad  dq_rms %.4f  "
                  "base %.1f mm  dquat %.4f  |v|max %.3f rad/s"
                  % (first["dq_max_rad"], first["dq_rms_rad"], first["dbase_mm"],
                     first["dquat"], first["v_max"]))
        if truth_probe is not None:
            got = truth_probe()
            if got is not None:
                est_err.append(np.asarray(st.base_pos - got[0], float))
        qpos = np.concatenate([st.base_pos, st.base_quat, st.q])
        R = _quat_to_mat(st.base_quat)
        qvel = np.concatenate([st.base_linvel, st.base_angvel, st.v])
        x_meas = np.concatenate([cb.mj_to_pin(qpos), cb.mj_to_pin_v(qvel, R)])
        u0, xs1 = mpc(min(k, len(us) - 1), x_meas)
        stats["steps"] += 1
        if u0 is None:
            stats["mpc_none"] += 1
            return xs[k][7:nq], np.zeros(27), us[k]
        return xs1[:nq][7:], xs1[nq:][6:], np.clip(u0, -tau_lim, tau_lim)

    # -- the live viewer ---------------------------------------------------
    # In-process only, and it says so rather than silently showing nothing:
    # over DDS the physics is in the twin's process and `lean_twin --viewer` is
    # the flag that reaches it.
    viewer = viewer_thread = None
    if args.viewer:                       # already validated against argv above
        import mujoco.viewer as _mjv
        cr.show_gripper(plant.m)      # visual only; the jaws have no dynamics
        # CAPTURE THE VIEWER'S THREAD SO IT CAN BE JOINED. `Handle.close()`
        # only calls `sim.exit()` -- it SIGNALS the render loop and returns
        # immediately, and mujoco keeps the thread private and daemonic. So
        # the interpreter exits while C++ is still destroying the GL context,
        # and the process dies on the way out: measured here as a reliable
        # SIGSEGV on close-then-exit and a `terminate called without an active
        # exception` abort under the `with` form, on every attempt, with and
        # without this file's RTLD_GLOBAL. It happens AFTER the run, so the
        # numbers and the video are already written and correct -- which is
        # what makes it worth fixing rather than living with: a study tool
        # that core-dumps at exit turns every wrapper script's `|| true` into
        # a place a real failure can hide.
        with plant.data_lock:
            viewer = _mjv.launch_passive(plant.m, plant.d)
        viewer_thread = _viewer_thread_of(_mjv)
        plant.viewer_lock = viewer.lock      # every mjData write goes under it
        print("[croco_twin] passive viewer open. At %s the maneuver is %.1f s "
              "of wall clock -- pass --realtime 0.25 to watch it."
              % ("free-run" if args.realtime is None
                 else "%.2gx" % args.realtime, len(us) * dt_plan
                 / (args.realtime or 1.0)))

    # -- state recorder for the video --------------------------------------
    # A qpos copy per period and nothing else. Rendering here would put a
    # 5-15 ms Renderer call inside a 20 ms period, which is how you measure a
    # deployment failure you caused yourself.
    qtrace = []
    qtmpl = None
    if args.video:
        qtmpl = cs.load(ik_margin=0.0)[1].qpos.copy()

    def record(row, st, cmd):
        if args.plant == "mujoco":
            qtrace.append(plant.d.qpos.copy())
        else:
            q = qtmpl.copy()
            q[0:3], q[3:7] = st.base_pos, st.base_quat
            q[7:7 + 27] = st.q
            qtrace.append(q)

    cfg = LoopConfig(ctrl_hz=1.0 / dt_plan, stale_s=args.stale_ms * 1e-3,
                     realtime=args.realtime)
    stance = cs.start_qpos(m, plan["start"])[7:] if args.bringup else None
    panel = None
    if args.gui:
        from croco.gui import Panel
        panel = Panel(mpc, port=args.gui, period_ms=1e3 * dt_plan)
        print("[croco_twin] panel on %s -- open it before the maneuver starts, "
              "the run is only %.1f s long" % (panel.url, len(us) * dt_plan))
    # One hook, several consumers. Each is individually optional and none of
    # them may raise into the control thread.
    hooks = []
    if monitor is not None:
        hooks.append(monitor)         # first: it annotates what the rest read
    if panel is not None:
        hooks.append(panel.on_step)
    if args.video:
        hooks.append(record)
    if viewer is not None:
        hooks.append(lambda row, st, cmd: viewer.sync())

    def on_step(row, st, cmd):
        for h in hooks:
            try:
                h(row, st, cmd)
            except Exception:                                    # noqa: BLE001
                pass

    loop = ControlLoop(plant, policy, stance=stance, cfg=cfg,
                       on_step=on_step if hooks else None)
    print("[croco_twin] %.0f Hz, horizon %d, %d iter(s), %d thread(s), "
          "%s bring-up" % (cfg.ctrl_hz, args.horizon, args.iters, args.threads,
                           "with" if args.bringup else "no"))
    t0 = time.monotonic()
    try:
        log = loop.run(kp, kd, max_seconds=args.max_seconds)
    except KeyboardInterrupt:
        log = loop.log
    finally:
        plant.close()
        if viewer is not None:
            viewer.close()
            if viewer_thread is not None:
                viewer_thread.join(timeout=5.0)

    if args.plant == "mujoco":
        print("[croco_twin] in-process outcome: pelvis z %.4f m  %s"
              % (plant.d.qpos[2], "FELL" if plant.d.qpos[2] < 0.55 else "upright"))
    solves = [r["solve_ms"] for r in log if "solve_ms" in r]
    ages = [1e3 * r["age"] for r in log if "age" in r]
    out = dict(
        wall_s=time.monotonic() - t0,
        periods=len(log), mpc_steps=stats["steps"],
        overruns=getattr(loop, "overruns", None),
        worst_overrun_ms=1e3 * getattr(loop, "worst_overrun_s", 0.0),
        watchdog_trips=getattr(loop, "watchdog_trips", None),
        safe_periods=sum(1 for r in log if r.get("phase") == "safe"),
        tau_saturated=sum(r.get("tau_sat", 0) for r in log),
        q_clipped=sum(r.get("q_clip", 0) for r in log),
        solve_ms_mean=float(np.mean(solves)) if solves else None,
        solve_ms_p95=float(np.percentile(solves, 95)) if solves else None,
        age_ms_p50=float(np.percentile(ages, 50)) if ages else None,
        age_ms_p95=float(np.percentile(ages, 95)) if ages else None,
        age_ms_max=float(np.max(ages)) if ages else None,
        stale_ms=args.stale_ms,
        nthreads_effective=int(mpc.problem.nthreads),
        recv=args.recv,
        cmd_topic=getattr(plant, "cmd_topic", None),
        via_safety=bool(args.via_safety),
        # A run whose OCP is not the one the plan was solved with says so in
        # its own artifact, or the grid quietly stops being comparable.
        ocp_overrides=(ocp_overrides(args) or None),
        **({} if monitor is None else monitor.summary()),
        **({} if panel is None else panel.summary()),
        recv_samples_per_poll=(
            None if getattr(plant, "recv_polls", 0) == 0
            else round(plant.recv_samples / plant.recv_polls, 2)),
        recv_empty_polls=getattr(plant, "recv_empty", None),
        est_err_mm_p50=(None if not est_err else float(
            1e3 * np.percentile(np.linalg.norm(est_err, axis=1), 50))),
        est_err_mm_p95=(None if not est_err else float(
            1e3 * np.percentile(np.linalg.norm(est_err, axis=1), 95))),
        est_err_mm_max=(None if not est_err else float(
            1e3 * np.max(np.linalg.norm(est_err, axis=1)))),
        est_err_mm_xyz=[[round(1e3 * c, 2) for c in e] for e in est_err],
        # The pacing belongs NEXT TO the overrun count it qualifies. A reader
        # who sees `overruns: 0` without seeing that the run was quarter speed
        # has been told something false by omission.
        realtime=args.realtime,
        realtime_note=(None if args.realtime is None or args.realtime >= 1.0
                       else "SLOWED to %.3gx: the solver had %.0f ms of wall "
                            "clock per %.0f ms control period. Overruns here "
                            "are NOT a deployment result." %
                            (args.realtime, 1e3 * dt_plan / args.realtime,
                             1e3 * dt_plan)),
        viewer=bool(viewer is not None),
        base_source=("GROUND TRUTH (rt/sim_state)" if args.base == "truth"
                     else "estimator (%s) + IMU attitude" % args.est_topic))
    if args.video:
        if not qtrace:
            print("[croco_twin] --video: no states were recorded (the loop "
                  "never reached a commanded period); nothing to render.")
        else:
            print("[croco_twin] rendering %d states -> %s"
                  % (len(qtrace), args.video))
            got = render_run(qtrace, plan, args.dir, args.video,
                             cam=args.video_cam, fps=args.video_fps,
                             dt_plan=dt_plan)
            out["video"] = got
            if got and panel is not None:
                panel.set_video(got)
                print("[croco_twin] video is in the panel at %s" % panel.url)

    if args.realtime is not None and args.realtime < 1.0:
        print("[croco_twin] SLOWED to %.3gx real time. The overrun count "
              "below is a counterfactual, not a deployment result."
              % args.realtime)
    if panel is not None and panel.dirty:
        print("[croco_twin] WEIGHTS WERE CHANGED LIVE (%d edits). This run is "
              "NOT the plan's cost function; see gui_weight_changes in --out."
              % len(panel.changes))
    print("[croco_twin] " + json.dumps(out, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(dict(summary=out, log=log), open(args.out, "w"), indent=1)
        print("[croco_twin] wrote %s" % args.out)
    if panel is not None:
        # HOLD THE PANEL OPEN. The maneuver is four seconds long and the
        # process would otherwise exit before a browser could finish loading
        # the page, which is how the first run of this was measured as
        # "HTTP 000". The run is over; the numbers are what you came to look at.
        print("[croco_twin] panel still serving at %s -- Ctrl-C to exit"
              % panel.url)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
        panel.close()
    return 0


def _quat_to_mat(q):
    import mujoco
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


if __name__ == "__main__":
    raise SystemExit(main())
