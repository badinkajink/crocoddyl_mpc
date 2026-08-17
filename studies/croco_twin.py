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
import ctypes
import json
import os
import sys
import threading
import time

sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

import numpy as np                                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "croco_ext"))
sys.path.insert(0, os.path.join(HERE, ".."))

import croco_bridge as cb                                       # noqa: E402
import contact_select as cs                                     # noqa: E402
import croco_replay as cr                                       # noqa: E402

from croco.control.mpc import MPC                               # noqa: E402
from croco.plant.dds_plant import (DDSPlant, PollingReceiver,   # noqa: E402
                                   assert_joint_order)
from croco.runtime.loop import ControlLoop, LoopConfig          # noqa: E402


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
def build(args):
    """The MPC and the reference plan, exactly as croco_replay builds them."""
    plan = json.load(open(os.path.join(args.dir, "plan_%s.json" % args.tag)))
    ocp, _ = cr.build_ocp(plan, args.dir)
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
TASKS = [
    ("brace+reach", "the certified maneuver: approach, brace, reach"),
    ("stand",       "legs only, no brace (subset=[])"),
    ("recover",     "brace released, back to the start pose (n_return>0)"),
]
SUBMODES = ["single-shot", "hold", "automode"]


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

    def __init__(self, name, run_dir, tag, note=""):
        self.name, self.run_dir, self.tag, self.note = name, run_dir, tag, note
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
        if self.ready:
            return None
        return ("no %s in %s -- this task has never been solved for this cell"
                % (os.path.basename(self.plan_path), self.run_dir))

    def build(self, args):
        """Build the OCP + MPC for this task. Slow (~20 s); call off-thread."""
        if self.mpc is not None:
            return self
        if not self.ready:
            raise FileNotFoundError(self.why_unavailable())
        ns = argparse.Namespace(**vars(args))
        ns.dir, ns.tag = self.run_dir, self.tag
        self.plan, self.mpc, self.xs, self.us = build(ns)
        return self


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
        self.args, self.tasks, self.plant = args, tasks, plant
        self.m, self.kp, self.kd, self.tau_lim = m, kp, kd, tau_lim
        self.hooks = hooks              # extra on_step consumers (panel, video)
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
        self.panel = None
        self.q0 = None                  # in-process reset pose
        self.target = self.target0 = None      # reach target, live-editable
        self.target_nodes = 0
        self.rot_base = None            # the plan's own gripper orientation
        self.rot_deg, self.rot_axis, self.rot_nodes = 0.0, "x", 0
        self._auto = [t for t, _ in TASKS]

    # -- state the browser sees -------------------------------------------
    def state(self):
        cur = self.tasks.get(self.task_name)
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
            tasks=[dict(name=n, note=note,
                        ready=bool(self.tasks[n].ready),
                        why=self.tasks[n].why_unavailable())
                   for n, note in TASKS if n in self.tasks],
            built=bool(cur and cur.built))

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
        task = task or self.tasks.get(self.task_name)
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
        task = task or self.tasks.get(self.task_name)
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
                if v in self.tasks:
                    self.task_name = v
                    self._skip = True     # take effect at the episode boundary
            elif name == "viewer":
                self._viewer_req = bool(payload.get("on"))
                return self._toggle_viewer(self._viewer_req)
            elif name == "quit":
                self.quit = True
                self._skip = True
            elif name == "render":
                threading.Thread(target=self.render, daemon=True).start()
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

    def render(self):
        """Render the last episode to the panel. Off the control thread.

        PAUSED PERIODS ARE ABSENT BY CONSTRUCTION, not by filtering: the
        recorder is an `on_step` hook and `ControlLoop` does not call `on_step`
        for a paused period, so a pause leaves no frames rather than a stretch
        of identical ones. A video of a run someone paused halfway is still a
        video of the trajectory, which is what makes it comparable to a replay.
        """
        q, task = list(self.qtrace), self.tasks.get(self.task_name)
        if not q or task is None or task.plan is None or not self.args.video:
            return
        self.set_status("rendering %d states ..." % len(q))
        try:
            got = render_run(q, task.plan, task.run_dir, self.args.video,
                             cam=self.args.video_cam, fps=self.args.video_fps,
                             dt_plan=task.plan["dt"])
            if got and self.panel is not None:
                self.panel.set_video(got)
        except Exception as exc:                                 # noqa: BLE001
            self.set_status("render failed: %s" % exc)
            return
        self.set_status("rendered %s" % os.path.basename(self.args.video))

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
            before = set(threading.enumerate())
            self.viewer = _mjv.launch_passive(self.plant.m, self.plant.d)
            self._viewer_thread = next(
                iter(set(threading.enumerate()) - before), None)
        elif not on and self.viewer is not None:
            self.close_viewer()
        self.push()

    def close_viewer(self):
        if self.viewer is None:
            return
        v, th = self.viewer, self._viewer_thread
        self.viewer, self._viewer_thread = None, None
        try:
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
            k = int(round(t / dt_plan))
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

        def record(row, st, cmd):
            if a.plant == "mujoco":
                self.qtrace.append(self.plant.d.qpos.copy())
            else:
                q = self._qtmpl.copy()
                q[0:3], q[3:7] = st.base_pos, st.base_quat
                q[7:7 + 27] = st.q
                self.qtrace.append(q)

        def on_step(row, st, cmd):
            for h in self.hooks:
                try:
                    h(row, st, cmd)
                except Exception:                                # noqa: BLE001
                    pass
            record(row, st, cmd)
            if self.viewer is not None:
                try:
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
            chained=bool(getattr(self, "_chained", False)),
            chain_dq_max_rad=stats.get("chain_dq_max_rad"),
            chain_dbase_mm=stats.get("chain_dbase_mm"),
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
            pelvis_z=(float(self.plant.d.qpos[2]) if a.plant == "mujoco"
                      else None))

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
            try:
                self.plant.write(self.plant.safe_hold(2.0))
            except Exception:                                    # noqa: BLE001
                pass
            if self.viewer is not None:
                try:
                    self.viewer.sync()
                except Exception:                                # noqa: BLE001
                    pass
            time.sleep(0.02)

    def run(self):
        """The supervisor. Runs until the panel (or Ctrl-C) says stop."""
        if self.args.plant != "mujoco":
            self._qtmpl = cs.load(ik_margin=0.0)[1].qpos.copy()
        while not self.quit:
            task = self.tasks.get(self.task_name)
            if task is None or not task.ready:
                self.set_status("%s is not available -- %s"
                                % (self.task_name,
                                   task and task.why_unavailable()))
                self.idle()
                continue
            if not task.built:
                self.set_status("building the %s OCP ..." % task.name)
                try:
                    task.build(self.args)
                except Exception as exc:                         # noqa: BLE001
                    task.error = str(exc)
                    self.set_status("%s failed to build: %s"
                                    % (task.name, exc))
                    self.idle()
                    continue
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
            self.set_status("running %s / %s (episode %d)"
                            % (task.name, self.submode, self.episode))
            self.runs.append(self.run_episode(task))
            self.push()
            if self.quit:
                break
            with self.lock:
                skipped, reset = self._skip, self._reset
                self._skip = False
            if self.submode == "automode" and not (skipped or reset):
                nxt = [n for n in self._auto
                       if n in self.tasks and self.tasks[n].ready]
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

    tasks = {}
    for name, note in TASKS:
        tag = args.tag if name == "brace+reach" else name
        tasks[name] = Task(name, args.dir, tag, note)
    if not tasks[args.task].ready:
        ready = [n for n in tasks if tasks[n].ready]
        raise SystemExit(
            "--task %s: %s\nAvailable in this cell: %s"
            % (args.task, tasks[args.task].why_unavailable(),
               ", ".join(ready) or "none"))

    # dt comes off the plan JSON, which is cheap to read -- the panel must
    # exist BEFORE the ~20 s OCP build, not after it, or the twenty seconds
    # look exactly like a hung page.
    dt_plan = json.load(open(tasks[args.task].plan_path))["dt"]

    plant, _base = _make_plant(args, m, tau_lim)
    panel = Panel(None, port=args.gui, period_ms=1e3 * dt_plan,
                  config=dict(
                      plant=args.plant, base=args.base, cell=args.dir,
                      tag=args.tag, horizon=args.horizon, iters=args.iters,
                      threads=args.threads,
                      dds=(None if args.plant == "mujoco"
                           else "domain %d / %s" % (args.domain, args.iface)),
                      video=args.video,
                      gl=os.environ.get("MUJOCO_GL", "(default)")))
    session = Session(args, tasks, plant, m, kp, kd, tau_lim,
                      hooks=[panel.on_step])
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
    ap.add_argument("--tag", default="elbow_palm")
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
                    choices=[t for t, _ in TASKS],
                    help="which maneuver the session starts on. A task is a "
                         "SOLVED PLAN in the cell directory, not a set of "
                         "weights: the phases differ by contact set, so "
                         "switching rebuilds the OCP. `stand` and `recover` "
                         "need plan_stand/plan_recover artifacts, which the "
                         "certified grid does not carry (every cell has "
                         "n_return=0).")
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
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--emit-qpos0", default=None,
                    help="write the plan's start qpos to this file and exit. "
                         "Feed it to `lean_twin --qpos0`: the twin must begin "
                         "where the plan begins, and no keyframe is that pose.")
    args = ap.parse_args()

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

    # VIDEO IS ON BY DEFAULT IN THE SESSION, next to the replay mp4 the cell
    # already carries (`replay_<tag>_mpc.mp4`) -- the two are the same kind of
    # artifact and belong side by side: one is what the plan does, the other
    # is what the loop did. --no-video opts out.
    #
    # NOT in the batch path, deliberately. twin_grid.sh runs this 26 times and
    # a default there would silently add 26 renders and 26 files to a
    # certified grid, changing what a batch run produces as a side effect of a
    # GUI convenience. Batch keeps --video opt-in, as it was.
    if args.no_video:
        args.video = None
    elif args.video is None and args.gui:
        args.video = os.path.join(args.dir, "twin_%s.mp4" % args.tag)

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
                         recv=args.recv)
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
        _before = set(threading.enumerate())
        viewer = _mjv.launch_passive(plant.m, plant.d)
        viewer_thread = next(iter(set(threading.enumerate()) - _before), None)
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
