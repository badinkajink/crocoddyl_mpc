"""The lean scene, served over Unitree DDS: `rt/lowstate` out, `rt/lowcmd` in.

WHAT THIS IS FOR.  Every crocoddyl result so far was measured with MuJoCo in the
SAME PROCESS as the controller: the loop read `d.qpos`, wrote `d.ctrl` and
stepped physics itself, so it could never be late, never miss a sample, and
never disagree with the plant about what time it was.  None of those hold on the
twin or on the robot.  This process puts the same physics on the far side of the
same wire the real H1-2 uses, so the deployment questions can be asked BEFORE
the scene and the estimator are also different:

    does the loop hold 200 Hz with a 10-17 ms solve in it
    is the 27-motor order right end to end
    does (q_des, kp, kd, tau_ff) mean the same thing on both sides
    what does the watchdog do when a solve overruns

WHAT IT IS NOT.  It is not the digital twin.  `h1_robocasa`/`h1_mujoco` are, and
they carry a kitchen, not the lean table -- which is the next problem, not this
one.  This is the same MJCF the plan was solved against, so a failure here is a
DEPLOYMENT failure and cannot be blamed on scene parity.  Keep it that way: the
moment this file starts diverging from `Lean_H12_Magpie.xml` it stops being able
to isolate anything.

TWO THINGS IT DOES DELIBERATELY.

  1. IT REWRITES THE ACTUATORS TO TORQUE.  The MJCF has 27 <position> servos with
     kp/kd baked into the model; the real robot (and `h1_robocasa`'s
     `unitree_interface.low_cmd_handler`) takes tau + kp*(q_des - q) +
     kd*(dq_des - dq) and honours WHATEVER GAINS THE COMMAND CARRIES.  Driving a
     position servo through the inversion `ctrl = q_des + (tau + kd*v_des)/kp`
     reproduces that law only while the commanded gains equal the model's, and a
     twin that silently ignores a gain change is a twin that cannot catch the
     bug where the controller sends the wrong ones.  So the servos are converted
     to plain force actuators at load time and the law is applied here, exactly
     as the hardware bridge applies it.  forcerange is untouched.
  2. IT DOES NOT PUBLISH THE BASE POSE unless asked.  `rt/lowstate` carries IMU
     and encoders, and that is all the robot has.  The lean OCP needs a base
     pose, so SOMETHING upstream must estimate one; `--publish-truth` puts the
     true pose on `rt/sim_state` as JSON -- the topic and encoding Isaac already
     uses -- purely so the deployment plumbing can be tested with the estimator
     held fixed at perfect.  It is off by default and every run that used it
     says so, because a controller that reads it works here and fails on the
     robot.  See the ground-truth rule in the repo's CLAUDE.md.

usage:
    python -m croco.twin.lean_twin --model .../Lean_H12_Magpie.xml --key stand
    ... --domain 1 --iface lo [--publish-truth] [--viewer] [--rate 500]
"""
from __future__ import annotations

import argparse
import json
import threading
import time

import numpy as np

from ..plant.dds_plant import JOINT_NAMES, TOPIC_LOWCMD, TOPIC_LOWSTATE

TOPIC_TRUTH = "rt/sim_state"
TOPIC_SPORT = "rt/sportmodestate"   # the fork's estimator bench speaks this
# pelvis -> IMU site. Must equal base_estimator_node.IMU_OFFSET and
# h12_control_node.cc's kImuOffset, or the two ends of the reconstruction
# disagree and every base pose is wrong by a quarter of a metre.
IMU_OFFSET = np.array([-0.04452, -0.01891, 0.27756])


def _quat_to_mat(q):
    import mujoco
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def to_torque_actuators(m):
    """<position> servos -> plain force actuators, keeping forcerange.

    A <position> actuator is gaintype=FIXED with gainprm[0]=kp and bias
    (0, -kp, -kd), i.e. force = kp*(ctrl - q) - kd*v. Setting gainprm[0]=1 and
    the bias to zero makes force = ctrl, so `ctrl` is a torque in N*m and the PD
    law can be applied by whoever sends the command -- which is the point.

    Returns the (kp, kd) that WERE in the model, because they are the gains the
    controller is expected to send back, and a run where the two disagree should
    be visible rather than absorbed.
    """
    import mujoco
    nu = len(JOINT_NAMES)
    kp = m.actuator_gainprm[:nu, 0].copy()
    kd = -m.actuator_biasprm[:nu, 2].copy()
    if not np.allclose(-m.actuator_biasprm[:nu, 1], kp):
        raise ValueError("actuators are not <position> servos with kp/kv; "
                         "refusing to guess what their gains mean")
    for i in range(nu):
        m.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED
        m.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_NONE
        m.actuator_gainprm[i, :] = 0.0
        m.actuator_gainprm[i, 0] = 1.0
        m.actuator_biasprm[i, :] = 0.0
        # ctrl is now a torque, so the position ctrlrange would clip it to
        # radians. The force limit still applies through forcerange.
        m.actuator_ctrllimited[i] = 0
    return kp, kd


def assert_motor_order(m):
    """The MJCF actuator order IS the Unitree motor order, checked not assumed.

    A silent permutation here drives the wrong limb, and nothing downstream --
    not the safety layer, not the plan -- would catch it.
    """
    import mujoco
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT,
                               m.actuator_trnid[i, 0])
             for i in range(len(JOINT_NAMES))]
    if names != JOINT_NAMES:
        bad = [(i, a, b) for i, (a, b) in enumerate(zip(names, JOINT_NAMES))
               if a != b]
        raise ValueError("MJCF actuator order != Unitree motor order; first "
                         "mismatches: %s" % bad[:4])


class LeanTwin:
    """Step the lean MJCF, publish `rt/lowstate`, apply `rt/lowcmd`."""

    def __init__(self, model_path, key="stand", domain=1, iface="lo",
                 rate_hz=500.0, publish_truth=False, cmd_timeout=0.5,
                 hold_until_cmd=True, qpos0=None, wait_for_cmd=True,
                 wait_timeout=300.0):
        import mujoco

        self._mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(model_path)
        self.d = mujoco.MjData(self.m)
        self.nu = len(JOINT_NAMES)
        assert_motor_order(self.m)
        self.kp_model, self.kd_model = to_torque_actuators(self.m)

        if qpos0 is not None:
            # THE PLAN'S START POSE, not a keyframe. A keyframe is not the same
            # thing: `contact_select.start_qpos` applies the study's stance
            # offset to it, and the plan's x0 is that pose refined by the IK, so
            # a twin reset to the raw keyframe puts the robot centimetres from
            # where the maneuver assumes it is -- relative to the TABLE, which
            # is the one distance the whole brace depends on. Measured: starting
            # from the raw `stand` keyframe, the robot fell every time
            # (pelvis 1.00 -> 0.06 m) while the same plan and the same MPC stand
            # at 0.956 m in-process. It reads as a controller failure and is an
            # initial-condition failure.
            q = np.loadtxt(qpos0) if isinstance(qpos0, str) else np.asarray(qpos0)
            if q.size != self.m.nq:
                raise ValueError("qpos0 has %d entries, model has nq=%d"
                                 % (q.size, self.m.nq))
            self.d.qpos[:] = q
            self.d.qvel[:] = 0.0
        else:
            kid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_KEY, key)
            if kid < 0:
                raise KeyError("no keyframe %r in %s" % (key, model_path))
            mujoco.mj_resetDataKeyframe(self.m, self.d, kid)
        mujoco.mj_forward(self.m, self.d)

        self.dt = float(self.m.opt.timestep)
        self.steps_per_pub = max(1, int(round(1.0 / (rate_hz * self.dt))))
        self.publish_truth = publish_truth
        self.cmd_timeout = cmd_timeout

        self.hold_until_cmd = hold_until_cmd
        self.wait_for_cmd = wait_for_cmd
        self.wait_timeout = wait_timeout
        self.q_hold = self.d.qpos[7:7 + self.nu].copy()
        self._lock = threading.Lock()
        self._tau = np.zeros(self.nu)
        self._last_cmd = None                 # monotonic; None = nothing yet
        self._gain_warned = False
        self.stats = dict(cmds=0, states=0, timeouts=0, gain_mismatch=0)
        # The twin is the only side that can see whether the robot is still
        # standing: the controller has joint angles and a base pose, not a
        # verdict. Without this a run reports 196 healthy control periods while
        # the robot lies on the floor.
        self.z0 = float(self.d.qpos[2])
        self._z_min = self.z0
        # SCOPE THE VERDICT TO THE COMMANDED INTERVAL. The controller retires
        # when its plan runs out and signs off with a damping stop, so the twin
        # keeps stepping a robot nobody is driving and it collapses -- every run
        # then reports "fell" for the one reason that says nothing about the
        # maneuver. What is being asked is whether the robot survived while it
        # was being COMMANDED, so that window is tracked separately.
        self._z_min_cmd = self.z0
        self._z_at_last_cmd = self.z0
        self._commanding = False

        from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,
                                                 ChannelPublisher,
                                                 ChannelSubscriber)
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

        if domain == 0:
            raise SystemExit(
                "ROS_DOMAIN_ID 0 is the REAL ROBOT's command bus. Refusing to "
                "publish rt/lowcmd-shaped traffic on it from a simulator.")
        ChannelFactoryInitialize(domain, iface)
        self._state_msg = unitree_hg_msg_dds__LowState_()
        self._pub = ChannelPublisher(TOPIC_LOWSTATE, LowState_)
        self._pub.Init()
        self._sub = ChannelSubscriber(TOPIC_LOWCMD, LowCmd_)
        self._sub.Init(self._on_lowcmd, 10)
        self._truth_pub = self._sport_pub = None
        if publish_truth:
            self._String_ = String_
            self._truth_pub = ChannelPublisher(TOPIC_TRUTH, String_)
            self._truth_pub.Init()
            # ALSO as SportModeState_ on rt/sportmodestate, which is what
            # h1_robocasa's h12_mujoco.py publishes and therefore what the
            # fork's whole estimator bench already speaks: sim_tag_anchor.py
            # and base_estimator_node --compare both read this topic and need
            # no adapter. Same gate, same warning -- one flag still turns all
            # ground truth on and off.
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
            from unitree_sdk2py.idl.default import (
                unitree_go_msg_dds__SportModeState_)
            self._sport_msg = unitree_go_msg_dds__SportModeState_()
            self._sport_pub = ChannelPublisher(TOPIC_SPORT, SportModeState_)
            self._sport_pub.Init()
            print("[lean_twin] GROUND TRUTH IS ON: base pose on %s. Any result "
                  "from this run is a plumbing result, not a deployment "
                  "result." % TOPIC_TRUTH)

    # -- wire -------------------------------------------------------------

    def _on_lowcmd(self, msg):
        """The hardware's law, applied here: tau + kp*(q-q_m) + kd*(dq-dq_m)."""
        with self._lock:
            q = self.d.qpos[7:7 + self.nu]
            v = self.d.qvel[6:6 + self.nu]
            tau = np.zeros(self.nu)
            kp_seen = np.zeros(self.nu)
            for i in range(self.nu):
                mc = msg.motor_cmd[i]
                if mc.mode != 1:
                    continue
                kp_seen[i] = mc.kp
                tau[i] = (mc.tau + mc.kp * (mc.q - q[i])
                          + mc.kd * (mc.dq - v[i]))
            self._tau[:] = tau
            self._last_cmd = time.monotonic()
            if not self._commanding:
                self._commanding = True
                self._z_min_cmd = float(self.d.qpos[2])
            self.stats["cmds"] += 1
            # Not an error -- the robot honours whatever it is sent -- but a
            # difference the in-process replay could not express, so it is
            # counted rather than absorbed.
            # A DAMPING STOP legitimately carries kp = 0 -- that is what a
            # watchdog trip looks like on the wire, not a controller sending the
            # wrong gains. Comparing it against the model's gains made the
            # warning fire on every safe period, i.e. exactly when it was least
            # useful.
            if (np.any(kp_seen != 0.0)
                    and not np.allclose(kp_seen, self.kp_model,
                                        rtol=1e-6, atol=1e-6)):
                self.stats["gain_mismatch"] += 1
                if not self._gain_warned:
                    self._gain_warned = True
                    print("[lean_twin] note: commanded kp differs from the "
                          "model's servo gains; applying what was SENT, as the "
                          "robot would.")

    def _publish_state(self):
        m, d = self.m, self.d
        s = self._state_msg
        s.tick = int(round(d.time / self.dt))
        q = d.qpos[7:7 + self.nu]
        v = d.qvel[6:6 + self.nu]
        f = d.actuator_force[:self.nu]
        for i in range(self.nu):
            ms = s.motor_state[i]
            ms.q, ms.dq, ms.tau_est = float(q[i]), float(v[i]), float(f[i])
        # IMU: pelvis attitude in world (w,x,y,z) and body-frame angular rate,
        # which is what the H1-2's IMU reports.
        s.imu_state.quaternion[:] = [float(x) for x in d.qpos[3:7]]
        R = d.xmat[1].reshape(3, 3) if m.nbody > 1 else np.eye(3)
        w_body = R.T @ d.qvel[3:6]
        s.imu_state.gyroscope[:] = [float(x) for x in w_body]
        self._pub.Write(s)
        self.stats["states"] += 1

    def _publish_truth(self):
        payload = json.dumps(dict(
            t=float(self.d.time),
            base_pos=[float(x) for x in self.d.qpos[0:3]],
            base_quat=[float(x) for x in self.d.qpos[3:7]],
            base_linvel=[float(x) for x in self.d.qvel[0:3]],
            base_angvel=[float(x) for x in self.d.qvel[3:6]]))
        msg = self._String_(data=payload)
        self._truth_pub.Write(msg)
        if self._sport_pub is not None:
            # THE IMU SITE, NOT THE PELVIS. `rt/sportmodestate` means the site
            # everywhere else in this stack -- the factory topic does, the
            # fork's estimator publishes it, and h12_control_node.cc undoes the
            # offset on the way in. Publishing the pelvis here would make
            # `--compare` read a constant 0.278 m error and call the estimator
            # broken when it was exact.
            sm = self._sport_msg
            R = _quat_to_mat(self.d.qpos[3:7])
            roff = R @ IMU_OFFSET
            p = self.d.qpos[0:3] + roff
            v = self.d.qvel[0:3] + np.cross(R @ self.d.qvel[3:6], roff)
            for k in range(3):
                sm.position[k] = float(p[k])
                sm.velocity[k] = float(v[k])
            sm.imu_state.quaternion[:] = [float(x) for x in self.d.qpos[3:7]]
            self._sport_pub.Write(sm)

    # -- run --------------------------------------------------------------

    def run(self, duration=None, viewer=False, realtime=True):
        """Advance physics in real time, publishing state and applying commands.

        REAL TIME IS THE POINT. Stepping as fast as possible would hide exactly
        the failure this process exists to expose -- a controller that cannot
        keep up. The loop sleeps to the wall clock and reports how far it
        drifted.
        """
        mujoco = self._mj
        v = None
        if viewer:
            import mujoco.viewer
            v = mujoco.viewer.launch_passive(self.m, self.d)
        if self.wait_for_cmd:
            # DO NOT START THE EPISODE BEFORE THE CONTROLLER EXISTS.
            # The first attempt held the start pose with a joint PD while the
            # controller built its OCP, on the theory that a hold is close
            # enough to a winch. It is not: x0 is a leaning posture whose
            # balance is the MPC's job, and a joint-space hold has no authority
            # over the floating base at all. Measured -- after 25 s of holding,
            # the robot had toppled and slid 1.14 m, ending at pelvis z 0.069 m
            # with an 88 deg pitch. The controller then started, correctly,
            # against a robot lying on the floor 1.44 m from where its plan
            # begins, and the run failed for that reason while looking exactly
            # like a control failure. So physics does not advance at all until
            # the first lowcmd. State is still published throughout, because the
            # controller has to be able to connect and read x0 before it can
            # produce a command to start on.
            print("[lean_twin] holding the episode at t=0 until the first "
                  "rt/lowcmd (physics is NOT advancing)")
            t_wait = time.monotonic()
            while self._last_cmd is None:
                if v is not None:
                    v.sync()
                self._publish_state()          # the controller needs a state to
                if self._truth_pub is not None:  # connect to, and to plan from
                    self._publish_truth()
                time.sleep(0.002)
                if time.monotonic() - t_wait > self.wait_timeout:
                    raise TimeoutError(
                        "no rt/lowcmd in %.0f s. The twin will not start an "
                        "episode nobody is driving." % self.wait_timeout)
            print("[lean_twin] first command after %.1f s -- episode starts"
                  % (time.monotonic() - t_wait))
        t_wall0 = time.monotonic()
        n = 0
        skipped = 0
        worst_lag = 0.0
        try:
            while duration is None or self.d.time < duration:
                with self._lock:
                    stale = (self._last_cmd is None
                             or time.monotonic() - self._last_cmd > self.cmd_timeout)
                    if self._last_cmd is None and self.hold_until_cmd:
                        # THE WINCH. Before the first command ever arrives the
                        # controller is still building its OCP -- seconds, not
                        # milliseconds -- and a robot given zero torque for that
                        # long is on the floor before the run starts. So the
                        # twin holds the start keyframe with the model's own
                        # gains and lets go on the first lowcmd, which is what
                        # the plan assumes anyway: every stress profile in this
                        # study is named for how the WINCH sets the robot down.
                        # This is the spawn hold, not a controller, and it is
                        # never re-entered once a command has been seen.
                        self.d.ctrl[:self.nu] = (
                            self.kp_model * (self.q_hold - self.d.qpos[7:7 + self.nu])
                            - self.kd_model * self.d.qvel[6:6 + self.nu])
                    elif stale:
                        # What the real twin does: no command, no torque. The
                        # robot collapses, which is the correct and visible
                        # consequence of a controller that stopped talking.
                        self.d.ctrl[:self.nu] = 0.0
                        self.stats["timeouts"] += 1
                    else:
                        self.d.ctrl[:self.nu] = self._tau
                    mujoco.mj_step(self.m, self.d)
                n += 1
                z = float(self.d.qpos[2])
                self._z_min = min(self._z_min, z)
                if self._commanding and not stale:
                    self._z_min_cmd = min(self._z_min_cmd, z)
                    self._z_at_last_cmd = z
                if n % self.steps_per_pub == 0:
                    self._publish_state()
                    if self._truth_pub is not None:
                        self._publish_truth()
                    if v is not None:
                        v.sync()
                if realtime:
                    lag = (time.monotonic() - t_wall0) - self.d.time
                    worst_lag = max(worst_lag, lag)
                    if lag < 0:
                        time.sleep(-lag)
        finally:
            if v is not None:
                v.close()
            self.close()
        self.stats["worst_lag_s"] = worst_lag
        self.stats["sim_time_s"] = float(self.d.time)
        self.stats["pelvis_z0"] = self.z0
        self.stats["pelvis_z_final"] = float(self.d.qpos[2])
        self.stats["pelvis_z_min"] = self._z_min
        self.stats["pelvis_z_at_last_cmd"] = self._z_at_last_cmd
        self.stats["pelvis_z_min_commanded"] = self._z_min_cmd
        # The SAME absolute threshold croco_replay.summarise uses (0.55 m), so a
        # twin run and a replay can be compared without translating between two
        # ideas of "fell" -- but over the COMMANDED window only. `fell_after` is
        # the uncontrolled collapse afterwards, which is expected and is not a
        # result.
        self.stats["fell"] = bool(self._z_min_cmd < 0.55)
        self.stats["fell_after_release"] = bool(self._z_min < 0.55
                                                and self._z_min_cmd >= 0.55)
        return self.stats

    def close(self):
        for h in ("_sub", "_pub", "_truth_pub", "_sport_pub"):
            c = getattr(self, h, None)
            if c is not None:
                try:
                    c.Close()
                except Exception:
                    pass
                setattr(self, h, None)


def _install_sigterm():
    """SIGTERM must print the outcome, not swallow it.

    A twin that is killed by whatever orchestrates the run -- a shell, a
    supervisor, a test harness -- otherwise dies before reporting whether the
    robot was still standing, which is the one number the controller side cannot
    produce for itself.
    """
    import signal

    def _raise(_sig, _frm):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _raise)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="Lean_H12_Magpie.xml")
    ap.add_argument("--key", default="stand",
                    help="start keyframe; ignored when --qpos0 is given")
    ap.add_argument("--qpos0", default=None,
                    help="full qpos to start from, as written by "
                         "`croco_twin.py --emit-qpos0`. USE THIS: the plan's x0 "
                         "is not any keyframe, and starting from the wrong pose "
                         "fails as a controller failure.")
    ap.add_argument("--domain", type=int, default=1,
                    help="DDS domain. 0 is the REAL ROBOT and is refused.")
    ap.add_argument("--iface", default="lo")
    ap.add_argument("--rate", type=float, default=500.0,
                    help="lowstate publish rate [Hz]")
    ap.add_argument("--duration", type=float, default=None,
                    help="sim seconds to run (default: until interrupted)")
    ap.add_argument("--cmd-timeout", type=float, default=0.5,
                    help="seconds without a lowcmd before torques are zeroed, "
                         "as the real twin does")
    ap.add_argument("--publish-truth", action="store_true",
                    help="publish the TRUE base pose on rt/sim_state. Off by "
                         "default: no robot has it, and a controller that reads "
                         "it works here and fails on hardware.")
    ap.add_argument("--no-hold", action="store_true",
                    help="do NOT hold the start keyframe before the first "
                         "command; the robot falls while the controller builds "
                         "its OCP, which is only what you want if you are "
                         "testing the fall")
    ap.add_argument("--no-wait", action="store_true",
                    help="start stepping immediately instead of waiting for the "
                         "first lowcmd. The robot then falls over while the "
                         "controller is still building, which is not a test of "
                         "anything.")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--free-run", action="store_true",
                    help="do not pace to the wall clock (hides late controllers)")
    a = ap.parse_args(argv)

    _install_sigterm()
    twin = LeanTwin(a.model, key=a.key, domain=a.domain, iface=a.iface,
                    rate_hz=a.rate, publish_truth=a.publish_truth,
                    cmd_timeout=a.cmd_timeout,
                    hold_until_cmd=not a.no_hold, qpos0=a.qpos0,
                    wait_for_cmd=not a.no_wait)
    print("[lean_twin] %s  dt=%.4f s  lowstate %.0f Hz  domain %d/%s"
          % (a.model.split("/")[-1], twin.dt, a.rate, a.domain, a.iface))
    try:
        stats = twin.run(duration=a.duration, viewer=a.viewer,
                         realtime=not a.free_run)
    except KeyboardInterrupt:
        stats = twin.stats
        stats.setdefault("pelvis_z0", twin.z0)
        stats.setdefault("pelvis_z_final", float(twin.d.qpos[2]))
        stats.setdefault("pelvis_z_min", twin._z_min)
        stats.setdefault("pelvis_z_at_last_cmd", twin._z_at_last_cmd)
        stats.setdefault("pelvis_z_min_commanded", twin._z_min_cmd)
        stats.setdefault("fell", bool(twin._z_min_cmd < 0.55))
        stats.setdefault("fell_after_release",
                         bool(twin._z_min < 0.55 and twin._z_min_cmd >= 0.55))
        stats.setdefault("sim_time_s", float(twin.d.time))
    print("[lean_twin] %s" % json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
