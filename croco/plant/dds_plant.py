"""The digital twin (and, unchanged, the real robot) over Unitree DDS.

THIS IS THE WHOLE DEPLOY PORT. `rt/lowstate` in, `rt/lowcmd` out, same wire
format against `h1_robocasa`/`h1_mujoco` and against the machine -- which is the
property the rest of the stack is built on and the reason nothing above this
file changes between sim and real.

WHAT IT DELIBERATELY DOES NOT DO.

  * NO SAFETY LAYER. `h12_safety_layer` already sits between this and the robot
    and owns the position/velocity/torque clipping and the e-stop. Re-clipping
    here would mean two authorities with two copies of the limits, and the one
    that is wrong would be this one. The plant reports its limits
    (`tau_limit`, `q_range`) so a controller can plan inside them, and then
    sends what it was given.
  * NO BASE ESTIMATION. `rt/lowstate` carries IMU and joint encoders, not a
    world pose. The base estimate arrives from whatever is running upstream --
    `h12_deploy_mjpc`'s estimator_node, FAST-LIO, or the tag anchor -- through
    `base_source`. If none is supplied `State.base_*` is None, and a controller
    that needs the base must fail rather than invent one.
  * NO CLOCK OF ITS OWN. `now()` is the lowstate tick times the twin's timestep,
    which is what the MJPC deploy node calls `use_twin_time`. Pacing a
    sim-coupled controller on the wall clock makes its plan index drift against
    the plant whenever the sim is not real-time, and the twin frequently is not.

RECEIVE IS POLLED, NOT CALLED BACK. unitree_sdk2py's default is a cyclonedds
listener that invokes a PYTHON callback, which hands the sample to a PYTHON
queue thread. Both hops need the GIL, and crocoddyl's bindings hold the GIL for
the whole solve (Boost.Python releases it nowhere; measured: a 112 ms solve
stops every other Python thread in the process for 112 ms). So on a control
loop whose period is mostly solve, the receive thread cannot run during the
part of the period when the samples are actually arriving, and the state the
controller reads is as old as the last gap it was allowed to breathe in -- 62
of 195 periods over the watchdog on the first twin run, with a 0.7 ms MEDIAN
age. It presents as a network problem and is not one: cyclonedds's own receive
threads are C and were never blocked; only the Python that drains them was.

Polling inverts it. The reader accumulates in C, and the control thread -- the
one that by definition holds the GIL when it wants a state -- takes the newest
sample itself, at the instant it needs it. No second thread, no queue, no
starvation, and the age becomes transport time rather than scheduling time.

ORDERING. `JOINT_ORDER` is the Unitree `LowCmd_.motor_cmd` index for each of the
27 joints in MJCF order. It is asserted against the model at construction rather
than trusted, because a silent permutation here is a robot that moves the wrong
limb and there is no downstream check that would catch it.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from .base import Plant, State

TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_LOWCMD = "rt/lowcmd"

# MJCF/CL_Assets joint order == h12_safety_layer JOINT_NAMES order == the Unitree
# hg motor index order for the H1-2. Kept explicit so the assertion below has
# something to check against.
JOINT_NAMES = [
    "left_hip_yaw_joint", "left_hip_pitch_joint", "left_hip_roll_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "torso_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


class PollingReceiver:
    """A DDS subscriber the caller drains, instead of one that calls back.

    WHY THE NAME MANGLING BELOW. unitree_sdk2py exposes exactly one read path,
    `ChannelSubscriber.Read()` -> cyclonedds `take_one()`, and `take_one()` with
    no timeout is an INFINITE BLOCKING iterator when the reader is empty -- it
    cannot be called from a control period. The `DataReader` underneath has the
    non-blocking `take(N)` this needs, and there is no public accessor for it.
    So it is reached by mangled name and ASSERTED, so that an upstream rename
    fails loudly here rather than silently falling back to a receive path whose
    whole problem is that it is invisible.

    `take(N)` returns oldest-first, so the newest sample is the LAST one. Taking
    it (rather than the first) is the entire point: between two control periods
    several samples arrive, and the stale ones are of no interest to anybody.
    """

    def __init__(self, topic, dtype, depth=64):
        from unitree_sdk2py.core.channel import ChannelSubscriber
        self.depth = depth
        self.n_last = 0
        self._sub = ChannelSubscriber(topic, dtype)
        self._sub.Init()                       # handler=None -> no listener
        try:
            self._rd = (self._sub._ChannelSubscriber__channel
                        ._Channel__reader._Reader__reader)
        except AttributeError as e:            # noqa: BLE001
            raise RuntimeError(
                "cannot reach the DataReader inside unitree_sdk2py's "
                "ChannelSubscriber (%s). Its internals changed; fix "
                "PollingReceiver rather than reverting to the callback path, "
                "which starves under the GIL." % e)
        from cyclonedds.internal import InvalidSample
        self._invalid = InvalidSample

    def latest(self):
        """The newest sample waiting, or None. Never blocks.

        `n_last` is how many were waiting. It is the honest health signal on
        this path: with the twin at 500 Hz and the loop at 50 Hz it should be
        ~10, and a run of zeros means the publisher stopped or the transport
        did -- which `age` alone can no longer tell you (see DDSPlant.read).
        """
        try:
            samples = self._rd.take(N=self.depth)
        except Exception:                                       # noqa: BLE001
            self.n_last = 0
            return None
        samples = samples or ()
        self.n_last = len(samples)
        for s in reversed(samples):
            if not isinstance(s, self._invalid):
                return s
        return None

    def close(self):
        try:
            self._sub.Close()
        except Exception:                                       # noqa: BLE001
            pass


class DDSPlant(Plant):
    """Drive `rt/lowcmd` from a Command; read `rt/lowstate` into a State.

    `base_source` is any callable returning
    (pos, quat_wxyz, linvel, angvel, age_seconds) or None -- e.g. a subscriber to
    the estimator's odometry. Keeping it injectable is what lets the same plant
    serve the twin (where a sim odom may be published) and the robot (where
    FAST-LIO or the tag anchor supplies it) without a branch in here.
    """

    OWNS_CLOCK = False          # the twin/robot advances itself; we just watch

    def __init__(self, network_interface=None, domain_id=None, twin_dt=None,
                 base_source=None, tau_limit=None, q_range=None,
                 stale_after=0.05, recv="poll"):
        self.nu = len(JOINT_NAMES)
        self.tau_limit = None if tau_limit is None else np.asarray(tau_limit, float)
        self.q_range = q_range
        self.base_source = base_source
        self.twin_dt = twin_dt
        self.stale_after = stale_after

        self._lock = threading.Lock()
        self._q = np.zeros(self.nu)
        self._v = np.zeros(self.nu)
        self._tau = np.zeros(self.nu)
        self._imu_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._imu_gyro = np.zeros(3)
        self._tick = 0
        self._stamp = None                  # monotonic time of the last sample
        self.recv_samples = self.recv_polls = self.recv_empty = 0

        from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,
                                                 ChannelPublisher,
                                                 ChannelSubscriber)
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.utils.crc import CRC

        if domain_id is not None or network_interface is not None:
            ChannelFactoryInitialize(domain_id or 0, network_interface or "")
        self._crc = CRC()
        self._cmd_msg = unitree_hg_msg_dds__LowCmd_()
        self._pub = ChannelPublisher(TOPIC_LOWCMD, LowCmd_)
        self._pub.Init()
        self.recv = recv
        if recv == "poll":
            self._recv = PollingReceiver(TOPIC_LOWSTATE, LowState_)
            self._sub = None
        elif recv == "callback":
            self._recv = None
            self._sub = ChannelSubscriber(TOPIC_LOWSTATE, LowState_)
            self._sub.Init(self._ingest, 10)   # kept for the A/B, not for use
        else:
            raise ValueError("recv must be 'poll' or 'callback'")

    # -- DDS ---------------------------------------------------------------

    def _ingest(self, msg):
        """Fold one lowstate into the cached state. Same body either way -- the
        receive STRATEGY changes, the decoding does not."""
        with self._lock:
            for i in range(self.nu):
                mm = msg.motor_state[i]
                self._q[i], self._v[i], self._tau[i] = mm.q, mm.dq, mm.tau_est
            self._imu_quat[:] = msg.imu_state.quaternion
            self._imu_gyro[:] = msg.imu_state.gyroscope
            self._tick = int(getattr(msg, "tick", self._tick + 1))
            self._stamp = time.monotonic()

    def _pump(self):
        """Take the newest lowstate, if polling. No-op on the callback path."""
        if self._recv is None:
            return
        msg = self._recv.latest()
        self.recv_samples += self._recv.n_last
        self.recv_polls += 1
        if msg is None:
            self.recv_empty += 1
        else:
            self._ingest(msg)

    def wait_for_state(self, timeout=5.0):
        """Block until a lowstate has arrived. Fail loudly, not silently at zero.

        Without this the first read() returns an all-zeros pose, the controller
        plans from a robot standing perfectly straight that does not exist, and
        the first command is a large step to somewhere else.
        """
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            self._pump()
            with self._lock:
                if self._stamp is not None:
                    return True
            time.sleep(0.005)
        raise TimeoutError(
            "no %s in %.1f s. Is the twin running, and is ROS_DOMAIN_ID the "
            "same on both sides? (0 is the REAL robot's bus.)"
            % (TOPIC_LOWSTATE, timeout))

    # -- Plant -------------------------------------------------------------

    def now(self) -> float:
        with self._lock:
            tick, stamp = self._tick, self._stamp
        if self.twin_dt:
            return tick * self.twin_dt
        return 0.0 if stamp is None else stamp

    def read(self) -> State:
        """WHAT `age` MEANS DEPENDS ON THE RECEIVE PATH, and it is worth being
        exact about because the watchdog reads it.

        callback: time since a background thread decoded a sample -- so it
                  measures arrival AND that thread's ability to get scheduled,
                  and under a GIL-holding solver the second term dominates.
        poll:     time since THIS thread last took a new sample. Within a
                  period there is no staleness to measure, by construction: the
                  newest sample in the reader is taken at the instant it is
                  wanted. The watchdog still does its job -- nothing new to
                  take means `_stamp` stops advancing and `age` grows -- but it
                  now detects a DEAD PUBLISHER rather than a busy interpreter,
                  which is the failure it was always meant to catch.

        Sub-period transport delay is no longer visible in `age` on the polling
        path. `recv_samples / recv_polls` is what to watch instead.
        """
        self._pump()
        with self._lock:
            q, v, tau = self._q.copy(), self._v.copy(), self._tau.copy()
            quat, gyro = self._imu_quat.copy(), self._imu_gyro.copy()
            stamp = self._stamp
        age = 0.0 if stamp is None else max(0.0, time.monotonic() - stamp)
        bp = bq = bv = bw = None
        if self.base_source is not None:
            got = self.base_source()
            if got is not None:
                bp, bq, bv, bw, b_age = got
                age = max(age, b_age)
        else:
            # IMU orientation is measured; POSITION is not, and is left None so
            # a controller that needs it cannot quietly proceed on zeros.
            bq, bw = quat, gyro
        return State(t=self.now(), q=q, v=v, base_pos=bp, base_quat=bq,
                     base_linvel=bv, base_angvel=bw, tau=tau, age=age)

    def write(self, cmd) -> None:
        if cmd.nu != self.nu:
            raise ValueError("command has nu=%d, lowcmd has %d motors"
                             % (cmd.nu, self.nu))
        m = self._cmd_msg
        for i in range(self.nu):
            mc = m.motor_cmd[i]
            mc.mode = 1
            mc.q = float(cmd.q_des[i])
            mc.dq = float(cmd.v_des[i])
            mc.kp = float(cmd.kp[i])
            mc.kd = float(cmd.kd[i])
            mc.tau = float(cmd.tau_ff[i])
        m.crc = self._crc.Crc(m)
        self._pub.Write(m)

    def close(self) -> None:
        if self._recv is not None:
            self._recv.close()
            self._recv = None
        for h in ("_sub", "_pub"):
            c = getattr(self, h, None)
            if c is not None:
                try:
                    c.Close()
                except Exception:
                    pass
                setattr(self, h, None)

    def safe_hold(self, kd=2.0):
        """kp = 0 exactly: the hardware accepts it, so no epsilon is needed."""
        from ..control.command import Command
        st = self.read()
        c = Command(q_des=st.q, kp=np.full(self.nu, 1.0), kd=np.full(self.nu, kd),
                    tau_ff=np.zeros(self.nu))
        c.kp[:] = 0.0                       # bypass Command's positive-kp rule
        return c


def assert_joint_order(model, nu=27):
    """The MJCF's actuator order IS JOINT_NAMES, checked rather than assumed."""
    import mujoco
    names = []
    for i in range(nu):
        j = model.actuator_trnid[i, 0]
        names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))
    if names != JOINT_NAMES:
        bad = [(i, a, b) for i, (a, b) in enumerate(zip(names, JOINT_NAMES))
               if a != b]
        raise ValueError(
            "MJCF actuator order does not match the Unitree motor order; "
            "commands would drive the wrong joints. First mismatches: %s"
            % bad[:4])
    return True
