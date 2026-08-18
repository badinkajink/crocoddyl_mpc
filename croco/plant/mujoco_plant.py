"""MuJoCo in this process, behind the Plant interface.

This is what every croco replay has always driven; the only new thing is that it
now says so through an interface the twin and the robot also implement, so the
same controller code runs against all three.

TWO THINGS IT REFUSES TO DO, both of which the study got wrong before.

  1. IT DOES NOT HAND OVER GROUND TRUTH. `d.qpos[:7]` is the exact base pose and
     no robot has it. The plant therefore reports the base through a `sense`
     callable -- by default a per-run bias plus white noise, matching how S17
     modelled the estimator -- and `sense=None` (exact) has to be asked for
     explicitly. This is the repo's ground-truth rule applied one level down: a
     controller that reads exact base pose works here and fails on hardware.

  2. IT DOES NOT INFLATE THE CONTACT MARGIN. `contact_select.load` defaults to a
     25 mm collision margin so the IK can see contacts before it enters them. In
     a DYNAMICS run that same inflation makes every geom generate force 25 mm
     before it touches, which is not the contact model the plan is being tested
     against -- S12's replays ran with it on. `ik_margin=0` is not optional here.
"""
from __future__ import annotations

import contextlib

import numpy as np

#: One shared no-op context: `_guard` is called every control period and a
#: fresh nullcontext per period is pure garbage.
_NULL = contextlib.nullcontext()

from .base import Plant, State


class MuJoCoPlant(Plant):
    """In-process physics. This process advances the clock."""

    OWNS_CLOCK = True

    def __init__(self, model, data, sense=None, tau_limit=None, nu=None):
        import mujoco
        import threading

        self.data_lock = threading.Lock()
        self._mj = mujoco
        self.m, self.d = model, data
        self.nu = int(nu if nu is not None else model.nu)
        self.sense = sense
        self.tau_limit = (None if tau_limit is None
                          else np.asarray(tau_limit, float))
        self.q_range = (model.jnt_range[1:self.nu + 1, 0].copy(),
                        model.jnt_range[1:self.nu + 1, 1].copy())
        self.kp, self.kd = self.servo_gains()

    # -- gains -------------------------------------------------------------

    def servo_gains(self):
        """(kp, kd) read OFF the model, never assumed.

        A <position> actuator carries kp in gainprm[0] and -kp in biasprm[1];
        if those disagree the actuator is not the servo this code thinks it is,
        and the MuJoCo inversion in Command.to_mujoco_ctrl would be wrong.
        """
        m = self.m
        kp = m.actuator_gainprm[:self.nu, 0].copy()
        kd = -m.actuator_biasprm[:self.nu, 2].copy()
        if not np.allclose(-m.actuator_biasprm[:self.nu, 1], kp):
            raise ValueError(
                "actuator gainprm/biasprm disagree: these are not <position> "
                "servos with kp/kv, so the (q_des,kp,kd,tau_ff) -> ctrl "
                "inversion does not hold for this model.")
        return kp, kd

    # -- Plant -------------------------------------------------------------

    def now(self) -> float:
        return float(self.d.time)

    def read(self) -> State:
        nq_robot = 7 + self.nu
        q = self.d.qpos[7:nq_robot].copy()
        v = self.d.qvel[6:6 + self.nu].copy()
        bp = self.d.qpos[0:3].copy()
        bq = self.d.qpos[3:7].copy()
        bv = self.d.qvel[0:3].copy()
        bw = self.d.qvel[3:6].copy()
        if self.sense is not None:
            q, v, bp, bq, bv, bw = self.sense(q, v, bp, bq, bv, bw)
        return State(t=self.now(), q=q, v=v, base_pos=bp, base_quat=bq,
                     base_linvel=bv, base_angvel=bw,
                     tau=self.d.actuator_force[:self.nu].copy(), age=0.0)

    def write(self, cmd) -> None:
        if cmd.nu != self.nu:
            raise ValueError("command has nu=%d, plant has nu=%d"
                             % (cmd.nu, self.nu))
        if not (np.allclose(cmd.kp, self.kp) and np.allclose(cmd.kd, self.kd)):
            # The inversion is only valid against THIS model's gains. Silently
            # accepting different ones produces a torque that is not the one the
            # controller asked for, and nothing downstream would notice.
            raise ValueError(
                "command gains differ from the model's actuator gains; the "
                "MuJoCo <position> inversion would not reproduce the commanded "
                "torque. Re-issue the command with plant.kp / plant.kd, or use "
                "a torque-actuated model.")
        with self.data_lock, self._guard():     # a write to mjData
            self.d.ctrl[:self.nu] = cmd.to_mujoco_ctrl()

    #: Set to `viewer.lock` (the BOUND METHOD, not an acquired lock) while a
    #: passive viewer is attached; None otherwise. See `_guard`.
    viewer_lock = None

    #: OUR lock over mjData, as opposed to mujoco's. Two different races need
    #: two different locks and they are not interchangeable:
    #:
    #:   viewer_lock  guards us against MUJOCO'S render thread, which lives in
    #:                C++ and honours only mujoco's own lock.
    #:   data_lock    guards us against OURSELVES -- the control thread inside
    #:                mj_step versus a socket thread inside launch_passive.
    #:                `launch_passive` calls mj_forward on this same mjData
    #:                before it returns a handle, so there is a window during
    #:                which the viewer is being created and no viewer_lock
    #:                exists yet to take. That window is where the panel's
    #:                viewer button crashed: measured, 2 of 3 sessions died
    #:                with the control thread in mujoco_plant.step and a
    #:                socket thread in viewer.py launch_passive.
    #:
    #: Always take data_lock OUTSIDE viewer_lock so the order is total.

    def _guard(self):
        """The passive viewer's lock, or a no-op context.

        WHY THIS EXISTS. `mujoco.viewer.launch_passive` runs its render loop on
        its OWN thread, reading the same `mjData` this process is stepping. The
        documented contract is that every mutation of the model or data happens
        under `viewer.lock()`:

            with viewer.lock():
                mj_step(m, d)
            viewer.sync()

        croco_twin stepped outside it. The result is a genuine data race, and
        it presents the way data races do -- an intermittent SIGSEGV with no
        traceback, only when a viewer happens to be open, which is why it
        survived every deterministic test and still killed real sessions. The
        cost is one uncontended lock acquire per control period when a viewer
        is attached, and exactly nothing when it is not.
        """
        lk = self.viewer_lock
        return lk() if lk is not None else _NULL

    def step(self, dt: float) -> None:
        n = max(1, int(round(dt / self.m.opt.timestep)))
        with self.data_lock, self._guard():
            for _ in range(n):
                self._mj.mj_step(self.m, self.d)

    def safe_hold(self, kd=2.0):
        """The closest thing a POSITION servo can do to a damping stop.

        The base class's safe hold is kp = 0, kd > 0: resist motion, drive
        toward nothing. A MuJoCo <position> actuator cannot express that -- its
        kp/kd are baked into the model and `write` rejects a command carrying
        different ones (rightly: the inversion would not reproduce the commanded
        torque). So here it is "hold exactly where you are, no feedforward",
        which resists deviation but is a stiff hold rather than a soft one.

        THIS IS A REAL DIFFERENCE BETWEEN THE PLANTS, not a detail. A watchdog
        trip in MuJoCo behaves better than the same trip on hardware, so a
        stale-state recovery that only ever gets tested here is not tested. The
        `kd` argument is accepted and ignored, deliberately, rather than
        silently reinterpreted.
        """
        from ..control.command import Command
        st = self.read()
        return Command(q_des=st.q, kp=self.kp, kd=self.kd,
                       tau_ff=np.zeros(self.nu))


def default_sense(rng, base_bias_m=0.01, base_bias_rad=0.01,
                  q_noise=0.0, v_noise=0.0):
    """The estimator error model S17 used: a per-run BIAS plus a white part.

    Base error on a real robot is IMU + leg kinematics + a contact assumption.
    It is slowly varying, so modelling it as white noise makes the controller
    look better than it is -- the bias is drawn once per run and stays.
    """
    bias_p = rng.normal(0.0, base_bias_m, 3)
    bias_r = rng.normal(0.0, base_bias_rad, 3)

    def sense(q, v, bp, bq, bv, bw):
        import mujoco
        qn = q + (rng.normal(0, q_noise, q.size) if q_noise else 0.0)
        vn = v + (rng.normal(0, v_noise, v.size) if v_noise else 0.0)
        dq, out = np.zeros(4), np.zeros(4)
        mujoco.mju_euler2Quat(dq, bias_r, "xyz")
        mujoco.mju_mulQuat(out, dq, bq)
        return qn, vn, bp + bias_p, out, bv, bw

    return sense
