"""The command the H1-2 actually takes, and the two plants that consume it.

THE ONE INTERFACE. Every controller in this stack -- the offline plan replayed
open loop, the Riccati feedback, the online MPC -- ultimately emits the same
four things per joint:

    (q_des, kp, kd, tau_ff)

because that is what `rt/lowcmd` carries and what the motor driver forms its
torque from:

    tau = kp (q_des - q) + kd (v_des - v) + tau_ff                        (1)

This module is that tuple and nothing else. It exists because the tuple was
previously implicit: `croco_replay` computed it correctly and then, on the last
line before stepping, collapsed it into MuJoCo's single `ctrl` scalar per joint.
The controller was already deployment-shaped and nobody could tell, so "port the
crocoddyl MPC to the robot" looked like a rewrite when it is a plant swap.

WHY THE MUJOCO COLLAPSE IS NOT A SIMPLIFICATION. A MuJoCo <position> actuator
emits

    tau = kp (ctrl - q) - kd v                                           (2)

-- one input, and the damping term references ZERO velocity, not v_des. Equating
(1) and (2) gives

    ctrl = q_des + (tau_ff + kd v_des) / kp                              (3)

which is exact, and is the substitution `to_mujoco_ctrl` performs. Two things
about it are load-bearing and have burned this study before:

  * IT IS NOT OPEN-LOOP TORQUE. The servo's own PD term survives on both sides,
    so a "feedforward torque" run still has a position loop underneath it doing
    the stabilising. S12 called this mode "torque inversion" and read its
    survival as evidence about the plan; most of it was the servo.
  * IT INFLATES THE SETPOINT. A joint that needs tau_ff near its limit asks for
    a setpoint tau_ff/kp away from where it wants to be -- tens of degrees on a
    low-gain wrist. That is harmless in MuJoCo and is NOT harmless on hardware,
    where the safety layer clips q_des against the joint range and will silently
    truncate the feedforward. `Command.clamp` exists for that reason.

So the collapse is a property of the MuJoCo PLANT, not of the controller, and it
belongs on the MuJoCo plant's side of the interface. The DDS plant sends the
four fields as they are.
"""
from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class Command:
    """One joint-space command for the whole robot, in MJCF/Unitree joint order.

    Every field is length-nu except where noted. `v_des` defaults to zero, which
    is the right default for a setpoint hold and the wrong one for tracking a
    trajectory -- pass the plan's velocity when there is one.
    """

    q_des: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau_ff: np.ndarray
    v_des: np.ndarray | None = None

    def __post_init__(self):
        self.q_des = np.asarray(self.q_des, float)
        n = self.q_des.size
        self.kp = np.broadcast_to(np.asarray(self.kp, float), (n,)).copy()
        self.kd = np.broadcast_to(np.asarray(self.kd, float), (n,)).copy()
        self.tau_ff = np.broadcast_to(np.asarray(self.tau_ff, float), (n,)).copy()
        self.v_des = (np.zeros(n) if self.v_des is None
                      else np.broadcast_to(np.asarray(self.v_des, float), (n,)).copy())
        if np.any(self.kp <= 0):
            raise ValueError("kp must be positive: the MuJoCo inversion divides "
                             "by it, and a zero-gain joint has no setpoint to "
                             "command. Use a torque-only plant instead.")

    @property
    def nu(self) -> int:
        return self.q_des.size

    def clamp(self, tau_lim=None, q_lo=None, q_hi=None):
        """Saturate the feedforward, then the setpoint. Returns (cmd, report).

        ORDER MATTERS. Clamping tau_ff first and the setpoint second is the same
        order the hardware applies them, so what this returns is what the robot
        will do rather than what the planner asked for. The report says how much
        was given up, because a run that silently saturates looks like a control
        failure and is a limits failure.
        """
        tau = self.tau_ff
        hit_tau = 0
        if tau_lim is not None:
            lim = np.broadcast_to(np.asarray(tau_lim, float), tau.shape)
            hit_tau = int(np.sum(np.abs(tau) > lim + 1e-9))
            tau = np.clip(tau, -lim, lim)
        q = self.q_des
        hit_q = 0
        if q_lo is not None and q_hi is not None:
            lo = np.broadcast_to(np.asarray(q_lo, float), q.shape)
            hi = np.broadcast_to(np.asarray(q_hi, float), q.shape)
            hit_q = int(np.sum((q < lo - 1e-9) | (q > hi + 1e-9)))
            q = np.clip(q, lo, hi)
        out = Command(q, self.kp, self.kd, tau, self.v_des)
        return out, {"tau_saturated": hit_tau, "q_clipped": hit_q}

    def to_mujoco_ctrl(self) -> np.ndarray:
        """Equation (3): the single <position> input equivalent to this command.

        Only correct against actuators whose gains are this command's kp/kd --
        `MuJoCoPlant` reads them off the model and does not assume.
        """
        return self.q_des + (self.tau_ff + self.kd * self.v_des) / self.kp

    @classmethod
    def hold(cls, q, kp, kd, tau_ff=None):
        """Hold a pose. The control experiment every replay is read against."""
        q = np.asarray(q, float)
        return cls(q, kp, kd, np.zeros(q.size) if tau_ff is None else tau_ff)
