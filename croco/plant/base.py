"""What a controller is allowed to know about the thing it is driving.

THE POINT OF THIS FILE. Deploying the crocoddyl MPC to the digital twin, and
then to the robot, does not need a new controller -- it needs a new PLANT. The
controller already emits `(q_des, kp, kd, tau_ff)` (see control/command.py); what
differs between "MuJoCo in this process", "the twin over DDS" and "the machine"
is only:

    how you read the state           read()
    where the command goes           write()
    what time it is                  now()

Three methods. Everything else the MJPC deploy node carries -- 2683 lines of
`deploy_common.cc` with an embedded `mjpc::Agent`, its planner thread, its task
registry and the clang-13/LTO/libc++ build that pins it to one container -- is
irrelevant here, because this stack never links libmjpc. It reads one MJCF and
plans in Pinocchio.

WHAT IS DELIBERATELY NOT IN THIS INTERFACE.

  * No "reset". The twin and the robot cannot be reset, and a controller that
    can only run against something resettable is a controller that only runs in
    simulation. Episode setup belongs to whoever built the plant.
  * No ground truth. `State.base` is an ESTIMATE on every plant including the
    MuJoCo one, which is why MuJoCoPlant takes a `sense` model rather than
    handing over `d.qpos`. A controller that reads exact base pose works in sim
    and fails on hardware, and the repo's own rule is that ground truth is never
    on by default.
  * No wall clock. `now()` is the PLANT's clock -- sim time under MuJoCo, the
    lowstate tick under DDS -- because latency compensation and the plan index
    are both functions of the plant's time, not the operator's.

STALENESS IS PART OF THE STATE, not an exception. A DDS plant can hand back the
same sample twice, or nothing at all, and a controller that cannot tell has no
way to fail safe. `State.age` is seconds since the sample was produced; the
runtime loop is expected to treat `age > stale_after` as a damping-stop, which
is what the MJPC node's H1 watchdog does at 50 ms.
"""
from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class State:
    """One measurement of the robot, as the controller is allowed to see it.

    q, v are the ACTUATED joints only, in MJCF/Unitree order (nu of them). The
    floating base is separate and explicitly an estimate.
    """

    t: float                      # plant clock, seconds
    q: np.ndarray                 # (nu,) joint positions
    v: np.ndarray                 # (nu,) joint velocities
    base_pos: np.ndarray | None = None      # (3,) world, ESTIMATE
    base_quat: np.ndarray | None = None     # (4,) w,x,y,z world, ESTIMATE
    base_linvel: np.ndarray | None = None   # (3,)
    base_angvel: np.ndarray | None = None   # (3,)
    tau: np.ndarray | None = None           # (nu,) measured torque, if the plant has it
    age: float = 0.0              # seconds since produced; 0 for in-process plants

    @property
    def has_base(self) -> bool:
        return self.base_pos is not None and self.base_quat is not None


class Plant:
    """The three things a controller needs. Subclasses implement all three.

    A plant is NOT required to be steppable: the DDS plant advances because the
    twin does, and `step()` is a no-op there. The runtime loop therefore paces
    itself on `now()` and never assumes it owns the clock.
    """

    nu: int = 0
    #: (nu,) actuator torque limits the plan should be clamped against, or None.
    tau_limit: np.ndarray | None = None
    #: ((nu,), (nu,)) joint range for setpoint clamping, or None.
    q_range: tuple | None = None

    def read(self) -> State:
        raise NotImplementedError

    def write(self, cmd) -> None:
        """Apply a control.Command. Must not block longer than a control period."""
        raise NotImplementedError

    def now(self) -> float:
        raise NotImplementedError

    def step(self, dt: float) -> None:
        """Advance a plant that this process owns. No-op for external plants."""

    def close(self) -> None:
        """Release sockets / participants. Safe to call twice."""

    # -- convenience shared by every plant ---------------------------------

    def safe_hold(self, kd=2.0):
        """The command to send when the state is stale or the loop is exiting.

        kp = 0, tau = 0, kd > 0 is a DAMPING STOP: the robot resists motion but
        is not driven toward any setpoint. Commanding a stale setpoint instead is
        how a watchdog turns a sensor dropout into a fall.
        """
        from ..control.command import Command
        n = self.nu
        # kp must be positive for Command's MuJoCo inversion, so a damping stop
        # is expressed as "hold where you are, with no feedforward" and the
        # plants that can express kp=0 natively (DDS) override this.
        st = self.read()
        return Command(q_des=st.q, kp=np.full(n, 1e-6), kd=np.full(n, kd),
                       tau_ff=np.zeros(n))
