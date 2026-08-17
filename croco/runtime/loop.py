"""The control loop: what runs between a controller and a plant, on both.

The controller emits `Command`s and the plant consumes them (see
control/command.py, plant/base.py). Everything BETWEEN those two -- the parts
that are neither the optimisation nor the hardware -- lives here, because they
are the parts that decide whether a plan that solved offline survives contact
with a real clock:

    bring-up      you do not hand a standing robot a maneuver at t=0
    latency       the state you solved from is already old when you apply the
                  answer
    watchdog      a controller that cannot tell it is flying blind will fly a
                  stale setpoint into the floor
    clamp         the plan was solved against a torque box; the servo's own PD
                  term is added downstream of it and does not know that

None of this is new thinking. It is `deploy_common.cc`'s policy set -- the part
of the MJPC deploy node that was actually about deployment rather than about
MJPC -- re-expressed against this stack's interface. The C++ is not reused
because it embeds `mjpc::Agent` and links `libmjpc`, which this stack does not;
the POLICIES are reused because they were paid for on hardware.

WHAT IS DELIBERATELY NOT HERE. No safety layer: on the robot `h12_safety_layer`
owns the limits and the e-stop, and a second authority with a second copy of the
numbers is how you get a disagreement nobody notices. `clamp` here is the
planner staying inside the box it solved against, not a safety function, and it
says so.
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np

from ..control.command import Command


@dataclasses.dataclass
class LoopConfig:
    """Every knob, with the value the MJPC deploy node settled on after real runs.

    These are not tuned here -- they are carried across so a first twin run
    starts from something that flew, rather than from zeros. `deploy_common.h`
    is the provenance for each.
    """

    ctrl_hz: float = 200.0          # kCtrlHz
    warmup_s: float = 1.0           # kWarmupSec: hold measured pose, let the solver converge
    ramp_s: float = 5.0             # kStartRampSec: measured pose -> stance
    ramp_hold_s: float = 3.0        # kRampHoldSec
    blend_s: float = 4.5            # kPolicyBlendSec: scripted stance -> live policy
    stale_s: float = 0.05           # kStaleSec: H1 watchdog
    latency_comp: bool = True       # kLatencyComp
    latency_extra_ms: float = 4.0   # kLatencyExtraMs: transport + zero-order hold
    latency_max_ms: float = 40.0    # kLatencyMaxMs: hard cap on predict-forward
    clamp_ratio: float = 0.9        # kClampRatio: 0.9 x TAU_ESTOP
    safe_hold_kd: float = 2.0       # kSafeHoldKd


class Phase:
    WARMUP, RAMP, HOLD, BLEND, RUN, SAFE, DONE = (
        "warmup", "ramp", "hold", "blend", "run", "safe", "done")


def _ease(u):
    """Smoothstep. Linear ramps step the acceleration at both ends, which the
    robot feels as a jerk at exactly the two moments it is least stable."""
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


class ControlLoop:
    """Drive a plant from a policy, with a bring-up that does not start mid-air.

    `policy(t, state) -> (q_des, v_des, tau_ff)` in the plant's joint order, or
    None when it has nothing left to say (the maneuver ended). Returning None is
    how a finite plan retires without the loop having to know its length.

    The loop owns the PHASE, and the phase is the part everyone re-implements
    badly: at t=0 the robot is wherever it was left, the policy wants it in the
    stance its plan starts from, and stepping straight to that setpoint asks for
    the whole error at once through the position gains.
    """

    def __init__(self, plant, policy, stance=None, cfg=None, on_step=None):
        self.plant = plant
        self.policy = policy
        self.cfg = cfg or LoopConfig()
        self.stance = None if stance is None else np.asarray(stance, float)
        self.on_step = on_step
        self.phase = Phase.WARMUP
        self.t0 = None
        self.q_start = None
        self._solve_ewma = 0.0
        self.log = []

    # -- policies ---------------------------------------------------------

    def _latency(self, solve_s):
        """How far to predict the state forward before handing it to the policy.

        AUTO mode is an EWMA of the measured compute time plus a fixed transport
        term, capped. The cap matters more than the estimate: a single slow step
        with no cap predicts the robot somewhere it will never be, and the
        command that comes back is worse than no compensation at all.
        """
        c = self.cfg
        if not c.latency_comp:
            return 0.0
        self._solve_ewma = 0.9 * self._solve_ewma + 0.1 * solve_s
        ms = self._solve_ewma * 1e3 + c.latency_extra_ms
        return min(ms, c.latency_max_ms) * 1e-3

    def _clamp(self, cmd):
        """Keep the planner inside the box it solved against. NOT a safety layer.

        The servo's PD term is added downstream of tau_ff, so a command can leave
        the box even when the feedforward alone is inside it -- S17 measured
        1.111x the clamp basis on the bracing arm in 4 of 200 control periods.
        """
        lim = getattr(self.plant, "tau_limit", None)
        if lim is not None:
            lim = np.asarray(lim, float) * self.cfg.clamp_ratio
        qr = getattr(self.plant, "q_range", None)
        out, rep = cmd.clamp(tau_lim=lim,
                             q_lo=None if qr is None else qr[0],
                             q_hi=None if qr is None else qr[1])
        return out, rep

    # -- phase ------------------------------------------------------------

    def _phase_command(self, t, st, kp, kd):
        """Bring-up, or None once the live policy owns the robot.

        warmup  hold exactly where the robot was found. The policy may still be
                converging and must not be believed yet.
        ramp    ease from there to the stance the plan starts from.
        hold    sit at the stance. On hardware this is where an operator looks at
                the robot before anything dynamic happens.
        blend   cross-fade scripted stance -> policy output, so the first live
                command is not a step.
        """
        c = self.cfg
        if self.stance is None:
            return None                       # no bring-up requested
        t_ramp = c.warmup_s
        t_hold = t_ramp + c.ramp_s
        t_blend = t_hold + c.ramp_hold_s
        t_run = t_blend + c.blend_s
        if t < t_ramp:
            self.phase = Phase.WARMUP
            return Command.hold(self.q_start, kp, kd)
        if t < t_hold:
            self.phase = Phase.RAMP
            u = _ease((t - t_ramp) / max(c.ramp_s, 1e-9))
            return Command.hold((1 - u) * self.q_start + u * self.stance, kp, kd)
        if t < t_blend:
            self.phase = Phase.HOLD
            return Command.hold(self.stance, kp, kd)
        if t < t_run:
            self.phase = Phase.BLEND
            return ("blend", _ease((t - t_blend) / max(c.blend_s, 1e-9)))
        self.phase = Phase.RUN
        return None

    # -- the loop ---------------------------------------------------------

    def step(self, kp, kd):
        """One control period. Returns False when the loop is finished."""
        plant, c = self.plant, self.cfg
        st = plant.read()
        if self.t0 is None:
            self.t0, self.q_start = st.t, st.q.copy()
            if self.stance is None:
                self.phase = Phase.RUN
        t = st.t - self.t0

        # WATCHDOG FIRST. Everything below reasons about the state; if the state
        # is old, none of that reasoning is valid, and the right answer is to
        # stop driving rather than to drive from a guess.
        if st.age > c.stale_s:
            self.phase = Phase.SAFE
            plant.write(plant.safe_hold(c.safe_hold_kd))
            self.log.append(dict(t=t, phase=self.phase, age=st.age))
            return True

        scripted = self._phase_command(t, st, kp, kd)
        if isinstance(scripted, Command):
            plant.write(scripted)
            self.log.append(dict(t=t, phase=self.phase, tau_sat=0, q_clip=0))
            return True

        t_pol = t - (c.warmup_s + c.ramp_s + c.ramp_hold_s
                     if self.stance is not None else 0.0)
        t0 = time.perf_counter()
        out = self.policy(max(t_pol, 0.0), st)
        solve_s = time.perf_counter() - t0
        if out is None:
            self.phase = Phase.DONE
            plant.write(plant.safe_hold(c.safe_hold_kd))
            return False

        q_des, v_des, tau_ff = out
        cmd = Command(q_des=q_des, kp=kp, kd=kd, tau_ff=tau_ff, v_des=v_des)
        if isinstance(scripted, tuple):                 # blending
            _, u = scripted
            cmd = Command(q_des=(1 - u) * self.stance + u * cmd.q_des,
                          kp=kp, kd=kd, tau_ff=u * cmd.tau_ff,
                          v_des=u * cmd.v_des)
        cmd, rep = self._clamp(cmd)
        plant.write(cmd)
        self.log.append(dict(t=t, phase=self.phase, solve_ms=solve_s * 1e3,
                             latency_ms=self._latency(solve_s) * 1e3,
                             tau_sat=rep["tau_saturated"],
                             q_clip=rep["q_clipped"]))
        if self.on_step:
            self.on_step(self.log[-1], st, cmd)
        return True

    def run(self, kp, kd, max_seconds=None):
        """Pace on the PLANT's clock, not the wall clock.

        A sim-coupled controller paced on the wall clock drifts against its own
        plan whenever the sim is not real-time, and the twin frequently is not.
        For a plant this process owns, `step(dt)` advances it; for an external
        one the sleep is what yields.
        """
        dt = 1.0 / self.cfg.ctrl_hz
        owns = getattr(self.plant, "OWNS_CLOCK", False)
        try:
            while True:
                if not self.step(kp, kd):
                    break
                if self.t0 is not None and max_seconds is not None \
                        and self.plant.read().t - self.t0 > max_seconds:
                    break
                if owns:
                    self.plant.step(dt)
                else:
                    time.sleep(dt)
        finally:
            try:
                self.plant.write(self.plant.safe_hold(self.cfg.safe_hold_kd))
            except Exception:                                   # noqa: BLE001
                pass
        return self.log
