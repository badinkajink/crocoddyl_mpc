#!/usr/bin/env python3
"""How long does this robot stay up under a CONSTANT joint-level command?

The number every replay result has to be read against, and it was never measured.
S12 concluded that its open-loop replays failed because the plan was open loop;
this asks what happens with no plan at all -- put the robot at a keyframe, command
that pose, and let go.

Two commands, because the difference between them is informative:

  ctrl = q                      the naive one.  The servo has to make its own
                                gravity torque out of tracking error, so every
                                loaded joint sags before it holds.
  ctrl = q + tau_gravity / kp   the servo is handed the exact static-equilibrium
                                torque the study's own QP solves for, so at t = 0
                                the pose is an exact equilibrium.

Neither holds, because neither can see the base: a standing biped is an inverted
pendulum and the ankle strategy that stabilises it is a function of the CoM, which
a joint-space setpoint does not contain.  What the second command buys is time.

usage: croco_plant.py [--key stand] [--seconds 8]
"""

import argparse

import numpy as np

import croco_bridge as cb          # first: sets RTLD_GLOBAL
import contact_select as cs
import croco_replay as cr
import mujoco


def survival(key="stand", seconds=8.0, gravity_ff=False, fall_z=0.55):
    m, d = cs.load(ik_margin=0.0)
    kp, _ = cr.servo_gains(m)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, key)
    mujoco.mj_resetDataKeyframe(m, d, kid)
    mujoco.mj_forward(m, d)
    tau = cs.equilibrium_qp(m, d, ())["tau"] if gravity_ff else np.zeros(m.nu)
    mujoco.mj_resetDataKeyframe(m, d, kid)
    q0 = d.qpos[7:cb.NQ_ROBOT].copy()
    d.ctrl[:] = q0 + tau / kp
    n = int(round(seconds / m.opt.timestep))
    for i in range(n):
        mujoco.mj_step(m, d)
        if d.qpos[2] < fall_z:
            return i * m.opt.timestep, float(d.qpos[2])
    return None, float(d.qpos[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="stand")
    ap.add_argument("--seconds", type=float, default=8.0)
    args = ap.parse_args()
    print(f"constant joint command from keyframe '{args.key}', "
          f"fall = pelvis below 0.55 m")
    for label, ff in (("ctrl = q", False), ("ctrl = q + tau_gravity/kp", True)):
        t, z = survival(args.key, args.seconds, ff)
        print(f"  {label:28s} -> "
              + (f"falls at {t:.2f} s" if t else
                 f"still up at {args.seconds:.1f} s (pelvis {z:.3f} m)"))


if __name__ == "__main__":
    main()
