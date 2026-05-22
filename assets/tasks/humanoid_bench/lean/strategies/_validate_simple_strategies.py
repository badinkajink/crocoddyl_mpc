#!/usr/bin/env python3
"""Closed-loop MPC validation of the standalone simple-task strategies.

Drives the real agent_server: for each new Strategy index, resets to home,
runs MPC closed-loop for a few sim-seconds, then checks the robot did not
fall and that the targeted joints moved toward the pose-library keyframe.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../../../../python")))
from mujoco_mpc import agent as agent_lib

BIN = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "../../../../../build/bin/agent_server"))
# qpos index -> target angle, per strategy (handless composite, base at [0:7])
TASKS = {
    6:  ("stand",            {}),
    7:  ("reach_forward",    {}),                       # reach cost; just check upright
    8:  ("crouch",           {10: 1.0, 16: 1.0, 8: -0.6, 14: -0.6}),
    9:  ("arms_sideways",    {21: 1.5, 28: -1.5}),
    10: ("arms_forward",     {20: -1.5, 27: -1.5}),
    11: ("arms_overhead",    {20: -2.8, 27: 2.8}),
    12: ("single_arm_raise", {28: -1.5}),
    13: ("lean_left",        {9: -0.25, 15: -0.25}),
    14: ("lean_right",       {9: 0.25, 15: 0.25}),
    15: ("torso_twist",      {19: 0.6}),
}
NSTEPS = int(os.environ.get("NSTEPS", "240"))

def reach_frac(home, final, target):
    # fraction of the way from home(=0) to target the joint actually moved
    if abs(target) < 1e-6: return 1.0
    return (final - home) / target

def main():
    print(f"agent_server: {BIN}\nNSTEPS={NSTEPS}\n")
    with agent_lib.Agent(task_id="Lean H12", server_binary_path=BIN) as ag:
        npass = 0
        for idx, (name, targ) in TASKS.items():
            ag.reset()
            ag.set_task_parameter("Strategy", float(idx))
            ag.set_task_parameter("Phase", -1.0)
            ag.step()  # triggers strategy load
            home = np.array(ag.get_state().qpos)
            t0 = time.time()
            for _ in range(NSTEPS):
                ag.planner_step(); ag.step()
            q = np.array(ag.get_state().qpos)
            metrics, phase = ag.get_metrics()
            finite = bool(np.all(np.isfinite(q)))
            basez = q[2] if finite else float("nan")
            fell = (not finite) or basez < 0.5
            fr = {f"q{i}": round(reach_frac(home[i], q[i], a), 2)
                  for i, a in targ.items()}
            ok = (not fell) and (not targ or
                                 np.mean([reach_frac(home[i], q[i], a)
                                          for i, a in targ.items()]) > 0.4)
            npass += ok
            print(f"[{idx:2d} {name:16s}] {'PASS' if ok else 'FAIL'} "
                  f"phase={phase!r} basez={basez:.2f} "
                  f"{'FELL ' if fell else ''}reach={fr} ({time.time()-t0:.0f}s)")
        print(f"\n{npass}/{len(TASKS)} strategies passed")
        return 0 if npass == len(TASKS) else 1

if __name__ == "__main__":
    sys.exit(main())
