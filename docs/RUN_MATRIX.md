# Run matrix — which command, for which plant

Three plants, and they are not interchangeable: each answers a different
question, costs a different amount of setup, and reports a different kind of
number. Pick by what you are trying to find out.

| | `--plant mujoco` | `--plant dds` | `--plant dds --via-safety` |
|---|---|---|---|
| **physics** | in this process | `lean_twin`, another process | same |
| **command path** | `d.ctrl` | `rt/lowcmd` | `rt/safety/lowcmd_in` → **h12_safety_layer** → `rt/lowcmd` |
| **answers** | is the controller right | does it survive the wire | would the robot's own relay let it happen |
| **needs** | nothing | a twin on the same DDS domain | a twin **and** the safety layer running |
| **reset button** | yes | no — the twin owns its physics | no |
| **viewer button** | yes | no — run `lean_twin --viewer` | no |
| **can go limp on you** | no | no | **yes** — the e-stop latches |
| **measured pelvis z** | 0.955 | 0.741 @1×, 0.890 @0.5× | not yet measured |

`--safety` is orthogonal to all three: it MONITORS the layer's limits and never
touches a command, so it is safe on every plant including the first. Turn it on
before you turn on `--via-safety`.

## 1. In-process — the default, and where to start

```bash
studies/croco_twin.py \
    --dir studies/runs/2026-08-17_session20_gui/cell \
    --tag elbow_palm --plant mujoco --gui
```

Panel on <http://127.0.0.1:8770/>. Add `--viewer` for MuJoCo's window (needs a
windowing GL backend — `MUJOCO_GL=egl` is offscreen and the button disables
itself), and `--realtime 0.25` to make it watchable.

Free-running by default, which means **there is no deadline and therefore no
overruns**. That is not a pass. Pin it to real time to find out:

```bash
... --plant mujoco --gui --realtime 1.0     # 22/199 periods over, measured
```

## 2. Over DDS — the twin, one wire away from the robot

Two processes. The twin owns the clock, so its `--realtime` and the
controller's must match; the controller cannot reach across and set it.

```bash
# terminal 1 — the twin
studies/croco_twin.py --dir <cell> --tag elbow_palm --emit-qpos0 /tmp/q0.txt
python -m croco.twin.lean_twin --qpos0 /tmp/q0.txt --realtime 1.0

# terminal 2 — the controller
studies/croco_twin.py --dir <cell> --tag elbow_palm --plant dds \
    --base truth --gui --realtime 1.0
```

`--base truth` reads `rt/sim_state`, which is ground truth and exists only in
simulation. `--base estimator` is the honest one and needs the estimator node
running.

## 3. Through the safety layer — what the robot's relay would allow

**Read this before running it.** The layer's e-stop LATCHES: on the first
violation `_publisher_loop` starts emitting `make_estop_cmd` (mode 0,
kp = kd = tau = 0) and never stops until the process is restarted. On the twin
that is a robot that goes limp mid-maneuver. Find out whether it would trip
*before* giving it authority:

```bash
# step 1 — monitor only. No authority, any plant, always safe.
studies/croco_twin.py --dir <cell> --tag elbow_palm --plant mujoco \
    --gui --safety default_safety_full
```

Red rules on the period plot mark the periods it would have tripped; the first
one is solid because it is the only one that would really have happened. The
run artifact carries `safety_estop_periods`, `safety_first_estop_why` and
`safety_clip_kinds`.

Measured on the certified brace+reach against `default_safety_full`:
**0 e-stop trips**, but **16–18 of 198 periods clipped** — `tau` (the layer's
`torque_ratio: 0.60` derates the URDF limits below what the MPC clamps to) and
`dq` (`velocity_ratio: 0.10` puts the band as low as 0.9 rad/s, under the
maneuver's own joint speeds). Clipping is silent and stops nothing, but a
clipped command is not the command the plan describes.

```bash
# step 2 — the layer in the path, for real.
python -m h12_safety_layer.script.safety_layer_main \
    --config default_safety_full.yaml            # terminal 3

studies/croco_twin.py --dir <cell> --tag elbow_palm --plant dds \
    --base truth --gui --safety --via-safety     # terminal 2
```

`--via-safety` only changes the topic commands are published to. If the layer
is not running, nothing reaches `rt/lowcmd` at all and the robot receives
nothing — that is the failure mode to expect, not a fall.

The safety layer is found the same way `CL_Assets` is: `$H12_SAFETY_LAYER`,
else `../core_ws/src/h12_safety_layer` relative to this repo. A sourced ROS
workspace's *installed* copy is used only as a last resort, because it is a
build of some past state of that source.

## Tasks, and why two of the three are usually greyed out

A task is a **solved plan** in the cell directory, not a set of weights: the
phases differ by contact set and no weight can add a contact. The certified
S18 grid only ever solved the braced maneuver, so `stand` and `recover` are
offered-but-disabled on every cell in it. Five seconds fixes that:

```bash
studies/solve_tasks.sh studies/runs/2026-08-16_session18/grid/x1050_y-235_z1098_dy+000
```

## `reachRot` is missing from the sliders

Those same cells were solved with `w_reach_rot = 0`, so the orientation cost
does not exist in the built models and cannot be added to one. Either rebuild
with the flag, which costs nothing and re-solves nothing:

```bash
studies/croco_twin.py --dir <cell> --tag elbow_palm --plant mujoco --gui \
    --reach-rot auto
```

or re-solve the cell with `solve_tasks.sh`. The override is recorded in the
run artifact as `ocp_overrides`, because the warm start still comes from a
plan solved without it.

## The interpreter is pinned

Running under the wrong crocoddyl build is a **SIGSEGV inside
`ShootingProblem`** with no traceback — measured 3/3 on this machine's conda
`(base)`, whose pip/cmeel wheels report the same versions as the working
conda-forge ones. The entry points therefore re-exec themselves into the
pinned interpreter and print one line saying so. `CROCO_PY=<python>` chooses
it; `CROCO_NO_REEXEC=1` turns it off.

## Solve time: which knob buys milliseconds

Measured on the certified cell, in process, 199 periods, all native extensions
built. `--realtime 1.0` so the deadline is real.

| horizon | mean | p95 | overruns | tau saturated | pelvis z |
|---|---|---|---|---|---|
| 10 | 4.91 ms | 7.82 | 2/200 | 116 | 0.9391 |
| 15 | 6.75 | 10.00 | 1/199 | 78 | 0.9372 |
| 20 | 7.96 | 12.55 | 2/199 | 69 | 0.9402 |
| **25** | **8.86** | **13.53** | **2/199** | **42** | **0.9470** |
| 30 | 12.72 | 21.04 | 22/199 | 7 | 0.9539 |
| 35 (default) | 14.42 | 24.55 | **52/199** | 12 | 0.9555 |

Solve time is **near-linear in the horizon**, and so is the deadline story:
the default 35 misses 52 of 199 periods at 1×, while 25 misses 2. What you pay
is posture — the pelvis sits ~9 mm lower at 25 and ~18 mm lower at 15, and
torque saturation climbs steeply as the horizon shortens (42 events at 25,
116 at 10). A short horizon does not see the brace coming, so it arrives late
and pushes harder.

**`--horizon 25` is the recommended operating point**: p95 comfortably inside
the 20 ms period, 26× fewer overruns, and a centimetre of posture.

The other two knobs:

| threads | mean | | iters | mean |
|---|---|---|---|---|
| 1 | 15.79 ms | | 1 (default) | 12.55 ms |
| 4 | 13.48 | | 2 | 23.32 |
| 12 | 11.38 | | 3 | 33.79 |
| 20 | 11.21 | | | |

Threading gives only **1.4× for 20 threads** and is flat past 12 — one of the
solver's three stages is parallel, so Amdahl caps it. Iterations are linear and
already at the floor. Neither is a lever; the horizon is.
