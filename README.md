# crocoddyl-mpc

A crocoddyl MPC runtime for the Unitree H1-2, and the study that produced it.

The robot leans on a table, braces on its forearm and palm, and reaches a
target it cannot reach standing. The plan is solved offline as an optimal
control problem; the same plan is then flown by a receding-horizon MPC against
three different plants — in-process MuJoCo, a digital twin on the far side of
Unitree DDS, and (by the same interface) the real robot.

**It does not link MJPC.** It reads one MJCF and plans in Pinocchio. The lean
task assets live in `assets/` because this study authored them, not because
anything here needs `libmjpc`.

## Layout

| Path | What it is |
|---|---|
| `croco/` | the runtime library — solver, plants, control loop, twin, panel |
| `croco/control/mpc.py` | the controller: warm-started BoxFDDP over a sliding window |
| `croco/plant/` | `MuJoCoPlant` (in-process) and `DDSPlant` (over the wire), one interface |
| `croco/runtime/loop.py` | bring-up, latency compensation, watchdog, clamp, pacing |
| `croco/gui/` | the live panel: plots, weight sliders, session controls |
| `croco/twin/lean_twin.py` | the digital twin — physics behind Unitree DDS |
| `studies/` | the scripts that plan, replay, score, sweep and document |
| `croco/safety/` | the H1-2 safety layer's limits, evaluated without giving it the actuators |
| `assets/tasks/` | the lean model and its include closure |
| `docs/index.html` | **documentation index** — start here |
| `docs/RUN_MATRIX.md` | which command goes with which plant |
| `docs/lean/` | session writeups, in date order |

## Install

crocoddyl and pinocchio are conda-forge packages with compiled Boost.Python
bindings. There are no usable wheels, so the environment is conda, not pip.

> **Do not use conda's `base` environment.** Its pip/cmeel crocoddyl wheel
> **SIGSEGVs inside `ShootingProblem`** — measured 3/3 on this machine, twice
> as a segfault and once as a `MemoryError`, always with no traceback. Both
> environments report crocoddyl 3.2.1 and pinocchio 4.1.0, so a version check
> clears it; the difference is which pinocchio's C++ ABI crocoddyl was linked
> against. This cost a session to find.
>
> The entry points now defend themselves: `croco/env.py` re-execs into the
> pinned interpreter before anything native is loaded and prints one line
> saying so. `CROCO_PY=<python>` chooses it, `CROCO_NO_REEXEC=1` turns it off.
> Name the env `croco` and it is found automatically.

```bash
conda create -n croco -c conda-forge python=3.12 crocoddyl pinocchio mujoco \
    "numpy<2" scipy matplotlib imageio quadprog
conda activate croco
pip install -e .                     # makes `croco` importable from anywhere
pip install -e ".[twin]"             # only if you want the DDS twin
```

You also need **CL_Assets** (the H1-2 meshes, MJCFs and the magpie URDF) as a
sibling checkout or pointed at explicitly:

```bash
export CL_ASSETS_DIR=/path/to/CL_Assets
```

**h12_safety_layer is optional, and resolved the same way.** `--safety` reads
that ROS package's YAML and its limit tables rather than carrying a copy of 27
joint limits that would go stale silently. It is found at
`$H12_SAFETY_LAYER`, else `../core_ws/src/h12_safety_layer` relative to this
repo. Nothing else here needs ROS, a colcon workspace, or the robot, so
without it you simply do not get that flag. It needs PyYAML:

```bash
pip install pyyaml
```

## Build

Two steps, both of which fail *quietly* if skipped — each is a performance
knob, not a correctness one, so nothing crashes; the loop just misses its
deadline.

```bash
studies/run_session.sh deps     # native extensions + the OpenMP crocoddyl
studies/run_session.sh stage    # stage the lean model out of assets/
studies/run_session.sh check    # <- read this before anything else
```

`check` is the green light. It reports the interpreter, whether contact
dynamics survive, whether both native extensions are built, DDS availability,
OpenMP, and both asset paths.

| Knob | Cost of skipping it | Measured |
|---|---|---|
| `croco_keepout` / `croco_passive` | keep-out activation falls back to Python | 85.7 → 16.8 ms mean solve |
| OpenMP `libcrocoddyl` | `nthreads` is silently pinned to 1 | p95 21.3 → 16.7 ms |

The stock conda-forge crocoddyl has no OpenMP; `run_session.sh deps` rebuilds
it into `~/opt/crocoddyl-omp` and every entry point `LD_PRELOAD`s it per
process. It is deliberately *not* exported into the shell — `LD_PRELOAD` is
inherited by every child, and preloading libcrocoddyl into `git` stops git
starting, which is how a recorded commit hash silently became "unknown" for a
whole session.

## Run

Everything needs a **cell**, and a fresh clone has none — `studies/runs/` is
not tracked. A cell is built in two steps, and the first is not optional:

```bash
# 1. make the cell: certify a contact mode at a reach target. Writes
#    modes.json (the target) and q_<mode>.txt (the pose each mode holds).
#    ~10 s. Without this, croco_run has nothing to plan toward.
studies/croco_modes.py --out studies/runs/mycell --target 1.05 -0.2348 1.0982

# 2. solve a plan in it. --dt 0.02 is this study's timestep; croco_run's own
#    default is 0.01, which is half the maneuver.
studies/croco_run.py --dir studies/runs/mycell --mode elbow+palm \
    --tag elbow_palm --dt 0.02 --n-approach 120 --n-braced 80
```

That gives the cell **one** task. The panel offers three, and the other two
are greyed out until they exist — a task is a solved plan, not a weight
preset. `studies/solve_tasks.sh studies/runs/mycell` solves whichever are
missing, in about five seconds, and skips any that already exist.

Then fly it:

```bash
studies/croco_twin.py --dir studies/runs/mycell --tag elbow_palm \
    --plant mujoco --gui
```

Measured on a cell built exactly this way: plan cost 24.73, reach error
0.32 mm, and the loop holds the pelvis at 0.955 m through all 198 periods.

**`docs/RUN_MATRIX.md` is the page to keep open**: which command goes with
which plant, what each one can tell you, and the two symptoms that look like
missing features but are missing artifacts.

### The panel (interactive)

```bash
studies/croco_twin.py --dir studies/runs/mycell --tag elbow_palm \
    --plant mujoco --gui
```

Opens `http://127.0.0.1:8770/` in about two seconds — before the OCP build, so
you can watch it. Episodes run until you stop them. The panel gives you
play/pause, reset, the MuJoCo viewer, speed, task and submode, a live reach
target, gripper roll, and a slider per cost term.

- **Tasks** are solved plans, not weight presets: the phases differ by *contact
  set*, so switching rebuilds the OCP (1.2 s) and rewinding costs 8.8 ms.
- **Submodes** — `single-shot` runs once and holds; `hold` freezes the plan
  index at the last node and keeps solving; `automode` chains tasks back to
  back without putting the robot back.
- **Speed** below 1× hands the solver wall clock the robot would not give it.
  The overrun count then stops being a deployment result, so it is recorded in
  the artifact and printed. Use it to ask *"would this survive if compute were
  free"*, not to make a number look better.

The viewer needs a windowing GL backend — `MUJOCO_GL=egl` is offscreen and the
button disables itself with that explanation.

### Batch (what the study runs)

Without `--gui` this is a one-shot: build, run once, print a JSON summary,
exit. That is what every recorded result came from, and what `twin_grid.sh`
drives 26 times.

```bash
studies/croco_twin.py --dir studies/runs/mycell --tag elbow_palm --plant mujoco --out r.json
studies/run_session.sh grid          # certify, plan, stress, collect
```

### Against the DDS twin

Two processes, and **both must agree on the speed factor** — the twin owns the
far clock, so `croco_twin --realtime` cannot reach it:

```bash
# terminal 1
python -m croco.twin.lean_twin --model $LEAN_TASK_DIR/Lean_H12_Magpie.xml \
    --key stand --publish-truth --realtime 1.0
# terminal 2
studies/croco_twin.py --dir studies/runs/mycell --tag elbow_palm --plant dds --base truth
```

`--base truth` reads the twin's **ground truth** base pose. It has to be typed
because it is the one privileged input in the loop: `rt/lowstate` carries no
base pose and no robot has one. `--base estimator` runs the proprioceptive
estimator instead, which is what the robot would use.

`ROS_DOMAIN_ID=0` is the real robot's command bus. The twin scripts refuse it.

### Through the safety layer

`h12_safety_layer` is the relay that sits between any controller and the real
H1-2. It clips commands and e-stops on out-of-range state. Two levels, and
they are not the same thing:

```bash
# MONITOR — no authority, safe on every plant, this is the one to start with
studies/croco_twin.py --dir studies/runs/mycell --tag elbow_palm --plant mujoco \
    --gui --safety default_safety_full

# IN THE PATH — publishes to rt/safety/lowcmd_in instead of rt/lowcmd,
# so a RUNNING layer clips and forwards. --plant dds only.
studies/croco_twin.py ... --plant dds --safety --via-safety
```

`--safety` computes the layer's own predicates on the state the loop already
read and marks the periods it *would* have tripped — red rules on the panel's
period plot, `safety_*` fields in the artifact. It never touches a command.

`--via-safety` gives the layer authority, and **its e-stop latches**: on the
first violation it emits `mode 0, kp = kd = tau = 0` forever, so the robot goes
limp mid-maneuver and stays limp. That is correct for hardware and ruinous for
a study run, which is why it is opt-in and why the monitor exists.

Measured on the certified brace+reach against `default_safety_full`: **0
e-stop trips**, but 16–18 of 198 periods would be **clipped** — `tau` (the
layer derates the URDF torque limits to 60%, below what the MPC clamps to) and
`dq` (a 10% velocity band, under the maneuver's own joint speeds). Clipping
stops nothing and is silent, but a clipped command is not the command the plan
describes.

## Gotchas

- **A fresh clone has no cells.** `studies/runs/` is untracked by design, and
  the panel's `stand`/`recover` entries stay greyed out until you solve them
  (`studies/solve_tasks.sh <cell>`).
- **`base` conda env segfaults.** Now caught and re-exec'd around. See Install.
- **No `reachRot` slider** means the cell was solved with `w_reach_rot = 0`,
  not that the feature is missing. `--reach-rot auto` adds it without
  re-solving anything.
- **`deps` before `check` before anything.** Both extensions must say native.
- **Task ≠ weights.** A cost that is not in the built models cannot be added to
  them — that is why plans are solved with `--reach-rot auto` at a token
  weight, so the gripper-orientation term *exists* and the panel can point it.
- **Moving the reach target live works within reason.** The offline warm start
  still descends toward the original target; a large relocation wants a
  re-solve, not a nudge.
- **Chained behaviours do not check their seams.** Transitions are measured
  (`chain_dq_max_rad`, `chain_dbase_mm`) and reported, not gated — up to
  1.1 rad and ~70 mm have been observed and survived, but nothing guarantees it.
