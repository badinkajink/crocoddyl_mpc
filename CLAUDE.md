# CLAUDE.md — crocoddyl_mpc working agreement

This repo is a **research study**, not a product: almost every file is an
argument backed by a measurement, and the comments carry the measurement. Read
the comment above a constant before changing the constant.

## 0. Definition of done — apply to every task, without being asked

A change is not finished when the code runs. It is finished when:

1. **It is measured.** No claim about behaviour ships without the number that
   supports it and the command that produced it. "Should be better" is not a
   result. If a measurement contradicts an earlier claim of mine, the
   contradiction goes in the write-up.
2. **Videos are rendered.** Any change that alters what the robot *does* gets
   an mp4 next to the cell it came from. The session path renders by default
   (`croco_twin --gui`, one per contact mode); batch runs need `--video`.
   A behavioural claim with no video is a claim nobody can check in ten seconds.
3. **It is documented where it will be found.** A measured trade-off goes in
   the docpage AND in a comment at the constant it justifies. `docs/index.html`
   gets the new page's row in the same commit.
4. **It is committed and pushed.** `git add -A && git commit && git push` on
   `crocoddyl-mpc`. This repo is gitignored in the `Humanoid_Simulation`
   superproject and is deliberately NOT a submodule, so nothing else pins it and
   an uncommitted working tree is the only copy that exists.
5. **Memory is updated** when a finding would cost more than ten minutes to
   rediscover.

Ask before: changing a default that a certified cell depends on, deleting run
artifacts, or force-pushing. Do not ask before committing ordinary work.

## 1. Environment — the parts that bite

```bash
export CL_ASSETS_DIR=../CL_Assets            # or the staged checkout
export STAGE_ROOT=studies/runs/_stage
export LEAN_TASK_DIR=$STAGE_ROOT/mjpc/tasks/humanoid_bench/lean
export MUJOCO_GL=egl                         # windowing GL only for --viewer
PY=~/miniconda3/envs/croco/bin/python        # NOT base: base segfaults
```

- **The `croco` conda env is mandatory.** crocoddyl in the base env segfaults
  inside contact dynamics. `studies/run_session.sh` and `solve_tasks.sh` resolve
  the interpreter themselves; match them.
- **Never `export LD_PRELOAD`.** The OpenMP libcrocoddyl at
  `~/opt/crocoddyl-omp/lib/libcrocoddyl.so` is selected per process, in this
  order: `timeout N env LD_PRELOAD=... $PY ...`. Exported, it drags libpinocchio
  into `git` and `timeout` and breaks both — which silently records
  `commit: "unknown"` in a plan's provenance. Do not preload the offline solve;
  it is ~4 s and gains nothing.
- **Bash tool calls have a default 2-minute timeout.** A 40 s hold plus the OCP
  build exceeds it. Pass an explicit timeout or background the job; a killed
  sweep has clobbered an artifact mid-run before.

## 2. The shape of the thing

- A **cell** is a directory with `modes.json` (a reach target + a certified `q*`
  per contact mode). Made by `croco_modes.py`. `croco_run.py` only solves *into*
  one.
- A **plan** is `plan_<tag>.json` + `xs_<tag>.npy` + `us_<tag>.npy`. A task is a
  plan, not a set of weights: **phases differ by contact set, and no weight can
  add a contact.**
- `croco_twin.py --gui` runs a session against the plans in a cell, with
  **task** (brace / stand / recover) and **contact mode** as separate dropdowns.
  Switching mode chains — the robot stays braced where it is.
- `--dt 0.02` always. `croco_run` defaults to 0.01, which silently halves the
  maneuver and changes the cost.

## 3. Standing decisions — do not silently revisit

| Decision | Why |
|---|---|
| Default mode is **elbow+forearm**, not elbow+palm | elbow+palm's palm carries **0.00 N** for an entire hold; the gripper sits 9–33 mm above the wood at `q*`. The static QP ranked it first by crediting a force that does not exist. |
| Default **contact Kp = 50** (`CONTACT_GAINS`) | At Kp = 0 a contact constrains velocity only, so brace drift is permanent and the hold never settles, in every mode. 50 settles it: sink 33 → 17 mm, tail rate 0.87 → −0.03 mm/s. **Not monotonic — 10 and 20 are worse than 0.** |
| MPC horizon stays at **35 nodes** | The user's call. Horizon is the only near-linear speed lever; do not trade it away for a marginal solve time. |
| `SITES5` and `ARM_SITES` keep their existing membership | Every certified cell and ranking in the study was computed over them. The `wrist` site is additive and opt-in via `croco_modes --sites`. |
| Ground truth is never a default | Same rule as the parent repo: a controller that consumes it works in sim and fails on hardware. |

## 4. Traps that have cost a session each

- **The e-stop is not what limits hardware reach, and the knee is.** Replaying
  every rollout through `h12_safety_layer`'s own check (`studies/estop_replay.py`,
  `relax_safety_split.yaml` -- what real bringup launches) clears all three
  channels: velocity peaks at **0.22x** the trip level, torque at **0.81x**, and
  the only position violations are **8 mrad** of MuJoCo soft-constraint overshoot.
  Torque trips are *structurally* impossible here -- every MJCF `forcerange` is
  0.18-0.90x its safety limit, so the sim reaches 129 cm while modelled weaker
  than the robot is permitted to be. But under the config as deployed
  (`position_offset = 0.0001`, zero tolerance) **100% of rollouts trip**, always
  on a knee, median 1.0 s: this robot stands with its knees on the hyperextension
  stop **34-67% of every episode**. Sim-side that is solver softness; on hardware
  it is two joints parked on a stop with no margin in the trip band.

- **`hand_x_settled` is world x; Allen's reach is base x.** The two have been
  compared directly, and the frames differ by exactly the sim's 19 cm base
  offset, so the difference cancels and the mismatch reads as agreement. The
  same target reconciliation that pins the real table to 2 mm also pins the real
  *robot*: its base stands **0.448 m** from the slab where the sim's stands
  **0.260 m** -- 19 cm further back, confirmed independently by the reach
  arithmetic (both hands reach the same table point, and 98.8 - 79.8 = 19.0 cm).
  Read from each robot's own base: targeted reach **sim 80 vs real 99 cm**, max
  reach **sim 110-116 vs real 99 cm**. The published "8 mm agreement at the
  target, 29 cm apart at max reach" is two frame errors. Unresolved until Allen
  gives base x at t = 0 (or a t = 0 AprilTag-to-base transform); nothing has been
  moved on the suspicion. See the star block at `brace_vs_stand.REALPOSE_TARGET`.

- **`Trunk Clear` is capped, and blind to load.** `lean_simple.cc` term 4 is
  `max(0, 0.05 - gap)` at weight 1500, and a rigid slab clamps `gap` at ~0 -- so
  the whole cost of lying on the table is **75 units and cannot grow**, and a
  feather-touch and a 402 N chest-load score identically. Against `Reach`
  (weight 400) that is 187 mm, so a torso pitch buying ~19 cm of hand travel
  *pays for it*. At the hardware's target the planner takes the trade in **12 of
  16** braced rollouts (68-402 N, median 150 N = 23% BW), nothing falls, and the
  rest buys **no accuracy** (8.6 vs 7.5 mm settled error). Clean runs park at
  +43 mm, just inside the +50 mm shoulder where the residual goes to zero. Any
  trunk-contact rate out of this task measures that ceiling, not the robot --
  hardware rested on 2 of 15, both failures. See the star comment at the term.

- **The declared contact set does not allocate load.** Contacts are prescribed
  kinematic constraints; forces fall out of the dynamics. A mode named
  `elbow+forearm` can carry 111 N through the elbow, 1.8 N through the forearm
  and 33 N through an *uncommanded* wrist pad. Always read `F_other` and
  `F_other_bodies` before believing a mode's name.
- **`left_wrist_pad` has contype/conaffinity 0 and collides anyway**, through an
  explicit `<contact><pair>`. It looks disabled and is the surface actually
  carrying the brace.
- **`show_gripper` moves `geom_group` only** — it has never touched
  contype/conaffinity. "The gripper collision is off" is false; the gripper is
  in the air.
- **Two plans claiming the same (kind, mode)** are resolved alphabetically.
  `discover_plans` now reports it instead of picking silently.
- **A chained episode reads `chain_k0` / `chain_k0_dist`.** A large `k0` means
  the plan's seed pose is wrong, not that projection is working.
