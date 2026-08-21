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
