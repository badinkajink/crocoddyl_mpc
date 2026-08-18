#!/usr/bin/env bash
# Solve the three task plans a croco_twin SESSION offers, into one cell.
#
# WHY THIS EXISTS. The GUI's task dropdown lists `stand` and `recover` greyed
# out on almost every cell, with the reason underneath ("no plan_stand.json in
# ..."). That is honest but useless on its own: a task is a SOLVED PLAN in the
# cell directory (plan_<tag>.json + xs_<tag>.npy + us_<tag>.npy), not a set of
# weights, because the phases differ by CONTACT SET and no weight can add a
# contact. The certified S18 grid carries only the braced maneuver -- every
# cell has n_return = 0 -- so the other two dropdown entries have never been
# solved anywhere in it. This script solves them.
#
# It does NOT create the cell. `modes.json` (the reach target and the certified
# q* per contact mode) must already be there, i.e. point this at a cell
# croco_modes.py produced -- an S18 grid cell, or the one run_session.sh makes.
#
# Cost: three offline solves, measured 2.6 s + 0.6 s + 1.0 s on the S20 cell.
# It SKIPS any task whose plan already exists, so on a certified cell it adds
# `stand` and `recover` and leaves the certified maneuver alone (FORCE=1 to
# re-solve everything). It also takes dt from the cell rather than from
# croco_run's default -- see the note by DT below.
#
#   studies/solve_tasks.sh runs/2026-08-16_session18/grid/x1050_y-235_z1098_dy+000
#   studies/croco_twin.py --dir <that cell> --tag elbow_palm --plant mujoco --gui
#
# The tags are what croco_twin looks for and are not free: `stand` and
# `recover` are hardcoded in its TASKS table (brace+reach uses --tag).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

CELL="${1:?usage: solve_tasks.sh <cell-dir> [brace-tag] [mode]}"
TAG="${2:-elbow_palm}"
MODE="${3:-elbow+palm}"
FORCE="${FORCE:-0}"

[ -f "$CELL/modes.json" ] || {
  echo "no $CELL/modes.json -- this is not a cell. Cells come from" >&2
  echo "croco_modes.py (run_session.sh makes one); this only adds plans." >&2
  exit 1; }

# Same resolver as run_session.sh and croco/env.py: the pinned interpreter is
# not a preference, it is what does not SIGSEGV in ShootingProblem.
_croco_py() {
  if [ -n "${CROCO_PY:-}" ]; then echo "$CROCO_PY"; return; fi
  for base in "${CONDA_EXE%/bin/conda}" "$HOME/miniconda3" "$HOME/anaconda3" \
              "$HOME/miniforge3" /opt/conda; do
    [ -x "$base/envs/croco/bin/python" ] && { echo "$base/envs/croco/bin/python"; return; }
  done
  command -v python3
}
PY="$(_croco_py)"

if [ -z "${CL_ASSETS_DIR:-}" ]; then
  for c in "$ROOT/../CL_Assets" "$ROOT/assets/CL_Assets" "$ROOT/build/_deps/cl_assets-src"; do
    [ -d "$c/mujoco_assets" ] && { CL_ASSETS_DIR="$(cd "$c" && pwd)"; break; }
  done
fi
: "${CL_ASSETS_DIR:?set CL_ASSETS_DIR to the CL_Assets checkout}"
export CL_ASSETS_DIR
export STAGE_ROOT="${STAGE_ROOT:-$HERE/runs/_stage}"
export LEAN_TASK_DIR="${LEAN_TASK_DIR:-$STAGE_ROOT/mjpc/tasks/humanoid_bench/lean}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# THE CELL'S OWN dt, NOT croco_run's DEFAULT. croco_run defaults to dt = 0.01
# and every cell in this study is 0.02, so solving "the missing tasks" with the
# default silently produces a HALF-LENGTH maneuver: the same 200 nodes over 2 s
# instead of 4, which doubles the joint velocities and torques and is a
# different problem wearing the same tag. Caught by re-solving a certified cell
# and finding its cost had moved from 24.733 to 23.717.
DT="$("$PY" "$HERE/../croco/_cell_dt.py" "$CELL" 2>/dev/null || echo 0.02)"
echo "dt = $DT (from this cell's existing plan)"

# NEVER OVERWRITE A SOLVED PLAN. The point of this script is to ADD the tasks a
# cell is missing; silently re-solving one it already has is how a certified
# grid cell stops being the thing it was certified as. FORCE=1 to re-solve.
run() {
  local what="$1" tag="$2"; shift 2
  if [ "$FORCE" != "1" ] && [ -f "$CELL/plan_$tag.json" ]; then
    echo; echo "=== $what -- SKIP, plan_$tag.json exists (FORCE=1 to re-solve) ==="
    return 0
  fi
  echo; echo "=== $what ==="
  "$PY" "$HERE/croco_run.py" --dir "$CELL" --tag "$tag" --dt "$DT" "$@"
}

# brace+reach -- the certified maneuver. --reach-rot auto is the default in
# croco_run now; naming it here is what makes the reachRot slider and the roll
# knob exist in the panel, at a weight small enough not to move the plan.
run "brace+reach -> plan_$TAG" "$TAG" \
    --mode "$MODE" --start stand \
    --n-approach 120 --n-braced 80 --reach-rot auto --w-reach-rot 1e-2

# stand -- legs only. An empty contact subset is a DIFFERENT problem, not the
# same one with the arm costs turned down.
run "stand -> plan_stand" stand \
    --mode legs_only --start stand \
    --n-approach 120 --n-braced 80 --reach-rot auto --w-reach-rot 1e-2

# recover -- starts braced and returns. n_approach 0 because there is nothing
# to approach: the robot is already on the table. --return-start is what makes
# the return phase regulate to the STAND pose rather than back to the brace.
run "recover -> plan_recover" recover \
    --mode "$MODE" --start forearm_brace_reach --return-start stand \
    --n-approach 0 --n-braced 20 --n-return 120 \
    --reach-rot auto --w-reach-rot 1e-2

echo
echo "=== $CELL now carries ==="
ls -1 "$CELL"/plan_*.json
echo
echo "All three are selectable in the panel's task dropdown:"
echo "  studies/croco_twin.py --dir $CELL --tag $TAG --plant mujoco --gui"
