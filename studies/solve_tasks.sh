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

run() { echo; echo "=== $1 ==="; shift; "$PY" "$HERE/croco_run.py" --dir "$CELL" "$@"; }

# brace+reach -- the certified maneuver. --reach-rot auto is the default in
# croco_run now; naming it here is what makes the reachRot slider and the roll
# knob exist in the panel, at a weight small enough not to move the plan.
run "brace+reach -> plan_$TAG" \
    --mode "$MODE" --start stand --tag "$TAG" \
    --n-approach 120 --n-braced 80 --reach-rot auto --w-reach-rot 1e-2

# stand -- legs only. An empty contact subset is a DIFFERENT problem, not the
# same one with the arm costs turned down.
run "stand -> plan_stand" \
    --mode legs_only --start stand --tag stand \
    --n-approach 120 --n-braced 80 --reach-rot auto --w-reach-rot 1e-2

# recover -- starts braced and returns. n_approach 0 because there is nothing
# to approach: the robot is already on the table. --return-start is what makes
# the return phase regulate to the STAND pose rather than back to the brace.
run "recover -> plan_recover" \
    --mode "$MODE" --start forearm_brace_reach --return-start stand \
    --tag recover --n-approach 0 --n-braced 20 --n-return 120 \
    --reach-rot auto --w-reach-rot 1e-2

echo
echo "=== $CELL now carries ==="
ls -1 "$CELL"/plan_*.json
echo
echo "All three are selectable in the panel's task dropdown:"
echo "  studies/croco_twin.py --dir $CELL --tag $TAG --plant mujoco --gui"
