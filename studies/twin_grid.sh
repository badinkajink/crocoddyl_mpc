#!/usr/bin/env bash
# Run every certified grid cell through the DDS twin, one cell per twin.
#
# WHAT THIS CLOSES.  S19's first twin run was ONE cell, and the docpage's whole
# answer to "is the twin equivalent to the grid" was: no, 1 run against 260.
# This closes the CELLS axis -- 26 plans, each solved in process and again over
# the wire, so the per-cell difference is attributable to deployment and to
# nothing else.  It does NOT close the disturbance-profile or seed axes:
# croco_twin has no push/noise plumbing, and adding it is a separate change.
#
# ONE TWIN PER CELL, DELIBERATELY.  The twin must start at the pose its plan
# starts from (--qpos0, emitted per cell), and it must not step physics before
# the controller exists -- a joint-space PD hold does not balance a leaning
# posture, so a twin left running through the next cell's ~25 s OCP build finds
# the robot on the floor.  Restarting is cheaper than making it resettable.
#
# PIDs are captured and killed by number.  `pkill -f lean_twin` matches this
# script's own command line and kills the shell running it.
#
# usage:  studies/twin_grid.sh [GRID_DIR] [OUT_DIR]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

GRID="${1:-$HERE/runs/2026-08-16_session18/grid}"
OUT="${2:-$HERE/runs/$(date +%Y-%m-%d)_session19/twingrid}"
# The interpreter. `croco` is a conda env with a crocoddyl whose CONTACT
# dynamics do not segfault -- `base`'s does, which cost a whole session to
# find (see the header). Resolved rather than hardcoded to one developer's
# absolute path, which is what it was: CROCO_PY wins, then a `croco` env under
# whatever conda is installed, then the active env, then python3. `croco_env.py`
# is what actually checks the interpreter is usable; this only has to find a
# plausible one.
_croco_py() {
  if [ -n "${CROCO_PY:-}" ]; then echo "$CROCO_PY"; return; fi
  for base in "${CONDA_EXE%/bin/conda}" "$HOME/miniconda3" "$HOME/anaconda3" \
              "$HOME/miniforge3" /opt/conda; do
    [ -x "$base/envs/croco/bin/python" ] && { echo "$base/envs/croco/bin/python"; return; }
  done
  [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ] \
    && { echo "$CONDA_PREFIX/bin/python"; return; }
  command -v python3
}
PY="$(_croco_py)"
OMP="${CROCO_OMP_LIB:-$HOME/opt/crocoddyl-omp/lib/libcrocoddyl.so.3.2.1}"
DOMAIN="${ROS_DOMAIN_ID:-1}"
IFACE="${GOLEM_IFACE:-lo}"
THREADS="${THREADS:-8}"

[ "$DOMAIN" = "0" ] && { echo "domain 0 is the REAL robot's bus. refusing."; exit 1; }
: "${CL_ASSETS_DIR:?set CL_ASSETS_DIR}"
export STAGE_ROOT="${STAGE_ROOT:-$HERE/runs/_stage}"
export LEAN_TASK_DIR="${LEAN_TASK_DIR:-$STAGE_ROOT/mjpc/tasks/humanoid_bench/lean}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p "$OUT"

# -u: the twin is killed by signal at the end of every cell, and a block-
# buffered stdout loses its whole log (including the outcome JSON) when it is.
pyrun() { env LD_PRELOAD="$OMP" "$PY" -u "$@"; }

# THE TWIN IS LAUNCHED WITHOUT THE FUNCTION, ON PURPOSE. Backgrounding a shell
# FUNCTION forks a subshell, so `$!` is the subshell's pid and not python's --
# the TERM went to the subshell, python was orphaned, and the next cell's twin
# came up alongside a twin that was still publishing lowstate at 500 Hz. Two
# twins on one topic is not a crash; it is a quietly worse run (overruns went
# 20 -> 54) and an outcome line that never prints. Launch it directly so `$!`
# is the process that has to die.

for cell in "$GRID"/*/; do
  name="$(basename "$cell")"
  plan="$(ls "$cell"/plan_*.json 2>/dev/null | head -1)" || true
  [ -n "${plan:-}" ] || { echo "$name  no plan, skipped"; continue; }
  tag="$(basename "$plan" .json)"; tag="${tag#plan_}"
  [ -f "$OUT/$name.json" ] && { echo "$name  done, skipped"; continue; }

  # The twin must BEGIN where the plan begins; no keyframe is that pose.
  pyrun studies/croco_twin.py --dir "$cell" --tag "$tag" \
        --emit-qpos0 "$OUT/$name.qpos0" >/dev/null 2>&1 \
    || { echo "$name  qpos0 FAILED"; continue; }

  env LD_PRELOAD="$OMP" "$PY" -u -m croco.twin.lean_twin \
        --model "$LEAN_TASK_DIR/Lean_H12_Magpie.xml" --key stand \
        --qpos0 "$OUT/$name.qpos0" --publish-truth \
        --domain "$DOMAIN" --iface "$IFACE" --duration 180 \
        > "$OUT/$name.twin.log" 2>&1 &
  TWIN=$!
  trap 'kill -TERM '"$TWIN"' 2>/dev/null || true' EXIT
  sleep 3

  pyrun studies/croco_twin.py --dir "$cell" --tag "$tag" --base truth \
        --threads "$THREADS" --domain "$DOMAIN" --iface "$IFACE" \
        --out "$OUT/$name.json" > "$OUT/$name.log" 2>&1 || true

  kill -TERM "$TWIN" 2>/dev/null || true
  wait "$TWIN" 2>/dev/null || true
  trap - EXIT
  # Never start the next cell while anything is still on the bus.
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    kill -0 "$TWIN" 2>/dev/null || break
    sleep 0.5
  done
  kill -KILL "$TWIN" 2>/dev/null || true

  z="$(sed -n 's/.*"pelvis_z_min_commanded": \([0-9.]*\).*/\1/p' "$OUT/$name.twin.log" | tail -1)"
  echo "$name  $tag  pelvis_min_cmd ${z:-?}"
done

pyrun studies/twin_grid.py "$OUT"
