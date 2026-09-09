#!/usr/bin/env bash
# Fly the crocoddyl braced lean at every table height the MJPC study has run.
#
# One process per face (croco_run mutates contact_select's globals and a
# crocoddyl OCP built against one slab must not be reused against another), and
# serial, under a CPU cap -- a parallel sweep has crashed this box before.
set -u
cd "$(dirname "$0")/.."
export CL_ASSETS_DIR="$PWD/../CL_Assets"
export LEAN_TASK_DIR="$PWD/studies/runs/_stage/mjpc/tasks/humanoid_bench/lean"
export MUJOCO_GL=egl
export CMPC_THREADS="${CMPC_THREADS:-6}"
PY=~/miniconda3/envs/croco/bin/python
OUT="${1:-studies/runs/2026-09-08_height_dyn}"
FACES="${FACES:-0.785 0.885 0.985 1.035 1.050 1.060 1.085}"
SEEDS="${SEEDS:-0,1,2}"
SECS="${SECS:-25}"
for f in $FACES; do
  echo "================ face $f ================"
  "$PY" -u studies/height_dynamic.py --face "$f" --tx "${TX:-1.06}" \
       --seeds "$SEEDS" --seconds "$SECS" --out "$OUT" 2>&1 | grep -v '^Warning:'
done
echo "ALL DONE"
