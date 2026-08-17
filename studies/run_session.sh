#!/usr/bin/env bash
# ONE command to reproduce a session: stage the model, check the interpreter,
# run the grid, render the videos.
#
# WHY THIS EXISTS.  Reproducing S18 took three separate discoveries that nothing
# in the repo recorded, and each one presented as a different bug:
#
#   1. build/ is root-owned under docker, so stage_assets.sh could not write ->
#      STAGE_ROOT.
#   2. croco_bridge defaulted its URDF to the cmake FetchContent tree, so a
#      machine that staged with CL_ASSETS_DIR got a staged MJCF and no URDF, and
#      failed inside urdfdom as "does not contain a valid URDF model".
#   3. `base` has a crocoddyl wheel set whose CONTACT dynamics SEGFAULT. Every
#      cell died with rc=-11 and no traceback, on the old model as well as the
#      new one, which reads exactly like an asset regression and is not.
#
# All three are now either handled or checked here, so the next re-run starts
# from a green light instead of from a bisect.
#
# usage:
#   studies/run_session.sh check                 # env + assets only
#   studies/run_session.sh stage                 # (re)stage the model
#   studies/run_session.sh deps                  # build the native extensions
#   studies/run_session.sh grid   [stages...]    # default: certify plan stress collect
#   studies/run_session.sh videos                # grid videos + gripper orientations
#   studies/run_session.sh all
#
# env:
#   CROCO_PY     interpreter (default: the `croco` conda env)
#   RUN          run directory (default: runs/<today>_session18)
#   SEEDS        stress seeds (default: 1 2 3 4 5)
#   PROFILES     stress profiles (default: nominal winch1)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

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
CROCO_PY="$(_croco_py)"
RUN="${RUN:-$HERE/runs/$(date +%Y-%m-%d)_session18}"
SEEDS="${SEEDS:-1 2 3 4 5}"
PROFILES="${PROFILES:-nominal winch1}"

# CL_Assets: prefer an explicit export, else the sibling checkout in the
# superproject, else the cmake FetchContent tree.
if [ -z "${CL_ASSETS_DIR:-}" ]; then
  for c in "$ROOT/../CL_Assets" "$ROOT/build/_deps/cl_assets-src"; do
    [ -d "$c/mujoco_assets" ] && { CL_ASSETS_DIR="$(cd "$c" && pwd)"; break; }
  done
fi
: "${CL_ASSETS_DIR:?set CL_ASSETS_DIR to the CL_Assets checkout}"
export CL_ASSETS_DIR

# Stage somewhere writable: build/ is commonly root-owned (docker bind mount).
export STAGE_ROOT="${STAGE_ROOT:-$HERE/runs/_stage}"
export LEAN_TASK_DIR="${LEAN_TASK_DIR:-$STAGE_ROOT/mjpc/tasks/humanoid_bench/lean}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

say() { printf '\n=== %s ===\n' "$*"; }

# --- the OpenMP libcrocoddyl -------------------------------------------------
# conda-forge's crocoddyl has no OpenMP, so `nthreads` is silently pinned to 1.
# The rebuilt one is selected PER PROCESS with LD_PRELOAD, because the bindings
# carry RPATH $ORIGIN/../../.. and RPATH beats LD_LIBRARY_PATH -- exporting a
# path does nothing at all. See croco_ext/build_crocoddyl_omp.sh.
#
# It is NOT exported into this shell's environment on purpose: LD_PRELOAD is
# inherited by every child, and preloading libcrocoddyl into `git` makes git
# fail to start -- which is how croco_speed.py's recorded commit silently became
# "unknown" for a whole session. Only `pyrun` sets it.
CROCO_OMP_LIB="${CROCO_OMP_LIB:-$HOME/opt/crocoddyl-omp/lib/libcrocoddyl.so.3.2.1}"

pyrun() {
  if [ -f "$CROCO_OMP_LIB" ]; then
    LD_PRELOAD="$CROCO_OMP_LIB" "$CROCO_PY" "$@"
  else
    "$CROCO_PY" "$@"
  fi
}

do_stage() {
  say "staging model  -> $STAGE_ROOT"
  bash "$HERE/stage_assets.sh"
}

do_deps() {
  # The native extensions are a 6x knob that fails SILENTLY: croco_geom falls
  # back to the Python keep-out activation with a one-line note, so an unbuilt
  # checkout runs correctly at 85 ms against a 20 ms control period. Built here
  # rather than documented, because "remember to build the extensions" is the
  # kind of instruction that gets skipped exactly once.
  if ls "$HERE"/croco_ext/croco_keepout*.so >/dev/null 2>&1 \
     && ls "$HERE"/croco_ext/croco_passive*.so >/dev/null 2>&1; then
    say "extensions already built"
  else
    say "building native extensions (keep-out is a 6x speed-up)"
    # ONE TARGET PER CALL. build.sh takes a single [keepout|passive|mfd], so
    # "keepout passive" built keepout and silently dropped passive -- and
    # croco_env then reported "passive native no", which reads as a missing
    # optional rather than as this step having half-failed. Exactly the trap
    # ecc41dc is named after.
    bash "$HERE/croco_ext/build.sh" keepout
    bash "$HERE/croco_ext/build.sh" passive
  fi
  # The OpenMP crocoddyl is the SECOND silent knob: without it every
  # `nthreads` request is pinned to 1 with a warning, and the p95 sits ON the
  # 20 ms control period instead of under it (16.7 ms vs 21.3 ms measured).
  # Worth 1.4x, not 6x -- Amdahl, see the S16 docpage -- but it is the
  # difference between missing a deadline sometimes and never.
  if [ -f "$CROCO_OMP_LIB" ]; then
    say "OpenMP crocoddyl present: $CROCO_OMP_LIB"
  else
    say "building the OpenMP crocoddyl (p95 21.3 -> 16.7 ms)"
    bash "$HERE/croco_ext/build_crocoddyl_omp.sh" || \
      echo "(OpenMP build failed; runs will use the stock single-threaded library)"
  fi
}

do_check() {
  say "environment"
  pyrun "$HERE/croco_env.py"
}

do_grid() {
  local stages=("$@")
  [ ${#stages[@]} -eq 0 ] && stages=(certify plan stress collect)
  mkdir -p "$RUN/grid"
  for s in "${stages[@]}"; do
    say "grid $s  -> $RUN/grid"
    case "$s" in
      stress)  pyrun "$HERE/croco_grid.py" stress --dir "$RUN/grid" \
                  --seeds $SEEDS --profiles $PROFILES ;;
      collect) pyrun "$HERE/croco_grid.py" collect --dir "$RUN/grid" \
                  --out "$RUN/grid.json" ;;
      *)       pyrun "$HERE/croco_grid.py" "$s" --dir "$RUN/grid" ;;
    esac
  done
}

do_videos() {
  say "grid videos -> $RUN/media"
  pyrun "$HERE/croco_grid.py" videos --dir "$RUN/grid" \
      --out "$RUN/media" || echo "(grid videos stage reported a failure)"
  say "gripper orientation videos -> $RUN/media"
  pyrun "$HERE/gripper_views.py" --out "$RUN/media"
}

cmd="${1:-all}"; shift || true
case "$cmd" in
  deps)   do_deps ;;
  check)  do_check ;;
  stage)  do_stage ;;
  grid)   do_deps; do_check; do_grid "$@" ;;
  videos) do_check; do_videos ;;
  all)    do_stage; do_deps; do_check; do_grid; do_videos ;;
  *) echo "usage: $0 {check|stage|deps|grid|videos|all}" >&2; exit 2 ;;
esac

say "done -- run dir $RUN"
