#!/usr/bin/env bash
# Stage the h1_2 / lean MuJoCo assets WITHOUT a full MJPC C++ build.
#
# WHY.  Every studies script loads its model out of
# `build/mjpc/tasks/humanoid_bench/lean/`, which normally only exists because
# cmake's `copy_model_resources` target built it.  That target also stages
# menagerie, dm_control, the panda, the quadrotor -- none of which this study
# touches -- and it is a hard dependency of libmjpc, so getting the lean model
# on a fresh machine otherwise means compiling all of MJPC.
#
# This reproduces EXACTLY the h1_2 + lean subset of the rules in
# mjpc/tasks/CMakeLists.txt (lines ~270-435), in the same order, so the staged
# model is byte-identical to what a real build produces:
#
#   meshes <- CL_Assets @ the pinned SHA, h1_2_modified <- _gen_h12_base_limits
#   (CL_Assets limits imported onto the vendored planner base), magpie variant
#   <- that file + h1_2_modified_magpie.xml.patch (grippers + brace pads).
#
# The one deliberate omission is meshes/skin (h1_mujoco, Avoid_H12 only) -- the
# lean model does not reference it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/mjpc/tasks"
# STAGE_ROOT relocates the staged tree. Default is this checkout's own build/,
# but that directory is often root-owned (docker creates it as a bind-mount
# target), so the study needs to be able to stage somewhere writable. Point
# LEAN_TASK_DIR at "$STAGE_ROOT/mjpc/tasks/humanoid_bench/lean" to run against
# whatever this produces -- see the echo at the end, which prints the exact
# export line.
STAGE_ROOT="${STAGE_ROOT:-$ROOT/build}"
DST="$STAGE_ROOT/mjpc/tasks"
CL="${CL_ASSETS_DIR:-$ROOT/build/_deps/cl_assets-src}"
PY="${PYTHON:-python3}"

[ -d "$CL/meshes/h1_2" ] || { echo "no CL_Assets at $CL" >&2; exit 1; }

mkdir -p "$DST"
cp -r "$SRC/." "$DST/"

HB="$DST/humanoid_bench"
mkdir -p "$HB/h1_2"
cp -r "$CL/meshes/h1_2/." "$HB/h1_2/"
cp "$CL/mujoco_assets/h1_2_handless.xml" "$HB/h1_2/h1_2.xml"
cp "$SRC/humanoid_bench/h1_2_base/magpie_h12.stl" "$HB/h1_2/magpie_h12.stl"

mkdir -p "$HB/meshes"
ln -sfn ../h1_2 "$HB/meshes/h1_2"

"$PY" "$SRC/humanoid_bench/h1_2_base/_gen_h12_base_limits.py" \
      "$CL/mujoco_assets/h1_2_handless.xml" \
      "$SRC/humanoid_bench/h1_2_base/h1_2_pos.xml" \
      "$HB/h1_2/h1_2_modified.xml"

cp "$HB/h1_2/h1_2_modified.xml" "$HB/h1_2/h1_2_modified_magpie.xml"
patch -p0 -f "$HB/h1_2/h1_2_modified_magpie.xml" \
      < "$SRC/humanoid_bench/h1_2_modified_magpie.xml.patch"
rm -f "$HB/h1_2/h1_2_modified_magpie.xml.rej"

# The mesh symlinks have to exist BEFORE the two magpie steps below: both of
# them load the staged model to check their own work, and the model only
# resolves through lean/meshes/h1_2.
for t in lean beginning; do
  rm -rf "$HB/$t/meshes"
  mkdir -p "$HB/$t/meshes"
  ln -sfn ../../h1_2 "$HB/$t/meshes/h1_2"
  ln -sfn meshes/h1_2 "$HB/$t/h1_2"
done

# Actuator torque budget, magpie-scoped -- mirrors the --forcerange real step in
# mjpc/tasks/CMakeLists.txt. (icra2026 shipped this as a patch hunk; it cannot
# apply after the CL limit import above rewrites the ctrlranges it quotes.)
"$PY" "$SRC/humanoid_bench/h1_2_base/_gen_h12_base_limits.py" \
      "$CL/mujoco_assets/h1_2_handless.xml" \
      "$HB/h1_2/h1_2_modified_magpie.xml" \
      "$HB/h1_2/h1_2_modified_magpie.xml" \
      --forcerange real

# The real magpie gripper, from CL_Assets -- mirrors the _gen_magpie_gripper.py
# step in mjpc/tasks/CMakeLists.txt. Welded, so nq/nv/nu and the inertia are
# unchanged; the script verifies that through the task model and refuses to
# write otherwise.
"$PY" "$SRC/humanoid_bench/h1_2_base/_gen_magpie_gripper.py" \
      "$CL/mujoco_assets/h1_2_magpie.xml" \
      "$HB/h1_2/h1_2_modified_magpie.xml" \
      "$HB/h1_2/h1_2_modified_magpie.xml" \
      --jaw open --collision cad \
      --meshdir "$HB/h1_2" \
      --verify-with "$HB/lean/Lean_H12_Magpie.xml"

echo "staged $HB/lean/Lean_H12_Magpie.xml"
echo "run the study against it with:"
echo "  export LEAN_TASK_DIR=$HB/lean"
