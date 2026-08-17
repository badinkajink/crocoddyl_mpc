#!/usr/bin/env bash
# Build the libcrocoddyl that HAS OpenMP, and leave the conda env untouched.
#
# WHY.  conda-forge ships crocoddyl without CROCODDYL_WITH_MULTITHREADING, so
# `ShootingProblem.nthreads = n` prints "multithreading support is not enabled"
# and pins the count to 1.  There is no conda-forge variant with it.  Measured
# on this machine at the deployed 35-node / 1-iteration configuration:
#
#     nthreads  1    14.4 ms mean   21.3 ms p95     <- stock conda-forge
#     nthreads 20    10.4 ms        16.7 ms         <- this build
#
# against a 20 ms control period.  That is the difference between a p95 ON the
# deadline and one under it.  The mean gain is only 1.4x and it is Amdahl-capped
# (S16: only the derivative sweep is parallel; the backward pass and the
# line-search rollouts are sequential by construction), so do not expect more by
# adding cores -- it is flat past about four.
#
# THREE THINGS THAT MAKE THIS A SAFE SWAP, all checked in S16 rather than hoped:
#
#   1. The define does not change the ABI.  `nthreads_` is an unconditional
#      member of ShootingProblemTpl; the define only gates #pragma omp lines.
#      So the stock Python bindings load this library unchanged and we build
#      -DBUILD_PYTHON_INTERFACE=OFF, which is most of the build time.
#   2. The bindings use RPATH ($ORIGIN/../../..), not RUNPATH, and RPATH beats
#      LD_LIBRARY_PATH -- so the obvious selection mechanism silently does
#      NOTHING.  LD_PRELOAD does work, because a preloaded object with the
#      matching SONAME satisfies the NEEDED entry.  That is why run_session.sh
#      selects this per-process rather than by exporting a path.
#   3. Building crocoddyl 3.2.1 against Pinocchio 4.1.0 hits the same pch.hpp
#      problem as any downstream translation unit, so croco_ext/pinocchio_compat
#      goes on the include path here too.
#
# ONE UPSTREAM PATCH IS NEEDED and it has nothing to do with threading:
# core/actions/{lqr,diff-lqr}.hxx initialise a member with
# `cond ? VectorXs::Zero(n) : VectorXs::Ones(n)`, whose branches are different
# Eigen expression types.  Newer Eigen rejects it and the build stops on a toy
# LQR action model this study never instantiates.  Both are wrapped in
# VectorXs(...) to force a common type.
#
# usage:  croco_ext/build_crocoddyl_omp.sh [--force]
# env:    CROCO_OMP_PREFIX  install prefix (default ~/opt/crocoddyl-omp)
#         CROCO_OMP_SRC     source tree    (default ~/opt/src/crocoddyl)
#         CONDA_PREFIX      the env to build AGAINST -- must be the one that
#                           will load the result, or the pinocchio ABI differs
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${CROCO_OMP_PREFIX:-$HOME/opt/crocoddyl-omp}"
SRC="${CROCO_OMP_SRC:-$HOME/opt/src/crocoddyl}"
VER=3.2.1
LIB="$PREFIX/lib/libcrocoddyl.so.$VER"
JOBS="${JOBS:-$(nproc)}"

if [ "${1:-}" != "--force" ] && [ -f "$LIB" ]; then
  echo "already built: $LIB"
  exit 0
fi

: "${CONDA_PREFIX:?activate the croco env first -- this links against ITS pinocchio}"
# The env that builds it must be the env that loads it. A mismatch here is not a
# link error, it is a segfault at the first contact solve.
"$CONDA_PREFIX/bin/python" - <<'PY'
import pinocchio, crocoddyl, sys
if crocoddyl.__version__ != "3.2.1":
    sys.exit("this env has crocoddyl %s; the patch and the SONAME below are for "
             "3.2.1" % crocoddyl.__version__)
print("building against pinocchio %s in %s" % (pinocchio.__version__, sys.prefix))
PY

if [ ! -d "$SRC/.git" ]; then
  echo "--- cloning crocoddyl v$VER -> $SRC"
  mkdir -p "$(dirname "$SRC")"
  git clone --recursive --depth 1 --branch "v$VER" \
      https://github.com/loco-3d/crocoddyl "$SRC"
fi

echo "--- patching the two LQR ternaries (unrelated to OpenMP; see header)"
for f in core/actions/lqr.hxx core/actions/diff-lqr.hxx; do
  p="$SRC/include/crocoddyl/$f"
  # Idempotent: only rewrites the raw form, so re-running is a no-op.
  sed -i -E 's/drift_free \? VectorXs::Zero\((n[qx])\) : VectorXs::Ones\(\1\)/drift_free ? VectorXs(VectorXs::Zero(\1)) : VectorXs(VectorXs::Ones(\1))/' "$p"
done

mkdir -p "$SRC/build"
cd "$SRC/build"
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_INSTALL_PREFIX="$PREFIX" \
         -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
         -DCMAKE_CXX_FLAGS="-I$HERE/pinocchio_compat -Wno-deprecated-declarations" \
         -DBUILD_WITH_MULTITHREADS=ON -DBUILD_WITH_NTHREADS=4 \
         -DBUILD_PYTHON_INTERFACE=OFF -DBUILD_TESTING=OFF \
         -DBUILD_BENCHMARK=OFF -DBUILD_EXAMPLES=OFF -DBUILD_WITH_IPOPT=OFF
make -j"$JOBS"
make install

# ASSERT THE BUILD TOOK. A library without the define still loads, still runs and
# is still slow -- exactly the silent-fallback failure croco_ext/*.so had.
echo "--- verifying multithreading is live"
LD_PRELOAD="$LIB" "$CONDA_PREFIX/bin/python" - <<'PY'
import numpy as np, crocoddyl, sys
p = crocoddyl.ShootingProblem(np.zeros(3), [crocoddyl.ActionModelUnicycle()] * 2,
                              crocoddyl.ActionModelUnicycle())
p.nthreads = 4
mapped = next((l.split()[-1] for l in open("/proc/self/maps")
               if "libcrocoddyl.so" in l), "?")
if int(p.nthreads) != 4:
    sys.exit("BUILT BUT NOT MULTITHREADED: nthreads pinned to %d (mapped %s)"
             % (p.nthreads, mapped))
print("OK: nthreads=%d, mapped %s" % (p.nthreads, mapped))
PY
echo "built $LIB"
