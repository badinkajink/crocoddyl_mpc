#!/usr/bin/env bash
# Build the C++ extensions against the conda `croco` env.
#
#   croco_keepout   box keep-out: the activation (S13) and the fused cost (S15)
#   croco_passive   the plant's passive joint torques as a native actuation model
#   croco_mfd       matrix-free Delassus benchmark -- reaches
#                   DelassusOperatorRigidBodySystemsTpl, which pinocchio 4.1.0
#                   ships as headers only and does NOT expose to Python
#
# No cmake: these are single translation units against headers that are all in
# one prefix, and a find_package dance would only hide which flags matter.  The
# flags that DO matter:
#
#   -I$PREFIX/include/eigen3      crocoddyl's headers include <Eigen/Dense>
#   -lcrocoddyl                   the extern-template instantiations of
#                                 ActivationModelAbstractTpl<double> live there,
#                                 and linking against a DIFFERENT libcrocoddyl
#                                 than the one crocoddyl's pywrap loaded would
#                                 give two typeinfos and a bases<> that never
#                                 matches at import time
#   -lpinocchio_default -leigenpy the same argument for pinocchio: the operator
#                                 is header-only but its Model/Data/Constraint
#                                 typeinfos have to be the ones the pinocchio
#                                 pywrap already registered, or bp::extract on a
#                                 Python pin.Model finds no converter
#   -lboost_python311             same registry as eigenpy/crocoddyl's bindings
#
# usage: croco_ext/build.sh [--clean] [keepout|passive|mfd]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The prefix has to be the `croco` env and not merely "whatever is activated".
# A login shell here has conda BASE activated, so CONDA_PREFIX is set and points
# somewhere with no crocoddyl in it -- a plain `${CONDA_PREFIX:-...}` default
# never fires and the build dies on `pinocchio/fwd.hpp: No such file`, which
# reads like a missing dependency rather than a wrong prefix.  So: take
# CROCO_PREFIX if given, else CONDA_PREFIX only if it actually has crocoddyl in
# it, else the known env, and say so if none of those work.
pick_prefix() {
  for p in "${CROCO_PREFIX:-}" "${CONDA_PREFIX:-}" "$HOME/miniconda3/envs/croco" \
           "$HOME/miniforge3/envs/croco"; do
    [ -n "$p" ] && [ -d "$p/include/crocoddyl" ] && { echo "$p"; return; }
  done
  echo "no conda prefix with include/crocoddyl found; set CROCO_PREFIX" >&2
  exit 1
}
PREFIX="$(pick_prefix)"
echo "building against $PREFIX"
PYVER="$("$PREFIX/bin/python" -c 'import sys; print("%d%d" % sys.version_info[:2])')"
PYDOT="$("$PREFIX/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
EXT="$("$PREFIX/bin/python" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
NPY="$("$PREFIX/bin/python" -c 'import numpy; print(numpy.get_include())')"

if [ "${1:-}" = "--clean" ]; then
  rm -f "$HERE/croco_keepout$EXT" "$HERE/croco_passive$EXT" "$HERE/croco_mfd$EXT"
  echo "removed built extensions"; exit 0
fi
WHICH="${1:-all}"

# `pinocchio_compat` LAST on the include path: crocoddyl 3.2.1's pch.hpp names six
# Pinocchio headers that Pinocchio 4.1.0 moved behind umbrella headers, and any
# TU including a crocoddyl multibody header dies on them.  See its README.
COMMON=(-O3 -DNDEBUG -fPIC -shared -std=c++17
        -I"$PREFIX/include" -I"$PREFIX/include/eigen3"
        -I"$PREFIX/include/python$PYDOT" -I"$NPY"
        -I"$HERE/pinocchio_compat"
        -L"$PREFIX/lib" "-lboost_python$PYVER" -Wl,-rpath,"$PREFIX/lib")

if [ "$WHICH" = "all" ] || [ "$WHICH" = "keepout" ]; then
  set -x
  g++ "${COMMON[@]}" "$HERE/keepout.cpp" -o "$HERE/croco_keepout$EXT" -lcrocoddyl
  set +x
  echo "built $HERE/croco_keepout$EXT"
fi

if [ "$WHICH" = "all" ] || [ "$WHICH" = "passive" ]; then
  # -leigenpy: unlike croco_keepout this one takes Eigen vectors across the
  # boundary (the 27 damping and friction coefficients), so it needs eigenpy's
  # numpy converters registered in the same process-wide registry.
  set -x
  g++ "${COMMON[@]}" "$HERE/actuation.cpp" -o "$HERE/croco_passive$EXT" \
      -lcrocoddyl -leigenpy
  set +x
  echo "built $HERE/croco_passive$EXT"
fi

if [ "$WHICH" = "all" ] || [ "$WHICH" = "mfd" ]; then
  # The -D flags are NOT optional and NOT tuning: they are
  # pinocchio::pinocchio_headers' INTERFACE_COMPILE_DEFINITIONS, verbatim from
  # $PREFIX/lib/cmake/pinocchio/pinocchioTargets.cmake.  The constraint
  # collection is a boost::variant over 25 types and the constraint visitors
  # call through 8-argument boost::fusion::invoke; at boost's defaults (20 and
  # 6) the whole header stack fails to instantiate.
  set -x
  g++ "${COMMON[@]}" \
      -DBOOST_MPL_LIMIT_LIST_SIZE=30 -DBOOST_MPL_LIMIT_VECTOR_SIZE=30 \
      -DBOOST_MPL_CFG_NO_PREPROCESSED_HEADERS \
      -DBOOST_FUSION_INVOKE_MAX_ARITY=12 \
      -DPINOCCHIO_ENABLE_TEMPLATE_INSTANTIATION \
      -Wno-deprecated-declarations \
      "$HERE/mfdelassus.cpp" -o "$HERE/croco_mfd$EXT" \
      -lpinocchio_default -leigenpy
  set +x
  echo "built $HERE/croco_mfd$EXT"
fi
