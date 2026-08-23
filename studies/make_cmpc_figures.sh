#!/usr/bin/env bash
# Everything downstream of the CMPC episodes, in one command.
#
# The episodes themselves (cmpc_brace_vs_stand.py cells / run) are NOT here:
# they are ~40 minutes and are the thing you re-run deliberately. This is the
# part you re-run because a figure was wrong, and it is idempotent.
#
# TWO INTERPRETERS, on purpose and not by accident. Scoring loads crocoddyl's
# pinned MuJoCo through the `croco` env; plotting needs matplotlib, which that
# env does not carry. Both read and write only files, so the split is safe --
# but running the scorer under the plotting interpreter would silently re-solve
# every frame against a different MuJoCo version than the run used, and the
# contact forces are the numbers most sensitive to that.
#
#   make_cmpc_figures.sh [--skip-analyze] [--skip-media]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$HERE"

: "${CL_ASSETS_DIR:=$ROOT/../CL_Assets}"
: "${STAGE_ROOT:=$HERE/runs/_stage}"
: "${LEAN_TASK_DIR:=$STAGE_ROOT/mjpc/tasks/humanoid_bench/lean}"
: "${MUJOCO_GL:=egl}"
export CL_ASSETS_DIR STAGE_ROOT LEAN_TASK_DIR MUJOCO_GL

PY_SCORE="${PY_SCORE:-$HOME/miniconda3/envs/croco/bin/python}"   # mujoco+numpy
PY_PLOT="${PY_PLOT:-$HOME/miniconda3/bin/python}"                # +matplotlib

CMPC_RUN="${CMPC_RUN:-$HERE/runs/2026-08-22_cmpc_bvs}"
MJPC_RUN="${MJPC_RUN:-$HERE/runs/2026-08-22_brace_vs_stand}"
OUT="${OUT:-$ROOT/paper/figures/cmpc_brace_reach}"
MEDIA="$OUT/media"

SKIP_ANALYZE=0; SKIP_MEDIA=0
for a in "$@"; do
  case "$a" in
    --skip-analyze) SKIP_ANALYZE=1 ;;
    --skip-media)   SKIP_MEDIA=1 ;;
    *) echo "unknown flag $a" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT" "$MEDIA"

if [ "$SKIP_ANALYZE" = 0 ]; then
  echo "=== scoring CMPC rollouts (append) ==="
  # --append: ~25 s per rollout, nearly all of it in the equilibrium-region LPs.
  "$PY_SCORE" brace_vs_stand.py analyze --run "$CMPC_RUN" --append
fi

echo "=== CMPC standalone figures ==="
"$PY_PLOT" brace_vs_stand.py plot --json "$CMPC_RUN/analysis.json" --out "$OUT"

echo "=== MJPC vs CMPC overlays ==="
"$PY_PLOT" mjpc_vs_cmpc.py --mjpc "$MJPC_RUN/analysis.json" \
    --cmpc "$CMPC_RUN/analysis.json" --out "$OUT" \
    --cmpc-manifest "$CMPC_RUN/manifest_all.json" \
    --mjpc-manifest "$MJPC_RUN"

echo "=== strategy strip ==="
"$PY_PLOT" bvs_strips.py --json "$CMPC_RUN/analysis.json" --run "$CMPC_RUN" \
    --out "$OUT"

echo "=== five-stage strip ==="
if [ -f "$CMPC_RUN/cycle_brace_r0.csv" ]; then
  "$PY_PLOT" cmpc_stage_strip.py --csv "$CMPC_RUN/cycle_brace_r0.csv" \
      --manifest "$CMPC_RUN/manifest_cycle.json" --out "$OUT"
else
  echo "  no cycle_brace_r0.csv -- run \`cmpc_brace_vs_stand.py run --stage cycle\`"
fi

if [ "$SKIP_MEDIA" = 0 ]; then
  echo "=== videos + contact sheets ==="
  # The same representative rollouts the strip draws, plus the full cycle: an
  # mp4 next to the figure is what makes a behavioural claim checkable in ten
  # seconds, which is this repo's standing rule.
  for tag in nominal2_brace_r0 nominal2_stand_r0 nominal_brace_r0 \
             maxreach_brace_r0 maxreach_stand_r0 cycle_brace_r0; do
    [ -f "$CMPC_RUN/$tag.csv" ] || continue
    "$PY_PLOT" simple_video.py "$CMPC_RUN/$tag.csv" --out "$MEDIA" --sheet \
        --speed 4 || echo "  (video failed for $tag)"
  done
fi

echo "=== session docpage ==="
"$PY_PLOT" cmpc_page.py --mjpc "$MJPC_RUN/analysis.json" \
    --cmpc "$CMPC_RUN/analysis.json" --figs "$OUT" \
    --out "$ROOT/docs/lean/2026-08-22_cmpc_vs_mjpc.html"

echo
echo "=== $OUT ==="
ls -1 "$OUT"
