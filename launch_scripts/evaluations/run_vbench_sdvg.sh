#!/usr/bin/env bash
# Launch V-Bench evaluation for generated Self-Forcing/SDVG videos.
#
# Common usage:
#   SDVG_ROOT=/path/to/videos RESUME=1 NGPUS=1 bash launch_scripts/evaluations/run_vbench_sdvg.sh
#
# The underlying evaluator expects each evaluated directory to contain .mp4 files
# and, when available, uses profile.json to recover prompts for V-Bench custom_input.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SDVG_ROOT="${SDVG_ROOT:-$REPO_ROOT/outputs/sdvg}"
EVAL_ROOT="${EVAL_ROOT:-$REPO_ROOT/eval}"
NGPUS="${NGPUS:-1}"
LIMIT="${LIMIT:-0}"
RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
VBENCH_DIMENSIONS="${VBENCH_DIMENSIONS:-subject_consistency background_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality}"

cmd=(
  bash
  "$REPO_ROOT/eval/run_vbench_sdvg.sh"
  --sdvg_root "$SDVG_ROOT"
  --eval_root "$EVAL_ROOT"
  --ngpus "$NGPUS"
  --limit "$LIMIT"
  --dimensions $VBENCH_DIMENSIONS
)

if [[ -n "${RESULTS_JSONL:-}" ]]; then
  cmd+=(--results_jsonl "$RESULTS_JSONL")
fi

if [[ -n "${VBENCH_VENV:-}" ]]; then
  export VBENCH_VENV
fi

if [[ -n "${VBENCH_EXTRA_ARGS:-}" ]]; then
  # Space-separated passthrough args for vbench/launch/evaluate.py.
  read -r -a extra_args <<< "$VBENCH_EXTRA_ARGS"
  for extra_arg in "${extra_args[@]}"; do
    cmd+=(--vbench_extra_arg "$extra_arg")
  done
fi

if [[ "$RESUME" == "1" || "$RESUME" == "true" ]]; then
  cmd+=(--resume)
fi

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  cmd+=(--dry_run)
fi

echo "Launching V-Bench evaluation:"
printf ' %q' "${cmd[@]}"
echo

exec "${cmd[@]}" "$@"
