#!/usr/bin/env bash
# V-Bench evaluation for the 20260701_103845 interleave-50 AR draft-head run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_NAME="eval_20260701_103845_allspan_bf16_inckv_interleave50_compare_draft_head_chunk_streaming_train_0_5"

export SDVG_ROOT="${SDVG_ROOT:-$REPO_ROOT/outputs/draft_head/test_runs/$RUN_NAME}"
export EVAL_ROOT="${EVAL_ROOT:-$REPO_ROOT/eval/vbench_runs/$RUN_NAME}"
export NGPUS="${NGPUS:-1}"
export RESUME="${RESUME:-1}"
export LIMIT="${LIMIT:-0}"

bash "$SCRIPT_DIR/run_vbench_sdvg.sh" "$@"
