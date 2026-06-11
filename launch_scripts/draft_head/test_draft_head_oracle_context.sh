#!/usr/bin/env bash
# Run a draft-head quality diagnostic with target-regenerated oracle context.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="${SF_VENV:-$PROJECT_ROOT/sf_venv}"
PYTHON="$VENV_DIR/bin/python"

MODEL_ROOT="${MODEL_ROOT:-/mnt/lanxiangh/models}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/self_forcing_dmd.yaml}"
TARGET_CHECKPOINT_PATH="${TARGET_CHECKPOINT_PATH:-$MODEL_ROOT/realtime-video/checkpoints/krea-realtime-video-14b.safetensors}"
DRAFT_HEAD_CHECKPOINT_PATH="${DRAFT_HEAD_CHECKPOINT_PATH:-$PROJECT_ROOT/outputs/draft_head/checkpoints/20260601_015502_3gpu_bs1_moviegen_draft_head_attention_clean_latent_flow_h5120_l3_heads40_ctx4680_pool1x1x1_ffn13824_steps1000_937_833_625_0_blocks9_targetinit8_16_24/final.pt}"

OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/draft_head/test_runs/draft_head_oracle_context_$(date +%Y%m%d_%H%M%S)}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
NUM_BLOCKS="${NUM_BLOCKS:-9}"
MAX_PROMPTS="${MAX_PROMPTS:-1}"
START_INDEX="${START_INDEX:-0}"
PROMPT_FILE="${PROMPT_FILE:-}"
PROMPT="${PROMPT:-A hyperrealistic close-up of ocean waves shimmering at sunset.}"
SEED="${SEED:-42}"
AGREEMENT_METRIC="${AGREEMENT_METRIC:-rmse}"
ROUTER_MODE="${ROUTER_MODE:-heuristic}"

LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/launch_scripts/logs}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
LOG_FILE="$LOG_DIR/$(basename "${BASH_SOURCE[0]}" .sh)_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }

cd "$PROJECT_ROOT"

for path in "$PYTHON" "$TARGET_CHECKPOINT_PATH" "$DRAFT_HEAD_CHECKPOINT_PATH" "$MODEL_ROOT/wan_models/Wan2.1-T2V-14B"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: required path missing: $path" >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

PROMPT_ARGS=(--prompt "$PROMPT")
if [[ -n "$PROMPT_FILE" ]]; then
  PROMPT_ARGS=(--prompt_file "$PROMPT_FILE" --start_index "$START_INDEX" --max_prompts "$MAX_PROMPTS")
fi

log "Draft-head oracle-context quality diagnostic"
log "Checkpoint: $DRAFT_HEAD_CHECKPOINT_PATH"
log "Output:     $OUTPUT_DIR"
log "Blocks:     $NUM_BLOCKS"
log "Metric:     $AGREEMENT_METRIC"

"$PYTHON" sdvg_inference.py "$ROUTER_MODE" \
  --config_path "$CONFIG_PATH" \
  --model_root "$MODEL_ROOT" \
  --target_checkpoint_path "$TARGET_CHECKPOINT_PATH" \
  --draft_head_checkpoint_path "$DRAFT_HEAD_CHECKPOINT_PATH" \
  --mode compare \
  --compare_mode draft_head \
  --draft_head_log_target_delta \
  --draft_head_oracle_context \
  --agreement_metric "$AGREEMENT_METRIC" \
  --num_blocks "$NUM_BLOCKS" \
  --seed "$SEED" \
  --output_dir "$OUTPUT_DIR" \
  "${PROMPT_ARGS[@]}" \
  2>&1 | tee -a "$LOG_FILE"

log "Done. Profile: $OUTPUT_DIR/profile.json"
