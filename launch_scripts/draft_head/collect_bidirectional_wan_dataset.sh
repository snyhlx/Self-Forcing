#!/usr/bin/env bash
# Collect Option-B full-video original-Wan latents for bidirectional draft-head training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="${SF_VENV:-$PROJECT_ROOT/sf_venv}"
PYTHON="$VENV_DIR/bin/python"

MODEL_ROOT="${MODEL_ROOT:-/mnt/lanxiangh/models}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/self_forcing_dmd.yaml}"
TARGET_MODEL_NAME="${TARGET_MODEL_NAME:-Wan2.1-T2V-14B}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/lanxiangh/data/ff_exec/bidirectional_wan_draft_head_dataset/tau_delta_0p0}"

CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
NUM_BLOCKS="${NUM_BLOCKS:-9}"
PROMPT_FILE="${PROMPT_FILE:-}"
PROMPT="${PROMPT:-A hyperrealistic close-up of ocean waves shimmering at sunset.}"
START_INDEX="${START_INDEX:-0}"
MAX_PROMPTS="${MAX_PROMPTS:-1}"
SEED="${SEED:-42}"
SHARD_SIZE="${SHARD_SIZE:-16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SAMPLING_STEPS="${SAMPLING_STEPS:-50}"
SAMPLE_SOLVER="${SAMPLE_SOLVER:-unipc}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"

LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/launch_scripts/logs}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
LOG_FILE="$LOG_DIR/$(basename "${BASH_SOURCE[0]}" .sh)_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }

cd "$PROJECT_ROOT"

for path in "$PYTHON" "$MODEL_ROOT/wan_models/$TARGET_MODEL_NAME"; do
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

log "Collecting Option-B bidirectional Wan dataset"
log "Output:  $OUTPUT_DIR"
log "Teacher: $TARGET_MODEL_NAME"
log "Blocks:  $NUM_BLOCKS"
log "Prompts: ${PROMPT_FILE:-single prompt} start=$START_INDEX max=$MAX_PROMPTS"

"$PYTHON" collect_bidirectional_wan_dataset.py \
  --output_dir "$OUTPUT_DIR" \
  --model_root "$MODEL_ROOT" \
  --config_path "$CONFIG_PATH" \
  --target_model_name "$TARGET_MODEL_NAME" \
  --num_blocks "$NUM_BLOCKS" \
  --seed "$SEED" \
  --shard_size "$SHARD_SIZE" \
  --batch_size "$BATCH_SIZE" \
  --sampling_steps "$SAMPLING_STEPS" \
  --sample_solver "$SAMPLE_SOLVER" \
  --guidance_scale "$GUIDANCE_SCALE" \
  "${PROMPT_ARGS[@]}" \
  2>&1 | tee -a "$LOG_FILE"

log "Done. Manifest: $OUTPUT_DIR/manifest.json"
