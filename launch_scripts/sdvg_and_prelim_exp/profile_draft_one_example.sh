#!/usr/bin/env bash
# Profile pure drafter generation and write timing breakdown JSON.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${SF_VENV:-$PROJECT_ROOT/sf_venv}"
PYTHON="$VENV_DIR/bin/python"

MODEL_ROOT="${MODEL_ROOT:-/mnt/lanxiangh/models}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/self_forcing_dmd.yaml}"
PROMPT_FILE="${PROMPT_FILE:-$PROJECT_ROOT/prompts/MovieGenVideoBench.txt}"
PROMPT_INDEX="${PROMPT_INDEX:-2}"
DRAFT_CHECKPOINT_PATH="${DRAFT_CHECKPOINT_PATH:-$MODEL_ROOT/Self-Forcing/checkpoints/self_forcing_dmd.pt}"
TARGET_CHECKPOINT_PATH="${TARGET_CHECKPOINT_PATH:-$MODEL_ROOT/realtime-video/checkpoints/krea-realtime-video-14b.safetensors}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
NUM_BLOCKS="${NUM_BLOCKS:-9}"
SEED="${SEED:-42}"
RUN_TAG="${RUN_TAG:-draft_only_blocks${NUM_BLOCKS}_prompt${PROMPT_INDEX}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/sdvg/profile_${RUN_TAG}}"

LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/launch_scripts/logs}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
LOG_FILE="$LOG_DIR/$(basename "${BASH_SOURCE[0]}" .sh)_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }

cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

log "Profiling draft_only"
log "Prompt:  $PROMPT_FILE index=$PROMPT_INDEX"
log "Blocks:  $NUM_BLOCKS"
log "Drafter: $DRAFT_CHECKPOINT_PATH"
log "Output:  $OUTPUT_DIR"

"$PYTHON" sdvg_inference.py \
  heuristic \
  --config_path "$CONFIG_PATH" \
  --model_root "$MODEL_ROOT" \
  --draft_checkpoint_path "$DRAFT_CHECKPOINT_PATH" \
  --target_checkpoint_path "$TARGET_CHECKPOINT_PATH" \
  --prompt_file "$PROMPT_FILE" \
  --start_index "$PROMPT_INDEX" \
  --max_prompts 1 \
  --output_dir "$OUTPUT_DIR" \
  --mode draft_only \
  --num_blocks "$NUM_BLOCKS" \
  --seed "$SEED" \
  2>&1 | tee -a "$LOG_FILE"

log "Profile JSON: $OUTPUT_DIR/profile.json"
log "Videos: $OUTPUT_DIR/prompt_$(printf '%04d' "$PROMPT_INDEX")_draft_only.mp4"
