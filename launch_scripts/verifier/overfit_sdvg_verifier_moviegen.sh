#!/usr/bin/env bash
# Collect a tiny MovieGenVideoBench latent block dataset and overfit a target-vs-draft verifier.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${SF_VENV:-$PROJECT_ROOT/sf_venv}"
PYTHON="$VENV_DIR/bin/python"

MODEL_ROOT="${MODEL_ROOT:-/mnt/lanxiangh/models}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/self_forcing_dmd.yaml}"
PROMPT_FILE="${PROMPT_FILE:-$PROJECT_ROOT/prompts/MovieGenVideoBench.txt}"
DRAFT_CHECKPOINT_PATH="${DRAFT_CHECKPOINT_PATH:-$MODEL_ROOT/Self-Forcing/checkpoints/self_forcing_dmd.pt}"
TARGET_CHECKPOINT_PATH="${TARGET_CHECKPOINT_PATH:-$MODEL_ROOT/realtime-video/checkpoints/krea-realtime-video-14b.safetensors}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/sdvg/verifier/moviegen_$(date +%Y%m%d_%H%M%S)}"
DATA_PATH="${DATA_PATH:-$OUTPUT_DIR/pairs.pt}"
VERIFIER_PATH="${VERIFIER_PATH:-$OUTPUT_DIR/verifier.pt}"

CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
START_INDEX="${START_INDEX:-0}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
NUM_BLOCKS="${NUM_BLOCKS:-3}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-32}"

LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/launch_scripts/logs}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
LOG_FILE="$LOG_DIR/$(basename "${BASH_SOURCE[0]}" .sh)_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }

cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

log "Collecting verifier pairs from MovieGen prompts=$PROMPT_FILE start=$START_INDEX max=$MAX_PROMPTS blocks=$NUM_BLOCKS"
log "Data:     $DATA_PATH"
log "Verifier: $VERIFIER_PATH"

"$PYTHON" sdvg_verifier_experiment.py collect \
  --config_path "$CONFIG_PATH" \
  --model_root "$MODEL_ROOT" \
  --draft_checkpoint_path "$DRAFT_CHECKPOINT_PATH" \
  --target_checkpoint_path "$TARGET_CHECKPOINT_PATH" \
  --prompt_file "$PROMPT_FILE" \
  --start_index "$START_INDEX" \
  --max_prompts "$MAX_PROMPTS" \
  --num_blocks "$NUM_BLOCKS" \
  --seed "$SEED" \
  --output_path "$DATA_PATH" \
  2>&1 | tee -a "$LOG_FILE"

log "Training verifier overfit loop"
"$PYTHON" sdvg_verifier_experiment.py train \
  --data_path "$DATA_PATH" \
  --output_path "$VERIFIER_PATH" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  2>&1 | tee -a "$LOG_FILE"

log "Done. Dataset: $DATA_PATH"
log "Done. Verifier: $VERIFIER_PATH"
