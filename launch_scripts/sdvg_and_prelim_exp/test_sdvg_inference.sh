#!/usr/bin/env bash
# Smoke test SDVG inference scaffold with heuristic routing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${SF_VENV:-$PROJECT_ROOT/sf_venv}"
PYTHON="$VENV_DIR/bin/python"

MODEL_ROOT="${MODEL_ROOT:-/mnt/lanxiangh/models}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/self_forcing_dmd.yaml}"
DRAFT_CHECKPOINT_PATH="${DRAFT_CHECKPOINT_PATH:-$MODEL_ROOT/Self-Forcing/checkpoints/self_forcing_dmd.pt}"
TARGET_CHECKPOINT_PATH="${TARGET_CHECKPOINT_PATH:-$MODEL_ROOT/realtime-video/checkpoints/krea-realtime-video-14b.safetensors}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/sdvg/smoke_$(date +%Y%m%d_%H%M%S)}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
NUM_BLOCKS="${NUM_BLOCKS:-3}"
ROUTER_MODE="${ROUTER_MODE:-heuristic}"
TAU="${TAU:-0.02}"
MODE="${MODE:-compare}"
SEED="${SEED:-42}"

LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/launch_scripts/logs}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
LOG_FILE="$LOG_DIR/$(basename "${BASH_SOURCE[0]}" .sh)_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }

cd "$PROJECT_ROOT"

for path in "$PYTHON" "$DRAFT_CHECKPOINT_PATH" "$TARGET_CHECKPOINT_PATH" "$MODEL_ROOT/wan_models/Wan2.1-T2V-1.3B" "$MODEL_ROOT/wan_models/Wan2.1-T2V-14B"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: required path missing: $path" >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

log "Mode:      $MODE"
log "Router:    $ROUTER_MODE tau=$TAU"
log "Blocks:    $NUM_BLOCKS"
log "Drafter:   $DRAFT_CHECKPOINT_PATH"
log "Target:    $TARGET_CHECKPOINT_PATH"
log "Output:    $OUTPUT_DIR"

"$PYTHON" sdvg_inference.py \
  "$ROUTER_MODE" \
  --config_path "$CONFIG_PATH" \
  --model_root "$MODEL_ROOT" \
  --draft_checkpoint_path "$DRAFT_CHECKPOINT_PATH" \
  --target_checkpoint_path "$TARGET_CHECKPOINT_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --mode "$MODE" \
  --tau "$TAU" \
  --num_blocks "$NUM_BLOCKS" \
  --seed "$SEED" \
  2>&1 | tee -a "$LOG_FILE"

log "Done. Profile: $OUTPUT_DIR/profile.json"
