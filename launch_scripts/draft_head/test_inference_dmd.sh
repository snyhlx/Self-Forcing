#!/usr/bin/env bash
# Smoke test: single-prompt CLI inference (Self-Forcing DMD checkpoint).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${SF_VENV:-$PROJECT_ROOT/sf_venv}"
PYTHON="${VENV_DIR}/bin/python"

MODEL_ROOT="${MODEL_ROOT:-/mnt/lanxiangh/models}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/self_forcing_dmd.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$MODEL_ROOT/Self-Forcing/checkpoints/self_forcing_dmd.pt}"
WAN_MODEL_DIR="${WAN_MODEL_DIR:-$MODEL_ROOT/wan_models/Wan2.1-T2V-1.3B}"
PROMPT_FILE="${PROMPT_FILE:-$PROJECT_ROOT/prompts/test_single.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/test_runs/inference_dmd_$(date +%Y%m%d_%H%M%S)}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-21}"
SEED="${SEED:-42}"

LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/launch_scripts/logs}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
LOG_FILE="$LOG_DIR/$(basename "${BASH_SOURCE[0]}" .sh)_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }

cd "$PROJECT_ROOT"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: venv not found at $VENV_DIR (expected sf_venv)." >&2
  echo "Setup: conda create -n self_forcing python=3.10 -y && pip install -r requirements.txt && python setup.py develop" >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "ERROR: checkpoint missing: $CHECKPOINT_PATH" >&2
  echo "Download: huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir $MODEL_ROOT/Self-Forcing" >&2
  exit 1
fi

if [[ ! -d "$WAN_MODEL_DIR" ]]; then
  echo "ERROR: Wan base model missing: $WAN_MODEL_DIR" >&2
  echo "Download: huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir $MODEL_ROOT/wan_models/Wan2.1-T2V-1.3B" >&2
  exit 1
fi

mkdir -p "$PROJECT_ROOT/wan_models"
ln -sfn "$WAN_MODEL_DIR" "$PROJECT_ROOT/wan_models/Wan2.1-T2V-1.3B"

mkdir -p "$(dirname "$PROMPT_FILE")"
if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "A hyperrealistic close-up of ocean waves shimmering at sunset, golden hour light." > "$PROMPT_FILE"
fi

log "Project: $PROJECT_ROOT"
log "Python:  $PYTHON ($("$PYTHON" -V 2>&1))"
log "GPU:     CUDA_VISIBLE_DEVICES=$CUDA_DEVICE"
log "Config:  $CONFIG_PATH"
log "Ckpt:    $CHECKPOINT_PATH"
log "Wan:     $WAN_MODEL_DIR"
log "Prompt:  $PROMPT_FILE"
log "Output:  $OUTPUT_DIR"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

"$PYTHON" inference.py \
  --config_path "$CONFIG_PATH" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --data_path "$PROMPT_FILE" \
  --output_folder "$OUTPUT_DIR" \
  --num_output_frames "$NUM_OUTPUT_FRAMES" \
  --seed "$SEED" \
  --use_ema \
  2>&1 | tee -a "$LOG_FILE"

log "Done. Videos in $OUTPUT_DIR"
ls -la "$OUTPUT_DIR" | tee -a "$LOG_FILE"
