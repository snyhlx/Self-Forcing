#!/usr/bin/env bash
# Smoke test: launch Flask/SocketIO demo server (manual stop with Ctrl+C).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${SF_VENV:-$PROJECT_ROOT/sf_venv}"
PYTHON="${VENV_DIR}/bin/python"

MODEL_ROOT="${MODEL_ROOT:-/mnt/lanxiangh/models}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/self_forcing_dmd.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$MODEL_ROOT/Self-Forcing/checkpoints/self_forcing_dmd.pt}"
WAN_MODEL_DIR="${WAN_MODEL_DIR:-$MODEL_ROOT/wan_models/Wan2.1-T2V-1.3B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5001}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"

LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/launch_scripts/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(basename "${BASH_SOURCE[0]}" .sh)_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }

cd "$PROJECT_ROOT"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: venv not found at $VENV_DIR" >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "ERROR: checkpoint missing: $CHECKPOINT_PATH" >&2
  exit 1
fi

if [[ ! -d "$WAN_MODEL_DIR" ]]; then
  echo "ERROR: Wan base model missing: $WAN_MODEL_DIR" >&2
  exit 1
fi

mkdir -p "$PROJECT_ROOT/wan_models"
ln -sfn "$WAN_MODEL_DIR" "$PROJECT_ROOT/wan_models/Wan2.1-T2V-1.3B"

log "Starting demo at http://${HOST}:${PORT}"
log "Wan model: $WAN_MODEL_DIR"
log "Checkpoint: $CHECKPOINT_PATH"
log "Open in browser, then stop with Ctrl+C"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

exec "$PYTHON" demo.py \
  --host "$HOST" \
  --port "$PORT" \
  --config_path "$CONFIG_PATH" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  2>&1 | tee -a "$LOG_FILE"
