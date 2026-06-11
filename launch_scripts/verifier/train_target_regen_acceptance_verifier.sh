#!/usr/bin/env bash
# Train the latent verifier to predict whether a draft block agrees with target regeneration.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${SF_VENV:-$PROJECT_ROOT/sf_venv}"
PYTHON="$VENV_DIR/bin/python"

DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/outputs/sdvg/verifier/target_regen_pairs.pt}"
OUTPUT_PATH="${OUTPUT_PATH:-$PROJECT_ROOT/outputs/sdvg/verifier/target_regen_acceptance_verifier.pt}"
TAU_DELTA="${TAU_DELTA:-}"
NUM_BLOCKS="${NUM_BLOCKS:-9}"
EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-32}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
LOG_EVERY="${LOG_EVERY:-25}"

LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/launch_scripts/logs}"
mkdir -p "$LOG_DIR" "$(dirname "$OUTPUT_PATH")"
LOG_FILE="$LOG_DIR/$(basename "${BASH_SOURCE[0]}" .sh)_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }

cd "$PROJECT_ROOT"

if [[ ! -f "$DATA_PATH" ]]; then
  echo "ERROR: missing target-regeneration pairs: $DATA_PATH" >&2
  echo "Generate them with SAVE_PAIRS=1 STORE_CONTEXT=1 bash launch_scripts/profile_target_regen_moviegen.sh" >&2
  exit 1
fi

TAU_ARGS=()
if [[ -n "$TAU_DELTA" ]]; then
  TAU_ARGS+=(--tau_delta "$TAU_DELTA")
fi

log "Training target-regeneration acceptance verifier"
log "Pairs:  $DATA_PATH"
log "Output: $OUTPUT_PATH"
log "Tau:    ${TAU_DELTA:-stored threshold from pairs}"
log "Epochs: $EPOCHS batch_size=$BATCH_SIZE"

"$PYTHON" sdvg_verifier_experiment.py train_acceptance \
  --data_path "$DATA_PATH" \
  --output_path "$OUTPUT_PATH" \
  "${TAU_ARGS[@]}" \
  --num_blocks "$NUM_BLOCKS" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --hidden_dim "$HIDDEN_DIM" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --log_every "$LOG_EVERY" \
  2>&1 | tee -a "$LOG_FILE"

log "Verifier checkpoint: $OUTPUT_PATH"
