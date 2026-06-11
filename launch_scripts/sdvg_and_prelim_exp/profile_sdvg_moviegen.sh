#!/usr/bin/env bash
# Profile SDVG on MovieGenVideoBench prompts, the benchmark used in the SDVG paper.
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
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
NUM_BLOCKS="${NUM_BLOCKS:-9}"
MAX_PROMPTS="${MAX_PROMPTS:-16}"
START_INDEX="${START_INDEX:-0}"
# ROUTER_MODE: heuristic, image_reward, latent_verifier
ROUTER_MODE="${ROUTER_MODE:-image_reward}"
REWARD_DEVICE="${REWARD_DEVICE:-cuda:0}"
REWARD_MODEL_NAME="${REWARD_MODEL_NAME:-ImageReward-v1.0}"
REWARD_DOWNLOAD_ROOT="${REWARD_DOWNLOAD_ROOT:-/mnt/lanxiangh/models/ImageReward}"
VERIFIER_CHECKPOINT_PATH="${VERIFIER_CHECKPOINT_PATH:-}"
VERIFIER_DEVICE="${VERIFIER_DEVICE:-cuda:0}"
REUSE_SCORING_DECODES_FOR_OUTPUT="${REUSE_SCORING_DECODES_FOR_OUTPUT:-1}"
if [[ "$ROUTER_MODE" == "image_reward" ]]; then
  TAU="${TAU:--0.5}"
  TAU_SWEEP="${TAU_SWEEP:--0.8 -0.6 -0.5 -0.4 -0.2 0.0}"
elif [[ "$ROUTER_MODE" == "latent_verifier" ]]; then
  TAU="${TAU:-0.0}"
  TAU_SWEEP="${TAU_SWEEP:-0.0}"
else
  TAU="${TAU:-0.02}"
  TAU_SWEEP="${TAU_SWEEP:-0.05 0.1 0.2 0.3 0.5 0.7 0.9}"
fi
MODE="${MODE:-compare}"
SEED="${SEED:-42}"
PROMPT_TAG="start${START_INDEX}_max${MAX_PROMPTS}"
TAU_TAG="${TAU_SWEEP:-tau${TAU}}"
TAU_TAG="$(printf '%s' "$TAU_TAG" | tr ' ' '_' | sed 's/-/neg_/g; s/\./p/g')"
RUN_TAG="${RUN_TAG:-${ROUTER_MODE}_blocks${NUM_BLOCKS}_${PROMPT_TAG}_${TAU_TAG}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/sdvg/moviegen_${RUN_TAG}}"

LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/launch_scripts/logs}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
LOG_FILE="$LOG_DIR/$(basename "${BASH_SOURCE[0]}" .sh)_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }

cd "$PROJECT_ROOT"

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "ERROR: prompt file missing: $PROMPT_FILE" >&2
  echo "Expected MovieGenVideoBench prompts at /root/Self-Forcing/prompts/MovieGenVideoBench.txt" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

log "Dataset:   $PROMPT_FILE"
log "Prompts:   start=$START_INDEX max=$MAX_PROMPTS (set MAX_PROMPTS=0 for all)"
log "Mode:      $MODE"
if [[ -n "$TAU_SWEEP" ]]; then
  log "Router:    $ROUTER_MODE tau_sweep=[$TAU_SWEEP]"
else
  log "Router:    $ROUTER_MODE tau=$TAU"
fi
if [[ "$ROUTER_MODE" == "image_reward" ]]; then
  log "Reward:    $REWARD_MODEL_NAME on $REWARD_DEVICE"
  log "Reward weights: $REWARD_DOWNLOAD_ROOT"
fi
if [[ "$ROUTER_MODE" == "latent_verifier" ]]; then
  log "Verifier:  $VERIFIER_CHECKPOINT_PATH on $VERIFIER_DEVICE"
fi
log "Reuse scoring decodes for output: $REUSE_SCORING_DECODES_FOR_OUTPUT"
log "Blocks:    $NUM_BLOCKS"
log "Output:    $OUTPUT_DIR"

REUSE_OUTPUT_ARGS=()
if [[ "$REUSE_SCORING_DECODES_FOR_OUTPUT" == "1" ]]; then
  REUSE_OUTPUT_ARGS+=(--reuse_scoring_decodes_for_output)
fi

VERIFIER_ARGS=()
if [[ "$ROUTER_MODE" == "latent_verifier" ]]; then
  if [[ -z "$VERIFIER_CHECKPOINT_PATH" ]]; then
    echo "ERROR: VERIFIER_CHECKPOINT_PATH is required when ROUTER_MODE=latent_verifier" >&2
    exit 1
  fi
  VERIFIER_ARGS+=(--verifier_checkpoint_path "$VERIFIER_CHECKPOINT_PATH" --verifier_device "$VERIFIER_DEVICE")
fi

run_one_tau() {
  local tau="$1"
  local tau_label
  tau_label="$(printf '%s' "$tau" | sed 's/-/neg_/g; s/\./p/g')"
  local run_output_dir="$OUTPUT_DIR/tau_${tau_label}"

  mkdir -p "$run_output_dir"
  log "=== Running tau=$tau -> $run_output_dir ==="

  "$PYTHON" sdvg_inference.py \
    "$ROUTER_MODE" \
    --config_path "$CONFIG_PATH" \
    --model_root "$MODEL_ROOT" \
    --draft_checkpoint_path "$DRAFT_CHECKPOINT_PATH" \
    --target_checkpoint_path "$TARGET_CHECKPOINT_PATH" \
    --prompt_file "$PROMPT_FILE" \
    --start_index "$START_INDEX" \
    --max_prompts "$MAX_PROMPTS" \
    --output_dir "$run_output_dir" \
    --mode "$MODE" \
    --tau "$tau" \
    --reward_device "$REWARD_DEVICE" \
    --reward_model_name "$REWARD_MODEL_NAME" \
    --reward_download_root "$REWARD_DOWNLOAD_ROOT" \
    "${VERIFIER_ARGS[@]}" \
    "${REUSE_OUTPUT_ARGS[@]}" \
    --num_blocks "$NUM_BLOCKS" \
    --seed "$SEED" \
    2>&1 | tee -a "$LOG_FILE"

  log "Profile JSON: $run_output_dir/profile.json"
}

if [[ -n "$TAU_SWEEP" ]]; then
  for tau in $TAU_SWEEP; do
    run_one_tau "$tau"
  done
else
  run_one_tau "$TAU"
fi

log "Sweep complete. Outputs saved under: $OUTPUT_DIR"
