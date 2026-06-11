#!/usr/bin/env bash
# Train a Wan-token DFlash-style draft head with clean-latent + FlowMatch loss.
#
# This head uses Wan patch/time/head modules and DFlash-style target hidden-state
# conditioning. Unlike the raw KV-cache experiment, it is initialized from target
# Wan blocks by default.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$SCRIPT_DIR/train_draft_head_clean_latent_mse+flow.sh"

if [[ ! -f "$TEMPLATE" ]]; then
  echo "ERROR: template launcher missing: $TEMPLATE" >&2
  exit 1
fi

export HEAD_TYPE="${HEAD_TYPE:-wan_dflash_attention}"
export LAYER_NAMES="${LAYER_NAMES:-blocks.8 blocks.16 blocks.24}"
export MANIFEST_PATH="${MANIFEST_PATH:-/mnt/lanxiangh/data/ff_exec/draft_head_full_dataset/tau_delta_0p0/manifest.json}"

export MODEL_ROOT="${MODEL_ROOT:-/mnt/lanxiangh/models}"
export INIT_TARGET_BLOCKS="${INIT_TARGET_BLOCKS-8 16 24}"
export INIT_TARGET_MODEL_NAME="${INIT_TARGET_MODEL_NAME:-Wan2.1-T2V-14B}"
export INIT_TARGET_CHECKPOINT_PATH="${INIT_TARGET_CHECKPOINT_PATH:-/mnt/lanxiangh/models/realtime-video/checkpoints/krea-realtime-video-14b.safetensors}"

export HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-5120}"
export NUM_LAYERS="${NUM_LAYERS:-3}"
export NUM_HEADS="${NUM_HEADS:-40}"
export MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-4680}"
export LATENT_POOL="${LATENT_POOL:-1 1 1}"
export FFN_DIM="${FFN_DIM:-13824}"
#export TRAINING_MODE="${TRAINING_MODE:-unrolled}"
export TRAINING_MODE="${TRAINING_MODE:-one_step}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export AMP_DTYPE="${AMP_DTYPE:-bf16}"
export LOSS_TYPE="${LOSS_TYPE:-clean_latent_flow}"
export CLEAN_LATENT_LOSS_WEIGHT="${CLEAN_LATENT_LOSS_WEIGHT:-1.0}"
export FLOW_LOSS_WEIGHT="${FLOW_LOSS_WEIGHT:-0.25}"

WAN_MODEL_DIR="$MODEL_ROOT/wan_models"
for model_name in Wan2.1-T2V-1.3B Wan2.1-T2V-14B; do
  if [[ ! -d "$WAN_MODEL_DIR/$model_name" ]]; then
    echo "ERROR: missing Wan model directory: $WAN_MODEL_DIR/$model_name" >&2
    echo "Set MODEL_ROOT to the parent containing wan_models/, e.g. MODEL_ROOT=/mnt/lanxiangh/models" >&2
    exit 1
  fi
done
if [[ ! -f "$INIT_TARGET_CHECKPOINT_PATH" ]]; then
  echo "ERROR: missing target initialization checkpoint: $INIT_TARGET_CHECKPOINT_PATH" >&2
  exit 1
fi

exec bash "$TEMPLATE"
