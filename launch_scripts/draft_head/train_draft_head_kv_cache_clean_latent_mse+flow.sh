#!/usr/bin/env bash
# Train the DFlash-style KV-cache draft head with clean-latent + FlowMatch loss.
#
# This intentionally delegates to train_draft_head_clean_latent_mse+flow.sh so
# optimizer/loss/DDP defaults stay in one place. The distinguishing defaults are:
#   HEAD_TYPE=kv_cache_attention
#   LAYER_NAMES=blocks.8 blocks.16 blocks.24
#   INIT_TARGET_BLOCKS=disabled
#   online target KV replay during training
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$SCRIPT_DIR/train_draft_head_clean_latent_mse+flow.sh"

if [[ ! -f "$TEMPLATE" ]]; then
  echo "ERROR: template launcher missing: $TEMPLATE" >&2
  exit 1
fi

export HEAD_TYPE="${HEAD_TYPE:-kv_cache_attention}"
export LAYER_NAMES="${LAYER_NAMES:-blocks.8 blocks.16 blocks.24}"

# KV cache is generated on the fly by replaying committed target context, so the
# existing full-feature draft-head dataset is enough. No stored-KV manifest is used.
DEFAULT_FULL_MANIFEST_PATH="/mnt/lanxiangh/data/ff_exec/draft_head_full_dataset/tau_delta_0p0/manifest.json"
export MANIFEST_PATH="${MANIFEST_PATH:-$DEFAULT_FULL_MANIFEST_PATH}"

# Target-block initialization copies Wan hidden-feature transformer weights into
# the older feature-attention head. It is not compatible with direct KV reuse.
export INIT_TARGET_BLOCKS="${INIT_TARGET_BLOCKS-}"

# Keep the high-capacity defaults aligned with the current single-step best runs.
export HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-5120}"
export NUM_LAYERS="${NUM_LAYERS:-3}"
export NUM_HEADS="${NUM_HEADS:-40}"
export MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-4680}"
export LATENT_POOL="${LATENT_POOL:-1 1 1}"
export FFN_DIM="${FFN_DIM:-13824}"
export TRAINING_MODE="${TRAINING_MODE:-unrolled}"
#export TRAINING_MODE="${TRAINING_MODE:-one_step}"
export BATCH_SIZE=4
export AMP_DTYPE="${AMP_DTYPE:-bf16}"
export LOSS_TYPE="${LOSS_TYPE:-clean_latent_flow}"
export CLEAN_LATENT_LOSS_WEIGHT="${CLEAN_LATENT_LOSS_WEIGHT:-1.0}"
export FLOW_LOSS_WEIGHT="${FLOW_LOSS_WEIGHT:-0.25}"
export MODEL_ROOT="${MODEL_ROOT:-/mnt/lanxiangh/models}"
export KV_CACHE_TARGET_MODEL_NAME="${KV_CACHE_TARGET_MODEL_NAME:-Wan2.1-T2V-14B}"
export KV_CACHE_TARGET_CHECKPOINT_PATH="${KV_CACHE_TARGET_CHECKPOINT_PATH:-/mnt/lanxiangh/models/realtime-video/checkpoints/krea-realtime-video-14b.safetensors}"
export ONLINE_KV_NOISE_SEED="${ONLINE_KV_NOISE_SEED:-42}"

WAN_MODEL_DIR="$MODEL_ROOT/wan_models"
for model_name in Wan2.1-T2V-1.3B Wan2.1-T2V-14B; do
  if [[ ! -d "$WAN_MODEL_DIR/$model_name" ]]; then
    echo "ERROR: missing Wan model directory: $WAN_MODEL_DIR/$model_name" >&2
    echo "Set MODEL_ROOT to the parent containing wan_models/, e.g. MODEL_ROOT=/mnt/lanxiangh/models" >&2
    exit 1
  fi
done
if [[ ! -f "$KV_CACHE_TARGET_CHECKPOINT_PATH" ]]; then
  echo "ERROR: missing KV-cache target checkpoint: $KV_CACHE_TARGET_CHECKPOINT_PATH" >&2
  exit 1
fi

exec bash "$TEMPLATE"
