#!/usr/bin/env bash
# Train a Wan-token DFlash-style draft head with clean-latent + FlowMatch + DMD loss.
#
# This wrapper keeps the Wan-DFlash target-initialized recipe and enables a
# conservative DMD term by default. Override any exported variable at launch.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="$SCRIPT_DIR/train_draft_head_wan_dflash_clean_latent_mse+flow.sh"

if [[ ! -f "$BASE_SCRIPT" ]]; then
  echo "ERROR: base launcher missing: $BASE_SCRIPT" >&2
  exit 1
fi

export LOSS_TYPE="${LOSS_TYPE:-clean_latent_flow}"
export CLEAN_LATENT_LOSS_WEIGHT="${CLEAN_LATENT_LOSS_WEIGHT:-1.0}"
export FLOW_LOSS_WEIGHT="${FLOW_LOSS_WEIGHT:-0.25}"
export DMD_LOSS_WEIGHT="${DMD_LOSS_WEIGHT:-0.01}"
export DMD_EVERY="${DMD_EVERY:-4}"
export DMD_GUIDANCE_SCALE="${DMD_GUIDANCE_SCALE:-3.0}"
export DMD_MIN_TIMESTEP="${DMD_MIN_TIMESTEP:-20}"
export DMD_MAX_TIMESTEP="${DMD_MAX_TIMESTEP:-980}"
export EPOCHS="${EPOCHS:-3}"

exec bash "$BASE_SCRIPT"
