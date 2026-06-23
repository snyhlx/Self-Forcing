#!/usr/bin/env bash
# Train the target-conditioned latent draft head from collected target_regen supervision.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="${SF_VENV:-$PROJECT_ROOT/sf_venv}"
PYTHON="$VENV_DIR/bin/python"

DEFAULT_MANIFEST_PATH="/mnt/lanxiangh/data/ff_exec/draft_head_full_dataset/tau_delta_0p0/manifest.json"
LEGACY_MANIFEST_PATH="$PROJECT_ROOT/outputs/sdvg/draft_head_dataset/moviegen_target_context/tau_delta_0p0/manifest.json"
MANIFEST_PATH_WAS_SET="${MANIFEST_PATH+x}"
MANIFEST_PATH="${MANIFEST_PATH:-$DEFAULT_MANIFEST_PATH}"
NUM_BLOCKS="${NUM_BLOCKS:-9}"
LAYER_NAMES="${LAYER_NAMES:-}"
HEAD_TYPE="${HEAD_TYPE:-attention}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-5120}"
NUM_LAYERS="${NUM_LAYERS:-3}"
NUM_HEADS="${NUM_HEADS:-40}"
NUM_RES_BLOCKS="${NUM_RES_BLOCKS:-2}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-4680}"
LATENT_POOL="${LATENT_POOL:-1 1 1}"
FFN_MULT="${FFN_MULT:-4}"
FFN_DIM="${FFN_DIM:-13824}"
DENOISING_STEP_LIST="${DENOISING_STEP_LIST:-1000 750 500 250 0}"
TIMESTEP_SHIFT="${TIMESTEP_SHIFT:-5.0}"
LOSS_TYPE="${LOSS_TYPE:-clean_latent_flow}"
CLEAN_LATENT_LOSS_WEIGHT="${CLEAN_LATENT_LOSS_WEIGHT:-1.0}"
FLOW_LOSS_WEIGHT="${FLOW_LOSS_WEIGHT:-1.0}"
DMD_LOSS_WEIGHT="${DMD_LOSS_WEIGHT:-0.0}"
DMD_EVERY="${DMD_EVERY:-1}"
DMD_MODEL_NAME="${DMD_MODEL_NAME:-Wan2.1-T2V-14B}"
DMD_CHECKPOINT_PATH="${DMD_CHECKPOINT_PATH:-/mnt/lanxiangh/models/realtime-video/checkpoints/krea-realtime-video-14b.safetensors}"
DMD_GUIDANCE_SCALE="${DMD_GUIDANCE_SCALE:-3.0}"
DMD_MIN_TIMESTEP="${DMD_MIN_TIMESTEP:-20}"
DMD_MAX_TIMESTEP="${DMD_MAX_TIMESTEP:-980}"
INIT_TARGET_BLOCKS_WAS_SET="${INIT_TARGET_BLOCKS+x}"
INIT_TARGET_BLOCKS="${INIT_TARGET_BLOCKS:-8 16 24}"
if [[ "$HEAD_TYPE" == "ar_bidirectional" ]] && [[ -z "$INIT_TARGET_BLOCKS_WAS_SET" ]]; then
  INIT_TARGET_BLOCKS=""
fi
INIT_TARGET_CHECKPOINT_PATH="${INIT_TARGET_CHECKPOINT_PATH:-/mnt/lanxiangh/models/realtime-video/checkpoints/krea-realtime-video-14b.safetensors}"
INIT_TARGET_MODEL_NAME="${INIT_TARGET_MODEL_NAME:-Wan2.1-T2V-14B}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/lanxiangh/models}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/self_forcing_dmd.yaml}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
VAL_FRACTION="${VAL_FRACTION:-0.05}"
MAX_RECORDS="${MAX_RECORDS:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && "$NUM_GPUS" -gt 1 ]]; then
  CUDA_DEVICE="$(seq -s, 0 $((NUM_GPUS - 1)))"
else
  CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
fi

STEP_TAG="$(printf '%s' "$DENOISING_STEP_LIST" | tr ' ' '_')"
CONFIG_TAG="moviegen_draft_head_${HEAD_TYPE}_${LOSS_TYPE}_h${HIDDEN_CHANNELS}_l${NUM_LAYERS}_heads${NUM_HEADS}_ctx${MAX_CONTEXT_TOKENS}_pool$(printf '%s' "$LATENT_POOL" | tr ' ' 'x')_ffn${FFN_DIM}_steps${STEP_TAG}_blocks${NUM_BLOCKS}"
if [[ "$HEAD_TYPE" == "conv" ]]; then
  CONFIG_TAG="moviegen_draft_head_${HEAD_TYPE}_${LOSS_TYPE}_h${HIDDEN_CHANNELS}_res${NUM_RES_BLOCKS}_steps${STEP_TAG}_blocks${NUM_BLOCKS}"
fi
if [[ -n "$INIT_TARGET_BLOCKS" ]]; then
  CONFIG_TAG="${CONFIG_TAG}_targetinit$(printf '%s' "$INIT_TARGET_BLOCKS" | tr ' ' '_')"
fi
if [[ "$DMD_LOSS_WEIGHT" != "0" && "$DMD_LOSS_WEIGHT" != "0.0" ]]; then
  CONFIG_TAG="${CONFIG_TAG}_dmd${DMD_LOSS_WEIGHT}"
fi
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/outputs/draft_head/checkpoints}"
RUN_DIR="${RUN_DIR:-$CHECKPOINT_ROOT/${RUN_TIMESTAMP}_${CONFIG_TAG}}"
OUTPUT_PATH="${OUTPUT_PATH:-$RUN_DIR/final.pt}"

LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/launch_scripts/logs}"
mkdir -p "$LOG_DIR" "$(dirname "$OUTPUT_PATH")"
LOG_FILE="$LOG_DIR/$(basename "${BASH_SOURCE[0]}" .sh)_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }

cd "$PROJECT_ROOT"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: python not found or not executable: $PYTHON" >&2
  exit 1
fi
if [[ -z "$MANIFEST_PATH_WAS_SET" && ! -f "$MANIFEST_PATH" && -f "$LEGACY_MANIFEST_PATH" ]]; then
  MANIFEST_PATH="$LEGACY_MANIFEST_PATH"
fi
if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "ERROR: draft-head manifest missing: $MANIFEST_PATH" >&2
  echo "Checked default pooled manifest: $DEFAULT_MANIFEST_PATH" >&2
  echo "Checked legacy manifest: $LEGACY_MANIFEST_PATH" >&2
  echo "Generate it with DRAFT_HEAD_DATASET_DIR=... DRAFT_HEAD_CAPTURE_LAYERS='blocks.X ...' bash launch_scripts/profile_target_regen_moviegen.sh" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

LAYER_ARGS=()
if [[ -n "$LAYER_NAMES" ]]; then
  # shellcheck disable=SC2206
  LAYER_ARRAY=($LAYER_NAMES)
  LAYER_ARGS+=(--layer_names "${LAYER_ARRAY[@]}")
fi

MAX_RECORD_ARGS=()
if [[ "$MAX_RECORDS" != "0" ]]; then
  MAX_RECORD_ARGS+=(--max_records "$MAX_RECORDS")
fi

# shellcheck disable=SC2206
DENOISING_STEP_ARRAY=($DENOISING_STEP_LIST)

TARGET_INIT_ARGS=()
if [[ -n "$INIT_TARGET_BLOCKS" ]]; then
  if [[ -z "$INIT_TARGET_CHECKPOINT_PATH" ]]; then
    echo "ERROR: INIT_TARGET_CHECKPOINT_PATH is required when INIT_TARGET_BLOCKS is set" >&2
    exit 1
  fi
  # shellcheck disable=SC2206
  INIT_TARGET_BLOCK_ARRAY=($INIT_TARGET_BLOCKS)
  TARGET_INIT_ARGS+=(--init_target_blocks "${INIT_TARGET_BLOCK_ARRAY[@]}")
  TARGET_INIT_ARGS+=(--init_target_checkpoint_path "$INIT_TARGET_CHECKPOINT_PATH")
  TARGET_INIT_ARGS+=(--init_target_model_name "$INIT_TARGET_MODEL_NAME")
  TARGET_INIT_ARGS+=(--model_root "$MODEL_ROOT")
  TARGET_INIT_ARGS+=(--config_path "$CONFIG_PATH")
fi

log "Training latent draft head"
log "Manifest: $MANIFEST_PATH"
log "Run dir:   $RUN_DIR"
log "Output:    $OUTPUT_PATH"
log "Layers:   ${LAYER_NAMES:-infer from manifest records}"
log "Head:     $HEAD_TYPE input_source=scheduled_latents hidden=$HIDDEN_CHANNELS layers=$NUM_LAYERS heads=$NUM_HEADS ffn_dim=$FFN_DIM"
log "Loss:     type=$LOSS_TYPE clean_latent=$CLEAN_LATENT_LOSS_WEIGHT flow=$FLOW_LOSS_WEIGHT dmd=$DMD_LOSS_WEIGHT every=$DMD_EVERY"
log "Schedule: [$DENOISING_STEP_LIST] timestep_shift=$TIMESTEP_SHIFT"
log "Target init: ${INIT_TARGET_BLOCKS:-disabled}"
log "GPUs:     $NUM_GPUS visible=$CUDA_VISIBLE_DEVICES"
log "Epochs:   $EPOCHS batch_size=$BATCH_SIZE lr=$LR"

RUNNER=("$PYTHON")
if [[ "$NUM_GPUS" -gt 1 ]]; then
  RUNNER=("$PYTHON" -m torch.distributed.run --nproc_per_node "$NUM_GPUS")
fi

"${RUNNER[@]}" train_ar_draft_head.py \
  --manifest_path "$MANIFEST_PATH" \
  --output_path "$OUTPUT_PATH" \
  --head_type "$HEAD_TYPE" \
  --num_blocks "$NUM_BLOCKS" \
  "${LAYER_ARGS[@]}" \
  --hidden_channels "$HIDDEN_CHANNELS" \
  --num_layers "$NUM_LAYERS" \
  --num_heads "$NUM_HEADS" \
  --num_res_blocks "$NUM_RES_BLOCKS" \
  --max_context_tokens "$MAX_CONTEXT_TOKENS" \
  --latent_pool $LATENT_POOL \
  --ffn_mult "$FFN_MULT" \
  --ffn_dim "$FFN_DIM" \
  --denoising_step_list "${DENOISING_STEP_ARRAY[@]}" \
  --timestep_shift "$TIMESTEP_SHIFT" \
  --loss_type "$LOSS_TYPE" \
  --clean_latent_loss_weight "$CLEAN_LATENT_LOSS_WEIGHT" \
  --flow_loss_weight "$FLOW_LOSS_WEIGHT" \
  --dmd_loss_weight "$DMD_LOSS_WEIGHT" \
  --dmd_every "$DMD_EVERY" \
  --dmd_model_name "$DMD_MODEL_NAME" \
  --dmd_checkpoint_path "$DMD_CHECKPOINT_PATH" \
  --dmd_guidance_scale "$DMD_GUIDANCE_SCALE" \
  --dmd_min_timestep "$DMD_MIN_TIMESTEP" \
  --dmd_max_timestep "$DMD_MAX_TIMESTEP" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --val_fraction "$VAL_FRACTION" \
  "${MAX_RECORD_ARGS[@]}" \
  "${TARGET_INIT_ARGS[@]}" \
  2>&1 | tee -a "$LOG_FILE"

log "Draft-head checkpoint: $OUTPUT_PATH"
