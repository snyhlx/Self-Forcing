#!/usr/bin/env bash
# Train the target-conditioned latent draft head with clean-latent + FlowMatch loss.
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
#TRAINING_MODE="${TRAINING_MODE:-unrolled}"
TRAINING_MODE="${TRAINING_MODE:-one_step}"
UNROLL_STEP_WEIGHTS="${UNROLL_STEP_WEIGHTS:-}"
UNROLL_NOISE_MODE="${UNROLL_NOISE_MODE:-fixed}"
#AMP_DTYPE="${AMP_DTYPE:-none}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
FREEZE_COPIED_WAN_EPOCHS="${FREEZE_COPIED_WAN_EPOCHS:-0}"
PARALLEL_STRATEGY="${PARALLEL_STRATEGY:-ddp}"
FSDP_MIN_NUM_PARAMS="${FSDP_MIN_NUM_PARAMS:-100000000}"
FSDP_MIXED_PRECISION="${FSDP_MIXED_PRECISION:-none}"
LOSS_TYPE="${LOSS_TYPE:-clean_latent_flow}"
CLEAN_LATENT_LOSS_WEIGHT="${CLEAN_LATENT_LOSS_WEIGHT:-1.0}"
FLOW_LOSS_WEIGHT="${FLOW_LOSS_WEIGHT:-0.25}"
DMD_LOSS_WEIGHT="${DMD_LOSS_WEIGHT:-0.0}"
DMD_EVERY="${DMD_EVERY:-1}"
DMD_MODEL_NAME="${DMD_MODEL_NAME:-Wan2.1-T2V-14B}"
DMD_CHECKPOINT_PATH="${DMD_CHECKPOINT_PATH:-/mnt/lanxiangh/models/realtime-video/checkpoints/krea-realtime-video-14b.safetensors}"
DMD_GUIDANCE_SCALE="${DMD_GUIDANCE_SCALE:-3.0}"
DMD_MIN_TIMESTEP="${DMD_MIN_TIMESTEP:-20}"
DMD_MAX_TIMESTEP="${DMD_MAX_TIMESTEP:-980}"
INIT_TARGET_BLOCKS_WAS_SET="${INIT_TARGET_BLOCKS+x}"
INIT_TARGET_BLOCKS="${INIT_TARGET_BLOCKS-8 16 24}"
if [[ "$HEAD_TYPE" == "kv_cache_attention" && -z "$INIT_TARGET_BLOCKS_WAS_SET" ]]; then
  INIT_TARGET_BLOCKS=""
fi
INIT_TARGET_CHECKPOINT_PATH="${INIT_TARGET_CHECKPOINT_PATH:-/mnt/lanxiangh/models/realtime-video/checkpoints/krea-realtime-video-14b.safetensors}"
INIT_TARGET_MODEL_NAME="${INIT_TARGET_MODEL_NAME:-Wan2.1-T2V-14B}"
INIT_DRAFT_HEAD_CHECKPOINT_PATH="${INIT_DRAFT_HEAD_CHECKPOINT_PATH:-}"
KV_CACHE_TARGET_MODEL_NAME="${KV_CACHE_TARGET_MODEL_NAME:-Wan2.1-T2V-14B}"
KV_CACHE_TARGET_CHECKPOINT_PATH="${KV_CACHE_TARGET_CHECKPOINT_PATH:-/mnt/lanxiangh/models/realtime-video/checkpoints/krea-realtime-video-14b.safetensors}"
ONLINE_KV_NOISE_SEED="${ONLINE_KV_NOISE_SEED:-42}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/lanxiangh/models}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/self_forcing_dmd.yaml}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
VAL_FRACTION="${VAL_FRACTION:-0.05}"
MAX_RECORDS="${MAX_RECORDS:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && "$NUM_GPUS" -gt 1 ]]; then
  CUDA_DEVICE="$(seq -s, 0 $((NUM_GPUS - 1)))"
else
  CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
fi

tag_slug() {
  printf '%s' "$1" | tr ' /.,:' '_____'
}

short_hash() {
  printf '%s' "$1" | sha1sum | cut -c1-8
}

STEP_TAG="$(tag_slug "$DENOISING_STEP_LIST")"
POOL_TAG="$(printf '%s' "$LATENT_POOL" | tr ' ' 'x')"
LR_TAG="$(tag_slug "$LR")"
CLEAN_TAG="$(tag_slug "$CLEAN_LATENT_LOSS_WEIGHT")"
FLOW_TAG="$(tag_slug "$FLOW_LOSS_WEIGHT")"
SHIFT_TAG="$(tag_slug "$TIMESTEP_SHIFT")"
AMP_TAG="$(tag_slug "$AMP_DTYPE")"
GC_TAG="$(tag_slug "$GRADIENT_CHECKPOINTING")"
FREEZE_WAN_TAG="$(tag_slug "$FREEZE_COPIED_WAN_EPOCHS")"
DMD_TAG="$(tag_slug "$DMD_LOSS_WEIGHT")"
FULL_CONFIG_TAG="dh_${HEAD_TYPE}_${LOSS_TYPE}_${TRAINING_MODE}_h${HIDDEN_CHANNELS}_l${NUM_LAYERS}_nh${NUM_HEADS}_ctx${MAX_CONTEXT_TOKENS}_pool${POOL_TAG}_ffm${FFN_MULT}_ffn${FFN_DIM}_st${STEP_TAG}_sh${SHIFT_TAG}_nb${NUM_BLOCKS}_bs${BATCH_SIZE}_gpu${NUM_GPUS}_ep${EPOCHS}_lr${LR_TAG}_wd$(tag_slug "$WEIGHT_DECAY")_val$(tag_slug "$VAL_FRACTION")_cl${CLEAN_TAG}_fl${FLOW_TAG}_dmd${DMD_TAG}_de${DMD_EVERY}_amp${AMP_TAG}_gc${GC_TAG}_fw${FREEZE_WAN_TAG}_par${PARALLEL_STRATEGY}"
if [[ -n "$LAYER_NAMES" ]]; then
  FULL_CONFIG_TAG="${FULL_CONFIG_TAG}_layers$(tag_slug "$LAYER_NAMES")"
fi
if [[ "$MAX_RECORDS" != "0" ]]; then
  FULL_CONFIG_TAG="${FULL_CONFIG_TAG}_maxrec${MAX_RECORDS}"
fi
if [[ "$TRAINING_MODE" == "unrolled" ]]; then
  FULL_CONFIG_TAG="${FULL_CONFIG_TAG}_unoise$(tag_slug "$UNROLL_NOISE_MODE")"
  if [[ -n "$UNROLL_STEP_WEIGHTS" ]]; then
    FULL_CONFIG_TAG="${FULL_CONFIG_TAG}_uw$(tag_slug "$UNROLL_STEP_WEIGHTS")"
  else
    FULL_CONFIG_TAG="${FULL_CONFIG_TAG}_uwauto"
  fi
fi
if [[ -n "$INIT_TARGET_BLOCKS" ]]; then
  FULL_CONFIG_TAG="${FULL_CONFIG_TAG}_tinit$(tag_slug "$INIT_TARGET_BLOCKS")"
fi
if [[ -n "$INIT_DRAFT_HEAD_CHECKPOINT_PATH" ]]; then
  INIT_DRAFT_TAG="$(basename "$(dirname "$INIT_DRAFT_HEAD_CHECKPOINT_PATH")")_$(basename "$INIT_DRAFT_HEAD_CHECKPOINT_PATH")"
  FULL_CONFIG_TAG="${FULL_CONFIG_TAG}_dinit$(tag_slug "$INIT_DRAFT_TAG")"
fi
if [[ "$HEAD_TYPE" == "kv_cache_attention" ]]; then
  FULL_CONFIG_TAG="${FULL_CONFIG_TAG}_onlinekv_ks${ONLINE_KV_NOISE_SEED}_ktm$(tag_slug "$KV_CACHE_TARGET_MODEL_NAME")"
fi
if [[ "$DMD_LOSS_WEIGHT" != "0" && "$DMD_LOSS_WEIGHT" != "0.0" ]]; then
  FULL_CONFIG_TAG="${FULL_CONFIG_TAG}_dgs$(tag_slug "$DMD_GUIDANCE_SCALE")_dt${DMD_MIN_TIMESTEP}-${DMD_MAX_TIMESTEP}_dm$(tag_slug "$DMD_MODEL_NAME")"
fi
CONFIG_HASH="$(short_hash "$FULL_CONFIG_TAG")"
CONFIG_TAG="dh_${HEAD_TYPE}_${LOSS_TYPE}_${TRAINING_MODE}_h${HIDDEN_CHANNELS}_ctx${MAX_CONTEXT_TOKENS}_p${POOL_TAG}_st${STEP_TAG}_bs${BATCH_SIZE}_g${NUM_GPUS}_lr${LR_TAG}_fl${FLOW_TAG}_amp${AMP_TAG}_gc${GC_TAG}_fw${FREEZE_WAN_TAG}_cfg${CONFIG_HASH}"
if [[ "$HEAD_TYPE" == "conv" ]]; then
  CONFIG_TAG="dh_${HEAD_TYPE}_${LOSS_TYPE}_${TRAINING_MODE}_h${HIDDEN_CHANNELS}_res${NUM_RES_BLOCKS}_st${STEP_TAG}_bs${BATCH_SIZE}_g${NUM_GPUS}_lr${LR_TAG}_fl${FLOW_TAG}_amp${AMP_TAG}_gc${GC_TAG}_fw${FREEZE_WAN_TAG}_cfg${CONFIG_HASH}"
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
  echo "Checked default manifest: $DEFAULT_MANIFEST_PATH" >&2
  echo "Checked legacy manifest: $LEGACY_MANIFEST_PATH" >&2
  echo "Generate it with DRAFT_HEAD_DATASET_DIR=... DRAFT_HEAD_CAPTURE_LAYERS='blocks.X ...' bash launch_scripts/sdvg_and_prelim_exp/profile_target_regen_moviegen.sh" >&2
  exit 1
fi
if [[ -n "$INIT_DRAFT_HEAD_CHECKPOINT_PATH" && ! -f "$INIT_DRAFT_HEAD_CHECKPOINT_PATH" ]]; then
  echo "ERROR: draft-head init checkpoint missing: $INIT_DRAFT_HEAD_CHECKPOINT_PATH" >&2
  exit 1
fi
if [[ "$HEAD_TYPE" == "kv_cache_attention" && ! -f "$KV_CACHE_TARGET_CHECKPOINT_PATH" ]]; then
  echo "ERROR: KV-cache target checkpoint missing: $KV_CACHE_TARGET_CHECKPOINT_PATH" >&2
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
UNROLL_ARGS=(--training_mode "$TRAINING_MODE" --unroll_noise_mode "$UNROLL_NOISE_MODE")
if [[ -n "$UNROLL_STEP_WEIGHTS" ]]; then
  # shellcheck disable=SC2206
  UNROLL_STEP_WEIGHT_ARRAY=($UNROLL_STEP_WEIGHTS)
  UNROLL_ARGS+=(--unroll_step_weights "${UNROLL_STEP_WEIGHT_ARRAY[@]}")
fi
MEMORY_ARGS=(--amp_dtype "$AMP_DTYPE")
if [[ "$GRADIENT_CHECKPOINTING" == "1" || "$GRADIENT_CHECKPOINTING" == "true" ]]; then
  MEMORY_ARGS+=(--gradient_checkpointing)
fi
DRAFT_INIT_ARGS=()
if [[ -n "$INIT_DRAFT_HEAD_CHECKPOINT_PATH" ]]; then
  DRAFT_INIT_ARGS+=(--init_draft_head_checkpoint_path "$INIT_DRAFT_HEAD_CHECKPOINT_PATH")
fi

KV_CACHE_ARGS=()
if [[ "$HEAD_TYPE" == "kv_cache_attention" ]]; then
  KV_CACHE_ARGS+=(--kv_cache_target_model_name "$KV_CACHE_TARGET_MODEL_NAME")
  KV_CACHE_ARGS+=(--kv_cache_target_checkpoint_path "$KV_CACHE_TARGET_CHECKPOINT_PATH")
  KV_CACHE_ARGS+=(--online_kv_noise_seed "$ONLINE_KV_NOISE_SEED")
fi

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

log "Training latent draft head with clean-latent + flow loss"
log "Manifest: $MANIFEST_PATH"
log "Run dir:   $RUN_DIR"
log "Output:    $OUTPUT_PATH"
log "Config:    $FULL_CONFIG_TAG"
log "Config hash: $CONFIG_HASH"
log "Layers:   ${LAYER_NAMES:-infer from manifest records}"
log "Head:     $HEAD_TYPE input_source=scheduled_latents hidden=$HIDDEN_CHANNELS layers=$NUM_LAYERS heads=$NUM_HEADS ffn_dim=$FFN_DIM"
log "Loss:     type=$LOSS_TYPE clean_latent=$CLEAN_LATENT_LOSS_WEIGHT flow=$FLOW_LOSS_WEIGHT dmd=$DMD_LOSS_WEIGHT every=$DMD_EVERY"
log "Training: mode=$TRAINING_MODE unroll_noise=$UNROLL_NOISE_MODE weights=${UNROLL_STEP_WEIGHTS:-auto}"
log "Memory:   amp_dtype=$AMP_DTYPE gradient_checkpointing=$GRADIENT_CHECKPOINTING freeze_copied_wan_epochs=$FREEZE_COPIED_WAN_EPOCHS"
log "Parallel: strategy=$PARALLEL_STRATEGY fsdp_min_num_params=$FSDP_MIN_NUM_PARAMS fsdp_mixed_precision=$FSDP_MIXED_PRECISION"
log "Schedule: [$DENOISING_STEP_LIST] timestep_shift=$TIMESTEP_SHIFT"
log "Target init: ${INIT_TARGET_BLOCKS:-disabled}"
log "Draft init: ${INIT_DRAFT_HEAD_CHECKPOINT_PATH:-disabled}"
if [[ "$HEAD_TYPE" == "kv_cache_attention" ]]; then
  log "Online KV: target_model=$KV_CACHE_TARGET_MODEL_NAME checkpoint=$KV_CACHE_TARGET_CHECKPOINT_PATH noise_seed=$ONLINE_KV_NOISE_SEED"
fi
log "GPUs:     $NUM_GPUS visible=$CUDA_VISIBLE_DEVICES"
log "Epochs:   $EPOCHS batch_size=$BATCH_SIZE lr=$LR"

RUNNER=("$PYTHON")
if [[ "$NUM_GPUS" -gt 1 ]]; then
  RUNNER=("$PYTHON" -m torch.distributed.run --standalone --max_restarts 0 --nproc_per_node "$NUM_GPUS")
fi

"${RUNNER[@]}" train_draft_head.py \
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
  "${UNROLL_ARGS[@]}" \
  "${MEMORY_ARGS[@]}" \
  --parallel_strategy "$PARALLEL_STRATEGY" \
  --fsdp_min_num_params "$FSDP_MIN_NUM_PARAMS" \
  --fsdp_mixed_precision "$FSDP_MIXED_PRECISION" \
  --freeze_copied_wan_epochs "$FREEZE_COPIED_WAN_EPOCHS" \
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
  "${DRAFT_INIT_ARGS[@]}" \
  "${KV_CACHE_ARGS[@]}" \
  "${TARGET_INIT_ARGS[@]}" \
  2>&1 | tee -a "$LOG_FILE"

log "Draft-head checkpoint: $OUTPUT_PATH"
