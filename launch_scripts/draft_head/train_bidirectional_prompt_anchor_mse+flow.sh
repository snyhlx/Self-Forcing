#!/usr/bin/env bash
# Train an all-at-once bidirectional draft head conditioned on prompt embedding
# and target chunk-0 clean latents as an anchor.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="${SF_VENV:-$PROJECT_ROOT/sf_venv}"
PYTHON="$VENV_DIR/bin/python"

TEACHER_SETUP="${TEACHER_SETUP:-bidirectional_wan}"
MANIFEST_PATH="${MANIFEST_PATH:-/mnt/lanxiangh/data/ff_exec/bidirectional_wan_draft_head_dataset/tau_delta_0p0/manifest.json}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/mnt/lanxiangh/data/cache/specgen}"
DATASET_INDEX_WORKERS="${DATASET_INDEX_WORKERS:-8}"
DATASET_CACHE_WAIT_SECONDS="${DATASET_CACHE_WAIT_SECONDS:-3600}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/lanxiangh/models}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/self_forcing_dmd.yaml}"
TARGET_MODEL_NAME="${TARGET_MODEL_NAME:-Wan2.1-T2V-14B}"
TARGET_CHECKPOINT_PATH="${TARGET_CHECKPOINT_PATH:-$MODEL_ROOT/wan_models/$TARGET_MODEL_NAME/diffusion_pytorch_model.safetensors.index.json}"
INIT_MODEL_NAME="${INIT_MODEL_NAME:-}"
INIT_DRAFT_HEAD_CHECKPOINT_PATH="${INIT_DRAFT_HEAD_CHECKPOINT_PATH:-}"
ANCHOR_NOISE_SEED="${ANCHOR_NOISE_SEED:-42}"
NUM_BLOCKS="${NUM_BLOCKS:-9}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-5120}"
NUM_LAYERS="${NUM_LAYERS:-6}"
NUM_HEADS="${NUM_HEADS:-40}"
FFN_DIM="${FFN_DIM:-13824}"
TEMPORAL_MIXER_LAYERS="${TEMPORAL_MIXER_LAYERS:-2}"
TEMPORAL_MIXER_FFN_DIM="${TEMPORAL_MIXER_FFN_DIM:-2048}"
INIT_TARGET_BLOCKS="${INIT_TARGET_BLOCKS:-0 8 16 24 32 39}"
PROMPT_DIM="${PROMPT_DIM:-4096}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
PARALLEL_STRATEGY="${PARALLEL_STRATEGY:-fsdp}"
FSDP_MIN_NUM_PARAMS="${FSDP_MIN_NUM_PARAMS:-100000000}"
FSDP_MIXED_PRECISION="${FSDP_MIXED_PRECISION:-none}"
DENOISING_STEP_LIST="${DENOISING_STEP_LIST:-1000 750 500 250 0}"
DENSE_SCHEDULE_STEPS="${DENSE_SCHEDULE_STEPS:-}"
TIMESTEP_SHIFT="${TIMESTEP_SHIFT:-5.0}"
PREDICTION_TYPE="${PREDICTION_TYPE:-flow}"
TRAINING_MODE="${TRAINING_MODE:-unrolled}"
ANCHOR_CONDITIONING="${ANCHOR_CONDITIONING:-none}"
RANDOM_TIMESTEP_SAMPLING="${RANDOM_TIMESTEP_SAMPLING:-uniform_schedule}"
LOGIT_NORMAL_MEAN="${LOGIT_NORMAL_MEAN:-0.0}"
LOGIT_NORMAL_STD="${LOGIT_NORMAL_STD:-1.0}"
UNROLL_STEP_WEIGHTS="${UNROLL_STEP_WEIGHTS:-}"
UNROLL_NOISE_MODE="${UNROLL_NOISE_MODE:-fixed}"
ROLLOUT_SOLVER="${ROLLOUT_SOLVER:-euler}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-0}"
ROLLOUT_SCHEDULE="${ROLLOUT_SCHEDULE:-}"
ROLLOUT_SIGMA_MAX="${ROLLOUT_SIGMA_MAX:-1600}"
ROLLOUT_SOLVER_SHIFT="${ROLLOUT_SOLVER_SHIFT:-8.0}"
TEACHER_TRAJECTORY_CACHE_DIR="${TEACHER_TRAJECTORY_CACHE_DIR:-/mnt/lanxiangh/data/ff_exec/teacher_trajectory_cache}"
TEACHER_TRAJECTORY_STEPS="${TEACHER_TRAJECTORY_STEPS:-5}"
TEACHER_TRAJECTORY_SOLVER="${TEACHER_TRAJECTORY_SOLVER:-unipc}"
TEACHER_TRAJECTORY_SHIFT="${TEACHER_TRAJECTORY_SHIFT:-}"
TEACHER_TRAJECTORY_DATASET_KEY="${TEACHER_TRAJECTORY_DATASET_KEY:-}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
VAL_FRACTION="${VAL_FRACTION:-0.05}"
OVERFIT_NUM_EXAMPLES="${OVERFIT_NUM_EXAMPLES:-0}"
OVERFIT_START_INDEX="${OVERFIT_START_INDEX:-0}"
CLEAN_LATENT_LOSS_WEIGHT="${CLEAN_LATENT_LOSS_WEIGHT:-1.0}"
FLOW_LOSS_WEIGHT="${FLOW_LOSS_WEIGHT:-0.25}"
DETAIL_LOSS_WEIGHT="${DETAIL_LOSS_WEIGHT:-0.0}"
TEMPORAL_DELTA_WEIGHT="${TEMPORAL_DELTA_WEIGHT:-0.0}"
BOUNDARY_WEIGHT="${BOUNDARY_WEIGHT:-0.0}"
DMD_LOSS_WEIGHT="${DMD_LOSS_WEIGHT:-0.0}"
DMD_FAKE_SCORE_LOSS_WEIGHT="${DMD_FAKE_SCORE_LOSS_WEIGHT:-1.0}"
DMD_WARMUP_STEPS="${DMD_WARMUP_STEPS:-0}"
DMD_STUDENT_UPDATE_FREQ="${DMD_STUDENT_UPDATE_FREQ:-5}"
DMD_FAKE_SCORE_LR="${DMD_FAKE_SCORE_LR:-1e-7}"
DMD_FAKE_SCORE_WEIGHT_DECAY="${DMD_FAKE_SCORE_WEIGHT_DECAY:-0.01}"
DMD_FAKE_SCORE_GRADIENT_CHECKPOINTING="${DMD_FAKE_SCORE_GRADIENT_CHECKPOINTING:-1}"
DMD_MODEL_NAME="${DMD_MODEL_NAME:-$TARGET_MODEL_NAME}"
DMD_TEACHER_MODEL_NAME="${DMD_TEACHER_MODEL_NAME:-$DMD_MODEL_NAME}"
DMD_FAKE_SCORE_MODEL_NAME="${DMD_FAKE_SCORE_MODEL_NAME:-$DMD_MODEL_NAME}"
DMD_TEACHER_CHECKPOINT_PATH="${DMD_TEACHER_CHECKPOINT_PATH:-}"
DMD_FAKE_SCORE_CHECKPOINT_PATH="${DMD_FAKE_SCORE_CHECKPOINT_PATH:-}"
DMD_GUIDANCE_SCALE="${DMD_GUIDANCE_SCALE:-5.0}"
DMD_TIMESTEP_SHIFT="${DMD_TIMESTEP_SHIFT:-5.0}"
DMD_MIN_TIMESTEP="${DMD_MIN_TIMESTEP:-20}"
DMD_MAX_TIMESTEP="${DMD_MAX_TIMESTEP:-980}"
DMD_TIME_DISTRIBUTION="${DMD_TIME_DISTRIBUTION:-shifted_uniform}"
DMD_LOGNORMAL_MEAN="${DMD_LOGNORMAL_MEAN:-0.0}"
DMD_LOGNORMAL_STD="${DMD_LOGNORMAL_STD:-1.6}"
DMD_FAKE_SCORE_WEIGHTING="${DMD_FAKE_SCORE_WEIGHTING:-scheduler_sigma}"
DMD_TARGET_CONVENTION="${DMD_TARGET_CONVENTION:-wan_rf_x0}"
DMD_CONSISTENCY_OBJECTIVE="${DMD_CONSISTENCY_OBJECTIVE:-none}"
DMD_CONSISTENCY_LOSS_WEIGHT="${DMD_CONSISTENCY_LOSS_WEIGHT:-0.0}"
DMD_CONSISTENCY_FD_EPSILON="${DMD_CONSISTENCY_FD_EPSILON:-1e-4}"
DMD_SCORE_SCOPE="${DMD_SCORE_SCOPE:-future}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
NUM_GPUS="${NUM_GPUS:-4}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-math}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: python not found or not executable: $PYTHON" >&2
  exit 1
fi
if [[ "$TEACHER_SETUP" != "bidirectional_wan" ]]; then
  echo "ERROR: TEACHER_SETUP must be bidirectional_wan for this launcher." >&2
  echo "This script is for Option B: bidirectional teacher -> bidirectional student." >&2
  exit 1
fi
if [[ "$MANIFEST_PATH" == *"draft_head_full_dataset_6layer_uniform"* || "$MANIFEST_PATH" == *"draft_head_full_dataset"* ]]; then
  echo "ERROR: MANIFEST_PATH points to the old AR/Krea draft-head dataset:" >&2
  echo "  $MANIFEST_PATH" >&2
  echo "Option B requires a dataset collected from original/non-causal Wan full-video teacher." >&2
  exit 1
fi
if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "ERROR: Option-B bidirectional Wan manifest missing: $MANIFEST_PATH" >&2
  echo "Generate/point MANIFEST_PATH to a dataset collected from original/non-causal Wan full-video teacher." >&2
  exit 1
fi
if [[ "$TARGET_CHECKPOINT_PATH" == *"realtime-video"* || "$TARGET_CHECKPOINT_PATH" == *"krea"* ]]; then
  echo "ERROR: TARGET_CHECKPOINT_PATH points to Krea/realtime causal checkpoint:" >&2
  echo "  $TARGET_CHECKPOINT_PATH" >&2
  echo "Option B should use original/non-causal Wan teacher weights under wan_models/$TARGET_MODEL_NAME." >&2
  exit 1
fi
if [[ ! -f "$TARGET_CHECKPOINT_PATH" ]]; then
  echo "ERROR: original Wan target checkpoint/index missing: $TARGET_CHECKPOINT_PATH" >&2
  exit 1
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && "$NUM_GPUS" -gt 1 ]]; then
  CUDA_DEVICE="$(seq -s, 0 $((NUM_GPUS - 1)))"
else
  CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
fi
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

tag_slug() {
  printf '%s' "$1" | tr ' /.,:' '_____'
}

LR_TAG="$(tag_slug "$LR")"
DELTA_TAG="$(tag_slug "$TEMPORAL_DELTA_WEIGHT")"
BOUNDARY_TAG="$(tag_slug "$BOUNDARY_WEIGHT")"
FLOW_TAG="$(tag_slug "$FLOW_LOSS_WEIGHT")"
DETAIL_TAG="$(tag_slug "$DETAIL_LOSS_WEIGHT")"
DMD_TAG="$(tag_slug "$DMD_LOSS_WEIGHT")"
if [[ "$ROLLOUT_SOLVER" == "unipc" && "$ROLLOUT_STEPS" != "0" ]]; then
  STEP_TAG="unipc${ROLLOUT_STEPS}_shift$(tag_slug "$ROLLOUT_SOLVER_SHIFT")"
elif [[ "$ROLLOUT_SOLVER" == "rcm" ]]; then
  EFFECTIVE_ROLLOUT_STEPS="$ROLLOUT_STEPS"
  if [[ "$EFFECTIVE_ROLLOUT_STEPS" == "0" ]]; then
    EFFECTIVE_ROLLOUT_STEPS="5"
  fi
  if [[ -n "$ROLLOUT_SCHEDULE" ]]; then
    STEP_TAG="rcm$(tag_slug "$ROLLOUT_SCHEDULE")"
  else
    STEP_TAG="rcm${EFFECTIVE_ROLLOUT_STEPS}_sigmamax$(tag_slug "$ROLLOUT_SIGMA_MAX")"
  fi
else
  STEP_TAG="$(tag_slug "$DENOISING_STEP_LIST")"
fi
DENSE_TAG="${DENSE_SCHEDULE_STEPS:-manual}"
OVERFIT_TAG="${OVERFIT_NUM_EXAMPLES:-0}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/outputs/draft_head/checkpoints}"
RUN_DIR="${RUN_DIR:-$CHECKPOINT_ROOT/${RUN_TIMESTAMP}_bidirectional_prompt_anchor_wan_targetinit_tempmix${TEMPORAL_MIXER_LAYERS}_${TRAINING_MODE}_h${HIDDEN_CHANNELS}_l${NUM_LAYERS}_st${STEP_TAG}_dense${DENSE_TAG}_overfit${OVERFIT_TAG}_bs${BATCH_SIZE}_g${NUM_GPUS}_lr${LR_TAG}_fl${FLOW_TAG}_dt${DETAIL_TAG}_td${DELTA_TAG}_bd${BOUNDARY_TAG}_dmd${DMD_TAG}}"
OUTPUT_PATH="${OUTPUT_PATH:-$RUN_DIR/final.pt}"

LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/launch_scripts/logs}"
mkdir -p "$LOG_DIR" "$(dirname "$OUTPUT_PATH")"
LOG_FILE="$LOG_DIR/$(basename "${BASH_SOURCE[0]}" .sh)_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }

cd "$PROJECT_ROOT"

log "Training bidirectional prompt-anchor draft head"
log "Teacher setup: $TEACHER_SETUP"
log "Manifest: $MANIFEST_PATH"
log "Dataset cache: $DATASET_CACHE_DIR index_workers=$DATASET_INDEX_WORKERS wait_seconds=$DATASET_CACHE_WAIT_SECONDS"
log "Run dir:   $RUN_DIR"
log "Output:    $OUTPUT_PATH"
log "GPUs:      $NUM_GPUS visible=$CUDA_VISIBLE_DEVICES"
log "Anchor:    conditioning=$ANCHOR_CONDITIONING online_target target=$TARGET_MODEL_NAME seed=$ANCHOR_NOISE_SEED"
log "Model:     wan hidden=$HIDDEN_CHANNELS layers=$NUM_LAYERS heads=$NUM_HEADS ffn_dim=$FFN_DIM prompt_dim=$PROMPT_DIM temporal_mixer_layers=$TEMPORAL_MIXER_LAYERS temporal_mixer_ffn_dim=$TEMPORAL_MIXER_FFN_DIM gradient_checkpointing=$GRADIENT_CHECKPOINTING"
log "Parallel:  strategy=$PARALLEL_STRATEGY fsdp_min_num_params=$FSDP_MIN_NUM_PARAMS fsdp_mixed_precision=$FSDP_MIXED_PRECISION"
log "Attention: backend=$ATTENTION_BACKEND"
log "Init:      model=${INIT_MODEL_NAME:-$TARGET_MODEL_NAME} target_blocks=[$INIT_TARGET_BLOCKS] draft_head_ckpt=${INIT_DRAFT_HEAD_CHECKPOINT_PATH:-none}"
log "Training:  mode=$TRAINING_MODE anchor_conditioning=$ANCHOR_CONDITIONING prediction_type=$PREDICTION_TYPE euler_steps=[$DENOISING_STEP_LIST] dense_schedule_steps=${DENSE_SCHEDULE_STEPS:-off} random_sampling=$RANDOM_TIMESTEP_SAMPLING logit_mean=$LOGIT_NORMAL_MEAN logit_std=$LOGIT_NORMAL_STD unroll_noise=$UNROLL_NOISE_MODE rollout_solver=$ROLLOUT_SOLVER rollout_steps=$ROLLOUT_STEPS rollout_schedule=${ROLLOUT_SCHEDULE:-default} rollout_sigma_max=$ROLLOUT_SIGMA_MAX rollout_shift=$ROLLOUT_SOLVER_SHIFT weights=${UNROLL_STEP_WEIGHTS:-auto} teacher_traj_steps=$TEACHER_TRAJECTORY_STEPS teacher_traj_solver=$TEACHER_TRAJECTORY_SOLVER teacher_traj_shift=${TEACHER_TRAJECTORY_SHIFT:-legacy} teacher_traj_dataset_key=${TEACHER_TRAJECTORY_DATASET_KEY:-auto} teacher_traj_cache=$TEACHER_TRAJECTORY_CACHE_DIR overfit_num=$OVERFIT_NUM_EXAMPLES overfit_start=$OVERFIT_START_INDEX num_workers=$NUM_WORKERS"
log "Loss:      clean=$CLEAN_LATENT_LOSS_WEIGHT flow=$FLOW_LOSS_WEIGHT detail=$DETAIL_LOSS_WEIGHT temporal_delta=$TEMPORAL_DELTA_WEIGHT boundary=$BOUNDARY_WEIGHT dmd=$DMD_LOSS_WEIGHT dmd_fake=$DMD_FAKE_SCORE_LOSS_WEIGHT"
log "DMD:       teacher_model=$DMD_TEACHER_MODEL_NAME fake_score_model=$DMD_FAKE_SCORE_MODEL_NAME teacher_ckpt=${DMD_TEACHER_CHECKPOINT_PATH:-pretrained-dir} fake_ckpt=${DMD_FAKE_SCORE_CHECKPOINT_PATH:-teacher/pretrained-init} guidance=$DMD_GUIDANCE_SCALE time_dist=$DMD_TIME_DISTRIBUTION shift=$DMD_TIMESTEP_SHIFT lognormal_mean=$DMD_LOGNORMAL_MEAN lognormal_std=$DMD_LOGNORMAL_STD fake_weighting=$DMD_FAKE_SCORE_WEIGHTING target_convention=$DMD_TARGET_CONVENTION consistency=$DMD_CONSISTENCY_OBJECTIVE consistency_weight=$DMD_CONSISTENCY_LOSS_WEIGHT fd_eps=$DMD_CONSISTENCY_FD_EPSILON t=[$DMD_MIN_TIMESTEP,$DMD_MAX_TIMESTEP] scope=$DMD_SCORE_SCOPE warmup=$DMD_WARMUP_STEPS student_update_freq=$DMD_STUDENT_UPDATE_FREQ fake_lr=$DMD_FAKE_SCORE_LR fake_gc=$DMD_FAKE_SCORE_GRADIENT_CHECKPOINTING"

RUNNER=("$PYTHON")
if [[ "$NUM_GPUS" -gt 1 ]]; then
  RUNNER=("$PYTHON" -m torch.distributed.run --standalone --max_restarts 0 --nproc_per_node "$NUM_GPUS")
fi

UNROLL_WEIGHT_ARGS=()
if [[ -n "$UNROLL_STEP_WEIGHTS" ]]; then
  # shellcheck disable=SC2206
  UNROLL_WEIGHT_ARRAY=($UNROLL_STEP_WEIGHTS)
  UNROLL_WEIGHT_ARGS+=(--unroll_step_weights "${UNROLL_WEIGHT_ARRAY[@]}")
fi
DENSE_SCHEDULE_ARGS=()
if [[ -n "$DENSE_SCHEDULE_STEPS" ]]; then
  DENSE_SCHEDULE_ARGS+=(--dense_schedule_steps "$DENSE_SCHEDULE_STEPS")
fi
MEMORY_ARGS=()
if [[ "$GRADIENT_CHECKPOINTING" == "1" || "$GRADIENT_CHECKPOINTING" == "true" ]]; then
  MEMORY_ARGS+=(--gradient_checkpointing)
fi
INIT_MODEL_ARGS=()
if [[ -n "$INIT_MODEL_NAME" ]]; then
  INIT_MODEL_ARGS+=(--init_model_name "$INIT_MODEL_NAME")
fi
if [[ -n "$INIT_DRAFT_HEAD_CHECKPOINT_PATH" ]]; then
  INIT_MODEL_ARGS+=(--init_draft_head_checkpoint_path "$INIT_DRAFT_HEAD_CHECKPOINT_PATH")
fi
INIT_TARGET_BLOCK_ARRAY=($INIT_TARGET_BLOCKS)
TEACHER_TRAJECTORY_SHIFT_ARGS=()
if [[ -n "$TEACHER_TRAJECTORY_SHIFT" ]]; then
  TEACHER_TRAJECTORY_SHIFT_ARGS+=(--teacher_trajectory_shift "$TEACHER_TRAJECTORY_SHIFT")
fi
TEACHER_TRAJECTORY_DATASET_ARGS=()
if [[ -n "$TEACHER_TRAJECTORY_DATASET_KEY" ]]; then
  TEACHER_TRAJECTORY_DATASET_ARGS+=(--teacher_trajectory_dataset_key "$TEACHER_TRAJECTORY_DATASET_KEY")
fi
DMD_MEMORY_ARGS=()
if [[ "$DMD_FAKE_SCORE_GRADIENT_CHECKPOINTING" == "1" || "$DMD_FAKE_SCORE_GRADIENT_CHECKPOINTING" == "true" ]]; then
  DMD_MEMORY_ARGS+=(--dmd_fake_score_gradient_checkpointing)
fi

"${RUNNER[@]}" train_bidirectional_draft_head.py \
  --manifest_path "$MANIFEST_PATH" \
  --output_path "$OUTPUT_PATH" \
  --dataset_cache_dir "$DATASET_CACHE_DIR" \
  --dataset_index_workers "$DATASET_INDEX_WORKERS" \
  --dataset_cache_wait_seconds "$DATASET_CACHE_WAIT_SECONDS" \
  --model_root "$MODEL_ROOT" \
  --config_path "$CONFIG_PATH" \
  --target_model_name "$TARGET_MODEL_NAME" \
  --target_checkpoint_path "$TARGET_CHECKPOINT_PATH" \
  "${INIT_MODEL_ARGS[@]}" \
  --anchor_noise_seed "$ANCHOR_NOISE_SEED" \
  --num_blocks "$NUM_BLOCKS" \
  --hidden_channels "$HIDDEN_CHANNELS" \
  --num_layers "$NUM_LAYERS" \
  --num_heads "$NUM_HEADS" \
  --ffn_dim "$FFN_DIM" \
  --temporal_mixer_layers "$TEMPORAL_MIXER_LAYERS" \
  --temporal_mixer_ffn_dim "$TEMPORAL_MIXER_FFN_DIM" \
  --init_target_blocks "${INIT_TARGET_BLOCK_ARRAY[@]}" \
  --prompt_dim "$PROMPT_DIM" \
  "${MEMORY_ARGS[@]}" \
  --parallel_strategy "$PARALLEL_STRATEGY" \
  --fsdp_min_num_params "$FSDP_MIN_NUM_PARAMS" \
  --fsdp_mixed_precision "$FSDP_MIXED_PRECISION" \
  --attention_backend "$ATTENTION_BACKEND" \
  --denoising_step_list $DENOISING_STEP_LIST \
  "${DENSE_SCHEDULE_ARGS[@]}" \
  --timestep_shift "$TIMESTEP_SHIFT" \
  --prediction_type "$PREDICTION_TYPE" \
  --training_mode "$TRAINING_MODE" \
  --anchor_conditioning "$ANCHOR_CONDITIONING" \
  --random_timestep_sampling "$RANDOM_TIMESTEP_SAMPLING" \
  --logit_normal_mean "$LOGIT_NORMAL_MEAN" \
  --logit_normal_std "$LOGIT_NORMAL_STD" \
  --unroll_noise_mode "$UNROLL_NOISE_MODE" \
  --rollout_solver "$ROLLOUT_SOLVER" \
  --rollout_steps "$ROLLOUT_STEPS" \
  --rollout_schedule "$ROLLOUT_SCHEDULE" \
  --rollout_sigma_max "$ROLLOUT_SIGMA_MAX" \
  --rollout_solver_shift "$ROLLOUT_SOLVER_SHIFT" \
  --teacher_trajectory_cache_dir "$TEACHER_TRAJECTORY_CACHE_DIR" \
  --teacher_trajectory_steps "$TEACHER_TRAJECTORY_STEPS" \
  --teacher_trajectory_solver "$TEACHER_TRAJECTORY_SOLVER" \
  "${TEACHER_TRAJECTORY_SHIFT_ARGS[@]}" \
  "${TEACHER_TRAJECTORY_DATASET_ARGS[@]}" \
  "${UNROLL_WEIGHT_ARGS[@]}" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --val_fraction "$VAL_FRACTION" \
  --overfit_num_examples "$OVERFIT_NUM_EXAMPLES" \
  --overfit_start_index "$OVERFIT_START_INDEX" \
  --clean_latent_loss_weight "$CLEAN_LATENT_LOSS_WEIGHT" \
  --flow_loss_weight "$FLOW_LOSS_WEIGHT" \
  --detail_loss_weight "$DETAIL_LOSS_WEIGHT" \
  --temporal_delta_weight "$TEMPORAL_DELTA_WEIGHT" \
  --boundary_weight "$BOUNDARY_WEIGHT" \
  --dmd_loss_weight "$DMD_LOSS_WEIGHT" \
  --dmd_fake_score_loss_weight "$DMD_FAKE_SCORE_LOSS_WEIGHT" \
  --dmd_warmup_steps "$DMD_WARMUP_STEPS" \
  --dmd_student_update_freq "$DMD_STUDENT_UPDATE_FREQ" \
  --dmd_fake_score_lr "$DMD_FAKE_SCORE_LR" \
  --dmd_fake_score_weight_decay "$DMD_FAKE_SCORE_WEIGHT_DECAY" \
  "${DMD_MEMORY_ARGS[@]}" \
  --dmd_model_name "$DMD_MODEL_NAME" \
  --dmd_teacher_model_name "$DMD_TEACHER_MODEL_NAME" \
  --dmd_fake_score_model_name "$DMD_FAKE_SCORE_MODEL_NAME" \
  --dmd_teacher_checkpoint_path "$DMD_TEACHER_CHECKPOINT_PATH" \
  --dmd_fake_score_checkpoint_path "$DMD_FAKE_SCORE_CHECKPOINT_PATH" \
  --dmd_guidance_scale "$DMD_GUIDANCE_SCALE" \
  --dmd_timestep_shift "$DMD_TIMESTEP_SHIFT" \
  --dmd_min_timestep "$DMD_MIN_TIMESTEP" \
  --dmd_max_timestep "$DMD_MAX_TIMESTEP" \
  --dmd_time_distribution "$DMD_TIME_DISTRIBUTION" \
  --dmd_lognormal_mean "$DMD_LOGNORMAL_MEAN" \
  --dmd_lognormal_std "$DMD_LOGNORMAL_STD" \
  --dmd_fake_score_weighting "$DMD_FAKE_SCORE_WEIGHTING" \
  --dmd_target_convention "$DMD_TARGET_CONVENTION" \
  --dmd_consistency_objective "$DMD_CONSISTENCY_OBJECTIVE" \
  --dmd_consistency_loss_weight "$DMD_CONSISTENCY_LOSS_WEIGHT" \
  --dmd_consistency_fd_epsilon "$DMD_CONSISTENCY_FD_EPSILON" \
  --dmd_score_scope "$DMD_SCORE_SCOPE" \
  --amp_dtype "$AMP_DTYPE" \
  2>&1 | tee -a "$LOG_FILE"

log "Bidirectional checkpoint: $OUTPUT_PATH"
