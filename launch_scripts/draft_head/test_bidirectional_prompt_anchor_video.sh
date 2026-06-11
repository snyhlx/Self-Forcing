#!/usr/bin/env bash
# Generate MP4s from a bidirectional prompt-anchor draft-head checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="${SF_VENV:-$PROJECT_ROOT/sf_venv}"
PYTHON="$VENV_DIR/bin/python"

TEACHER_SETUP="${TEACHER_SETUP:-bidirectional_wan}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/lanxiangh/models}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/self_forcing_dmd.yaml}"
TARGET_MODEL_NAME="${TARGET_MODEL_NAME:-Wan2.1-T2V-14B}"
TARGET_CHECKPOINT_PATH="${TARGET_CHECKPOINT_PATH:-$MODEL_ROOT/wan_models/$TARGET_MODEL_NAME/diffusion_pytorch_model.safetensors.index.json}"
DRAFT_HEAD_CHECKPOINT_PATH="${DRAFT_HEAD_CHECKPOINT_PATH:-$PROJECT_ROOT/outputs/draft_head/checkpoints/20260606_142446_bidirectional_prompt_anchor_wan_targetinit_unrolled_h5120_l6_st1000_750_500_250_0_bs1_g4_lr5e-4_fl0_25_td0_0_bd0_0/epoch_0002.pt}"

OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/draft_head/test_runs/bidirectional_prompt_anchor_$(date +%Y%m%d_%H%M%S)}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
NUM_BLOCKS="${NUM_BLOCKS:-9}"
MAX_PROMPTS="${MAX_PROMPTS:-1}"
START_INDEX="${START_INDEX:-0}"
PROMPT_FILE="${PROMPT_FILE:-}"
PROMPT="${PROMPT:-A hyperrealistic close-up of ocean waves shimmering at sunset.}"
SEED="${SEED:-42}"
FPS="${FPS:-16}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
TARGET_REFINE_TIMESTEP="${TARGET_REFINE_TIMESTEP:-0}"
SAVE_RAW_VIDEO="${SAVE_RAW_VIDEO:-0}"
SAVE_TARGET_VIDEO="${SAVE_TARGET_VIDEO:-0}"
TRAINING_MODE="${TRAINING_MODE:-}"
PREDICTION_TYPE="${PREDICTION_TYPE:-}"
ANCHOR_CONDITIONING="${ANCHOR_CONDITIONING:-}"
UNROLL_NOISE_MODE="${UNROLL_NOISE_MODE:-}"
DENOISING_STEP_LIST="${DENOISING_STEP_LIST:-}"
HEAD_SAMPLING_STEPS="${HEAD_SAMPLING_STEPS:-}"
HEAD_SOLVER="${HEAD_SOLVER:-}"
HEAD_SOLVER_SHIFT="${HEAD_SOLVER_SHIFT:-}"
TEACHER_SAMPLING_STEPS="${TEACHER_SAMPLING_STEPS:-}"
VIDEO_MANIFEST_PATH="${VIDEO_MANIFEST_PATH:-}"
VIDEO_DATASET_INDEX="${VIDEO_DATASET_INDEX:-}"
VIDEO_PROMPT_INDEX="${VIDEO_PROMPT_INDEX:-}"
VIDEO_SPLIT="${VIDEO_SPLIT:-all}"
VIDEO_SPLIT_INDEX="${VIDEO_SPLIT_INDEX:-0}"

LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/launch_scripts/logs}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
LOG_FILE="$LOG_DIR/$(basename "${BASH_SOURCE[0]}" .sh)_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }

cd "$PROJECT_ROOT"

for path in "$PYTHON" "$TARGET_CHECKPOINT_PATH" "$DRAFT_HEAD_CHECKPOINT_PATH" "$MODEL_ROOT/wan_models/Wan2.1-T2V-14B"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: required path missing: $path" >&2
    exit 1
  fi
done
if [[ "$TEACHER_SETUP" != "bidirectional_wan" ]]; then
  echo "ERROR: TEACHER_SETUP must be bidirectional_wan for this launcher." >&2
  echo "This script is for Option B: bidirectional teacher -> bidirectional student." >&2
  exit 1
fi
if [[ "$TARGET_CHECKPOINT_PATH" == *"realtime-video"* || "$TARGET_CHECKPOINT_PATH" == *"krea"* ]]; then
  echo "ERROR: TARGET_CHECKPOINT_PATH points to Krea/realtime causal checkpoint:" >&2
  echo "  $TARGET_CHECKPOINT_PATH" >&2
  echo "Option B should use original/non-causal Wan teacher weights under wan_models/$TARGET_MODEL_NAME." >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

PROMPT_ARGS=(--prompt "$PROMPT")
if [[ -n "$PROMPT_FILE" ]]; then
  PROMPT_ARGS=(--prompt_file "$PROMPT_FILE" --start_index "$START_INDEX" --max_prompts "$MAX_PROMPTS")
fi

OPTIONAL_ARGS=()
if [[ -n "$TRAINING_MODE" ]]; then
  OPTIONAL_ARGS+=(--training_mode "$TRAINING_MODE")
fi
if [[ -n "$PREDICTION_TYPE" ]]; then
  OPTIONAL_ARGS+=(--prediction_type "$PREDICTION_TYPE")
fi
if [[ -n "$ANCHOR_CONDITIONING" ]]; then
  OPTIONAL_ARGS+=(--anchor_conditioning "$ANCHOR_CONDITIONING")
fi
if [[ -n "$UNROLL_NOISE_MODE" ]]; then
  OPTIONAL_ARGS+=(--unroll_noise_mode "$UNROLL_NOISE_MODE")
fi
if [[ -n "$DENOISING_STEP_LIST" ]]; then
  # shellcheck disable=SC2206
  DENOISING_STEP_ARRAY=($DENOISING_STEP_LIST)
  OPTIONAL_ARGS+=(--denoising_step_list "${DENOISING_STEP_ARRAY[@]}")
fi
if [[ -n "$HEAD_SAMPLING_STEPS" ]]; then
  OPTIONAL_ARGS+=(--head_sampling_steps "$HEAD_SAMPLING_STEPS")
fi
if [[ -n "$HEAD_SOLVER" ]]; then
  OPTIONAL_ARGS+=(--head_solver "$HEAD_SOLVER")
fi
if [[ -n "$HEAD_SOLVER_SHIFT" ]]; then
  OPTIONAL_ARGS+=(--head_solver_shift "$HEAD_SOLVER_SHIFT")
fi
if [[ -n "$TEACHER_SAMPLING_STEPS" ]]; then
  OPTIONAL_ARGS+=(--teacher_sampling_steps "$TEACHER_SAMPLING_STEPS")
fi
if [[ -n "$VIDEO_MANIFEST_PATH" ]]; then
  OPTIONAL_ARGS+=(--video_manifest_path "$VIDEO_MANIFEST_PATH")
fi
if [[ -n "$VIDEO_DATASET_INDEX" ]]; then
  OPTIONAL_ARGS+=(--video_dataset_index "$VIDEO_DATASET_INDEX")
fi
if [[ -n "$VIDEO_PROMPT_INDEX" ]]; then
  OPTIONAL_ARGS+=(--video_prompt_index "$VIDEO_PROMPT_INDEX")
fi
if [[ -n "$VIDEO_MANIFEST_PATH" ]]; then
  OPTIONAL_ARGS+=(--video_split "$VIDEO_SPLIT" --video_split_index "$VIDEO_SPLIT_INDEX")
fi
if [[ "$TARGET_REFINE_TIMESTEP" -gt 0 ]]; then
  OPTIONAL_ARGS+=(--target_refine_timestep "$TARGET_REFINE_TIMESTEP")
fi
if [[ "$SAVE_RAW_VIDEO" == "1" || "$SAVE_RAW_VIDEO" == "true" ]]; then
  OPTIONAL_ARGS+=(--save_raw_video)
fi
if [[ "$SAVE_TARGET_VIDEO" == "1" || "$SAVE_TARGET_VIDEO" == "true" ]]; then
  OPTIONAL_ARGS+=(--save_target_video)
fi

log "Bidirectional prompt-anchor video diagnostic"
log "Checkpoint: $DRAFT_HEAD_CHECKPOINT_PATH"
log "Output:     $OUTPUT_DIR"
log "Blocks:     $NUM_BLOCKS"
log "Prompts:    ${PROMPT_FILE:-single prompt} start=$START_INDEX max=$MAX_PROMPTS"
log "Manifest:   ${VIDEO_MANIFEST_PATH:-off} dataset_index=${VIDEO_DATASET_INDEX:-default} prompt_index=${VIDEO_PROMPT_INDEX:-off} split=$VIDEO_SPLIT split_index=$VIDEO_SPLIT_INDEX"
log "Prediction: ${PREDICTION_TYPE:-checkpoint/default}"
log "Anchor conditioning: ${ANCHOR_CONDITIONING:-checkpoint/default}"
log "Head steps: ${HEAD_SAMPLING_STEPS:-checkpoint/default}"
log "Head solver: ${HEAD_SOLVER:-euler} shift=${HEAD_SOLVER_SHIFT:-8.0}"
log "Teacher steps: ${TEACHER_SAMPLING_STEPS:-default}"
log "Refine:     target_timestep=$TARGET_REFINE_TIMESTEP save_raw=$SAVE_RAW_VIDEO save_target=$SAVE_TARGET_VIDEO"

"$PYTHON" eval_bidirectional_draft_head.py \
  --checkpoint_path "$DRAFT_HEAD_CHECKPOINT_PATH" \
  --video_output_dir "$OUTPUT_DIR" \
  --model_root "$MODEL_ROOT" \
  --config_path "$CONFIG_PATH" \
  --teacher_setup "$TEACHER_SETUP" \
  --target_model_name "$TARGET_MODEL_NAME" \
  --target_checkpoint_path "$TARGET_CHECKPOINT_PATH" \
  --num_blocks "$NUM_BLOCKS" \
  --anchor_noise_seed "$SEED" \
  --amp_dtype "$AMP_DTYPE" \
  --fps "$FPS" \
  "${OPTIONAL_ARGS[@]}" \
  "${PROMPT_ARGS[@]}" \
  2>&1 | tee -a "$LOG_FILE"

log "Done. Profile: $OUTPUT_DIR/profile.json"
