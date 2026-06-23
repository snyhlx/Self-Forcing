#!/usr/bin/env bash
# Collect AR teacher trajectories from the Krea realtime 14B checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="${SF_VENV:-$PROJECT_ROOT/sf_venv}"
PYTHON="$VENV_DIR/bin/python"

MODEL_ROOT="${MODEL_ROOT:-/mnt/lanxiangh/models}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/self_forcing_dmd.yaml}"
PROMPT_FILE="${PROMPT_FILE:-}"
PROMPT_MANIFEST_PATH="${PROMPT_MANIFEST_PATH:-/mnt/lanxiangh/data/ff_exec/bidirectional_wan_draft_head_dataset/tau_delta_0p0_50steps_legacy/manifest.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/lanxiangh/data/ff_exec/ar_teacher_trajectory_cache/krea14b_unipc5_shift8_$(date +%Y%m%d_%H%M%S)}"

# This is the Wan architecture name used by the wrapper. The actual AR teacher
# weights are Krea realtime 14B via KREA_CHECKPOINT_PATH.
TARGET_MODEL_NAME="${TARGET_MODEL_NAME:-Wan2.1-T2V-14B}"
KREA_CHECKPOINT_PATH="${KREA_CHECKPOINT_PATH:-$MODEL_ROOT/realtime-video/checkpoints/krea-realtime-video-14b.safetensors}"

NUM_GPUS="${NUM_GPUS:-8}"
START_INDEX="${START_INDEX:-0}"
MAX_PROMPTS="${MAX_PROMPTS:-800}"
NUM_BLOCKS="${NUM_BLOCKS:-7}"
COLLECT_START_BLOCK="${COLLECT_START_BLOCK:-1}"
DENOISING_STEP_LIST="${DENOISING_STEP_LIST:-999 969 922 841 666}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.0}"
SEED="${SEED:-42}"

LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: python not found or not executable: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$KREA_CHECKPOINT_PATH" ]]; then
  echo "ERROR: Krea realtime checkpoint missing: $KREA_CHECKPOINT_PATH" >&2
  exit 1
fi
if [[ -n "$PROMPT_FILE" && ! -f "$PROMPT_FILE" ]]; then
  echo "ERROR: prompt file missing: $PROMPT_FILE" >&2
  exit 1
fi
if [[ -z "$PROMPT_FILE" && ! -f "$PROMPT_MANIFEST_PATH" ]]; then
  echo "ERROR: prompt manifest missing: $PROMPT_MANIFEST_PATH" >&2
  exit 1
fi

cd "$PROJECT_ROOT"

echo "Collecting AR teacher trajectories"
echo "Output root: $OUTPUT_ROOT"
echo "Teacher architecture: $TARGET_MODEL_NAME"
echo "Krea checkpoint: $KREA_CHECKPOINT_PATH"
echo "Prompt file: ${PROMPT_FILE:-disabled}"
echo "Prompt manifest: ${PROMPT_MANIFEST_PATH:-disabled}"
echo "Prompt selection: start=$START_INDEX max=$MAX_PROMPTS shards=$NUM_GPUS"
echo "Blocks: num_blocks=$NUM_BLOCKS collect_start_block=$COLLECT_START_BLOCK"
echo "Schedule: [$DENOISING_STEP_LIST] guidance=$GUIDANCE_SCALE seed=$SEED"

PROMPT_ARGS=()
if [[ -n "$PROMPT_FILE" ]]; then
  PROMPT_ARGS+=(--prompt_file "$PROMPT_FILE")
else
  PROMPT_ARGS+=(--prompt_manifest_path "$PROMPT_MANIFEST_PATH")
fi

pids=()
for ((shard_index = 0; shard_index < NUM_GPUS; shard_index++)); do
  CUDA_VISIBLE_DEVICES="$shard_index" "$PYTHON" collect_ar_teacher_trajectory.py \
    --output_dir "$OUTPUT_ROOT/shard_$shard_index" \
    --model_root "$MODEL_ROOT" \
    --config_path "$CONFIG_PATH" \
    --target_model_name "$TARGET_MODEL_NAME" \
    --target_checkpoint_path "$KREA_CHECKPOINT_PATH" \
    "${PROMPT_ARGS[@]}" \
    --start_index "$START_INDEX" \
    --max_prompts "$MAX_PROMPTS" \
    --num_prompt_shards "$NUM_GPUS" \
    --prompt_shard_index "$shard_index" \
    --num_blocks "$NUM_BLOCKS" \
    --collect_start_block "$COLLECT_START_BLOCK" \
    --denoising_step_list "$DENOISING_STEP_LIST" \
    --guidance_scale "$GUIDANCE_SCALE" \
    --seed "$SEED" \
    --progress_path "$OUTPUT_ROOT/progress_$shard_index.json" \
    > "$LOG_DIR/shard_$shard_index.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid_index in "${!pids[@]}"; do
  if ! wait "${pids[$pid_index]}"; then
    echo "ERROR: shard_$pid_index failed; see $LOG_DIR/shard_$pid_index.log" >&2
    failed=1
  fi
done
if [[ "$failed" != "0" ]]; then
  exit 1
fi
echo "Done: $OUTPUT_ROOT"
echo "Shard manifests:"
ls "$OUTPUT_ROOT"/shard_*/manifest.json
