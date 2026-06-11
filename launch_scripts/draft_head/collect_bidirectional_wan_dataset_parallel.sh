#!/usr/bin/env bash
# Collect Option-B full-video original-Wan latents in parallel and consolidate one manifest.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="${SF_VENV:-$PROJECT_ROOT/sf_venv}"
PYTHON="$VENV_DIR/bin/python"

MODEL_ROOT="${MODEL_ROOT:-/mnt/lanxiangh/models}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/self_forcing_dmd.yaml}"
TARGET_MODEL_NAME="${TARGET_MODEL_NAME:-Wan2.1-T2V-14B}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/lanxiangh/data/ff_exec/bidirectional_wan_draft_head_dataset/tau_delta_0p0}"

NUM_GPUS="${NUM_GPUS:-4}"
CUDA_DEVICES="${CUDA_DEVICES:-0 1 2 3}"
NUM_BLOCKS="${NUM_BLOCKS:-9}"
PROMPT_FILE="${PROMPT_FILE:-}"
PROMPT="${PROMPT:-A hyperrealistic close-up of ocean waves shimmering at sunset.}"
START_INDEX="${START_INDEX:-0}"
MAX_PROMPTS="${MAX_PROMPTS:-1}"
SEED="${SEED:-42}"
SHARD_SIZE="${SHARD_SIZE:-16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SAMPLING_STEPS="${SAMPLING_STEPS:-50}"
SAMPLE_SOLVER="${SAMPLE_SOLVER:-unipc}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"
MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-60}"

LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/launch_scripts/logs}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
LOG_FILE="$LOG_DIR/$(basename "${BASH_SOURCE[0]}" .sh)_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }

cd "$PROJECT_ROOT"

for path in "$PYTHON" "$MODEL_ROOT/wan_models/$TARGET_MODEL_NAME"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: required path missing: $path" >&2
    exit 1
  fi
done

# shellcheck disable=SC2206
DEVICE_ARRAY=($CUDA_DEVICES)
if [[ "${#DEVICE_ARRAY[@]}" -lt "$NUM_GPUS" ]]; then
  echo "ERROR: CUDA_DEVICES must provide at least NUM_GPUS entries." >&2
  exit 1
fi

PROMPT_ARGS=(--prompt "$PROMPT")
if [[ -n "$PROMPT_FILE" ]]; then
  PROMPT_ARGS=(--prompt_file "$PROMPT_FILE" --start_index "$START_INDEX" --max_prompts "$MAX_PROMPTS")
fi

log "Parallel Option-B bidirectional Wan dataset collection"
log "Output:  $OUTPUT_DIR"
log "Teacher: $TARGET_MODEL_NAME"
log "GPUs:    $CUDA_DEVICES num=$NUM_GPUS"
log "Blocks:  $NUM_BLOCKS"
log "Prompts: ${PROMPT_FILE:-single prompt} start=$START_INDEX max=$MAX_PROMPTS"

PIDS=()
PART_DIRS=()
PROGRESS_PATHS=()
for ((rank = 0; rank < NUM_GPUS; rank++)); do
  device="${DEVICE_ARRAY[$rank]}"
  part_dir="$OUTPUT_DIR/part_$(printf '%03d' "$rank")"
  part_log="$LOG_DIR/$(basename "${BASH_SOURCE[0]}" .sh)_part$(printf '%03d' "$rank")_$(date +%Y%m%d_%H%M%S).log"
  progress_path="$OUTPUT_DIR/progress_part_$(printf '%03d' "$rank").json"
  mkdir -p "$part_dir"
  PART_DIRS+=("$part_dir")
  PROGRESS_PATHS+=("$progress_path")
  rm -f "$progress_path"
  log "Starting collector rank=$rank device=$device part=$part_dir"
  (
    export CUDA_VISIBLE_DEVICES="$device"
    "$PYTHON" collect_bidirectional_wan_dataset.py \
      --output_dir "$part_dir" \
      --model_root "$MODEL_ROOT" \
      --config_path "$CONFIG_PATH" \
      --target_model_name "$TARGET_MODEL_NAME" \
      --num_blocks "$NUM_BLOCKS" \
      --seed "$SEED" \
      --shard_size "$SHARD_SIZE" \
      --batch_size "$BATCH_SIZE" \
      --progress_path "$progress_path" \
      --sampling_steps "$SAMPLING_STEPS" \
      --sample_solver "$SAMPLE_SOLVER" \
      --guidance_scale "$GUIDANCE_SCALE" \
      --num_prompt_shards "$NUM_GPUS" \
      --prompt_shard_index "$rank" \
      "${PROMPT_ARGS[@]}"
  ) >"$part_log" 2>&1 &
  PIDS+=("$!")
done

while true; do
  running=0
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      running=1
      break
    fi
  done
  total_done=0
  total_expected=0
  status_parts=()
  for ((rank = 0; rank < NUM_GPUS; rank++)); do
    progress_path="${PROGRESS_PATHS[$rank]}"
    if [[ -f "$progress_path" ]]; then
      progress_values="$("$PYTHON" - "$progress_path" <<'PY' || true
import json
import sys
try:
    data = json.loads(open(sys.argv[1], encoding="utf-8").read())
    print(data.get("done", 0), data.get("total", 0))
except Exception:
    print("0 0")
PY
)"
      read -r done expected <<<"$progress_values"
      total_done=$((total_done + done))
      total_expected=$((total_expected + expected))
      status_parts+=("r${rank}:${done}/${expected}")
    else
      status_parts+=("r${rank}:starting")
    fi
  done
  if [[ "$total_expected" -gt 0 ]]; then
    log "Progress: ${total_done}/${total_expected} prompts (${status_parts[*]})"
  else
    log "Progress: workers starting (${status_parts[*]})"
  fi
  if [[ "$running" -eq 0 ]]; then
    break
  fi
  sleep "$MONITOR_INTERVAL_SECONDS"
done

failed=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "ERROR: at least one collection worker failed. Check part logs in $LOG_DIR." >&2
  exit 1
fi

log "All collectors finished; consolidating manifests"
"$PYTHON" consolidate_bidirectional_wan_dataset.py \
  --output_dir "$OUTPUT_DIR" \
  --part_dirs "${PART_DIRS[@]}" \
  2>&1 | tee -a "$LOG_FILE"

log "Done. Consolidated manifest: $OUTPUT_DIR/manifest.json"
