#!/usr/bin/env bash
# Offline precompute compact teacher trajectory cache for Option-B training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="${SF_VENV:-$PROJECT_ROOT/sf_venv}"
PYTHON="$VENV_DIR/bin/python"

MANIFEST_PATH="${MANIFEST_PATH:-/mnt/lanxiangh/data/ff_exec/bidirectional_wan_draft_head_dataset/tau_delta_0p0/manifest.json}"
CACHE_DIR="${TEACHER_TRAJECTORY_CACHE_DIR:-/mnt/lanxiangh/data/ff_exec/teacher_trajectory_cache}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/mnt/lanxiangh/data/cache/specgen}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/lanxiangh/models}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/self_forcing_dmd.yaml}"
TARGET_MODEL_NAME="${TARGET_MODEL_NAME:-Wan2.1-T2V-14B}"
NUM_BLOCKS="${NUM_BLOCKS:-9}"
SEED="${SEED:-42}"
TRAJECTORY_STEPS="${TEACHER_TRAJECTORY_STEPS:-5}"
TRAJECTORY_SOLVER="${TEACHER_TRAJECTORY_SOLVER:-unipc}"
SPLIT="${SPLIT:-all}"
SPLIT_INDEX_START="${SPLIT_INDEX_START:-0}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
VAL_FRACTION="${VAL_FRACTION:-0.05}"
NUM_GPUS="${NUM_GPUS:-4}"
CUDA_DEVICES="${CUDA_DEVICES:-0 1 2 3}"
DATASET_INDEX_WORKERS="${DATASET_INDEX_WORKERS:-8}"
DATASET_CACHE_WAIT_SECONDS="${DATASET_CACHE_WAIT_SECONDS:-3600}"

LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/launch_scripts/logs}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/precompute_teacher_trajectory_cache_parallel_${RUN_TIMESTAMP}.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: python not found or not executable: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "ERROR: manifest missing: $MANIFEST_PATH" >&2
  exit 1
fi

# shellcheck disable=SC2206
DEVICE_ARRAY=($CUDA_DEVICES)
if [[ "${#DEVICE_ARRAY[@]}" -lt "$NUM_GPUS" ]]; then
  echo "ERROR: CUDA_DEVICES must provide at least NUM_GPUS entries." >&2
  exit 1
fi

cd "$PROJECT_ROOT"
log "Precomputing teacher trajectory cache"
log "Manifest: $MANIFEST_PATH"
log "Cache:    $CACHE_DIR"
log "Split:    $SPLIT start=$SPLIT_INDEX_START max=$MAX_EXAMPLES"
log "Steps:    $TRAJECTORY_STEPS solver=$TRAJECTORY_SOLVER"
log "GPUs:     $CUDA_DEVICES num=$NUM_GPUS"

pids=()
for rank in $(seq 0 $((NUM_GPUS - 1))); do
  device="${DEVICE_ARRAY[$rank]}"
  progress_path="$LOG_DIR/precompute_teacher_trajectory_${RUN_TIMESTAMP}_rank${rank}.json"
  part_log="$LOG_DIR/precompute_teacher_trajectory_${RUN_TIMESTAMP}_rank${rank}.log"
  log "Starting cache worker rank=$rank device=$device"
  (
    export CUDA_VISIBLE_DEVICES="$device"
    "$PYTHON" precompute_bidirectional_teacher_trajectory_cache.py \
      --manifest_path "$MANIFEST_PATH" \
      --cache_dir "$CACHE_DIR" \
      --dataset_cache_dir "$DATASET_CACHE_DIR" \
      --dataset_index_workers "$DATASET_INDEX_WORKERS" \
      --dataset_cache_wait_seconds "$DATASET_CACHE_WAIT_SECONDS" \
      --model_root "$MODEL_ROOT" \
      --config_path "$CONFIG_PATH" \
      --target_model_name "$TARGET_MODEL_NAME" \
      --num_blocks "$NUM_BLOCKS" \
      --seed "$SEED" \
      --trajectory_steps "$TRAJECTORY_STEPS" \
      --trajectory_solver "$TRAJECTORY_SOLVER" \
      --split "$SPLIT" \
      --split_index_start "$SPLIT_INDEX_START" \
      --max_examples "$MAX_EXAMPLES" \
      --val_fraction "$VAL_FRACTION" \
      --num_shards "$NUM_GPUS" \
      --shard_index "$rank" \
      --progress_path "$progress_path" \
      2>&1 | tee "$part_log"
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  log "ERROR: one or more cache workers failed"
  exit 1
fi

log "Teacher trajectory cache precompute complete"
