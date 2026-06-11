#!/usr/bin/env bash
# Compare target-only generation with whole target-block regeneration verification.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="${SF_VENV:-$PROJECT_ROOT/sf_venv}"
PYTHON="$VENV_DIR/bin/python"

MODEL_ROOT="${MODEL_ROOT:-/mnt/lanxiangh/models}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/self_forcing_dmd.yaml}"
PROMPT_FILE="${PROMPT_FILE:-$PROJECT_ROOT/prompts/MovieGenVideoBench.txt}"
DRAFT_CHECKPOINT_PATH="${DRAFT_CHECKPOINT_PATH:-$MODEL_ROOT/Self-Forcing/checkpoints/self_forcing_dmd.pt}"
TARGET_CHECKPOINT_PATH="${TARGET_CHECKPOINT_PATH:-$MODEL_ROOT/realtime-video/checkpoints/krea-realtime-video-14b.safetensors}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
NUM_BLOCKS="${NUM_BLOCKS:-9}"
MAX_PROMPTS="${MAX_PROMPTS:-16}"
START_INDEX="${START_INDEX:-0}"
AGREEMENT_METRIC="${AGREEMENT_METRIC:-rmse}"
TAU_DELTA="${TAU_DELTA:-0.5}"
TAU_DELTA_SWEEP="${TAU_DELTA_SWEEP:-$TAU_DELTA}"
SAVE_PAIRS="${SAVE_PAIRS:-0}"
STORE_CONTEXT="${STORE_CONTEXT:-0}"
REUSE_SCORING_DECODES_FOR_OUTPUT="${REUSE_SCORING_DECODES_FOR_OUTPUT:-1}"
DRAFT_HEAD_DATASET_DIR="${DRAFT_HEAD_DATASET_DIR:-}"
DRAFT_HEAD_CAPTURE_LAYERS="${DRAFT_HEAD_CAPTURE_LAYERS:-}"
DRAFT_HEAD_SHARD_SIZE="${DRAFT_HEAD_SHARD_SIZE:-128}"
DRAFT_HEAD_FEATURE_STORAGE="${DRAFT_HEAD_FEATURE_STORAGE:-pooled}"
DRAFT_HEAD_CONTEXT_SOURCE="${DRAFT_HEAD_CONTEXT_SOURCE:-target_features}"
DRAFT_HEAD_COLLECTION_MODE="${DRAFT_HEAD_COLLECTION_MODE:-draft_target}"
DRAFT_HEAD_PARALLEL_DEVICES="${DRAFT_HEAD_PARALLEL_DEVICES:-}"
DRAFT_HEAD_PARALLEL_CHILD="${DRAFT_HEAD_PARALLEL_CHILD:-0}"
MODE="${MODE:-}"
if [[ -z "$MODE" ]]; then
  if [[ -n "$DRAFT_HEAD_DATASET_DIR" ]]; then
    case "$DRAFT_HEAD_COLLECTION_MODE" in
      draft_target)
        MODE="target_regen"
        ;;
      target_only)
        MODE="target_only"
        ;;
      *)
        echo "ERROR: DRAFT_HEAD_COLLECTION_MODE must be draft_target or target_only, got: $DRAFT_HEAD_COLLECTION_MODE" >&2
        exit 1
        ;;
    esac
  else
    MODE="compare"
  fi
fi
SEED="${SEED:-42}"

PROMPT_TAG="start${START_INDEX}_max${MAX_PROMPTS}"
TAU_TAG="$(printf '%s' "$TAU_DELTA_SWEEP" | tr ' ' '_' | sed 's/-/neg_/g; s/\./p/g')"
RUN_TAG="${RUN_TAG:-target_regen_${AGREEMENT_METRIC}_blocks${NUM_BLOCKS}_${PROMPT_TAG}_${TAU_TAG}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/sdvg/moviegen_${RUN_TAG}}"

LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/launch_scripts/logs}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
LOG_FILE="$LOG_DIR/$(basename "${BASH_SOURCE[0]}" .sh)_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }

cd "$PROJECT_ROOT"

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "ERROR: prompt file missing: $PROMPT_FILE" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

log "Dataset:   $PROMPT_FILE"
log "Prompts:   start=$START_INDEX max=$MAX_PROMPTS"
log "Mode:      $MODE"
log "Agreement: $AGREEMENT_METRIC tau_delta_sweep=[$TAU_DELTA_SWEEP]"
log "Save pairs: $SAVE_PAIRS (STORE_CONTEXT=$STORE_CONTEXT)"
if [[ -n "$DRAFT_HEAD_DATASET_DIR" ]]; then
  log "Draft-head collection mode: $DRAFT_HEAD_COLLECTION_MODE"
  log "Draft-head dataset: $DRAFT_HEAD_DATASET_DIR"
  log "Draft-head capture layers: $DRAFT_HEAD_CAPTURE_LAYERS"
  log "Draft-head feature storage: $DRAFT_HEAD_FEATURE_STORAGE"
  log "Draft-head context source: $DRAFT_HEAD_CONTEXT_SOURCE"
  if [[ "$MAX_PROMPTS" == "0" ]]; then
    PROMPT_COUNT="$("$PYTHON" - <<PY
from pathlib import Path
path = Path("$PROMPT_FILE")
start = int("$START_INDEX")
prompts = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
print(max(0, len(prompts) - start))
PY
)"
  else
    PROMPT_COUNT="$MAX_PROMPTS"
  fi
  EXPECTED_RECORDS=$(( PROMPT_COUNT * (NUM_BLOCKS - 1) ))
  log "Expected draft-head records per tau: about $EXPECTED_RECORDS (block 0 has no prior target context)"
fi
log "Reuse block decodes for output: $REUSE_SCORING_DECODES_FOR_OUTPUT"
log "Blocks:    $NUM_BLOCKS"
log "Output:    $OUTPUT_DIR"

REUSE_OUTPUT_ARGS=()
if [[ "$REUSE_SCORING_DECODES_FOR_OUTPUT" == "1" ]]; then
  REUSE_OUTPUT_ARGS+=(--reuse_scoring_decodes_for_output)
fi

CONTEXT_ARGS=()
if [[ "$STORE_CONTEXT" == "1" ]]; then
  CONTEXT_ARGS+=(--store_target_regen_context)
fi

if [[ -n "$DRAFT_HEAD_DATASET_DIR" && -z "$DRAFT_HEAD_CAPTURE_LAYERS" ]]; then
  echo "ERROR: DRAFT_HEAD_CAPTURE_LAYERS is required when DRAFT_HEAD_DATASET_DIR is set" >&2
  exit 1
fi

prompt_count() {
  "$PYTHON" - <<PY
from pathlib import Path
path = Path("$PROMPT_FILE")
start = int("$START_INDEX")
max_prompts = int("$MAX_PROMPTS")
prompts = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
remaining = max(0, len(prompts) - start)
print(remaining if max_prompts == 0 else min(max_prompts, remaining))
PY
}

merge_parallel_draft_head_manifests() {
  "$PYTHON" - <<PY
import json
import shutil
from pathlib import Path

root = Path("$DRAFT_HEAD_DATASET_DIR")
devices = "$DRAFT_HEAD_PARALLEL_DEVICES".replace(",", " ").split()
taus = "$TAU_DELTA_SWEEP".split()

for tau in taus:
    tau_label = tau.replace("-", "neg_").replace(".", "p")
    final_dir = root / f"tau_delta_{tau_label}"
    final_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    total_records = 0
    source_manifests = []
    shard_size = int("$DRAFT_HEAD_SHARD_SIZE")
    for worker_index, _device in enumerate(devices):
        manifest_path = root / "_parallel_shards" / f"worker_{worker_index}" / f"tau_delta_{tau_label}" / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_manifests.append(str(manifest_path))
        total_records += int(manifest.get("num_records", 0))
        shard_size = int(manifest.get("shard_size", shard_size))
        for shard in manifest.get("shards", []):
            source_shard = manifest_path.parent / shard["path"]
            merged_name = f"worker_{worker_index}_{source_shard.name}"
            shutil.copy2(source_shard, final_dir / merged_name)
            shards.append({"path": merged_name, "num_records": int(shard["num_records"])})
    if not shards:
        raise SystemExit(f"No shard manifests found for tau_delta_{tau_label} under {root / '_parallel_shards'}")
    merged = {
        "format": "sdvg_draft_head_v1",
        "num_records": total_records,
        "shard_size": shard_size,
        "shards": shards,
        "source_manifests": source_manifests,
    }
    (final_dir / "manifest.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"Merged {len(shards)} shards / {total_records} records -> {final_dir / 'manifest.json'}")
PY
}

if [[ -n "$DRAFT_HEAD_PARALLEL_DEVICES" && "$DRAFT_HEAD_PARALLEL_CHILD" != "1" ]]; then
  if [[ -z "$DRAFT_HEAD_DATASET_DIR" ]]; then
    echo "ERROR: DRAFT_HEAD_PARALLEL_DEVICES requires DRAFT_HEAD_DATASET_DIR" >&2
    exit 1
  fi
  # shellcheck disable=SC2206
  PARALLEL_DEVICE_ARRAY=(${DRAFT_HEAD_PARALLEL_DEVICES//,/ })
  TOTAL_PROMPTS="$(prompt_count)"
  WORKERS="${#PARALLEL_DEVICE_ARRAY[@]}"
  if [[ "$WORKERS" -le 0 || "$TOTAL_PROMPTS" -le 0 ]]; then
    echo "ERROR: no parallel devices or no prompts to process" >&2
    exit 1
  fi
  log "Parallel draft-head collection: devices=[$DRAFT_HEAD_PARALLEL_DEVICES] workers=$WORKERS prompts=$TOTAL_PROMPTS"
  PIDS=()
  for worker_index in "${!PARALLEL_DEVICE_ARRAY[@]}"; do
    device="${PARALLEL_DEVICE_ARRAY[$worker_index]}"
    worker_start=$(( START_INDEX + (TOTAL_PROMPTS * worker_index) / WORKERS ))
    worker_end=$(( START_INDEX + (TOTAL_PROMPTS * (worker_index + 1)) / WORKERS ))
    worker_count=$(( worker_end - worker_start ))
    if [[ "$worker_count" -le 0 ]]; then
      continue
    fi
    worker_output_dir="$OUTPUT_DIR/parallel_worker_${worker_index}"
    worker_dataset_dir="$DRAFT_HEAD_DATASET_DIR/_parallel_shards/worker_${worker_index}"
    log "Launching worker $worker_index device=$device start=$worker_start max=$worker_count"
    (
      CUDA_VISIBLE_DEVICES="$device" \
      DRAFT_HEAD_PARALLEL_DEVICES="" \
      DRAFT_HEAD_PARALLEL_CHILD=1 \
      START_INDEX="$worker_start" \
      MAX_PROMPTS="$worker_count" \
      OUTPUT_DIR="$worker_output_dir" \
      DRAFT_HEAD_DATASET_DIR="$worker_dataset_dir" \
      bash "$0"
    ) &
    PIDS+=("$!")
  done
  for pid in "${PIDS[@]}"; do
    wait "$pid"
  done
  merge_parallel_draft_head_manifests | tee -a "$LOG_FILE"
  log "Parallel sweep complete. Merged dataset saved under: $DRAFT_HEAD_DATASET_DIR"
  exit 0
fi

run_one_tau() {
  local tau="$1"
  local tau_label
  tau_label="$(printf '%s' "$tau" | sed 's/-/neg_/g; s/\./p/g')"
  local run_output_dir="$OUTPUT_DIR/tau_delta_${tau_label}"

  mkdir -p "$run_output_dir"
  log "=== Running tau_delta=$tau -> $run_output_dir ==="

  PAIR_ARGS=()
  if [[ "$SAVE_PAIRS" == "1" ]]; then
    PAIR_ARGS+=(--target_regen_pairs_path "target_regen_pairs.pt")
  fi
  DRAFT_HEAD_ARGS=()
  if [[ -n "$DRAFT_HEAD_DATASET_DIR" ]]; then
    DRAFT_HEAD_ARGS+=(--draft_head_dataset_dir "$DRAFT_HEAD_DATASET_DIR/tau_delta_${tau_label}")
    DRAFT_HEAD_ARGS+=(--draft_head_shard_size "$DRAFT_HEAD_SHARD_SIZE")
    DRAFT_HEAD_ARGS+=(--draft_head_feature_storage "$DRAFT_HEAD_FEATURE_STORAGE")
    DRAFT_HEAD_ARGS+=(--draft_head_context_source "$DRAFT_HEAD_CONTEXT_SOURCE")
    # shellcheck disable=SC2206
    CAPTURE_LAYER_ARRAY=($DRAFT_HEAD_CAPTURE_LAYERS)
    DRAFT_HEAD_ARGS+=(--draft_head_capture_layers "${CAPTURE_LAYER_ARRAY[@]}")
  fi

  "$PYTHON" sdvg_inference.py \
    heuristic \
    --config_path "$CONFIG_PATH" \
    --model_root "$MODEL_ROOT" \
    --draft_checkpoint_path "$DRAFT_CHECKPOINT_PATH" \
    --target_checkpoint_path "$TARGET_CHECKPOINT_PATH" \
    --prompt_file "$PROMPT_FILE" \
    --start_index "$START_INDEX" \
    --max_prompts "$MAX_PROMPTS" \
    --output_dir "$run_output_dir" \
    --mode "$MODE" \
    --compare_mode target_regen \
    --tau "$tau" \
    --agreement_metric "$AGREEMENT_METRIC" \
    "${PAIR_ARGS[@]}" \
    "${CONTEXT_ARGS[@]}" \
    "${DRAFT_HEAD_ARGS[@]}" \
    "${REUSE_OUTPUT_ARGS[@]}" \
    --num_blocks "$NUM_BLOCKS" \
    --seed "$SEED" \
    2>&1 | tee -a "$LOG_FILE"

  log "Profile JSON: $run_output_dir/profile.json"
}

for tau in $TAU_DELTA_SWEEP; do
  run_one_tau "$tau"
done

log "Sweep complete. Outputs saved under: $OUTPUT_DIR"
