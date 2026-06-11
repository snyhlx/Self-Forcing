#!/usr/bin/env bash
# Create/use a uv VBench env, then evaluate all SDVG video directories.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VBENCH_VENV="${VBENCH_VENV:-/root/vbench_venv}"
VBENCH_BIN="${VBENCH_BIN:-$VBENCH_VENV/bin/vbench}"
PYTHON_BIN="${PYTHON_BIN:-$VBENCH_VENV/bin/python}"

if [[ ! -x "$VBENCH_BIN" ]]; then
  echo "VBench not found at $VBENCH_BIN; creating uv env at $VBENCH_VENV"
  uv venv "$VBENCH_VENV" --python 3.11
  uv pip install --python "$PYTHON_BIN" -U pip
  uv pip install --python "$PYTHON_BIN" vbench
fi

export HF_HOME="${HF_HOME:-/mnt/lanxiangh/models/hf_cache}"
export PATH="$VBENCH_VENV/bin:$PATH"

exec "$PYTHON_BIN" "$SCRIPT_DIR/eval_vbench_sdvg.py" \
  --vbench_bin "$VBENCH_BIN" \
  --vbench_python "$PYTHON_BIN" \
  "$@"
