#!/usr/bin/env bash
set -euo pipefail
ROOT="/scratch/f004ndc/Kernel PE"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=3
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/experiment3_repair_t5b/olmo-2-7b_${TS}"
mkdir -p "$LOG_DIR"
PY="$ROOT/.venv/bin/python"
{
  echo "[$(date)] START theory5b olmo-2-7b on GPU3"
  "$PY" experiment3/theory5b_boundary_detection.py --model olmo-2-7b --device cuda:0
  echo "[$(date)] END theory5b olmo-2-7b"
} 2>&1 | tee "$LOG_DIR/theory5b.log"
