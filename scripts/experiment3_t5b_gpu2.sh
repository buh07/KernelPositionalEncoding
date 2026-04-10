#!/usr/bin/env bash
set -euo pipefail
ROOT="/scratch/f004ndc/Kernel PE"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=2
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/experiment3_repair_t5b/llama-3.1-8b_${TS}"
mkdir -p "$LOG_DIR"
PY="$ROOT/.venv/bin/python"
{
  echo "[$(date)] START theory5b llama-3.1-8b on GPU2"
  "$PY" experiment3/theory5b_boundary_detection.py --model llama-3.1-8b --device cuda:0
  echo "[$(date)] END theory5b llama-3.1-8b"
} 2>&1 | tee "$LOG_DIR/theory5b.log"
