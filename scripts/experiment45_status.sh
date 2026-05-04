#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "== tmux sessions =="
tmux ls || true

echo ""
echo "== GPU snapshot =="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo ""
echo "== Exp4 latest log tail =="
if [ -f "$ROOT_DIR/logs/experiment4/full_program_latest/exp4_gpu0.log" ]; then
  tail -n 40 "$ROOT_DIR/logs/experiment4/full_program_latest/exp4_gpu0.log"
else
  echo "No Exp4 full-program log yet."
fi

echo ""
echo "== Exp5 latest log tail =="
if [ -f "$ROOT_DIR/logs/experiment5/full_program_latest/exp5_gpu1.log" ]; then
  tail -n 40 "$ROOT_DIR/logs/experiment5/full_program_latest/exp5_gpu1.log"
else
  echo "No Exp5 full-program log yet."
fi
