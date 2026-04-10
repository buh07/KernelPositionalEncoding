#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/f004ndc/Kernel PE"
cd "$ROOT"

LOG="${1:-$ROOT/logs/experiment3_refresh/postrun_$(date +%Y%m%d_%H%M%S).log}"

echo "[$(date)] Waiting for exp3_refresh_llama and exp3_refresh_olmo to finish" | tee -a "$LOG"

while tmux has-session -t exp3_refresh_llama 2>/dev/null || tmux has-session -t exp3_refresh_olmo 2>/dev/null; do
  echo "[$(date)] Refresh sessions still running" >> "$LOG"
  sleep 120
done

echo "[$(date)] Refresh sessions ended; running postrun validation" | tee -a "$LOG"
./scripts/experiment3_refresh_postrun.sh 2>&1 | tee -a "$LOG"
