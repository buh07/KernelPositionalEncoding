#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/Kernel PE"
RUN_ROOT="results/reinforce_exp/runs/neurips_ext_20260427_125221"
cd "$ROOT"
echo "[r16] waiting for non-RoPE theory8 reports"
while [ ! -f "${RUN_ROOT}/theory8_position_ablation_nonrope/gpt2-small/report.json" ] || [ ! -f "${RUN_ROOT}/theory8_position_ablation_nonrope/gpt2-medium/report.json" ]; do
  sleep 30
done
.venv/bin/python -u reinforce_exp/exp_new_r16_ablation_r2_regression.py \
  --output-root "${RUN_ROOT}/exp_new_r16_ablation_r2_regression" \
  --nonrope-theory8-root "${RUN_ROOT}/theory8_position_ablation_nonrope" --n-boot 10000 --seed 20260427
