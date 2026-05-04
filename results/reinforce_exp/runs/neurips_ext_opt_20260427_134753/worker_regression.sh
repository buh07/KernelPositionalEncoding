#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/Kernel PE"
RUN_ROOT="${1:?run_root}"
cd "$ROOT"
echo "[r16] running 3-model RoPE regression (GPT-2 excluded: non-RoPE)"
.venv/bin/python -u reinforce_exp/exp_new_r16_ablation_r2_regression.py \
  --output-root "$RUN_ROOT/exp_new_r16_ablation_r2_regression" \
  --n-boot 10000 --seed 20260427
