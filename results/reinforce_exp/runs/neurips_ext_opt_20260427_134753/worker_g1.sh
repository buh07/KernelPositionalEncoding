#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/Kernel PE"
RUN_ROOT="${1:?run_root}"
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u experiment3/phase2/exp3p2c_redundancy_quantification.py \
  --model olmo-2-7b --device cuda:0 --output-root "$RUN_ROOT/exp3p2c_redundancy_quantification" \
  --fractions 0,1,2,5,10,15,20,25,50 --sort-orders high_to_low,low_to_high --num-seeds 3 --synthetic-count 100 \
  --batch-size-synth 8 --ntp-count-per-seed 100 --ntp-seq-len 512 --batch-size-ntp 4
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u reinforce_exp/exp_new_r14_kernel_permutation.py \
  --model olmo-2-7b --device cuda:0 --output-root "$RUN_ROOT/exp_new_r14_kernel_permutation" \
  --eval-seqs 100 --seq-len 512 --n-permutations 5 --permutation-seed 20260417 --batch-size 4 --n-boot 5000 \
  --per-head-eval-count 12 --per-head-eval-seqs 32
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u experiment3/theory8_position_ablation.py \
  --model gpt2-small --device cuda:0 --output-dir "$RUN_ROOT/theory8_position_ablation_nonrope" \
  --estimation-seqs 50 --eval-seqs 100 --seq-len 512
for SEED in 1 9 17; do
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u reinforce_exp/exp_r2b_olmo_boundary_power.py run-seed \
    --model olmo-2-7b --seed "$SEED" --domains wiki,code,dialogue --device cuda:0 --num-sequences 64 \
    --seq-len 512 --top-k-dims 16 --synthetic-target-per-cell 1200 --output-root "$RUN_ROOT/exp_r2b_olmo_boundary_power"
done
