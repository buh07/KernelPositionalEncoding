#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/Kernel PE"
RUN_ROOT="results/reinforce_exp/runs/neurips_ext_20260427_125221"
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=6 .venv/bin/python -u reinforce_exp/exp_new_r12_ordering_control.py \
  --model llama-3.1-8b --device cuda:0 --output-root "$RUN_ROOT/exp_new_r12_ordering_control" \
  --fractions 0,1,2,5,10,15,20,25,50 --num-orderings 10 --ordering-seed-base 20260417 --num-seeds 3 \
  --synthetic-count 100 --batch-size-synth 8 --ntp-count-per-seed 100 --ntp-seq-len 512 --batch-size-ntp 4
CUDA_VISIBLE_DEVICES=6 .venv/bin/python -u experiment3/theory8_position_ablation.py \
  --model gpt2-small --device cuda:0 --output-dir "$RUN_ROOT/theory8_position_ablation_nonrope" \
  --estimation-seqs 50 --eval-seqs 100 --seq-len 512
for SEED in 15 16 17 18 19; do
  CUDA_VISIBLE_DEVICES=6 .venv/bin/python -u reinforce_exp/exp_r2b_olmo_boundary_power.py run-seed \
    --model olmo-2-7b --seed "$SEED" --domains wiki,code,dialogue --device cuda:0 --num-sequences 64 \
    --seq-len 512 --top-k-dims 16 --synthetic-target-per-cell 1200 --output-root "$RUN_ROOT/exp_r2b_olmo_boundary_power"
done
