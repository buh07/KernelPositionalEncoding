#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
RUN_ID="neurips_ext_20260427_125221"
RUN_ROOT="results/reinforce_exp/runs/${RUN_ID}"

mk_session() {
  local name="$1"
  local body="$2"
  if tmux has-session -t "$name" 2>/dev/null; then
    echo "[skip] session exists: $name"
    return 0
  fi
  tmux new-session -d -s "$name" "bash -lc '$body'"
  echo "[ok] launched: $name"
}

mk_session "n26ext_${RUN_ID}_g0" '
set -euo pipefail
cd "'"$ROOT"'"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiment3/phase2/exp3p2c_redundancy_quantification.py \
  --model llama-3.1-8b --device cuda:0 --output-root "'"$RUN_ROOT"'"/exp3p2c_redundancy_quantification \
  --fractions 0,1,2,5,10,15,20,25,50 --sort-orders high_to_low,low_to_high --num-seeds 3 --synthetic-count 100 \
  --batch-size-synth 8 --ntp-count-per-seed 100 --ntp-seq-len 512 --batch-size-ntp 4
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u reinforce_exp/exp_new_r14_kernel_permutation.py \
  --model llama-3.1-8b --device cuda:0 --output-root "'"$RUN_ROOT"'"/exp_new_r14_kernel_permutation \
  --eval-seqs 100 --seq-len 512 --n-permutations 5 --permutation-seed 20260417 --batch-size 4 --n-boot 5000 \
  --per-head-eval-count 12 --per-head-eval-seqs 32
for SEED in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u reinforce_exp/exp_r2b_olmo_boundary_power.py run-seed \
    --model olmo-2-7b --seed "$SEED" --domains wiki,code,dialogue --device cuda:0 --num-sequences 64 \
    --seq-len 512 --top-k-dims 16 --synthetic-target-per-cell 1200 --output-root "'"$RUN_ROOT"'"/exp_r2b_olmo_boundary_power
 done
'

mk_session "n26ext_${RUN_ID}_g1" '
set -euo pipefail
cd "'"$ROOT"'"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u experiment3/phase2/exp3p2c_redundancy_quantification.py \
  --model olmo-2-7b --device cuda:0 --output-root "'"$RUN_ROOT"'"/exp3p2c_redundancy_quantification \
  --fractions 0,1,2,5,10,15,20,25,50 --sort-orders high_to_low,low_to_high --num-seeds 3 --synthetic-count 100 \
  --batch-size-synth 8 --ntp-count-per-seed 100 --ntp-seq-len 512 --batch-size-ntp 4
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u reinforce_exp/exp_new_r14_kernel_permutation.py \
  --model mistral-7b-v0.1 --device cuda:0 --output-root "'"$RUN_ROOT"'"/exp_new_r14_kernel_permutation \
  --eval-seqs 100 --seq-len 512 --n-permutations 5 --permutation-seed 20260417 --batch-size 4 --n-boot 5000 \
  --per-head-eval-count 12 --per-head-eval-seqs 32
for SEED in 5 6 7 8 9; do
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u reinforce_exp/exp_r2b_olmo_boundary_power.py run-seed \
    --model olmo-2-7b --seed "$SEED" --domains wiki,code,dialogue --device cuda:0 --num-sequences 64 \
    --seq-len 512 --top-k-dims 16 --synthetic-target-per-cell 1200 --output-root "'"$RUN_ROOT"'"/exp_r2b_olmo_boundary_power
 done
'

mk_session "n26ext_${RUN_ID}_g2" '
set -euo pipefail
cd "'"$ROOT"'"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -u experiment3/phase2/exp3p2c_redundancy_quantification.py \
  --model mistral-7b-v0.1 --device cuda:0 --output-root "'"$RUN_ROOT"'"/exp3p2c_redundancy_quantification \
  --fractions 0,1,2,5,10,15,20,25,50 --sort-orders high_to_low,low_to_high --num-seeds 3 --synthetic-count 100 \
  --batch-size-synth 8 --ntp-count-per-seed 100 --ntp-seq-len 512 --batch-size-ntp 4
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -u reinforce_exp/exp_new_r12_mistral.py \
  --device cuda:0 --output-root "'"$RUN_ROOT"'"/exp_new_r12_ordering_control \
  --fractions 0,1,2,5,10,15,20,25,50 --num-orderings 10 --num-seeds 3 --synthetic-count 100 \
  --batch-size-synth 8 --ntp-count-per-seed 100 --ntp-seq-len 512 --batch-size-ntp 4
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -u reinforce_exp/exp_new_r14_kernel_permutation.py \
  --model olmo-2-7b --device cuda:0 --output-root "'"$RUN_ROOT"'"/exp_new_r14_kernel_permutation \
  --eval-seqs 100 --seq-len 512 --n-permutations 5 --permutation-seed 20260417 --batch-size 4 --n-boot 5000 \
  --per-head-eval-count 12 --per-head-eval-seqs 32
for SEED in 10 11 12 13 14; do
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python -u reinforce_exp/exp_r2b_olmo_boundary_power.py run-seed \
    --model olmo-2-7b --seed "$SEED" --domains wiki,code,dialogue --device cuda:0 --num-sequences 64 \
    --seq-len 512 --top-k-dims 16 --synthetic-target-per-cell 1200 --output-root "'"$RUN_ROOT"'"/exp_r2b_olmo_boundary_power
 done
'

mk_session "n26ext_${RUN_ID}_g6" '
set -euo pipefail
cd "'"$ROOT"'"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=6 .venv/bin/python -u reinforce_exp/exp_new_r12_ordering_control.py \
  --model llama-3.1-8b --device cuda:0 --output-root "'"$RUN_ROOT"'"/exp_new_r12_ordering_control \
  --fractions 0,1,2,5,10,15,20,25,50 --num-orderings 10 --ordering-seed-base 20260417 --num-seeds 3 \
  --synthetic-count 100 --batch-size-synth 8 --ntp-count-per-seed 100 --ntp-seq-len 512 --batch-size-ntp 4
CUDA_VISIBLE_DEVICES=6 .venv/bin/python -u experiment3/theory8_position_ablation.py \
  --model gpt2-small --device cuda:0 --output-dir "'"$RUN_ROOT"'"/theory8_position_ablation_nonrope \
  --estimation-seqs 50 --eval-seqs 100 --seq-len 512
for SEED in 15 16 17 18 19; do
  CUDA_VISIBLE_DEVICES=6 .venv/bin/python -u reinforce_exp/exp_r2b_olmo_boundary_power.py run-seed \
    --model olmo-2-7b --seed "$SEED" --domains wiki,code,dialogue --device cuda:0 --num-sequences 64 \
    --seq-len 512 --top-k-dims 16 --synthetic-target-per-cell 1200 --output-root "'"$RUN_ROOT"'"/exp_r2b_olmo_boundary_power
 done
'

mk_session "n26ext_${RUN_ID}_g7" '
set -euo pipefail
cd "'"$ROOT"'"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -u reinforce_exp/exp_new_r12_ordering_control.py \
  --model olmo-2-7b --device cuda:0 --output-root "'"$RUN_ROOT"'"/exp_new_r12_ordering_control \
  --fractions 0,1,2,5,10,15,20,25,50 --num-orderings 10 --ordering-seed-base 20260417 --num-seeds 3 \
  --synthetic-count 100 --batch-size-synth 8 --ntp-count-per-seed 100 --ntp-seq-len 512 --batch-size-ntp 4
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -u experiment3/theory8_position_ablation.py \
  --model gpt2-medium --device cuda:0 --output-dir "'"$RUN_ROOT"'"/theory8_position_ablation_nonrope \
  --estimation-seqs 50 --eval-seqs 100 --seq-len 512
for SEED in 20 21 22 23; do
  CUDA_VISIBLE_DEVICES=7 .venv/bin/python -u reinforce_exp/exp_r2b_olmo_boundary_power.py run-seed \
    --model olmo-2-7b --seed "$SEED" --domains wiki,code,dialogue --device cuda:0 --num-sequences 64 \
    --seq-len 512 --top-k-dims 16 --synthetic-target-per-cell 1200 --output-root "'"$RUN_ROOT"'"/exp_r2b_olmo_boundary_power
 done
'

mk_session "n26ext_${RUN_ID}_r2b_finalize" '
set -euo pipefail
cd "'"$ROOT"'"
echo "[r2b-finalize] waiting for 24 seed manifests under '"$RUN_ROOT"'/exp_r2b_olmo_boundary_power/manifests"
while true; do
  N=$(ls "'"$RUN_ROOT"'"/exp_r2b_olmo_boundary_power/manifests/seed_*.json 2>/dev/null | wc -l || true)
  if [ "$N" -ge 24 ]; then
    break
  fi
  sleep 30
done
.venv/bin/python -u reinforce_exp/exp_r2b_olmo_boundary_power.py finalize-domain \
  --model olmo-2-7b --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 \
  --domains wiki,code,dialogue --output-root "'"$RUN_ROOT"'"/exp_r2b_olmo_boundary_power --n-boot 5000 --n-perm 5000 --seed 13
.venv/bin/python -u reinforce_exp/exp_r2b_olmo_boundary_power.py finalize-claim \
  --model olmo-2-7b --output-root "'"$RUN_ROOT"'"/exp_r2b_olmo_boundary_power
'

mk_session "n26ext_${RUN_ID}_regression" '
set -euo pipefail
cd "'"$ROOT"'"
echo "[r16] waiting for non-RoPE theory8 reports"
while [ ! -f "'"$RUN_ROOT"'"/theory8_position_ablation_nonrope/gpt2-small/report.json" ] || [ ! -f "'"$RUN_ROOT"'"/theory8_position_ablation_nonrope/gpt2-medium/report.json" ]; do
  sleep 30
done
.venv/bin/python -u reinforce_exp/exp_new_r16_ablation_r2_regression.py \
  --output-root "'"$RUN_ROOT"'"/exp_new_r16_ablation_r2_regression \
  --nonrope-theory8-root "'"$RUN_ROOT"'"/theory8_position_ablation_nonrope --n-boot 10000 --seed 20260427
'

echo "[launch] complete"
tmux ls | rg "n26ext_${RUN_ID}" || true
