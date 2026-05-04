#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

SESSIONS=(
  reinforce2_r12_g2
  reinforce2_r12_g3
  reinforce2_r2b_g4
  reinforce2_r2b_g5_r14
  reinforce2_r2b_g6_r14
  reinforce2_r2b_finalize
  reinforce2_r5b_g7
)

for s in "${SESSIONS[@]}"; do
  tmux kill-session -t "$s" 2>/dev/null || true
done

echo "[preflight] GPU status"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

# NEW-R12 + NEW-R15 chained on GPU 2 (Llama)
tmux new-session -d -s reinforce2_r12_g2 "bash -lc '
  set -euo pipefail
  cd \"$ROOT\"
  CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp/exp_new_r12_ordering_control.py \
    --model llama-3.1-8b --device cuda:0 --output-root results/reinforce_exp/exp_new_r12_ordering_control \
    --fractions 0,1,2,5,10,15,20,25,50 --num-orderings 10 --num-seeds 3 --synthetic-count 100 --batch-size-synth 8 \
    --ntp-count-per-seed 100 --ntp-seq-len 512 --batch-size-ntp 4
  CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp/exp_new_r15_head_stability.py \
    --model llama-3.1-8b --device cuda:0 --output-root results/reinforce_exp/exp_new_r15_head_stability \
    --seq-len 512 --num-sequences 500 --quantile 0.25 --n-boot 5000 --seed 17
'"

# NEW-R12 + NEW-R15 chained on GPU 3 (OLMo)
tmux new-session -d -s reinforce2_r12_g3 "bash -lc '
  set -euo pipefail
  cd \"$ROOT\"
  CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp/exp_new_r12_ordering_control.py \
    --model olmo-2-7b --device cuda:0 --output-root results/reinforce_exp/exp_new_r12_ordering_control \
    --fractions 0,1,2,5,10,15,20,25,50 --num-orderings 10 --num-seeds 3 --synthetic-count 100 --batch-size-synth 8 \
    --ntp-count-per-seed 100 --ntp-seq-len 512 --batch-size-ntp 4
  CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp/exp_new_r15_head_stability.py \
    --model olmo-2-7b --device cuda:0 --output-root results/reinforce_exp/exp_new_r15_head_stability \
    --seq-len 512 --num-sequences 500 --quantile 0.25 --n-boot 5000 --seed 17
'"

# R2B seed workers on GPUs 4/5/6
tmux new-session -d -s reinforce2_r2b_g4 "bash -lc '
  set -euo pipefail
  cd \"$ROOT\"
  for SEED in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp/exp_r2b_olmo_boundary_power.py run-seed \
      --model olmo-2-7b --seed \"\$SEED\" --domains wiki,code,dialogue --device cuda:0 \
      --num-sequences 64 --seq-len 512 --top-k-dims 16 --synthetic-target-per-cell 1200 \
      --output-root results/reinforce_exp/exp_r2b_olmo_boundary_power
  done
'"

tmux new-session -d -s reinforce2_r2b_g5_r14 "bash -lc '
  set -euo pipefail
  cd \"$ROOT\"
  for SEED in 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp/exp_r2b_olmo_boundary_power.py run-seed \
      --model olmo-2-7b --seed \"\$SEED\" --domains wiki,code,dialogue --device cuda:0 \
      --num-sequences 64 --seq-len 512 --top-k-dims 16 --synthetic-target-per-cell 1200 \
      --output-root results/reinforce_exp/exp_r2b_olmo_boundary_power
  done
  CUDA_VISIBLE_DEVICES=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp/exp_new_r14_kernel_permutation.py \
    --model llama-3.1-8b --device cuda:0 --output-root results/reinforce_exp/exp_new_r14_kernel_permutation \
    --eval-seqs 100 --seq-len 512 --n-permutations 5 --permutation-seed 20260417 --batch-size 4 --n-boot 5000 \
    --per-head-eval-count 12 --per-head-eval-seqs 32
'"

tmux new-session -d -s reinforce2_r2b_g6_r14 "bash -lc '
  set -euo pipefail
  cd \"$ROOT\"
  for SEED in 8 9 10 11; do
    CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp/exp_r2b_olmo_boundary_power.py run-seed \
      --model olmo-2-7b --seed \"\$SEED\" --domains wiki,code,dialogue --device cuda:0 \
      --num-sequences 64 --seq-len 512 --top-k-dims 16 --synthetic-target-per-cell 1200 \
      --output-root results/reinforce_exp/exp_r2b_olmo_boundary_power
  done
  CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp/exp_new_r14_kernel_permutation.py \
    --model olmo-2-7b --device cuda:0 --output-root results/reinforce_exp/exp_new_r14_kernel_permutation \
    --eval-seqs 100 --seq-len 512 --n-permutations 5 --permutation-seed 20260417 --batch-size 4 --n-boot 5000 \
    --per-head-eval-count 12 --per-head-eval-seqs 32
'"

# R2B finalization watcher (CPU)
tmux new-session -d -s reinforce2_r2b_finalize "bash -lc '
  set -euo pipefail
  cd \"$ROOT\"
  echo \"[R2B-finalize] waiting for 12 seed manifests...\"
  while true; do
    N=\$(ls results/reinforce_exp/exp_r2b_olmo_boundary_power/manifests/seed_*.json 2>/dev/null | wc -l || true)
    if [ \"\$N\" -ge 12 ]; then
      break
    fi
    sleep 30
  done
  .venv/bin/python -u reinforce_exp/exp_r2b_olmo_boundary_power.py finalize-domain \
    --model olmo-2-7b --seeds 0,1,2,3,4,5,6,7,8,9,10,11 --domains wiki,code,dialogue \
    --output-root results/reinforce_exp/exp_r2b_olmo_boundary_power --n-boot 5000 --n-perm 5000 --seed 13
  .venv/bin/python -u reinforce_exp/exp_r2b_olmo_boundary_power.py finalize-claim \
    --model olmo-2-7b --output-root results/reinforce_exp/exp_r2b_olmo_boundary_power
'"

# R5B on GPU 7 (both models sequentially)
tmux new-session -d -s reinforce2_r5b_g7 "bash -lc '
  set -euo pipefail
  cd \"$ROOT\"
  CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp/exp_r5b_regime_alignment.py \
    --model all --device-map llama-3.1-8b:cuda:0,olmo-2-7b:cuda:0 --num-seeds 12 --seq-len 512 --batch-size 4 \
    --count-scale 1.0 --output-root results/reinforce_exp/exp_r5b_regime_alignment
'"

echo "[launch] started priority reinforce_exp sessions"
tmux ls | rg 'reinforce2_' || true
echo "[monitor] bash reinforce_exp/status_priority_tmux.sh"
