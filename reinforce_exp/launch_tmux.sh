#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

SESSIONS=(
  reinforce_r1_g2
  reinforce_r1_g3
  reinforce_r1_g4
  reinforce_r1_finalize
  reinforce_r2
  reinforce_r3
  reinforce_r4
  reinforce_r5
)

for s in "${SESSIONS[@]}"; do
  tmux kill-session -t "$s" 2>/dev/null || true
done

echo "[preflight] GPU status"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

# EXP-R1 workers (12 seeds across GPUs 2/3/4)
tmux new-session -d -s reinforce_r1_g2 "bash -lc 'cd \"$ROOT\" && CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp/exp_r1_multidomain_gate.py run-seed --model llama-3.1-8b --seed 0 --domains wiki,code,dialogue --device cuda:0 --num-sequences 48 --synthetic-target-per-cell 1000 --output-root results/reinforce_exp/exp_r1_multidomain_gate && CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp/exp_r1_multidomain_gate.py run-seed --model llama-3.1-8b --seed 1 --domains wiki,code,dialogue --device cuda:0 --num-sequences 48 --synthetic-target-per-cell 1000 --output-root results/reinforce_exp/exp_r1_multidomain_gate && CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp/exp_r1_multidomain_gate.py run-seed --model llama-3.1-8b --seed 2 --domains wiki,code,dialogue --device cuda:0 --num-sequences 48 --synthetic-target-per-cell 1000 --output-root results/reinforce_exp/exp_r1_multidomain_gate && CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp/exp_r1_multidomain_gate.py run-seed --model llama-3.1-8b --seed 3 --domains wiki,code,dialogue --device cuda:0 --num-sequences 48 --synthetic-target-per-cell 1000 --output-root results/reinforce_exp/exp_r1_multidomain_gate'"

tmux new-session -d -s reinforce_r1_g3 "bash -lc 'cd \"$ROOT\" && CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp/exp_r1_multidomain_gate.py run-seed --model llama-3.1-8b --seed 4 --domains wiki,code,dialogue --device cuda:0 --num-sequences 48 --synthetic-target-per-cell 1000 --output-root results/reinforce_exp/exp_r1_multidomain_gate && CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp/exp_r1_multidomain_gate.py run-seed --model llama-3.1-8b --seed 5 --domains wiki,code,dialogue --device cuda:0 --num-sequences 48 --synthetic-target-per-cell 1000 --output-root results/reinforce_exp/exp_r1_multidomain_gate && CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp/exp_r1_multidomain_gate.py run-seed --model llama-3.1-8b --seed 6 --domains wiki,code,dialogue --device cuda:0 --num-sequences 48 --synthetic-target-per-cell 1000 --output-root results/reinforce_exp/exp_r1_multidomain_gate && CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp/exp_r1_multidomain_gate.py run-seed --model llama-3.1-8b --seed 7 --domains wiki,code,dialogue --device cuda:0 --num-sequences 48 --synthetic-target-per-cell 1000 --output-root results/reinforce_exp/exp_r1_multidomain_gate'"

tmux new-session -d -s reinforce_r1_g4 "bash -lc 'cd \"$ROOT\" && CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp/exp_r1_multidomain_gate.py run-seed --model llama-3.1-8b --seed 8 --domains wiki,code,dialogue --device cuda:0 --num-sequences 48 --synthetic-target-per-cell 1000 --output-root results/reinforce_exp/exp_r1_multidomain_gate && CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp/exp_r1_multidomain_gate.py run-seed --model llama-3.1-8b --seed 9 --domains wiki,code,dialogue --device cuda:0 --num-sequences 48 --synthetic-target-per-cell 1000 --output-root results/reinforce_exp/exp_r1_multidomain_gate && CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp/exp_r1_multidomain_gate.py run-seed --model llama-3.1-8b --seed 10 --domains wiki,code,dialogue --device cuda:0 --num-sequences 48 --synthetic-target-per-cell 1000 --output-root results/reinforce_exp/exp_r1_multidomain_gate && CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp/exp_r1_multidomain_gate.py run-seed --model llama-3.1-8b --seed 11 --domains wiki,code,dialogue --device cuda:0 --num-sequences 48 --synthetic-target-per-cell 1000 --output-root results/reinforce_exp/exp_r1_multidomain_gate'"

# EXP-R1 finalize watcher
tmux new-session -d -s reinforce_r1_finalize "bash -lc 'cd \"$ROOT\" && bash reinforce_exp/r1_finalize_watch.sh'"

# EXP-R2 dependence-aware reanalysis (CPU)
tmux new-session -d -s reinforce_r2 "bash -lc 'cd \"$ROOT\" && .venv/bin/python -u reinforce_exp/exp_r2_dependence_reanalysis.py --output-root results/reinforce_exp/exp_r2_dependence_reanalysis --n-boot 5000 --n-perm 5000 --seed 7'"

# EXP-R3 core replication on Mistral
tmux new-session -d -s reinforce_r3 "bash -lc 'cd \"$ROOT\" && CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp/exp_r3_core_replication.py --model mistral-7b-v0.1 --device cuda:0 --num-sequences-b 32 --synthetic-b 600 --num-sequences-j 160 --output-root results/reinforce_exp/exp_r3_core_replication'"

# EXP-R4 PE-scheme contrast
tmux new-session -d -s reinforce_r4 "bash -lc 'cd \"$ROOT\" && CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp/exp_r4_pe_scheme_contrast.py --device cuda:0 --num-sequences 24 --seq-len 512 --top-k-dims 16 --synthetic-target-per-cell 300 --seed 0 --output-root results/reinforce_exp/exp_r4_pe_scheme_contrast'"

# EXP-R5 task-grounded specialization
tmux new-session -d -s reinforce_r5 "bash -lc 'cd \"$ROOT\" && CUDA_VISIBLE_DEVICES=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp/exp_r5_task_grounded_specialization.py --model all --device-map llama-3.1-8b:cuda:0,olmo-2-7b:cuda:0 --num-seeds 6 --seq-len 512 --batch-size 4 --count-scale 1.0 --output-root results/reinforce_exp/exp_r5_task_grounded'"

echo "[launch] started reinforce_exp tmux sessions"
tmux ls | rg 'reinforce_' || true

echo "[monitor] bash reinforce_exp/status_tmux.sh"
