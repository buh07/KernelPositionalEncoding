#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
PY="$ROOT/.venv/bin/python"
LOG_DIR="$ROOT/results/cs_fix_logs"
mkdir -p "$LOG_DIR"

# Pre-stage c1 artifacts so Exp7B reruns only recompute corrected calibration/stat summaries.
mkdir -p "$ROOT/results/experiment7_csfix_single/exp7b_phase_transition"
mkdir -p "$ROOT/results/experiment7_csfix_global/exp7b_phase_transition"
rsync -a --delete "$ROOT/results/experiment7/exp7b_phase_transition/c1_runs/" "$ROOT/results/experiment7_csfix_single/exp7b_phase_transition/c1_runs/"
rsync -a --delete "$ROOT/results/experiment7/exp7b_phase_transition/c1_runs/" "$ROOT/results/experiment7_csfix_global/exp7b_phase_transition/c1_runs/"

for s in csfix_g0_exp7b_single csfix_g1_exp7b_global csfix_g2_exp7a csfix_g3_e31b; do
  tmux kill-session -t "$s" 2>/dev/null || true
 done

# GPU0: Exp7B corrected single-condition calibration (matches manuscript wording)
tmux new-session -d -s csfix_g0_exp7b_single "cd '$ROOT' && CUDA_VISIBLE_DEVICES=0 '$PY' -m experiment7.run 7b --output-root 'results/experiment7_csfix_single' --device cuda:0 --models llama-3.1-8b,olmo-2-7b --seq-lens 256,512,1024 --calibration-mode single_condition --calibration-model llama-3.1-8b --calibration-seq-len 256 |& tee '$LOG_DIR/csfix_g0_exp7b_single.log'"

# GPU1: Exp7B global-fit reference (for explicit text/code reconciliation)
tmux new-session -d -s csfix_g1_exp7b_global "cd '$ROOT' && CUDA_VISIBLE_DEVICES=1 '$PY' -m experiment7.run 7b --output-root 'results/experiment7_csfix_global' --device cuda:0 --models llama-3.1-8b,olmo-2-7b --seq-lens 256,512,1024 --calibration-mode global_fit --calibration-model llama-3.1-8b --calibration-seq-len 256 |& tee '$LOG_DIR/csfix_g1_exp7b_global.log'"

# GPU2: Exp7A corrected model-level (non-pseudoreplicated primary stats)
tmux new-session -d -s csfix_g2_exp7a "cd '$ROOT' && CUDA_VISIBLE_DEVICES=2 '$PY' -m experiment7.run 7a --output-root 'results/experiment7_csfix_single' --tracka-md 'experiment1/experiment1results.md' --bootstrap-samples 20000 --bootstrap-seed 42 |& tee '$LOG_DIR/csfix_g2_exp7a.log'"

# GPU3: E31B corrected sign/text + claim impact status
tmux new-session -d -s csfix_g3_e31b "cd '$ROOT' && CUDA_VISIBLE_DEVICES=3 '$PY' reinforce_exp3/scripts/run_e31b_coherence_refresh.py --output-root 'results/reinforce_exp3/E31b_coherence_refresh' --e31a-table 'results/reinforce_exp3/E31a_breadth_consolidation/model_breadth_r2_table.csv' --seq-len 512 |& tee '$LOG_DIR/csfix_g3_e31b.log'"

echo "Launched sessions:"
tmux ls | rg 'csfix_g[0-3]_' || true
