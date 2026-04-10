#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

NUM_SEQ="${1:-16}"
SYNTH="${2:-300}"

for s in exp3p2_tok_audit_llama exp3p2_tok_audit_olmo; do
  tmux kill-session -t "$s" 2>/dev/null || true
done

tmux new-session -d -s exp3p2_tok_audit_llama \
  "cd '$ROOT' && CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' .venv/bin/python -u experiment3/phase2/exp3p2b_tokenizer_audit.py --model llama-3.1-8b --device cuda:0 --num-sequences '$NUM_SEQ' --seq-len 512 --top-k-dims 16 --synthetic-target-per-cell '$SYNTH' --seed 0 --output-root results/experiment3_phase2/exp3p2b_tokenizer_audit"

tmux new-session -d -s exp3p2_tok_audit_olmo \
  "cd '$ROOT' && CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' .venv/bin/python -u experiment3/phase2/exp3p2b_tokenizer_audit.py --model olmo-2-7b --device cuda:0 --num-sequences '$NUM_SEQ' --seq-len 512 --top-k-dims 16 --synthetic-target-per-cell '$SYNTH' --seed 0 --output-root results/experiment3_phase2/exp3p2b_tokenizer_audit"

echo "[launch] tokenizer audit sessions started"
tmux ls | rg 'exp3p2_tok_audit' || true
