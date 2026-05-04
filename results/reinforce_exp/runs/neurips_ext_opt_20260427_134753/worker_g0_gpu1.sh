#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/Kernel PE"
RUN_ROOT="${1:?run_root}"
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[g0-gpu1] starting R12 mistral_b on GPU 1 (CUDA_VISIBLE_DEVICES=1)"
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u reinforce_exp/exp_new_r12_mistral.py \
  --device cuda:0 --output-root "$RUN_ROOT/exp_new_r12_ordering_control_shards/mistral_b" \
  --fractions 0,1,2,5,10,15,20,25,50 --num-orderings 10 --ordering-id-start 5 --ordering-id-stop 10 --ordering-seed-base 20260417 \
  --num-seeds 3 --synthetic-count 100 --batch-size-synth 8 --ntp-count-per-seed 100 --ntp-seq-len 512 --batch-size-ntp 4
echo "[g0-gpu1] R12 mistral_b done; starting R2B seeds 0 8 16 on GPU 1"
for SEED in 0 8 16; do
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u reinforce_exp/exp_r2b_olmo_boundary_power.py run-seed \
    --model olmo-2-7b --seed "$SEED" --domains wiki,code,dialogue --device cuda:0 --num-sequences 64 \
    --seq-len 512 --top-k-dims 16 --synthetic-target-per-cell 1200 --output-root "$RUN_ROOT/exp_r2b_olmo_boundary_power"
done
echo "[g0-gpu1] all done"
