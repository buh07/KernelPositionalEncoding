#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/Kernel PE"
RUN_ROOT="${1:?run_root}"
cd "$ROOT"
echo "[r12-merge] waiting for shard summaries"
need=(
  "$RUN_ROOT/exp_new_r12_ordering_control_shards/llama_a/llama-3.1-8b/summary.json"
  "$RUN_ROOT/exp_new_r12_ordering_control_shards/llama_b/llama-3.1-8b/summary.json"
  "$RUN_ROOT/exp_new_r12_ordering_control_shards/olmo_a/olmo-2-7b/summary.json"
  "$RUN_ROOT/exp_new_r12_ordering_control_shards/olmo_b/olmo-2-7b/summary.json"
  "$RUN_ROOT/exp_new_r12_ordering_control_shards/mistral_a/mistral-7b-v0.1/summary.json"
  "$RUN_ROOT/exp_new_r12_ordering_control_shards/mistral_b/mistral-7b-v0.1/summary.json"
)
while true; do
  ok=1
  for p in "${need[@]}"; do
    if [ ! -f "$p" ]; then ok=0; break; fi
  done
  [ "$ok" -eq 1 ] && break
  sleep 30
done
.venv/bin/python -u reinforce_exp/exp_new_r12_merge_shards.py --model llama-3.1-8b \
  --shard-roots "$RUN_ROOT/exp_new_r12_ordering_control_shards/llama_a,$RUN_ROOT/exp_new_r12_ordering_control_shards/llama_b" \
  --output-root "$RUN_ROOT/exp_new_r12_ordering_control" --expected-ordering-start 0 --expected-ordering-stop 10
.venv/bin/python -u reinforce_exp/exp_new_r12_merge_shards.py --model olmo-2-7b \
  --shard-roots "$RUN_ROOT/exp_new_r12_ordering_control_shards/olmo_a,$RUN_ROOT/exp_new_r12_ordering_control_shards/olmo_b" \
  --output-root "$RUN_ROOT/exp_new_r12_ordering_control" --expected-ordering-start 0 --expected-ordering-stop 10
.venv/bin/python -u reinforce_exp/exp_new_r12_merge_shards.py --model mistral-7b-v0.1 \
  --shard-roots "$RUN_ROOT/exp_new_r12_ordering_control_shards/mistral_a,$RUN_ROOT/exp_new_r12_ordering_control_shards/mistral_b" \
  --output-root "$RUN_ROOT/exp_new_r12_ordering_control" --expected-ordering-start 0 --expected-ordering-stop 10
