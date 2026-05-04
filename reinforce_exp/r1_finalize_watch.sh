#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

MODEL="llama-3.1-8b"
OUT_ROOT="results/reinforce_exp/exp_r1_multidomain_gate"
DOMAINS=(wiki code dialogue)
SEEDS=(0 1 2 3 4 5 6 7 8 9 10 11)

while true; do
  missing=0
  for d in "${DOMAINS[@]}"; do
    for s in "${SEEDS[@]}"; do
      p="$OUT_ROOT/domain_${d}/seed${s}/${MODEL}/synthetic_boundary_results.json"
      if [[ ! -f "$p" ]]; then
        missing=1
        break
      fi
    done
    if [[ "$missing" -eq 1 ]]; then
      break
    fi
  done
  if [[ "$missing" -eq 0 ]]; then
    break
  fi
  sleep 120
done

.venv/bin/python -u reinforce_exp/exp_r1_multidomain_gate.py finalize-domain \
  --model "$MODEL" \
  --domains "wiki,code,dialogue" \
  --seeds "0,1,2,3,4,5,6,7,8,9,10,11" \
  --output-root "$OUT_ROOT"

.venv/bin/python -u reinforce_exp/exp_r1_multidomain_gate.py finalize-multidomain \
  --model "$MODEL" \
  --domains "wiki,code,dialogue" \
  --output-root "$OUT_ROOT"
