#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"

echo "[tmux]"
tmux ls 2>/dev/null | grep 'rexp3_e28' || echo "no rexp3_e28 sessions"

echo
for f in reinforce_exp3/logs/rexp3_e28_g0_llama.log \
         reinforce_exp3/logs/rexp3_e28_g1_mistral.log \
         reinforce_exp3/logs/rexp3_e28_g2_olmo.log \
         reinforce_exp3/logs/rexp3_e28_g3_finalize.log; do
  echo "===== $f (tail -n 20) ====="
  [[ -f "$f" ]] && tail -n 20 "$f" || echo "(missing)"
  echo
done

echo "[artifacts]"
for m in llama-3.1-8b mistral-7b-v0.1 olmo-2-7b; do
  p="results/reinforce_exp3/E28_probe_scope_battery/$m/summary.json"
  if [[ -f "$p" ]]; then
    echo "- $m: ready"
  else
    echo "- $m: pending"
  fi
done
[[ -f results/reinforce_exp3/E28_probe_scope_battery/cross_model_probe_scope_summary.json ]] \
  && echo "- cross-model finalize: ready" \
  || echo "- cross-model finalize: pending"
