#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

echo "[tmux]"
tmux ls 2>/dev/null | rg 'exp3p2_tok_audit' || echo "(none)"

echo
echo "[aggregate]"
p="results/experiment3_phase2/exp3p2b_tokenizer_audit/tokenizer_audit_report.json"
if [[ -f "$p" ]]; then
  jq '{timestamp, verdict}' "$p"
else
  echo "pending"
fi

echo
echo "[per-model]"
for m in llama-3.1-8b olmo-2-7b; do
  q="results/experiment3_phase2/exp3p2b_tokenizer_audit/${m}/tokenizer_audit_report.json"
  if [[ -f "$q" ]]; then
    echo "--- $m"
    jq '{model, verdict, feature_summaries}' "$q"
  else
    echo "--- $m: pending"
  fi
done
