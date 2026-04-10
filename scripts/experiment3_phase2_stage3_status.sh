#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

echo "=== tmux sessions ==="
tmux ls || true

echo
echo "=== GPUs ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

echo
for s in exp3p2_stage3_llama exp3p2_stage3_olmo; do
  echo "===== $s ====="
  if tmux has-session -t "$s" 2>/dev/null; then
    tmux capture-pane -p -t "$s" | tail -n 80
  else
    echo "(not running)"
  fi
  echo
done

echo "=== artifact check ==="
python - <<'PY'
from pathlib import Path
import json

root = Path('/jumbo/lisp/f004ndc/Kernel PE')
models = ['llama-3.1-8b', 'olmo-2-7b']
checks = [
    ('3P2-I report', 'results/experiment3_phase2/exp3p2i_tokenizer_corpus/{m}/invariance_report.json'),
    ('Idea4 report', 'results/experiment3_phase2/idea4_structural_ambiguity/{m}/ambiguity_report.json'),
    ('Idea4 rows', 'results/experiment3_phase2/idea4_structural_ambiguity/{m}/per_item_scores.parquet'),
    ('3P2-C.2 report', 'results/experiment3_phase2/exp3p2c_redundancy_quantification/{m}/nonlinearity_test.json'),
    ('3P2-C.2 rows', 'results/experiment3_phase2/exp3p2c_redundancy_quantification/{m}/simultaneous_ablation_results.parquet'),
]
for m in models:
    print(f'\n[{m}]')
    for label, pat in checks:
        p = root / pat.format(m=m)
        print(f'  {label:16s}: {"OK" if p.exists() else "MISSING"} ({p})')
    inv = root / f'results/experiment3_phase2/exp3p2i_tokenizer_corpus/{m}/invariance_report.json'
    if inv.exists():
        d = json.loads(inv.read_text())
        print(f"  invariance verdict: {d.get('invariance_verdict')}")
PY
