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
for s in exp3p2_stage4_llama exp3p2_stage4_olmo; do
  echo "===== $s ====="
  if tmux has-session -t "$s" 2>/dev/null; then
    tmux capture-pane -p -t "$s" | tail -n 120
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
    ('3P2-A report', 'results/experiment3_phase2/exp3p2a_positional_broadcast/{m}/a_report.json'),
    ('3P2-A tasks', 'results/experiment3_phase2/exp3p2a_positional_broadcast/{m}/task_battery_results.parquet'),
    ('3P2-J summary', 'results/experiment3_phase2/exp3p2j_conditional_regimes/{m}/regime_summary.json'),
    ('3P2-J rows', 'results/experiment3_phase2/exp3p2j_conditional_regimes/{m}/conditional_effects.parquet'),
    ('Idea6 summary', 'results/experiment3_phase2/idea6_math_si_channels/{m}/intervention_summary.json'),
    ('Idea6 rows', 'results/experiment3_phase2/idea6_math_si_channels/{m}/math_channel_results.parquet'),
    ('Idea4 report', 'results/experiment3_phase2/idea4_structural_ambiguity_normed_stage4/{m}/ambiguity_report.json'),
]

for m in models:
    print(f'\n[{m}]')
    for label, pat in checks:
        p = root / pat.format(m=m)
        print(f'  {label:16s}: {"OK" if p.exists() else "MISSING"} ({p})')

    j = root / f'results/experiment3_phase2/exp3p2j_conditional_regimes/{m}/regime_summary.json'
    if j.exists():
        d = json.loads(j.read_text())
        print(f"  3P2-J supports_e10: {d.get('verdict', {}).get('supports_e10_conditional_specialization')}")

    i6 = root / f'results/experiment3_phase2/idea6_math_si_channels/{m}/intervention_summary.json'
    if i6.exists():
        d = json.loads(i6.read_text())
        print(f"  Idea6 any_improve: {d.get('verdict', {}).get('any_si_channel_improves_math')}")
PY
