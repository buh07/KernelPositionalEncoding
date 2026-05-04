#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${1:-reinforce_exp2_full}"
RUNS_DIR="$ROOT_DIR/results/reinforce_exp2/pipeline_runs"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session: $SESSION (running)"
  tmux list-windows -t "$SESSION"
  echo
  echo "Panes:"
  tmux list-panes -a -F '#{session_name}:#{window_name}.#{pane_index} #{pane_current_command} #{pane_current_path}' | grep "^${SESSION}:" || true
else
  echo "Session: $SESSION (not running)"
fi

echo
latest_record="$(ls -1t "$RUNS_DIR"/tmux_full_*.json 2>/dev/null | head -n 1 || true)"
if [[ -z "$latest_record" ]]; then
  echo "No tmux run records found under $RUNS_DIR"
  exit 0
fi

echo "Latest run record: $latest_record"
python - "$latest_record" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
data = json.loads(p.read_text(encoding="utf-8"))
print(f"run_id={data.get('run_id')} n_events={data.get('n_events')} n_failures={data.get('n_failures')}")
rows = data.get("rows", [])
if rows:
    print("recent events:")
    for row in rows[-8:]:
        print(f"  {row.get('timestamp')} {row.get('stage')} {row.get('item')} {row.get('status')} {row.get('message')}")
marker_dir = data.get("marker_dir")
if marker_dir:
    mp = Path(marker_dir)
    if mp.exists():
        print(f"markers ({marker_dir}):")
        for f in sorted(mp.iterdir()):
            print(f"  {f.name}")
PY
