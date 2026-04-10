#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/f004ndc/Kernel PE"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
LOG_ROOT="$ROOT/logs/experiment3_refresh"

if [[ $# -ge 2 ]]; then
  LLAMA_RUN="$1"
  OLMO_RUN="$2"
else
  LLAMA_RUN="$(find "$LOG_ROOT" -maxdepth 1 -type d -name 'llama-3.1-8b_*' | sort | tail -n 1)"
  OLMO_RUN="$(find "$LOG_ROOT" -maxdepth 1 -type d -name 'olmo-2-7b_*' | sort | tail -n 1)"
fi

if [[ -z "${LLAMA_RUN:-}" || -z "${OLMO_RUN:-}" ]]; then
  echo "ERROR: could not determine latest refresh run directories."
  exit 1
fi

echo "Using run directories:"
echo "  llama: $LLAMA_RUN"
echo "  olmo : $OLMO_RUN"

for run_dir in "$LLAMA_RUN" "$OLMO_RUN"; do
  if ! grep -q "COMPLETE all steps" "$run_dir/_session.log"; then
    echo "ERROR: run not complete yet: $run_dir"
    exit 1
  fi
done

echo
echo "[1/5] Scanning refresh logs for failure signatures..."
FAIL_PATTERNS="Traceback|CUDA out of memory|RuntimeError|ERROR|Exception"
set +e
grep -RInE "$FAIL_PATTERNS" "$LLAMA_RUN" "$OLMO_RUN"
grep_rc=$?
set -e
if [[ $grep_rc -eq 0 ]]; then
  echo "ERROR: found failure markers in logs."
  exit 1
fi
echo "OK: no failure signatures found."

echo
echo "[2/5] Verifying Theory 3 forced recomputation (no Phase 1 skip)..."
for run_dir in "$LLAMA_RUN" "$OLMO_RUN"; do
  if grep -q "Phase 1: SKIPPED" "$run_dir/theory3.log"; then
    echo "ERROR: Theory 3 phase 1 was skipped in $run_dir/theory3.log"
    exit 1
  fi
done
echo "OK: Theory 3 phase 1 executed for both models."

echo
echo "[3/5] Artifact timestamps for affected theories..."
for theory in theory1_si_circuits theory3_crossterm theory5_subword_ablation theory5b_boundary_detection theory7b_activation_patching theory10_feeder_specificity; do
  for model in llama-3.1-8b olmo-2-7b; do
    json_path=""
    if [[ -f "$ROOT/results/experiment3/$theory/$model/analysis.json" ]]; then
      json_path="$ROOT/results/experiment3/$theory/$model/analysis.json"
    elif [[ -f "$ROOT/results/experiment3/$theory/$model/report.json" ]]; then
      json_path="$ROOT/results/experiment3/$theory/$model/report.json"
    fi
    if [[ -n "$json_path" ]]; then
      printf '%s | %s | %s | %s\n' "$theory" "$model" "$json_path" "$(date -r "$json_path" '+%Y-%m-%d %H:%M:%S %Z')"
    else
      printf '%s | %s | MISSING\n' "$theory" "$model"
    fi
  done
done

echo
echo "[4/5] Running Experiment 3 tests..."
"$PY" -m pytest -q tests/test_experiment3*

echo
echo "[5/5] Regenerating experiment3results.md..."
"$PY" experiment3/regenerate_results_report.py

echo
echo "Post-run validation complete."
