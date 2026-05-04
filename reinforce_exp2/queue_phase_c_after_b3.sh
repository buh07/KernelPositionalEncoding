#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/jumbo/lisp/f004ndc/Kernel PE}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/results/reinforce_exp2}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/logs/reinforce_exp2}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-0}" # 0 = wait indefinitely

# Requested GPU policy: available GPUs are 0 and 2-7.
# We keep C2 on a stable map with llama on GPU0, and run C1 on separate devices.
C1_DEVICE_MAP="${C1_DEVICE_MAP:-llama-3.1-8b:cuda:2,olmo-2-7b:cuda:6}"
C1_FALLBACK_DEVICE_MAP="${C1_FALLBACK_DEVICE_MAP:-llama-3.1-8b:cuda:0,olmo-2-7b:cuda:6}"
C2_DEVICE_MAP="${C2_DEVICE_MAP:-llama-3.1-8b:cuda:0,olmo-2-7b:cuda:4,mistral-7b-v0.1:cuda:5}"

B3_OUT="$RESULTS_ROOT/B3_cluster_ablation"
C1_OUT="$RESULTS_ROOT/C1_olmo_causal_trace"
C2_OUT="$RESULTS_ROOT/C2_carrier_class_predictions"

mkdir -p "$LOG_ROOT/C1_olmo_causal_trace" "$LOG_ROOT/C2_carrier_class_predictions"

echo "[queue_c] $(date '+%Y-%m-%d %H:%M:%S %Z') waiting for B3 completion artifacts at: $B3_OUT"
echo "[queue_c] poll interval: ${POLL_SECONDS}s"
echo "[queue_c] C1 primary map: $C1_DEVICE_MAP"
echo "[queue_c] C2 map: $C2_DEVICE_MAP"

start_epoch="$(date +%s)"

b3_ready() {
  [[ -f "$B3_OUT/summary.json" ]] &&
  [[ -f "$B3_OUT/manifest.json" ]] &&
  [[ -f "$B3_OUT/claim_impact.json" ]] &&
  [[ -f "$B3_OUT/cluster_ablation_curve.parquet" ]] &&
  [[ -f "$B3_OUT/mechanism_disambiguation_verdict.json" ]]
}

b3_running() {
  pgrep -f "run_b3_cluster_ablation.py" >/dev/null 2>&1
}

while true; do
  if b3_ready; then
    echo "[queue_c] $(date '+%Y-%m-%d %H:%M:%S %Z') B3 artifacts detected; proceeding to C phase."
    break
  fi

  now_epoch="$(date +%s)"
  elapsed="$((now_epoch - start_epoch))"
  if [[ "$MAX_WAIT_SECONDS" -gt 0 && "$elapsed" -ge "$MAX_WAIT_SECONDS" ]]; then
    echo "[queue_c] ERROR: timed out waiting for B3 after ${elapsed}s."
    exit 2
  fi

  if ! b3_running; then
    echo "[queue_c] ERROR: B3 process not running and B3 artifacts are incomplete."
    echo "[queue_c] Missing at least one required B3 file in $B3_OUT."
    exit 2
  fi

  echo "[queue_c] $(date '+%Y-%m-%d %H:%M:%S %Z') still waiting (elapsed ${elapsed}s)..."
  sleep "$POLL_SECONDS"
done

calib_sha="$(
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
p = Path("/jumbo/lisp/f004ndc/Kernel PE/results/reinforce_exp2/calibration_splits/calibration_v1_manifest.json")
if p.exists():
    try:
        print(json.loads(p.read_text(encoding="utf-8")).get("sha256_ids_parquet", ""))
    except Exception:
        print("")
else:
    print("")
PY
)"

ts="$(date '+%Y-%m-%d_%H-%M-%S')"
c1_log="$LOG_ROOT/C1_olmo_causal_trace/${ts}__queued_after_b3.log"
c2_log="$LOG_ROOT/C2_carrier_class_predictions/${ts}__queued_after_b3.log"

run_c1() {
  local device_map="$1"
  local log_path="$2"
  local dep_json
  dep_json="$(printf '{"depends_on":["B3_cluster_ablation","B2b_cluster_alignment","B4_confound_isolation"],"trigger":"queued_after_b3","device_map":"%s","output_root":"%s"}' "$device_map" "$C1_OUT")"
  env \
    REINFORCE_EXP2_RUN_MODE="full" \
    REINFORCE_EXP2_DEPENDENCY_INPUTS_JSON="$dep_json" \
    REINFORCE_EXP2_RESUME_DECISION="run" \
    REINFORCE_EXP2_LOG_PATH="$log_path" \
    REINFORCE_EXP2_CALIBRATION_SPLIT_SHA256="$calib_sha" \
    "$PYTHON_BIN" "$ROOT_DIR/reinforce_exp2/scripts/run_c1_olmo_causal_trace.py" \
      --models "llama-3.1-8b,olmo-2-7b" \
      --device-map "$device_map" \
      --output-root "$C1_OUT"
}

run_c2() {
  local device_map="$1"
  local log_path="$2"
  local dep_json
  dep_json="$(printf '{"depends_on":["B3_cluster_ablation","B2b_cluster_alignment","B4_confound_isolation"],"trigger":"queued_after_b3","device_map":"%s","output_root":"%s"}' "$device_map" "$C2_OUT")"
  env \
    REINFORCE_EXP2_RUN_MODE="full" \
    REINFORCE_EXP2_DEPENDENCY_INPUTS_JSON="$dep_json" \
    REINFORCE_EXP2_RESUME_DECISION="run" \
    REINFORCE_EXP2_LOG_PATH="$log_path" \
    REINFORCE_EXP2_CALIBRATION_SPLIT_SHA256="$calib_sha" \
    "$PYTHON_BIN" "$ROOT_DIR/reinforce_exp2/scripts/run_c2_carrier_class_predictions.py" \
      --models "llama-3.1-8b,olmo-2-7b,mistral-7b-v0.1" \
      --device-map "$device_map" \
      --output-root "$C2_OUT"
}

echo "[queue_c] launching C1 + C2 in parallel."
echo "[queue_c] C1 log: $c1_log"
echo "[queue_c] C2 log: $c2_log"

set +e
run_c1 "$C1_DEVICE_MAP" "$c1_log" >"$c1_log" 2>&1 &
c1_pid=$!
run_c2 "$C2_DEVICE_MAP" "$c2_log" >"$c2_log" 2>&1 &
c2_pid=$!

wait "$c1_pid"
c1_rc=$?
wait "$c2_pid"
c2_rc=$?
set -e

if [[ "$c1_rc" -ne 0 ]]; then
  echo "[queue_c] C1 primary map failed (rc=$c1_rc). Attempting fallback map: $C1_FALLBACK_DEVICE_MAP"
  c1_log_fb="$LOG_ROOT/C1_olmo_causal_trace/${ts}__queued_after_b3_fallback.log"
  if run_c1 "$C1_FALLBACK_DEVICE_MAP" "$c1_log_fb" >"$c1_log_fb" 2>&1; then
    c1_rc=0
    c1_log="$c1_log_fb"
    echo "[queue_c] C1 fallback succeeded."
  else
    c1_rc=$?
    echo "[queue_c] C1 fallback failed (rc=$c1_rc)."
  fi
fi

echo "[queue_c] C1 rc=$c1_rc log=$c1_log"
echo "[queue_c] C2 rc=$c2_rc log=$c2_log"

if [[ "$c1_rc" -ne 0 || "$c2_rc" -ne 0 ]]; then
  echo "[queue_c] ERROR: one or more C tasks failed."
  exit 3
fi

echo "[queue_c] validating core artifacts for C1/C2."
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import sys
ROOT = Path("/jumbo/lisp/f004ndc/Kernel PE")
sys.path.insert(0, str(ROOT))
from reinforce_exp2.scripts._shared import validate_core_artifacts  # noqa: E402

checks = [
    (
        "C1_olmo_causal_trace",
        ROOT / "results/reinforce_exp2/C1_olmo_causal_trace",
        [
            ROOT / "results/reinforce_exp2/C1_olmo_causal_trace/causal_trace_matrix.parquet",
            ROOT / "results/reinforce_exp2/C1_olmo_causal_trace/restoration_ratio_summary.json",
            ROOT / "results/reinforce_exp2/C1_olmo_causal_trace/interaction_report.json",
        ],
    ),
    (
        "C2_carrier_class_predictions",
        ROOT / "results/reinforce_exp2/C2_carrier_class_predictions",
        [
            ROOT / "results/reinforce_exp2/C2_carrier_class_predictions/position_shuffle_results.parquet",
            ROOT / "results/reinforce_exp2/C2_carrier_class_predictions/long_context_results.parquet",
            ROOT / "results/reinforce_exp2/C2_carrier_class_predictions/carrier_class_report.json",
        ],
    ),
]

ok_all = True
for name, out_dir, tables in checks:
    ok, errs = validate_core_artifacts(out_dir, required_table_paths=tables)
    if ok:
        print(f"[queue_c] {name}: artifact validation PASSED")
    else:
        ok_all = False
        print(f"[queue_c] {name}: artifact validation FAILED")
        for e in errs:
            print(f"  - {e}")

if not ok_all:
    raise SystemExit(4)
PY

echo "[queue_c] $(date '+%Y-%m-%d %H:%M:%S %Z') phase C queued run finished successfully."
