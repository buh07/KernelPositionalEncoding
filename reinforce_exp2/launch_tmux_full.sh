#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_SELF="$ROOT_DIR/reinforce_exp2/launch_tmux_full.sh"
SESSION_DEFAULT="reinforce_exp2_full"
PYTHON_DEFAULT="$ROOT_DIR/.venv/bin/python"
LOG_ROOT_DEFAULT="$ROOT_DIR/logs/reinforce_exp2"
PIPELINE_RUNS_DIR="$ROOT_DIR/results/reinforce_exp2/pipeline_runs"
B1_EXECUTION_MODE_DEFAULT="split_parallel"
B1_PARTIAL_ROOT_DEFAULT=""
B1_RESUME_PARTIALS_DEFAULT="true"
B3_EXECUTION_MODE_DEFAULT="split_parallel"
B3_PARTIAL_ROOT_DEFAULT=""
B3_RESUME_PARTIALS_DEFAULT="true"

WORKER=""
RUN_ID=""
SESSION="$SESSION_DEFAULT"
PYTHON_BIN="$PYTHON_DEFAULT"
LOG_ROOT="$LOG_ROOT_DEFAULT"
B1_EXECUTION_MODE="$B1_EXECUTION_MODE_DEFAULT"
B1_PARTIAL_ROOT="$B1_PARTIAL_ROOT_DEFAULT"
B1_RESUME_PARTIALS="$B1_RESUME_PARTIALS_DEFAULT"
B3_EXECUTION_MODE="$B3_EXECUTION_MODE_DEFAULT"
B3_PARTIAL_ROOT="$B3_PARTIAL_ROOT_DEFAULT"
B3_RESUME_PARTIALS="$B3_RESUME_PARTIALS_DEFAULT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --worker)
      WORKER="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --session)
      SESSION="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --log-root)
      LOG_ROOT="$2"
      shift 2
      ;;
    --b1-execution-mode)
      B1_EXECUTION_MODE="$2"
      shift 2
      ;;
    --b1-partial-root)
      B1_PARTIAL_ROOT="$2"
      shift 2
      ;;
    --b1-resume-partials)
      B1_RESUME_PARTIALS="$2"
      shift 2
      ;;
    --b3-execution-mode)
      B3_EXECUTION_MODE="$2"
      shift 2
      ;;
    --b3-partial-root)
      B3_PARTIAL_ROOT="$2"
      shift 2
      ;;
    --b3-resume-partials)
      B3_RESUME_PARTIALS="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="full_$(date +%Y%m%d_%H%M%S)"
fi

MARKER_DIR="$PIPELINE_RUNS_DIR/tmux_markers_${RUN_ID}"
EVENTS_LOG="$MARKER_DIR/events.log"
RUN_RECORD="$PIPELINE_RUNS_DIR/tmux_full_${RUN_ID}.json"

mkdir -p "$MARKER_DIR" "$LOG_ROOT" "$PIPELINE_RUNS_DIR"

record_event() {
  local stage="$1"
  local item="$2"
  local status="$3"
  local log_path="${4:-}"
  local message="${5:-}"
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  message="$(echo "$message" | tr '\t\n' '  ')"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$ts" "$stage" "$item" "$status" "$log_path" "$message" >> "$EVENTS_LOG"
}

write_run_record() {
  ROOT_DIR_ENV="$ROOT_DIR" RUN_ID_ENV="$RUN_ID" SESSION_ENV="$SESSION" EVENTS_LOG_ENV="$EVENTS_LOG" RUN_RECORD_ENV="$RUN_RECORD" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT_DIR_ENV"])
run_id = os.environ["RUN_ID_ENV"]
session = os.environ["SESSION_ENV"]
events_path = Path(os.environ["EVENTS_LOG_ENV"])
out_path = Path(os.environ["RUN_RECORD_ENV"])

rows = []
if events_path.exists():
    for line in events_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        ts, stage, item, status, log_path, msg = parts[:6]
        rows.append(
            {
                "timestamp": ts,
                "stage": stage,
                "item": item,
                "status": status,
                "log_path": log_path,
                "message": msg,
            }
        )

n_failures = sum(1 for r in rows if r.get("status") in {"failed", "dependency_failed"})
payload = {
    "timestamp": rows[-1]["timestamp"] if rows else None,
    "run_id": run_id,
    "session": session,
    "mode": "full",
    "strict_todo": True,
    "resume_policy": "strict_artifact_check",
    "log_root": str(root / "logs" / "reinforce_exp2"),
    "marker_dir": str(root / "results" / "reinforce_exp2" / "pipeline_runs" / f"tmux_markers_{run_id}"),
    "n_events": len(rows),
    "n_failures": int(n_failures),
    "rows": rows,
}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

patch_schema_flag() {
  local out_root="$1"
  local passed="$2"
  ROOT_DIR_ENV="$ROOT_DIR" "$PYTHON_BIN" - "$out_root" "$passed" <<'PY'
import os
import sys
from pathlib import Path

root = Path(os.environ["ROOT_DIR_ENV"])
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from reinforce_exp2.scripts._shared import patch_schema_validation_flag

out_dir = Path(sys.argv[1])
passed = str(sys.argv[2]).strip().lower() in {"1", "true", "yes", "y"}
patch_schema_validation_flag(out_dir, passed)
PY
}

mark_done() {
  local stage="$1"
  touch "$MARKER_DIR/${stage}.done"
  record_event "$stage" "$stage" "done" "" "stage complete"
  write_run_record
}

mark_failed() {
  local stage="$1"
  local msg="$2"
  echo "$msg" > "$MARKER_DIR/${stage}.failed"
  record_event "$stage" "$stage" "failed" "" "$msg"
  write_run_record
}

wait_for_stage() {
  local dep="$1"
  while true; do
    if [[ -f "$MARKER_DIR/${dep}.done" ]]; then
      return 0
    fi
    if compgen -G "$MARKER_DIR/*.failed" > /dev/null; then
      return 1
    fi
    sleep 5
  done
}

artifact_ready() {
  local exp_id="$1"
  local out_dir="$2"
  ROOT_DIR_ENV="$ROOT_DIR" "$PYTHON_BIN" - "$exp_id" "$out_dir" <<'PY'
import os
import sys
from pathlib import Path

root = Path(os.environ["ROOT_DIR_ENV"])
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from reinforce_exp2.scripts.run_pipeline import _artifact_ready

exp_id = str(sys.argv[1])
out_dir = Path(sys.argv[2])
ok, errs, _ = _artifact_ready(exp_id, out_dir)
if ok:
    raise SystemExit(0)
for e in errs:
    print(e)
raise SystemExit(1)
PY
}

run_task() {
  local stage="$1"
  local exp_id="$2"
  local out_root="$3"
  shift 3

  local ts
  ts="$(date '+%Y-%m-%d_%H-%M-%S')"
  local log_dir="$LOG_ROOT/$exp_id"
  local log_path="$log_dir/$ts.log"
  mkdir -p "$log_dir"

  if artifact_ready "$exp_id" "$out_root" >/dev/null 2>&1; then
    record_event "$stage" "$exp_id" "skipped_existing" "$log_path" "strict artifact check passed"
    write_run_record
    return 0
  fi

  record_event "$stage" "$exp_id" "running" "$log_path" "starting"
  write_run_record

  local dep_json
  dep_json="{\"stage\":\"$stage\",\"run_id\":\"$RUN_ID\",\"output_root\":\"$out_root\"}"
  local calib_sha=""
  local calib_manifest="$ROOT_DIR/results/reinforce_exp2/calibration_splits/calibration_v1_manifest.json"
  if [[ -f "$calib_manifest" ]]; then
    calib_sha="$("$PYTHON_BIN" - "$calib_manifest" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
try:
    payload = json.loads(p.read_text(encoding="utf-8"))
except Exception:
    payload = {}
print(str(payload.get("sha256_ids_parquet", "")).strip())
PY
)"
  fi

  set +e
  REINFORCE_EXP2_RUN_MODE=full \
  REINFORCE_EXP2_DEPENDENCY_INPUTS_JSON="$dep_json" \
  REINFORCE_EXP2_RESUME_DECISION=rerun_missing_or_invalid \
  REINFORCE_EXP2_LOG_PATH="$log_path" \
  REINFORCE_EXP2_CALIBRATION_SPLIT_SHA256="$calib_sha" \
  "$PYTHON_BIN" "$@" > "$log_path" 2>&1
  local rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    record_event "$stage" "$exp_id" "failed" "$log_path" "command exit code $rc"
    write_run_record
    return $rc
  fi

  if ! artifact_ready "$exp_id" "$out_root" >> "$log_path" 2>&1; then
    patch_schema_flag "$out_root" false
    record_event "$stage" "$exp_id" "failed" "$log_path" "post-run strict artifact validation failed"
    write_run_record
    return 3
  fi

  patch_schema_flag "$out_root" true
  record_event "$stage" "$exp_id" "ok" "$log_path" "completed"
  write_run_record
  return 0
}

run_preflight_stage() {
  local stage="preflight"
  run_task "$stage" "preflight" "$ROOT_DIR/results/reinforce_exp2/preflight" \
    "$ROOT_DIR/reinforce_exp2/scripts/run_preflight.py" \
    --output-root "$ROOT_DIR/results/reinforce_exp2/preflight" \
    --python-bin "$PYTHON_BIN" || return 1

  run_task "$stage" "calibration_v1" "$ROOT_DIR/results/reinforce_exp2/calibration_splits" \
    "$ROOT_DIR/reinforce_exp2/scripts/build_calibration_v1.py" \
    --output-root "$ROOT_DIR/results/reinforce_exp2/calibration_splits" \
    --model llama-3.1-8b || return 1

  mark_done "$stage"
}

run_phase_a_stage() {
  local stage="phase_a"
  if ! wait_for_stage "preflight"; then
    mark_failed "$stage" "dependency_failed: preflight"
    return 1
  fi

  run_task "$stage" "A0_claim_hygiene" "$ROOT_DIR/results/reinforce_exp2/A0_claim_hygiene" \
    "$ROOT_DIR/reinforce_exp2/scripts/run_a0_claim_hygiene.py" \
    --output-root "$ROOT_DIR/results/reinforce_exp2/A0_claim_hygiene" || return 1

  run_task "$stage" "A1_evidence_consolidation" "$ROOT_DIR/results/reinforce_exp2/A1_evidence_consolidation" \
    "$ROOT_DIR/reinforce_exp2/scripts/run_a1_evidence_consolidation.py" \
    --output-root "$ROOT_DIR/results/reinforce_exp2/A1_evidence_consolidation" || return 1

  run_task "$stage" "A2_prediction_layer" "$ROOT_DIR/results/reinforce_exp2/A2_prediction_layer" \
    "$ROOT_DIR/reinforce_exp2/scripts/run_a2_prediction_layer.py" \
    --output-root "$ROOT_DIR/results/reinforce_exp2/A2_prediction_layer" || return 1

  mark_done "$stage"
}

run_b1_split_parallel_stage() {
  local stage="$1"
  local b1_out="$ROOT_DIR/results/reinforce_exp2/B1_kernel_taxonomy"
  local b1_partial="$B1_PARTIAL_ROOT"
  if [[ -z "$b1_partial" ]]; then
    b1_partial="$b1_out/partials"
  fi

  local models=("llama-3.1-8b" "olmo-2-7b" "mistral-7b-v0.1")
  local devices=("cuda:0" "cuda:2" "cuda:3")
  local pids=()
  local metas=()

  local calib_sha=""
  local calib_manifest="$ROOT_DIR/results/reinforce_exp2/calibration_splits/calibration_v1_manifest.json"
  if [[ -f "$calib_manifest" ]]; then
    calib_sha="$("$PYTHON_BIN" - "$calib_manifest" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    payload = json.loads(p.read_text(encoding="utf-8"))
except Exception:
    payload = {}
print(str(payload.get("sha256_ids_parquet", "")).strip())
PY
)"
  fi

  for idx in "${!models[@]}"; do
    local model="${models[$idx]}"
    local dev="${devices[$idx]}"
    local ts
    ts="$(date '+%Y-%m-%d_%H-%M-%S')"
    local log_dir="$LOG_ROOT/B1_kernel_taxonomy"
    local log_path="$log_dir/${ts}__per_model_${model}.log"
    mkdir -p "$log_dir"

    record_event "$stage" "b1_per_model_start" "running" "$log_path" "model=$model device=$dev"
    write_run_record

    set +e
    REINFORCE_EXP2_RUN_MODE=full \
    REINFORCE_EXP2_RESUME_DECISION="internal_partial_resume" \
    REINFORCE_EXP2_LOG_PATH="$log_path" \
    REINFORCE_EXP2_CALIBRATION_SPLIT_SHA256="$calib_sha" \
    "$PYTHON_BIN" "$ROOT_DIR/reinforce_exp2/scripts/run_b1_kernel_taxonomy.py" \
      --execution-mode per_model \
      --single-model "$model" \
      --models "llama-3.1-8b,olmo-2-7b,mistral-7b-v0.1" \
      --output-root "$b1_out" \
      --partial-root "$b1_partial" \
      --resume-partials "$B1_RESUME_PARTIALS" \
      --calibration-root "$ROOT_DIR/results/reinforce_exp2/calibration_splits" \
      --device-map "$model:$dev" > "$log_path" 2>&1 &
    local pid=$!
    set -e
    pids+=("$pid")
    metas+=("$model|$log_path")
  done

  local rc_any=0
  for idx in "${!pids[@]}"; do
    local pid="${pids[$idx]}"
    local meta="${metas[$idx]}"
    local model="${meta%%|*}"
    local log_path="${meta#*|}"
    set +e
    wait "$pid"
    local rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
      rc_any=1
      record_event "$stage" "b1_per_model_end" "failed" "$log_path" "model=$model rc=$rc"
    else
      record_event "$stage" "b1_per_model_end" "ok" "$log_path" "model=$model rc=0"
    fi
    write_run_record
  done

  if [[ $rc_any -ne 0 ]]; then
    return 1
  fi

  local ts
  ts="$(date '+%Y-%m-%d_%H-%M-%S')"
  local agg_log="$LOG_ROOT/B1_kernel_taxonomy/${ts}__aggregate.log"
  record_event "$stage" "b1_aggregate_start" "running" "$agg_log" "aggregate finalize"
  write_run_record

  local dep_json
  dep_json="{\"stage\":\"$stage\",\"run_id\":\"$RUN_ID\",\"output_root\":\"$b1_out\",\"b1_substep\":\"aggregate\"}"

  set +e
  REINFORCE_EXP2_RUN_MODE=full \
  REINFORCE_EXP2_DEPENDENCY_INPUTS_JSON="$dep_json" \
  REINFORCE_EXP2_RESUME_DECISION=run \
  REINFORCE_EXP2_LOG_PATH="$agg_log" \
  REINFORCE_EXP2_CALIBRATION_SPLIT_SHA256="$calib_sha" \
  "$PYTHON_BIN" "$ROOT_DIR/reinforce_exp2/scripts/run_b1_kernel_taxonomy.py" \
    --execution-mode aggregate \
    --models "llama-3.1-8b,olmo-2-7b,mistral-7b-v0.1" \
    --output-root "$b1_out" \
    --partial-root "$b1_partial" \
    --resume-partials "$B1_RESUME_PARTIALS" \
    --calibration-root "$ROOT_DIR/results/reinforce_exp2/calibration_splits" > "$agg_log" 2>&1
  local rc_agg=$?
  set -e

  if [[ $rc_agg -ne 0 ]]; then
    record_event "$stage" "b1_aggregate_end" "failed" "$agg_log" "aggregate finalize failed rc=$rc_agg"
    write_run_record
    return 1
  fi

  if ! artifact_ready "B1_kernel_taxonomy" "$b1_out" >> "$agg_log" 2>&1; then
    patch_schema_flag "$b1_out" false
    record_event "$stage" "b1_aggregate_end" "failed" "$agg_log" "aggregate artifacts failed strict validation"
    write_run_record
    return 1
  fi
  patch_schema_flag "$b1_out" true
  record_event "$stage" "b1_aggregate_end" "ok" "$agg_log" "aggregate finalize complete"
  write_run_record
  return 0
}

run_b3_split_parallel_stage() {
  local stage="$1"
  local b3_out="$ROOT_DIR/results/reinforce_exp2/B3_cluster_ablation"
  local b3_partial="$B3_PARTIAL_ROOT"
  if [[ -z "$b3_partial" ]]; then
    b3_partial="$b3_out/partials"
  fi

  local models=("llama-3.1-8b" "olmo-2-7b" "mistral-7b-v0.1")
  local devices=("cuda:0" "cuda:2" "cuda:3")
  local pids=()
  local metas=()

  local calib_sha=""
  local calib_manifest="$ROOT_DIR/results/reinforce_exp2/calibration_splits/calibration_v1_manifest.json"
  if [[ -f "$calib_manifest" ]]; then
    calib_sha="$("$PYTHON_BIN" - "$calib_manifest" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    payload = json.loads(p.read_text(encoding="utf-8"))
except Exception:
    payload = {}
print(str(payload.get("sha256_ids_parquet", "")).strip())
PY
)"
  fi

  for idx in "${!models[@]}"; do
    local model="${models[$idx]}"
    local dev="${devices[$idx]}"
    local ts
    ts="$(date '+%Y-%m-%d_%H-%M-%S')"
    local log_dir="$LOG_ROOT/B3_cluster_ablation"
    local log_path="$log_dir/${ts}__per_model_${model}.log"
    mkdir -p "$log_dir"

    record_event "$stage" "b3_per_model_start" "running" "$log_path" "model=$model device=$dev"
    write_run_record

    set +e
    REINFORCE_EXP2_RUN_MODE=full \
    REINFORCE_EXP2_RESUME_DECISION="internal_partial_resume" \
    REINFORCE_EXP2_LOG_PATH="$log_path" \
    REINFORCE_EXP2_CALIBRATION_SPLIT_SHA256="$calib_sha" \
    "$PYTHON_BIN" "$ROOT_DIR/reinforce_exp2/scripts/run_b3_cluster_ablation.py" \
      --execution-mode per_model \
      --single-model "$model" \
      --models "llama-3.1-8b,olmo-2-7b,mistral-7b-v0.1" \
      --output-root "$b3_out" \
      --partial-root "$b3_partial" \
      --resume-partials "$B3_RESUME_PARTIALS" \
      --b1-root "$ROOT_DIR/results/reinforce_exp2/B1_kernel_taxonomy" \
      --b2a-root "$ROOT_DIR/results/reinforce_exp2/B2a_head_alignment" \
      --calibration-root "$ROOT_DIR/results/reinforce_exp2/calibration_splits" \
      --device-map "$model:$dev" > "$log_path" 2>&1 &
    local pid=$!
    set -e
    pids+=("$pid")
    metas+=("$model|$log_path")
  done

  local rc_any=0
  for idx in "${!pids[@]}"; do
    local pid="${pids[$idx]}"
    local meta="${metas[$idx]}"
    local model="${meta%%|*}"
    local log_path="${meta#*|}"
    set +e
    wait "$pid"
    local rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
      rc_any=1
      record_event "$stage" "b3_per_model_end" "failed" "$log_path" "model=$model rc=$rc"
    else
      record_event "$stage" "b3_per_model_end" "ok" "$log_path" "model=$model rc=0"
    fi
    write_run_record
  done

  if [[ $rc_any -ne 0 ]]; then
    return 1
  fi

  local ts
  ts="$(date '+%Y-%m-%d_%H-%M-%S')"
  local agg_log="$LOG_ROOT/B3_cluster_ablation/${ts}__aggregate.log"
  record_event "$stage" "b3_aggregate_start" "running" "$agg_log" "aggregate finalize"
  write_run_record

  local dep_json
  dep_json="{\"stage\":\"$stage\",\"run_id\":\"$RUN_ID\",\"output_root\":\"$b3_out\",\"b3_substep\":\"aggregate\"}"

  set +e
  REINFORCE_EXP2_RUN_MODE=full \
  REINFORCE_EXP2_DEPENDENCY_INPUTS_JSON="$dep_json" \
  REINFORCE_EXP2_RESUME_DECISION=run \
  REINFORCE_EXP2_LOG_PATH="$agg_log" \
  REINFORCE_EXP2_CALIBRATION_SPLIT_SHA256="$calib_sha" \
  "$PYTHON_BIN" "$ROOT_DIR/reinforce_exp2/scripts/run_b3_cluster_ablation.py" \
    --execution-mode aggregate \
    --models "llama-3.1-8b,olmo-2-7b,mistral-7b-v0.1" \
    --output-root "$b3_out" \
    --partial-root "$b3_partial" \
    --resume-partials "$B3_RESUME_PARTIALS" \
    --b1-root "$ROOT_DIR/results/reinforce_exp2/B1_kernel_taxonomy" \
    --b2a-root "$ROOT_DIR/results/reinforce_exp2/B2a_head_alignment" \
    --calibration-root "$ROOT_DIR/results/reinforce_exp2/calibration_splits" > "$agg_log" 2>&1
  local rc_agg=$?
  set -e

  if [[ $rc_agg -ne 0 ]]; then
    record_event "$stage" "b3_aggregate_end" "failed" "$agg_log" "aggregate finalize failed rc=$rc_agg"
    write_run_record
    return 1
  fi

  if ! artifact_ready "B3_cluster_ablation" "$b3_out" >> "$agg_log" 2>&1; then
    patch_schema_flag "$b3_out" false
    record_event "$stage" "b3_aggregate_end" "failed" "$agg_log" "aggregate artifacts failed strict validation"
    write_run_record
    return 1
  fi
  patch_schema_flag "$b3_out" true
  record_event "$stage" "b3_aggregate_end" "ok" "$agg_log" "aggregate finalize complete"
  write_run_record
  return 0
}

run_phase_b_core_stage() {
  local stage="phase_b_core"
  if ! wait_for_stage "preflight"; then
    mark_failed "$stage" "dependency_failed: preflight"
    return 1
  fi

  set +e
  run_task "$stage" "B2a_head_alignment" "$ROOT_DIR/results/reinforce_exp2/B2a_head_alignment" \
    "$ROOT_DIR/reinforce_exp2/scripts/run_b2a_head_alignment.py" \
    --models llama-3.1-8b,olmo-2-7b,mistral-7b-v0.1 \
    --output-root "$ROOT_DIR/results/reinforce_exp2/B2a_head_alignment" &
  local pid_b2a=$!

  if [[ "$B1_EXECUTION_MODE" == "split_parallel" ]]; then
    run_b1_split_parallel_stage "$stage" &
  else
    run_task "$stage" "B1_kernel_taxonomy" "$ROOT_DIR/results/reinforce_exp2/B1_kernel_taxonomy" \
      "$ROOT_DIR/reinforce_exp2/scripts/run_b1_kernel_taxonomy.py" \
      --execution-mode full \
      --models llama-3.1-8b,olmo-2-7b,mistral-7b-v0.1 \
      --output-root "$ROOT_DIR/results/reinforce_exp2/B1_kernel_taxonomy" \
      --partial-root "${B1_PARTIAL_ROOT:-$ROOT_DIR/results/reinforce_exp2/B1_kernel_taxonomy/partials}" \
      --resume-partials "$B1_RESUME_PARTIALS" \
      --calibration-root "$ROOT_DIR/results/reinforce_exp2/calibration_splits" \
      --device-map "llama-3.1-8b:cuda:0,olmo-2-7b:cuda:2,mistral-7b-v0.1:cuda:3" &
  fi
  local pid_b1=$!

  wait "$pid_b2a"; local rc1=$?
  wait "$pid_b1"; local rc2=$?
  set -e

  if [[ $rc1 -ne 0 || $rc2 -ne 0 ]]; then
    mark_failed "$stage" "B core failed (B2a rc=$rc1, B1 rc=$rc2)"
    return 1
  fi

  mark_done "$stage"
}

run_phase_b_adv_stage() {
  local stage="phase_b_adv"
  if ! wait_for_stage "phase_b_core"; then
    mark_failed "$stage" "dependency_failed: phase_b_core"
    return 1
  fi

  set +e
  run_task "$stage" "B2b_cluster_alignment" "$ROOT_DIR/results/reinforce_exp2/B2b_cluster_alignment" \
    "$ROOT_DIR/reinforce_exp2/scripts/run_b2b_cluster_alignment.py" \
    --models llama-3.1-8b,olmo-2-7b,mistral-7b-v0.1 \
    --output-root "$ROOT_DIR/results/reinforce_exp2/B2b_cluster_alignment" \
    --b1-root "$ROOT_DIR/results/reinforce_exp2/B1_kernel_taxonomy" \
    --b2a-root "$ROOT_DIR/results/reinforce_exp2/B2a_head_alignment" &
  local pid_b2b=$!

  run_task "$stage" "B4_confound_isolation" "$ROOT_DIR/results/reinforce_exp2/B4_confound_isolation" \
    "$ROOT_DIR/reinforce_exp2/scripts/run_b4_confound_isolation.py" \
    --models llama-3.1-8b,olmo-2-7b,mistral-7b-v0.1 \
    --output-root "$ROOT_DIR/results/reinforce_exp2/B4_confound_isolation" \
    --b1-root "$ROOT_DIR/results/reinforce_exp2/B1_kernel_taxonomy" \
    --calibration-root "$ROOT_DIR/results/reinforce_exp2/calibration_splits" \
    --device-map "llama-3.1-8b:cuda:0,olmo-2-7b:cuda:2,mistral-7b-v0.1:cuda:3" &
  local pid_b4=$!

  wait "$pid_b2b"; local rc1=$?
  wait "$pid_b4"; local rc2=$?
  set -e

  if [[ $rc1 -ne 0 || $rc2 -ne 0 ]]; then
    mark_failed "$stage" "B advanced pre-B3 failed (B2b rc=$rc1, B4 rc=$rc2)"
    return 1
  fi

  if [[ "$B3_EXECUTION_MODE" == "split_parallel" ]]; then
    run_b3_split_parallel_stage "$stage" || return 1
  else
    run_task "$stage" "B3_cluster_ablation" "$ROOT_DIR/results/reinforce_exp2/B3_cluster_ablation" \
      "$ROOT_DIR/reinforce_exp2/scripts/run_b3_cluster_ablation.py" \
      --execution-mode full \
      --models llama-3.1-8b,olmo-2-7b,mistral-7b-v0.1 \
      --output-root "$ROOT_DIR/results/reinforce_exp2/B3_cluster_ablation" \
      --partial-root "${B3_PARTIAL_ROOT:-$ROOT_DIR/results/reinforce_exp2/B3_cluster_ablation/partials}" \
      --resume-partials "$B3_RESUME_PARTIALS" \
      --b1-root "$ROOT_DIR/results/reinforce_exp2/B1_kernel_taxonomy" \
      --b2a-root "$ROOT_DIR/results/reinforce_exp2/B2a_head_alignment" \
      --calibration-root "$ROOT_DIR/results/reinforce_exp2/calibration_splits" \
      --device-map "llama-3.1-8b:cuda:0,olmo-2-7b:cuda:2,mistral-7b-v0.1:cuda:3" || return 1
  fi

  mark_done "$stage"
  touch "$MARKER_DIR/B.done"
  ROOT_DIR_ENV="$ROOT_DIR" "$PYTHON_BIN" - "$RUN_ID" <<'PY'
import json
import os
import sys
from pathlib import Path
root = Path(os.environ["ROOT_DIR_ENV"])
run_id = sys.argv[1]
path = root / "results" / "reinforce_exp2" / "pipeline_runs" / "B_COMPLETE_full.marker"
payload = {"run_id": run_id, "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S")}
path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

run_phase_c_stage() {
  local stage="phase_c"
  if ! wait_for_stage "phase_b_adv"; then
    mark_failed "$stage" "dependency_failed: phase_b_adv"
    return 1
  fi

  # C1 and C2 run sequentially to preserve deterministic memory pressure.
  set +e
  run_task "$stage" "C1_olmo_causal_trace" "$ROOT_DIR/results/reinforce_exp2/C1_olmo_causal_trace" \
    "$ROOT_DIR/reinforce_exp2/scripts/run_c1_olmo_causal_trace.py" \
    --models llama-3.1-8b,olmo-2-7b \
    --device-map "llama-3.1-8b:cuda:0,olmo-2-7b:cuda:4" \
    --n-examples 3000 \
    --output-root "$ROOT_DIR/results/reinforce_exp2/C1_olmo_causal_trace"
  local rc1=$?

  run_task "$stage" "C2_carrier_class_predictions" "$ROOT_DIR/results/reinforce_exp2/C2_carrier_class_predictions" \
    "$ROOT_DIR/reinforce_exp2/scripts/run_c2_carrier_class_predictions.py" \
    --models llama-3.1-8b,olmo-2-7b,mistral-7b-v0.1 \
    --device-map "llama-3.1-8b:cuda:0,olmo-2-7b:cuda:5,mistral-7b-v0.1:cuda:6" \
    --output-root "$ROOT_DIR/results/reinforce_exp2/C2_carrier_class_predictions"
  local rc2=$?
  set -e

  if [[ $rc1 -ne 0 || $rc2 -ne 0 ]]; then
    mark_failed "$stage" "C phase failed (C1 rc=$rc1, C2 rc=$rc2)"
    return 1
  fi

  mark_done "$stage"
  touch "$MARKER_DIR/all.done"
  record_event "$stage" "pipeline" "ok" "" "all stages complete"
  write_run_record
}

run_monitor_stage() {
  local stage="monitor"
  while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] heartbeat run_id=$RUN_ID session=$SESSION"
    ls -1 "$MARKER_DIR" | sed 's/^/  marker: /' || true
    if [[ -f "$EVENTS_LOG" ]]; then
      echo "  recent events:"
      tail -n 12 "$EVENTS_LOG" | sed 's/^/    /'
    fi
    sleep 60
  done
}

run_worker() {
  case "$WORKER" in
    preflight)
      run_preflight_stage || { mark_failed "preflight" "preflight stage failed"; exit 1; }
      ;;
    phase_a)
      run_phase_a_stage || exit 1
      ;;
    phase_b_core)
      run_phase_b_core_stage || exit 1
      ;;
    phase_b_adv)
      run_phase_b_adv_stage || exit 1
      ;;
    phase_c)
      run_phase_c_stage || exit 1
      ;;
    monitor)
      run_monitor_stage
      ;;
    *)
      echo "Unknown worker: $WORKER" >&2
      exit 2
      ;;
  esac
}

if [[ -n "$WORKER" ]]; then
  run_worker
  exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already exists. Use reinforce_exp2/status_tmux.sh to inspect." >&2
  exit 0
fi

record_event "launcher" "session" "running" "" "creating tmux session"
write_run_record

tmux new-session -d -s "$SESSION" -n preflight "bash -lc 'cd \"$ROOT_DIR\" && \"$SCRIPT_SELF\" --worker preflight --run-id \"$RUN_ID\" --session \"$SESSION\" --log-root \"$LOG_ROOT\" --python \"$PYTHON_BIN\" --b1-execution-mode \"$B1_EXECUTION_MODE\" --b1-partial-root \"$B1_PARTIAL_ROOT\" --b1-resume-partials \"$B1_RESUME_PARTIALS\" --b3-execution-mode \"$B3_EXECUTION_MODE\" --b3-partial-root \"$B3_PARTIAL_ROOT\" --b3-resume-partials \"$B3_RESUME_PARTIALS\"'"
tmux new-window -t "$SESSION" -n phase_a "bash -lc 'cd \"$ROOT_DIR\" && \"$SCRIPT_SELF\" --worker phase_a --run-id \"$RUN_ID\" --session \"$SESSION\" --log-root \"$LOG_ROOT\" --python \"$PYTHON_BIN\" --b1-execution-mode \"$B1_EXECUTION_MODE\" --b1-partial-root \"$B1_PARTIAL_ROOT\" --b1-resume-partials \"$B1_RESUME_PARTIALS\" --b3-execution-mode \"$B3_EXECUTION_MODE\" --b3-partial-root \"$B3_PARTIAL_ROOT\" --b3-resume-partials \"$B3_RESUME_PARTIALS\"'"
tmux new-window -t "$SESSION" -n phase_b_core "bash -lc 'cd \"$ROOT_DIR\" && \"$SCRIPT_SELF\" --worker phase_b_core --run-id \"$RUN_ID\" --session \"$SESSION\" --log-root \"$LOG_ROOT\" --python \"$PYTHON_BIN\" --b1-execution-mode \"$B1_EXECUTION_MODE\" --b1-partial-root \"$B1_PARTIAL_ROOT\" --b1-resume-partials \"$B1_RESUME_PARTIALS\" --b3-execution-mode \"$B3_EXECUTION_MODE\" --b3-partial-root \"$B3_PARTIAL_ROOT\" --b3-resume-partials \"$B3_RESUME_PARTIALS\"'"
tmux new-window -t "$SESSION" -n phase_b_adv "bash -lc 'cd \"$ROOT_DIR\" && \"$SCRIPT_SELF\" --worker phase_b_adv --run-id \"$RUN_ID\" --session \"$SESSION\" --log-root \"$LOG_ROOT\" --python \"$PYTHON_BIN\" --b1-execution-mode \"$B1_EXECUTION_MODE\" --b1-partial-root \"$B1_PARTIAL_ROOT\" --b1-resume-partials \"$B1_RESUME_PARTIALS\" --b3-execution-mode \"$B3_EXECUTION_MODE\" --b3-partial-root \"$B3_PARTIAL_ROOT\" --b3-resume-partials \"$B3_RESUME_PARTIALS\"'"
tmux new-window -t "$SESSION" -n phase_c "bash -lc 'cd \"$ROOT_DIR\" && \"$SCRIPT_SELF\" --worker phase_c --run-id \"$RUN_ID\" --session \"$SESSION\" --log-root \"$LOG_ROOT\" --python \"$PYTHON_BIN\" --b1-execution-mode \"$B1_EXECUTION_MODE\" --b1-partial-root \"$B1_PARTIAL_ROOT\" --b1-resume-partials \"$B1_RESUME_PARTIALS\" --b3-execution-mode \"$B3_EXECUTION_MODE\" --b3-partial-root \"$B3_PARTIAL_ROOT\" --b3-resume-partials \"$B3_RESUME_PARTIALS\"'"
tmux new-window -t "$SESSION" -n monitor "bash -lc 'cd \"$ROOT_DIR\" && \"$SCRIPT_SELF\" --worker monitor --run-id \"$RUN_ID\" --session \"$SESSION\" --log-root \"$LOG_ROOT\" --python \"$PYTHON_BIN\" --b1-execution-mode \"$B1_EXECUTION_MODE\" --b1-partial-root \"$B1_PARTIAL_ROOT\" --b1-resume-partials \"$B1_RESUME_PARTIALS\" --b3-execution-mode \"$B3_EXECUTION_MODE\" --b3-partial-root \"$B3_PARTIAL_ROOT\" --b3-resume-partials \"$B3_RESUME_PARTIALS\"'"

tmux select-window -t "$SESSION:monitor"

echo "Launched session '$SESSION' with run_id=$RUN_ID"
echo "Markers: $MARKER_DIR"
echo "Run record: $RUN_RECORD"
echo "Attach: tmux attach -t $SESSION"
