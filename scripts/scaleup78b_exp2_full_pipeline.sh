#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <run_tag> <feas_run_id> [output_root]" >&2
  exit 2
fi

RUN_TAG="$1"
FEAS_RUN_ID="$2"
OUT_ROOT="${3:-results/experiment2}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_PROFILE="${MODEL_PROFILE:-scaleup_78b}"
MODEL_ALLOWLIST="${MODEL_ALLOWLIST:-}"
H12_ENDPOINT_POLICY="${H12_ENDPOINT_POLICY:-co_primary_raw_headroom}"
LONG_TASK_FEASIBILITY_POLICY="${LONG_TASK_FEASIBILITY_POLICY:-retrieval_fallback}"
FEAS_OFFSETS="${FEAS_OFFSETS:-8,12,16,24,32,48,64,96,128}"
LOCK_CANDIDATES="${LOCK_CANDIDATES:-8,12,16,24,32,48,64,96,128}"
FLOOR_THRESHOLD="${FLOOR_THRESHOLD:-0.15}"
BATCH_SIZE_SYNTH="${BATCH_SIZE_SYNTH:-24}"
BATCH_SIZE_TIER1="${BATCH_SIZE_TIER1:-24}"
POSTHOC_BATCH_SIZE_SYNTH="${POSTHOC_BATCH_SIZE_SYNTH:-24}"
POSTHOC_BATCH_SIZE_TIER1="${POSTHOC_BATCH_SIZE_TIER1:-24}"
P2A_SYNTHETIC_COUNT="${P2A_SYNTHETIC_COUNT:-50}"
POLL_SECONDS="${POLL_SECONDS:-120}"
P2A_PILOT_MODE="${P2A_PILOT_MODE:-floor-prepass}"
PHASE2B_SKIP_POSTHOC="${PHASE2B_SKIP_POSTHOC:-0}"
PHASE2B_ALLOW_CENTERED_PENDING_REANALYZE="${PHASE2B_ALLOW_CENTERED_PENDING_REANALYZE:-0}"
P2B_INTERVENTION_PROFILE="${P2B_INTERVENTION_PROFILE:-full}"
P2B_TRACK_A_ENABLED="${P2B_TRACK_A_ENABLED:-true}"
P2B_CORE_SYNTH_ONLY="${P2B_CORE_SYNTH_ONLY:-0}"
P2B_GPU_LIST="${P2B_GPU_LIST:-0 1 2}"
P2B_EXEC_SYNTHETIC_COUNT="${P2B_EXEC_SYNTHETIC_COUNT:-200}"
P2B_EXEC_TIER1_COUNT="${P2B_EXEC_TIER1_COUNT:-100}"
GPU_LIST_NORMALIZED="${P2B_GPU_LIST//,/ }"
read -r -a SCALEUP_GPUS <<< "$GPU_LIST_NORMALIZED"
if (( ${#SCALEUP_GPUS[@]} == 0 )); then
  SCALEUP_GPUS=(0 1 2)
fi

P2A_PILOT_RUN_ID="${P2A_PILOT_RUN_ID:-exp2_scaleup78b_p2a_pilot_${RUN_TAG}}"
P2A_RUN_ID="${P2A_RUN_ID:-exp2_scaleup78b_p2a_${RUN_TAG}}"
P2B_RUN_ID="${P2B_RUN_ID:-exp2_scaleup78b_p2b_${RUN_TAG}}"

LOG_DIR="logs/scaleup/${RUN_TAG}"
mkdir -p "$LOG_DIR" logs/experiment2
LOG_FILE="${LOG_DIR}/exp2_full_pipeline.log"
exec > >(tee -a "$LOG_FILE") 2>&1

compute_seed_ranges() {
  local max_seed="$1"
  local workers="$2"
  .venv/bin/python - <<'PY' "$max_seed" "$workers"
import sys
max_seed = int(sys.argv[1])
workers = max(1, int(sys.argv[2]))
total = max_seed + 1
workers = min(workers, total)
base = total // workers
rem = total % workers
start = 0
parts = []
for i in range(workers):
    size = base + (1 if i < rem else 0)
    end = start + size - 1
    parts.append(f"{start}:{end}")
    start = end + 1
print(" ".join(parts))
PY
}

wait_for_tmux_group() {
  local prefix="$1"
  shift
  local gpus=("$@")
  while true; do
    local alive=0
    for gpu in "${gpus[@]}"; do
      if tmux has-session -t "${prefix}_g${gpu}" 2>/dev/null; then
        alive=$((alive + 1))
      fi
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] wait_group=${prefix} workers_alive=${alive}"
    if (( alive == 0 )); then
      break
    fi
    sleep "$POLL_SECONDS"
  done
}

wait_for_sessions_gone() {
  local sessions=("$@")
  while true; do
    local alive=0
    local alive_names=()
    for s in "${sessions[@]}"; do
      if tmux has-session -t "$s" 2>/dev/null; then
        alive=$((alive + 1))
        alive_names+=("$s")
      fi
    done
    if (( alive == 0 )); then
      echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] wait_sessions_complete=true sessions=${sessions[*]}"
      break
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] wait_sessions_complete=false active=${alive_names[*]}"
    sleep "$POLL_SECONDS"
  done
}

kill_tmux_group() {
  local prefix="$1"
  shift
  local gpus=("$@")
  for gpu in "${gpus[@]}"; do
    tmux kill-session -t "${prefix}_g${gpu}" 2>/dev/null || true
  done
}

launch_feasibility_seed_shards() {
  local manifest="$1"
  local gpus=("${SCALEUP_GPUS[@]}")
  kill_tmux_group "scaleup78b_feas" "${gpus[@]}"
  local ranges_raw
  ranges_raw="$(compute_seed_ranges 2 "${#gpus[@]}")"
  read -r -a ranges <<< "$ranges_raw"
  local active_gpus=()
  for i in "${!ranges[@]}"; do
    local gpu="${gpus[$i]}"
    local r="${ranges[$i]}"
    local s0="${r%:*}"
    local s1="${r#*:}"
    local worker_log="${LOG_DIR}/scaleup78b_feas_g${gpu}.log"
    tmux new-session -d -s "scaleup78b_feas_g${gpu}" \
      "cd '$ROOT_DIR' && CUDA_VISIBLE_DEVICES=${gpu} PYTHONUNBUFFERED=1 .venv/bin/python experiment2/run.py --mode execute --model-profile '$MODEL_PROFILE' --model-allowlist '$MODEL_ALLOWLIST' --manifest '$manifest' --device cuda --output-root '$OUT_ROOT' --kernel-engine optimized --centered-compute defer --feasibility-task-only --synthetic-eval-mode restricted --candidate-size 10 --h12-endpoint-policy '$H12_ENDPOINT_POLICY' --floor-threshold '$FLOOR_THRESHOLD' --batch-size-synth '$BATCH_SIZE_SYNTH' --batch-size-tier1 '$BATCH_SIZE_TIER1' --seed-start '$s0' --seed-end '$s1' --print-summary |& tee '$worker_log'"
    active_gpus+=("$gpu")
  done
  wait_for_tmux_group "scaleup78b_feas" "${active_gpus[@]}"
}

launch_execute_seed_shards() {
  local prefix="$1"
  local manifest="$2"
  local floor_thr="$3"
  local synthetic_count="${4:-}"
  local gpus=("${SCALEUP_GPUS[@]}")
  kill_tmux_group "$prefix" "${gpus[@]}"
  local ranges_raw
  ranges_raw="$(compute_seed_ranges 6 "${#gpus[@]}")"
  read -r -a ranges <<< "$ranges_raw"
  local synth_opt=""
  if [[ -n "$synthetic_count" ]]; then
    synth_opt="--synthetic-count ${synthetic_count}"
  fi
  local active_gpus=()
  for i in "${!ranges[@]}"; do
    local gpu="${gpus[$i]}"
    local r="${ranges[$i]}"
    local s0="${r%:*}"
    local s1="${r#*:}"
    local worker_log="${LOG_DIR}/${prefix}_g${gpu}.log"
    tmux new-session -d -s "${prefix}_g${gpu}" \
      "cd '$ROOT_DIR' && CUDA_VISIBLE_DEVICES=${gpu} PYTHONUNBUFFERED=1 .venv/bin/python experiment2/run.py --mode execute --model-profile '$MODEL_PROFILE' --model-allowlist '$MODEL_ALLOWLIST' --manifest '$manifest' --device cuda --output-root '$OUT_ROOT' --kernel-engine optimized --centered-compute defer --synthetic-eval-mode restricted --candidate-size 10 --h12-endpoint-policy '$H12_ENDPOINT_POLICY' --floor-threshold '$floor_thr' --batch-size-synth '$BATCH_SIZE_SYNTH' --batch-size-tier1 '$BATCH_SIZE_TIER1' ${synth_opt} --seed-start '$s0' --seed-end '$s1' --prune-floor-limited-interventions --print-summary |& tee '$worker_log'"
    active_gpus+=("$gpu")
  done
  wait_for_tmux_group "$prefix" "${active_gpus[@]}"
}

launch_floor_prepass_seed_shards() {
  local prefix="$1"
  local manifest="$2"
  local floor_thr="$3"
  local synthetic_count="${4:-}"
  local gpus=("${SCALEUP_GPUS[@]}")
  kill_tmux_group "$prefix" "${gpus[@]}"
  local ranges_raw
  ranges_raw="$(compute_seed_ranges 6 "${#gpus[@]}")"
  read -r -a ranges <<< "$ranges_raw"
  local synth_opt=""
  if [[ -n "$synthetic_count" ]]; then
    synth_opt="--synthetic-count ${synthetic_count}"
  fi
  local active_gpus=()
  for i in "${!ranges[@]}"; do
    local gpu="${gpus[$i]}"
    local r="${ranges[$i]}"
    local s0="${r%:*}"
    local s1="${r#*:}"
    local worker_log="${LOG_DIR}/${prefix}_g${gpu}.log"
    tmux new-session -d -s "${prefix}_g${gpu}" \
      "cd '$ROOT_DIR' && CUDA_VISIBLE_DEVICES=${gpu} PYTHONUNBUFFERED=1 .venv/bin/python experiment2/run.py --mode floor-prepass --model-profile '$MODEL_PROFILE' --model-allowlist '$MODEL_ALLOWLIST' --manifest '$manifest' --device cuda --output-root '$OUT_ROOT' --synthetic-eval-mode restricted --candidate-size 10 --h12-endpoint-policy '$H12_ENDPOINT_POLICY' --floor-threshold '$floor_thr' --batch-size-synth '$BATCH_SIZE_SYNTH' --batch-size-tier1 '$BATCH_SIZE_TIER1' ${synth_opt} --seed-start '$s0' --seed-end '$s1' --print-summary |& tee '$worker_log'"
    active_gpus+=("$gpu")
  done
  wait_for_tmux_group "$prefix" "${active_gpus[@]}"
}

launch_posthoc_seed_shards() {
  local prefix="$1"
  local phase="$2"
  local run_id="$3"
  local gpus=("${SCALEUP_GPUS[@]}")
  kill_tmux_group "$prefix" "${gpus[@]}"
  local ranges_raw
  ranges_raw="$(compute_seed_ranges 6 "${#gpus[@]}")"
  read -r -a ranges <<< "$ranges_raw"
  local active_gpus=()
  for i in "${!ranges[@]}"; do
    local gpu="${gpus[$i]}"
    local r="${ranges[$i]}"
    local s0="${r%:*}"
    local s1="${r#*:}"
    local worker_log="${LOG_DIR}/${prefix}_g${gpu}.log"
    tmux new-session -d -s "${prefix}_g${gpu}" \
      "cd '$ROOT_DIR' && CUDA_VISIBLE_DEVICES=${gpu} PYTHONUNBUFFERED=1 .venv/bin/python experiment2/run.py --mode posthoc-centered --model-profile '$MODEL_PROFILE' --model-allowlist '$MODEL_ALLOWLIST' --phase '$phase' --run-id '$run_id' --device cuda --output-root '$OUT_ROOT' --batch-size-synth '$POSTHOC_BATCH_SIZE_SYNTH' --batch-size-tier1 '$POSTHOC_BATCH_SIZE_TIER1' --strict-posthoc --seed-start '$s0' --seed-end '$s1' --print-summary |& tee '$worker_log'"
    active_gpus+=("$gpu")
  done
  wait_for_tmux_group "$prefix" "${active_gpus[@]}"
}

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] scaleup-exp2-full-pipeline start run_tag=${RUN_TAG} feas_run_id=${FEAS_RUN_ID} model_profile=${MODEL_PROFILE} model_allowlist=${MODEL_ALLOWLIST:-all}"

stale_sessions=()
for gpu in "${SCALEUP_GPUS[@]}"; do
  stale_sessions+=("scaleup78b_g${gpu}")
done
wait_for_sessions_gone "${stale_sessions[@]}"

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] building feasibility manifest run_id=${FEAS_RUN_ID}"
PYTHONPATH=. .venv/bin/python experiment2/run.py \
  --mode feasibility-build \
  --model-profile "$MODEL_PROFILE" \
  --model-allowlist "$MODEL_ALLOWLIST" \
  --h12-endpoint-policy "$H12_ENDPOINT_POLICY" \
  --run-id "$FEAS_RUN_ID" \
  --output-root "$OUT_ROOT" \
  --device cuda \
  --feasibility-offsets "$FEAS_OFFSETS" \
  --lock-candidate-offsets "$LOCK_CANDIDATES" \
  --floor-threshold "$FLOOR_THRESHOLD" \
  --print-summary | tee "${LOG_DIR}/exp2_feas_build.log"

FEAS_MANIFEST="$OUT_ROOT/feasibility/$FEAS_RUN_ID/manifest.jsonl"
launch_feasibility_seed_shards "$FEAS_MANIFEST"

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] finalizing feasibility outputs run_id=${FEAS_RUN_ID}"
PYTHONPATH=. .venv/bin/python experiment2/run.py \
  --mode feasibility-finalize \
  --model-profile "$MODEL_PROFILE" \
  --model-allowlist "$MODEL_ALLOWLIST" \
  --run-id "$FEAS_RUN_ID" \
  --output-root "$OUT_ROOT" \
  --feasibility-offsets "$FEAS_OFFSETS" \
  --lock-candidate-offsets "$LOCK_CANDIDATES" \
  --long-task-feasibility-policy "$LONG_TASK_FEASIBILITY_POLICY" \
  --floor-threshold "$FLOOR_THRESHOLD" \
  --print-summary | tee "${LOG_DIR}/exp2_feas_finalize.log"

PYTHONPATH=. .venv/bin/python experiment2/run.py \
  --mode lock-long-offsets \
  --model-profile "$MODEL_PROFILE" \
  --model-allowlist "$MODEL_ALLOWLIST" \
  --output-root "$OUT_ROOT" \
  --sweep-run-id "$FEAS_RUN_ID" \
  --floor-threshold "$FLOOR_THRESHOLD" \
  --feasibility-offsets "$FEAS_OFFSETS" \
  --lock-candidate-offsets "$LOCK_CANDIDATES" \
  --long-task-feasibility-policy "$LONG_TASK_FEASIBILITY_POLICY" \
  --apply \
  --print-summary | tee "${LOG_DIR}/exp2_lock_long_offsets.log"

LOCK_STATUS="$(.venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('experiment2/long_offset_lock.json')
if not p.exists():
    print('missing')
else:
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        print('invalid')
    else:
        print(str(data.get('status', 'unknown')))
PY
)"
if [[ "$LOCK_STATUS" != "ok" ]]; then
  echo "Lock status is not ok (status=${LOCK_STATUS}); aborting full scale-up pipeline." >&2
  exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] building phase2a baseline-only pilot run_id=${P2A_PILOT_RUN_ID}"
PYTHONPATH=. .venv/bin/python experiment2/run.py \
  --mode build \
  --phase phase2a \
  --model-profile "$MODEL_PROFILE" \
  --model-allowlist "$MODEL_ALLOWLIST" \
  --run-id "$P2A_PILOT_RUN_ID" \
  --output-root "$OUT_ROOT" \
  --phase2a-baseline-only \
  --print-summary | tee "${LOG_DIR}/exp2_phase2a_pilot_build.log"

P2A_PILOT_MANIFEST="$OUT_ROOT/phase2a/$P2A_PILOT_RUN_ID/manifest.jsonl"
if [[ "$P2A_PILOT_MODE" == "floor-prepass" ]]; then
  launch_floor_prepass_seed_shards "scaleup78b_p2a_pilot" "$P2A_PILOT_MANIFEST" "$FLOOR_THRESHOLD" "$P2A_SYNTHETIC_COUNT"
else
  launch_execute_seed_shards "scaleup78b_p2a_pilot" "$P2A_PILOT_MANIFEST" "$FLOOR_THRESHOLD" "$P2A_SYNTHETIC_COUNT"
  PYTHONPATH=. .venv/bin/python experiment2/run.py \
    --mode reanalyze \
    --model-profile "$MODEL_PROFILE" \
    --model-allowlist "$MODEL_ALLOWLIST" \
    --phase phase2a \
    --run-id "$P2A_PILOT_RUN_ID" \
    --output-root "$OUT_ROOT" \
    --print-summary | tee "${LOG_DIR}/exp2_phase2a_pilot_reanalyze.log"
fi

PYTHONPATH=. .venv/bin/python experiment2/run.py \
  --mode calibrate-floor \
  --model-profile "$MODEL_PROFILE" \
  --model-allowlist "$MODEL_ALLOWLIST" \
  --run-id "$P2A_PILOT_RUN_ID" \
  --output-root "$OUT_ROOT" \
  --phase2a-floor-min-pass-rate 0.70 \
  --print-summary | tee "${LOG_DIR}/exp2_phase2a_pilot_floor_calibration.log"

SELECTED_FLOOR="$(.venv/bin/python - <<'PY' "$OUT_ROOT" "$P2A_PILOT_RUN_ID"
import json, sys
from pathlib import Path
out_root = Path(sys.argv[1])
run_id = sys.argv[2]
p = out_root / 'phase2a' / run_id / 'floor_recalibration.json'
if not p.exists():
    print('')
    raise SystemExit(0)
obj = json.loads(p.read_text(encoding='utf-8'))
val = obj.get('selected_floor_threshold')
print('' if val is None else str(val))
PY
)"
if [[ -z "$SELECTED_FLOOR" ]]; then
  echo "Phase 2A pilot failed floor recalibration; selected threshold is null. Aborting before full runs." >&2
  exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] phase2a full run with floor_threshold=${SELECTED_FLOOR} run_id=${P2A_RUN_ID}"
PYTHONPATH=. .venv/bin/python experiment2/run.py \
  --mode build \
  --phase phase2a \
  --model-profile "$MODEL_PROFILE" \
  --model-allowlist "$MODEL_ALLOWLIST" \
  --run-id "$P2A_RUN_ID" \
  --output-root "$OUT_ROOT" \
  --print-summary | tee "${LOG_DIR}/exp2_phase2a_build.log"

P2A_MANIFEST="$OUT_ROOT/phase2a/$P2A_RUN_ID/manifest.jsonl"
launch_execute_seed_shards "scaleup78b_p2a_exec" "$P2A_MANIFEST" "$SELECTED_FLOOR" "$P2A_SYNTHETIC_COUNT"
launch_posthoc_seed_shards "scaleup78b_p2a_posthoc" "phase2a" "$P2A_RUN_ID"

PYTHONPATH=. .venv/bin/python experiment2/run.py \
  --mode reanalyze \
  --model-profile "$MODEL_PROFILE" \
  --model-allowlist "$MODEL_ALLOWLIST" \
  --phase phase2a \
  --run-id "$P2A_RUN_ID" \
  --output-root "$OUT_ROOT" \
  --print-summary | tee "${LOG_DIR}/exp2_phase2a_reanalyze.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] launching phase2b full pipeline run_id=${P2B_RUN_ID} floor_threshold=${SELECTED_FLOOR}"
P2B_GPU_CSV="$(IFS=,; echo "${SCALEUP_GPUS[*]}")"
P2B_POSTHOC_WORKERS="${#SCALEUP_GPUS[@]}"
MODEL_PROFILE="$MODEL_PROFILE" \
MODEL_ALLOWLIST="$MODEL_ALLOWLIST" \
H12_ENDPOINT_POLICY="$H12_ENDPOINT_POLICY" \
FLOOR_THRESHOLD="$SELECTED_FLOOR" \
INTERVENTION_PROFILE="$P2B_INTERVENTION_PROFILE" \
TRACK_A_ENABLED="$P2B_TRACK_A_ENABLED" \
CORE_SYNTH_ONLY="$P2B_CORE_SYNTH_ONLY" \
GPU_LIST="$GPU_LIST_NORMALIZED" \
EXEC_SYNTHETIC_COUNT="$P2B_EXEC_SYNTHETIC_COUNT" \
EXEC_TIER1_COUNT="$P2B_EXEC_TIER1_COUNT" \
CHUNK_SIZE=1 \
BATCH_SIZE_SYNTH="$BATCH_SIZE_SYNTH" \
BATCH_SIZE_TIER1="$BATCH_SIZE_TIER1" \
POSTHOC_BATCH_SIZE_SYNTH="$POSTHOC_BATCH_SIZE_SYNTH" \
POSTHOC_BATCH_SIZE_TIER1="$POSTHOC_BATCH_SIZE_TIER1" \
POSTHOC_GPUS="$P2B_GPU_CSV" \
POSTHOC_MAX_WORKERS="$P2B_POSTHOC_WORKERS" \
SKIP_POSTHOC="$PHASE2B_SKIP_POSTHOC" \
ALLOW_CENTERED_PENDING_REANALYZE="$PHASE2B_ALLOW_CENTERED_PENDING_REANALYZE" \
bash scripts/experiment2_phase2b_pipeline_launch.sh "$P2B_RUN_ID" "$OUT_ROOT" "none" | tee "${LOG_DIR}/exp2_phase2b_pipeline.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] scaleup-exp2-full-pipeline complete run_tag=${RUN_TAG} p2a_run_id=${P2A_RUN_ID} p2b_run_id=${P2B_RUN_ID}"
