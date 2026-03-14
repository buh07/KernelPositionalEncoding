#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${1:-exp2_phase2b_fastq_v8_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${2:-results/experiment2}"
SUPERSEDE_RUN_ID="${3:-exp2_phase2b_fastq_20260307_121615}"
CHUNK_SIZE="${CHUNK_SIZE:-1}"
POLL_SECONDS="${POLL_SECONDS:-60}"
QUEUE_FILE="logs/experiment2/${RUN_ID}_phase2b_queue.tsv"

KERNEL_ENGINE="${KERNEL_ENGINE:-optimized}"
CENTERED_COMPUTE="${CENTERED_COMPUTE:-defer}"
SYNTHETIC_EVAL_MODE="${SYNTHETIC_EVAL_MODE:-restricted}"
CANDIDATE_SIZE="${CANDIDATE_SIZE:-10}"
FLOOR_THRESHOLD="${FLOOR_THRESHOLD:-0.15}"
MODEL_PROFILE="${MODEL_PROFILE:-legacy_1b}"
MODEL_ALLOWLIST="${MODEL_ALLOWLIST:-}"
H12_ENDPOINT_POLICY="${H12_ENDPOINT_POLICY:-raw_primary}"
TRACK_A_ENABLED="${TRACK_A_ENABLED:-true}"
INTERVENTION_PROFILE="${INTERVENTION_PROFILE:-full}"
BATCH_SIZE_SYNTH="${BATCH_SIZE_SYNTH:-24}"
BATCH_SIZE_TIER1="${BATCH_SIZE_TIER1:-24}"
EXEC_SYNTHETIC_COUNT="${EXEC_SYNTHETIC_COUNT:-200}"
EXEC_TIER1_COUNT="${EXEC_TIER1_COUNT:-100}"
POSTHOC_BATCH_SIZE_SYNTH="${POSTHOC_BATCH_SIZE_SYNTH:-24}"
POSTHOC_BATCH_SIZE_TIER1="${POSTHOC_BATCH_SIZE_TIER1:-24}"
PRUNE_FLOOR_LIMITED="${PRUNE_FLOOR_LIMITED:-1}"
PREPASS_PARALLEL="${PREPASS_PARALLEL:-1}"
SKIP_POSTHOC="${SKIP_POSTHOC:-0}"
ALLOW_CENTERED_PENDING_REANALYZE="${ALLOW_CENTERED_PENDING_REANALYZE:-0}"
CORE_SYNTH_ONLY="${CORE_SYNTH_ONLY:-0}"
GPU_LIST_RAW="${GPU_LIST:-0 1 2}"
GPU_LIST_NORMALIZED="${GPU_LIST_RAW//,/ }"
read -r -a GPUS <<< "$GPU_LIST_NORMALIZED"
if (( ${#GPUS[@]} == 0 )); then
  GPUS=(0 1 2)
fi
GPU_LIST="$(printf "%s " "${GPUS[@]}")"
GPU_LIST="${GPU_LIST% }"
GPU_CSV="$(IFS=,; echo "${GPUS[*]}")"
POSTHOC_GPUS="${POSTHOC_GPUS:-$GPU_CSV}"
POSTHOC_MAX_WORKERS="${POSTHOC_MAX_WORKERS:-${#GPUS[@]}}"

mkdir -p logs/experiment2

compute_seed_ranges() {
  local total_seeds="$1"
  local workers="$2"
  python - "$total_seeds" "$workers" <<'PY'
import sys
total = int(sys.argv[1])
workers = max(1, int(sys.argv[2]))
workers = min(workers, total)
base = total // workers
extra = total % workers
start = 0
for idx in range(workers):
    size = base + (1 if idx < extra else 0)
    end = start + size - 1
    print(f"{start}:{end}")
    start = end + 1
PY
}

# Stop legacy + queue workers before relaunch.
for gpu in "${GPUS[@]}"; do
  for sess in "exp2b_g${gpu}" "exp2q_g${gpu}" "exp2b_prepass_g${gpu}" "exp2b_posthoc_g${gpu}" "exp2p_g${gpu}"; do
    tmux kill-session -t "$sess" 2>/dev/null || true
  done
done
for sess in exp2b_g0 exp2b_g1 exp2b_g2 exp2q_g0 exp2q_g1 exp2q_g2 exp2b_prepass_g0 exp2b_prepass_g1 exp2b_prepass_g2 exp2b_posthoc_g0 exp2b_posthoc_g1 exp2b_posthoc_g2 exp2p_g0 exp2p_g1 exp2p_g2; do
  tmux kill-session -t "$sess" 2>/dev/null || true
done

# Mark superseded run for audit traceability.
if [[ -n "$SUPERSEDE_RUN_ID" && -d "$OUT_ROOT/phase2b/$SUPERSEDE_RUN_ID" ]]; then
  cat > "$OUT_ROOT/phase2b/$SUPERSEDE_RUN_ID/superseded_by_fixv8.json" <<JSON
{
  "superseded": true,
  "superseded_at": "$(date '+%Y-%m-%dT%H:%M:%S%z')",
  "superseded_by_run_id": "$RUN_ID",
  "note": "Superseded by Phase2B floor-prepass + manifest-pruned fast queue relaunch (v8)."
}
JSON
fi

# Build fresh phase2b full manifest.
CORE_ONLY_ARGS=()
if [[ "$CORE_SYNTH_ONLY" == "1" ]]; then
  CORE_ONLY_ARGS+=(--phase2b-core-synthetic-only)
fi

PYTHONPATH=. .venv/bin/python experiment2/run.py \
  --mode build \
  --phase phase2b \
  --model-profile "$MODEL_PROFILE" \
  --model-allowlist "$MODEL_ALLOWLIST" \
  --intervention-profile "$INTERVENTION_PROFILE" \
  "${CORE_ONLY_ARGS[@]}" \
  --device cuda \
  --run-id "$RUN_ID" \
  --output-root "$OUT_ROOT" \
  --print-summary | tee "logs/experiment2/${RUN_ID}_build_phase2b.log"

PHASE_DIR="$OUT_ROOT/phase2b/$RUN_ID"
FULL_MANIFEST="$PHASE_DIR/manifest.jsonl"
PRUNED_MANIFEST="$PHASE_DIR/manifest_pruned.jsonl"

# Precompute floor decisions from baseline rows only.
if [[ "$PREPASS_PARALLEL" == "1" ]]; then
  mapfile -t seed_ranges < <(compute_seed_ranges 7 "${#GPUS[@]}")
  for idx in "${!seed_ranges[@]}"; do
    gpu="${GPUS[$idx]}"
    IFS=: read -r s0 s1 <<< "${seed_ranges[$idx]}"
    tmux new-session -d -s "exp2b_prepass_g${gpu}" \
      "cd \"$ROOT_DIR\" && CUDA_VISIBLE_DEVICES=${gpu} PYTHONUNBUFFERED=1 .venv/bin/python experiment2/run.py --mode floor-prepass --model-profile \"$MODEL_PROFILE\" --model-allowlist \"$MODEL_ALLOWLIST\" --manifest \"$FULL_MANIFEST\" --device cuda --output-root \"$OUT_ROOT\" --synthetic-eval-mode \"$SYNTHETIC_EVAL_MODE\" --candidate-size \"$CANDIDATE_SIZE\" --h12-endpoint-policy \"$H12_ENDPOINT_POLICY\" --track-a-enabled \"$TRACK_A_ENABLED\" --intervention-profile \"$INTERVENTION_PROFILE\" --floor-threshold \"$FLOOR_THRESHOLD\" --batch-size-synth \"$BATCH_SIZE_SYNTH\" --batch-size-tier1 \"$BATCH_SIZE_TIER1\" --seed-start ${s0} --seed-end ${s1} --print-summary |& tee \"logs/experiment2/${RUN_ID}_g${gpu}_floor_prepass_seed${s0}${s1}.log\""
  done
  while true; do
    alive=0
    for gpu in "${GPUS[@]}"; do
      sess="exp2b_prepass_g${gpu}"
      if tmux has-session -t "$sess" 2>/dev/null; then
        alive=$((alive + 1))
      fi
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] floor-prepass-progress run_id=$RUN_ID workers_alive=$alive"
    if (( alive == 0 )); then
      break
    fi
    sleep "$POLL_SECONDS"
  done
else
  PYTHONPATH=. .venv/bin/python experiment2/run.py \
    --mode floor-prepass \
    --model-profile "$MODEL_PROFILE" \
    --model-allowlist "$MODEL_ALLOWLIST" \
    --manifest "$FULL_MANIFEST" \
    --device cuda \
    --output-root "$OUT_ROOT" \
    --synthetic-eval-mode "$SYNTHETIC_EVAL_MODE" \
    --candidate-size "$CANDIDATE_SIZE" \
    --h12-endpoint-policy "$H12_ENDPOINT_POLICY" \
    --track-a-enabled "$TRACK_A_ENABLED" \
    --intervention-profile "$INTERVENTION_PROFILE" \
    --floor-threshold "$FLOOR_THRESHOLD" \
    --batch-size-synth "$BATCH_SIZE_SYNTH" \
    --batch-size-tier1 "$BATCH_SIZE_TIER1" \
    --print-summary | tee "logs/experiment2/${RUN_ID}_floor_prepass.log"
fi

# Prune non-baseline floor-limited rows from execution manifest.
PYTHONPATH=. .venv/bin/python scripts/experiment2_phase2b_manifest_prune.py \
  --run-id "$RUN_ID" \
  --output-root "$OUT_ROOT" \
  --input-manifest "$FULL_MANIFEST" | tee "logs/experiment2/${RUN_ID}_manifest_prune.log"

if [[ ! -f "$PRUNED_MANIFEST" ]]; then
  echo "Missing pruned manifest: $PRUNED_MANIFEST" >&2
  exit 1
fi

# Launch execute queue workers using pruned manifest.
KERNEL_ENGINE="$KERNEL_ENGINE" \
CENTERED_COMPUTE="$CENTERED_COMPUTE" \
SYNTHETIC_EVAL_MODE="$SYNTHETIC_EVAL_MODE" \
CANDIDATE_SIZE="$CANDIDATE_SIZE" \
FLOOR_THRESHOLD="$FLOOR_THRESHOLD" \
BATCH_SIZE_SYNTH="$BATCH_SIZE_SYNTH" \
BATCH_SIZE_TIER1="$BATCH_SIZE_TIER1" \
SYNTHETIC_COUNT="$EXEC_SYNTHETIC_COUNT" \
TIER1_COUNT="$EXEC_TIER1_COUNT" \
H12_ENDPOINT_POLICY="$H12_ENDPOINT_POLICY" \
MODEL_PROFILE="$MODEL_PROFILE" \
MODEL_ALLOWLIST="$MODEL_ALLOWLIST" \
TRACK_A_ENABLED="$TRACK_A_ENABLED" \
INTERVENTION_PROFILE="$INTERVENTION_PROFILE" \
GPU_LIST="$GPU_LIST" \
PRUNE_FLOOR_LIMITED="$PRUNE_FLOOR_LIMITED" \
bash scripts/experiment2_phase2b_queue_launch.sh "$RUN_ID" "$OUT_ROOT" "$QUEUE_FILE" "$CHUNK_SIZE" "$PRUNED_MANIFEST"

# Monitor execute queue; restart missing workers while pending jobs exist.
PENDING_FILE="${QUEUE_FILE%.tsv}.pending.tsv"
FAIL_FILE="${QUEUE_FILE%.tsv}.failures.tsv"
DONE_FILE="${QUEUE_FILE%.tsv}.done.tsv"

while true; do
  pending=0
  done_count=0
  fail_count=0
  [[ -f "$PENDING_FILE" ]] && pending="$(wc -l < "$PENDING_FILE" | tr -d ' ')"
  [[ -f "$DONE_FILE" ]] && done_count="$(wc -l < "$DONE_FILE" | tr -d ' ')"
  [[ -f "$FAIL_FILE" ]] && fail_count="$(wc -l < "$FAIL_FILE" | tr -d ' ')"

  alive=0
  for gpu in $GPU_LIST; do
    if tmux has-session -t "exp2q_g${gpu}" 2>/dev/null; then
      alive=$((alive + 1))
    elif (( pending > 0 )); then
      tmux new-session -d -s "exp2q_g${gpu}" \
        "cd \"$ROOT_DIR\" && KERNEL_ENGINE=$KERNEL_ENGINE CENTERED_COMPUTE=$CENTERED_COMPUTE SYNTHETIC_EVAL_MODE=$SYNTHETIC_EVAL_MODE CANDIDATE_SIZE=$CANDIDATE_SIZE FLOOR_THRESHOLD=$FLOOR_THRESHOLD BATCH_SIZE_SYNTH=$BATCH_SIZE_SYNTH BATCH_SIZE_TIER1=$BATCH_SIZE_TIER1 SYNTHETIC_COUNT=$EXEC_SYNTHETIC_COUNT TIER1_COUNT=$EXEC_TIER1_COUNT H12_ENDPOINT_POLICY=$H12_ENDPOINT_POLICY MODEL_PROFILE=$MODEL_PROFILE MODEL_ALLOWLIST=$MODEL_ALLOWLIST TRACK_A_ENABLED=$TRACK_A_ENABLED INTERVENTION_PROFILE=$INTERVENTION_PROFILE PRUNE_FLOOR_LIMITED=$PRUNE_FLOOR_LIMITED bash scripts/experiment2_phase2b_queue_worker.sh ${gpu} \"$QUEUE_FILE\" \"$OUT_ROOT\""
      alive=$((alive + 1))
    fi
  done

  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] execute-progress run_id=$RUN_ID pending=$pending done_jobs=$done_count fail_jobs=$fail_count workers_alive=$alive"

  if (( pending == 0 && alive == 0 )); then
    break
  fi

  sleep "$POLL_SECONDS"
done

if [[ -f "$FAIL_FILE" ]] && [[ -s "$FAIL_FILE" ]]; then
  echo "Execute queue completed with failures. Review: $FAIL_FILE" >&2
  exit 1
fi

# Posthoc centered fill via lock-based queue + autoscale.
if [[ "$SKIP_POSTHOC" != "1" ]]; then
  POSTHOC_QUEUE_FILE="logs/experiment2/${RUN_ID}_phase2b_posthoc_queue.tsv"
  STRICT_POSTHOC=1 \
  DISABLE_EXAMPLE_CACHE=0 \
  POSTHOC_MAX_WORKERS="$POSTHOC_MAX_WORKERS" \
  BATCH_SIZE_SYNTH="$POSTHOC_BATCH_SIZE_SYNTH" \
  BATCH_SIZE_TIER1="$POSTHOC_BATCH_SIZE_TIER1" \
  MODEL_PROFILE="$MODEL_PROFILE" \
  MODEL_ALLOWLIST="$MODEL_ALLOWLIST" \
  bash scripts/experiment2_phase2b_posthoc_autoscale.sh "$RUN_ID" "$OUT_ROOT" "$POSTHOC_QUEUE_FILE" "$POSTHOC_GPUS" "$POLL_SECONDS"
else
  echo "Skipping posthoc-centered stage for run_id=$RUN_ID (SKIP_POSTHOC=1)"
fi

# Hard guard: do not run reanalysis while centered backlog remains.
PENDING_CENTERED="$(.venv/bin/python - <<'PY' "$OUT_ROOT" "$RUN_ID"
import glob, json, sys
out_root = sys.argv[1]
run_id = sys.argv[2]
files = glob.glob(f"{out_root}/phase2b/{run_id}/**/run_config.json", recursive=True)
pending = 0
for fp in files:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if bool(payload.get("centered_pending", False)):
            pending += 1
    except Exception:
        pass
print(pending)
PY
)"
if [[ "$PENDING_CENTERED" != "0" ]]; then
  if [[ "$ALLOW_CENTERED_PENDING_REANALYZE" == "1" ]]; then
    echo "Proceeding with reanalyze despite centered_pending=$PENDING_CENTERED (ALLOW_CENTERED_PENDING_REANALYZE=1)"
  else
    echo "Blocked reanalyze: centered_pending remains $PENDING_CENTERED for run_id=$RUN_ID" >&2
    exit 1
  fi
fi

# Reanalysis stage.
PYTHONPATH=. .venv/bin/python experiment2/run.py \
  --mode reanalyze \
  --model-profile "$MODEL_PROFILE" \
  --model-allowlist "$MODEL_ALLOWLIST" \
  --phase phase2b \
  --run-id "$RUN_ID" \
  --output-root "$OUT_ROOT" \
  --print-summary | tee "logs/experiment2/${RUN_ID}_reanalyze_phase2b.log"

echo "Pipeline complete for run_id=$RUN_ID"
