#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

LOG_DIR="$ROOT/logs/experiment3_phase2_chain/chain_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

log() {
  echo "[$(date)] $*" | tee -a "$LOG_DIR/chain.log"
}

wait_for_file() {
  local path="$1"
  local label="$2"
  local poll="${3:-120}"
  until [[ -f "$path" ]]; do
    log "waiting for $label: $path"
    sleep "$poll"
  done
  log "ready: $label"
}

log "chain start"

# Wait for all multiseed B outputs (Phase 2)
for s in 0 1 2 3 4 5 6; do
  wait_for_file "results/experiment3_phase2/exp3p2b_trivial_feature_control_multiseed/seed${s}/llama-3.1-8b/synthetic_boundary_results.json" "3P2-B seed${s}" 120
done
log "all 3P2-B multiseed artifacts present"

# Wait for pooled adjudication summary emitted by finalize watcher
wait_for_file "results/experiment3_phase2/exp3p2b_trivial_feature_control/llama-3.1-8b/multiseed_gate_summary.json" "3P2-B pooled adjudication" 60

# Launch 3P2-J long-span repair (Phase 4) unless already present
if [[ -f "results/experiment3_phase2/exp3p2j_conditional_regimes_longspan_repair/llama-3.1-8b/regime_summary.json" && \
      -f "results/experiment3_phase2/exp3p2j_conditional_regimes_longspan_repair/olmo-2-7b/regime_summary.json" && \
      -f "results/experiment3_phase2/exp3p2j_conditional_regimes_longspan_repair/llama-3.1-8b/regime_coverage_manifest.json" && \
      -f "results/experiment3_phase2/exp3p2j_conditional_regimes_longspan_repair/olmo-2-7b/regime_coverage_manifest.json" ]]; then
  log "3P2-J long-span repair artifacts already present; skipping launch"
else
  log "launching 3P2-J long-span repair"
  bash scripts/experiment3_phase2_j_longspan_launch_tmux.sh 160 16 64 256 | tee -a "$LOG_DIR/launch_j.log"
fi

wait_for_file "results/experiment3_phase2/exp3p2j_conditional_regimes_longspan_repair/llama-3.1-8b/regime_summary.json" "3P2-J llama summary" 180
wait_for_file "results/experiment3_phase2/exp3p2j_conditional_regimes_longspan_repair/olmo-2-7b/regime_summary.json" "3P2-J olmo summary" 180
wait_for_file "results/experiment3_phase2/exp3p2j_conditional_regimes_longspan_repair/llama-3.1-8b/regime_coverage_manifest.json" "3P2-J llama coverage" 60
wait_for_file "results/experiment3_phase2/exp3p2j_conditional_regimes_longspan_repair/olmo-2-7b/regime_coverage_manifest.json" "3P2-J olmo coverage" 60

# Ensure non-RoPE summary exists (Phase 5)
wait_for_file "results/experiment3_phase2/exp3p2k_non_rope_control/non_rope_control_summary.json" "non-RoPE summary" 120

# Launch tokenizer audit last (Phase 3) unless already present
if [[ -f "results/experiment3_phase2/exp3p2b_tokenizer_audit/llama-3.1-8b/tokenizer_audit_report.json" && \
      -f "results/experiment3_phase2/exp3p2b_tokenizer_audit/olmo-2-7b/tokenizer_audit_report.json" && \
      -f "results/experiment3_phase2/exp3p2b_tokenizer_audit/tokenizer_audit_report.json" ]]; then
  log "tokenizer audit artifacts already present; skipping launch"
else
  log "launching tokenizer audit"
  bash scripts/experiment3_phase2_tokenizer_audit_launch_tmux.sh 16 300 | tee -a "$LOG_DIR/launch_tokenizer_audit.log"
fi

wait_for_file "results/experiment3_phase2/exp3p2b_tokenizer_audit/llama-3.1-8b/tokenizer_audit_report.json" "tokenizer audit llama" 180
wait_for_file "results/experiment3_phase2/exp3p2b_tokenizer_audit/olmo-2-7b/tokenizer_audit_report.json" "tokenizer audit olmo" 180
wait_for_file "results/experiment3_phase2/exp3p2b_tokenizer_audit/tokenizer_audit_report.json" "tokenizer audit aggregate" 60

log "chain complete"
