#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/f004ndc/Kernel PE"
cd "$ROOT"

LOG_ROOT="$ROOT/logs/experiment3_phase2_stage1"
GATE_OUT="$ROOT/results/experiment3_phase2/phase2_governance/gate_g1_decision.json"

if [[ $# -ge 2 ]]; then
  LLAMA_RUN="$1"
  OLMO_RUN="$2"
else
  LLAMA_RUN="$(find "$LOG_ROOT" -maxdepth 1 -type d -name 'llama-3.1-8b_*' | sort | tail -n 1)"
  OLMO_RUN="$(find "$LOG_ROOT" -maxdepth 1 -type d -name 'olmo-2-7b_*' | sort | tail -n 1)"
fi

if [[ -z "${LLAMA_RUN:-}" || -z "${OLMO_RUN:-}" ]]; then
  echo "ERROR: could not determine Stage 1 run directories."
  exit 1
fi

echo "Using Stage 1 run directories:"
echo "  llama: $LLAMA_RUN"
echo "  olmo : $OLMO_RUN"

for run_dir in "$LLAMA_RUN" "$OLMO_RUN"; do
  if ! grep -q "COMPLETE Stage 1" "$run_dir/_session.log"; then
    echo "ERROR: run not complete yet: $run_dir"
    exit 1
  fi
done

echo "[1/3] Scanning Stage 1 logs for failure markers..."
FAIL_PATTERNS="Traceback|CUDA out of memory|RuntimeError|ERROR|Exception"
set +e
grep -RInE "$FAIL_PATTERNS" "$LLAMA_RUN" "$OLMO_RUN"
grep_rc=$?
set -e
if [[ $grep_rc -eq 0 ]]; then
  echo "ERROR: found failure markers in Stage 1 logs."
  exit 1
fi
echo "OK: no failure markers found."

echo "[2/3] Verifying Stage 1 artifact presence..."
for model in llama-3.1-8b olmo-2-7b; do
  for p in \
    "results/experiment3_phase2/exp3p2e_t5_t5b_reconciliation/$model/t5_t5b_reconciliation.json" \
    "results/experiment3_phase2/exp3p2e_t5_t5b_reconciliation/$model/position_type_breakdown.parquet" \
    "results/experiment3_phase2/exp3p2e_t5_t5b_reconciliation/$model/boundary_mediation_analysis.json" \
    "results/experiment3_phase2/exp3p2f_proxy_decomposition/$model/proxy_decomposition.json" \
    "results/experiment3_phase2/exp3p2f_proxy_decomposition/$model/per_head_features.parquet" \
    "results/experiment3_phase2/exp3p2b_trivial_feature_control/$model/space_prefix_classifier.json" \
    "results/experiment3_phase2/exp3p2b_trivial_feature_control/$model/post_ablation_t5b_a.json" \
    "results/experiment3_phase2/exp3p2b_trivial_feature_control/$model/synthetic_boundary_results.json" \
    "results/experiment3_phase2/exp3p2b_trivial_feature_control/$model/adversarial_sequences.parquet" \
    "results/experiment3_phase2/exp3p2b_trivial_feature_control/$model/offset_group_boundary_scores.json"
  do
    if [[ ! -f "$ROOT/$p" ]]; then
      echo "ERROR: missing expected artifact: $p"
      exit 1
    fi
  done
done
echo "OK: all expected Stage 1 artifacts are present."

echo "[3/3] Emitting Gate G1 decision artifact..."
"$ROOT/.venv/bin/python" - <<'PY'
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

ROOT = Path("/scratch/f004ndc/Kernel PE")
MODELS = ["llama-3.1-8b", "olmo-2-7b"]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def evaluate_model(model: str) -> dict[str, Any]:
    f_path = ROOT / "results/experiment3_phase2/exp3p2f_proxy_decomposition" / model / "proxy_decomposition.json"
    b_post_path = ROOT / "results/experiment3_phase2/exp3p2b_trivial_feature_control" / model / "post_ablation_t5b_a.json"
    b_syn_path = ROOT / "results/experiment3_phase2/exp3p2b_trivial_feature_control" / model / "synthetic_boundary_results.json"

    f_data = load_json(f_path)
    b_post = load_json(b_post_path)
    b_syn = load_json(b_syn_path)

    median_delta = as_float(f_data.get("delta_r2", {}).get("median"))
    ci_count = int(f_data.get("collapse_rule_evaluation", {}).get("ci_includes_zero_count", 0))
    proxy_collapse = bool(math.isfinite(median_delta) and median_delta < 0.02 and ci_count >= 2)

    d_val = as_float(
        b_post.get("high_vs_low_attn_to_prev_last", {}).get("cohens_d")
    )
    prefix_following = bool(b_syn.get("prefix_following_artifact_flag", False))
    boundary_artifact = bool((math.isfinite(d_val) and d_val < 0.20) or prefix_following)

    return {
        "model": model,
        "proxy_decomposition_path": str(f_path),
        "post_ablation_t5b_a_path": str(b_post_path),
        "synthetic_boundary_results_path": str(b_syn_path),
        "proxy_collapse_eval": {
            "median_delta_r2": median_delta,
            "ci_includes_zero_count": ci_count,
            "rule_triggered": proxy_collapse,
        },
        "boundary_artifact_eval": {
            "cohens_d_post_control": d_val,
            "prefix_following_artifact_flag": prefix_following,
            "rule_triggered": boundary_artifact,
        },
    }


tier1_results = [evaluate_model(m) for m in MODELS]
proxy_collapse_global = all(x["proxy_collapse_eval"]["rule_triggered"] for x in tier1_results)
boundary_artifact_global = any(x["boundary_artifact_eval"]["rule_triggered"] for x in tier1_results)

deferred: list[str] = []
if proxy_collapse_global:
    deferred.extend(["3P2-G", "3P2-H", "3P2-J"])
if boundary_artifact_global:
    deferred.append("3P2-I")

continue_to_stage2 = not (proxy_collapse_global or boundary_artifact_global)

payload = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "tier1_results": tier1_results,
    "proxy_collapse_rule": {
        "definition": "median(delta_r2) < 0.02 and CI includes 0 for >=2 outcomes in both models",
        "triggered": proxy_collapse_global,
    },
    "boundary_artifact_rule": {
        "definition": "post-control d < 0.20 OR prefix-following artifact",
        "triggered": boundary_artifact_global,
    },
    "continue_to_stage2": continue_to_stage2,
    "deferred_experiments": deferred,
    "gating_note": "Emitted immediately after 3P2-B completion and before any Stage 2 launch.",
}

out_path = ROOT / "results/experiment3_phase2/phase2_governance/gate_g1_decision.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

print(f"Wrote: {out_path}")
print(f"continue_to_stage2={continue_to_stage2}")
print(f"deferred_experiments={deferred}")
PY

echo "Gate G1 finalization complete."
echo "Output: $GATE_OUT"
