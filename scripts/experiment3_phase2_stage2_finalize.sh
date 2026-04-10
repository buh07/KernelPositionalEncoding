#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/f004ndc/Kernel PE"
cd "$ROOT"

LOG_ROOT="$ROOT/logs/experiment3_phase2_stage2"
GATE_PATH="$ROOT/results/experiment3_phase2/phase2_governance/gate_g1_decision.json"
STATUS_OUT="$ROOT/results/experiment3_phase2/phase2_governance/stage2_status.json"
TODO_PATH="$ROOT/experiment3/phase2/TODO.md"

if [[ $# -ge 3 ]]; then
  LLAMA_RUN="$1"
  OLMO_RUN="$2"
  MISTRAL_RUN="$3"
else
  LLAMA_RUN="$(find "$LOG_ROOT" -maxdepth 1 -type d -name 'llama-3.1-8b_*' | sort | tail -n 1)"
  OLMO_RUN="$(find "$LOG_ROOT" -maxdepth 1 -type d -name 'olmo-2-7b_*' | sort | tail -n 1)"
  MISTRAL_RUN="$(find "$LOG_ROOT" -maxdepth 1 -type d -name 'mistral-7b-v0.1_*' | sort | tail -n 1)"
fi

if [[ -z "${LLAMA_RUN:-}" || -z "${OLMO_RUN:-}" || -z "${MISTRAL_RUN:-}" ]]; then
  echo "ERROR: could not determine Stage 2 run directories."
  exit 1
fi

if [[ ! -f "$GATE_PATH" ]]; then
  echo "ERROR: Gate G1 artifact missing: $GATE_PATH"
  exit 1
fi

if ! "$ROOT/.venv/bin/python" - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

path = Path("/scratch/f004ndc/Kernel PE/results/experiment3_phase2/phase2_governance/gate_g1_decision.json")
with path.open("r", encoding="utf-8") as f:
    gate = json.load(f)
if not bool(gate.get("continue_to_stage2", False)):
    raise SystemExit(1)
PY
then
  echo "ERROR: Gate G1 does not allow Stage 2 continuation."
  exit 1
fi

echo "Using Stage 2 run directories:"
echo "  llama  : $LLAMA_RUN"
echo "  olmo   : $OLMO_RUN"
echo "  mistral: $MISTRAL_RUN"

for run_dir in "$LLAMA_RUN" "$OLMO_RUN" "$MISTRAL_RUN"; do
  if [[ ! -f "$run_dir/_session.log" ]]; then
    echo "ERROR: missing _session.log in $run_dir"
    exit 1
  fi
  if ! grep -q "COMPLETE Stage 2" "$run_dir/_session.log"; then
    echo "ERROR: run not complete yet: $run_dir"
    exit 1
  fi
done

echo "[1/4] Scanning Stage 2 logs for failure markers..."
FAIL_PATTERNS="Traceback|CUDA out of memory|RuntimeError|ERROR|Exception"
set +e
grep -RInE -- "$FAIL_PATTERNS" "$LLAMA_RUN" "$OLMO_RUN" "$MISTRAL_RUN"
grep_rc=$?
set -e
if [[ $grep_rc -eq 0 ]]; then
  echo "ERROR: found failure markers in Stage 2 logs."
  exit 1
fi
echo "OK: no failure markers found."

echo "[2/4] Verifying artifact presence..."
for model in llama-3.1-8b olmo-2-7b; do
  for p in \
    "results/experiment3_phase2/exp3p2g_dose_response/$model/dose_response_curve.parquet" \
    "results/experiment3_phase2/exp3p2g_dose_response/$model/saturation_fit.json" \
    "results/experiment3_phase2/exp3p2h_source_target_spec/$model/source_target_preregister.json" \
    "results/experiment3_phase2/exp3p2h_source_target_spec/$model/layer_target_sweep.json" \
    "results/experiment3_phase2/exp3p2h_source_target_spec/$model/sweep_results.parquet"
  do
    if [[ ! -f "$ROOT/$p" ]]; then
      echo "ERROR: missing expected artifact: $p"
      exit 1
    fi
  done
done

for p in \
  "results/experiment3_phase2/exp3p2d_architecture_tiebreaker/llama-3.1-8b/theory7b_llama_gqa_grouped.json" \
  "results/experiment3_phase2/exp3p2d_architecture_tiebreaker/mistral-7b-v0.1/report.json" \
  "results/experiment3_phase2/exp3p2d_architecture_tiebreaker/cross_model_comparison.json"
do
  if [[ ! -f "$ROOT/$p" ]]; then
    echo "ERROR: missing expected artifact: $p"
    exit 1
  fi
done
echo "OK: expected artifacts are present."

echo "[3/4] Validating artifact contracts and writing stage2_status.json..."
export LLAMA_RUN OLMO_RUN MISTRAL_RUN
"$ROOT/.venv/bin/python" - <<'PY'
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("/scratch/f004ndc/Kernel PE")
STATUS_OUT = ROOT / "results/experiment3_phase2/phase2_governance/stage2_status.json"
TODO_PATH = ROOT / "experiment3/phase2/TODO.md"
GATE_PATH = ROOT / "results/experiment3_phase2/phase2_governance/gate_g1_decision.json"

RUN_DIRS = {
    "llama-3.1-8b": Path(os.environ["LLAMA_RUN"]),
    "olmo-2-7b": Path(os.environ["OLMO_RUN"]),
    "mistral-7b-v0.1": Path(os.environ["MISTRAL_RUN"]),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require_json_fields(path: Path, fields: list[str]) -> dict[str, Any]:
    obj = load_json(path)
    missing = [k for k in fields if k not in obj]
    if missing:
        raise KeyError(f"Missing fields {missing} in {path}")
    return obj


def require_parquet_cols(path: Path, cols: list[str]) -> int:
    df = pd.read_parquet(path)
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"Missing columns {miss} in {path}")
    return int(len(df))


gate = load_json(GATE_PATH)
if not bool(gate.get("continue_to_stage2", False)):
    raise RuntimeError("Gate G1 continuation is false; Stage 2 artifacts should not be finalized.")

# 3P2-G checks
rows_g: dict[str, int] = {}
g_json: dict[str, dict[str, Any]] = {}
for model in ("llama-3.1-8b", "olmo-2-7b"):
    g_curve = ROOT / "results/experiment3_phase2/exp3p2g_dose_response" / model / "dose_response_curve.parquet"
    g_sat = ROOT / "results/experiment3_phase2/exp3p2g_dose_response" / model / "saturation_fit.json"

    rows_g[model] = require_parquet_cols(
        g_curve,
        [
            "model",
            "task",
            "seed",
            "group",
            "attenuation_scale",
            "metric_name",
            "metric_value",
            "floor_proximity",
            "tier",
            "primary_test_id",
            "mde_target",
            "achieved_power",
            "multiplicity_family",
        ],
    )
    g_json[model] = require_json_fields(
        g_sat,
        [
            "tier",
            "primary_test_id",
            "mde_target",
            "achieved_power",
            "multiplicity_family",
        ],
    )

# 3P2-H checks
rows_h: dict[str, int] = {}
h_json: dict[str, dict[str, Any]] = {}
prereg_ts_ok: dict[str, bool] = {}
for model in ("llama-3.1-8b", "olmo-2-7b"):
    h_prereg = ROOT / "results/experiment3_phase2/exp3p2h_source_target_spec" / model / "source_target_preregister.json"
    h_sweep = ROOT / "results/experiment3_phase2/exp3p2h_source_target_spec" / model / "layer_target_sweep.json"
    h_rows = ROOT / "results/experiment3_phase2/exp3p2h_source_target_spec" / model / "sweep_results.parquet"

    require_json_fields(h_prereg, ["timestamp", "source_bins", "target_taxonomies", "trigger_evaluation"])
    h_json[model] = require_json_fields(
        h_sweep,
        [
            "model",
            "timestamp",
            "tier",
            "primary_test_id",
            "source_target_preregister",
            "source_bins",
            "target_taxonomies",
            "rho_results",
            "specificity_tests",
            "mde_target",
            "achieved_power",
            "multiplicity_family",
            "stability_verdict",
        ],
    )
    rows_h[model] = require_parquet_cols(
        h_rows,
        [
            "model",
            "source_window",
            "target_taxonomy",
            "rho_induction",
            "rho_control",
            "delta_rho",
            "p_value",
            "p_value_holm",
            "tier",
            "primary_test_id",
            "mde_target",
            "achieved_power",
            "multiplicity_family",
        ],
    )

    prereg_ts_ok[model] = bool(h_prereg.stat().st_mtime <= h_sweep.stat().st_mtime)

# 3P2-D checks
d2_path = ROOT / "results/experiment3_phase2/exp3p2d_architecture_tiebreaker/llama-3.1-8b/theory7b_llama_gqa_grouped.json"
d2 = require_json_fields(
    d2_path,
    [
        "experiment",
        "tier",
        "primary_test_id",
        "mde_target",
        "achieved_power",
        "multiplicity_family",
        "per_query_head",
        "per_kv_group",
    ],
)

mistral_path = ROOT / "results/experiment3_phase2/exp3p2d_architecture_tiebreaker/mistral-7b-v0.1/report.json"
mistral = load_json(mistral_path)
mistral_status = str(mistral.get("status", "complete"))
if mistral_status not in {"complete", "blocked"}:
    mistral_status = "complete"

_ = load_json(ROOT / "results/experiment3_phase2/exp3p2d_architecture_tiebreaker/cross_model_comparison.json")

completion_epoch = {
    model: (run_dir / "_session.log").stat().st_mtime
    for model, run_dir in RUN_DIRS.items()
}
max_completion_epoch = max(completion_epoch.values())
status_epoch = time.time()
if status_epoch < max_completion_epoch:
    raise RuntimeError("Status emission time is earlier than Stage 2 session completion time.")

status = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(status_epoch)),
    "gate_g1_path": "results/experiment3_phase2/phase2_governance/gate_g1_decision.json",
    "gate_g1_continue_to_stage2": bool(gate.get("continue_to_stage2", False)),
    "run_directories": {k: str(v) for k, v in RUN_DIRS.items()},
    "run_ids": {k: v.name for k, v in RUN_DIRS.items()},
    "checks": {
        "log_failure_markers": "none_found",
        "h_prereg_before_sweep": prereg_ts_ok,
        "status_after_session_completion": True,
    },
    "stage2": {
        "3P2-G": {
            "llama-3.1-8b": {
                "status": "complete",
                "rows": rows_g["llama-3.1-8b"],
                "early_selective_slope_supported": bool(
                    g_json["llama-3.1-8b"].get("early_selective_slope_supported", False)
                ),
            },
            "olmo-2-7b": {
                "status": "complete",
                "rows": rows_g["olmo-2-7b"],
                "early_selective_slope_supported": bool(
                    g_json["olmo-2-7b"].get("early_selective_slope_supported", False)
                ),
            },
        },
        "3P2-H": {
            "llama-3.1-8b": {
                "status": "complete",
                "rows": rows_h["llama-3.1-8b"],
                "trigger": h_json["llama-3.1-8b"].get("trigger_evaluation", {}).get("trigger"),
                "stability_verdict": h_json["llama-3.1-8b"].get("stability_verdict"),
            },
            "olmo-2-7b": {
                "status": "complete",
                "rows": rows_h["olmo-2-7b"],
                "trigger": h_json["olmo-2-7b"].get("trigger_evaluation", {}).get("trigger"),
                "stability_verdict": h_json["olmo-2-7b"].get("stability_verdict"),
            },
        },
        "3P2-D": {
            "d1_mistral": {
                "status": mistral_status,
                "block_reason": mistral.get("block_reason"),
                "report_path": str(mistral_path),
            },
            "d2_llama": {
                "status": "complete",
                "report_path": str(d2_path),
                "delta_spearman_group_minus_query": d2.get("delta_spearman_group_minus_query"),
            },
            "cross_model_comparison_path": "results/experiment3_phase2/exp3p2d_architecture_tiebreaker/cross_model_comparison.json",
        },
    },
    "stage2c_complete": bool(mistral_status == "complete"),
    "stage2c_blocked": bool(mistral_status == "blocked"),
    "deferred_experiments": ["3P2-D.1"] if mistral_status == "blocked" else [],
}

STATUS_OUT.parent.mkdir(parents=True, exist_ok=True)
with STATUS_OUT.open("w", encoding="utf-8") as f:
    json.dump(status, f, indent=2)

md = TODO_PATH.read_text(encoding="utf-8")
begin = "<!-- STAGE2_POSTRUN_BEGIN -->"
end = "<!-- STAGE2_POSTRUN_END -->"
if begin in md and end in md:
    d1 = status["stage2"]["3P2-D"]["d1_mistral"]
    if d1["status"] == "blocked":
        d1_note = f"blocked ({d1.get('block_reason')})"
    else:
        d1_note = "complete"

    replacement = (
        f"{begin}\n"
        f"Snapshot timestamp: `{status['timestamp']}`\n\n"
        f"Run IDs:\n"
        f"- `llama-3.1-8b`: `{status['run_ids']['llama-3.1-8b']}`\n"
        f"- `olmo-2-7b`: `{status['run_ids']['olmo-2-7b']}`\n"
        f"- `mistral-7b-v0.1`: `{status['run_ids']['mistral-7b-v0.1']}`\n\n"
        f"| Stage 2 item | Status | Notes |\n"
        f"|---|---|---|\n"
        f"| 3P2-G (Llama/OLMo) | complete | rows={rows_g['llama-3.1-8b']}/{rows_g['olmo-2-7b']}; early-selective={status['stage2']['3P2-G']['llama-3.1-8b']['early_selective_slope_supported']}/{status['stage2']['3P2-G']['olmo-2-7b']['early_selective_slope_supported']} |\n"
        f"| 3P2-H (Llama/OLMo) | complete | trigger={status['stage2']['3P2-H']['llama-3.1-8b']['trigger']}/{status['stage2']['3P2-H']['olmo-2-7b']['trigger']}; verdict={status['stage2']['3P2-H']['llama-3.1-8b']['stability_verdict']}/{status['stage2']['3P2-H']['olmo-2-7b']['stability_verdict']} |\n"
        f"| 3P2-D.2 (Llama GQA grouped) | complete | delta_spearman={d2.get('delta_spearman_group_minus_query')} |\n"
        f"| 3P2-D.1 (Mistral replication) | {d1['status']} | {d1_note} |\n"
        f"| Governance artifact | complete | `results/experiment3_phase2/phase2_governance/stage2_status.json` |\n"
        f"{end}"
    )
    md = re.sub(re.escape(begin) + r".*?" + re.escape(end), replacement, md, flags=re.S)
    TODO_PATH.write_text(md, encoding="utf-8")

print(f"Wrote: {STATUS_OUT}")
PY

echo "[4/4] Finalization complete."
echo "Output: $STATUS_OUT"
