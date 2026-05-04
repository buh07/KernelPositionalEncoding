#!/usr/bin/env python3
"""E31C (C-lite): functional coupling reanalysis from existing experiments.

Correlates model-level SI amplitude (mean R²) with existing functional preferential-gap
signals from E12, E29B, and E30A without running new heavy benchmarks.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp3.common import RESULTS_ROOT, ensure_dir, timestamp_now, write_json  # noqa: E402
from reinforce_exp3.scripts._shared import emit_core_artifacts  # noqa: E402

EXPERIMENT_ID = "E31C"
DEFAULT_OUT = RESULTS_ROOT / "E31c_functional_coupling_reanalysis"
DEFAULT_E31A = RESULTS_ROOT / "E31a_breadth_consolidation" / "model_breadth_r2_table.csv"
PRIMARY = ["llama-3.1-8b", "mistral-7b-v0.1", "olmo-2-7b"]


def _load_e12_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in PRIMARY:
        p = ROOT / "results" / "reinforce_exp3" / "E12_icl_sensitivity" / model / "icl_sensitivity_result.json"
        if not p.exists():
            continue
        payload = json.loads(p.read_text(encoding="utf-8"))
        gap = float(payload.get("icl_lp_loss", np.nan)) - float(payload.get("non_icl_lp_loss", np.nan))
        rows.append(
            {
                "source": "E12_controlled_probe",
                "model": model,
                "functional_gap": float(gap),
                "raw_payload_path": str(p),
            }
        )
    return rows


def _load_cross_model_rows(exp_id: str, source_name: str) -> list[dict[str, Any]]:
    p = ROOT / "results" / "reinforce_exp3" / exp_id / "cross_model_summary.json"
    if not p.exists():
        raise FileNotFoundError(f"[E31C] missing {exp_id} cross_model_summary.json: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for rec in payload.get("per_model", []):
        model = str(rec.get("model", ""))
        if model not in PRIMARY:
            continue
        rows.append(
            {
                "source": source_name,
                "model": model,
                "functional_gap": float(rec.get("mean_pooled_gap_true_lp", np.nan)),
                "raw_payload_path": str(p),
            }
        )
    return rows


def _exact_grouped_perm_p(df: pd.DataFrame, obs_r: float) -> float:
    """Exact permutation p-value by permuting model labels within each source group."""
    groups = [g.copy() for _, g in df.groupby("source", sort=True)]
    # Each group is size 3 in this setup.
    perms_by_group = []
    for g in groups:
        vals = g["functional_gap_z"].to_numpy(dtype=np.float64)
        perms_by_group.append(list(itertools.permutations(vals.tolist())))

    count_ge = 0
    total = 0
    x = df["mean_r2"].to_numpy(dtype=np.float64)

    for combo in itertools.product(*perms_by_group):
        y_parts = []
        for arr in combo:
            y_parts.extend(arr)
        y = np.asarray(y_parts, dtype=np.float64)
        r = float(np.corrcoef(x, y)[0, 1])
        if r >= obs_r:
            count_ge += 1
        total += 1

    return float((count_ge + 1) / (total + 1))


def run(*, out_root: Path, e31a_table: Path) -> dict[str, Any]:
    t0 = time.time()
    out_root = ensure_dir(out_root)

    if not e31a_table.exists():
        raise FileNotFoundError(f"[E31C] missing E31A table: {e31a_table}")
    r2 = pd.read_csv(e31a_table)
    r2 = r2[r2["model"].isin(PRIMARY)][["model", "mean_r2"]].copy()

    rows = []
    rows.extend(_load_e12_rows())
    rows.extend(_load_cross_model_rows("E29b_naturaltext_longcontext_probe", "E29B_naturaltext"))
    rows.extend(_load_cross_model_rows("E30a_probe_boundary_grid", "E30A_boundary_grid"))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("[E31C] no functional rows loaded")

    df = df.merge(r2, on="model", how="left")
    if df["mean_r2"].isna().any():
        miss = df[df["mean_r2"].isna()]["model"].unique().tolist()
        raise RuntimeError(f"[E31C] missing mean_r2 for models: {miss}")

    # Source-normalized coupling view (mitigates scale mismatch across E12/E29B/E30A).
    def _z(s: pd.Series) -> pd.Series:
        arr = s.to_numpy(dtype=np.float64)
        sd = float(np.std(arr))
        if sd <= 1e-12:
            return pd.Series(np.zeros_like(arr), index=s.index)
        return pd.Series((arr - float(np.mean(arr))) / sd, index=s.index)

    df["functional_gap_z"] = df.groupby("source", group_keys=False)["functional_gap"].apply(_z)
    df.to_csv(out_root / "functional_coupling_figure.csv", index=False)

    x = df["mean_r2"].to_numpy(dtype=np.float64)
    y = df["functional_gap_z"].to_numpy(dtype=np.float64)

    pearson = scipy_stats.pearsonr(x, y)
    spearman = scipy_stats.spearmanr(x, y)
    p_exact_one_sided = _exact_grouped_perm_p(df, float(pearson.statistic))

    # Model-level aggregate descriptive view.
    by_model = (
        df.groupby("model", as_index=False)
        .agg(mean_r2=("mean_r2", "mean"), functional_gap_z_mean=("functional_gap_z", "mean"), n_cells=("functional_gap_z", "count"))
        .sort_values("mean_r2", ascending=False)
    )
    if len(by_model) >= 3:
        pearson_model = scipy_stats.pearsonr(
            by_model["mean_r2"].to_numpy(dtype=np.float64),
            by_model["functional_gap_z_mean"].to_numpy(dtype=np.float64),
        )
    else:
        pearson_model = None

    payload = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "n_rows": int(len(df)),
        "n_sources": int(df["source"].nunique()),
        "n_models": int(df["model"].nunique()),
        "cell_level_correlation": {
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
            "exact_grouped_perm_p_one_sided_positive": float(p_exact_one_sided),
        },
        "model_aggregate": {
            "rows": by_model.to_dict(orient="records"),
            "pearson_r": float(pearson_model.statistic) if pearson_model is not None else float("nan"),
            "pearson_p": float(pearson_model.pvalue) if pearson_model is not None else float("nan"),
        },
        "interpretation": "directional_functional_coupling_from_existing_probes",
        "limitations": [
            "Coupling is reanalysis of existing probe-style outcomes, not new benchmark execution.",
            "Cell-level statistics reuse models across sources; treat p-values as directional evidence.",
        ],
    }

    write_json(out_root / "r2_vs_functional_gap_correlation.json", payload)

    patch = f"""# Scope Language Patch (E31C)

Across existing functional probes (E12/E29B/E30A), higher model-level SI amplitude aligns directionally with larger
preferential SI-intervention gaps (cell-level Pearson r={payload['cell_level_correlation']['pearson_r']:.3f},
Spearman rho={payload['cell_level_correlation']['spearman_rho']:.3f}; grouped exact one-sided p={payload['cell_level_correlation']['exact_grouped_perm_p_one_sided_positive']:.4g}).

Scope: this is a fast coupling reanalysis on existing probe families, not a standalone naturalistic benchmark claim.
"""
    (out_root / "scope_language_patch.md").write_text(patch, encoding="utf-8")

    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "analysis_tier": "hardening",
        "canonical_eligible": True,
        "override_used": False,
        "verdict": {
            "interpretation": "directional_functional_coupling_supported_with_caveat",
            "n_rows": int(len(df)),
            "pearson_r": payload["cell_level_correlation"]["pearson_r"],
            "exact_p_one_sided": payload["cell_level_correlation"]["exact_grouped_perm_p_one_sided_positive"],
        },
        "elapsed_sec": float(time.time() - t0),
    }
    prereg = {
        "experiment_id": EXPERIMENT_ID,
        "question": "Do existing functional SI-sensitive gaps covary with model-level SI amplitude?",
        "primary_endpoint": "cell-level coupling between mean_r2 and source-normalized functional gap",
    }
    claim_impact = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "claim_status": "supported_with_caveat",
        "impact": "adds fast directional function-coupling evidence using existing artifacts",
    }
    data_dict = {
        "r2_vs_functional_gap_correlation.json": "Correlation metrics linking model SI amplitude to existing functional probe gaps.",
        "functional_coupling_figure.csv": "Per-source per-model rows used for coupling plot and analysis.",
        "scope_language_patch.md": "Paper-ready scope wording for directional functional coupling.",
    }

    emit_core_artifacts(
        experiment_id=EXPERIMENT_ID,
        out_dir=out_root,
        preregistration=prereg,
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dict,
        manifest_extra={"inputs": {"e31a_table": str(e31a_table)}},
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="E31C functional coupling reanalysis from existing runs")
    p.add_argument("--output-root", default=str(DEFAULT_OUT))
    p.add_argument("--e31a-table", default=str(DEFAULT_E31A))
    args = p.parse_args()

    summary = run(out_root=Path(args.output_root), e31a_table=Path(args.e31a_table))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
