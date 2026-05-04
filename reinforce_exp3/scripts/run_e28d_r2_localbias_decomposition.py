#!/usr/bin/env python3
"""E28d — R² metric decomposition against local-distance bias null.

For each head:
- R²_full from head_r2_summary (existing metric)
- R²_local from fitting kernel g(Δ) with exponential distance-decay null:
    g(Δ) ≈ a * exp(-b|Δ|) + c
- R²_excess = max(0, R²_full - R²_local)

Then test whether E28a/E26-style disruption tracks R²_excess at bin level.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp3.common import (  # noqa: E402
    PRIMARY_MODELS,
    RESULTS_ROOT,
    ensure_dir,
    timestamp_now,
    write_json,
)
from reinforce_exp3.scripts._shared import (  # noqa: E402
    emit_core_artifacts,
    enforce_coverage_contract,
    parse_models_arg,
    read_json,
)

EXPERIMENT_ID = "E28D"
DEFAULT_OUT = RESULTS_ROOT / "E28d_r2_localbias_decomposition"
E28A_DEFAULT = RESULTS_ROOT / "E28a_e26_20bin_exact"

_KERNEL_CANDIDATES = [
    "results/reinforce_exp/exp_r3_core_replication/{model}/theory8_position_ablation/{model}/estimated_kernels.json",
    "results/experiment3/theory8_position_ablation/{model}/estimated_kernels.json",
]
_R2_PATH = "results/experiment3/theory1_si_circuits/{model}/head_r2_summary.parquet"


def _load_kernels(model_name: str) -> dict[tuple[int, int], np.ndarray]:
    for template in _KERNEL_CANDIDATES:
        p = ROOT / template.format(model=model_name)
        if not p.exists():
            continue
        raw = json.loads(p.read_text(encoding="utf-8"))
        out: dict[tuple[int, int], np.ndarray] = {}
        for k, v in raw.items():
            if not (k.startswith("L") and "H" in k):
                continue
            left, right = k[1:].split("H", 1)
            out[(int(left), int(right))] = np.asarray(v, dtype=np.float64)
        if out:
            return out
    raise FileNotFoundError(f"[E28d] No estimated_kernels.json for {model_name}")


def _load_r2_data(model_name: str) -> pd.DataFrame:
    p = ROOT / _R2_PATH.format(model=model_name)
    if not p.exists():
        raise FileNotFoundError(f"[E28d] Missing R² data: {p}")
    df = pd.read_parquet(p)
    required = {"layer", "head", "mean_r2"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"[E28d] Missing columns in R² data: {df.columns.tolist()}")
    return df


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    if y.size == 0:
        return float("nan")
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    ss_res = float(np.sum((y - yhat) ** 2))
    return float(1.0 - ss_res / ss_tot)


def _fit_local_decay_r2(g: np.ndarray) -> float:
    y = np.asarray(g, dtype=np.float64)
    n = int(y.shape[0])
    if n < 4:
        return float("nan")

    x = np.arange(n, dtype=np.float64)
    best = -np.inf

    # Grid over b; solve linear least squares for a,c at each b.
    for b in np.linspace(0.0, 1.5, 81):
        phi = np.exp(-b * x)
        X = np.column_stack([phi, np.ones_like(phi)])
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            yhat = X @ beta
            r = _r2(y, yhat)
            if np.isfinite(r) and r > best:
                best = float(r)
        except Exception:
            continue

    if not np.isfinite(best):
        return float("nan")
    return float(max(min(best, 1.0), -1.0))


def run_model(*, model_name: str, out_root: Path, e28a_root: Path) -> dict[str, Any]:
    model_dir = ensure_dir(out_root / model_name)

    kernels = _load_kernels(model_name)
    r2_df = _load_r2_data(model_name).copy()
    r2_df["layer"] = r2_df["layer"].astype(int)
    r2_df["head"] = r2_df["head"].astype(int)

    rows: list[dict[str, Any]] = []
    for r in r2_df.itertuples(index=False):
        key = (int(r.layer), int(r.head))
        g = kernels.get(key)
        if g is None:
            continue
        r2_local = _fit_local_decay_r2(g)
        r2_full = float(r.mean_r2)
        r2_excess = max(0.0, r2_full - (r2_local if np.isfinite(r2_local) else 0.0))
        rows.append(
            {
                "layer": int(r.layer),
                "head": int(r.head),
                "r2_full": r2_full,
                "r2_local": float(r2_local),
                "r2_excess": float(r2_excess),
            }
        )

    head_df = pd.DataFrame(rows)
    if head_df.empty:
        raise RuntimeError(f"[E28d] no head decomposition rows for {model_name}")
    head_df.to_parquet(model_dir / "head_r2_decomposition.parquet", index=False)

    # Join E28a bin assignments and per-bin disruption.
    assign_p = e28a_root / model_name / "head_bin_assignments.parquet"
    bins_p = e28a_root / model_name / "dose_response_bins.parquet"
    if not assign_p.exists() or not bins_p.exists():
        raise RuntimeError(
            f"[E28d] missing E28a artifacts for {model_name}: {assign_p} / {bins_p}. Run E28a first."
        )

    assign_df = pd.read_parquet(assign_p)
    bin_df = pd.read_parquet(bins_p)

    merged = head_df.merge(assign_df[["layer", "head", "bin"]], on=["layer", "head"], how="inner")
    if merged.empty:
        raise RuntimeError(f"[E28d] decomposition-assignment merge empty for {model_name}")

    bin_stats = (
        merged.groupby("bin", as_index=False)
        .agg(
            mean_r2_full=("r2_full", "mean"),
            mean_r2_local=("r2_local", "mean"),
            mean_r2_excess=("r2_excess", "mean"),
            n_heads=("r2_full", "count"),
        )
        .merge(bin_df[["bin", "per_head_true_delta", "mean_r2"]], on="bin", how="left")
    )

    # Correlations against disruption.
    x_full = bin_stats["mean_r2_full"].to_numpy(dtype=np.float64)
    x_excess = bin_stats["mean_r2_excess"].to_numpy(dtype=np.float64)
    y = bin_stats["per_head_true_delta"].to_numpy(dtype=np.float64)

    rho_full, p_full = scipy_stats.spearmanr(x_full, y)
    rho_excess, p_excess = scipy_stats.spearmanr(x_excess, y)

    bin_stats.to_parquet(model_dir / "bin_level_decomposition_vs_disruption.parquet", index=False)

    rec = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "model": model_name,
        "n_bins": int(len(bin_stats)),
        "rho_full_vs_disruption": float(rho_full),
        "p_full_vs_disruption": float(p_full),
        "rho_excess_vs_disruption": float(rho_excess),
        "p_excess_vs_disruption": float(p_excess),
        "supports_nontrivial_excess_tracking": bool(np.isfinite(rho_excess) and rho_excess > 0 and p_excess < 0.10),
    }
    write_json(model_dir / "summary.json", rec)
    return rec


def _finalize(models: list[str], out_root: Path) -> dict[str, Any]:
    rows = []
    for m in models:
        p = out_root / m / "summary.json"
        if not p.exists():
            raise RuntimeError(f"[E28d] hard_fail_reason: missing summary for {m}: {p}")
        rows.append(read_json(p))

    enforce_coverage_contract(
        experiment_id=EXPERIMENT_ID,
        observed_models=[r.get("model", "") for r in rows],
        required_models=models,
    )

    n_pass = sum(1 for r in rows if bool(r.get("supports_nontrivial_excess_tracking", False)))
    if n_pass == len(rows):
        interp = "r2_excess_tracks_disruption_supported"
        status = "supported"
    elif n_pass >= 2:
        interp = "r2_excess_tracks_disruption_supported_with_caveat"
        status = "supported_with_caveat"
    elif n_pass >= 1:
        interp = "r2_excess_tracks_disruption_mixed"
        status = "mixed"
    else:
        interp = "r2_excess_tracks_disruption_not_supported"
        status = "not_supported"

    cross = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "n_models": int(len(rows)),
        "n_model_pass": int(n_pass),
        "interpretation": interp,
        "per_model": rows,
    }
    write_json(out_root / "cross_model_summary.json", cross)

    prereg = {
        "experiment_id": EXPERIMENT_ID,
        "question": "Does disruption track SI structure beyond simple local distance-decay bias?",
        "primary_hypothesis": "Bin-level disruption tracks R²_excess positively.",
        "primary_endpoints": [
            "rho_excess_vs_disruption",
            "p_excess_vs_disruption",
        ],
    }

    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "analysis_tier": "hardening",
        "canonical_eligible": True,
        "override_used": False,
        "verdict": {
            "interpretation": interp,
            "n_model_pass": int(n_pass),
            "n_models": int(len(rows)),
        },
        "limitations": [
            "R²_local is fitted on kernel profiles and is an approximation to local-bias explainability.",
            "Correlation uses bin-level disruption summaries from E28a, not fresh head-level interventions.",
        ],
    }

    claim_impact = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "claim_status": status,
        "supports_main_text": status in ("supported", "supported_with_caveat", "mixed"),
        "notes": [
            "Provides decomposition-based evidence on whether SI effects exceed simple local-distance decay.",
        ],
    }

    data_dictionary = {
        "experiment_id": EXPERIMENT_ID,
        "tables": [
            {
                "path": "<model>/head_r2_decomposition.parquet",
                "description": "Per-head decomposition into full/local/excess R².",
                "columns": [],
            },
            {
                "path": "<model>/bin_level_decomposition_vs_disruption.parquet",
                "description": "Bin-level means and disruption linkage used for correlations.",
                "columns": [],
            },
        ],
    }

    emit_core_artifacts(
        experiment_id=EXPERIMENT_ID,
        out_dir=out_root,
        preregistration=prereg,
        manifest_extra={"models": models},
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )
    return cross


def main() -> None:
    p = argparse.ArgumentParser(description="E28d: R² local-bias decomposition", allow_abbrev=False)
    p.add_argument("--models", default=",".join(PRIMARY_MODELS))
    p.add_argument("--output-root", default=str(DEFAULT_OUT))
    p.add_argument("--e28a-root", default=str(E28A_DEFAULT))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--finalize-only", action="store_true")
    p.add_argument("--no-finalize", action="store_true")
    args = p.parse_args()

    if args.finalize_only and args.no_finalize:
        raise RuntimeError("[E28d] hard_fail_reason: --finalize-only and --no-finalize are mutually exclusive")

    models = parse_models_arg(args.models, default=PRIMARY_MODELS)
    out_root = ensure_dir(Path(args.output_root))
    e28a_root = Path(args.e28a_root)

    if args.finalize_only:
        cross = _finalize(models, out_root)
        print(f"[E28d] Finalized. interpretation={cross['interpretation']}", flush=True)
        return

    for model_name in models:
        print(f"[E28d] Running decomposition for {model_name}", flush=True)
        run_model(model_name=model_name, out_root=out_root, e28a_root=e28a_root)

    if args.no_finalize:
        print("[E28d] Shard complete (no finalize).", flush=True)
        return

    cross = _finalize(models, out_root)
    print(f"[E28d] Done. interpretation={cross['interpretation']}", flush=True)


if __name__ == "__main__":
    main()
