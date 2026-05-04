#!/usr/bin/env python3
"""E30B — Local-bias null-family decomposition for SI metric validity.

Extends E28D by fitting multiple local-bias null families per head:
- exponential decay: a * exp(-b|Δ|) + c
- power-law decay: a * (|Δ| + 1)^(-b) + c
- piecewise local-window: in-window mean + out-window mean (best window)

Then compute R²_excess_family = max(0, R²_full - max(R²_local_family)).
Primary check: does bin-level disruption track R²_excess_family?
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

EXPERIMENT_ID = "E30B"
DEFAULT_OUT = RESULTS_ROOT / "E30b_localbias_null_family"
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
    raise FileNotFoundError(f"[E30B] No estimated_kernels.json for {model_name}")


def _load_r2_data(model_name: str) -> pd.DataFrame:
    p = ROOT / _R2_PATH.format(model=model_name)
    if not p.exists():
        raise FileNotFoundError(f"[E30B] Missing R² data: {p}")
    df = pd.read_parquet(p)
    required = {"layer", "head", "mean_r2"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"[E30B] Missing columns in R² data: {df.columns.tolist()}")
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


def _fit_exp_r2(y: np.ndarray) -> float:
    n = int(y.shape[0])
    x = np.arange(n, dtype=np.float64)
    best = -np.inf
    for b in np.linspace(0.0, 1.5, 81):
        phi = np.exp(-b * x)
        X = np.column_stack([phi, np.ones_like(phi)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        best = max(best, _r2(y, yhat))
    return float(best)


def _fit_powerlaw_r2(y: np.ndarray) -> float:
    n = int(y.shape[0])
    x = np.arange(n, dtype=np.float64)
    best = -np.inf
    for b in np.linspace(0.0, 4.0, 121):
        phi = np.power(x + 1.0, -b)
        X = np.column_stack([phi, np.ones_like(phi)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        best = max(best, _r2(y, yhat))
    return float(best)


def _fit_piecewise_window_r2(y: np.ndarray) -> float:
    n = int(y.shape[0])
    if n < 4:
        return float("nan")
    best = -np.inf
    for w in range(1, min(n - 1, 64) + 1):
        in_mask = np.arange(n) <= w
        out_mask = ~in_mask
        if not np.any(out_mask):
            continue
        in_mean = float(np.mean(y[in_mask]))
        out_mean = float(np.mean(y[out_mask]))
        yhat = np.where(in_mask, in_mean, out_mean)
        best = max(best, _r2(y, yhat))
    return float(best)


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
        y = np.asarray(g, dtype=np.float64)
        if y.size < 4:
            continue

        r2_exp = _fit_exp_r2(y)
        r2_pow = _fit_powerlaw_r2(y)
        r2_win = _fit_piecewise_window_r2(y)
        r2_local_best = float(np.nanmax([r2_exp, r2_pow, r2_win]))

        r2_full = float(r.mean_r2)
        r2_excess = max(0.0, r2_full - r2_local_best)

        rows.append(
            {
                "layer": int(r.layer),
                "head": int(r.head),
                "r2_full": r2_full,
                "r2_local_exp": float(r2_exp),
                "r2_local_powerlaw": float(r2_pow),
                "r2_local_window": float(r2_win),
                "r2_local_best": float(r2_local_best),
                "r2_excess_family": float(r2_excess),
            }
        )

    head_df = pd.DataFrame(rows)
    if head_df.empty:
        raise RuntimeError(f"[E30B] no head decomposition rows for {model_name}")
    head_df.to_parquet(model_dir / "head_r2_localbias_family_decomposition.parquet", index=False)

    assign_p = e28a_root / model_name / "head_bin_assignments.parquet"
    bins_p = e28a_root / model_name / "dose_response_bins.parquet"
    if not assign_p.exists() or not bins_p.exists():
        raise RuntimeError(
            f"[E30B] missing E28a artifacts for {model_name}: {assign_p} / {bins_p}. Run E28a first."
        )

    assign_df = pd.read_parquet(assign_p)
    bin_df = pd.read_parquet(bins_p)

    merged = head_df.merge(assign_df[["layer", "head", "bin"]], on=["layer", "head"], how="inner")
    if merged.empty:
        raise RuntimeError(f"[E30B] decomposition-assignment merge empty for {model_name}")

    bin_stats = (
        merged.groupby("bin", as_index=False)
        .agg(
            mean_r2_full=("r2_full", "mean"),
            mean_r2_local_best=("r2_local_best", "mean"),
            mean_r2_excess_family=("r2_excess_family", "mean"),
            n_heads=("r2_full", "count"),
        )
        .merge(bin_df[["bin", "per_head_true_delta", "mean_r2"]], on="bin", how="left")
    )

    x_full = bin_stats["mean_r2_full"].to_numpy(dtype=np.float64)
    x_excess = bin_stats["mean_r2_excess_family"].to_numpy(dtype=np.float64)
    y = bin_stats["per_head_true_delta"].to_numpy(dtype=np.float64)

    rho_full, p_full = scipy_stats.spearmanr(x_full, y)
    rho_excess, p_excess = scipy_stats.spearmanr(x_excess, y)

    bin_stats.to_parquet(model_dir / "bin_level_localbias_family_vs_disruption.parquet", index=False)

    rec = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "model": model_name,
        "n_bins": int(len(bin_stats)),
        "rho_full_vs_disruption": float(rho_full),
        "p_full_vs_disruption": float(p_full),
        "rho_excess_family_vs_disruption": float(rho_excess),
        "p_excess_family_vs_disruption": float(p_excess),
        "supports_nontrivial_excess_family_tracking": bool(
            np.isfinite(rho_excess) and rho_excess > 0 and p_excess < 0.10
        ),
    }
    write_json(model_dir / "summary.json", rec)
    return rec


def _finalize(models: list[str], out_root: Path) -> dict[str, Any]:
    rows = []
    for m in models:
        p = out_root / m / "summary.json"
        if not p.exists():
            raise RuntimeError(f"[E30B] hard_fail_reason: missing summary for {m}: {p}")
        rows.append(read_json(p))

    enforce_coverage_contract(
        experiment_id=EXPERIMENT_ID,
        observed_models=[r.get("model", "") for r in rows],
        required_models=models,
    )

    n_pass = sum(1 for r in rows if bool(r.get("supports_nontrivial_excess_family_tracking", False)))
    if n_pass == len(rows):
        interp = "localbias_family_excess_tracks_disruption_supported"
        status = "supported"
    elif n_pass >= 2:
        interp = "localbias_family_excess_tracks_disruption_supported_with_caveat"
        status = "supported_with_caveat"
    elif n_pass >= 1:
        interp = "localbias_family_excess_tracks_disruption_mixed"
        status = "mixed"
    else:
        interp = "localbias_family_excess_tracks_disruption_not_supported"
        status = "not_supported"

    cross = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "n_models": int(len(rows)),
        "n_model_pass": int(n_pass),
        "interpretation": interp,
        "claim_status": status,
        "per_model": rows,
    }
    write_json(out_root / "cross_model_summary.json", cross)

    prereg = {
        "experiment_id": EXPERIMENT_ID,
        "question": "Does SI-linked disruption tracking survive richer local-bias null families?",
        "primary_hypothesis": "Bin-level disruption tracks R²_excess over best local-bias family fit.",
        "primary_endpoints": [
            "rho_excess_family_vs_disruption",
            "p_excess_family_vs_disruption",
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
    }

    claim_impact = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "claim_status": status,
        "impacts": [
            "Directly strengthens or bounds the local-attention-bias alternative for the SI R² metric.",
            "Supports main-text interpretation of Result I as non-trivial SI-linked head-level structure.",
        ],
    }

    data_dictionary = {
        "head_r2_localbias_family_decomposition.parquet": {
            "description": "Per-head full R² and multi-family local-bias null fits.",
        },
        "bin_level_localbias_family_vs_disruption.parquet": {
            "description": "Bin-level link between local-bias-family excess R² and disruption cost.",
        },
    }

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "generated_at": timestamp_now(),
        "models": models,
        "output_root": str(out_root),
        "files": [
            "cross_model_summary.json",
            "summary.json",
            "manifest.json",
            "preregistration.json",
            "claim_impact.json",
            "data_dictionary.json",
        ],
    }

    emit_core_artifacts(
        experiment_id=EXPERIMENT_ID,
        out_dir=out_root,
        preregistration=prereg,
        manifest=manifest,
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )
    return cross


def main() -> None:
    p = argparse.ArgumentParser(description="E30B: local-bias null-family decomposition", allow_abbrev=False)
    p.add_argument("--models", default="all")
    p.add_argument("--output-root", default=str(DEFAULT_OUT))
    p.add_argument("--e28a-root", default=str(E28A_DEFAULT))
    p.add_argument("--finalize-only", action="store_true")
    p.add_argument("--no-finalize", action="store_true")
    args = p.parse_args()

    if args.finalize_only and args.no_finalize:
        raise RuntimeError("[E30B] hard_fail_reason: --finalize-only and --no-finalize are mutually exclusive")

    models = parse_models_arg(args.models, default=PRIMARY_MODELS)
    out_root = ensure_dir(Path(args.output_root))
    e28a_root = Path(args.e28a_root)

    if args.finalize_only:
        cross = _finalize(models=models, out_root=out_root)
        print(f"[E30B] Finalized from shard outputs. interpretation={cross['interpretation']}", flush=True)
        return

    for model in models:
        run_model(model_name=model, out_root=out_root, e28a_root=e28a_root)

    if args.no_finalize:
        print("[E30B] Shard run complete (no finalize).", flush=True)
        return

    cross = _finalize(models=models, out_root=out_root)
    print(
        f"[E30B] Done. interpretation={cross['interpretation']} "
        f"n_model_pass={cross['n_model_pass']}/{cross['n_models']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
