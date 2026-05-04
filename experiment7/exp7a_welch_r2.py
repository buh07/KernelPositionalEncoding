#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from experiment4.common import now_timestamp, write_json

# Track-A models used by Exp7A at snapshot time.
MODEL_META: dict[str, dict[str, Any]] = {
    "gpt2-small": {"pe_kind": "absolute", "head_dim": 64, "rope_theta": 10000.0},
    "gpt2-medium": {"pe_kind": "absolute", "head_dim": 64, "rope_theta": 10000.0},
    "olmo-1b": {"pe_kind": "rope", "head_dim": 128, "rope_theta": 10000.0},
    "llama-3.2-1b": {"pe_kind": "rope", "head_dim": 64, "rope_theta": 500000.0},
    "tinyllama-1.1b": {"pe_kind": "rope", "head_dim": 64, "rope_theta": 10000.0},
    "tinyllama-nope-1.1b": {"pe_kind": "none", "head_dim": 64, "rope_theta": 10000.0},
}


def _extract_markdown_table(tracka_md_path: Path) -> pd.DataFrame:
    text = tracka_md_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    wanted = {"model", "dataset", "len", "early_mean_r2", "overall_mean_r2", "mean_pooled_r2"}

    for i in range(len(lines) - 2):
        line = lines[i].strip()
        sep = lines[i + 1].strip()
        if not (line.startswith("|") and sep.startswith("|")):
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        if not wanted.issubset(set(cols)):
            continue

        rows: list[list[str]] = []
        j = i + 2
        while j < len(lines):
            raw = lines[j].strip()
            if not raw.startswith("|"):
                break
            vals = [c.strip() for c in raw.strip("|").split("|")]
            if len(vals) != len(cols):
                break
            rows.append(vals)
            j += 1

        if not rows:
            continue
        df = pd.DataFrame(rows, columns=cols)
        for c in ["len", "early_mean_r2", "overall_mean_r2", "mean_pooled_r2"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["len", "overall_mean_r2"]).copy()
        df = df.rename(columns={"len": "seq_len"})
        df["seq_len"] = df["seq_len"].astype(int)
        return df

    raise RuntimeError(f"Could not locate Track A summary table in {tracka_md_path}")


def _rope_frequencies(head_dim: int, theta_base: float) -> np.ndarray:
    d2 = max(1, int(head_dim) // 2)
    idx = np.arange(d2, dtype=np.float64)
    base = float(theta_base) if float(theta_base) > 0 else 10000.0
    return base ** (-2.0 * idx / float(head_dim))


def _coherence_profile(*, head_dim: int, seq_len: int, pe_kind: str, rope_theta: float) -> pd.DataFrame:
    deltas = np.arange(1, max(2, int(seq_len)), dtype=np.int64)
    if str(pe_kind) == "none":
        mu = np.zeros_like(deltas, dtype=np.float64)
    else:
        freqs = _rope_frequencies(head_dim=head_dim, theta_base=rope_theta)
        # Proxy geometry profile: |mean_k exp(i * delta * freq_k)|
        ph = np.outer(deltas.astype(np.float64), freqs)
        mu = np.abs(np.exp(1j * ph).mean(axis=1))
    return pd.DataFrame({"delta": deltas, "mu_delta": mu})


def _welch_mu(head_dim: int, seq_len: int) -> float:
    d = float(max(1, int(head_dim)))
    n = float(max(2, int(seq_len)))
    return float(math.sqrt(max(0.0, (n - d) / (d * (n - 1.0)))))


def _perm_pvalue(x: np.ndarray, y: np.ndarray, *, n_perm: int = 20000, seed: int = 42) -> float:
    if len(x) < 3:
        return float("nan")
    rng = np.random.default_rng(int(seed))
    obs = float(np.corrcoef(x, y)[0, 1])
    n = len(x)
    c = 0
    for _ in range(int(n_perm)):
        yp = y[rng.permutation(n)]
        r = float(np.corrcoef(x, yp)[0, 1])
        if abs(r) >= abs(obs):
            c += 1
    return float((c + 1) / (int(n_perm) + 1))


def _safe_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    if len(x) < 3 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan"), float("nan"), float("nan"), float("nan")
    pear = scipy_stats.pearsonr(x, y)
    spe = scipy_stats.spearmanr(x, y)
    return float(pear.statistic), float(pear.pvalue), float(spe.statistic), float(spe.pvalue)


def _loo_loglinear_predict(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # log(y+eps)=a+b*log(x+eps), leave-one-model-out predictions.
    n = len(x)
    eps = 1e-8
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        xx = x[keep]
        yy = y[keep]
        if len(xx) < 3 or np.std(xx) <= 1e-12 or np.std(yy) <= 1e-12:
            continue
        X = np.column_stack([np.ones(len(xx)), np.log(xx + eps)])
        beta, *_ = np.linalg.lstsq(X, np.log(yy + eps), rcond=None)
        pred_log = beta[0] + beta[1] * np.log(x[i] + eps)
        out[i] = float(np.exp(pred_log))
    return out


def run_7a(
    *,
    output_root: Path,
    tracka_md_path: Path,
    bootstrap_seed: int = 42,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    out_dir = output_root / "exp7a_welch_r2"
    out_dir.mkdir(parents=True, exist_ok=True)

    track = _extract_markdown_table(tracka_md_path)

    # Model x seq coherence proxy profiles.
    profile_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    models = sorted(track["model"].astype(str).unique().tolist())
    seq_lens = sorted(int(v) for v in track["seq_len"].dropna().astype(int).unique().tolist())

    for model in models:
        meta = MODEL_META.get(model)
        if meta is None:
            raise KeyError(f"Missing MODEL_META entry for model={model}")
        head_dim = int(meta["head_dim"])
        pe_kind = str(meta["pe_kind"])
        rope_theta = float(meta["rope_theta"])

        for seq_len in seq_lens:
            prof = _coherence_profile(head_dim=head_dim, seq_len=seq_len, pe_kind=pe_kind, rope_theta=rope_theta)
            prof.insert(0, "seq_len", int(seq_len))
            prof.insert(0, "model", model)
            profile_rows.append(prof)

            mu1 = float(prof.loc[prof["delta"] == 1, "mu_delta"].iloc[0])
            mu_max = float(prof["mu_delta"].max())
            mu_w = _welch_mu(head_dim=head_dim, seq_len=seq_len)
            eta = float(mu_max / mu_w) if mu_w > 0 else float("nan")
            summary_rows.append(
                {
                    "model": model,
                    "seq_len": int(seq_len),
                    "head_dim": int(head_dim),
                    "pe_kind": pe_kind,
                    "rope_theta": float(rope_theta),
                    "mu1": mu1,
                    "mu_max": mu_max,
                    "mu_welch": float(mu_w),
                    "eta_welch_gap": eta,
                }
            )

    profile_df = pd.concat(profile_rows, ignore_index=True)
    sum_df = pd.DataFrame(summary_rows)
    profile_df.to_parquet(out_dir / "coherence_profiles.parquet", index=False)
    sum_df.to_parquet(out_dir / "welch_gap_summary.parquet", index=False)

    # Build row-level exploratory table (still pseudo-replicated across datasets/seq-lens).
    merged_rows = track.merge(sum_df[["model", "seq_len", "mu1", "mu_max", "eta_welch_gap"]], on=["model", "seq_len"], how="left")
    merged_rows["mu1_sq"] = merged_rows["mu1"].astype(float) ** 2

    # Primary model-level analysis (mitigates pseudo-replication).
    model_df = (
        merged_rows.groupby("model", as_index=False)
        .agg(
            mean_r2=("overall_mean_r2", "mean"),
            mean_eta=("eta_welch_gap", "mean"),
            mean_mu1=("mu1", "mean"),
            n_rows=("model", "size"),
        )
        .sort_values("mean_r2", ascending=False)
        .reset_index(drop=True)
    )

    x_model = model_df["mean_eta"].to_numpy(dtype=np.float64)
    y_model = model_df["mean_r2"].to_numpy(dtype=np.float64)
    p_r, p_p, s_r, s_p = _safe_corr(x_model, y_model)
    p_perm = _perm_pvalue(x_model, y_model, n_perm=max(2000, int(bootstrap_samples)), seed=int(bootstrap_seed))

    # Leave-one-model-out predictive check to avoid in-sample fit/eval leakage.
    y_pred_loo = _loo_loglinear_predict(x_model, y_model)
    model_df["predicted_r2_loo"] = y_pred_loo
    mask_pred = np.isfinite(y_pred_loo)
    lp_r, lp_p, ls_r, ls_p = _safe_corr(y_pred_loo[mask_pred], y_model[mask_pred]) if mask_pred.sum() >= 3 else (float("nan"), float("nan"), float("nan"), float("nan"))

    # Row-level exploratory (pseudo-replicated; reported as non-primary).
    x_row = merged_rows["eta_welch_gap"].to_numpy(dtype=np.float64)
    y_row = merged_rows["overall_mean_r2"].to_numpy(dtype=np.float64)
    rp_r, rp_p, rs_r, rs_p = _safe_corr(x_row, y_row)

    # Back-compatible per-row table for downstream references.
    row_table = merged_rows[[
        "model",
        "dataset",
        "seq_len",
        "early_mean_r2",
        "overall_mean_r2",
        "mean_pooled_r2",
        "mu1",
        "mu_max",
        "eta_welch_gap",
        "mu1_sq",
    ]].copy()

    # Add model-level LOO predictions down to row table (constant by model).
    pred_map = dict(zip(model_df["model"].astype(str), model_df["predicted_r2_loo"].astype(float)))
    row_table["predicted_r2"] = row_table["model"].map(pred_map).astype(float)
    row_table.to_parquet(out_dir / "r2_prediction_table.parquet", index=False)

    model_df.to_parquet(out_dir / "model_level_prediction_table.parquet", index=False)

    results = {
        "timestamp": now_timestamp(),
        "experiment": "7A",
        "status": "completed",
        "input_source": str(tracka_md_path),
        "geometry_source": "analytic_proxy_schedule_by_head_dim_theta",
        "analysis_scope": {
            "primary": "model_level",
            "exploratory": "row_level_pseudoreplicated",
        },
        "n_track_a_rows": int(len(merged_rows)),
        "n_models": int(model_df.shape[0]),
        "seq_lens": [int(x) for x in seq_lens],
        "fit_alpha": None,
        "prediction_vs_observed": {
            "pearson_r": lp_r,
            "pearson_p": lp_p,
            "spearman_rho": ls_r,
            "spearman_p": ls_p,
            "method": "leave_one_model_out_loglinear",
            "n_valid": int(mask_pred.sum()),
        },
        "mu1_vs_observed": {
            "pearson_r": _safe_corr(model_df["mean_mu1"].to_numpy(dtype=np.float64), y_model)[0],
            "pearson_p": _safe_corr(model_df["mean_mu1"].to_numpy(dtype=np.float64), y_model)[1],
            "spearman_rho": _safe_corr(model_df["mean_mu1"].to_numpy(dtype=np.float64), y_model)[2],
            "spearman_p": _safe_corr(model_df["mean_mu1"].to_numpy(dtype=np.float64), y_model)[3],
            "scope": "model_level",
        },
        "primary_model_level_correlation": {
            "pearson_r": p_r,
            "pearson_p": p_p,
            "spearman_rho": s_r,
            "spearman_p": s_p,
            "perm_p_two_sided": p_perm,
            "n_models": int(model_df.shape[0]),
        },
        "exploratory_row_level_correlation": {
            "pearson_r": rp_r,
            "pearson_p": rp_p,
            "spearman_rho": rs_r,
            "spearman_p": rs_p,
            "n_rows": int(len(merged_rows)),
            "warning": "row-level entries are pseudo-replicated across datasets/lengths",
        },
        "notes": [
            "Exp7A coherence features are analytic proxy geometry descriptors, not learned PE matrix extraction.",
            "Primary adjudication should use model-level statistics; row-level correlations are exploratory.",
        ],
        "artifacts": {
            "coherence_profiles": str(out_dir / "coherence_profiles.parquet"),
            "welch_gap_summary": str(out_dir / "welch_gap_summary.parquet"),
            "prediction_table": str(out_dir / "r2_prediction_table.parquet"),
            "model_level_prediction_table": str(out_dir / "model_level_prediction_table.parquet"),
        },
    }
    write_json(out_dir / "r2_prediction_results.json", results)

    write_json(
        out_dir / "run_manifest.json",
        {
            "timestamp": now_timestamp(),
            "experiment": "7A",
            "status": "completed",
            "track_a_source": str(tracka_md_path),
            "bootstrap_seed": int(bootstrap_seed),
            "bootstrap_samples": int(bootstrap_samples),
            "primary_scope": "model_level",
            "artifacts": {
                "coherence_profiles": str(out_dir / "coherence_profiles.parquet"),
                "welch_gap_summary": str(out_dir / "welch_gap_summary.parquet"),
                "r2_prediction_results": str(out_dir / "r2_prediction_results.json"),
                "prediction_table": str(out_dir / "r2_prediction_table.parquet"),
                "model_level_prediction_table": str(out_dir / "model_level_prediction_table.parquet"),
                "run_manifest": str(out_dir / "run_manifest.json"),
            },
        },
    )
    return results
