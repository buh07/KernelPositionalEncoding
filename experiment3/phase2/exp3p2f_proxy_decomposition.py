#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


MODELS = ("llama-3.1-8b", "olmo-2-7b")
BOOTSTRAP_N = 1000
PERMUTATION_N = 400
BOOTSTRAP_SEED = 1729
PERMUTATION_SEED = 2718


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    yhat = np.asarray(y_pred, dtype=np.float64)
    if len(y) == 0 or not np.isfinite(y).all() or not np.isfinite(yhat).all():
        return float("nan")
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(np.sum((y - yhat) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _fit_ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    yhat = x @ beta
    return beta, yhat, _r2(y, yhat)


def _build_design(df: pd.DataFrame, include_r2: bool) -> np.ndarray:
    layer_dummies = pd.get_dummies(df["layer"].astype(int), prefix="layer", drop_first=True, dtype=float)
    cols = [
        "attention_entropy_proxy",
        "dominant_offset_proxy",
        "head_norm_proxy",
    ]
    cont = df[cols].astype(float).copy()
    for c in cols:
        if cont[c].isna().all():
            cont[c] = 0.0
        else:
            cont[c] = cont[c].fillna(float(cont[c].median()))

    parts = [layer_dummies.reset_index(drop=True), cont.reset_index(drop=True)]
    if include_r2:
        r2_col = df[["mean_r2"]].astype(float).copy()
        if r2_col["mean_r2"].isna().all():
            r2_col["mean_r2"] = 0.0
        else:
            r2_col["mean_r2"] = r2_col["mean_r2"].fillna(float(r2_col["mean_r2"].median()))
        parts.append(r2_col.reset_index(drop=True))

    x_body = pd.concat(parts, axis=1)
    x = np.column_stack([np.ones(len(df), dtype=np.float64), x_body.to_numpy(dtype=np.float64)])
    return x


def _bootstrap_delta_r2(df: pd.DataFrame, outcome_col: str, rng: np.random.RandomState) -> tuple[float, float]:
    vals: list[float] = []
    n = len(df)
    for _ in range(BOOTSTRAP_N):
        idx = rng.randint(0, n, n)
        bdf = df.iloc[idx].reset_index(drop=True)
        y = bdf[outcome_col].to_numpy(dtype=np.float64)
        _, _, base_r2 = _fit_ols(y, _build_design(bdf, include_r2=False))
        _, _, full_r2 = _fit_ols(y, _build_design(bdf, include_r2=True))
        vals.append(full_r2 - base_r2)
    lo, hi = np.percentile(vals, [2.5, 97.5]).tolist()
    return float(lo), float(hi)


def _permute_delta_r2(df: pd.DataFrame, outcome_col: str, rng: np.random.RandomState) -> float:
    y = df[outcome_col].to_numpy(dtype=np.float64)
    _, _, base_r2 = _fit_ols(y, _build_design(df, include_r2=False))
    _, _, full_r2 = _fit_ols(y, _build_design(df, include_r2=True))
    observed = full_r2 - base_r2

    null_vals: list[float] = []
    for _ in range(PERMUTATION_N):
        pdf = df.copy()
        shuffled = []
        for _, grp in pdf.groupby("layer"):
            vals = grp["mean_r2"].to_numpy(dtype=np.float64).copy()
            rng.shuffle(vals)
            shuffled.append(pd.Series(vals, index=grp.index))
        pdf["mean_r2"] = pd.concat(shuffled).sort_index().to_numpy()
        _, _, p_full_r2 = _fit_ols(y, _build_design(pdf, include_r2=True))
        null_vals.append(p_full_r2 - base_r2)
    null_arr = np.asarray(null_vals, dtype=np.float64)
    p_perm = float((np.sum(null_arr >= observed) + 1) / (len(null_arr) + 1))
    return p_perm


def _approx_power_for_corr(n: int, effect_r: float = 0.15, alpha: float = 0.05) -> float:
    if n <= 3:
        return float("nan")
    z_eff = np.arctanh(effect_r) * np.sqrt(n - 3)
    z_alpha = scipy_stats.norm.ppf(1 - alpha / 2.0)
    power = scipy_stats.norm.cdf(z_eff - z_alpha)
    return float(np.clip(power, 0.0, 1.0))


def _prepare_feature_table(model: str) -> pd.DataFrame:
    r2_path = Path("results/experiment3/theory1_si_circuits") / model / "per_sequence_r2.parquet"
    prev_path = Path("results/experiment3/theory7_induction_feeders") / model / "prev_token_scores.parquet"
    bscore_path = Path("results/experiment3/theory5b_boundary_detection") / model / "boundary_attention_scores.parquet"
    t7b_patch_path = Path("results/experiment3/theory7b_activation_patching") / model / "patching_results.parquet"
    t10_patch_path = Path("results/experiment3/theory10_feeder_specificity") / model / "patching_results.parquet"

    required = [r2_path, prev_path, bscore_path, t7b_patch_path, t10_patch_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs for 3P2-F ({model}): {missing}")

    r2_seq = pd.read_parquet(r2_path)
    r2_feat = (
        r2_seq.groupby(["layer", "head"], as_index=False)["r2"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_r2", "std": "r2_std"})
    )

    prev_df = pd.read_parquet(prev_path)[["layer", "head", "prev_token_score"]].copy()
    bscore_df = pd.read_parquet(bscore_path)[["layer", "head", "boundary_attn_score"]].copy()

    t7b_patch = pd.read_parquet(t7b_patch_path)
    t7b_outcome = (
        t7b_patch.groupby(["source_layer", "source_head"], as_index=False)["mean_disruption"]
        .mean()
        .rename(
            columns={
                "source_layer": "layer",
                "source_head": "head",
                "mean_disruption": "outcome_t7b_disruption",
            }
        )
    )

    t10_patch = pd.read_parquet(t10_patch_path)
    t10_ind = (
        t10_patch.loc[t10_patch["target_group"] == "induction"]
        .groupby(["source_layer", "source_head"], as_index=False)["mean_disruption"]
        .mean()
        .rename(
            columns={
                "source_layer": "layer",
                "source_head": "head",
                "mean_disruption": "outcome_t10_induction_disruption",
            }
        )
    )
    t10_rand = (
        t10_patch.loc[t10_patch["target_group"] == "random_mid"]
        .groupby(["source_layer", "source_head"], as_index=False)["mean_disruption"]
        .mean()
        .rename(
            columns={
                "source_layer": "layer",
                "source_head": "head",
                "mean_disruption": "outcome_t10_random_mid_disruption",
            }
        )
    )

    df = r2_feat.merge(prev_df, on=["layer", "head"], how="left")
    df = df.merge(bscore_df, on=["layer", "head"], how="left")
    df = df.merge(t7b_outcome, on=["layer", "head"], how="left")
    df = df.merge(t10_ind, on=["layer", "head"], how="left")
    df = df.merge(t10_rand, on=["layer", "head"], how="left")

    # Proxy controls for the base model.
    p = np.clip(df["prev_token_score"].fillna(0.0).to_numpy(dtype=np.float64), 1e-6, 1 - 1e-6)
    df["attention_entropy_proxy"] = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    df["dominant_offset_proxy"] = df["prev_token_score"].fillna(0.0).astype(float)
    df["head_norm_proxy"] = np.sqrt(np.square(df["mean_r2"].fillna(0.0)) + np.square(df["r2_std"].fillna(0.0)))
    df["model"] = model
    return df


def _analyze_outcome(df: pd.DataFrame, outcome_col: str, rng_boot: np.random.RandomState, rng_perm: np.random.RandomState) -> dict[str, Any]:
    adf = df[["layer", "head", "mean_r2", "attention_entropy_proxy", "dominant_offset_proxy", "head_norm_proxy", outcome_col]].dropna()
    n = len(adf)
    if n < 20:
        return {
            "n_heads": n,
            "base_r2": float("nan"),
            "full_r2": float("nan"),
            "delta_r2": float("nan"),
            "delta_r2_ci_95": [float("nan"), float("nan")],
            "delta_r2_perm_p": float("nan"),
            "r2_unique_contribution_proxy": float("nan"),
            "ci_includes_zero": True,
        }

    y = adf[outcome_col].to_numpy(dtype=np.float64)
    x_base = _build_design(adf, include_r2=False)
    x_full = _build_design(adf, include_r2=True)

    beta_base, yhat_base, base_r2 = _fit_ols(y, x_base)
    beta_full, yhat_full, full_r2 = _fit_ols(y, x_full)
    delta = float(full_r2 - base_r2)

    ci_lo, ci_hi = _bootstrap_delta_r2(adf, outcome_col, rng_boot)
    perm_p = _permute_delta_r2(adf, outcome_col, rng_perm)

    # Standardized proxy effect for mean_r2 (last column in full model body).
    y_std = np.std(y, ddof=1)
    r2_std = np.std(adf["mean_r2"].to_numpy(dtype=np.float64), ddof=1)
    beta_r2 = float(beta_full[-1]) if len(beta_full) else float("nan")
    std_beta = float(beta_r2 * r2_std / y_std) if y_std > 0 and np.isfinite(beta_r2) else float("nan")

    return {
        "n_heads": int(n),
        "base_r2": float(base_r2),
        "full_r2": float(full_r2),
        "delta_r2": delta,
        "delta_r2_ci_95": [float(ci_lo), float(ci_hi)],
        "delta_r2_perm_p": float(perm_p),
        "r2_unique_contribution_proxy": std_beta,
        "ci_includes_zero": bool(ci_lo <= 0.0 <= ci_hi),
    }


def run(model: str, output_root: Path) -> None:
    out_dir = output_root / model
    out_dir.mkdir(parents=True, exist_ok=True)

    full_table = _prepare_feature_table(model)
    full_table.to_parquet(out_dir / "per_head_features.parquet", index=False)

    rng_boot = np.random.RandomState(BOOTSTRAP_SEED)
    rng_perm = np.random.RandomState(PERMUTATION_SEED)

    outcomes = {
        "t7b_disruption": "outcome_t7b_disruption",
        "t10_induction_disruption": "outcome_t10_induction_disruption",
        "t10_random_mid_disruption": "outcome_t10_random_mid_disruption",
        "t5b_boundary_score": "boundary_attn_score",
    }

    outcome_results: dict[str, dict[str, Any]] = {}
    deltas: list[float] = []
    unique_effects: list[float] = []
    ci_include_zero_count = 0
    total_valid = 0
    n_heads_min = None

    for outcome_name, outcome_col in outcomes.items():
        res = _analyze_outcome(full_table, outcome_col, rng_boot, rng_perm)
        outcome_results[outcome_name] = res
        d = _safe_float(res.get("delta_r2"))
        if np.isfinite(d):
            deltas.append(d)
            total_valid += 1
        if bool(res.get("ci_includes_zero", True)):
            ci_include_zero_count += 1
        u = _safe_float(res.get("r2_unique_contribution_proxy"))
        if np.isfinite(u):
            unique_effects.append(u)
        n_val = int(res.get("n_heads", 0))
        if n_heads_min is None or n_val < n_heads_min:
            n_heads_min = n_val

    median_delta = float(np.median(deltas)) if deltas else float("nan")
    median_unique = float(np.median(unique_effects)) if unique_effects else float("nan")

    collapse = bool(
        np.isfinite(median_delta)
        and median_delta < 0.02
        and ci_include_zero_count >= 2
    )

    achieved_power = _approx_power_for_corr(int(n_heads_min or 0), effect_r=0.15)
    proxy_json = {
        "experiment": "3P2-F_proxy_decomposition",
        "model": model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tier": "tier1_confirmatory_core",
        "primary_test_id": "3P2-F",
        "base_model_spec": {
            "formula": "outcome ~ C(layer) + attention_entropy_proxy + dominant_offset_proxy + head_norm_proxy",
            "notes": "Layer encoded as one-hot dummies.",
        },
        "full_model_spec": {
            "formula": "outcome ~ C(layer) + attention_entropy_proxy + dominant_offset_proxy + head_norm_proxy + mean_r2",
            "notes": "Nested model with mean_r2 appended to base controls.",
        },
        "outcomes": outcome_results,
        "delta_r2": {
            "per_outcome": {k: _safe_float(v.get("delta_r2")) for k, v in outcome_results.items()},
            "median": median_delta,
        },
        "partial_effect_r2": {
            "per_outcome_standardized_proxy": {
                k: _safe_float(v.get("r2_unique_contribution_proxy"))
                for k, v in outcome_results.items()
            },
            "median_standardized_proxy": median_unique,
        },
        "mde_target": {
            "delta_r2": 0.02,
            "effect_type": "incremental_variance_explained",
        },
        "achieved_power": achieved_power,
        "multiplicity_family": "tier1_holm_primary_tests",
        "retained_vs_collapsed_verdict": "collapsed" if collapse else "retained",
        "collapse_rule_evaluation": {
            "median_delta_r2_lt_0_02": bool(np.isfinite(median_delta) and median_delta < 0.02),
            "ci_includes_zero_count": int(ci_include_zero_count),
            "n_outcomes_evaluated": int(total_valid),
            "rule_triggered": collapse,
        },
        "inputs": {
            "t1_per_sequence_r2": f"results/experiment3/theory1_si_circuits/{model}/per_sequence_r2.parquet",
            "t7_prev_token_scores": f"results/experiment3/theory7_induction_feeders/{model}/prev_token_scores.parquet",
            "t5b_boundary_scores": f"results/experiment3/theory5b_boundary_detection/{model}/boundary_attention_scores.parquet",
            "t7b_patching_results": f"results/experiment3/theory7b_activation_patching/{model}/patching_results.parquet",
            "t10_patching_results": f"results/experiment3/theory10_feeder_specificity/{model}/patching_results.parquet",
        },
    }
    _write_json(out_dir / "proxy_decomposition.json", proxy_json)
    print(f"[3P2-F] {model}: wrote artifacts to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3P2-F: Proxy decomposition for R-squared")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument(
        "--output-root",
        default="results/experiment3_phase2/exp3p2f_proxy_decomposition",
        help="Output root directory for 3P2-F artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(model=args.model, output_root=Path(args.output_root))


if __name__ == "__main__":
    main()
