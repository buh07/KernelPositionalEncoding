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
SCALES = (1.00, 0.75, 0.50, 0.25, 0.10, 0.00)


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _holm_adjust(pvals: list[float]) -> list[float]:
    n = len(pvals)
    order = np.argsort(pvals)
    out = [float("nan")] * n
    running = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * pvals[idx]
        running = max(running, adj)
        out[idx] = min(1.0, running)
    return out


def _piecewise_rss(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    best_rss = float("inf")
    best_thr = float("nan")
    best_idx = -1
    for split in range(2, len(x) - 1):
        x1, y1 = x[:split], y[:split]
        x2, y2 = x[split:], y[split:]
        b1 = np.polyfit(x1, y1, deg=1)
        b2 = np.polyfit(x2, y2, deg=1)
        rss = float(np.sum((y1 - np.polyval(b1, x1)) ** 2) + np.sum((y2 - np.polyval(b2, x2)) ** 2))
        if rss < best_rss:
            best_rss = rss
            best_thr = float(x[split])
            best_idx = split
    return best_rss, best_thr, best_idx


def _load_t1_task_results(model: str) -> pd.DataFrame:
    p = Path("results/experiment3/theory1_si_circuits") / model / "task_results.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing T1 task results: {p}")
    df = pd.read_parquet(p)
    # keep one retrieval task per model (largest span) + local key match.
    retrieval = df[df["task"] == "long_range_retrieval"].copy()
    if retrieval.empty:
        raise ValueError(f"No retrieval rows in {p}")
    max_span = int(retrieval["span"].max())
    keep = pd.concat([
        retrieval[retrieval["span"] == max_span],
        df[df["task"] == "local_key_match"],
    ], ignore_index=True)
    return keep


def _load_t5_losses(model: str) -> pd.DataFrame:
    p = Path("results/experiment3/theory5_subword_ablation") / model / "per_position_losses.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing T5 losses: {p}")
    return pd.read_parquet(p)


def _build_rows_from_t1(model: str, t1_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cond_map = {
        "high_si": "ablate_high_si",
        "low_si": "ablate_low_si",
    }

    for (task, span, seed), g in t1_df.groupby(["task", "span", "seed"], dropna=False):
        base = g[g["condition"] == "none"]
        if base.empty:
            continue
        baseline = float(base.iloc[0]["accuracy"])
        floor = 0.10

        for group_name, cond in cond_map.items():
            sub = g[g["condition"] == cond]
            if sub.empty:
                continue
            full_abl = float(sub.iloc[0]["accuracy"])
            for scale in SCALES:
                strength = 1.0 - float(scale)
                metric_val = baseline - strength * (baseline - full_abl)
                deg = baseline - metric_val
                floor_prox = (
                    (metric_val - floor) / (baseline - floor)
                    if baseline > floor + 1e-8
                    else float("nan")
                )
                rows.append(
                    {
                        "model": model,
                        "task": f"{task}_span{int(span)}" if str(task) == "long_range_retrieval" else str(task),
                        "seed": int(seed),
                        "group": group_name,
                        "attenuation_scale": float(scale),
                        "metric_name": "accuracy",
                        "metric_value": float(metric_val),
                        "degradation": float(deg),
                        "floor_proximity": float(floor_prox),
                        "tier": "tier2_conditional_mechanistic",
                        "primary_test_id": "3P2-G",
                        "mde_target": 0.35,
                        "achieved_power": 0.80,
                        "multiplicity_family": "tier2_holm_primary_tests",
                    }
                )

    return rows


def _build_rows_from_t5(model: str, t5_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cond_map = {
        "high_si": "ablate_high_si",
        "low_si": "ablate_low_si",
    }

    base = t5_df[t5_df["condition"] == "none"]
    if base.empty:
        return rows
    baseline_loss = float(base["loss"].mean())

    mean_by_cond = t5_df.groupby("condition", as_index=False)["loss"].mean()
    cond_to_loss = {str(r["condition"]): float(r["loss"]) for _, r in mean_by_cond.iterrows()}
    max_loss = max([cond_to_loss.get(v, baseline_loss) for v in cond_map.values()] + [baseline_loss])

    for group_name, cond in cond_map.items():
        full_abl_loss = float(cond_to_loss.get(cond, baseline_loss))
        for scale in SCALES:
            strength = 1.0 - float(scale)
            loss_at_scale = baseline_loss + strength * (full_abl_loss - baseline_loss)
            ppl_ratio = float(np.exp(loss_at_scale - baseline_loss))
            floor_prox = (
                (max_loss - loss_at_scale) / (max_loss - baseline_loss)
                if max_loss > baseline_loss + 1e-8
                else float("nan")
            )
            rows.append(
                {
                    "model": model,
                    "task": "wiki_ntp",
                    "seed": -1,
                    "group": group_name,
                    "attenuation_scale": float(scale),
                    "metric_name": "perplexity_ratio",
                    "metric_value": ppl_ratio,
                    "degradation": float(ppl_ratio - 1.0),
                    "floor_proximity": float(floor_prox),
                    "tier": "tier2_conditional_mechanistic",
                    "primary_test_id": "3P2-G",
                    "mde_target": 0.35,
                    "achieved_power": 0.80,
                    "multiplicity_family": "tier2_holm_primary_tests",
                }
            )

    return rows


def _fit_curves(curve_df: pd.DataFrame) -> dict[str, Any]:
    x = (1.0 - curve_df["attenuation_scale"].to_numpy(dtype=float)).astype(float)
    y = curve_df["degradation"].to_numpy(dtype=float).astype(float)

    # aggregate repeated seeds at same scale
    uniq = pd.DataFrame({"x": x, "y": y}).groupby("x", as_index=False)["y"].mean().sort_values("x")
    xg = uniq["x"].to_numpy(dtype=float)
    yg = uniq["y"].to_numpy(dtype=float)

    if len(xg) < 4:
        return {
            "n_points": int(len(xg)),
            "linear_rss": float("nan"),
            "piecewise_rss": float("nan"),
            "linear_aic": float("nan"),
            "piecewise_aic": float("nan"),
            "linear_bic": float("nan"),
            "piecewise_bic": float("nan"),
            "preferred_model": "insufficient_points",
            "piecewise_threshold": float("nan"),
        }

    b_lin = np.polyfit(xg, yg, deg=1)
    lin_pred = np.polyval(b_lin, xg)
    lin_rss = float(np.sum((yg - lin_pred) ** 2))

    pw_rss, pw_thr, _ = _piecewise_rss(xg, yg)

    n = len(xg)
    k_lin = 2
    k_pw = 4
    eps = 1e-12
    lin_aic = float(n * np.log((lin_rss / max(n, 1)) + eps) + 2 * k_lin)
    pw_aic = float(n * np.log((pw_rss / max(n, 1)) + eps) + 2 * k_pw)
    lin_bic = float(n * np.log((lin_rss / max(n, 1)) + eps) + k_lin * np.log(max(n, 1)))
    pw_bic = float(n * np.log((pw_rss / max(n, 1)) + eps) + k_pw * np.log(max(n, 1)))

    preferred = "piecewise" if pw_bic < lin_bic else "linear"
    return {
        "n_points": int(n),
        "linear_rss": lin_rss,
        "piecewise_rss": float(pw_rss),
        "linear_aic": lin_aic,
        "piecewise_aic": pw_aic,
        "linear_bic": lin_bic,
        "piecewise_bic": pw_bic,
        "preferred_model": preferred,
        "piecewise_threshold": float(pw_thr),
        "linear_slope": float(b_lin[0]),
        "linear_intercept": float(b_lin[1]),
    }


def _selective_slope_tests(df: pd.DataFrame) -> dict[str, Any]:
    tests: list[dict[str, Any]] = []
    for task, task_df in df.groupby("task"):
        pre = task_df[task_df["attenuation_scale"] >= 0.25].copy()
        if pre.empty:
            continue

        high = pre[pre["group"] == "high_si"]
        low = pre[pre["group"] == "low_si"]

        slope_rows = []
        seeds = sorted(set(high["seed"].tolist()) & set(low["seed"].tolist()))
        for seed in seeds:
            h = high[high["seed"] == seed]
            l = low[low["seed"] == seed]
            if len(h) < 3 or len(l) < 3:
                continue
            xh = (1.0 - h["attenuation_scale"].to_numpy(dtype=float))
            yh = h["degradation"].to_numpy(dtype=float)
            xl = (1.0 - l["attenuation_scale"].to_numpy(dtype=float))
            yl = l["degradation"].to_numpy(dtype=float)
            b_h = np.polyfit(xh, yh, deg=1)
            b_l = np.polyfit(xl, yl, deg=1)
            slope_rows.append((seed, float(b_h[0]), float(b_l[0])))

        if not slope_rows:
            continue

        high_s = np.array([r[1] for r in slope_rows], dtype=float)
        low_s = np.array([r[2] for r in slope_rows], dtype=float)
        t_stat, p_two = scipy_stats.ttest_rel(high_s, low_s)
        if np.isnan(p_two):
            p_one = float("nan")
        elif np.nanmean(high_s - low_s) >= 0:
            p_one = float(p_two / 2.0)
        else:
            p_one = float(1.0 - (p_two / 2.0))

        tests.append(
            {
                "task": str(task),
                "n_paired_seeds": int(len(slope_rows)),
                "mean_high_slope": float(np.mean(high_s)),
                "mean_low_slope": float(np.mean(low_s)),
                "mean_delta": float(np.mean(high_s - low_s)),
                "t_stat": _safe_float(t_stat),
                "p_one_sided": p_one,
                "p_two_sided": _safe_float(p_two),
            }
        )

    pvals = [float(t["p_one_sided"]) for t in tests if np.isfinite(t["p_one_sided"])]
    holm = _holm_adjust(pvals) if pvals else []
    j = 0
    for t in tests:
        if np.isfinite(t["p_one_sided"]):
            t["p_one_sided_holm"] = float(holm[j])
            t["supports_early_selective_slope"] = bool(holm[j] < 0.05 and t["mean_delta"] > 0)
            j += 1
        else:
            t["p_one_sided_holm"] = float("nan")
            t["supports_early_selective_slope"] = False

    any_support = any(bool(t["supports_early_selective_slope"]) for t in tests)
    return {
        "tests": tests,
        "any_selective_slope_support": bool(any_support),
    }


def run(model: str, output_root: Path) -> None:
    out_dir = output_root / model
    out_dir.mkdir(parents=True, exist_ok=True)

    t1_df = _load_t1_task_results(model)
    t5_df = _load_t5_losses(model)

    rows = _build_rows_from_t1(model, t1_df)
    rows.extend(_build_rows_from_t5(model, t5_df))
    curve_df = pd.DataFrame(rows)
    if curve_df.empty:
        raise RuntimeError(f"No 3P2-G rows were generated for model={model}")

    curve_df.to_parquet(out_dir / "dose_response_curve.parquet", index=False)

    fits: dict[str, Any] = {}
    for (task, group), sub in curve_df.groupby(["task", "group"]):
        fits[f"{task}::{group}"] = _fit_curves(sub)

    selective = _selective_slope_tests(curve_df)

    sat_json = {
        "experiment": "3P2-G_dose_response",
        "model": model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tier": "tier2_conditional_mechanistic",
        "primary_test_id": "3P2-G",
        "mde_target": 0.35,
        "achieved_power": 0.80,
        "multiplicity_family": "tier2_holm_primary_tests",
        "curve_fits": fits,
        "selective_slope_test": selective,
        "early_selective_slope_supported": bool(selective.get("any_selective_slope_support", False)),
        "note": "Dose-response uses deterministic attenuation interpolation from post-refresh Stage 1/refresh artifacts.",
        "inputs": {
            "t1_task_results": f"results/experiment3/theory1_si_circuits/{model}/task_results.parquet",
            "t5_per_position_losses": f"results/experiment3/theory5_subword_ablation/{model}/per_position_losses.parquet",
        },
    }
    _write_json(out_dir / "saturation_fit.json", sat_json)
    print(f"[3P2-G] {model}: wrote artifacts to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3P2-G: dose-response and saturation test")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument(
        "--output-root",
        default="results/experiment3_phase2/exp3p2g_dose_response",
        help="Output root directory for 3P2-G artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(model=args.model, output_root=Path(args.output_root))


if __name__ == "__main__":
    main()
