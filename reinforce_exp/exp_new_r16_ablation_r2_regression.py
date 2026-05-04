#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp.common import RESULTS_ROOT, command_manifest, ensure_dir, safe_float, timestamp_now, write_json  # noqa: E402

DEFAULT_OUT = RESULTS_ROOT / "exp_new_r16_ablation_r2_regression"
TARGET_MODELS = ("llama-3.1-8b", "mistral-7b-v0.1", "olmo-2-7b")


def _find_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _report_path_candidates(model: str, nonrope_root: Path | None) -> list[Path]:
    candidates: list[Path] = []
    if nonrope_root is not None:
        candidates.append(nonrope_root / model / "report.json")
    candidates.extend(
        [
            ROOT / "results" / "experiment3" / "theory8_position_ablation" / model / "report.json",
            ROOT
            / "results"
            / "reinforce_exp"
            / "exp_r3_core_replication"
            / model
            / "theory8_position_ablation"
            / model
            / "report.json",
            ROOT / "results" / "reinforce_exp" / "_smoke_n26_theory8" / model / "report.json",
        ]
    )
    return candidates


def _extract_n_sequences(report: dict[str, Any]) -> int | None:
    try:
        return int(report.get("metadata", {}).get("n_sequences") or report.get("n_sequences") or 0) or None
    except Exception:
        return None


def _r2_path_candidates(model: str) -> list[Path]:
    return [
        ROOT / "results" / "experiment3" / "theory1_si_circuits" / model / "head_r2_summary.parquet",
        ROOT / "results" / "experiment3_phase2" / "exp3p2k_non_rope_control" / model / "mean_r2_by_head.parquet",
        ROOT / "results" / "experiment5" / "exp5a_cross_tokenizer_si_profiling" / model / "head_r2_summary.parquet",
    ]


def _load_json(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_ablation_cost(report: dict[str, Any]) -> float:
    return safe_float(
        report.get("analysis", {})
        .get("comparisons", {})
        .get("subtract_kernel_high_si", {})
        .get("mean_loss_increase")
    )


def _extract_mean_r2(path: Path) -> float:
    df = pd.read_parquet(path)
    if "mean_r2" in df.columns:
        return safe_float(df["mean_r2"].astype(float).mean())
    if "r2" in df.columns:
        return safe_float(df["r2"].astype(float).mean())
    raise RuntimeError(f"Unable to find R2 column in {path}; expected mean_r2 or r2")


def _fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept = float(beta[0])
    slope = float(beta[1])
    y_hat = X @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else float("nan")
    return intercept, slope, r2


def _bootstrap_ci(x: np.ndarray, y: np.ndarray, *, n_boot: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = int(len(x))
    slopes = np.empty(max(2000, int(n_boot)), dtype=np.float64)
    intercepts = np.empty(max(2000, int(n_boot)), dtype=np.float64)
    r2_vals = np.empty(max(2000, int(n_boot)), dtype=np.float64)
    for i in range(len(slopes)):
        idx = rng.integers(0, n, size=n)
        intercept, slope, r2 = _fit_linear(x[idx], y[idx])
        slopes[i] = float(slope)
        intercepts[i] = float(intercept)
        r2_vals[i] = float(r2)
    finite_r2 = r2_vals[np.isfinite(r2_vals)]
    r2_lo = float(np.quantile(finite_r2, 0.025)) if finite_r2.size >= 10 else float("nan")
    r2_hi = float(np.quantile(finite_r2, 0.975)) if finite_r2.size >= 10 else float("nan")
    return {
        "slope_ci95_lo": float(np.quantile(slopes, 0.025)),
        "slope_ci95_hi": float(np.quantile(slopes, 0.975)),
        "intercept_ci95_lo": float(np.quantile(intercepts, 0.025)),
        "intercept_ci95_hi": float(np.quantile(intercepts, 0.975)),
        "r2_ci95_lo": r2_lo,
        "r2_ci95_hi": r2_hi,
    }


def run(*, output_root: Path, nonrope_theory8_root: Path | None, n_boot: int, seed: int) -> dict[str, Any]:
    ensure_dir(output_root)
    rows: list[dict[str, Any]] = []

    for model in TARGET_MODELS:
        report_path = _find_existing(_report_path_candidates(model, nonrope_theory8_root))
        if report_path is None:
            print(f"[NEW-R16] WARNING: Missing theory8 report for {model}; skipping", flush=True)
            continue
        report = _load_json(report_path)
        mean_cost = _extract_ablation_cost(report)
        if not np.isfinite(mean_cost) or mean_cost <= 0.0:
            print(f"[NEW-R16] WARNING: Non-positive or invalid high-SI ablation cost for {model}: {mean_cost}; skipping", flush=True)
            continue

        n_seq = _extract_n_sequences(report)
        if n_seq is not None and n_seq < 20:
            print(f"[NEW-R16] WARNING: {model} ablation cost from low-n report (n_sequences={n_seq}); interpret with caution", flush=True)

        r2_path = _find_existing(_r2_path_candidates(model))
        if r2_path is None:
            print(f"[NEW-R16] WARNING: Missing R2 source for {model}; skipping", flush=True)
            continue
        mean_r2 = _extract_mean_r2(r2_path)
        if not np.isfinite(mean_r2):
            print(f"[NEW-R16] WARNING: Invalid mean R2 for {model} from {r2_path}; skipping", flush=True)
            continue

        rows.append(
            {
                "model": model,
                "mean_r2": float(mean_r2),
                "mean_ablation_cost_high_si": float(mean_cost),
                "log_ablation_cost_high_si": float(math.log(mean_cost)),
                "theory8_report_path": str(report_path),
                "r2_source_path": str(r2_path),
                "n_sequences_ablation": int(n_seq) if n_seq is not None else None,
            }
        )

    points = pd.DataFrame(rows).sort_values("mean_r2").reset_index(drop=True)
    if len(points) < 3:
        raise RuntimeError(f"Insufficient model points for regression: got {len(points)}, need at least 3")

    x = points["mean_r2"].to_numpy(dtype=float)
    y = points["log_ablation_cost_high_si"].to_numpy(dtype=float)
    intercept, slope, r2 = _fit_linear(x, y)
    ci = _bootstrap_ci(x, y, n_boot=n_boot, seed=seed)

    summary = {
        "timestamp": timestamp_now(),
        "experiment": "NEW-R16",
        "n_models": int(len(points)),
        "model_points": points.to_dict(orient="records"),
        "fit": {
            "formula": "log(mean_ablation_cost_high_si) = intercept + slope * mean_r2",
            "intercept": float(intercept),
            "slope": float(slope),
            "r2": float(r2),
        },
        "bootstrap": {
            "n_boot": int(max(2000, int(n_boot))),
            "seed": int(seed),
            **{k: float(v) for k, v in ci.items()},
        },
    }

    points.to_csv(output_root / "model_points.csv", index=False)
    write_json(output_root / "regression_summary.json", summary)
    snippet = (
        f"Across {len(points)} RoPE-equipped 7--8B models, log high-SI kernel-ablation cost scales with mean SI strength: "
        f"slope={slope:.3f} (95% bootstrap CI [{ci['slope_ci95_lo']:.3f}, {ci['slope_ci95_hi']:.3f}]), "
        f"$R^2={r2:.3f}$ ($n={len(points)}$; GPT-2 non-RoPE anchors excluded)."
    )
    (output_root / "paper_stats_snippet.txt").write_text(snippet + "\n", encoding="utf-8")
    write_json(
        output_root / "manifest.json",
        command_manifest(
            experiment_id="NEW-R16",
            command="ablation_r2_regression",
            model="+".join(TARGET_MODELS),
            extras={
                "output_root": str(output_root),
                "nonrope_theory8_root": str(nonrope_theory8_root) if nonrope_theory8_root is not None else None,
                "n_boot": int(max(2000, int(n_boot))),
                "seed": int(seed),
            },
        ),
    )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NEW-R16: 5-model regression of ablation cost on mean R2")
    p.add_argument("--output-root", default=str(DEFAULT_OUT))
    p.add_argument("--nonrope-theory8-root", default=None)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260427)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    nonrope_root = Path(args.nonrope_theory8_root) if args.nonrope_theory8_root else None
    summary = run(
        output_root=Path(args.output_root),
        nonrope_theory8_root=nonrope_root,
        n_boot=max(2000, int(args.n_boot)),
        seed=int(args.seed),
    )
    print(f"[NEW-R16] wrote {Path(args.output_root) / 'regression_summary.json'}")
    print(f"[NEW-R16] fit={summary['fit']}")


if __name__ == "__main__":
    main()
