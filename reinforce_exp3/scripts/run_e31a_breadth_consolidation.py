#!/usr/bin/env python3
"""E31A (A-lite): breadth consolidation from existing head-level R² artifacts.

Builds one canonical non-stale, non-smoke model breadth table from already-run outputs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp3.common import RESULTS_ROOT, ensure_dir, timestamp_now, write_json  # noqa: E402
from reinforce_exp3.scripts._shared import emit_core_artifacts  # noqa: E402

EXPERIMENT_ID = "E31A"
DEFAULT_OUT = RESULTS_ROOT / "E31a_breadth_consolidation"

# Canonical source map (explicitly avoids stale/smoke trees).
MODEL_SOURCES: dict[str, str] = {
    "llama-3.1-8b": "results/experiment3/theory1_si_circuits/llama-3.1-8b/head_r2_summary.parquet",
    "mistral-7b-v0.1": "results/experiment3/theory1_si_circuits/mistral-7b-v0.1/head_r2_summary.parquet",
    "olmo-2-7b": "results/experiment3/theory1_si_circuits/olmo-2-7b/head_r2_summary.parquet",
    "gemma-2-9b": "results/reinforce_exp3/E5_fourth_model/gemma-2-9b/head_r2_summary.parquet",
    "qwen2.5-7b": "results/reinforce_exp3/E29c_qwen_anchor_quickcheck/qwen2.5-7b/head_r2_summary.parquet",
    "gpt2-small": "results/experiment3/theory1_si_circuits/gpt2-small/head_r2_summary.parquet",
    "gpt2-medium": "results/experiment5/exp5a_cross_tokenizer_si_profiling/gpt2-medium/head_r2_summary.parquet",
    "pythia-1.4b": "results/experiment5/exp5a_cross_tokenizer_si_profiling/pythia-1.4b/head_r2_summary.parquet",
    "pythia-410m": "results/experiment5/exp5a_cross_tokenizer_si_profiling/pythia-410m/head_r2_summary.parquet",
    "tinyllama-1.1b": "results/experiment3/theory1_si_circuits/tinyllama-1.1b/head_r2_summary.parquet",
    "tinyllama-nope-1.1b": "results/experiment3/theory1_si_circuits/tinyllama-nope-1.1b/head_r2_summary.parquet",
}

MODEL_FAMILY: dict[str, str] = {
    "llama-3.1-8b": "llama",
    "tinyllama-1.1b": "llama",
    "tinyllama-nope-1.1b": "llama",
    "mistral-7b-v0.1": "mistral",
    "olmo-2-7b": "olmo",
    "gemma-2-9b": "gemma",
    "qwen2.5-7b": "qwen",
    "gpt2-small": "gpt2",
    "gpt2-medium": "gpt2",
    "pythia-1.4b": "pythia",
    "pythia-410m": "pythia",
}


def _load_model_r2(model: str, rel_path: str) -> pd.DataFrame:
    p = ROOT / rel_path
    if not p.exists():
        raise FileNotFoundError(f"[E31A] missing artifact for {model}: {p}")
    df = pd.read_parquet(p)
    if "mean_r2" not in df.columns:
        if "r2" in df.columns:
            df = df.rename(columns={"r2": "mean_r2"})
        else:
            raise RuntimeError(f"[E31A] mean_r2 missing in {p}")
    out = df[["mean_r2"]].copy()
    out["model"] = model
    out["source_path"] = str(p)
    out["family"] = MODEL_FAMILY.get(model, "other")
    return out


def _spread_stats(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "min": float("nan"),
            "max": float("nan"),
            "range": float("nan"),
            "ratio_max_over_min": float("nan"),
        }
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    eps = 1e-9
    return {
        "n": int(arr.size),
        "min": min_v,
        "max": max_v,
        "range": float(max_v - min_v),
        "ratio_max_over_min": float(max_v / max(min_v, eps)),
    }


def run(out_root: Path) -> dict[str, Any]:
    t0 = time.time()
    out_root = ensure_dir(out_root)

    rows: list[pd.DataFrame] = []
    for model, rel in MODEL_SOURCES.items():
        rows.append(_load_model_r2(model, rel))

    all_df = pd.concat(rows, ignore_index=True)

    by_model = (
        all_df.groupby(["model", "family", "source_path"], as_index=False)["mean_r2"]
        .agg(
            n_heads="count",
            mean_r2="mean",
            median_r2="median",
            q25_r2=lambda s: float(np.quantile(np.asarray(s, dtype=np.float64), 0.25)),
            q75_r2=lambda s: float(np.quantile(np.asarray(s, dtype=np.float64), 0.75)),
            std_r2="std",
            min_r2="min",
            max_r2="max",
        )
        .sort_values("mean_r2", ascending=False)
        .reset_index(drop=True)
    )

    by_model.to_csv(out_root / "model_breadth_r2_table.csv", index=False)

    # Family spread summary
    fam_payload: dict[str, Any] = {}
    for fam, g in by_model.groupby("family"):
        vals = [float(x) for x in g["mean_r2"].tolist()]
        fam_payload[fam] = {
            "n_models": int(len(g)),
            "models": g["model"].tolist(),
            **_spread_stats(vals),
        }

    overall = _spread_stats([float(x) for x in by_model["mean_r2"].tolist()])
    family_summary = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "n_models": int(len(by_model)),
        "overall": overall,
        "by_family": fam_payload,
    }
    write_json(out_root / "family_spread_summary.json", family_summary)

    # Paper-ready digest
    top = by_model.iloc[0]
    bot = by_model.iloc[-1]
    digest = f"""# E31A Breadth Digest

- Canonical breadth consolidation over **{len(by_model)}** non-stale model artifacts spanning **{len(set(by_model['family']))}** families.
- Mean SI amplitude spread across this panel: **{overall['ratio_max_over_min']:.2f}x** (min={overall['min']:.3f}, max={overall['max']:.3f}).
- Highest mean R² model: **{top['model']}** ({top['mean_r2']:.3f}); lowest: **{bot['model']}** ({bot['mean_r2']:.3f}).
- This supports a broad heterogeneity statement on the sampled panel; it does not, by itself, identify causal drivers.
"""
    (out_root / "breadth_claim_digest.md").write_text(digest, encoding="utf-8")

    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "analysis_tier": "hardening",
        "canonical_eligible": True,
        "override_used": False,
        "verdict": {
            "interpretation": "breadth_panel_consolidated",
            "n_models": int(len(by_model)),
            "n_families": int(len(set(by_model["family"]))),
            "spread_ratio_max_over_min": float(overall["ratio_max_over_min"]),
        },
        "elapsed_sec": float(time.time() - t0),
    }

    prereg = {
        "experiment_id": EXPERIMENT_ID,
        "question": "What is the non-stale breadth of observed SI amplitude across already-run models?",
        "primary_endpoint": "model-level mean_r2 spread on canonical artifacts",
    }
    claim_impact = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "claim_status": "supported",
        "impact": "extends breadth panel without new heavy inference",
    }
    data_dict = {
        "model_breadth_r2_table.csv": "Per-model head-level R² summary statistics on canonical artifacts.",
        "family_spread_summary.json": "Within-family and overall spread summaries for mean SI amplitude.",
        "breadth_claim_digest.md": "Paper-ready prose digest for breadth heterogeneity claim.",
    }

    emit_core_artifacts(
        experiment_id=EXPERIMENT_ID,
        out_dir=out_root,
        preregistration=prereg,
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dict,
        manifest_extra={
            "models": sorted(MODEL_SOURCES.keys()),
            "output_files": [
                "model_breadth_r2_table.csv",
                "family_spread_summary.json",
                "breadth_claim_digest.md",
            ],
        },
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="E31A breadth consolidation from existing R² artifacts")
    p.add_argument("--output-root", default=str(DEFAULT_OUT))
    args = p.parse_args()

    out_root = Path(args.output_root)
    summary = run(out_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
