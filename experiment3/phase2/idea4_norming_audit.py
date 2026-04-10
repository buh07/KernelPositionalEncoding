#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _audit_model(model_dir: Path) -> dict[str, Any]:
    rows_path = model_dir / "per_item_scores.parquet"
    if not rows_path.exists():
        raise FileNotFoundError(f"Missing {rows_path}")

    df = pd.read_parquet(rows_path)
    if "condition" not in df.columns:
        raise RuntimeError("per_item_scores missing condition column")

    none = df[df["condition"] == "none"].copy()
    if none.empty:
        raise RuntimeError("No condition=none rows found")

    labeled = none[none["correct_option"].notna()].copy()
    neutral = none[none["correct_option"].isna()].copy()

    for col in ["variant_id", "context_label", "pred_option", "family", "relation_type"]:
        if col not in none.columns:
            raise RuntimeError(f"Missing required column: {col}")

    # Per-variant context consistency checks on labeled contexts.
    pairs = labeled[labeled["context_label"].isin(["context_a", "context_b"])].copy()
    grouped = pairs.pivot_table(
        index=["variant_id", "family", "relation_type"],
        columns="context_label",
        values=["pred_option", "is_correct", "correct_margin", "p_a", "entropy"],
        aggfunc="first",
    ).reset_index()

    if ("pred_option", "context_a") in grouped.columns and ("pred_option", "context_b") in grouped.columns:
        grouped[("derived", "flip")] = (
            grouped[("pred_option", "context_a")].astype(str)
            != grouped[("pred_option", "context_b")].astype(str)
        )
    else:
        grouped[("derived", "flip")] = False

    if ("is_correct", "context_a") in grouped.columns and ("is_correct", "context_b") in grouped.columns:
        grouped[("derived", "both_correct")] = (
            grouped[("is_correct", "context_a")].astype(float) > 0.5
        ) & (
            grouped[("is_correct", "context_b")].astype(float) > 0.5)
    else:
        grouped[("derived", "both_correct")] = False

    if ("correct_margin", "context_a") in grouped.columns and ("correct_margin", "context_b") in grouped.columns:
        grouped[("derived", "mean_correct_margin")] = (
            grouped[("correct_margin", "context_a")].astype(float)
            + grouped[("correct_margin", "context_b")].astype(float)
        ) / 2.0
    else:
        grouped[("derived", "mean_correct_margin")] = np.nan

    # Flatten columns for serialization/parquet.
    flat = grouped.copy()
    flat.columns = [
        c if isinstance(c, str) else (c[0] if c[1] == "" else f"{c[0]}__{c[1]}")
        for c in flat.columns
    ]

    # Flags for weakly normed variants.
    flat["flag_failed_context_a"] = flat.get("is_correct__context_a", np.nan).astype(float) <= 0.5
    flat["flag_failed_context_b"] = flat.get("is_correct__context_b", np.nan).astype(float) <= 0.5
    flat["flag_no_flip"] = ~flat.get("derived__flip", False).astype(bool)
    flat["flag_low_margin"] = flat.get("derived__mean_correct_margin", np.nan).astype(float) < 0.0
    flat["flag_any"] = (
        flat["flag_failed_context_a"]
        | flat["flag_failed_context_b"]
        | flat["flag_no_flip"]
        | flat["flag_low_margin"]
    )

    family_summary: dict[str, Any] = {}
    for fam, g in flat.groupby("family", as_index=False):
        family_summary[str(fam)] = {
            "n_variants": int(len(g)),
            "context_a_accuracy": _safe_float(np.nanmean(g.get("is_correct__context_a", np.nan).to_numpy(dtype=float))),
            "context_b_accuracy": _safe_float(np.nanmean(g.get("is_correct__context_b", np.nan).to_numpy(dtype=float))),
            "both_correct_rate": _safe_float(np.nanmean(g.get("derived__both_correct", np.nan).to_numpy(dtype=float))),
            "flip_rate": _safe_float(np.nanmean(g.get("derived__flip", np.nan).to_numpy(dtype=float))),
            "flag_rate": _safe_float(np.nanmean(g["flag_any"].to_numpy(dtype=float))),
            "mean_correct_margin": _safe_float(np.nanmean(g.get("derived__mean_correct_margin", np.nan).to_numpy(dtype=float))),
        }

    neutral_summary = {
        "n_neutral_rows": int(len(neutral)),
        "neutral_entropy_mean": _safe_float(neutral["entropy"].astype(float).mean()) if len(neutral) else float("nan"),
        "neutral_p_a_mean": _safe_float(neutral["p_a"].astype(float).mean()) if len(neutral) else float("nan"),
        "neutral_p_a_std": _safe_float(neutral["p_a"].astype(float).std()) if len(neutral) else float("nan"),
    }

    overall = {
        "n_variants": int(len(flat)),
        "context_a_accuracy": _safe_float(np.nanmean(flat.get("is_correct__context_a", np.nan).to_numpy(dtype=float))),
        "context_b_accuracy": _safe_float(np.nanmean(flat.get("is_correct__context_b", np.nan).to_numpy(dtype=float))),
        "both_correct_rate": _safe_float(np.nanmean(flat.get("derived__both_correct", np.nan).to_numpy(dtype=float))),
        "flip_rate": _safe_float(np.nanmean(flat.get("derived__flip", np.nan).to_numpy(dtype=float))),
        "flagged_variant_rate": _safe_float(np.nanmean(flat["flag_any"].to_numpy(dtype=float))),
        "mean_correct_margin": _safe_float(np.nanmean(flat.get("derived__mean_correct_margin", np.nan).to_numpy(dtype=float))),
    }

    # Save detailed flags.
    flagged = flat[flat["flag_any"]].copy()
    retained = flat[~flat["flag_any"]].copy()
    flat.to_parquet(model_dir / "norming_variant_table.parquet", index=False)
    flagged.to_parquet(model_dir / "norming_flagged_variants.parquet", index=False)
    retained.to_parquet(model_dir / "norming_retained_variants.parquet", index=False)

    return {
        "model": model_dir.name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "audit_type": "idea4_internal_norming_audit",
        "based_on_condition": "none",
        "overall": overall,
        "neutral_summary": neutral_summary,
        "family_summary": family_summary,
        "n_flagged_variants": int(len(flagged)),
        "n_retained_variants": int(len(retained)),
        "retained_variant_ids": sorted(str(v) for v in retained["variant_id"].astype(str).tolist()),
        "limitations": [
            "Internal audit only; does not replace external human norming.",
            "Uses model-internal consistency criteria, which can be model-specific.",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Internal norming audit for Idea 4 stimuli/results")
    ap.add_argument(
        "--root",
        default="results/experiment3_phase2/idea4_structural_ambiguity",
        help="Root containing per-model idea4 outputs",
    )
    ap.add_argument("--models", default="llama-3.1-8b,olmo-2-7b")
    ap.add_argument("--write-allowlists", action="store_true")
    ap.add_argument(
        "--consensus-min-models",
        type=int,
        default=2,
        help="Minimum number of models in which a variant must be retained for consensus allowlist.",
    )
    ap.add_argument(
        "--min-variants-per-relation-type",
        type=int,
        default=8,
        help="Balanced allowlist check threshold per relation type.",
    )
    ap.add_argument("--allowlist-prefix", default="normed_allowlist")
    args = ap.parse_args()

    root = ROOT / args.root
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    summary: dict[str, Any] = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "models": {}}
    retained_by_model: dict[str, set[str]] = {}
    for m in models:
        model_dir = root / m
        rep = _audit_model(model_dir)
        _write_json(model_dir / "norming_audit.json", rep)
        summary["models"][m] = rep
        retained = set(str(v) for v in rep.get("retained_variant_ids", []))
        retained_by_model[m] = retained
        if args.write_allowlists:
            model_allow = model_dir / f"{args.allowlist_prefix}.txt"
            with model_allow.open("w", encoding="utf-8") as f:
                for v in sorted(retained):
                    f.write(f"{v}\n")

    # Build a consensus allowlist across models.
    counts: dict[str, int] = {}
    for kept in retained_by_model.values():
        for v in kept:
            counts[v] = counts.get(v, 0) + 1
    consensus_min = max(1, int(args.consensus_min_models))
    consensus = sorted(v for v, c in counts.items() if c >= consensus_min)

    # Balance check by relation type using first model's retained table as canonical mapping.
    relation_map: dict[str, str] = {}
    if models:
        canonical = root / models[0] / "norming_variant_table.parquet"
        if canonical.exists():
            cdf = pd.read_parquet(canonical)
            for _, r in cdf.iterrows():
                relation_map[str(r["variant_id"])] = str(r["relation_type"])
    relation_counts: dict[str, int] = {}
    for v in consensus:
        rt = relation_map.get(v, "unknown")
        relation_counts[rt] = relation_counts.get(rt, 0) + 1
    min_rt_count = min(relation_counts.values()) if relation_counts else 0
    balanced_ok = bool(min_rt_count >= int(args.min_variants_per_relation_type))

    summary["allowlists"] = {
        "consensus_min_models": consensus_min,
        "consensus_count": int(len(consensus)),
        "consensus_relation_counts": relation_counts,
        "consensus_balanced_min_count": int(min_rt_count),
        "consensus_balanced_ok": balanced_ok,
        "min_variants_per_relation_type": int(args.min_variants_per_relation_type),
    }
    if args.write_allowlists:
        consensus_path = root / f"{args.allowlist_prefix}_consensus.txt"
        with consensus_path.open("w", encoding="utf-8") as f:
            for v in consensus:
                f.write(f"{v}\n")
        summary["allowlists"]["consensus_path"] = str(consensus_path)

    _write_json(root / "norming_audit_summary.json", summary)


if __name__ == "__main__":
    main()
