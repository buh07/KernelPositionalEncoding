#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiment2.stats import holm_bonferroni, paired_sign_flip_pvalue


def _pair_from_intervention(name: str) -> int | None:
    if not isinstance(name, str):
        return None
    if name.startswith("ablate_pair_"):
        try:
            return int(name.split("_")[-1])
        except ValueError:
            return None
    return None


def _safe_mean(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    n = np.asarray(num, dtype=np.float64)
    d = np.asarray(den, dtype=np.float64)
    out = np.full_like(n, np.nan, dtype=np.float64)
    mask = np.isfinite(n) & np.isfinite(d) & (np.abs(d) > 1e-12)
    out[mask] = n[mask] / d[mask]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Report expanded pair-sweep effects with Holm correction.")
    ap.add_argument("--aggregate-task-metrics", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--families-json", type=Path, required=True)
    ap.add_argument("--pair-indices", type=str, default="0,8,16,24,32,40,48,56")
    ap.add_argument("--floor-threshold", type=float, default=0.15)
    ap.add_argument("--chance-default", type=float, default=0.10)
    args = ap.parse_args()

    df = pd.read_parquet(args.aggregate_task_metrics)
    needed = {
        "split",
        "model",
        "task",
        "span",
        "seed",
        "intervention",
        "random_draw",
        "mean_accuracy",
        "floor_limited",
    }
    missing = sorted(needed - set(df.columns))
    if missing:
        raise RuntimeError(f"Missing required columns in aggregate metrics: {missing}")
    if "chance_accuracy" not in df.columns:
        df["chance_accuracy"] = np.nan

    families = json.loads(args.families_json.read_text(encoding="utf-8"))
    pair_indices = [int(tok.strip()) for tok in str(args.pair_indices).split(",") if tok.strip()]
    floor_threshold = float(args.floor_threshold)

    df = df[df["split"] == "synthetic"].copy()
    df["pair_idx"] = df["intervention"].map(_pair_from_intervention)

    pair_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []

    for fam in families:
        model = str(fam["model"])
        task = str(fam["task"])
        spans = fam.get("spans")
        fam_df = df[(df["model"] == model) & (df["task"] == task)].copy()
        if spans is not None:
            span_values = [int(s) for s in spans]
            fam_df = fam_df[fam_df["span"].isin(span_values)].copy()
        else:
            span_values = sorted(int(s) for s in fam_df["span"].dropna().unique().tolist())
            if not span_values:
                span_values = [0]
        if fam_df.empty:
            continue

        for span in span_values:
            span_df = fam_df[fam_df["span"] == span].copy()
            if span_df.empty:
                continue

            base = span_df[span_df["intervention"] == "none"][["seed", "mean_accuracy", "chance_accuracy", "floor_limited"]].copy()
            if base.empty:
                continue
            base = base.rename(
                columns={
                    "mean_accuracy": "none_accuracy",
                    "chance_accuracy": "none_chance_accuracy",
                    "floor_limited": "none_floor_limited",
                }
            )

            base["chance_effective"] = base["none_chance_accuracy"].where(base["none_chance_accuracy"].notna(), float(args.chance_default))
            base["headroom"] = base["none_accuracy"] - base["chance_effective"]

            floor_excluded = int(span_df["floor_limited"].fillna(False).astype(bool).sum())
            none_pass_rate = float((base["none_accuracy"] >= floor_threshold).mean())
            family_rows.append(
                {
                    "model": model,
                    "task": task,
                    "span": int(span),
                    "baseline_n": int(len(base)),
                    "none_accuracy_mean": float(base["none_accuracy"].mean()),
                    "none_accuracy_min": float(base["none_accuracy"].min()),
                    "none_pass_rate_at_floor": none_pass_rate,
                    "none_floor_limited_rate": float(base["none_floor_limited"].fillna(False).astype(bool).mean()),
                    "excluded_floor_rows": floor_excluded,
                    "floor_threshold": floor_threshold,
                }
            )

            random = span_df[span_df["intervention"] == "random_pair"][["seed", "mean_accuracy"]].copy()
            rand_by_seed = random.groupby("seed", as_index=False).agg(random_mean_accuracy=("mean_accuracy", "mean"))
            merged_rand = base.merge(rand_by_seed, on="seed", how="inner")
            random_drop = (
                merged_rand["none_accuracy"].to_numpy(dtype=np.float64)
                - merged_rand["random_mean_accuracy"].to_numpy(dtype=np.float64)
            )

            raw_p_map: dict[str, float] = {}
            head_p_map: dict[str, float] = {}
            diff_p_map: dict[str, float] = {}
            temp_records: list[dict[str, Any]] = []

            for pair_idx in pair_indices:
                target = span_df[span_df["intervention"] == f"ablate_pair_{pair_idx}"][["seed", "mean_accuracy"]].copy()
                if target.empty:
                    continue
                merged = base.merge(target.rename(columns={"mean_accuracy": "target_accuracy"}), on="seed", how="inner")
                if merged.empty:
                    continue

                none_acc = merged["none_accuracy"].to_numpy(dtype=np.float64)
                tgt_acc = merged["target_accuracy"].to_numpy(dtype=np.float64)
                headroom = merged["headroom"].to_numpy(dtype=np.float64)
                drop_abs = none_acc - tgt_acc
                drop_base = _safe_ratio(drop_abs, none_acc)
                drop_head = _safe_ratio(drop_abs, headroom)

                rand_join = merged_rand.merge(
                    merged[["seed", "target_accuracy"]],
                    on="seed",
                    how="inner",
                )
                target_minus_random = (
                    rand_join["target_accuracy"].to_numpy(dtype=np.float64)
                    - rand_join["random_mean_accuracy"].to_numpy(dtype=np.float64)
                )

                p_raw = float(paired_sign_flip_pvalue(drop_abs))
                p_head = float(paired_sign_flip_pvalue(drop_head))
                p_diff = float(paired_sign_flip_pvalue(target_minus_random))

                key = str(pair_idx)
                raw_p_map[key] = p_raw
                head_p_map[key] = p_head
                diff_p_map[key] = p_diff

                mean_raw = _safe_mean(drop_abs)
                mean_head = _safe_mean(drop_head)
                temp_records.append(
                    {
                        "model": model,
                        "task": task,
                        "span": int(span),
                        "pair_idx": int(pair_idx),
                        "n_seeds": int(len(merged)),
                        "effect_raw": mean_raw,
                        "effect_over_baseline": _safe_mean(drop_base),
                        "effect_over_headroom": mean_head,
                        "effect_target_minus_random": _safe_mean(target_minus_random),
                        "p_raw": p_raw,
                        "p_headroom": p_head,
                        "p_target_minus_random": p_diff,
                        "direction_agreement_raw_vs_headroom": bool(
                            np.isfinite(mean_raw)
                            and np.isfinite(mean_head)
                            and np.sign(mean_raw) == np.sign(mean_head)
                        ),
                    }
                )

            raw_adj = holm_bonferroni(raw_p_map)
            head_adj = holm_bonferroni(head_p_map)
            diff_adj = holm_bonferroni(diff_p_map)

            family_size = int(len(temp_records))
            for rec in temp_records:
                key = str(rec["pair_idx"])
                rec["p_raw_holm"] = float(raw_adj.get(key, float("nan")))
                rec["p_headroom_holm"] = float(head_adj.get(key, float("nan")))
                rec["p_target_minus_random_holm"] = float(diff_adj.get(key, float("nan")))
                rec["mc_method"] = "holm"
                rec["mc_family"] = f"{model}|{task}|span={int(span)}"
                rec["mc_family_size"] = family_size
                pair_rows.append(rec)

    pair_df = pd.DataFrame(pair_rows)
    family_df = pd.DataFrame(family_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_df.to_parquet(args.output_dir / "pair_effects.parquet", engine="pyarrow", index=False)
    family_df.to_parquet(args.output_dir / "floor_exclusion_summary.parquet", engine="pyarrow", index=False)

    summary = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "source": str(args.aggregate_task_metrics),
        "mc_method": "holm",
        "floor_threshold": floor_threshold,
        "pair_count_reported": int(len(pair_df)),
        "families_reported": int(family_df.shape[0]),
        "co_primary_metrics": ["effect_raw", "effect_over_headroom"],
        "notes": [
            "Exploratory-only; non-retroactive to confirmatory adjudication.",
            "Holm correction is applied within each pre-specified family model/task/span.",
            "Direction agreement compares sign(effect_raw) vs sign(effect_over_headroom).",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    memo_lines = [
        "# Expanded Pair-Sweep Decision Memo",
        "",
        f"Source: `{args.aggregate_task_metrics}`",
        "Policy: exploratory-only; non-retroactive to confirmatory claims.",
        "Multiple comparisons: Holm within each pre-specified model/task/span family.",
        "",
        "## Family floor/evaluability summary",
    ]
    if family_df.empty:
        memo_lines.append("- no evaluable families found in source artifact")
    else:
        for row in family_df.sort_values(["model", "task", "span"]).to_dict(orient="records"):
            memo_lines.append(
                f"- `{row['model']}` / `{row['task']}` / span `{int(row['span'])}`: "
                f"none_pass_rate={float(row['none_pass_rate_at_floor']):.3f}, "
                f"excluded_floor_rows={int(row['excluded_floor_rows'])}"
            )
    memo_lines.append("")
    memo_lines.append("## Interpretation guardrails")
    memo_lines.append("- Report both raw and headroom-normalized effects together.")
    memo_lines.append("- Do not overinterpret pairs with floor-limited baselines.")
    memo_lines.append("- Treat pair-level outcomes as exploratory spectral profiling.")
    (args.output_dir / "decision_memo.md").write_text("\n".join(memo_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
