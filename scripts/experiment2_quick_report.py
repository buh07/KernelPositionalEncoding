#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _parse_int_csv(raw: str) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for part in str(raw).split(","):
        tok = part.strip()
        if not tok:
            continue
        val = int(tok)
        if val not in seen:
            out.append(val)
            seen.add(val)
    return out


def _parse_str_csv(raw: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for part in str(raw).split(","):
        tok = part.strip()
        if not tok:
            continue
        if tok not in seen:
            out.append(tok)
            seen.add(tok)
    return out


def _trend_non_decreasing(values: list[float]) -> bool:
    finite = [float(v) for v in values if np.isfinite(v)]
    if len(finite) <= 1:
        return False
    return bool(all((b - a) >= -1e-12 for a, b in zip(finite[:-1], finite[1:])))


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize quick fixed-span stress-test runs.")
    ap.add_argument("--aggregate-task-metrics", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--model", type=str, default="llama-3.1-8b")
    ap.add_argument("--task", type=str, default="long_range_retrieval")
    ap.add_argument("--spans", type=str, default="32,48,64")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument(
        "--interventions",
        type=str,
        required=True,
        help="CSV interventions including none, e.g. none,ablate_high_strong,ablate_low_strong,random_strong",
    )
    ap.add_argument(
        "--monotonic-interventions",
        type=str,
        default="ablate_high_strong,ablate_low_strong",
        help="Subset to evaluate monotonicity for 32->48->64 style checks.",
    )
    args = ap.parse_args()

    spans = _parse_int_csv(args.spans)
    seeds = set(_parse_int_csv(args.seeds))
    interventions = _parse_str_csv(args.interventions)
    monotonic_interventions = _parse_str_csv(args.monotonic_interventions)

    df = pd.read_parquet(args.aggregate_task_metrics)
    need_cols = {"split", "model", "task", "span", "seed", "intervention", "mean_accuracy"}
    missing = sorted(need_cols - set(df.columns))
    if missing:
        raise RuntimeError(f"Missing required columns in aggregate task metrics: {missing}")

    df = df[
        (df["split"] == "synthetic")
        & (df["model"] == args.model)
        & (df["task"] == args.task)
        & (df["span"].isin(spans))
        & (df["seed"].isin(sorted(seeds)))
        & (df["intervention"].isin(interventions))
    ].copy()
    if df.empty:
        raise RuntimeError("No rows matched requested quick-report filters.")

    if "chance_accuracy" not in df.columns:
        df["chance_accuracy"] = np.nan
    chance_default = float(df["chance_accuracy"].dropna().median()) if df["chance_accuracy"].notna().any() else 0.10

    grouped = (
        df.groupby(["span", "seed", "intervention"], as_index=False)
        .agg(mean_accuracy=("mean_accuracy", "mean"), chance_accuracy=("chance_accuracy", "mean"))
    )

    base = grouped[grouped["intervention"] == "none"][["span", "seed", "mean_accuracy", "chance_accuracy"]].rename(
        columns={"mean_accuracy": "none_accuracy", "chance_accuracy": "none_chance_accuracy"}
    )
    if base.empty:
        raise RuntimeError("Matched rows did not include none baselines.")

    effect_rows: list[dict[str, Any]] = []
    for intervention in interventions:
        if intervention == "none":
            continue
        sub = grouped[grouped["intervention"] == intervention][["span", "seed", "mean_accuracy", "chance_accuracy"]].rename(
            columns={"mean_accuracy": "ablated_accuracy", "chance_accuracy": "ablated_chance_accuracy"}
        )
        merged = base.merge(sub, on=["span", "seed"], how="inner")
        if merged.empty:
            continue
        for _, row in merged.iterrows():
            none_acc = float(row["none_accuracy"])
            abl_acc = float(row["ablated_accuracy"])
            chance = float(row["none_chance_accuracy"]) if np.isfinite(row["none_chance_accuracy"]) else chance_default
            drop = none_acc - abl_acc
            headroom = none_acc - chance
            effect_rows.append(
                {
                    "span": int(row["span"]),
                    "seed": int(row["seed"]),
                    "intervention": str(intervention),
                    "none_accuracy": none_acc,
                    "ablated_accuracy": abl_acc,
                    "chance_accuracy": chance,
                    "drop_abs": drop,
                    "drop_over_baseline": drop / max(none_acc, 1e-6),
                    "drop_over_headroom": drop / max(headroom, 1e-6),
                }
            )

    eff = pd.DataFrame(effect_rows)
    if eff.empty:
        raise RuntimeError("No matched intervention rows could be compared to none baseline.")

    by_span = (
        eff.groupby(["span", "intervention"], as_index=False)
        .agg(
            n=("seed", "count"),
            none_accuracy_mean=("none_accuracy", "mean"),
            ablated_accuracy_mean=("ablated_accuracy", "mean"),
            drop_abs_mean=("drop_abs", "mean"),
            drop_abs_std=("drop_abs", "std"),
            drop_over_baseline_mean=("drop_over_baseline", "mean"),
            drop_over_headroom_mean=("drop_over_headroom", "mean"),
        )
        .sort_values(["intervention", "span"])
    )

    mono_block: dict[str, Any] = {}
    for intervention in monotonic_interventions:
        sub = by_span[by_span["intervention"] == intervention].set_index("span")
        abs_vals = [float(sub.loc[s, "drop_abs_mean"]) if s in sub.index else float("nan") for s in spans]
        base_vals = [float(sub.loc[s, "drop_over_baseline_mean"]) if s in sub.index else float("nan") for s in spans]
        head_vals = [float(sub.loc[s, "drop_over_headroom_mean"]) if s in sub.index else float("nan") for s in spans]
        mono_block[intervention] = {
            "spans": [int(s) for s in spans],
            "drop_abs_mean": abs_vals,
            "drop_over_baseline_mean": base_vals,
            "drop_over_headroom_mean": head_vals,
            "non_decreasing_drop_abs": _trend_non_decreasing(abs_vals),
            "non_decreasing_drop_over_baseline": _trend_non_decreasing(base_vals),
            "non_decreasing_drop_over_headroom": _trend_non_decreasing(head_vals),
        }

    summary = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "source": str(args.aggregate_task_metrics),
        "filters": {
            "model": args.model,
            "task": args.task,
            "spans": [int(x) for x in spans],
            "seeds": sorted(int(x) for x in seeds),
            "interventions": interventions,
        },
        "chance_accuracy_default": chance_default,
        "effects_by_span_intervention": by_span.to_dict(orient="records"),
        "monotonic_checks": mono_block,
        "notes": [
            "drop_abs = none_accuracy - ablated_accuracy",
            "drop_over_baseline = drop_abs / none_accuracy",
            "drop_over_headroom = drop_abs / (none_accuracy - chance_accuracy)",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    eff.to_parquet(args.output_dir / "effects_per_seed.parquet", engine="pyarrow", index=False)
    by_span.to_parquet(args.output_dir / "effects_by_span.parquet", engine="pyarrow", index=False)

    memo_lines = [
        "# Quick Stress-Test Decision Memo",
        "",
        f"Source: `{args.aggregate_task_metrics}`",
        f"Model/task: `{args.model}` / `{args.task}`",
        f"Spans: {spans}",
        f"Interventions: {interventions}",
        "",
        "## Monotonic checks",
    ]
    for name, block in mono_block.items():
        memo_lines.append(
            f"- `{name}`: abs={block['non_decreasing_drop_abs']}, "
            f"baseline_norm={block['non_decreasing_drop_over_baseline']}, "
            f"headroom_norm={block['non_decreasing_drop_over_headroom']}"
        )
    memo_lines.append("")
    memo_lines.append("## Notes")
    memo_lines.append("- This report is exploratory and non-gating.")
    memo_lines.append("- Co-primary interpretation uses both raw and headroom-normalized drops.")
    (args.output_dir / "decision_memo.md").write_text("\n".join(memo_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
