#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge semi-natural quick probe shard outputs.")
    ap.add_argument("--inputs", type=str, required=True, help="Comma-separated shard directories.")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    shard_dirs = [Path(p.strip()) for p in str(args.inputs).split(",") if p.strip()]
    if not shard_dirs:
        raise RuntimeError("No shard input directories provided.")

    frames = []
    for d in shard_dirs:
        p = d / "aggregate_task_metrics.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing shard aggregate file: {p}")
        frames.append(pd.read_parquet(p))
    df = pd.concat(frames, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output_dir / "aggregate_task_metrics.parquet", engine="pyarrow", index=False)

    summary = (
        df.groupby(["span", "intervention"], as_index=False)
        .agg(
            n=("seed", "count"),
            mean_accuracy_restricted=("mean_accuracy_restricted", "mean"),
            mean_accuracy_full_vocab=("mean_accuracy_full_vocab", "mean"),
            mean_nll_restricted=("mean_nll_restricted", "mean"),
            mean_nll_full_vocab=("mean_nll_full_vocab", "mean"),
        )
        .sort_values(["intervention", "span"])
    )
    payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "input_shards": [str(d) for d in shard_dirs],
        "row_count": int(len(df)),
        "summary_rows": summary.to_dict(orient="records"),
        "exploratory_only": True,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    memo = ["# Semi-Natural Bridge Decision Memo (Merged Shards)", ""]
    for row in summary.to_dict(orient="records"):
        memo.append(
            f"- span={int(row['span'])}, intervention={row['intervention']}: "
            f"restricted_acc={float(row['mean_accuracy_restricted']):.4f}, "
            f"full_vocab_acc={float(row['mean_accuracy_full_vocab']):.4f}"
        )
    (args.output_dir / "decision_memo.md").write_text("\n".join(memo) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
