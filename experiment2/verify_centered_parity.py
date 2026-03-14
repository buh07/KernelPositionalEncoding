from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _collect_centered(root: Path, phase: str, run_id: str) -> pd.DataFrame:
    phase_root = root / phase / run_id
    rows = []
    for p in phase_root.glob("**/kernel_track_b_centered_summary.parquet"):
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        if df.empty:
            continue
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    key_cols = ["model", "split", "task", "intervention", "seed", "seq_len", "layer", "head"]
    return out[key_cols + ["mean_r2"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify centered-kernel parity between two Experiment2 runs.")
    parser.add_argument("--output-root", default="results/experiment2")
    parser.add_argument("--phase", required=True)
    parser.add_argument("--run-a", required=True, help="Inline reference run_id.")
    parser.add_argument("--run-b", required=True, help="Defer+posthoc run_id.")
    args = parser.parse_args()

    root = Path(args.output_root)
    a = _collect_centered(root, args.phase, args.run_a)
    b = _collect_centered(root, args.phase, args.run_b)
    if a.empty or b.empty:
        raise SystemExit("Missing centered summary data in one or both runs.")

    key_cols = ["model", "split", "task", "intervention", "seed", "seq_len", "layer", "head"]
    merged = a.merge(b, on=key_cols, how="inner", suffixes=("_a", "_b"))
    if merged.empty:
        raise SystemExit("No overlapping centered rows between runs.")
    delta = (merged["mean_r2_a"] - merged["mean_r2_b"]).abs().to_numpy(dtype=np.float64)
    mae = float(np.mean(delta))
    max_abs = float(np.max(delta))
    print(
        f"phase={args.phase} overlap_rows={len(merged)} "
        f"mae={mae:.3e} max_abs={max_abs:.3e}"
    )


if __name__ == "__main__":
    main()
