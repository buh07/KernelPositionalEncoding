#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from experiment5.common import now_timestamp, write_json
from experiment5.config import RESULTS_ROOT
from experiment5.pipeline import run_5c


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 5C: synthetic tokenizer perturbation battery")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--num-sequences", type=int, default=36)
    p.add_argument("--positions-per-type", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-root", default=str(RESULTS_ROOT / "exp5c_synthetic_tokenizer_perturbation"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    summary = run_5c(
        device=str(args.device),
        output_root=out_root,
        seq_len=int(args.seq_len),
        num_sequences=int(args.num_sequences),
        positions_per_type=int(args.positions_per_type),
        seed=int(args.seed),
    )
    run_status = str(summary.get("status", "completed")) if isinstance(summary, dict) else "completed"
    manifest = {
        "timestamp": now_timestamp(),
        "experiment": "5C",
        "status": run_status,
        "device": str(args.device),
        "seq_len": int(args.seq_len),
        "num_sequences": int(args.num_sequences),
        "positions_per_type": int(args.positions_per_type),
        "seed": int(args.seed),
        "artifacts": {
            "perturbation_sensitivity_matrix": str(out_root / "perturbation_sensitivity_matrix.json"),
            "perturbation_results": str(out_root / "perturbation_results.parquet"),
        },
        "summary": summary,
    }
    write_json(out_root / "run_manifest.json", manifest)
    print(f"[5C] {run_status}: {out_root / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
