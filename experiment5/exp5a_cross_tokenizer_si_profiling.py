#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from experiment5.common import now_timestamp, write_json
from experiment5.config import RESULTS_ROOT
from experiment5.pipeline import run_5a


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 5A: cross-tokenizer SI profiling")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--num-sequences", type=int, default=24)
    p.add_argument("--top-k-dims", type=int, default=64)
    p.add_argument("--synthetic-target-per-cell", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-root", default=str(RESULTS_ROOT / "exp5a_cross_tokenizer_si_profiling"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    summary = run_5a(
        device=str(args.device),
        output_root=out_root,
        seq_len=int(args.seq_len),
        num_sequences=int(args.num_sequences),
        top_k_dims=int(args.top_k_dims),
        synthetic_target_per_cell=int(args.synthetic_target_per_cell),
        seed=int(args.seed),
    )
    run_status = str(summary.get("status", "completed")) if isinstance(summary, dict) else "completed"
    manifest = {
        "timestamp": now_timestamp(),
        "experiment": "5A",
        "status": run_status,
        "device": str(args.device),
        "seq_len": int(args.seq_len),
        "num_sequences": int(args.num_sequences),
        "top_k_dims": int(args.top_k_dims),
        "synthetic_target_per_cell": int(args.synthetic_target_per_cell),
        "seed": int(args.seed),
        "artifacts": {
            "cross_tokenizer_r2_profiles": str(out_root / "cross_tokenizer_r2_profiles.parquet"),
            "tokenizer_entanglement_matrix": str(out_root / "tokenizer_entanglement_matrix.json"),
        },
        "summary": summary,
    }
    write_json(out_root / "run_manifest.json", manifest)
    print(f"[5A] {run_status}: {out_root / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
