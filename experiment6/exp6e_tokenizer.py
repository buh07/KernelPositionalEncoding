#!/usr/bin/env python3
"""CLI entrypoint for Experiment 6E: SI-Optimized Tokenizer Design."""
from __future__ import annotations

import argparse
from pathlib import Path

from experiment4.common import now_timestamp, write_json
from experiment6.config import RESULTS_ROOT, TARGET_MODELS
from experiment6.pipeline import run_6e


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 6E: SI-Optimized Tokenizer")
    p.add_argument("--model", choices=TARGET_MODELS, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-root", default=str(RESULTS_ROOT / "exp6e_tokenizer"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_root = Path(args.output_root)
    summary = run_6e(
        model_name=str(args.model),
        device=str(args.device),
        output_root=out_root,
        seed=int(args.seed),
    )

    manifest = {
        "timestamp": now_timestamp(),
        "experiment": "6E",
        "status": "completed",
        "model": str(args.model),
        "device": str(args.device),
        "seed": int(args.seed),
        "summary": summary,
    }
    write_json(out_root / str(args.model) / "run_manifest.json", manifest)
    print(f"[6E] completed: {out_root / str(args.model) / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
