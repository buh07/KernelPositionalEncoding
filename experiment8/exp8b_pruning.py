#!/usr/bin/env python3
"""CLI entrypoint for Experiment 8B: Tokenizer-Aware Pruning."""
from __future__ import annotations

import argparse
from pathlib import Path

from experiment4.common import now_timestamp, write_json
from experiment8.config import ABLATION_FRACTIONS, RESULTS_ROOT, TARGET_MODELS_8B
from experiment8.pipeline import run_8b


# Cross-model pairings for the mismatch test.
CROSS_MODEL_MAP = {
    "llama-3.1-8b": "olmo-2-7b",
    "olmo-2-7b": "llama-3.1-8b",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 8B: Tokenizer-Aware Pruning")
    p.add_argument("--model", choices=TARGET_MODELS_8B, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cross-model", default=None,
                   help="Model to use for mismatched SI mask (default: auto from CROSS_MODEL_MAP)")
    p.add_argument("--output-root", default=str(RESULTS_ROOT / "exp8b_pruning"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cross = args.cross_model or CROSS_MODEL_MAP.get(str(args.model))
    out_root = Path(args.output_root)

    summary = run_8b(
        model_name=str(args.model),
        device=str(args.device),
        output_root=out_root,
        fractions=ABLATION_FRACTIONS,
        cross_model_name=cross,
    )

    manifest = {
        "timestamp": now_timestamp(),
        "experiment": "8B",
        "status": "completed",
        "model": str(args.model),
        "device": str(args.device),
        "cross_model": cross,
        "fractions": list(ABLATION_FRACTIONS),
        "summary": summary,
    }
    write_json(out_root / str(args.model) / "run_manifest.json", manifest)
    print(f"[8B] completed: {out_root / str(args.model) / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
