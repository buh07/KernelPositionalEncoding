#!/usr/bin/env python3
"""CLI entrypoint for Experiment 8A: SI Boundary Alignment Score."""
from __future__ import annotations

import argparse
from pathlib import Path

from experiment4.common import now_timestamp, write_json
from experiment8.config import RESULTS_ROOT, SIBAS_NUM_SEQUENCES, SIBAS_SEQ_LEN, TARGET_MODELS_8A
from experiment8.pipeline import run_8a


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 8A: SIBAS")
    p.add_argument("--model", choices=TARGET_MODELS_8A, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-sequences", type=int, default=SIBAS_NUM_SEQUENCES)
    p.add_argument("--seq-len", type=int, default=SIBAS_SEQ_LEN)
    p.add_argument("--output-root", default=str(RESULTS_ROOT / "exp8a_sibas"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)

    summary = run_8a(
        model_name=str(args.model),
        device=str(args.device),
        output_root=out_root,
        num_sequences=int(args.num_sequences),
        seq_len=int(args.seq_len),
    )

    manifest = {
        "timestamp": now_timestamp(),
        "experiment": "8A",
        "status": "completed",
        "model": str(args.model),
        "device": str(args.device),
        "num_sequences": int(args.num_sequences),
        "seq_len": int(args.seq_len),
        "summary": summary,
    }
    write_json(out_root / str(args.model) / "run_manifest.json", manifest)
    print(f"[8A] completed: {out_root / str(args.model) / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
