#!/usr/bin/env python3
"""Experiment 7 dispatcher.

Usage:
  python -m experiment7.run --list
  python -m experiment7.run 7a
  python -m experiment7.run 7b --device cuda:0
  python -m experiment7.run all --device cuda:0
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiment4.common import now_timestamp, write_json
from experiment7.exp7a_welch_r2 import run_7a
from experiment7.exp7b_phase_transition import DEFAULT_MODELS, DEFAULT_SEQ_LENS, run_7b


ORDERED: tuple[str, ...] = ("7a", "7b")


def _parse_csv(raw: str) -> list[str]:
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


def _parse_int_csv(raw: str) -> list[int]:
    return [int(tok.strip()) for tok in str(raw).split(",") if tok.strip()]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 7 dispatcher")
    p.add_argument("experiment", nargs="?", choices=[*ORDERED, "all"])
    p.add_argument("--list", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-root", default="results/experiment7")
    p.add_argument("--tracka-md", default="experiment1/experiment1results.md")
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--seq-lens", default=",".join(str(x) for x in DEFAULT_SEQ_LENS))
    p.add_argument("--calibration-mode", choices=["single_condition", "global_fit"], default="single_condition")
    p.add_argument("--calibration-model", default="")
    p.add_argument("--calibration-seq-len", type=int, default=0)
    p.add_argument("--epsilon", type=float, default=0.01)
    p.add_argument("--sparsity-sequences", type=int, default=6)
    p.add_argument("--c1-num-seeds", type=int, default=3)
    p.add_argument("--c1-synthetic-count", type=int, default=64)
    p.add_argument("--c1-ntp-count-per-seed", type=int, default=64)
    p.add_argument("--c1-batch-size-synth", type=int, default=8)
    p.add_argument("--c1-batch-size-ntp", type=int, default=2)
    p.add_argument("--force-rerun-c1", action="store_true")
    p.add_argument("--reuse-effective-sparsity", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.list or args.experiment is None:
        print("Experiment 7 entrypoints:")
        print("  7a  exp7a_welch_r2")
        print("  7b  exp7b_phase_transition")
        print("  all 7a -> 7b")
        return

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    models = _parse_csv(args.models)
    seq_lens = _parse_int_csv(args.seq_lens)

    queue = ORDERED if args.experiment == "all" else (str(args.experiment),)
    status_rows: list[dict[str, object]] = []

    for exp in queue:
        if exp == "7a":
            print("[exp7] running 7A (Welch/coherence vs R²)", flush=True)
            rep = run_7a(
                output_root=out_root,
                tracka_md_path=Path(args.tracka_md),
                bootstrap_seed=int(args.bootstrap_seed),
                bootstrap_samples=int(args.bootstrap_samples),
            )
            status_rows.append({"experiment": "7A", "status": str(rep.get("status", "completed"))})
        elif exp == "7b":
            print(f"[exp7] running 7B (phase transition) on device={args.device}", flush=True)
            rep = run_7b(
                output_root=out_root,
                device=str(args.device),
                models=models,
                seq_lens=seq_lens,
                calibration_mode=str(args.calibration_mode),
                calibration_model=(str(args.calibration_model).strip() or None),
                calibration_seq_len=(int(args.calibration_seq_len) if int(args.calibration_seq_len) > 0 else None),
                epsilon=float(args.epsilon),
                sparsity_sequences=int(args.sparsity_sequences),
                c1_num_seeds=int(args.c1_num_seeds),
                c1_synthetic_count=int(args.c1_synthetic_count),
                c1_ntp_count_per_seed=int(args.c1_ntp_count_per_seed),
                c1_batch_size_synth=int(args.c1_batch_size_synth),
                c1_batch_size_ntp=int(args.c1_batch_size_ntp),
                force_rerun_c1=bool(args.force_rerun_c1),
                reuse_effective_sparsity=bool(args.reuse_effective_sparsity),
            )
            status_rows.append({"experiment": "7B", "status": str(rep.get("status", "completed"))})
        else:
            raise RuntimeError(f"Unsupported experiment key: {exp}")

    manifest = {
        "timestamp": now_timestamp(),
        "requested": str(args.experiment),
        "queue": list(queue),
        "device": str(args.device),
        "models": models,
        "seq_lens": [int(x) for x in seq_lens],
        "status_rows": status_rows,
    }
    write_json(out_root / "run_manifest.json", manifest)
    print(f"[exp7] completed queue: {queue}")
    print(f"[exp7] manifest: {out_root / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
