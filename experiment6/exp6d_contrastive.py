#!/usr/bin/env python3
"""CLI entrypoint for Experiment 6D: Contrastive Channel Assignment."""
from __future__ import annotations

import argparse
from pathlib import Path

from experiment4.common import now_timestamp, write_json
from experiment4.lora import LoRAHyperParams
from experiment4.pipeline import TrainConfig
from experiment6.config import RESULTS_ROOT, TARGET_MODELS
from experiment6.pipeline import run_6d


def _parse_seeds(raw: str) -> list[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 6D: Contrastive Channel Assignment")
    p.add_argument("--model", choices=TARGET_MODELS, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--output-root", default=str(RESULTS_ROOT / "exp6d_contrastive"))

    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--aux-interval", type=int, default=20)
    p.add_argument("--math-per-task", type=int, default=120)
    p.add_argument("--control-fraction", type=float, default=0.20)
    p.add_argument("--lambda-contrast", type=float, default=0.05,
                   help="Weight for the contrastive channel variance loss")

    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--si-rank", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = _parse_seeds(args.seeds)

    train_cfg = TrainConfig(
        max_steps=int(args.max_steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        warmup_steps=int(args.warmup_steps),
        grad_clip=float(args.grad_clip),
        max_length=int(args.max_length),
        log_every=int(args.log_every),
        aux_interval=int(args.aux_interval),
        math_per_task=int(args.math_per_task),
        control_fraction=float(args.control_fraction),
    )
    hp = LoRAHyperParams(
        rank=int(args.lora_rank),
        alpha=float(args.lora_alpha),
        dropout=float(args.lora_dropout),
        si_rank=int(args.si_rank),
    )

    out_root = Path(args.output_root)
    summary = run_6d(
        model_name=str(args.model),
        device=str(args.device),
        output_root=out_root,
        seeds=seeds,
        train_cfg=train_cfg,
        lora_hp=hp,
        lambda_contrast=float(args.lambda_contrast),
    )

    manifest = {
        "timestamp": now_timestamp(),
        "experiment": "6D",
        "status": "completed",
        "model": str(args.model),
        "device": str(args.device),
        "seeds": seeds,
        "train_config": train_cfg.__dict__,
        "lora": hp.__dict__,
        "lambda_contrast": float(args.lambda_contrast),
        "summary": summary,
    }
    write_json(out_root / str(args.model) / "run_manifest.json", manifest)
    print(f"[6D] completed: {out_root / str(args.model) / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
