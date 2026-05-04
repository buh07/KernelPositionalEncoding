#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from experiment4.config import RESULTS_ROOT, TARGET_MODELS
from experiment4.common import now_timestamp, write_json
from experiment4.lora import LoRAHyperParams
from experiment4.pipeline import TrainConfig, run_4c_trajectory


def _parse_checkpoints(raw: str, max_steps: int) -> tuple[int, ...]:
    raw_s = str(raw).strip().lower()
    if raw_s == "auto":
        n_points = 7  # default: 6 intervals (e.g., 0,50,...,300 for max_steps=300)
        vals = sorted({int(round(i * max_steps / max(1, n_points - 1))) for i in range(n_points)})
        vals[0] = 0
        vals[-1] = int(max_steps)
        return tuple(int(x) for x in vals)

    vals: list[int] = []
    for tok in [x.strip() for x in str(raw).split(",") if x.strip()]:
        v = int(tok)
        if v < 0:
            v = int(max_steps)
        if v > int(max_steps):
            v = int(max_steps)
        vals.append(v)
    if 0 not in vals:
        vals.insert(0, 0)
    if int(max_steps) not in vals:
        vals.append(int(max_steps))
    uniq = sorted(set(vals))
    return tuple(int(x) for x in uniq)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 4C: SI trajectory during standard FT")
    p.add_argument("--model", choices=TARGET_MODELS, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-root", default=str(RESULTS_ROOT / "exp4c_si_trajectory_during_ft"))

    p.add_argument("--max-steps", type=int, default=300)
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

    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--si-rank", type=int, default=1)

    p.add_argument(
        "--checkpoints",
        default="auto",
        help="Comma-separated checkpoint steps, -1 for final step, or 'auto' (default).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
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
    checkpoints = _parse_checkpoints(args.checkpoints, max_steps=int(args.max_steps))

    out_root = Path(args.output_root)
    summary = run_4c_trajectory(
        model_name=str(args.model),
        device=str(args.device),
        output_root=out_root,
        train_cfg=train_cfg,
        lora_hp=hp,
        checkpoints=checkpoints,
    )

    manifest = {
        "timestamp": now_timestamp(),
        "experiment": "4C",
        "status": "completed",
        "model": str(args.model),
        "device": str(args.device),
        "train_config": train_cfg.__dict__,
        "lora": hp.__dict__,
        "checkpoints": [int(x) for x in checkpoints],
        "artifacts": {
            "trajectory": str(out_root / str(args.model) / "si_trajectory_during_ft.parquet"),
            "summary": str(out_root / str(args.model) / "si_trajectory_summary.json"),
        },
        "summary": summary,
    }
    write_json(out_root / str(args.model) / "run_manifest.json", manifest)
    print(f"[4C] completed: {out_root / str(args.model) / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
