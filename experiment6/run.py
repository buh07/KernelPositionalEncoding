#!/usr/bin/env python3
"""Experiment 6 dispatcher.

Usage:
    python -m experiment6.run --list
    python -m experiment6.run 6a --model tinyllama-1.1b --device cuda:0
    python -m experiment6.run all --model all --device-map tinyllama-1.1b:cuda:0,llama-3.1-8b:cuda:0,olmo-2-7b:cuda:1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EXPERIMENTS: dict[str, str] = {
    "6a": "experiment6.exp6a_gradient_routing",
    "6b": "experiment6.exp6b_anti_localization",
    "6c": "experiment6.exp6c_distillation",
    "6d": "experiment6.exp6d_contrastive",
    "6e": "experiment6.exp6e_tokenizer",
}

ORDERED_ALL: tuple[str, ...] = ("6a", "6b", "6c", "6d", "6e")
TARGET_MODELS: tuple[str, ...] = ("tinyllama-1.1b", "llama-3.1-8b", "olmo-2-7b")


def _parse_device_map(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for token in [x.strip() for x in str(raw).split(",") if x.strip()]:
        model, dev = token.split(":", 1)
        out[model.strip()] = dev.strip()
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 6 sub-experiments")
    parser.add_argument("experiment", nargs="?", choices=[*EXPERIMENTS.keys(), "all"])
    parser.add_argument("--model", default="all", choices=["all", *TARGET_MODELS])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="tinyllama-1.1b:cuda:0,llama-3.1-8b:cuda:0,olmo-2-7b:cuda:1")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--list", action="store_true", help="List experiment entrypoints")
    return parser.parse_args()


def run_one(experiment_id: str, model: str, device: str, seeds: str) -> int:
    module = EXPERIMENTS[experiment_id]
    cmd = [sys.executable, "-m", module, "--model", model, "--device", device]
    # 6E has --seed (single) instead of --seeds
    if experiment_id == "6e":
        first_seed = seeds.split(",")[0].strip()
        cmd.extend(["--seed", first_seed])
    else:
        cmd.extend(["--seeds", seeds])
    print(f"\n{'=' * 72}")
    print(f"Running Experiment {experiment_id.upper()} for model={model} device={device}")
    print(f"{'=' * 72}\n")
    return subprocess.call(cmd)


def main() -> None:
    args = parse_args()
    if args.list or args.experiment is None:
        print("Available Experiment 6 entrypoints:")
        for key, val in EXPERIMENTS.items():
            print(f"  {key:<4} {val}")
        print(f"Recommended full order: {' -> '.join(ORDERED_ALL)}")
        return

    model_targets = list(TARGET_MODELS) if args.model == "all" else [args.model]
    device_map = _parse_device_map(args.device_map)
    queue = ORDERED_ALL if args.experiment == "all" else (args.experiment,)

    for exp_id in queue:
        for model in model_targets:
            device = device_map.get(model, args.device)
            rc = run_one(exp_id, model, device, args.seeds)
            if rc != 0:
                print(f"\n*** {exp_id} failed for {model} with exit code {rc} ***")
                sys.exit(rc)


if __name__ == "__main__":
    main()
