#!/usr/bin/env python3
"""Experiment 4 dispatcher.

Usage:
    python -m experiment4.run --list
    python -m experiment4.run 4c --model llama-3.1-8b --device cuda:0
    python -m experiment4.run all --model all --device-map llama-3.1-8b:cuda:0,olmo-2-7b:cuda:1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EXPERIMENT4_DIR = Path(__file__).resolve().parent

EXPERIMENTS: dict[str, str] = {
    "4c": "experiment4.exp4c_si_trajectory_during_ft",
    "4a": "experiment4.exp4a_si_aware_lora",
    "4b": "experiment4.exp4b_si_loss_augmented",
}

ORDERED_ALL: tuple[str, ...] = ("4c", "4a", "4b")
TARGET_MODELS: tuple[str, ...] = ("llama-3.1-8b", "olmo-2-7b")


def _parse_device_map(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for token in [x.strip() for x in str(raw).split(",") if x.strip()]:
        model, dev = token.split(":", 1)
        out[model.strip()] = dev.strip()
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 4 scaffold entrypoints")
    parser.add_argument("experiment", nargs="?", choices=[*EXPERIMENTS.keys(), "all"])
    parser.add_argument("--model", default="all", choices=["all", *TARGET_MODELS])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="llama-3.1-8b:cuda:0,olmo-2-7b:cuda:1")
    parser.add_argument("--list", action="store_true", help="List experiment entrypoints")
    return parser.parse_args()


def run_one(experiment_id: str, model: str, device: str) -> int:
    module = EXPERIMENTS[experiment_id]
    cmd = [sys.executable, "-m", module, "--model", model, "--device", device]
    print(f"\n{'=' * 72}")
    print(f"Running Experiment {experiment_id} scaffold for model={model} device={device}")
    print(f"{'=' * 72}\n")
    return subprocess.call(cmd)


def main() -> None:
    args = parse_args()
    if args.list or args.experiment is None:
        print("Available Experiment 4 entrypoints:")
        for key, val in EXPERIMENTS.items():
            print(f"  {key:<4} {val}")
        print("Recommended full order: 4c -> 4a -> 4b")
        return

    model_targets = list(TARGET_MODELS) if args.model == "all" else [args.model]
    device_map = _parse_device_map(args.device_map)
    queue = ORDERED_ALL if args.experiment == "all" else (args.experiment,)

    for exp_id in queue:
        for model in model_targets:
            device = device_map.get(model, args.device)
            rc = run_one(exp_id, model, device)
            if rc != 0:
                print(f"\n*** {exp_id} failed for {model} with exit code {rc} ***")
                sys.exit(rc)


if __name__ == "__main__":
    main()
