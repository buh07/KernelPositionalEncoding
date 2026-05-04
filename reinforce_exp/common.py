#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results" / "reinforce_exp"
LOGS_ROOT = ROOT / "logs" / "reinforce_exp"


def timestamp_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def safe_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_csv_ints(raw: str) -> list[int]:
    out: list[int] = []
    for tok in str(raw).split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(int(t))
    return sorted(set(out))


def parse_csv_strs(raw: str) -> list[str]:
    out: list[str] = []
    for tok in str(raw).split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(t)
    return out


def command_manifest(
    *,
    experiment_id: str,
    command: str,
    model: str,
    seed_set: list[int] | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "timestamp": timestamp_now(),
        "experiment_id": experiment_id,
        "cwd": str(ROOT),
        "python": os.environ.get("PYTHON", str(ROOT / ".venv" / "bin" / "python")),
        "command": command,
        "model": model,
    }
    if seed_set is not None:
        payload["seed_set"] = [int(x) for x in seed_set]
    if extras:
        payload.update(extras)
    return payload
