#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import hashlib
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results" / "reinforce_exp2"
LOGS_ROOT = ROOT / "logs" / "reinforce_exp2"
SCHEMAS_ROOT = PACKAGE_ROOT / "schemas"
ASSETS_ROOT = ROOT / "shared_storage" / "reinforce_exp2_assets"
PRIMARY_MODELS = ("llama-3.1-8b", "olmo-2-7b", "mistral-7b-v0.1")


def timestamp_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def maybe_file_sha256(path: Path) -> str | None:
    try:
        if path.exists():
            return file_sha256(path)
    except Exception:
        return None
    return None


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


def finite_or_nan(x: Any) -> float:
    v = safe_float(x)
    return v if np.isfinite(v) else float("nan")


def parse_csv_ints(raw: str) -> list[int]:
    vals: list[int] = []
    for tok in str(raw).split(","):
        t = tok.strip()
        if not t:
            continue
        vals.append(int(t))
    return sorted(set(vals))


def parse_csv_strs(raw: str) -> list[str]:
    vals: list[str] = []
    for tok in str(raw).split(","):
        t = tok.strip()
        if not t:
            continue
        vals.append(t)
    return vals


def command_manifest(
    *,
    experiment_id: str,
    command: str,
    model: str,
    analysis_tier: str = "confirmatory",
    canonical_eligible: bool = True,
    override_used: bool = False,
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
        "analysis_tier": analysis_tier,
        "canonical_eligible": bool(canonical_eligible),
        "override_used": bool(override_used),
    }
    if seed_set is not None:
        payload["seed_set"] = [int(x) for x in seed_set]
    if extras:
        payload.update(extras)
    return payload


def attach_run_metadata(
    payload: dict[str, Any],
    *,
    run_mode: str | None = None,
    dependency_inputs: dict[str, Any] | None = None,
    resume_decision: str | None = None,
    schema_validation_passed: bool | None = None,
    log_path: str | None = None,
    calibration_split_sha256: str | None = None,
) -> dict[str, Any]:
    out = dict(payload)
    if run_mode is not None:
        out["run_mode"] = str(run_mode)
    if dependency_inputs is not None:
        out["dependency_inputs"] = dependency_inputs
    if resume_decision is not None:
        out["resume_decision"] = str(resume_decision)
    if schema_validation_passed is not None:
        out["schema_validation_passed"] = bool(schema_validation_passed)
    if log_path is not None:
        out["log_path"] = str(log_path)
    if calibration_split_sha256 is not None:
        out["calibration_split_sha256"] = str(calibration_split_sha256)
    return out


def run_subprocess(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd or ROOT), check=check, text=True, capture_output=True)


def to_head_key(layer: int, head: int) -> str:
    return f"L{int(layer)}H{int(head)}"


def parse_head_key(head_key: str) -> tuple[int, int]:
    if not (head_key.startswith("L") and "H" in head_key):
        raise ValueError(f"Invalid head key: {head_key}")
    left, right = head_key[1:].split("H", 1)
    return int(left), int(right)


def load_head_groups(model_name: str) -> dict[str, Any]:
    path = ROOT / "results" / "experiment3" / "theory1_si_circuits" / model_name / "head_groups.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path)


def load_kernels(model_name: str) -> dict[str, np.ndarray]:
    candidates = [
        ROOT / "results" / "experiment3" / "theory8_position_ablation" / model_name / "estimated_kernels.json",
        ROOT / "results" / "reinforce_exp" / "exp_r3_core_replication" / model_name / "theory8_position_ablation" / model_name / "estimated_kernels.json",
    ]
    for p in candidates:
        if p.exists():
            raw = read_json(p)
            out: dict[str, np.ndarray] = {}
            for k, v in raw.items():
                if isinstance(v, list) and k.startswith("L") and "H" in k:
                    out[str(k)] = np.asarray(v, dtype=np.float64)
            if out:
                return out
    raise FileNotFoundError(f"No kernel file found for {model_name}. Tried: {candidates}")


def mean_or_nan(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))
