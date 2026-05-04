#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results" / "reinforce_exp3"
LOGS_ROOT = ROOT / "logs" / "reinforce_exp3"
SCHEMAS_ROOT = PACKAGE_ROOT / "schemas"

PRIMARY_MODELS = ("llama-3.1-8b", "olmo-2-7b", "mistral-7b-v0.1")

# Paths to upstream experiment artifacts used as inputs
B1_RESULTS = ROOT / "results" / "reinforce_exp2" / "B1_kernel_taxonomy"
EXP3_SI_CIRCUITS = ROOT / "results" / "experiment3" / "theory1_si_circuits"
EXP3P2C_ROOT = ROOT / "results" / "experiment3_phase2" / "exp3p2c_redundancy_quantification"
EXP3P2B_ROOT = ROOT / "results" / "experiment3_phase2" / "exp3p2b_trivial_feature_control_multiseed"
R12_RESULTS = ROOT / "results" / "reinforce_exp" / "exp_new_r12_ordering_control"


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


def finite_or_nan(x: Any) -> float:
    v = safe_float(x)
    return v if np.isfinite(v) else float("nan")


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


def to_head_key(layer: int, head: int) -> str:
    return f"L{int(layer)}H{int(head)}"


def parse_head_key(head_key: str) -> tuple[int, int]:
    if not (head_key.startswith("L") and "H" in head_key):
        raise ValueError(f"Invalid head key: {head_key}")
    left, right = head_key[1:].split("H", 1)
    return int(left), int(right)


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
    upstream_sha256s: dict[str, str] | None = None,
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
    if upstream_sha256s is not None:
        out["upstream_sha256s"] = upstream_sha256s
    return out


def load_cluster_membership() -> "pd.DataFrame":  # type: ignore[name-defined]
    import pandas as pd
    p = B1_RESULTS / "cluster_membership.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"B1 cluster_membership.parquet not found at {p}. "
            "Run reinforce_exp2 B1_kernel_taxonomy first."
        )
    return pd.read_parquet(p)


def load_r2_summary(model_name: str) -> "pd.DataFrame":  # type: ignore[name-defined]
    import pandas as pd
    p = EXP3_SI_CIRCUITS / model_name / "head_r2_summary.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_parquet(p)
    if "mean_r2" not in df.columns and "r2" in df.columns:
        df = df.rename(columns={"r2": "mean_r2"})
    return df[["layer", "head", "mean_r2"]].copy()


def load_head_groups(model_name: str) -> dict[str, Any]:
    p = EXP3_SI_CIRCUITS / model_name / "head_groups.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return read_json(p)


def mean_or_nan(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))
