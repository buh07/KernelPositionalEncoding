#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from jsonschema import Draft202012Validator

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp2.common import (  # noqa: E402
    SCHEMAS_ROOT,
    LOGS_ROOT,
    PRIMARY_MODELS,
    RESULTS_ROOT,
    attach_run_metadata,
    command_manifest,
    ensure_dir,
    load_head_groups,
    load_kernels,
    parse_csv_strs,
    read_json,
    safe_float,
    timestamp_now,
    to_head_key,
    write_json,
)

CORE_SCHEMA_FILES: dict[str, str] = {
    "preregistration.json": "preregistration.schema.json",
    "manifest.json": "artifact_manifest.schema.json",
    "summary.json": "summary.schema.json",
    "claim_impact.json": "claim_impact.schema.json",
    "data_dictionary.json": "data_dictionary.schema.json",
}


def _pipeline_context_from_env() -> dict[str, Any]:
    out: dict[str, Any] = {}
    run_mode = os.environ.get("REINFORCE_EXP2_RUN_MODE", "").strip()
    if run_mode:
        out["run_mode"] = run_mode
    dep_raw = os.environ.get("REINFORCE_EXP2_DEPENDENCY_INPUTS_JSON", "").strip()
    if dep_raw:
        try:
            out["dependency_inputs"] = json.loads(dep_raw)
        except Exception:
            out["dependency_inputs"] = {"raw": dep_raw}
    resume_decision = os.environ.get("REINFORCE_EXP2_RESUME_DECISION", "").strip()
    if resume_decision:
        out["resume_decision"] = resume_decision
    schema_passed = os.environ.get("REINFORCE_EXP2_SCHEMA_VALIDATION_PASSED", "").strip().lower()
    if schema_passed in {"true", "false"}:
        out["schema_validation_passed"] = schema_passed == "true"
    log_path = os.environ.get("REINFORCE_EXP2_LOG_PATH", "").strip()
    if log_path:
        out["log_path"] = log_path
    calib_hash = os.environ.get("REINFORCE_EXP2_CALIBRATION_SPLIT_SHA256", "").strip()
    if calib_hash:
        out["calibration_split_sha256"] = calib_hash
    return out


def validate_json_against_schema(path: Path, schema_path: Path) -> tuple[bool, list[str]]:
    if not path.exists():
        return False, [f"missing file: {path}"]
    if not schema_path.exists():
        return False, [f"missing schema: {schema_path}"]
    try:
        payload = read_json(path)
    except Exception as exc:
        return False, [f"json parse failed for {path}: {type(exc).__name__}: {exc}"]
    try:
        schema = read_json(schema_path)
        validator = Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(payload), key=lambda e: str(e.path))
        if errors:
            msgs = []
            for err in errors:
                loc = ".".join(str(x) for x in err.path) or "<root>"
                msgs.append(f"{path.name}@{loc}: {err.message}")
            return False, msgs
        return True, []
    except Exception as exc:
        return False, [f"schema validation failed for {path}: {type(exc).__name__}: {exc}"]


def validate_core_artifacts(out_dir: Path, required_table_paths: list[Path] | None = None) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for artifact_name, schema_name in CORE_SCHEMA_FILES.items():
        ok, msgs = validate_json_against_schema(out_dir / artifact_name, SCHEMAS_ROOT / schema_name)
        if not ok:
            errors.extend(msgs)
    if required_table_paths:
        for p in required_table_paths:
            if not Path(p).exists():
                errors.append(f"missing required table artifact: {p}")
    return len(errors) == 0, errors


def patch_schema_validation_flag(out_dir: Path, passed: bool) -> None:
    """Set schema_validation_passed on core run-metadata artifacts in-place."""
    for name in ("manifest.json", "summary.json", "claim_impact.json"):
        path = Path(out_dir) / name
        if not path.exists():
            continue
        try:
            payload = read_json(path)
        except Exception:
            continue
        payload["schema_validation_passed"] = bool(passed)
        write_json(path, payload)


def parse_models_arg(raw: str, default: Iterable[str] = PRIMARY_MODELS) -> list[str]:
    if not str(raw).strip():
        return list(default)
    if str(raw).strip().lower() == "all":
        return list(default)
    vals = parse_csv_strs(raw)
    return vals if vals else list(default)


def parse_device_map(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in [x.strip() for x in str(raw).split(",") if x.strip()]:
        if ":" not in tok:
            continue
        model, dev = tok.split(":", 1)
        out[model.strip()] = dev.strip()
    return out


def build_default_experiment_dirs(experiment_id: str) -> tuple[Path, Path]:
    res_dir = ensure_dir(RESULTS_ROOT / experiment_id)
    log_dir = ensure_dir(LOGS_ROOT / experiment_id)
    return res_dir, log_dir


def emit_core_artifacts(
    *,
    experiment_id: str,
    out_dir: Path,
    preregistration: dict[str, Any],
    manifest: dict[str, Any],
    summary: dict[str, Any],
    claim_impact: dict[str, Any],
    data_dictionary: dict[str, Any],
) -> None:
    ctx = _pipeline_context_from_env()

    manifest_payload = attach_run_metadata(
        manifest,
        run_mode=ctx.get("run_mode"),
        dependency_inputs=ctx.get("dependency_inputs"),
        resume_decision=ctx.get("resume_decision"),
        schema_validation_passed=ctx.get("schema_validation_passed"),
        log_path=ctx.get("log_path"),
        calibration_split_sha256=ctx.get("calibration_split_sha256"),
    )
    summary_payload = attach_run_metadata(
        summary,
        run_mode=ctx.get("run_mode"),
        dependency_inputs=ctx.get("dependency_inputs"),
        resume_decision=ctx.get("resume_decision"),
        schema_validation_passed=ctx.get("schema_validation_passed"),
        log_path=ctx.get("log_path"),
        calibration_split_sha256=ctx.get("calibration_split_sha256"),
    )
    claim_payload = attach_run_metadata(
        claim_impact,
        run_mode=ctx.get("run_mode"),
        dependency_inputs=ctx.get("dependency_inputs"),
        resume_decision=ctx.get("resume_decision"),
        schema_validation_passed=ctx.get("schema_validation_passed"),
        log_path=ctx.get("log_path"),
        calibration_split_sha256=ctx.get("calibration_split_sha256"),
    )

    write_json(out_dir / "preregistration.json", preregistration)
    write_json(out_dir / "manifest.json", manifest_payload)
    write_json(out_dir / "summary.json", summary_payload)
    write_json(out_dir / "claim_impact.json", claim_payload)
    write_json(out_dir / "data_dictionary.json", data_dictionary)


def load_r2_summary(model_name: str) -> pd.DataFrame:
    p = ROOT / "results" / "experiment3" / "theory1_si_circuits" / model_name / "head_r2_summary.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_parquet(p)
    cols = set(df.columns)
    if "mean_r2" not in cols and "r2" in cols:
        df = df.rename(columns={"r2": "mean_r2"})
    needed = ["layer", "head", "mean_r2"]
    return df[needed].copy()


def load_boundary_scores(model_name: str) -> pd.DataFrame:
    p = ROOT / "results" / "experiment3" / "theory5b_boundary_detection" / model_name / "boundary_attention_scores.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["layer", "head", "boundary_attn_score"])
    df = pd.read_parquet(p)
    if "boundary_attn_score" not in df.columns:
        return pd.DataFrame(columns=["layer", "head", "boundary_attn_score"])
    return df[["layer", "head", "boundary_attn_score"]].copy()


def load_task_aligned_effects(model_name: str) -> pd.DataFrame:
    p = ROOT / "results" / "reinforce_exp" / "exp_r5b_regime_alignment" / model_name / "task_aligned_effects.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return float("nan")
    v1 = np.var(x, ddof=1)
    v2 = np.var(y, ddof=1)
    sp = math.sqrt(max(((n1 - 1) * v1 + (n2 - 1) * v2) / max(1, (n1 + n2 - 2)), 1e-12))
    d = float((np.mean(x) - np.mean(y)) / sp)
    j = 1.0 - 3.0 / max(1.0, (4.0 * (n1 + n2) - 9.0))
    return float(j * d)


def holm_adjust_dict(pvals: dict[str, float]) -> dict[str, float]:
    vals = [(k, safe_float(v)) for k, v in pvals.items() if np.isfinite(safe_float(v))]
    if not vals:
        return {}
    vals = sorted(vals, key=lambda kv: kv[1])
    m = len(vals)
    out: dict[str, float] = {}
    prev = 0.0
    for i, (k, p) in enumerate(vals):
        adj = min(1.0, max(prev, (m - i) * p))
        out[k] = float(adj)
        prev = adj
    return out


def ivw_meta(effects: list[float], variances: list[float]) -> tuple[float, float, float]:
    eff = np.asarray(effects, dtype=np.float64)
    var = np.asarray(variances, dtype=np.float64)
    keep = np.isfinite(eff) & np.isfinite(var) & (var > 0)
    if not np.any(keep):
        return float("nan"), float("nan"), float("nan")
    w = 1.0 / var[keep]
    mu = float(np.sum(w * eff[keep]) / np.sum(w))
    se = float(math.sqrt(1.0 / np.sum(w)))
    return mu, mu - 1.96 * se, mu + 1.96 * se


def run_shell(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd or ROOT), check=True, text=True, capture_output=True)


def make_common_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description, allow_abbrev=False)
    p.add_argument("--models", default=",".join(PRIMARY_MODELS))
    p.add_argument("--analysis-tier", choices=["confirmatory", "exploratory"], default="confirmatory")
    p.add_argument("--override-used", action="store_true")
    p.add_argument("--canonical-eligible", action="store_true", default=True)
    p.add_argument("--output-root", default=str(RESULTS_ROOT))
    return p


def head_dataframe_from_kernels(model_name: str) -> pd.DataFrame:
    kernels = load_kernels(model_name)
    rows: list[dict[str, Any]] = []
    for hk, vec in kernels.items():
        if not (hk.startswith("L") and "H" in hk):
            continue
        left, right = hk[1:].split("H", 1)
        layer, head = int(left), int(right)
        rows.append(
            {
                "model": model_name,
                "layer": layer,
                "head": head,
                "head_key": hk,
                "kernel": np.asarray(vec, dtype=np.float64),
            }
        )
    if not rows:
        raise RuntimeError(f"No kernels loaded for {model_name}")
    return pd.DataFrame(rows)


def attach_common_head_features(model_name: str, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    r2 = load_r2_summary(model_name)
    boundary = load_boundary_scores(model_name)
    out = out.merge(r2, on=["layer", "head"], how="left")
    out = out.merge(boundary, on=["layer", "head"], how="left")
    out["boundary_attn_score"] = out["boundary_attn_score"].astype(float)
    out["mean_r2"] = out["mean_r2"].astype(float)

    groups = load_head_groups(model_name)
    high = {to_head_key(int(x["layer"]), int(x["head"])) for x in groups.get("high_si", [])}
    low = {to_head_key(int(x["layer"]), int(x["head"])) for x in groups.get("low_si", [])}
    out["is_high_si"] = out["head_key"].isin(high)
    out["is_low_si"] = out["head_key"].isin(low)

    return out


def ttest_one_sided_greater(sample: np.ndarray) -> tuple[float, float, float]:
    sample = np.asarray(sample, dtype=np.float64)
    sample = sample[np.isfinite(sample)]
    if sample.size < 2:
        return float("nan"), float("nan"), float("nan")
    t_stat, p_two = scipy_stats.ttest_1samp(sample, popmean=0.0, nan_policy="omit")
    if not np.isfinite(t_stat) or not np.isfinite(p_two):
        return float("nan"), float("nan"), float("nan")
    p_one = float(p_two / 2.0) if float(t_stat) > 0 else float(1.0 - p_two / 2.0)
    return float(t_stat), float(p_two), p_one
