#!/usr/bin/env python3
"""Shared utilities for reinforce_exp3 scripts: artifact emission, validation, statistics."""
from __future__ import annotations

import itertools
import json
import math
import os
import hashlib
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

from reinforce_exp3.common import (  # noqa: E402
    PRIMARY_MODELS,
    RESULTS_ROOT,
    SCHEMAS_ROOT,
    attach_run_metadata,
    command_manifest,
    ensure_dir,
    parse_csv_strs,
    read_json,
    safe_float,
    timestamp_now,
    write_json,
)

CORE_SCHEMA_FILES: dict[str, str] = {
    "preregistration.json": "preregistration.schema.json",
    "manifest.json": "artifact_manifest.schema.json",
    "summary.json": "summary.schema.json",
    "claim_impact.json": "claim_impact.schema.json",
    "data_dictionary.json": "data_dictionary.schema.json",
}


class CoverageContractError(RuntimeError):
    """Raised when a finalize/run coverage contract is violated."""


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

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
            msgs = [
                f"{path.name}@{'.'.join(str(x) for x in e.path) or '<root>'}: {e.message}"
                for e in errors
            ]
            return False, msgs
        return True, []
    except Exception as exc:
        return False, [f"schema validation failed for {path}: {type(exc).__name__}: {exc}"]


def validate_core_artifacts(
    out_dir: Path,
    required_table_paths: list[Path] | None = None,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for artifact_name, schema_name in CORE_SCHEMA_FILES.items():
        ok, msgs = validate_json_against_schema(
            out_dir / artifact_name, SCHEMAS_ROOT / schema_name
        )
        if not ok:
            errors.extend(msgs)
    if required_table_paths:
        for p in required_table_paths:
            if not Path(p).exists():
                errors.append(f"missing required table artifact: {p}")
    return len(errors) == 0, errors


def patch_schema_validation_flag(out_dir: Path, passed: bool) -> None:
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


# ---------------------------------------------------------------------------
# Core artifact emission
# ---------------------------------------------------------------------------

def _pipeline_context_from_env() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, field in [
        ("REINFORCE_EXP3_RUN_MODE", "run_mode"),
        ("REINFORCE_EXP3_RESUME_DECISION", "resume_decision"),
        ("REINFORCE_EXP3_LOG_PATH", "log_path"),
    ]:
        val = os.environ.get(key, "").strip()
        if val:
            out[field] = val
    dep_raw = os.environ.get("REINFORCE_EXP3_DEPENDENCY_INPUTS_JSON", "").strip()
    if dep_raw:
        try:
            out["dependency_inputs"] = json.loads(dep_raw)
        except Exception:
            out["dependency_inputs"] = {"raw": dep_raw}
    schema_passed = os.environ.get("REINFORCE_EXP3_SCHEMA_VALIDATION_PASSED", "").strip().lower()
    if schema_passed in {"true", "false"}:
        out["schema_validation_passed"] = schema_passed == "true"
    return out


def emit_core_artifacts(
    *,
    experiment_id: str,
    out_dir: Path,
    preregistration: dict[str, Any],
    manifest_extra: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
    summary: dict[str, Any],
    claim_impact: dict[str, Any],
    data_dictionary: dict[str, Any],
) -> None:
    ctx = _pipeline_context_from_env()
    ensure_dir(out_dir)

    # Build manifest from manifest_extra if manifest not provided directly
    if manifest is None:
        manifest = {
            "experiment_id": experiment_id,
            "timestamp": timestamp_now(),
            **(manifest_extra or {}),
        }

    def _attach(d: dict[str, Any]) -> dict[str, Any]:
        return attach_run_metadata(
            d,
            run_mode=ctx.get("run_mode"),
            dependency_inputs=ctx.get("dependency_inputs"),
            resume_decision=ctx.get("resume_decision"),
            schema_validation_passed=ctx.get("schema_validation_passed"),
            log_path=ctx.get("log_path"),
        )

    write_json(out_dir / "preregistration.json", preregistration)
    write_json(out_dir / "manifest.json", _attach(manifest))
    write_json(out_dir / "summary.json", _attach(summary))
    write_json(out_dir / "claim_impact.json", _attach(claim_impact))
    write_json(out_dir / "data_dictionary.json", data_dictionary)


def stream_seed(seed: int, stream_id: str | int) -> int:
    """Derive a deterministic child seed from a root seed and stream id."""
    base = int(seed) & 0xFFFFFFFF
    sid = str(stream_id).encode("utf-8")
    digest = hashlib.sha256(sid).hexdigest()
    stream_component = int(digest[:8], 16) & 0xFFFFFFFF
    return int((base ^ stream_component) & 0xFFFFFFFF)


def rng_for_stream(seed: int, stream_id: str | int) -> np.random.Generator:
    """Deterministic RNG with stable stream derivation policy."""
    return np.random.default_rng(stream_seed(seed, stream_id))


def enforce_coverage_contract(
    *,
    experiment_id: str,
    observed_models: Iterable[str],
    required_models: Iterable[str] | None = None,
    observed_tasks: Iterable[str] | None = None,
    required_tasks: Iterable[str] | None = None,
    observed_counts: dict[str, int] | None = None,
    min_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Validate required model/task/count coverage and raise on violation."""
    observed_model_set = set(str(m) for m in observed_models)
    required_model_set = set(str(m) for m in (required_models or []))
    observed_task_set = set(str(t) for t in (observed_tasks or []))
    required_task_set = set(str(t) for t in (required_tasks or []))
    observed_counts = dict(observed_counts or {})
    min_counts = dict(min_counts or {})

    errors: list[str] = []
    if required_model_set:
        missing_models = sorted(required_model_set - observed_model_set)
        if missing_models:
            errors.append(f"missing_models={missing_models}")
    if required_task_set:
        missing_tasks = sorted(required_task_set - observed_task_set)
        if missing_tasks:
            errors.append(f"missing_tasks={missing_tasks}")
    for key, min_val in min_counts.items():
        have = int(observed_counts.get(key, 0))
        need = int(min_val)
        if have < need:
            errors.append(f"count[{key}]={have} < required={need}")

    contract = {
        "experiment_id": experiment_id,
        "required_models": sorted(required_model_set),
        "observed_models": sorted(observed_model_set),
        "required_tasks": sorted(required_task_set),
        "observed_tasks": sorted(observed_task_set),
        "observed_counts": {str(k): int(v) for k, v in observed_counts.items()},
        "min_counts": {str(k): int(v) for k, v in min_counts.items()},
        "passed": len(errors) == 0,
    }
    if errors:
        contract["hard_fail_reason"] = "; ".join(errors)
        raise CoverageContractError(contract["hard_fail_reason"])
    return contract


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def parse_models_arg(raw: str, default: Iterable[str] = PRIMARY_MODELS) -> list[str]:
    if not str(raw).strip() or str(raw).strip().lower() == "all":
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


def synthetic_token_sequences(
    *,
    tokenizer: Any,
    n: int,
    seq_len: int,
    seed: int,
) -> list[list[int]]:
    """Generate deterministic synthetic token-id sequences using non-special ids."""
    rng = np.random.default_rng(int(seed))
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    if vocab_size <= 10:
        raise RuntimeError("hard_fail_reason: tokenizer vocab_size is invalid for synthetic sequence generation")

    special = set()
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        v = getattr(tokenizer, attr, None)
        if v is not None:
            special.add(int(v))

    valid_ids = [i for i in range(vocab_size) if i not in special]
    if len(valid_ids) < 32:
        raise RuntimeError("hard_fail_reason: too few non-special tokens for synthetic sequence generation")

    arr = np.asarray(valid_ids, dtype=np.int64)
    out: list[list[int]] = []
    for _ in range(int(n)):
        idx = rng.integers(0, len(arr), size=int(seq_len), endpoint=False)
        out.append(arr[idx].astype(int).tolist())
    return out


def ensure_profile_sequence_cache(
    *,
    tokenizer: Any,
    model_name: str,
    min_sequences: int,
    seq_len: int,
    seed: int = 20260501,
) -> dict[str, Any]:
    """Create reinforce_exp3-local synthetic profile cache when natural cache is absent.

    This function never writes into canonical Experiment 1 data paths.
    """
    canonical_dir = ROOT / "data" / "experiment1" / "wiki40b_en_pre2019" / str(model_name)
    canonical = sorted(canonical_dir.rglob("*.jsonl")) if canonical_dir.exists() else []
    natural = [
        p for p in canonical
        if ("synthetic" not in p.name.lower() and "auto_profile_sequences" not in p.name.lower())
    ]
    if natural:
        return {"status": "existing", "path": str(natural[0]), "created": False, "source": "canonical"}

    data_dir = (
        ROOT
        / "results"
        / "reinforce_exp3"
        / "_synthetic_profile_cache"
        / "experiment1"
        / "wiki40b_en_pre2019"
        / str(model_name)
    )
    existing = sorted(data_dir.rglob("*.jsonl")) if data_dir.exists() else []
    if existing:
        return {"status": "existing", "path": str(existing[0]), "created": False, "source": "reinforce_exp3_synthetic"}

    ensure_dir(data_dir)
    n = max(int(min_sequences), 64)
    L = max(int(seq_len), 32)
    seqs = synthetic_token_sequences(tokenizer=tokenizer, n=n, seq_len=L, seed=int(seed))

    out_path = data_dir / "auto_profile_sequences.synthetic.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for i, toks in enumerate(seqs):
            row = {
                "tokens": toks,
                "source": "reinforce_exp3_auto_profile_cache",
                "index": int(i),
            }
            f.write(json.dumps(row) + "\n")
    return {
        "status": "generated",
        "path": str(out_path),
        "created": True,
        "num_sequences": int(len(seqs)),
        "seq_len": int(L),
        "source": "reinforce_exp3_synthetic",
    }


# ---------------------------------------------------------------------------
# Curve-fitting utilities (mirrors exp3p2c / R12 conventions)
# ---------------------------------------------------------------------------

def _fit_linear(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = x.size
    if n < 2:
        return {"a": float("nan"), "b": float("nan"), "rss": float("nan"), "n": n}
    beta = np.polyfit(x, y, deg=1)
    pred = np.polyval(beta, x)
    rss = float(np.sum((y - pred) ** 2))
    return {"a": float(beta[1]), "b": float(beta[0]), "rss": rss, "n": n}


def _fit_piecewise(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Hinge model: y = a + b1*x + b2*max(0, x - tau). Grid-search tau then L-BFGS-B."""
    from scipy.optimize import minimize

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = x.size
    if n < 4:
        return {"a": float("nan"), "b1": float("nan"), "b2": float("nan"),
                "tau": float("nan"), "rss": float("nan"), "n": n}

    def _rss(tau: float) -> float:
        hinge = np.maximum(0.0, x - tau)
        X = np.column_stack([np.ones(n), x, hinge])
        try:
            coef, res, _, _ = np.linalg.lstsq(X, y, rcond=None)
            if res.size > 0:
                return float(res[0])
            pred = X @ coef
            return float(np.sum((y - pred) ** 2))
        except Exception:
            return float("inf")

    lo, hi = float(np.percentile(x, 10)), float(np.percentile(x, 90))
    grid = np.linspace(lo, hi, 30)
    best_tau = float(grid[np.argmin([_rss(t) for t in grid])])

    res = minimize(
        lambda t: _rss(float(t[0])),
        x0=[best_tau],
        bounds=[(lo, hi)],
        method="L-BFGS-B",
        options={"maxiter": 200, "ftol": 1e-12},
    )
    tau = float(res.x[0])

    hinge = np.maximum(0.0, x - tau)
    X = np.column_stack([np.ones(n), x, hinge])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    rss = float(np.sum((y - pred) ** 2))
    return {
        "a": float(coef[0]), "b1": float(coef[1]), "b2": float(coef[2]),
        "tau": tau, "rss": rss, "n": n,
    }


def _fit_logistic(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Sigmoid: y = L / (1 + exp(-k*(x - x0))). Grid init + curve_fit."""
    from scipy.optimize import curve_fit

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = x.size
    if n < 4:
        return {"L": float("nan"), "k": float("nan"), "x0": float("nan"),
                "rss": float("nan"), "n": n}

    def sigmoid(xv: np.ndarray, L: float, k: float, x0: float) -> np.ndarray:
        return L / (1.0 + np.exp(-k * (xv - x0)))

    best_rss = float("inf")
    best_params = (float(np.max(y)), 10.0, float(np.median(x)))
    for x0_init in np.linspace(float(np.percentile(x, 10)), float(np.percentile(x, 90)), 6):
        for k_init in [5.0, 20.0, 50.0]:
            try:
                popt, _ = curve_fit(
                    sigmoid, x, y,
                    p0=[float(np.max(y)), k_init, x0_init],
                    maxfev=2000,
                    bounds=([0, 0, float(x.min())], [float(np.max(y)) * 3, 200, float(x.max())]),
                )
                pred = sigmoid(x, *popt)
                rss = float(np.sum((y - pred) ** 2))
                if rss < best_rss:
                    best_rss = rss
                    best_params = tuple(float(p) for p in popt)  # type: ignore[assignment]
            except Exception:
                continue

    return {"L": best_params[0], "k": best_params[1], "x0": best_params[2],
            "rss": best_rss, "n": n}


def _bic(rss: float, n: int, k: int) -> float:
    if not (np.isfinite(rss) and n > k and rss > 0):
        return float("nan")
    sigma2 = rss / n
    return float(n * math.log(sigma2) + k * math.log(n))


def fit_three_models(
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    """Fit linear, threshold-piecewise, and logistic-sigmoid; return BIC votes."""
    lin = _fit_linear(x, y)
    pie = _fit_piecewise(x, y)
    sig = _fit_logistic(x, y)

    n = int(np.sum(np.isfinite(np.asarray(x)) & np.isfinite(np.asarray(y))))
    bic_lin = _bic(lin["rss"], n, k=2)
    bic_pie = _bic(pie["rss"], n, k=4)
    bic_sig = _bic(sig["rss"], n, k=3)

    bics = {"linear": bic_lin, "threshold_piecewise": bic_pie, "logistic_sigmoid": bic_sig}
    finite = {k: v for k, v in bics.items() if np.isfinite(v)}
    preferred = min(finite, key=lambda k: finite[k]) if finite else "linear"

    return {
        "linear": {**lin, "bic": safe_float(bic_lin)},
        "threshold_piecewise": {**pie, "bic": safe_float(bic_pie)},
        "logistic_sigmoid": {**sig, "bic": safe_float(bic_sig)},
        "preferred_model": preferred,
        "bic_delta_linear_minus_threshold": safe_float(bic_lin - bic_pie),
        "bic_delta_logistic_minus_threshold": safe_float(bic_sig - bic_pie),
        "n_points": n,
    }


def ordering_verdict(per_key_fits: dict[str, Any]) -> dict[str, Any]:
    preferred = [str(v.get("preferred_model", "linear")) for v in per_key_fits.values()]
    th = sum(1 for p in preferred if p == "threshold_piecewise")
    li = sum(1 for p in preferred if p == "linear")
    lo = sum(1 for p in preferred if p == "logistic_sigmoid")
    total = max(1, th + li + lo)
    return {
        "threshold_votes": int(th),
        "linear_votes": int(li),
        "logistic_votes": int(lo),
        "n_fits": total,
        "threshold_majority": bool(th > (li + lo)),
    }


# ---------------------------------------------------------------------------
# Statistical utilities
# ---------------------------------------------------------------------------

def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)[np.isfinite(np.asarray(x, dtype=float))]
    y = np.asarray(y, dtype=float)[np.isfinite(np.asarray(y, dtype=float))]
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return float("nan")
    sp = math.sqrt(
        max(((n1 - 1) * float(np.var(x, ddof=1)) + (n2 - 1) * float(np.var(y, ddof=1)))
            / max(1, n1 + n2 - 2), 1e-12)
    )
    return float((float(np.mean(x)) - float(np.mean(y))) / sp)


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 5000,
    seed: int = 0,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(boots, alpha))
    hi = float(np.quantile(boots, 1.0 - alpha))
    return float(np.mean(values)), lo, hi


def permutation_test_one_sided_less(
    a: np.ndarray,
    b: np.ndarray,
    n_perm: int = 10000,
    seed: int = 0,
) -> tuple[float, float]:
    """One-sided permutation test: H_A: mean(a) < mean(b).
    Returns (observed_diff, p_one_sided) where diff = mean(a) - mean(b).
    """
    a = np.asarray(a, dtype=float)[np.isfinite(np.asarray(a, dtype=float))]
    b = np.asarray(b, dtype=float)[np.isfinite(np.asarray(b, dtype=float))]
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan")
    observed = float(np.mean(a) - np.mean(b))
    combined = np.concatenate([a, b])
    na = a.size
    n_total = combined.size
    rng = np.random.default_rng(int(seed))
    null = np.empty(int(n_perm), dtype=float)
    for i in range(int(n_perm)):
        perm = rng.permutation(n_total)
        grp_a = combined[perm[:na]]
        grp_b = combined[perm[na:]]
        null[i] = float(np.mean(grp_a) - np.mean(grp_b))
    p_one = float(np.mean(null <= observed))
    return observed, p_one


def spearman_with_ci(
    x: Any,
    y: Any,
    n_boot: int = 5000,
    seed: int = 0,
) -> dict[str, float]:
    """Returns dict with keys: rho, pval, ci_lo, ci_hi."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    nan_result: dict[str, float] = {"rho": float("nan"), "pval": float("nan"),
                                     "ci_lo": float("nan"), "ci_hi": float("nan")}
    if x.size < 3:
        return nan_result
    rho, p_two = scipy_stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    idx = np.arange(x.size)
    boot_rhos = []
    for _ in range(n_boot):
        bi = rng.choice(idx, size=idx.size, replace=True)
        r, _ = scipy_stats.spearmanr(x[bi], y[bi])
        if np.isfinite(r):
            boot_rhos.append(float(r))
    if len(boot_rhos) < 10:
        return {"rho": float(rho), "pval": float(p_two), "ci_lo": float("nan"), "ci_hi": float("nan")}
    arr = np.array(boot_rhos)
    return {
        "rho": float(rho),
        "pval": float(p_two),
        "ci_lo": float(np.quantile(arr, 0.025)),
        "ci_hi": float(np.quantile(arr, 0.975)),
    }


def jaccard(set_a: set, set_b: set) -> float:
    union = len(set_a | set_b)
    if union == 0:
        return float("nan")
    return float(len(set_a & set_b) / union)


def holm_adjust_dict(pval_dict: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni correction for a dict of {key: p-value}.

    Returns a dict of {key: adjusted_p} with the same keys.
    """
    if not pval_dict:
        return {}
    keys = list(pval_dict.keys())
    pvals = [pval_dict[k] for k in keys]
    n = len(pvals)
    # Sort ascending
    order = sorted(range(n), key=lambda i: pvals[i])
    adjusted = [0.0] * n
    for rank, idx in enumerate(order):
        adjusted[idx] = min(1.0, pvals[idx] * (n - rank))
    # Enforce monotonicity: adjusted[i] >= adjusted[j] for rank[i] > rank[j]
    for i in range(1, n):
        prev_idx = order[i - 1]
        curr_idx = order[i]
        if adjusted[curr_idx] < adjusted[prev_idx]:
            adjusted[curr_idx] = adjusted[prev_idx]
    return {keys[i]: adjusted[i] for i in range(n)}


# ---------------------------------------------------------------------------
# Project-level model loading and correct head ablation
# ---------------------------------------------------------------------------

def load_model_for_exp(
    model_name: str,
    device: str = "cpu",
    attn_implementation: str | None = None,
):
    """Load model and tokenizer via project infrastructure (resolves local weights).

    Returns (model, tokenizer) with model already moved to device and in eval mode.
    Raises KeyError if model_name is not in the project MODELS registry.

    attn_implementation: if provided (e.g. "eager"), bypasses the LRU-cached project
    loader and loads directly from the local weights path using AutoModelForCausalLM.
    Required for models that use SDPA by default (e.g. OLMo-2) when
    output_attentions=True is needed — SDPA does not support returning attention
    weights, but "eager" attention does.
    """
    from experiment3.theory1_si_circuits import MODELS  # noqa: PLC0415
    from shared.models.loading import (  # noqa: PLC0415
        load_model as _proj_load_model,
        load_tokenizer as _proj_load_tokenizer,
    )

    spec = MODELS[model_name]
    tokenizer = _proj_load_tokenizer(spec)

    if attn_implementation is not None:
        loaded = _proj_load_model(
            spec,
            device_map=device,
            attn_implementation=attn_implementation,
        )
        model = loaded.model
        model.eval()
    else:
        loaded = _proj_load_model(spec)
        model = loaded.model
        model.eval()
        model.to(device)

    return model, tokenizer


from contextlib import contextmanager as _contextmanager


@_contextmanager
def head_output_ablation(
    model: Any,
    heads_to_zero: list[tuple[int, int]],
):
    """Zero specified attention heads via o_proj.register_forward_pre_hook.

    Wraps experiment3.theory1_si_circuits.head_output_ablation, converting
    (layer, head) int tuples to HeadID objects. This is the correct ablation
    method: it zeroes each head's contribution *before* the output projection,
    not after (which would zero mixed-head dimensions meaninglessly).
    """
    from experiment3.theory1_si_circuits import (  # noqa: PLC0415
        head_output_ablation as _exp3_ablation,
        HeadID,
    )
    head_ids = [HeadID(layer=int(l), head=int(h)) for l, h in heads_to_zero]
    with _exp3_ablation(model, head_ids):
        yield
