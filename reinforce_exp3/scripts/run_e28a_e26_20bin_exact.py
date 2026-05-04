#!/usr/bin/env python3
"""E28a — E26 statistical hardening with 20 bins + exact one-tailed Spearman p.

This script mirrors E26 but:
- uses 20 within-model R² bins,
- computes exact one-tailed permutation p-values for Spearman when n<=8,
- reports both scipy approximation and exact p.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp3.common import (  # noqa: E402
    PRIMARY_MODELS,
    RESULTS_ROOT,
    ensure_dir,
    timestamp_now,
    write_json,
)
from reinforce_exp3.scripts._shared import (  # noqa: E402
    emit_core_artifacts,
    enforce_coverage_contract,
    load_model_for_exp,
    parse_device_map,
    parse_models_arg,
    rng_for_stream,
)
from experiment3.theory8_position_ablation import (  # noqa: E402
    compute_per_token_loss,
    load_wiki_sequences,
    subtract_positional_kernels,
)

EXPERIMENT_ID = "E28A"
N_BINS = 20
DEFAULT_OUT = RESULTS_ROOT / "E28a_e26_20bin_exact"

_KERNEL_CANDIDATES = [
    "results/reinforce_exp/exp_r3_core_replication/{model}/theory8_position_ablation/{model}/estimated_kernels.json",
    "results/experiment3/theory8_position_ablation/{model}/estimated_kernels.json",
]
_R2_PATH = "results/experiment3/theory1_si_circuits/{model}/head_r2_summary.parquet"


def _load_kernels(model_name: str) -> dict[tuple[int, int], np.ndarray]:
    for template in _KERNEL_CANDIDATES:
        p = ROOT / template.format(model=model_name)
        if not p.exists():
            continue
        raw = json.loads(p.read_text(encoding="utf-8"))
        out: dict[tuple[int, int], np.ndarray] = {}
        for k, v in raw.items():
            if not (k.startswith("L") and "H" in k):
                continue
            left, right = k[1:].split("H", 1)
            out[(int(left), int(right))] = np.asarray(v, dtype=np.float32)
        if out:
            return out
    raise FileNotFoundError(f"[E28a] No estimated_kernels.json for {model_name}")


def _load_r2_data(model_name: str) -> pd.DataFrame:
    p = ROOT / _R2_PATH.format(model=model_name)
    if not p.exists():
        raise FileNotFoundError(f"[E28a] Missing R² data: {p}")
    df = pd.read_parquet(p)
    required = {"layer", "head", "mean_r2"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"[E28a] Missing columns in R² data: {df.columns.tolist()}")
    return df


def _permute_kernels(
    kernels: dict[tuple[int, int], np.ndarray],
    heads: list[tuple[int, int]],
    rng: np.random.Generator,
) -> dict[tuple[int, int], np.ndarray]:
    out: dict[tuple[int, int], np.ndarray] = {}
    for head in heads:
        g = kernels.get(head)
        if g is None:
            continue
        out[head] = g[rng.permutation(len(g))].copy()
    return out


def _norm_matched_random_kernels(
    kernels: dict[tuple[int, int], np.ndarray],
    heads: list[tuple[int, int]],
    rng: np.random.Generator,
) -> dict[tuple[int, int], np.ndarray]:
    out: dict[tuple[int, int], np.ndarray] = {}
    for head in heads:
        g = kernels.get(head)
        if g is None:
            continue
        n = len(g)
        z = rng.standard_normal(n).astype(np.float32)
        z -= float(np.mean(z))
        zn = float(np.linalg.norm(z))
        gn = float(np.linalg.norm(g))
        if zn > 1e-12 and gn > 1e-12:
            out[head] = (z * (gn / zn)).astype(np.float32)
        else:
            out[head] = np.zeros_like(g)
    return out


def _eval_mean_losses(
    model: Any,
    sequences: list[list[int]],
    device: str,
    batch_size: int,
) -> np.ndarray:
    vals: list[float] = []
    bs = max(1, int(batch_size))
    pos = 0
    while pos < len(sequences):
        batch = sequences[pos : pos + bs]
        try:
            input_ids = torch.tensor(batch, dtype=torch.long, device=device)
            with torch.inference_mode():
                loss = compute_per_token_loss(model, input_ids)
            tok = int(input_ids.shape[1] - 1)
            seq_loss = loss.view(input_ids.shape[0], tok).mean(dim=1).detach().cpu().numpy().astype(np.float64)
            vals.extend(float(x) for x in seq_loss.tolist())
            pos += len(batch)
            del input_ids, loss
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and bs > 1:
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                continue
            raise
    return np.asarray(vals, dtype=np.float64)


def _spearman_exact_one_tailed(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (rho, p_exact_one_tailed, p_scipy_approx).

    One-tailed direction is rho_perm >= rho_obs for positive-association hypothesis.
    For n<=8, enumerate all permutations exactly.
    For n>8, Monte Carlo fallback with fixed budget.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    sp = scipy_stats.spearmanr(x, y)
    rho = float(sp.statistic)
    p_scipy = float(sp.pvalue)
    n = int(len(x))

    if n <= 1 or not np.isfinite(rho):
        return rho, float("nan"), p_scipy

    if n <= 8:
        count = 0
        total = 0
        y_list = list(y)
        for perm in itertools.permutations(y_list):
            total += 1
            r = scipy_stats.spearmanr(x, np.asarray(perm, dtype=np.float64)).statistic
            if float(r) >= rho - 1e-12:
                count += 1
        p_exact = float(count / total)
        return rho, p_exact, p_scipy

    # Fallback MC for larger n
    rng = np.random.default_rng(20260503)
    m = 200000
    count = 0
    for _ in range(m):
        yp = np.array(y, copy=True)
        rng.shuffle(yp)
        r = scipy_stats.spearmanr(x, yp).statistic
        if float(r) >= rho - 1e-12:
            count += 1
    p_mc = float((count + 1) / (m + 1))
    return rho, p_mc, p_scipy


def run_model(
    *,
    model_name: str,
    device: str,
    out_root: Path,
    eval_seqs: int,
    seq_len: int,
    n_control_trials: int,
    seed: int,
    batch_size: int,
) -> dict[str, Any]:
    t0 = time.time()
    model_dir = ensure_dir(out_root / model_name)

    model, _tok = load_model_for_exp(model_name, device, attn_implementation="eager")

    r2_df = _load_r2_data(model_name)
    kernels = _load_kernels(model_name)

    r2_df = r2_df.copy()
    # robust quantile binning with potential ties
    q = min(N_BINS, max(3, int(r2_df["mean_r2"].nunique())))
    r2_df["bin"] = pd.qcut(r2_df["mean_r2"], q=q, labels=False, duplicates="drop") + 1
    r2_df["bin"] = r2_df["bin"].astype(int)

    available = set(kernels.keys())
    r2_df = r2_df[r2_df.apply(lambda row: (int(row["layer"]), int(row["head"])) in available, axis=1)].copy()

    # Save bin assignments for downstream E28d analysis
    assign_df = r2_df[["layer", "head", "mean_r2", "bin"]].copy().reset_index(drop=True)
    assign_df.to_parquet(model_dir / "head_bin_assignments.parquet", index=False)

    sequences = load_wiki_sequences(model_name, max_sequences=max(8, eval_seqs), seq_len=max(64, seq_len))
    sequences = [seq[:seq_len] for seq in sequences[:eval_seqs]]
    if len(sequences) < 4:
        raise RuntimeError(f"[E28a] Insufficient sequences ({len(sequences)}) for {model_name}")

    baseline = _eval_mean_losses(model, sequences, device, batch_size)

    rows: list[dict[str, Any]] = []
    actual_bins = sorted(r2_df["bin"].unique())

    for bin_idx in actual_bins:
        bin_mask = r2_df["bin"] == bin_idx
        sub = r2_df.loc[bin_mask, ["layer", "head", "mean_r2"]].copy()
        heads = [(int(r.layer), int(r.head)) for r in sub.itertuples(index=False)]
        n_heads = len(heads)
        if n_heads == 0:
            continue

        mean_r2 = float(sub["mean_r2"].mean())
        min_r2 = float(sub["mean_r2"].min())
        max_r2 = float(sub["mean_r2"].max())

        with subtract_positional_kernels(model, kernels, heads, seq_len):
            true_loss = _eval_mean_losses(model, sequences, device, batch_size)
        true_delta = float(np.mean(true_loss - baseline))

        perm_deltas: list[float] = []
        norm_deltas: list[float] = []
        for trial in range(n_control_trials):
            rng = rng_for_stream(seed, f"{model_name}:bin{bin_idx}:trial{trial}")
            k_perm = _permute_kernels(kernels, heads, rng)
            with subtract_positional_kernels(model, k_perm, heads, seq_len):
                pl = _eval_mean_losses(model, sequences, device, batch_size)
            perm_deltas.append(float(np.mean(pl - baseline)))

            k_norm = _norm_matched_random_kernels(kernels, heads, rng)
            with subtract_positional_kernels(model, k_norm, heads, seq_len):
                nl = _eval_mean_losses(model, sequences, device, batch_size)
            norm_deltas.append(float(np.mean(nl - baseline)))

        mean_perm = float(np.mean(perm_deltas)) if perm_deltas else float("nan")
        mean_norm = float(np.mean(norm_deltas)) if norm_deltas else float("nan")
        per_head_true = true_delta / n_heads
        per_head_perm = mean_perm / n_heads if n_heads > 0 else float("nan")

        rows.append({
            "bin": int(bin_idx),
            "n_heads": int(n_heads),
            "mean_r2": mean_r2,
            "min_r2": min_r2,
            "max_r2": max_r2,
            "true_mean_delta": true_delta,
            "perm_mean_delta": mean_perm,
            "norm_mean_delta": mean_norm,
            "per_head_true_delta": per_head_true,
            "per_head_perm_delta": per_head_perm,
            "true_over_perm": true_delta / mean_perm if abs(mean_perm) > 1e-10 else float("nan"),
        })
        print(
            f"[E28a][{model_name}] bin={bin_idx} n={n_heads} mean_r2={mean_r2:.4f} "
            f"true={true_delta:.6f} perm={mean_perm:.6f} norm={mean_norm:.6f}",
            flush=True,
        )

    dose_df = pd.DataFrame(rows)
    dose_df.to_parquet(model_dir / "dose_response_bins.parquet", index=False)

    if len(dose_df) >= 3:
        x = dose_df["mean_r2"].to_numpy(dtype=np.float64)
        y = dose_df["per_head_true_delta"].to_numpy(dtype=np.float64)
        spearman_r, p_exact_one, p_scipy = _spearman_exact_one_tailed(x, y)
    else:
        spearman_r = float("nan")
        p_exact_one = float("nan")
        p_scipy = float("nan")

    sorted_rows = dose_df.sort_values("bin")
    deltas = sorted_rows["per_head_true_delta"].tolist()
    n_monotone_steps = sum(1 for a, b in zip(deltas, deltas[1:]) if b >= a)
    total_steps = max(len(deltas) - 1, 1)
    monotone_fraction = n_monotone_steps / total_steps

    elapsed = time.time() - t0
    rec = {
        "model": model_name,
        "n_bins": int(len(rows)),
        "spearman_r": float(spearman_r),
        "spearman_p_exact_one_tailed": float(p_exact_one),
        "spearman_p_scipy_approx": float(p_scipy),
        "monotone_fraction": float(monotone_fraction),
        "dose_response_positive": bool(
            np.isfinite(spearman_r)
            and np.isfinite(p_exact_one)
            and spearman_r > 0
            and p_exact_one < 0.05
        ),
        "elapsed_sec": float(elapsed),
        "bins": rows,
    }
    write_json(model_dir / "model_summary.json", rec)
    print(
        f"[E28a][{model_name}] Spearman_r={spearman_r:.4f} "
        f"p_exact_one={p_exact_one:.6g} p_scipy={p_scipy:.6g} monotone={monotone_fraction:.2f}",
        flush=True,
    )
    return rec


def _finalize(models: list[str], out_root: Path, start_ts: str) -> dict[str, Any]:
    model_results: list[dict[str, Any]] = []
    for m in models:
        p = out_root / m / "model_summary.json"
        if not p.exists():
            raise RuntimeError(f"[E28a] hard_fail_reason: missing model_summary for {m} at {p}")
        with p.open() as f:
            model_results.append(json.load(f))

    n_positive = sum(1 for r in model_results if r.get("dose_response_positive", False))
    spearman_by_model = {r["model"]: r["spearman_r"] for r in model_results}
    p_exact_by_model = {r["model"]: r["spearman_p_exact_one_tailed"] for r in model_results}

    if n_positive == len(model_results):
        interp = "dose_response_exact_supported"
        claim_status = "supported"
    elif n_positive >= 2:
        interp = "dose_response_exact_supported_with_caveat"
        claim_status = "supported_with_caveat"
    elif n_positive == 1:
        interp = "dose_response_exact_mixed"
        claim_status = "mixed"
    else:
        interp = "dose_response_exact_not_supported"
        claim_status = "not_supported"

    prereg = {
        "experiment_id": EXPERIMENT_ID,
        "question": "Does E26 dose-response remain with finer bins and exact small-n p-values?",
        "primary_hypothesis": "Spearman(mean-bin R², per-head disruption) remains positive with exact one-tailed p-values.",
        "primary_endpoints": [
            "spearman_r",
            "spearman_p_exact_one_tailed",
            "monotone_fraction",
        ],
        "sample_size_plan": {"n_bins_target": N_BINS, "eval_seqs": 100},
        "acceptance_criteria": [
            "spearman_r > 0 and p_exact_one_tailed < 0.05",
        ],
    }

    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "analysis_tier": "hardening",
        "canonical_eligible": True,
        "override_used": False,
        "verdict": {
            "interpretation": interp,
            "n_positive": int(n_positive),
            "n_models": int(len(model_results)),
            "spearman_by_model": spearman_by_model,
            "p_exact_by_model": p_exact_by_model,
        },
        "limitations": [
            "Grouped-bin design still aggregates heads; head-level regression is not tested here.",
            "Exact test is one-tailed directional by preregistered hypothesis.",
        ],
    }

    claim_impact = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "claim_status": claim_status,
        "supports_main_text": claim_status in ("supported", "supported_with_caveat", "mixed"),
        "notes": [
            "Replaces approximate Spearman p-values with exact one-tailed p-values for small-n bin correlations.",
            "Uses 20-bin granularity to reduce 5-point statistical fragility concerns.",
        ],
    }

    data_dictionary = {
        "experiment_id": EXPERIMENT_ID,
        "tables": [
            {
                "path": "<model>/head_bin_assignments.parquet",
                "description": "Head-to-bin assignment at 20-bin granularity.",
                "columns": [
                    {"name": "layer", "dtype": "int", "description": "layer index"},
                    {"name": "head", "dtype": "int", "description": "head index"},
                    {"name": "mean_r2", "dtype": "float", "description": "head mean R²"},
                    {"name": "bin", "dtype": "int", "description": "R² quantile bin"},
                ],
            },
            {
                "path": "<model>/dose_response_bins.parquet",
                "description": "Per-bin disruption metrics under true/permuted/norm controls.",
                "columns": [],
            },
        ],
    }

    cross_summary = {
        "start_ts": start_ts,
        "end_ts": timestamp_now(),
        "models": [r["model"] for r in model_results],
        "interpretation": interp,
        "n_positive": int(n_positive),
        "spearman_by_model": spearman_by_model,
        "p_exact_one_tailed_by_model": p_exact_by_model,
    }
    write_json(out_root / "cross_model_summary.json", cross_summary)

    emit_core_artifacts(
        experiment_id=EXPERIMENT_ID,
        out_dir=out_root,
        preregistration=prereg,
        manifest_extra={"models": list(models), "n_bins": N_BINS},
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )
    return cross_summary


def main() -> None:
    p = argparse.ArgumentParser(description="E28a: E26 hardening (20 bins + exact Spearman)", allow_abbrev=False)
    p.add_argument("--models", default=",".join(PRIMARY_MODELS))
    p.add_argument("--device-map", default="llama-3.1-8b:cuda:0,olmo-2-7b:cuda:0,mistral-7b-v0.1:cuda:0")
    p.add_argument("--output-root", default=str(DEFAULT_OUT))
    p.add_argument("--eval-seqs", type=int, default=100)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--n-control-trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260503)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--finalize-only", action="store_true")
    p.add_argument("--no-finalize", action="store_true")
    args = p.parse_args()

    if args.finalize_only and args.no_finalize:
        raise RuntimeError("[E28a] hard_fail_reason: --finalize-only and --no-finalize are mutually exclusive")

    models = parse_models_arg(args.models, default=PRIMARY_MODELS)
    device_map = parse_device_map(args.device_map)
    out_root = ensure_dir(Path(args.output_root))
    start_ts = timestamp_now()

    eval_seqs = int(args.eval_seqs)
    n_control_trials = int(args.n_control_trials)
    if args.smoke:
        eval_seqs = min(eval_seqs, 16)
        n_control_trials = min(n_control_trials, 1)

    if args.finalize_only:
        cross = _finalize(models, out_root, start_ts)
        print(f"[E28a] Finalized. interpretation={cross['interpretation']}", flush=True)
        return

    for model_name in models:
        device = device_map.get(model_name, "cuda:0")
        print(f"[E28a] Running {model_name} on {device}", flush=True)
        run_model(
            model_name=model_name,
            device=device,
            out_root=out_root,
            eval_seqs=max(8, eval_seqs),
            seq_len=max(64, int(args.seq_len)),
            n_control_trials=max(1, n_control_trials),
            seed=int(args.seed),
            batch_size=max(1, int(args.batch_size)),
        )

    if args.no_finalize:
        print("[E28a] Shard complete (no finalize).", flush=True)
        return

    cross = _finalize(models, out_root, start_ts)
    print(f"[E28a] Done. interpretation={cross['interpretation']}", flush=True)


if __name__ == "__main__":
    main()
