#!/usr/bin/env python3
"""E3 — Continuous-Metric Variants for Synthetic Tasks.

Addresses the Schaeffer et al. metric-artifact objection: binary accuracy
metrics can produce spurious threshold effects. This experiment re-runs the
cumulative ablation BIC analysis using continuous log-probability metrics for
both synthetic tasks (local key-match and long-range retrieval), showing that
threshold-piecewise preference holds in continuous form as well.

Usage:
    python reinforce_exp3/scripts/run_e3_continuous_metrics.py \
        --models llama-3.1-8b,olmo-2-7b,mistral-7b-v0.1 \
        --device-map llama-3.1-8b:cuda:0,olmo-2-7b:cuda:3,mistral-7b-v0.1:cuda:1
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp3.common import (  # noqa: E402
    B1_RESULTS,
    EXP3_SI_CIRCUITS,
    PRIMARY_MODELS,
    RESULTS_ROOT,
    ensure_dir,
    read_json,
    timestamp_now,
    write_json,
)
from reinforce_exp3.scripts._shared import (  # noqa: E402
    enforce_coverage_contract,
    emit_core_artifacts,
    fit_three_models,
    head_output_ablation,
    load_model_for_exp,
    parse_device_map,
    parse_models_arg,
)

OUT_ROOT = RESULTS_ROOT / "E3_continuous_metric_variants"

# Fraction grid matching 3P2-C with additional fine points near typical threshold
FRACTIONS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50)

N_SEEDS = 5
N_KEYMATCH_EXAMPLES = 200
N_RETRIEVAL_EXAMPLES = 200
KV_PAIR_COUNT = 10
SEED_BASE = 20260429


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _generate_keymatch_prompts(
    tokenizer: Any,
    n: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Build local key-match prompts: [key] [val] fillers [key] → predict val."""
    vocab = list(tokenizer.get_vocab().values())
    prompts: list[dict[str, Any]] = []
    for _ in range(n):
        key_id = rng.choice(vocab)
        val_id = rng.choice(vocab)
        n_filler = rng.randint(4, 12)
        filler_ids = [rng.choice(vocab) for _ in range(n_filler)]
        input_ids = [key_id, val_id] + filler_ids + [key_id]
        prompts.append({"input_ids": input_ids, "target_token_id": val_id})
    return prompts


def _generate_retrieval_prompts(
    tokenizer: Any,
    n: int,
    kv_pairs: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Build long-range retrieval prompts: N (key val) pairs then query key → predict val."""
    vocab = list(tokenizer.get_vocab().values())
    prompts: list[dict[str, Any]] = []
    for _ in range(n):
        keys = [rng.choice(vocab) for _ in range(kv_pairs)]
        vals = [rng.choice(vocab) for _ in range(kv_pairs)]
        query_idx = rng.randint(0, kv_pairs - 1)
        input_ids: list[int] = []
        for k, v in zip(keys, vals):
            input_ids.extend([k, v])
        input_ids.append(keys[query_idx])
        prompts.append({
            "input_ids": input_ids,
            "target_token_id": vals[query_idx],
        })
    return prompts


# ---------------------------------------------------------------------------
# Prior log-prob estimation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _estimate_token_priors(
    model: Any,
    tokenizer: Any,
    n_samples: int,
    rng: random.Random,
    device: str,
) -> dict[int, float]:
    """Estimate unconditional log P(token) for normalization via short random prompts."""
    vocab_ids = list(tokenizer.get_vocab().values())
    log_prob_sum: dict[int, float] = {}
    counts: dict[int, int] = {}
    model.eval()
    with torch.no_grad():
        for _ in range(n_samples):
            prompt = [rng.choice(vocab_ids) for _ in range(8)]
            input_tensor = torch.tensor([prompt], device=device, dtype=torch.long)
            out = model(input_tensor)
            logits = out.logits[0, -1]
            log_probs_t = torch.log_softmax(logits, dim=-1).cpu()
            for tid in vocab_ids:
                log_prob_sum[tid] = log_prob_sum.get(tid, 0.0) + float(log_probs_t[tid])
                counts[tid] = counts.get(tid, 0) + 1
    priors: dict[int, float] = {}
    for tid in vocab_ids:
        if counts.get(tid, 0) > 0:
            priors[tid] = log_prob_sum[tid] / counts[tid]
    return priors


# ---------------------------------------------------------------------------
# Per-head R² ordering
# ---------------------------------------------------------------------------

def _load_r2_and_order(model_name: str) -> pd.DataFrame:
    """Load per-head R² summary; return descending-R² sorted DataFrame."""
    summary_path = EXP3_SI_CIRCUITS / model_name / "head_r2_summary.parquet"
    if not summary_path.exists():
        alt = B1_RESULTS / "cluster_membership.parquet"
        df = pd.read_parquet(alt)
        df = df[df["model"] == model_name].copy()
    else:
        df = pd.read_parquet(summary_path)
        if "model" not in df.columns:
            df["model"] = model_name
    df = df.sort_values("mean_r2", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Continuous metric evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_keymatch_logprob(
    model: Any,
    prompts: list[dict[str, Any]],
    device: str,
    priors: dict[int, float],
) -> float:
    """Compute mean normalized log-prob for local key-match task."""
    model.eval()
    scores: list[float] = []
    for p in prompts:
        input_ids = p["input_ids"]
        target_id = p["target_token_id"]
        input_tensor = torch.tensor([input_ids], device=device, dtype=torch.long)
        out = model(input_tensor)
        logits = out.logits[0, -1]
        log_probs = torch.log_softmax(logits, dim=-1).cpu()
        raw = float(log_probs[target_id])
        prior = priors.get(target_id, float("nan"))
        score = raw - prior if not np.isnan(prior) else raw
        scores.append(score)
    return float(np.mean(scores))


@torch.no_grad()
def _eval_retrieval_logprob(
    model: Any,
    prompts: list[dict[str, Any]],
    device: str,
    priors: dict[int, float],
) -> float:
    """Compute mean normalized log-prob for long-range retrieval task."""
    model.eval()
    scores: list[float] = []
    for p in prompts:
        input_ids = p["input_ids"]
        target_id = p["target_token_id"]
        input_tensor = torch.tensor([input_ids], device=device, dtype=torch.long)
        out = model(input_tensor)
        logits = out.logits[0, -1]
        log_probs = torch.log_softmax(logits, dim=-1).cpu()
        raw = float(log_probs[target_id])
        prior = priors.get(target_id, float("nan"))
        score = raw - prior if not np.isnan(prior) else raw
        scores.append(score)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Per-model run
# ---------------------------------------------------------------------------

def run_model(
    model_name: str,
    device: str,
    out_dir: Path,
    fractions: tuple[float, ...] = FRACTIONS,
    n_seeds: int = N_SEEDS,
) -> dict[str, Any]:
    print(f"[E3] Loading {model_name} on {device}", flush=True)
    model, tokenizer = load_model_for_exp(model_name, device)

    r2_df = _load_r2_and_order(model_name)
    total_heads = len(r2_df)
    ordered_heads = list(zip(r2_df["layer"].tolist(), r2_df["head"].tolist()))
    print(f"[E3] {model_name}: {total_heads} heads loaded", flush=True)

    model_out = ensure_dir(out_dir / model_name)
    rng0 = random.Random(SEED_BASE + 99999)

    print(f"[E3] Estimating token priors for {model_name}", flush=True)
    priors = _estimate_token_priors(model, tokenizer, n_samples=100, rng=rng0, device=device)

    rows_km: list[dict[str, Any]] = []
    rows_rt: list[dict[str, Any]] = []

    for seed_offset in range(n_seeds):
        seed = SEED_BASE + seed_offset
        rng = random.Random(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        km_prompts = _generate_keymatch_prompts(tokenizer, N_KEYMATCH_EXAMPLES, rng)
        rt_prompts = _generate_retrieval_prompts(tokenizer, N_RETRIEVAL_EXAMPLES, KV_PAIR_COUNT, rng)

        prev_n = -1
        for frac in fractions:
            n_ablate = int(round(frac * total_heads))
            if n_ablate == prev_n:
                continue
            prev_n = n_ablate

            ablated = ordered_heads[:n_ablate]
            with head_output_ablation(model, ablated):
                km_score = _eval_keymatch_logprob(model, km_prompts, device, priors)
                rt_score = _eval_retrieval_logprob(model, rt_prompts, device, priors)

            rows_km.append({
                "model": model_name, "seed": seed, "fraction": frac,
                "n_ablated": n_ablate, "logprob_normalized": km_score,
            })
            rows_rt.append({
                "model": model_name, "seed": seed, "fraction": frac,
                "n_ablated": n_ablate, "logprob_normalized": rt_score,
            })
            print(
                f"[E3] {model_name} seed={seed} frac={frac:.2f} "
                f"km={km_score:.4f} rt={rt_score:.4f}",
                flush=True,
            )

    km_df = pd.DataFrame(rows_km)
    rt_df = pd.DataFrame(rows_rt)
    km_df.to_parquet(model_out / "local_keymatch_logprob_curve.parquet", index=False)
    rt_df.to_parquet(model_out / "longrange_logprob_curve.parquet", index=False)

    def _agg_mean(df: pd.DataFrame) -> tuple[list[float], list[float]]:
        agg = df.groupby("fraction")["logprob_normalized"].mean().reset_index()
        agg = agg.sort_values("fraction")
        return agg["fraction"].tolist(), agg["logprob_normalized"].tolist()

    km_fracs, km_vals = _agg_mean(km_df)
    rt_fracs, rt_vals = _agg_mean(rt_df)

    km_fit = fit_three_models(km_fracs, km_vals)
    rt_fit = fit_three_models(rt_fracs, rt_vals)
    km_verdict = str(km_fit.get("preferred_model", "linear"))
    rt_verdict = str(rt_fit.get("preferred_model", "linear"))

    bic_votes = {
        "local_keymatch_logprob": {
            "preferred_model": km_verdict,
            "linear_bic": km_fit["linear"]["bic"],
            "threshold_piecewise_bic": km_fit["threshold_piecewise"]["bic"],
            "logistic_sigmoid_bic": km_fit["logistic_sigmoid"]["bic"],
        },
        "longrange_retrieval_logprob": {
            "preferred_model": rt_verdict,
            "linear_bic": rt_fit["linear"]["bic"],
            "threshold_piecewise_bic": rt_fit["threshold_piecewise"]["bic"],
            "logistic_sigmoid_bic": rt_fit["logistic_sigmoid"]["bic"],
        },
    }
    write_json(model_out / "bic_votes_continuous.json", bic_votes)
    print(f"[E3] {model_name}: km_verdict={km_verdict} rt_verdict={rt_verdict}", flush=True)

    return {
        "model": model_name,
        "km_fit": km_fit,
        "rt_fit": rt_fit,
        "km_verdict": km_verdict,
        "rt_verdict": rt_verdict,
    }


# ---------------------------------------------------------------------------
# Cross-model aggregation
# ---------------------------------------------------------------------------

def aggregate_results(
    model_results: list[dict[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    new_cells: list[dict[str, Any]] = []
    km_votes_linear = 0
    rt_votes_linear = 0

    for r in model_results:
        for task, key in [("local_keymatch_logprob", "km"), ("longrange_retrieval_logprob", "rt")]:
            verdict = r[f"{key}_verdict"]
            cell = {
                "model": r["model"], "task": task, "metric_type": "continuous",
                "preferred_model": verdict, "is_linear": verdict == "linear",
            }
            new_cells.append(cell)
            if task.startswith("local"):
                km_votes_linear += int(verdict == "linear")
            else:
                rt_votes_linear += int(verdict == "linear")

    total_new = len(new_cells)
    linear_new = km_votes_linear + rt_votes_linear

    if linear_new == 0:
        outcome = "metric_artifact_refuted"
        note = f"0/{total_new} continuous-metric cells prefer linear BIC. Schaeffer objection refuted."
    elif linear_new <= total_new // 4:
        outcome = "metric_artifact_partial"
        note = f"{linear_new}/{total_new} continuous-metric cells prefer linear BIC. Threshold holds in majority."
    else:
        outcome = "metric_artifact_present"
        note = f"{linear_new}/{total_new} continuous-metric cells prefer linear BIC. Metric artifact may be a factor."

    summary = {
        "n_new_cells": total_new, "n_new_linear_votes": linear_new,
        "km_linear_votes": km_votes_linear, "rt_linear_votes": rt_votes_linear,
        "outcome": outcome, "note": note, "new_cells": new_cells,
    }
    write_json(out_dir / "cross_model_bic_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Artifact emission
# ---------------------------------------------------------------------------

def _emit_artifacts(
    models: list[str],
    out_dir: Path,
    cross_model: dict[str, Any],
    start_ts: str,
    n_seeds: int,
) -> None:
    outcome = cross_model["outcome"]
    claim_status = "supported" if outcome == "metric_artifact_refuted" else (
        "mixed" if outcome == "metric_artifact_partial" else "not_supported"
    )

    claim_impact = {
        "experiment_id": "E3",
        "claim_addressed": (
            "Result III threshold preference holds under continuous (log-probability) "
            "metrics, refuting the Schaeffer et al. metric-artifact objection"
        ),
        "claim_status": claim_status,
        "supports_main_text": claim_status in ("supported", "mixed"),
        "outcome_summary": cross_model["note"],
        "n_new_bic_cells": cross_model["n_new_cells"],
        "n_new_linear_votes": cross_model["n_new_linear_votes"],
        "notes": [cross_model["note"]],
    }

    preregistration = {
        "experiment_id": "E3",
        "hypothesis": (
            "Threshold-piecewise BIC preference persists when binary accuracy is replaced "
            "by continuous log-probability metrics."
        ),
        "primary_criterion": f"0/{2 * len(models)} linear BIC votes in continuous-metric cells",
        "models": models, "fractions": list(FRACTIONS), "n_seeds": int(n_seeds),
        "tasks": ["local_keymatch_logprob", "longrange_retrieval_logprob"],
        "metric_type": "continuous (normalized log-probability)",
        "timestamp": start_ts,
    }

    summary = {
        "experiment_id": "E3", "status": "complete", "models_run": models,
        "n_new_bic_cells": cross_model["n_new_cells"],
        "linear_votes_new": cross_model["n_new_linear_votes"],
        "outcome": cross_model["outcome"],
        "timestamp_start": start_ts, "timestamp_end": timestamp_now(),
    }

    data_dictionary = {
        "experiment_id": "E3",
        "tables": [
            {
                "path": "<model>/local_keymatch_logprob_curve.parquet",
                "description": "Continuous key-match log-prob scores per seed/fraction",
                "columns": [
                    {"name": "model", "dtype": "str", "description": "Model name"},
                    {"name": "seed", "dtype": "int", "description": "Random seed"},
                    {"name": "fraction", "dtype": "float", "description": "Ablation fraction"},
                    {"name": "n_ablated", "dtype": "int", "description": "Number of heads ablated"},
                    {"name": "logprob_normalized", "dtype": "float",
                     "description": "Mean normalized log-prob (log P - log P_prior)"},
                ],
            },
            {
                "path": "<model>/longrange_logprob_curve.parquet",
                "description": "Continuous retrieval log-prob scores per seed/fraction",
                "columns": [
                    {"name": "model", "dtype": "str", "description": "Model name"},
                    {"name": "seed", "dtype": "int", "description": "Random seed"},
                    {"name": "fraction", "dtype": "float", "description": "Ablation fraction"},
                    {"name": "n_ablated", "dtype": "int", "description": "Number of heads ablated"},
                    {"name": "logprob_normalized", "dtype": "float",
                     "description": "Mean normalized log-prob (log P - log P_prior)"},
                ],
            },
        ],
    }

    emit_core_artifacts(
        out_dir=out_dir, experiment_id="E3",
        preregistration=preregistration,
        manifest_extra={
            "models": models, "fractions": list(FRACTIONS), "n_seeds": int(n_seeds),
            "n_keymatch_examples": N_KEYMATCH_EXAMPLES,
            "n_retrieval_examples": N_RETRIEVAL_EXAMPLES,
            "kv_pair_count": KV_PAIR_COUNT, "seed_base": SEED_BASE,
        },
        summary=summary, claim_impact=claim_impact, data_dictionary=data_dictionary,
    )


# ---------------------------------------------------------------------------
# Finalize from per-model shards
# ---------------------------------------------------------------------------

def _load_model_result_from_artifacts(
    model_name: str,
    out_dir: Path,
) -> dict[str, Any]:
    model_dir = out_dir / model_name
    votes_path = model_dir / "bic_votes_continuous.json"
    if not votes_path.exists():
        raise RuntimeError(f"[E3] hard_fail_reason: missing per-model artifact {votes_path}")
    votes = read_json(votes_path)
    km = votes.get("local_keymatch_logprob", {})
    rt = votes.get("longrange_retrieval_logprob", {})
    km_v = str(km.get("preferred_model", "linear"))
    rt_v = str(rt.get("preferred_model", "linear"))
    return {
        "model": model_name,
        "km_verdict": km_v,
        "rt_verdict": rt_v,
        "km_fit": {
            "preferred_model": km_v,
            "linear": {"bic": km.get("linear_bic")},
            "threshold_piecewise": {"bic": km.get("threshold_piecewise_bic")},
            "logistic_sigmoid": {"bic": km.get("logistic_sigmoid_bic")},
        },
        "rt_fit": {
            "preferred_model": rt_v,
            "linear": {"bic": rt.get("linear_bic")},
            "threshold_piecewise": {"bic": rt.get("threshold_piecewise_bic")},
            "logistic_sigmoid": {"bic": rt.get("logistic_sigmoid_bic")},
        },
    }


def finalize_from_shards(
    *,
    models: list[str],
    out_dir: Path,
    start_ts: str,
    n_seeds: int,
) -> dict[str, Any]:
    model_results = [_load_model_result_from_artifacts(m, out_dir) for m in models]
    enforce_coverage_contract(
        experiment_id="E3",
        observed_models=[r["model"] for r in model_results],
        required_models=models,
        observed_tasks=["local_keymatch_logprob", "longrange_retrieval_logprob"],
        required_tasks=["local_keymatch_logprob", "longrange_retrieval_logprob"],
        observed_counts={"continuous_cells": len(model_results) * 2},
        min_counts={"continuous_cells": len(models) * 2},
    )
    cross_model = aggregate_results(model_results, out_dir)
    _emit_artifacts(models, out_dir, cross_model, start_ts, n_seeds=n_seeds)
    return cross_model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="E3: Continuous-metric BIC ablation to refute Schaeffer metric-artifact objection",
        allow_abbrev=False,
    )
    p.add_argument("--models", default=",".join(["llama-3.1-8b", "olmo-2-7b", "mistral-7b-v0.1"]))
    p.add_argument(
        "--device-map",
        default="llama-3.1-8b:cuda:0,olmo-2-7b:cuda:0,mistral-7b-v0.1:cuda:0",
    )
    p.add_argument("--output-root", default=str(OUT_ROOT))
    p.add_argument("--n-seeds", type=int, default=N_SEEDS)
    p.add_argument("--finalize-only", action="store_true",
                   help="Read per-model shard outputs and emit cross-model artifacts only.")
    p.add_argument("--no-finalize", action="store_true",
                   help="Run per-model computations but skip cross-model finalize emission.")
    args = p.parse_args()

    models = parse_models_arg(args.models, default=PRIMARY_MODELS)
    device_map = parse_device_map(args.device_map)
    out_dir = ensure_dir(Path(args.output_root))
    start_ts = timestamp_now()

    print(f"[E3] Starting at {start_ts}", flush=True)
    print(f"[E3] Models: {models}", flush=True)

    if args.finalize_only and args.no_finalize:
        raise RuntimeError("[E3] hard_fail_reason: --finalize-only and --no-finalize are mutually exclusive")

    if args.finalize_only:
        cross_model = finalize_from_shards(
            models=models,
            out_dir=out_dir,
            start_ts=start_ts,
            n_seeds=args.n_seeds,
        )
        print(f"[E3] Finalized from shards. Outcome: {cross_model['outcome']}", flush=True)
        return

    model_results: list[dict[str, Any]] = []
    for model_name in models:
        device = device_map.get(model_name, "cuda:0")
        result = run_model(model_name=model_name, device=device, out_dir=out_dir,
                           fractions=FRACTIONS, n_seeds=args.n_seeds)
        model_results.append(result)

    if args.no_finalize:
        print("[E3] Shard run complete (no finalize).", flush=True)
        return

    cross_model = finalize_from_shards(
        models=models,
        out_dir=out_dir,
        start_ts=start_ts,
        n_seeds=args.n_seeds,
    )

    print(f"[E3] Done. Outcome: {cross_model['outcome']}", flush=True)


if __name__ == "__main__":
    main()
