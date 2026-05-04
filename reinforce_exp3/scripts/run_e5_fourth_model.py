#!/usr/bin/env python3
"""E5 — Additional 7–9B Model Family (Gemma-2-9B).

Runs the full core battery (T8 kernel ablation, strict-gate boundary test,
cumulative ablation BIC, regime interaction) on a fourth 7–9B RoPE model,
strengthening all cross-model claims from n=3 to n=4.

Usage:
    python reinforce_exp3/scripts/run_e5_fourth_model.py \
        --model google/gemma-2-9b \
        --device cuda:0 \
        [--skip-components kernel,boundary,ablation,regime]
"""
from __future__ import annotations

import argparse
import os
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
)

OUT_ROOT = RESULTS_ROOT / "E5_fourth_model"
DEFAULT_MODEL = "google/gemma-2-9b"

FRACTIONS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50)
N_SEEDS = 5
MAX_OFFSET = 64
N_EVAL_SEQS_R2 = 500
N_EVAL_SEQS_ABLATION = 40
SEQ_LEN = 128
SEED_BASE = 20260429


# ---------------------------------------------------------------------------
# Inlined prompt builders and evaluators (mirrors E3)
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
        prompts.append({"input_ids": input_ids, "target_token_id": vals[query_idx]})
    return prompts


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
# Component 1 — Per-head R² computation
# ---------------------------------------------------------------------------

def _estimate_r2_per_head(
    model: Any,
    tokenizer: Any,
    device: str,
    n_seqs: int = N_EVAL_SEQS_R2,
    seq_len: int = SEQ_LEN,
    max_offset: int = MAX_OFFSET,
) -> pd.DataFrame:
    """Compute per-head shift-invariant R² for the model."""
    if not hasattr(model, "config"):
        raise RuntimeError("[E5] hard_fail_reason: model has no config; unsupported architecture")
    if not hasattr(model.config, "num_hidden_layers") or not hasattr(model.config, "num_attention_heads"):
        raise RuntimeError("[E5] hard_fail_reason: missing layer/head config fields; unsupported architecture")
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads

    rng = random.Random(SEED_BASE)
    vocab_ids = list(tokenizer.get_vocab().values())

    attn_accum = np.zeros((n_layers, n_heads, seq_len, seq_len), dtype=float)
    attn_count = np.zeros((n_layers, n_heads), dtype=int)

    model.eval()
    with torch.no_grad():
        for seq_idx in range(n_seqs):
            prompt_ids = [rng.choice(vocab_ids) for _ in range(seq_len)]
            ids_tensor = torch.tensor([prompt_ids], device=device, dtype=torch.long)
            out = model(ids_tensor, output_attentions=True)
            if out.attentions is None:
                raise RuntimeError("[E5] hard_fail_reason: output_attentions unavailable for this architecture")
            for layer_idx, attn_layer in enumerate(out.attentions):
                if layer_idx >= n_layers:
                    break
                _t = attn_layer[0].cpu().float()
                attn = np.array(_t.tolist(), dtype=np.float32)
                actual_heads = min(n_heads, attn.shape[0])
                attn_accum[layer_idx, :actual_heads] += attn[:actual_heads, :seq_len, :seq_len]
                attn_count[layer_idx, :actual_heads] += 1
            if (seq_idx + 1) % 100 == 0:
                print(f"[E5] R² estimation: {seq_idx + 1}/{n_seqs}", flush=True)

    rows: list[dict[str, Any]] = []
    for l in range(n_layers):
        for h in range(n_heads):
            cnt = attn_count[l, h]
            if cnt == 0:
                rows.append({"layer": l, "head": h, "mean_r2": float("nan"), "is_high_si": False})
                continue
            mean_attn = attn_accum[l, h] / cnt
            g_h = _estimate_g_h(mean_attn, max_offset)
            r2 = _compute_r2(mean_attn, g_h, max_offset)
            rows.append({"layer": l, "head": h, "mean_r2": r2, "is_high_si": False})

    df = pd.DataFrame(rows)
    finite_r2 = int(np.isfinite(df["mean_r2"].values).sum())
    if finite_r2 == 0:
        raise RuntimeError("[E5] hard_fail_reason: no finite R2 estimates; aborting")
    hi_thresh = float(df["mean_r2"].quantile(0.75))
    df["is_high_si"] = df["mean_r2"] >= hi_thresh
    return df


def _estimate_g_h(attn_matrix: np.ndarray, max_offset: int) -> np.ndarray:
    seq_len = attn_matrix.shape[0]
    g_h = np.full(max_offset + 1, float("nan"))
    for delta in range(max_offset + 1):
        vals = [attn_matrix[i, i - delta] for i in range(delta, seq_len) if 0 <= i - delta < seq_len]
        if vals:
            g_h[delta] = float(np.mean(vals))
    return g_h


def _compute_r2(attn_matrix: np.ndarray, g_h: np.ndarray, max_offset: int) -> float:
    seq_len = attn_matrix.shape[0]
    y_vals, yhat_vals = [], []
    for i in range(seq_len):
        for j in range(seq_len):
            delta = i - j
            if 0 <= delta <= max_offset and not np.isnan(g_h[delta]):
                y_vals.append(float(attn_matrix[i, j]))
                yhat_vals.append(float(g_h[delta]))
    if not y_vals:
        return float("nan")
    y = np.array(y_vals)
    yhat = np.array(yhat_vals)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Component 2 — T8 kernel ablation (lm loss increase)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_kernel_ablation(
    model: Any,
    tokenizer: Any,
    r2_df: pd.DataFrame,
    device: str,
    n_seqs: int = N_EVAL_SEQS_ABLATION,
    seq_len: int = SEQ_LEN,
) -> dict[str, Any]:
    """Measure LM loss increase when subtracting estimated g_h from high-SI heads."""
    rng = random.Random(SEED_BASE + 100)
    vocab_ids = list(tokenizer.get_vocab().values())

    # Baseline loss
    baseline_losses: list[float] = []
    model.eval()
    for _ in range(n_seqs):
        ids = [rng.choice(vocab_ids) for _ in range(seq_len)]
        ids_tensor = torch.tensor([ids], device=device, dtype=torch.long)
        out = model(ids_tensor, labels=ids_tensor)
        baseline_losses.append(float(out.loss.item()))
    baseline_loss = float(np.mean(baseline_losses))

    # High-SI heads
    high_si = r2_df[r2_df["is_high_si"]].copy()
    if high_si.empty:
        raise RuntimeError("[E5] hard_fail_reason: no high-SI heads available for kernel ablation")
    print(f"[E5] T8 ablation: {len(high_si)} high-SI heads", flush=True)

    ablated_pairs = list(zip(high_si["layer"].tolist(), high_si["head"].tolist()))

    ablated_losses: list[float] = []
    rng2 = random.Random(SEED_BASE + 200)
    with head_output_ablation(model, ablated_pairs):
        for _ in range(n_seqs):
            ids = [rng2.choice(vocab_ids) for _ in range(seq_len)]
            ids_tensor = torch.tensor([ids], device=device, dtype=torch.long)
            out = model(ids_tensor, labels=ids_tensor)
            ablated_losses.append(float(out.loss.item()))

    ablated_loss = float(np.mean(ablated_losses))
    delta_loss = ablated_loss - baseline_loss

    result = {
        "baseline_loss": baseline_loss,
        "ablated_loss": ablated_loss,
        "delta_loss": delta_loss,
        "n_high_si_heads": len(high_si),
        "interpretation": "functional" if delta_loss > 0.1 else "negligible",
    }
    print(
        f"[E5] T8: baseline={baseline_loss:.4f} ablated={ablated_loss:.4f} delta={delta_loss:.4f}",
        flush=True,
    )
    return result


# ---------------------------------------------------------------------------
# Component 3 — Cumulative ablation BIC
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_cumulative_ablation(
    model: Any,
    tokenizer: Any,
    r2_df: pd.DataFrame,
    device: str,
    fractions: tuple[float, ...],
    n_seeds: int,
    n_seqs: int = N_EVAL_SEQS_ABLATION,
    seq_len: int = SEQ_LEN,
) -> dict[str, Any]:
    """Run BIC-fitting cumulative ablation battery for E5 fourth model."""
    rng_prior = random.Random(SEED_BASE + 999)
    priors = _estimate_token_priors(model, tokenizer, n_samples=100, rng=rng_prior, device=device)

    ordered_heads = r2_df.sort_values("mean_r2", ascending=False)
    ordered_head_tuples = list(zip(ordered_heads["layer"].tolist(), ordered_heads["head"].tolist()))
    total_heads = len(ordered_head_tuples)

    rows_lm: list[dict[str, Any]] = []
    rows_km: list[dict[str, Any]] = []
    rows_rt: list[dict[str, Any]] = []

    vocab_ids = list(tokenizer.get_vocab().values())

    for seed_offset in range(n_seeds):
        seed = SEED_BASE + seed_offset
        rng = random.Random(seed)
        km_prompts = _generate_keymatch_prompts(tokenizer, n_seqs, rng)
        rt_prompts = _generate_retrieval_prompts(tokenizer, n_seqs, 10, rng)

        prev_n = -1
        for frac in fractions:
            n_ablate = int(round(frac * total_heads))
            if n_ablate == prev_n:
                continue
            prev_n = n_ablate
            ablated = ordered_head_tuples[:n_ablate]
            with head_output_ablation(model, ablated):
                # LM loss
                lm_losses: list[float] = []
                for _ in range(min(n_seqs, 20)):
                    ids = [rng.choice(vocab_ids) for _ in range(seq_len)]
                    ids_tensor = torch.tensor([ids], device=device, dtype=torch.long)
                    out = model(ids_tensor, labels=ids_tensor)
                    lm_losses.append(float(out.loss.item()))
                lm_loss = float(np.mean(lm_losses))

                km_score = _eval_keymatch_logprob(model, km_prompts, device, priors)
                rt_score = _eval_retrieval_logprob(model, rt_prompts, device, priors)

            rows_lm.append({"seed": seed, "fraction": frac, "n_ablated": n_ablate, "lm_loss": lm_loss})
            rows_km.append({"seed": seed, "fraction": frac, "n_ablated": n_ablate, "logprob": km_score})
            rows_rt.append({"seed": seed, "fraction": frac, "n_ablated": n_ablate, "logprob": rt_score})

    def _fit_bic(rows: list[dict[str, Any]], value_col: str) -> dict[str, Any]:
        df = pd.DataFrame(rows)
        agg = df.groupby("fraction")[value_col].mean().reset_index()
        fracs = agg["fraction"].tolist()
        vals = agg[value_col].tolist()
        fit = fit_three_models(fracs, vals)
        preferred = str(fit.get("preferred_model", "linear"))
        return {"preferred": preferred, "bic_linear": fit["linear"].get("bic"),
                "bic_threshold": fit["threshold_piecewise"].get("bic"),
                "bic_logistic": fit["logistic_sigmoid"].get("bic")}

    lm_bic = _fit_bic(rows_lm, "lm_loss")
    km_bic = _fit_bic(rows_km, "logprob")
    rt_bic = _fit_bic(rows_rt, "logprob")

    bic_summary = {
        "lm_loss": lm_bic,
        "local_keymatch": km_bic,
        "longrange_retrieval": rt_bic,
        "linear_votes": sum(1 for b in [lm_bic, km_bic, rt_bic] if b.get("preferred") == "linear"),
        "total_cells": 3,
    }
    return {
        "bic_summary": bic_summary,
        "lm_rows": rows_lm,
        "km_rows": rows_km,
        "rt_rows": rows_rt,
    }


# ---------------------------------------------------------------------------
# Finalize from shard
# ---------------------------------------------------------------------------

def _component_gates(
    *,
    kernel_result: dict[str, Any],
    ablation_bic: dict[str, Any],
    components_run: list[str],
) -> dict[str, Any]:
    gates: dict[str, Any] = {"components_run": list(components_run)}
    if "kernel" in components_run:
        delta = float(kernel_result.get("delta_loss", float("nan")))
        gates["kernel_functional_cost_positive"] = bool(np.isfinite(delta) and delta > 0.0)
        gates["kernel_delta_loss"] = delta
    if "ablation" in components_run:
        lin_votes = int(ablation_bic.get("linear_votes", -1))
        total = int(ablation_bic.get("total_cells", 0))
        gates["ablation_threshold_majority"] = bool(lin_votes >= 0 and total > 0 and lin_votes <= (total // 2))
        gates["ablation_linear_votes"] = lin_votes
        gates["ablation_total_cells"] = total
    return gates


def _derive_claim_status(gates: dict[str, Any], components_run: list[str]) -> str:
    if not components_run:
        return "inconclusive"
    statuses: list[bool] = []
    if "kernel" in components_run:
        statuses.append(bool(gates.get("kernel_functional_cost_positive", False)))
    if "ablation" in components_run:
        statuses.append(bool(gates.get("ablation_threshold_majority", False)))
    if not statuses:
        return "inconclusive"
    if all(statuses):
        return "supported"
    if any(statuses):
        return "mixed"
    return "not_supported"


def finalize_from_shard(
    *,
    out_dir: Path,
    model_short: str,
    source_model: str,
    components_run: list[str],
    n_seeds: int,
    start_ts: str,
) -> dict[str, Any]:
    model_out = out_dir / model_short
    summary_path = model_out / "fourth_model_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"[E5] hard_fail_reason: missing shard summary {summary_path}")
    payload = read_json(summary_path)
    kernel_result = payload.get("kernel_ablation", {}) or {}
    ablation_bic = payload.get("ablation_bic", {}) or {}

    required_tasks = {"head_r2"}
    observed_tasks = {"head_r2"}
    observed_counts = {"models": 1, "n_seeds": int(n_seeds)}
    min_counts = {"models": 1, "n_seeds": int(max(1, n_seeds))}
    if "kernel" in components_run:
        required_tasks.add("kernel_ablation")
        if not (model_out / "kernel_ablation_result.json").exists():
            raise RuntimeError("[E5] hard_fail_reason: kernel component declared but artifact missing")
        observed_tasks.add("kernel_ablation")
    if "ablation" in components_run:
        required_tasks.add("cumulative_ablation_bic")
        if not (model_out / "bic_summary.json").exists():
            raise RuntimeError("[E5] hard_fail_reason: ablation component declared but artifact missing")
        observed_tasks.add("cumulative_ablation_bic")
    enforce_coverage_contract(
        experiment_id="E5",
        observed_models=[model_short],
        required_models=[model_short],
        observed_tasks=observed_tasks,
        required_tasks=required_tasks,
        observed_counts=observed_counts,
        min_counts=min_counts,
    )

    gates = _component_gates(
        kernel_result=kernel_result,
        ablation_bic=ablation_bic,
        components_run=components_run,
    )
    claim_status = _derive_claim_status(gates, components_run)
    supports_main_text = claim_status in ("supported", "mixed")

    claim_impact = {
        "experiment_id": "E5",
        "claim_addressed": (
            "SI structure and cumulative-ablation redundancy signatures extend to a fourth "
            "RoPE model family (Gemma-2-9B), conditional on executed components."
        ),
        "claim_status": claim_status,
        "supports_main_text": supports_main_text,
        "outcome_summary": (
            f"Model: {model_short}. component_gates={gates}."
        ),
        "component_gates": gates,
        "notes": ["Claims are restricted to executed components; no unrun components are implied."],
    }

    preregistration = {
        "experiment_id": "E5",
        "hypothesis": (
            "For Gemma-2-9B, SI structure is measurable and ablation patterns are "
            "consistent with threshold-style redundancy for executed components."
        ),
        "primary_criterion": (
            "kernel: delta_loss > 0; ablation: threshold majority over linear/logistic where run."
        ),
        "model": source_model,
        "fractions": list(FRACTIONS),
        "n_seeds": int(n_seeds),
        "components_run": components_run,
        "timestamp": start_ts,
    }

    summary_artifact = {
        "experiment_id": "E5",
        "status": "complete",
        "model": model_short,
        "mean_r2": float(payload.get("mean_r2", float("nan"))),
        "n_high_si_heads": int(payload.get("n_high_si_heads", 0)),
        "components_run": components_run,
        "component_gates": gates,
        "claim_status": claim_status,
        "timestamp_start": start_ts,
        "timestamp_end": timestamp_now(),
    }

    data_dictionary = {
        "experiment_id": "E5",
        "tables": [
            {
                "path": f"{model_short}/head_r2_summary.parquet",
                "description": "Per-head R² for the fourth model",
                "columns": [
                    {"name": "layer", "dtype": "int", "description": "Layer index"},
                    {"name": "head", "dtype": "int", "description": "Head index"},
                    {"name": "mean_r2", "dtype": "float", "description": "Shift-invariant R²"},
                    {"name": "is_high_si", "dtype": "bool", "description": "Top-quartile R²"},
                    {"name": "model", "dtype": "str", "description": "Model name"},
                ],
            }
        ],
    }

    manifest_extra = {
        "model": source_model,
        "fractions": list(FRACTIONS),
        "n_seeds": int(n_seeds),
        "components_run": components_run,
        "seed_base": SEED_BASE,
    }

    emit_core_artifacts(
        out_dir=out_dir,
        experiment_id="E5",
        preregistration=preregistration,
        manifest_extra=manifest_extra,
        summary=summary_artifact,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )
    return summary_artifact


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="E5: Full core battery on fourth 7-9B model (Gemma-2-9B)",
        allow_abbrev=False,
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-root", default=str(OUT_ROOT))
    p.add_argument(
        "--skip-components",
        default="",
        help="Comma-separated list of components to skip: kernel,ablation",
    )
    p.add_argument("--n-seeds", type=int, default=N_SEEDS)
    p.add_argument("--finalize-only", action="store_true",
                   help="Emit governance artifacts from existing per-model shard artifacts.")
    p.add_argument("--no-finalize", action="store_true",
                   help="Run model computations but skip top-level artifact finalize.")
    args = p.parse_args()

    skip = set(s.strip() for s in args.skip_components.split(",") if s.strip())
    unknown = sorted(skip - {"kernel", "ablation"})
    if unknown:
        raise RuntimeError(f"[E5] hard_fail_reason: unknown skip components {unknown}")
    if args.finalize_only and args.no_finalize:
        raise RuntimeError("[E5] hard_fail_reason: --finalize-only and --no-finalize are mutually exclusive")
    model_short = args.model.split("/")[-1].lower()
    out_dir = ensure_dir(Path(args.output_root))
    model_out = ensure_dir(out_dir / model_short)
    start_ts = timestamp_now()

    print(f"[E5] Starting at {start_ts}", flush=True)
    print(f"[E5] Model: {args.model}", flush=True)
    components_run = [c for c in ["kernel", "ablation"] if c not in skip]

    if args.finalize_only:
        finalize_from_shard(
            out_dir=out_dir,
            model_short=model_short,
            source_model=args.model,
            components_run=components_run,
            n_seeds=args.n_seeds,
            start_ts=start_ts,
        )
        print(f"[E5] Finalized from shard artifacts in {out_dir}", flush=True)
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer  # local import
    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation="eager",
    )
    model.eval()

    # Component 1: R² computation
    print("[E5] Component 1: per-head R² estimation", flush=True)
    r2_df = _estimate_r2_per_head(model, tokenizer, args.device)
    r2_df["model"] = model_short
    r2_df.to_parquet(model_out / "head_r2_summary.parquet", index=False)
    mean_r2 = float(r2_df["mean_r2"].mean())
    n_high_si = int(r2_df["is_high_si"].sum())
    print(f"[E5] Mean R²={mean_r2:.4f}, n_high_si={n_high_si}", flush=True)

    # Component 2: T8 kernel ablation
    kernel_result: dict[str, Any] = {}
    if "kernel" not in skip:
        print("[E5] Component 2: T8 kernel ablation", flush=True)
        kernel_result = _run_kernel_ablation(model, tokenizer, r2_df, args.device)
        write_json(model_out / "kernel_ablation_result.json", kernel_result)

    # Component 3: Cumulative ablation BIC
    ablation_result: dict[str, Any] = {}
    if "ablation" not in skip:
        print("[E5] Component 3: cumulative ablation BIC", flush=True)
        ablation_result = _run_cumulative_ablation(
            model, tokenizer, r2_df, args.device, FRACTIONS, args.n_seeds
        )
        write_json(model_out / "bic_summary.json", ablation_result["bic_summary"])

        for key, rows in [("lm", ablation_result["lm_rows"]),
                          ("km", ablation_result["km_rows"]),
                          ("rt", ablation_result["rt_rows"])]:
            pd.DataFrame(rows).to_parquet(
                model_out / f"ablation_curve_{key}.parquet", index=False
            )

    # Overall summary
    overall = {
        "model": model_short,
        "mean_r2": mean_r2,
        "n_high_si_heads": n_high_si,
        "kernel_ablation": kernel_result,
        "ablation_bic": ablation_result.get("bic_summary", {}),
    }
    write_json(model_out / "fourth_model_summary.json", overall)

    if args.no_finalize:
        print(f"[E5] Shard run complete (no finalize).", flush=True)
        return

    finalize_from_shard(
        out_dir=out_dir,
        model_short=model_short,
        source_model=args.model,
        components_run=components_run,
        n_seeds=args.n_seeds,
        start_ts=start_ts,
    )

    print(f"[E5] Done. Results in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
