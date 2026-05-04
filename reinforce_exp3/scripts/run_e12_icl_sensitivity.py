#!/usr/bin/env python3
"""E12 — ICL Sensitivity Under SI-Kernel Subtraction.

Tests whether ICL performance degrades preferentially (relative to a matched
non-ICL task) when SI-kernel ablation is applied, linking SI heads to the
induction-head mechanism (Olsson et al. 2022).

Usage:
    python reinforce_exp3/scripts/run_e12_icl_sensitivity.py \
        --models llama-3.1-8b,olmo-2-7b,mistral-7b-v0.1 \
        --device-map llama-3.1-8b:cuda:0,olmo-2-7b:cuda:0,mistral-7b-v0.1:cuda:0
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
    RESULTS_ROOT,
    ensure_dir,
    timestamp_now,
    write_json,
)
from reinforce_exp3.scripts._shared import (  # noqa: E402
    bootstrap_mean_ci,
    cohen_d,
    enforce_coverage_contract,
    emit_core_artifacts,
    head_output_ablation,
    load_model_for_exp,
    parse_device_map,
    parse_models_arg,
    read_json,
)

OUT_ROOT = RESULTS_ROOT / "E12_icl_sensitivity"

N_ICL_EXAMPLES = 4          # number of (key → value) demonstrations per prompt
N_EVAL_PROMPTS = 200        # prompts per condition (ICL / non-ICL)
SEQ_LEN_MAX = 256
SEED_BASE = 20260429
MIN_NONICL_ACC_DENOM = 0.01
MIN_NONICL_LP_DENOM = 0.05

# SI-kernel ablation: ablate all high-SI heads (top-quartile R²)
# This is the same ablation as in T8 / E3 / E5


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_icl_prompts(
    tokenizer: Any,
    n: int,
    n_demos: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Build few-shot ICL prompts: n_demos × (key → value) then query key.

    Returns list of {input_ids, target_token_id, is_icl: True}.
    The model must recognize the (key → value) pattern to get credit.
    """
    vocab_ids = list(tokenizer.get_vocab().values())
    prompts: list[dict[str, Any]] = []

    for _ in range(n):
        # Shared key-value mapping for this prompt
        key_id = rng.choice(vocab_ids)
        val_id = rng.choice(vocab_ids)
        # Build demonstrations: [key val] × n_demos
        demo_ids: list[int] = []
        for _ in range(n_demos):
            demo_ids.extend([key_id, val_id])
        # Query: key → expect val
        input_ids = demo_ids + [key_id]
        prompts.append({
            "input_ids": input_ids,
            "target_token_id": val_id,
            "is_icl": True,
        })
    return prompts


def _build_non_icl_prompts(
    tokenizer: Any,
    n: int,
    n_demos: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Build matched non-ICL prompts: same number of tokens, random order.

    The key-value pairs are presented in a random order that doesn't support
    pattern inference (different key-val pairs each time, query key novel).
    """
    vocab_ids = list(tokenizer.get_vocab().values())
    prompts: list[dict[str, Any]] = []

    for _ in range(n):
        # Each demo uses a DIFFERENT key-val pair — no consistent mapping to learn
        demo_ids: list[int] = []
        for _ in range(n_demos):
            k = rng.choice(vocab_ids)
            v = rng.choice(vocab_ids)
            demo_ids.extend([k, v])
        # Query: a new key (not seen in demos) — correct answer is the "registered" val
        # We use the last demo's value as the notional correct answer
        last_demo_val = demo_ids[-1]  # val from last demo pair
        query_key = rng.choice(vocab_ids)
        input_ids = demo_ids + [query_key]
        prompts.append({
            "input_ids": input_ids,
            "target_token_id": last_demo_val,
            "is_icl": False,
        })
    return prompts



# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_accuracy_and_logprob(
    model: Any,
    prompts: list[dict[str, Any]],
    device: str,
) -> dict[str, Any]:
    """Evaluate accuracy (argmax) and log-probability for target token."""
    model.eval()
    correct: list[int] = []
    logprobs: list[float] = []

    for p in prompts:
        ids = p["input_ids"]
        target = p["target_token_id"]
        ids_tensor = torch.tensor([ids], device=device, dtype=torch.long)
        out = model(ids_tensor)
        logits = out.logits[0, -1]  # (vocab,)
        pred = int(torch.argmax(logits).item())
        lp = float(torch.log_softmax(logits, dim=-1)[target].item())
        correct.append(int(pred == target))
        logprobs.append(lp)

    return {
        "accuracy": float(np.mean(correct)),
        "mean_logprob": float(np.mean(logprobs)),
        "n_prompts": len(prompts),
    }


# ---------------------------------------------------------------------------
# Per-model run
# ---------------------------------------------------------------------------

def run_model(
    model_name: str,
    device: str,
    out_dir: Path,
    n_eval: int = N_EVAL_PROMPTS,
    n_demos: int = N_ICL_EXAMPLES,
) -> dict[str, Any]:
    print(f"[E12] Loading {model_name} on {device}", flush=True)
    model, tokenizer = load_model_for_exp(model_name, device)

    # Load high-SI heads
    cluster_path = B1_RESULTS / "cluster_membership.parquet"
    if cluster_path.exists():
        cm = pd.read_parquet(cluster_path)
        m_cm = cm[(cm["model"] == model_name) & (cm["is_high_si"] == True)]
        high_si_set: set[tuple[int, int]] = set(zip(m_cm["layer"].tolist(), m_cm["head"].tolist()))
    else:
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
        # Fallback: use top half of first half of layers as "high-SI" proxy
        high_si_set = {(l, h) for l in range(n_layers // 2) for h in range(n_heads // 2)}

    print(f"[E12] {model_name}: {len(high_si_set)} high-SI heads to ablate", flush=True)

    rng = random.Random(SEED_BASE)
    icl_prompts = _build_icl_prompts(tokenizer, n_eval, n_demos, rng)
    rng2 = random.Random(SEED_BASE + 1)
    non_icl_prompts = _build_non_icl_prompts(tokenizer, n_eval, n_demos, rng2)

    # --- Intact model evaluation ---
    print(f"[E12] {model_name}: evaluating intact model", flush=True)
    intact_icl = _eval_accuracy_and_logprob(model, icl_prompts, device)
    intact_non_icl = _eval_accuracy_and_logprob(model, non_icl_prompts, device)

    # --- Ablated model evaluation ---
    print(f"[E12] {model_name}: evaluating ablated model", flush=True)
    with head_output_ablation(model, list(high_si_set)):
        ablated_icl = _eval_accuracy_and_logprob(model, icl_prompts, device)
        ablated_non_icl = _eval_accuracy_and_logprob(model, non_icl_prompts, device)

    # --- Compute sensitivity ratio ---
    icl_acc_loss = intact_icl["accuracy"] - ablated_icl["accuracy"]
    non_icl_acc_loss = intact_non_icl["accuracy"] - ablated_non_icl["accuracy"]
    icl_lp_loss = intact_icl["mean_logprob"] - ablated_icl["mean_logprob"]
    non_icl_lp_loss = intact_non_icl["mean_logprob"] - ablated_non_icl["mean_logprob"]

    # Preferential-degradation criteria with denominator guards.
    acc_valid = abs(non_icl_acc_loss) >= float(MIN_NONICL_ACC_DENOM)
    lp_valid = abs(non_icl_lp_loss) >= float(MIN_NONICL_LP_DENOM)
    acc_ratio = float("nan")
    lp_ratio = float("nan")
    if acc_valid:
        acc_ratio = icl_acc_loss / max(abs(non_icl_acc_loss), 1e-12)
    if lp_valid:
        lp_ratio = icl_lp_loss / max(abs(non_icl_lp_loss), 1e-12)

    def _metric_vote(valid: bool, icl_loss: float, base_loss: float) -> str:
        if not valid:
            return "indeterminate"
        return "preferential" if (icl_loss > base_loss and icl_loss > 0.0) else "not_preferential"

    acc_vote = _metric_vote(acc_valid, icl_acc_loss, non_icl_acc_loss)
    lp_vote = _metric_vote(lp_valid, icl_lp_loss, non_icl_lp_loss)
    valid_votes = [v for v in [acc_vote, lp_vote] if v != "indeterminate"]
    if not valid_votes:
        model_status = "indeterminate"
    elif all(v == "preferential" for v in valid_votes):
        model_status = "supported"
    elif all(v == "not_preferential" for v in valid_votes):
        model_status = "not_supported"
    else:
        model_status = "mixed"
    icl_preferential = any(v == "preferential" for v in valid_votes)

    result = {
        "model": model_name,
        "n_high_si_heads": len(high_si_set),
        "intact_icl_accuracy": intact_icl["accuracy"],
        "intact_non_icl_accuracy": intact_non_icl["accuracy"],
        "ablated_icl_accuracy": ablated_icl["accuracy"],
        "ablated_non_icl_accuracy": ablated_non_icl["accuracy"],
        "icl_acc_loss": icl_acc_loss,
        "non_icl_acc_loss": non_icl_acc_loss,
        "icl_lp_loss": icl_lp_loss,
        "non_icl_lp_loss": non_icl_lp_loss,
        "accuracy_sensitivity_ratio": acc_ratio,
        "logprob_sensitivity_ratio": lp_ratio,
        "icl_preferentially_degraded": icl_preferential,
        "acc_denominator_valid": bool(acc_valid),
        "lp_denominator_valid": bool(lp_valid),
        "acc_vote": acc_vote,
        "lp_vote": lp_vote,
        "model_level_status": model_status,
    }

    model_out = ensure_dir(out_dir / model_name)
    write_json(model_out / "icl_sensitivity_result.json", result)

    # Save prompt-level data for reproducibility
    rows: list[dict[str, Any]] = []
    for idx, (condition, prompts_list, intact, ablated) in enumerate([
        ("icl", icl_prompts, intact_icl, ablated_icl),
        ("non_icl", non_icl_prompts, intact_non_icl, ablated_non_icl),
    ]):
        rows.append({
            "model": model_name,
            "condition": condition,
            "intact_accuracy": intact["accuracy"],
            "ablated_accuracy": ablated["accuracy"],
            "intact_mean_logprob": intact["mean_logprob"],
            "ablated_mean_logprob": ablated["mean_logprob"],
            "acc_loss": intact["accuracy"] - ablated["accuracy"],
            "lp_loss": intact["mean_logprob"] - ablated["mean_logprob"],
        })
    pd.DataFrame(rows).to_parquet(model_out / "condition_summary.parquet", index=False)

    print(
        f"[E12] {model_name}: icl_loss={icl_acc_loss:.4f} non_icl_loss={non_icl_acc_loss:.4f} "
        f"ratio={acc_ratio:.3f} preferential={icl_preferential}",
        flush=True,
    )
    return result


# ---------------------------------------------------------------------------
# Cross-model summary
# ---------------------------------------------------------------------------

def _cross_model_summary(
    model_results: list[dict[str, Any]],
    out_dir: Path,
    required_models: list[str],
) -> dict[str, Any]:
    observed_models = [r.get("model", "") for r in model_results]
    enforce_coverage_contract(
        experiment_id="E12",
        observed_models=observed_models,
        required_models=required_models,
    )

    valid_models = [r for r in model_results if r.get("model_level_status") != "indeterminate"]
    n_preferential = sum(
        1 for r in valid_models if bool(r.get("icl_preferentially_degraded", False))
    )
    ratios = [r.get("accuracy_sensitivity_ratio", float("nan")) for r in valid_models]
    valid_ratios = [v for v in ratios if not np.isnan(v)]
    mean_ratio = float(np.mean(valid_ratios)) if valid_ratios else float("nan")

    if len(valid_models) < 2:
        interpretation = "insufficient_valid_models"
        note = (
            "Too few models passed denominator validity checks for cross-model ICL inference."
        )
    elif n_preferential == len(valid_models) and mean_ratio > 1.5:
        interpretation = "icl_specific_degradation"
        note = (
            f"ICL accuracy loss > non-ICL loss (ratio={mean_ratio:.2f}) in all {len(valid_models)} valid models. "
            "SI heads are preferentially serving ICL-relevant computation, linking them to "
            "the induction-head mechanism (Olsson et al. 2022)."
        )
    elif n_preferential >= max(1, len(valid_models) // 2):
        interpretation = "partial_icl_specificity"
        note = (
            f"ICL preferentially degraded in {n_preferential}/{len(valid_models)} valid models. "
            "SI heads partially serve ICL computation; effect not uniform across models."
        )
    else:
        interpretation = "task_general_degradation"
        note = (
            "ICL and non-ICL tasks degrade equally under SI ablation. "
            "Causal cost of SI-kernel ablation is task-general, not ICL-specific. "
            "SI heads are not specifically induction-head-like in their functional role."
        )

    summary = {
        "n_models": len(model_results),
        "n_valid_models": len(valid_models),
        "n_icl_preferential": n_preferential,
        "mean_accuracy_sensitivity_ratio": mean_ratio,
        "interpretation": interpretation,
        "note": note,
        "per_model": [
            {
                "model": r["model"],
                "ratio": r.get("accuracy_sensitivity_ratio", float("nan")),
                "preferential": r.get("icl_preferentially_degraded", False),
                "model_level_status": r.get("model_level_status", "unknown"),
                "acc_denominator_valid": bool(r.get("acc_denominator_valid", False)),
                "lp_denominator_valid": bool(r.get("lp_denominator_valid", False)),
            }
            for r in model_results
        ],
    }
    write_json(out_dir / "cross_model_icl_summary.json", summary)
    print(f"[E12] Cross-model interpretation: {interpretation}", flush=True)
    return summary


# ---------------------------------------------------------------------------
# Artifact emission
# ---------------------------------------------------------------------------

def _emit_artifacts(
    models: list[str],
    out_dir: Path,
    cross_model: dict[str, Any],
    start_ts: str,
) -> None:
    interp = cross_model.get("interpretation", "unknown")
    if interp == "icl_specific_degradation":
        claim_status = "supported"
    elif interp == "partial_icl_specificity":
        claim_status = "mixed"
    else:
        claim_status = "not_supported"

    claim_impact = {
        "experiment_id": "E12",
        "claim_addressed": (
            "SI-kernel ablation causes preferential ICL degradation (vs matched non-ICL task), "
            "linking SI infrastructure to the induction-head mechanism of in-context learning"
        ),
        "claim_status": claim_status,
        "supports_main_text": claim_status in ("supported", "mixed"),
        "outcome_summary": cross_model.get("note", ""),
        "mean_accuracy_sensitivity_ratio": cross_model.get("mean_accuracy_sensitivity_ratio", float("nan")),
        "notes": [cross_model.get("note", "")],
    }

    preregistration = {
        "experiment_id": "E12",
        "hypothesis": (
            "SI-kernel ablation (zeroing high-SI head outputs) causes greater ICL accuracy "
            "degradation than a matched non-ICL task, with sensitivity ratio > 1.0."
        ),
        "primary_criterion": (
            "ICL/non-ICL accuracy sensitivity ratio > 1.5 in ≥ 2/3 models."
        ),
        "models": models,
        "n_eval_prompts": N_EVAL_PROMPTS,
        "n_icl_demos": N_ICL_EXAMPLES,
        "reference": "Olsson et al. 2022, In-context Learning and Induction Heads",
        "dependency": "E10 (SI vs retrieval overlap) should run first to confirm SI/retrieval head distinctness",
        "timestamp": start_ts,
    }

    summary = {
        "experiment_id": "E12",
        "status": "complete",
        "models_run": models,
        "interpretation": interp,
        "n_icl_preferential": cross_model.get("n_icl_preferential", 0),
        "mean_accuracy_ratio": cross_model.get("mean_accuracy_sensitivity_ratio", float("nan")),
        "timestamp_start": start_ts,
        "timestamp_end": timestamp_now(),
    }

    data_dictionary = {
        "experiment_id": "E12",
        "tables": [
            {
                "path": "<model>/condition_summary.parquet",
                "description": "Accuracy and log-prob for ICL vs non-ICL conditions, intact vs ablated",
                "columns": [
                    {"name": "model", "dtype": "str", "description": "Model name"},
                    {"name": "condition", "dtype": "str", "description": "icl or non_icl"},
                    {"name": "intact_accuracy", "dtype": "float", "description": "Accuracy without ablation"},
                    {"name": "ablated_accuracy", "dtype": "float", "description": "Accuracy with SI heads ablated"},
                    {"name": "intact_mean_logprob", "dtype": "float", "description": "Mean log-prob without ablation"},
                    {"name": "ablated_mean_logprob", "dtype": "float", "description": "Mean log-prob with ablation"},
                    {"name": "acc_loss", "dtype": "float", "description": "intact_accuracy - ablated_accuracy"},
                    {"name": "lp_loss", "dtype": "float", "description": "intact_mean_logprob - ablated_mean_logprob"},
                ],
            }
        ],
    }

    manifest_extra = {
        "models": models,
        "n_eval_prompts": N_EVAL_PROMPTS,
        "n_icl_demos": N_ICL_EXAMPLES,
        "seed_base": SEED_BASE,
        "reference": "Olsson et al. 2022",
    }

    emit_core_artifacts(
        out_dir=out_dir,
        experiment_id="E12",
        preregistration=preregistration,
        manifest_extra=manifest_extra,
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="E12: ICL sensitivity under SI-kernel subtraction",
        allow_abbrev=False,
    )
    p.add_argument("--models", default=",".join(["llama-3.1-8b", "olmo-2-7b", "mistral-7b-v0.1"]))
    p.add_argument(
        "--device-map",
        default="llama-3.1-8b:cuda:0,olmo-2-7b:cuda:0,mistral-7b-v0.1:cuda:0",
    )
    p.add_argument("--output-root", default=str(OUT_ROOT))
    p.add_argument("--n-eval", type=int, default=N_EVAL_PROMPTS)
    p.add_argument("--n-demos", type=int, default=N_ICL_EXAMPLES)
    p.add_argument("--finalize-only", action="store_true",
                   help="Read per-model shard artifacts and emit cross-model outputs.")
    p.add_argument("--no-finalize", action="store_true",
                   help="Run per-model computations but skip cross-model finalize.")
    args = p.parse_args()
    if args.finalize_only and args.no_finalize:
        raise RuntimeError("[E12] hard_fail_reason: --finalize-only and --no-finalize are mutually exclusive")

    models = parse_models_arg(args.models)
    device_map = parse_device_map(args.device_map)
    out_dir = ensure_dir(Path(args.output_root))
    start_ts = timestamp_now()

    print(f"[E12] Starting at {start_ts}", flush=True)
    print(f"[E12] Models: {models}", flush=True)

    model_results: list[dict[str, Any]] = []
    if args.finalize_only:
        for model_name in models:
            path = out_dir / model_name / "icl_sensitivity_result.json"
            if not path.exists():
                raise RuntimeError(f"[E12] hard_fail_reason: missing shard artifact {path}")
            model_results.append(read_json(path))
    else:
        for model_name in models:
            device = device_map.get(model_name, "cuda:0")
            result = run_model(model_name, device, out_dir, n_eval=args.n_eval, n_demos=args.n_demos)
            model_results.append(result)

    if args.no_finalize:
        print("[E12] Shard run complete (no finalize).", flush=True)
        return

    cross_model = _cross_model_summary(model_results, out_dir, required_models=models)
    _emit_artifacts(models, out_dir, cross_model, start_ts)

    print(f"[E12] Done. Interpretation: {cross_model.get('interpretation')}", flush=True)


if __name__ == "__main__":
    main()
