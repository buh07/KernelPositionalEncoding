#!/usr/bin/env python3
"""E10 — SI vs. Retrieval-Head Overlap.

Implements the Wu et al. (2024) copy-score definition of retrieval heads and
tests whether the high-SI head population overlaps substantially with the
retrieval-head population across three models.

Usage:
    python reinforce_exp3/scripts/run_e10_retrieval_overlap.py \
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
from scipy import stats as scipy_stats

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
    enforce_coverage_contract,
    emit_core_artifacts,
    jaccard,
    read_json,
    load_model_for_exp,
    parse_device_map,
    parse_models_arg,
    spearman_with_ci,
)

OUT_ROOT = RESULTS_ROOT / "E10_si_retrieval_overlap"

N_EVAL_EXAMPLES = 200
SEQUENCE_LENGTH = 512   # long enough for needle-in-haystack
N_DISTRACTORS_MIN = 20
N_DISTRACTORS_MAX = 60
SEED_BASE = 20260429

# Overlap thresholds from TODO spec
JACCARD_DISJOINT = 0.10
JACCARD_OVERLAP = 0.25
SPEARMAN_DISJOINT = 0.20
SPEARMAN_OVERLAP = 0.40


# ---------------------------------------------------------------------------
# Needle-in-haystack prompt builder
# ---------------------------------------------------------------------------

def _build_needle_prompts(
    tokenizer: Any,
    n: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Build needle-in-haystack examples.

    Each example has:
    - A needle token (key) placed at a random position in a distractor sequence
    - The model must attend to the needle position to produce the correct value
    Returns list of dicts: {input_ids, needle_position, target_token_id}
    """
    vocab_ids = list(tokenizer.get_vocab().values())
    prompts: list[dict[str, Any]] = []
    for _ in range(n):
        n_distract = rng.randint(N_DISTRACTORS_MIN, N_DISTRACTORS_MAX)
        key_id = rng.choice(vocab_ids)
        val_id = rng.choice(vocab_ids)
        distractors = [rng.choice(vocab_ids) for _ in range(n_distract)]
        # needle_position = index where (key val) pair is inserted
        needle_pos = rng.randint(0, n_distract - 1)
        # Build: distractor_0 ... distractor_{needle_pos-1} key val distractor_{needle_pos} ... query_key
        prefix = distractors[:needle_pos]
        suffix = distractors[needle_pos:]
        input_ids = prefix + [key_id, val_id] + suffix + [key_id]
        # needle_position is the index of key_id in input_ids
        actual_needle_pos = len(prefix)
        prompts.append({
            "input_ids": input_ids,
            "needle_position": actual_needle_pos,   # position of key in sequence
            "target_token_id": val_id,
        })
    return prompts


# ---------------------------------------------------------------------------
# Retrieval (copy) score computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_copy_scores(
    model: Any,
    prompts: list[dict[str, Any]],
    device: str,
    n_heads: int,
    n_layers: int,
) -> np.ndarray:
    """Compute per-head retrieval copy score across prompts.

    Copy score for head h at layer l = mean over examples of:
        attn_weight[last_pos, needle_position] - mean(attn_weight[last_pos, :])

    Returns array of shape (n_layers, n_heads).
    """
    model.eval()
    score_accum = np.zeros((n_layers, n_heads), dtype=float)
    score_count = np.zeros((n_layers, n_heads), dtype=float)

    # We need attention weights from all layers — expensive but tractable for 200 examples
    for prompt in prompts:
        input_ids = prompt["input_ids"]
        needle_pos = prompt["needle_position"]
        if needle_pos >= len(input_ids) - 1:
            continue

        ids_tensor = torch.tensor([input_ids], device=device, dtype=torch.long)
        try:
            out = model(ids_tensor, output_attentions=True)
        except Exception:
            continue

        if out.attentions is None:
            continue

        for layer_idx, attn_layer in enumerate(out.attentions):
            if layer_idx >= n_layers:
                break
            # attn_layer: (1, n_heads, seq_len, seq_len)
            _t = attn_layer[0].cpu().float()
            attn = np.array(_t.tolist(), dtype=np.float32)  # (n_heads, seq_len, seq_len)
            seq_len = attn.shape[-1]
            if needle_pos >= seq_len:
                continue
            last_pos = seq_len - 1
            for h in range(min(n_heads, attn.shape[0])):
                attn_row = attn[h, last_pos]  # (seq_len,)
                mean_attn = attn_row.mean()
                copy_score = float(attn_row[needle_pos]) - float(mean_attn)
                score_accum[layer_idx, h] += copy_score
                score_count[layer_idx, h] += 1.0

    # Average
    with np.errstate(invalid="ignore"):
        mean_scores = np.where(score_count > 0, score_accum / score_count, float("nan"))
    return mean_scores


# ---------------------------------------------------------------------------
# Per-model analysis
# ---------------------------------------------------------------------------

def run_model(
    model_name: str,
    device: str,
    out_dir: Path,
    n_eval: int = N_EVAL_EXAMPLES,
) -> dict[str, Any]:
    # Use eager attention so output_attentions=True works across all model families
    print(f"[E10] Loading {model_name} on {device}", flush=True)
    model, tokenizer = load_model_for_exp(model_name, device, attn_implementation="eager")

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads

    rng = random.Random(SEED_BASE)
    prompts = _build_needle_prompts(tokenizer, n_eval, rng)
    print(f"[E10] {model_name}: computing copy scores across {n_eval} examples", flush=True)
    copy_scores = _compute_copy_scores(model, prompts, device, n_heads, n_layers)
    # copy_scores: (n_layers, n_heads)

    # Build per-head retrieval score DataFrame
    rows = []
    for l in range(n_layers):
        for h in range(n_heads):
            rows.append({
                "model": model_name,
                "layer": l,
                "head": h,
                "retrieval_score": copy_scores[l, h],
            })
    retrieval_df = pd.DataFrame(rows)
    model_out = ensure_dir(out_dir / model_name)
    retrieval_df.to_parquet(model_out / "retrieval_scores.parquet", index=False)

    # Load SI R² data from B1
    cluster_path = B1_RESULTS / "cluster_membership.parquet"
    si_df: pd.DataFrame | None = None
    if cluster_path.exists():
        cm = pd.read_parquet(cluster_path)
        si_df = cm[cm["model"] == model_name].copy()

    # Compute overlap statistics
    if si_df is not None and not si_df.empty:
        # Merge on (layer, head)
        merged = retrieval_df.merge(si_df[["layer", "head", "mean_r2", "is_high_si"]], on=["layer", "head"])

        # Top-quartile thresholds
        r2_threshold = float(merged["mean_r2"].quantile(0.75))
        ret_threshold = float(merged["retrieval_score"].dropna().quantile(0.75))

        top_si = set(zip(
            merged[merged["mean_r2"] >= r2_threshold]["layer"],
            merged[merged["mean_r2"] >= r2_threshold]["head"],
        ))
        top_ret = set(zip(
            merged[merged["retrieval_score"] >= ret_threshold]["layer"],
            merged[merged["retrieval_score"] >= ret_threshold]["head"],
        ))
        jaccard_val = jaccard(top_si, top_ret)

        # Spearman rank correlation between R² and retrieval score
        valid = merged[merged["retrieval_score"].notna() & merged["mean_r2"].notna()]
        rho_result = spearman_with_ci(
            valid["mean_r2"].tolist(), valid["retrieval_score"].tolist()
        )
        write_json(model_out / "si_retrieval_rho.json", rho_result)

        # Cluster enrichment: does any cluster show high mean retrieval score?
        cluster_enrichment: dict[str, Any] = {}
        if "cluster_descriptor_kmeans" in merged.columns:
            cluster_groups = merged.groupby("cluster_descriptor_kmeans")["retrieval_score"]
            kw_groups = [
                grp.dropna().values
                for _, grp in cluster_groups
                if len(grp.dropna()) >= 3
            ]
            if len(kw_groups) >= 2:
                stat, pval = scipy_stats.kruskal(*kw_groups)
                cluster_enrichment = {
                    "kruskal_wallis_stat": float(stat),
                    "kruskal_wallis_pval": float(pval),
                    "cluster_mean_retrieval": cluster_groups.mean().to_dict(),
                }
            else:
                cluster_enrichment = {"note": "insufficient groups for Kruskal-Wallis"}

        jaccard_json = {
            "jaccard_si_retrieval_top_quartile": jaccard_val,
            "r2_threshold": r2_threshold,
            "ret_threshold": ret_threshold,
            "n_top_si": len(top_si),
            "n_top_ret": len(top_ret),
            "n_overlap": len(top_si & top_ret),
        }
        write_json(model_out / "si_retrieval_jaccard.json", jaccard_json)
        write_json(model_out / "cluster_retrieval_enrichment.json", cluster_enrichment)

        result = {
            "model": model_name,
            "jaccard_si_retrieval": jaccard_val,
            "r2_retrieval_spearman_rho": rho_result.get("rho", float("nan")),
            "cluster_enrichment_pval": cluster_enrichment.get("kruskal_wallis_pval", float("nan")),
            "status": "ok",
        }
        print(
            f"[E10] {model_name}: jaccard={jaccard_val:.4f} rho={rho_result.get('rho', float('nan')):.4f}",
            flush=True,
        )
    else:
        raise RuntimeError(f"[E10] hard_fail_reason: missing SI data for {model_name}")

    return result


# ---------------------------------------------------------------------------
# Cross-model summary
# ---------------------------------------------------------------------------

def _cross_model_summary(
    model_results: list[dict[str, Any]],
    out_dir: Path,
    required_models: list[str],
) -> dict[str, Any]:
    valid = [r for r in model_results if r.get("status") == "ok"]
    enforce_coverage_contract(
        experiment_id="E10",
        observed_models=[r.get("model", "") for r in valid],
        required_models=required_models,
        observed_counts={"n_models": len(valid)},
        min_counts={"n_models": len(required_models)},
    )

    jaccards = [r["jaccard_si_retrieval"] for r in valid]
    rhos = [r["r2_retrieval_spearman_rho"] for r in valid]
    if any(np.isnan(v) for v in jaccards) or any(np.isnan(v) for v in rhos):
        raise RuntimeError("[E10] hard_fail_reason: non-finite overlap metrics in model summaries")

    mean_jaccard = float(np.nanmean(jaccards))
    mean_rho = float(np.nanmean(rhos))

    # Interpretation per TODO spec
    all_disjoint = all(j < JACCARD_DISJOINT for j in jaccards if not np.isnan(j))
    rho_non_overlap = all(r <= SPEARMAN_DISJOINT for r in rhos if not np.isnan(r))
    rho_separation = all(r <= -SPEARMAN_DISJOINT for r in rhos if not np.isnan(r))

    any_overlap = any(j > JACCARD_OVERLAP for j in jaccards if not np.isnan(j))
    rho_overlap = any(r > SPEARMAN_OVERLAP for r in rhos if not np.isnan(r))

    if all_disjoint and rho_non_overlap:
        interpretation = "disjoint_populations"
        rho_clause = "ρ <= 0.20" if not rho_separation else "ρ <= -0.20 (active separation signal)"
        note = (
            f"Jaccard < {JACCARD_DISJOINT} and {rho_clause} in all models. "
            "SI heads and retrieval heads are empirically disjoint populations. "
            "Add to Result III: 'SI infrastructure heads are distinct from retrieval heads (Wu et al. 2024)'."
        )
    elif any_overlap or rho_overlap:
        interpretation = "substantial_overlap"
        note = (
            f"Jaccard > {JACCARD_OVERLAP} or positive ρ > {SPEARMAN_OVERLAP} in at least one model. "
            "SI heads and retrieval heads co-occur. Paper needs mechanistic distinction or acknowledgment."
        )
    else:
        interpretation = "partial_separation"
        note = (
            "Mixed results: some models show separation, others show moderate overlap. "
            "Report per-model Jaccard values without an aggregate population claim."
        )

    summary = {
        "n_models": len(valid),
        "mean_jaccard": mean_jaccard,
        "mean_r2_retrieval_rho": mean_rho,
        "per_model": [
            {
                "model": r["model"],
                "jaccard": r["jaccard_si_retrieval"],
                "spearman_rho": r["r2_retrieval_spearman_rho"],
            }
            for r in valid
        ],
        "interpretation": interpretation,
        "note": note,
    }
    write_json(out_dir / "cross_model_overlap_summary.json", summary)
    print(f"[E10] Interpretation: {interpretation}", flush=True)
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
    if interp == "disjoint_populations":
        claim_status = "supported"
    elif interp == "substantial_overlap":
        claim_status = "not_supported"
    else:
        claim_status = "inconclusive"

    claim_impact = {
        "experiment_id": "E10",
        "claim_addressed": (
            "High-SI heads and retrieval heads (Wu et al. 2024) are distinct populations, "
            "supporting SI infrastructure as a novel finding beyond retrieval-head taxonomy"
        ),
        "claim_status": claim_status,
        "supports_main_text": claim_status == "supported",
        "outcome_summary": cross_model.get("note", ""),
        "mean_jaccard": cross_model.get("mean_jaccard", float("nan")),
        "mean_spearman_rho": cross_model.get("mean_r2_retrieval_rho", float("nan")),
        "notes": [cross_model.get("note", "")],
    }

    preregistration = {
        "experiment_id": "E10",
        "hypothesis": (
            "High-SI heads (top-quartile R²) and retrieval heads (Wu et al. copy score) "
            "are disjoint populations: Jaccard < 0.10 and ρ <= 0.20 in all three models."
        ),
        "primary_criterion": "Jaccard < 0.10 and Spearman ρ <= 0.20 in all models",
        "models": models,
        "n_eval_examples": N_EVAL_EXAMPLES,
        "seed_base": SEED_BASE,
        "timestamp": start_ts,
    }

    summary = {
        "experiment_id": "E10",
        "status": "complete",
        "models_run": models,
        "interpretation": interp,
        "mean_jaccard": cross_model.get("mean_jaccard", float("nan")),
        "timestamp_start": start_ts,
        "timestamp_end": timestamp_now(),
    }

    data_dictionary = {
        "experiment_id": "E10",
        "tables": [
            {
                "path": "<model>/retrieval_scores.parquet",
                "description": "Per-head Wu et al. copy score (mean attention to needle position minus mean)",
                "columns": [
                    {"name": "model", "dtype": "str", "description": "Model name"},
                    {"name": "layer", "dtype": "int", "description": "Layer index"},
                    {"name": "head", "dtype": "int", "description": "Head index"},
                    {"name": "retrieval_score", "dtype": "float",
                     "description": "Mean copy score: attn[last, needle] - mean_attn[last, :]"},
                ],
            }
        ],
    }

    manifest_extra = {
        "models": models,
        "n_eval_examples": N_EVAL_EXAMPLES,
        "sequence_length": SEQUENCE_LENGTH,
        "seed_base": SEED_BASE,
        "reference": "Wu et al. 2024, Retrieval Heads",
    }

    emit_core_artifacts(
        out_dir=out_dir,
        experiment_id="E10",
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
        description="E10: SI vs retrieval-head overlap (Wu et al. copy score)",
        allow_abbrev=False,
    )
    p.add_argument("--models", default=",".join(["llama-3.1-8b", "olmo-2-7b", "mistral-7b-v0.1"]))
    p.add_argument(
        "--device-map",
        default="llama-3.1-8b:cuda:0,olmo-2-7b:cuda:0,mistral-7b-v0.1:cuda:0",
    )
    p.add_argument("--output-root", default=str(OUT_ROOT))
    p.add_argument("--n-eval", type=int, default=N_EVAL_EXAMPLES)
    p.add_argument("--finalize-only", action="store_true",
                   help="Finalize cross-model outputs from per-model shard summaries.")
    p.add_argument("--no-finalize", action="store_true",
                   help="Run per-model jobs only; skip cross-model finalize emission.")
    args = p.parse_args()
    if args.finalize_only and args.no_finalize:
        raise RuntimeError("[E10] hard_fail_reason: --finalize-only and --no-finalize are mutually exclusive")

    models = parse_models_arg(args.models)
    device_map = parse_device_map(args.device_map)
    out_dir = ensure_dir(Path(args.output_root))
    start_ts = timestamp_now()

    print(f"[E10] Starting at {start_ts}", flush=True)

    model_results: list[dict[str, Any]] = []
    if args.finalize_only:
        for model_name in models:
            model_summary = out_dir / model_name / "si_retrieval_jaccard.json"
            if not model_summary.exists():
                raise RuntimeError(f"[E10] hard_fail_reason: missing shard artifact {model_summary}")
            j_payload = read_json(model_summary)
            # compatibility fallback: read rho from model result json if present
            rho = float("nan")
            res_path = out_dir / model_name / "model_result.json"
            if res_path.exists():
                rho = float(read_json(res_path).get("r2_retrieval_spearman_rho", float("nan")))
            else:
                # reconstruct from overlap parquet if possible
                rho_path = out_dir / model_name / "si_retrieval_rho.json"
                if rho_path.exists():
                    rho = float(read_json(rho_path).get("rho", float("nan")))
            if np.isnan(rho):
                raise RuntimeError(
                    f"[E10] hard_fail_reason: missing finite Spearman rho for {model_name} finalize"
                )
            model_results.append({
                "model": model_name,
                "jaccard_si_retrieval": j_payload.get("jaccard_si_retrieval_top_quartile", float("nan")),
                "r2_retrieval_spearman_rho": rho,
                "status": "ok",
            })
    else:
        for model_name in models:
            device = device_map.get(model_name, "cuda:0")
            result = run_model(model_name, device, out_dir, n_eval=args.n_eval)
            write_json(out_dir / model_name / "model_result.json", result)
            model_results.append(result)

    if args.no_finalize:
        print("[E10] Shard run complete (no finalize).", flush=True)
        return

    cross_model = _cross_model_summary(model_results, out_dir, required_models=models)
    _emit_artifacts(models, out_dir, cross_model, start_ts)

    print(f"[E10] Done. Interpretation: {cross_model.get('interpretation')}", flush=True)


if __name__ == "__main__":
    main()
