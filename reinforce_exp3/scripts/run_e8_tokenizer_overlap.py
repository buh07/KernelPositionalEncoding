#!/usr/bin/env python3
"""E8 — Tokenizer-Overlap SI Comparison.

Asks whether pairwise tokenizer similarity correlates with SI distribution
similarity across the three primary models. Pure reanalysis — no GPU needed.

Usage:
    python reinforce_exp3/scripts/run_e8_tokenizer_overlap.py \
        --models llama-3.1-8b,olmo-2-7b,mistral-7b-v0.1
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
    emit_core_artifacts,
    jaccard,
    parse_models_arg,
    spearman_with_ci,
)

OUT_ROOT = RESULTS_ROOT / "E8_tokenizer_overlap_si"


# ---------------------------------------------------------------------------
# Tokenizer vocabulary loading
# ---------------------------------------------------------------------------

def _load_tokenizer_vocab(model_name: str) -> dict[str, int]:
    """Return {token_str: token_id} dict for a model's tokenizer."""
    try:
        import sys as _sys
        _root = Path(__file__).resolve().parents[2]
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        from experiment3.theory1_si_circuits import MODELS
        from shared.models.loading import load_tokenizer as _proj_load_tok
        spec = MODELS[model_name]
        tok = _proj_load_tok(spec)
        return dict(tok.get_vocab())
    except Exception as exc:
        print(f"[E8] Warning: could not load tokenizer for {model_name}: {exc}", flush=True)
        return {}


def _get_boundary_token_fraction(vocab: dict[str, int]) -> float:
    """Fraction of vocabulary tokens that are word-initial (space-prefix)."""
    if not vocab:
        return float("nan")
    n_boundary = sum(
        1 for t in vocab
        if t.startswith(" ") or t.startswith("▁") or t.startswith("Ġ")
    )
    return n_boundary / len(vocab)


def _compute_bpe_merge_rank_similarity(
    vocab_a: dict[str, int],
    vocab_b: dict[str, int],
) -> float:
    """Spearman correlation of token IDs for shared tokens (proxy for merge rank)."""
    shared = set(vocab_a.keys()) & set(vocab_b.keys())
    if len(shared) < 10:
        return float("nan")
    shared_list = sorted(shared)
    ids_a = [vocab_a[t] for t in shared_list]
    ids_b = [vocab_b[t] for t in shared_list]
    rho, _ = scipy_stats.spearmanr(ids_a, ids_b)
    return float(rho)


# ---------------------------------------------------------------------------
# SI distribution similarity metrics
# ---------------------------------------------------------------------------

def _load_r2_per_head(models: list[str]) -> dict[str, pd.DataFrame]:
    """Load per-head R² for each model from cluster_membership parquet."""
    path = B1_RESULTS / "cluster_membership.parquet"
    if not path.exists():
        raise FileNotFoundError(f"cluster_membership.parquet not found at {path}")
    cm = pd.read_parquet(path)
    result: dict[str, pd.DataFrame] = {}
    for m in models:
        df_m = cm[cm["model"] == m].copy()
        if df_m.empty:
            print(f"[E8] Warning: no R² data for {m}", flush=True)
        result[m] = df_m
    return result


def _kl_divergence_r2(r2_a: np.ndarray, r2_b: np.ndarray, n_bins: int = 30) -> float:
    """KL divergence between two R² distributions using histogram density estimate."""
    r2_a = r2_a[np.isfinite(r2_a)]
    r2_b = r2_b[np.isfinite(r2_b)]
    if len(r2_a) < 5 or len(r2_b) < 5:
        return float("nan")

    eps = 1e-10
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    hist_a, _ = np.histogram(r2_a, bins=bins, density=True)
    hist_b, _ = np.histogram(r2_b, bins=bins, density=True)
    hist_a = hist_a + eps
    hist_b = hist_b + eps
    hist_a /= hist_a.sum()
    hist_b /= hist_b.sum()
    kl = float(np.sum(hist_a * np.log(hist_a / hist_b)))
    return kl


def _r2_rank_spearman(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
) -> float:
    """Spearman rank correlation of per-head R² by (layer, head) position."""
    df_a = df_a.set_index(["layer", "head"])["mean_r2"]
    df_b = df_b.set_index(["layer", "head"])["mean_r2"]
    common_idx = df_a.index.intersection(df_b.index)
    if len(common_idx) < 10:
        return float("nan")
    rho, _ = scipy_stats.spearmanr(df_a.loc[common_idx].values, df_b.loc[common_idx].values)
    return float(rho)


# ---------------------------------------------------------------------------
# Per-pair analysis
# ---------------------------------------------------------------------------

def _analyze_pair(
    model_a: str,
    model_b: str,
    vocabs: dict[str, dict[str, int]],
    r2_dfs: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    vocab_a = vocabs[model_a]
    vocab_b = vocabs[model_b]
    df_a = r2_dfs[model_a]
    df_b = r2_dfs[model_b]

    # Tokenizer overlap metrics
    tokens_a = set(vocab_a.keys())
    tokens_b = set(vocab_b.keys())
    jaccard_overlap = jaccard(tokens_a, tokens_b)
    merge_rank_rho = _compute_bpe_merge_rank_similarity(vocab_a, vocab_b)
    boundary_frac_a = _get_boundary_token_fraction(vocab_a)
    boundary_frac_b = _get_boundary_token_fraction(vocab_b)
    boundary_frac_diff = abs(boundary_frac_a - boundary_frac_b)

    # SI distribution similarity metrics
    r2_a = df_a["mean_r2"].values if not df_a.empty else np.array([])
    r2_b = df_b["mean_r2"].values if not df_b.empty else np.array([])
    kl_div = _kl_divergence_r2(r2_a, r2_b)
    r2_rank_rho = _r2_rank_spearman(df_a, df_b)
    mean_r2_gap = (
        abs(float(np.nanmean(r2_a)) - float(np.nanmean(r2_b)))
        if len(r2_a) > 0 and len(r2_b) > 0
        else float("nan")
    )

    result = {
        "model_a": model_a,
        "model_b": model_b,
        "tokenizer": {
            "vocab_size_a": len(vocab_a),
            "vocab_size_b": len(vocab_b),
            "jaccard_overlap": jaccard_overlap,
            "merge_rank_spearman_rho": merge_rank_rho,
            "boundary_frac_a": boundary_frac_a,
            "boundary_frac_b": boundary_frac_b,
            "boundary_frac_diff": boundary_frac_diff,
        },
        "si_distribution": {
            "kl_divergence": kl_div,
            "r2_rank_spearman_rho": r2_rank_rho,
            "mean_r2_gap": mean_r2_gap,
            "mean_r2_a": float(np.nanmean(r2_a)) if len(r2_a) > 0 else float("nan"),
            "mean_r2_b": float(np.nanmean(r2_b)) if len(r2_b) > 0 else float("nan"),
        },
    }
    return result


# ---------------------------------------------------------------------------
# Cross-pair correlation
# ---------------------------------------------------------------------------

def _summarize_tokenizer_si_correlation(
    pair_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Test whether tokenizer similarity correlates with SI similarity (n=3 pairs)."""
    # Extract vectors
    jaccard_vals = [r["tokenizer"]["jaccard_overlap"] for r in pair_results]
    merge_rho_vals = [r["tokenizer"]["merge_rank_spearman_rho"] for r in pair_results]
    kl_vals = [r["si_distribution"]["kl_divergence"] for r in pair_results]
    r2_rho_vals = [r["si_distribution"]["r2_rank_spearman_rho"] for r in pair_results]
    r2_gap_vals = [r["si_distribution"]["mean_r2_gap"] for r in pair_results]

    def _safe_spearman(x: list[float], y: list[float]) -> float:
        x_arr = np.array(x)
        y_arr = np.array(y)
        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        if mask.sum() < 3:
            return float("nan")
        rho, _ = scipy_stats.spearmanr(x_arr[mask], y_arr[mask])
        return float(rho)

    # With n=3 pairs, these correlations are only interpretable if monotone and large
    corr_jaccard_kl = _safe_spearman(jaccard_vals, kl_vals)
    corr_jaccard_r2gap = _safe_spearman(jaccard_vals, r2_gap_vals)
    corr_mergerho_kl = _safe_spearman(merge_rho_vals, kl_vals)

    # Interpretation
    meaningful_threshold = 0.8
    any_large = any(
        abs(v) >= meaningful_threshold
        for v in [corr_jaccard_kl, corr_jaccard_r2gap, corr_mergerho_kl]
        if not np.isnan(v)
    )

    if any_large:
        interpretation = "tokenizer_similarity_correlates_with_si"
        note = (
            "At least one tokenizer–SI correlation has |ρ| ≥ 0.8 (n=3 pairs). "
            "Tokenizer is a plausible driver of SI distribution variation. "
            "Note: n=3 is insufficient for inference; treat as directional signal only."
        )
    else:
        interpretation = "no_clear_tokenizer_si_correlation"
        note = (
            "No tokenizer similarity metric strongly correlates with SI distribution similarity "
            "(|ρ| < 0.8 across all metrics; n=3 pairs). "
            "Cross-model SI variation not explained by tokenizer overlap in this analysis."
        )

    return {
        "n_pairs": len(pair_results),
        "correlation_jaccard_vs_kl_divergence": corr_jaccard_kl,
        "correlation_jaccard_vs_r2_gap": corr_jaccard_r2gap,
        "correlation_merge_rank_vs_kl": corr_mergerho_kl,
        "interpretation": interpretation,
        "note": note,
        "caveat": "n=3 model pairs; no valid statistical inference possible in isolation.",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="E8: Tokenizer-overlap vs SI-distribution similarity analysis",
        allow_abbrev=False,
    )
    p.add_argument("--models", default=",".join(["llama-3.1-8b", "olmo-2-7b", "mistral-7b-v0.1"]))
    p.add_argument("--output-root", default=str(OUT_ROOT))
    args = p.parse_args()

    models = parse_models_arg(args.models)
    out_dir = ensure_dir(Path(args.output_root))
    start_ts = timestamp_now()
    print(f"[E8] Starting at {start_ts}", flush=True)

    # Load tokenizer vocabularies
    print("[E8] Loading tokenizer vocabularies", flush=True)
    vocabs: dict[str, dict[str, int]] = {}
    for m in models:
        vocabs[m] = _load_tokenizer_vocab(m)
        print(f"[E8] {m}: {len(vocabs[m])} tokens", flush=True)

    # Load R² data
    print("[E8] Loading per-head R² data", flush=True)
    try:
        r2_dfs = _load_r2_per_head(models)
    except FileNotFoundError as exc:
        print(f"[E8] ERROR: {exc}", flush=True)
        sys.exit(1)

    # Pairwise analysis
    pair_results: list[dict[str, Any]] = []
    for ma, mb in itertools.combinations(models, 2):
        print(f"[E8] Analyzing pair: {ma} vs {mb}", flush=True)
        result = _analyze_pair(ma, mb, vocabs, r2_dfs)
        pair_results.append(result)
        print(
            f"[E8]   jaccard={result['tokenizer']['jaccard_overlap']:.4f} "
            f"kl={result['si_distribution']['kl_divergence']:.4f} "
            f"r2_gap={result['si_distribution']['mean_r2_gap']:.4f}",
            flush=True,
        )

    write_json(out_dir / "tokenizer_overlap_matrix.json", {"pairs": pair_results})

    # SI distribution similarity matrix
    si_matrix = {
        "models": models,
        "pairs": [
            {
                "model_a": r["model_a"],
                "model_b": r["model_b"],
                **r["si_distribution"],
            }
            for r in pair_results
        ],
    }
    write_json(out_dir / "si_distribution_similarity_matrix.json", si_matrix)

    # Correlation summary
    correlation_summary = _summarize_tokenizer_si_correlation(pair_results)
    write_json(out_dir / "correlation_summary.json", correlation_summary)
    print(f"[E8] Interpretation: {correlation_summary['interpretation']}", flush=True)

    # Build parquet for reproducibility
    rows = []
    for r in pair_results:
        rows.append({
            "model_a": r["model_a"],
            "model_b": r["model_b"],
            "jaccard_overlap": r["tokenizer"]["jaccard_overlap"],
            "merge_rank_rho": r["tokenizer"]["merge_rank_spearman_rho"],
            "boundary_frac_diff": r["tokenizer"]["boundary_frac_diff"],
            "kl_divergence": r["si_distribution"]["kl_divergence"],
            "r2_rank_rho": r["si_distribution"]["r2_rank_spearman_rho"],
            "mean_r2_gap": r["si_distribution"]["mean_r2_gap"],
        })
    pd.DataFrame(rows).to_parquet(out_dir / "pairwise_overlap_table.parquet", index=False)

    # Emit governance artifacts
    interp = correlation_summary["interpretation"]
    claim_status = "mixed" if interp == "tokenizer_similarity_correlates_with_si" else "inconclusive"

    claim_impact = {
        "experiment_id": "E8",
        "claim_addressed": (
            "Cross-model SI strength variation may be partially explained by tokenizer "
            "vocabulary overlap and merge-rank similarity"
        ),
        "claim_status": claim_status,
        "supports_main_text": False,
        "outcome_summary": correlation_summary["note"],
        "notes": [
            correlation_summary["note"],
            correlation_summary["caveat"],
        ],
    }

    preregistration = {
        "experiment_id": "E8",
        "hypothesis": (
            "Models with more similar tokenizers (higher Jaccard overlap, higher merge-rank ρ) "
            "will have more similar SI distributions (lower KL divergence, lower mean-R² gap)."
        ),
        "primary_criterion": (
            "At least one tokenizer–SI cross-pair correlation has |ρ| ≥ 0.8. "
            "Caveat: n=3 pairs is insufficient for inference; result is directional only."
        ),
        "models": models,
        "timestamp": start_ts,
    }

    summary = {
        "experiment_id": "E8",
        "status": "complete",
        "models_analyzed": models,
        "n_pairs": len(pair_results),
        "interpretation": interp,
        "timestamp_start": start_ts,
        "timestamp_end": timestamp_now(),
    }

    data_dictionary = {
        "experiment_id": "E8",
        "tables": [
            {
                "path": "pairwise_overlap_table.parquet",
                "description": "Pairwise tokenizer-overlap and SI-similarity metrics for all model pairs",
                "columns": [
                    {"name": "model_a", "dtype": "str", "description": "First model in pair"},
                    {"name": "model_b", "dtype": "str", "description": "Second model in pair"},
                    {"name": "jaccard_overlap", "dtype": "float", "description": "Jaccard vocabulary overlap"},
                    {"name": "merge_rank_rho", "dtype": "float", "description": "Spearman ρ of token IDs for shared tokens"},
                    {"name": "boundary_frac_diff", "dtype": "float", "description": "|boundary_token_fraction_A - boundary_token_fraction_B|"},
                    {"name": "kl_divergence", "dtype": "float", "description": "KL divergence of per-head R² distributions"},
                    {"name": "r2_rank_rho", "dtype": "float", "description": "Spearman ρ of per-head R² rank (by layer-head position)"},
                    {"name": "mean_r2_gap", "dtype": "float", "description": "|mean_R²_A - mean_R²_B|"},
                ],
            }
        ],
    }

    manifest_extra = {
        "models": models,
        "data_source": str(B1_RESULTS / "cluster_membership.parquet"),
    }

    emit_core_artifacts(
        out_dir=out_dir,
        experiment_id="E8",
        preregistration=preregistration,
        manifest_extra=manifest_extra,
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )

    print(f"[E8] Done. Results in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
