#!/usr/bin/env python3
"""E0 — Cluster-Sequential vs. Interleaved Ablation.

Tests whether the threshold-piecewise capacity collapse in Result III is driven by
cluster-boundary depletion (mechanism 2) vs. genuine distributed redundancy.

Three ablation orderings are compared per model:
  sequential  — exhaust smallest cluster first, then next, then largest (within each
                cluster, heads ranked by R² descending)
  interleaved — round-robin across clusters, one head per cluster per round (within
                each cluster, heads ranked by R² descending)
  r2ranked    — standard descending-R² ordering (reference / replication of 3P2-C)

Primary test: does sequential ordering produce a statistically earlier collapse
(lower tau) than interleaved? H_A: tau_sequential < tau_interleaved.

Usage:
    python reinforce_exp3/scripts/run_e0_cluster_sequential.py \\
        --models llama-3.1-8b,olmo-2-7b,mistral-7b-v0.1 \\
        --device-map "llama-3.1-8b:cuda:0,olmo-2-7b:cuda:1,mistral-7b-v0.1:cuda:2" \\
        --num-seeds 5 --synthetic-count 100 --ntp-count-per-seed 100
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp3.common import (  # noqa: E402
    B1_RESULTS,
    PRIMARY_MODELS,
    RESULTS_ROOT,
    command_manifest,
    ensure_dir,
    load_cluster_membership,
    load_head_groups,
    read_json,
    load_r2_summary,
    safe_float,
    timestamp_now,
    write_json,
)
from reinforce_exp3.scripts._shared import (  # noqa: E402
    emit_core_artifacts,
    fit_three_models,
    holm_adjust_dict,
    ordering_verdict,
    parse_device_map,
    parse_models_arg,
    permutation_test_one_sided_less,
    stream_seed,
)

from experiment2.tasks import build_token_pools  # noqa: E402
from experiment3.phase2.exp3p2c_redundancy_quantification import (  # noqa: E402
    _build_synthetic_cells,
    _evaluate_ntp_losses,
    _load_ranked_heads,
    _prepare_ntp_seed_chunks,
)
from experiment3.theory1_si_circuits import MODELS, RETRIEVAL_SPANS, HeadID, evaluate_task_battery  # noqa: E402
from shared.models.loading import load_model, load_tokenizer  # noqa: E402

DEFAULT_OUT = RESULTS_ROOT / "E0_cluster_sequential_ablation"
EXPERIMENT_ID = "E0"

# Ablation fractions (percent of high-SI heads removed), finer near expected threshold
FRACTIONS = (0, 5, 10, 15, 20, 25, 30, 40, 50)
MIN_TAU_FITS_PER_ORDERING = 3
PERM_ROOT_SEED = 20260429


def _condition_cache_path(cache_dir: Path, ordering_name: str, fraction_pct: int) -> Path:
    return cache_dir / f"{ordering_name}_f{int(fraction_pct):02d}.parquet"


def _load_condition_cache(cache_path: Path) -> pd.DataFrame | None:
    if not cache_path.exists():
        return None
    return pd.read_parquet(cache_path)


def _write_condition_cache(cache_path: Path, df: pd.DataFrame) -> None:
    df.to_parquet(cache_path, index=False)


# ---------------------------------------------------------------------------
# Ordering constructors
# ---------------------------------------------------------------------------

def _build_sequential_ordering(
    cluster_df: pd.DataFrame,
    r2_df: pd.DataFrame,
    model_name: str,
) -> list[tuple[int, int]]:
    """Exhaust smallest cluster first, then next-smallest, largest last.
    Within each cluster, heads sorted descending by R².
    """
    mdf = cluster_df[cluster_df["model"] == model_name].copy()
    mdf = mdf[mdf["is_high_si"] == True].copy()  # noqa: E712

    r2_map = {
        (int(row.layer), int(row.head)): float(row.mean_r2)
        for row in r2_df.itertuples()
    }

    clusters = sorted(mdf["cluster_descriptor_kmeans"].unique())
    cluster_sizes = {c: int((mdf["cluster_descriptor_kmeans"] == c).sum()) for c in clusters}
    # Order: smallest cluster first
    ordered_clusters = sorted(clusters, key=lambda c: cluster_sizes[c])

    result: list[tuple[int, int]] = []
    for c in ordered_clusters:
        heads = [
            (int(row.layer), int(row.head))
            for row in mdf[mdf["cluster_descriptor_kmeans"] == c].itertuples()
        ]
        heads.sort(key=lambda lh: r2_map.get(lh, 0.0), reverse=True)
        result.extend(heads)
    return result


def _build_interleaved_ordering(
    cluster_df: pd.DataFrame,
    r2_df: pd.DataFrame,
    model_name: str,
) -> list[tuple[int, int]]:
    """Round-robin across clusters: one head per cluster per round.
    Within each cluster, heads pre-sorted by R² descending; each round takes next.
    """
    mdf = cluster_df[cluster_df["model"] == model_name].copy()
    mdf = mdf[mdf["is_high_si"] == True].copy()  # noqa: E712

    r2_map = {
        (int(row.layer), int(row.head)): float(row.mean_r2)
        for row in r2_df.itertuples()
    }

    clusters = sorted(mdf["cluster_descriptor_kmeans"].unique())
    queues: list[list[tuple[int, int]]] = []
    for c in clusters:
        heads = [
            (int(row.layer), int(row.head))
            for row in mdf[mdf["cluster_descriptor_kmeans"] == c].itertuples()
        ]
        heads.sort(key=lambda lh: r2_map.get(lh, 0.0), reverse=True)
        queues.append(heads)

    result: list[tuple[int, int]] = []
    # Drain round-robin until all queues empty
    while any(q for q in queues):
        for q in queues:
            if q:
                result.append(q.pop(0))
    return result


def _build_r2ranked_ordering(
    cluster_df: pd.DataFrame,
    r2_df: pd.DataFrame,
    model_name: str,
) -> list[tuple[int, int]]:
    """Standard descending-R² ordering (same as 3P2-C reference)."""
    mdf = cluster_df[cluster_df["model"] == model_name].copy()
    mdf = mdf[mdf["is_high_si"] == True].copy()  # noqa: E712

    r2_map = {
        (int(row.layer), int(row.head)): float(row.mean_r2)
        for row in r2_df.itertuples()
    }
    heads = [(int(row.layer), int(row.head)) for row in mdf.itertuples()]
    heads.sort(key=lambda lh: r2_map.get(lh, 0.0), reverse=True)
    return heads


def _heads_for_fraction(ordering: list[Any], fraction_pct: int) -> list[Any]:
    if int(fraction_pct) <= 0:
        return []
    n_total = len(ordering)
    n_select = max(1, int(round((float(fraction_pct) / 100.0) * n_total)))
    return list(ordering[:n_select])


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def run_model(
    *,
    model_name: str,
    device: str,
    output_root: Path,
    fractions: tuple[int, ...],
    num_seeds: int,
    synthetic_count: int,
    batch_size_synth: int,
    ntp_count_per_seed: int,
    ntp_seq_len: int,
    batch_size_ntp: int,
    skip_ntp: bool,
    n_perm: int,
) -> dict[str, Any]:
    print(f"\n[E0] model={model_name} device={device}", flush=True)
    t0 = time.time()
    out_dir = output_root / model_name
    ensure_dir(out_dir)
    cache_dir = out_dir / "condition_cache"
    ensure_dir(cache_dir)

    # --- Load cluster assignments ---
    cluster_df = load_cluster_membership()
    r2_df = load_r2_summary(model_name)

    # --- Build three orderings ---
    seq_ordering = _build_sequential_ordering(cluster_df, r2_df, model_name)
    int_ordering = _build_interleaved_ordering(cluster_df, r2_df, model_name)
    r2_ordering = _build_r2ranked_ordering(cluster_df, r2_df, model_name)

    print(
        f"[E0] orderings built: sequential={len(seq_ordering)} "
        f"interleaved={len(int_ordering)} r2ranked={len(r2_ordering)}",
        flush=True,
    )

    orderings = {
        "sequential": [HeadID(l, h) for l, h in seq_ordering],
        "interleaved": [HeadID(l, h) for l, h in int_ordering],
        "r2ranked": [HeadID(l, h) for l, h in r2_ordering],
    }

    expected_cache_paths = [
        _condition_cache_path(cache_dir, ordering_name, int(frac))
        for ordering_name in orderings
        for frac in fractions
    ]
    all_rows: list[pd.DataFrame] = []
    if expected_cache_paths and all(p.exists() for p in expected_cache_paths):
        print(
            f"[E0][{model_name}] all {len(expected_cache_paths)} conditions restored from cache",
            flush=True,
        )
        all_rows = [pd.read_parquet(p) for p in expected_cache_paths]
    else:
        # --- Load model ---
        model_spec = MODELS[model_name]
        loaded = load_model(model_spec)
        model = loaded.model.to(device)
        model.eval()
        tokenizer = load_tokenizer(model_spec)

        retrieval_candidates = RETRIEVAL_SPANS.get(model_name, ())
        retrieval_span = 48 if 48 in retrieval_candidates else int(max(retrieval_candidates))

        vocab_size = int(tokenizer.vocab_size)
        special_ids = [
            int(x) for x in [
                getattr(tokenizer, attr, None)
                for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
            ]
            if x is not None
        ]
        pools = build_token_pools(model_name, vocab_size, special_ids)
        seeds = range(max(1, int(num_seeds)))

        task_configs, prebuilt_examples = _build_synthetic_cells(
            model_name=model_name,
            pools=pools,
            seeds=seeds,
            synthetic_count=max(1, int(synthetic_count)),
            retrieval_span=int(retrieval_span),
        )

        ntp_seed_chunks = ntp_coverage = None
        if not skip_ntp:
            ntp_seed_chunks, ntp_coverage = _prepare_ntp_seed_chunks(
                tokenizer=tokenizer,
                model_name=model_name,
                num_seeds=max(1, int(num_seeds)),
                ntp_count_per_seed=max(1, int(ntp_count_per_seed)),
                seq_len=max(64, int(ntp_seq_len)),
            )

        synth_batch_ceiling_cache: dict[tuple[str, int], int] = {}
        ntp_batch_state: dict[str, int] = {"wiki_ntp": max(1, int(batch_size_ntp))}

        # --- Evaluate each ordering at each fraction ---
        for ordering_name, ordering in orderings.items():
            n_heads_total = len(ordering)
            for frac in fractions:
                heads_to_ablate = _heads_for_fraction(ordering, int(frac))
                cond_name = f"{ordering_name}_f{int(frac):02d}"
                cache_path = _condition_cache_path(cache_dir, ordering_name, int(frac))
                cached = _load_condition_cache(cache_path)
                if cached is not None and len(cached) > 0:
                    all_rows.append(cached)
                    print(
                        f"[E0][{model_name}] ordering={ordering_name} frac={frac}% cache=hit",
                        flush=True,
                    )
                    continue
                print(
                    f"[E0][{model_name}] ordering={ordering_name} frac={frac}% "
                    f"n_heads={len(heads_to_ablate)}",
                    flush=True,
                )

                synth_rows = evaluate_task_battery(
                    model=model,
                    tokenizer=tokenizer,
                    model_spec=model_spec,
                    device=device,
                    heads_to_zero=heads_to_ablate,
                    condition_name=cond_name,
                    seeds=seeds,
                    retrieval_spans=(retrieval_span,),
                    pools=pools,
                    synthetic_count=max(1, int(synthetic_count)),
                    batch_size=max(1, int(batch_size_synth)),
                    task_configs=task_configs,
                    prebuilt_examples=prebuilt_examples,
                    batch_ceiling_cache=synth_batch_ceiling_cache,
                )
                synth_df = pd.DataFrame(synth_rows)
                synth_df["ordering_type"] = ordering_name
                synth_df["ablation_fraction"] = int(frac)
                synth_df["n_heads_ablated"] = int(len(heads_to_ablate))
                synth_df["n_heads_total"] = int(n_heads_total)
                synth_df["metric_name"] = "accuracy"
                synth_df["metric_value"] = synth_df["accuracy"].astype(float)
                pieces = [synth_df]

                if not skip_ntp:
                    ntp_rows = _evaluate_ntp_losses(
                        model=model,
                        model_name=model_name,
                        tokenizer=tokenizer,
                        device=device,
                        heads_to_zero=heads_to_ablate,
                        num_seeds=max(1, int(num_seeds)),
                        ntp_count_per_seed=max(1, int(ntp_count_per_seed)),
                        seq_len=max(64, int(ntp_seq_len)),
                        batch_size=max(1, int(batch_size_ntp)),
                        seed_chunks=ntp_seed_chunks,
                        coverage_metadata=ntp_coverage,
                        batch_state=ntp_batch_state,
                    )
                    ntp_df = pd.DataFrame(ntp_rows)
                    ntp_df["ordering_type"] = ordering_name
                    ntp_df["ablation_fraction"] = int(frac)
                    ntp_df["n_heads_ablated"] = int(len(heads_to_ablate))
                    ntp_df["n_heads_total"] = int(n_heads_total)
                    ntp_df["accuracy"] = np.nan
                    ntp_df["n_targets"] = np.nan
                    ntp_df["n_correct"] = np.nan
                    pieces.append(ntp_df)

                cond_df = pd.concat(pieces, ignore_index=True, sort=False)
                _write_condition_cache(cache_path, cond_df)
                all_rows.append(cond_df)

    full_df = pd.concat(all_rows, ignore_index=True, sort=False)
    full_df["model"] = model_name

    # --- Compute degradation relative to no-ablation baseline per ordering ---
    baseline = (
        full_df[full_df["ablation_fraction"] == 0]
        .groupby(["ordering_type", "task", "seed", "metric_name"], as_index=False)["metric_value"]
        .mean()
        .rename(columns={"metric_value": "baseline_metric_value"})
    )
    full_df = full_df.merge(
        baseline, on=["ordering_type", "task", "seed", "metric_name"], how="left"
    )
    is_acc = full_df["metric_name"] == "accuracy"
    is_loss = full_df["metric_name"] == "loss"
    full_df["degradation"] = np.nan
    full_df.loc[is_acc, "degradation"] = (
        full_df.loc[is_acc, "baseline_metric_value"] - full_df.loc[is_acc, "metric_value"]
    )
    full_df.loc[is_loss, "degradation"] = (
        full_df.loc[is_loss, "metric_value"] - full_df.loc[is_loss, "baseline_metric_value"]
    )

    full_df.to_parquet(out_dir / "ablation_curves_all_orderings.parquet", index=False)

    # --- Fit threshold model per ordering × task and extract tau ---
    fits_by_ordering: dict[str, dict[str, Any]] = {}
    tau_values: dict[str, list[float]] = {o: [] for o in orderings}

    for ordering_name in orderings:
        odf = full_df[full_df["ordering_type"] == ordering_name]
        fits: dict[str, Any] = {}
        for (task, metric_name), gdf in odf.groupby(["task", "metric_name"], sort=True):
            agg = (
                gdf.groupby("ablation_fraction", as_index=False)["degradation"]
                .mean()
                .sort_values("ablation_fraction")
            )
            x = agg["ablation_fraction"].values / 100.0  # normalize to [0,1]
            y = agg["degradation"].values
            fit = fit_three_models(x, y)
            key = f"{task}::{metric_name}"
            fits[key] = fit
            tau = safe_float(fit.get("threshold_piecewise", {}).get("tau", float("nan")))
            if np.isfinite(tau):
                tau_values[ordering_name].append(tau)
        fits_by_ordering[ordering_name] = fits

    write_json(out_dir / "curve_fits_by_ordering.json", fits_by_ordering)

    # --- Compute tau comparison statistics ---
    tau_seq = np.array(tau_values["sequential"], dtype=float)
    tau_int = np.array(tau_values["interleaved"], dtype=float)
    tau_r2 = np.array(tau_values["r2ranked"], dtype=float)

    diff_seq_vs_int, p_seq_lt_int = permutation_test_one_sided_less(
        tau_seq, tau_int, n_perm=int(n_perm), seed=stream_seed(PERM_ROOT_SEED, f"{model_name}:seq_vs_int")
    )
    diff_seq_vs_r2, p_seq_lt_r2 = permutation_test_one_sided_less(
        tau_seq, tau_r2, n_perm=int(n_perm), seed=stream_seed(PERM_ROOT_SEED, f"{model_name}:seq_vs_r2")
    )
    diff_int_vs_r2, p_int_lt_r2 = permutation_test_one_sided_less(
        tau_int, tau_r2, n_perm=int(n_perm), seed=stream_seed(PERM_ROOT_SEED, f"{model_name}:int_vs_r2")
    )

    mean_tau_seq = float(np.nanmean(tau_seq)) if tau_seq.size > 0 else float("nan")
    mean_tau_int = float(np.nanmean(tau_int)) if tau_int.size > 0 else float("nan")
    mean_tau_r2 = float(np.nanmean(tau_r2)) if tau_r2.size > 0 else float("nan")

    # BIC vote tallies per ordering
    bic_verdicts = {o: ordering_verdict(fits_by_ordering[o]) for o in orderings}

    tau_comparison = {
        "timestamp": timestamp_now(),
        "model": model_name,
        "mean_tau_sequential": safe_float(mean_tau_seq),
        "mean_tau_interleaved": safe_float(mean_tau_int),
        "mean_tau_r2ranked": safe_float(mean_tau_r2),
        "diff_seq_minus_int": safe_float(diff_seq_vs_int),
        "p_seq_lt_int_one_sided": safe_float(p_seq_lt_int),
        "diff_seq_minus_r2": safe_float(diff_seq_vs_r2),
        "p_seq_lt_r2_one_sided": safe_float(p_seq_lt_r2),
        "diff_int_minus_r2": safe_float(diff_int_vs_r2),
        "p_int_lt_r2_one_sided": safe_float(p_int_lt_r2),
        "n_tau_fits_sequential": int(tau_seq.size),
        "n_tau_fits_interleaved": int(tau_int.size),
        "n_tau_fits_r2ranked": int(tau_r2.size),
        "n_perm": int(n_perm),
        "bic_verdicts": bic_verdicts,
        "acceptance_criteria": {
            "cluster_structure_evidence": (
                "p_seq_lt_int < 0.05 (Holm) in >= 2/3 models AND "
                "AUC_sequential >= 1.2 * AUC_interleaved"
            ),
            "redundancy_consistent": (
                "|mean_tau_seq - mean_tau_int| < 0.05 AND p_seq_lt_int > 0.20"
            ),
        },
        "rng_policy": {
            "root_seed": int(PERM_ROOT_SEED),
            "stream_ids": ["seq_vs_int", "seq_vs_r2", "int_vs_r2"],
            "stream_seed_derivation": "stream_seed(root_seed, f'{model}:{stream_id}')",
        },
    }

    min_tau = min(int(tau_seq.size), int(tau_int.size), int(tau_r2.size))
    if min_tau < int(MIN_TAU_FITS_PER_ORDERING):
        raise RuntimeError(
            f"[E0] hard_fail_reason: tau_fit_coverage_too_low for {model_name}: "
            f"seq={tau_seq.size}, int={tau_int.size}, r2={tau_r2.size}, "
            f"required_min={MIN_TAU_FITS_PER_ORDERING}"
        )

    # Preliminary verdict for this model
    if np.isfinite(p_seq_lt_int) and float(p_seq_lt_int) < 0.05 and np.isfinite(diff_seq_vs_int) and float(diff_seq_vs_int) < -0.05:
        model_verdict = "cluster_structure_consistent"
    elif np.isfinite(p_seq_lt_int) and float(p_seq_lt_int) > 0.20 and abs(safe_float(diff_seq_vs_int)) < 0.05:
        model_verdict = "redundancy_consistent"
    else:
        model_verdict = "inconclusive"

    tau_comparison["model_verdict"] = model_verdict
    write_json(out_dir / "tau_comparison.json", tau_comparison)

    runtime = float(time.time() - t0)
    print(f"[E0][{model_name}] done in {runtime:.1f}s  verdict={model_verdict}", flush=True)

    return {
        "model": model_name,
        "tau_comparison": tau_comparison,
        "runtime_sec": runtime,
        "bic_verdicts": bic_verdicts,
    }


# ---------------------------------------------------------------------------
# Cross-model aggregation
# ---------------------------------------------------------------------------

def aggregate_results(per_model: dict[str, Any], output_root: Path) -> dict[str, Any]:
    raw_pvals = {
        m: safe_float(r.get("tau_comparison", {}).get("p_seq_lt_int_one_sided", float("nan")))
        for m, r in per_model.items()
    }
    finite_raw = {k: v for k, v in raw_pvals.items() if np.isfinite(v)}
    holm = holm_adjust_dict(finite_raw) if finite_raw else {}

    for m, adj_p in holm.items():
        per_model[m]["tau_comparison"]["p_seq_lt_int_holm"] = safe_float(adj_p)
        tau_path = output_root / m / "tau_comparison.json"
        if tau_path.exists():
            payload = read_json(tau_path)
            payload["p_seq_lt_int_holm"] = safe_float(adj_p)
            write_json(tau_path, payload)

    verdicts = {m: r.get("tau_comparison", {}).get("model_verdict", "unknown")
                for m, r in per_model.items()}
    for m in verdicts:
        raw_p = safe_float(per_model[m].get("tau_comparison", {}).get("p_seq_lt_int_one_sided", float("nan")))
        adj_p = safe_float(per_model[m].get("tau_comparison", {}).get("p_seq_lt_int_holm", float("nan")))
        diff = safe_float(per_model[m].get("tau_comparison", {}).get("diff_seq_minus_int", float("nan")))
        if np.isfinite(adj_p) and np.isfinite(diff):
            if float(adj_p) < 0.05 and float(diff) < -0.05:
                verdicts[m] = "cluster_structure_consistent"
            elif float(adj_p) > 0.20 and abs(float(diff)) < 0.05:
                verdicts[m] = "redundancy_consistent"
            else:
                verdicts[m] = "inconclusive"
    n_redundancy = sum(1 for v in verdicts.values() if v == "redundancy_consistent")
    n_cluster = sum(1 for v in verdicts.values() if v == "cluster_structure_consistent")
    n_inconclusive = sum(1 for v in verdicts.values() if v == "inconclusive")

    if n_redundancy >= 2:
        overall = "redundancy_consistent"
    elif n_cluster >= 2:
        overall = "cluster_structure_consistent"
    else:
        overall = "inconclusive"

    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "per_model_verdicts": verdicts,
        "p_seq_lt_int_raw": raw_pvals,
        "p_seq_lt_int_holm": holm,
        "n_redundancy_consistent": n_redundancy,
        "n_cluster_structure_consistent": n_cluster,
        "n_inconclusive": n_inconclusive,
        "overall_verdict": overall,
    }
    write_json(output_root / "cross_model_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Artifact emission
# ---------------------------------------------------------------------------

def _emit_artifacts(output_root: Path, models: list[str], summary: dict[str, Any]) -> None:
    preregistration = {
        "experiment_id": EXPERIMENT_ID,
        "question": (
            "Does exhausting SI heads cluster-by-cluster (sequential ordering) produce "
            "earlier threshold-style collapse than interleaving removals across clusters?"
        ),
        "primary_hypothesis": (
            "If genuine distributed redundancy drives the threshold, sequential and "
            "interleaved orderings produce statistically indistinguishable collapse fractions. "
            "If cluster-boundary depletion drives the threshold, sequential ordering collapses "
            "significantly earlier (lower tau)."
        ),
        "primary_endpoints": [
            "p_seq_lt_int > 0.20 in all three models (redundancy consistent)",
            "|mean_tau_seq - mean_tau_int| < 0.05 in all three models",
        ],
        "secondary_endpoints": [
            "BIC vote tally per ordering type matches R12 r2-ranked tally",
            "AUC under degradation curve indistinguishable across sequential/interleaved",
        ],
        "model_list": list(models),
        "dataset_sources": [
            "experiment3/theory1_si_circuits (R² rankings)",
            "reinforce_exp2/B1_kernel_taxonomy (cluster assignments)",
            "Wikipedia held-out sequences for NTP loss",
        ],
        "inclusion_exclusion_rules": [
            "Include all high-SI heads (top quartile by R²) per model",
            "Use cluster_descriptor_kmeans column from B1 cluster_membership.parquet",
            "Exclude models where B1 cluster_membership.parquet is unavailable",
        ],
        "sample_size_plan": {
            "fractions": list(FRACTIONS),
            "num_seeds": 5,
            "n_perm_tau_test": 10000,
        },
        "seed_plan": {"seed_base": 0, "n_perm": 10000},
        "stopping_rule": "Run all three orderings at all fractions; no early stopping.",
        "multiplicity_family": ["Holm-Bonferroni across three models for tau comparison"],
        "acceptance_criteria": [
            "Redundancy consistent: p_seq_lt_int > 0.20 AND |delta_tau| < 0.05 in all models",
            "Cluster structure: p_seq_lt_int < 0.05 (Holm) in >= 2/3 models AND AUC ratio >= 1.2",
        ],
        "fallback_interpretation_if_null": (
            "If inconclusive: report that the experiment could not distinguish redundancy "
            "from cluster-boundary depletion at current power. Retain Result III as "
            "'threshold-style capacity collapse' without mechanistic attribution."
        ),
    }

    overall_verdict = summary.get("overall_verdict", "inconclusive")
    claim_status = (
        "supported" if overall_verdict == "redundancy_consistent"
        else "not_supported" if overall_verdict == "cluster_structure_consistent"
        else "inconclusive"
    )

    emit_core_artifacts(
        experiment_id=EXPERIMENT_ID,
        out_dir=output_root,
        preregistration=preregistration,
        manifest=command_manifest(
            experiment_id=EXPERIMENT_ID,
            command="cluster_sequential_vs_interleaved",
            model="+".join(models),
            analysis_tier="confirmatory",
            canonical_eligible=True,
            override_used=False,
            extras={
                "fractions": list(FRACTIONS),
                "upstream_b1": str(B1_RESULTS / "cluster_membership.parquet"),
            },
        ),
        summary={
            "timestamp": timestamp_now(),
            "experiment_id": EXPERIMENT_ID,
            "analysis_tier": "confirmatory",
            "canonical_eligible": True,
            "override_used": False,
            "verdict": summary,
            "limitations": [
                "Cluster imbalance (Llama cluster 1 = 77% of high-SI heads) reduces "
                "discriminative power for that model.",
                "Only 3 cluster types per model; within-cluster variance uncharacterized.",
                "Permutation test power depends on n_tau_fits; underpowered if few "
                "task × metric combinations pass threshold-piecewise fit.",
            ],
        },
        claim_impact={
            "timestamp": timestamp_now(),
            "experiment_id": EXPERIMENT_ID,
            "claim_status": claim_status,
            "supports_main_text": overall_verdict == "redundancy_consistent",
            "strict_only": True,
            "notes": [
                f"Overall verdict: {overall_verdict}",
                "redundancy_consistent → add ordering-invariance sentence to Result III.",
                "cluster_structure_consistent → demote Result III claim; report mechanism ambiguity.",
                "inconclusive → retain current language; note B3 and E0 both inconclusive.",
            ],
        },
        data_dictionary={
            "experiment_id": EXPERIMENT_ID,
            "tables": [
                {
                    "path": "<model>/ablation_curves_all_orderings.parquet",
                    "description": "Full degradation curves for all three orderings, all fractions, all seeds and tasks.",
                    "columns": [
                        {"name": "ordering_type", "dtype": "str", "description": "sequential | interleaved | r2ranked"},
                        {"name": "ablation_fraction", "dtype": "int", "description": "Percentage of high-SI heads ablated (0-50)"},
                        {"name": "task", "dtype": "str", "description": "Task name (local_key_match, long_range_retrieval, wiki_ntp)"},
                        {"name": "metric_name", "dtype": "str", "description": "accuracy or loss"},
                        {"name": "metric_value", "dtype": "float", "description": "Raw metric value at this fraction"},
                        {"name": "degradation", "dtype": "float", "description": "Metric change relative to 0% ablation baseline"},
                        {"name": "seed", "dtype": "int", "description": "Evaluation seed"},
                        {"name": "n_heads_ablated", "dtype": "int", "description": "Number of heads zeroed in this condition"},
                        {"name": "model", "dtype": "str", "description": "Model name"},
                    ],
                },
                {
                    "path": "<model>/tau_comparison.json",
                    "description": "Per-model tau statistics and permutation test results.",
                    "columns": [],
                },
            ],
        },
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="E0: Cluster-sequential vs interleaved ablation",
        allow_abbrev=False,
    )
    p.add_argument("--models", default=",".join(PRIMARY_MODELS))
    p.add_argument(
        "--device-map",
        default="llama-3.1-8b:cuda:0,olmo-2-7b:cuda:0,mistral-7b-v0.1:cuda:0",
    )
    p.add_argument("--output-root", default=str(DEFAULT_OUT))
    p.add_argument("--fractions", default=",".join(str(f) for f in FRACTIONS))
    p.add_argument("--num-seeds", type=int, default=5)
    p.add_argument("--synthetic-count", type=int, default=100)
    p.add_argument("--batch-size-synth", type=int, default=8)
    p.add_argument("--ntp-count-per-seed", type=int, default=100)
    p.add_argument("--ntp-seq-len", type=int, default=512)
    p.add_argument("--batch-size-ntp", type=int, default=4)
    p.add_argument("--skip-ntp", action="store_true")
    p.add_argument("--n-perm", type=int, default=10000,
                   help="Number of permutations for tau comparison test")
    p.add_argument("--no-finalize", action="store_true",
                   help="Skip cross-model aggregation; write per-model results only")
    p.add_argument("--finalize-only", action="store_true",
                   help="Read existing per-model results and emit cross-model artifacts only")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    ensure_dir(output_root)

    models = parse_models_arg(args.models)

    if args.finalize_only:
        print(f"[E0] Finalize-only mode: reading per-model results for {models}", flush=True)
        per_model_results: dict[str, Any] = {}
        for model_name in models:
            tau_path = output_root / model_name / "tau_comparison.json"
            if not tau_path.exists():
                raise FileNotFoundError(f"[E0] finalize-only: missing {tau_path}")
            per_model_results[model_name] = {"tau_comparison": read_json(tau_path)}
        summary = aggregate_results(per_model_results, output_root)
        _emit_artifacts(output_root, models, summary)
        print(f"[E0] Finalize complete. Overall verdict: {summary['overall_verdict']}", flush=True)
        return

    device_map = parse_device_map(args.device_map)
    fractions_pct = tuple(
        int(x) for x in args.fractions.split(",") if x.strip().isdigit()
    )

    per_model_results = {}
    for model_name in models:
        device = device_map.get(model_name, "cuda:0")
        result = run_model(
            model_name=model_name,
            device=device,
            output_root=output_root,
            fractions=fractions_pct,
            num_seeds=max(1, args.num_seeds),
            synthetic_count=max(1, args.synthetic_count),
            batch_size_synth=max(1, args.batch_size_synth),
            ntp_count_per_seed=max(1, args.ntp_count_per_seed),
            ntp_seq_len=max(64, args.ntp_seq_len),
            batch_size_ntp=max(1, args.batch_size_ntp),
            skip_ntp=bool(args.skip_ntp),
            n_perm=max(100, args.n_perm),
        )
        per_model_results[model_name] = result

    if args.no_finalize:
        print(f"[E0] Shard complete (no-finalize). Per-model results written.", flush=True)
        return

    summary = aggregate_results(per_model_results, output_root)
    _emit_artifacts(output_root, models, summary)

    print(f"\n[E0] Complete. Overall verdict: {summary['overall_verdict']}", flush=True)
    print(f"[E0] Results: {output_root}", flush=True)


if __name__ == "__main__":
    main()
