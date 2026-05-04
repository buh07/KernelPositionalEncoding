#!/usr/bin/env python3
"""E2 — Random-Order Sensitivity Beyond R12 Grid.

Two variations on NEW-R12:
  Variation 1 (fine-grid): Re-run existing 10 random orderings from R12 with a
    denser ablation fraction grid near the model-specific threshold (adds 6 evaluation
    points between 8% and 22%). Tests whether Llama's 7/10 partial-replication result
    is an artefact of the coarse original grid.

  Variation 2 (batch-size): Run 5 new random orderings for OLMo (strongest R12 result)
    with batch-size 5 (remove 5 heads per step instead of 1). Tests whether the
    threshold preference is sensitive to removal granularity.

Usage:
    python reinforce_exp3/scripts/run_e2_ordering_sensitivity.py \\
        --models llama-3.1-8b,olmo-2-7b,mistral-7b-v0.1 \\
        --device-map "llama-3.1-8b:cuda:0,olmo-2-7b:cuda:1,mistral-7b-v0.1:cuda:2"
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp3.common import (  # noqa: E402
    PRIMARY_MODELS,
    R12_RESULTS,
    RESULTS_ROOT,
    command_manifest,
    ensure_dir,
    read_json,
    safe_float,
    timestamp_now,
    write_json,
)
from reinforce_exp3.scripts._shared import (  # noqa: E402
    enforce_coverage_contract,
    emit_core_artifacts,
    fit_three_models,
    ordering_verdict,
    parse_device_map,
    parse_models_arg,
)

from experiment2.tasks import build_token_pools  # noqa: E402
from experiment3.phase2.exp3p2c_redundancy_quantification import (  # noqa: E402
    _build_synthetic_cells,
    _evaluate_ntp_losses,
    _load_ranked_heads,
    _prepare_ntp_seed_chunks,
)
from experiment3.theory1_si_circuits import MODELS, RETRIEVAL_SPANS, evaluate_task_battery  # noqa: E402
from shared.models.loading import load_model, load_tokenizer  # noqa: E402

DEFAULT_OUT = RESULTS_ROOT / "E2_ordering_sensitivity"
EXPERIMENT_ID = "E2"

# Fine-grained grid: adds points between 8% and 22% to bracket the typical threshold
FINE_FRACTIONS = (0, 1, 2, 5, 8, 10, 12, 15, 18, 20, 22, 25, 30, 40, 50)
ORIGINAL_FRACTIONS = (0, 1, 2, 5, 10, 15, 20, 25, 50)

# Batch-size variation (OLMo only, 5 new orderings)
BATCH5_MODELS = ("olmo-2-7b",)
N_BATCH5_ORDERINGS = 5
BATCH5_SEED_BASE = 20260501  # distinct from R12 seed base (20260417)
EXPECTED_R12_ORDERINGS = 10


def _effective_shard_bounds(
    start: int | None,
    stop: int | None,
    total: int,
) -> tuple[int, int]:
    lo = 0 if start is None else max(0, int(start))
    hi = int(total) if stop is None else min(int(total), int(stop))
    if hi < lo:
        raise RuntimeError(
            f"[E2] hard_fail_reason: invalid shard bounds start={lo} stop={hi} total={total}"
        )
    return lo, hi


def _is_in_shard(ordering_id: int, start: int | None, stop: int | None, total: int) -> bool:
    lo, hi = _effective_shard_bounds(start, stop, total)
    return lo <= int(ordering_id) < hi


def _condition_cache_path(cache_dir: Path, condition_name: str) -> Path:
    return cache_dir / f"{condition_name}.parquet"


def _load_condition_cache(cache_path: Path) -> pd.DataFrame | None:
    if not cache_path.exists():
        return None
    return pd.read_parquet(cache_path)


def _write_condition_cache(cache_path: Path, df: pd.DataFrame) -> None:
    df.to_parquet(cache_path, index=False)


def _load_r12_orderings(model_name: str) -> list[dict[str, Any]] | None:
    """Load ordering permutation indices from existing R12 results."""
    p = R12_RESULTS / model_name / "ordering_payloads.json"
    if not p.exists():
        return None
    data = read_json(p)
    return data.get("orderings", [])


def _heads_for_fraction_batch(
    ordering: list[Any], fraction_pct: int, batch_size: int = 1
) -> list[Any]:
    """Select heads up to `fraction_pct`% of ordering, rounded to nearest batch_size."""
    if int(fraction_pct) <= 0:
        return []
    n_total = len(ordering)
    n_target = max(1, int(round((float(fraction_pct) / 100.0) * n_total)))
    # Round up to nearest batch_size
    n_select = int(np.ceil(n_target / max(1, batch_size))) * max(1, batch_size)
    n_select = min(n_select, n_total)
    return list(ordering[:n_select])


def _run_variation1_finegrid(
    *,
    model_name: str,
    device: str,
    output_root: Path,
    num_seeds: int,
    synthetic_count: int,
    batch_size_synth: int,
    ntp_count_per_seed: int,
    ntp_seq_len: int,
    batch_size_ntp: int,
    skip_ntp: bool,
    ordering_id_start: int | None,
    ordering_id_stop: int | None,
) -> dict[str, Any]:
    """Re-run R12 orderings with fine-grid fractions."""
    print(f"\n[E2-V1] model={model_name} fine-grid variation", flush=True)
    out_dir = output_root / model_name / "finegrid"
    ensure_dir(out_dir)
    cache_dir = out_dir / "condition_cache"
    ensure_dir(cache_dir)

    r12_orderings = _load_r12_orderings(model_name)
    if not r12_orderings:
        raise RuntimeError(f"[E2-V1] hard_fail_reason: missing R12 ordering payloads for {model_name}")
    if len(r12_orderings) != EXPECTED_R12_ORDERINGS:
        raise RuntimeError(
            f"[E2-V1] hard_fail_reason: expected {EXPECTED_R12_ORDERINGS} orderings for {model_name}, "
            f"found {len(r12_orderings)}"
        )

    selected_orderings: list[dict[str, Any]] = []
    for ord_payload in r12_orderings:
        ordering_id = int(ord_payload["ordering_id"])
        if _is_in_shard(
            ordering_id=ordering_id,
            start=ordering_id_start,
            stop=ordering_id_stop,
            total=EXPECTED_R12_ORDERINGS,
        ):
            selected_orderings.append(ord_payload)
    if not selected_orderings:
        raise RuntimeError(
            f"[E2-V1] hard_fail_reason: empty ordering shard for {model_name} "
            f"with bounds start={ordering_id_start} stop={ordering_id_stop}"
        )

    new_fractions = [f for f in FINE_FRACTIONS if f not in ORIGINAL_FRACTIONS]
    expected_cache_paths = [
        _condition_cache_path(
            cache_dir,
            f"finegrid_o{int(ord_payload['ordering_id']):02d}_f{int(frac):02d}",
        )
        for ord_payload in selected_orderings
        for frac in new_fractions
    ]
    all_rows: list[pd.DataFrame] = []
    if expected_cache_paths and all(p.exists() for p in expected_cache_paths):
        print(
            f"[E2-V1][{model_name}] all {len(expected_cache_paths)} conditions restored from cache",
            flush=True,
        )
        all_rows = [pd.read_parquet(p) for p in expected_cache_paths]
    else:
        ranked_heads, _ = _load_ranked_heads(model_name)

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

        synth_batch_cache: dict[tuple[str, int], int] = {}
        ntp_batch_state: dict[str, int] = {"wiki_ntp": max(1, int(batch_size_ntp))}

        for ord_payload in selected_orderings:
            ordering_id = int(ord_payload["ordering_id"])
            perm_idx = [int(x) for x in ord_payload["head_index_order"]]
            ordering = [ranked_heads[i] for i in perm_idx]

            # Only evaluate fractions that are NEW (not in original grid)

            for frac in new_fractions:
                heads = _heads_for_fraction_batch(ordering, int(frac), batch_size=1)
                cond_name = f"finegrid_o{ordering_id:02d}_f{int(frac):02d}"
                cache_path = _condition_cache_path(cache_dir, cond_name)
                cached = _load_condition_cache(cache_path)
                if cached is not None and len(cached) > 0:
                    all_rows.append(cached)
                    print(
                        f"[E2-V1][{model_name}] ordering={ordering_id} frac={frac}% cache=hit",
                        flush=True,
                    )
                    continue
                print(
                    f"[E2-V1][{model_name}] ordering={ordering_id} frac={frac}% heads={len(heads)}",
                    flush=True,
                )

                synth_rows = evaluate_task_battery(
                    model=model,
                    tokenizer=tokenizer,
                    model_spec=model_spec,
                    device=device,
                    heads_to_zero=heads,
                    condition_name=cond_name,
                    seeds=seeds,
                    retrieval_spans=(retrieval_span,),
                    pools=pools,
                    synthetic_count=max(1, int(synthetic_count)),
                    batch_size=max(1, int(batch_size_synth)),
                    task_configs=task_configs,
                    prebuilt_examples=prebuilt_examples,
                    batch_ceiling_cache=synth_batch_cache,
                )
                synth_df = pd.DataFrame(synth_rows)
                synth_df["ordering_id"] = int(ordering_id)
                synth_df["ablation_fraction"] = int(frac)
                synth_df["n_heads_ablated"] = int(len(heads))
                synth_df["variation"] = "finegrid_new_points"
                synth_df["metric_name"] = "accuracy"
                synth_df["metric_value"] = synth_df["accuracy"].astype(float)
                pieces = [synth_df]

                if not skip_ntp:
                    ntp_rows = _evaluate_ntp_losses(
                        model=model,
                        model_name=model_name,
                        tokenizer=tokenizer,
                        device=device,
                        heads_to_zero=heads,
                        num_seeds=max(1, int(num_seeds)),
                        ntp_count_per_seed=max(1, int(ntp_count_per_seed)),
                        seq_len=max(64, int(ntp_seq_len)),
                        batch_size=max(1, int(batch_size_ntp)),
                        seed_chunks=ntp_seed_chunks,
                        coverage_metadata=ntp_coverage,
                        batch_state=ntp_batch_state,
                    )
                    ntp_df = pd.DataFrame(ntp_rows)
                    ntp_df["ordering_id"] = int(ordering_id)
                    ntp_df["ablation_fraction"] = int(frac)
                    ntp_df["n_heads_ablated"] = int(len(heads))
                    ntp_df["variation"] = "finegrid_new_points"
                    ntp_df["accuracy"] = np.nan
                    pieces.append(ntp_df)

                cond_df = pd.concat(pieces, ignore_index=True, sort=False)
                _write_condition_cache(cache_path, cond_df)
                all_rows.append(cond_df)

    if not all_rows:
        raise RuntimeError(f"[E2-V1] hard_fail_reason: no finegrid rows generated for {model_name}")

    new_df = pd.concat(all_rows, ignore_index=True, sort=False)
    new_df["model"] = model_name
    shard_lo, shard_hi = _effective_shard_bounds(
        ordering_id_start, ordering_id_stop, EXPECTED_R12_ORDERINGS
    )
    shard_tag = f"o{shard_lo:02d}_{shard_hi:02d}"
    new_df.to_parquet(out_dir / f"finegrid_new_points_{shard_tag}.parquet", index=False)

    # --- Merge with existing R12 curve data and refit ---
    r12_curve_path = R12_RESULTS / model_name / "cumulative_ablation_curve_random.parquet"
    ordering_votes: list[dict[str, Any]] = []

    if r12_curve_path.exists():
        r12_df = pd.read_parquet(r12_curve_path)
        # Standardise column name
        if "metric_value" not in r12_df.columns and "degradation" in r12_df.columns:
            r12_df["metric_value"] = np.nan
        merged = pd.concat([r12_df, new_df], ignore_index=True, sort=False)
    else:
        print(f"[E2-V1] R12 curve parquet not found for {model_name}; fitting new points only.",
              flush=True)
        merged = new_df

    # Refit per ordering × task (only for this shard's ordering IDs)
    selected_ordering_ids = {int(p["ordering_id"]) for p in selected_orderings}
    for ordering_id, odf in merged.groupby("ordering_id", sort=True):
        if int(ordering_id) not in selected_ordering_ids:
            continue
        fits: dict[str, Any] = {}
        for (task, metric_name), gdf in odf.groupby(["task", "metric_name"], sort=True):
            agg = (
                gdf.groupby("ablation_fraction", as_index=False)["degradation"]
                .mean()
                .sort_values("ablation_fraction")
                .dropna(subset=["degradation"])
            )
            if len(agg) < 4:
                continue
            x = agg["ablation_fraction"].values / 100.0
            y = agg["degradation"].values
            key = f"{task}::{metric_name}"
            fits[key] = fit_three_models(x, y)
        if fits:
            verdict = ordering_verdict(fits)
            ordering_votes.append({
                "ordering_id": int(ordering_id),
                "grid": "fine",
                **verdict,
            })

    n_majority = sum(1 for r in ordering_votes if bool(r.get("threshold_majority", False)))
    n_total = len(ordering_votes)
    frac_majority = float(n_majority / max(1, n_total))

    result = {
        "model": model_name,
        "variation": "finegrid",
        "n_orderings_global_expected": int(len(r12_orderings)),
        "ordering_id_range": {"start_inclusive": shard_lo, "stop_exclusive": shard_hi},
        "n_orderings_expected": int(len(selected_orderings)),
        "n_orderings_refit": n_total,
        "n_threshold_majority": n_majority,
        "fraction_threshold_majority": frac_majority,
        "robustness_assessment": (
            "strong_replication" if frac_majority >= 0.8
            else "partial_replication" if frac_majority >= 0.5
            else "failure"
        ),
        "ordering_votes": ordering_votes,
        "ordering_seed_provenance": "from reinforce_exp/exp_new_r12_ordering_control ordering_payloads.json",
    }
    if int(n_total) != int(len(selected_orderings)):
        raise RuntimeError(
            f"[E2-V1] hard_fail_reason: refit ordering coverage mismatch for {model_name}: "
            f"refit={n_total}, expected={len(selected_orderings)}"
        )
    write_json(out_dir / f"finegrid_bic_votes_{shard_tag}.json", result)
    return result


def _run_variation2_batchsize(
    *,
    model_name: str,
    device: str,
    output_root: Path,
    num_seeds: int,
    synthetic_count: int,
    batch_size_synth: int,
    ntp_count_per_seed: int,
    ntp_seq_len: int,
    batch_size_ntp: int,
    skip_ntp: bool,
    ordering_id_start: int | None,
    ordering_id_stop: int | None,
) -> dict[str, Any]:
    """Run 5 new random orderings with batch_size=5 (remove 5 heads per ablation step)."""
    print(f"\n[E2-V2] model={model_name} batch-size-5 variation", flush=True)
    out_dir = output_root / model_name / "batch5"
    ensure_dir(out_dir)
    cache_dir = out_dir / "condition_cache"
    ensure_dir(cache_dir)

    selected_ordering_ids = [
        oid for oid in range(N_BATCH5_ORDERINGS)
        if _is_in_shard(
            ordering_id=oid,
            start=ordering_id_start,
            stop=ordering_id_stop,
            total=N_BATCH5_ORDERINGS,
        )
    ]
    if not selected_ordering_ids:
        raise RuntimeError(
            f"[E2-V2] hard_fail_reason: empty ordering shard for {model_name} "
            f"with bounds start={ordering_id_start} stop={ordering_id_stop}"
        )

    expected_cache_paths = [
        _condition_cache_path(cache_dir, f"batch5_o{int(oid):02d}_f{int(frac):02d}")
        for oid in selected_ordering_ids
        for frac in FINE_FRACTIONS
    ]
    all_rows: list[pd.DataFrame] = []
    ordering_seed_map: dict[int, int] = {}
    if expected_cache_paths and all(p.exists() for p in expected_cache_paths):
        print(
            f"[E2-V2][{model_name}] all {len(expected_cache_paths)} conditions restored from cache",
            flush=True,
        )
        all_rows = [pd.read_parquet(p) for p in expected_cache_paths]
        for oid in selected_ordering_ids:
            ordering_seed_map[int(oid)] = int(BATCH5_SEED_BASE + int(oid) * 1013)
    else:
        ranked_heads, _ = _load_ranked_heads(model_name)

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

        synth_batch_cache: dict[tuple[str, int], int] = {}
        ntp_batch_state: dict[str, int] = {"wiki_ntp": max(1, int(batch_size_ntp))}
        n_heads = len(ranked_heads)

        for local_id in selected_ordering_ids:
            perm_seed = BATCH5_SEED_BASE + local_id * 1013
            ordering_seed_map[int(local_id)] = int(perm_seed)
            rng = np.random.default_rng(perm_seed)
            perm_idx = rng.permutation(n_heads).tolist()
            ordering = [ranked_heads[i] for i in perm_idx]

            for frac in FINE_FRACTIONS:
                heads = _heads_for_fraction_batch(ordering, int(frac), batch_size=5)
                cond_name = f"batch5_o{local_id:02d}_f{int(frac):02d}"
                cache_path = _condition_cache_path(cache_dir, cond_name)
                cached = _load_condition_cache(cache_path)
                if cached is not None and len(cached) > 0:
                    all_rows.append(cached)
                    print(
                        f"[E2-V2][{model_name}] ordering={local_id} frac={frac}% cache=hit",
                        flush=True,
                    )
                    continue
                print(
                    f"[E2-V2][{model_name}] ordering={local_id} frac={frac}% heads={len(heads)}",
                    flush=True,
                )

                synth_rows = evaluate_task_battery(
                    model=model,
                    tokenizer=tokenizer,
                    model_spec=model_spec,
                    device=device,
                    heads_to_zero=heads,
                    condition_name=cond_name,
                    seeds=seeds,
                    retrieval_spans=(retrieval_span,),
                    pools=pools,
                    synthetic_count=max(1, int(synthetic_count)),
                    batch_size=max(1, int(batch_size_synth)),
                    task_configs=task_configs,
                    prebuilt_examples=prebuilt_examples,
                    batch_ceiling_cache=synth_batch_cache,
                )
                synth_df = pd.DataFrame(synth_rows)
                synth_df["ordering_id"] = int(local_id)
                synth_df["ablation_fraction"] = int(frac)
                synth_df["n_heads_ablated"] = int(len(heads))
                synth_df["variation"] = "batch5"
                synth_df["metric_name"] = "accuracy"
                synth_df["metric_value"] = synth_df["accuracy"].astype(float)
                pieces = [synth_df]

                if not skip_ntp:
                    ntp_rows = _evaluate_ntp_losses(
                        model=model,
                        model_name=model_name,
                        tokenizer=tokenizer,
                        device=device,
                        heads_to_zero=heads,
                        num_seeds=max(1, int(num_seeds)),
                        ntp_count_per_seed=max(1, int(ntp_count_per_seed)),
                        seq_len=max(64, int(ntp_seq_len)),
                        batch_size=max(1, int(batch_size_ntp)),
                        seed_chunks=ntp_seed_chunks,
                        coverage_metadata=ntp_coverage,
                        batch_state=ntp_batch_state,
                    )
                    ntp_df = pd.DataFrame(ntp_rows)
                    ntp_df["ordering_id"] = int(local_id)
                    ntp_df["ablation_fraction"] = int(frac)
                    ntp_df["n_heads_ablated"] = int(len(heads))
                    ntp_df["variation"] = "batch5"
                    ntp_df["accuracy"] = np.nan
                    pieces.append(ntp_df)

                cond_df = pd.concat(pieces, ignore_index=True, sort=False)
                _write_condition_cache(cache_path, cond_df)
                all_rows.append(cond_df)

    full_df = pd.concat(all_rows, ignore_index=True, sort=False)
    full_df["model"] = model_name

    baseline = (
        full_df[full_df["ablation_fraction"] == 0]
        .groupby(["ordering_id", "task", "seed", "metric_name"], as_index=False)["metric_value"]
        .mean()
        .rename(columns={"metric_value": "baseline_metric_value"})
    )
    full_df = full_df.merge(
        baseline, on=["ordering_id", "task", "seed", "metric_name"], how="left"
    )
    full_df["degradation"] = np.nan
    is_acc = full_df["metric_name"] == "accuracy"
    is_loss = full_df["metric_name"] == "loss"
    full_df.loc[is_acc, "degradation"] = (
        full_df.loc[is_acc, "baseline_metric_value"] - full_df.loc[is_acc, "metric_value"]
    )
    full_df.loc[is_loss, "degradation"] = (
        full_df.loc[is_loss, "metric_value"] - full_df.loc[is_loss, "baseline_metric_value"]
    )
    shard_lo, shard_hi = _effective_shard_bounds(
        ordering_id_start, ordering_id_stop, N_BATCH5_ORDERINGS
    )
    shard_tag = f"o{shard_lo:02d}_{shard_hi:02d}"
    full_df.to_parquet(out_dir / f"batch5_curves_{shard_tag}.parquet", index=False)

    ordering_votes: list[dict[str, Any]] = []
    for ordering_id, odf in full_df.groupby("ordering_id", sort=True):
        fits: dict[str, Any] = {}
        for (task, metric_name), gdf in odf.groupby(["task", "metric_name"], sort=True):
            agg = (
                gdf.groupby("ablation_fraction", as_index=False)["degradation"]
                .mean()
                .sort_values("ablation_fraction")
                .dropna(subset=["degradation"])
            )
            if len(agg) < 4:
                continue
            x = agg["ablation_fraction"].values / 100.0
            y = agg["degradation"].values
            fits[f"{task}::{metric_name}"] = fit_three_models(x, y)
        if fits:
            verdict = ordering_verdict(fits)
            ordering_votes.append({"ordering_id": int(ordering_id), "batch_size": 5, **verdict})

    n_majority = sum(1 for r in ordering_votes if bool(r.get("threshold_majority", False)))
    n_total = len(ordering_votes)

    result = {
        "model": model_name,
        "variation": "batch5",
        "n_orderings_global_expected": int(N_BATCH5_ORDERINGS),
        "ordering_id_range": {"start_inclusive": shard_lo, "stop_exclusive": shard_hi},
        "n_orderings_expected": int(len(selected_ordering_ids)),
        "n_orderings": n_total,
        "n_threshold_majority": n_majority,
        "fraction_threshold_majority": float(n_majority / max(1, n_total)),
        "ordering_votes": ordering_votes,
        "ordering_seed_map": ordering_seed_map,
    }
    if int(n_total) != int(len(selected_ordering_ids)):
        raise RuntimeError(
            f"[E2-V2] hard_fail_reason: batch5 ordering coverage mismatch for {model_name}: "
            f"got={n_total}, expected={len(selected_ordering_ids)}"
        )
    write_json(out_dir / f"batch5_bic_votes_{shard_tag}.json", result)
    return result


def _merge_ordering_vote_shards(
    *,
    model_name: str,
    variation_dir: Path,
    base_name: str,
    total_expected: int,
    count_field: str,
) -> dict[str, Any]:
    canonical_path = variation_dir / f"{base_name}.json"
    if canonical_path.exists():
        payload = read_json(canonical_path)
        votes = payload.get("ordering_votes", [])
        if len(votes) == total_expected:
            return payload

    shard_paths = sorted(variation_dir.glob(f"{base_name}_o*.json"))
    if not shard_paths:
        raise FileNotFoundError(
            f"[E2] finalize-only: missing {base_name} shards for {model_name} in {variation_dir}"
        )

    merged_votes: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for sp in shard_paths:
        payload = read_json(sp)
        for vote in payload.get("ordering_votes", []):
            oid = int(vote["ordering_id"])
            if oid in seen_ids:
                raise RuntimeError(
                    f"[E2] hard_fail_reason: duplicate ordering_id={oid} while merging {base_name} for {model_name}"
                )
            seen_ids.add(oid)
            merged_votes.append(vote)

    if len(merged_votes) != int(total_expected):
        raise RuntimeError(
            f"[E2] hard_fail_reason: incomplete shard merge for {model_name}/{base_name}: "
            f"got={len(merged_votes)} expected={total_expected}"
        )

    merged_votes = sorted(merged_votes, key=lambda v: int(v["ordering_id"]))
    n_majority = int(sum(1 for v in merged_votes if bool(v.get("threshold_majority", False))))
    frac_majority = float(n_majority / max(1, len(merged_votes)))

    merged: dict[str, Any] = {
        "model": model_name,
        "ordering_votes": merged_votes,
        "n_threshold_majority": n_majority,
        "fraction_threshold_majority": frac_majority,
    }
    if count_field == "n_orderings_refit":
        merged.update(
            {
                "variation": "finegrid",
                "n_orderings_global_expected": int(total_expected),
                "n_orderings_expected": int(total_expected),
                "n_orderings_refit": int(len(merged_votes)),
                "robustness_assessment": (
                    "strong_replication" if frac_majority >= 0.8
                    else "partial_replication" if frac_majority >= 0.5
                    else "failure"
                ),
                "ordering_seed_provenance": (
                    "from reinforce_exp/exp_new_r12_ordering_control ordering_payloads.json"
                ),
            }
        )
    else:
        merged.update(
            {
                "variation": "batch5",
                "n_orderings_global_expected": int(total_expected),
                "n_orderings_expected": int(total_expected),
                "n_orderings": int(len(merged_votes)),
            }
        )
        # Preserve seed map if present across shards.
        ordering_seed_map: dict[str, int] = {}
        for sp in shard_paths:
            payload = read_json(sp)
            for k, v in payload.get("ordering_seed_map", {}).items():
                ordering_seed_map[str(k)] = int(v)
        if ordering_seed_map:
            merged["ordering_seed_map"] = ordering_seed_map

    write_json(canonical_path, merged)
    return merged


def _emit_artifacts(output_root: Path, models: list[str], all_results: dict[str, Any]) -> None:
    emit_core_artifacts(
        experiment_id=EXPERIMENT_ID,
        out_dir=output_root,
        preregistration={
            "experiment_id": EXPERIMENT_ID,
            "question": (
                "Is the 26/30 threshold-majority result from NEW-R12 sensitive to "
                "ablation fraction grid coarseness or removal batch size?"
            ),
            "primary_hypothesis": (
                "The threshold preference is not a discretization artefact: fine-grid "
                "re-fitting does not decrease threshold-majority counts, and batch-size-5 "
                "orderings maintain threshold-majority preference in OLMo."
            ),
            "primary_endpoints": [
                "Llama fine-grid threshold-majority count >= original count (7/10)",
                "OLMo batch-5 threshold majority in >= 4/5 orderings",
            ],
            "secondary_endpoints": [
                "Mistral fine-grid count does not decrease from 9/10",
            ],
            "model_list": list(models),
            "dataset_sources": [
                "reinforce_exp/exp_new_r12_ordering_control (existing R12 orderings)",
                "Wikipedia held-out sequences",
            ],
            "inclusion_exclusion_rules": [
                "Variation 1: use existing R12 ordering payloads; skip model if not found",
                "Variation 2: OLMo only; 5 fresh orderings with batch_size=5",
            ],
            "sample_size_plan": {"n_orderings_v1": 10, "n_orderings_v2": 5, "num_seeds": 3},
            "seed_plan": {"v2_seed_base": BATCH5_SEED_BASE},
            "stopping_rule": "Run all designated orderings; no early stopping.",
            "multiplicity_family": ["No multiplicity correction; exploratory sensitivity check"],
            "acceptance_criteria": [
                "Fine-grid Llama count >= 7/10",
                "Batch-5 OLMo count >= 4/5",
            ],
            "fallback_interpretation_if_null": (
                "If Llama fine-grid count decreases: grid coarseness partially explains "
                "partial-replication; report updated count. "
                "If OLMo batch-5 fails: note batch-size sensitivity as a limitation."
            ),
        },
        manifest=command_manifest(
            experiment_id=EXPERIMENT_ID,
            command="ordering_sensitivity",
            model="+".join(models),
            analysis_tier="exploratory",
            canonical_eligible=False,
            override_used=False,
            extras={
                "fine_fractions": list(FINE_FRACTIONS),
                "batch5_models": list(BATCH5_MODELS),
                "seed_provenance": {
                    "v1": "ordering payload IDs from R12",
                    "v2": f"BATCH5_SEED_BASE={BATCH5_SEED_BASE}, seed=base+ordering_id*1013",
                },
            },
        ),
        summary={
            "timestamp": timestamp_now(),
            "experiment_id": EXPERIMENT_ID,
            "analysis_tier": "exploratory",
            "canonical_eligible": False,
            "override_used": False,
            "verdict": all_results,
            "limitations": [
                "Variation 1 depends on R12 ordering payloads existing on disk.",
                "Batch-size sensitivity test is OLMo-only at n=5 orderings.",
            ],
        },
        claim_impact={
            "timestamp": timestamp_now(),
            "experiment_id": EXPERIMENT_ID,
            "claim_status": "pending",
            "supports_main_text": False,
            "strict_only": False,
            "notes": [
                "Update R12 table in paper if Llama fine-grid count changes.",
                "Report batch-size insensitivity as robustness note if OLMo batch-5 passes.",
            ],
        },
        data_dictionary={
            "experiment_id": EXPERIMENT_ID,
            "tables": [
                {
                    "path": "<model>/finegrid/finegrid_bic_votes.json",
                    "description": "BIC vote tallies after re-fitting with fine-grid fractions.",
                    "columns": [],
                },
                {
                    "path": "olmo-2-7b/batch5/batch5_curves.parquet",
                    "description": "Degradation curves for batch-size-5 orderings in OLMo.",
                    "columns": [
                        {"name": "ordering_id", "dtype": "int", "description": "Local ordering index (0-4)"},
                        {"name": "ablation_fraction", "dtype": "int", "description": "Percentage of heads ablated (batch-5 rounded)"},
                        {"name": "task", "dtype": "str", "description": "Task name"},
                        {"name": "degradation", "dtype": "float", "description": "Performance change vs baseline"},
                    ],
                },
            ],
        },
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="E2: Random-order sensitivity (fine-grid + batch-size variation)",
        allow_abbrev=False,
    )
    p.add_argument("--models", default=",".join(PRIMARY_MODELS))
    p.add_argument(
        "--device-map",
        default="llama-3.1-8b:cuda:0,olmo-2-7b:cuda:0,mistral-7b-v0.1:cuda:0",
    )
    p.add_argument("--output-root", default=str(DEFAULT_OUT))
    p.add_argument("--num-seeds", type=int, default=3)
    p.add_argument("--synthetic-count", type=int, default=100)
    p.add_argument("--batch-size-synth", type=int, default=8)
    p.add_argument("--ntp-count-per-seed", type=int, default=100)
    p.add_argument("--ntp-seq-len", type=int, default=512)
    p.add_argument("--batch-size-ntp", type=int, default=4)
    p.add_argument("--skip-ntp", action="store_true")
    p.add_argument("--skip-v1", action="store_true", help="Skip fine-grid variation")
    p.add_argument("--skip-v2", action="store_true", help="Skip batch-size variation")
    p.add_argument(
        "--ordering-id-start",
        type=int,
        default=None,
        help="Inclusive ordering-id lower bound for deterministic sharding (default: full range).",
    )
    p.add_argument(
        "--ordering-id-stop",
        type=int,
        default=None,
        help="Exclusive ordering-id upper bound for deterministic sharding (default: full range).",
    )
    p.add_argument("--no-finalize", action="store_true",
                   help="Skip cross-model summary; write per-model results only")
    p.add_argument("--finalize-only", action="store_true",
                   help="Read existing per-model results and emit cross-model artifacts only")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    ensure_dir(output_root)

    models = parse_models_arg(args.models)

    if args.finalize_only:
        print(f"[E2] Finalize-only mode: reading per-model results for {models}", flush=True)
        all_results: dict[str, Any] = {}
        for model_name in models:
            finegrid_dir = output_root / model_name / "finegrid"
            if finegrid_dir.exists():
                fg_payload = _merge_ordering_vote_shards(
                    model_name=model_name,
                    variation_dir=finegrid_dir,
                    base_name="finegrid_bic_votes",
                    total_expected=EXPECTED_R12_ORDERINGS,
                    count_field="n_orderings_refit",
                )
                all_results.setdefault(model_name, {})["finegrid"] = fg_payload

            batch5_dir = output_root / model_name / "batch5"
            if batch5_dir.exists():
                b5_payload = _merge_ordering_vote_shards(
                    model_name=model_name,
                    variation_dir=batch5_dir,
                    base_name="batch5_bic_votes",
                    total_expected=N_BATCH5_ORDERINGS,
                    count_field="n_orderings",
                )
                all_results.setdefault(model_name, {})["batch5"] = b5_payload

            if model_name not in all_results:
                raise FileNotFoundError(
                    f"[E2] finalize-only: no results found for {model_name} in {output_root}"
                )
        enforce_coverage_contract(
            experiment_id=EXPERIMENT_ID,
            observed_models=all_results.keys(),
            required_models=models,
        )
        write_json(output_root / "updated_r12_table.json", {
            "timestamp": timestamp_now(),
            "experiment_id": EXPERIMENT_ID,
            "per_model": all_results,
            "note": (
                "Compare n_threshold_majority values with original R12 results "
                "(Llama 7/10, Mistral 9/10, OLMo 10/10) to determine if fine-grid "
                "changes the aggregate 26/30 count."
            ),
        })
        _emit_artifacts(output_root, models, all_results)
        print(f"[E2] Finalize complete. Results: {output_root}", flush=True)
        return

    device_map = parse_device_map(args.device_map)

    all_results = {}

    common_kwargs = dict(
        num_seeds=max(1, args.num_seeds),
        synthetic_count=max(1, args.synthetic_count),
        batch_size_synth=max(1, args.batch_size_synth),
        ntp_count_per_seed=max(1, args.ntp_count_per_seed),
        ntp_seq_len=max(64, args.ntp_seq_len),
        batch_size_ntp=max(1, args.batch_size_ntp),
        skip_ntp=bool(args.skip_ntp),
    )

    if not args.skip_v1:
        for model_name in models:
            device = device_map.get(model_name, "cuda:0")
            r = _run_variation1_finegrid(
                model_name=model_name,
                device=device,
                output_root=output_root,
                ordering_id_start=args.ordering_id_start,
                ordering_id_stop=args.ordering_id_stop,
                **common_kwargs,
            )
            all_results.setdefault(model_name, {})["finegrid"] = r

    if not args.skip_v2:
        for model_name in BATCH5_MODELS:
            if model_name not in models:
                continue
            device = device_map.get(model_name, "cuda:0")
            r = _run_variation2_batchsize(
                model_name=model_name,
                device=device,
                output_root=output_root,
                ordering_id_start=args.ordering_id_start,
                ordering_id_stop=args.ordering_id_stop,
                **common_kwargs,
            )
            all_results.setdefault(model_name, {})["batch5"] = r

    if args.no_finalize:
        print(f"[E2] Shard complete (no-finalize). Per-model results written.", flush=True)
        return

    enforce_coverage_contract(
        experiment_id=EXPERIMENT_ID,
        observed_models=all_results.keys(),
        required_models=models,
    )

    write_json(output_root / "updated_r12_table.json", {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "per_model": all_results,
        "note": (
            "Compare n_threshold_majority values with original R12 results "
            "(Llama 7/10, Mistral 9/10, OLMo 10/10) to determine if fine-grid "
            "changes the aggregate 26/30 count."
        ),
    })

    _emit_artifacts(output_root, models, all_results)
    print(f"\n[E2] Complete. Results: {output_root}", flush=True)


if __name__ == "__main__":
    main()
