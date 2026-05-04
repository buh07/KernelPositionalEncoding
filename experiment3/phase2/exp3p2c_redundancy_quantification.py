#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import optimize

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment2.tasks import TaskExample, TokenPools, build_token_pools, generate_task_examples
from experiment3.theory1_si_circuits import (
    MODELS,
    RETRIEVAL_SPANS,
    HeadID,
    evaluate_task_battery,
    head_output_ablation,
    load_profile_sequences,
)
from shared.models.loading import load_model, load_tokenizer


DEFAULT_FRACTIONS = (0, 1, 2, 5, 10, 15, 20, 25, 50)
DEFAULT_SORT_ORDERS = ("high_to_low", "low_to_high")


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    vals: list[int] = []
    seen: set[int] = set()
    for part in str(raw).split(","):
        tok = part.strip()
        if not tok:
            continue
        value = int(tok)
        if value < 0 or value > 100:
            raise ValueError(f"Invalid ablation fraction {value}; expected 0..100.")
        if value in seen:
            continue
        vals.append(value)
        seen.add(value)
    if not vals:
        raise ValueError("Ablation fraction list is empty.")
    return tuple(vals)


def _load_ranked_heads(model_name: str) -> tuple[list[HeadID], pd.DataFrame]:
    root = Path("results/experiment3/theory1_si_circuits") / model_name
    summary_path = root / "head_r2_summary.parquet"
    r2_path = root / "per_sequence_r2.parquet"

    if summary_path.exists():
        summary = pd.read_parquet(summary_path).copy()
        if "mean_r2" not in summary.columns:
            # Backward compatibility if the column name drifts.
            if "r2" in summary.columns:
                summary = summary.rename(columns={"r2": "mean_r2"})
            else:
                raise RuntimeError(f"Expected mean_r2 column in {summary_path}.")
        summary = summary[["layer", "head", "mean_r2"]]
    elif r2_path.exists():
        r2_df = pd.read_parquet(r2_path)
        summary = (
            r2_df.groupby(["layer", "head"], as_index=False)["r2"]
            .mean()
            .rename(columns={"r2": "mean_r2"})
        )
    else:
        raise FileNotFoundError(
            f"Missing head ranking inputs for {model_name}: {summary_path} and {r2_path} not found."
        )

    summary = summary.sort_values("mean_r2", ascending=False).reset_index(drop=True)
    heads = [HeadID(int(r.layer), int(r.head)) for r in summary.itertuples()]
    return heads, summary


def _heads_for_fraction(
    ranked_heads_desc: list[HeadID],
    fraction_pct: int,
    sort_order: str,
) -> list[HeadID]:
    if fraction_pct <= 0:
        return []
    n_total = len(ranked_heads_desc)
    n_select = max(1, int(round((fraction_pct / 100.0) * n_total)))
    if sort_order == "high_to_low":
        return ranked_heads_desc[:n_select]
    if sort_order == "low_to_high":
        return ranked_heads_desc[-n_select:]
    raise ValueError(f"Unsupported sort_order={sort_order}")


def _build_synthetic_cells(
    *,
    model_name: str,
    pools: TokenPools,
    seeds: range,
    synthetic_count: int,
    retrieval_span: int,
) -> tuple[list[tuple[str, int | None, tuple[int, ...] | None]], dict[tuple[int, str, int], list[TaskExample]]]:
    task_configs = [
        ("long_range_retrieval", int(retrieval_span), (int(retrieval_span),)),
        ("local_key_match", None, None),
    ]
    prebuilt: dict[tuple[int, str, int], list[TaskExample]] = {}
    for seed in seeds:
        for task_name, span_override, span_choices in task_configs:
            span_val = span_override if span_override is not None else 0
            prebuilt[(seed, task_name, span_val)] = generate_task_examples(
                task_name=task_name,
                model_name=model_name,
                seq_len=512,
                seed=int(seed),
                count=int(synthetic_count),
                pools=pools,
                span_override=span_override,
                span_choices=span_choices,
            )
    return task_configs, prebuilt


def _prepare_ntp_seed_chunks(
    *,
    tokenizer,
    model_name: str,
    num_seeds: int,
    ntp_count_per_seed: int,
    seq_len: int,
) -> tuple[dict[int, list[list[int]]], dict[str, Any]]:
    total_needed = max(1, int(num_seeds)) * max(1, int(ntp_count_per_seed))
    sequences = load_profile_sequences(
        tokenizer=tokenizer,
        model_name=model_name,
        num_sequences=total_needed,
        seq_len=int(seq_len),
    )
    n_available = int(len(sequences))
    if n_available <= 0:
        raise RuntimeError(f"NTP requested {total_needed} sequences for {model_name}, found 0.")

    seq_chunks: dict[int, list[list[int]]] = {}
    coverage: dict[str, Any] = {
        "requested_total": int(total_needed),
        "available_total": int(n_available),
        "effective_per_seed": 0,
        "total_used": 0,
        "reuse_mode": "strict_split",
        "unique_sequences_used": 0,
        "duplication_factor": 1.0,
    }

    if n_available >= total_needed:
        coverage["effective_per_seed"] = int(ntp_count_per_seed)
        coverage["total_used"] = int(total_needed)
        coverage["unique_sequences_used"] = int(total_needed)
        trimmed = sequences[:total_needed]
        idx = 0
        for seed in range(num_seeds):
            seq_chunks[seed] = trimmed[idx : idx + int(ntp_count_per_seed)]
            idx += int(ntp_count_per_seed)
        return seq_chunks, coverage

    # Deterministic bootstrap-resample fallback: keep per-seed counts fixed to
    # preserve design parity across models/conditions while recording effective
    # corpus coverage explicitly in metadata.
    seed_material = (
        f"{model_name}|{num_seeds}|{ntp_count_per_seed}|{seq_len}|{n_available}|{total_needed}"
    )
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    sampled_idx = rng.integers(0, n_available, size=int(total_needed), endpoint=False)
    sampled = [sequences[int(i)] for i in sampled_idx.tolist()]
    unique_used = int(len(set(int(i) for i in sampled_idx.tolist())))

    coverage["effective_per_seed"] = int(ntp_count_per_seed)
    coverage["total_used"] = int(total_needed)
    coverage["reuse_mode"] = "bootstrap_reuse"
    coverage["unique_sequences_used"] = int(unique_used)
    coverage["duplication_factor"] = float(total_needed / max(1, unique_used))

    print(
        "  [NTP] requested="
        f"{total_needed} available={n_available}; using bootstrap_reuse "
        f"(per_seed={ntp_count_per_seed}, unique={unique_used}, "
        f"dup_factor={coverage['duplication_factor']:.3f})",
        flush=True,
    )

    idx = 0
    for seed_i in range(num_seeds):
        seq_chunks[seed_i] = sampled[idx : idx + int(ntp_count_per_seed)]
        idx += int(ntp_count_per_seed)
    return seq_chunks, coverage


def _evaluate_ntp_losses(
    *,
    model,
    model_name: str,
    tokenizer,
    device: str,
    heads_to_zero: list[HeadID],
    num_seeds: int,
    ntp_count_per_seed: int,
    seq_len: int,
    batch_size: int,
    seed_chunks: dict[int, list[list[int]]] | None = None,
    coverage_metadata: dict[str, Any] | None = None,
    batch_state: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    if seed_chunks is None or coverage_metadata is None:
        seed_chunks, coverage_metadata = _prepare_ntp_seed_chunks(
            tokenizer=tokenizer,
            model_name=model_name,
            num_seeds=int(num_seeds),
            ntp_count_per_seed=int(ntp_count_per_seed),
            seq_len=int(seq_len),
        )

    rows: list[dict[str, Any]] = []
    requested_bs = max(1, int(batch_size))
    if batch_state is None:
        eff_bs = int(requested_bs)
    else:
        eff_bs = max(1, int(batch_state.get("wiki_ntp", requested_bs)))
    with head_output_ablation(model, heads_to_zero):
        for seed in range(num_seeds):
            seed_seqs = seed_chunks[seed]
            losses: list[float] = []
            pos = 0
            while pos < len(seed_seqs):
                batch_tokens = seed_seqs[pos : pos + eff_bs]
                try:
                    input_ids = torch.tensor(batch_tokens, dtype=torch.long, device=device)
                    with torch.inference_mode():
                        outputs = model(input_ids=input_ids, use_cache=False)
                        logits = outputs.logits[:, :-1, :]
                        labels = input_ids[:, 1:]
                        token_losses = F.cross_entropy(
                            logits.reshape(-1, logits.shape[-1]),
                            labels.reshape(-1),
                            reduction="none",
                        )
                        token_losses = token_losses.view(labels.shape[0], labels.shape[1])
                        # Ensure NumPy conversion is supported across mixed-precision model dtypes.
                        token_losses = token_losses.float()
                        seq_losses = token_losses.mean(dim=1).detach().cpu().tolist()
                        losses.extend(float(x) for x in seq_losses)
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower() and eff_bs > 1:
                        torch.cuda.empty_cache()
                        eff_bs = max(1, eff_bs // 2)
                        if batch_state is not None:
                            batch_state["wiki_ntp"] = int(eff_bs)
                        continue
                    raise
                finally:
                    if "input_ids" in locals():
                        del input_ids
                    if "outputs" in locals():
                        del outputs
                    if "logits" in locals():
                        del logits
                    if "labels" in locals():
                        del labels
                    if "token_losses" in locals():
                        del token_losses

                pos += len(batch_tokens)

            mean_loss = float(np.mean(losses))
            ppl = float(np.exp(mean_loss))
            rows.append(
                {
                    "task": "wiki_ntp",
                    "span": 0,
                    "seed": int(seed),
                    "metric_name": "loss",
                    "metric_value": mean_loss,
                    "aux_metric_name": "ppl",
                    "aux_metric_value": ppl,
                    "n_examples": int(len(losses)),
                    "requested_batch_size_ntp": int(requested_bs),
                    "effective_batch_size_ntp": int(eff_bs),
                    "ntp_requested_total": int(coverage_metadata.get("requested_total", 0)),
                    "ntp_available_total": int(coverage_metadata.get("available_total", 0)),
                    "ntp_effective_per_seed": int(coverage_metadata.get("effective_per_seed", 0)),
                    "ntp_reuse_mode": str(coverage_metadata.get("reuse_mode", "unknown")),
                }
            )
    if batch_state is not None:
        prev_bs = int(batch_state.get("wiki_ntp", eff_bs))
        batch_state["wiki_ntp"] = min(prev_bs, int(eff_bs))
    return rows


def _piecewise_rss(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    best_rss = float("inf")
    best_threshold = float("nan")
    if len(x) < 4:
        return float("nan"), float("nan")
    for split in range(2, len(x) - 1):
        x1, y1 = x[:split], y[:split]
        x2, y2 = x[split:], y[split:]
        b1 = np.polyfit(x1, y1, deg=1)
        b2 = np.polyfit(x2, y2, deg=1)
        rss = float(np.sum((y1 - np.polyval(b1, x1)) ** 2) + np.sum((y2 - np.polyval(b2, x2)) ** 2))
        if rss < best_rss:
            best_rss = rss
            best_threshold = float(x[split])
    return best_rss, best_threshold


def _bic_from_rss(*, rss: float, n: int, k: int, eps: float = 1e-12) -> float:
    return float(n * np.log((float(rss) / max(1, n)) + eps) + int(k) * np.log(max(1, n)))


def _logistic_fn(x: np.ndarray, L: float, k: float, x0: float, c: float) -> np.ndarray:
    return c + (L / (1.0 + np.exp(-k * (x - x0))))


def _fit_logistic_sigmoid(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    n = int(len(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    y_range = float(max(y_max - y_min, 1e-8))
    p0 = np.asarray([y_range, 8.0, 0.2, y_min], dtype=float)
    lower = np.asarray([0.0, 1e-3, 0.0, y_min - 2.0 * y_range], dtype=float)
    upper = np.asarray([10.0 * y_range, 100.0, 1.0, y_max + 2.0 * y_range], dtype=float)
    try:
        popt, _ = optimize.curve_fit(
            _logistic_fn,
            x,
            y,
            p0=p0,
            bounds=(lower, upper),
            maxfev=20000,
        )
        y_hat = _logistic_fn(x, *popt)
        rss = float(np.sum((y - y_hat) ** 2))
        bic = _bic_from_rss(rss=rss, n=n, k=4)
        return {
            "valid": True,
            "L": float(popt[0]),
            "k": float(popt[1]),
            "x0": float(popt[2]),
            "c": float(popt[3]),
            "rss": float(rss),
            "bic": float(bic),
        }
    except Exception as exc:
        return {
            "valid": False,
            "rss": float("nan"),
            "bic": float("nan"),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _fit_curve_models(curve_df: pd.DataFrame) -> dict[str, Any]:
    grouped = (
        curve_df.groupby("ablation_fraction", as_index=False)["degradation"]
        .mean()
        .sort_values("ablation_fraction")
        .reset_index(drop=True)
    )
    x = grouped["ablation_fraction"].to_numpy(dtype=float) / 100.0
    y = grouped["degradation"].to_numpy(dtype=float)

    if len(x) < 4:
        return {
            "n_points": int(len(x)),
            "preferred_model": "insufficient_points",
        }

    b_lin = np.polyfit(x, y, deg=1)
    y_lin = np.polyval(b_lin, x)
    rss_lin = float(np.sum((y - y_lin) ** 2))
    rss_pw, threshold = _piecewise_rss(x, y)

    n = len(x)
    bic_lin = _bic_from_rss(rss=rss_lin, n=n, k=2)
    bic_pw = _bic_from_rss(rss=rss_pw, n=n, k=4)
    logistic = _fit_logistic_sigmoid(x, y)

    bic_candidates = {
        "linear": bic_lin,
        "threshold_piecewise": bic_pw,
        "logistic_sigmoid": float(logistic.get("bic", float("nan"))),
    }
    valid_items = [(name, bic) for name, bic in bic_candidates.items() if np.isfinite(float(bic))]
    preferred = min(valid_items, key=lambda item: item[1])[0] if valid_items else "insufficient_points"

    return {
        "n_points": int(n),
        "linear": {
            "slope": float(b_lin[0]),
            "intercept": float(b_lin[1]),
            "rss": float(rss_lin),
            "bic": bic_lin,
        },
        "threshold_piecewise": {
            "rss": float(rss_pw),
            "bic": bic_pw,
            "best_threshold_fraction": float(threshold),
        },
        "logistic_sigmoid": logistic,
        "bic_candidates": {k: _safe_float(v) for k, v in bic_candidates.items()},
        "preferred_model": preferred,
    }


def run_model(
    *,
    model_name: str,
    device: str,
    output_root: Path,
    fractions: tuple[int, ...],
    sort_orders: tuple[str, ...],
    num_seeds: int,
    synthetic_count: int,
    batch_size_synth: int,
    ntp_count_per_seed: int,
    ntp_seq_len: int,
    batch_size_ntp: int,
    skip_ntp: bool,
    use_ntp_sequence_cache: bool = True,
) -> None:
    print(f"\n[3P2-C.1] model={model_name} device={device}")
    t_model = time.time()

    model_spec = MODELS[model_name]
    loaded = load_model(model_spec)
    model = loaded.model.to(device)
    tokenizer = load_tokenizer(model_spec)

    ranked_heads, r2_summary = _load_ranked_heads(model_name)
    print(f"  Loaded R² ranking for {len(ranked_heads)} heads.")

    retrieval_candidates = RETRIEVAL_SPANS.get(model_name, ())
    retrieval_span = 48 if 48 in retrieval_candidates else int(max(retrieval_candidates))
    print(f"  Retrieval span for C.1: {retrieval_span}")

    vocab_size = int(tokenizer.vocab_size)
    special_ids = [getattr(tokenizer, attr, None) for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")]
    special_ids = [int(x) for x in special_ids if x is not None]
    pools = build_token_pools(model_name, vocab_size, special_ids)
    seeds = range(max(1, int(num_seeds)))
    task_configs, prebuilt_examples = _build_synthetic_cells(
        model_name=model_name,
        pools=pools,
        seeds=seeds,
        synthetic_count=max(1, int(synthetic_count)),
        retrieval_span=int(retrieval_span),
    )

    out_dir = output_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    condition_rows_cache: dict[tuple[str, int], pd.DataFrame] = {}
    synth_batch_ceiling_cache: dict[tuple[str, int], int] = {}
    ntp_batch_state: dict[str, int] = {"wiki_ntp": max(1, int(batch_size_ntp))}
    ntp_seed_chunks: dict[int, list[list[int]]] | None = None
    ntp_coverage: dict[str, Any] | None = None
    if not skip_ntp and bool(use_ntp_sequence_cache):
        ntp_seed_chunks, ntp_coverage = _prepare_ntp_seed_chunks(
            tokenizer=tokenizer,
            model_name=model_name,
            num_seeds=max(1, int(num_seeds)),
            ntp_count_per_seed=max(1, int(ntp_count_per_seed)),
            seq_len=max(64, int(ntp_seq_len)),
        )

    total_conditions = len(sort_orders) * len(fractions)
    completed = 0
    start = time.time()

    for sort_order in sort_orders:
        for frac in fractions:
            heads = _heads_for_fraction(ranked_heads, int(frac), sort_order)
            cond_name = f"{sort_order}_f{int(frac):02d}"
            completed += 1
            elapsed = max(1e-6, time.time() - start)
            avg = elapsed / completed
            eta = avg * (total_conditions - completed)
            print(
                f"  [{completed}/{total_conditions}] {cond_name}: "
                f"{len(heads)} heads | eta {int(eta)}s",
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
                batch_ceiling_cache=synth_batch_ceiling_cache,
            )
            synth_df = pd.DataFrame(synth_rows)
            synth_df["sort_order"] = sort_order
            synth_df["ablation_fraction"] = int(frac)
            synth_df["n_heads_ablated"] = int(len(heads))
            synth_df["metric_name"] = "accuracy"
            synth_df["metric_value"] = synth_df["accuracy"].astype(float)
            synth_df["aux_metric_name"] = "none"
            synth_df["aux_metric_value"] = np.nan
            condition_rows = [synth_df]

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
                    seed_chunks=ntp_seed_chunks if bool(use_ntp_sequence_cache) else None,
                    coverage_metadata=ntp_coverage if bool(use_ntp_sequence_cache) else None,
                    batch_state=ntp_batch_state,
                )
                ntp_df = pd.DataFrame(ntp_rows)
                ntp_df["condition"] = cond_name
                ntp_df["sort_order"] = sort_order
                ntp_df["ablation_fraction"] = int(frac)
                ntp_df["n_heads_ablated"] = int(len(heads))
                ntp_df["accuracy"] = np.nan
                ntp_df["n_targets"] = np.nan
                ntp_df["n_correct"] = np.nan
                ntp_df["task"] = ntp_df["task"].astype(str)
                condition_rows.append(ntp_df)

            merged_df = pd.concat(condition_rows, ignore_index=True, sort=False)
            condition_rows_cache[(sort_order, int(frac))] = merged_df.copy()
            all_rows.append(merged_df)

    full_df = pd.concat(all_rows, ignore_index=True, sort=False)

    # Baselines are fraction==0 per sort order.
    baseline = (
        full_df[full_df["ablation_fraction"] == 0]
        .groupby(["sort_order", "task", "seed", "metric_name"], as_index=False)["metric_value"]
        .mean()
        .rename(columns={"metric_value": "baseline_metric_value"})
    )
    full_df = full_df.merge(
        baseline,
        on=["sort_order", "task", "seed", "metric_name"],
        how="left",
    )

    full_df["degradation"] = np.nan
    is_accuracy = full_df["metric_name"] == "accuracy"
    full_df.loc[is_accuracy, "degradation"] = (
        full_df.loc[is_accuracy, "baseline_metric_value"] - full_df.loc[is_accuracy, "metric_value"]
    )
    is_loss = full_df["metric_name"] == "loss"
    full_df.loc[is_loss, "degradation"] = (
        full_df.loc[is_loss, "metric_value"] - full_df.loc[is_loss, "baseline_metric_value"]
    )

    full_df["tier"] = "tier2_conditional_mechanistic"
    full_df["primary_test_id"] = "3P2-C.1"
    full_df["mde_target"] = 0.35
    full_df["achieved_power"] = 0.80
    full_df["multiplicity_family"] = "tier2_holm_primary_tests"
    full_df["model"] = model_name

    out_curve = out_dir / "cumulative_ablation_curve.parquet"
    full_df.to_parquet(out_curve, index=False)
    print(f"  wrote {out_curve}")

    fit_payload: dict[str, Any] = {
        "experiment": "3P2-C.1_cumulative_ablation_curve",
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tier": "tier2_conditional_mechanistic",
        "primary_test_id": "3P2-C.1",
        "mde_target": 0.35,
        "achieved_power": 0.80,
        "multiplicity_family": "tier2_holm_primary_tests",
        "curve_fits": {},
    }

    grouped = full_df.groupby(["sort_order", "task", "metric_name"], as_index=False)
    for row in grouped:
        (sort_order, task, metric_name), gdf = row
        fit = _fit_curve_models(gdf)
        key = f"{sort_order}::{task}::{metric_name}"
        fit_payload["curve_fits"][key] = fit

    # Aggregate support signal for E2.
    preferred = [x.get("preferred_model") for x in fit_payload["curve_fits"].values()]
    threshold_votes = sum(1 for x in preferred if x == "threshold_piecewise")
    linear_votes = sum(1 for x in preferred if x == "linear")
    logistic_votes = sum(1 for x in preferred if x == "logistic_sigmoid")
    fit_payload["summary"] = {
        "threshold_votes": int(threshold_votes),
        "linear_votes": int(linear_votes),
        "logistic_votes": int(logistic_votes),
        "supports_redundancy_threshold_pattern": bool(threshold_votes > max(linear_votes, logistic_votes)),
        "note": (
            "Threshold preference supports E2 redundancy; "
            "linear preference supports E1-style gradual non-specific degradation."
        ),
        "runtime_seconds": float(time.time() - t_model),
        "effective_batch_size_synth": {
            f"{task}::span{span}": int(bs)
            for (task, span), bs in sorted(synth_batch_ceiling_cache.items())
        },
        "effective_batch_size_ntp": int(ntp_batch_state.get("wiki_ntp", max(1, int(batch_size_ntp)))),
        "ntp_coverage": ntp_coverage if not skip_ntp else None,
        "ntp_sequence_cache_enabled": bool(use_ntp_sequence_cache and (not skip_ntp)),
    }

    out_fit = out_dir / "curve_fit_comparison.json"
    _write_json(out_fit, fit_payload)
    print(f"  wrote {out_fit}")

    _write_json(
        out_dir / "manifest.json",
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "experiment": "3P2-C.1",
            "model": model_name,
            "device": device,
            "fractions": [int(x) for x in fractions],
            "sort_orders": [str(x) for x in sort_orders],
            "num_seeds": int(num_seeds),
            "synthetic_count": int(synthetic_count),
            "ntp_count_per_seed": int(ntp_count_per_seed),
            "ntp_seq_len": int(ntp_seq_len),
            "batch_size_synth": int(batch_size_synth),
            "batch_size_ntp_requested": int(batch_size_ntp),
            "effective_batch_size_synth": fit_payload["summary"].get("effective_batch_size_synth"),
            "effective_batch_size_ntp": fit_payload["summary"].get("effective_batch_size_ntp"),
            "ntp_coverage": fit_payload["summary"].get("ntp_coverage"),
            "ntp_sequence_cache_enabled": bool(fit_payload["summary"].get("ntp_sequence_cache_enabled", False)),
            "runtime_seconds": float(fit_payload["summary"].get("runtime_seconds", float("nan"))),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3P2-C.1: cumulative ablation curve")
    parser.add_argument("--model", default="all", choices=["all"] + sorted(MODELS.keys()))
    parser.add_argument("--device", default="cuda:0", help="Device for single-model mode.")
    parser.add_argument("--output-root", default="results/experiment3_phase2/exp3p2c_redundancy_quantification")
    parser.add_argument("--fractions", default="0,1,2,5,10,15,20,25,50")
    parser.add_argument("--sort-orders", default="high_to_low,low_to_high")
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--synthetic-count", type=int, default=100)
    parser.add_argument("--batch-size-synth", type=int, default=8)
    parser.add_argument("--ntp-count-per-seed", type=int, default=100)
    parser.add_argument("--ntp-seq-len", type=int, default=512)
    parser.add_argument("--batch-size-ntp", type=int, default=4)
    parser.add_argument("--skip-ntp", action="store_true")
    parser.add_argument("--disable-ntp-seq-cache", action="store_true")
    parser.add_argument(
        "--device-map",
        default="llama-3.1-8b:cuda:0,olmo-2-7b:cuda:1",
        help="Comma-separated model:device map used when --model all.",
    )
    return parser.parse_args()


def _parse_device_map(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in str(raw).split(","):
        token = part.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid device map token '{token}'. Expected model:device")
        model, device = token.split(":", 1)
        out[model.strip()] = device.strip()
    return out


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    fractions = _parse_int_tuple(args.fractions)
    sort_orders = tuple(tok.strip() for tok in str(args.sort_orders).split(",") if tok.strip())
    for order in sort_orders:
        if order not in DEFAULT_SORT_ORDERS:
            raise ValueError(f"Unsupported sort order: {order}")

    models = sorted(MODELS.keys()) if args.model == "all" else [args.model]
    device_map = _parse_device_map(args.device_map) if args.model == "all" else {}

    for model_name in models:
        device = device_map.get(model_name, args.device)
        run_model(
            model_name=model_name,
            device=device,
            output_root=output_root,
            fractions=fractions,
            sort_orders=sort_orders,
            num_seeds=max(1, int(args.num_seeds)),
            synthetic_count=max(1, int(args.synthetic_count)),
            batch_size_synth=max(1, int(args.batch_size_synth)),
            ntp_count_per_seed=max(1, int(args.ntp_count_per_seed)),
            ntp_seq_len=max(64, int(args.ntp_seq_len)),
            batch_size_ntp=max(1, int(args.batch_size_ntp)),
            skip_ntp=bool(args.skip_ntp),
            use_ntp_sequence_cache=not bool(args.disable_ntp_seq_cache),
        )


if __name__ == "__main__":
    main()
