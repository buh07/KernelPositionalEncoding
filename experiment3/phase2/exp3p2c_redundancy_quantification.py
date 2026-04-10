#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

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
) -> list[dict[str, Any]]:
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

    # Graceful degradation when the tokenized corpus has fewer sequences than requested.
    # Keep seed-stratified evaluation by assigning an equal per-seed quota when possible.
    seq_chunks: dict[int, list[list[int]]] = {}
    if n_available >= int(num_seeds):
        per_seed = min(int(ntp_count_per_seed), n_available // int(num_seeds))
        per_seed = max(1, int(per_seed))
        usable = int(per_seed) * int(num_seeds)
        if usable < total_needed:
            print(
                f"  [NTP] requested={total_needed} available={n_available}; "
                f"using per_seed={per_seed} (total={usable})",
                flush=True,
            )
        trimmed = sequences[:usable]
        idx = 0
        for seed in range(num_seeds):
            seq_chunks[seed] = trimmed[idx : idx + per_seed]
            idx += per_seed
    else:
        print(
            f"  [NTP] requested={total_needed} available={n_available} (< num_seeds={num_seeds}); "
            "reusing sequences across seeds",
            flush=True,
        )
        for seed in range(num_seeds):
            seq_chunks[seed] = [sequences[seed % n_available]]

    rows: list[dict[str, Any]] = []
    eff_bs = max(1, int(batch_size))
    with head_output_ablation(model, heads_to_zero):
        for seed in range(num_seeds):
            seed_seqs = seq_chunks[seed]
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
                        seq_losses = token_losses.mean(dim=1).detach().cpu().numpy().tolist()
                        losses.extend(float(x) for x in seq_losses)
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower() and eff_bs > 1:
                        torch.cuda.empty_cache()
                        eff_bs = max(1, eff_bs // 2)
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
                }
            )
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
    eps = 1e-12
    k_lin = 2
    k_pw = 4
    bic_lin = float(n * np.log((rss_lin / max(1, n)) + eps) + k_lin * np.log(max(1, n)))
    bic_pw = float(n * np.log((rss_pw / max(1, n)) + eps) + k_pw * np.log(max(1, n)))
    preferred = "threshold_piecewise" if bic_pw < bic_lin else "linear"

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
    fit_payload["summary"] = {
        "threshold_votes": int(threshold_votes),
        "linear_votes": int(linear_votes),
        "supports_redundancy_threshold_pattern": bool(threshold_votes > linear_votes),
        "note": (
            "Threshold preference supports E2 redundancy; "
            "linear preference supports E1-style gradual non-specific degradation."
        ),
        "runtime_seconds": float(time.time() - t_model),
    }

    out_fit = out_dir / "curve_fit_comparison.json"
    _write_json(out_fit, fit_payload)
    print(f"  wrote {out_fit}")


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
        )


if __name__ == "__main__":
    main()
