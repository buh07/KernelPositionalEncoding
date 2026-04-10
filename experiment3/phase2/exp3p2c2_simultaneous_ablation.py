#!/usr/bin/env python3
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

from experiment2.tasks import TaskExample, TokenPools, build_token_pools, generate_task_examples
from experiment3.phase2.exp3p2c_redundancy_quantification import (  # noqa: E402
    _evaluate_ntp_losses,
    _load_ranked_heads,
)
from experiment3.theory1_si_circuits import (  # noqa: E402
    MODELS,
    RETRIEVAL_SPANS,
    HeadID,
    evaluate_task_battery,
)
from shared.models.loading import load_model, load_tokenizer  # noqa: E402


TARGET_MODELS = ("llama-3.1-8b", "olmo-2-7b")
DEFAULT_FRACTIONS = (0, 25, 50, 75)


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
        v = int(tok)
        if v < 0 or v > 100:
            raise ValueError(f"Invalid fraction {v}; expected 0..100")
        if v in seen:
            continue
        vals.append(v)
        seen.add(v)
    if not vals:
        raise ValueError("Empty fractions list")
    return tuple(vals)


def _heads_for_fraction(ranked_heads_desc: list[HeadID], fraction_pct: int) -> list[HeadID]:
    if fraction_pct <= 0:
        return []
    n_total = len(ranked_heads_desc)
    n_select = max(1, int(round((fraction_pct / 100.0) * n_total)))
    return ranked_heads_desc[:n_select]


def _build_task_cells(
    *,
    model_name: str,
    pools: TokenPools,
    seeds: range,
    synthetic_count: int,
    retrieval_spans: tuple[int, ...],
) -> tuple[list[tuple[str, int | None, tuple[int, ...] | None]], dict[tuple[int, str, int], list[TaskExample]]]:
    task_configs: list[tuple[str, int | None, tuple[int, ...] | None]] = []
    for span in retrieval_spans:
        task_configs.append(("long_range_retrieval", int(span), (int(span),)))
    task_configs.append(("local_key_match", None, None))

    prebuilt: dict[tuple[int, str, int], list[TaskExample]] = {}
    for seed in seeds:
        for task_name, span_override, span_choices in task_configs:
            span_val = int(span_override) if span_override is not None else 0
            prebuilt[(int(seed), task_name, span_val)] = generate_task_examples(
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


def _bootstrap_excess(d25: np.ndarray, d75: np.ndarray, *, n_boot: int, seed: int) -> dict[str, Any]:
    d25 = np.asarray(d25, dtype=float)
    d75 = np.asarray(d75, dtype=float)
    mask = np.isfinite(d25) & np.isfinite(d75)
    d25 = d25[mask]
    d75 = d75[mask]

    if d25.size == 0:
        return {
            "n": 0,
            "mean_25": float("nan"),
            "mean_75": float("nan"),
            "ratio_75_over_25": float("nan"),
            "mean_excess_over_3x": float("nan"),
            "ci_95_excess": [float("nan"), float("nan")],
            "p_one_sided_excess": float("nan"),
            "supports_superlinear": False,
        }

    excess = d75 - 3.0 * d25
    mean_25 = float(np.mean(d25))
    mean_75 = float(np.mean(d75))
    ratio = float(mean_75 / max(abs(mean_25), 1e-8))
    obs = float(np.mean(excess))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, excess.size, size=(n_boot, excess.size))
    boot = excess[idx].mean(axis=1)
    p_one = float((np.sum(boot <= 0.0) + 1) / (n_boot + 1))

    return {
        "n": int(excess.size),
        "mean_25": mean_25,
        "mean_75": mean_75,
        "ratio_75_over_25": ratio,
        "mean_excess_over_3x": obs,
        "ci_95_excess": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "p_one_sided_excess": p_one,
        "supports_superlinear": bool(p_one < 0.05 and obs > 0.0),
    }


def _run_model(
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
    bootstrap_samples: int,
) -> dict[str, Any]:
    print(f"\n[3P2-C.2] model={model_name} device={device}", flush=True)
    t_model = time.time()

    model_spec = MODELS[model_name]
    loaded = load_model(model_spec)
    model = loaded.model.to(device)
    tokenizer = load_tokenizer(model_spec)

    ranked_heads, _ = _load_ranked_heads(model_name)
    retrieval_spans = tuple(int(x) for x in RETRIEVAL_SPANS.get(model_name, (48,)))

    vocab_size = int(tokenizer.vocab_size)
    special_ids = [getattr(tokenizer, a, None) for a in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")]
    special_ids = [int(x) for x in special_ids if x is not None]
    pools = build_token_pools(model_name, vocab_size, special_ids)

    seeds = range(max(1, int(num_seeds)))
    task_configs, prebuilt_examples = _build_task_cells(
        model_name=model_name,
        pools=pools,
        seeds=seeds,
        synthetic_count=max(1, int(synthetic_count)),
        retrieval_spans=retrieval_spans,
    )

    out_dir = output_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_all: list[pd.DataFrame] = []

    for frac in fractions:
        heads = _heads_for_fraction(ranked_heads, int(frac))
        condition_name = f"top_r2_f{int(frac):02d}"
        print(
            f"  [3P2-C.2] fraction={frac}% heads={len(heads)}",
            flush=True,
        )

        synth_rows = evaluate_task_battery(
            model=model,
            tokenizer=tokenizer,
            model_spec=model_spec,
            device=device,
            heads_to_zero=heads,
            condition_name=condition_name,
            seeds=seeds,
            retrieval_spans=retrieval_spans,
            pools=pools,
            synthetic_count=max(1, int(synthetic_count)),
            batch_size=max(1, int(batch_size_synth)),
            task_configs=task_configs,
            prebuilt_examples=prebuilt_examples,
        )
        synth_df = pd.DataFrame(synth_rows)
        synth_df["ablation_fraction"] = int(frac)
        synth_df["n_heads_ablated"] = int(len(heads))
        synth_df["metric_name"] = "accuracy"
        synth_df["metric_value"] = synth_df["accuracy"].astype(float)
        synth_df["aux_metric_name"] = "none"
        synth_df["aux_metric_value"] = np.nan
        frames = [synth_df]

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
            ntp_df["condition"] = condition_name
            ntp_df["ablation_fraction"] = int(frac)
            ntp_df["n_heads_ablated"] = int(len(heads))
            ntp_df["accuracy"] = np.nan
            ntp_df["n_targets"] = np.nan
            ntp_df["n_correct"] = np.nan
            ntp_df["task"] = ntp_df["task"].astype(str)
            frames.append(ntp_df)

        merged = pd.concat(frames, ignore_index=True, sort=False)
        rows_all.append(merged)

    full_df = pd.concat(rows_all, ignore_index=True, sort=False)

    baseline = (
        full_df[full_df["ablation_fraction"] == 0]
        .groupby(["task", "seed", "metric_name"], as_index=False)["metric_value"]
        .mean()
        .rename(columns={"metric_value": "baseline_metric_value"})
    )
    full_df = full_df.merge(
        baseline,
        on=["task", "seed", "metric_name"],
        how="left",
    )

    full_df["degradation"] = np.nan
    is_acc = full_df["metric_name"] == "accuracy"
    full_df.loc[is_acc, "degradation"] = (
        full_df.loc[is_acc, "baseline_metric_value"] - full_df.loc[is_acc, "metric_value"]
    )
    is_loss = full_df["metric_name"] == "loss"
    full_df.loc[is_loss, "degradation"] = (
        full_df.loc[is_loss, "metric_value"] - full_df.loc[is_loss, "baseline_metric_value"]
    )

    full_df["model"] = model_name
    full_df["tier"] = "tier2_conditional_mechanistic"
    full_df["primary_test_id"] = "3P2-C.2"
    full_df["mde_target"] = 0.35
    full_df["achieved_power"] = 0.80
    full_df["multiplicity_family"] = "tier2_holm_primary_tests"

    out_parquet = out_dir / "simultaneous_ablation_results.parquet"
    full_df.to_parquet(out_parquet, index=False)
    print(f"  wrote {out_parquet}", flush=True)

    test_payload: dict[str, Any] = {
        "experiment": "3P2-C.2_all_high_si_simultaneous_ablation",
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tier": "tier2_conditional_mechanistic",
        "primary_test_id": "3P2-C.2",
        "mde_target": 0.35,
        "achieved_power": 0.80,
        "multiplicity_family": "tier2_holm_primary_tests",
        "fractions": list(int(x) for x in fractions),
        "nonlinearity_rule": "75% degradation > 3x 25% degradation (paired bootstrap)",
        "per_task_metric": {},
        "pooled_by_metric": {},
    }

    # Per-task test.
    for (task, metric_name), g in full_df.groupby(["task", "metric_name"], as_index=False):
        wide = (
            g[g["ablation_fraction"].isin([25, 75])]
            .pivot_table(index=["seed"], columns="ablation_fraction", values="degradation", aggfunc="mean")
            .reset_index(drop=True)
        )
        d25 = wide[25].to_numpy(dtype=float) if 25 in wide.columns else np.array([], dtype=float)
        d75 = wide[75].to_numpy(dtype=float) if 75 in wide.columns else np.array([], dtype=float)
        test_payload["per_task_metric"][f"{task}::{metric_name}"] = _bootstrap_excess(
            d25,
            d75,
            n_boot=max(500, int(bootstrap_samples)),
            seed=2026,
        )

    # Pooled by metric.
    for metric_name, g in full_df.groupby("metric_name", as_index=False):
        wide = (
            g[g["ablation_fraction"].isin([25, 75])]
            .groupby(["task", "seed", "ablation_fraction"], as_index=False)["degradation"]
            .mean()
            .pivot_table(index=["task", "seed"], columns="ablation_fraction", values="degradation", aggfunc="first")
            .reset_index()
        )
        d25 = wide[25].to_numpy(dtype=float) if 25 in wide.columns else np.array([], dtype=float)
        d75 = wide[75].to_numpy(dtype=float) if 75 in wide.columns else np.array([], dtype=float)
        test_payload["pooled_by_metric"][str(metric_name)] = _bootstrap_excess(
            d25,
            d75,
            n_boot=max(500, int(bootstrap_samples)),
            seed=2027,
        )

    superlinear_votes = sum(1 for v in test_payload["per_task_metric"].values() if bool(v.get("supports_superlinear", False)))
    total_votes = len(test_payload["per_task_metric"])
    test_payload["summary"] = {
        "superlinear_votes": int(superlinear_votes),
        "total_votes": int(total_votes),
        "supports_nonlinearity": bool(superlinear_votes > max(0, total_votes // 2)),
        "runtime_seconds": float(time.time() - t_model),
    }

    out_json = out_dir / "nonlinearity_test.json"
    _write_json(out_json, test_payload)
    print(f"  wrote {out_json}", flush=True)

    return test_payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3P2-C.2: simultaneous high-SI ablation nonlinearity stress test")
    p.add_argument("--model", choices=["all", *TARGET_MODELS], default="all")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--device-map",
        default="llama-3.1-8b:cuda:0,olmo-2-7b:cuda:1",
        help="Comma-separated model:device map used when --model all",
    )
    p.add_argument("--fractions", default="0,25,50,75")
    p.add_argument("--num-seeds", type=int, default=3)
    p.add_argument("--synthetic-count", type=int, default=100)
    p.add_argument("--batch-size-synth", type=int, default=8)
    p.add_argument("--ntp-count-per-seed", type=int, default=100)
    p.add_argument("--ntp-seq-len", type=int, default=512)
    p.add_argument("--batch-size-ntp", type=int, default=4)
    p.add_argument("--skip-ntp", action="store_true")
    p.add_argument("--bootstrap-samples", type=int, default=5000)
    p.add_argument(
        "--output-root",
        default="results/experiment3_phase2/exp3p2c_redundancy_quantification",
    )
    return p.parse_args()


def _parse_device_map(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in [t.strip() for t in raw.split(",") if t.strip()]:
        model, device = tok.split(":", 1)
        out[model.strip()] = device.strip()
    return out


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    models = list(TARGET_MODELS) if args.model == "all" else [args.model]
    device_map = _parse_device_map(args.device_map) if args.model == "all" else {}
    fractions = _parse_int_tuple(args.fractions)

    summary: dict[str, Any] = {}
    for model_name in models:
        device = device_map.get(model_name, args.device)
        rep = _run_model(
            model_name=model_name,
            device=device,
            output_root=output_root,
            fractions=fractions,
            num_seeds=max(1, int(args.num_seeds)),
            synthetic_count=max(1, int(args.synthetic_count)),
            batch_size_synth=max(1, int(args.batch_size_synth)),
            ntp_count_per_seed=max(1, int(args.ntp_count_per_seed)),
            ntp_seq_len=max(64, int(args.ntp_seq_len)),
            batch_size_ntp=max(1, int(args.batch_size_ntp)),
            skip_ntp=bool(args.skip_ntp),
            bootstrap_samples=max(500, int(args.bootstrap_samples)),
        )
        summary[model_name] = rep

    _write_json(output_root / "c2_nonlinearity_summary.json", {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": summary,
    })


if __name__ == "__main__":
    main()
