#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp.common import RESULTS_ROOT, command_manifest, ensure_dir, safe_float, timestamp_now, write_json  # noqa: E402
from experiment2.tasks import build_token_pools  # noqa: E402
from experiment3.phase2.exp3p2c_redundancy_quantification import (  # noqa: E402
    _build_synthetic_cells,
    _evaluate_ntp_losses,
    _fit_curve_models,
    _load_ranked_heads,
    _parse_int_tuple,
    _prepare_ntp_seed_chunks,
)
from experiment3.theory1_si_circuits import MODELS, RETRIEVAL_SPANS, evaluate_task_battery  # noqa: E402
from shared.models.loading import load_model, load_tokenizer  # noqa: E402

DEFAULT_OUT = RESULTS_ROOT / "exp_new_r12_ordering_control"
TARGET_MODELS = ("llama-3.1-8b", "olmo-2-7b")


def _heads_for_fraction(ordering: list[Any], fraction_pct: int) -> list[Any]:
    if int(fraction_pct) <= 0:
        return []
    n_total = len(ordering)
    n_select = max(1, int(round((float(fraction_pct) / 100.0) * n_total)))
    return list(ordering[:n_select])


def _ordering_verdict(per_key_fits: dict[str, Any]) -> dict[str, Any]:
    preferred = [str(v.get("preferred_model")) for v in per_key_fits.values()]
    th = sum(1 for p in preferred if p == "threshold_piecewise")
    li = sum(1 for p in preferred if p == "linear")
    lo = sum(1 for p in preferred if p == "logistic_sigmoid")
    votes_total = int(th + li + lo)

    th_vs_lin_wins = 0
    th_vs_lin_comp = 0
    th_vs_log_wins = 0
    th_vs_log_comp = 0
    for payload in per_key_fits.values():
        bic_th = safe_float(payload.get("threshold_piecewise", {}).get("bic"))
        bic_lin = safe_float(payload.get("linear", {}).get("bic"))
        bic_log = safe_float(payload.get("logistic_sigmoid", {}).get("bic"))
        if np.isfinite(bic_th) and np.isfinite(bic_lin):
            th_vs_lin_comp += 1
            th_vs_lin_wins += int(bic_th < bic_lin)
        if np.isfinite(bic_th) and np.isfinite(bic_log):
            th_vs_log_comp += 1
            th_vs_log_wins += int(bic_th < bic_log)

    frac = float(th / max(1, votes_total))
    return {
        "threshold_votes": int(th),
        "linear_votes": int(li),
        "logistic_votes": int(lo),
        "n_fits": int(len(preferred)),
        "threshold_fraction": frac,
        "threshold_majority": bool(th > (li + lo)),
        "threshold_majority_over_alternatives": bool(th > (li + lo)),
        "threshold_vs_linear_wins": int(th_vs_lin_wins),
        "threshold_vs_linear_comparisons": int(th_vs_lin_comp),
        "threshold_vs_logistic_wins": int(th_vs_log_wins),
        "threshold_vs_logistic_comparisons": int(th_vs_log_comp),
    }


def _mean_bic_delta_pairwise(fits: dict[str, Any], *, left: str, right: str) -> float:
    vals: list[float] = []
    for payload in fits.values():
        bic_left = safe_float(payload.get(left, {}).get("bic"))
        bic_right = safe_float(payload.get(right, {}).get("bic"))
        if np.isfinite(bic_left) and np.isfinite(bic_right):
            vals.append(float(bic_left - bic_right))
    if not vals:
        return float("nan")
    return float(np.mean(np.asarray(vals, dtype=float)))


def _mean_bic_delta(fits: dict[str, Any]) -> float:
    # >0 means threshold outperforms linear (kept for backward compatibility).
    return _mean_bic_delta_pairwise(fits, left="linear", right="threshold_piecewise")


def _load_canonical_fit(model_name: str) -> dict[str, Any] | None:
    paths = [
        ROOT / "results" / "experiment3_phase2" / "exp3p2c_redundancy_quantification" / model_name / "curve_fit_comparison.json",
        ROOT / "results" / "reinforce_exp" / "exp_r3_core_replication" / model_name / "exp3p2c_redundancy_quantification" / model_name / "curve_fit_comparison.json",
    ]
    for p in paths:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
    return None


def run_model(
    *,
    model_name: str,
    device: str,
    output_root: Path,
    fractions: tuple[int, ...],
    num_orderings: int,
    ordering_seed_base: int,
    ordering_id_start: int = 0,
    ordering_id_stop: int | None = None,
    num_seeds: int,
    synthetic_count: int,
    batch_size_synth: int,
    ntp_count_per_seed: int,
    ntp_seq_len: int,
    batch_size_ntp: int,
    skip_ntp: bool,
) -> dict[str, Any]:
    print(f"\n[NEW-R12] model={model_name} device={device}", flush=True)
    t_model = time.time()
    out_dir = output_root / model_name
    ensure_dir(out_dir)

    model_spec = MODELS[model_name]
    loaded = load_model(model_spec)
    model = loaded.model.to(device)
    model.eval()
    tokenizer = load_tokenizer(model_spec)

    ranked_heads, _ = _load_ranked_heads(model_name)
    n_heads = len(ranked_heads)
    print(f"[NEW-R12] loaded ranked heads: {n_heads}", flush=True)

    retrieval_candidates = RETRIEVAL_SPANS.get(model_name, ())
    retrieval_span = 48 if 48 in retrieval_candidates else int(max(retrieval_candidates))

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

    start_id = max(0, int(ordering_id_start))
    default_stop = max(1, int(num_orderings))
    stop_id = int(ordering_id_stop) if ordering_id_stop is not None else int(default_stop)
    if stop_id <= start_id:
        raise ValueError(f"Invalid ordering shard range: [{start_id}, {stop_id})")
    ordering_ids = list(range(start_id, stop_id))

    ordering_payloads: dict[int, dict[str, Any]] = {}
    all_rows: list[pd.DataFrame] = []
    synth_batch_ceiling_cache: dict[tuple[str, int], int] = {}
    ntp_batch_state: dict[str, int] = {"wiki_ntp": max(1, int(batch_size_ntp))}
    ntp_seed_chunks: dict[int, list[list[int]]] | None = None
    ntp_coverage: dict[str, Any] | None = None
    if not skip_ntp:
        ntp_seed_chunks, ntp_coverage = _prepare_ntp_seed_chunks(
            tokenizer=tokenizer,
            model_name=model_name,
            num_seeds=max(1, int(num_seeds)),
            ntp_count_per_seed=max(1, int(ntp_count_per_seed)),
            seq_len=max(64, int(ntp_seq_len)),
        )

    for ordering_pos, ordering_id in enumerate(ordering_ids):
        perm_seed = int(ordering_seed_base + ordering_id * 1009)
        perm_rng = np.random.default_rng(perm_seed)
        perm_idx = perm_rng.permutation(n_heads).tolist()
        ordering = [ranked_heads[i] for i in perm_idx]

        for frac in fractions:
            heads = _heads_for_fraction(ordering, int(frac))
            cond_name = f"random_o{ordering_id:02d}_f{int(frac):02d}"
            print(
                f"[NEW-R12][{model_name}] ordering={ordering_pos+1}/{len(ordering_ids)} "
                f"(global_id={ordering_id}) "
                f"fraction={int(frac):>2}% heads={len(heads)}",
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
            synth_df["ordering_type"] = "random"
            synth_df["ordering_id"] = int(ordering_id)
            synth_df["order_seed"] = int(perm_seed)
            synth_df["ablation_fraction"] = int(frac)
            synth_df["n_heads_ablated"] = int(len(heads))
            synth_df["metric_name"] = "accuracy"
            synth_df["metric_value"] = synth_df["accuracy"].astype(float)
            synth_df["aux_metric_name"] = "none"
            synth_df["aux_metric_value"] = np.nan
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
                ntp_df["condition"] = cond_name
                ntp_df["ordering_type"] = "random"
                ntp_df["ordering_id"] = int(ordering_id)
                ntp_df["order_seed"] = int(perm_seed)
                ntp_df["ablation_fraction"] = int(frac)
                ntp_df["n_heads_ablated"] = int(len(heads))
                ntp_df["accuracy"] = np.nan
                ntp_df["n_targets"] = np.nan
                ntp_df["n_correct"] = np.nan
                ntp_df["task"] = ntp_df["task"].astype(str)
                pieces.append(ntp_df)

            all_rows.append(pd.concat(pieces, ignore_index=True, sort=False))

        ordering_payloads[int(ordering_id)] = {
            "ordering_id": int(ordering_id),
            "order_seed": int(perm_seed),
            "head_index_order": [int(x) for x in perm_idx],
        }

    if not all_rows:
        raise RuntimeError("[NEW-R12] no rows collected")

    full_df = pd.concat(all_rows, ignore_index=True, sort=False)
    baseline = (
        full_df[full_df["ablation_fraction"] == 0]
        .groupby(["ordering_id", "task", "seed", "metric_name"], as_index=False)["metric_value"]
        .mean()
        .rename(columns={"metric_value": "baseline_metric_value"})
    )
    full_df = full_df.merge(
        baseline,
        on=["ordering_id", "task", "seed", "metric_name"],
        how="left",
    )
    full_df["degradation"] = np.nan
    is_acc = full_df["metric_name"] == "accuracy"
    full_df.loc[is_acc, "degradation"] = full_df.loc[is_acc, "baseline_metric_value"] - full_df.loc[is_acc, "metric_value"]
    is_loss = full_df["metric_name"] == "loss"
    full_df.loc[is_loss, "degradation"] = full_df.loc[is_loss, "metric_value"] - full_df.loc[is_loss, "baseline_metric_value"]
    full_df["primary_test_id"] = "NEW-R12"
    full_df["model"] = model_name

    out_curve = out_dir / "cumulative_ablation_curve_random.parquet"
    full_df.to_parquet(out_curve, index=False)

    per_ordering_fits: dict[str, dict[str, Any]] = {}
    ordering_rows: list[dict[str, Any]] = []
    ordering_bic_delta: list[float] = []
    ordering_bic_delta_logistic: list[float] = []

    for ordering_id, odf in full_df.groupby("ordering_id", sort=True):
        fits: dict[str, Any] = {}
        for (task, metric_name), gdf in odf.groupby(["task", "metric_name"], sort=True):
            key = f"{task}::{metric_name}"
            fits[key] = _fit_curve_models(gdf)
        per_ordering_fits[str(int(ordering_id))] = fits
        verdict = _ordering_verdict(fits)
        bic_delta = _mean_bic_delta(fits)
        bic_delta_log = _mean_bic_delta_pairwise(fits, left="logistic_sigmoid", right="threshold_piecewise")
        ordering_bic_delta.append(bic_delta)
        ordering_bic_delta_logistic.append(bic_delta_log)
        ordering_rows.append(
            {
                "ordering_id": int(ordering_id),
                "order_seed": int(ordering_payloads[int(ordering_id)]["order_seed"]) if int(ordering_id) in ordering_payloads else None,
                **verdict,
                "mean_bic_delta_linear_minus_threshold": safe_float(bic_delta),
                "mean_bic_delta_logistic_minus_threshold": safe_float(bic_delta_log),
            }
        )

    n_ord = len(ordering_rows)
    n_majority = sum(1 for r in ordering_rows if bool(r.get("threshold_majority", False)))
    frac_majority = float(n_majority / max(1, n_ord))

    if frac_majority >= 0.8:
        robustness = "strong_replication"
    elif frac_majority >= 0.5:
        robustness = "partial_replication"
    else:
        robustness = "failure"

    finite_bic = np.asarray([x for x in ordering_bic_delta if np.isfinite(x)], dtype=float)
    finite_bic_logistic = np.asarray([x for x in ordering_bic_delta_logistic if np.isfinite(x)], dtype=float)

    aggregate = {
        "timestamp": timestamp_now(),
        "experiment": "NEW-R12",
        "model": model_name,
        "n_orderings": int(n_ord),
        "ordering_id_start": int(start_id),
        "ordering_id_stop": int(stop_id),
        "n_threshold_majority": int(n_majority),
        "fraction_threshold_majority": frac_majority,
        "robustness_assessment": robustness,
        "acceptance_thresholds": {
            "strong_replication_ge": 0.8,
            "partial_replication_ge": 0.5,
        },
        "mean_random_bic_delta_linear_minus_threshold": safe_float(np.mean(finite_bic) if finite_bic.size else float("nan")),
        "std_random_bic_delta_linear_minus_threshold": safe_float(np.std(finite_bic, ddof=1) if finite_bic.size >= 2 else float("nan")),
        "mean_random_bic_delta_logistic_minus_threshold": safe_float(
            np.mean(finite_bic_logistic) if finite_bic_logistic.size else float("nan")
        ),
        "std_random_bic_delta_logistic_minus_threshold": safe_float(
            np.std(finite_bic_logistic, ddof=1) if finite_bic_logistic.size >= 2 else float("nan")
        ),
        "effective_batch_size_synth": {
            f"{task}::span{span}": int(bs)
            for (task, span), bs in sorted(synth_batch_ceiling_cache.items())
        },
        "effective_batch_size_ntp": int(ntp_batch_state.get("wiki_ntp", max(1, int(batch_size_ntp)))),
        "ntp_coverage": ntp_coverage if not skip_ntp else None,
    }

    canonical = _load_canonical_fit(model_name)
    canonical_vs_random: dict[str, Any] = {
        "timestamp": timestamp_now(),
        "experiment": "NEW-R12",
        "model": model_name,
        "canonical_curve_fit_path": None,
        "canonical_mean_bic_delta_linear_minus_threshold": float("nan"),
        "canonical_mean_bic_delta_logistic_minus_threshold": float("nan"),
        "random_mean_bic_delta_linear_minus_threshold": aggregate["mean_random_bic_delta_linear_minus_threshold"],
        "random_mean_bic_delta_logistic_minus_threshold": aggregate["mean_random_bic_delta_logistic_minus_threshold"],
        "random_distribution": [safe_float(x) for x in ordering_bic_delta],
        "random_distribution_logistic_minus_threshold": [safe_float(x) for x in ordering_bic_delta_logistic],
        "ranked_ordering_outlier": None,
    }
    if canonical is not None:
        canonical_fits = canonical.get("curve_fits", {})
        can_delta = _mean_bic_delta(canonical_fits)
        can_delta_log = _mean_bic_delta_pairwise(canonical_fits, left="logistic_sigmoid", right="threshold_piecewise")
        canonical_vs_random["canonical_mean_bic_delta_linear_minus_threshold"] = safe_float(can_delta)
        canonical_vs_random["canonical_mean_bic_delta_logistic_minus_threshold"] = safe_float(can_delta_log)
        canonical_vs_random["canonical_curve_fit_path"] = str(
            ROOT / "results" / "experiment3_phase2" / "exp3p2c_redundancy_quantification" / model_name / "curve_fit_comparison.json"
        )
        arr = np.asarray([x for x in ordering_bic_delta if np.isfinite(x)], dtype=float)
        if np.isfinite(can_delta) and arr.size >= 2:
            z = float((can_delta - float(np.mean(arr))) / max(float(np.std(arr, ddof=1)), 1e-8))
            canonical_vs_random["ranked_ordering_outlier"] = {
                "z_score": safe_float(z),
                "abs_z_gt_2": bool(abs(z) > 2.0),
            }

    write_json(out_dir / "ordering_vote_summary.json", {"rows": ordering_rows, "per_ordering_fits": per_ordering_fits})
    write_json(out_dir / "aggregate_robustness.json", aggregate)
    write_json(out_dir / "canonical_vs_random_comparison.json", canonical_vs_random)
    write_json(
        out_dir / "ordering_payloads.json",
        {"orderings": [ordering_payloads[k] for k in sorted(ordering_payloads.keys())]},
    )

    report = {
        "timestamp": timestamp_now(),
        "experiment": "NEW-R12",
        "model": model_name,
        "output_dir": str(out_dir),
        "runtime_sec": float(time.time() - t_model),
        "aggregate": aggregate,
        "canonical_vs_random": canonical_vs_random,
        "runtime_metadata": {
            "ordering_id_start": int(start_id),
            "ordering_id_stop": int(stop_id),
            "effective_batch_size_synth": aggregate.get("effective_batch_size_synth"),
            "effective_batch_size_ntp": aggregate.get("effective_batch_size_ntp"),
            "ntp_coverage": aggregate.get("ntp_coverage"),
        },
    }
    write_json(out_dir / "summary.json", report)
    return report


def _parse_device_map(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in [x.strip() for x in str(raw).split(",") if x.strip()]:
        model, dev = tok.split(":", 1)
        out[model.strip()] = dev.strip()
    return out


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.output_root)
    ensure_dir(out_root)
    device_map = _parse_device_map(args.device_map)
    reports: dict[str, Any] = {}

    for model in TARGET_MODELS:
        dev = device_map.get(model, "cuda:0")
        cmd = [
            str(ROOT / ".venv" / "bin" / "python"),
            "-u",
            "reinforce_exp/exp_new_r12_ordering_control.py",
            "--model",
            model,
            "--device",
            dev,
            "--output-root",
            str(out_root),
            "--fractions",
            str(args.fractions),
            "--num-orderings",
            str(args.num_orderings),
            "--ordering-seed-base",
            str(args.ordering_seed_base),
            "--ordering-id-start",
            str(args.ordering_id_start),
            "--num-seeds",
            str(args.num_seeds),
            "--synthetic-count",
            str(args.synthetic_count),
            "--batch-size-synth",
            str(args.batch_size_synth),
            "--ntp-count-per-seed",
            str(args.ntp_count_per_seed),
            "--ntp-seq-len",
            str(args.ntp_seq_len),
            "--batch-size-ntp",
            str(args.batch_size_ntp),
        ]
        if args.ordering_id_stop is not None:
            cmd.extend(["--ordering-id-stop", str(args.ordering_id_stop)])
        if bool(args.skip_ntp):
            cmd.append("--skip-ntp")
        print("[NEW-R12] exec:", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(ROOT), check=True)
        s_path = out_root / model / "summary.json"
        reports[model] = json.loads(s_path.read_text(encoding="utf-8"))

    payload = {"timestamp": timestamp_now(), "experiment": "NEW-R12", "models": reports}
    write_json(out_root / "aggregate_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NEW-R12: random-order ablation control for C.1")
    p.add_argument("--model", choices=["all", *TARGET_MODELS], default="all")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device-map", default="llama-3.1-8b:cuda:0,olmo-2-7b:cuda:1")
    p.add_argument("--output-root", default=str(DEFAULT_OUT))
    p.add_argument("--fractions", default="0,1,2,5,10,15,20,25,50")
    p.add_argument("--num-orderings", type=int, default=10)
    p.add_argument("--ordering-seed-base", type=int, default=20260417)
    p.add_argument("--ordering-id-start", type=int, default=0)
    p.add_argument("--ordering-id-stop", type=int, default=None)
    p.add_argument("--num-seeds", type=int, default=3)
    p.add_argument("--synthetic-count", type=int, default=100)
    p.add_argument("--batch-size-synth", type=int, default=8)
    p.add_argument("--ntp-count-per-seed", type=int, default=100)
    p.add_argument("--ntp-seq-len", type=int, default=512)
    p.add_argument("--batch-size-ntp", type=int, default=4)
    p.add_argument("--skip-ntp", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    ensure_dir(out_root)

    if args.model == "all":
        agg = run_all(args)
        write_json(
            out_root / "manifest.json",
            command_manifest(
                experiment_id="NEW-R12",
                command="all_models",
                model="+".join(TARGET_MODELS),
                extras={
                    "fractions": list(int(x) for x in _parse_int_tuple(args.fractions)),
                    "num_orderings": int(args.num_orderings),
                    "ordering_id_start": int(args.ordering_id_start),
                    "ordering_id_stop": int(args.ordering_id_stop) if args.ordering_id_stop is not None else None,
                    "num_seeds": int(args.num_seeds),
                    "output_root": str(out_root),
                },
            ),
        )
        print(f"[NEW-R12] wrote {out_root / 'aggregate_summary.json'}")
        print(f"[NEW-R12] models={list(agg['models'].keys())}")
        return

    rep = run_model(
        model_name=str(args.model),
        device=str(args.device),
        output_root=out_root,
        fractions=_parse_int_tuple(args.fractions),
        num_orderings=max(1, int(args.num_orderings)),
        ordering_seed_base=int(args.ordering_seed_base),
        ordering_id_start=max(0, int(args.ordering_id_start)),
        ordering_id_stop=int(args.ordering_id_stop) if args.ordering_id_stop is not None else None,
        num_seeds=max(1, int(args.num_seeds)),
        synthetic_count=max(1, int(args.synthetic_count)),
        batch_size_synth=max(1, int(args.batch_size_synth)),
        ntp_count_per_seed=max(1, int(args.ntp_count_per_seed)),
        ntp_seq_len=max(64, int(args.ntp_seq_len)),
        batch_size_ntp=max(1, int(args.batch_size_ntp)),
        skip_ntp=bool(args.skip_ntp),
    )
    write_json(
        out_root / str(args.model) / "manifest.json",
        command_manifest(
            experiment_id="NEW-R12",
            command="single_model",
            model=str(args.model),
            extras={
                "device": str(args.device),
                "fractions": list(int(x) for x in _parse_int_tuple(args.fractions)),
                "num_orderings": int(args.num_orderings),
                "ordering_id_start": int(args.ordering_id_start),
                "ordering_id_stop": int(args.ordering_id_stop) if args.ordering_id_stop is not None else None,
                "num_seeds": int(args.num_seeds),
                "output_root": str(out_root),
                "effective_batch_size_synth": rep.get("runtime_metadata", {}).get("effective_batch_size_synth"),
                "effective_batch_size_ntp": rep.get("runtime_metadata", {}).get("effective_batch_size_ntp"),
                "ntp_coverage": rep.get("runtime_metadata", {}).get("ntp_coverage"),
            },
        ),
    )
    print(f"[NEW-R12] wrote {out_root / str(args.model) / 'summary.json'}")
    print(f"[NEW-R12] robustness={rep['aggregate']['robustness_assessment']}")


if __name__ == "__main__":
    main()
