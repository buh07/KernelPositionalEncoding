#!/usr/bin/env python3
"""E17 — Norm-Matched Specificity Control for SI Kernel Subtraction.

Extends permutation specificity with a third arm: norm-matched random kernel
perturbations. Tests whether true SI-kernel subtraction produces larger
performance degradation than both offset-permuted and norm-matched random
perturbations on the same high-SI head set.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp3.common import (  # noqa: E402
    PRIMARY_MODELS,
    RESULTS_ROOT,
    ensure_dir,
    timestamp_now,
    write_json,
)
from reinforce_exp3.scripts._shared import (  # noqa: E402
    enforce_coverage_contract,
    emit_core_artifacts,
    load_model_for_exp,
    parse_device_map,
    parse_models_arg,
    read_json,
)
from experiment3.theory8_position_ablation import (  # noqa: E402
    MODELS,
    compute_per_token_loss,
    load_head_groups,
    load_wiki_sequences,
    subtract_positional_kernels,
)

OUT_ROOT = RESULTS_ROOT / "E17_normmatched_specificity"


def _load_kernels(model_name: str) -> tuple[dict[tuple[int, int], np.ndarray], str]:
    candidates = [
        ROOT / "results" / "reinforce_exp" / "exp_r3_core_replication" / model_name / "theory8_position_ablation" / model_name / "estimated_kernels.json",
        ROOT / "results" / "experiment3" / "theory8_position_ablation" / model_name / "estimated_kernels.json",
    ]
    for p in candidates:
        if not p.exists():
            continue
        raw = json.loads(p.read_text(encoding="utf-8"))
        out: dict[tuple[int, int], np.ndarray] = {}
        for k, v in raw.items():
            if not (k.startswith("L") and "H" in k):
                continue
            left, right = k[1:].split("H", 1)
            out[(int(left), int(right))] = np.asarray(v, dtype=np.float32)
        if out:
            return out, str(p)
    raise FileNotFoundError(
        f"[E17] No estimated_kernels.json found for {model_name}; checked: {candidates}"
    )


def _permute_kernels(
    kernels: dict[tuple[int, int], np.ndarray],
    heads: list[tuple[int, int]],
    rng: np.random.Generator,
) -> dict[tuple[int, int], np.ndarray]:
    out: dict[tuple[int, int], np.ndarray] = {}
    for head in heads:
        g = kernels.get(head)
        if g is None:
            continue
        idx = rng.permutation(len(g))
        out[head] = g[idx].copy()
    return out


def _norm_matched_random_kernels(
    kernels: dict[tuple[int, int], np.ndarray],
    heads: list[tuple[int, int]],
    rng: np.random.Generator,
) -> dict[tuple[int, int], np.ndarray]:
    out: dict[tuple[int, int], np.ndarray] = {}
    for head in heads:
        g = kernels.get(head)
        if g is None:
            continue
        n = int(len(g))
        if n <= 0:
            continue
        z = rng.standard_normal(n).astype(np.float32)
        z -= float(np.mean(z))
        zn = float(np.linalg.norm(z))
        gn = float(np.linalg.norm(g))
        if zn <= 1e-12 or gn <= 1e-12:
            out[head] = np.zeros_like(g, dtype=np.float32)
        else:
            out[head] = (z * (gn / zn)).astype(np.float32)
    return out


def _eval_sequence_mean_losses(
    model: Any,
    sequences: list[list[int]],
    device: str,
    batch_size: int,
) -> np.ndarray:
    vals: list[float] = []
    bs = max(1, int(batch_size))
    pos = 0
    while pos < len(sequences):
        batch = sequences[pos : pos + bs]
        try:
            input_ids = torch.tensor(batch, dtype=torch.long, device=device)
            with torch.inference_mode():
                loss = compute_per_token_loss(model, input_ids)
            tok_per_seq = int(input_ids.shape[1] - 1)
            seq_loss = (
                loss.view(input_ids.shape[0], tok_per_seq)
                .mean(dim=1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            vals.extend(float(x) for x in seq_loss.tolist())
            pos += len(batch)
            del input_ids, loss
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and bs > 1:
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                continue
            raise
    return np.asarray(vals, dtype=np.float64)


def _bootstrap_diff_ci(diff: np.ndarray, *, n_boot: int, seed: int) -> dict[str, float]:
    if diff.size == 0:
        return {
            "mean": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "p_one_gt_zero": float("nan"),
        }
    rng = np.random.default_rng(int(seed))
    n = int(diff.size)
    m = max(2000, int(n_boot))
    boot = np.empty(m, dtype=np.float64)
    for i in range(m):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(np.mean(diff[idx]))
    return {
        "mean": float(np.mean(diff)),
        "ci_lo": float(np.quantile(boot, 0.025)),
        "ci_hi": float(np.quantile(boot, 0.975)),
        "p_one_gt_zero": float((np.sum(boot <= 0.0) + 1) / (m + 1)),
    }


def _bootstrap_arm_diff_ci(
    true_delta: np.ndarray,
    ctrl_stack: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    """Hierarchical bootstrap over sequences and control trials.

    true_delta: [n_seq]
    ctrl_stack: [n_trials, n_seq]
    """
    td = np.asarray(true_delta, dtype=np.float64)
    cs = np.asarray(ctrl_stack, dtype=np.float64)
    if td.ndim != 1 or cs.ndim != 2:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "p_one_gt_zero": float("nan")}
    n_seq = int(td.shape[0])
    n_trials = int(cs.shape[0])
    if n_seq == 0 or n_trials == 0 or cs.shape[1] != n_seq:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "p_one_gt_zero": float("nan")}

    obs = float(np.mean(td) - np.mean(cs))
    rng = np.random.default_rng(int(seed))
    m = max(2000, int(n_boot))
    boot = np.empty(m, dtype=np.float64)
    for i in range(m):
        seq_idx = rng.integers(0, n_seq, size=n_seq)
        trial_idx = rng.integers(0, n_trials, size=n_trials)
        true_mean = float(np.mean(td[seq_idx]))
        ctrl_mean = float(np.mean(cs[trial_idx][:, seq_idx]))
        boot[i] = true_mean - ctrl_mean
    return {
        "mean": obs,
        "ci_lo": float(np.quantile(boot, 0.025)),
        "ci_hi": float(np.quantile(boot, 0.975)),
        "p_one_gt_zero": float((np.sum(boot <= 0.0) + 1) / (m + 1)),
    }


def _set_eager_attention(model: Any) -> None:
    try:
        model.config._attn_implementation = "eager"
    except Exception:
        pass
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            try:
                layer.self_attn.config._attn_implementation = "eager"
            except Exception:
                continue


def run_model(
    *,
    model_name: str,
    device: str,
    out_root: Path,
    eval_seqs: int,
    seq_len: int,
    n_permutations: int,
    seed: int,
    batch_size: int,
    n_boot: int,
    per_head_eval_count: int,
    per_head_eval_seqs: int,
) -> dict[str, Any]:
    t0 = time.time()
    model_dir = ensure_dir(out_root / model_name)

    # Use non-cached loader path for eager attention compatibility.
    model, _tokenizer_unused = load_model_for_exp(model_name, device, attn_implementation="eager")
    _set_eager_attention(model)

    head_groups = load_head_groups(model_name)
    high_heads = [(int(l), int(h)) for (l, h) in head_groups["high_si"]]
    kernels, kernel_path = _load_kernels(model_name)

    sequences = load_wiki_sequences(model_name, max_sequences=max(8, int(eval_seqs)), seq_len=max(64, int(seq_len)))
    sequences = [seq[: int(seq_len)] for seq in sequences[: int(eval_seqs)]]
    if len(sequences) < 4:
        raise RuntimeError(f"[E17] insufficient sequences ({len(sequences)}) for model={model_name}")

    baseline = _eval_sequence_mean_losses(model=model, sequences=sequences, device=device, batch_size=batch_size)

    with subtract_positional_kernels(model, kernels, high_heads, int(seq_len)):
        true_loss = _eval_sequence_mean_losses(model=model, sequences=sequences, device=device, batch_size=batch_size)
    true_delta = true_loss - baseline

    perm_rows: list[dict[str, Any]] = []
    norm_rows: list[dict[str, Any]] = []
    perm_stack: list[np.ndarray] = []
    norm_stack: list[np.ndarray] = []

    for idx in range(max(1, int(n_permutations))):
        perm_seed = int(seed) + idx * 7919 + 11
        norm_seed = int(seed) + idx * 7919 + 29
        rng_perm = np.random.default_rng(perm_seed)
        rng_norm = np.random.default_rng(norm_seed)

        perm_k = _permute_kernels(kernels, high_heads, rng_perm)
        with subtract_positional_kernels(model, perm_k, high_heads, int(seq_len)):
            perm_loss = _eval_sequence_mean_losses(model=model, sequences=sequences, device=device, batch_size=batch_size)
        perm_delta = perm_loss - baseline
        perm_stack.append(perm_delta)
        perm_rows.append(
            {
                "trial": int(idx),
                "trial_seed": int(perm_seed),
                "mean_loss_delta": float(np.mean(perm_delta)),
                "std_loss_delta": float(np.std(perm_delta, ddof=1)) if len(perm_delta) > 1 else float("nan"),
            }
        )

        norm_k = _norm_matched_random_kernels(kernels, high_heads, rng_norm)
        with subtract_positional_kernels(model, norm_k, high_heads, int(seq_len)):
            norm_loss = _eval_sequence_mean_losses(model=model, sequences=sequences, device=device, batch_size=batch_size)
        norm_delta = norm_loss - baseline
        norm_stack.append(norm_delta)
        norm_rows.append(
            {
                "trial": int(idx),
                "trial_seed": int(norm_seed),
                "mean_loss_delta": float(np.mean(norm_delta)),
                "std_loss_delta": float(np.std(norm_delta, ddof=1)) if len(norm_delta) > 1 else float("nan"),
            }
        )
        print(
            f"[E17] {model_name} trial={idx+1}/{n_permutations} "
            f"true={float(np.mean(true_delta)):.6f} "
            f"perm={float(np.mean(perm_delta)):.6f} norm={float(np.mean(norm_delta)):.6f}",
            flush=True,
        )

    perm_mean_seq = np.mean(np.stack(perm_stack, axis=0), axis=0)
    norm_mean_seq = np.mean(np.stack(norm_stack, axis=0), axis=0)

    diff_true_perm = true_delta - perm_mean_seq
    diff_true_norm = true_delta - norm_mean_seq

    perm_stack_arr = np.stack(perm_stack, axis=0)
    norm_stack_arr = np.stack(norm_stack, axis=0)
    ci_perm = _bootstrap_arm_diff_ci(true_delta, perm_stack_arr, n_boot=n_boot, seed=seed + 101)
    ci_norm = _bootstrap_arm_diff_ci(true_delta, norm_stack_arr, n_boot=n_boot, seed=seed + 202)

    true_mean = float(np.mean(true_delta))
    perm_mean = float(np.mean(perm_mean_seq))
    norm_mean = float(np.mean(norm_mean_seq))
    ratio_perm = true_mean / max(abs(perm_mean), 1e-8)
    ratio_norm = true_mean / max(abs(norm_mean), 1e-8)

    # Optional per-head breakdown (first K high-SI heads).
    per_head_rows: list[dict[str, Any]] = []
    small_n = max(4, int(per_head_eval_seqs))
    seq_small = sequences[:small_n]
    base_small = baseline[:small_n]
    for hidx, head in enumerate(high_heads[: max(0, int(per_head_eval_count))]):
        with subtract_positional_kernels(model, kernels, [head], int(seq_len)):
            hi_loss = _eval_sequence_mean_losses(model=model, sequences=seq_small, device=device, batch_size=batch_size)
        hi_delta = hi_loss - base_small

        rng_h = np.random.default_rng(seed + 3000 + hidx)
        perm_h = _permute_kernels(kernels, [head], rng_h)
        with subtract_positional_kernels(model, perm_h, [head], int(seq_len)):
            perm_h_loss = _eval_sequence_mean_losses(model=model, sequences=seq_small, device=device, batch_size=batch_size)
        perm_h_delta = perm_h_loss - base_small

        norm_h = _norm_matched_random_kernels(kernels, [head], rng_h)
        with subtract_positional_kernels(model, norm_h, [head], int(seq_len)):
            norm_h_loss = _eval_sequence_mean_losses(model=model, sequences=seq_small, device=device, batch_size=batch_size)
        norm_h_delta = norm_h_loss - base_small

        per_head_rows.append(
            {
                "layer": int(head[0]),
                "head": int(head[1]),
                "true_mean_delta": float(np.mean(hi_delta)),
                "permuted_mean_delta": float(np.mean(perm_h_delta)),
                "normmatched_mean_delta": float(np.mean(norm_h_delta)),
                "ratio_true_over_permuted": float(np.mean(hi_delta) / max(abs(float(np.mean(perm_h_delta))), 1e-8)),
                "ratio_true_over_normmatched": float(np.mean(hi_delta) / max(abs(float(np.mean(norm_h_delta))), 1e-8)),
            }
        )

    pd.DataFrame(perm_rows).to_parquet(model_dir / "permuted_arm_trials.parquet", index=False)
    pd.DataFrame(norm_rows).to_parquet(model_dir / "normmatched_arm_trials.parquet", index=False)
    pd.DataFrame(per_head_rows).to_parquet(model_dir / "per_head_specificity.parquet", index=False)

    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": "E17",
        "model": model_name,
        "runtime_sec": float(time.time() - t0),
        "n_sequences": int(len(sequences)),
        "n_high_si_heads": int(len(high_heads)),
        "kernel_source": kernel_path,
        "headline": {
            "true_mean_loss_delta": true_mean,
            "permuted_mean_loss_delta": perm_mean,
            "normmatched_mean_loss_delta": norm_mean,
            "ratio_true_over_permuted": ratio_perm,
            "ratio_true_over_normmatched": ratio_norm,
            "true_minus_permuted_ci95": [ci_perm["ci_lo"], ci_perm["ci_hi"]],
            "true_minus_normmatched_ci95": [ci_norm["ci_lo"], ci_norm["ci_hi"]],
            "supports_specificity": bool(ci_perm["ci_lo"] > 0.0 and ci_norm["ci_lo"] > 0.0),
        },
    }
    write_json(model_dir / "summary.json", summary)
    write_json(
        model_dir / "specificity_tests.json",
        {
            "true_vs_permuted": ci_perm,
            "true_vs_normmatched": ci_norm,
            "ratio_true_over_permuted": ratio_perm,
            "ratio_true_over_normmatched": ratio_norm,
        },
    )
    return summary


def _load_model_summary(model_name: str, out_root: Path) -> dict[str, Any]:
    p = out_root / model_name / "summary.json"
    if not p.exists():
        raise RuntimeError(f"[E17] hard_fail_reason: missing shard artifact {p}")
    return read_json(p)


def _cross_model_finalize(models: list[str], out_root: Path, start_ts: str) -> dict[str, Any]:
    rows = [_load_model_summary(m, out_root) for m in models]
    enforce_coverage_contract(
        experiment_id="E17",
        observed_models=[r.get("model", "") for r in rows],
        required_models=models,
    )

    supports = [bool(r.get("headline", {}).get("supports_specificity", False)) for r in rows]
    ratios_perm = [float(r.get("headline", {}).get("ratio_true_over_permuted", float("nan"))) for r in rows]
    ratios_norm = [float(r.get("headline", {}).get("ratio_true_over_normmatched", float("nan"))) for r in rows]

    n_supported = int(sum(1 for x in supports if x))
    if n_supported == len(rows):
        claim_status = "supported"
        interp = "all_models_specific"
        note = "True SI-kernel subtraction exceeds both permuted and norm-matched controls in all models."
    elif n_supported >= max(1, len(rows) - 1):
        claim_status = "supported_with_caveat"
        interp = "mostly_specific"
        note = "Specificity holds in most models; one model remains borderline."
    elif n_supported >= 1:
        claim_status = "mixed"
        interp = "partial_specific"
        note = "Specificity is model-conditional under norm-matched controls."
    else:
        claim_status = "not_supported"
        interp = "not_specific"
        note = "True subtraction does not consistently exceed norm-matched controls."

    cross_model = {
        "timestamp": timestamp_now(),
        "experiment_id": "E17",
        "n_models": int(len(rows)),
        "n_supported": int(n_supported),
        "mean_ratio_true_over_permuted": float(np.nanmean(np.asarray(ratios_perm, dtype=float))),
        "mean_ratio_true_over_normmatched": float(np.nanmean(np.asarray(ratios_norm, dtype=float))),
        "interpretation": interp,
        "note": note,
        "per_model": rows,
    }
    write_json(out_root / "cross_model_specificity_summary.json", cross_model)

    claim_impact = {
        "timestamp": timestamp_now(),
        "experiment_id": "E17",
        "claim_status": claim_status,
        "supports_main_text": claim_status in ("supported", "supported_with_caveat", "mixed"),
        "notes": [note],
        "outcome_summary": note,
    }

    prereg = {
        "experiment_id": "E17",
        "question": "Does SI-kernel subtraction exceed both offset-permuted and norm-matched random perturbation controls?",
        "primary_hypothesis": "True SI subtraction causes larger loss increase than both controls.",
        "primary_endpoints": [
            "true_minus_permuted_mean_delta",
            "true_minus_normmatched_mean_delta",
        ],
        "secondary_endpoints": [
            "ratio_true_over_permuted",
            "ratio_true_over_normmatched",
            "per_head_specificity_subset",
        ],
        "model_list": models,
        "dataset_sources": ["wiki40b_en_pre2019 tokenized sequences (existing local cache)"],
        "inclusion_exclusion_rules": [
            "Require >=4 evaluation sequences per model",
            "Use top-quartile high-SI head set from existing head groups",
        ],
        "sample_size_plan": {
            "eval_sequences_per_model": "configured by --eval-seqs",
            "permutation_trials": "configured by --n-permutations",
        },
        "seed_plan": {"base_seed": 20260501},
        "stopping_rule": "Run fixed configured trials; no adaptive stopping",
        "multiplicity_family": ["per-model bootstrap comparisons"],
        "acceptance_criteria": [
            "ci95(true-permuted) > 0 and ci95(true-normmatched) > 0 in all models => supported",
        ],
        "fallback_interpretation_if_null": "Specificity remains incomplete; treat T8 as broader head-importance perturbation.",
    }

    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": "E17",
        "analysis_tier": "confirmatory",
        "canonical_eligible": True,
        "override_used": False,
        "verdict": {
            "interpretation": interp,
            "n_supported": int(n_supported),
            "n_models": int(len(rows)),
        },
        "limitations": [
            "Norm-matched control perturbs Toeplitz kernels in logit space and still induces softmax renormalization.",
        ],
    }

    data_dictionary = {
        "experiment_id": "E17",
        "tables": [
            {
                "path": "<model>/permuted_arm_trials.parquet",
                "description": "Per-trial mean loss deltas for offset-permuted control arm.",
                "columns": [
                    {"name": "trial", "dtype": "int", "description": "trial index"},
                    {"name": "trial_seed", "dtype": "int", "description": "trial seed"},
                    {"name": "mean_loss_delta", "dtype": "float", "description": "mean per-sequence loss increase"},
                    {"name": "std_loss_delta", "dtype": "float", "description": "std per-sequence loss increase"},
                ],
            },
            {
                "path": "<model>/normmatched_arm_trials.parquet",
                "description": "Per-trial mean loss deltas for norm-matched random control arm.",
                "columns": [
                    {"name": "trial", "dtype": "int", "description": "trial index"},
                    {"name": "trial_seed", "dtype": "int", "description": "trial seed"},
                    {"name": "mean_loss_delta", "dtype": "float", "description": "mean per-sequence loss increase"},
                    {"name": "std_loss_delta", "dtype": "float", "description": "std per-sequence loss increase"},
                ],
            },
            {
                "path": "<model>/per_head_specificity.parquet",
                "description": "Per-head subset comparison between true/permuted/normmatched deltas.",
                "columns": [
                    {"name": "layer", "dtype": "int", "description": "layer index"},
                    {"name": "head", "dtype": "int", "description": "head index"},
                    {"name": "true_mean_delta", "dtype": "float", "description": "true SI subtraction mean delta"},
                    {"name": "permuted_mean_delta", "dtype": "float", "description": "permuted control mean delta"},
                    {"name": "normmatched_mean_delta", "dtype": "float", "description": "norm-matched control mean delta"},
                    {"name": "ratio_true_over_permuted", "dtype": "float", "description": "true/permuted ratio"},
                    {"name": "ratio_true_over_normmatched", "dtype": "float", "description": "true/normmatched ratio"},
                ],
            },
        ],
    }

    emit_core_artifacts(
        experiment_id="E17",
        out_dir=out_root,
        preregistration=prereg,
        manifest_extra={"models": models},
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )
    return cross_model


def main() -> None:
    p = argparse.ArgumentParser(description="E17: norm-matched specificity control", allow_abbrev=False)
    p.add_argument("--models", default=",".join(PRIMARY_MODELS))
    p.add_argument(
        "--device-map",
        default="llama-3.1-8b:cuda:0,olmo-2-7b:cuda:0,mistral-7b-v0.1:cuda:0",
    )
    p.add_argument("--output-root", default=str(OUT_ROOT))
    p.add_argument("--eval-seqs", type=int, default=100)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--n-permutations", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260501)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--per-head-eval-count", type=int, default=12)
    p.add_argument("--per-head-eval-seqs", type=int, default=32)
    p.add_argument("--smoke", action="store_true", help="Low-cost smoke mode")
    p.add_argument("--finalize-only", action="store_true")
    p.add_argument("--no-finalize", action="store_true")
    args = p.parse_args()

    if args.finalize_only and args.no_finalize:
        raise RuntimeError("[E17] hard_fail_reason: --finalize-only and --no-finalize are mutually exclusive")

    models = parse_models_arg(args.models, default=PRIMARY_MODELS)
    device_map = parse_device_map(args.device_map)
    out_root = ensure_dir(Path(args.output_root))
    start_ts = timestamp_now()

    eval_seqs = int(args.eval_seqs)
    n_perms = int(args.n_permutations)
    n_boot = int(args.n_boot)
    per_head_eval_count = int(args.per_head_eval_count)
    per_head_eval_seqs = int(args.per_head_eval_seqs)
    if args.smoke:
        eval_seqs = min(eval_seqs, 16)
        n_perms = min(n_perms, 2)
        n_boot = min(n_boot, 1000)
        per_head_eval_count = min(per_head_eval_count, 4)
        per_head_eval_seqs = min(per_head_eval_seqs, 8)

    if args.finalize_only:
        cross = _cross_model_finalize(models, out_root, start_ts)
        print(f"[E17] Finalized from shards. interpretation={cross['interpretation']}", flush=True)
        return

    for model_name in models:
        device = device_map.get(model_name, "cuda:0")
        run_model(
            model_name=model_name,
            device=device,
            out_root=out_root,
            eval_seqs=max(8, eval_seqs),
            seq_len=max(64, int(args.seq_len)),
            n_permutations=max(1, n_perms),
            seed=int(args.seed),
            batch_size=max(1, int(args.batch_size)),
            n_boot=max(1000, n_boot),
            per_head_eval_count=max(0, per_head_eval_count),
            per_head_eval_seqs=max(4, per_head_eval_seqs),
        )

    if args.no_finalize:
        print("[E17] Shard run complete (no finalize).", flush=True)
        return

    cross = _cross_model_finalize(models, out_root, start_ts)
    print(f"[E17] Done. interpretation={cross['interpretation']}", flush=True)


if __name__ == "__main__":
    main()
