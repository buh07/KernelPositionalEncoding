#!/usr/bin/env python3
"""E29A — Kernel-transplant specificity test for Result-I tautology hardening.

Idea
----
If disruption sensitivity is SI-structure-specific (not merely generic importance),
then applying transplanted high-SI kernels onto low-SI heads should increase
low-head disruption beyond low-self kernels (with norm-matching).

Per model, evaluate on wiki sequences:
  - low_self                  : subtract low-head kernels on low heads
  - low_transplant_normmatched: subtract transplanted high-head kernels on low heads
  - low_random_normmatched    : subtract norm-matched random kernels on low heads
  - high_self                 : reference subtraction on high heads

Primary endpoint
----------------
  delta_transplant_vs_lowself = low_transplant_normmatched - low_self
Pass if bootstrap CI lower bound > 0 and one-sided p < 0.05.

Cross-model interpretation
--------------------------
  supported if Llama and Mistral pass; OLMo is directional anchor.
"""
from __future__ import annotations

import argparse
import json
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
    PRIMARY_MODELS,
    RESULTS_ROOT,
    ensure_dir,
    timestamp_now,
    write_json,
)
from reinforce_exp3.scripts._shared import (  # noqa: E402
    emit_core_artifacts,
    enforce_coverage_contract,
    load_model_for_exp,
    parse_device_map,
    parse_models_arg,
)
from experiment3.theory8_position_ablation import (  # noqa: E402
    compute_per_token_loss,
    load_head_groups,
    load_wiki_sequences,
    subtract_positional_kernels,
)

EXPERIMENT_ID = "E29A"
DEFAULT_OUT = RESULTS_ROOT / "E29a_kernel_transplant_specificity"

_KERNEL_CANDIDATES = [
    "results/reinforce_exp/exp_r3_core_replication/{model}/theory8_position_ablation/{model}/estimated_kernels.json",
    "results/experiment3/theory8_position_ablation/{model}/estimated_kernels.json",
]


def _load_kernels(model_name: str) -> dict[tuple[int, int], np.ndarray]:
    for template in _KERNEL_CANDIDATES:
        p = ROOT / template.format(model=model_name)
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
            return out
    raise FileNotFoundError(f"[E29A] No estimated_kernels.json for {model_name}")


def _eval_mean_losses(model: Any, sequences: list[list[int]], device: str, batch_size: int) -> np.ndarray:
    vals: list[float] = []
    bs = max(1, int(batch_size))
    pos = 0
    while pos < len(sequences):
        batch = sequences[pos : pos + bs]
        try:
            input_ids = torch.tensor(batch, dtype=torch.long, device=device)
            with torch.inference_mode():
                loss = compute_per_token_loss(model, input_ids)
            tok = int(input_ids.shape[1] - 1)
            seq_loss = loss.view(input_ids.shape[0], tok).mean(dim=1).detach().cpu().numpy().astype(np.float64)
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


def _norm_match_to(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    src = np.asarray(source, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.float32)
    src = src - float(np.mean(src))
    s_norm = float(np.linalg.norm(src))
    t_norm = float(np.linalg.norm(tgt))
    if s_norm <= 1e-12 or t_norm <= 1e-12:
        return np.zeros_like(tgt)
    return (src * (t_norm / s_norm)).astype(np.float32)


def _build_transplant_kernels(
    *,
    kernels: dict[tuple[int, int], np.ndarray],
    high_heads: list[tuple[int, int]],
    low_heads: list[tuple[int, int]],
) -> dict[tuple[int, int], np.ndarray]:
    """Map high-head kernels onto low heads with per-head norm matching.

    Matching is done by sorted layer/head order and reused modulo high set size.
    """
    if not high_heads or not low_heads:
        return {}
    hs = sorted(high_heads)
    ls = sorted(low_heads)
    out: dict[tuple[int, int], np.ndarray] = {}
    for i, lhead in enumerate(ls):
        hhead = hs[i % len(hs)]
        if hhead not in kernels or lhead not in kernels:
            continue
        out[lhead] = _norm_match_to(kernels[lhead], kernels[hhead])
    return out


def _build_random_normmatched_kernels(
    *,
    kernels: dict[tuple[int, int], np.ndarray],
    low_heads: list[tuple[int, int]],
    seed: int,
) -> dict[tuple[int, int], np.ndarray]:
    rng = np.random.default_rng(int(seed))
    out: dict[tuple[int, int], np.ndarray] = {}
    for head in low_heads:
        g = kernels.get(head)
        if g is None:
            continue
        z = rng.standard_normal(len(g)).astype(np.float32)
        out[head] = _norm_match_to(g, z)
    return out


def _bootstrap_diff_ci(diff: np.ndarray, n_boot: int = 5000, seed: int = 20260504) -> dict[str, float]:
    d = np.asarray(diff, dtype=np.float64)
    if d.size == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "p_one_gt_zero": float("nan")}
    rng = np.random.default_rng(int(seed))
    n = int(d.size)
    boot = np.empty(max(1000, int(n_boot)), dtype=np.float64)
    for i in range(boot.size):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(np.mean(d[idx]))
    # one-sided p for H1: mean(diff) > 0
    p_one = float((np.sum(boot <= 0.0) + 1) / (boot.size + 1))
    return {
        "mean": float(np.mean(d)),
        "ci_lo": float(np.quantile(boot, 0.025)),
        "ci_hi": float(np.quantile(boot, 0.975)),
        "p_one_gt_zero": p_one,
    }


def run_model(
    *,
    model_name: str,
    device: str,
    out_root: Path,
    eval_seqs: int,
    seq_len: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    t0 = time.time()
    model_dir = ensure_dir(out_root / model_name)

    model, _tok = load_model_for_exp(model_name, device, attn_implementation="eager")
    kernels = _load_kernels(model_name)

    head_groups = load_head_groups(model_name)
    high_heads = [(int(l), int(h)) for (l, h) in head_groups.get("high_si", []) if (int(l), int(h)) in kernels]
    low_heads = [(int(l), int(h)) for (l, h) in head_groups.get("low_si", []) if (int(l), int(h)) in kernels]
    if not high_heads or not low_heads:
        raise RuntimeError(f"[E29A] missing high/low SI heads for {model_name}")

    # layer-matched paired subset to reduce layer confound
    high_by_layer: dict[int, list[tuple[int, int]]] = {}
    low_by_layer: dict[int, list[tuple[int, int]]] = {}
    for lh in high_heads:
        high_by_layer.setdefault(lh[0], []).append(lh)
    for lh in low_heads:
        low_by_layer.setdefault(lh[0], []).append(lh)

    matched_high: list[tuple[int, int]] = []
    matched_low: list[tuple[int, int]] = []
    for layer in sorted(set(high_by_layer.keys()) & set(low_by_layer.keys())):
        hs = sorted(high_by_layer[layer])
        ls = sorted(low_by_layer[layer])
        k = min(len(hs), len(ls))
        matched_high.extend(hs[:k])
        matched_low.extend(ls[:k])

    if len(matched_low) < 32:
        raise RuntimeError(f"[E29A] insufficient matched high/low pairs for {model_name}: {len(matched_low)}")

    sequences = load_wiki_sequences(model_name, max_sequences=max(8, eval_seqs), seq_len=max(64, seq_len))
    sequences = [seq[:seq_len] for seq in sequences[:eval_seqs]]
    if len(sequences) < 4:
        raise RuntimeError(f"[E29A] insufficient sequences for {model_name}: {len(sequences)}")

    baseline = _eval_mean_losses(model, sequences, device, batch_size)

    # low self
    low_self_k = {h: kernels[h] for h in matched_low if h in kernels}
    with subtract_positional_kernels(model, low_self_k, matched_low, int(seq_len)):
        loss_low_self = _eval_mean_losses(model, sequences, device, batch_size)

    # low transplant (high->low, norm-matched to low norm)
    low_transplant_k = _build_transplant_kernels(kernels=kernels, high_heads=matched_high, low_heads=matched_low)
    with subtract_positional_kernels(model, low_transplant_k, matched_low, int(seq_len)):
        loss_low_transplant = _eval_mean_losses(model, sequences, device, batch_size)

    # low random norm-matched
    low_rand_k = _build_random_normmatched_kernels(kernels=kernels, low_heads=matched_low, seed=seed)
    with subtract_positional_kernels(model, low_rand_k, matched_low, int(seq_len)):
        loss_low_rand = _eval_mean_losses(model, sequences, device, batch_size)

    # high self reference
    high_self_k = {h: kernels[h] for h in matched_high if h in kernels}
    with subtract_positional_kernels(model, high_self_k, matched_high, int(seq_len)):
        loss_high_self = _eval_mean_losses(model, sequences, device, batch_size)

    delta_low_self = loss_low_self - baseline
    delta_low_transplant = loss_low_transplant - baseline
    delta_low_rand = loss_low_rand - baseline
    delta_high_self = loss_high_self - baseline

    diff_trans_vs_low = delta_low_transplant - delta_low_self
    ci = _bootstrap_diff_ci(diff_trans_vs_low, seed=seed + 11)

    trial_df = pd.DataFrame(
        {
            "seq_idx": np.arange(len(delta_low_self), dtype=int),
            "delta_low_self": delta_low_self,
            "delta_low_transplant_normmatched": delta_low_transplant,
            "delta_low_random_normmatched": delta_low_rand,
            "delta_high_self": delta_high_self,
            "diff_transplant_minus_lowself": diff_trans_vs_low,
        }
    )
    trial_df.to_parquet(model_dir / "kernel_transplant_trials.parquet", index=False)

    write_json(
        model_dir / "headset_summary.json",
        {
            "n_matched_pairs": int(len(matched_low)),
            "matched_high_heads": [[int(l), int(h)] for (l, h) in matched_high],
            "matched_low_heads": [[int(l), int(h)] for (l, h) in matched_low],
        },
    )

    rec = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "model": model_name,
        "n_eval_seqs": int(len(sequences)),
        "n_matched_pairs": int(len(matched_low)),
        "low_self_mean_delta": float(np.mean(delta_low_self)),
        "low_transplant_normmatched_mean_delta": float(np.mean(delta_low_transplant)),
        "low_random_normmatched_mean_delta": float(np.mean(delta_low_rand)),
        "high_self_mean_delta": float(np.mean(delta_high_self)),
        "delta_transplant_vs_lowself": float(np.mean(diff_trans_vs_low)),
        "ci_delta_transplant_vs_lowself": [float(ci["ci_lo"]), float(ci["ci_hi"])],
        "p_one_gt_zero": float(ci["p_one_gt_zero"]),
        "pass_model": bool(float(ci["ci_lo"]) > 0.0 and float(ci["p_one_gt_zero"]) < 0.05),
        "elapsed_sec": float(time.time() - t0),
    }
    write_json(model_dir / "summary.json", rec)
    print(
        f"[E29A][{model_name}] d(trans-low)={rec['delta_transplant_vs_lowself']:.6f} "
        f"ci=[{ci['ci_lo']:.6f},{ci['ci_hi']:.6f}] p_one={ci['p_one_gt_zero']:.6f} pass={rec['pass_model']}",
        flush=True,
    )
    return rec


def _finalize(models: list[str], out_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for m in models:
        p = out_root / m / "summary.json"
        if not p.exists():
            raise RuntimeError(f"[E29A] hard_fail_reason: missing summary for {m}: {p}")
        rows.append(json.loads(p.read_text(encoding="utf-8")))

    enforce_coverage_contract(
        experiment_id=EXPERIMENT_ID,
        observed_models=[r.get("model", "") for r in rows],
        required_models=models,
    )

    by_model = {r["model"]: r for r in rows}
    lm_pass = bool(by_model.get("llama-3.1-8b", {}).get("pass_model", False))
    ms_pass = bool(by_model.get("mistral-7b-v0.1", {}).get("pass_model", False))
    ol_pass = bool(by_model.get("olmo-2-7b", {}).get("pass_model", False))

    if lm_pass and ms_pass:
        if ol_pass:
            interp = "kernel_transplant_specificity_supported_all_models"
            claim_status = "supported"
        else:
            interp = "kernel_transplant_specificity_supported_core_models"
            claim_status = "supported_with_caveat"
    elif lm_pass or ms_pass:
        interp = "kernel_transplant_specificity_mixed"
        claim_status = "mixed"
    else:
        interp = "kernel_transplant_specificity_not_supported"
        claim_status = "not_supported"

    cross = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "interpretation": interp,
        "claim_status": claim_status,
        "n_models": int(len(rows)),
        "n_pass": int(sum(1 for r in rows if bool(r.get("pass_model", False)))),
        "per_model": rows,
    }
    write_json(out_root / "cross_model_summary.json", cross)

    prereg = {
        "experiment_id": EXPERIMENT_ID,
        "question": "Does transplanted SI kernel structure increase low-head disruption beyond low-self kernels?",
        "primary_hypothesis": "delta_transplant_vs_lowself > 0 (one-sided) under norm-matched transplant.",
        "primary_endpoints": [
            "delta_transplant_vs_lowself",
            "bootstrap_CI_delta_transplant_vs_lowself",
            "p_one_gt_zero",
        ],
        "acceptance_criteria": ["model pass if CI lower bound > 0 and p_one_gt_zero < 0.05"],
    }

    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "analysis_tier": "hardening",
        "canonical_eligible": True,
        "override_used": False,
        "verdict": {
            "interpretation": interp,
            "n_pass": int(cross["n_pass"]),
            "n_models": int(cross["n_models"]),
        },
        "limitations": [
            "Kernel transplant remains an evaluation-time perturbation, not a training counterfactual.",
            "Layer-matched pairing controls structure partially; full circuit-level causality remains open.",
        ],
    }

    claim_impact = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "claim_status": claim_status,
        "supports_main_text": claim_status in ("supported", "supported_with_caveat", "mixed"),
        "notes": [
            "Direct anti-tautology hardening for Result-I specificity.",
            "Positive transplant-over-low-self supports SI-structure-specific disruption signal.",
        ],
    }

    data_dictionary = {
        "experiment_id": EXPERIMENT_ID,
        "tables": [
            {
                "path": "<model>/kernel_transplant_trials.parquet",
                "description": "Sequence-level deltas for low-self, low-transplant, low-random, and high-self arms.",
                "columns": [],
            },
        ],
    }

    emit_core_artifacts(
        experiment_id=EXPERIMENT_ID,
        out_dir=out_root,
        preregistration=prereg,
        manifest_extra={"models": models},
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )
    return cross


def main() -> None:
    p = argparse.ArgumentParser(description="E29A: kernel transplant specificity", allow_abbrev=False)
    p.add_argument("--models", default=",".join(PRIMARY_MODELS))
    p.add_argument("--device-map", default="llama-3.1-8b:cuda:0,olmo-2-7b:cuda:0,mistral-7b-v0.1:cuda:0")
    p.add_argument("--output-root", default=str(DEFAULT_OUT))
    p.add_argument("--eval-seqs", type=int, default=96)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260504)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--finalize-only", action="store_true")
    p.add_argument("--no-finalize", action="store_true")
    args = p.parse_args()

    if args.finalize_only and args.no_finalize:
        raise RuntimeError("[E29A] hard_fail_reason: --finalize-only and --no-finalize are mutually exclusive")

    models = parse_models_arg(args.models, default=PRIMARY_MODELS)
    device_map = parse_device_map(args.device_map)
    out_root = ensure_dir(Path(args.output_root))

    eval_seqs = int(args.eval_seqs)
    batch_size = int(args.batch_size)
    if args.smoke:
        eval_seqs = min(eval_seqs, 16)
        batch_size = min(batch_size, 2)

    if args.finalize_only:
        cross = _finalize(models, out_root)
        print(f"[E29A] Finalized. interpretation={cross['interpretation']}", flush=True)
        return

    for model_name in models:
        device = device_map.get(model_name, "cuda:0")
        print(f"[E29A] Running {model_name} on {device}", flush=True)
        run_model(
            model_name=model_name,
            device=device,
            out_root=out_root,
            eval_seqs=max(8, eval_seqs),
            seq_len=max(64, int(args.seq_len)),
            batch_size=max(1, batch_size),
            seed=int(args.seed),
        )

    if args.no_finalize:
        print("[E29A] Shard complete (no finalize).", flush=True)
        return

    cross = _finalize(models, out_root)
    print(f"[E29A] Done. interpretation={cross['interpretation']}", flush=True)


if __name__ == "__main__":
    main()
