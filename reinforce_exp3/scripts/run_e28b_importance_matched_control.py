#!/usr/bin/env python3
"""E28b — Importance-matched non-SI control for SI-ranked kernel subtraction.

Design:
1) Compute non-SI importance proxy per head via generic single-head output zeroing
   on a calibration slice (head_output_ablation, not SI-kernel-based).
2) Build matched-cardinality non-SI head set by proxy ranking.
3) Compare SI-kernel subtraction disruption on high-SI heads vs importance-matched
   non-SI heads (equal head count).
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
    head_output_ablation,
)
from experiment3.theory8_position_ablation import (  # noqa: E402
    compute_per_token_loss,
    load_head_groups,
    load_wiki_sequences,
    subtract_positional_kernels,
)

EXPERIMENT_ID = "E28B"
DEFAULT_OUT = RESULTS_ROOT / "E28b_importance_matched_control"

_KERNEL_CANDIDATES = [
    "results/reinforce_exp/exp_r3_core_replication/{model}/theory8_position_ablation/{model}/estimated_kernels.json",
    "results/experiment3/theory8_position_ablation/{model}/estimated_kernels.json",
]
_R2_PATH = "results/experiment3/theory1_si_circuits/{model}/head_r2_summary.parquet"


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
    raise FileNotFoundError(f"[E28b] No estimated_kernels.json for {model_name}")


def _load_r2_data(model_name: str) -> pd.DataFrame:
    p = ROOT / _R2_PATH.format(model=model_name)
    if not p.exists():
        raise FileNotFoundError(f"[E28b] Missing R² data: {p}")
    df = pd.read_parquet(p)
    required = {"layer", "head", "mean_r2"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"[E28b] Missing columns in R² data: {df.columns.tolist()}")
    return df


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


def _bootstrap_diff_ci(diff: np.ndarray, n_boot: int = 5000, seed: int = 20260503) -> dict[str, float]:
    d = np.asarray(diff, dtype=np.float64)
    if d.size == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "p_one_gt_zero": float("nan")}
    rng = np.random.default_rng(int(seed))
    n = int(d.size)
    boot = np.empty(max(1000, int(n_boot)), dtype=np.float64)
    for i in range(boot.size):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(np.mean(d[idx]))
    return {
        "mean": float(np.mean(d)),
        "ci_lo": float(np.quantile(boot, 0.025)),
        "ci_hi": float(np.quantile(boot, 0.975)),
        "p_one_gt_zero": float((np.sum(boot <= 0.0) + 1) / (boot.size + 1)),
    }


def _compute_head_importance_proxy(
    *,
    model: Any,
    model_name: str,
    heads: list[tuple[int, int]],
    cal_sequences: list[list[int]],
    device: str,
    batch_size: int,
    cache_path: Path,
) -> pd.DataFrame:
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        have = {(int(r.layer), int(r.head)) for r in df.itertuples(index=False)}
        if all(h in have for h in heads):
            return df

    baseline = _eval_mean_losses(model, cal_sequences, device, batch_size)
    rows: list[dict[str, Any]] = []
    n = len(heads)
    for i, (layer, head) in enumerate(heads, start=1):
        with head_output_ablation(model, [(layer, head)]):
            ablated = _eval_mean_losses(model, cal_sequences, device, batch_size)
        delta = np.abs(ablated - baseline)
        rows.append(
            {
                "layer": int(layer),
                "head": int(head),
                "importance_abs_delta_mean": float(np.mean(delta)),
                "importance_abs_delta_median": float(np.median(delta)),
            }
        )
        if i % 64 == 0 or i == n:
            print(f"[E28b][{model_name}] importance proxy progress: {i}/{n}", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(cache_path, index=False)
    return df


def run_model(
    *,
    model_name: str,
    device: str,
    out_root: Path,
    eval_seqs: int,
    cal_eval_seqs: int,
    seq_len: int,
    batch_size: int,
) -> dict[str, Any]:
    t0 = time.time()
    model_dir = ensure_dir(out_root / model_name)

    model, _tok = load_model_for_exp(model_name, device, attn_implementation="eager")
    kernels = _load_kernels(model_name)

    r2_df = _load_r2_data(model_name).copy()
    available = set(kernels.keys())
    r2_df = r2_df[r2_df.apply(lambda row: (int(row["layer"]), int(row["head"])) in available, axis=1)].copy()

    head_groups = load_head_groups(model_name)
    high_heads = [(int(l), int(h)) for (l, h) in head_groups.get("high_si", []) if (int(l), int(h)) in available]
    if not high_heads:
        raise RuntimeError(f"[E28b] no high_si heads for {model_name}")
    high_set = set(high_heads)

    all_heads = [(int(r.layer), int(r.head)) for r in r2_df[["layer", "head"]].itertuples(index=False)]
    non_si_heads = [h for h in all_heads if h not in high_set]
    if len(non_si_heads) < len(high_heads):
        raise RuntimeError(f"[E28b] insufficient non-SI heads for {model_name}")

    sequences = load_wiki_sequences(model_name, max_sequences=max(8, eval_seqs), seq_len=max(64, seq_len))
    eval_sequences = [seq[:seq_len] for seq in sequences[:eval_seqs]]
    cal_sequences = [seq[:seq_len] for seq in sequences[:cal_eval_seqs]]
    if len(eval_sequences) < 4 or len(cal_sequences) < 4:
        raise RuntimeError(f"[E28b] insufficient eval/cal sequences for {model_name}")

    proxy_df = _compute_head_importance_proxy(
        model=model,
        model_name=model_name,
        heads=all_heads,
        cal_sequences=cal_sequences,
        device=device,
        batch_size=batch_size,
        cache_path=model_dir / "head_importance_proxy.parquet",
    )

    proxy_non = proxy_df[proxy_df.apply(lambda r: (int(r["layer"]), int(r["head"])) in set(non_si_heads), axis=1)].copy()
    proxy_non = proxy_non.sort_values("importance_abs_delta_mean", ascending=False)
    importance_heads = [(int(r.layer), int(r.head)) for r in proxy_non.head(len(high_heads)).itertuples(index=False)]

    write_json(
        model_dir / "selected_head_sets.json",
        {
            "n_high_si": int(len(high_heads)),
            "n_importance_non_si": int(len(importance_heads)),
            "high_si_heads": [[int(l), int(h)] for (l, h) in high_heads],
            "importance_non_si_heads": [[int(l), int(h)] for (l, h) in importance_heads],
        },
    )

    baseline = _eval_mean_losses(model, eval_sequences, device, batch_size)

    with subtract_positional_kernels(model, kernels, high_heads, int(seq_len)):
        loss_si = _eval_mean_losses(model, eval_sequences, device, batch_size)
    with subtract_positional_kernels(model, kernels, importance_heads, int(seq_len)):
        loss_imp = _eval_mean_losses(model, eval_sequences, device, batch_size)

    delta_si = loss_si - baseline
    delta_imp = loss_imp - baseline
    diff = delta_si - delta_imp

    ci = _bootstrap_diff_ci(diff)

    trial_df = pd.DataFrame(
        {
            "seq_idx": np.arange(len(delta_si), dtype=int),
            "delta_si": delta_si,
            "delta_importance": delta_imp,
            "delta_diff": diff,
        }
    )
    trial_df.to_parquet(model_dir / "si_vs_importance_trials.parquet", index=False)

    rec = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "model": model_name,
        "n_eval_seqs": int(len(eval_sequences)),
        "n_cal_eval_seqs": int(len(cal_sequences)),
        "n_high_si_heads": int(len(high_heads)),
        "n_importance_non_si_heads": int(len(importance_heads)),
        "si_mean_delta": float(np.mean(delta_si)),
        "importance_mean_delta": float(np.mean(delta_imp)),
        "delta_true_si_vs_importance": float(np.mean(diff)),
        "ci_delta_true_si_vs_importance": [float(ci["ci_lo"]), float(ci["ci_hi"])],
        "p_one_gt_zero": float(ci["p_one_gt_zero"]),
        "pass_model": bool(float(ci["ci_lo"]) > 0.0),
        "elapsed_sec": float(time.time() - t0),
    }
    write_json(model_dir / "summary.json", rec)
    print(
        f"[E28b][{model_name}] si_mean={rec['si_mean_delta']:.6f} imp_mean={rec['importance_mean_delta']:.6f} "
        f"diff={rec['delta_true_si_vs_importance']:.6f} ci=[{ci['ci_lo']:.6f},{ci['ci_hi']:.6f}] "
        f"p_one={ci['p_one_gt_zero']:.6f}",
        flush=True,
    )
    return rec


def _finalize(models: list[str], out_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for m in models:
        p = out_root / m / "summary.json"
        if not p.exists():
            raise RuntimeError(f"[E28b] hard_fail_reason: missing summary for {m}: {p}")
        rows.append(json.loads(p.read_text(encoding="utf-8")))

    enforce_coverage_contract(
        experiment_id=EXPERIMENT_ID,
        observed_models=[r.get("model", "") for r in rows],
        required_models=models,
    )

    n_pass = sum(1 for r in rows if bool(r.get("pass_model", False)))
    pooled_diff = float(np.mean([float(r["delta_true_si_vs_importance"]) for r in rows])) if rows else float("nan")

    if n_pass == len(rows):
        interp = "si_over_importance_supported"
        claim_status = "supported"
    elif n_pass >= 2:
        interp = "si_over_importance_supported_with_caveat"
        claim_status = "supported_with_caveat"
    elif n_pass >= 1:
        interp = "si_over_importance_mixed"
        claim_status = "mixed"
    else:
        interp = "si_over_importance_not_supported"
        claim_status = "not_supported"

    cross = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "n_models": int(len(rows)),
        "n_model_pass": int(n_pass),
        "interpretation": interp,
        "pooled_mean_delta_true_si_vs_importance": pooled_diff,
        "per_model": rows,
    }
    write_json(out_root / "cross_model_summary.json", cross)

    prereg = {
        "experiment_id": EXPERIMENT_ID,
        "question": "Does SI-ranked kernel subtraction disrupt more than importance-matched non-SI controls?",
        "primary_hypothesis": "delta_true_si_vs_importance > 0 under matched-cardinality design.",
        "primary_endpoints": [
            "delta_true_si_vs_importance",
            "bootstrap_CI_delta_true_si_vs_importance",
        ],
        "acceptance_criteria": ["model pass if CI lower bound > 0"],
        "design": {
            "importance_proxy": "single-head generic head-output zeroing on calibration sequences",
            "matching": "equal head count, non-SI pool only",
        },
    }

    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "analysis_tier": "hardening",
        "canonical_eligible": True,
        "override_used": False,
        "verdict": {
            "interpretation": interp,
            "n_model_pass": int(n_pass),
            "n_models": int(len(rows)),
            "pooled_mean_delta": pooled_diff,
        },
        "limitations": [
            "Importance proxy is calibration-slice dependent and based on single-head zeroing.",
            "Kernel subtraction remains an intervention-time assay, not a training-time counterfactual.",
        ],
    }

    claim_impact = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "claim_status": claim_status,
        "supports_main_text": claim_status in ("supported", "supported_with_caveat", "mixed"),
        "notes": [
            "Directly tests whether SI-ranked disruption exceeds a matched generic-importance control.",
            "Used to bound Result-I specificity language against generic head-importance explanations.",
        ],
    }

    data_dictionary = {
        "experiment_id": EXPERIMENT_ID,
        "tables": [
            {
                "path": "<model>/head_importance_proxy.parquet",
                "description": "Per-head generic importance proxy from single-head output zeroing.",
                "columns": [
                    {"name": "layer", "dtype": "int", "description": "layer index"},
                    {"name": "head", "dtype": "int", "description": "head index"},
                    {"name": "importance_abs_delta_mean", "dtype": "float", "description": "mean absolute calibration loss delta"},
                    {"name": "importance_abs_delta_median", "dtype": "float", "description": "median absolute calibration loss delta"},
                ],
            },
            {
                "path": "<model>/si_vs_importance_trials.parquet",
                "description": "Per-sequence deltas for SI vs importance-matched interventions.",
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
    p = argparse.ArgumentParser(description="E28b: importance-matched control", allow_abbrev=False)
    p.add_argument("--models", default=",".join(PRIMARY_MODELS))
    p.add_argument("--device-map", default="llama-3.1-8b:cuda:0,olmo-2-7b:cuda:0,mistral-7b-v0.1:cuda:0")
    p.add_argument("--output-root", default=str(DEFAULT_OUT))
    p.add_argument("--eval-seqs", type=int, default=64)
    p.add_argument("--cal-eval-seqs", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--finalize-only", action="store_true")
    p.add_argument("--no-finalize", action="store_true")
    args = p.parse_args()

    if args.finalize_only and args.no_finalize:
        raise RuntimeError("[E28b] hard_fail_reason: --finalize-only and --no-finalize are mutually exclusive")

    models = parse_models_arg(args.models, default=PRIMARY_MODELS)
    device_map = parse_device_map(args.device_map)
    out_root = ensure_dir(Path(args.output_root))

    eval_seqs = int(args.eval_seqs)
    cal_eval_seqs = int(args.cal_eval_seqs)
    batch_size = int(args.batch_size)
    if args.smoke:
        eval_seqs = min(eval_seqs, 12)
        cal_eval_seqs = min(cal_eval_seqs, 4)
        batch_size = min(batch_size, 2)

    if args.finalize_only:
        cross = _finalize(models, out_root)
        print(f"[E28b] Finalized. interpretation={cross['interpretation']}", flush=True)
        return

    for model_name in models:
        device = device_map.get(model_name, "cuda:0")
        print(f"[E28b] Running {model_name} on {device}", flush=True)
        run_model(
            model_name=model_name,
            device=device,
            out_root=out_root,
            eval_seqs=max(8, eval_seqs),
            cal_eval_seqs=max(4, cal_eval_seqs),
            seq_len=max(64, int(args.seq_len)),
            batch_size=max(1, batch_size),
        )

    if args.no_finalize:
        print("[E28b] Shard complete (no finalize).", flush=True)
        return

    cross = _finalize(models, out_root)
    print(f"[E28b] Done. interpretation={cross['interpretation']}", flush=True)


if __name__ == "__main__":
    main()
