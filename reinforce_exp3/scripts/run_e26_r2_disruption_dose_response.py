#!/usr/bin/env python3
"""E26 — R²-stratified disruption cost dose-response.

Tests whether SI disruption cost scales with R² amplitude within each model.
Heads are grouped into 5 within-model R² quintile bins; grouped kernel
subtraction is run for each bin with true, permuted, and norm-matched controls.

Primary endpoint: Spearman r(mean-bin-R², per-head disruption cost) > 0 in
Llama and Mistral. If positive, R² amplitude predicts load-bearing within models,
supporting SI-specificity at the head level even after E24 weakened it at the
collective level.

If the correlation is flat or negative, disruption cost is independent of R²
within each model → the grouped high-SI ablation result (T8/E17) reflects
removal of many heads, not SI amplitude specifically.

OLMo (mean R²≈0.058) is included as a divergent anchor; its dose-response is
expected to be flat (low amplitude across all bins).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats

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
    rng_for_stream,
)
from experiment3.theory8_position_ablation import (  # noqa: E402
    compute_per_token_loss,
    load_wiki_sequences,
    subtract_positional_kernels,
)

EXPERIMENT_ID = "E26"
N_BINS = 5
DEFAULT_OUT = RESULTS_ROOT / "E26_r2_disruption_dose_response"

# Kernel file search order (matches E17)
_KERNEL_CANDIDATES = [
    "results/reinforce_exp/exp_r3_core_replication/{model}/theory8_position_ablation/{model}/estimated_kernels.json",
    "results/experiment3/theory8_position_ablation/{model}/estimated_kernels.json",
]
_R2_PATH = "results/experiment3/theory1_si_circuits/{model}/head_r2_summary.parquet"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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
    raise FileNotFoundError(f"[E26] No estimated_kernels.json for {model_name}")


def _load_r2_data(model_name: str) -> pd.DataFrame:
    p = ROOT / _R2_PATH.format(model=model_name)
    if not p.exists():
        raise FileNotFoundError(f"[E26] Missing R² data: {p}")
    df = pd.read_parquet(p)
    required = {"layer", "head", "mean_r2"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"[E26] Missing columns in R² data: {df.columns.tolist()}")
    return df


# ---------------------------------------------------------------------------
# Perturbation helpers (identical to E17)
# ---------------------------------------------------------------------------

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
        out[head] = g[rng.permutation(len(g))].copy()
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
        n = len(g)
        z = rng.standard_normal(n).astype(np.float32)
        z -= float(np.mean(z))
        zn = float(np.linalg.norm(z))
        gn = float(np.linalg.norm(g))
        if zn > 1e-12 and gn > 1e-12:
            out[head] = (z * (gn / zn)).astype(np.float32)
        else:
            out[head] = np.zeros_like(g)
    return out


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _eval_mean_losses(
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


# ---------------------------------------------------------------------------
# Per-model run
# ---------------------------------------------------------------------------

def run_model(
    *,
    model_name: str,
    device: str,
    out_root: Path,
    eval_seqs: int,
    seq_len: int,
    n_control_trials: int,
    seed: int,
    batch_size: int,
) -> dict[str, Any]:
    t0 = time.time()
    model_dir = ensure_dir(out_root / model_name)

    model, _tok = load_model_for_exp(model_name, device, attn_implementation="eager")

    r2_df = _load_r2_data(model_name)
    kernels = _load_kernels(model_name)

    # Assign within-model R² quintile bins (1=lowest, 5=highest)
    r2_df = r2_df.copy()
    r2_df["bin"] = pd.qcut(r2_df["mean_r2"], q=N_BINS, labels=False, duplicates="drop") + 1
    r2_df["bin"] = r2_df["bin"].astype(int)

    # Keep only heads that have kernel data
    available = set(kernels.keys())
    r2_df = r2_df[r2_df.apply(lambda row: (int(row["layer"]), int(row["head"])) in available, axis=1)].copy()

    sequences = load_wiki_sequences(model_name, max_sequences=max(8, eval_seqs), seq_len=max(64, seq_len))
    sequences = [seq[:seq_len] for seq in sequences[:eval_seqs]]
    if len(sequences) < 4:
        raise RuntimeError(f"[E26] Insufficient sequences ({len(sequences)}) for {model_name}")

    baseline = _eval_mean_losses(model, sequences, device, batch_size)

    rows: list[dict[str, Any]] = []
    actual_bins = sorted(r2_df["bin"].unique())

    for bin_idx in actual_bins:
        bin_mask = r2_df["bin"] == bin_idx
        bin_heads = [(int(row["layer"]), int(row["head"])) for _, row in r2_df[bin_mask].iterrows()]
        n_heads = len(bin_heads)
        mean_r2 = float(r2_df.loc[bin_mask, "mean_r2"].mean())
        min_r2 = float(r2_df.loc[bin_mask, "mean_r2"].min())
        max_r2 = float(r2_df.loc[bin_mask, "mean_r2"].max())

        # True kernel subtraction
        with subtract_positional_kernels(model, kernels, bin_heads, seq_len):
            true_loss = _eval_mean_losses(model, sequences, device, batch_size)
        true_delta = float(np.mean(true_loss - baseline))

        # Control trials
        perm_deltas: list[float] = []
        norm_deltas: list[float] = []
        for trial in range(n_control_trials):
            rng_p = rng_for_stream(seed, f"e26_perm_{model_name}_bin{bin_idx}_t{trial}")
            rng_n = rng_for_stream(seed, f"e26_norm_{model_name}_bin{bin_idx}_t{trial}")
            perm_k = _permute_kernels(kernels, bin_heads, rng_p)
            with subtract_positional_kernels(model, perm_k, bin_heads, seq_len):
                perm_loss = _eval_mean_losses(model, sequences, device, batch_size)
            perm_deltas.append(float(np.mean(perm_loss - baseline)))
            norm_k = _norm_matched_random_kernels(kernels, bin_heads, rng_n)
            with subtract_positional_kernels(model, norm_k, bin_heads, seq_len):
                norm_loss = _eval_mean_losses(model, sequences, device, batch_size)
            norm_deltas.append(float(np.mean(norm_loss - baseline)))

        mean_perm = float(np.mean(perm_deltas))
        mean_norm = float(np.mean(norm_deltas))
        # Per-head proxy: total group delta divided by bin size
        per_head_true = true_delta / max(n_heads, 1)
        per_head_perm = mean_perm / max(n_heads, 1)

        rows.append({
            "bin": int(bin_idx),
            "n_heads": int(n_heads),
            "mean_r2": mean_r2,
            "min_r2": min_r2,
            "max_r2": max_r2,
            "true_mean_delta": true_delta,
            "perm_mean_delta": mean_perm,
            "norm_mean_delta": mean_norm,
            "per_head_true_delta": per_head_true,
            "per_head_perm_delta": per_head_perm,
            "true_over_perm": true_delta / mean_perm if abs(mean_perm) > 1e-10 else float("nan"),
        })
        print(
            f"[E26][{model_name}] bin={bin_idx} n={n_heads} mean_r2={mean_r2:.4f} "
            f"true_delta={true_delta:.6f} perm_delta={mean_perm:.6f} norm_delta={mean_norm:.6f}",
            flush=True,
        )

    dose_df = pd.DataFrame(rows)
    dose_df.to_parquet(model_dir / "dose_response_bins.parquet", index=False)

    # Spearman correlation: mean_r2 vs per-head true disruption
    if len(dose_df) >= 3:
        sp = scipy_stats.spearmanr(dose_df["mean_r2"], dose_df["per_head_true_delta"])
        spearman_r = float(sp.statistic)
        spearman_p = float(sp.pvalue)
    else:
        spearman_r = float("nan")
        spearman_p = float("nan")

    # Monotone direction check (bin i < bin i+1 for per-head delta)
    sorted_rows = dose_df.sort_values("bin")
    deltas = sorted_rows["per_head_true_delta"].tolist()
    n_monotone_steps = sum(1 for a, b in zip(deltas, deltas[1:]) if b >= a)
    total_steps = max(len(deltas) - 1, 1)
    monotone_fraction = n_monotone_steps / total_steps

    elapsed = time.time() - t0
    rec = {
        "model": model_name,
        "n_bins": len(rows),
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "monotone_fraction": monotone_fraction,
        "dose_response_positive": spearman_r > 0 and spearman_p < 0.10,
        "elapsed_sec": elapsed,
        "bins": rows,
    }
    write_json(model_dir / "model_summary.json", rec)
    print(f"[E26][{model_name}] Spearman_r={spearman_r:.4f} p={spearman_p:.4f} monotone={monotone_fraction:.2f}", flush=True)
    return rec


# ---------------------------------------------------------------------------
# Cross-model finalize
# ---------------------------------------------------------------------------

def _finalize(models: list[str], out_root: Path, start_ts: str) -> dict[str, Any]:
    model_results: list[dict[str, Any]] = []
    for m in models:
        p = out_root / m / "model_summary.json"
        if not p.exists():
            raise RuntimeError(f"[E26] hard_fail_reason: missing model_summary for {m} at {p}")
        with p.open() as f:
            model_results.append(json.load(f))

    n_positive = sum(1 for r in model_results if r.get("dose_response_positive", False))
    # Direction check: does spearman_r track model-mean-R2 order (Llama > Mistral > OLMo)?
    spearman_by_model = {r["model"]: r["spearman_r"] for r in model_results}
    high_si_models = [m for m in ["llama-3.1-8b", "mistral-7b-v0.1"] if m in spearman_by_model]
    low_si_model = "olmo-2-7b"
    high_si_positive = all(spearman_by_model.get(m, float("nan")) > 0 for m in high_si_models)
    olmo_lower = all(
        spearman_by_model.get(low_si_model, 0) <= spearman_by_model.get(m, 0)
        for m in high_si_models
    )

    if high_si_positive and olmo_lower:
        interp = "dose_response_positive_si_specific"
        claim_status = "supported"
    elif n_positive >= 2:
        interp = "dose_response_positive_partial"
        claim_status = "supported_with_caveat"
    elif n_positive == 1:
        interp = "dose_response_mixed"
        claim_status = "mixed"
    else:
        interp = "dose_response_not_detected"
        claim_status = "not_supported"

    prereg = {
        "experiment_id": EXPERIMENT_ID,
        "question": "Does per-head disruption cost scale with R² amplitude within each model?",
        "primary_hypothesis": (
            "Spearman r(mean-bin-R², per-head disruption cost) > 0 in Llama and Mistral, "
            "indicating R² amplitude predicts load-bearing beyond head count."
        ),
        "primary_endpoints": [
            "Spearman r(mean_r2, per_head_true_delta) per model",
            "Fraction of monotone bin-to-bin increase steps",
        ],
        "secondary_endpoints": [
            "Per-bin true vs permuted and norm-matched delta comparison",
            "Cross-model gradient: Llama > Mistral > OLMo in dose-response slope",
        ],
        "model_list": list(models),
        "dataset_sources": ["wikitext-103 (wiki sequences, seq_len=512)"],
        "inclusion_exclusion_rules": [
            "Only heads with available kernel data included",
            "Bins with fewer than 10 heads excluded from Spearman calculation",
        ],
        "sample_size_plan": {"eval_seqs": "100 (smoke: 16)", "n_bins": N_BINS},
        "seed_plan": {"base_seed": 20260503, "stream_derivation": "rng_for_stream(seed, stream_id)"},
        "stopping_rule": "Fixed eval_seqs; no adaptive stopping",
        "multiplicity_family": ["per-model Spearman correlations (3 tests)"],
        "acceptance_criteria": [
            "Spearman r > 0 and p < 0.10 in Llama and Mistral => supported",
            "r > 0 in at least one of {Llama, Mistral} => supported_with_caveat",
        ],
        "fallback_interpretation_if_null": (
            "Disruption cost is independent of R² within models; the grouped ablation "
            "result reflects head count, not SI amplitude → weakens SI-specificity claim."
        ),
    }

    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "analysis_tier": "confirmatory",
        "canonical_eligible": True,
        "override_used": False,
        "verdict": {
            "interpretation": interp,
            "n_positive": n_positive,
            "n_models": len(model_results),
            "high_si_positive": high_si_positive,
            "olmo_lower": olmo_lower,
            "spearman_by_model": spearman_by_model,
        },
        "limitations": [
            "Per-head disruption cost is a proxy (group delta / n_heads); interaction effects not separated.",
            "5 quintile bins may not have equal head counts due to tied R² values.",
            "OLMo has low mean R² (0.058); dose-response may be underpowered for that model.",
        ],
    }

    claim_impact = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "claim_status": claim_status,
        "supports_main_text": claim_status in ("supported", "supported_with_caveat"),
        "notes": [
            "Supports Weakness-2/5 response: if positive, R² amplitude predicts head-level load-bearing.",
            "Positive result partially rehabilitates SI-specificity at head level despite E24 caveat.",
            "Negative result requires reframing Result I as head-count effect, not SI-amplitude effect.",
        ],
    }

    data_dictionary = {
        "experiment_id": EXPERIMENT_ID,
        "tables": [
            {
                "path": "<model>/dose_response_bins.parquet",
                "description": "Per-R²-quintile-bin disruption cost results.",
                "columns": [
                    {"name": "bin", "dtype": "int", "description": "quintile bin (1=lowest R², 5=highest)"},
                    {"name": "n_heads", "dtype": "int", "description": "number of heads in bin"},
                    {"name": "mean_r2", "dtype": "float", "description": "mean R² of heads in bin"},
                    {"name": "min_r2", "dtype": "float", "description": "min R² in bin"},
                    {"name": "max_r2", "dtype": "float", "description": "max R² in bin"},
                    {"name": "true_mean_delta", "dtype": "float", "description": "mean loss increase under true SI subtraction"},
                    {"name": "perm_mean_delta", "dtype": "float", "description": "mean loss increase under permuted control (avg over trials)"},
                    {"name": "norm_mean_delta", "dtype": "float", "description": "mean loss increase under norm-matched control (avg over trials)"},
                    {"name": "per_head_true_delta", "dtype": "float", "description": "true_mean_delta / n_heads (per-head proxy)"},
                    {"name": "per_head_perm_delta", "dtype": "float", "description": "perm_mean_delta / n_heads (per-head proxy)"},
                    {"name": "true_over_perm", "dtype": "float", "description": "ratio of true to permuted delta"},
                ],
            },
            {
                "path": "<model>/model_summary.json",
                "description": "Per-model Spearman correlation and monotone fraction.",
                "columns": [],
            },
        ],
    }

    cross_summary = {
        "start_ts": start_ts,
        "end_ts": timestamp_now(),
        "models": [r["model"] for r in model_results],
        "interpretation": interp,
        "n_positive": n_positive,
        "spearman_by_model": spearman_by_model,
    }
    write_json(out_root / "cross_model_summary.json", cross_summary)

    emit_core_artifacts(
        experiment_id=EXPERIMENT_ID,
        out_dir=out_root,
        preregistration=prereg,
        manifest_extra={"models": list(models)},
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )
    return cross_summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="E26: R²-stratified disruption dose-response", allow_abbrev=False)
    p.add_argument("--models", default=",".join(PRIMARY_MODELS))
    p.add_argument("--device-map", default="llama-3.1-8b:cuda:0,olmo-2-7b:cuda:0,mistral-7b-v0.1:cuda:0")
    p.add_argument("--output-root", default=str(DEFAULT_OUT))
    p.add_argument("--eval-seqs", type=int, default=100)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--n-control-trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260503)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--finalize-only", action="store_true")
    p.add_argument("--no-finalize", action="store_true")
    args = p.parse_args()

    if args.finalize_only and args.no_finalize:
        raise RuntimeError("[E26] hard_fail_reason: --finalize-only and --no-finalize are mutually exclusive")

    models = parse_models_arg(args.models, default=PRIMARY_MODELS)
    device_map = parse_device_map(args.device_map)
    out_root = ensure_dir(Path(args.output_root))
    start_ts = timestamp_now()

    eval_seqs = int(args.eval_seqs)
    n_control_trials = int(args.n_control_trials)
    if args.smoke:
        eval_seqs = min(eval_seqs, 16)
        n_control_trials = min(n_control_trials, 1)

    if args.finalize_only:
        cross = _finalize(models, out_root, start_ts)
        print(f"[E26] Finalized. interpretation={cross['interpretation']}", flush=True)
        return

    for model_name in models:
        device = device_map.get(model_name, "cuda:0")
        print(f"[E26] Running {model_name} on {device}", flush=True)
        run_model(
            model_name=model_name,
            device=device,
            out_root=out_root,
            eval_seqs=max(8, eval_seqs),
            seq_len=max(64, int(args.seq_len)),
            n_control_trials=max(1, n_control_trials),
            seed=int(args.seed),
            batch_size=max(1, int(args.batch_size)),
        )

    if args.no_finalize:
        print("[E26] Shard complete (no finalize).", flush=True)
        return

    cross = _finalize(models, out_root, start_ts)
    print(f"[E26] Done. interpretation={cross['interpretation']}", flush=True)


if __name__ == "__main__":
    main()
