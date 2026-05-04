#!/usr/bin/env python3
"""E31D-Mistral: Mistral replication of naturalistic SI vs importance-matched non-SI control.

Identical protocol to E31D (Llama), applied to Mistral-7B-v0.1.
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

from reinforce_exp3.common import RESULTS_ROOT, ensure_dir, timestamp_now, write_json  # noqa: E402
from reinforce_exp3.scripts._shared import emit_core_artifacts, load_model_for_exp  # noqa: E402
from reinforce_exp3.scripts.run_e29b_naturaltext_longcontext_probe import (  # noqa: E402
    FAMILIES,
    _bootstrap_mean_ci,
    _build_family_prompts,
    _eval_target_metrics,
    _load_kernels,
    _permute_kernels,
)
from reinforce_exp3.scripts.run_e31d_llama_naturalistic_importance_control import (  # noqa: E402
    _one_sided_signflip_p,
)
from experiment3.theory8_position_ablation import subtract_positional_kernels  # noqa: E402

EXPERIMENT_ID = "E31D_Mistral"
MODEL = "mistral-7b-v0.1"
DEFAULT_OUT = RESULTS_ROOT / "E31d_mistral_naturalistic_importance_control"


def _load_importance_heads_mistral() -> list[tuple[int, int]]:
    sel = (
        ROOT
        / "results"
        / "reinforce_exp3"
        / "E28b_importance_matched_control"
        / MODEL
        / "selected_head_sets.json"
    )
    if sel.exists():
        payload = json.loads(sel.read_text(encoding="utf-8"))
        heads = payload.get("importance_non_si_heads", [])
        out = [(int(x[0]), int(x[1])) for x in heads]
        if out:
            return out

    proxy_path = (
        ROOT
        / "results"
        / "reinforce_exp3"
        / "E28b_importance_matched_control"
        / MODEL
        / "head_importance_proxy.parquet"
    )
    r2_path = ROOT / "results" / "experiment3" / "theory1_si_circuits" / MODEL / "head_r2_summary.parquet"
    groups_path = ROOT / "results" / "experiment3" / "theory1_si_circuits" / MODEL / "head_groups.json"
    if not (proxy_path.exists() and r2_path.exists() and groups_path.exists()):
        raise FileNotFoundError(f"[E31D-Mistral] Missing E28B artifacts for {MODEL}")

    proxy = pd.read_parquet(proxy_path)
    groups = json.loads(groups_path.read_text(encoding="utf-8"))
    high = {(int(d["layer"]), int(d["head"])) for d in groups.get("high_si", [])}

    proxy["lh"] = list(zip(proxy["layer"].astype(int), proxy["head"].astype(int)))
    non = proxy[~proxy["lh"].isin(high)].copy()
    non = non.sort_values("importance_abs_delta_mean", ascending=False)

    n_high = len(high)
    out = [(int(r.layer), int(r.head)) for r in non.head(n_high).itertuples(index=False)]
    if not out:
        raise RuntimeError(f"[E31D-Mistral] Could not derive importance-matched non-SI heads for {MODEL}")
    return out


def run(
    *,
    device: str,
    out_root: Path,
    total_prompts: int,
    batch_size: int,
    seq_len: int,
    n_control_trials: int,
    seed: int,
    smoke: bool,
) -> dict[str, Any]:
    t0 = time.time()
    out_root = ensure_dir(out_root)

    n_total = int(total_prompts)
    if smoke:
        n_total = min(n_total, 12)
        n_control_trials = min(n_control_trials, 1)

    model_dir = ensure_dir(out_root / MODEL)

    model, tokenizer = load_model_for_exp(MODEL, device=device, attn_implementation="eager")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = int(tokenizer.pad_token_id)

    kernels = _load_kernels(MODEL)

    groups_path = ROOT / "results" / "experiment3" / "theory1_si_circuits" / MODEL / "head_groups.json"
    groups = json.loads(groups_path.read_text(encoding="utf-8"))
    high_heads = [
        (int(d["layer"]), int(d["head"]))
        for d in groups.get("high_si", [])
        if (int(d["layer"]), int(d["head"])) in kernels
    ]
    if not high_heads:
        raise RuntimeError(f"[E31D-Mistral] no usable high-SI heads for {MODEL}")

    imp_heads = [h for h in _load_importance_heads_mistral() if h in kernels]
    if not imp_heads:
        raise RuntimeError(f"[E31D-Mistral] no usable importance-matched non-SI heads for {MODEL}")

    k = min(len(imp_heads), len(high_heads))
    imp_heads = imp_heads[:k]
    high_heads = high_heads[:k]

    n_fam = len(FAMILIES)
    n_per_family = max(8, n_total // max(1, n_fam))

    family_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []

    for fidx, spec in enumerate(FAMILIES):
        probe_prompts, ctrl_prompts = _build_family_prompts(
            model_name=MODEL,
            tokenizer=tokenizer,
            spec=spec,
            n_prompts=n_per_family,
            seed=seed + fidx * 97,
        )

        intact_probe = _eval_target_metrics(model=model, prompts=probe_prompts, device=device, pad_token_id=pad_id, batch_size=batch_size)
        intact_ctrl = _eval_target_metrics(model=model, prompts=ctrl_prompts, device=device, pad_token_id=pad_id, batch_size=batch_size)

        max_len = int(max(seq_len, spec.context_len))

        with subtract_positional_kernels(model, kernels, high_heads, max_len):
            si_probe = _eval_target_metrics(model=model, prompts=probe_prompts, device=device, pad_token_id=pad_id, batch_size=batch_size)
            si_ctrl = _eval_target_metrics(model=model, prompts=ctrl_prompts, device=device, pad_token_id=pad_id, batch_size=batch_size)

        perm_probe_runs: list[np.ndarray] = []
        perm_ctrl_runs: list[np.ndarray] = []
        for trial in range(max(1, int(n_control_trials))):
            rng = np.random.default_rng(seed + 8000 + fidx * 31 + trial)
            k_perm = _permute_kernels(kernels, high_heads, rng)
            with subtract_positional_kernels(model, k_perm, high_heads, max_len):
                pp = _eval_target_metrics(model=model, prompts=probe_prompts, device=device, pad_token_id=pad_id, batch_size=batch_size)
                pc = _eval_target_metrics(model=model, prompts=ctrl_prompts, device=device, pad_token_id=pad_id, batch_size=batch_size)
            perm_probe_runs.append(pp["lp"])
            perm_ctrl_runs.append(pc["lp"])

        perm_probe_lp = np.mean(np.vstack(perm_probe_runs), axis=0)
        perm_ctrl_lp = np.mean(np.vstack(perm_ctrl_runs), axis=0)

        with subtract_positional_kernels(model, kernels, imp_heads, max_len):
            imp_probe = _eval_target_metrics(model=model, prompts=probe_prompts, device=device, pad_token_id=pad_id, batch_size=batch_size)
            imp_ctrl = _eval_target_metrics(model=model, prompts=ctrl_prompts, device=device, pad_token_id=pad_id, batch_size=batch_size)

        loss_si_probe = intact_probe["lp"] - si_probe["lp"]
        loss_si_ctrl = intact_ctrl["lp"] - si_ctrl["lp"]
        gap_si = loss_si_probe - loss_si_ctrl

        loss_imp_probe = intact_probe["lp"] - imp_probe["lp"]
        loss_imp_ctrl = intact_ctrl["lp"] - imp_ctrl["lp"]
        gap_imp = loss_imp_probe - loss_imp_ctrl

        loss_perm_probe = intact_probe["lp"] - perm_probe_lp
        loss_perm_ctrl = intact_ctrl["lp"] - perm_ctrl_lp
        gap_perm = loss_perm_probe - loss_perm_ctrl

        gap_si_minus_imp = gap_si - gap_imp
        fam_ci = _bootstrap_mean_ci(gap_si_minus_imp, n_boot=3000, seed=seed + 5000 + fidx)
        fam_row = {
            "family": spec.name,
            "context_len": int(spec.context_len),
            "n_prompts": int(len(gap_si)),
            "mean_gap_si": float(np.mean(gap_si)),
            "mean_gap_imp": float(np.mean(gap_imp)),
            "mean_gap_perm": float(np.mean(gap_perm)),
            "mean_delta_si_minus_imp": float(np.mean(gap_si_minus_imp)),
            "delta_si_minus_imp_ci": [float(fam_ci["ci_lo"]), float(fam_ci["ci_hi"])],
            "family_si_gt_imp_pass": bool(float(np.mean(gap_si_minus_imp)) > 0.0),
        }
        family_rows.append(fam_row)

        for i in range(len(gap_si)):
            trial_rows.append({
                "family": spec.name,
                "prompt_idx": int(i),
                "gap_si": float(gap_si[i]),
                "gap_imp": float(gap_imp[i]),
                "gap_perm": float(gap_perm[i]),
                "delta_si_minus_imp": float(gap_si_minus_imp[i]),
                "delta_si_minus_perm": float(gap_si[i] - gap_perm[i]),
            })

        print(
            f"[E31D-Mistral][{spec.name}] mean_gap_si={np.mean(gap_si):.4f} "
            f"mean_gap_imp={np.mean(gap_imp):.4f} mean_delta={np.mean(gap_si_minus_imp):.4f}",
            flush=True,
        )

    trials_df = pd.DataFrame(trial_rows)
    trials_df.to_parquet(model_dir / "mistral_naturalistic_arm_trials.parquet", index=False)

    d_si_imp = trials_df["delta_si_minus_imp"].to_numpy(dtype=np.float64)
    d_si_perm = trials_df["delta_si_minus_perm"].to_numpy(dtype=np.float64)

    ci_si_imp = _bootstrap_mean_ci(d_si_imp, n_boot=6000, seed=seed + 9001)
    ci_si_perm = _bootstrap_mean_ci(d_si_perm, n_boot=6000, seed=seed + 9002)

    p_si_imp = _one_sided_signflip_p(d_si_imp, n_perm=30000, seed=seed + 9101)
    p_si_perm = _one_sided_signflip_p(d_si_perm, n_perm=30000, seed=seed + 9102)

    result = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "model": MODEL,
        "smoke": bool(smoke),
        "n_high_si_heads": int(len(high_heads)),
        "n_importance_non_si_heads": int(len(imp_heads)),
        "n_families": int(len(family_rows)),
        "n_trials": int(len(trials_df)),
        "mean_delta_si_minus_imp": float(ci_si_imp["mean"]),
        "delta_si_minus_imp_ci": [float(ci_si_imp["ci_lo"]), float(ci_si_imp["ci_hi"])],
        "p_one_sided_si_gt_imp": float(p_si_imp),
        "mean_delta_si_minus_perm": float(ci_si_perm["mean"]),
        "delta_si_minus_perm_ci": [float(ci_si_perm["ci_lo"]), float(ci_si_perm["ci_hi"])],
        "p_one_sided_si_gt_perm": float(p_si_perm),
        "directional_support_si_gt_importance": bool(float(ci_si_imp["ci_lo"]) > 0.0),
        "directional_support_si_gt_perm": bool(float(ci_si_perm["ci_lo"]) > 0.0),
        "family_results": family_rows,
        "elapsed_sec": float(time.time() - t0),
    }

    write_json(model_dir / "summary.json", result)
    write_json(out_root / "mistral_naturalistic_importance_vs_si.json", result)

    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "analysis_tier": "hardening",
        "canonical_eligible": True,
        "verdict": {
            "interpretation": "mistral_naturalistic_si_vs_importance_supported"
            if bool(result["directional_support_si_gt_importance"])
            else "mistral_naturalistic_si_vs_importance_mixed",
            "directional_support": bool(result["directional_support_si_gt_importance"]),
            "n_trials": int(result["n_trials"]),
        },
        "elapsed_sec": float(time.time() - t0),
    }

    prereg = {
        "experiment_id": EXPERIMENT_ID,
        "question": "On semi-naturalistic prompts, is SI-targeted ablation more disruptive than importance-matched non-SI ablation in Mistral?",
        "primary_endpoint": "mean_delta_si_minus_imp",
    }
    claim_impact = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "claim_status": "supported" if bool(result["directional_support_si_gt_importance"]) else "mixed",
        "impact": "Mistral replication of E31D naturalistic SI vs importance-matched control",
    }
    data_dict = {
        "mistral_naturalistic_importance_vs_si.json": "Top-line Mistral naturalistic SI vs importance-matched summary.",
        "mistral_naturalistic_arm_trials.parquet": "Prompt-level trial table.",
    }

    emit_core_artifacts(
        experiment_id=EXPERIMENT_ID,
        out_dir=out_root,
        preregistration=prereg,
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dict,
        manifest_extra={"model": MODEL, "smoke": bool(smoke), "total_prompts": int(n_total)},
    )

    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="E31D-Mistral naturalistic SI vs importance-matched control")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-root", default=str(DEFAULT_OUT))
    p.add_argument("--total-prompts", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--n-control-trials", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260504)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    run(
        device=str(args.device),
        out_root=Path(args.output_root),
        total_prompts=int(args.total_prompts),
        batch_size=int(args.batch_size),
        seq_len=int(args.seq_len),
        n_control_trials=int(args.n_control_trials),
        seed=int(args.seed),
        smoke=bool(args.smoke),
    )


if __name__ == "__main__":
    main()
