#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats

from experiment3.theory1_si_circuits import MODELS as THEORY_MODELS, load_profile_sequences
from experiment4.common import now_timestamp, safe_float, write_json
from shared.attention.adapters import get_adapter
from shared.models.loading import load_model, load_tokenizer


DEFAULT_MODELS: tuple[str, ...] = ("llama-3.1-8b", "olmo-2-7b")
DEFAULT_SEQ_LENS: tuple[int, ...] = (256, 512, 1024)


def _load_high_si_heads(model_name: str) -> list[tuple[int, int]]:
    path = Path("results/experiment3/theory1_si_circuits") / model_name / "head_groups.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("high_si", [])
    out: list[tuple[int, int]] = []
    for row in rows:
        out.append((int(row["layer"]), int(row["head"])))
    if not out:
        raise RuntimeError(f"Empty high_si head list in {path}")
    return out


def _n_total_heads(model_name: str) -> int:
    summary_path = Path("results/experiment3/theory1_si_circuits") / model_name / "head_r2_summary.parquet"
    if summary_path.exists():
        return int(len(pd.read_parquet(summary_path)))
    r2_path = Path("results/experiment3/theory1_si_circuits") / model_name / "per_sequence_r2.parquet"
    if r2_path.exists():
        df = pd.read_parquet(r2_path)
        return int(df[["layer", "head"]].drop_duplicates().shape[0])
    raise FileNotFoundError(f"Missing head cardinality inputs for {model_name}")


def _estimate_effective_sparsity(
    *,
    model_name: str,
    seq_len: int,
    device: str,
    epsilon: float,
    num_sequences: int,
) -> pd.DataFrame:
    spec = THEORY_MODELS[model_name]
    try:
        loaded = load_model(spec, attn_implementation="eager")
    except Exception:
        loaded = load_model(spec)
    model = loaded.model.to(device)
    tokenizer = load_tokenizer(spec)
    adapter = get_adapter(spec)
    high_si = _load_high_si_heads(model_name)

    seqs = load_profile_sequences(
        tokenizer=tokenizer,
        model_name=model_name,
        num_sequences=max(1, int(num_sequences)),
        seq_len=max(64, int(seq_len)),
    )
    if not seqs:
        raise RuntimeError(f"No profiling sequences available for {model_name} seq_len={seq_len}")

    rows: list[dict[str, Any]] = []
    adapter.register(model)
    try:
        with torch.inference_mode():
            for seq_idx, tokens in enumerate(seqs):
                input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
                capture = adapter.capture(
                    model,
                    input_ids=input_ids,
                    include_logits=True,
                    return_token_logits=False,
                    capture_attention=True,
                    output_device="cpu",
                )
                logits = capture.logits  # [layers, heads, seq, seq]
                for layer, head in high_si:
                    l = logits[layer, head].to(dtype=torch.float32)
                    probs = torch.softmax(l, dim=-1)
                    # Number of positions with non-negligible attention mass.
                    counts = (probs > float(epsilon)).sum(dim=-1).to(dtype=torch.float32)
                    rows.append(
                        {
                            "model": model_name,
                            "seq_len": int(seq_len),
                            "sequence_id": int(seq_idx),
                            "layer": int(layer),
                            "head": int(head),
                            "effective_sparsity": float(counts.mean().item()),
                            "epsilon": float(epsilon),
                        }
                    )
                del capture, input_ids
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        adapter.cleanup()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    return df


def _run_c1_for_len(
    *,
    model_name: str,
    seq_len: int,
    device: str,
    output_root: Path,
    num_seeds: int,
    synthetic_count: int,
    ntp_count_per_seed: int,
    batch_size_synth: int,
    batch_size_ntp: int,
    force: bool,
) -> Path:
    model_dir = output_root / f"len_{int(seq_len)}" / model_name
    curve_fit = model_dir / "curve_fit_comparison.json"
    curve_data = model_dir / "cumulative_ablation_curve.parquet"
    if not force and curve_fit.exists() and curve_data.exists():
        return model_dir

    cmd = [
        sys.executable,
        "-m",
        "experiment3.phase2.exp3p2c_redundancy_quantification",
        "--model",
        model_name,
        "--device",
        str(device),
        "--output-root",
        str(output_root / f"len_{int(seq_len)}"),
        "--fractions",
        "0,1,2,5,10,15,20,25,50",
        "--sort-orders",
        "high_to_low",
        "--num-seeds",
        str(int(num_seeds)),
        "--synthetic-count",
        str(int(synthetic_count)),
        "--batch-size-synth",
        str(int(batch_size_synth)),
        "--ntp-count-per-seed",
        str(int(ntp_count_per_seed)),
        "--ntp-seq-len",
        str(int(seq_len)),
        "--batch-size-ntp",
        str(int(batch_size_ntp)),
    ]
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"3P2-C.1 subprocess failed for model={model_name} len={seq_len} rc={rc}")
    return model_dir


def _extract_observed_thresholds(
    *,
    model_name: str,
    seq_len: int,
    model_dir: Path,
    n_total_heads: int,
) -> dict[str, Any]:
    fit_path = model_dir / "curve_fit_comparison.json"
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    fits = fit.get("curve_fits", {})
    key = "high_to_low::wiki_ntp::loss"
    if key not in fits:
        cand = [k for k in fits if k.startswith("high_to_low::") and k.endswith("::loss")]
        if not cand:
            raise RuntimeError(f"No high_to_low loss key in {fit_path}")
        key = sorted(cand)[0]

    frac = safe_float(fits[key]["threshold_piecewise"]["best_threshold_fraction"])
    frac = float(max(0.0, min(1.0, frac)))
    observed_mstar = float(frac * float(n_total_heads))

    curve_path = model_dir / "cumulative_ablation_curve.parquet"
    curve_df = pd.read_parquet(curve_path)
    curve_df = curve_df[
        (curve_df["sort_order"].astype(str) == "high_to_low")
        & (curve_df["task"].astype(str) == "wiki_ntp")
        & (curve_df["metric_name"].astype(str) == "loss")
    ].copy()
    nearest_m = float("nan")
    nearest_frac = float("nan")
    if not curve_df.empty:
        curve_df["frac_norm"] = curve_df["ablation_fraction"].astype(float) / 100.0
        curve_df["abs_delta"] = (curve_df["frac_norm"] - frac).abs()
        row = curve_df.sort_values("abs_delta").iloc[0]
        nearest_m = float(row["n_heads_ablated"])
        nearest_frac = float(row["frac_norm"])

    return {
        "model": model_name,
        "seq_len": int(seq_len),
        "fit_key": key,
        "threshold_fraction": frac,
        "n_total_heads": int(n_total_heads),
        "observed_mstar_fractional": observed_mstar,
        "observed_mstar_nearest_heads": nearest_m,
        "observed_mstar_nearest_fraction": nearest_frac,
    }


def run_7b(
    *,
    output_root: Path,
    device: str,
    models: list[str],
    seq_lens: list[int],
    calibration_mode: str = "single_condition",
    calibration_model: str | None = None,
    calibration_seq_len: int | None = None,
    epsilon: float = 0.01,
    sparsity_sequences: int = 6,
    c1_num_seeds: int = 3,
    c1_synthetic_count: int = 64,
    c1_ntp_count_per_seed: int = 64,
    c1_batch_size_synth: int = 8,
    c1_batch_size_ntp: int = 2,
    force_rerun_c1: bool = False,
    reuse_effective_sparsity: bool = False,
) -> dict[str, Any]:
    out_dir = output_root / "exp7b_phase_transition"
    out_dir.mkdir(parents=True, exist_ok=True)
    c1_root = out_dir / "c1_runs"
    c1_root.mkdir(parents=True, exist_ok=True)

    sparsity_rows: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, Any]] = []

    for model_name in models:
        n_heads = _n_total_heads(model_name)
        for seq_len in seq_lens:
            model_dir = _run_c1_for_len(
                model_name=model_name,
                seq_len=int(seq_len),
                device=device,
                output_root=c1_root,
                num_seeds=c1_num_seeds,
                synthetic_count=c1_synthetic_count,
                ntp_count_per_seed=c1_ntp_count_per_seed,
                batch_size_synth=c1_batch_size_synth,
                batch_size_ntp=c1_batch_size_ntp,
                force=force_rerun_c1,
            )
            threshold_rows.append(
                _extract_observed_thresholds(
                    model_name=model_name,
                    seq_len=int(seq_len),
                    model_dir=model_dir,
                    n_total_heads=n_heads,
                )
            )
            if not reuse_effective_sparsity:
                s_df = _estimate_effective_sparsity(
                    model_name=model_name,
                    seq_len=int(seq_len),
                    device=device,
                    epsilon=float(epsilon),
                    num_sequences=int(sparsity_sequences),
                )
                sparsity_rows.append(s_df)

    if reuse_effective_sparsity:
        eff_path = out_dir / "effective_sparsity.parquet"
        if not eff_path.exists():
            raise FileNotFoundError(
                f"reuse_effective_sparsity=True requested but missing {eff_path}"
            )
        sparsity_df = pd.read_parquet(eff_path)
    else:
        sparsity_df = pd.concat(sparsity_rows, ignore_index=True, sort=False)
        sparsity_df.to_parquet(out_dir / "effective_sparsity.parquet", index=False)

    s_summary = (
        sparsity_df.groupby(["model", "seq_len"], as_index=False)["effective_sparsity"]
        .mean()
        .rename(columns={"effective_sparsity": "s_eff_high_si_mean"})
    )
    thr_df = pd.DataFrame(threshold_rows)
    merged = thr_df.merge(s_summary, on=["model", "seq_len"], how="left")

    merged["x_predictor"] = merged.apply(
        lambda r: float(r["s_eff_high_si_mean"]) * math.log(max(1.000001, float(r["seq_len"]) / max(1e-6, float(r["s_eff_high_si_mean"])))),
        axis=1,
    )
    x = merged["x_predictor"].to_numpy(dtype=np.float64)
    y = merged["observed_mstar_fractional"].to_numpy(dtype=np.float64)
    denom = float(np.dot(x, x))
    c_hat_global = float(np.dot(x, y) / denom) if denom > 0 else float("nan")
    if np.isfinite(c_hat_global):
        c_hat_global = float(max(0.0, c_hat_global))

    cal_model = str(calibration_model or (models[0] if models else ""))
    cal_seq = int(calibration_seq_len if calibration_seq_len is not None else (seq_lens[0] if seq_lens else 256))
    cal_mask = (merged["model"].astype(str) == cal_model) & (merged["seq_len"].astype(int) == int(cal_seq))
    if cal_mask.any():
        cal_row = merged.loc[cal_mask].iloc[0]
        x_cal = float(cal_row["x_predictor"])
        y_cal = float(cal_row["observed_mstar_fractional"])
        c_hat_single = float(y_cal / x_cal) if abs(x_cal) > 1e-12 else float("nan")
    else:
        c_hat_single = float("nan")

    use_single = str(calibration_mode).strip().lower() == "single_condition" and np.isfinite(c_hat_single)
    c_hat = float(c_hat_single if use_single else c_hat_global)

    merged["calibration_mode_used"] = "single_condition" if use_single else "global_fit"
    merged["c_hat"] = c_hat
    merged["c_hat_global"] = c_hat_global
    merged["c_hat_single"] = c_hat_single
    merged["predicted_mstar"] = merged["x_predictor"] * c_hat

    pred = merged["predicted_mstar"].to_numpy(dtype=np.float64)
    obs = merged["observed_mstar_fractional"].to_numpy(dtype=np.float64)
    if len(pred) >= 2 and float(np.std(pred)) > 1e-12 and float(np.std(obs)) > 1e-12:
        pear = scipy_stats.pearsonr(pred, obs)
        spear = scipy_stats.spearmanr(pred, obs)
        pear_r, pear_p = float(pear.statistic), float(pear.pvalue)
        spear_r, spear_p = float(spear.statistic), float(spear.pvalue)
    else:
        pear_r = float("nan")
        pear_p = float("nan")
        spear_r = float("nan")
        spear_p = float("nan")
    cal_excluded = merged.loc[~cal_mask].copy() if cal_mask.any() else merged.copy()
    if not cal_excluded.empty:
        pred_oos = cal_excluded["predicted_mstar"].to_numpy(dtype=np.float64)
        obs_oos = cal_excluded["observed_mstar_fractional"].to_numpy(dtype=np.float64)
        if len(pred_oos) >= 2 and float(np.std(pred_oos)) > 1e-12 and float(np.std(obs_oos)) > 1e-12:
            pear_oos = scipy_stats.pearsonr(pred_oos, obs_oos)
            spear_oos = scipy_stats.spearmanr(pred_oos, obs_oos)
            pear_r_oos, pear_p_oos = float(pear_oos.statistic), float(pear_oos.pvalue)
            spear_r_oos, spear_p_oos = float(spear_oos.statistic), float(spear_oos.pvalue)
        else:
            pear_r_oos = float("nan")
            pear_p_oos = float("nan")
            spear_r_oos = float("nan")
            spear_p_oos = float("nan")
    else:
        pear_r_oos = float("nan")
        pear_p_oos = float("nan")
        spear_r_oos = float("nan")
        spear_p_oos = float("nan")

    merged.to_parquet(out_dir / "mstar_predictions.parquet", index=False)

    summary = {
        "timestamp": now_timestamp(),
        "experiment": "7B",
        "status": "completed",
        "models": models,
        "seq_lens": [int(x) for x in seq_lens],
        "epsilon": float(epsilon),
        "calibration": {
            "requested_mode": str(calibration_mode),
            "mode_used": "single_condition" if use_single else "global_fit",
            "model": cal_model,
            "seq_len": int(cal_seq),
            "c_hat": float(c_hat),
            "c_hat_single": float(c_hat_single),
            "c_hat_global": float(c_hat_global),
            "single_condition_available": bool(cal_mask.any()),
        },
        "prediction_vs_observed": {
            "pearson_r": pear_r,
            "pearson_p": pear_p,
            "spearman_rho": spear_r,
            "spearman_p": spear_p,
        },
        "prediction_vs_observed_out_of_sample": {
            "excluded_condition": {"model": cal_model, "seq_len": int(cal_seq)},
            "pearson_r": pear_r_oos,
            "pearson_p": pear_p_oos,
            "spearman_rho": spear_r_oos,
            "spearman_p": spear_p_oos,
            "n_conditions": int(len(cal_excluded)),
        },
        "artifacts": {
            "effective_sparsity": str(out_dir / "effective_sparsity.parquet"),
            "mstar_predictions": str(out_dir / "mstar_predictions.parquet"),
            "c1_root": str(c1_root),
        },
    }
    write_json(out_dir / "phase_transition_scaling.json", summary)
    write_json(
        out_dir / "run_manifest.json",
        {
            "timestamp": now_timestamp(),
            "experiment": "7B",
            "status": "completed",
            "device": str(device),
            "models": models,
            "seq_lens": [int(x) for x in seq_lens],
            "epsilon": float(epsilon),
            "calibration": {
                "requested_mode": str(calibration_mode),
                "mode_used": "single_condition" if use_single else "global_fit",
                "model": cal_model,
                "seq_len": int(cal_seq),
                "c_hat": float(c_hat),
                "c_hat_single": float(c_hat_single),
                "c_hat_global": float(c_hat_global),
            },
            "c1_params": {
                "num_seeds": int(c1_num_seeds),
                "synthetic_count": int(c1_synthetic_count),
                "ntp_count_per_seed": int(c1_ntp_count_per_seed),
                "batch_size_synth": int(c1_batch_size_synth),
                "batch_size_ntp": int(c1_batch_size_ntp),
                "force_rerun_c1": bool(force_rerun_c1),
                "reuse_effective_sparsity": bool(reuse_effective_sparsity),
            },
            "artifacts": {
                "effective_sparsity": str(out_dir / "effective_sparsity.parquet"),
                "mstar_predictions": str(out_dir / "mstar_predictions.parquet"),
                "phase_transition_scaling": str(out_dir / "phase_transition_scaling.json"),
                "run_manifest": str(out_dir / "run_manifest.json"),
            },
        },
    )
    return summary
