#!/usr/bin/env python3
"""E20 — Heterogeneity Variance Decomposition (NEW-R20).

Implements two modes:
  - proxy: feasible short-run matched-training variance decomposition
  - fullscale: long-run scaffold (execution guarded by explicit ack flag)
"""
from __future__ import annotations

import argparse
import math
import time
import sys
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
    ensure_profile_sequence_cache,
    emit_core_artifacts,
    enforce_coverage_contract,
    parse_device_map,
    parse_models_arg,
    read_json,
)
from experiment3.theory1_si_circuits import MODELS  # noqa: E402
from experiment4.lora import LoRAHyperParams  # noqa: E402
from experiment4.pipeline import (  # noqa: E402
    AuxLossConfig,
    TrainConfig,
    _train_single_run,
    post_ft_si_audit,
    run_c1_mini_ablation,
)

OUT_ROOT = RESULTS_ROOT / "E20_heterogeneity_variance_decomp"

PROXY_DEFAULTS = {
    "max_steps": 120,
    "batch_size": 1,
    "lr": 2e-4,
    "weight_decay": 0.0,
    "warmup_steps": 12,
    "grad_clip": 1.0,
    "max_length": 256,
    "log_every": 20,
    "aux_interval": 20,
    "math_per_task": 96,
    "control_fraction": 0.20,
    "c1_synth_count": 48,
}

FULLSCALE_DEFAULTS = {
    "max_steps": 1200,
    "batch_size": 1,
    "lr": 2e-4,
    "weight_decay": 0.0,
    "warmup_steps": 120,
    "grad_clip": 1.0,
    "max_length": 256,
    "log_every": 25,
    "aux_interval": 25,
    "math_per_task": 400,
    "control_fraction": 0.20,
    "c1_synth_count": 160,
}


def _parse_seed_list(raw: str) -> list[int]:
    vals = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise RuntimeError("[E20] hard_fail_reason: empty --seed-list")
    return sorted(set(vals))


def _defaults_for_mode(mode: str, smoke: bool) -> dict[str, Any]:
    d = dict(PROXY_DEFAULTS if mode == "proxy" else FULLSCALE_DEFAULTS)
    if smoke:
        d["max_steps"] = min(int(d["max_steps"]), 24)
        d["math_per_task"] = min(int(d["math_per_task"]), 24)
        d["c1_synth_count"] = min(int(d["c1_synth_count"]), 16)
    return d


def _build_train_cfg(mode: str, smoke: bool) -> tuple[TrainConfig, int]:
    d = _defaults_for_mode(mode, smoke)
    cfg = TrainConfig(
        max_steps=int(d["max_steps"]),
        batch_size=int(d["batch_size"]),
        lr=float(d["lr"]),
        weight_decay=float(d["weight_decay"]),
        warmup_steps=int(d["warmup_steps"]),
        grad_clip=float(d["grad_clip"]),
        max_length=int(d["max_length"]),
        log_every=int(d["log_every"]),
        aux_interval=int(d["aux_interval"]),
        math_per_task=int(d["math_per_task"]),
        control_fraction=float(d["control_fraction"]),
    )
    return cfg, int(d["c1_synth_count"])


def _lora_hp() -> LoRAHyperParams:
    return LoRAHyperParams(
        rank=8,
        alpha=16.0,
        dropout=0.0,
        si_rank=1,
        si_amplified_rank=16,
        non_si_reduced_rank=4,
    )


def _require_head_groups(model_name: str) -> None:
    p = ROOT / "results" / "experiment3" / "theory1_si_circuits" / model_name / "head_groups.json"
    if not p.exists():
        raise RuntimeError(
            f"[E20] hard_fail_reason: missing baseline SI metadata for {model_name}: {p}. "
            "Run Experiment 3 Theory1 profiling first."
        )


def _required_control_count(train_cfg: TrainConfig) -> int:
    n_math = int(train_cfg.math_per_task) * 5
    frac = float(train_cfg.control_fraction)
    denom = max(1e-8, 1.0 - frac)
    n_control = int(round((frac / denom) * n_math))
    return max(1, n_control)


def _extract_c1_metrics(df: pd.DataFrame) -> dict[str, float]:
    out = {
        "c1_overall_degradation_mean": float("nan"),
        "c1_local_keymatch_deg": float("nan"),
        "c1_longrange_deg": float("nan"),
    }
    if df is None or df.empty:
        return out

    frac_vals = sorted(df["ablation_fraction"].dropna().unique().tolist())
    target_frac = 20 if 20 in frac_vals else int(frac_vals[-1])
    sub = df[df["ablation_fraction"] == target_frac].copy()
    if sub.empty:
        return out

    out["c1_overall_degradation_mean"] = float(np.nanmean(sub["degradation"].to_numpy(dtype=float)))

    lk = sub[sub["task"].astype(str) == "local_key_match"]
    if not lk.empty:
        out["c1_local_keymatch_deg"] = float(np.nanmean(lk["degradation"].to_numpy(dtype=float)))

    lr = sub[sub["task"].astype(str) == "long_range_retrieval"]
    if not lr.empty:
        out["c1_longrange_deg"] = float(np.nanmean(lr["degradation"].to_numpy(dtype=float)))

    return out


def _extract_boundary_metrics(audit: dict[str, Any], *, model_name: str, seed: int) -> tuple[float, bool]:
    boundary = audit.get("boundary", None)
    if not isinstance(boundary, dict) or ("post_ablation_d" not in boundary):
        raise RuntimeError(
            "[E20] hard_fail_reason: post_ft_si_audit output missing required key "
            f"'boundary.post_ablation_d' for model={model_name} seed={seed}. "
            "Audit schema may have changed; aborting to prevent silent NaN boundary metrics."
        )
    return (
        float(boundary.get("post_ablation_d", float("nan"))),
        bool(boundary.get("prefix_following_artifact_flag", False)),
    )


def run_seed(
    *,
    model_name: str,
    seed: int,
    mode: str,
    device: str,
    out_root: Path,
    smoke: bool,
) -> dict[str, Any]:
    _require_head_groups(model_name)

    train_cfg, c1_synth = _build_train_cfg(mode, smoke)
    hp = _lora_hp()
    aux_cfg = AuxLossConfig(mode="none", lambda_preserve=0.0, lambda_routing=0.0)

    seed_dir = ensure_dir(out_root / model_name / f"seed_{seed}")
    t0 = time.time()

    model, tokenizer, train_summary = _train_single_run(
        model_name=model_name,
        condition="d_full_qlora_baseline",
        seed=int(seed),
        device=device,
        output_dir=seed_dir / "train_run",
        train_cfg=train_cfg,
        lora_hp=hp,
        aux_cfg=aux_cfg,
    )

    audit = post_ft_si_audit(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        device=device,
        output_dir=seed_dir / "post_ft_audit",
        r2_sequences=8 if smoke else 16,
        seq_len=256,
        synthetic_target_per_cell=32 if smoke else 96,
    )

    c1_path = seed_dir / "c1_mini_ablation.parquet"
    c1_df = run_c1_mini_ablation(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        device=device,
        output_path=c1_path,
        fractions=(0, 5, 10, 20),
        num_seeds=1,
        synthetic_count=max(8, int(c1_synth)),
        batch_size=2,
    )
    c1_metrics = _extract_c1_metrics(c1_df)
    boundary_d, boundary_artifact_flag = _extract_boundary_metrics(
        audit,
        model_name=model_name,
        seed=int(seed),
    )

    try:
        del model
    except Exception:
        pass
    try:
        del tokenizer
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rec = {
        "timestamp": timestamp_now(),
        "experiment_id": "E20",
        "mode": mode,
        "model": model_name,
        "seed": int(seed),
        "runtime_sec": float(time.time() - t0),
        "train_elapsed_sec": float(train_summary.get("elapsed_sec", float("nan"))),
        "final_total_loss": float(train_summary.get("final_total_loss", float("nan"))),
        "si_mean_r2": float(audit.get("r2_summary", {}).get("mean", float("nan"))),
        "si_high_mean_r2": float(audit.get("r2_summary", {}).get("high_mean", float("nan"))),
        "boundary_d": boundary_d,
        "boundary_artifact_flag": boundary_artifact_flag,
        **c1_metrics,
    }
    write_json(seed_dir / "summary.json", rec)
    write_json(seed_dir / "train_summary.json", train_summary)
    return rec


def _load_seed_summary(model_name: str, seed: int, out_root: Path) -> dict[str, Any]:
    p = out_root / model_name / f"seed_{seed}" / "summary.json"
    if not p.exists():
        raise RuntimeError(f"[E20] hard_fail_reason: missing shard artifact {p}")
    return read_json(p)


def _nanvar(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    return float(np.var(arr, ddof=1))


def _finalize(models: list[str], seeds: list[int], out_root: Path, mode: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for model in models:
        for seed in seeds:
            rows.append(_load_seed_summary(model, seed, out_root))

    enforce_coverage_contract(
        experiment_id="E20",
        observed_models=[str(r.get("model", "")) for r in rows],
        required_models=models,
        observed_counts={m: sum(1 for r in rows if str(r.get("model")) == m) for m in models},
        min_counts={m: int(len(seeds)) for m in models},
    )

    df = pd.DataFrame(rows)
    ensure_dir(out_root)
    df.to_parquet(out_root / "per_seed_metrics.parquet", index=False)

    per_model_rows: list[dict[str, Any]] = []
    model_means: list[float] = []
    within_vars: list[float] = []
    for model in models:
        sub = df[df["model"] == model].copy()
        si_vals = sub["si_mean_r2"].to_numpy(dtype=float)
        c1_vals = sub["c1_overall_degradation_mean"].to_numpy(dtype=float)
        rec = {
            "model": model,
            "n_seeds": int(len(sub)),
            "mean_si_r2": float(np.nanmean(si_vals)),
            "std_si_r2": float(np.nanstd(si_vals, ddof=1)) if np.isfinite(np.nanstd(si_vals, ddof=1)) else float("nan"),
            "mean_c1_deg": float(np.nanmean(c1_vals)),
            "within_seed_var_si": _nanvar(si_vals.tolist()),
        }
        per_model_rows.append(rec)
        model_means.append(rec["mean_si_r2"])
        within_vars.append(rec["within_seed_var_si"])

    between_model_var = _nanvar(model_means)
    within_seed_var = float(np.nanmean(np.asarray(within_vars, dtype=float)))
    ratio = float(between_model_var / within_seed_var) if np.isfinite(between_model_var) and np.isfinite(within_seed_var) and within_seed_var > 0 else float("nan")

    if np.isfinite(between_model_var) and np.isfinite(within_seed_var):
        if between_model_var > (1.25 * within_seed_var):
            interpretation = "model_family_dominant"
            claim_status = "supported_with_caveat"
            note = "Between-model SI variance exceeds within-model seed variance."
        elif within_seed_var > (1.25 * between_model_var):
            interpretation = "run_variance_dominant"
            claim_status = "mixed"
            note = "Within-model seed variance is comparable to or exceeds between-model variance."
        else:
            interpretation = "variance_components_overlap"
            claim_status = "inconclusive"
            note = "Between-model and within-model variance components overlap materially."
    else:
        interpretation = "insufficient_variance_data"
        claim_status = "pending"
        note = "Insufficient finite variance statistics for decomposition."

    cross = {
        "timestamp": timestamp_now(),
        "experiment_id": "E20",
        "mode": mode,
        "n_models": int(len(models)),
        "seeds_per_model": int(len(seeds)),
        "between_model_var_si": between_model_var,
        "within_model_seed_var_si": within_seed_var,
        "between_over_within_ratio": ratio,
        "interpretation": interpretation,
        "note": note,
        "per_model": per_model_rows,
    }
    write_json(out_root / "cross_model_variance_summary.json", cross)

    claim_impact = {
        "timestamp": timestamp_now(),
        "experiment_id": "E20",
        "claim_status": claim_status,
        "supports_main_text": claim_status in ("supported", "supported_with_caveat", "mixed"),
        "notes": [note],
        "outcome_summary": note,
    }

    prereg = {
        "experiment_id": "E20",
        "mode": mode,
        "question": "Does cross-model SI heterogeneity exceed matched-setup training-run variance?",
        "primary_hypothesis": "Between-model SI variance exceeds within-model seed variance under matched proxy training.",
        "primary_endpoints": [
            "between_model_var_si",
            "within_model_seed_var_si",
            "between_over_within_ratio",
        ],
        "secondary_endpoints": [
            "mean_c1_deg",
            "boundary_d",
            "si_high_mean_r2",
        ],
        "model_list": models,
        "seed_list": [int(s) for s in seeds],
        "sample_size_plan": {
            "seeds_per_model": int(len(seeds)),
            "models": int(len(models)),
        },
        "acceptance_criteria": [
            "between_model_var_si > 1.25 * within_model_seed_var_si indicates model-family-dominant heterogeneity",
        ],
        "fallback_interpretation_if_null": "Heterogeneity remains descriptive; variance decomposition does not isolate model-family dominance.",
    }

    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": "E20",
        "analysis_tier": "confirmatory",
        "canonical_eligible": True,
        "override_used": False,
        "verdict": {
            "interpretation": interpretation,
            "between_over_within_ratio": ratio,
            "n_models": int(len(models)),
            "seeds_per_model": int(len(seeds)),
        },
        "limitations": [
            "Proxy training recipe is short-horizon and does not replace true pretraining-scale variance decomposition.",
            "Head-group priors are inherited from baseline SI profiling artifacts.",
        ],
    }

    data_dictionary = {
        "experiment_id": "E20",
        "tables": [
            {
                "path": "per_seed_metrics.parquet",
                "description": "Per-seed matched-training outcomes for variance decomposition.",
                "columns": [
                    {"name": "model", "dtype": "str", "description": "model name"},
                    {"name": "seed", "dtype": "int", "description": "training seed"},
                    {"name": "si_mean_r2", "dtype": "float", "description": "post-FT SI mean R2"},
                    {"name": "si_high_mean_r2", "dtype": "float", "description": "post-FT high-SI mean R2"},
                    {"name": "boundary_d", "dtype": "float", "description": "post-FT boundary effect size (Cohen's d)"},
                    {"name": "boundary_artifact_flag", "dtype": "bool", "description": "synthetic boundary artifact guard flag"},
                    {"name": "c1_overall_degradation_mean", "dtype": "float", "description": "post-FT mini-ablation mean degradation"},
                    {"name": "runtime_sec", "dtype": "float", "description": "wall-clock runtime"},
                ],
            }
        ],
    }

    emit_core_artifacts(
        experiment_id="E20",
        out_dir=out_root,
        preregistration=prereg,
        manifest_extra={"models": models, "seeds": [int(s) for s in seeds], "mode": mode},
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )
    return cross


def main() -> None:
    p = argparse.ArgumentParser(description="E20: heterogeneity variance decomposition", allow_abbrev=False)
    p.add_argument("--mode", choices=["proxy", "fullscale"], default="proxy")
    p.add_argument("--models", default=",".join(PRIMARY_MODELS))
    p.add_argument(
        "--device-map",
        default="llama-3.1-8b:cuda:0,olmo-2-7b:cuda:0,mistral-7b-v0.1:cuda:0",
    )
    p.add_argument("--seed-list", default="0,1,2")
    p.add_argument("--output-root", default=str(OUT_ROOT))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--finalize-only", action="store_true")
    p.add_argument("--no-finalize", action="store_true")
    p.add_argument("--ack-fullscale-execution", action="store_true")
    args = p.parse_args()

    if args.finalize_only and args.no_finalize:
        raise RuntimeError("[E20] hard_fail_reason: --finalize-only and --no-finalize are mutually exclusive")
    if args.mode == "fullscale" and (not args.ack_fullscale_execution) and (not args.finalize_only):
        raise RuntimeError(
            "[E20] hard_fail_reason: fullscale mode requires explicit --ack-fullscale-execution flag"
        )

    models = parse_models_arg(args.models, default=PRIMARY_MODELS)
    device_map = parse_device_map(args.device_map)
    seeds = _parse_seed_list(args.seed_list)
    out_root = ensure_dir(Path(args.output_root))

    if args.finalize_only:
        cross = _finalize(models=models, seeds=seeds, out_root=out_root, mode=args.mode)
        print(f"[E20] Finalized from shards. interpretation={cross['interpretation']}", flush=True)
        return

    # Preflight: ensure baseline SI metadata and tokenized control-profile cache.
    train_cfg_for_preflight, _ = _build_train_cfg(args.mode, bool(args.smoke))
    n_control = _required_control_count(train_cfg_for_preflight)
    for model_name in models:
        _require_head_groups(model_name)
        spec = MODELS.get(model_name)
        if spec is None:
            raise RuntimeError(
                f"[E20] hard_fail_reason: model '{model_name}' missing from theory1 MODELS registry"
            )
        from shared.models.loading import load_tokenizer as _proj_load_tokenizer  # noqa: PLC0415

        tokenizer = _proj_load_tokenizer(spec)
        cache_status = ensure_profile_sequence_cache(
            tokenizer=tokenizer,
            model_name=model_name,
            min_sequences=max(128, n_control + 32),
            seq_len=int(train_cfg_for_preflight.max_length),
            seed=20260501,
        )
        print(f"[E20] preflight model={model_name}: {cache_status}", flush=True)

    for model_name in models:
        if model_name not in PRIMARY_MODELS:
            raise RuntimeError(
                f"[E20] hard_fail_reason: model '{model_name}' unsupported for NEW-R20; "
                f"expected subset of {list(PRIMARY_MODELS)}"
            )
        device = device_map.get(model_name, "cuda:0")
        for seed in seeds:
            print(f"[E20] Running model={model_name} seed={seed} mode={args.mode} device={device}", flush=True)
            run_seed(
                model_name=model_name,
                seed=int(seed),
                mode=args.mode,
                device=device,
                out_root=out_root,
                smoke=bool(args.smoke),
            )

    if args.no_finalize:
        print("[E20] Shard run complete (no finalize).", flush=True)
        return

    cross = _finalize(models=models, seeds=seeds, out_root=out_root, mode=args.mode)
    print(f"[E20] Done. interpretation={cross['interpretation']}", flush=True)


if __name__ == "__main__":
    main()
