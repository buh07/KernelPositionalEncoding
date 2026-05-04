#!/usr/bin/env python3
"""E21 — Matched PE-family contrast (NEW-R21).

Implements two modes:
  - proxy: feasible matched RoPE-vs-NoPE short-run training/eval contrast
  - fullscale: long-run scaffold (execution guarded by explicit ack flag)
"""
from __future__ import annotations

import argparse
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
    RESULTS_ROOT,
    ensure_dir,
    timestamp_now,
    write_json,
)
from reinforce_exp3.scripts._shared import (  # noqa: E402
    ensure_profile_sequence_cache,
    emit_core_artifacts,
    enforce_coverage_contract,
    load_model_for_exp,
    parse_device_map,
    parse_models_arg,
    read_json,
    synthetic_token_sequences,
)
from experiment4.lora import LoRAHyperParams  # noqa: E402
from experiment4.pipeline import (  # noqa: E402
    AuxLossConfig,
    TrainConfig,
    _train_single_run,
    post_ft_si_audit,
    run_c1_mini_ablation,
)
from experiment3.theory1_si_circuits import (  # noqa: E402
    MODELS,
    classify_heads,
    compute_per_head_r2,
    load_profile_sequences,
)
from shared.attention.adapters import get_adapter  # noqa: E402

OUT_ROOT = RESULTS_ROOT / "E21_pe_family_training_contrast"
R21_PROXY_MODELS = ("tinyllama-1.1b", "tinyllama-nope-1.1b")

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
        raise RuntimeError("[E21] hard_fail_reason: empty --seed-list")
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


def _ensure_head_groups(model_name: str, device: str, seq_len: int = 256, n_seq: int = 16) -> dict[str, Any]:
    out_dir = ROOT / "results" / "experiment3" / "theory1_si_circuits" / model_name
    head_groups_path = out_dir / "head_groups.json"
    r2_summary_path = out_dir / "head_r2_summary.parquet"
    if head_groups_path.exists():
        return {
            "status": "existing",
            "path": str(head_groups_path),
            "r2_summary_present": bool(r2_summary_path.exists()),
        }

    if model_name not in MODELS:
        raise RuntimeError(
            f"[E21] hard_fail_reason: model '{model_name}' is not registered in theory1 MODELS. "
            "Add a ModelSpec entry before running NEW-R21."
        )

    try:
        model, tokenizer = load_model_for_exp(model_name, device=device, attn_implementation="eager")
    except Exception as exc:
        raise RuntimeError(
            f"[E21] hard_fail_reason: failed to load model '{model_name}'. "
            "If this is tinyllama-nope, verify Hugging Face connectivity and local cache/weights availability. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    cache_status = ensure_profile_sequence_cache(
        tokenizer=tokenizer,
        model_name=model_name,
        min_sequences=max(128, int(n_seq) * 8),
        seq_len=max(128, int(seq_len)),
        seed=20260501,
    )

    seqs: list[list[int]] = []
    try:
        seqs = load_profile_sequences(tokenizer=tokenizer, model_name=model_name, num_sequences=int(n_seq), seq_len=int(seq_len))
    except Exception:
        seqs = []
    if not seqs:
        seqs = synthetic_token_sequences(
            tokenizer=tokenizer,
            n=max(8, int(n_seq)),
            seq_len=max(128, int(seq_len)),
            seed=20260501,
        )

    adapter = get_adapter(MODELS[model_name])
    try:
        adapter.register(model)
        r2_df = compute_per_head_r2(
            model=model,
            adapter=adapter,
            tokenizer=tokenizer,
            model_spec=MODELS[model_name],
            device=device,
            sequences=seqs,
        )
    except Exception as exc:
        raise RuntimeError(
            f"[E21] hard_fail_reason: failed during SI headgroup derivation for '{model_name}'. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        try:
            adapter.cleanup()
        except Exception:
            pass

    high_si, low_si, mean_r2 = classify_heads(r2_df)
    if len(high_si) == 0 or len(low_si) == 0:
        raise RuntimeError(f"[E21] hard_fail_reason: failed to derive non-empty SI head groups for {model_name}")

    ensure_dir(out_dir)
    if head_groups_path.exists():
        # Avoid mutating shared baseline priors if a concurrent run produced them first.
        return {
            "status": "existing",
            "path": str(head_groups_path),
            "r2_summary_present": bool(r2_summary_path.exists()),
            "note": "head_groups existed at write time; skipped overwrite",
        }
    write_json(
        head_groups_path,
        {
            "timestamp": timestamp_now(),
            "source": "E21_auto_profile",
            "high_si": [{"layer": int(h.layer), "head": int(h.head)} for h in high_si],
            "low_si": [{"layer": int(h.layer), "head": int(h.head)} for h in low_si],
        },
    )
    r2_df.to_parquet(out_dir / "per_sequence_r2.parquet", index=False)
    mean_r2.to_parquet(r2_summary_path, index=False)

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

    return {
        "status": "generated",
        "path": str(head_groups_path),
        "num_sequences": int(len(seqs)),
        "profile_cache": cache_status,
    }


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
            "[E21] hard_fail_reason: post_ft_si_audit output missing required key "
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

    pe_family = "rope" if model_name == "tinyllama-1.1b" else "nope"
    rec = {
        "timestamp": timestamp_now(),
        "experiment_id": "E21",
        "mode": mode,
        "model": model_name,
        "pe_family": pe_family,
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
        raise RuntimeError(f"[E21] hard_fail_reason: missing shard artifact {p}")
    return read_json(p)


def _bootstrap_ci(vals: np.ndarray, n_boot: int = 4000, seed: int = 20260501) -> dict[str, float]:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    rng = np.random.default_rng(int(seed))
    boot = np.empty(int(n_boot), dtype=float)
    n = int(arr.size)
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(np.mean(arr[idx]))
    return {
        "mean": float(np.mean(arr)),
        "ci_lo": float(np.quantile(boot, 0.025)),
        "ci_hi": float(np.quantile(boot, 0.975)),
    }


def _finalize(models: list[str], seeds: list[int], out_root: Path, mode: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for model in models:
        for seed in seeds:
            rows.append(_load_seed_summary(model, seed, out_root))

    enforce_coverage_contract(
        experiment_id="E21",
        observed_models=[str(r.get("model", "")) for r in rows],
        required_models=models,
        observed_counts={m: sum(1 for r in rows if str(r.get("model")) == m) for m in models},
        min_counts={m: int(len(seeds)) for m in models},
    )

    df = pd.DataFrame(rows)
    ensure_dir(out_root)
    df.to_parquet(out_root / "per_seed_metrics.parquet", index=False)

    rope = df[df["model"] == "tinyllama-1.1b"].copy()
    nope = df[df["model"] == "tinyllama-nope-1.1b"].copy()
    paired = rope[["seed", "si_mean_r2", "c1_overall_degradation_mean", "boundary_d"]].merge(
        nope[["seed", "si_mean_r2", "c1_overall_degradation_mean", "boundary_d"]],
        on="seed",
        suffixes=("_rope", "_nope"),
        how="inner",
    )
    if paired.empty:
        raise RuntimeError("[E21] hard_fail_reason: no paired seed rows for rope/nope comparison")

    paired["delta_si_mean_r2"] = paired["si_mean_r2_rope"] - paired["si_mean_r2_nope"]
    paired["delta_c1_deg"] = paired["c1_overall_degradation_mean_rope"] - paired["c1_overall_degradation_mean_nope"]
    paired["delta_boundary_d"] = paired["boundary_d_rope"] - paired["boundary_d_nope"]
    paired.to_parquet(out_root / "paired_seed_deltas.parquet", index=False)

    si_ci = _bootstrap_ci(paired["delta_si_mean_r2"].to_numpy(dtype=float))
    c1_ci = _bootstrap_ci(paired["delta_c1_deg"].to_numpy(dtype=float), seed=20260502)
    b_ci = _bootstrap_ci(paired["delta_boundary_d"].to_numpy(dtype=float), seed=20260503)

    pos = int(np.sum(paired["delta_si_mean_r2"].to_numpy(dtype=float) > 0))
    n = int(len(paired))

    if np.isfinite(si_ci["ci_lo"]) and si_ci["ci_lo"] > 0:
        interpretation = "rope_gt_nope_proxy"
        claim_status = "proxy_specific"
        note = "Proxy matched training shows higher SI mean R2 for RoPE than NoPE across paired seeds."
    elif np.isfinite(si_ci["mean"]) and si_ci["mean"] > 0:
        interpretation = "rope_gt_nope_uncertain"
        claim_status = "mixed"
        note = "RoPE-NoPE SI difference is positive on average but uncertainty interval overlaps zero."
    elif np.isfinite(si_ci["ci_hi"]) and si_ci["ci_hi"] < 0:
        interpretation = "nope_ge_rope"
        claim_status = "not_supported"
        note = "Proxy matched training does not support RoPE > NoPE for SI mean R2."
    else:
        interpretation = "contrast_inconclusive"
        claim_status = "inconclusive"
        note = "RoPE-vs-NoPE proxy contrast remains inconclusive."

    per_model_rows = []
    for model in models:
        sub = df[df["model"] == model]
        per_model_rows.append(
            {
                "model": model,
                "n_seeds": int(len(sub)),
                "mean_si_r2": float(np.nanmean(sub["si_mean_r2"].to_numpy(dtype=float))),
                "mean_c1_deg": float(np.nanmean(sub["c1_overall_degradation_mean"].to_numpy(dtype=float))),
                "mean_boundary_d": float(np.nanmean(sub["boundary_d"].to_numpy(dtype=float))),
            }
        )

    cross = {
        "timestamp": timestamp_now(),
        "experiment_id": "E21",
        "mode": mode,
        "n_paired_seeds": n,
        "n_positive_delta_si": pos,
        "si_delta_summary": si_ci,
        "c1_delta_summary": c1_ci,
        "boundary_delta_summary": b_ci,
        "interpretation": interpretation,
        "note": note,
        "per_model": per_model_rows,
    }
    write_json(out_root / "cross_family_contrast_summary.json", cross)

    claim_impact = {
        "timestamp": timestamp_now(),
        "experiment_id": "E21",
        "claim_status": claim_status,
        "supports_main_text": claim_status in ("supported", "supported_with_caveat", "proxy_specific", "mixed"),
        "notes": [note, "Proxy result is not equivalent to pretraining-scale PE-family attribution."],
        "outcome_summary": note,
    }

    prereg = {
        "experiment_id": "E21",
        "mode": mode,
        "question": "Under matched proxy training, does RoPE yield stronger SI structure than NoPE?",
        "primary_hypothesis": "TinyLlama RoPE has higher post-FT SI mean R2 than TinyLlama-NoPE under matched recipe.",
        "primary_endpoints": [
            "delta_si_mean_r2",
            "delta_c1_deg",
        ],
        "secondary_endpoints": [
            "delta_boundary_d",
            "si_high_mean_r2",
        ],
        "model_list": models,
        "seed_list": [int(s) for s in seeds],
        "sample_size_plan": {
            "paired_seeds": int(len(seeds)),
            "models": int(len(models)),
        },
        "acceptance_criteria": [
            "bootstrap ci95(delta_si_mean_r2) > 0 indicates RoPE> NoPE under proxy conditions",
        ],
        "fallback_interpretation_if_null": "PE-family attribution remains open pending fullscale matched training.",
    }

    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": "E21",
        "analysis_tier": "confirmatory",
        "canonical_eligible": True,
        "override_used": False,
        "verdict": {
            "interpretation": interpretation,
            "n_paired_seeds": n,
            "si_delta_mean": si_ci["mean"],
            "si_delta_ci95": [si_ci["ci_lo"], si_ci["ci_hi"]],
        },
        "limitations": [
            "Proxy recipe is short-horizon and not equivalent to full pretraining-scale PE-family comparison.",
            "Head-group priors for NoPE may be auto-derived from synthetic fallback profiling when natural-profile caches are unavailable.",
        ],
    }

    data_dictionary = {
        "experiment_id": "E21",
        "tables": [
            {
                "path": "per_seed_metrics.parquet",
                "description": "Per-seed outcomes for RoPE and NoPE runs.",
                "columns": [
                    {"name": "model", "dtype": "str", "description": "model name"},
                    {"name": "pe_family", "dtype": "str", "description": "PE family label used for contrast pairing"},
                    {"name": "seed", "dtype": "int", "description": "training seed"},
                    {"name": "si_mean_r2", "dtype": "float", "description": "post-FT SI mean R2"},
                    {"name": "boundary_d", "dtype": "float", "description": "post-FT boundary effect size (Cohen's d)"},
                    {"name": "boundary_artifact_flag", "dtype": "bool", "description": "synthetic boundary artifact guard flag"},
                    {"name": "c1_overall_degradation_mean", "dtype": "float", "description": "post-FT mini-ablation mean degradation"},
                ],
            },
            {
                "path": "paired_seed_deltas.parquet",
                "description": "Paired seed deltas (RoPE - NoPE) for primary contrast metrics.",
                "columns": [
                    {"name": "seed", "dtype": "int", "description": "paired seed"},
                    {"name": "delta_si_mean_r2", "dtype": "float", "description": "SI mean R2 difference"},
                    {"name": "delta_c1_deg", "dtype": "float", "description": "C1 degradation difference"},
                    {"name": "delta_boundary_d", "dtype": "float", "description": "boundary effect size difference"},
                ],
            },
        ],
    }

    emit_core_artifacts(
        experiment_id="E21",
        out_dir=out_root,
        preregistration=prereg,
        manifest_extra={"models": models, "seeds": [int(s) for s in seeds], "mode": mode},
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )
    return cross


def main() -> None:
    p = argparse.ArgumentParser(description="E21: PE-family matched training contrast", allow_abbrev=False)
    p.add_argument("--mode", choices=["proxy", "fullscale"], default="proxy")
    p.add_argument("--models", default=",".join(R21_PROXY_MODELS))
    p.add_argument(
        "--device-map",
        default="tinyllama-1.1b:cuda:0,tinyllama-nope-1.1b:cuda:0",
    )
    p.add_argument("--seed-list", default="0,1,2")
    p.add_argument("--output-root", default=str(OUT_ROOT))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--finalize-only", action="store_true")
    p.add_argument("--no-finalize", action="store_true")
    p.add_argument("--ack-fullscale-execution", action="store_true")
    args = p.parse_args()

    if args.finalize_only and args.no_finalize:
        raise RuntimeError("[E21] hard_fail_reason: --finalize-only and --no-finalize are mutually exclusive")
    if args.mode == "fullscale" and (not args.ack_fullscale_execution) and (not args.finalize_only):
        raise RuntimeError(
            "[E21] hard_fail_reason: fullscale mode requires explicit --ack-fullscale-execution flag"
        )

    models = parse_models_arg(args.models, default=R21_PROXY_MODELS)
    device_map = parse_device_map(args.device_map)
    seeds = _parse_seed_list(args.seed_list)
    out_root = ensure_dir(Path(args.output_root))

    for m in models:
        if m not in R21_PROXY_MODELS:
            raise RuntimeError(
                f"[E21] hard_fail_reason: model '{m}' unsupported for NEW-R21; expected subset of {list(R21_PROXY_MODELS)}"
            )

    if args.finalize_only:
        cross = _finalize(models=models, seeds=seeds, out_root=out_root, mode=args.mode)
        print(f"[E21] Finalized from shards. interpretation={cross['interpretation']}", flush=True)
        return

    # Preflight: ensure tokenized profile cache and required SI metadata.
    train_cfg_for_preflight, _ = _build_train_cfg(args.mode, bool(args.smoke))
    n_math = int(train_cfg_for_preflight.math_per_task) * 5
    frac = float(train_cfg_for_preflight.control_fraction)
    n_control = max(1, int(round((frac / max(1e-8, 1.0 - frac)) * n_math)))

    preflight_rows = []
    for model_name in models:
        device = device_map.get(model_name, "cuda:0")
        spec = MODELS.get(model_name)
        if spec is None:
            raise RuntimeError(
                f"[E21] hard_fail_reason: model '{model_name}' missing from theory1 MODELS registry"
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
        status = _ensure_head_groups(model_name=model_name, device=device, seq_len=256, n_seq=16 if args.smoke else 24)
        row = {"model": model_name, "profile_cache": cache_status, **status}
        preflight_rows.append(row)
        print(f"[E21] preflight model={model_name}: {row}", flush=True)
    write_json(out_root / "headgroup_preflight.json", {"timestamp": timestamp_now(), "rows": preflight_rows})

    for model_name in models:
        device = device_map.get(model_name, "cuda:0")
        for seed in seeds:
            print(f"[E21] Running model={model_name} seed={seed} mode={args.mode} device={device}", flush=True)
            run_seed(
                model_name=model_name,
                seed=int(seed),
                mode=args.mode,
                device=device,
                out_root=out_root,
                smoke=bool(args.smoke),
            )

    if args.no_finalize:
        print("[E21] Shard run complete (no finalize).", flush=True)
        return

    cross = _finalize(models=models, seeds=seeds, out_root=out_root, mode=args.mode)
    print(f"[E21] Done. interpretation={cross['interpretation']}", flush=True)


if __name__ == "__main__":
    main()
