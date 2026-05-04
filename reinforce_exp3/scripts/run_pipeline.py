#!/usr/bin/env python3
"""reinforce_exp3 pipeline orchestrator.

Usage:
    python reinforce_exp3/scripts/run_pipeline.py --experiments E0,E2,E3 --device-map "llama-3.1-8b:cuda:0,olmo-2-7b:cuda:1,mistral-7b-v0.1:cuda:2"
    python reinforce_exp3/scripts/run_pipeline.py --experiments all --device-map "llama-3.1-8b:cuda:0"
    python reinforce_exp3/scripts/run_pipeline.py --experiments E9   # heatmaps only, no GPU needed

Tier priorities:
    must_have  : E0, E3, E6, E17, E18, E19, E22
    quick      : E2, E8, E9, E10
    long       : E4, E5, E7, E11, E12
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp3.common import (  # noqa: E402
    RESULTS_ROOT,
    ensure_dir,
    timestamp_now,
    write_json,
)
from reinforce_exp3.scripts._shared import parse_device_map  # noqa: E402

SCRIPTS_DIR = Path(__file__).resolve().parent
PYTHON = str(ROOT / ".venv" / "bin" / "python")

# Ordered by recommended execution priority (see TODO.md)
EXPERIMENT_ORDER = [
    "E9", "E3", "E0", "E6", "E2", "E10", "E8", "E4", "E5", "E7", "E11", "E12",
    "E17", "E18", "E19", "E22", "E23", "E24", "E24B", "E25", "E26", "E27", "E28",
    "E28A", "E28B", "E28D", "E28C",
    "E29A", "E29B", "E29C",
    "E30A", "E30B",
    "E20", "E21",
]

TIER_MAP: dict[str, str] = {
    "E0": "must_have",
    "E3": "must_have",
    "E6": "must_have",
    "E17": "must_have",
    "E18": "must_have",
    "E19": "must_have",
    "E22": "must_have",
    "E23": "must_have",
    "E24": "quick",
    "E24B": "quick",
    "E25": "quick",
    "E26": "quick",
    "E27": "quick",
    "E28": "quick",
    "E28A": "quick",
    "E28B": "quick",
    "E28C": "quick",
    "E28D": "quick",
    "E29A": "quick",
    "E29B": "long",
    "E29C": "quick",
    "E30A": "quick",
    "E30B": "quick",
    "E2": "quick",
    "E8": "quick",
    "E9": "quick",
    "E10": "quick",
    "E4": "long",
    "E5": "long",
    "E7": "long",
    "E11": "long",
    "E12": "long",
    "E20": "long",
    "E21": "long",
}

SCRIPT_MAP: dict[str, str] = {
    "E0": "run_e0_cluster_sequential.py",
    "E2": "run_e2_ordering_sensitivity.py",
    "E3": "run_e3_continuous_metrics.py",
    "E4": "run_e4_checkpoint_trajectory.py",
    "E5": "run_e5_fourth_model.py",
    "E6": "run_e6_llama_boundary.py",
    "E7": "run_e7_nonenglish_boundary.py",
    "E8": "run_e8_tokenizer_overlap.py",
    "E9": "run_e9_si_heatmaps.py",
    "E10": "run_e10_retrieval_overlap.py",
    "E11": "run_e11_rope_freq_decomp.py",
    "E12": "run_e12_icl_sensitivity.py",
    "E17": "run_e17_normmatched_specificity.py",
    "E18": "run_e18_icl_format_confound.py",
    "E19": "run_e19_si_score_robustness.py",
    "E22": "run_e22_task_conditional_specificity.py",
    "E23": "run_e23_stage_sensitive_si_pretraining.py",
    "E24": "run_e24_non_si_matched_baseline.py",
    "E24B": "run_e24b_si_vs_non_si_contrast.py",
    "E25": "run_e25_cross_model_comparability.py",
    "E26": "run_e26_r2_disruption_dose_response.py",
    "E27": "run_e27_absolute_threshold_robustness.py",
    "E28": "run_e28_probe_scope_battery.py",
    "E28A": "run_e28a_e26_20bin_exact.py",
    "E28B": "run_e28b_importance_matched_control.py",
    "E28C": "run_e28c_probe_diversity_transfer.py",
    "E28D": "run_e28d_r2_localbias_decomposition.py",
    "E29A": "run_e29a_kernel_transplant_specificity.py",
    "E29B": "run_e29b_naturaltext_longcontext_probe.py",
    "E29C": "run_e29c_qwen_anchor_quickcheck.py",
    "E30A": "run_e30a_probe_boundary_grid.py",
    "E30B": "run_e30b_localbias_null_family.py",
    "E20": "run_e20_heterogeneity_variance_decomp.py",
    "E21": "run_e21_pe_family_training_contrast.py",
}

OUTPUT_ROOT_MAP: dict[str, str] = {
    "E20": "E20_heterogeneity_variance_decomp",
    "E21": "E21_pe_family_training_contrast",
    "E23": "E23_stage_sensitive_si_pretraining",
    "E24": "E24_non_si_matched_baseline",
    "E24B": "E24b_si_vs_non_si_contrast",
    "E25": "E25_cross_model_comparability",
    "E26": "E26_r2_disruption_dose_response",
    "E27": "E27_absolute_threshold_robustness",
    "E28": "E28_probe_scope_battery",
    "E28A": "E28a_e26_20bin_exact",
    "E28B": "E28b_importance_matched_control",
    "E28C": "E28c_probe_diversity_transfer",
    "E28D": "E28d_r2_localbias_decomposition",
    "E29A": "E29a_kernel_transplant_specificity",
    "E29B": "E29b_naturaltext_longcontext_probe",
    "E29C": "E29c_qwen_anchor_quickcheck",
    "E30A": "E30a_probe_boundary_grid",
    "E30B": "E30b_localbias_null_family",
}

# Which experiments need a --models / --device-map passthrough
NEEDS_MODELS = {"E0", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11", "E12", "E17", "E18", "E19", "E22", "E24", "E24B", "E25", "E26", "E27", "E28", "E28A", "E28B", "E28C", "E28D", "E29A", "E29B", "E29C", "E30A", "E30B", "E20", "E21"}
NEEDS_DEVICE = {"E0", "E2", "E3", "E4", "E5", "E6", "E7", "E10", "E11", "E12", "E17", "E18", "E19", "E22", "E26", "E27", "E28", "E28A", "E28B", "E28C", "E29A", "E29B", "E29C", "E30A", "E20", "E21"}
EXPERIMENT_MODELS_OVERRIDE: dict[str, list[str]] = {
    "E20": ["llama-3.1-8b", "mistral-7b-v0.1", "olmo-2-7b"],
    "E21": ["tinyllama-1.1b", "tinyllama-nope-1.1b"],
    "E29C": ["qwen2.5-7b"],
}


def _run_experiment(
    exp_id: str,
    *,
    models: list[str],
    device_map: dict[str, str],
    extra_args: list[str],
) -> dict[str, Any]:
    script = SCRIPTS_DIR / SCRIPT_MAP[exp_id]
    cmd = [PYTHON, "-u", str(script)]
    run_models = EXPERIMENT_MODELS_OVERRIDE.get(exp_id, models)
    run_device_map = {m: device_map[m] for m in run_models if m in device_map}

    if exp_id in NEEDS_MODELS:
        cmd += ["--models", ",".join(run_models)]
    if exp_id in NEEDS_DEVICE:
        dm_str = ",".join(f"{m}:{d}" for m, d in run_device_map.items())
        cmd += ["--device-map", dm_str]

    out_dir = OUTPUT_ROOT_MAP.get(
        exp_id,
        f"E{exp_id[1:]}_{'_'.join(SCRIPT_MAP[exp_id].split('_')[2:-1])}",
    )
    cmd += ["--output-root", str(RESULTS_ROOT / out_dir)]
    cmd += extra_args

    print(f"\n[PIPELINE] Starting {exp_id} (tier={TIER_MAP[exp_id]})", flush=True)
    print(f"[PIPELINE] cmd: {' '.join(cmd)}", flush=True)
    t0 = time.time()

    result = subprocess.run(cmd, cwd=str(ROOT), check=False, text=True)
    elapsed = time.time() - t0

    status = "success" if result.returncode == 0 else f"failed(rc={result.returncode})"
    print(f"[PIPELINE] {exp_id} finished: {status} in {elapsed:.1f}s", flush=True)
    return {"experiment_id": exp_id, "status": status, "elapsed_sec": elapsed, "returncode": result.returncode}


def _resolve_experiments(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(EXPERIMENT_ORDER)
    tiers = {"must_have", "quick", "long"}
    raw_ids = [x.strip().upper() for x in raw.split(",") if x.strip()]
    resolved: list[str] = []
    for rid in raw_ids:
        if rid in SCRIPT_MAP:
            resolved.append(rid)
        elif rid.lower() in tiers:
            resolved.extend(
                [e for e in EXPERIMENT_ORDER if TIER_MAP.get(e) == rid.lower()]
            )
        else:
            print(f"[PIPELINE] WARNING: unknown experiment id '{rid}', skipping", flush=True)
    # Preserve order and deduplicate
    seen: set[str] = set()
    out: list[str] = []
    for e in resolved:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="reinforce_exp3 pipeline: launch one or more experiments",
        allow_abbrev=False,
    )
    p.add_argument(
        "--experiments",
        default="must_have",
        help=(
            "Comma-separated experiment IDs (E0,E3,E6) or tier names "
            "(must_have, quick, long) or 'all'. Default: must_have"
        ),
    )
    p.add_argument(
        "--models",
        default=",".join(["llama-3.1-8b", "olmo-2-7b", "mistral-7b-v0.1"]),
        help="Comma-separated model names",
    )
    p.add_argument(
        "--device-map",
        default="llama-3.1-8b:cuda:0,olmo-2-7b:cuda:0,mistral-7b-v0.1:cuda:0",
        help="Model-to-device mapping: 'llama-3.1-8b:cuda:0,olmo-2-7b:cuda:1,...'",
    )
    p.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Halt pipeline if any experiment returns non-zero exit code",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    args = p.parse_args()

    experiments = _resolve_experiments(args.experiments)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    device_map = parse_device_map(args.device_map)

    print(f"[PIPELINE] reinforce_exp3 started at {timestamp_now()}", flush=True)
    print(f"[PIPELINE] experiments={experiments}", flush=True)
    print(f"[PIPELINE] models={models}", flush=True)
    print(f"[PIPELINE] device_map={device_map}", flush=True)

    ensure_dir(RESULTS_ROOT)
    run_log: list[dict[str, Any]] = []

    for exp_id in experiments:
        if args.dry_run:
            script = SCRIPTS_DIR / SCRIPT_MAP[exp_id]
            print(f"[DRY-RUN] Would run: {exp_id} via {script}", flush=True)
            continue
        rec = _run_experiment(exp_id, models=models, device_map=device_map, extra_args=[])
        run_log.append(rec)
        if args.stop_on_failure and rec["returncode"] != 0:
            print(f"[PIPELINE] Halting: {exp_id} failed with rc={rec['returncode']}", flush=True)
            break

    if not args.dry_run:
        summary_path = RESULTS_ROOT / "pipeline_run_log.json"
        write_json(summary_path, {
            "timestamp": timestamp_now(),
            "experiments_run": experiments,
            "results": run_log,
        })
        n_ok = sum(1 for r in run_log if r["status"] == "success")
        print(f"\n[PIPELINE] Done: {n_ok}/{len(run_log)} succeeded. Log: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
