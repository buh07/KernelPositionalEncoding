from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str in sys.path:
    sys.path.remove(project_root_str)
sys.path.insert(0, project_root_str)

from experiment1.config import DEFAULT_MODEL_PROFILE, SUPPORTED_MODEL_PROFILES
from experiment2.config import model_names_for_profile, model_sets_for_profile
from experiment2.execution import ExecutionConfig, reanalyze_run, run_manifest
from experiment2.head_groups import freeze_phase2d_head_groups
from experiment2.long_offsets import LOCK_PATH, load_long_offset_bundle
from experiment2.posthoc_centered import (
    PosthocCenteredConfig,
    PosthocQueueWorkerConfig,
    run_posthoc_centered,
    run_posthoc_queue_worker,
)

LEGACY_PHASE2A_MODEL = "llama-3.2-1b"
SCALEUP_PHASE2A_MODEL = "llama-3.1-8b"
_DEFAULT_MODEL_SETS = model_sets_for_profile(DEFAULT_MODEL_PROFILE)
_DEFAULT_ROPE_MODELS = tuple(_DEFAULT_MODEL_SETS["rope"])
_DEFAULT_ABSPE_MODELS = tuple(_DEFAULT_MODEL_SETS["abspe"])
_DEFAULT_NOPE_MODELS = tuple(_DEFAULT_MODEL_SETS["nope"])
_DEFAULT_PHASE2A_MODEL = LEGACY_PHASE2A_MODEL if LEGACY_PHASE2A_MODEL in _DEFAULT_ROPE_MODELS else (
    _DEFAULT_ROPE_MODELS[0] if _DEFAULT_ROPE_MODELS else LEGACY_PHASE2A_MODEL
)
_DEFAULT_PHASE2D_MODELS = tuple(m for m in ("olmo-1b", "llama-3.2-1b") if m in _DEFAULT_ROPE_MODELS) or tuple(
    _DEFAULT_ROPE_MODELS[:2]
)

SYNTHETIC_TASKS = ("local_copy_offset", "local_key_match", "delayed_copy", "long_range_retrieval")
INTERVENTIONS_7 = (
    "none",
    "ablate_high_medium",
    "ablate_high_strong",
    "ablate_low_medium",
    "ablate_low_strong",
    "random_medium",
    "random_strong",
)
INTERVENTIONS_STRONG = ("none", "ablate_high_strong", "ablate_low_strong", "random_strong")
INTERVENTION_PROFILES = ("full", "strong_only")
TIER1_DATASETS = ("wiki40b_en_pre2019", "codesearchnet_python_snapshot")
DEFAULT_RANDOM_DRAWS_CONFIRMATORY = 3
DEFAULT_TIER1_SEEDS_PHASE2B = 7
DEFAULT_FEASIBILITY_LONG_OFFSETS = (8, 12, 16, 24, 32, 48, 64, 96, 128)
DEFAULT_LOCK_CANDIDATE_OFFSETS = (8, 12, 16, 24, 32, 48, 64, 96, 128)
DEFAULT_LONG_TASK_FEASIBILITY_POLICY = "strict_two_task"
LONG_TASK_FEASIBILITY_POLICIES = ("strict_two_task", "retrieval_fallback")


@dataclass(frozen=True)
class RunCell:
    phase: str
    split: str
    model: str
    task: str
    seq_len: int
    intervention: str
    seed: int
    dataset: str | None = None
    scope: str | None = None
    head_group: str | None = None
    span: int | None = None
    random_draw: int | None = None
    notes: str | None = None
    device: str | None = None
    run_id: str | None = None


@dataclass(frozen=True)
class BuildOptions:
    random_draws_confirmatory: int = DEFAULT_RANDOM_DRAWS_CONFIRMATORY
    tier1_seeds_phase2b: int = DEFAULT_TIER1_SEEDS_PHASE2B
    phase2a_baseline_only: bool = False
    model_profile: str = DEFAULT_MODEL_PROFILE
    model_allowlist: tuple[str, ...] | None = None
    rope_models: tuple[str, ...] = _DEFAULT_ROPE_MODELS
    abspe_models: tuple[str, ...] = _DEFAULT_ABSPE_MODELS
    nope_models: tuple[str, ...] = _DEFAULT_NOPE_MODELS
    phase2a_model: str = _DEFAULT_PHASE2A_MODEL
    phase2d_models: tuple[str, ...] = _DEFAULT_PHASE2D_MODELS
    intervention_profile: str = "full"
    phase2b_core_synthetic_only: bool = False


def _phase2a_anchor_model_for_profile(model_profile: str, rope_models: tuple[str, ...]) -> str:
    preferred = LEGACY_PHASE2A_MODEL if model_profile == "legacy_1b" else SCALEUP_PHASE2A_MODEL
    if preferred in rope_models:
        return preferred
    if rope_models:
        return rope_models[0]
    raise RuntimeError(f"Model profile '{model_profile}' has no RoPE models for Phase 2A.")


def _phase2d_models_for_profile(model_profile: str, rope_models: tuple[str, ...]) -> tuple[str, ...]:
    if model_profile == "legacy_1b":
        desired = tuple(m for m in ("olmo-1b", "llama-3.2-1b") if m in rope_models)
        if desired:
            return desired
    return tuple(rope_models[:2] if len(rope_models) >= 2 else rope_models)


def _resolve_build_options(
    *,
    model_profile: str,
    random_draws_confirmatory: int,
    tier1_seeds_phase2b: int,
    phase2a_baseline_only: bool,
    intervention_profile: str,
    phase2b_core_synthetic_only: bool,
    model_allowlist: tuple[str, ...] | None = None,
) -> BuildOptions:
    sets = model_sets_for_profile(model_profile)
    rope_models = tuple(sets["rope"])
    abspe_models = tuple(sets["abspe"])
    nope_models = tuple(sets["nope"])
    if model_allowlist:
        allow = set(str(name) for name in model_allowlist)
        rope_models = tuple(name for name in rope_models if name in allow)
        abspe_models = tuple(name for name in abspe_models if name in allow)
        nope_models = tuple(name for name in nope_models if name in allow)
    if not (rope_models or abspe_models or nope_models):
        raise RuntimeError(
            f"Model allowlist removed all models for profile='{model_profile}'. "
            f"allowlist={list(model_allowlist or [])}"
        )
    if str(intervention_profile) not in INTERVENTION_PROFILES:
        raise RuntimeError(
            f"Unsupported intervention profile '{intervention_profile}'. "
            f"Supported: {INTERVENTION_PROFILES}"
        )
    return BuildOptions(
        random_draws_confirmatory=max(1, int(random_draws_confirmatory)),
        tier1_seeds_phase2b=max(1, int(tier1_seeds_phase2b)),
        phase2a_baseline_only=bool(phase2a_baseline_only),
        model_profile=model_profile,
        model_allowlist=(tuple(model_allowlist) if model_allowlist else None),
        rope_models=rope_models,
        abspe_models=abspe_models,
        nope_models=nope_models,
        phase2a_model=_phase2a_anchor_model_for_profile(model_profile, rope_models),
        phase2d_models=_phase2d_models_for_profile(model_profile, rope_models),
        intervention_profile=str(intervention_profile),
        phase2b_core_synthetic_only=bool(phase2b_core_synthetic_only),
    )


def _default_build_options() -> BuildOptions:
    return _resolve_build_options(
        model_profile=DEFAULT_MODEL_PROFILE,
        random_draws_confirmatory=DEFAULT_RANDOM_DRAWS_CONFIRMATORY,
        tier1_seeds_phase2b=DEFAULT_TIER1_SEEDS_PHASE2B,
        phase2a_baseline_only=False,
        intervention_profile="full",
        phase2b_core_synthetic_only=False,
        model_allowlist=None,
    )


def _normalize_model_allowlist(raw: str, *, model_profile: str) -> tuple[str, ...] | None:
    tokens: list[str] = []
    seen: set[str] = set()
    for part in str(raw or "").split(","):
        token = part.strip()
        if not token:
            continue
        if token in seen:
            continue
        tokens.append(token)
        seen.add(token)
    if not tokens:
        return None
    valid = set(model_names_for_profile(model_profile))
    invalid = [name for name in tokens if name not in valid]
    if invalid:
        raise SystemExit(
            f"--model-allowlist has names not in profile '{model_profile}': {invalid}. "
            f"Valid={sorted(valid)}"
        )
    return tuple(tokens)


def _parse_bool_text(value: str, *, flag_name: str) -> bool:
    text = str(value).strip().lower()
    if text not in {"true", "false"}:
        raise SystemExit(f"{flag_name} must be one of: true,false (got '{value}')")
    return text == "true"


def _effective_long_task_feasibility_policy(*, model_profile: str, requested_policy: str) -> str:
    policy = str(requested_policy).strip() or DEFAULT_LONG_TASK_FEASIBILITY_POLICY
    if policy not in LONG_TASK_FEASIBILITY_POLICIES:
        raise RuntimeError(
            f"Unsupported long-task feasibility policy '{policy}'. "
            f"Supported: {LONG_TASK_FEASIBILITY_POLICIES}"
        )
    # Retrieval fallback amendment is scale-up only; keep legacy semantics frozen.
    if policy == "retrieval_fallback" and str(model_profile) != "scaleup_78b":
        return "strict_two_task"
    return policy


def _confirmatory_long_tasks_for_profile(model_profile: str) -> tuple[str, ...]:
    default_long_tasks = ("delayed_copy", "long_range_retrieval")
    if str(model_profile) != "scaleup_78b":
        return default_long_tasks
    bundle = load_long_offset_bundle(strict=False)
    active = tuple(task for task in bundle.active_long_tasks if task in default_long_tasks)
    return active or default_long_tasks


def _parse_int_tuple(raw: str, *, flag_name: str) -> tuple[int, ...]:
    values: list[int] = []
    seen: set[int] = set()
    for part in str(raw).split(","):
        token = part.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise SystemExit(f"{flag_name} must be a comma-separated list of integers; got '{token}'.") from exc
        if value <= 0:
            raise SystemExit(f"{flag_name} requires positive integers; got {value}.")
        if value not in seen:
            values.append(value)
            seen.add(value)
    if not values:
        raise SystemExit(f"{flag_name} produced an empty list.")
    return tuple(values)


def _expand_interventions(interventions: Iterable[str], *, random_draws: int) -> list[tuple[str, int | None]]:
    out: list[tuple[str, int | None]] = []
    for intervention in interventions:
        if intervention.startswith("random_"):
            for draw_idx in range(max(1, int(random_draws))):
                out.append((intervention, draw_idx))
        else:
            out.append((intervention, None))
    return out


def resolve_device(device_arg: str, allow_cpu_fallback: bool) -> str:
    if device_arg != "auto":
        if device_arg.startswith("cuda") and not torch.cuda.is_available() and not allow_cpu_fallback:
            raise RuntimeError(f"Requested device '{device_arg}', but CUDA is unavailable.")
        if device_arg.startswith("cuda") and not torch.cuda.is_available() and allow_cpu_fallback:
            return "cpu"
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if allow_cpu_fallback:
        return "cpu"
    raise RuntimeError("Device set to auto, but CUDA is unavailable and CPU fallback is disabled.")


def build_phase_2a(opts: BuildOptions | None = None) -> list[RunCell]:
    opts = opts or _default_build_options()
    rows: list[RunCell] = []
    model = str(opts.phase2a_model)
    long_tasks = _confirmatory_long_tasks_for_profile(opts.model_profile)
    tasks = ("local_key_match",) + tuple(long_tasks)
    interventions = ("none",) if bool(opts.phase2a_baseline_only) else INTERVENTIONS_7
    expanded = _expand_interventions(interventions, random_draws=opts.random_draws_confirmatory)
    for task in tasks:
        for intervention, random_draw in expanded:
            for seed in range(7):
                rows.append(
                    RunCell(
                        phase="phase2a",
                        split="synthetic",
                        model=model,
                        task=task,
                        seq_len=1024,
                        intervention=intervention,
                        random_draw=random_draw,
                        seed=seed,
                    )
                )
    return rows


def build_phase_2a_long_pilot(opts: BuildOptions | None = None) -> list[RunCell]:
    opts = opts or _default_build_options()
    rows: list[RunCell] = []
    tasks = ("delayed_copy", "long_range_retrieval")
    for model in opts.rope_models:
        for task in tasks:
            for seed in range(7):
                rows.append(
                    RunCell(
                        phase="phase2a_long_pilot",
                        split="synthetic",
                        model=model,
                        task=task,
                        seq_len=1024,
                        intervention="none",
                        random_draw=None,
                        seed=seed,
                        notes="phase2a_long_offset_calibration_pilot",
                    )
                )
    return rows


def build_phase_2b(opts: BuildOptions | None = None) -> list[RunCell]:
    opts = opts or _default_build_options()
    rows: list[RunCell] = []
    intervention_set = INTERVENTIONS_7 if str(opts.intervention_profile) == "full" else INTERVENTIONS_STRONG
    expanded_all = _expand_interventions(intervention_set, random_draws=opts.random_draws_confirmatory)
    expanded_strong = _expand_interventions(INTERVENTIONS_STRONG, random_draws=opts.random_draws_confirmatory)
    synthetic_tasks = ("local_copy_offset", "local_key_match") + _confirmatory_long_tasks_for_profile(opts.model_profile)

    # Synthetic confirmatory core.
    for model in opts.rope_models:
        for task in synthetic_tasks:
            for intervention, random_draw in expanded_all:
                for seed in range(7):
                    rows.append(
                        RunCell(
                            phase="phase2b",
                            split="synthetic",
                            model=model,
                            task=task,
                            seq_len=1024,
                            intervention=intervention,
                            random_draw=random_draw,
                            seed=seed,
                        )
                    )
    if bool(opts.phase2b_core_synthetic_only):
        return rows

    # Tier-1 transfer panel.
    for model in opts.rope_models:
        for dataset in TIER1_DATASETS:
            for intervention, random_draw in expanded_strong:
                for seed in range(opts.tier1_seeds_phase2b):
                    rows.append(
                        RunCell(
                            phase="phase2b",
                            split="tier1_ppl",
                            model=model,
                            task="tier1_stratified_ppl",
                            seq_len=1024,
                            intervention=intervention,
                            random_draw=random_draw,
                            seed=seed,
                            dataset=dataset,
                        )
                    )

    # Confirmatory medium-range transition mini-panel.
    sentinel_tasks = ("copy_offset_bridge", "retrieval_bridge")
    sentinel_spans = (32, 64, 96)
    for model in opts.rope_models:
        for task in sentinel_tasks:
            for span in sentinel_spans:
                for intervention, random_draw in expanded_strong:
                    for seed in range(7):
                        rows.append(
                            RunCell(
                                phase="phase2b",
                                split="span_bridge",
                                model=model,
                                task=task,
                                seq_len=1024,
                                intervention=intervention,
                                random_draw=random_draw,
                                seed=seed,
                                span=span,
                                notes="phase2b_confirmatory_medium_range_transition_panel",
                            )
                        )
    return rows


def build_phase_2c(opts: BuildOptions | None = None) -> list[RunCell]:
    opts = opts or _default_build_options()
    rows: list[RunCell] = []
    # Base matrix
    for model in opts.rope_models:
        for task in SYNTHETIC_TASKS:
            for intervention in INTERVENTIONS_7:
                for seed in range(5):
                    rows.append(
                        RunCell(
                            phase="phase2c",
                            split="synthetic",
                            model=model,
                            task=task,
                            seq_len=256,
                            intervention=intervention,
                            seed=seed,
                        )
                    )
    for model in opts.abspe_models:
        for task in SYNTHETIC_TASKS:
            for intervention in INTERVENTIONS_7:
                for seed in range(5):
                    rows.append(
                        RunCell(
                            phase="phase2c",
                            split="synthetic",
                            model=model,
                            task=task,
                            seq_len=1024,
                            intervention=intervention,
                            seed=seed,
                            notes="abspe_dct_analog",
                        )
                    )
    for model in opts.nope_models:
        for task in SYNTHETIC_TASKS:
            for seed in range(5):
                rows.append(
                    RunCell(
                        phase="phase2c",
                        split="synthetic",
                        model=model,
                        task=task,
                        seq_len=1024,
                        intervention="none",
                            seed=seed,
                        )
                    )
    if bool(opts.phase2b_core_synthetic_only):
        return rows
    for model in opts.rope_models:
        for dataset in TIER1_DATASETS:
            for intervention in INTERVENTIONS_STRONG:
                rows.append(
                    RunCell(
                        phase="phase2c",
                        split="tier1_ppl",
                        model=model,
                        task="tier1_stratified_ppl",
                        seq_len=1024,
                        intervention=intervention,
                        seed=0,
                        dataset=dataset,
                    )
                )
    for model in opts.abspe_models:
        for dataset in TIER1_DATASETS:
            for intervention in INTERVENTIONS_STRONG:
                rows.append(
                    RunCell(
                        phase="phase2c",
                        split="tier1_ppl",
                        model=model,
                        task="tier1_stratified_ppl",
                        seq_len=1024,
                        intervention=intervention,
                        seed=0,
                        dataset=dataset,
                        notes="abspe_dct_analog",
                    )
                )
    for model in opts.nope_models:
        for dataset in TIER1_DATASETS:
            rows.append(
                RunCell(
                    phase="phase2c",
                    split="tier1_ppl",
                    model=model,
                    task="tier1_stratified_ppl",
                    seq_len=1024,
                    intervention="none",
                    seed=0,
                    dataset=dataset,
                )
            )

    # Medium-range bridge
    bridge_tasks = ("copy_offset_bridge", "retrieval_bridge")
    bridge_spans = (32, 64, 96)
    for model in opts.rope_models:
        for task in bridge_tasks:
            for span in bridge_spans:
                for intervention in INTERVENTIONS_STRONG:
                    for seed in range(5):
                        rows.append(
                            RunCell(
                                phase="phase2c",
                                split="span_bridge",
                                model=model,
                                task=task,
                                seq_len=1024,
                                intervention=intervention,
                                seed=seed,
                                span=span,
                            )
                        )
    return rows


def build_phase_2d(opts: BuildOptions | None = None) -> list[RunCell]:
    opts = opts or _default_build_options()
    rows: list[RunCell] = []
    models = tuple(opts.phase2d_models)
    scopes = ("early-only", "deep-only")
    head_groups = ("high-kernel", "low-kernel")
    interventions = (
        "ablate_high_medium",
        "ablate_high_strong",
        "ablate_low_medium",
        "ablate_low_strong",
        "random_medium",
        "random_strong",
    )
    for model in models:
        for task in SYNTHETIC_TASKS:
            for scope in scopes:
                for head_group in head_groups:
                    for intervention in interventions:
                        for seed in range(5):
                            rows.append(
                                RunCell(
                                    phase="phase2d",
                                    split="mechanistic",
                                    model=model,
                                    task=task,
                                    seq_len=1024,
                                    intervention=intervention,
                                    seed=seed,
                                    scope=scope,
                                    head_group=head_group,
                                )
                            )
    return rows


PHASE_BUILDERS = {
    "phase2a": build_phase_2a,
    "phase2a_long_pilot": build_phase_2a_long_pilot,
    "phase2b": build_phase_2b,
    "phase2c": build_phase_2c,
    "phase2d": build_phase_2d,
}
DEFAULT_BUILD_PHASES = ("phase2a", "phase2b", "phase2c", "phase2d")


def build_cells(phases: Iterable[str], opts: BuildOptions) -> list[RunCell]:
    rows: list[RunCell] = []
    for phase in phases:
        rows.extend(PHASE_BUILDERS[phase](opts))
    return rows


def summarize(rows: list[RunCell]) -> dict[str, object]:
    per_phase: dict[str, int] = {}
    per_split: dict[str, int] = {}
    for row in rows:
        per_phase[row.phase] = per_phase.get(row.phase, 0) + 1
        per_split[row.split] = per_split.get(row.split, 0) + 1
    return {
        "total_cells": len(rows),
        "per_phase": dict(sorted(per_phase.items())),
        "per_split": dict(sorted(per_split.items())),
    }


def _require_confirmatory_long_offset_lock(*, phases: list[str]) -> None:
    if not any(p in {"phase2a", "phase2b"} for p in phases):
        return
    load_long_offset_bundle(strict=True)


def _build_feasibility_rows(*, run_id: str, device: str, offsets: Iterable[int]) -> list[RunCell]:
    return _build_feasibility_rows_for_models(
        run_id=run_id,
        device=device,
        offsets=offsets,
        rope_models=_DEFAULT_ROPE_MODELS,
    )


def _build_feasibility_rows_for_models(
    *,
    run_id: str,
    device: str,
    offsets: Iterable[int],
    rope_models: tuple[str, ...],
) -> list[RunCell]:
    rows: list[RunCell] = []
    offset_values = tuple(int(x) for x in offsets)
    for model in rope_models:
        for task in ("delayed_copy", "long_range_retrieval"):
            for span in offset_values:
                for seed in range(3):
                    rows.append(
                        RunCell(
                            phase="feasibility",
                            split="synthetic",
                            model=model,
                            task=task,
                            seq_len=1024,
                            intervention="none",
                            seed=seed,
                            span=span,
                            notes=f"long_offset_feasibility_baseline_only|offsets={','.join(str(x) for x in offset_values)}",
                            device=device,
                            run_id=run_id,
                        )
                    )
    return rows


def _build_feasibility_manifest(
    *,
    output_root: Path,
    run_id: str,
    device: str,
    model_profile: str,
    feasibility_offsets: tuple[int, ...],
    model_allowlist: tuple[str, ...] | None = None,
) -> tuple[list[RunCell], Path]:
    model_sets = model_sets_for_profile(model_profile)
    rope_models = tuple(model_sets["rope"])
    if model_allowlist:
        allow = set(str(name) for name in model_allowlist)
        rope_models = tuple(name for name in rope_models if name in allow)
    if not rope_models:
        raise RuntimeError(
            f"Feasibility manifest has no RoPE models after allowlist filter. "
            f"profile={model_profile} allowlist={list(model_allowlist or [])}"
        )
    rows = _build_feasibility_rows_for_models(
        run_id=run_id,
        device=device,
        offsets=feasibility_offsets,
        rope_models=rope_models,
    )
    output_paths = write_manifest(rows, output_root=output_root, run_id=run_id, device=device)
    manifest_path = output_paths["feasibility_manifest"]
    return rows, manifest_path


def _finalize_feasibility_outputs(
    *,
    output_root: Path,
    run_id: str,
    threshold: float,
    feasibility_offsets: tuple[int, ...],
    lock_candidate_offsets: tuple[int, ...],
    long_task_feasibility_policy: str,
    model_profile: str,
    model_allowlist: tuple[str, ...] | None = None,
) -> dict[str, object]:
    # Ensure aggregate artifacts are rebuilt from whatever shard-execute rows currently exist.
    reanalyze_result = reanalyze_run(output_root=output_root, run_id=run_id, phases=["feasibility"])
    phase_root = output_root / "feasibility" / run_id
    task_path = phase_root / "aggregate_task_metrics.parquet"
    if not task_path.exists():
        raise RuntimeError(
            "Feasibility finalize requires aggregate task metrics. "
            f"Missing: {task_path}. Run feasibility execute first."
        )
    task_df = pd.read_parquet(task_path)
    recommendation = _compute_long_offset_lock_recommendation(
        task_df,
        threshold=float(threshold),
        feasibility_offsets=feasibility_offsets,
        candidate_order=lock_candidate_offsets,
        long_task_feasibility_policy=long_task_feasibility_policy,
        model_profile=model_profile,
        model_allowlist=model_allowlist,
    )
    recommendation_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_run_id": run_id,
        **recommendation,
    }
    (phase_root / "lock_recommendation_preview.json").write_text(
        json.dumps(recommendation_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(recommendation["support_table_rows"]).to_parquet(
        phase_root / "offset_pass_rates.parquet",
        engine="pyarrow",
        index=False,
    )
    manifest_path = phase_root / "manifest.jsonl"
    manifest_rows = 0
    if manifest_path.exists():
        manifest_rows = sum(1 for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {
        "run_id": run_id,
        "mode": "feasibility-finalize",
        "reanalyze_result": reanalyze_result,
        "manifest_rows": int(manifest_rows),
        "recommendation_preview": {
            "status": recommendation_payload["status"],
            "selected_long_offsets": recommendation_payload["selected_long_offsets"],
            "fallback_spans": recommendation_payload["fallback_spans"],
            "feasibility_offsets": recommendation_payload["feasibility_offsets"],
            "candidate_order": recommendation_payload["candidate_order"],
            "active_long_tasks": recommendation_payload.get("active_long_tasks", []),
            "long_task_policy_applied": recommendation_payload.get("long_task_policy_applied", "strict_two_task"),
            "lock_resolution_mode": recommendation_payload.get("lock_resolution_mode", "strict_two_task"),
            "effective_models": recommendation_payload.get("effective_models", []),
            "model_allowlist": recommendation_payload.get("model_allowlist", []),
        },
        "outputs": {
            "manifest": str(manifest_path),
            "aggregate_task_metrics": str(task_path),
            "offset_pass_rates": str(phase_root / "offset_pass_rates.parquet"),
            "lock_recommendation_preview": str(phase_root / "lock_recommendation_preview.json"),
        },
    }


def _compute_long_offset_lock_recommendation(
    task_df: pd.DataFrame,
    *,
    threshold: float,
    feasibility_offsets: Iterable[int],
    candidate_order: Iterable[int],
    long_task_feasibility_policy: str = DEFAULT_LONG_TASK_FEASIBILITY_POLICY,
    model_profile: str = DEFAULT_MODEL_PROFILE,
    min_pass_rate: float = 2.0 / 3.0,
    model_allowlist: tuple[str, ...] | None = None,
) -> dict[str, object]:
    df = task_df.copy()
    needed = {"model", "task", "span", "seed", "intervention", "mean_accuracy"}
    if not needed.issubset(set(df.columns)):
        missing = sorted(needed - set(df.columns))
        raise RuntimeError(f"Feasibility task metrics missing required columns: {missing}")
    feasibility_values = tuple(int(x) for x in feasibility_offsets)
    candidate_values = tuple(int(x) for x in candidate_order)
    if not feasibility_values:
        raise RuntimeError("feasibility_offsets cannot be empty.")
    if not candidate_values:
        raise RuntimeError("candidate_order cannot be empty.")
    if any(span not in feasibility_values for span in candidate_values):
        raise RuntimeError(
            "lock-candidate offsets must be a subset of feasibility offsets. "
            f"feasibility_offsets={feasibility_values} candidate_order={candidate_values}"
        )

    df = df[
        (df["intervention"] == "none")
        & (df["task"].isin(["delayed_copy", "long_range_retrieval"]))
        & (df["span"].isin(feasibility_values))
    ].copy()
    model_allowlist_values = tuple(str(x) for x in (model_allowlist or tuple()))
    if model_allowlist_values:
        df = df[df["model"].isin(model_allowlist_values)].copy()
        if df.empty:
            raise RuntimeError(
                "No feasibility rows remain after model-allowlist filtering. "
                f"allowlist={model_allowlist_values}"
            )
    if df.empty:
        raise RuntimeError(
            "No feasibility rows found for delayed_copy/long_range_retrieval at configured offsets. "
            f"Offsets={feasibility_values}"
        )
    df["pass_floor"] = (df["mean_accuracy"].astype(float) >= float(threshold)).astype(int)
    grouped = (
        df.groupby(["span", "model", "task"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            pass_rate=("pass_floor", "mean"),
            mean_accuracy=("mean_accuracy", "mean"),
            min_accuracy=("mean_accuracy", "min"),
            max_accuracy=("mean_accuracy", "max"),
        )
        .sort_values(["span", "model", "task"])
        .reset_index(drop=True)
    )
    effective_models = sorted(str(m) for m in grouped["model"].dropna().unique().tolist())

    observed_spans = sorted(int(s) for s in grouped["span"].unique())
    missing_candidates = [int(s) for s in candidate_values if int(s) not in set(observed_spans)]
    if missing_candidates:
        raise RuntimeError(
            "Feasibility results are missing candidate offsets required for lock computation: "
            f"{missing_candidates}. Observed spans={observed_spans}"
        )

    applied_policy = _effective_long_task_feasibility_policy(
        model_profile=model_profile,
        requested_policy=long_task_feasibility_policy,
    )
    required_tasks = ("delayed_copy", "long_range_retrieval")
    strict_feasible_by_offset: dict[int, bool] = {}
    retrieval_feasible_by_offset: dict[int, bool] = {}
    task_feasible_by_span: dict[int, dict[str, bool]] = {}
    task_model_support_rows: list[dict[str, object]] = []
    for span in sorted(int(s) for s in feasibility_values):
        span_task_map: dict[str, bool] = {}
        for task_name in required_tasks:
            sub = grouped[(grouped["span"] == span) & (grouped["task"] == task_name)].copy()
            if sub.empty:
                model_support = 0.0
                model_pass_count = 0
                models_observed = 0
                task_feasible = False
            else:
                model_pass = (sub["pass_rate"].astype(float) >= float(min_pass_rate)).astype(int)
                model_support = float(model_pass.mean())
                model_pass_count = int(model_pass.sum())
                models_observed = int(sub["model"].nunique())
                task_feasible = bool(model_support >= float(min_pass_rate))
            span_task_map[task_name] = bool(task_feasible)
            task_model_support_rows.append(
                {
                    "span": int(span),
                    "task": str(task_name),
                    "models_observed": int(models_observed),
                    "models_passing": int(model_pass_count),
                    "model_support_rate": float(model_support),
                    "task_feasible": bool(task_feasible),
                    "rule": "task_feasible_if_model_support_rate_ge_min_pass_rate",
                    "min_pass_rate": float(min_pass_rate),
                }
            )
        task_feasible_by_span[span] = span_task_map
        strict_feasible_by_offset[span] = bool(all(span_task_map.get(task_name, False) for task_name in required_tasks))
        retrieval_feasible_by_offset[span] = bool(span_task_map.get("long_range_retrieval", False))

    strict_selected: list[int] = []
    for span in candidate_values:
        if strict_feasible_by_offset.get(span, False):
            strict_selected.append(span)
        else:
            break

    retrieval_selected: list[int] = []
    for span in candidate_values:
        if retrieval_feasible_by_offset.get(span, False):
            retrieval_selected.append(span)
        else:
            break

    selected: list[int]
    active_long_tasks: list[str]
    resolution_mode: str
    if applied_policy == "retrieval_fallback":
        if len(strict_selected) >= 2:
            selected = list(strict_selected)
            active_long_tasks = list(required_tasks)
            resolution_mode = "strict_two_task"
        elif len(retrieval_selected) >= 2:
            selected = list(retrieval_selected)
            active_long_tasks = ["long_range_retrieval"]
            resolution_mode = "retrieval_only_fallback"
        else:
            selected = list(strict_selected if strict_selected else retrieval_selected)
            active_long_tasks = list(required_tasks) if strict_selected else ["long_range_retrieval"]
            resolution_mode = "insufficient_feasible_offsets"
    else:
        selected = list(strict_selected)
        active_long_tasks = list(required_tasks)
        resolution_mode = "strict_two_task"

    status = "ok" if len(selected) >= 2 else "fail_insufficient_feasible_offsets"
    fallback = selected[:1] if len(selected) >= 1 else []
    return {
        "status": status,
        "threshold": float(threshold),
        "min_pass_rate": float(min_pass_rate),
        "lock_policy": "per_task_model_majority",
        "long_task_policy_requested": str(long_task_feasibility_policy),
        "long_task_policy_applied": str(applied_policy),
        "model_profile": str(model_profile),
        "model_allowlist": [str(x) for x in model_allowlist_values],
        "effective_models": effective_models,
        "lock_resolution_mode": resolution_mode,
        "active_long_tasks": list(active_long_tasks),
        "task_feasible_rule": "offset_feasible_if_each_long_task_has_models_passing_fraction_ge_min_pass_rate",
        "feasibility_offsets": [int(x) for x in feasibility_values],
        "candidate_order": [int(x) for x in candidate_values],
        "feasible_by_offset": {str(k): bool(v) for k, v in strict_feasible_by_offset.items()},
        "strict_feasible_by_offset": {str(k): bool(v) for k, v in strict_feasible_by_offset.items()},
        "retrieval_only_feasible_by_offset": {str(k): bool(v) for k, v in retrieval_feasible_by_offset.items()},
        "task_feasible_by_offset": {
            str(k): {str(task): bool(v) for task, v in task_map.items()} for k, task_map in task_feasible_by_span.items()
        },
        "strict_selected_long_offsets": [int(x) for x in strict_selected],
        "fallback_selected_long_offsets": [int(x) for x in retrieval_selected],
        "selected_long_offsets": [int(x) for x in selected],
        "fallback_spans": [int(x) for x in fallback],
        "support_table_rows": grouped.to_dict(orient="records"),
        "task_model_support_rows": task_model_support_rows,
    }


def _write_long_offset_lock(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    LOCK_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _phase2a_floor_calibration_accuracy_from_aggregate(task_path: Path) -> pd.Series:
    task_df = pd.read_parquet(task_path)
    none_rows = task_df[
        (task_df["split"].isin(["synthetic", "span_bridge", "mechanistic"]))
        & (task_df["intervention"] == "none")
    ].copy()
    if none_rows.empty:
        raise RuntimeError("Phase 2A floor calibration found no none-intervention synthetic rows.")
    if "eval_mode" in none_rows.columns:
        restricted_rows = none_rows[none_rows["eval_mode"] == "restricted"]
        if not restricted_rows.empty:
            none_rows = restricted_rows
    return none_rows["mean_accuracy"].astype(float)


def _phase2a_floor_calibration_accuracy_from_floor_decisions(phase_root: Path) -> pd.Series:
    floor_dir = phase_root / "floor_decisions"
    files = sorted(floor_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"Missing Phase 2A floor decision files under: {floor_dir}")

    accuracies: list[float] = []
    for fp in files:
        payload = json.loads(fp.read_text(encoding="utf-8"))
        key = str(payload.get("key", ""))
        parts = key.split("|")
        split = parts[1] if len(parts) > 1 else ""
        if split not in {"synthetic", "span_bridge", "mechanistic"}:
            continue
        baseline = payload.get("baseline_accuracy")
        if baseline is None:
            continue
        try:
            accuracies.append(float(baseline))
        except Exception:
            continue

    if not accuracies:
        raise RuntimeError(
            "Phase 2A floor calibration found no valid baseline_accuracy values in floor decisions."
        )
    return pd.Series(accuracies, dtype=float)


def calibrate_phase2a_floor_threshold(
    *,
    output_root: Path,
    run_id: str,
    min_pass_rate: float = 0.70,
) -> dict[str, object]:
    phase_root = output_root / "phase2a" / run_id
    task_path = phase_root / "aggregate_task_metrics.parquet"
    calibration_source: str
    if task_path.exists():
        acc = _phase2a_floor_calibration_accuracy_from_aggregate(task_path)
        calibration_source = "aggregate_task_metrics"
    else:
        acc = _phase2a_floor_calibration_accuracy_from_floor_decisions(phase_root)
        calibration_source = "floor_decisions_baseline_accuracy"

    pass_rate_015 = float((acc >= 0.15).mean())
    pass_rate_013 = float((acc >= 0.13).mean())

    selected_threshold: float | None
    status: str
    if pass_rate_015 >= min_pass_rate:
        selected_threshold = 0.15
        status = "pass_keep_0.15"
    elif pass_rate_013 >= min_pass_rate:
        selected_threshold = 0.13
        status = "pass_lower_0.13"
    else:
        selected_threshold = None
        status = "fail_floor_limited"

    payload: dict[str, object] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "phase": "phase2a",
        "run_id": run_id,
        "calibration_source": calibration_source,
        "min_pass_rate": float(min_pass_rate),
        "none_rows_evaluated": int(len(acc)),
        "pass_rate_at_0.15": pass_rate_015,
        "pass_rate_at_0.13": pass_rate_013,
        "status": status,
        "selected_floor_threshold": selected_threshold,
        "proceed_to_phase2b": bool(selected_threshold is not None),
        "rule": "keep_0.15_if_pass_rate>=0.70_else_use_0.13_if_pass_rate>=0.70_else_stop",
    }
    out_path = phase_root / "floor_recalibration.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def write_manifest(rows: list[RunCell], output_root: Path, run_id: str, device: str) -> dict[str, Path]:
    out_paths: dict[str, Path] = {}
    per_phase: dict[str, list[RunCell]] = {}
    for row in rows:
        row_device = RunCell(**{**asdict(row), "device": device, "run_id": run_id})
        per_phase.setdefault(row.phase, []).append(row_device)

    for phase, phase_rows in per_phase.items():
        phase_dir = output_root / phase / run_id
        phase_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = phase_dir / "manifest.jsonl"
        summary_path = phase_dir / "manifest_summary.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            for row in phase_rows:
                handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")
        payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "phase": phase,
            "run_id": run_id,
            "device": device,
            "counts": summarize(phase_rows),
        }
        summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        out_paths[f"{phase}_manifest"] = manifest_path
        out_paths[f"{phase}_summary"] = summary_path
    if "phase2d" in per_phase:
        phase2d_models = tuple(sorted({str(row.model) for row in per_phase["phase2d"]}))
        frozen = freeze_phase2d_head_groups(results_root=output_root.parent, run_id=run_id, models=phase2d_models)
        out_paths["phase2d_frozen_head_groups"] = frozen
    return out_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 2 runner with explicit GPU device policy.")
    parser.add_argument(
        "--mode",
        choices=[
            "build",
            "execute",
            "floor-prepass",
            "reanalyze",
            "posthoc-centered",
            "posthoc-queue-worker",
            "calibrate-floor",
            "feasibility-build",
            "feasibility-sweep",
            "feasibility-finalize",
            "lock-long-offsets",
        ],
        default="build",
        help=(
            "build: generate manifests; execute: run cells from an existing manifest; "
            "floor-prepass: compute baseline floor decisions only from an existing manifest; "
            "reanalyze: recompute aggregate/gate outputs; posthoc-centered: fill deferred centered kernel metrics; "
            "posthoc-queue-worker: persistent lock-claim worker for posthoc queue jobs; "
            "calibrate-floor: apply Phase 2A pilot floor-threshold rule; "
            "feasibility-build: build baseline-only long-offset feasibility manifest only; "
            "feasibility-sweep: run baseline-only long-offset feasibility pilot end-to-end; "
            "feasibility-finalize: aggregate feasibility outputs and write lock recommendation preview; "
            "lock-long-offsets: derive/apply long-offset lock artifact from feasibility outputs."
        ),
    )
    parser.add_argument(
        "--phase",
        choices=["phase2a", "phase2a_long_pilot", "phase2b", "phase2c", "phase2d", "all"],
        default="all",
        help="Which phase matrix to build.",
    )
    parser.add_argument(
        "--model-profile",
        choices=SUPPORTED_MODEL_PROFILES,
        default=DEFAULT_MODEL_PROFILE,
        help="Model profile for build/feasibility matrix generation (default: legacy_1b).",
    )
    parser.add_argument(
        "--model-allowlist",
        default="",
        help=(
            "Optional comma-separated model-name allowlist applied to build/feasibility/lock matrix generation "
            "and recorded in execution artifacts for auditability."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Execution device to store in manifest rows (e.g., cuda, cuda:0, cpu, auto). Default: cuda.",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="If set, auto/cuda requests fall back to CPU when CUDA is unavailable.",
    )
    parser.add_argument(
        "--output-root",
        default="results/experiment2",
        help="Root directory for manifest output.",
    )
    parser.add_argument(
        "--run-id",
        default=datetime.now().strftime("run_%Y%m%d_%H%M%S"),
        help="Run identifier under each phase directory.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print summary JSON to stdout.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to manifest.jsonl for --mode execute.",
    )
    parser.add_argument(
        "--queue-file",
        default=None,
        help="Path to queue .tsv for --mode posthoc-queue-worker.",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Optional cap on number of manifest rows to execute.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Optional starting index into manifest rows for execution.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run cells even if run_config.json already exists.",
    )
    parser.add_argument(
        "--synthetic-count",
        type=int,
        default=200,
        help="Synthetic sequences per cell for execution mode.",
    )
    parser.add_argument(
        "--tier1-count",
        type=int,
        default=100,
        help="Tier1 sequences per cell for execution mode.",
    )
    parser.add_argument(
        "--batch-size-synth",
        type=int,
        default=8,
        help="Synthetic/mechanistic/span-bridge batch size for execute/posthoc modes.",
    )
    parser.add_argument(
        "--batch-size-tier1",
        type=int,
        default=8,
        help="Tier1 batch size for execute/posthoc modes.",
    )
    parser.add_argument(
        "--disable-example-cache",
        action="store_true",
        help="Disable synthetic example cache reuse in execute/posthoc modes.",
    )
    parser.add_argument(
        "--kernel-engine",
        choices=["legacy", "optimized"],
        default="optimized",
        help="Kernel-fit engine for execute mode. optimized is vectorized and parity-gated against legacy.",
    )
    parser.add_argument(
        "--kernel-centered-mode",
        choices=["shared_mean", "legacy_per_sequence"],
        default="shared_mean",
        help="Centered diagnostic mode for Track-B-style kernel metric in Experiment 2.",
    )
    parser.add_argument(
        "--centered-compute",
        choices=["inline", "defer"],
        default="inline",
        help="inline: compute centered kernels during execute (default); defer: skip centered kernels and mark cells pending for posthoc fill.",
    )
    parser.add_argument(
        "--feasibility-task-only",
        action="store_true",
        help=(
            "Execute feasibility rows in task-only mode (skip Track A/Track B kernel extraction and centered pipeline). "
            "Safe for lock decisions because locking uses baseline task accuracy only."
        ),
    )
    parser.add_argument(
        "--random-draws-confirmatory",
        type=int,
        default=DEFAULT_RANDOM_DRAWS_CONFIRMATORY,
        help="Number of random draws per seed for random_* interventions in confirmatory phases.",
    )
    parser.add_argument(
        "--tier1-seeds-phase2b",
        type=int,
        default=DEFAULT_TIER1_SEEDS_PHASE2B,
        help="Tier-1 seeds for Phase 2B confirmatory transfer cells.",
    )
    parser.add_argument(
        "--phase2a-baseline-only",
        action="store_true",
        help="Build Phase 2A as baseline-only pilot (none intervention only).",
    )
    parser.add_argument(
        "--phase2b-core-synthetic-only",
        action="store_true",
        help=(
            "Build Phase 2B synthetic core only (skip tier1/span_bridge rows). "
            "Used for exploratory throughput runs such as Option C."
        ),
    )
    parser.add_argument(
        "--allow-phase2a-reuse",
        action="store_true",
        help="Allow Phase 2A->2B reuse when strict compatibility/provenance checks pass (default: disabled).",
    )
    parser.add_argument(
        "--prune-floor-limited-interventions",
        action="store_true",
        help=(
            "Skip non-baseline synthetic/span-bridge/mechanistic cells when baseline floor gate is failed. "
            "Writes lightweight skip artifacts for audit and keeps confirmatory semantics unchanged."
        ),
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=None,
        help="Optional inclusive seed lower-bound filter for execute mode.",
    )
    parser.add_argument(
        "--seed-end",
        type=int,
        default=None,
        help="Optional inclusive seed upper-bound filter for execute mode.",
    )
    parser.add_argument(
        "--strict-posthoc",
        action="store_true",
        help="Require persisted execution counts for posthoc centered fill; fail if missing.",
    )
    parser.add_argument(
        "--model-filter",
        default="",
        help="Optional comma-separated model names for --mode posthoc-centered.",
    )
    parser.add_argument(
        "--preferred-model",
        default="",
        help="Optional preferred model for --mode posthoc-queue-worker claim locality.",
    )
    parser.add_argument(
        "--synthetic-eval-mode",
        choices=["restricted", "full_vocab"],
        default="restricted",
        help="Evaluation mode for synthetic/span-bridge/mechanistic splits during execute mode.",
    )
    parser.add_argument(
        "--candidate-size",
        type=int,
        default=10,
        help="Candidate set size when --synthetic-eval-mode=restricted (default: 10).",
    )
    parser.add_argument(
        "--h12-endpoint-policy",
        choices=["raw_primary", "co_primary_raw_headroom"],
        default="raw_primary",
        help=(
            "H1/H2 endpoint interpretation policy recorded in run artifacts. "
            "raw_primary preserves legacy behavior; co_primary_raw_headroom records co-primary raw+headroom policy for scale-up runs."
        ),
    )
    parser.add_argument(
        "--intervention-profile",
        choices=INTERVENTION_PROFILES,
        default="full",
        help=(
            "Phase2B intervention profile for manifest build and execution metadata. "
            "full keeps all confirmatory interventions; strong_only keeps {none,ablate_high_strong,ablate_low_strong,random_strong}."
        ),
    )
    parser.add_argument(
        "--track-a-enabled",
        choices=["true", "false"],
        default="true",
        help=(
            "Enable Track A kernel collection during execute/floor-prepass. "
            "Use false for Option C exploratory core throughput runs (Track B raw remains collected)."
        ),
    )
    parser.add_argument(
        "--floor-threshold",
        type=float,
        default=0.15,
        help="Floor gate baseline accuracy threshold for synthetic/span-bridge/mechanistic splits.",
    )
    parser.add_argument(
        "--phase2a-floor-min-pass-rate",
        type=float,
        default=0.70,
        help="Minimum none-cell pass rate used by --mode calibrate-floor (default: 0.70).",
    )
    parser.add_argument(
        "--sweep-run-id",
        default="",
        help="Feasibility sweep run id for --mode lock-long-offsets.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="When used with --mode lock-long-offsets, write experiment2/long_offset_lock.json.",
    )
    parser.add_argument(
        "--feasibility-offsets",
        default=",".join(str(x) for x in DEFAULT_FEASIBILITY_LONG_OFFSETS),
        help=(
            "Comma-separated long offsets for feasibility sweep rows (used by --mode feasibility-sweep, "
            "and as expected offset set for --mode lock-long-offsets)."
        ),
    )
    parser.add_argument(
        "--lock-candidate-offsets",
        default=",".join(str(x) for x in DEFAULT_LOCK_CANDIDATE_OFFSETS),
        help=(
            "Comma-separated candidate long offsets considered for contiguous lock selection "
            "(must be a subset of --feasibility-offsets)."
        ),
    )
    parser.add_argument(
        "--long-task-feasibility-policy",
        choices=LONG_TASK_FEASIBILITY_POLICIES,
        default=DEFAULT_LONG_TASK_FEASIBILITY_POLICY,
        help=(
            "Long-task feasibility lock policy. strict_two_task requires both delayed_copy and long_range_retrieval; "
            "retrieval_fallback allows long_range_retrieval-only fallback when strict policy fails (scaleup_78b only)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.seed_start is not None) and (args.seed_end is not None) and (int(args.seed_start) > int(args.seed_end)):
        raise SystemExit("--seed-start cannot be greater than --seed-end")
    resolved_device = resolve_device(args.device, allow_cpu_fallback=args.allow_cpu_fallback)
    model_allowlist = _normalize_model_allowlist(str(args.model_allowlist), model_profile=str(args.model_profile))
    track_a_enabled = _parse_bool_text(str(args.track_a_enabled), flag_name="--track-a-enabled")
    optionc_exploratory_core = bool(
        str(args.model_profile) == "scaleup_78b"
        and tuple(model_allowlist or tuple()) == ("llama-3.1-8b",)
        and str(args.intervention_profile) == "strong_only"
    )
    if args.mode == "build":
        phases = list(DEFAULT_BUILD_PHASES) if args.phase == "all" else [args.phase]
        _require_confirmatory_long_offset_lock(phases=phases)
        build_opts = _resolve_build_options(
            model_profile=str(args.model_profile),
            random_draws_confirmatory=args.random_draws_confirmatory,
            tier1_seeds_phase2b=args.tier1_seeds_phase2b,
            phase2a_baseline_only=bool(args.phase2a_baseline_only),
            intervention_profile=str(args.intervention_profile),
            phase2b_core_synthetic_only=bool(args.phase2b_core_synthetic_only),
            model_allowlist=model_allowlist,
        )
        rows = build_cells(phases, build_opts)
        output_root = Path(args.output_root)
        output_paths = write_manifest(rows, output_root=output_root, run_id=args.run_id, device=resolved_device)

        summary = {
            "run_id": args.run_id,
            "phases": phases,
            "model_profile": str(args.model_profile),
            "model_allowlist": list(model_allowlist or []),
            "intervention_profile": str(args.intervention_profile),
            "device": resolved_device,
            "counts": summarize(rows),
            "outputs": {k: str(v) for k, v in sorted(output_paths.items())},
        }
        if args.print_summary:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print(f"Experiment 2 manifest written: run_id={args.run_id}, phases={','.join(phases)}, device={resolved_device}")
    elif args.mode == "feasibility-build":
        output_root = Path(args.output_root)
        feasibility_offsets = _parse_int_tuple(args.feasibility_offsets, flag_name="--feasibility-offsets")
        lock_candidate_offsets = _parse_int_tuple(args.lock_candidate_offsets, flag_name="--lock-candidate-offsets")
        if any(offset not in feasibility_offsets for offset in lock_candidate_offsets):
            raise SystemExit(
                "--lock-candidate-offsets must be a subset of --feasibility-offsets. "
                f"feasibility={feasibility_offsets} lock_candidates={lock_candidate_offsets}"
            )
        rows, manifest_path = _build_feasibility_manifest(
            output_root=output_root,
            run_id=args.run_id,
            device=resolved_device,
            model_profile=str(args.model_profile),
            feasibility_offsets=feasibility_offsets,
            model_allowlist=model_allowlist,
        )
        summary = {
            "run_id": args.run_id,
            "mode": "feasibility-build",
            "model_profile": str(args.model_profile),
            "model_allowlist": list(model_allowlist or []),
            "counts": summarize(rows),
            "outputs": {
                "manifest": str(manifest_path),
                "manifest_summary": str(manifest_path.parent / "manifest_summary.json"),
            },
        }
        if args.print_summary:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print(
                "Experiment 2 feasibility manifest built: "
                f"run_id={args.run_id} rows={summary['counts']['total_cells']}"
            )
    elif args.mode == "feasibility-finalize":
        output_root = Path(args.output_root)
        feasibility_offsets = _parse_int_tuple(args.feasibility_offsets, flag_name="--feasibility-offsets")
        lock_candidate_offsets = _parse_int_tuple(args.lock_candidate_offsets, flag_name="--lock-candidate-offsets")
        if any(offset not in feasibility_offsets for offset in lock_candidate_offsets):
            raise SystemExit(
                "--lock-candidate-offsets must be a subset of --feasibility-offsets. "
                f"feasibility={feasibility_offsets} lock_candidates={lock_candidate_offsets}"
            )
        summary = _finalize_feasibility_outputs(
            output_root=output_root,
            run_id=args.run_id,
            threshold=float(args.floor_threshold),
            feasibility_offsets=feasibility_offsets,
            lock_candidate_offsets=lock_candidate_offsets,
            long_task_feasibility_policy=str(args.long_task_feasibility_policy),
            model_profile=str(args.model_profile),
            model_allowlist=model_allowlist,
        )
        if args.print_summary:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            preview = summary["recommendation_preview"]
            print(
                "Experiment 2 feasibility finalize complete: "
                f"run_id={args.run_id} status={preview['status']} selected={preview['selected_long_offsets']}"
            )
    elif args.mode == "feasibility-sweep":
        output_root = Path(args.output_root)
        feasibility_offsets = _parse_int_tuple(args.feasibility_offsets, flag_name="--feasibility-offsets")
        lock_candidate_offsets = _parse_int_tuple(args.lock_candidate_offsets, flag_name="--lock-candidate-offsets")
        if any(offset not in feasibility_offsets for offset in lock_candidate_offsets):
            raise SystemExit(
                "--lock-candidate-offsets must be a subset of --feasibility-offsets. "
                f"feasibility={feasibility_offsets} lock_candidates={lock_candidate_offsets}"
            )
        rows, manifest_path = _build_feasibility_manifest(
            output_root=output_root,
            run_id=args.run_id,
            device=resolved_device,
            model_profile=str(args.model_profile),
            feasibility_offsets=feasibility_offsets,
            model_allowlist=model_allowlist,
        )
        exec_cfg = ExecutionConfig(
            device=resolved_device,
            output_root=output_root,
            kernel_engine=args.kernel_engine,
            kernel_centered_mode=args.kernel_centered_mode,
            centered_compute="defer",
            feasibility_task_only=True,
            synthetic_count=args.synthetic_count,
            tier1_count=args.tier1_count,
            synthetic_eval_mode="restricted",
            candidate_size=10,
            h12_endpoint_policy=str(args.h12_endpoint_policy),
            floor_threshold=float(args.floor_threshold),
            allow_phase2a_reuse=False,
            prune_floor_limited_interventions=bool(args.prune_floor_limited_interventions),
            batch_size_synth=max(1, int(args.batch_size_synth)),
            batch_size_tier1=max(1, int(args.batch_size_tier1)),
            enable_example_cache=not bool(args.disable_example_cache),
            force=args.force,
            model_allowlist=tuple(model_allowlist or ()),
            track_a_enabled=track_a_enabled,
            intervention_profile=str(args.intervention_profile),
            optionc_exploratory_core=optionc_exploratory_core,
        )
        execute_result = run_manifest(Path(manifest_path), config=exec_cfg)
        finalize_summary = _finalize_feasibility_outputs(
            output_root=output_root,
            run_id=args.run_id,
            threshold=float(args.floor_threshold),
            feasibility_offsets=feasibility_offsets,
            lock_candidate_offsets=lock_candidate_offsets,
            long_task_feasibility_policy=str(args.long_task_feasibility_policy),
            model_profile=str(args.model_profile),
            model_allowlist=model_allowlist,
        )
        summary = {
            "run_id": args.run_id,
            "mode": "feasibility-sweep",
            "model_profile": str(args.model_profile),
            "model_allowlist": list(model_allowlist or []),
            "counts": summarize(rows),
            "execute_result": execute_result,
            "finalize_result": finalize_summary,
        }
        if args.print_summary:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            preview = finalize_summary["recommendation_preview"]
            print(
                "Experiment 2 feasibility sweep complete: "
                f"run_id={args.run_id} status={preview['status']} "
                f"selected={preview['selected_long_offsets']}"
            )
    elif args.mode == "lock-long-offsets":
        if not str(args.sweep_run_id).strip():
            raise SystemExit("--sweep-run-id is required for --mode lock-long-offsets")
        feasibility_offsets = _parse_int_tuple(args.feasibility_offsets, flag_name="--feasibility-offsets")
        lock_candidate_offsets = _parse_int_tuple(args.lock_candidate_offsets, flag_name="--lock-candidate-offsets")
        if any(offset not in feasibility_offsets for offset in lock_candidate_offsets):
            raise SystemExit(
                "--lock-candidate-offsets must be a subset of --feasibility-offsets. "
                f"feasibility={feasibility_offsets} lock_candidates={lock_candidate_offsets}"
            )
        output_root = Path(args.output_root)
        sweep_root = output_root / "feasibility" / str(args.sweep_run_id)
        task_path = sweep_root / "aggregate_task_metrics.parquet"
        if not task_path.exists():
            raise FileNotFoundError(
                f"Missing feasibility aggregate task metrics: {task_path}. "
                "Run --mode feasibility-sweep first."
            )
        task_df = pd.read_parquet(task_path)
        recommendation = _compute_long_offset_lock_recommendation(
            task_df,
            threshold=float(args.floor_threshold),
            feasibility_offsets=feasibility_offsets,
            candidate_order=lock_candidate_offsets,
            long_task_feasibility_policy=str(args.long_task_feasibility_policy),
            model_profile=str(args.model_profile),
            model_allowlist=model_allowlist,
        )
        lock_payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source_run_id": str(args.sweep_run_id),
            "threshold": float(args.floor_threshold),
            "status": recommendation["status"],
            "lock_policy": recommendation["lock_policy"],
            "long_task_policy": recommendation.get("long_task_policy_applied", "strict_two_task"),
            "lock_resolution_mode": recommendation.get("lock_resolution_mode", "strict_two_task"),
            "active_long_tasks": recommendation.get("active_long_tasks", ("delayed_copy", "long_range_retrieval")),
            "task_feasible_rule": recommendation["task_feasible_rule"],
            "model_profile": str(args.model_profile),
            "model_allowlist": [str(x) for x in (model_allowlist or tuple())],
            "effective_models": recommendation.get("effective_models", []),
            "selected_long_offsets": recommendation["selected_long_offsets"],
            "strict_selected_long_offsets": recommendation.get("strict_selected_long_offsets", recommendation["selected_long_offsets"]),
            "fallback_selected_long_offsets": recommendation.get("fallback_selected_long_offsets", recommendation["selected_long_offsets"]),
            "fallback_spans": recommendation["fallback_spans"],
            "min_pass_rate": recommendation["min_pass_rate"],
            "feasibility_offsets": recommendation["feasibility_offsets"],
            "candidate_order": recommendation["candidate_order"],
            "feasible_by_offset": recommendation["feasible_by_offset"],
            "strict_feasible_by_offset": recommendation.get("strict_feasible_by_offset", recommendation["feasible_by_offset"]),
            "retrieval_only_feasible_by_offset": recommendation.get("retrieval_only_feasible_by_offset", {}),
            "task_feasible_by_offset": recommendation["task_feasible_by_offset"],
        }
        (sweep_root / "lock_recommendation.json").write_text(
            json.dumps(
                {
                    **lock_payload,
                    "support_table_rows": recommendation["support_table_rows"],
                    "task_model_support_rows": recommendation["task_model_support_rows"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        lock_hash = None
        if bool(args.apply):
            lock_hash = _write_long_offset_lock(lock_payload)
        summary = {
            "mode": "lock-long-offsets",
            "sweep_run_id": str(args.sweep_run_id),
            "model_profile": str(args.model_profile),
            "model_allowlist": [str(x) for x in (model_allowlist or tuple())],
            "effective_models": recommendation.get("effective_models", []),
            "status": lock_payload["status"],
            "selected_long_offsets": lock_payload["selected_long_offsets"],
            "fallback_spans": lock_payload["fallback_spans"],
            "active_long_tasks": lock_payload.get("active_long_tasks", []),
            "long_task_policy": lock_payload.get("long_task_policy", "strict_two_task"),
            "lock_resolution_mode": lock_payload.get("lock_resolution_mode", "strict_two_task"),
            "applied": bool(args.apply),
            "lock_path": str(LOCK_PATH),
            "lock_hash": lock_hash,
            "recommendation_path": str(sweep_root / "lock_recommendation.json"),
        }
        if args.print_summary:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print(
                "Experiment 2 long-offset lock recommendation complete: "
                f"sweep_run_id={args.sweep_run_id} status={lock_payload['status']} "
                f"applied={bool(args.apply)}"
            )
    elif args.mode == "execute":
        if not args.manifest:
            raise SystemExit("--manifest is required for --mode execute")
        exec_cfg = ExecutionConfig(
            device=resolved_device,
            output_root=Path(args.output_root),
            kernel_engine=args.kernel_engine,
            kernel_centered_mode=args.kernel_centered_mode,
            centered_compute=args.centered_compute,
            feasibility_task_only=bool(args.feasibility_task_only),
            max_cells=args.max_cells,
            start_index=args.start_index,
            synthetic_count=args.synthetic_count,
            tier1_count=args.tier1_count,
            synthetic_eval_mode=args.synthetic_eval_mode,
            candidate_size=max(2, int(args.candidate_size)),
            h12_endpoint_policy=str(args.h12_endpoint_policy),
            floor_threshold=float(args.floor_threshold),
            allow_phase2a_reuse=bool(args.allow_phase2a_reuse),
            prune_floor_limited_interventions=bool(args.prune_floor_limited_interventions),
            batch_size_synth=max(1, int(args.batch_size_synth)),
            batch_size_tier1=max(1, int(args.batch_size_tier1)),
            seed_start=(None if args.seed_start is None else int(args.seed_start)),
            seed_end=(None if args.seed_end is None else int(args.seed_end)),
            enable_example_cache=not bool(args.disable_example_cache),
            force=args.force,
            model_allowlist=tuple(model_allowlist or ()),
            track_a_enabled=track_a_enabled,
            intervention_profile=str(args.intervention_profile),
            optionc_exploratory_core=optionc_exploratory_core,
        )
        result = run_manifest(Path(args.manifest), config=exec_cfg)
        if args.print_summary:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(
                "Experiment 2 execution complete: "
                f"status={result['status']} processed={result['processed']} skipped={result['skipped']} failures={result['failures']}"
            )
    elif args.mode == "floor-prepass":
        if not args.manifest:
            raise SystemExit("--manifest is required for --mode floor-prepass")
        exec_cfg = ExecutionConfig(
            device=resolved_device,
            output_root=Path(args.output_root),
            kernel_engine=args.kernel_engine,
            kernel_centered_mode=args.kernel_centered_mode,
            centered_compute="defer",
            max_cells=args.max_cells,
            start_index=args.start_index,
            synthetic_count=args.synthetic_count,
            tier1_count=args.tier1_count,
            synthetic_eval_mode=args.synthetic_eval_mode,
            candidate_size=max(2, int(args.candidate_size)),
            h12_endpoint_policy=str(args.h12_endpoint_policy),
            floor_threshold=float(args.floor_threshold),
            allow_phase2a_reuse=False,
            prune_floor_limited_interventions=False,
            batch_size_synth=max(1, int(args.batch_size_synth)),
            batch_size_tier1=max(1, int(args.batch_size_tier1)),
            seed_start=(None if args.seed_start is None else int(args.seed_start)),
            seed_end=(None if args.seed_end is None else int(args.seed_end)),
            enable_example_cache=not bool(args.disable_example_cache),
            floor_prepass_only=True,
            force=args.force,
            model_allowlist=tuple(model_allowlist or ()),
            track_a_enabled=track_a_enabled,
            intervention_profile=str(args.intervention_profile),
            optionc_exploratory_core=optionc_exploratory_core,
        )
        result = run_manifest(Path(args.manifest), config=exec_cfg)
        if args.print_summary:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(
                "Experiment 2 floor-prepass complete: "
                f"status={result['status']} processed_baselines={result.get('processed_baselines', 0)} "
                f"floor_limited={result.get('floor_limited', 0)} failures={result.get('failures', 0)}"
            )
    elif args.mode == "reanalyze":
        phases = list(PHASE_BUILDERS.keys()) if args.phase == "all" else [args.phase]
        result = reanalyze_run(output_root=Path(args.output_root), run_id=args.run_id, phases=phases)
        if args.print_summary:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            ok = [p for p, s in result["phases"].items() if s == "ok"]
            missing = [p for p, s in result["phases"].items() if s != "ok"]
            print(
                "Experiment 2 reanalysis complete: "
                f"run_id={args.run_id} updated={','.join(ok) if ok else 'none'} missing={','.join(missing) if missing else 'none'}"
            )
    elif args.mode == "calibrate-floor":
        result = calibrate_phase2a_floor_threshold(
            output_root=Path(args.output_root),
            run_id=args.run_id,
            min_pass_rate=float(args.phase2a_floor_min_pass_rate),
        )
        if args.print_summary:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            threshold_txt = (
                "none (stop at pilot)"
                if result.get("selected_floor_threshold") is None
                else f"{result['selected_floor_threshold']:.2f}"
            )
            print(
                "Experiment 2 floor calibration complete: "
                f"run_id={args.run_id} status={result['status']} selected_floor_threshold={threshold_txt}"
            )
    elif args.mode == "posthoc-centered":
        if args.phase == "all":
            raise SystemExit("--mode posthoc-centered requires a single --phase")
        model_filter = frozenset(
            m.strip() for m in str(args.model_filter).split(",") if m.strip()
        ) or None
        posthoc_cfg = PosthocCenteredConfig(
            output_root=Path(args.output_root),
            phase=args.phase,
            run_id=args.run_id,
            device=resolved_device,
            force=args.force,
            synthetic_count=args.synthetic_count,
            tier1_count=args.tier1_count,
            batch_size_synth=max(1, int(args.batch_size_synth)),
            batch_size_tier1=max(1, int(args.batch_size_tier1)),
            strict_posthoc=args.strict_posthoc,
            model_filter=model_filter,
            seed_start=(None if args.seed_start is None else int(args.seed_start)),
            seed_end=(None if args.seed_end is None else int(args.seed_end)),
            enable_example_cache=not bool(args.disable_example_cache),
        )
        result = run_posthoc_centered(posthoc_cfg)
        if args.print_summary:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(
                "Experiment 2 posthoc-centered complete: "
                f"phase={args.phase} run_id={args.run_id} processed={result['processed']} skipped={result.get('skipped', 0)}"
            )
    elif args.mode == "posthoc-queue-worker":
        if args.phase == "all":
            raise SystemExit("--mode posthoc-queue-worker requires a single --phase")
        if not args.queue_file:
            raise SystemExit("--queue-file is required for --mode posthoc-queue-worker")
        worker_cfg = PosthocQueueWorkerConfig(
            queue_file=Path(args.queue_file),
            output_root=Path(args.output_root),
            phase=args.phase,
            device=resolved_device,
            batch_size_synth=max(1, int(args.batch_size_synth)),
            batch_size_tier1=max(1, int(args.batch_size_tier1)),
            strict_posthoc=bool(args.strict_posthoc),
            force=bool(args.force),
            enable_example_cache=not bool(args.disable_example_cache),
            preferred_model=(str(args.preferred_model).strip() or None),
        )
        result = run_posthoc_queue_worker(worker_cfg)
        if args.print_summary:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(
                "Experiment 2 posthoc queue worker complete: "
                f"status={result['status']} done_jobs={result['done_jobs']} "
                f"failed_jobs={result['failed_jobs']} pending_jobs={result['pending_jobs']}"
            )
    else:
        raise SystemExit(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
