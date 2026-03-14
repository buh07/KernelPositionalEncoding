from __future__ import annotations

import hashlib
import json
import random
import shutil
import gc
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import torch

from experiment1.paths import tokenized_path
from experiment1.track_a import SequenceRecord as TrackASequenceRecord
from experiment2.analysis import write_phase_analysis
from experiment2.config import (
    DATASET_BY_NAME,
    DEFAULT_SYNTHETIC_COUNT,
    DEFAULT_TIER1_COUNT,
    MODEL_BY_NAME,
    Experiment2Paths,
)
from experiment2.flooring import FloorDecision, floor_key, fallback_spans_for_task, intervention_rank
from experiment2.example_cache import load_cached_examples, make_cache_key, write_cached_examples
from experiment2.head_groups import load_target_heads
from experiment2.interventions import (
    AbsPEDCTInterventionContext,
    InterventionAudit,
    InterventionPlan,
    RopeQKInterventionContext,
    build_abspe_dct_plan,
    build_rope_intervention_plan,
)
from experiment2.kernels import KernelMetricsAccumulator
from experiment2.long_offsets import load_long_offset_bundle
from experiment2.provenance import (
    generator_hash,
    intervention_hash,
    model_tokenizer_hash,
    provenance_match,
    provenance_payload,
    row_hash,
)
from experiment2.summarize import write_norm_family_decomposition, write_protocol_revision_log
from experiment2.summarize import write_claim_guard, write_h4_interaction_exploratory
from experiment2.tasks import TaskExample, build_token_pools, generate_task_examples
from experiment2.tier1 import (
    build_frozen_bins,
    dependency_delta,
    load_frozen_bins,
    local_concentration_scores,
    stratified_metrics,
    summarize_frozen_bins,
)
from shared.attention import get_adapter
from shared.models.loading import load_model, load_tokenizer
from shared.specs import SequenceLengthSpec
from shared.utils.logging import get_logger

LOGGER = get_logger("experiment2.execution")


@dataclass
class ExecutionConfig:
    device: str
    output_root: Path
    kernel_engine: str = "optimized"
    kernel_centered_mode: str = "shared_mean"
    centered_compute: str = "inline"
    max_cells: int | None = None
    start_index: int = 0
    synthetic_count: int = DEFAULT_SYNTHETIC_COUNT
    tier1_count: int = DEFAULT_TIER1_COUNT
    synthetic_eval_mode: str = "restricted"
    candidate_size: int = 10
    candidate_policy_version: str = "restricted_candidates_v1_structured_first"
    floor_threshold: float = 0.15
    allow_phase2a_reuse: bool = False
    prune_floor_limited_interventions: bool = False
    batch_size_synth: int = 8
    batch_size_tier1: int = 8
    seed_start: int | None = None
    seed_end: int | None = None
    enable_example_cache: bool = True
    floor_prepass_only: bool = False
    force: bool = False
    feasibility_task_only: bool = False
    h12_endpoint_policy: str = "raw_primary"
    model_allowlist: tuple[str, ...] = tuple()
    track_a_enabled: bool = True
    intervention_profile: str = "full"
    optionc_exploratory_core: bool = False


@dataclass
class _RunState:
    floor_cache: dict[str, FloorDecision] = field(default_factory=dict)
    reuse_events: list[dict[str, Any]] = field(default_factory=list)
    resource_cache: dict[tuple[str, str], "_CachedResources"] = field(default_factory=dict)
    batch_size_memory: dict[tuple[str, str], int] = field(default_factory=dict)
    intervention_plan_cache: dict[tuple[Any, ...], dict[str, Any]] = field(default_factory=dict)


@dataclass
class _CachedResources:
    model_spec: Any
    model: Any
    tokenizer: Any
    adapter: Any
    pools: Any


def run_manifest(manifest_path: Path, config: ExecutionConfig) -> dict[str, Any]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if config.model_allowlist:
        allow = set(str(name) for name in config.model_allowlist)
        rows = [row for row in rows if str(row.get("model")) in allow]
    if (config.seed_start is not None) or (config.seed_end is not None):
        s0 = int(config.seed_start) if config.seed_start is not None else None
        s1 = int(config.seed_end) if config.seed_end is not None else None
        filtered: list[dict[str, Any]] = []
        for row in rows:
            if "seed" not in row:
                continue
            seed = int(row["seed"])
            if s0 is not None and seed < s0:
                continue
            if s1 is not None and seed > s1:
                continue
            filtered.append(row)
        rows = filtered
    if config.start_index:
        rows = rows[config.start_index :]
    if config.max_cells is not None:
        rows = rows[: config.max_cells]
    if not rows:
        return {"status": "no_rows", "manifest": str(manifest_path), "processed": 0}

    if config.floor_prepass_only:
        return _run_floor_prepass(rows=rows, manifest_path=manifest_path, config=config)

    rows.sort(key=_row_sort_key)
    state = _RunState()

    processed = 0
    skipped = 0
    failures = 0
    per_phase: dict[str, int] = {}
    for idx, row in enumerate(rows):
        try:
            if _execute_cell(row, config, state):
                processed += 1
                per_phase[row["phase"]] = per_phase.get(row["phase"], 0) + 1
            else:
                skipped += 1
        except Exception as exc:
            failures += 1
            LOGGER.exception("Experiment2 cell failed idx=%s row=%s error=%s", idx, row, exc)

    run_id = str(rows[0].get("run_id") or _infer_run_id_from_manifest(manifest_path))
    for phase in sorted(per_phase):
        _aggregate_phase_outputs(
            config.output_root,
            phase=phase,
            run_id=run_id,
            reuse_events=[e for e in state.reuse_events if e.get("phase") == phase],
        )
    return {
        "status": "ok" if failures == 0 else "partial_failure",
        "manifest": str(manifest_path),
        "processed": processed,
        "skipped": skipped,
        "failures": failures,
        "per_phase": per_phase,
        "model_allowlist": [str(x) for x in config.model_allowlist],
    }


def _run_floor_prepass(*, rows: list[dict[str, Any]], manifest_path: Path, config: ExecutionConfig) -> dict[str, Any]:
    rows.sort(key=_row_sort_key)
    state = _RunState()
    processed = 0
    skipped = 0
    failures = 0
    floor_limited = 0
    floor_not_limited = 0
    per_phase: dict[str, int] = {}
    per_model: dict[str, int] = {}

    for idx, row in enumerate(rows):
        split = str(row.get("split"))
        if split not in {"synthetic", "span_bridge", "mechanistic"}:
            skipped += 1
            continue
        if str(row.get("intervention")) != "none":
            skipped += 1
            continue
        try:
            resources = _get_cached_resources(row=row, config=config, state=state)
            decision = _resolve_floor_decision(
                row=row,
                config=config,
                model=resources.model,
                model_spec=resources.model_spec,
                adapter=resources.adapter,
                pools=resources.pools,
                state=state,
            )
            processed += 1
            per_phase[str(row.get("phase"))] = per_phase.get(str(row.get("phase")), 0) + 1
            per_model[str(row.get("model"))] = per_model.get(str(row.get("model")), 0) + 1
            if decision.floor_limited:
                floor_limited += 1
            else:
                floor_not_limited += 1
        except Exception as exc:
            failures += 1
            LOGGER.exception("Experiment2 floor-prepass failed idx=%s row=%s error=%s", idx, row, exc)

    run_id = str(rows[0].get("run_id") or _infer_run_id_from_manifest(manifest_path))
    return {
        "status": "ok" if failures == 0 else "partial_failure",
        "mode": "floor-prepass",
        "manifest": str(manifest_path),
        "run_id": run_id,
        "processed_baselines": processed,
        "skipped_rows": skipped,
        "failures": failures,
        "floor_limited": floor_limited,
        "floor_not_limited": floor_not_limited,
        "per_phase": per_phase,
        "per_model": per_model,
    }


def reanalyze_run(*, output_root: Path, run_id: str, phases: list[str]) -> dict[str, Any]:
    completed: dict[str, str] = {}
    blocked_phases: list[str] = []
    for phase in phases:
        phase_root = output_root / phase / run_id
        if not phase_root.exists():
            completed[phase] = "missing"
            continue
        reuse_events: list[dict[str, Any]] = []
        reuse_path = phase_root / "reuse_decisions.json"
        if reuse_path.exists():
            try:
                payload = json.loads(reuse_path.read_text(encoding="utf-8"))
                reuse_events = list(payload.get("events", []))
            except Exception:
                reuse_events = []
        _aggregate_phase_outputs(output_root, phase=phase, run_id=run_id, reuse_events=reuse_events)
        gate_path = phase_root / "gate_evaluation.json"
        status = "ok"
        if gate_path.exists():
            try:
                gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
                if str(gate_payload.get("status")) == "blocked_centered_pending":
                    status = "blocked_centered_pending"
                    blocked_phases.append(phase)
            except Exception:
                status = "ok"
        completed[phase] = status
    top_status = "ok" if not blocked_phases else "blocked_centered_pending"
    return {
        "status": top_status,
        "run_id": run_id,
        "output_root": str(output_root),
        "phases": completed,
        "blocked_centered_pending_phases": blocked_phases,
    }


def _row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    split_order = {
        # Prioritize confirmatory synthetic cells first; bridge rows are
        # independent and can run later without changing results.
        "synthetic": 0,
        "tier1_ppl": 1,
        "span_bridge": 2,
        "mechanistic": 3,
    }
    split_rank = split_order.get(str(row.get("split")), 99)
    return (
        str(row.get("phase")),
        str(row.get("model")),
        split_rank,
        str(row.get("split")),
        str(row.get("dataset")),
        str(row.get("task")),
        int(row.get("seq_len", 0)),
        int(row.get("seed", 0)),
        int(row.get("span", -1) or -1),
        intervention_rank(str(row.get("intervention", ""))),
    )


def _batch_size_for_row(row: dict[str, Any], config: ExecutionConfig) -> int:
    if row.get("split") == "tier1_ppl":
        return max(1, int(config.batch_size_tier1))
    base = max(1, int(config.batch_size_synth))
    split = str(row.get("split"))
    seq_len = int(row.get("seq_len", 0))
    model_name = str(row.get("model"))
    phase_name = str(row.get("phase"))
    # Scale-up Phase 2A at 1024 tokens is OOM-prone on 7-8B models. Start at
    # batch size 1 to avoid wasteful retry passes.
    if (
        phase_name == "phase2a"
        and split in {"synthetic", "span_bridge", "mechanistic"}
        and seq_len >= 1024
        and model_name in {"olmo-2-7b", "llama-3.1-8b", "gemma-7b"}
    ):
        return 1
    # Phase 2B span-bridge baselines are especially memory-heavy at 7-8B scale.
    # Start directly at batch size 1 to avoid wasting one OOM retry pass.
    if (
        phase_name == "phase2b"
        and split in {"synthetic", "span_bridge", "mechanistic"}
        and seq_len >= 1024
        and model_name in {"olmo-2-7b", "llama-3.1-8b", "gemma-7b"}
    ):
        return 1
    # Long 1024-token dependency tasks are consistently OOM-prone at larger
    # batch sizes on 1B-class models; start conservatively to avoid retry waste.
    if (
        split in {"synthetic", "span_bridge", "mechanistic"}
        and seq_len >= 1024
        and str(row.get("task")) in {"delayed_copy", "long_range_retrieval", "retrieval_bridge", "copy_offset_bridge"}
    ):
        return min(base, 2)
    return base


def _iter_example_batches(examples: list[TaskExample], batch_size: int) -> list[list[TaskExample]]:
    return [examples[i : i + batch_size] for i in range(0, len(examples), batch_size)]


def _prepare_batch_tensors(batch: list[TaskExample], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    if not batch:
        raise RuntimeError("Cannot prepare tensors for empty example batch.")
    seq_lens = {len(ex.tokens) for ex in batch}
    if len(seq_lens) != 1:
        raise RuntimeError(f"Mixed sequence lengths in batch are unsupported: {sorted(seq_lens)}")
    seq_len = int(next(iter(seq_lens)))
    input_ids = torch.tensor([ex.tokens for ex in batch], dtype=torch.long, device=device)
    attention_mask = torch.ones((len(batch), seq_len), dtype=torch.long, device=device)
    return input_ids, attention_mask


def _token_logits_for_index(token_logits: torch.Tensor | None, index: int) -> torch.Tensor:
    if token_logits is None:
        raise RuntimeError("Missing token logits.")
    if token_logits.ndim == 2:
        if index != 0:
            raise RuntimeError("Requested non-zero batch index from single-example token logits.")
        return token_logits
    if token_logits.ndim == 3:
        return token_logits[index]
    raise RuntimeError(f"Unexpected token logits shape {tuple(token_logits.shape)}")


def _capture_logits_batched(logits: torch.Tensor | None) -> torch.Tensor:
    if logits is None:
        raise RuntimeError("Missing capture logits.")
    if logits.ndim == 4:
        return logits.unsqueeze(1)
    if logits.ndim == 5:
        return logits
    raise RuntimeError(f"Unexpected capture logits shape {tuple(logits.shape)}")


def _empty_kernel_summary_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "model",
            "phase",
            "split",
            "dataset",
            "task",
            "span",
            "intervention",
            "random_draw",
            "seed",
            "seq_len",
            "layer",
            "head",
            "mean_r2",
            "std_r2",
            "count",
        ]
    )


def _get_cached_resources(row: dict[str, Any], config: ExecutionConfig, state: _RunState) -> _CachedResources:
    model_name = str(row["model"])
    key = (model_name, str(config.device))
    cached = state.resource_cache.get(key)
    if cached is not None:
        return cached
    model_spec = MODEL_BY_NAME[model_name]
    loader = load_model(model_spec)
    model = loader.model.to(config.device)
    model.eval()
    tokenizer = load_tokenizer(model_spec)
    adapter = get_adapter(model_spec)
    pools = build_token_pools(
        model_name=model_spec.name,
        vocab_size=tokenizer.vocab_size,
        special_ids=getattr(tokenizer, "all_special_ids", []),
    )
    cached = _CachedResources(
        model_spec=model_spec,
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        pools=pools,
    )
    state.resource_cache[key] = cached
    return cached


def _execute_cell(row: dict[str, Any], config: ExecutionConfig, state: _RunState) -> bool:
    out_dir = _condition_dir(config.output_root, row)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = out_dir / "run_config.json"
    if run_config_path.exists() and not config.force:
        LOGGER.info(
            "Experiment2 skip existing phase=%s model=%s task=%s intervention=%s seed=%s",
            row["phase"],
            row["model"],
            row["task"],
            row["intervention"],
            row["seed"],
        )
        return False

    resources = _get_cached_resources(row=row, config=config, state=state)
    model_spec = resources.model_spec
    model = resources.model
    tokenizer = resources.tokenizer
    adapter = resources.adapter
    pools = resources.pools

    prov = provenance_payload(
        generator=generator_hash(),
        intervention=intervention_hash(),
        model_tokenizer=model_tokenizer_hash(model_spec, model, tokenizer),
    )
    long_offset_bundle = load_long_offset_bundle(strict=False)
    phase_name = str(row.get("phase"))
    is_task_metrics_only = bool(config.feasibility_task_only)
    is_feasibility_task_only = bool(is_task_metrics_only and phase_name == "feasibility")
    if is_feasibility_task_only and str(row.get("intervention")) != "none":
        raise RuntimeError(
            "feasibility_task_only mode expects baseline-only cells (intervention='none'). "
            f"Got intervention={row.get('intervention')}"
        )

    reused = _maybe_reuse_from_phase2a(row=row, out_dir=out_dir, config=config, provenance=prov, state=state)
    if reused:
        return True

    if is_feasibility_task_only:
        floor_decision = FloorDecision(
            key=floor_key(row),
            baseline_accuracy=1.0,
            fallback_accuracy=None,
            fallback_applied=False,
            floor_limited=False,
            fallback_spans=tuple(),
        )
        fallback_span_choices = tuple()
    else:
        floor_decision = _resolve_floor_decision(
            row=row,
            config=config,
            model=model,
            model_spec=model_spec,
            adapter=adapter,
            pools=pools,
            state=state,
        )
        fallback_span_choices = floor_decision.fallback_spans if floor_decision.fallback_applied else tuple()

    should_prune_floor_limited = bool(
        config.prune_floor_limited_interventions
        and (not is_feasibility_task_only)
        and str(row.get("split")) in {"synthetic", "span_bridge", "mechanistic"}
        and str(row.get("intervention")) != "none"
        and bool(floor_decision.floor_limited)
    )
    if should_prune_floor_limited:
        run_payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "row": row,
            "device": config.device,
            "kernel_engine": config.kernel_engine,
            "kernel_centered_mode": config.kernel_centered_mode,
            "centered_compute": config.centered_compute,
            "centered_pending": False,
            "feasibility_task_only": False,
            "execution_status": "skipped_floor_limited",
            "skip_reason": "floor_limited_non_baseline_pruned",
            "floor_prune_enabled": True,
            "implementation": "experiment2.execution.v8_floor_prune",
            "synthetic_count_used": int(config.synthetic_count),
            "tier1_count_used": int(config.tier1_count),
            "synthetic_eval_mode": str(config.synthetic_eval_mode),
            "h12_endpoint_policy": str(config.h12_endpoint_policy),
            "track_a_enabled": bool(config.track_a_enabled),
            "intervention_profile": str(config.intervention_profile),
            "optionc_exploratory_core": bool(config.optionc_exploratory_core),
            "candidate_size": int(config.candidate_size),
            "candidate_policy_version": str(config.candidate_policy_version),
            "floor_threshold": float(config.floor_threshold),
            "allow_phase2a_reuse": bool(config.allow_phase2a_reuse),
            "model_allowlist": [str(x) for x in config.model_allowlist],
            "long_offset_lock_hash": long_offset_bundle.lock_hash,
            "long_offsets_used": [int(x) for x in long_offset_bundle.long_offsets],
            "fallback_spans_used": [int(x) for x in long_offset_bundle.fallback_spans],
            "example_cache_enabled": bool(config.enable_example_cache),
            **prov,
            "floor": floor_decision.to_json_dict(),
        }
        run_config_path.write_text(json.dumps(run_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        skip_payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "row": row,
            "reason": "floor_limited_non_baseline_pruned",
            "non_evaluable_policy": "confirmatory_excludes_floor_limited",
            "floor": floor_decision.to_json_dict(),
            "row_hash": row_hash(row),
            "long_offset_lock_hash": long_offset_bundle.lock_hash,
        }
        (out_dir / "skip_record.json").write_text(
            json.dumps(skip_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return True

    examples = _load_or_build_examples(
        row=row,
        config=config,
        pools=pools,
        span_choices=fallback_span_choices,
        generator_hash_value=prov["generator_hash"],
    )
    if not examples:
        raise RuntimeError(f"No examples built for row={row}")

    plan, intervention_cm = _build_intervention(
        model,
        model_spec,
        row,
        results_root=config.output_root.parent,
        run_id=str(row.get("run_id", "")),
        cache=state.intervention_plan_cache,
    )
    audit = InterventionAudit(plan=plan, notes="Intervention not applied (none)")

    split_name = str(row.get("split"))
    phase_name = str(row.get("phase"))
    intervention_name = str(row.get("intervention"))
    # Kernel summaries are only needed for synthetic/mechanistic tracks.
    # Span-bridge and tier1 rely on task metrics, so skip heavy attention capture.
    capture_kernels = (
        (not is_task_metrics_only)
        and split_name in {"synthetic", "mechanistic"}
        and not (phase_name == "phase2a" and intervention_name != "none")
    )

    kernel_accum = None
    if capture_kernels:
        kernel_accum = KernelMetricsAccumulator(
            model_spec,
            engine=config.kernel_engine,
            centered_mode=config.kernel_centered_mode,
        )

    all_acc: list[float] = []
    all_nll: list[float] = []
    all_acc_full_vocab: list[float] = []
    all_nll_full_vocab: list[float] = []
    all_acc_restricted: list[float] = []
    all_nll_restricted: list[float] = []
    all_targets = 0
    no_match_rates: list[float] = []
    no_match_targets = 0
    no_match_examples = 0
    per_pos_rows: list[dict[str, Any]] = []
    baseline_bin_rows: list[dict[str, Any]] = []
    batch_memory_key = (str(row["model"]), str(row.get("split")))
    effective_batch_size = int(state.batch_size_memory.get(batch_memory_key, _batch_size_for_row(row, config)))
    phase2a_nonbaseline = bool(phase_name == "phase2a" and intervention_name != "none")

    def _run_cell_passes(batch_size: int) -> None:
        nonlocal audit, all_targets, no_match_targets, no_match_examples
        nonlocal all_acc, all_nll, all_acc_full_vocab, all_nll_full_vocab, all_acc_restricted, all_nll_restricted
        nonlocal per_pos_rows, baseline_bin_rows, no_match_rates
        all_acc = []
        all_nll = []
        all_acc_full_vocab = []
        all_nll_full_vocab = []
        all_acc_restricted = []
        all_nll_restricted = []
        all_targets = 0
        no_match_rates = []
        no_match_targets = 0
        no_match_examples = 0
        per_pos_rows = []
        baseline_bin_rows = []
        if kernel_accum is not None:
            kernel_accum.track_a.clear()
            kernel_accum.track_b_raw.clear()
            kernel_accum.track_b_centered.clear()
            kernel_accum._sum_q = None
            kernel_accum._sum_k = None
            kernel_accum._count_tokens = 0

        batches = _iter_example_batches(examples, batch_size)

        # Pass 1: task metrics (+ Track A/raw/shared-mean accumulation when enabled).
        for batch in batches:
            input_ids, attention_mask = _prepare_batch_tensors(batch, config.device)
            capture = None
            batch_logits = None
            if is_task_metrics_only:
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                if hasattr(outputs, "logits"):
                    batch_logits = outputs.logits
                elif isinstance(outputs, (tuple, list)) and outputs:
                    batch_logits = outputs[0]
                else:
                    raise RuntimeError("Unexpected model output while computing task-only feasibility logits.")
                if batch_logits is None or batch_logits.ndim != 3:
                    raise RuntimeError(f"Unexpected logits shape in feasibility task-only mode: {tuple(batch_logits.shape)}")
            else:
                # Track A computation requires attention logits. Keep logits on
                # whenever Track A is enabled and kernel accumulation is active.
                # Tier1 baseline also needs logits for concentration/bin diagnostics.
                needs_attention_logits = bool(
                    (kernel_accum is not None and bool(config.track_a_enabled))
                    or (split_name == "tier1_ppl" and intervention_name == "none")
                )
                needs_attention_capture = bool(kernel_accum is not None or needs_attention_logits)
                capture = adapter.capture(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    include_logits=needs_attention_logits,
                    return_token_logits=True,
                    capture_attention=needs_attention_capture,
                    output_device=config.device,
                )
            for local_idx, ex in enumerate(batch):
                if is_task_metrics_only:
                    token_logits = batch_logits[local_idx]
                else:
                    token_logits = _token_logits_for_index(capture.token_logits, local_idx)
                metrics, per_pos = _evaluate_example_from_token_logits(
                    token_logits,
                    ex,
                    split=str(row.get("split")),
                    pools=pools,
                    synthetic_eval_mode=str(config.synthetic_eval_mode),
                    candidate_size=int(config.candidate_size),
                    candidate_policy_version=str(config.candidate_policy_version),
                )
                all_acc.append(metrics["accuracy"])
                all_nll.append(metrics["mean_nll"])
                all_acc_full_vocab.append(metrics["full_vocab_accuracy"])
                all_nll_full_vocab.append(metrics["full_vocab_mean_nll"])
                if metrics["restricted_accuracy"] is not None:
                    all_acc_restricted.append(float(metrics["restricted_accuracy"]))
                if metrics["restricted_mean_nll"] is not None:
                    all_nll_restricted.append(float(metrics["restricted_mean_nll"]))
                all_targets += metrics["num_targets"]
                per_pos_rows.extend(per_pos)

                if ex.task_name == "local_key_match":
                    rate = float(ex.task_params.get("no_match_rate", 0.0))
                    no_match_rates.append(rate)
                    if ex.has_no_match:
                        no_match_examples += 1
                        no_match_targets += sum(1 for t in ex.target_tokens if t == ex.task_params.get("no_match_token", -1))

                if (
                    (not is_task_metrics_only)
                    and row.get("split") == "tier1_ppl"
                    and row.get("intervention") == "none"
                    and capture.logits is not None
                ):
                    logits_b = _capture_logits_batched(capture.logits)
                    layer_logits = logits_b[:, local_idx]
                    local_scores_w8 = local_concentration_scores(layer_logits, target_positions=ex.target_positions, window=8)
                    local_scores_w16 = local_concentration_scores(layer_logits, target_positions=ex.target_positions, window=16)
                    local_scores_w32 = local_concentration_scores(layer_logits, target_positions=ex.target_positions, window=32)
                    for rec in per_pos:
                        pos = int(rec["position"])
                        baseline_bin_rows.append(
                            {
                                "example_id": ex.id,
                                "position": pos,
                                "local_score": float(local_scores_w16.get(pos, float("nan"))),
                                "local_score_w8": float(local_scores_w8.get(pos, float("nan"))),
                                "local_score_w16": float(local_scores_w16.get(pos, float("nan"))),
                                "local_score_w32": float(local_scores_w32.get(pos, float("nan"))),
                                "baseline_nll": float(rec["nll"]),
                            }
                        )

            if kernel_accum is not None and capture is not None and (not phase2a_nonbaseline):
                if bool(config.track_a_enabled):
                    kernel_accum.update_track_a_raw(capture)
                else:
                    kernel_accum.update_track_b_raw_from_qk(capture)
                if config.centered_compute == "inline" and config.kernel_centered_mode == "shared_mean":
                    kernel_accum.accumulate_shared_means(capture)
                elif config.centered_compute == "inline":
                    kernel_accum.update_centered(capture)
            del capture
            del batch_logits

        # Pass 2: centered kernel for shared-mean mode.
        if (
            kernel_accum is not None
            and (not phase2a_nonbaseline)
            and config.centered_compute == "inline"
            and config.kernel_centered_mode == "shared_mean"
        ):
            shared_q, shared_k = kernel_accum.finalize_shared_means()
            for batch in batches:
                input_ids, attention_mask = _prepare_batch_tensors(batch, config.device)
                capture = adapter.capture(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    include_logits=False,
                    return_token_logits=False,
                    output_device=config.device,
                )
                kernel_accum.update_centered(capture, shared_q_mean=shared_q, shared_k_mean=shared_k)
                del capture

    if is_task_metrics_only:
        with intervention_cm:
            bs = effective_batch_size
            while True:
                try:
                    _run_cell_passes(bs)
                    effective_batch_size = bs
                    state.batch_size_memory[batch_memory_key] = int(bs)
                    break
                except torch.cuda.OutOfMemoryError:
                    if not config.device.startswith("cuda") or bs <= 1:
                        raise
                    bs = max(1, bs // 2)
                    LOGGER.warning("OOM in cell; retrying with reduced batch_size=%s row=%s", bs, row)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                except RuntimeError as exc:
                    if ("CUDA out of memory" not in str(exc)) or (not config.device.startswith("cuda")) or bs <= 1:
                        raise
                    bs = max(1, bs // 2)
                    LOGGER.warning("Runtime OOM in cell; retrying with reduced batch_size=%s row=%s", bs, row)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
    else:
        with intervention_cm:
            adapter.register(model)
            try:
                bs = effective_batch_size
                while True:
                    try:
                        _run_cell_passes(bs)
                        effective_batch_size = bs
                        state.batch_size_memory[batch_memory_key] = int(bs)
                        break
                    except torch.cuda.OutOfMemoryError:
                        if not config.device.startswith("cuda") or bs <= 1:
                            raise
                        bs = max(1, bs // 2)
                        LOGGER.warning("OOM in cell; retrying with reduced batch_size=%s row=%s", bs, row)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                    except RuntimeError as exc:
                        if ("CUDA out of memory" not in str(exc)) or (not config.device.startswith("cuda")) or bs <= 1:
                            raise
                        bs = max(1, bs // 2)
                        LOGGER.warning("Runtime OOM in cell; retrying with reduced batch_size=%s row=%s", bs, row)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()

                if isinstance(intervention_cm, RopeQKInterventionContext):
                    audit = intervention_cm.audit()
                elif isinstance(intervention_cm, AbsPEDCTInterventionContext):
                    audit = intervention_cm.audit()
            finally:
                adapter.cleanup()
    if config.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    task_row = {
        "phase": row["phase"],
        "split": row["split"],
        "model": row["model"],
        "dataset": row.get("dataset"),
        "task": row["task"],
        "span": row.get("span"),
        "seq_len": row["seq_len"],
        "intervention": row["intervention"],
        "random_draw": row.get("random_draw"),
        "seed": row["seed"],
        "num_sequences": len(examples),
        "effective_batch_size": int(effective_batch_size),
        "num_targets": all_targets,
        "mean_accuracy": float(sum(all_acc) / max(len(all_acc), 1)),
        "mean_nll": float(sum(all_nll) / max(len(all_nll), 1)),
        "mean_accuracy_full_vocab": float(sum(all_acc_full_vocab) / max(len(all_acc_full_vocab), 1)),
        "mean_nll_full_vocab": float(sum(all_nll_full_vocab) / max(len(all_nll_full_vocab), 1)),
        "mean_accuracy_restricted": (
            float(sum(all_acc_restricted) / max(len(all_acc_restricted), 1)) if all_acc_restricted else None
        ),
        "mean_nll_restricted": (
            float(sum(all_nll_restricted) / max(len(all_nll_restricted), 1)) if all_nll_restricted else None
        ),
        "eval_mode": (
            "full_vocab"
            if str(row.get("split")) == "tier1_ppl"
            else str(config.synthetic_eval_mode)
        ),
        "candidate_count": (
            None
            if str(row.get("split")) == "tier1_ppl" or str(config.synthetic_eval_mode) == "full_vocab"
            else int(config.candidate_size)
        ),
        "chance_accuracy": (
            None
            if str(row.get("split")) == "tier1_ppl" or str(config.synthetic_eval_mode) == "full_vocab"
            else float(1.0 / max(int(config.candidate_size), 1))
        ),
        "candidate_policy_version": (
            "full_vocab"
            if str(row.get("split")) == "tier1_ppl" or str(config.synthetic_eval_mode) == "full_vocab"
            else str(config.candidate_policy_version)
        ),
        "baseline_accuracy": float(floor_decision.baseline_accuracy),
        "fallback_applied": bool(floor_decision.fallback_applied),
        "floor_limited": bool(floor_decision.floor_limited),
        "floor_rule_version": floor_decision.rule_version,
        "norm_drift_pass": bool(audit.norm_drift_pass) if audit.norm_drift_pass is not None else True,
        "plan_target": audit.plan.target,
        "plan_severity": audit.plan.severity,
        "plan_random_draw": audit.plan.random_draw,
        "overlap_high_fraction": float(audit.plan.overlap_high_fraction),
        "overlap_low_fraction": float(audit.plan.overlap_low_fraction),
        "overlap_high_jaccard": float(audit.plan.overlap_high_jaccard),
        "overlap_low_jaccard": float(audit.plan.overlap_low_jaccard),
        "mean_no_match_rate": float(sum(no_match_rates) / max(len(no_match_rates), 1)) if no_match_rates else 0.0,
        "no_match_examples": int(no_match_examples),
        "no_match_targets": int(no_match_targets),
        "row_hash": row_hash(row),
        "phase2a_poc": bool(row.get("phase") == "phase2a"),
    }
    task_metrics = pd.DataFrame([task_row])

    if kernel_accum is not None:
        a_df, b_raw_df, b_cent_df = kernel_accum.to_dataframes(
            model=row["model"],
            phase=row["phase"],
            split=row["split"],
            dataset=row.get("dataset"),
            task=row["task"],
            span=row.get("span"),
            intervention=row["intervention"],
            random_draw=row.get("random_draw"),
            seed=row["seed"],
            seq_len=row["seq_len"],
        )
        if config.centered_compute == "defer":
            b_cent_df = _empty_kernel_summary_df()
    else:
        a_df = _empty_kernel_summary_df()
        b_raw_df = _empty_kernel_summary_df()
        b_cent_df = _empty_kernel_summary_df()

    run_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "row": row,
        "device": config.device,
        "kernel_engine": config.kernel_engine,
        "kernel_centered_mode": config.kernel_centered_mode,
        "centered_compute": config.centered_compute,
        "centered_pending": bool((kernel_accum is not None) and config.centered_compute == "defer"),
        "feasibility_task_only": bool(is_feasibility_task_only),
        "execution_status": "executed",
        "skip_reason": None,
        "floor_prune_enabled": bool(config.prune_floor_limited_interventions),
        "implementation": "experiment2.execution.v8_floor_prune",
        "synthetic_count_used": int(config.synthetic_count),
        "tier1_count_used": int(config.tier1_count),
        "synthetic_eval_mode": str(config.synthetic_eval_mode),
        "h12_endpoint_policy": str(config.h12_endpoint_policy),
        "track_a_enabled": bool(config.track_a_enabled),
        "intervention_profile": str(config.intervention_profile),
        "optionc_exploratory_core": bool(config.optionc_exploratory_core),
        "candidate_size": int(config.candidate_size),
        "candidate_policy_version": str(config.candidate_policy_version),
        "floor_threshold": float(config.floor_threshold),
        "allow_phase2a_reuse": bool(config.allow_phase2a_reuse),
        "model_allowlist": [str(x) for x in config.model_allowlist],
        "long_offset_lock_hash": long_offset_bundle.lock_hash,
        "long_offsets_used": [int(x) for x in long_offset_bundle.long_offsets],
        "fallback_spans_used": [int(x) for x in long_offset_bundle.fallback_spans],
        "effective_batch_size": int(effective_batch_size),
        "batch_size_synth_config": int(config.batch_size_synth),
        "batch_size_tier1_config": int(config.batch_size_tier1),
        "example_cache_enabled": bool(config.enable_example_cache),
        "phase2a_efficiency_profile": (
            "capture_pruned_nonbaseline_count50"
            if str(row.get("phase")) == "phase2a"
            else None
        ),
        "phase2b_confirmatory_collection_unchanged": bool(str(row.get("phase")) == "phase2b"),
        **prov,
        "floor": floor_decision.to_json_dict(),
    }
    run_config_path.write_text(json.dumps(run_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    task_metrics.to_parquet(out_dir / "task_metrics.parquet", engine="pyarrow", index=False)
    a_df.to_parquet(out_dir / "kernel_track_a_summary.parquet", engine="pyarrow", index=False)
    b_raw_df.to_parquet(out_dir / "kernel_track_b_raw_summary.parquet", engine="pyarrow", index=False)
    b_cent_df.to_parquet(out_dir / "kernel_track_b_centered_summary.parquet", engine="pyarrow", index=False)
    (out_dir / "intervention_audit.json").write_text(json.dumps(audit.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if row.get("split") == "tier1_ppl":
        _write_tier1_artifacts(
            row=row,
            config=config,
            out_dir=out_dir,
            per_position_rows=per_pos_rows,
            baseline_bin_rows=baseline_bin_rows,
        )

    return True


def _floor_decision_path(output_root: Path, row: dict[str, Any]) -> Path:
    run_id = str(row.get("run_id") or "unknown_run")
    return output_root / row["phase"] / run_id / "floor_decisions" / f"{floor_key(row)}.json"


def _resolve_floor_decision(
    *,
    row: dict[str, Any],
    config: ExecutionConfig,
    model,
    model_spec,
    adapter,
    pools,
    state: _RunState,
) -> FloorDecision:
    if row.get("split") not in {"synthetic", "span_bridge", "mechanistic"}:
        return FloorDecision(
            key=floor_key(row),
            baseline_accuracy=1.0,
            fallback_accuracy=None,
            fallback_applied=False,
            floor_limited=False,
            fallback_spans=tuple(),
            scoped_noop_baseline_accuracy=None,
            scoped_noop_delta_vs_global=None,
            scoped_noop_consistent=None,
        )

    key = floor_key(row)
    if key in state.floor_cache:
        return state.floor_cache[key]

    path = _floor_decision_path(config.output_root, row)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        dec = FloorDecision(
            key=payload["key"],
            baseline_accuracy=float(payload["baseline_accuracy"]),
            fallback_accuracy=(None if payload.get("fallback_accuracy") is None else float(payload["fallback_accuracy"])),
            fallback_applied=bool(payload.get("fallback_applied", False)),
            floor_limited=bool(payload.get("floor_limited", False)),
            fallback_spans=tuple(int(x) for x in payload.get("fallback_spans", [])),
            floor_baseline_scope_policy=str(payload.get("floor_baseline_scope_policy", "global_task_feasibility")),
            scoped_noop_baseline_accuracy=(
                None
                if (
                    payload.get("scoped_noop_baseline_accuracy") is None
                    and payload.get("scoped_baseline_accuracy") is None
                )
                else float(payload.get("scoped_noop_baseline_accuracy", payload.get("scoped_baseline_accuracy")))
            ),
            scoped_noop_delta_vs_global=(
                None if payload.get("scoped_noop_delta_vs_global") is None else float(payload.get("scoped_noop_delta_vs_global"))
            ),
            scoped_noop_consistent=(
                None
                if (
                    payload.get("scoped_noop_consistent") is None
                    and payload.get("scoped_floor_limited") is None
                )
                else bool(payload.get("scoped_noop_consistent", True))
            ),
            threshold=float(payload.get("threshold", config.floor_threshold)),
            rule_version=str(payload.get("rule_version", "v1")),
        )
        state.floor_cache[key] = dec
        return dec

    baseline_acc = _quick_baseline_accuracy(
        row=row,
        config=config,
        model=model,
        adapter=adapter,
        pools=pools,
        span_choices=tuple(),
    )
    fallback_spans = fallback_spans_for_task(str(row.get("task")), int(row.get("seq_len", 0)))
    fallback_acc = None
    fallback_applied = False
    floor_limited = baseline_acc < config.floor_threshold
    if floor_limited and fallback_spans:
        fallback_applied = True
        fallback_acc = _quick_baseline_accuracy(
            row=row,
            config=config,
            model=model,
            adapter=adapter,
            pools=pools,
            span_choices=fallback_spans,
        )
        floor_limited = fallback_acc < config.floor_threshold

    scoped_noop_baseline_accuracy: float | None = None
    scoped_noop_delta_vs_global: float | None = None
    scoped_noop_consistent: bool | None = None
    if (
        row.get("phase") == "phase2d"
        and row.get("scope")
        and row.get("head_group")
        and getattr(model_spec, "pe_scheme", None) == "RoPE"
    ):
        scoped_rows = dict(row)
        scoped_rows["intervention"] = "none"
        target_scope = str(row.get("scope"))
        target_group = str(row.get("head_group"))
        target_map = load_target_heads(
            results_root=config.output_root.parent,
            run_id=str(row.get("run_id", "")),
            model=model_spec.name,
            scope=target_scope,
            head_group=target_group,
        )
        head_dim = int(model.config.hidden_size // model.config.num_attention_heads)
        no_op_plan = InterventionPlan(
            name="scope_baseline_noop",
            kind="rope_qk",
            severity="none",
            target="none",
            removed_indices=tuple(),
            reference_indices_high=tuple(),
            reference_indices_low=tuple(),
            overlap_high_count=0,
            overlap_low_count=0,
            overlap_high_fraction=0.0,
            overlap_low_fraction=0.0,
            overlap_high_jaccard=0.0,
            overlap_low_jaccard=0.0,
            nested_medium_in_strong=True,
            band_size=0,
            component_count=max(head_dim // 2, 0),
        )
        scoped_cm = RopeQKInterventionContext(
            model,
            no_op_plan,
            target_query_heads_by_layer=target_map,
            target_scope=target_scope,
            target_head_group=target_group,
        )
        scoped_span_choices = fallback_spans if fallback_applied else tuple()
        scoped_noop_baseline_accuracy = _quick_baseline_accuracy(
            row=scoped_rows,
            config=config,
            model=model,
            adapter=adapter,
            pools=pools,
            span_choices=scoped_span_choices,
            intervention_cm=scoped_cm,
        )
        scoped_noop_delta_vs_global = float(scoped_noop_baseline_accuracy - baseline_acc)
        scoped_noop_consistent = bool(abs(scoped_noop_delta_vs_global) <= 1e-6)

    dec = FloorDecision(
        key=key,
        baseline_accuracy=float(baseline_acc),
        fallback_accuracy=(None if fallback_acc is None else float(fallback_acc)),
        fallback_applied=fallback_applied,
        floor_limited=floor_limited,
        fallback_spans=fallback_spans if fallback_applied else tuple(),
        floor_baseline_scope_policy="global_task_feasibility",
        scoped_noop_baseline_accuracy=scoped_noop_baseline_accuracy,
        scoped_noop_delta_vs_global=scoped_noop_delta_vs_global,
        scoped_noop_consistent=scoped_noop_consistent,
        threshold=float(config.floor_threshold),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dec.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    state.floor_cache[key] = dec
    return dec


def _quick_baseline_accuracy(
    *,
    row: dict[str, Any],
    config: ExecutionConfig,
    model,
    adapter,
    pools,
    span_choices: tuple[int, ...],
    intervention_cm: Any | None = None,
) -> float:
    examples = _build_examples(row, config=config, pools=pools, span_choices=span_choices)
    all_acc: list[float] = []
    batch_size = _batch_size_for_row(row, config)
    cm = intervention_cm if intervention_cm is not None else nullcontext()
    with cm:
        adapter.register(model)
        try:
            bs = int(batch_size)
            while True:
                try:
                    all_acc = []
                    with torch.no_grad():
                        for batch in _iter_example_batches(examples, bs):
                            inp, mask = _prepare_batch_tensors(batch, config.device)
                            capture = adapter.capture(
                                model,
                                input_ids=inp,
                                attention_mask=mask,
                                include_logits=False,
                                return_token_logits=True,
                                capture_attention=False,
                                output_device=config.device,
                            )
                            for local_idx, ex in enumerate(batch):
                                token_logits = _token_logits_for_index(capture.token_logits, local_idx)
                                acc = _evaluate_example_accuracy_only(
                                    token_logits,
                                    ex,
                                    split=str(row.get("split")),
                                    pools=pools,
                                    synthetic_eval_mode=str(config.synthetic_eval_mode),
                                    candidate_size=int(config.candidate_size),
                                    candidate_policy_version=str(config.candidate_policy_version),
                                )
                                all_acc.append(acc)
                    return float(sum(all_acc) / max(len(all_acc), 1))
                except torch.cuda.OutOfMemoryError:
                    if not config.device.startswith("cuda") or bs <= 1:
                        raise
                    bs = max(1, bs // 2)
                    LOGGER.warning("OOM in quick baseline; retrying with reduced batch_size=%s row=%s", bs, row)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                except RuntimeError as exc:
                    if ("CUDA out of memory" not in str(exc)) or (not config.device.startswith("cuda")) or bs <= 1:
                        raise
                    bs = max(1, bs // 2)
                    LOGGER.warning("Runtime OOM in quick baseline; retrying with reduced batch_size=%s row=%s", bs, row)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
        finally:
            adapter.cleanup()


def _evaluate_example_accuracy_only(
    token_logits: torch.Tensor | None,
    example: TaskExample,
    *,
    split: str,
    pools,
    synthetic_eval_mode: str,
    candidate_size: int,
    candidate_policy_version: str,
) -> float:
    if token_logits is None:
        raise RuntimeError(f"Missing token logits for example {example.id}.")
    if token_logits.ndim != 2:
        raise RuntimeError(f"Expected [seq, vocab] token logits, got shape={tuple(token_logits.shape)}")

    eval_mode = _resolve_eval_mode(split=str(split), synthetic_eval_mode=str(synthetic_eval_mode))
    use_restricted = eval_mode == "restricted"
    sequence_value_tokens: list[int] | None = None
    if use_restricted and str(example.task_name) in {"local_key_match", "long_range_retrieval", "retrieval_bridge"}:
        values_set = {int(v) for v in pools.values}
        sequence_value_tokens = sorted({int(tok) for tok in example.tokens if int(tok) in values_set})

    total = 0
    correct = 0
    for pos, target in zip(example.target_positions, example.target_tokens):
        if pos <= 0 or pos >= token_logits.shape[0]:
            continue
        target_int = int(target)
        pred_vec = token_logits[pos - 1]
        if use_restricted:
            candidate_ids = _build_restricted_candidates(
                example=example,
                target=target_int,
                position=int(pos),
                pools=pools,
                candidate_size=int(candidate_size),
                candidate_policy_version=str(candidate_policy_version),
                sequence_value_tokens=sequence_value_tokens,
            )
            cand_tensor = torch.tensor(candidate_ids, dtype=torch.long, device=pred_vec.device)
            best_idx = int(torch.argmax(pred_vec.index_select(0, cand_tensor)).item())
            pred_id = int(candidate_ids[best_idx])
        else:
            pred_id = int(torch.argmax(pred_vec).item())
        correct += int(pred_id == target_int)
        total += 1
    return float(correct / max(total, 1))


def _build_examples(
    row: dict[str, Any],
    config: ExecutionConfig,
    pools,
    span_choices: tuple[int, ...] = tuple(),
) -> list[TaskExample]:
    if row["split"] in {"synthetic", "span_bridge", "mechanistic"}:
        span_override = int(row["span"]) if row.get("span") is not None else None
        return generate_task_examples(
            task_name=row["task"],
            model_name=row["model"],
            seq_len=int(row["seq_len"]),
            seed=int(row["seed"]),
            count=config.synthetic_count,
            pools=pools,
            span_override=span_override,
            span_choices=span_choices or None,
        )
    if row["split"] == "tier1_ppl":
        return _load_tier1_examples(
            model_name=row["model"],
            dataset_name=row["dataset"],
            seq_len=int(row["seq_len"]),
            seed=int(row["seed"]),
            count=config.tier1_count,
        )
    raise ValueError(f"Unsupported split for execution: {row['split']}")


def _load_or_build_examples(
    *,
    row: dict[str, Any],
    config: ExecutionConfig,
    pools,
    span_choices: tuple[int, ...],
    generator_hash_value: str,
) -> list[TaskExample]:
    split = str(row.get("split"))
    if split not in {"synthetic", "span_bridge", "mechanistic"} or not config.enable_example_cache:
        return _build_examples(row, config=config, pools=pools, span_choices=span_choices)

    cache_root = config.output_root / "cache" / "synthetic_examples"
    key = make_cache_key(
        generator_hash=generator_hash_value,
        model=str(row["model"]),
        task=str(row["task"]),
        seq_len=int(row["seq_len"]),
        seed=int(row["seed"]),
        span=(None if row.get("span") is None else int(row.get("span"))),
        synthetic_count=int(config.synthetic_count),
        fallback_spans=tuple(int(x) for x in span_choices),
    )
    cached = load_cached_examples(cache_root, key)
    if cached is not None:
        return cached
    built = _build_examples(row, config=config, pools=pools, span_choices=span_choices)
    write_cached_examples(
        cache_root=cache_root,
        key=key,
        examples=built,
        metadata={
            "generator_hash": generator_hash_value,
            "model": str(row["model"]),
            "task": str(row["task"]),
            "seq_len": int(row["seq_len"]),
            "seed": int(row["seed"]),
            "span": (None if row.get("span") is None else int(row.get("span"))),
            "synthetic_count": int(config.synthetic_count),
            "fallback_spans": [int(x) for x in span_choices],
        },
    )
    return built


def _tier1_bins_path(config: ExecutionConfig, row: dict[str, Any]) -> Path:
    run_id = str(row.get("run_id") or "unknown_run")
    phase_root = config.output_root / row["phase"] / run_id
    name = f"{row['model']}__{row['dataset']}__len_{row['seq_len']}__seed_{row['seed']}"
    return phase_root / "tier1_bins" / f"{name}.parquet"


def _write_tier1_artifacts(
    *,
    row: dict[str, Any],
    config: ExecutionConfig,
    out_dir: Path,
    per_position_rows: list[dict[str, Any]],
    baseline_bin_rows: list[dict[str, Any]],
) -> None:
    per_df = pd.DataFrame(per_position_rows)
    bins_path = _tier1_bins_path(config, row)

    if row.get("intervention") == "none":
        base_df = pd.DataFrame(baseline_bin_rows)
        summary = build_frozen_bins(base_df, out_path=bins_path)
        bins = load_frozen_bins(bins_path)
    else:
        bins = load_frozen_bins(bins_path)
        summary = summarize_frozen_bins(bins)

    table = stratified_metrics(per_df, bins)
    dep_delta = dependency_delta(table)
    table = table.assign(
        phase=row["phase"],
        model=row["model"],
        dataset=row["dataset"],
        seq_len=row["seq_len"],
        intervention=row["intervention"],
        random_draw=row.get("random_draw"),
        seed=row["seed"],
        h5_interpretable=bool(summary.interpretable),
        bin_separation_gap=float(summary.separation_gap),
        binning_window_agreement=float(summary.window_agreement),
        binning_robust_pass=bool(summary.robustness_pass),
        dependency_delta=float(dep_delta),
    )
    table.to_parquet(out_dir / "tier1_stratified_metrics.parquet", engine="pyarrow", index=False)


def _load_tier1_examples(
    *,
    model_name: str,
    dataset_name: str,
    seq_len: int,
    seed: int,
    count: int,
) -> list[TaskExample]:
    model_spec = MODEL_BY_NAME[model_name]
    dataset_spec = DATASET_BY_NAME[dataset_name]
    data_root = Experiment2Paths.default().data_root
    jsonl = tokenized_path(data_root, model_spec, dataset_spec, SequenceLengthSpec(tokens=seq_len))
    if not jsonl.exists():
        raise FileNotFoundError(f"Tier1 source missing tokenized file: {jsonl}")
    eval_rows: list[dict[str, Any]] = []
    with jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") != "eval":
                continue
            eval_rows.append(row)
    if len(eval_rows) < count:
        raise RuntimeError(
            f"Tier1 examples short for {model_name}/{dataset_name}: got {len(eval_rows)} expected {count}"
        )

    rng = random.Random(f"tier1|{model_name}|{dataset_name}|{seq_len}|{seed}")
    selected_idx = sorted(rng.sample(range(len(eval_rows)), k=count))
    examples: list[TaskExample] = []
    for idx in selected_idx:
        row = eval_rows[idx]
        rec = TrackASequenceRecord(seq_id=row["id"], tokens=row["tokens"], split=row["split"])
        target_positions = list(range(1, len(rec.tokens)))
        target_tokens = [rec.tokens[pos] for pos in target_positions]
        examples.append(
            TaskExample(
                id=f"{model_name}:{dataset_name}:{seed}:{rec.seq_id}:tier1",
                task_name="tier1_stratified_ppl",
                tokens=rec.tokens,
                target_positions=target_positions,
                target_tokens=target_tokens,
                dependency_span=0,
                task_class="tier1",
                seed=seed,
                model=model_name,
                length=seq_len,
                task_params={"dataset": dataset_name},
                pair_count=None,
                query_key=None,
                distractor_key=None,
                match_rule="all_positions",
                has_no_match=False,
            )
        )
    return examples


def _build_intervention(
    model,
    model_spec,
    row: dict[str, Any],
    *,
    results_root: Path,
    run_id: str,
    cache: dict[tuple[Any, ...], dict[str, Any]] | None = None,
) -> tuple[InterventionPlan, Any]:
    intervention = row["intervention"]
    seed = int(row["seed"])
    random_draw = None if row.get("random_draw") is None else int(row.get("random_draw"))
    if intervention == "none":
        plan = InterventionPlan(
            name="none",
            kind="none",
            severity="none",
            target="none",
            removed_indices=tuple(),
            reference_indices_high=tuple(),
            reference_indices_low=tuple(),
            overlap_high_count=0,
            overlap_low_count=0,
            overlap_high_fraction=0.0,
            overlap_low_fraction=0.0,
            overlap_high_jaccard=0.0,
            overlap_low_jaccard=0.0,
            nested_medium_in_strong=True,
            band_size=0,
            component_count=0,
        )
        return plan, nullcontext()
    target_scope = str(row.get("scope")) if row.get("scope") else None
    target_group = str(row.get("head_group")) if row.get("head_group") else None
    phase_name = str(row.get("phase")) if row.get("phase") else None
    cache_key: tuple[Any, ...] | None = None
    cached_payload: dict[str, Any] | None = None
    if cache is not None:
        cache_key = (
            str(model_spec.name),
            str(model_spec.pe_scheme),
            str(intervention),
            int(seed),
            random_draw,
            phase_name,
            target_scope,
            target_group,
            str(run_id),
        )
        cached_payload = cache.get(cache_key)
    if model_spec.pe_scheme == "RoPE":
        if cached_payload is not None:
            plan = cached_payload["plan"]
            target_map = cached_payload.get("target_map")
            target_scope = cached_payload.get("target_scope")
            target_group = cached_payload.get("target_group")
        else:
            head_dim = model.config.hidden_size // model.config.num_attention_heads
            plan = build_rope_intervention_plan(
                head_dim=head_dim,
                intervention=intervention,
                seed=seed,
                random_draw=random_draw,
            )
            target_map = None
            if phase_name == "phase2d" and target_scope and target_group:
                target_map = load_target_heads(
                    results_root=results_root,
                    run_id=run_id,
                    model=model_spec.name,
                    scope=target_scope,
                    head_group=target_group,
                )
            if cache is not None and cache_key is not None:
                cache[cache_key] = {
                    "plan": plan,
                    "target_map": target_map,
                    "target_scope": target_scope,
                    "target_group": target_group,
                }
        return plan, RopeQKInterventionContext(
            model,
            plan,
            target_query_heads_by_layer=target_map,
            target_scope=target_scope,
            target_head_group=target_group,
        )
    if model_spec.pe_scheme == "learned-absolute":
        if cached_payload is not None:
            plan = cached_payload["plan"]
        else:
            n_positions = int(getattr(model.transformer.wpe.weight, "shape")[0])
            plan = build_abspe_dct_plan(
                n_positions=n_positions,
                intervention=intervention,
                seed=seed,
                random_draw=random_draw,
            )
            if cache is not None and cache_key is not None:
                cache[cache_key] = {"plan": plan}
        return plan, AbsPEDCTInterventionContext(model, plan)
    raise ValueError(
        f"Intervention {intervention} requested for non-intervenable model {model_spec.name} ({model_spec.pe_scheme})."
    )


def _stable_int_seed(*parts: object) -> int:
    text = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _resolve_eval_mode(*, split: str, synthetic_eval_mode: str) -> str:
    if split in {"synthetic", "span_bridge", "mechanistic"} and synthetic_eval_mode == "restricted":
        return "restricted"
    return "full_vocab"


def _task_structured_distractors(
    example: TaskExample,
    *,
    target: int,
    position: int,
    pools,
    sequence_value_tokens: list[int] | None = None,
) -> list[int]:
    # TODO(experiment2-efficiency): Memoization for this helper is intentionally deferred.
    # Candidate construction has been measured as a minor cost versus model forward passes.
    task_name = str(example.task_name)
    if task_name in {"local_copy_offset", "delayed_copy", "copy_offset_bridge"}:
        out: list[int] = []
        offset_raw = example.task_params.get("offset")
        if isinstance(offset_raw, int):
            for tgt_pos in example.target_positions:
                src_pos = int(tgt_pos) - int(offset_raw)
                if 0 <= src_pos < len(example.tokens):
                    tok = int(example.tokens[src_pos])
                    if tok != int(target):
                        out.append(tok)
            for tok in example.target_tokens:
                tok_i = int(tok)
                if tok_i != int(target):
                    out.append(tok_i)
            # Include local context around the current query position.
            for delta in (-1, 1, -2, 2):
                p = int(position) + delta
                if 0 <= p < len(example.tokens):
                    tok = int(example.tokens[p])
                    if tok != int(target):
                        out.append(tok)
        return out
    if task_name == "local_key_match":
        values_src = sequence_value_tokens or []
        values = [int(tok) for tok in values_src if int(tok) != int(target)]
        out: list[int] = []
        if int(pools.no_match_token) != int(target):
            out.append(int(pools.no_match_token))
        out.extend(values)
        return out
    if task_name in {"long_range_retrieval", "retrieval_bridge"}:
        values_src = sequence_value_tokens or []
        return [int(tok) for tok in values_src if int(tok) != int(target)]
    return []


def _build_restricted_candidates(
    *,
    example: TaskExample,
    target: int,
    position: int,
    pools,
    candidate_size: int,
    candidate_policy_version: str,
    sequence_value_tokens: list[int] | None = None,
) -> tuple[int, ...]:
    target_int = int(target)
    if candidate_size < 2:
        raise RuntimeError(f"candidate_size must be >=2, got {candidate_size}")
    selected: list[int] = [target_int]
    seen = {target_int}

    structured = _task_structured_distractors(
        example,
        target=target_int,
        position=position,
        pools=pools,
        sequence_value_tokens=sequence_value_tokens,
    )
    for token in structured:
        tok = int(token)
        if tok in seen:
            continue
        selected.append(tok)
        seen.add(tok)
        if len(selected) >= candidate_size:
            return tuple(selected[:candidate_size])

    rng = random.Random(
        _stable_int_seed(
            "restricted_candidates",
            candidate_policy_version,
            example.id,
            example.task_name,
            position,
            target_int,
            candidate_size,
        )
    )
    filler_pool = pools.filler
    attempts = 0
    max_attempts = max(len(filler_pool) * 3, 256)
    while len(selected) < candidate_size and attempts < max_attempts:
        tok = int(filler_pool[rng.randrange(len(filler_pool))])
        attempts += 1
        if tok in seen:
            continue
        selected.append(tok)
        seen.add(tok)
    if len(selected) < candidate_size:
        for tok in filler_pool:
            token = int(tok)
            if token in seen:
                continue
            selected.append(token)
            seen.add(token)
            if len(selected) >= candidate_size:
                break
    if len(selected) < candidate_size:
        raise RuntimeError(
            f"Unable to build {candidate_size}-way candidates for {example.id} position={position} "
            f"under policy={candidate_policy_version}; only {len(selected)} unique tokens available."
        )
    return tuple(selected[:candidate_size])


def _evaluate_example_from_token_logits(
    token_logits: torch.Tensor | None,
    example: TaskExample,
    *,
    split: str,
    pools,
    synthetic_eval_mode: str,
    candidate_size: int,
    candidate_policy_version: str,
) -> tuple[dict[str, float | int | str | None], list[dict[str, Any]]]:
    if token_logits is None:
        raise RuntimeError(f"Missing token logits for example {example.id}.")
    logits = token_logits
    if logits.ndim != 2:
        raise RuntimeError(f"Expected [seq, vocab] token logits, got shape={tuple(logits.shape)}")

    eval_mode = _resolve_eval_mode(split=str(split), synthetic_eval_mode=str(synthetic_eval_mode))
    use_restricted = eval_mode == "restricted"
    log_probs = torch.log_softmax(logits, dim=-1)
    sequence_value_tokens: list[int] | None = None
    if use_restricted and str(example.task_name) in {"local_key_match", "long_range_retrieval", "retrieval_bridge"}:
        values_set = {int(v) for v in pools.values}
        sequence_value_tokens = sorted({int(tok) for tok in example.tokens if int(tok) in values_set})
    total = 0
    correct = 0
    nll_total = 0.0
    full_correct = 0
    full_nll_total = 0.0
    restricted_correct = 0
    restricted_nll_total = 0.0
    per_pos: list[dict[str, Any]] = []
    for pos, target in zip(example.target_positions, example.target_tokens):
        if pos <= 0 or pos >= logits.shape[0]:
            continue
        target_int = int(target)
        pred_vec = logits[pos - 1]
        full_pred_id = int(torch.argmax(pred_vec).item())
        full_corr = int(full_pred_id == target_int)
        full_nll = -float(log_probs[pos - 1, target_int].item())
        full_correct += full_corr
        full_nll_total += full_nll

        if use_restricted:
            candidate_ids = _build_restricted_candidates(
                example=example,
                target=target_int,
                position=int(pos),
                pools=pools,
                candidate_size=int(candidate_size),
                candidate_policy_version=str(candidate_policy_version),
                sequence_value_tokens=sequence_value_tokens,
            )
            cand_tensor = torch.tensor(candidate_ids, dtype=torch.long, device=pred_vec.device)
            cand_logits = pred_vec.index_select(0, cand_tensor)
            cand_log_probs = torch.log_softmax(cand_logits, dim=-1)
            best_idx = int(torch.argmax(cand_logits).item())
            pred_id = int(candidate_ids[best_idx])
            corr = int(pred_id == target_int)
            target_local_idx = candidate_ids.index(target_int)
            nll = -float(cand_log_probs[target_local_idx].item())
            chance = float(1.0 / len(candidate_ids))
            restricted_correct += corr
            restricted_nll_total += nll
            candidate_count = int(len(candidate_ids))
        else:
            pred_id = full_pred_id
            corr = full_corr
            nll = full_nll
            chance = None
            candidate_count = None

        correct += corr
        nll_total += nll
        total += 1
        per_pos.append(
            {
                "example_id": example.id,
                "position": int(pos),
                "target": target_int,
                "pred": pred_id,
                "correct": corr,
                "nll": nll,
                "eval_mode": eval_mode,
                "candidate_count": candidate_count,
                "chance_accuracy": chance,
                "pred_full_vocab": full_pred_id,
                "correct_full_vocab": full_corr,
                "nll_full_vocab": full_nll,
            }
        )
    if total == 0:
        raise RuntimeError(f"Example {example.id} produced zero valid target evaluations.")
    return {
        "accuracy": correct / total,
        "mean_nll": nll_total / total,
        "num_targets": total,
        "eval_mode": eval_mode,
        "candidate_count": (int(candidate_size) if use_restricted else None),
        "chance_accuracy": (float(1.0 / max(int(candidate_size), 1)) if use_restricted else None),
        "full_vocab_accuracy": full_correct / total,
        "full_vocab_mean_nll": full_nll_total / total,
        "restricted_accuracy": (restricted_correct / total if use_restricted else None),
        "restricted_mean_nll": (restricted_nll_total / total if use_restricted else None),
    }, per_pos


def _condition_dir(output_root: Path, row: dict[str, Any]) -> Path:
    run_id = row.get("run_id") or "unknown_run"
    parts = [output_root, row["phase"], run_id, row["model"]]
    if row.get("dataset"):
        parts.append(row["dataset"])
    parts.extend([row["task"], f"len_{row['seq_len']}"])
    if row.get("span") is not None:
        parts.append(f"span_{int(row['span'])}")
    parts.append(row["intervention"])
    if row.get("random_draw") is not None:
        parts.append(f"draw_{int(row['random_draw'])}")
    parts.append(f"seed_{row['seed']}")
    out = Path(parts[0])
    for p in parts[1:]:
        out = out / str(p)
    return out


def _infer_run_id_from_manifest(manifest_path: Path) -> str:
    # .../<phase>/<run_id>/manifest.jsonl
    if manifest_path.parent.name:
        return manifest_path.parent.name
    return "unknown_run"


def _copy_cell_artifacts(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in [
        "task_metrics.parquet",
        "kernel_track_a_summary.parquet",
        "kernel_track_b_raw_summary.parquet",
        "kernel_track_b_centered_summary.parquet",
        "intervention_audit.json",
    ]:
        src_path = src / name
        if src_path.exists():
            shutil.copy2(src_path, dst / name)


def _maybe_reuse_from_phase2a(
    *,
    row: dict[str, Any],
    out_dir: Path,
    config: ExecutionConfig,
    provenance: dict[str, str],
    state: _RunState,
) -> bool:
    if row.get("phase") != "phase2b":
        return False
    if row.get("split") != "synthetic":
        return False
    if row.get("model") != "llama-3.2-1b":
        return False
    if row.get("task") not in {"local_key_match", "delayed_copy", "long_range_retrieval"}:
        return False
    if int(row.get("seq_len", 0)) != 1024:
        return False
    if not bool(config.allow_phase2a_reuse):
        state.reuse_events.append(
            {
                "phase": "phase2b",
                "model": row.get("model"),
                "task": row.get("task"),
                "seed": row.get("seed"),
                "intervention": row.get("intervention"),
                "random_draw": row.get("random_draw"),
                "status": "rerun",
                "reason": "reuse_disabled_by_config",
            }
        )
        return False

    src_row = dict(row)
    src_row["phase"] = "phase2a"
    src_dir = _condition_dir(config.output_root, src_row)
    src_cfg = src_dir / "run_config.json"
    if not src_cfg.exists():
        state.reuse_events.append(
            {
                "phase": "phase2b",
                "model": row.get("model"),
                "task": row.get("task"),
                "seed": row.get("seed"),
                "intervention": row.get("intervention"),
                "status": "rerun",
                "reason": "missing_phase2a_source",
            }
        )
        return False

    src_payload = json.loads(src_cfg.read_text(encoding="utf-8"))
    src_prov = {
        "generator_hash": src_payload.get("generator_hash"),
        "intervention_hash": src_payload.get("intervention_hash"),
        "model_tokenizer_hash": src_payload.get("model_tokenizer_hash"),
    }
    config_mismatch: list[str] = []
    expected_str = {
        "synthetic_eval_mode": str(config.synthetic_eval_mode),
        "candidate_policy_version": str(config.candidate_policy_version),
    }
    for key, expected in expected_str.items():
        actual = str(src_payload.get(key, ""))
        if actual != expected:
            config_mismatch.append(key)
    expected_int = {
        "candidate_size": int(config.candidate_size),
    }
    for key, expected in expected_int.items():
        actual_raw = src_payload.get(key, None)
        try:
            actual = int(actual_raw)
        except Exception:
            actual = None
        if actual != expected:
            config_mismatch.append(key)
    floor_actual_raw = src_payload.get("floor_threshold", None)
    try:
        floor_actual = float(floor_actual_raw)
    except Exception:
        floor_actual = float("nan")
    if not (np.isfinite(floor_actual) and abs(floor_actual - float(config.floor_threshold)) <= 1e-12):
        config_mismatch.append("floor_threshold")
    expected_lock_hash = load_long_offset_bundle(strict=False).lock_hash
    actual_lock_hash = str(src_payload.get("long_offset_lock_hash", ""))
    if actual_lock_hash != expected_lock_hash:
        config_mismatch.append("long_offset_lock_hash")
    if config_mismatch:
        state.reuse_events.append(
            {
                "phase": "phase2b",
                "model": row.get("model"),
                "task": row.get("task"),
                "seed": row.get("seed"),
                "intervention": row.get("intervention"),
                "random_draw": row.get("random_draw"),
                "status": "rerun",
                "reason": "config_mismatch",
                "mismatch_fields": sorted(set(config_mismatch)),
                "source": {
                    "synthetic_eval_mode": src_payload.get("synthetic_eval_mode"),
                    "candidate_size": src_payload.get("candidate_size"),
                    "candidate_policy_version": src_payload.get("candidate_policy_version"),
                    "floor_threshold": src_payload.get("floor_threshold"),
                    "long_offset_lock_hash": src_payload.get("long_offset_lock_hash"),
                },
                "target": {
                    "synthetic_eval_mode": config.synthetic_eval_mode,
                    "candidate_size": config.candidate_size,
                    "candidate_policy_version": config.candidate_policy_version,
                    "floor_threshold": config.floor_threshold,
                    "long_offset_lock_hash": expected_lock_hash,
                },
            }
        )
        return False
    if not provenance_match(src_prov, provenance):
        state.reuse_events.append(
            {
                "phase": "phase2b",
                "model": row.get("model"),
                "task": row.get("task"),
                "seed": row.get("seed"),
                "intervention": row.get("intervention"),
                "status": "rerun",
                "reason": "provenance_mismatch",
                "source": src_prov,
                "target": provenance,
            }
        )
        return False

    _copy_cell_artifacts(src_dir, out_dir)
    centered_out = out_dir / "kernel_track_b_centered_summary.parquet"
    centered_pending = bool(config.centered_compute == "defer" and not centered_out.exists())
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "row": row,
        "device": config.device,
        "kernel_engine": config.kernel_engine,
        "kernel_centered_mode": config.kernel_centered_mode,
        "centered_compute": config.centered_compute,
        "centered_pending": centered_pending,
        "execution_status": "reused_phase2a",
        "skip_reason": None,
        "floor_prune_enabled": bool(config.prune_floor_limited_interventions),
        "implementation": "experiment2.execution.v8_floor_prune",
        "synthetic_count_used": int(src_payload.get("synthetic_count_used", config.synthetic_count)),
        "tier1_count_used": int(src_payload.get("tier1_count_used", config.tier1_count)),
        "synthetic_eval_mode": str(src_payload.get("synthetic_eval_mode", config.synthetic_eval_mode)),
        "h12_endpoint_policy": str(config.h12_endpoint_policy),
        "candidate_size": int(src_payload.get("candidate_size", config.candidate_size)),
        "candidate_policy_version": str(src_payload.get("candidate_policy_version", config.candidate_policy_version)),
        "floor_threshold": float(src_payload.get("floor_threshold", config.floor_threshold)),
        "allow_phase2a_reuse": bool(config.allow_phase2a_reuse),
        "model_allowlist": [str(x) for x in config.model_allowlist],
        "long_offset_lock_hash": str(src_payload.get("long_offset_lock_hash", load_long_offset_bundle(strict=False).lock_hash)),
        "long_offsets_used": list(src_payload.get("long_offsets_used", [])),
        "fallback_spans_used": list(src_payload.get("fallback_spans_used", [])),
        **provenance,
        "reused_from": str(src_dir),
    }
    (out_dir / "run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    state.reuse_events.append(
        {
            "phase": "phase2b",
            "model": row.get("model"),
            "task": row.get("task"),
            "seed": row.get("seed"),
            "intervention": row.get("intervention"),
            "status": "reused",
            "source": str(src_dir),
        }
    )
    return True


def _aggregate_phase_outputs(output_root: Path, *, phase: str, run_id: str, reuse_events: list[dict[str, Any]]) -> None:
    phase_root = output_root / phase / run_id
    if not phase_root.exists():
        return
    analysis_supported = phase in {"phase2a", "phase2b", "phase2c", "phase2d"}
    manifest_path = phase_root / "manifest.jsonl"
    manifest_rows = 0
    if manifest_path.exists():
        manifest_rows = sum(1 for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip())

    def _concat(pattern: str) -> pd.DataFrame:
        frames = [pd.read_parquet(p) for p in phase_root.glob(pattern)]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    task_df = _concat("**/task_metrics.parquet")
    if not task_df.empty:
        task_df.to_parquet(phase_root / "aggregate_task_metrics.parquet", engine="pyarrow", index=False)

    # Include tier1 stratified details if present.
    tier1_df = _concat("**/tier1_stratified_metrics.parquet")
    if not tier1_df.empty:
        tier1_df.to_parquet(phase_root / "aggregate_tier1_stratified_metrics.parquet", engine="pyarrow", index=False)

    a_df = _concat("**/kernel_track_a_summary.parquet")
    b_raw_df = _concat("**/kernel_track_b_raw_summary.parquet")
    b_cent_df = _concat("**/kernel_track_b_centered_summary.parquet")
    kernel_frames = []
    if not a_df.empty:
        kernel_frames.append(a_df.assign(metric="track_a"))
    if not b_raw_df.empty:
        kernel_frames.append(b_raw_df.assign(metric="track_b_raw"))
    if not b_cent_df.empty:
        kernel_frames.append(b_cent_df.assign(metric="track_b_centered"))
    if kernel_frames:
        pd.concat(kernel_frames, ignore_index=True).to_parquet(
            phase_root / "aggregate_kernel_metrics.parquet",
            engine="pyarrow",
            index=False,
        )

    pending_centered = 0
    run_config_rows = 0
    executed_rows = 0
    skipped_floor_limited_rows = 0
    endpoint_policy_counts: dict[str, int] = {}
    for cfg_path in phase_root.glob("**/run_config.json"):
        run_config_rows += 1
        try:
            payload = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        status = str(payload.get("execution_status", "executed"))
        if status == "skipped_floor_limited":
            skipped_floor_limited_rows += 1
        else:
            executed_rows += 1
        if bool(payload.get("centered_pending", False)):
            pending_centered += 1
        endpoint_policy = str(payload.get("h12_endpoint_policy", "raw_primary"))
        endpoint_policy_counts[endpoint_policy] = endpoint_policy_counts.get(endpoint_policy, 0) + 1

    if endpoint_policy_counts:
        selected_endpoint_policy = sorted(
            endpoint_policy_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]
    else:
        selected_endpoint_policy = "raw_primary"
    endpoint_policy_mixed = bool(len(endpoint_policy_counts) > 1)

    missing_rows = max(0, int(manifest_rows) - int(run_config_rows))
    coverage_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "phase": phase,
        "run_id": run_id,
        "manifest_rows": int(manifest_rows),
        "run_config_rows": int(run_config_rows),
        "executed_rows": int(executed_rows),
        "skipped_floor_limited_rows": int(skipped_floor_limited_rows),
        "missing_rows": int(missing_rows),
        "h12_endpoint_policy": str(selected_endpoint_policy),
        "h12_endpoint_policy_observed_counts": {str(k): int(v) for k, v in sorted(endpoint_policy_counts.items())},
        "h12_endpoint_policy_mixed": bool(endpoint_policy_mixed),
    }
    (phase_root / "execution_coverage.json").write_text(
        json.dumps(coverage_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if not analysis_supported:
        gate_payload = {
            "phase": phase,
            "status": "descriptive_only_non_protocol_phase",
            "note": "Analysis/gating pipeline is defined for phase2a-phase2d only.",
            "h12_endpoint_policy": str(selected_endpoint_policy),
            "h12_endpoint_policy_observed_counts": {str(k): int(v) for k, v in sorted(endpoint_policy_counts.items())},
            "h12_endpoint_policy_mixed": bool(endpoint_policy_mixed),
        }
        decision_payload = {
            "phase": phase,
            "run_id": run_id,
            "status": "descriptive_only_non_protocol_phase",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "h12_endpoint_policy": str(selected_endpoint_policy),
            "h12_endpoint_policy_observed_counts": {str(k): int(v) for k, v in sorted(endpoint_policy_counts.items())},
            "h12_endpoint_policy_mixed": bool(endpoint_policy_mixed),
        }
        (phase_root / "gate_evaluation.json").write_text(json.dumps(gate_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (phase_root / "decision_summary.json").write_text(
            json.dumps(decision_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif pending_centered > 0:
        gate_payload = {
            "phase": phase,
            "status": "blocked_centered_pending",
            "centered_pending_cells": int(pending_centered),
            "note": "Run has deferred centered kernels pending. Execute posthoc-centered before confirmatory analysis.",
            "h12_endpoint_policy": str(selected_endpoint_policy),
            "h12_endpoint_policy_observed_counts": {str(k): int(v) for k, v in sorted(endpoint_policy_counts.items())},
            "h12_endpoint_policy_mixed": bool(endpoint_policy_mixed),
        }
        decision_payload = {
            "phase": phase,
            "run_id": run_id,
            "status": "blocked_centered_pending",
            "centered_pending_cells": int(pending_centered),
            "h12_endpoint_policy": str(selected_endpoint_policy),
            "h12_endpoint_policy_observed_counts": {str(k): int(v) for k, v in sorted(endpoint_policy_counts.items())},
            "h12_endpoint_policy_mixed": bool(endpoint_policy_mixed),
        }
        (phase_root / "gate_evaluation.json").write_text(json.dumps(gate_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (phase_root / "decision_summary.json").write_text(
            json.dumps(decision_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        write_phase_analysis(phase_root)
        write_norm_family_decomposition(phase_root)
        write_h4_interaction_exploratory(phase_root)
        write_protocol_revision_log(phase_root)

    gate_path = phase_root / "gate_evaluation.json"
    if gate_path.exists():
        try:
            gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
            gate_payload["floor_pruned_rows_non_evaluable"] = int(skipped_floor_limited_rows)
            gate_payload.setdefault("h12_endpoint_policy", str(selected_endpoint_policy))
            gate_payload.setdefault(
                "h12_endpoint_policy_observed_counts",
                {str(k): int(v) for k, v in sorted(endpoint_policy_counts.items())},
            )
            gate_payload.setdefault("h12_endpoint_policy_mixed", bool(endpoint_policy_mixed))
            if skipped_floor_limited_rows > 0:
                gate_payload["floor_pruning_policy_note"] = (
                    "Rows skipped due to floor-limited non-baseline pruning are non-evaluable by confirmatory policy."
                )
            gate_path.write_text(json.dumps(gate_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except Exception:
            LOGGER.exception("Failed to annotate gate payload with floor-pruning metadata: %s", gate_path)
    write_claim_guard(phase_root)

    reuse_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "phase": phase,
        "run_id": run_id,
        "events": reuse_events,
        "counts": {
            "reused": int(sum(1 for e in reuse_events if e.get("status") == "reused")),
            "rerun": int(sum(1 for e in reuse_events if e.get("status") != "reused")),
        },
    }
    (phase_root / "reuse_decisions.json").write_text(json.dumps(reuse_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Ensure decision summary exists even for non-confirmatory phases.
    if not (phase_root / "decision_summary.json").exists():
        (phase_root / "decision_summary.json").write_text(
            json.dumps(
                {
                    "phase": phase,
                    "status": "descriptive_only",
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "h12_endpoint_policy": str(selected_endpoint_policy),
                    "h12_endpoint_policy_observed_counts": {str(k): int(v) for k, v in sorted(endpoint_policy_counts.items())},
                    "h12_endpoint_policy_mixed": bool(endpoint_policy_mixed),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        decision_path = phase_root / "decision_summary.json"
        try:
            decision_payload = json.loads(decision_path.read_text(encoding="utf-8"))
            decision_payload.setdefault("h12_endpoint_policy", str(selected_endpoint_policy))
            decision_payload.setdefault(
                "h12_endpoint_policy_observed_counts",
                {str(k): int(v) for k, v in sorted(endpoint_policy_counts.items())},
            )
            decision_payload.setdefault("h12_endpoint_policy_mixed", bool(endpoint_policy_mixed))
            decision_path.write_text(json.dumps(decision_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except Exception:
            LOGGER.exception("Failed to annotate decision payload with endpoint policy metadata: %s", decision_path)

    _write_promotion_guard(output_root=output_root, run_id=run_id)


def _write_promotion_guard(*, output_root: Path, run_id: str) -> None:
    phase2b_gate = output_root / "phase2b" / run_id / "gate_evaluation.json"
    phase2c_gate = output_root / "phase2c" / run_id / "gate_evaluation.json"
    phase2d_gate = output_root / "phase2d" / run_id / "gate_evaluation.json"
    phase2b_confirmatory_pass: bool | None = None
    if phase2b_gate.exists():
        try:
            payload = json.loads(phase2b_gate.read_text(encoding="utf-8"))
            value = payload.get("confirmatory_success")
            if isinstance(value, bool):
                phase2b_confirmatory_pass = value
        except Exception:
            phase2b_confirmatory_pass = None
    phase2c_seed_ready: bool | None = None
    if phase2c_gate.exists():
        try:
            payload = json.loads(phase2c_gate.read_text(encoding="utf-8"))
            value = payload.get("headline_7_seed_ready")
            if isinstance(value, bool):
                phase2c_seed_ready = value
        except Exception:
            phase2c_seed_ready = None
    phase2d_seed_ready: bool | None = None
    if phase2d_gate.exists():
        try:
            payload = json.loads(phase2d_gate.read_text(encoding="utf-8"))
            value = payload.get("headline_7_seed_ready")
            if isinstance(value, bool):
                phase2d_seed_ready = value
        except Exception:
            phase2d_seed_ready = None
    observed: bool | str = bool(phase2b_confirmatory_pass) if phase2b_confirmatory_pass is not None else "unknown"
    phase2c_seed_observed: bool | str = bool(phase2c_seed_ready) if phase2c_seed_ready is not None else "unknown"
    phase2d_seed_observed: bool | str = bool(phase2d_seed_ready) if phase2d_seed_ready is not None else "unknown"
    phase2c_eligible = bool((phase2b_confirmatory_pass is True) and (phase2c_seed_ready is True))
    phase2d_eligible = bool((phase2b_confirmatory_pass is True) and (phase2d_seed_ready is True))
    guard = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "advancement_policy": "analysis_only",
        "phase2b_confirmatory_pass_required_for_promotion": True,
        "phase2b_confirmatory_pass_observed": observed,
        "headline_claim_requires_7_seed_rerun_for_phase2c_d": True,
        "phase2c_headline_7_seed_ready_observed": phase2c_seed_observed,
        "phase2d_headline_7_seed_ready_observed": phase2d_seed_observed,
        "downstream": {
            "phase2c_confirmatory_promotion_eligible": phase2c_eligible,
            "phase2d_confirmatory_promotion_eligible": phase2d_eligible,
        },
    }
    shared_root = output_root / run_id
    shared_root.mkdir(parents=True, exist_ok=True)
    (shared_root / "promotion_guard.json").write_text(json.dumps(guard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for phase in ("phase2a", "phase2b", "phase2c", "phase2d"):
        phase_root = output_root / phase / run_id
        if phase_root.exists():
            (phase_root / "promotion_guard.json").write_text(
                json.dumps({**guard, "phase": phase}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
