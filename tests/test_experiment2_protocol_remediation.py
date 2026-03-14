from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from experiment2.analysis import _class_contrast, _h3_specificity, _kernel_deltas, _task_effects, evaluate_phase
from experiment2.kernels import KernelMetricsAccumulator
from experiment2.run import (
    BuildOptions,
    _build_feasibility_rows,
    _compute_long_offset_lock_recommendation,
    _resolve_build_options,
    build_phase_2a,
    build_phase_2a_long_pilot,
    build_phase_2b,
    calibrate_phase2a_floor_threshold,
)
from experiment2.stats import bootstrap_bca_ci_sensitivity
from experiment2.tasks import TaskExample, TokenPools, generate_task_examples
from shared.specs import ModelSpec

try:
    from experiment2.execution import (
        ExecutionConfig,
        _RunState,
        _aggregate_phase_outputs,
        _batch_size_for_row,
        _build_restricted_candidates,
        _condition_dir,
        _execute_cell,
        _evaluate_example_accuracy_only,
        _evaluate_example_from_token_logits,
        _maybe_reuse_from_phase2a,
        _quick_baseline_accuracy,
        run_manifest,
    )
except Exception:
    ExecutionConfig = None
    _RunState = None
    _aggregate_phase_outputs = None
    _batch_size_for_row = None
    _build_restricted_candidates = None
    _condition_dir = None
    _execute_cell = None
    _evaluate_example_accuracy_only = None
    _evaluate_example_from_token_logits = None
    _maybe_reuse_from_phase2a = None
    _quick_baseline_accuracy = None
    run_manifest = None


def _toy_pools() -> TokenPools:
    return TokenPools(
        filler=tuple(range(2000, 9000)),
        keys=tuple(range(100, 700)),
        values=tuple(range(800, 1400)),
        reserve=(1500, 1501, 1502, 1503, 1504),
        query_marker=1500,
        retrieval_marker=1501,
        no_match_token=1502,
    )


def test_task_effects_keeps_dataset_isolated() -> None:
    df = pd.DataFrame(
        [
            {
                "model": "m",
                "task": "tier1_stratified_ppl",
                "seq_len": 1024,
                "seed": 0,
                "split": "tier1_ppl",
                "dataset": "wiki",
                "intervention": "none",
                "mean_accuracy": 0.5,
                "mean_nll": 1.0,
                "floor_limited": False,
            },
            {
                "model": "m",
                "task": "tier1_stratified_ppl",
                "seq_len": 1024,
                "seed": 0,
                "split": "tier1_ppl",
                "dataset": "code",
                "intervention": "none",
                "mean_accuracy": 0.6,
                "mean_nll": 1.2,
                "floor_limited": False,
            },
            {
                "model": "m",
                "task": "tier1_stratified_ppl",
                "seq_len": 1024,
                "seed": 0,
                "split": "tier1_ppl",
                "dataset": "wiki",
                "intervention": "ablate_high_strong",
                "mean_accuracy": 0.4,
                "mean_nll": 1.5,
                "floor_limited": False,
            },
        ]
    )
    out = _task_effects(df)
    assert len(out) == len(df)
    wiki_ablate = out[(out["dataset"] == "wiki") & (out["intervention"] == "ablate_high_strong")].iloc[0]
    assert float(wiki_ablate["none_nll"]) == 1.0


def test_task_effects_adds_headroom_normalized_columns() -> None:
    df = pd.DataFrame(
        [
            {
                "model": "m",
                "task": "local_key_match",
                "seq_len": 1024,
                "seed": 0,
                "split": "synthetic",
                "intervention": "none",
                "mean_accuracy": 0.40,
                "mean_nll": 1.0,
                "floor_limited": False,
                "chance_accuracy": 0.10,
                "candidate_count": 10,
            },
            {
                "model": "m",
                "task": "local_key_match",
                "seq_len": 1024,
                "seed": 0,
                "split": "synthetic",
                "intervention": "ablate_high_strong",
                "mean_accuracy": 0.30,
                "mean_nll": 1.1,
                "floor_limited": False,
                "chance_accuracy": 0.10,
                "candidate_count": 10,
            },
        ]
    )
    out = _task_effects(df)
    row = out[out["intervention"] == "ablate_high_strong"].iloc[0]
    assert np.isclose(float(row["chance_accuracy_effective"]), 0.10)
    assert np.isclose(float(row["baseline_headroom"]), 0.30)
    assert np.isclose(float(row["drop_acc"]), 0.10)
    assert np.isclose(float(row["drop_acc_over_baseline"]), 0.25)
    assert np.isclose(float(row["drop_acc_over_headroom"]), 1.0 / 3.0)


def test_class_contrast_supports_value_col() -> None:
    eff = pd.DataFrame(
        [
            {"model": "m", "seed": 0, "task_class": "short", "intervention": "ablate_low_strong", "drop_acc": 0.04, "drop_acc_over_headroom": 0.20},
            {"model": "m", "seed": 0, "task_class": "long", "intervention": "ablate_low_strong", "drop_acc": 0.03, "drop_acc_over_headroom": 0.40},
        ]
    )
    raw = _class_contrast(eff, "ablate_low_strong", "h2")
    norm = _class_contrast(eff, "ablate_low_strong", "h2", value_col="drop_acc_over_headroom")
    assert np.isclose(float(raw.iloc[0]["contrast"]), -0.01)
    assert np.isclose(float(norm.iloc[0]["contrast"]), 0.20)


def test_kernel_deltas_respects_span() -> None:
    df = pd.DataFrame(
        [
            {
                "model": "m",
                "task": "copy_offset_bridge",
                "seq_len": 1024,
                "seed": 0,
                "split": "span_bridge",
                "span": 32,
                "metric": "track_a",
                "layer": 0,
                "head": 0,
                "intervention": "none",
                "mean_r2": 0.90,
            },
            {
                "model": "m",
                "task": "copy_offset_bridge",
                "seq_len": 1024,
                "seed": 0,
                "split": "span_bridge",
                "span": 64,
                "metric": "track_a",
                "layer": 0,
                "head": 0,
                "intervention": "none",
                "mean_r2": 0.50,
            },
            {
                "model": "m",
                "task": "copy_offset_bridge",
                "seq_len": 1024,
                "seed": 0,
                "split": "span_bridge",
                "span": 32,
                "metric": "track_a",
                "layer": 0,
                "head": 0,
                "intervention": "ablate_high_strong",
                "mean_r2": 0.80,
            },
        ]
    )
    out = _kernel_deltas(df)
    row = out[(out["span"] == 32) & (out["intervention"] == "ablate_high_strong")].iloc[0]
    assert abs(float(row["delta_r2_abs"]) - 0.10) < 1e-9


def test_phase2a_builder_updated_counts_and_tasks() -> None:
    rows = build_phase_2a(BuildOptions(random_draws_confirmatory=3, tier1_seeds_phase2b=7))
    assert len(rows) == 231
    tasks = {r.task for r in rows}
    assert tasks == {"local_key_match", "delayed_copy", "long_range_retrieval"}
    assert "local_copy_offset" not in tasks


def test_phase2a_builder_baseline_only_pilot_count() -> None:
    rows = build_phase_2a(BuildOptions(random_draws_confirmatory=3, tier1_seeds_phase2b=7, phase2a_baseline_only=True))
    assert len(rows) == 21
    assert all(r.intervention == "none" for r in rows)
    assert all(r.random_draw is None for r in rows)


def test_phase2a_long_pilot_builder_count() -> None:
    rows = build_phase_2a_long_pilot()
    assert len(rows) == 42
    assert all(r.phase == "phase2a_long_pilot" for r in rows)
    assert all(r.intervention == "none" for r in rows)
    assert {r.task for r in rows} == {"delayed_copy", "long_range_retrieval"}
    assert {r.model for r in rows} == {"llama-3.2-1b", "olmo-1b", "tinyllama-1.1b"}


def test_phase2b_builder_updated_counts_and_random_draw_axis() -> None:
    rows = build_phase_2b(BuildOptions(random_draws_confirmatory=3, tier1_seeds_phase2b=7))
    assert len(rows) == 1932
    random_rows = [r for r in rows if r.intervention.startswith("random_")]
    assert random_rows
    assert all(r.random_draw in {0, 1, 2} for r in random_rows)


def test_phase2b_builder_core_synthetic_only_flag() -> None:
    rows = build_phase_2b(
        BuildOptions(
            random_draws_confirmatory=3,
            tier1_seeds_phase2b=7,
            phase2b_core_synthetic_only=True,
        )
    )
    assert len(rows) == 924
    assert {r.split for r in rows} == {"synthetic"}


def test_phase2b_builder_strong_only_profile_has_expected_optionc_synthetic_size(monkeypatch) -> None:
    monkeypatch.setattr(
        "experiment2.run.load_long_offset_bundle",
        lambda strict=False: SimpleNamespace(active_long_tasks=("long_range_retrieval",)),
    )
    rows = build_phase_2b(
        BuildOptions(
            random_draws_confirmatory=3,
            tier1_seeds_phase2b=7,
            model_profile="scaleup_78b",
            rope_models=("llama-3.1-8b",),
            abspe_models=(),
            nope_models=(),
            phase2a_model="llama-3.1-8b",
            phase2d_models=("llama-3.1-8b",),
            intervention_profile="strong_only",
            phase2b_core_synthetic_only=True,
        )
    )
    synth = [r for r in rows if r.split == "synthetic"]
    assert len(synth) == 126
    assert {r.task for r in synth} == {"local_copy_offset", "local_key_match", "long_range_retrieval"}
    assert all(r.intervention in {"none", "ablate_high_strong", "ablate_low_strong", "random_strong"} for r in synth)
    random_rows = [r for r in synth if r.intervention == "random_strong"]
    assert random_rows
    assert all(r.random_draw in {0, 1, 2} for r in random_rows)


def test_build_feasibility_rows_count_and_none_only() -> None:
    rows = _build_feasibility_rows(
        run_id="sweep_test",
        device="cuda",
        offsets=(8, 12, 16, 24, 32, 48, 64, 96, 128),
    )
    assert len(rows) == 162
    assert all(r.phase == "feasibility" for r in rows)
    assert all(r.intervention == "none" for r in rows)
    assert {r.task for r in rows} == {"delayed_copy", "long_range_retrieval"}
    assert {int(r.span) for r in rows} == {8, 12, 16, 24, 32, 48, 64, 96, 128}


def test_long_offset_lock_recommendation_selects_contiguous_prefix() -> None:
    rows = []
    for model in ("llama-3.2-1b", "olmo-1b", "tinyllama-1.1b"):
        for task in ("delayed_copy", "long_range_retrieval"):
            for span in (8, 12, 16, 24, 32, 48, 64, 96, 128):
                for seed in range(3):
                    if span in (8, 12, 16, 24):
                        acc = 0.30
                    else:
                        acc = 0.10
                    rows.append(
                        {
                            "model": model,
                            "task": task,
                            "span": span,
                            "seed": seed,
                            "intervention": "none",
                            "mean_accuracy": acc,
                        }
                    )
    rec = _compute_long_offset_lock_recommendation(
        pd.DataFrame(rows),
        threshold=0.15,
        feasibility_offsets=(8, 12, 16, 24, 32, 48, 64, 96, 128),
        candidate_order=(8, 12, 16, 24, 32, 48, 64, 96, 128),
    )
    assert rec["status"] == "ok"
    assert rec["selected_long_offsets"] == [8, 12, 16, 24]
    assert rec["fallback_spans"] == [8]
    assert rec["lock_policy"] == "per_task_model_majority"


def test_long_offset_lock_recommendation_fails_when_only_one_feasible_offset() -> None:
    rows = []
    for model in ("llama-3.2-1b", "olmo-1b", "tinyllama-1.1b"):
        for task in ("delayed_copy", "long_range_retrieval"):
            for span in (8, 12, 16, 24, 32, 48, 64, 96, 128):
                for seed in range(3):
                    acc = 0.30 if span == 8 else 0.10
                    rows.append(
                        {
                            "model": model,
                            "task": task,
                            "span": span,
                            "seed": seed,
                            "intervention": "none",
                            "mean_accuracy": acc,
                        }
                    )
    rec = _compute_long_offset_lock_recommendation(
        pd.DataFrame(rows),
        threshold=0.15,
        feasibility_offsets=(8, 12, 16, 24, 32, 48, 64, 96, 128),
        candidate_order=(8, 12, 16, 24, 32, 48, 64, 96, 128),
    )
    assert rec["status"] == "fail_insufficient_feasible_offsets"
    assert rec["selected_long_offsets"] == [8]
    assert rec["fallback_spans"] == [8]


def test_long_offset_lock_recommendation_uses_per_task_model_majority() -> None:
    rows = []
    for model in ("llama-3.2-1b", "olmo-1b", "tinyllama-1.1b"):
        for task in ("delayed_copy", "long_range_retrieval"):
            for span in (8, 12, 16):
                for seed in range(3):
                    acc = 0.10
                    if span in (8, 12):
                        if task == "delayed_copy":
                            if model in {"llama-3.2-1b", "olmo-1b"}:
                                acc = 0.30
                        else:
                            acc = 0.30
                    rows.append(
                        {
                            "model": model,
                            "task": task,
                            "span": span,
                            "seed": seed,
                            "intervention": "none",
                            "mean_accuracy": acc,
                        }
                    )
    rec = _compute_long_offset_lock_recommendation(
        pd.DataFrame(rows),
        threshold=0.15,
        feasibility_offsets=(8, 12, 16),
        candidate_order=(8, 12, 16),
    )
    assert rec["status"] == "ok"
    assert rec["selected_long_offsets"] == [8, 12]
    assert rec["fallback_spans"] == [8]


def test_long_offset_lock_recommendation_retrieval_fallback_scaleup() -> None:
    rows = []
    models = ("llama-3.1-8b", "olmo-2-7b", "gemma-7b")
    for model in models:
        for task in ("delayed_copy", "long_range_retrieval"):
            for span in (8, 12, 16):
                for seed in range(3):
                    acc = 0.10
                    if task == "long_range_retrieval" and span in (8, 12, 16):
                        acc = 0.30
                    if task == "delayed_copy" and span == 8 and model in {"llama-3.1-8b", "olmo-2-7b"}:
                        acc = 0.30
                    rows.append(
                        {
                            "model": model,
                            "task": task,
                            "span": span,
                            "seed": seed,
                            "intervention": "none",
                            "mean_accuracy": acc,
                        }
                    )
    rec = _compute_long_offset_lock_recommendation(
        pd.DataFrame(rows),
        threshold=0.15,
        feasibility_offsets=(8, 12, 16),
        candidate_order=(8, 12, 16),
        long_task_feasibility_policy="retrieval_fallback",
        model_profile="scaleup_78b",
    )
    assert rec["status"] == "ok"
    assert rec["long_task_policy_applied"] == "retrieval_fallback"
    assert rec["lock_resolution_mode"] == "retrieval_only_fallback"
    assert rec["strict_selected_long_offsets"] == [8]
    assert rec["fallback_selected_long_offsets"] == [8, 12, 16]
    assert rec["selected_long_offsets"] == [8, 12, 16]
    assert rec["active_long_tasks"] == ["long_range_retrieval"]


def test_long_offset_lock_retrieval_fallback_is_disabled_for_legacy_profile() -> None:
    rows = []
    models = ("llama-3.2-1b", "olmo-1b", "tinyllama-1.1b")
    for model in models:
        for task in ("delayed_copy", "long_range_retrieval"):
            for span in (8, 12):
                for seed in range(3):
                    acc = 0.30 if (task == "long_range_retrieval" and span in (8, 12)) else 0.10
                    rows.append(
                        {
                            "model": model,
                            "task": task,
                            "span": span,
                            "seed": seed,
                            "intervention": "none",
                            "mean_accuracy": acc,
                        }
                    )
    rec = _compute_long_offset_lock_recommendation(
        pd.DataFrame(rows),
        threshold=0.15,
        feasibility_offsets=(8, 12),
        candidate_order=(8, 12),
        long_task_feasibility_policy="retrieval_fallback",
        model_profile="legacy_1b",
    )
    assert rec["long_task_policy_applied"] == "strict_two_task"
    assert rec["lock_resolution_mode"] == "strict_two_task"
    assert rec["status"] == "fail_insufficient_feasible_offsets"
    assert rec["active_long_tasks"] == ["delayed_copy", "long_range_retrieval"]


def test_scaleup_build_uses_lock_active_long_tasks(monkeypatch) -> None:
    monkeypatch.setattr(
        "experiment2.run.load_long_offset_bundle",
        lambda strict=False: SimpleNamespace(active_long_tasks=("long_range_retrieval",)),
    )
    opts = BuildOptions(
        random_draws_confirmatory=3,
        tier1_seeds_phase2b=7,
        model_profile="scaleup_78b",
        rope_models=("olmo-2-7b", "llama-3.1-8b", "gemma-7b"),
        abspe_models=(),
        nope_models=(),
        phase2a_model="llama-3.1-8b",
        phase2d_models=("olmo-2-7b", "llama-3.1-8b"),
    )
    p2a_rows = build_phase_2a(opts)
    p2b_rows = build_phase_2b(opts)
    assert {r.task for r in p2a_rows} == {"local_key_match", "long_range_retrieval"}
    synthetic_tasks = {r.task for r in p2b_rows if r.split == "synthetic"}
    assert synthetic_tasks == {"local_copy_offset", "local_key_match", "long_range_retrieval"}


def test_build_options_model_allowlist_filters_profile_models() -> None:
    opts = _resolve_build_options(
        model_profile="scaleup_78b",
        random_draws_confirmatory=3,
        tier1_seeds_phase2b=7,
        phase2a_baseline_only=False,
        intervention_profile="full",
        phase2b_core_synthetic_only=False,
        model_allowlist=("llama-3.1-8b",),
    )
    assert opts.rope_models == ("llama-3.1-8b",)
    assert opts.phase2a_model == "llama-3.1-8b"
    assert opts.model_allowlist == ("llama-3.1-8b",)


def test_long_offset_lock_recommendation_respects_model_allowlist() -> None:
    rows = []
    for model in ("llama-3.1-8b", "olmo-2-7b", "gemma-7b"):
        for task in ("delayed_copy", "long_range_retrieval"):
            for span in (32, 48, 64):
                for seed in range(3):
                    acc = 0.10
                    if model == "llama-3.1-8b" and task == "long_range_retrieval":
                        acc = 0.30
                    rows.append(
                        {
                            "model": model,
                            "task": task,
                            "span": span,
                            "seed": seed,
                            "intervention": "none",
                            "mean_accuracy": acc,
                        }
                    )
    rec = _compute_long_offset_lock_recommendation(
        pd.DataFrame(rows),
        threshold=0.15,
        feasibility_offsets=(32, 48, 64),
        candidate_order=(32, 48, 64),
        long_task_feasibility_policy="retrieval_fallback",
        model_profile="scaleup_78b",
        model_allowlist=("llama-3.1-8b",),
    )
    assert rec["status"] == "ok"
    assert rec["lock_resolution_mode"] == "retrieval_only_fallback"
    assert rec["active_long_tasks"] == ["long_range_retrieval"]
    assert rec["selected_long_offsets"] == [32, 48, 64]
    assert rec["model_allowlist"] == ["llama-3.1-8b"]
    assert rec["effective_models"] == ["llama-3.1-8b"]


def test_copy_tasks_prevent_chained_targets() -> None:
    pools = _toy_pools()
    examples = generate_task_examples(
        task_name="local_copy_offset",
        model_name="toy-model",
        seq_len=256,
        seed=0,
        count=16,
        pools=pools,
    )
    for ex in examples:
        offset = int(ex.task_params["offset"])
        targets = set(ex.target_positions)
        assert all((t - offset) not in targets for t in targets)
        assert all((t + offset) not in targets for t in targets)


def test_delayed_copy_honors_span_override() -> None:
    pools = _toy_pools()
    examples = generate_task_examples(
        task_name="delayed_copy",
        model_name="toy-model",
        seq_len=256,
        seed=0,
        count=8,
        pools=pools,
        span_override=32,
    )
    assert all(int(ex.task_params["offset"]) == 32 for ex in examples)
    assert all(int(ex.dependency_span) == 32 for ex in examples)


def test_long_range_retrieval_honors_span_override() -> None:
    pools = _toy_pools()
    examples = generate_task_examples(
        task_name="long_range_retrieval",
        model_name="toy-model",
        seq_len=1024,
        seed=0,
        count=4,
        pools=pools,
        span_override=32,
    )
    assert all(int(ex.task_params["min_gap"]) == 32 for ex in examples)
    assert all(int(ex.dependency_span) == 32 for ex in examples)


def test_local_key_match_has_no_key_overwrite() -> None:
    pools = _toy_pools()
    examples = generate_task_examples(
        task_name="local_key_match",
        model_name="toy-model",
        seq_len=256,
        seed=0,
        count=16,
        pools=pools,
    )
    assert all(int(ex.task_params.get("key_overwrite_count", -1)) == 0 for ex in examples)


def test_restricted_candidates_deterministic_and_include_target() -> None:
    if _build_restricted_candidates is None:
        pytest.skip("experiment2.execution import unavailable in this environment")
    pools = _toy_pools()
    example = generate_task_examples(
        task_name="local_copy_offset",
        model_name="toy-model",
        seq_len=256,
        seed=1,
        count=1,
        pools=pools,
    )[0]
    pos = int(example.target_positions[0])
    target = int(example.target_tokens[0])
    c1 = _build_restricted_candidates(
        example=example,
        target=target,
        position=pos,
        pools=pools,
        candidate_size=10,
        candidate_policy_version="restricted_candidates_v1_structured_first",
    )
    c2 = _build_restricted_candidates(
        example=example,
        target=target,
        position=pos,
        pools=pools,
        candidate_size=10,
        candidate_policy_version="restricted_candidates_v1_structured_first",
    )
    assert c1 == c2
    assert len(c1) == 10
    assert len(set(c1)) == 10
    assert target in set(c1)


def test_copy_candidates_include_structured_non_filler_distractors() -> None:
    if _build_restricted_candidates is None:
        pytest.skip("experiment2.execution import unavailable in this environment")
    pools = _toy_pools()
    example = TaskExample(
        id="toy:copy",
        task_name="local_copy_offset",
        tokens=[10, 11, 12, 13, 14, 15, 16, 17],
        target_positions=[2, 3, 4, 5],
        target_tokens=[11, 12, 13, 14],
        dependency_span=1,
        task_class="short",
        seed=0,
        model="toy-model",
        length=8,
        task_params={"offset": 1},
        pair_count=None,
        query_key=None,
        distractor_key=None,
        match_rule="copy_offset_1",
        has_no_match=False,
    )
    candidates = _build_restricted_candidates(
        example=example,
        target=11,
        position=2,
        pools=pools,
        candidate_size=10,
        candidate_policy_version="restricted_candidates_v1_structured_first",
    )
    offset = int(example.task_params["offset"])
    structured_expected = {
        int(example.tokens[int(t) - offset])
        for t in example.target_positions
        if int(t) - offset >= 0 and int(example.tokens[int(t) - offset]) != 11
    }
    structured_expected.update(int(tok) for tok in example.target_tokens if int(tok) != 11)
    structured_expected.update(int(example.tokens[p]) for p in (0, 1, 3, 4) if 0 <= p < len(example.tokens) and int(example.tokens[p]) != 11)
    assert set(candidates).intersection(structured_expected)


def test_restricted_candidates_key_match_include_no_match_when_non_target() -> None:
    if _build_restricted_candidates is None:
        pytest.skip("experiment2.execution import unavailable in this environment")
    pools = _toy_pools()
    examples = generate_task_examples(
        task_name="local_key_match",
        model_name="toy-model",
        seq_len=256,
        seed=2,
        count=8,
        pools=pools,
    )
    chosen = None
    for ex in examples:
        seq_values = {int(tok) for tok in ex.tokens if int(tok) in set(pools.values)}
        for pos, tgt in zip(ex.target_positions, ex.target_tokens):
            if int(tgt) != int(pools.no_match_token) and any(v != int(tgt) for v in seq_values):
                chosen = (ex, int(pos), int(tgt))
                break
        if chosen is not None:
            break
    assert chosen is not None
    ex, pos, target = chosen
    seq_value_tokens = sorted({int(tok) for tok in ex.tokens if int(tok) in set(pools.values)})
    candidates = _build_restricted_candidates(
        example=ex,
        target=target,
        position=pos,
        pools=pools,
        candidate_size=10,
        candidate_policy_version="restricted_candidates_v1_structured_first",
        sequence_value_tokens=seq_value_tokens,
    )
    assert int(pools.no_match_token) in set(candidates)
    assert any((tok in pools.values) and (tok != target) for tok in candidates)


def test_tier1_split_stays_full_vocab_when_synthetic_eval_restricted() -> None:
    if _evaluate_example_from_token_logits is None:
        pytest.skip("experiment2.execution import unavailable in this environment")
    example = TaskExample(
        id="toy:tier1",
        task_name="tier1_stratified_ppl",
        tokens=[10, 11, 12, 13, 14],
        target_positions=[1, 2, 3, 4],
        target_tokens=[11, 12, 13, 14],
        dependency_span=0,
        task_class="tier1",
        seed=0,
        model="toy-model",
        length=5,
        task_params={"dataset": "toy"},
        pair_count=None,
        query_key=None,
        distractor_key=None,
        match_rule="all_positions",
        has_no_match=False,
    )
    pools = _toy_pools()
    logits = torch.randn(5, 9000, dtype=torch.float32)
    metrics, _ = _evaluate_example_from_token_logits(
        logits,
        example,
        split="tier1_ppl",
        pools=pools,
        synthetic_eval_mode="restricted",
        candidate_size=10,
        candidate_policy_version="restricted_candidates_v1_structured_first",
    )
    assert metrics["eval_mode"] == "full_vocab"
    assert metrics["candidate_count"] is None
    assert metrics["restricted_accuracy"] is None


def test_h3_specificity_filter_excludes_other_task_class() -> None:
    df = pd.DataFrame(
        [
            {
                "model": "m",
                "task_class": "other",
                "metric": "track_b_raw",
                "intervention": "ablate_high_strong",
                "seed": 0,
                "delta_r2_abs": 0.2,
            },
            {
                "model": "m",
                "task_class": "other",
                "metric": "track_b_raw",
                "intervention": "random_strong",
                "seed": 0,
                "delta_r2_abs": 0.1,
            },
            {
                "model": "m",
                "task_class": "short",
                "metric": "track_b_raw",
                "intervention": "ablate_high_strong",
                "seed": 0,
                "delta_r2_abs": 0.2,
            },
            {
                "model": "m",
                "task_class": "short",
                "metric": "track_b_raw",
                "intervention": "random_strong",
                "seed": 0,
                "delta_r2_abs": 0.1,
            },
        ]
    )
    out = _h3_specificity(df, allowed_task_classes={"short", "long"}, allowed_metrics={"track_b_raw"})
    assert set(out["task_class"].unique().tolist()) <= {"short", "long"}


def test_h3_specificity_requires_paired_seed_inner_join() -> None:
    df = pd.DataFrame(
        [
            {"model": "m", "task_class": "short", "metric": "track_b_raw", "intervention": "ablate_high_strong", "seed": 0, "delta_r2_abs": 0.20},
            {"model": "m", "task_class": "short", "metric": "track_b_raw", "intervention": "ablate_high_strong", "seed": 1, "delta_r2_abs": 0.10},
            {"model": "m", "task_class": "short", "metric": "track_b_raw", "intervention": "random_strong", "seed": 1, "delta_r2_abs": 0.05},
            {"model": "m", "task_class": "short", "metric": "track_b_raw", "intervention": "random_strong", "seed": 2, "delta_r2_abs": 0.06},
        ]
    )
    out = _h3_specificity(df, allowed_task_classes={"short"}, allowed_metrics={"track_b_raw"})
    strong = out[(out["severity"] == "strong") & (out["target"] == "high")].iloc[0]
    assert int(strong["paired_seed_count"]) == 1
    assert bool(strong["insufficient_pairing"]) is True
    assert "diff_ci_low" in out.columns
    assert "severity_pass_ci" in out.columns


def test_h3_prefilter_equivalence_to_internal_filtering() -> None:
    rows = []
    for seed in range(7):
        rows.extend(
            [
                {"model": "m", "task_class": "short", "metric": "track_b_raw", "intervention": "ablate_high_strong", "seed": seed, "delta_r2_abs": 0.20},
                {"model": "m", "task_class": "short", "metric": "track_b_raw", "intervention": "random_strong", "seed": seed, "delta_r2_abs": 0.08},
                {"model": "m", "task_class": "long", "metric": "track_b_raw", "intervention": "ablate_low_strong", "seed": seed, "delta_r2_abs": 0.19},
                {"model": "m", "task_class": "long", "metric": "track_b_raw", "intervention": "random_strong", "seed": seed, "delta_r2_abs": 0.09},
                {"model": "m", "task_class": "short", "metric": "track_a", "intervention": "ablate_high_strong", "seed": seed, "delta_r2_abs": 0.15},
                {"model": "m", "task_class": "other", "metric": "track_b_raw", "intervention": "ablate_high_strong", "seed": seed, "delta_r2_abs": 0.30},
            ]
        )
    df = pd.DataFrame(rows)
    internal = _h3_specificity(df, allowed_task_classes={"short", "long"}, allowed_metrics={"track_b_raw"})
    prefiltered = _h3_specificity(
        df[df["task_class"].isin(["short", "long"]) & (df["metric"] == "track_b_raw")].copy()
    )
    key_cols = ["model", "task_class", "target", "severity", "metric"]
    cmp_cols = key_cols + ["S_ratio", "S_diff", "h3_group_pass"]
    a = internal[cmp_cols].sort_values(key_cols).reset_index(drop=True)
    b = prefiltered[cmp_cols].sort_values(key_cols).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-12, rtol=1e-12)


def test_kernel_engine_legacy_vs_optimized_parity() -> None:
    torch_seed = 1234
    import torch

    torch.manual_seed(torch_seed)
    layers, batch, heads, seq, dim = 2, 2, 3, 8, 8
    q = torch.randn(layers, batch, heads, seq, dim, dtype=torch.float32)
    k = torch.randn(layers, batch, heads, seq, dim, dtype=torch.float32)
    logits = torch.randn(layers, batch, heads, seq, seq, dtype=torch.float32)
    capture = SimpleNamespace(q=q, k=k, logits=logits)

    spec = ModelSpec(name="toy", hf_id="toy", norm="LayerNorm", pe_scheme="RoPE")
    legacy = KernelMetricsAccumulator(spec, engine="legacy", centered_mode="shared_mean")
    opt = KernelMetricsAccumulator(spec, engine="optimized", centered_mode="shared_mean")

    legacy.update_track_a_raw(capture)
    opt.update_track_a_raw(capture)
    legacy.accumulate_shared_means(capture)
    opt.accumulate_shared_means(capture)
    lq, lk = legacy.finalize_shared_means()
    oq, ok = opt.finalize_shared_means()
    legacy.update_centered(capture, shared_q_mean=lq, shared_k_mean=lk)
    opt.update_centered(capture, shared_q_mean=oq, shared_k_mean=ok)

    for key in sorted(legacy.track_a):
        assert abs(legacy.track_a[key].mean() - opt.track_a[key].mean()) < 1e-10
    for key in sorted(legacy.track_b_raw):
        assert abs(legacy.track_b_raw[key].mean() - opt.track_b_raw[key].mean()) < 1e-10
    for key in sorted(legacy.track_b_centered):
        assert abs(legacy.track_b_centered[key].mean() - opt.track_b_centered[key].mean()) < 1e-10


def test_bootstrap_bca_ci_sensitivity_deterministic() -> None:
    values = np.array([0.1, 0.12, 0.09, 0.11, 0.13, 0.08, 0.1], dtype=np.float64)
    a = bootstrap_bca_ci_sensitivity(values, seeds=(0, 1, 2), b=500)
    b = bootstrap_bca_ci_sensitivity(values, seeds=(0, 1, 2), b=500)
    assert a == b
    assert a["seeds_evaluated"] == 3


def test_bootstrap_bca_ci_sensitivity_default_budget() -> None:
    values = np.array([0.1, 0.12, 0.09, 0.11, 0.13, 0.08, 0.1], dtype=np.float64)
    out = bootstrap_bca_ci_sensitivity(values)
    assert out["seeds_evaluated"] == 3


def test_phase2a_floor_calibration_keeps_015_when_pass_rate_sufficient(tmp_path) -> None:
    phase_root = tmp_path / "phase2a" / "runx"
    phase_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(10):
        rows.append(
            {
                "split": "synthetic",
                "intervention": "none",
                "mean_accuracy": (0.20 if idx < 8 else 0.10),
                "eval_mode": "restricted",
            }
        )
    pd.DataFrame(rows).to_parquet(phase_root / "aggregate_task_metrics.parquet", engine="pyarrow", index=False)
    out = calibrate_phase2a_floor_threshold(output_root=tmp_path, run_id="runx", min_pass_rate=0.70)
    assert out["status"] == "pass_keep_0.15"
    assert float(out["selected_floor_threshold"]) == 0.15


def test_phase2a_floor_calibration_can_lower_to_013(tmp_path) -> None:
    phase_root = tmp_path / "phase2a" / "runy"
    phase_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(10):
        rows.append(
            {
                "split": "synthetic",
                "intervention": "none",
                "mean_accuracy": (0.14 if idx < 8 else 0.10),
                "eval_mode": "restricted",
            }
        )
    pd.DataFrame(rows).to_parquet(phase_root / "aggregate_task_metrics.parquet", engine="pyarrow", index=False)
    out = calibrate_phase2a_floor_threshold(output_root=tmp_path, run_id="runy", min_pass_rate=0.70)
    assert out["status"] == "pass_lower_0.13"
    assert float(out["selected_floor_threshold"]) == 0.13


def test_phase2a_floor_calibration_from_floor_decisions(tmp_path) -> None:
    phase_root = tmp_path / "phase2a" / "runz"
    floor_dir = phase_root / "floor_decisions"
    floor_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(10):
        payload = {
            "key": f"phase2a|synthetic|llama-3.1-8b|local_key_match|1024|{idx}|None",
            "baseline_accuracy": (0.20 if idx < 8 else 0.10),
            "fallback_accuracy": None,
            "fallback_applied": False,
            "floor_limited": False,
            "threshold": 0.15,
        }
        (floor_dir / f"k{idx}.json").write_text(json.dumps(payload), encoding="utf-8")
    out = calibrate_phase2a_floor_threshold(output_root=tmp_path, run_id="runz", min_pass_rate=0.70)
    assert out["status"] == "pass_keep_0.15"
    assert float(out["selected_floor_threshold"]) == 0.15
    assert str(out["calibration_source"]) == "floor_decisions_baseline_accuracy"


def test_phase2a_gate_contains_h3_and_advancement_fields(tmp_path) -> None:
    phase_root = tmp_path / "phase2a" / "runx"
    phase_root.mkdir(parents=True, exist_ok=True)
    task_df = pd.DataFrame(
        [
            {"model": "llama-3.2-1b", "task": "local_key_match", "seq_len": 1024, "seed": s, "split": "synthetic", "intervention": "none", "mean_accuracy": 0.9, "mean_nll": 0.5, "floor_limited": False}
            for s in range(7)
        ]
        + [
            {"model": "llama-3.2-1b", "task": "delayed_copy", "seq_len": 1024, "seed": s, "split": "synthetic", "intervention": "none", "mean_accuracy": 0.9, "mean_nll": 0.5, "floor_limited": False}
            for s in range(7)
        ]
        + [
            {"model": "llama-3.2-1b", "task": "local_key_match", "seq_len": 1024, "seed": s, "split": "synthetic", "intervention": "ablate_high_strong", "mean_accuracy": 0.75, "mean_nll": 0.7, "floor_limited": False}
            for s in range(7)
        ]
        + [
            {"model": "llama-3.2-1b", "task": "delayed_copy", "seq_len": 1024, "seed": s, "split": "synthetic", "intervention": "ablate_low_strong", "mean_accuracy": 0.75, "mean_nll": 0.7, "floor_limited": False}
            for s in range(7)
        ]
        + [
            {"model": "llama-3.2-1b", "task": "local_key_match", "seq_len": 1024, "seed": s, "split": "synthetic", "intervention": "random_strong", "mean_accuracy": 0.83, "mean_nll": 0.6, "floor_limited": False}
            for s in range(7)
        ]
        + [
            {"model": "llama-3.2-1b", "task": "delayed_copy", "seq_len": 1024, "seed": s, "split": "synthetic", "intervention": "random_strong", "mean_accuracy": 0.83, "mean_nll": 0.6, "floor_limited": False}
            for s in range(7)
        ]
    )
    kernel_rows = []
    for s in range(7):
        for task in ("local_key_match", "delayed_copy"):
            for intervention, r2 in (("none", 0.6), ("ablate_high_strong", 0.5), ("ablate_low_strong", 0.5), ("random_strong", 0.55)):
                kernel_rows.append(
                    {
                        "model": "llama-3.2-1b",
                        "task": task,
                        "seq_len": 1024,
                        "seed": s,
                        "split": "synthetic",
                        "metric": "track_b_raw",
                        "layer": 0,
                        "head": 0,
                        "intervention": intervention,
                        "mean_r2": r2,
                    }
                )
    kernel_df = pd.DataFrame(kernel_rows)
    task_df.to_parquet(phase_root / "aggregate_task_metrics.parquet", engine="pyarrow", index=False)
    kernel_df.to_parquet(phase_root / "aggregate_kernel_metrics.parquet", engine="pyarrow", index=False)

    out = evaluate_phase(phase_root)
    gate = out.gate_payload
    assert "h3_confirmatory_track_b_raw" in gate
    assert "ci_seed_sensitivity" in gate
    assert "h1" in gate and "h2" in gate and "criteria" in gate
    assert "headroom_normalized_diagnostic" in gate
    assert gate["headroom_normalized_diagnostic"]["advisory_only"] is True
    assert "task_class_headroom_summary" in gate
    assert "headroom_confound_risk" in gate
    assert "span_overlap_diagnostic" in gate
    assert np.isnan(gate["h3_confirmatory_track_b_raw"]["pass_rate_ci"])
    assert gate["advancement_policy"] == "analysis_only"
    assert gate["phase2b_confirmatory_pass_required_for_promotion"] is True


def test_phase2c_defaults_to_ineligible_without_phase2b_gate(tmp_path) -> None:
    phase_root = tmp_path / "phase2c" / "runx"
    phase_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame().to_parquet(phase_root / "aggregate_task_metrics.parquet", engine="pyarrow", index=False)
    pd.DataFrame().to_parquet(phase_root / "aggregate_kernel_metrics.parquet", engine="pyarrow", index=False)
    out = evaluate_phase(phase_root)
    gate = out.gate_payload
    assert gate["advancement_policy"] == "analysis_only"
    assert gate["phase2b_confirmatory_pass_observed"] == "unknown"
    assert gate["confirmatory_promotion_eligible"] is False


def test_phase2b_gate_has_pooled_h1_h2_inference(tmp_path) -> None:
    phase_root = tmp_path / "phase2b" / "runx"
    phase_root.mkdir(parents=True, exist_ok=True)
    models = ["olmo-1b", "llama-3.2-1b", "tinyllama-1.1b"]
    short_tasks = ["local_copy_offset", "local_key_match"]
    long_tasks = ["delayed_copy", "long_range_retrieval"]
    task_rows = []
    kernel_rows = []
    for model in models:
        for seed in range(7):
            for task in short_tasks + long_tasks:
                base_acc = 0.90
                task_rows.append(
                    {
                        "model": model,
                        "task": task,
                        "seq_len": 1024,
                        "seed": seed,
                        "split": "synthetic",
                        "intervention": "none",
                        "mean_accuracy": base_acc,
                        "mean_nll": 0.5,
                        "floor_limited": False,
                    }
                )
                high_acc = 0.70 if task in short_tasks else 0.85
                low_acc = 0.85 if task in short_tasks else 0.70
                rand_acc = 0.82
                for intervention, acc in (
                    ("ablate_high_strong", high_acc),
                    ("ablate_low_strong", low_acc),
                    ("random_strong", rand_acc),
                ):
                    task_rows.append(
                        {
                            "model": model,
                            "task": task,
                            "seq_len": 1024,
                            "seed": seed,
                            "split": "synthetic",
                            "intervention": intervention,
                            "mean_accuracy": acc,
                            "mean_nll": 0.7,
                            "floor_limited": False,
                        }
                    )
            for task in short_tasks + long_tasks:
                for intervention, r2 in (
                    ("none", 0.60),
                    ("ablate_high_strong", 0.45 if task in short_tasks else 0.52),
                    ("ablate_low_strong", 0.52 if task in short_tasks else 0.45),
                    ("random_strong", 0.54),
                ):
                    for metric in ("track_b_raw", "track_a", "track_b_centered"):
                        kernel_rows.append(
                            {
                                "model": model,
                                "task": task,
                                "seq_len": 1024,
                                "seed": seed,
                                "split": "synthetic",
                                "metric": metric,
                                "layer": 0,
                                "head": 0,
                                "intervention": intervention,
                                "mean_r2": r2,
                            }
                        )
    pd.DataFrame(task_rows).to_parquet(phase_root / "aggregate_task_metrics.parquet", engine="pyarrow", index=False)
    pd.DataFrame(kernel_rows).to_parquet(phase_root / "aggregate_kernel_metrics.parquet", engine="pyarrow", index=False)
    out = evaluate_phase(phase_root)
    gate = out.gate_payload
    assert gate["h12_endpoint_policy"] == "raw_primary"
    assert gate["h12_endpoint_policy_mixed"] is False
    assert gate["h12_endpoint_policy_observed_counts"] == {}
    assert "h1" in gate and "h2" in gate
    assert "primary_fixed_effects" in gate
    assert "cluster_sensitivity" in gate
    assert "restricted_rank_diagnostic" in gate
    assert gate["h3_gate_policy"] == "point_estimate_confirmatory_ci_diagnostic"
    assert gate["h3_ci_gated_pass_rate"] == gate["h3_rate_ci"] == gate["h3_pass_rate_ci"]
    assert "not CI of pass rate" in gate["h3_ci_gated_pass_rate_interpretation"]
    assert gate["cluster_sensitivity"]["evaluable"] is False
    assert gate["cluster_sensitivity_flag"] is False
    assert gate["cluster_h1_evaluable"] is False
    assert gate["cluster_h2_evaluable"] is False
    assert gate["cluster_any_evaluable"] is False
    assert gate["cluster_evaluable_policy"] == "any_hypothesis_evaluable_or"
    assert gate["task_family_difficulty_diagnostic"]["advisory_only"] is True
    assert gate["h1"]["label"] in {"pass", "imprecise_pass", "fail"}
    assert gate["h2"]["label"] in {"pass", "imprecise_pass", "fail"}
    assert "p_holm" in gate["h1"] and "p_holm" in gate["h2"]
    assert "h3_confirmatory_track_b_raw_per_draw_diagnostic" in gate
    assert "headroom_normalized_diagnostic" in gate
    assert gate["headroom_normalized_diagnostic"]["advisory_only"] is True
    assert "task_class_headroom_summary" in gate
    assert isinstance(gate["headroom_confound_risk"], bool)
    assert "feasibility_conditioned_subset" in gate
    subset = gate["feasibility_conditioned_subset"]
    assert subset["exploratory_only"] is True
    assert subset["subset_models"] == ["llama-3.2-1b", "olmo-1b"]
    assert subset["status"] == "ok"
    assert subset["n_contrasts"]["h1"] == 14
    assert subset["n_contrasts"]["h2"] == 14
    required = [
        "replication_2_of_3",
        "cross_task_agreement_2_of_3",
        "content_gated_safeguard_2_of_3",
        "specificity_advantage_2_of_3",
        "kernel_specificity_50pct",
        "pooled_primary_inference_holm_0_05",
    ]
    assert gate["confirmatory_success"] == all(bool(gate["criteria"][k]) for k in required)
    assert out.decision_payload["hypotheses"]["H1"] in {"pass", "imprecise_pass", "fail"}
    assert out.decision_payload["hypotheses"]["H2"] in {"pass", "imprecise_pass", "fail"}
    assert out.decision_payload["h12_endpoint_policy"] == "raw_primary"
    assert any("h3_ci_gated_pass_rate" in note for note in out.decision_payload["reporting_interpretation_notes"])
    assert any("Headroom-normalized H1/H2 diagnostics are advisory only" in note for note in out.decision_payload["reporting_interpretation_notes"])


def test_phase2b_h12_endpoint_policy_propagates_from_run_configs(tmp_path) -> None:
    phase_root = tmp_path / "phase2b" / "run_policy"
    phase_root.mkdir(parents=True, exist_ok=True)
    task_rows = []
    kernel_rows = []
    model = "llama-3.2-1b"
    short_tasks = ["local_copy_offset", "local_key_match"]
    long_tasks = ["delayed_copy", "long_range_retrieval"]
    for seed in range(7):
        for task in short_tasks + long_tasks:
            task_rows.append(
                {
                    "model": model,
                    "task": task,
                    "seq_len": 1024,
                    "seed": seed,
                    "split": "synthetic",
                    "intervention": "none",
                    "mean_accuracy": 0.90,
                    "mean_nll": 0.5,
                    "floor_limited": False,
                }
            )
            for intervention, acc in (
                ("ablate_high_strong", 0.70 if task in short_tasks else 0.85),
                ("ablate_low_strong", 0.85 if task in short_tasks else 0.70),
                ("random_strong", 0.82),
            ):
                task_rows.append(
                    {
                        "model": model,
                        "task": task,
                        "seq_len": 1024,
                        "seed": seed,
                        "split": "synthetic",
                        "intervention": intervention,
                        "mean_accuracy": acc,
                        "mean_nll": 0.7,
                        "floor_limited": False,
                    }
                )
            for intervention, r2 in (
                ("none", 0.60),
                ("ablate_high_strong", 0.45 if task in short_tasks else 0.52),
                ("ablate_low_strong", 0.52 if task in short_tasks else 0.45),
                ("random_strong", 0.54),
            ):
                kernel_rows.append(
                    {
                        "model": model,
                        "task": task,
                        "seq_len": 1024,
                        "seed": seed,
                        "split": "synthetic",
                        "metric": "track_b_raw",
                        "layer": 0,
                        "head": 0,
                        "intervention": intervention,
                        "mean_r2": r2,
                    }
                )
        cfg_dir = phase_root / f"cfg_seed_{seed}"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "row": {"seed": seed, "model": model},
                    "execution_status": "executed",
                    "centered_pending": False,
                    "h12_endpoint_policy": "co_primary_raw_headroom",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    pd.DataFrame(task_rows).to_parquet(phase_root / "aggregate_task_metrics.parquet", engine="pyarrow", index=False)
    pd.DataFrame(kernel_rows).to_parquet(phase_root / "aggregate_kernel_metrics.parquet", engine="pyarrow", index=False)
    out = evaluate_phase(phase_root)
    gate = out.gate_payload
    assert gate["h12_endpoint_policy"] == "co_primary_raw_headroom"
    assert gate["h12_endpoint_policy_mixed"] is False
    assert gate["h12_endpoint_policy_observed_counts"]["co_primary_raw_headroom"] == 7
    assert gate["h12_endpoint_adjudication"]["policy"] == "co_primary_raw_headroom"
    assert out.decision_payload["h12_endpoint_policy"] == "co_primary_raw_headroom"
    assert out.decision_payload["h12_endpoint_adjudication"]["policy"] == "co_primary_raw_headroom"


def test_headroom_confound_flag_trips_on_direction_mismatch(tmp_path) -> None:
    phase_root = tmp_path / "phase2b" / "run_headroom"
    phase_root.mkdir(parents=True, exist_ok=True)
    short_tasks = ["local_copy_offset", "local_key_match"]
    long_tasks = ["delayed_copy", "long_range_retrieval"]
    task_rows = []
    kernel_rows = []
    for seed in range(7):
        for task in short_tasks + long_tasks:
            base_acc = 0.90 if task in short_tasks else 0.15
            task_rows.append(
                {
                    "model": "llama-3.2-1b",
                    "task": task,
                    "seq_len": 1024,
                    "seed": seed,
                    "split": "synthetic",
                    "intervention": "none",
                    "mean_accuracy": base_acc,
                    "mean_nll": 0.5,
                    "floor_limited": False,
                    "chance_accuracy": 0.10,
                    "candidate_count": 10,
                }
            )
            for intervention, drop in (
                ("ablate_high_strong", 0.08 if task in short_tasks else 0.02),
                ("ablate_low_strong", 0.05 if task in short_tasks else 0.04),
                ("random_strong", 0.02),
            ):
                task_rows.append(
                    {
                        "model": "llama-3.2-1b",
                        "task": task,
                        "seq_len": 1024,
                        "seed": seed,
                        "split": "synthetic",
                        "intervention": intervention,
                        "mean_accuracy": base_acc - drop,
                        "mean_nll": 0.7,
                        "floor_limited": False,
                        "chance_accuracy": 0.10,
                        "candidate_count": 10,
                    }
                )
            for intervention, r2 in (
                ("none", 0.60),
                ("ablate_high_strong", 0.48),
                ("ablate_low_strong", 0.49),
                ("random_strong", 0.53),
            ):
                kernel_rows.append(
                    {
                        "model": "llama-3.2-1b",
                        "task": task,
                        "seq_len": 1024,
                        "seed": seed,
                        "split": "synthetic",
                        "metric": "track_b_raw",
                        "layer": 0,
                        "head": 0,
                        "intervention": intervention,
                        "mean_r2": r2,
                    }
                )

    pd.DataFrame(task_rows).to_parquet(phase_root / "aggregate_task_metrics.parquet", engine="pyarrow", index=False)
    pd.DataFrame(kernel_rows).to_parquet(phase_root / "aggregate_kernel_metrics.parquet", engine="pyarrow", index=False)
    gate = evaluate_phase(phase_root).gate_payload
    assert gate["headroom_confound_risk"] is True
    diag = gate["headroom_normalized_diagnostic"]
    assert diag["headroom_normalized"]["h2"]["raw_direction_agree"] is False


def test_phase2b_cluster_any_evaluable_or_semantics(tmp_path) -> None:
    phase_root = tmp_path / "phase2b" / "run_cluster_or"
    phase_root.mkdir(parents=True, exist_ok=True)
    models = [f"model{i}" for i in range(5)]
    short_tasks = ["local_copy_offset", "local_key_match"]
    long_tasks = ["delayed_copy", "long_range_retrieval"]
    task_rows = []
    kernel_rows = []
    for model in models:
        seed = 0
        for task in short_tasks + long_tasks:
            task_rows.append(
                {
                    "model": model,
                    "task": task,
                    "seq_len": 1024,
                    "seed": seed,
                    "split": "synthetic",
                    "intervention": "none",
                    "mean_accuracy": 0.90,
                    "mean_nll": 0.5,
                    "floor_limited": False,
                }
            )
            high_acc = 0.74 if task in short_tasks else 0.84
            rand_acc = 0.82
            for intervention, acc in (
                ("ablate_high_strong", high_acc),
                ("random_strong", rand_acc),
            ):
                task_rows.append(
                    {
                        "model": model,
                        "task": task,
                        "seq_len": 1024,
                        "seed": seed,
                        "split": "synthetic",
                        "intervention": intervention,
                        "mean_accuracy": acc,
                        "mean_nll": 0.7,
                        "floor_limited": False,
                    }
                )
        for task in short_tasks + long_tasks:
            for intervention, r2 in (
                ("none", 0.60),
                ("ablate_high_strong", 0.45 if task in short_tasks else 0.52),
                ("random_strong", 0.54),
            ):
                kernel_rows.append(
                    {
                        "model": model,
                        "task": task,
                        "seq_len": 1024,
                        "seed": seed,
                        "split": "synthetic",
                        "metric": "track_b_raw",
                        "layer": 0,
                        "head": 0,
                        "intervention": intervention,
                        "mean_r2": r2,
                    }
                )
    pd.DataFrame(task_rows).to_parquet(phase_root / "aggregate_task_metrics.parquet", engine="pyarrow", index=False)
    pd.DataFrame(kernel_rows).to_parquet(phase_root / "aggregate_kernel_metrics.parquet", engine="pyarrow", index=False)
    out = evaluate_phase(phase_root)
    gate = out.gate_payload
    assert gate["cluster_h1_evaluable"] is True
    assert gate["cluster_h2_evaluable"] is False
    assert gate["cluster_any_evaluable"] is True
    assert gate["cluster_sensitivity"]["evaluable"] is True


def test_phase2b_cross_task_sign_adapts_when_single_long_task_present(tmp_path) -> None:
    phase_root = tmp_path / "phase2b" / "run_single_long"
    phase_root.mkdir(parents=True, exist_ok=True)
    models = ["olmo-2-7b", "llama-3.1-8b", "gemma-7b"]
    short_tasks = ["local_copy_offset", "local_key_match"]
    long_tasks = ["long_range_retrieval"]
    task_rows = []
    kernel_rows = []
    for model in models:
        for seed in range(7):
            for task in short_tasks + long_tasks:
                base_acc = 0.90
                task_rows.append(
                    {
                        "model": model,
                        "task": task,
                        "seq_len": 1024,
                        "seed": seed,
                        "split": "synthetic",
                        "intervention": "none",
                        "mean_accuracy": base_acc,
                        "mean_nll": 0.5,
                        "floor_limited": False,
                    }
                )
                high_acc = 0.70 if task in short_tasks else 0.80
                low_acc = 0.84 if task in short_tasks else 0.72
                rand_acc = 0.82
                for intervention, acc in (
                    ("ablate_high_strong", high_acc),
                    ("ablate_low_strong", low_acc),
                    ("random_strong", rand_acc),
                ):
                    task_rows.append(
                        {
                            "model": model,
                            "task": task,
                            "seq_len": 1024,
                            "seed": seed,
                            "split": "synthetic",
                            "intervention": intervention,
                            "mean_accuracy": acc,
                            "mean_nll": 0.7,
                            "floor_limited": False,
                        }
                    )
                for intervention, r2 in (
                    ("none", 0.60),
                    ("ablate_high_strong", 0.46 if task in short_tasks else 0.50),
                    ("ablate_low_strong", 0.52 if task in short_tasks else 0.44),
                    ("random_strong", 0.54),
                ):
                    kernel_rows.append(
                        {
                            "model": model,
                            "task": task,
                            "seq_len": 1024,
                            "seed": seed,
                            "split": "synthetic",
                            "metric": "track_b_raw",
                            "layer": 0,
                            "head": 0,
                            "intervention": intervention,
                            "mean_r2": r2,
                        }
                    )
    pd.DataFrame(task_rows).to_parquet(phase_root / "aggregate_task_metrics.parquet", engine="pyarrow", index=False)
    pd.DataFrame(kernel_rows).to_parquet(phase_root / "aggregate_kernel_metrics.parquet", engine="pyarrow", index=False)
    gate = evaluate_phase(phase_root).gate_payload
    assert gate["active_long_tasks"] == ["long_range_retrieval"]
    assert gate["cross_task_sign_policy"] == "adaptive_by_active_long_task_count"
    assert gate["cross_task_sign_rule_version"] == "v2_single_long_adaptive"
    assert gate["criteria"]["cross_task_agreement_2_of_3"] is True


def test_phase2b_single_model_marked_exploratory_only(tmp_path) -> None:
    phase_root = tmp_path / "phase2b" / "single_model"
    phase_root.mkdir(parents=True, exist_ok=True)
    model = "llama-3.1-8b"
    short_tasks = ["local_copy_offset", "local_key_match"]
    long_tasks = ["long_range_retrieval"]
    task_rows = []
    kernel_rows = []
    for seed in range(7):
        for task in short_tasks + long_tasks:
            task_rows.append(
                {
                    "model": model,
                    "task": task,
                    "seq_len": 1024,
                    "seed": seed,
                    "split": "synthetic",
                    "intervention": "none",
                    "mean_accuracy": 0.9,
                    "mean_nll": 0.5,
                    "floor_limited": False,
                }
            )
            for intervention, acc in (
                ("ablate_high_strong", 0.70 if task in short_tasks else 0.82),
                ("ablate_low_strong", 0.84 if task in short_tasks else 0.72),
                ("random_strong", 0.82),
            ):
                task_rows.append(
                    {
                        "model": model,
                        "task": task,
                        "seq_len": 1024,
                        "seed": seed,
                        "split": "synthetic",
                        "intervention": intervention,
                        "mean_accuracy": acc,
                        "mean_nll": 0.7,
                        "floor_limited": False,
                    }
                )
            for intervention, r2 in (
                ("none", 0.60),
                ("ablate_high_strong", 0.46 if task in short_tasks else 0.50),
                ("ablate_low_strong", 0.52 if task in short_tasks else 0.44),
                ("random_strong", 0.54),
            ):
                kernel_rows.append(
                    {
                        "model": model,
                        "task": task,
                        "seq_len": 1024,
                        "seed": seed,
                        "split": "synthetic",
                        "metric": "track_b_raw",
                        "layer": 0,
                        "head": 0,
                        "intervention": intervention,
                        "mean_r2": r2,
                    }
                )
    pd.DataFrame(task_rows).to_parquet(phase_root / "aggregate_task_metrics.parquet", engine="pyarrow", index=False)
    pd.DataFrame(kernel_rows).to_parquet(phase_root / "aggregate_kernel_metrics.parquet", engine="pyarrow", index=False)
    out = evaluate_phase(phase_root)
    gate = out.gate_payload
    assert gate["observed_model_count"] == 1
    assert gate["confirmatory_applicability"] is False
    assert gate["exploratory_reason"] == "single_model_design"
    assert gate["confirmatory_promotion_eligible"] is False
    assert out.decision_payload["confirmatory_applicability"] is False
    assert out.decision_payload["exploratory_reason"] == "single_model_design"


def test_phase2b_h3_point_gate_can_pass_when_ci_diagnostic_fails(monkeypatch, tmp_path) -> None:
    phase_root = tmp_path / "phase2b" / "runx"
    phase_root.mkdir(parents=True, exist_ok=True)
    models = ["olmo-1b", "llama-3.2-1b", "tinyllama-1.1b"]
    short_tasks = ["local_copy_offset", "local_key_match"]
    long_tasks = ["delayed_copy", "long_range_retrieval"]
    task_rows = []
    kernel_rows = []
    for model in models:
        for seed in range(7):
            for task in short_tasks + long_tasks:
                task_rows.append(
                    {
                        "model": model,
                        "task": task,
                        "seq_len": 1024,
                        "seed": seed,
                        "split": "synthetic",
                        "intervention": "none",
                        "mean_accuracy": 0.90,
                        "mean_nll": 0.5,
                        "floor_limited": False,
                    }
                )
                high_acc = 0.70 if task in short_tasks else 0.85
                low_acc = 0.85 if task in short_tasks else 0.70
                rand_acc = 0.82
                for intervention, acc in (
                    ("ablate_high_strong", high_acc),
                    ("ablate_low_strong", low_acc),
                    ("random_strong", rand_acc),
                ):
                    task_rows.append(
                        {
                            "model": model,
                            "task": task,
                            "seq_len": 1024,
                            "seed": seed,
                            "split": "synthetic",
                            "intervention": intervention,
                            "mean_accuracy": acc,
                            "mean_nll": 0.7,
                            "floor_limited": False,
                        }
                    )
            for task in short_tasks + long_tasks:
                for intervention, r2 in (
                    ("none", 0.60),
                    ("ablate_high_strong", 0.45 if task in short_tasks else 0.52),
                    ("ablate_low_strong", 0.52 if task in short_tasks else 0.45),
                    ("random_strong", 0.54),
                ):
                    kernel_rows.append(
                        {
                            "model": model,
                            "task": task,
                            "seq_len": 1024,
                            "seed": seed,
                            "split": "synthetic",
                            "metric": "track_b_raw",
                            "layer": 0,
                            "head": 0,
                            "intervention": intervention,
                            "mean_r2": r2,
                        }
                    )
    pd.DataFrame(task_rows).to_parquet(phase_root / "aggregate_task_metrics.parquet", engine="pyarrow", index=False)
    pd.DataFrame(kernel_rows).to_parquet(phase_root / "aggregate_kernel_metrics.parquet", engine="pyarrow", index=False)

    from experiment2 import analysis as analysis_mod

    orig_precision = analysis_mod.precision_label

    def _patched_precision(*, effect, ci_low, threshold, direction="positive"):
        if float(threshold) == 0.0:
            return "imprecise_pass"
        return orig_precision(effect=effect, ci_low=ci_low, threshold=threshold, direction=direction)

    monkeypatch.setattr("experiment2.analysis.precision_label", _patched_precision)
    out = evaluate_phase(phase_root)
    gate = out.gate_payload
    assert gate["h3_rate_point"] >= 0.5
    assert np.isfinite(gate["h3_rate_ci"]) and gate["h3_rate_ci"] < 0.5
    assert gate["criteria"]["kernel_specificity_50pct"] is True
    assert gate["h3_ci_diagnostic_flag"] is True


def test_phase2c_requires_7_seed_ready_for_promotion(tmp_path) -> None:
    run_id = "runx"
    phase2b_root = tmp_path / "phase2b" / run_id
    phase2b_root.mkdir(parents=True, exist_ok=True)
    (phase2b_root / "gate_evaluation.json").write_text('{"confirmatory_success": true}\n', encoding="utf-8")

    phase2c_root = tmp_path / "phase2c" / run_id
    phase2c_root.mkdir(parents=True, exist_ok=True)
    task_rows = []
    for seed in range(5):
        task_rows.append(
            {
                "model": "olmo-1b",
                "task": "local_copy_offset",
                "seq_len": 256,
                "seed": seed,
                "split": "synthetic",
                "intervention": "none",
                "mean_accuracy": 0.9,
                "mean_nll": 0.5,
                "floor_limited": False,
            }
        )
    pd.DataFrame(task_rows).to_parquet(phase2c_root / "aggregate_task_metrics.parquet", engine="pyarrow", index=False)
    pd.DataFrame().to_parquet(phase2c_root / "aggregate_kernel_metrics.parquet", engine="pyarrow", index=False)
    out = evaluate_phase(phase2c_root)
    gate = out.gate_payload
    assert gate["phase2b_confirmatory_pass_observed"] is True
    assert gate["headline_claim_requires_7_seed_rerun"] is True
    assert gate["headline_7_seed_ready"] is False
    assert gate["confirmatory_promotion_eligible"] is False


def test_phase2b_confirmatory_requires_pooled_significance(monkeypatch, tmp_path) -> None:
    phase_root = tmp_path / "phase2b" / "runx"
    phase_root.mkdir(parents=True, exist_ok=True)
    models = ["olmo-1b", "llama-3.2-1b", "tinyllama-1.1b"]
    short_tasks = ["local_copy_offset", "local_key_match"]
    long_tasks = ["delayed_copy", "long_range_retrieval"]
    task_rows = []
    kernel_rows = []
    for model in models:
        for seed in range(7):
            for task in short_tasks + long_tasks:
                task_rows.append(
                    {
                        "model": model,
                        "task": task,
                        "seq_len": 1024,
                        "seed": seed,
                        "split": "synthetic",
                        "intervention": "none",
                        "mean_accuracy": 0.90,
                        "mean_nll": 0.5,
                        "floor_limited": False,
                    }
                )
                high_acc = 0.74 if task in short_tasks else 0.82
                low_acc = 0.82 if task in short_tasks else 0.74
                rand_acc = 0.80
                for intervention, acc in (
                    ("ablate_high_strong", high_acc),
                    ("ablate_low_strong", low_acc),
                    ("random_strong", rand_acc),
                ):
                    task_rows.append(
                        {
                            "model": model,
                            "task": task,
                            "seq_len": 1024,
                            "seed": seed,
                            "split": "synthetic",
                            "intervention": intervention,
                            "mean_accuracy": acc,
                            "mean_nll": 0.7,
                            "floor_limited": False,
                        }
                    )
            for task in short_tasks + long_tasks:
                for intervention, r2 in (
                    ("none", 0.60),
                    ("ablate_high_strong", 0.44 if task in short_tasks else 0.52),
                    ("ablate_low_strong", 0.52 if task in short_tasks else 0.44),
                    ("random_strong", 0.53),
                ):
                    kernel_rows.append(
                        {
                            "model": model,
                            "task": task,
                            "seq_len": 1024,
                            "seed": seed,
                            "split": "synthetic",
                            "metric": "track_b_raw",
                            "layer": 0,
                            "head": 0,
                            "intervention": intervention,
                            "mean_r2": r2,
                        }
                    )
    pd.DataFrame(task_rows).to_parquet(phase_root / "aggregate_task_metrics.parquet", engine="pyarrow", index=False)
    pd.DataFrame(kernel_rows).to_parquet(phase_root / "aggregate_kernel_metrics.parquet", engine="pyarrow", index=False)

    monkeypatch.setattr("experiment2.analysis.paired_sign_flip_pvalue", lambda values, max_random_flips=100_000: 0.80)
    out = evaluate_phase(phase_root)
    gate = out.gate_payload
    assert gate["criteria"]["pooled_primary_inference_holm_0_05"] is False
    assert gate["confirmatory_success"] is False


def test_run_manifest_seed_filter_execute_range(monkeypatch, tmp_path) -> None:
    if run_manifest is None or ExecutionConfig is None:
        pytest.skip("experiment2.execution import unavailable in this environment")
    manifest = tmp_path / "manifest.jsonl"
    rows = []
    for seed in range(7):
        rows.append(
            {
                "phase": "phase2a",
                "split": "synthetic",
                "model": "llama-3.2-1b",
                "task": "local_key_match",
                "seq_len": 1024,
                "intervention": "none",
                "seed": seed,
                "device": "cuda",
                "run_id": "runx",
            }
        )
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    seen: list[int] = []

    def _fake_execute(row, config, state):
        seen.append(int(row["seed"]))
        return True

    monkeypatch.setattr("experiment2.execution._execute_cell", _fake_execute)
    monkeypatch.setattr("experiment2.execution._aggregate_phase_outputs", lambda *args, **kwargs: None)

    cfg = ExecutionConfig(
        device="cpu",
        output_root=tmp_path / "out",
        seed_start=2,
        seed_end=4,
    )
    out = run_manifest(manifest, cfg)
    assert out["processed"] == 3
    assert sorted(seen) == [2, 3, 4]


def test_batch_size_heuristic_for_long_tasks() -> None:
    if _batch_size_for_row is None or ExecutionConfig is None:
        pytest.skip("experiment2.execution import unavailable in this environment")
    cfg = ExecutionConfig(device="cpu", output_root=Path("."), batch_size_synth=8, batch_size_tier1=8)
    long_row = {"split": "synthetic", "task": "delayed_copy", "seq_len": 1024}
    short_row = {"split": "synthetic", "task": "local_key_match", "seq_len": 1024}
    scaleup_phase2a_row = {
        "phase": "phase2a",
        "split": "synthetic",
        "task": "local_key_match",
        "seq_len": 1024,
        "model": "llama-3.1-8b",
    }
    scaleup_phase2b_row = {
        "phase": "phase2b",
        "split": "synthetic",
        "task": "local_key_match",
        "seq_len": 1024,
        "model": "llama-3.1-8b",
    }
    assert _batch_size_for_row(long_row, cfg) == 2
    assert _batch_size_for_row(short_row, cfg) == 8
    assert _batch_size_for_row(scaleup_phase2a_row, cfg) == 1
    assert _batch_size_for_row(scaleup_phase2b_row, cfg) == 1


def _phase2a_capture_test_harness(
    monkeypatch,
    tmp_path,
    *,
    intervention: str,
    phase: str = "phase2a",
    track_a_enabled: bool = True,
) -> tuple[list[dict[str, bool]], list[str]]:
    from contextlib import nullcontext

    if _execute_cell is None or ExecutionConfig is None or _RunState is None:
        pytest.skip("experiment2.execution import unavailable in this environment")
    from experiment2.flooring import FloorDecision
    from experiment2.interventions import build_rope_intervention_plan

    class _DummyAdapter:
        def __init__(self) -> None:
            self.calls: list[dict[str, bool]] = []

        def register(self, model) -> None:
            return None

        def cleanup(self) -> None:
            return None

        def capture(
            self,
            model,
            *,
            input_ids,
            attention_mask,
            include_logits,
            return_token_logits,
            output_device,
            capture_attention=True,
            **kwargs,
        ):
            self.calls.append(
                {
                    "include_logits": bool(include_logits),
                    "capture_attention": bool(capture_attention),
                    "return_token_logits": bool(return_token_logits),
                }
            )
            bsz, seq_len = input_ids.shape
            vocab = 64
            token_logits = torch.zeros((bsz, seq_len, vocab), dtype=torch.float32)
            return SimpleNamespace(token_logits=token_logits, logits=None)

    kernel_updates: list[str] = []

    class _DummyKernelAccum:
        def __init__(self, *args, **kwargs) -> None:
            self.track_a = []
            self.track_b_raw = []
            self.track_b_centered = []
            self._sum_q = None
            self._sum_k = None
            self._count_tokens = 0

        def update_track_a_raw(self, capture) -> None:
            kernel_updates.append("track_a")

        def update_track_b_raw_from_qk(self, capture) -> None:
            kernel_updates.append("track_b_raw")

        def accumulate_shared_means(self, capture) -> None:
            kernel_updates.append("shared")

        def update_centered(self, capture, shared_q_mean=None, shared_k_mean=None) -> None:
            kernel_updates.append("centered")

        def finalize_shared_means(self):
            return torch.zeros((1,), dtype=torch.float32), torch.zeros((1,), dtype=torch.float32)

        def to_dataframes(self, **kwargs):
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    adapter = _DummyAdapter()
    monkeypatch.setattr(
        "experiment2.execution._get_cached_resources",
        lambda row, config, state: SimpleNamespace(
            model_spec=SimpleNamespace(name=row["model"]),
            model=object(),
            tokenizer=object(),
            adapter=adapter,
            pools=object(),
        ),
    )
    monkeypatch.setattr("experiment2.execution.model_tokenizer_hash", lambda model_spec, model, tokenizer: "mh")
    monkeypatch.setattr(
        "experiment2.execution.provenance_payload",
        lambda **kwargs: {"generator_hash": "g", "intervention_hash": "i", "model_tokenizer_hash": "m"},
    )
    monkeypatch.setattr("experiment2.execution._maybe_reuse_from_phase2a", lambda **kwargs: False)
    monkeypatch.setattr(
        "experiment2.execution.load_long_offset_bundle",
        lambda strict=False: SimpleNamespace(lock_hash="lockhash", long_offsets=(8, 12), fallback_spans=(8,)),
    )
    monkeypatch.setattr(
        "experiment2.execution._resolve_floor_decision",
        lambda **kwargs: FloorDecision(
            key="k",
            baseline_accuracy=0.4,
            fallback_accuracy=None,
            fallback_applied=False,
            floor_limited=False,
            fallback_spans=tuple(),
        ),
    )
    example = TaskExample(
        id="toy:phase2a",
        task_name="local_key_match",
        tokens=[10, 11, 12, 13],
        target_positions=[2],
        target_tokens=[12],
        dependency_span=1,
        task_class="short",
        seed=0,
        model="llama-3.1-8b",
        length=4,
        task_params={},
        pair_count=None,
        query_key=None,
        distractor_key=None,
        match_rule="toy",
        has_no_match=False,
    )
    monkeypatch.setattr("experiment2.execution._load_or_build_examples", lambda **kwargs: [example])
    monkeypatch.setattr(
        "experiment2.execution._build_intervention",
        lambda model, model_spec, row, results_root, run_id, cache: (
            build_rope_intervention_plan(head_dim=8, intervention=str(row.get("intervention")), seed=int(row.get("seed", 0))),
            nullcontext(),
        ),
    )
    monkeypatch.setattr("experiment2.execution.KernelMetricsAccumulator", _DummyKernelAccum)
    monkeypatch.setattr(
        "experiment2.execution._evaluate_example_from_token_logits",
        lambda *args, **kwargs: (
            {
                "accuracy": 1.0,
                "mean_nll": 0.0,
                "full_vocab_accuracy": 1.0,
                "full_vocab_mean_nll": 0.0,
                "restricted_accuracy": 1.0,
                "restricted_mean_nll": 0.0,
                "num_targets": 1,
            },
            [{"position": 2, "nll": 0.0}],
        ),
    )

    row = {
        "phase": phase,
        "split": "synthetic",
        "model": "llama-3.1-8b",
        "task": "local_key_match",
        "seq_len": 1024,
        "intervention": intervention,
        "seed": 0,
        "random_draw": None,
        "run_id": f"run_{phase}_{intervention}",
    }
    cfg = ExecutionConfig(
        device="cpu",
        output_root=tmp_path / "results" / "experiment2",
        centered_compute="defer",
        track_a_enabled=bool(track_a_enabled),
    )
    state = _RunState()
    out = _execute_cell(row=row, config=cfg, state=state)
    assert out is True
    return adapter.calls, kernel_updates


def test_phase2a_nonbaseline_capture_pruned_for_speed(monkeypatch, tmp_path) -> None:
    calls, kernel_updates = _phase2a_capture_test_harness(
        monkeypatch,
        tmp_path,
        intervention="ablate_high_strong",
        phase="phase2a",
    )
    assert calls
    assert all(call["include_logits"] is False for call in calls)
    assert all(call["capture_attention"] is False for call in calls)
    assert all(call["return_token_logits"] is True for call in calls)
    assert kernel_updates == []


def test_phase2a_baseline_keeps_capture_and_kernel_updates(monkeypatch, tmp_path) -> None:
    calls, kernel_updates = _phase2a_capture_test_harness(
        monkeypatch,
        tmp_path,
        intervention="none",
        phase="phase2a",
    )
    assert calls
    assert any(call["include_logits"] is True for call in calls)
    assert any(call["capture_attention"] is True for call in calls)
    assert "track_a" in kernel_updates


def test_phase2b_nonbaseline_path_unchanged(monkeypatch, tmp_path) -> None:
    calls, kernel_updates = _phase2a_capture_test_harness(
        monkeypatch,
        tmp_path,
        intervention="ablate_high_strong",
        phase="phase2b",
    )
    assert calls
    assert any(call["include_logits"] is True for call in calls)
    assert any(call["capture_attention"] is True for call in calls)
    assert "track_a" in kernel_updates


def test_phase2b_track_a_disabled_keeps_track_b_raw_without_attention_logits(monkeypatch, tmp_path) -> None:
    calls, kernel_updates = _phase2a_capture_test_harness(
        monkeypatch,
        tmp_path,
        intervention="ablate_high_strong",
        phase="phase2b",
        track_a_enabled=False,
    )
    assert calls
    assert all(call["include_logits"] is False for call in calls)
    assert any(call["capture_attention"] is True for call in calls)
    assert "track_b_raw" in kernel_updates
    assert "track_a" not in kernel_updates


def test_quick_baseline_uses_adapter_capture_path() -> None:
    if _quick_baseline_accuracy is None or ExecutionConfig is None:
        pytest.skip("experiment2.execution import unavailable in this environment")

    pools = TokenPools(
        filler=tuple(range(20, 220)),
        keys=tuple(range(220, 320)),
        values=tuple(range(320, 420)),
        reserve=(420, 421, 422, 423, 424),
        query_marker=420,
        retrieval_marker=421,
        no_match_token=422,
    )
    row = {
        "split": "synthetic",
        "task": "local_copy_offset",
        "model": "toy-model",
        "seq_len": 64,
        "seed": 0,
    }

    class _DummyModel:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("Direct model forward should not be used in quick baseline path.")

    class _DummyAdapter:
        def __init__(self) -> None:
            self.capture_calls = 0
            self.output_devices: list[str] = []
            self.capture_attention_flags: list[bool] = []

        def register(self, model) -> None:
            return None

        def cleanup(self) -> None:
            return None

        def capture(
            self,
            model,
            *,
            input_ids,
            attention_mask,
            include_logits,
            return_token_logits,
            output_device,
            capture_attention=True,
            **kwargs,
        ):
            self.capture_calls += 1
            self.output_devices.append(str(output_device))
            self.capture_attention_flags.append(bool(capture_attention))
            bsz, seq_len = input_ids.shape
            vocab = 512
            logits = torch.zeros((bsz, seq_len, vocab), dtype=torch.float32, device="cpu")
            for b in range(bsz):
                for pos in range(1, seq_len):
                    tok = int(input_ids[b, pos].item())
                    if 0 <= tok < vocab:
                        logits[b, pos - 1, tok] = 10.0
            return SimpleNamespace(token_logits=logits)

    cfg = ExecutionConfig(
        device="cpu",
        output_root=Path("."),
        synthetic_count=4,
        synthetic_eval_mode="restricted",
        candidate_size=10,
        batch_size_synth=2,
    )
    adapter = _DummyAdapter()
    acc = _quick_baseline_accuracy(
        row=row,
        config=cfg,
        model=_DummyModel(),
        adapter=adapter,
        pools=pools,
        span_choices=tuple(),
    )
    assert adapter.capture_calls > 0
    assert set(adapter.output_devices) == {"cpu"}
    assert set(adapter.capture_attention_flags) == {False}
    assert 0.0 <= float(acc) <= 1.0


def test_accuracy_only_baseline_match_full_evaluator_for_restricted() -> None:
    if _evaluate_example_accuracy_only is None or _evaluate_example_from_token_logits is None:
        pytest.skip("experiment2.execution import unavailable in this environment")

    pools = TokenPools(
        filler=tuple(range(20, 220)),
        keys=tuple(range(220, 320)),
        values=tuple(range(320, 420)),
        reserve=(420, 421, 422, 423, 424),
        query_marker=420,
        retrieval_marker=421,
        no_match_token=422,
    )
    example = generate_task_examples(
        task_name="long_range_retrieval",
        model_name="llama-3.2-1b",
        seq_len=512,
        seed=1,
        count=1,
        pools=pools,
        span_override=12,
    )[0]
    vocab = 512
    logits = torch.zeros((len(example.tokens), vocab), dtype=torch.float32)
    for pos, tok in zip(example.target_positions, example.target_tokens):
        if 0 < pos < len(example.tokens):
            logits[pos - 1, int(tok)] = 4.0

    acc_only = _evaluate_example_accuracy_only(
        logits,
        example,
        split="synthetic",
        pools=pools,
        synthetic_eval_mode="restricted",
        candidate_size=10,
        candidate_policy_version="restricted_candidates_v1_structured_first",
    )
    metrics, _ = _evaluate_example_from_token_logits(
        logits,
        example,
        split="synthetic",
        pools=pools,
        synthetic_eval_mode="restricted",
        candidate_size=10,
        candidate_policy_version="restricted_candidates_v1_structured_first",
    )
    assert float(acc_only) == pytest.approx(float(metrics["accuracy"]), abs=1e-12)


def test_phase2a_reuse_disabled_by_default(tmp_path) -> None:
    if (
        ExecutionConfig is None
        or _RunState is None
        or _condition_dir is None
        or _maybe_reuse_from_phase2a is None
    ):
        pytest.skip("experiment2.execution import unavailable in this environment")

    output_root = tmp_path / "results" / "experiment2"
    run_id = "runx"
    row = {
        "phase": "phase2b",
        "split": "synthetic",
        "model": "llama-3.2-1b",
        "task": "local_key_match",
        "seq_len": 1024,
        "intervention": "ablate_high_strong",
        "random_draw": None,
        "seed": 0,
        "run_id": run_id,
    }
    cfg = ExecutionConfig(device="cpu", output_root=output_root, allow_phase2a_reuse=False)
    out_dir = output_root / "phase2b" / run_id / "dummy_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    state = _RunState()
    reused = _maybe_reuse_from_phase2a(
        row=row,
        out_dir=out_dir,
        config=cfg,
        provenance={"generator_hash": "g", "intervention_hash": "i", "model_tokenizer_hash": "m"},
        state=state,
    )
    assert reused is False
    assert state.reuse_events
    assert state.reuse_events[-1]["reason"] == "reuse_disabled_by_config"


def test_phase2a_reuse_config_mismatch_forces_rerun(tmp_path) -> None:
    if (
        ExecutionConfig is None
        or _RunState is None
        or _condition_dir is None
        or _maybe_reuse_from_phase2a is None
    ):
        pytest.skip("experiment2.execution import unavailable in this environment")

    output_root = tmp_path / "results" / "experiment2"
    run_id = "runx"
    row = {
        "phase": "phase2b",
        "split": "synthetic",
        "model": "llama-3.2-1b",
        "task": "local_key_match",
        "seq_len": 1024,
        "intervention": "ablate_high_strong",
        "random_draw": None,
        "seed": 0,
        "run_id": run_id,
    }
    src_row = dict(row)
    src_row["phase"] = "phase2a"
    src_dir = _condition_dir(output_root, src_row)
    src_dir.mkdir(parents=True, exist_ok=True)
    src_payload = {
        "candidate_size": 11,
        "synthetic_eval_mode": "restricted",
        "candidate_policy_version": "restricted_candidates_v1_structured_first",
        "floor_threshold": 0.15,
        "generator_hash": "g",
        "intervention_hash": "i",
        "model_tokenizer_hash": "m",
    }
    (src_dir / "run_config.json").write_text(json.dumps(src_payload), encoding="utf-8")
    out_dir = output_root / "phase2b" / run_id / "dummy_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = ExecutionConfig(
        device="cpu",
        output_root=output_root,
        allow_phase2a_reuse=True,
        candidate_size=10,
        synthetic_eval_mode="restricted",
        candidate_policy_version="restricted_candidates_v1_structured_first",
        floor_threshold=0.15,
    )
    state = _RunState()
    reused = _maybe_reuse_from_phase2a(
        row=row,
        out_dir=out_dir,
        config=cfg,
        provenance={"generator_hash": "g", "intervention_hash": "i", "model_tokenizer_hash": "m"},
        state=state,
    )
    assert reused is False
    assert state.reuse_events
    event = state.reuse_events[-1]
    assert event["reason"] == "config_mismatch"
    assert "candidate_size" in event["mismatch_fields"]


def test_execute_cell_prunes_floor_limited_nonbaseline(monkeypatch, tmp_path) -> None:
    if _execute_cell is None or ExecutionConfig is None or _RunState is None:
        pytest.skip("experiment2.execution import unavailable in this environment")
    from experiment2.flooring import FloorDecision

    row = {
        "phase": "phase2b",
        "split": "synthetic",
        "model": "llama-3.2-1b",
        "task": "delayed_copy",
        "seq_len": 1024,
        "intervention": "ablate_high_strong",
        "seed": 0,
        "random_draw": None,
        "run_id": "run_floor_prune",
    }
    cfg = ExecutionConfig(
        device="cpu",
        output_root=tmp_path / "results" / "experiment2",
        prune_floor_limited_interventions=True,
    )
    state = _RunState()

    monkeypatch.setattr(
        "experiment2.execution._get_cached_resources",
        lambda row, config, state: SimpleNamespace(model_spec=SimpleNamespace(name=row["model"]), model=object(), tokenizer=object(), adapter=object(), pools=object()),
    )
    monkeypatch.setattr("experiment2.execution.model_tokenizer_hash", lambda model_spec, model, tokenizer: "mh")
    monkeypatch.setattr("experiment2.execution.provenance_payload", lambda **kwargs: {"generator_hash": "g", "intervention_hash": "i", "model_tokenizer_hash": "m"})
    monkeypatch.setattr("experiment2.execution._maybe_reuse_from_phase2a", lambda **kwargs: False)
    monkeypatch.setattr(
        "experiment2.execution.load_long_offset_bundle",
        lambda strict=False: SimpleNamespace(lock_hash="lockhash", long_offsets=(8, 12), fallback_spans=(8,)),
    )
    monkeypatch.setattr(
        "experiment2.execution._resolve_floor_decision",
        lambda **kwargs: FloorDecision(
            key="k",
            baseline_accuracy=0.12,
            fallback_accuracy=None,
            fallback_applied=False,
            floor_limited=True,
            fallback_spans=tuple(),
        ),
    )

    out = _execute_cell(row=row, config=cfg, state=state)
    assert out is True
    out_dir = _condition_dir(cfg.output_root, row)
    run_cfg = json.loads((out_dir / "run_config.json").read_text(encoding="utf-8"))
    assert run_cfg["execution_status"] == "skipped_floor_limited"
    assert run_cfg["centered_pending"] is False
    assert (out_dir / "skip_record.json").exists()
    assert not (out_dir / "task_metrics.parquet").exists()


def test_aggregate_writes_execution_coverage_and_gate_floor_prune_note(monkeypatch, tmp_path) -> None:
    if _aggregate_phase_outputs is None:
        pytest.skip("experiment2.execution import unavailable in this environment")

    run_id = "run_cov"
    phase = "phase2b"
    phase_root = tmp_path / "results" / "experiment2" / phase / run_id
    phase_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = [
        {"phase": phase, "split": "synthetic", "model": "llama-3.2-1b", "task": "local_key_match", "seq_len": 1024, "intervention": "none", "seed": 0, "run_id": run_id},
        {"phase": phase, "split": "synthetic", "model": "llama-3.2-1b", "task": "local_key_match", "seq_len": 1024, "intervention": "ablate_high_strong", "seed": 0, "run_id": run_id},
    ]
    (phase_root / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in manifest_rows) + "\n", encoding="utf-8")

    for idx, status in enumerate(["executed", "skipped_floor_limited"]):
        out_dir = phase_root / "llama-3.2-1b" / "local_key_match" / "len_1024" / ("none" if idx == 0 else "ablate_high_strong") / f"seed_{idx}"
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "row": {
                "phase": phase,
                "split": "synthetic",
                "model": "llama-3.2-1b",
                "task": "local_key_match",
                "seq_len": 1024,
                "intervention": "none" if idx == 0 else "ablate_high_strong",
                "seed": idx,
                "run_id": run_id,
            },
            "execution_status": status,
            "centered_pending": False,
        }
        (out_dir / "run_config.json").write_text(json.dumps(payload), encoding="utf-8")

    def _fake_analysis(phase_root_path):
        gate_path = Path(phase_root_path) / "gate_evaluation.json"
        gate_path.write_text(json.dumps({"phase": phase, "confirmatory_success": False}), encoding="utf-8")
        (Path(phase_root_path) / "decision_summary.json").write_text(json.dumps({"phase": phase}), encoding="utf-8")

    monkeypatch.setattr("experiment2.execution.write_phase_analysis", _fake_analysis)
    monkeypatch.setattr("experiment2.execution.write_norm_family_decomposition", lambda phase_root_path: None)
    monkeypatch.setattr("experiment2.execution.write_h4_interaction_exploratory", lambda phase_root_path: None)
    monkeypatch.setattr("experiment2.execution.write_protocol_revision_log", lambda phase_root_path: None)
    monkeypatch.setattr("experiment2.execution.write_claim_guard", lambda phase_root_path: None)
    monkeypatch.setattr("experiment2.execution._write_promotion_guard", lambda output_root, run_id: None)

    _aggregate_phase_outputs(tmp_path / "results" / "experiment2", phase=phase, run_id=run_id, reuse_events=[])
    coverage = json.loads((phase_root / "execution_coverage.json").read_text(encoding="utf-8"))
    assert coverage["manifest_rows"] == 2
    assert coverage["run_config_rows"] == 2
    assert coverage["executed_rows"] == 1
    assert coverage["skipped_floor_limited_rows"] == 1
    assert coverage["missing_rows"] == 0
    gate = json.loads((phase_root / "gate_evaluation.json").read_text(encoding="utf-8"))
    assert gate["floor_pruned_rows_non_evaluable"] == 1
