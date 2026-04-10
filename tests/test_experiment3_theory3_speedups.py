from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import experiment3.theory3_crossterm_correlation as theory3


def test_batched_diagonal_means_matches_legacy_per_head() -> None:
    torch.manual_seed(0)
    logits = torch.randn(3, 5, 17, 17, dtype=torch.float32)
    max_delta = logits.shape[-1] - 1

    batched = theory3.extract_diagonal_means_batched(logits, max_delta).cpu().numpy()
    legacy = np.zeros_like(batched, dtype=np.float64)

    for layer_idx in range(logits.shape[0]):
        for head_idx in range(logits.shape[1]):
            legacy[layer_idx, head_idx] = theory3.extract_diagonal_means(
                logits[layer_idx, head_idx], max_delta
            )

    assert np.allclose(batched, legacy, atol=1e-12, rtol=1e-12)


def test_prefix_causal_energy_matches_masked_matmul() -> None:
    torch.manual_seed(0)
    q_pair = torch.randn(4, 6, 19, 2, dtype=torch.float32)
    k_pair = torch.randn(4, 6, 19, 2, dtype=torch.float32)

    prefix_energy = theory3.strict_causal_pair_energy_prefix(q_pair, k_pair)

    seq_len = q_pair.shape[2]
    strict_causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=-1)
    baseline = torch.zeros(q_pair.shape[0], q_pair.shape[1], dtype=torch.float32)
    for layer_idx in range(q_pair.shape[0]):
        for head_idx in range(q_pair.shape[1]):
            contrib = torch.matmul(q_pair[layer_idx, head_idx], k_pair[layer_idx, head_idx].T)
            baseline[layer_idx, head_idx] = (contrib[strict_causal] ** 2).sum()

    assert torch.allclose(prefix_energy, baseline, atol=1e-4, rtol=1e-6)


def test_resume_action_resolution_default_reuse_both_present() -> None:
    actions = theory3.resolve_artifact_actions(
        gap_exists=True,
        total_exists=True,
        skip_phase1=False,
        force_recompute_gap_kernels=False,
        force_recompute_total_energy=False,
    )
    assert actions == {
        "load_gap": True,
        "compute_gap": False,
        "load_total": True,
        "compute_total": False,
        "need_model": False,
    }


def test_resume_action_resolution_gap_only_present() -> None:
    actions = theory3.resolve_artifact_actions(
        gap_exists=True,
        total_exists=False,
        skip_phase1=False,
        force_recompute_gap_kernels=False,
        force_recompute_total_energy=False,
    )
    assert actions["load_gap"] is True
    assert actions["compute_gap"] is False
    assert actions["load_total"] is False
    assert actions["compute_total"] is True
    assert actions["need_model"] is True


def test_resume_action_resolution_no_artifacts() -> None:
    actions = theory3.resolve_artifact_actions(
        gap_exists=False,
        total_exists=False,
        skip_phase1=False,
        force_recompute_gap_kernels=False,
        force_recompute_total_energy=False,
    )
    assert actions["load_gap"] is False
    assert actions["compute_gap"] is True
    assert actions["load_total"] is False
    assert actions["compute_total"] is True
    assert actions["need_model"] is True


def test_resume_action_resolution_force_flags() -> None:
    actions = theory3.resolve_artifact_actions(
        gap_exists=True,
        total_exists=True,
        skip_phase1=False,
        force_recompute_gap_kernels=True,
        force_recompute_total_energy=True,
    )
    assert actions["load_gap"] is False
    assert actions["compute_gap"] is True
    assert actions["load_total"] is False
    assert actions["compute_total"] is True
    assert actions["need_model"] is True


def test_resume_action_resolution_skip_phase1_requires_gap() -> None:
    with pytest.raises(FileNotFoundError):
        theory3.resolve_artifact_actions(
            gap_exists=False,
            total_exists=True,
            skip_phase1=True,
            force_recompute_gap_kernels=False,
            force_recompute_total_energy=False,
        )


def test_resume_action_resolution_skip_phase1_conflicts_with_force_gap() -> None:
    with pytest.raises(ValueError):
        theory3.resolve_artifact_actions(
            gap_exists=True,
            total_exists=True,
            skip_phase1=True,
            force_recompute_gap_kernels=True,
            force_recompute_total_energy=False,
        )
