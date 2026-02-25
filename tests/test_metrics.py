from __future__ import annotations

import torch

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiment1.track_a import _shift_metrics
from experiment1.track_b import DiagonalStats
from experiment1.shift_kernels import get_kernel_estimator


def test_shift_metrics_perfect_shift_invariance():
    seq_len = 4
    matrix = torch.zeros(seq_len, seq_len)
    for t in range(seq_len):
        for s in range(seq_len):
            if t > s:
                matrix[t, s] = float(t - s)
    g_values, r2, sse, sst = _shift_metrics(matrix)
    assert len(g_values) == seq_len - 1
    assert abs(r2 - 1.0) < 1e-6
    assert sse < 1e-6
    assert sst > 0


def test_diagonal_stats_r2():
    stats = DiagonalStats(seq_len=5, estimator=get_kernel_estimator("learned-absolute"))
    matrix = torch.zeros(5, 5)
    for delta in range(1, 5):
        for t in range(delta, 5):
            s = t - delta
            matrix[t, s] = float(delta * 2)
    stats.update(matrix)
    fit = stats.fit()
    assert abs(fit.r2 - 1.0) < 1e-6
    means = fit.g_values
    assert len(means) == 4
    assert all(abs(mean - (idx + 1) * 2) < 1e-6 for idx, mean in enumerate(means))
