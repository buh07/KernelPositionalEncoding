from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiment3.stats_utils import (
    dependent_corr_williams_test,
    partial_correlation_with_intercept,
    validate_paired_samples,
)


def test_partial_correlation_with_intercept_matches_manual_residualization() -> None:
    rng = np.random.default_rng(7)
    n = 300
    z = rng.normal(loc=5.0, scale=2.0, size=n)
    u = rng.normal(size=n)
    eps = rng.normal(scale=0.3, size=n)

    x = 10.0 + 1.8 * z + u
    y = -3.0 + 2.2 * z + u + eps

    result = partial_correlation_with_intercept(x, y, z)
    assert result.df == n - 3

    design = np.column_stack([np.ones(n), z])
    beta_x, *_ = np.linalg.lstsq(design, x, rcond=None)
    beta_y, *_ = np.linalg.lstsq(design, y, rcond=None)
    x_resid = x - design @ beta_x
    y_resid = y - design @ beta_y
    manual_r = np.corrcoef(x_resid, y_resid)[0, 1]

    assert np.isclose(result.r, manual_r, atol=1e-12)
    assert result.p_value < 1e-10


def test_dependent_corr_williams_detects_difference() -> None:
    rng = np.random.default_rng(123)
    n = 500
    x = rng.normal(size=n)
    y_strong = x + rng.normal(scale=0.25, size=n)
    y_weak = x + rng.normal(scale=1.5, size=n)

    r_xy = float(np.corrcoef(x, y_strong)[0, 1])
    r_xz = float(np.corrcoef(x, y_weak)[0, 1])
    r_yz = float(np.corrcoef(y_strong, y_weak)[0, 1])

    res = dependent_corr_williams_test(r_xy=r_xy, r_xz=r_xz, r_yz=r_yz, n=n)
    assert res.statistic > 0
    assert res.p_value < 1e-6
    assert res.df == n - 3


def test_validate_paired_samples_guardrails() -> None:
    with pytest.raises(ValueError):
        validate_paired_samples(np.array([1.0, 2.0]), np.array([1.0]))

    with pytest.raises(ValueError):
        validate_paired_samples(
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            sample_ids_x=["a", "b"],
            sample_ids_y=["a", "c"],
        )

    # This should not raise.
    validate_paired_samples(
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0]),
        sample_ids_x=["a", "b"],
        sample_ids_y=["a", "b"],
    )
