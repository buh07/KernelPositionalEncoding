from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import stats as scipy_stats


@dataclass(frozen=True)
class PartialCorrelationResult:
    r: float
    p_value: float
    n: int
    df: int


@dataclass(frozen=True)
class DependentCorrelationTestResult:
    statistic: float
    p_value: float
    df: int
    method: str


def _as_float_1d(values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D input, got shape={arr.shape}")
    return arr


def residualize_with_intercept(
    y: Sequence[float] | np.ndarray,
    covariates: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Residualize y on covariates using OLS with an intercept."""
    y_arr = _as_float_1d(y)
    cov_arr = np.asarray(covariates, dtype=np.float64)
    if cov_arr.ndim == 1:
        cov_arr = cov_arr.reshape(-1, 1)
    if cov_arr.ndim != 2:
        raise ValueError(f"Expected 1D/2D covariates, got shape={cov_arr.shape}")
    if cov_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"Mismatched lengths: y={y_arr.shape[0]} covariates={cov_arr.shape[0]}"
        )

    x = np.column_stack([np.ones(cov_arr.shape[0], dtype=np.float64), cov_arr])
    beta, *_ = np.linalg.lstsq(x, y_arr, rcond=None)
    fitted = x @ beta
    return y_arr - fitted


def partial_correlation_with_intercept(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    covariates: Sequence[float] | np.ndarray,
) -> PartialCorrelationResult:
    """Partial Pearson correlation r(x, y | covariates).

    Uses intercept-based residualization and computes p-value with
    df = n - k - 2 (k = number of control covariates).
    """
    x_arr = _as_float_1d(x)
    y_arr = _as_float_1d(y)
    cov_arr = np.asarray(covariates, dtype=np.float64)

    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(f"Mismatched lengths: x={len(x_arr)} y={len(y_arr)}")

    if cov_arr.ndim == 1:
        k = 1
    elif cov_arr.ndim == 2:
        k = cov_arr.shape[1]
    else:
        raise ValueError(f"Expected 1D/2D covariates, got shape={cov_arr.shape}")

    n = x_arr.shape[0]
    df = n - k - 2
    if n < 4 or df <= 0:
        return PartialCorrelationResult(r=float("nan"), p_value=float("nan"), n=n, df=df)

    x_resid = residualize_with_intercept(x_arr, cov_arr)
    y_resid = residualize_with_intercept(y_arr, cov_arr)

    if np.std(x_resid) < 1e-12 or np.std(y_resid) < 1e-12:
        return PartialCorrelationResult(r=float("nan"), p_value=float("nan"), n=n, df=df)

    r = float(np.corrcoef(x_resid, y_resid)[0, 1])
    r = max(min(r, 1.0), -1.0)

    if abs(r) >= 1.0:
        p = 0.0
    else:
        t_stat = r * math.sqrt(df / max(1e-30, 1.0 - r * r))
        p = float(2.0 * scipy_stats.t.sf(abs(t_stat), df))

    return PartialCorrelationResult(r=r, p_value=p, n=n, df=df)


def dependent_corr_williams_test(
    *,
    r_xy: float,
    r_xz: float,
    r_yz: float,
    n: int,
) -> DependentCorrelationTestResult:
    """Williams' test for difference between overlapping dependent correlations.

    Tests H0: corr(x, y) == corr(x, z), where y and z are measured on same n.
    """
    if n <= 3:
        return DependentCorrelationTestResult(
            statistic=float("nan"), p_value=float("nan"), df=n - 3, method="williams"
        )

    r_xy = float(np.clip(r_xy, -0.999999999999, 0.999999999999))
    r_xz = float(np.clip(r_xz, -0.999999999999, 0.999999999999))
    r_yz = float(np.clip(r_yz, -0.999999999999, 0.999999999999))

    k = 1.0 - r_xy * r_xy - r_xz * r_xz - r_yz * r_yz + 2.0 * r_xy * r_xz * r_yz
    if k <= 1e-30:
        return DependentCorrelationTestResult(
            statistic=float("nan"), p_value=float("nan"), df=n - 3, method="williams"
        )

    num = (r_xy - r_xz) * math.sqrt((n - 1.0) * (1.0 + r_yz))
    den_a = (2.0 * (n - 1.0) / (n - 3.0)) * k
    den_b = ((r_xy + r_xz) ** 2 / 4.0) * ((1.0 - r_yz) ** 3)
    den = math.sqrt(max(1e-30, den_a + den_b))

    t_stat = num / den
    df = n - 3
    p = float(2.0 * scipy_stats.t.sf(abs(t_stat), df))
    return DependentCorrelationTestResult(
        statistic=float(t_stat), p_value=p, df=df, method="williams"
    )


def fisher_z(r: float) -> float:
    r = float(np.clip(r, -0.999999999999, 0.999999999999))
    return float(np.arctanh(r))


def inverse_fisher_z(z: float) -> float:
    return float(np.tanh(z))


def fisher_z_mean(correlations: Iterable[float]) -> float:
    vals = [float(c) for c in correlations if np.isfinite(c)]
    if not vals:
        return float("nan")
    z_vals = np.array([fisher_z(v) for v in vals], dtype=np.float64)
    return inverse_fisher_z(float(np.mean(z_vals)))


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    valid = [(k, float(v)) for k, v in p_values.items() if np.isfinite(v)]
    ordered = sorted(valid, key=lambda kv: kv[1])
    m = len(ordered)
    adjusted: dict[str, float] = {k: float("nan") for k in p_values}
    prev = 0.0
    for i, (key, p_val) in enumerate(ordered):
        adj = min(1.0, (m - i) * p_val)
        adj = max(adj, prev)
        adjusted[key] = adj
        prev = adj
    return adjusted


def fdr_bh_adjust(p_values: dict[str, float]) -> dict[str, float]:
    valid = [(k, float(v)) for k, v in p_values.items() if np.isfinite(v)]
    ordered = sorted(valid, key=lambda kv: kv[1])
    m = len(ordered)
    adjusted: dict[str, float] = {k: float("nan") for k in p_values}
    if m == 0:
        return adjusted

    q_vals = np.zeros(m, dtype=np.float64)
    for i, (_, p_val) in enumerate(ordered, start=1):
        q_vals[i - 1] = p_val * m / i
    q_vals = np.minimum.accumulate(q_vals[::-1])[::-1]
    q_vals = np.clip(q_vals, 0.0, 1.0)

    for (key, _), q in zip(ordered, q_vals):
        adjusted[key] = float(q)
    return adjusted


def validate_paired_samples(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    *,
    sample_ids_x: Sequence[str] | Sequence[int] | None = None,
    sample_ids_y: Sequence[str] | Sequence[int] | None = None,
) -> None:
    """Guardrail for paired tests.

    Raises ValueError for unequal length or mismatched sample identity.
    """
    x_arr = _as_float_1d(x)
    y_arr = _as_float_1d(y)
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(f"Paired samples require equal lengths: {len(x_arr)} vs {len(y_arr)}")

    if (sample_ids_x is None) ^ (sample_ids_y is None):
        raise ValueError("Provide both sample_ids_x and sample_ids_y for paired identity checks")

    if sample_ids_x is not None and sample_ids_y is not None:
        if len(sample_ids_x) != len(sample_ids_y):
            raise ValueError(
                f"Paired sample id arrays must match length: {len(sample_ids_x)} vs {len(sample_ids_y)}"
            )
        if list(sample_ids_x) != list(sample_ids_y):
            raise ValueError("Paired sample ids do not align")
