from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Protocol, TypeGuard, cast

import numpy as np
import numpy.typing as npt
import torch
from scipy import stats as scipy_stats

from si_rebuttal.controls import make_pcg64

DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_SPEARMAN_PERMUTATIONS = 200_000
FloatArray = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]


@dataclass(frozen=True)
class BootstrapResult:
    point_estimate: float
    ci_low: float
    ci_high: float
    samples: FloatArray


@dataclass(frozen=True)
class SpearmanPermutationResult:
    rho_observed: float
    p_one_sided: float
    p_two_sided_scipy: float
    permutation_count: int
    permutation_exceedance_count: int


class _SpearmanResult(Protocol):
    statistic: object
    pvalue: object


class _SpearmanrFn(Protocol):
    def __call__(
        self, a: Sequence[float] | npt.ArrayLike, b: Sequence[float] | npt.ArrayLike
    ) -> _SpearmanResult: ...


class _RankdataFn(Protocol):
    def __call__(
        self, a: Sequence[float] | npt.ArrayLike, *, method: str = "average"
    ) -> Sequence[float] | npt.ArrayLike: ...


class _IntegerGenerator(Protocol):
    def integers(
        self,
        low: int,
        high: int | None = None,
        *,
        size: tuple[int, int],
        endpoint: bool = False,
        dtype: type[np.int64],
    ) -> Int64Array: ...


def _is_spearman_result(value: object) -> TypeGuard[_SpearmanResult]:
    return hasattr(value, "statistic") and hasattr(value, "pvalue")


def _read_spearman_result(value: object) -> tuple[float, float]:
    if not _is_spearman_result(value):
        raise ValueError("observed Spearman result is missing statistic or pvalue.")
    statistic = value.statistic
    pvalue = value.pvalue
    if not isinstance(statistic, Real):
        raise ValueError("observed Spearman statistic is not numeric.")
    if not isinstance(pvalue, Real):
        raise ValueError("observed Spearman p-value is not numeric.")
    return float(statistic), float(pvalue)


def _as_finite_1d(values: Sequence[float] | npt.ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape [n].")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def average_control_trials_per_sequence(
    trials: Sequence[Sequence[float]] | npt.ArrayLike,
) -> FloatArray:
    values = np.asarray(trials, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("trials must have shape [num_trials, num_sequences].")
    if values.shape[0] == 0:
        raise ValueError("trials must include at least one control trial.")
    if not np.isfinite(values).all():
        raise ValueError("trials contain non-finite values.")
    return values.mean(axis=0)


def paired_percentile_bootstrap(
    baseline: Sequence[float] | npt.ArrayLike,
    condition: Sequence[float] | npt.ArrayLike,
    *,
    seed: int,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> BootstrapResult:
    baseline_values = _as_finite_1d(baseline, name="baseline")
    condition_values = _as_finite_1d(condition, name="condition")
    if baseline_values.shape != condition_values.shape:
        raise ValueError("baseline and condition must have the same shape.")
    rng = cast(_IntegerGenerator, make_pcg64(seed))
    n = baseline_values.shape[0]
    indices = rng.integers(0, n, size=(int(n_resamples), n), endpoint=False, dtype=np.int64)
    deltas = condition_values[indices] - baseline_values[indices]
    samples = deltas.mean(axis=1)
    point = float(np.mean(condition_values - baseline_values))
    percentiles = np.asarray(
        np.percentile(samples, [2.5, 97.5], method="linear"),
        dtype=np.float64,
    )
    ci_low = float(percentiles[0])
    ci_high = float(percentiles[1])
    return BootstrapResult(
        point_estimate=point,
        ci_low=ci_low,
        ci_high=ci_high,
        samples=samples,
    )


def _pearson_corr_rows(
    x_centered: FloatArray, x_norm: float, y_rank_rows: FloatArray
) -> FloatArray:
    y_centered = y_rank_rows - y_rank_rows.mean(axis=1, keepdims=True)
    y_norm = np.linalg.norm(y_centered, axis=1)
    if np.any(y_norm == 0.0):
        raise ValueError("permuted response ranks produced zero norm.")
    numerators = y_centered @ x_centered
    return numerators / (x_norm * y_norm)


def _permuted_rank_rows(
    rng: np.random.Generator,
    values: FloatArray,
    *,
    batch_size: int,
) -> FloatArray:
    permutations = np.stack(
        [rng.permutation(values.shape[0]) for _ in range(int(batch_size))], axis=0
    )
    return values[permutations]


def monte_carlo_spearman_positive(
    x: Sequence[float] | npt.ArrayLike,
    y: Sequence[float] | npt.ArrayLike,
    *,
    seed: int,
    n_permutations: int = DEFAULT_SPEARMAN_PERMUTATIONS,
    chunk_size: int = 5_000,
) -> SpearmanPermutationResult:
    x_values = _as_finite_1d(x, name="x")
    y_values = _as_finite_1d(y, name="y")
    if x_values.shape != y_values.shape:
        raise ValueError("x and y must have the same shape.")
    spearmanr = cast(_SpearmanrFn, scipy_stats.spearmanr)
    rankdata = cast(_RankdataFn, scipy_stats.rankdata)
    rho_observed, p_two_sided = _read_spearman_result(spearmanr(x_values, y_values))
    if not np.isfinite(rho_observed) or not np.isfinite(p_two_sided):
        raise ValueError("observed Spearman statistic is non-finite.")
    x_ranks = np.asarray(rankdata(x_values, method="average"), dtype=np.float64)
    y_ranks = np.asarray(rankdata(y_values, method="average"), dtype=np.float64)
    x_centered = x_ranks - x_ranks.mean()
    x_norm = float(np.linalg.norm(x_centered))
    if x_norm == 0.0:
        raise ValueError("x ranks have zero norm.")
    rng = make_pcg64(seed)
    exceedances = 0
    remaining = int(n_permutations)
    while remaining > 0:
        batch = min(int(chunk_size), remaining)
        permuted_y = _permuted_rank_rows(rng, y_ranks, batch_size=batch)
        rho_batch = _pearson_corr_rows(x_centered, x_norm, permuted_y)
        if not np.isfinite(rho_batch).all():
            raise ValueError("permuted Spearman statistics are non-finite.")
        exceedances += int(np.count_nonzero(rho_batch >= rho_observed))
        remaining -= batch
    p_one_sided = float((1 + exceedances) / (int(n_permutations) + 1))
    return SpearmanPermutationResult(
        rho_observed=rho_observed,
        p_one_sided=p_one_sided,
        p_two_sided_scipy=p_two_sided,
        permutation_count=int(n_permutations),
        permutation_exceedance_count=exceedances,
    )


def reconstruct_attention_probabilities(
    q: torch.Tensor,
    k: torch.Tensor,
    additive_mask: torch.Tensor,
) -> torch.Tensor:
    if q.ndim != 2 or k.ndim != 2:
        raise ValueError(f"Expected q and k to be rank-2, got {q.ndim=} {k.ndim=}.")
    if q.shape != k.shape:
        raise ValueError(f"q and k must share shape, got {tuple(q.shape)} vs {tuple(k.shape)}.")
    if additive_mask.shape != (q.shape[0], q.shape[0]):
        raise ValueError(
            "additive_mask must match the query/key sequence dimensions: "
            f"mask={tuple(additive_mask.shape)} seq={q.shape[0]}."
        )
    if torch.isnan(additive_mask).any() or torch.isposinf(additive_mask).any():
        raise ValueError("Manual reconstruction requires finite values or -inf in additive_mask.")
    if not (torch.isfinite(q).all() and torch.isfinite(k).all()):
        raise ValueError("Manual reconstruction requires finite q and k tensors.")
    scale = 1.0 / math.sqrt(q.shape[-1])
    logits = torch.matmul(q.to(torch.float32), k.to(torch.float32).transpose(0, 1)) * scale
    logits = logits + additive_mask.to(torch.float32)
    probs = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probs).all():
        raise ValueError("Manual reconstruction produced non-finite probabilities.")
    return probs
