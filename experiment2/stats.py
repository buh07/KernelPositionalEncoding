from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EffectInference:
    effect: float
    ci_low: float
    ci_high: float
    p_value: float
    p_value_holm: float | None = None


def paired_deltas(baseline: np.ndarray, condition: np.ndarray) -> np.ndarray:
    if baseline.shape != condition.shape:
        raise ValueError(f"Mismatched paired arrays: {baseline.shape} vs {condition.shape}")
    return condition - baseline


def _safe_norm_ppf(p: float) -> float:
    # Acklam rational approximation.
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def _safe_norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    return _safe_norm_ppf(p)


def bootstrap_bca_ci(values: np.ndarray, *, b: int = 20_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return float("nan"), float("nan")
    theta_hat = float(np.mean(x))
    if n == 1:
        return theta_hat, theta_hat

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(b, n))
    boot = np.mean(x[idx], axis=1)

    p_less = np.clip(np.mean(boot < theta_hat), 1e-10, 1 - 1e-10)
    z0 = _safe_norm_ppf(float(p_less))

    jack = np.empty(n, dtype=np.float64)
    for i in range(n):
        jack[i] = np.mean(np.delete(x, i))
    jack_mean = float(np.mean(jack))
    num = float(np.sum((jack_mean - jack) ** 3))
    den = float(np.sum((jack_mean - jack) ** 2))
    if den <= 0:
        acc = 0.0
    else:
        acc = num / (6.0 * (den ** 1.5))

    z_alpha1 = _safe_norm_ppf(alpha / 2)
    z_alpha2 = _safe_norm_ppf(1 - alpha / 2)

    def adj(z: float) -> float:
        denom = 1 - acc * (z0 + z)
        if abs(denom) < 1e-12:
            return float(np.clip(_safe_norm_cdf(z0), 0.0, 1.0))
        return float(np.clip(_safe_norm_cdf(z0 + (z0 + z) / denom), 0.0, 1.0))

    a1 = adj(z_alpha1)
    a2 = adj(z_alpha2)
    lo = float(np.quantile(boot, a1))
    hi = float(np.quantile(boot, a2))
    return lo, hi


def bootstrap_bca_ci_sensitivity(
    values: np.ndarray,
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    b: int = 2_000,
    alpha: float = 0.05,
) -> dict[str, float | int | bool]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "seeds_evaluated": 0,
            "ci_low_min": float("nan"),
            "ci_low_max": float("nan"),
            "ci_high_min": float("nan"),
            "ci_high_max": float("nan"),
            "ci_low_spread": float("nan"),
            "ci_high_spread": float("nan"),
            "ci_seed_sensitivity_flag": True,
        }
    lows: list[float] = []
    highs: list[float] = []
    for seed in seeds:
        lo, hi = bootstrap_bca_ci(x, b=b, alpha=alpha, seed=int(seed))
        lows.append(float(lo))
        highs.append(float(hi))
    ci_low_min = float(np.min(lows))
    ci_low_max = float(np.max(lows))
    ci_high_min = float(np.min(highs))
    ci_high_max = float(np.max(highs))
    low_spread = float(ci_low_max - ci_low_min)
    high_spread = float(ci_high_max - ci_high_min)
    # Conservative stability heuristic for reporting only.
    flag = bool((low_spread > 0.01) or (high_spread > 0.01))
    return {
        "seeds_evaluated": int(len(seeds)),
        "ci_low_min": ci_low_min,
        "ci_low_max": ci_low_max,
        "ci_high_min": ci_high_min,
        "ci_high_max": ci_high_max,
        "ci_low_spread": low_spread,
        "ci_high_spread": high_spread,
        "ci_seed_sensitivity_flag": flag,
    }


def paired_sign_flip_pvalue(values: np.ndarray, *, max_random_flips: int = 100_000) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    x = x[x != 0.0]
    n = x.size
    if n == 0:
        return 1.0
    t_obs = abs(float(np.mean(x)))

    if n <= 18:
        ge = 0
        total = 0
        for bits in itertools.product([-1.0, 1.0], repeat=n):
            t = abs(float(np.mean(x * np.asarray(bits))))
            ge += int(t >= t_obs - 1e-15)
            total += 1
        return ge / max(total, 1)

    rng = np.random.default_rng(0)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(max_random_flips, n), replace=True)
    t = np.abs(np.mean(signs * x.reshape(1, -1), axis=1))
    return float(np.mean(t >= t_obs - 1e-15))


def holm_bonferroni(p_values: dict[str, float], *, alpha: float = 0.05) -> dict[str, float]:
    valid = [(k, v) for k, v in p_values.items() if v == v]
    ordered = sorted(valid, key=lambda kv: kv[1])
    m = len(ordered)
    adjusted: dict[str, float] = {k: float("nan") for k in p_values}
    prev = 0.0
    for i, (k, p) in enumerate(ordered):
        adj = min(1.0, (m - i) * p)
        adj = max(adj, prev)
        adjusted[k] = adj
        prev = adj
    return adjusted


def mde_two_sided_normal(
    *,
    sd: float,
    n: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Approximate minimum detectable mean effect for paired contrasts.

    Uses a normal-approximation design equation:
      MDE = (z_{1-alpha/2} + z_power) * sd / sqrt(n)
    """
    if not np.isfinite(sd) or sd < 0 or n <= 0:
        return float("nan")
    z_alpha = norm_ppf(1.0 - alpha / 2.0)
    z_power = norm_ppf(power)
    if not np.isfinite(z_alpha) or not np.isfinite(z_power):
        return float("nan")
    return float((z_alpha + z_power) * sd / math.sqrt(float(n)))


def precision_label(*, effect: float, ci_low: float, threshold: float, direction: str = "positive") -> str:
    if not np.isfinite(effect) or not np.isfinite(ci_low):
        return "fail"
    if direction == "positive":
        if effect < threshold:
            return "fail"
        if ci_low > 0:
            return "pass"
        return "imprecise_pass"
    if direction == "negative":
        if effect > -threshold:
            return "fail"
        if ci_low < 0:
            return "pass"
        return "imprecise_pass"
    raise ValueError(f"Unsupported direction: {direction}")
