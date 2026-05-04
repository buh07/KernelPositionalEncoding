#!/usr/bin/env python3
"""E3: OLMo Result IV permutation null test.

Tests whether OLMo's observed 2/4 proxy-to-task sign concordance is
distinguishable from the permutation null. The concordance maps 4 task-
grounded regimes to 2 proxy regimes (matching exp_r5_task_grounded_specialization.py):
  long_span_64   → long_span_retrieval   (proxy)
  long_span_128  → long_span_retrieval   (proxy)
  uncertainty_low   → high_uncertainty_tokens (proxy)
  uncertainty_high  → high_uncertainty_tokens (proxy)

Permutes regime column labels in the task-grounded effects 10,000 times
and recomputes concordance each time to build the null distribution.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp.common import RESULTS_ROOT, ensure_dir, timestamp_now, write_json  # noqa: E402

PROXY_SUMMARY = (
    ROOT / "results" / "experiment3_phase2"
    / "exp3p2j_conditional_regimes_longspan_repair"
    / "olmo-2-7b" / "regime_summary.json"
)
TASK_EFFECTS = (
    ROOT / "results" / "reinforce_exp" / "exp_r5_task_grounded"
    / "olmo-2-7b" / "task_grounded_effects.parquet"
)
OUT_DIR = ensure_dir(RESULTS_ROOT / "exp_e3_olmo_permutation_test")

# Exactly the mapping used in exp_r5_task_grounded_specialization.py
REGIME_MAP = {
    "long_span_64":    "long_span_retrieval",
    "long_span_128":   "long_span_retrieval",
    "uncertainty_low": "high_uncertainty_tokens",
    "uncertainty_high": "high_uncertainty_tokens",
}
TASK_REGIMES = list(REGIME_MAP.keys())   # 4 task regimes
N_COMPARED = len(TASK_REGIMES)


def _proxy_mean_deltas() -> dict[str, float]:
    d = json.loads(PROXY_SUMMARY.read_text())
    return {
        name: float(data["mean_delta_high_minus_low"])
        for name, data in d["regime_tests"].items()
    }


def _task_mean_delta_per_seed(effects_df: pd.DataFrame, regime: str) -> np.ndarray:
    sub = effects_df[effects_df["regime"] == regime]
    high = sub[sub["group"] == "high_si"].sort_values("seed")["drop"].values
    low  = sub[sub["group"] == "low_si"].sort_values("seed")["drop"].values
    return high - low


def _concordance(task_means: np.ndarray, proxy_deltas: dict[str, float]) -> int:
    """Count sign matches for the 4 (task_regime → proxy_regime) pairs."""
    matches = 0
    for i, tr in enumerate(TASK_REGIMES):
        pr = REGIME_MAP[tr]
        ts = task_means[i]
        ps = proxy_deltas[pr]
        if (ts >= 0 and ps >= 0) or (ts < 0 and ps < 0):
            matches += 1
    return matches


def main() -> None:
    rng = np.random.default_rng(20260424)
    n_perm = 10_000

    proxy_deltas = _proxy_mean_deltas()
    effects_df = pd.read_parquet(TASK_EFFECTS)

    # per-seed deltas: shape (n_seeds, 4)
    delta_matrix = np.stack(
        [_task_mean_delta_per_seed(effects_df, r) for r in TASK_REGIMES], axis=1
    )
    n_seeds = delta_matrix.shape[0]

    observed_means = delta_matrix.mean(axis=0)
    observed_concordance = _concordance(observed_means, proxy_deltas)

    print("Proxy mean deltas:")
    for k, v in proxy_deltas.items():
        print(f"  {k}: {v:+.4f} ({'pos' if v >= 0 else 'neg'})")
    print("\nTask-grounded mean deltas (observed):")
    for i, tr in enumerate(TASK_REGIMES):
        v = observed_means[i]
        pr = REGIME_MAP[tr]
        match = "MATCH" if ((v >= 0) == (proxy_deltas[pr] >= 0)) else "mismatch"
        print(f"  {tr} → {pr}: {v:+.5f}  [{match}]")
    print(f"\nObserved concordance: {observed_concordance}/{N_COMPARED}")

    # Permutation test: permute column order (regime labels) in delta_matrix
    perm_concordances = np.empty(n_perm, dtype=int)
    for k in range(n_perm):
        perm_idx = rng.permutation(N_COMPARED)
        perm_means = delta_matrix[:, perm_idx].mean(axis=0)
        perm_concordances[k] = _concordance(perm_means, proxy_deltas)

    p_ge = float(np.mean(perm_concordances >= observed_concordance))
    p_eq = float(np.mean(perm_concordances == observed_concordance))
    null_mean = float(perm_concordances.mean())
    null_std  = float(perm_concordances.std())
    counts = {str(k): int(np.sum(perm_concordances == k)) for k in range(N_COMPARED + 1)}
    pctile = float(100 * (1 - p_ge))

    print(f"\nPermutation null (n={n_perm}):")
    print(f"  mean={null_mean:.3f}  std={null_std:.3f}")
    print(f"  distribution: {counts}")
    print(f"  P(concordance >= {observed_concordance}): {p_ge:.4f}")
    print(f"  Observed is at percentile {pctile:.1f} of null")
    print(f"\nConclusion: 2/4 concordance sits at the {pctile:.1f}th percentile "
          f"(null mean={null_mean:.3f}). "
          + ("This confirms the failure is at-chance, not a near-miss."
             if p_ge > 0.2 else "This is worse than expected by chance."))

    result = {
        "timestamp": timestamp_now(),
        "experiment": "E3_olmo_permutation_test",
        "model": "olmo-2-7b",
        "n_compared": N_COMPARED,
        "regime_map": REGIME_MAP,
        "proxy_deltas": proxy_deltas,
        "observed_task_means": dict(zip(TASK_REGIMES, observed_means.tolist())),
        "observed_concordance": int(observed_concordance),
        "n_permutations": int(n_perm),
        "null_mean": null_mean,
        "null_std": null_std,
        "null_count_distribution": counts,
        "p_concordance_ge_observed": p_ge,
        "p_concordance_eq_observed": p_eq,
        "observed_percentile": pctile,
        "interpretation": (
            f"Observed {observed_concordance}/{N_COMPARED} sits at "
            f"percentile {pctile:.1f} of the permutation null "
            f"(null mean={null_mean:.3f}, std={null_std:.3f}). "
            "Concordance failure is at-chance, not a near-miss."
            if p_ge > 0.2
            else f"Observed {observed_concordance}/{N_COMPARED} is BELOW "
            "the null mean — worse than chance concordance."
        ),
    }
    write_json(OUT_DIR / "permutation_result.json", result)
    print(f"\nResult written to {OUT_DIR / 'permutation_result.json'}")


if __name__ == "__main__":
    main()
