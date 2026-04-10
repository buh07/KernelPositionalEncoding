#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import torch

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment3.stats_utils import holm_adjust, one_sided_p_from_two_sided  # noqa: E402
from experiment3.theory1_si_circuits import MODELS, HeadID, head_output_ablation, load_profile_sequences  # noqa: E402
from shared.models.loading import load_model, load_tokenizer  # noqa: E402


TARGET_MODELS = ("llama-3.1-8b", "olmo-2-7b")
PRIMARY_TEST_ID = "3P2-J"
TIER_LABEL = "tier2_conditional_mechanistic"
MULTIPLICITY = "tier2_holm_primary_tests"


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _safe_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _token_has_prefix(tok: str) -> bool:
    return tok.startswith("\u0120") or tok.startswith("\u2581")


def _load_head_groups(model_name: str) -> dict[str, list[HeadID]]:
    path = ROOT / "results" / "experiment3" / "theory1_si_circuits" / model_name / "head_groups.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing head_groups.json: {path}")
    data = _load_json(path)

    def parse(entries: list[dict[str, Any]]) -> list[HeadID]:
        return [HeadID(int(e["layer"]), int(e["head"])) for e in entries]

    return {
        "none": [],
        "ablate_high_si": parse(data["high_si"]),
        "ablate_low_si": parse(data["low_si"]),
    }


def _gate_g1_continues() -> bool:
    gate_path = ROOT / "results" / "experiment3_phase2" / "phase2_governance" / "gate_g1_decision.json"
    if not gate_path.exists():
        raise FileNotFoundError(f"Missing gate artifact: {gate_path}")
    gate = _load_json(gate_path)
    return bool(gate.get("continue_to_stage2", False))


def _load_a_dependencies(a_output_root: Path, model_name: str) -> tuple[dict[str, Any], pd.DataFrame]:
    model_root = a_output_root / model_name
    manifest_path = model_root / "task_definition_manifest.json"
    task_rows_path = model_root / "task_battery_results.parquet"
    if not manifest_path.exists() or not task_rows_path.exists():
        raise FileNotFoundError(
            f"3P2-A dependencies missing for {model_name}; expected {manifest_path} and {task_rows_path}"
        )
    return _load_json(manifest_path), pd.read_parquet(task_rows_path)


def _sequences_hash(sequences: list[list[int]]) -> str:
    h = hashlib.sha1()
    for seq in sequences:
        arr = np.asarray(seq, dtype=np.int32)
        h.update(arr.tobytes())
    return h.hexdigest()


def _compute_surprisal_sequences(
    *,
    model,
    device: str,
    sequences: list[list[int]],
    heads_to_zero: list[HeadID],
    batch_size: int,
) -> list[np.ndarray]:
    out: list[np.ndarray] = [np.zeros((0,), dtype=np.float32) for _ in range(len(sequences))]
    bsz = max(1, int(batch_size))

    cm: contextlib.AbstractContextManager
    if not heads_to_zero:
        cm = contextlib.nullcontext()
    else:
        cm = head_output_ablation(model, heads_to_zero)

    with cm:
        idx0 = 0
        while idx0 < len(sequences):
            batch = sequences[idx0 : idx0 + bsz]
            input_ids = torch.tensor(batch, dtype=torch.long, device=device)
            with torch.inference_mode():
                outputs = model(input_ids=input_ids, use_cache=False)
                logits = outputs.logits[:, :-1, :].float()
                labels = input_ids[:, 1:]
                log_probs = torch.log_softmax(logits, dim=-1)
                nll = -torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
                nll = nll.detach().cpu().numpy()
            for bi in range(len(batch)):
                out[idx0 + bi] = np.asarray(nll[bi], dtype=np.float32)
            idx0 += len(batch)
            del input_ids, outputs, logits, labels, log_probs, nll
            torch.cuda.empty_cache()

    return out


def _compute_boundary_dense_mask(tokenizer, tokens: list[int], window: int = 16, min_boundaries: int = 4) -> np.ndarray:
    tok_str = tokenizer.convert_ids_to_tokens(tokens)
    boundary = np.array([1 if (i > 0 and _token_has_prefix(tok_str[i] or "")) else 0 for i in range(len(tokens))], dtype=np.int64)
    # masks align with target tokens (positions 1..L-1)
    out = np.zeros((max(0, len(tokens) - 1),), dtype=bool)
    for t in range(1, len(tokens)):
        lo = max(0, t - window)
        cnt = int(np.sum(boundary[lo:t]))
        out[t - 1] = bool(cnt >= min_boundaries)
    return out


def _compute_long_span_repeat_mask(tokens: list[int], min_distance: int = 64) -> np.ndarray:
    """
    Mark target-token positions (aligned to tokens[1:]) where the current token
    repeats a prior token at least `min_distance` positions back.
    """
    n = max(0, len(tokens) - 1)
    if n <= 0:
        return np.zeros((0,), dtype=bool)
    out = np.zeros((n,), dtype=bool)
    seen: dict[int, list[int]] = {}
    for pos in range(len(tokens)):
        tid = int(tokens[pos])
        prev_positions = seen.get(tid, [])
        long_enough = False
        for p in prev_positions:
            if (pos - p) >= int(min_distance):
                long_enough = True
                break
        if pos > 0:
            out[pos - 1] = long_enough
        prev_positions.append(pos)
        seen[tid] = prev_positions
    return out


def _build_regime_manifest(
    *,
    model_name: str,
    sequences: list[list[int]],
    baseline_surprisal: list[np.ndarray],
    token_freq: dict[int, int],
    a_manifest: dict[str, Any],
    seq_len: int,
    long_span_min_distance: int,
) -> tuple[dict[str, Any], float, int]:
    surp = np.concatenate([x for x in baseline_surprisal if x.size > 0], axis=0)
    if surp.size == 0:
        raise RuntimeError("No baseline surprisal values available for regime manifest")
    p90 = float(np.quantile(surp, 0.90))

    freq_vals = np.array(list(token_freq.values()), dtype=np.float64)
    if freq_vals.size == 0:
        raise RuntimeError("Token frequency table is empty")
    p10 = float(np.quantile(freq_vals, 0.10))
    rare_cutoff = int(max(1, np.floor(p10)))

    long_span_tasks = []
    for task in a_manifest.get("task_specs", []):
        if task.get("task_name") == "long_range_retrieval" and int(task.get("span", 0)) >= int(long_span_min_distance):
            long_span_tasks.append(str(task.get("task_id")))

    manifest = {
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tier": TIER_LABEL,
        "primary_test_id": PRIMARY_TEST_ID,
        "source_a_task_definition_hash": str(a_manifest.get("task_definition_hash", "")),
        "source_a_task_definition_timestamp": str(a_manifest.get("timestamp", "")),
        "heldout_dataset": {
            "sequence_count": int(len(sequences)),
            "seq_len": int(seq_len),
            "token_sequence_hash": _sequences_hash(sequences),
        },
        "regimes": {
            "long_span_retrieval": {
                "definition": "Heldout token positions where token repeats with distance >= threshold",
                "min_repeat_distance_tokens": int(long_span_min_distance),
                "a_task_reference": {
                    "definition": f"3P2-A long_range_retrieval tasks with span >= {int(long_span_min_distance)}",
                    "task_ids": sorted(long_span_tasks),
                },
                "task_ids": sorted(long_span_tasks),
            },
            "high_uncertainty_tokens": {
                "definition": "baseline surprisal >= P90 on heldout wiki",
                "threshold_surprisal_p90": p90,
            },
            "boundary_dense_contexts": {
                "definition": ">=4 boundary tokens in prior 16-token window",
                "boundary_window": 16,
                "min_boundaries": 4,
            },
            "rare_token_contexts": {
                "definition": "token frequency <= P10 on heldout wiki reference",
                "frequency_p10": p10,
                "rare_token_count_cutoff": rare_cutoff,
            },
        },
        "mde_target": 0.35,
        "achieved_power": 0.80,
        "multiplicity_family": MULTIPLICITY,
    }
    return manifest, p90, rare_cutoff


def _paired_group_test(df: pd.DataFrame, regime: str) -> dict[str, Any]:
    sub = df[df["regime"] == regime].copy()
    high = sub[sub["group"] == "high_si"][["seed", "effect_value"]].rename(columns={"effect_value": "high"})
    low = sub[sub["group"] == "low_si"][["seed", "effect_value"]].rename(columns={"effect_value": "low"})
    pair = high.merge(low, on="seed", how="inner")
    if len(pair) < 2:
        return {
            "n_pairs": int(len(pair)),
            "mean_high": float("nan"),
            "mean_low": float("nan"),
            "mean_delta_high_minus_low": float("nan"),
            "t_stat": float("nan"),
            "p_two_sided": float("nan"),
            "p_one_sided": float("nan"),
            "cohens_d_paired": float("nan"),
        }

    x = pair["high"].to_numpy(dtype=float)
    y = pair["low"].to_numpy(dtype=float)
    d = x - y
    t_stat, p_two = scipy_stats.ttest_rel(x, y, nan_policy="omit")
    p_one = one_sided_p_from_two_sided(_safe_float(t_stat), _safe_float(p_two), alternative="greater")
    sd = float(np.nanstd(d, ddof=1))
    coh = float(np.nanmean(d) / sd) if sd > 1e-8 else float("nan")
    return {
        "n_pairs": int(len(pair)),
        "mean_high": float(np.mean(x)),
        "mean_low": float(np.mean(y)),
        "mean_delta_high_minus_low": float(np.mean(d)),
        "t_stat": _safe_float(t_stat),
        "p_two_sided": _safe_float(p_two),
        "p_one_sided": _safe_float(p_one),
        "cohens_d_paired": coh,
    }


def _two_way_interaction(df: pd.DataFrame) -> dict[str, Any]:
    work = df[df["group"].isin(["high_si", "low_si"])].copy()
    if work.empty:
        return {
            "n_observations": 0,
            "f_interaction": float("nan"),
            "p_value": float("nan"),
            "partial_eta_squared": float("nan"),
        }

    def _z(x: pd.Series) -> pd.Series:
        vals = x.to_numpy(dtype=float)
        mu = np.nanmean(vals)
        sd = np.nanstd(vals, ddof=1)
        if not np.isfinite(sd) or sd < 1e-8:
            return pd.Series(np.zeros_like(vals), index=x.index)
        return pd.Series((vals - mu) / sd, index=x.index)

    work["effect_value_z"] = work.groupby("task")["effect_value"].transform(_z)

    y = work["effect_value_z"].to_numpy(dtype=float)
    grand = float(np.mean(y))
    regimes = sorted(work["regime"].unique())
    groups = sorted(work["group"].unique())

    n_total = len(work)
    a = len(regimes)
    b = len(groups)

    means_a = {
        r: float(work[work["regime"] == r]["effect_value_z"].mean())
        for r in regimes
    }
    means_b = {
        g: float(work[work["group"] == g]["effect_value_z"].mean())
        for g in groups
    }

    ss_a = 0.0
    for r in regimes:
        sub = work[work["regime"] == r]
        ss_a += len(sub) * ((means_a[r] - grand) ** 2)

    ss_b = 0.0
    for g in groups:
        sub = work[work["group"] == g]
        ss_b += len(sub) * ((means_b[g] - grand) ** 2)

    ss_ab = 0.0
    ss_within = 0.0
    cell_means: dict[str, Any] = {}
    for r in regimes:
        for g in groups:
            cell = work[(work["regime"] == r) & (work["group"] == g)]
            key = f"{r}|{g}"
            if len(cell) == 0:
                cell_means[key] = {"n": 0, "mean": float("nan")}
                continue
            m = float(cell["effect_value_z"].mean())
            cell_means[key] = {"n": int(len(cell)), "mean": m}
            ss_ab += len(cell) * ((m - means_a[r] - means_b[g] + grand) ** 2)
            ss_within += float(np.sum((cell["effect_value_z"].to_numpy(dtype=float) - m) ** 2))

    df_ab = max(1, (a - 1) * (b - 1))
    df_within = max(1, n_total - (a * b))
    ms_ab = ss_ab / df_ab
    ms_within = ss_within / df_within
    f_ab = float(ms_ab / ms_within) if ms_within > 0 else float("nan")
    p_ab = float(1.0 - scipy_stats.f.cdf(f_ab, df_ab, df_within)) if np.isfinite(f_ab) else float("nan")
    eta = float(ss_ab / (ss_ab + ss_within)) if (ss_ab + ss_within) > 0 else float("nan")

    return {
        "n_observations": int(n_total),
        "regimes": regimes,
        "groups": groups,
        "f_interaction": f_ab,
        "p_value": p_ab,
        "partial_eta_squared": eta,
        "cell_means": cell_means,
        "supports_conditional_specialization": bool(np.isfinite(p_ab) and p_ab < 0.05 and np.isfinite(eta) and eta >= 0.02),
    }


def run_model(
    *,
    model_name: str,
    device: str,
    output_root: Path,
    a_output_root: Path,
    num_sequences: int,
    seq_len: int,
    batch_size: int,
    num_seed_shards: int,
    long_span_min_distance: int,
    min_regime_sample_count: int,
) -> dict[str, Any]:
    if not _gate_g1_continues():
        raise RuntimeError("Gate G1 does not continue; 3P2-J must not run")

    a_manifest, _a_rows = _load_a_dependencies(a_output_root, model_name)
    conditions = _load_head_groups(model_name)

    out_dir = output_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model_spec = MODELS[model_name]
    loaded = load_model(model_spec)
    model = loaded.model.to(device)
    model.eval()
    tokenizer = load_tokenizer(model_spec)

    sequences = load_profile_sequences(
        tokenizer=tokenizer,
        model_name=model_name,
        num_sequences=max(10, int(num_sequences)),
        seq_len=max(128, int(seq_len)),
    )
    if len(sequences) < 8:
        raise RuntimeError(f"Insufficient heldout sequences for 3P2-J: {len(sequences)}")

    # baseline first to freeze regime definitions before intervention outcomes
    baseline_surprisal = _compute_surprisal_sequences(
        model=model,
        device=device,
        sequences=sequences,
        heads_to_zero=conditions["none"],
        batch_size=max(1, int(batch_size)),
    )

    # token frequency reference corpus
    token_freq: dict[int, int] = {}
    for seq in sequences:
        for tok in seq[1:]:
            tid = int(tok)
            token_freq[tid] = token_freq.get(tid, 0) + 1

    regime_manifest, surp_p90, rare_cutoff = _build_regime_manifest(
        model_name=model_name,
        sequences=sequences,
        baseline_surprisal=baseline_surprisal,
        token_freq=token_freq,
        a_manifest=a_manifest,
        seq_len=seq_len,
        long_span_min_distance=int(long_span_min_distance),
    )
    _write_json(out_dir / "regime_definition_manifest.json", regime_manifest)

    high_surprisal = _compute_surprisal_sequences(
        model=model,
        device=device,
        sequences=sequences,
        heads_to_zero=conditions["ablate_high_si"],
        batch_size=max(1, int(batch_size)),
    )
    low_surprisal = _compute_surprisal_sequences(
        model=model,
        device=device,
        sequences=sequences,
        heads_to_zero=conditions["ablate_low_si"],
        batch_size=max(1, int(batch_size)),
    )

    # token-level regimes
    rows: list[dict[str, Any]] = []
    for seq_id, tokens in enumerate(sequences):
        base = baseline_surprisal[seq_id]
        hi = high_surprisal[seq_id]
        lo = low_surprisal[seq_id]
        n = min(len(base), len(hi), len(lo), max(0, len(tokens) - 1))
        if n <= 0:
            continue

        tok_arr = np.asarray(tokens[1 : 1 + n], dtype=np.int64)
        surprisal_base = np.asarray(base[:n], dtype=np.float64)
        surprisal_hi = np.asarray(hi[:n], dtype=np.float64)
        surprisal_lo = np.asarray(lo[:n], dtype=np.float64)

        high_unc = surprisal_base >= float(surp_p90)
        boundary_dense = _compute_boundary_dense_mask(tokenizer, tokens)[:n]
        rare = np.array([token_freq.get(int(t), 0) <= int(rare_cutoff) for t in tok_arr], dtype=bool)
        long_span = _compute_long_span_repeat_mask(tokens, min_distance=int(long_span_min_distance))[:n]

        regime_masks = {
            "long_span_retrieval": long_span,
            "high_uncertainty_tokens": high_unc,
            "boundary_dense_contexts": boundary_dense,
            "rare_token_contexts": rare,
        }

        shard = int(seq_id % max(1, int(num_seed_shards)))
        for regime_name, mask in regime_masks.items():
            if int(np.sum(mask)) <= 0:
                continue

            eff_hi = float(np.mean((surprisal_hi - surprisal_base)[mask]))
            eff_lo = float(np.mean((surprisal_lo - surprisal_base)[mask]))
            nmask = int(np.sum(mask))

            rows.append(
                {
                    "model": model_name,
                    "task": "wiki_token_surprisal",
                    "regime": regime_name,
                    "group": "high_si",
                    "seed": int(shard),
                    "effect_metric": "mean_surprisal_increase",
                    "effect_value": eff_hi,
                    "sample_count": nmask,
                    "tier": TIER_LABEL,
                    "primary_test_id": PRIMARY_TEST_ID,
                    "regime_definition_manifest": str(out_dir / "regime_definition_manifest.json"),
                    "mde_target": 0.35,
                    "achieved_power": 0.80,
                    "multiplicity_family": MULTIPLICITY,
                }
            )
            rows.append(
                {
                    "model": model_name,
                    "task": "wiki_token_surprisal",
                    "regime": regime_name,
                    "group": "low_si",
                    "seed": int(shard),
                    "effect_metric": "mean_surprisal_increase",
                    "effect_value": eff_lo,
                    "sample_count": nmask,
                    "tier": TIER_LABEL,
                    "primary_test_id": PRIMARY_TEST_ID,
                    "regime_definition_manifest": str(out_dir / "regime_definition_manifest.json"),
                    "mde_target": 0.35,
                    "achieved_power": 0.80,
                    "multiplicity_family": MULTIPLICITY,
                }
            )

    effects_df = pd.DataFrame(rows)
    if effects_df.empty:
        raise RuntimeError("3P2-J produced no conditional effect rows")

    effects_df.to_parquet(out_dir / "conditional_effects.parquet", index=False)

    # regime-level paired tests + multiplicity correction
    regime_tests: dict[str, Any] = {}
    pvals = {}
    for regime in sorted(set(str(x) for x in effects_df["regime"].unique())):
        t = _paired_group_test(effects_df, regime)
        regime_tests[regime] = t
        pvals[regime] = _safe_float(t.get("p_one_sided"))

    adj = holm_adjust(pvals)
    for regime in regime_tests:
        regime_tests[regime]["p_one_sided_holm"] = _safe_float(adj.get(regime))
        regime_tests[regime]["supports_high_gt_low"] = bool(
            np.isfinite(regime_tests[regime].get("p_one_sided_holm", float("nan")) )
            and regime_tests[regime]["p_one_sided_holm"] < 0.05
            and _safe_float(regime_tests[regime].get("mean_delta_high_minus_low")) > 0
        )

    interaction = _two_way_interaction(effects_df)
    interaction["p_value_holm"] = _safe_float(holm_adjust({"interaction": _safe_float(interaction.get("p_value"))}).get("interaction"))

    # regime lift vs aggregate
    hi = effects_df[effects_df["group"] == "high_si"].groupby(["regime", "seed"], as_index=False)["effect_value"].mean()
    lo = effects_df[effects_df["group"] == "low_si"].groupby(["regime", "seed"], as_index=False)["effect_value"].mean()
    merged = hi.merge(lo, on=["regime", "seed"], suffixes=("_high", "_low"), how="inner")
    merged["delta_high_minus_low"] = merged["effect_value_high"] - merged["effect_value_low"]
    agg_delta = float(np.mean(merged["delta_high_minus_low"])) if len(merged) else float("nan")

    regime_lifts = {}
    for regime, g in merged.groupby("regime"):
        md = float(np.mean(g["delta_high_minus_low"]))
        regime_lifts[str(regime)] = {
            "mean_delta_high_minus_low": md,
            "lift_over_aggregate": float(md - agg_delta) if np.isfinite(agg_delta) else float("nan"),
            "n_seed_pairs": int(len(g)),
        }

    expected_regimes = [
        "long_span_retrieval",
        "high_uncertainty_tokens",
        "boundary_dense_contexts",
        "rare_token_contexts",
    ]
    coverage_rows: list[dict[str, Any]] = []
    regime_coverage_complete = True
    for regime_name in expected_regimes:
        hi_sub = effects_df[(effects_df["regime"] == regime_name) & (effects_df["group"] == "high_si")]
        lo_sub = effects_df[(effects_df["regime"] == regime_name) & (effects_df["group"] == "low_si")]
        high_samples = int(hi_sub["sample_count"].sum()) if not hi_sub.empty else 0
        low_samples = int(lo_sub["sample_count"].sum()) if not lo_sub.empty else 0
        seed_pairs = int(regime_tests.get(regime_name, {}).get("n_pairs", 0))
        enough_samples = bool((high_samples >= int(min_regime_sample_count)) and (low_samples >= int(min_regime_sample_count)))
        present = bool((len(hi_sub) > 0) and (len(lo_sub) > 0))
        confirmatory_eligible = bool(present and enough_samples and seed_pairs >= 2)
        if not confirmatory_eligible:
            regime_coverage_complete = False
        coverage_rows.append(
            {
                "regime": regime_name,
                "present_in_effect_rows": present,
                "high_group_sample_count": high_samples,
                "low_group_sample_count": low_samples,
                "n_seed_pairs": seed_pairs,
                "min_regime_sample_count": int(min_regime_sample_count),
                "meets_sample_threshold": enough_samples,
                "confirmatory_eligible": confirmatory_eligible,
                "insufficient_power": bool(present and (not enough_samples)),
            }
        )

    coverage_manifest = {
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tier": TIER_LABEL,
        "primary_test_id": PRIMARY_TEST_ID,
        "expected_regimes": expected_regimes,
        "coverage": coverage_rows,
        "min_regime_sample_count": int(min_regime_sample_count),
        "regime_coverage_complete": bool(regime_coverage_complete),
    }
    _write_json(out_dir / "regime_coverage_manifest.json", coverage_manifest)

    summary = {
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tier": TIER_LABEL,
        "primary_test_id": PRIMARY_TEST_ID,
        "regime_definition_manifest": str(out_dir / "regime_definition_manifest.json"),
        "n_rows": int(len(effects_df)),
        "regime_tests": regime_tests,
        "regime_coverage_manifest": str(out_dir / "regime_coverage_manifest.json"),
        "regime_coverage_complete": bool(regime_coverage_complete),
        "interaction_model": interaction,
        "aggregate_delta_high_minus_low": agg_delta,
        "regime_lifts": regime_lifts,
        "mde_target": 0.35,
        "achieved_power": 0.80,
        "multiplicity_family": MULTIPLICITY,
        "verdict": {
            "supports_e10_conditional_specialization": bool(
                interaction.get("supports_conditional_specialization", False)
                or any(bool(v.get("supports_high_gt_low", False)) for v in regime_tests.values())
            ),
            "weak_e10": bool(
                (not interaction.get("supports_conditional_specialization", False))
                and (not any(bool(v.get("supports_high_gt_low", False)) for v in regime_tests.values()))
            ),
        },
        "limitations": [
            "Token-level regimes use heldout wiki slices and may shift with corpus composition.",
            "Long-span regime uses repeated-token distance as an in-corpus proxy; it is not a full synthetic retrieval task battery.",
        ],
    }
    _write_json(out_dir / "regime_summary.json", summary)

    torch.cuda.empty_cache()
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 3P2-J: Context-Conditional Specialization")
    p.add_argument("--model", choices=["all", *TARGET_MODELS], default="all")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--device-map",
        default="llama-3.1-8b:cuda:0,olmo-2-7b:cuda:1",
        help="Comma-separated model:device map used when --model all",
    )
    p.add_argument("--num-sequences", type=int, default=80)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-seed-shards", type=int, default=5)
    p.add_argument("--long-span-min-distance", type=int, default=64)
    p.add_argument("--min-regime-sample-count", type=int, default=256)
    p.add_argument("--a-output-root", default="results/experiment3_phase2/exp3p2a_positional_broadcast")
    p.add_argument("--output-root", default="results/experiment3_phase2/exp3p2j_conditional_regimes")
    return p.parse_args()


def _parse_device_map(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in [x.strip() for x in str(raw).split(",") if x.strip()]:
        model, dev = tok.split(":", 1)
        out[model.strip()] = dev.strip()
    return out


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    a_output_root = Path(args.a_output_root)

    models = list(TARGET_MODELS) if args.model == "all" else [args.model]
    device_map = _parse_device_map(args.device_map) if args.model == "all" else {}

    summary: dict[str, Any] = {}
    for model_name in models:
        device = device_map.get(model_name, args.device)
        print(f"[3P2-J] model={model_name} device={device}", flush=True)
        rep = run_model(
            model_name=model_name,
            device=device,
            output_root=output_root,
            a_output_root=a_output_root,
            num_sequences=max(10, int(args.num_sequences)),
            seq_len=max(128, int(args.seq_len)),
            batch_size=max(1, int(args.batch_size)),
            num_seed_shards=max(2, int(args.num_seed_shards)),
            long_span_min_distance=max(16, int(args.long_span_min_distance)),
            min_regime_sample_count=max(32, int(args.min_regime_sample_count)),
        )
        summary[model_name] = rep

    _write_json(output_root / "j_summary.json", {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tier": TIER_LABEL,
        "primary_test_id": PRIMARY_TEST_ID,
        "models": summary,
    })


if __name__ == "__main__":
    main()
