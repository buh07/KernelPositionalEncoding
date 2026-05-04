#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp.common import RESULTS_ROOT, command_manifest, ensure_dir, safe_float, timestamp_now, write_json  # noqa: E402
from experiment3.stats_utils import holm_adjust, one_sided_p_from_two_sided  # noqa: E402
from experiment3.theory1_si_circuits import MODELS, HeadID, head_output_ablation, load_profile_sequences  # noqa: E402
from shared.models.loading import load_model, load_tokenizer  # noqa: E402

DEFAULT_OUT = RESULTS_ROOT / "exp_r3_core_replication" / "conditional_regimes_generic"


def _token_has_prefix(tok: str) -> bool:
    return tok.startswith("\u0120") or tok.startswith("\u2581")


def _load_head_groups(model_name: str) -> dict[str, list[HeadID]]:
    path = ROOT / "results" / "experiment3" / "theory1_si_circuits" / model_name / "head_groups.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing head_groups.json: {path}")
    import json

    raw = json.loads(path.read_text(encoding="utf-8"))

    def parse(entries: list[dict[str, Any]]) -> list[HeadID]:
        return [HeadID(int(e["layer"]), int(e["head"])) for e in entries]

    return {
        "none": [],
        "ablate_high_si": parse(raw["high_si"]),
        "ablate_low_si": parse(raw["low_si"]),
    }


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


def _boundary_dense_mask(tokenizer, tokens: list[int], window: int = 16, min_boundaries: int = 4) -> np.ndarray:
    tok_str = tokenizer.convert_ids_to_tokens(tokens)
    boundary = np.array([1 if (i > 0 and _token_has_prefix(tok_str[i] or "")) else 0 for i in range(len(tokens))], dtype=np.int64)
    out = np.zeros((max(0, len(tokens) - 1),), dtype=bool)
    for t in range(1, len(tokens)):
        lo = max(0, t - window)
        cnt = int(np.sum(boundary[lo:t]))
        out[t - 1] = bool(cnt >= min_boundaries)
    return out


def _long_span_repeat_mask(tokens: list[int], min_distance: int = 64) -> np.ndarray:
    n = max(0, len(tokens) - 1)
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
            "p_one_sided": float("nan"),
            "cohens_d": float("nan"),
        }

    delta = pair["high"].to_numpy(dtype=np.float64) - pair["low"].to_numpy(dtype=np.float64)
    t_stat, t_p_two = scipy_stats.ttest_1samp(delta, popmean=0.0, nan_policy="omit")
    p_one = one_sided_p_from_two_sided(float(t_stat), float(t_p_two), alternative="greater")
    d = float(np.mean(delta) / max(np.std(delta, ddof=1), 1e-8))
    return {
        "n_pairs": int(len(pair)),
        "mean_high": float(pair["high"].mean()),
        "mean_low": float(pair["low"].mean()),
        "mean_delta_high_minus_low": float(np.mean(delta)),
        "p_one_sided": float(p_one),
        "cohens_d": d,
    }


def _two_way_interaction(df: pd.DataFrame) -> dict[str, Any]:
    work = df.copy()
    work = work[np.isfinite(work["effect_value"].to_numpy(dtype=float))]
    if work.empty:
        return {
            "supports_conditional_specialization": False,
            "f_interaction": float("nan"),
            "p_value": float("nan"),
            "partial_eta_squared": float("nan"),
            "n_observations": 0,
        }

    work["effect_value_z"] = (work["effect_value"] - work["effect_value"].mean()) / max(work["effect_value"].std(ddof=1), 1e-8)
    regimes = sorted(str(x) for x in work["regime"].unique())
    groups = sorted(str(x) for x in work["group"].unique())
    a = len(regimes)
    b = len(groups)
    n_total = len(work)

    means_a = {r: float(work.loc[work["regime"] == r, "effect_value_z"].mean()) for r in regimes}
    means_b = {g: float(work.loc[work["group"] == g, "effect_value_z"].mean()) for g in groups}
    grand = float(work["effect_value_z"].mean())

    ss_ab = 0.0
    ss_within = 0.0
    for r in regimes:
        for g in groups:
            cell = work[(work["regime"] == r) & (work["group"] == g)]
            if len(cell) == 0:
                continue
            m = float(cell["effect_value_z"].mean())
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
        "f_interaction": f_ab,
        "p_value": p_ab,
        "partial_eta_squared": eta,
        "supports_conditional_specialization": bool(np.isfinite(p_ab) and p_ab < 0.05 and np.isfinite(eta) and eta >= 0.02),
    }


def run_model(
    *,
    model_name: str,
    device: str,
    output_root: Path,
    num_sequences: int,
    seq_len: int,
    batch_size: int,
    num_seed_shards: int,
    long_span_min_distance: int,
    min_regime_sample_count: int,
) -> dict[str, Any]:
    conditions = _load_head_groups(model_name)
    out_dir = output_root / model_name
    ensure_dir(out_dir)

    model_spec = MODELS[model_name]
    loaded = load_model(model_spec)
    model = loaded.model.to(device)
    model.eval()
    tokenizer = load_tokenizer(model_spec)

    sequences = load_profile_sequences(
        tokenizer=tokenizer,
        model_name=model_name,
        num_sequences=max(12, int(num_sequences)),
        seq_len=max(128, int(seq_len)),
    )
    if len(sequences) < 8:
        raise RuntimeError(f"Insufficient sequences for generic conditional regimes: {len(sequences)}")

    baseline = _compute_surprisal_sequences(model=model, device=device, sequences=sequences, heads_to_zero=conditions["none"], batch_size=batch_size)
    high = _compute_surprisal_sequences(model=model, device=device, sequences=sequences, heads_to_zero=conditions["ablate_high_si"], batch_size=batch_size)
    low = _compute_surprisal_sequences(model=model, device=device, sequences=sequences, heads_to_zero=conditions["ablate_low_si"], batch_size=batch_size)

    # Build token frequency reference from heldout set.
    token_freq: dict[int, int] = {}
    for seq in sequences:
        for tok in seq[1:]:
            tid = int(tok)
            token_freq[tid] = token_freq.get(tid, 0) + 1

    surp_all = np.concatenate([x for x in baseline if x.size > 0], axis=0)
    p90 = float(np.quantile(surp_all, 0.90))
    freq_vals = np.asarray(list(token_freq.values()), dtype=np.float64)
    p10 = float(np.quantile(freq_vals, 0.10))
    rare_cutoff = int(max(1, np.floor(p10)))

    rows: list[dict[str, Any]] = []
    for seq_id, tokens in enumerate(sequences):
        b = baseline[seq_id]
        h = high[seq_id]
        l = low[seq_id]
        n = min(len(b), len(h), len(l), max(0, len(tokens) - 1))
        if n <= 0:
            continue

        tok_arr = np.asarray(tokens[1 : 1 + n], dtype=np.int64)
        b = np.asarray(b[:n], dtype=np.float64)
        h = np.asarray(h[:n], dtype=np.float64)
        l = np.asarray(l[:n], dtype=np.float64)

        masks = {
            "long_span_retrieval": _long_span_repeat_mask(tokens, min_distance=long_span_min_distance)[:n],
            "high_uncertainty_tokens": (b >= float(p90)),
            "boundary_dense_contexts": _boundary_dense_mask(tokenizer, tokens)[:n],
            "rare_token_contexts": np.array([token_freq.get(int(t), 0) <= int(rare_cutoff) for t in tok_arr], dtype=bool),
        }

        shard = int(seq_id % max(1, int(num_seed_shards)))
        for regime, mask in masks.items():
            nmask = int(np.sum(mask))
            if nmask <= 0:
                continue
            eff_hi = float(np.mean((h - b)[mask]))
            eff_lo = float(np.mean((l - b)[mask]))
            rows.append(
                {
                    "model": model_name,
                    "task": "wiki_token_surprisal",
                    "regime": regime,
                    "group": "high_si",
                    "seed": shard,
                    "effect_metric": "mean_surprisal_increase",
                    "effect_value": eff_hi,
                    "sample_count": nmask,
                }
            )
            rows.append(
                {
                    "model": model_name,
                    "task": "wiki_token_surprisal",
                    "regime": regime,
                    "group": "low_si",
                    "seed": shard,
                    "effect_metric": "mean_surprisal_increase",
                    "effect_value": eff_lo,
                    "sample_count": nmask,
                }
            )

    effects = pd.DataFrame(rows)
    if effects.empty:
        raise RuntimeError("No conditional effects generated")
    effects.to_parquet(out_dir / "conditional_effects.parquet", index=False)

    regime_tests: dict[str, Any] = {}
    pvals = {}
    for regime in sorted(str(x) for x in effects["regime"].unique()):
        rep = _paired_group_test(effects, regime)
        regime_tests[regime] = rep
        pvals[regime] = safe_float(rep.get("p_one_sided"))

    adj = holm_adjust(pvals)
    for regime in regime_tests:
        regime_tests[regime]["p_one_sided_holm"] = safe_float(adj.get(regime))
        regime_tests[regime]["supports_high_gt_low"] = bool(
            np.isfinite(regime_tests[regime]["p_one_sided_holm"])
            and regime_tests[regime]["p_one_sided_holm"] < 0.05
            and safe_float(regime_tests[regime].get("mean_delta_high_minus_low")) > 0
        )

    interaction = _two_way_interaction(effects)
    interaction["p_value_holm"] = safe_float(holm_adjust({"interaction": safe_float(interaction.get("p_value"))}).get("interaction"))

    coverage_rows: list[dict[str, Any]] = []
    expected_regimes = ["long_span_retrieval", "high_uncertainty_tokens", "boundary_dense_contexts", "rare_token_contexts"]
    regime_coverage_complete = True
    for regime_name in expected_regimes:
        hi_sub = effects[(effects["regime"] == regime_name) & (effects["group"] == "high_si")]
        lo_sub = effects[(effects["regime"] == regime_name) & (effects["group"] == "low_si")]
        high_samples = int(hi_sub["sample_count"].sum()) if not hi_sub.empty else 0
        low_samples = int(lo_sub["sample_count"].sum()) if not lo_sub.empty else 0
        seed_pairs = int(regime_tests.get(regime_name, {}).get("n_pairs", 0))
        enough_samples = bool(high_samples >= int(min_regime_sample_count) and low_samples >= int(min_regime_sample_count))
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
                "confirmatory_eligible": confirmatory_eligible,
            }
        )

    coverage_manifest = {
        "timestamp": timestamp_now(),
        "model": model_name,
        "coverage": coverage_rows,
        "regime_coverage_complete": regime_coverage_complete,
        "min_regime_sample_count": int(min_regime_sample_count),
    }
    write_json(out_dir / "regime_coverage_manifest.json", coverage_manifest)

    summary = {
        "timestamp": timestamp_now(),
        "model": model_name,
        "n_rows": int(len(effects)),
        "regime_tests": regime_tests,
        "interaction_model": interaction,
        "regime_coverage_complete": regime_coverage_complete,
        "regime_coverage_manifest": str(out_dir / "regime_coverage_manifest.json"),
        "regime_definition": {
            "long_span_min_distance": int(long_span_min_distance),
            "surprisal_p90": p90,
            "rare_frequency_cutoff": rare_cutoff,
        },
        "verdict": {
            "supports_conditional_specialization": bool(
                interaction.get("supports_conditional_specialization", False)
                or any(bool(v.get("supports_high_gt_low", False)) for v in regime_tests.values())
            )
        },
    }
    write_json(out_dir / "regime_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXP-R3 helper: generic conditional-regime analysis")
    p.add_argument("--model", required=True, choices=sorted(MODELS.keys()))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-sequences", type=int, default=160)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-seed-shards", type=int, default=16)
    p.add_argument("--long-span-min-distance", type=int, default=64)
    p.add_argument("--min-regime-sample-count", type=int, default=256)
    p.add_argument("--output-root", default=str(DEFAULT_OUT))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    ensure_dir(out_root)

    rep = run_model(
        model_name=str(args.model),
        device=str(args.device),
        output_root=out_root,
        num_sequences=max(12, int(args.num_sequences)),
        seq_len=max(128, int(args.seq_len)),
        batch_size=max(1, int(args.batch_size)),
        num_seed_shards=max(2, int(args.num_seed_shards)),
        long_span_min_distance=max(16, int(args.long_span_min_distance)),
        min_regime_sample_count=max(32, int(args.min_regime_sample_count)),
    )

    write_json(
        out_root / str(args.model) / "manifest.json",
        command_manifest(
            experiment_id="EXP-R3-J",
            command="conditional_regimes_generic",
            model=str(args.model),
            extras={
                "output_root": str(out_root),
                "num_sequences": int(args.num_sequences),
                "num_seed_shards": int(args.num_seed_shards),
                "long_span_min_distance": int(args.long_span_min_distance),
            },
        ),
    )

    print(f"[EXP-R3-J] wrote {out_root / str(args.model) / 'regime_summary.json'}")
    print(f"[EXP-R3-J] verdict={rep['verdict']}")


if __name__ == "__main__":
    main()
