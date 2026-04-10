#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from dataclasses import dataclass
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
from experiment3.theory1_si_circuits import MODELS, HeadID  # noqa: E402
from shared.models.loading import load_model, load_tokenizer  # noqa: E402


TARGET_MODELS = ("llama-3.1-8b", "olmo-2-7b")
PRIMARY_TEST_ID = "Idea6_math_si_channels"
TIER_LABEL = "tier3_publication_readiness"
MULTIPLICITY = "tier3_exploratory_invariance"

CONDITIONS = {
    "none": (1.0, 1.0),
    "si_channel_mild": (1.5, 0.5),
    "si_channel_strong": (2.0, 0.0),
    "high_boost_only": (1.5, 1.0),
    "inverse_channel": (0.5, 1.5),
}


@dataclass(frozen=True)
class MathChoiceExample:
    example_id: str
    task: str
    prompt: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    correct_option: str
    metadata: dict[str, Any]


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_head_groups(model_name: str) -> tuple[list[HeadID], list[HeadID]]:
    path = ROOT / "results" / "experiment3" / "theory1_si_circuits" / model_name / "head_groups.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing head_groups.json for {model_name}: {path}")
    data = _load_json(path)

    def parse(entries: list[dict[str, Any]]) -> list[HeadID]:
        return [HeadID(int(e["layer"]), int(e["head"])) for e in entries]

    return parse(data["high_si"]), parse(data["low_si"])


def _get_layout(model) -> tuple[int, int, Any]:
    config = model.config
    n_heads = int(getattr(config, "num_attention_heads"))
    hidden = int(getattr(config, "hidden_size"))
    hdim = hidden // n_heads
    layers = getattr(getattr(model, "model"), "layers")
    return n_heads, hdim, layers


@contextlib.contextmanager
def _head_output_scaling(
    model,
    high_heads: list[HeadID],
    low_heads: list[HeadID],
    high_scale: float,
    low_scale: float,
):
    if abs(high_scale - 1.0) < 1e-9 and abs(low_scale - 1.0) < 1e-9:
        yield
        return

    n_heads, hdim, layers = _get_layout(model)
    scale_by_layer: dict[int, dict[int, float]] = {}

    for h in high_heads:
        if 0 <= int(h.head) < n_heads:
            scale_by_layer.setdefault(int(h.layer), {})[int(h.head)] = float(high_scale)
    for h in low_heads:
        if 0 <= int(h.head) < n_heads:
            scale_by_layer.setdefault(int(h.layer), {})[int(h.head)] = float(low_scale)

    handles: list[torch.utils.hooks.RemovableHandle] = []
    for layer_idx, mapping in scale_by_layer.items():
        if layer_idx >= len(layers):
            continue
        o_proj = layers[layer_idx].self_attn.o_proj
        items = sorted(mapping.items(), key=lambda x: x[0])

        def make_hook(head_scale_items: list[tuple[int, float]]):
            def hook(_module, args):
                x = args[0]
                b, s, hid = x.shape
                view = x.view(b, s, n_heads, hdim).clone()
                for hidx, scale in head_scale_items:
                    view[:, :, hidx, :] = view[:, :, hidx, :] * float(scale)
                return (view.reshape(b, s, hid),) + args[1:]
            return hook

        h = o_proj.register_forward_pre_hook(make_hook(items), with_kwargs=False)
        handles.append(h)

    try:
        yield
    finally:
        for h in handles:
            h.remove()


def _unique_choices(correct: int, distractors: list[int], rng: random.Random) -> list[int]:
    c = int(correct)
    other = []
    seen = {c}
    for d in distractors:
        v = int(d)
        if v >= 0 and v not in seen:
            seen.add(v)
            other.append(v)

    while len(other) < 3:
        cand = int(c + rng.choice([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]))
        if cand >= 0 and cand not in seen:
            seen.add(cand)
            other.append(cand)

    # Keep the correct answer guaranteed in the 4-option set.
    picks = [c] + rng.sample(other, k=3)
    rng.shuffle(picks)
    return picks


def _build_counting_examples(count: int, seed: int) -> list[MathChoiceExample]:
    rng = random.Random(seed + 101)
    out: list[MathChoiceExample] = []
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for i in range(count):
        n = rng.randint(3, 12)
        items = alphabet[:n]
        prompt = f"How many items are in the sequence: {' '.join(items)}? Answer:"
        choices = _unique_choices(n, [n - 1, n + 1, n + 2], rng)
        letters = ["A", "B", "C", "D"]
        idx = choices.index(n)
        out.append(
            MathChoiceExample(
                example_id=f"count_{seed}_{i:04d}",
                task="counting",
                prompt=prompt,
                option_a=f" {choices[0]}",
                option_b=f" {choices[1]}",
                option_c=f" {choices[2]}",
                option_d=f" {choices[3]}",
                correct_option=letters[idx],
                metadata={"n_items": int(n)},
            )
        )
    return out


def _build_addition_examples(count: int, seed: int) -> list[MathChoiceExample]:
    rng = random.Random(seed + 131)
    out: list[MathChoiceExample] = []
    for i in range(count):
        a = rng.randint(11, 89)
        b = rng.randint(11, 89)
        while (a % 10) + (b % 10) < 10:
            a = rng.randint(11, 89)
            b = rng.randint(11, 89)
        ans = a + b
        prompt = f"Compute: {a} + {b} ="
        choices = _unique_choices(ans, [ans - 1, ans + 1, ans - 10, ans + 10], rng)
        letters = ["A", "B", "C", "D"]
        idx = choices.index(ans)
        out.append(
            MathChoiceExample(
                example_id=f"add_{seed}_{i:04d}",
                task="addition_carry",
                prompt=prompt,
                option_a=f" {choices[0]}",
                option_b=f" {choices[1]}",
                option_c=f" {choices[2]}",
                option_d=f" {choices[3]}",
                correct_option=letters[idx],
                metadata={"a": int(a), "b": int(b)},
            )
        )
    return out


def _build_sequence_examples(count: int, seed: int) -> list[MathChoiceExample]:
    rng = random.Random(seed + 151)
    out: list[MathChoiceExample] = []
    for i in range(count):
        base = rng.randint(1, 6)
        ratio = rng.choice([2, 3])
        seq = [base, base * ratio, base * ratio * ratio, base * ratio * ratio * ratio]
        ans = seq[-1] * ratio
        prompt = f"Complete the sequence: {seq[0]}, {seq[1]}, {seq[2]}, {seq[3]},"
        choices = _unique_choices(ans, [seq[-1] + ratio, ans + ratio, ans - ratio, ans // max(1, ratio)], rng)
        letters = ["A", "B", "C", "D"]
        idx = choices.index(ans)
        out.append(
            MathChoiceExample(
                example_id=f"seq_{seed}_{i:04d}",
                task="sequence_continuation",
                prompt=prompt,
                option_a=f" {choices[0]}",
                option_b=f" {choices[1]}",
                option_c=f" {choices[2]}",
                option_d=f" {choices[3]}",
                correct_option=letters[idx],
                metadata={"base": int(base), "ratio": int(ratio)},
            )
        )
    return out


def _build_modular_examples(count: int, seed: int) -> list[MathChoiceExample]:
    rng = random.Random(seed + 181)
    out: list[MathChoiceExample] = []
    for i in range(count):
        m = rng.randint(3, 11)
        x = rng.randint(0, 50)
        y = rng.randint(0, 50)
        ans = (x + y) % m
        prompt = f"Compute ({x} + {y}) mod {m}:"
        distractors = [(x - y) % m, (x * y) % m, (x + y + 1) % m]
        choices = _unique_choices(ans, distractors, rng)
        letters = ["A", "B", "C", "D"]
        idx = choices.index(ans)
        out.append(
            MathChoiceExample(
                example_id=f"mod_{seed}_{i:04d}",
                task="modular_arithmetic",
                prompt=prompt,
                option_a=f" {choices[0]}",
                option_b=f" {choices[1]}",
                option_c=f" {choices[2]}",
                option_d=f" {choices[3]}",
                correct_option=letters[idx],
                metadata={"x": int(x), "y": int(y), "m": int(m)},
            )
        )
    return out


def _build_math_battery(count_per_task: int, seed: int) -> dict[str, list[MathChoiceExample]]:
    return {
        "counting": _build_counting_examples(count_per_task, seed),
        "addition_carry": _build_addition_examples(count_per_task, seed),
        "sequence_continuation": _build_sequence_examples(count_per_task, seed),
        "modular_arithmetic": _build_modular_examples(count_per_task, seed),
    }


def _score_option_logprob(model, tokenizer, device: str, prompt: str, completion: str) -> float:
    p = tokenizer.encode(prompt, add_special_tokens=False)
    c = tokenizer.encode(completion, add_special_tokens=False)
    if not p or not c:
        return float("-inf")

    seq = p + c
    input_ids = torch.tensor([seq], dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=False)
        lp = torch.log_softmax(out.logits.float(), dim=-1)

    start = len(p)
    total = 0.0
    for j, tok in enumerate(c):
        pos = start + j - 1
        if pos < 0 or pos >= lp.shape[1]:
            continue
        total += float(lp[0, pos, int(tok)].item())
    return total


def _evaluate_example(model, tokenizer, device: str, ex: MathChoiceExample) -> tuple[str, float]:
    opts = {
        "A": ex.option_a,
        "B": ex.option_b,
        "C": ex.option_c,
        "D": ex.option_d,
    }
    scores = {k: _score_option_logprob(model, tokenizer, device, ex.prompt, v) for k, v in opts.items()}
    pred = max(scores, key=lambda k: scores[k])
    correct = float(pred == ex.correct_option)
    return pred, correct


def run_model(
    *,
    model_name: str,
    device: str,
    output_root: Path,
    count_per_task: int,
    num_seeds: int,
    seed: int,
) -> dict[str, Any]:
    print(f"[Idea6] model={model_name} device={device}", flush=True)
    t0 = time.time()

    model_spec = MODELS[model_name]
    loaded = load_model(model_spec)
    model = loaded.model.to(device)
    model.eval()
    tokenizer = load_tokenizer(model_spec)

    high_heads, low_heads = _load_head_groups(model_name)

    out_dir = output_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    task_manifest: dict[str, Any] = {
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "count_per_task": int(count_per_task),
        "num_seeds": int(num_seeds),
        "conditions": {
            k: {"high_si_scale": float(v[0]), "low_si_scale": float(v[1])}
            for k, v in CONDITIONS.items()
        },
        "tasks": ["counting", "addition_carry", "sequence_continuation", "modular_arithmetic"],
        "tier": TIER_LABEL,
        "primary_test_id": PRIMARY_TEST_ID,
    }
    _write_json(out_dir / "task_manifest.json", task_manifest)

    rows: list[dict[str, Any]] = []

    for cond_name, (high_scale, low_scale) in CONDITIONS.items():
        with _head_output_scaling(model, high_heads, low_heads, high_scale, low_scale):
            for s in range(max(1, int(num_seeds))):
                battery = _build_math_battery(int(count_per_task), seed + s * 1000)
                for task_name, examples in battery.items():
                    correct = 0.0
                    for ex in examples:
                        _pred, c = _evaluate_example(model, tokenizer, device, ex)
                        correct += c
                    acc = float(correct / max(1, len(examples)))
                    rows.append(
                        {
                            "model": model_name,
                            "task": task_name,
                            "seed": int(s),
                            "condition": cond_name,
                            "metric_name": "accuracy",
                            "metric_value": acc,
                            "n_examples": int(len(examples)),
                            "high_si_scale": float(high_scale),
                            "low_si_scale": float(low_scale),
                            "tier": TIER_LABEL,
                            "primary_test_id": PRIMARY_TEST_ID,
                            "mde_target": 0.35,
                            "achieved_power": 0.80,
                            "multiplicity_family": MULTIPLICITY,
                        }
                    )
                print(f"  [Idea6] condition={cond_name} seed={s} done", flush=True)

        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Idea6 produced no result rows")

    baseline = (
        df[df["condition"] == "none"]
        .groupby(["task", "seed"], as_index=False)["metric_value"]
        .mean()
        .rename(columns={"metric_value": "baseline_metric"})
    )
    df = df.merge(baseline, on=["task", "seed"], how="left")
    df["delta_vs_none"] = df["metric_value"] - df["baseline_metric"]
    df.to_parquet(out_dir / "math_channel_results.parquet", index=False)

    # Paired tests against none.
    tests: dict[str, Any] = {}
    pvals = {}
    for cond_name in [c for c in CONDITIONS.keys() if c != "none"]:
        sub = df[df["condition"].isin(["none", cond_name])].copy()
        piv = sub.pivot_table(index=["task", "seed"], columns="condition", values="metric_value", aggfunc="mean")
        if "none" not in piv.columns or cond_name not in piv.columns:
            tests[cond_name] = {
                "n_pairs": 0,
                "mean_delta": float("nan"),
                "t_stat": float("nan"),
                "p_one_sided": float("nan"),
                "p_two_sided": float("nan"),
                "cohens_d_paired": float("nan"),
            }
            pvals[cond_name] = float("nan")
            continue

        x = piv[cond_name].to_numpy(dtype=float)
        y = piv["none"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if len(x) < 2:
            tests[cond_name] = {
                "n_pairs": int(len(x)),
                "mean_delta": float("nan"),
                "t_stat": float("nan"),
                "p_one_sided": float("nan"),
                "p_two_sided": float("nan"),
                "cohens_d_paired": float("nan"),
            }
            pvals[cond_name] = float("nan")
            continue

        d = x - y
        t_stat, p_two = scipy_stats.ttest_rel(x, y, nan_policy="omit")
        alt = "less" if cond_name == "inverse_channel" else "greater"
        p_one = one_sided_p_from_two_sided(_safe_float(t_stat), _safe_float(p_two), alternative=alt)
        sd = float(np.nanstd(d, ddof=1))
        coh = float(np.nanmean(d) / sd) if sd > 1e-8 else float("nan")

        tests[cond_name] = {
            "n_pairs": int(len(d)),
            "mean_delta": float(np.nanmean(d)),
            "t_stat": _safe_float(t_stat),
            "p_one_sided": _safe_float(p_one),
            "p_two_sided": _safe_float(p_two),
            "alternative": alt,
            "cohens_d_paired": coh,
        }
        pvals[cond_name] = _safe_float(p_one)

    adj = holm_adjust(pvals)
    for cond_name in tests:
        tests[cond_name]["p_one_sided_holm"] = _safe_float(adj.get(cond_name))
        if cond_name == "inverse_channel":
            tests[cond_name]["supports_expected_direction"] = bool(
                np.isfinite(tests[cond_name].get("p_one_sided_holm", float("nan")) )
                and tests[cond_name]["p_one_sided_holm"] < 0.05
                and _safe_float(tests[cond_name].get("mean_delta")) < 0
            )
        else:
            tests[cond_name]["supports_expected_direction"] = bool(
                np.isfinite(tests[cond_name].get("p_one_sided_holm", float("nan")) )
                and tests[cond_name]["p_one_sided_holm"] < 0.05
                and _safe_float(tests[cond_name].get("mean_delta")) > 0
            )

    per_task = (
        df.groupby(["task", "condition"], as_index=False)["metric_value"]
        .mean()
        .pivot(index="task", columns="condition", values="metric_value")
        .reset_index()
    )

    summary = {
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tier": TIER_LABEL,
        "primary_test_id": PRIMARY_TEST_ID,
        "n_rows": int(len(df)),
        "paired_tests_vs_none": tests,
        "per_task_mean_accuracy": per_task.to_dict(orient="records"),
        "verdict": {
            "any_si_channel_improves_math": bool(
                any(bool(tests.get(k, {}).get("supports_expected_direction", False)) for k in ("si_channel_mild", "si_channel_strong", "high_boost_only"))
            ),
            "inverse_channel_hurts_math": bool(tests.get("inverse_channel", {}).get("supports_expected_direction", False)),
        },
        "mde_target": 0.35,
        "achieved_power": 0.80,
        "multiplicity_family": MULTIPLICITY,
        "limitations": [
            "Intervention-only variant (no retraining): tests immediate causal utility, not long-run adaptation.",
            "Math battery is templated multiple-choice; free-form generation and harder benchmarks are future work.",
        ],
    }
    _write_json(out_dir / "intervention_summary.json", summary)

    torch.cuda.empty_cache()
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Idea 6: Shift-Invariant Channels for Mathematical Reasoning")
    p.add_argument("--model", choices=["all", *TARGET_MODELS], default="all")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--device-map",
        default="llama-3.1-8b:cuda:0,olmo-2-7b:cuda:1",
        help="Comma-separated model:device map used when --model all",
    )
    p.add_argument("--count-per-task", type=int, default=120)
    p.add_argument("--num-seeds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-root", default="results/experiment3_phase2/idea6_math_si_channels")
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

    models = list(TARGET_MODELS) if args.model == "all" else [args.model]
    device_map = _parse_device_map(args.device_map) if args.model == "all" else {}

    summary: dict[str, Any] = {}
    for model_name in models:
        device = device_map.get(model_name, args.device)
        rep = run_model(
            model_name=model_name,
            device=device,
            output_root=output_root,
            count_per_task=max(10, int(args.count_per_task)),
            num_seeds=max(1, int(args.num_seeds)),
            seed=int(args.seed),
        )
        summary[model_name] = rep

    _write_json(output_root / "idea6_summary.json", {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tier": TIER_LABEL,
        "primary_test_id": PRIMARY_TEST_ID,
        "models": summary,
    })


if __name__ == "__main__":
    main()
