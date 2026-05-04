#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats as scipy_stats

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp.common import RESULTS_ROOT, command_manifest, ensure_dir, safe_float, timestamp_now, write_json  # noqa: E402
from experiment3.theory8_position_ablation import (  # noqa: E402
    MODELS,
    compute_per_token_loss,
    load_head_groups,
    load_wiki_sequences,
    subtract_positional_kernels,
)
from shared.models.loading import load_model  # noqa: E402

TARGET_MODELS = ("llama-3.1-8b", "mistral-7b-v0.1", "olmo-2-7b")
DEFAULT_OUT = RESULTS_ROOT / "exp_new_r14_kernel_permutation"


def _load_kernels(model_name: str) -> tuple[dict[tuple[int, int], np.ndarray], str]:
    candidates = [
        ROOT / "results" / "reinforce_exp" / "exp_r3_core_replication" / model_name / "theory8_position_ablation" / model_name / "estimated_kernels.json",
        ROOT / "results" / "experiment3" / "theory8_position_ablation" / model_name / "estimated_kernels.json",
    ]
    for p in candidates:
        if not p.exists():
            continue
        raw = json.loads(p.read_text(encoding="utf-8"))
        out: dict[tuple[int, int], np.ndarray] = {}
        for k, v in raw.items():
            if not (k.startswith("L") and "H" in k):
                continue
            left, right = k[1:].split("H", 1)
            layer = int(left)
            head = int(right)
            out[(layer, head)] = np.asarray(v, dtype=np.float32)
        if out:
            return out, str(p)
    raise FileNotFoundError(f"No estimated_kernels.json found for {model_name} in expected locations: {candidates}")


def _permute_kernels(
    *,
    kernels: dict[tuple[int, int], np.ndarray],
    heads: list[tuple[int, int]],
    rng: np.random.Generator,
) -> dict[tuple[int, int], np.ndarray]:
    out: dict[tuple[int, int], np.ndarray] = {}
    for head in heads:
        g = kernels.get(head)
        if g is None:
            continue
        perm_idx = rng.permutation(len(g))
        out[head] = g[perm_idx].copy()
    return out


def _evaluate_sequence_mean_losses(
    *,
    model,
    sequences: list[list[int]],
    device: str,
    batch_size: int,
) -> np.ndarray:
    vals: list[float] = []
    bs = max(1, int(batch_size))
    pos = 0
    while pos < len(sequences):
        batch = sequences[pos : pos + bs]
        try:
            input_ids = torch.tensor(batch, dtype=torch.long, device=device)
            with torch.inference_mode():
                loss = compute_per_token_loss(model, input_ids)
            tok_per_seq = int(input_ids.shape[1] - 1)
            seq_loss = loss.view(input_ids.shape[0], tok_per_seq).mean(dim=1).numpy().astype(np.float64)
            vals.extend(float(x) for x in seq_loss.tolist())
            pos += len(batch)
            del input_ids, loss
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and bs > 1:
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                continue
            raise
    return np.asarray(vals, dtype=np.float64)


def _bootstrap_diff_ci(
    diff: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float, float]:
    if diff.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = int(diff.size)
    boot = np.empty(max(2000, int(n_boot)), dtype=np.float64)
    for i in range(len(boot)):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(np.mean(diff[idx]))
    mean_diff = float(np.mean(diff))
    ci_lo = float(np.quantile(boot, 0.025))
    ci_hi = float(np.quantile(boot, 0.975))
    p_one = float((np.sum(boot <= 0.0) + 1) / (len(boot) + 1))
    return mean_diff, ci_lo, ci_hi, p_one


def run_model(
    *,
    model_name: str,
    device: str,
    output_root: Path,
    eval_seqs: int,
    seq_len: int,
    n_permutations: int,
    permutation_seed: int,
    batch_size: int,
    n_boot: int,
    per_head_eval_count: int,
    per_head_eval_seqs: int,
) -> dict[str, Any]:
    t0 = time.time()
    out_dir = output_root / model_name
    ensure_dir(out_dir)

    model_spec = MODELS[model_name]
    loaded = load_model(model_spec)
    model = loaded.model.to(device)
    model.eval()
    model.config._attn_implementation = "eager"
    for layer in model.model.layers:
        layer.self_attn.config._attn_implementation = "eager"

    head_groups = load_head_groups(model_name)
    high_heads = [(int(l), int(h)) for (l, h) in head_groups["high_si"]]
    kernels, kernel_path = _load_kernels(model_name)

    sequences = load_wiki_sequences(model_name, max_sequences=max(8, int(eval_seqs)), seq_len=max(64, int(seq_len)))
    sequences = [seq[: int(seq_len)] for seq in sequences[: int(eval_seqs)]]
    if len(sequences) < 4:
        raise RuntimeError(f"[NEW-R14] insufficient sequences ({len(sequences)}) for model={model_name}")

    baseline = _evaluate_sequence_mean_losses(model=model, sequences=sequences, device=device, batch_size=batch_size)

    with subtract_positional_kernels(model, kernels, high_heads, int(seq_len)):
        true_loss = _evaluate_sequence_mean_losses(model=model, sequences=sequences, device=device, batch_size=batch_size)
    true_delta = true_loss - baseline

    perm_seed_rows: list[dict[str, Any]] = []
    perm_delta_stack: list[np.ndarray] = []
    for pidx in range(max(1, int(n_permutations))):
        rng = np.random.default_rng(int(permutation_seed) + pidx * 7919)
        perm_kernels = _permute_kernels(kernels=kernels, heads=high_heads, rng=rng)
        with subtract_positional_kernels(model, perm_kernels, high_heads, int(seq_len)):
            perm_loss = _evaluate_sequence_mean_losses(model=model, sequences=sequences, device=device, batch_size=batch_size)
        perm_delta = perm_loss - baseline
        perm_delta_stack.append(perm_delta)
        perm_seed_rows.append(
            {
                "perm_id": int(pidx),
                "perm_seed": int(permutation_seed + pidx * 7919),
                "mean_loss_delta": safe_float(np.mean(perm_delta)),
                "std_loss_delta": safe_float(np.std(perm_delta, ddof=1)),
            }
        )
        print(
            f"[NEW-R14][{model_name}] perm={pidx+1}/{n_permutations} "
            f"mean_delta={float(np.mean(perm_delta)):.6f}",
            flush=True,
        )

    perm_matrix = np.stack(perm_delta_stack, axis=0) if perm_delta_stack else np.zeros((1, len(true_delta)), dtype=np.float64)
    perm_mean_delta_seq = np.mean(perm_matrix, axis=0)
    diff = true_delta - perm_mean_delta_seq

    mean_diff, ci_lo, ci_hi, p_boot = _bootstrap_diff_ci(diff, n_boot=int(n_boot), seed=int(permutation_seed) + 123)
    t_stat, p_two = scipy_stats.ttest_1samp(diff, popmean=0.0, nan_policy="omit")
    p_one_t = float(p_two / 2.0) if np.isfinite(t_stat) and float(t_stat) > 0 else float(1.0 - p_two / 2.0) if np.isfinite(p_two) else float("nan")

    true_mean = float(np.mean(true_delta))
    perm_mean = float(np.mean(perm_mean_delta_seq))
    specificity_ratio = float(true_mean / max(abs(perm_mean), 1e-8))

    sampled_heads = high_heads[: max(0, int(per_head_eval_count))]
    per_head_rows: list[dict[str, Any]] = []
    if sampled_heads:
        seq_small = sequences[: max(4, int(per_head_eval_seqs))]
        base_small = baseline[: len(seq_small)]
        for hidx, head in enumerate(sampled_heads):
            one_head = [head]
            with subtract_positional_kernels(model, kernels, one_head, int(seq_len)):
                hi_loss = _evaluate_sequence_mean_losses(model=model, sequences=seq_small, device=device, batch_size=batch_size)
            hi_delta = hi_loss - base_small

            local_perm_rows: list[float] = []
            for pidx in range(min(3, max(1, int(n_permutations)))):
                rng = np.random.default_rng(int(permutation_seed) + hidx * 10007 + pidx * 29)
                perm_k = _permute_kernels(kernels=kernels, heads=one_head, rng=rng)
                with subtract_positional_kernels(model, perm_k, one_head, int(seq_len)):
                    lo_loss = _evaluate_sequence_mean_losses(model=model, sequences=seq_small, device=device, batch_size=batch_size)
                local_perm_rows.append(float(np.mean(lo_loss - base_small)))

            perm_head_mean = float(np.mean(np.asarray(local_perm_rows, dtype=np.float64))) if local_perm_rows else float("nan")
            per_head_rows.append(
                {
                    "layer": int(head[0]),
                    "head": int(head[1]),
                    "true_mean_delta": safe_float(np.mean(hi_delta)),
                    "permuted_mean_delta": safe_float(perm_head_mean),
                    "specificity_ratio": safe_float(float(np.mean(hi_delta) / max(abs(perm_head_mean), 1e-8)) if np.isfinite(perm_head_mean) else float("nan")),
                }
            )

    permutation_vs_true = {
        "timestamp": timestamp_now(),
        "experiment": "NEW-R14",
        "model": model_name,
        "kernel_source": kernel_path,
        "n_sequences": int(len(sequences)),
        "n_high_si_heads": int(len(high_heads)),
        "true_mean_loss_delta": safe_float(true_mean),
        "permuted_mean_loss_delta": safe_float(perm_mean),
        "specificity_ratio_true_over_permuted": safe_float(specificity_ratio),
        "permutations": perm_seed_rows,
    }
    specificity_test = {
        "timestamp": timestamp_now(),
        "experiment": "NEW-R14",
        "model": model_name,
        "mean_diff_true_minus_permuted": safe_float(mean_diff),
        "bootstrap_ci95": [safe_float(ci_lo), safe_float(ci_hi)],
        "p_one_bootstrap_true_gt_permuted": safe_float(p_boot),
        "t_stat": safe_float(t_stat),
        "p_one_ttest_true_gt_permuted": safe_float(p_one_t),
        "supports_specificity": bool(np.isfinite(ci_lo) and ci_lo > 0.0),
    }
    per_head = {
        "timestamp": timestamp_now(),
        "experiment": "NEW-R14",
        "model": model_name,
        "n_heads_evaluated": int(len(per_head_rows)),
        "rows": per_head_rows,
    }

    write_json(out_dir / "permutation_vs_true_comparison.json", permutation_vs_true)
    write_json(out_dir / "specificity_test.json", specificity_test)
    write_json(out_dir / "per_head_specificity.json", per_head)

    report = {
        "timestamp": timestamp_now(),
        "experiment": "NEW-R14",
        "model": model_name,
        "runtime_sec": float(time.time() - t0),
        "headline": {
            "true_mean_delta": safe_float(true_mean),
            "permuted_mean_delta": safe_float(perm_mean),
            "specificity_ratio": safe_float(specificity_ratio),
            "supports_specificity": bool(specificity_test["supports_specificity"]),
        },
    }
    write_json(out_dir / "summary.json", report)
    return report


def _parse_device_map(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in [x.strip() for x in str(raw).split(",") if x.strip()]:
        model, dev = tok.split(":", 1)
        out[model.strip()] = dev.strip()
    return out


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.output_root)
    ensure_dir(out_root)
    reports: dict[str, Any] = {}
    device_map = _parse_device_map(args.device_map)
    for model in TARGET_MODELS:
        dev = device_map.get(model, "cuda:0")
        cmd = [
            str(ROOT / ".venv" / "bin" / "python"),
            "-u",
            "reinforce_exp/exp_new_r14_kernel_permutation.py",
            "--model",
            str(model),
            "--device",
            str(dev),
            "--output-root",
            str(out_root),
            "--eval-seqs",
            str(args.eval_seqs),
            "--seq-len",
            str(args.seq_len),
            "--n-permutations",
            str(args.n_permutations),
            "--permutation-seed",
            str(args.permutation_seed),
            "--batch-size",
            str(args.batch_size),
            "--n-boot",
            str(args.n_boot),
            "--per-head-eval-count",
            str(args.per_head_eval_count),
            "--per-head-eval-seqs",
            str(args.per_head_eval_seqs),
        ]
        print("[NEW-R14] exec:", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(ROOT), check=True)
        reports[model] = json.loads((out_root / model / "summary.json").read_text(encoding="utf-8"))

    payload = {"timestamp": timestamp_now(), "experiment": "NEW-R14", "models": reports}
    write_json(out_root / "aggregate_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NEW-R14: SI kernel permutation specificity control")
    p.add_argument("--model", choices=["all", *TARGET_MODELS], default="all")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device-map", default="llama-3.1-8b:cuda:0,mistral-7b-v0.1:cuda:1,olmo-2-7b:cuda:2")
    p.add_argument("--output-root", default=str(DEFAULT_OUT))
    p.add_argument("--eval-seqs", type=int, default=100)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--n-permutations", type=int, default=5)
    p.add_argument("--permutation-seed", type=int, default=20260417)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--per-head-eval-count", type=int, default=12)
    p.add_argument("--per-head-eval-seqs", type=int, default=32)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    ensure_dir(out_root)

    if args.model == "all":
        agg = run_all(args)
        write_json(
            out_root / "manifest.json",
            command_manifest(
                experiment_id="NEW-R14",
                command="all_models",
                model="+".join(TARGET_MODELS),
                extras={
                    "eval_seqs": int(args.eval_seqs),
                    "n_permutations": int(args.n_permutations),
                    "output_root": str(out_root),
                },
            ),
        )
        print(f"[NEW-R14] wrote {out_root / 'aggregate_summary.json'}")
        print(f"[NEW-R14] models={list(agg['models'].keys())}")
        return

    rep = run_model(
        model_name=str(args.model),
        device=str(args.device),
        output_root=out_root,
        eval_seqs=max(8, int(args.eval_seqs)),
        seq_len=max(64, int(args.seq_len)),
        n_permutations=max(1, int(args.n_permutations)),
        permutation_seed=int(args.permutation_seed),
        batch_size=max(1, int(args.batch_size)),
        n_boot=max(1000, int(args.n_boot)),
        per_head_eval_count=max(0, int(args.per_head_eval_count)),
        per_head_eval_seqs=max(4, int(args.per_head_eval_seqs)),
    )
    write_json(
        out_root / str(args.model) / "manifest.json",
        command_manifest(
            experiment_id="NEW-R14",
            command="single_model",
            model=str(args.model),
            extras={
                "device": str(args.device),
                "eval_seqs": int(args.eval_seqs),
                "n_permutations": int(args.n_permutations),
                "output_root": str(out_root),
            },
        ),
    )
    print(f"[NEW-R14] wrote {out_root / str(args.model) / 'summary.json'}")
    print(f"[NEW-R14] supports_specificity={rep['headline']['supports_specificity']}")


if __name__ == "__main__":
    main()
