#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment3.theory1_si_circuits import MODELS, HeadID, head_output_ablation  # noqa: E402
from experiment3.theory5b_boundary_detection import (  # noqa: E402
    compute_word_boundaries,
    load_wiki_sequences,
    parse_head_list,
)
from shared.models.loading import load_model, load_tokenizer  # noqa: E402


TARGET_MODELS = ("llama-3.1-8b", "olmo-2-7b")


@dataclass
class SplitBuffers:
    x_train: list[np.ndarray]
    y_boundary_train: list[np.ndarray]
    y_distance_train: list[np.ndarray]
    x_test: list[np.ndarray]
    y_boundary_test: list[np.ndarray]
    y_distance_test: list[np.ndarray]


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _parse_device_map(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid device map token '{token}'. Expected model:device")
        model, device = token.split(":", 1)
        out[model.strip()] = device.strip()
    return out


def _load_head_groups(model_name: str, model, *, seed: int) -> dict[str, list[HeadID]]:
    head_groups_path = ROOT / "results" / "experiment3" / "theory1_si_circuits" / model_name / "head_groups.json"
    if not head_groups_path.exists():
        raise FileNotFoundError(f"Missing head groups for {model_name}: {head_groups_path}")

    with head_groups_path.open("r", encoding="utf-8") as f:
        groups = json.load(f)

    high_si = parse_head_list(groups["high_si"])
    low_si = parse_head_list(groups["low_si"])

    n_layers = int(getattr(model.config, "num_hidden_layers"))
    n_heads = int(getattr(model.config, "num_attention_heads"))
    all_heads = [HeadID(layer=l, head=h) for l in range(n_layers) for h in range(n_heads)]

    high_set = {(h.layer, h.head) for h in high_si}
    low_set = {(h.layer, h.head) for h in low_si}
    blocked = high_set | low_set
    candidates = [h for h in all_heads if (h.layer, h.head) not in blocked]
    if len(candidates) < len(high_si):
        # Fallback if complement is too small: avoid high-SI only.
        candidates = [h for h in all_heads if (h.layer, h.head) not in high_set]
    if len(candidates) < len(high_si):
        raise RuntimeError(
            f"Could not sample random matched heads for {model_name}: "
            f"need {len(high_si)}, have {len(candidates)} candidates."
        )

    rng = random.Random(seed + 17)
    random_matched = rng.sample(candidates, k=len(high_si))

    return {
        "high_si": high_si,
        "low_si": low_si,
        "random_matched": random_matched,
    }


def _boundary_labels(tokenizer, token_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns labels for token positions 1..T-1:
      - boundary indicator (0/1)
      - nearest-boundary distance bin: 0,1,2,3(>=3)
    """
    word_ids = compute_word_boundaries(tokenizer, token_ids)
    seq_len = len(word_ids)
    if seq_len < 2:
        raise ValueError("Sequence too short for boundary labels.")

    positions = np.arange(1, seq_len, dtype=np.int32)
    is_boundary = np.array([1 if word_ids[t] != word_ids[t - 1] else 0 for t in positions], dtype=np.int64)

    boundary_positions = positions[is_boundary == 1]
    if boundary_positions.size == 0:
        nearest = np.full_like(positions, fill_value=3, dtype=np.int64)
    else:
        # vectorized minimum absolute distance to any boundary.
        dmat = np.abs(positions[:, None] - boundary_positions[None, :])
        nearest_raw = np.min(dmat, axis=1).astype(np.int64)
        nearest = np.minimum(nearest_raw, 3)

    return is_boundary, nearest


def _bootstrap_accuracy_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = y_true.size
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    acc = (y_pred[idx] == y_true[idx]).mean(axis=1)
    return float(np.quantile(acc, 0.025)), float(np.quantile(acc, 0.975))


def _majority_chance(y: np.ndarray) -> float:
    y = np.asarray(y)
    if y.size == 0:
        return float("nan")
    vals, counts = np.unique(y, return_counts=True)
    del vals
    return float(np.max(counts) / y.size)


def _pseudo_r2(y_true: np.ndarray, proba: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    if y_true.size == 0:
        return float("nan")
    classes, counts = np.unique(y_true, return_counts=True)
    priors = counts.astype(np.float64) / counts.sum()
    class_to_idx = {int(c): i for i, c in enumerate(classes)}
    null = np.zeros((y_true.size, classes.size), dtype=np.float64)
    for i in range(y_true.size):
        null[i, :] = priors
    ll_full = log_loss(y_true, proba, labels=classes)
    ll_null = log_loss(y_true, null, labels=classes)
    if not np.isfinite(ll_full) or not np.isfinite(ll_null) or ll_null <= 0:
        return float("nan")
    return float(1.0 - (ll_full / ll_null))


def _fit_probe(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    multi_class: bool,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
        return {
            "status": "insufficient_label_variation",
            "n_train": int(y_train.size),
            "n_test": int(y_test.size),
            "accuracy": float("nan"),
            "chance": float("nan"),
            "delta_over_chance": float("nan"),
            "ci_95_accuracy": [float("nan"), float("nan")],
            "ci_95_delta_over_chance": [float("nan"), float("nan")],
            "pseudo_r2": float("nan"),
        }

    clf = LogisticRegression(
        max_iter=1000,
        random_state=seed,
        multi_class="multinomial" if multi_class else "auto",
        solver="lbfgs",
    )
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    y_proba = clf.predict_proba(x_test)

    acc = float(accuracy_score(y_test, y_pred))
    chance = _majority_chance(y_test)
    delta = float(acc - chance)
    ci_acc = _bootstrap_accuracy_ci(y_test, y_pred, n_boot=bootstrap_samples, seed=seed + 101)
    ci_delta = [float(ci_acc[0] - chance), float(ci_acc[1] - chance)]
    pr2 = _pseudo_r2(y_test, y_proba)

    return {
        "status": "ok",
        "n_train": int(y_train.size),
        "n_test": int(y_test.size),
        "accuracy": acc,
        "chance": chance,
        "delta_over_chance": delta,
        "ci_95_accuracy": [float(ci_acc[0]), float(ci_acc[1])],
        "ci_95_delta_over_chance": ci_delta,
        "pseudo_r2": pr2,
    }


def _append_group_split(
    buf: SplitBuffers,
    *,
    x_proj: np.ndarray,
    y_boundary: np.ndarray,
    y_distance: np.ndarray,
    is_train: bool,
) -> None:
    if is_train:
        buf.x_train.append(x_proj)
        buf.y_boundary_train.append(y_boundary)
        buf.y_distance_train.append(y_distance)
    else:
        buf.x_test.append(x_proj)
        buf.y_boundary_test.append(y_boundary)
        buf.y_distance_test.append(y_distance)


def _concat_or_empty(arrs: list[np.ndarray], width: int | None = None) -> np.ndarray:
    if not arrs:
        if width is None:
            return np.empty((0,), dtype=np.float32)
        return np.empty((0, width), dtype=np.float32)
    return np.concatenate(arrs, axis=0)


def run_model(
    *,
    model_name: str,
    device: str,
    output_root: Path,
    sequence_count: int,
    seq_len: int,
    batch_size: int,
    projection_dim: int,
    train_fraction: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    print(f"\n[3P2-C.3] model={model_name} device={device}", flush=True)
    t0 = time.time()

    model_spec = MODELS[model_name]
    loaded = load_model(model_spec)
    model = loaded.model.to(device)
    model.eval()
    tokenizer = load_tokenizer(model_spec)

    groups = _load_head_groups(model_name, model, seed=seed)
    group_names = ("high_si", "low_si", "random_matched")

    seqs = load_wiki_sequences(model_name, max_sequences=max(1, int(sequence_count)), seq_len=max(64, int(seq_len)))
    if len(seqs) < max(2, int(sequence_count)):
        print(f"  WARNING: requested {sequence_count} sequences, got {len(seqs)}", flush=True)
    if len(seqs) < 2:
        raise RuntimeError(f"Need at least 2 sequences for train/test split; got {len(seqs)}")

    n_seq = len(seqs)
    n_train_seq = max(1, min(n_seq - 1, int(math.floor(train_fraction * n_seq))))
    split_note = {"n_sequences_total": n_seq, "n_sequences_train": n_train_seq, "n_sequences_test": n_seq - n_train_seq}
    print(
        f"  sequences={n_seq} train={n_train_seq} test={n_seq - n_train_seq} "
        f"projection_dim={projection_dim}",
        flush=True,
    )

    hidden_size = int(getattr(model.config, "hidden_size"))
    torch.manual_seed(seed + 23)
    proj = torch.randn(hidden_size, projection_dim, device=device, dtype=torch.float32) / math.sqrt(float(projection_dim))

    buffers: dict[str, SplitBuffers] = {
        g: SplitBuffers([], [], [], [], [], [])
        for g in group_names
    }

    seq_labels_boundary: list[np.ndarray] = []
    seq_labels_distance: list[np.ndarray] = []
    for toks in seqs:
        yb, yd = _boundary_labels(tokenizer, toks)
        seq_labels_boundary.append(yb)
        seq_labels_distance.append(yd)

    with torch.inference_mode():
        for start in range(0, n_seq, max(1, int(batch_size))):
            end = min(n_seq, start + max(1, int(batch_size)))
            batch_tokens = seqs[start:end]
            batch = torch.tensor(batch_tokens, dtype=torch.long, device=device)

            base_out = model(input_ids=batch, use_cache=False, output_hidden_states=True)
            base_h = base_out.hidden_states[-1][:, 1:, :].float()

            for group_name in group_names:
                with head_output_ablation(model, groups[group_name]):
                    abl_out = model(input_ids=batch, use_cache=False, output_hidden_states=True)
                abl_h = abl_out.hidden_states[-1][:, 1:, :].float()
                contrib = base_h - abl_h
                proj_feat = torch.einsum("bth,hp->btp", contrib, proj).detach().cpu().numpy().astype(np.float32)

                for i, seq_idx in enumerate(range(start, end)):
                    yb = seq_labels_boundary[seq_idx]
                    yd = seq_labels_distance[seq_idx]
                    x_i = proj_feat[i]
                    m = min(x_i.shape[0], yb.shape[0], yd.shape[0])
                    if m <= 0:
                        continue
                    is_train = seq_idx < n_train_seq
                    _append_group_split(
                        buffers[group_name],
                        x_proj=x_i[:m, :],
                        y_boundary=yb[:m],
                        y_distance=yd[:m],
                        is_train=is_train,
                    )

            print(f"  processed sequences {end}/{n_seq}", flush=True)

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "experiment": "3P2-C.3_low_si_contribution_probe",
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tier": "tier2_conditional_mechanistic",
        "primary_test_id": "3P2-C.3",
        "mde_target": 0.35,
        "achieved_power": 0.8,
        "multiplicity_family": "tier2_holm_primary_tests",
        "config": {
            "sequence_count": int(sequence_count),
            "seq_len": int(seq_len),
            "batch_size": int(batch_size),
            "projection_dim": int(projection_dim),
            "train_fraction": float(train_fraction),
            "bootstrap_samples": int(bootstrap_samples),
            "seed": int(seed),
            "optional_stress_test_half_attenuation": False,
            "feature_definition": "last-layer residual contribution proxy via (intact - group-ablated), random-projected",
        },
        "split": split_note,
        "group_sizes": {g: int(len(groups[g])) for g in group_names},
        "group_metrics": {},
    }

    for group_name in group_names:
        buf = buffers[group_name]
        x_train = _concat_or_empty(buf.x_train, width=projection_dim)
        x_test = _concat_or_empty(buf.x_test, width=projection_dim)
        yb_train = _concat_or_empty(buf.y_boundary_train).astype(np.int64)
        yb_test = _concat_or_empty(buf.y_boundary_test).astype(np.int64)
        yd_train = _concat_or_empty(buf.y_distance_train).astype(np.int64)
        yd_test = _concat_or_empty(buf.y_distance_test).astype(np.int64)

        boundary_res = _fit_probe(
            x_train=x_train,
            y_train=yb_train,
            x_test=x_test,
            y_test=yb_test,
            multi_class=False,
            seed=seed + 211,
            bootstrap_samples=bootstrap_samples,
        )
        distance_res = _fit_probe(
            x_train=x_train,
            y_train=yd_train,
            x_test=x_test,
            y_test=yd_test,
            multi_class=True,
            seed=seed + 307,
            bootstrap_samples=bootstrap_samples,
        )

        summary["group_metrics"][group_name] = {
            "boundary_indicator": boundary_res,
            "nearest_boundary_bin": distance_res,
        }

        for target_name, res in (("boundary_indicator", boundary_res), ("nearest_boundary_bin", distance_res)):
            rows.append(
                {
                    "model": model_name,
                    "group": group_name,
                    "target": target_name,
                    "status": str(res.get("status")),
                    "accuracy": _safe_float(res.get("accuracy")),
                    "chance": _safe_float(res.get("chance")),
                    "delta_over_chance": _safe_float(res.get("delta_over_chance")),
                    "ci_95_accuracy_low": _safe_float((res.get("ci_95_accuracy") or [float("nan"), float("nan")])[0]),
                    "ci_95_accuracy_high": _safe_float((res.get("ci_95_accuracy") or [float("nan"), float("nan")])[1]),
                    "ci_95_delta_low": _safe_float((res.get("ci_95_delta_over_chance") or [float("nan"), float("nan")])[0]),
                    "ci_95_delta_high": _safe_float((res.get("ci_95_delta_over_chance") or [float("nan"), float("nan")])[1]),
                    "pseudo_r2": _safe_float(res.get("pseudo_r2")),
                    "n_train": int(res.get("n_train", 0)),
                    "n_test": int(res.get("n_test", 0)),
                    "projection_dim": int(projection_dim),
                    "tier": "tier2_conditional_mechanistic",
                    "primary_test_id": "3P2-C.3",
                    "mde_target": 0.35,
                    "achieved_power": 0.8,
                    "multiplicity_family": "tier2_holm_primary_tests",
                }
            )

    def _delta(group: str, target: str) -> float:
        return _safe_float(summary["group_metrics"].get(group, {}).get(target, {}).get("delta_over_chance"))

    baselines_ok = True
    for g in group_names:
        for target in ("boundary_indicator", "nearest_boundary_bin"):
            if not (_delta(g, target) > 0.10):
                baselines_ok = False

    low_within_10pct_any = False
    for target in ("boundary_indicator", "nearest_boundary_bin"):
        high = _safe_float(summary["group_metrics"].get("high_si", {}).get(target, {}).get("accuracy"))
        low = _safe_float(summary["group_metrics"].get("low_si", {}).get(target, {}).get("accuracy"))
        if np.isfinite(high) and high > 0 and np.isfinite(low):
            rel = low / high
            if rel >= 0.90:
                low_within_10pct_any = True

    summary["acceptance_checks"] = {
        "probe_baselines_over_chance_10pp_all_groups_targets": bool(baselines_ok),
        "low_si_within_10pct_relative_of_high_si_for_any_target": bool(low_within_10pct_any),
    }
    summary["interpretive_rule"] = (
        "Low-SI redundancy support requires low-SI decode performance "
        "within 10% relative of high-SI for at least one positional target."
    )
    summary["summary"] = {
        "supports_e2_redundancy_rule": bool(low_within_10pct_any),
        "runtime_seconds": float(time.time() - t0),
    }

    out_dir = output_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_rows_path = out_dir / "low_si_probe_results.parquet"
    probe_json_path = out_dir / "low_si_contribution_probe.json"
    pd.DataFrame(rows).to_parquet(probe_rows_path, index=False)
    _write_json(probe_json_path, summary)

    print(f"  wrote {probe_rows_path}", flush=True)
    print(f"  wrote {probe_json_path}", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3P2-C.3: low-SI contribution probe (non-destructive)")
    p.add_argument("--model", choices=["all", *TARGET_MODELS], default="all")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--device-map",
        default="",
        help="Comma-separated model:device map used when --model all",
    )
    p.add_argument("--output-root", default="results/experiment3_phase2/exp3p2c_redundancy_quantification")
    p.add_argument("--sequence-count", type=int, default=48)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--projection-dim", type=int, default=128)
    p.add_argument("--train-fraction", type=float, default=0.8)
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    models = list(TARGET_MODELS) if args.model == "all" else [args.model]
    device_map = _parse_device_map(args.device_map) if args.model == "all" else {}

    reports: dict[str, Any] = {}
    for model_name in models:
        device = device_map.get(model_name, args.device)
        reports[model_name] = run_model(
            model_name=model_name,
            device=device,
            output_root=output_root,
            sequence_count=max(2, int(args.sequence_count)),
            seq_len=max(64, int(args.seq_len)),
            batch_size=max(1, int(args.batch_size)),
            projection_dim=max(8, int(args.projection_dim)),
            train_fraction=float(max(0.5, min(0.95, args.train_fraction))),
            bootstrap_samples=max(200, int(args.bootstrap_samples)),
            seed=int(args.seed),
        )

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": reports,
    }
    _write_json(output_root / "c3_low_si_probe_summary.json", summary)
    print(f"[3P2-C.3] wrote {output_root / 'c3_low_si_probe_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
