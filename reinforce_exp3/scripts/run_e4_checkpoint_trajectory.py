#!/usr/bin/env python3
"""E4 — Checkpoint SI-Emergence Trajectory (OLMo-2-7B).

Computes per-head R² at multiple OLMo-2-7B training checkpoints to determine
whether SI strength is learned during training or present from initialization.

Usage:
    python reinforce_exp3/scripts/run_e4_checkpoint_trajectory.py \
        --checkpoints "allenai/OLMo-2-7B-step1000,allenai/OLMo-2-7B-step50000,allenai/OLMo-2-7B" \
        --device cuda:0 \
        [--n-eval-seqs 500]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp3.common import (  # noqa: E402
    B1_RESULTS,
    EXP3_SI_CIRCUITS,
    RESULTS_ROOT,
    ensure_dir,
    timestamp_now,
    write_json,
)
from reinforce_exp3.scripts._shared import (  # noqa: E402
    emit_core_artifacts,
    parse_device_map,
)

OUT_ROOT = RESULTS_ROOT / "E4_checkpoint_trajectory"
MODEL_NAME = "olmo-2-7b"

# OLMo-2-1124-7B stage checkpoints on HuggingFace.
# Intermediate checkpoints use "hf:allenai/OLMo-2-1124-7B@<branch>" syntax.
# The final checkpoint is the project-local "olmo-2-7b" (same model, local weights).
DEFAULT_CHECKPOINTS = [
    "hf:allenai/OLMo-2-1124-7B@stage2-ingredient3-step1000-tokens5B",
    "hf:allenai/OLMo-2-1124-7B@stage2-ingredient3-step3000-tokens13B",
    "hf:allenai/OLMo-2-1124-7B@stage2-ingredient3-step7000-tokens30B",
    "hf:allenai/OLMo-2-1124-7B@stage2-ingredient3-step11000-tokens47B",
    "olmo-2-7b",  # final checkpoint — project local weights
]

N_EVAL_SEQS = 500
SEQ_LEN = 128
MAX_OFFSET = 64   # max offset Δ to consider for g_h estimation
SEED_BASE = 20260429


# ---------------------------------------------------------------------------
# R² estimation for a single checkpoint
# ---------------------------------------------------------------------------

def _estimate_g_h(
    attn_matrix: np.ndarray,  # (seq_len, seq_len) mean attention weights
    max_offset: int,
) -> np.ndarray:
    """Estimate g_h(Δ) = mean attention weight as a function of offset Δ.

    Returns array of shape (max_offset + 1,) indexed by Δ = 0, 1, ..., max_offset.
    """
    seq_len = attn_matrix.shape[0]
    g_h = np.full(max_offset + 1, float("nan"))
    for delta in range(max_offset + 1):
        vals = []
        for i in range(seq_len):
            j = i - delta
            if 0 <= j < seq_len:
                vals.append(attn_matrix[i, j])
        if vals:
            g_h[delta] = float(np.mean(vals))
    return g_h


def _compute_head_r2(
    attn_matrix: np.ndarray,  # (seq_len, seq_len)
    g_h: np.ndarray,
    max_offset: int,
) -> float:
    """Compute R² of fitting g_h(i-j) to the observed attention matrix.

    R² = 1 - SS_res / SS_tot over all (i,j) pairs where 0 <= i-j <= max_offset.
    """
    y_vals: list[float] = []
    yhat_vals: list[float] = []
    seq_len = attn_matrix.shape[0]
    for i in range(seq_len):
        for j in range(seq_len):
            delta = i - j
            if 0 <= delta <= max_offset and not np.isnan(g_h[delta]):
                y_vals.append(float(attn_matrix[i, j]))
                yhat_vals.append(float(g_h[delta]))
    if not y_vals:
        return float("nan")
    y = np.array(y_vals)
    yhat = np.array(yhat_vals)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


@torch.no_grad()
def compute_r2_at_checkpoint(
    checkpoint_path: str,
    device: str,
    n_seqs: int = N_EVAL_SEQS,
    seq_len: int = SEQ_LEN,
    max_offset: int = MAX_OFFSET,
) -> pd.DataFrame:
    """Compute per-head R² for a single OLMo-2-7B checkpoint.

    Returns DataFrame with columns: layer, head, r2.
    """
    print(f"[E4] Loading checkpoint: {checkpoint_path}", flush=True)

    from reinforce_exp3.scripts._shared import load_model_for_exp  # noqa: PLC0415
    from experiment3.theory1_si_circuits import MODELS as _PROJ_MODELS  # noqa: PLC0415

    if checkpoint_path in _PROJ_MODELS:
        # Project-registered model name (e.g. "olmo-2-7b") — use local weights.
        # Must use eager attention so output_attentions=True works (OLMo uses SDPA).
        model, tokenizer = load_model_for_exp(
            checkpoint_path, device, attn_implementation="eager"
        )
    elif checkpoint_path.startswith("hf:"):
        # Format: "hf:<repo_id>@<revision>"
        hf_spec = checkpoint_path[3:]  # strip "hf:"
        if "@" in hf_spec:
            repo_id, revision = hf_spec.split("@", 1)
        else:
            repo_id, revision = hf_spec, "main"
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
        print(f"[E4]   HF repo={repo_id} revision={revision}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            revision=revision,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="eager",
        )
        model.eval()
    else:
        # Bare HF repo path — no revision
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="eager",
        )
        model.eval()

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads

    rng = random.Random(SEED_BASE)
    vocab_ids = list(tokenizer.get_vocab().values())

    # Accumulate mean attention matrices per (layer, head)
    attn_accum = np.zeros((n_layers, n_heads, seq_len, seq_len), dtype=float)
    attn_count = np.zeros((n_layers, n_heads), dtype=int)

    for seq_idx in range(n_seqs):
        prompt_ids = [rng.choice(vocab_ids) for _ in range(seq_len)]
        ids_tensor = torch.tensor([prompt_ids], device=device, dtype=torch.long)
        try:
            out = model(ids_tensor, output_attentions=True)
        except Exception as exc:
            print(f"[E4] Warning: forward pass failed for seq {seq_idx}: {exc}", flush=True)
            continue

        if out.attentions is None:
            continue

        for layer_idx, attn_layer in enumerate(out.attentions):
            if layer_idx >= n_layers:
                break
            _t = attn_layer[0].cpu().float()
            # Use tolist() to avoid torch-numpy bridge incompatibility (numpy 2.x)
            attn = np.array(_t.tolist(), dtype=np.float32)  # (n_heads, seq_len, seq_len)
            actual_heads = min(n_heads, attn.shape[0])
            attn_accum[layer_idx, :actual_heads] += attn[:actual_heads]
            attn_count[layer_idx, :actual_heads] += 1

        if (seq_idx + 1) % 50 == 0:
            print(f"[E4] {checkpoint_path}: {seq_idx + 1}/{n_seqs} sequences", flush=True)

    # Compute R² per head
    rows: list[dict[str, Any]] = []
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            cnt = attn_count[layer_idx, head_idx]
            if cnt == 0:
                rows.append({"layer": layer_idx, "head": head_idx, "r2": float("nan")})
                continue
            mean_attn = attn_accum[layer_idx, head_idx] / cnt  # (seq_len, seq_len)
            g_h = _estimate_g_h(mean_attn, max_offset)
            r2 = _compute_head_r2(mean_attn, g_h, max_offset)
            rows.append({"layer": layer_idx, "head": head_idx, "r2": r2})

    del model  # free GPU memory before loading next checkpoint
    torch.cuda.empty_cache()

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Trajectory analysis
# ---------------------------------------------------------------------------

def _extract_step_from_checkpoint(checkpoint_path: str) -> int:
    """Parse training step from checkpoint path.

    Handles formats:
    - "hf:allenai/OLMo-2-1124-7B@stage2-ingredient3-step3000-tokens13B" → 3000
    - "allenai/OLMo-2-7B-step50000" → 50000
    - "olmo-2-7b" (project local name) → -1 (final checkpoint)
    """
    import re
    m = re.search(r"step(\d+)", checkpoint_path, re.IGNORECASE)
    if m:
        return int(m.group(1))
    parts = checkpoint_path.replace("-", "_").split("_")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return -1  # final checkpoint (no step number)


def _cluster_trajectories(
    r2_over_time: np.ndarray,   # (n_checkpoints, n_heads)
    n_checkpoints: int,
    final_high_si_mask: np.ndarray,  # (n_heads,) bool
    final_low_si_mask: np.ndarray,   # (n_heads,) bool
) -> dict[str, Any]:
    """Summarize trajectory shape for high-SI and low-SI head groups."""
    hi_trajectories = r2_over_time[:, final_high_si_mask]   # (n_checkpoints, n_hi_heads)
    lo_trajectories = r2_over_time[:, final_low_si_mask]     # (n_checkpoints, n_lo_heads)

    mean_hi = float(np.nanmean(hi_trajectories, axis=1).tolist()[-1]) if hi_trajectories.size > 0 else float("nan")
    mean_lo = float(np.nanmean(lo_trajectories, axis=1).tolist()[-1]) if lo_trajectories.size > 0 else float("nan")

    # Categorize each head by trajectory type
    # "architectural": starts high (>0.3) at step 0
    # "learned": starts low (<0.1) at step 0 and grows to >0.3 at final
    # "suppressed": starts high (>0.3) at step 0 and ends low (<0.1)
    # "stable_low": stays low throughout
    trajectory_types: dict[str, int] = {
        "architectural": 0,
        "learned": 0,
        "suppressed": 0,
        "stable_low": 0,
        "mixed": 0,
    }

    if n_checkpoints < 2:
        return {"n_checkpoints": n_checkpoints, "trajectory_types": trajectory_types}

    n_total_heads = r2_over_time.shape[1]
    for h_idx in range(n_total_heads):
        traj = r2_over_time[:, h_idx]
        if np.all(np.isnan(traj)):
            continue
        r2_start = float(traj[0]) if not np.isnan(traj[0]) else float("nan")
        r2_end = float(traj[-1]) if not np.isnan(traj[-1]) else float("nan")
        if np.isnan(r2_start) or np.isnan(r2_end):
            continue
        if r2_start > 0.3 and r2_end > 0.3:
            trajectory_types["architectural"] += 1
        elif r2_start < 0.1 and r2_end > 0.3:
            trajectory_types["learned"] += 1
        elif r2_start > 0.3 and r2_end < 0.1:
            trajectory_types["suppressed"] += 1
        elif r2_start < 0.1 and r2_end < 0.1:
            trajectory_types["stable_low"] += 1
        else:
            trajectory_types["mixed"] += 1

    # Paper interpretation
    n_learned = trajectory_types["learned"]
    n_architectural = trajectory_types["architectural"]
    if n_learned > n_architectural:
        learned_framing = "supported"
        note = (
            f"{n_learned} heads show learned trajectories (start low, grow high) vs. "
            f"{n_architectural} architectural (start high, stay high). "
            "SI strength is primarily learned rather than architecturally imposed."
        )
    elif n_architectural > n_learned * 2:
        learned_framing = "not_supported"
        note = (
            f"{n_architectural} heads show architectural trajectories vs. {n_learned} learned. "
            "Much of OLMo's SI structure appears geometrically induced by RoPE initialization."
        )
    else:
        learned_framing = "mixed"
        note = (
            f"Mixed trajectory types: {n_learned} learned, {n_architectural} architectural, "
            f"{trajectory_types['mixed']} mixed. No clear dominant mechanism."
        )

    return {
        "n_checkpoints": n_checkpoints,
        "trajectory_types": trajectory_types,
        "learned_framing_of_si": learned_framing,
        "note": note,
        "mean_r2_hi_si_final": mean_hi,
        "mean_r2_lo_si_final": mean_lo,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="E4: Checkpoint SI-emergence trajectory for OLMo-2-7B",
        allow_abbrev=False,
    )
    p.add_argument(
        "--checkpoints",
        default=",".join(DEFAULT_CHECKPOINTS),
        help=(
            "Comma-separated HuggingFace checkpoint paths in chronological order. "
            "Example: 'allenai/OLMo-2-7B-step1000,allenai/OLMo-2-7B-step50000,allenai/OLMo-2-7B'"
        ),
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-root", default=str(OUT_ROOT))
    p.add_argument("--n-eval-seqs", type=int, default=N_EVAL_SEQS)
    args = p.parse_args()

    checkpoints = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
    out_dir = ensure_dir(Path(args.output_root))
    model_out = ensure_dir(out_dir / MODEL_NAME)
    start_ts = timestamp_now()

    print(f"[E4] Starting at {start_ts}", flush=True)
    print(f"[E4] Checkpoints: {checkpoints}", flush=True)

    # Save checkpoint metadata
    checkpoint_metadata = {
        "model": MODEL_NAME,
        "checkpoints": checkpoints,
        "training_steps": [_extract_step_from_checkpoint(c) for c in checkpoints],
        "n_eval_seqs": args.n_eval_seqs,
    }
    write_json(model_out / "checkpoint_metadata.json", checkpoint_metadata)

    # Compute R² at each checkpoint
    checkpoint_r2_dfs: list[pd.DataFrame] = []
    for ckpt in checkpoints:
        df = compute_r2_at_checkpoint(
            ckpt, args.device, n_seqs=args.n_eval_seqs
        )
        df["checkpoint"] = ckpt
        df["training_step"] = _extract_step_from_checkpoint(ckpt)
        checkpoint_r2_dfs.append(df)

    all_r2_df = pd.concat(checkpoint_r2_dfs, ignore_index=True)
    all_r2_df.to_parquet(model_out / "r2_per_checkpoint.parquet", index=False)

    # Build (n_checkpoints, n_heads) trajectory matrix
    n_checkpoints = len(checkpoints)
    # Identify unique (layer, head) pairs
    if checkpoint_r2_dfs:
        ref_df = checkpoint_r2_dfs[0]
        head_tuples = list(zip(ref_df["layer"].tolist(), ref_df["head"].tolist()))
        n_total_heads = len(head_tuples)
    else:
        head_tuples = []
        n_total_heads = 0

    r2_matrix = np.full((n_checkpoints, n_total_heads), float("nan"))
    for ck_idx, df in enumerate(checkpoint_r2_dfs):
        for h_idx, (layer, head) in enumerate(head_tuples):
            row = df[(df["layer"] == layer) & (df["head"] == head)]
            if not row.empty:
                r2_matrix[ck_idx, h_idx] = float(row["r2"].iloc[0])

    # Identify final-checkpoint high-SI and low-SI masks
    final_r2 = r2_matrix[-1] if n_checkpoints > 0 else np.array([])
    if len(final_r2) > 0:
        hi_thresh = float(np.nanpercentile(final_r2[np.isfinite(final_r2)], 75))
        lo_thresh = float(np.nanpercentile(final_r2[np.isfinite(final_r2)], 25))
        final_high_si = final_r2 >= hi_thresh
        final_low_si = final_r2 <= lo_thresh
    else:
        final_high_si = np.array([], dtype=bool)
        final_low_si = np.array([], dtype=bool)

    # Build trajectory DataFrame for paper figure
    traj_rows: list[dict[str, Any]] = []
    for ck_idx, ckpt in enumerate(checkpoints):
        step = _extract_step_from_checkpoint(ckpt)
        hi_mean = float(np.nanmean(r2_matrix[ck_idx, final_high_si])) if final_high_si.sum() > 0 else float("nan")
        lo_mean = float(np.nanmean(r2_matrix[ck_idx, final_low_si])) if final_low_si.sum() > 0 else float("nan")
        traj_rows.append({
            "checkpoint": ckpt,
            "training_step": step,
            "mean_r2_top_quartile_heads": hi_mean,
            "mean_r2_bottom_quartile_heads": lo_mean,
        })
    pd.DataFrame(traj_rows).to_parquet(model_out / "trajectory_clustering.parquet", index=False)

    # Trajectory type clustering
    trajectory_summary = _cluster_trajectories(
        r2_matrix, n_checkpoints, final_high_si, final_low_si
    )
    write_json(model_out / "trajectory_fit_summary.json", trajectory_summary)

    print(f"[E4] SI emergence: {trajectory_summary.get('learned_framing_of_si', 'n/a')}", flush=True)

    # Generate trajectory figure
    _save_trajectory_figure(traj_rows, out_dir / "figures" / "fig_r2_training_trajectory.pdf")

    # Emit governance artifacts
    learned_status = trajectory_summary.get("learned_framing_of_si", "unknown")
    claim_status_map = {"supported": "supported", "not_supported": "not_supported", "mixed": "mixed"}
    claim_status = claim_status_map.get(learned_status, "inconclusive")

    claim_impact = {
        "experiment_id": "E4",
        "claim_addressed": (
            "SI strength is a learned property (R² grows during training) rather than "
            "architecturally imposed by RoPE initialization"
        ),
        "claim_status": claim_status,
        "supports_main_text": claim_status in ("supported", "mixed"),
        "outcome_summary": trajectory_summary.get("note", ""),
        "trajectory_types": trajectory_summary.get("trajectory_types", {}),
        "notes": [trajectory_summary.get("note", "")],
    }

    preregistration = {
        "experiment_id": "E4",
        "hypothesis": (
            "For OLMo-2-7B, heads that are high-SI at the final checkpoint start with low R² "
            "at early checkpoints and grow to high R² during training (learned specialization). "
            "If architectural, R² would be high from initialization."
        ),
        "primary_criterion": (
            "n_learned_trajectory_heads > n_architectural_trajectory_heads"
        ),
        "model": MODEL_NAME,
        "checkpoints": checkpoints,
        "n_eval_seqs": args.n_eval_seqs,
        "timestamp": start_ts,
    }

    summary = {
        "experiment_id": "E4",
        "status": "complete",
        "model": MODEL_NAME,
        "n_checkpoints": n_checkpoints,
        "learned_framing_verdict": learned_status,
        "timestamp_start": start_ts,
        "timestamp_end": timestamp_now(),
    }

    data_dictionary = {
        "experiment_id": "E4",
        "tables": [
            {
                "path": f"{MODEL_NAME}/r2_per_checkpoint.parquet",
                "description": "Per-head R² at each training checkpoint",
                "columns": [
                    {"name": "checkpoint", "dtype": "str", "description": "HuggingFace checkpoint path"},
                    {"name": "training_step", "dtype": "int", "description": "Training step number"},
                    {"name": "layer", "dtype": "int", "description": "Layer index"},
                    {"name": "head", "dtype": "int", "description": "Head index"},
                    {"name": "r2", "dtype": "float", "description": "Shift-invariant R²"},
                ],
            },
            {
                "path": f"{MODEL_NAME}/trajectory_clustering.parquet",
                "description": "Mean R² over training steps for top/bottom-quartile head groups",
                "columns": [
                    {"name": "checkpoint", "dtype": "str", "description": "Checkpoint path"},
                    {"name": "training_step", "dtype": "int", "description": "Training step"},
                    {"name": "mean_r2_top_quartile_heads", "dtype": "float",
                     "description": "Mean R² for heads in final-checkpoint top quartile"},
                    {"name": "mean_r2_bottom_quartile_heads", "dtype": "float",
                     "description": "Mean R² for heads in final-checkpoint bottom quartile"},
                ],
            },
        ],
    }

    manifest_extra = {
        "model": MODEL_NAME,
        "checkpoints": checkpoints,
        "n_eval_seqs": args.n_eval_seqs,
        "seq_len": SEQ_LEN,
        "max_offset": MAX_OFFSET,
        "seed_base": SEED_BASE,
    }

    emit_core_artifacts(
        out_dir=out_dir,
        experiment_id="E4",
        preregistration=preregistration,
        manifest_extra=manifest_extra,
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )

    print(f"[E4] Done. Results in {out_dir}", flush=True)


def _save_trajectory_figure(
    traj_rows: list[dict[str, Any]],
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[E4] matplotlib not available; skipping trajectory figure", flush=True)
        return

    if not traj_rows:
        return

    steps = [r["training_step"] for r in traj_rows]
    hi_means = [r["mean_r2_top_quartile_heads"] for r in traj_rows]
    lo_means = [r["mean_r2_bottom_quartile_heads"] for r in traj_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, hi_means, "o-", label="Top-quartile SI heads (final ckpt)", color="steelblue")
    ax.plot(steps, lo_means, "s--", label="Bottom-quartile SI heads (final ckpt)", color="coral")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean R² (shift-invariant)")
    ax.set_title("OLMo-2-7B SI Structure Emergence During Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(str(out_path), format="pdf", dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as _plt
    _plt.close(fig)
    print(f"[E4] Saved trajectory figure: {out_path}", flush=True)


if __name__ == "__main__":
    main()
