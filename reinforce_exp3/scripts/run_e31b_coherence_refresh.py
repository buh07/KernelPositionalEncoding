#!/usr/bin/env python3
"""E31B (B-lite): coherence-gap correlation refresh using expanded model set."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp3.common import RESULTS_ROOT, ensure_dir, timestamp_now, write_json  # noqa: E402
from reinforce_exp3.scripts._shared import emit_core_artifacts  # noqa: E402

EXPERIMENT_ID = "E31B"
DEFAULT_OUT = RESULTS_ROOT / "E31b_coherence_refresh"
DEFAULT_E31A = RESULTS_ROOT / "E31a_breadth_consolidation" / "model_breadth_r2_table.csv"

# PE-type metadata and HF IDs for local config lookup.
MODEL_META: dict[str, dict[str, Any]] = {
    "llama-3.1-8b": {"hf_id": "meta-llama/Meta-Llama-3.1-8B", "pe_kind": "rope"},
    "mistral-7b-v0.1": {"hf_id": "mistralai/Mistral-7B-v0.1", "pe_kind": "rope"},
    "olmo-2-7b": {"hf_id": "allenai/OLMo-2-1124-7B", "pe_kind": "rope"},
    "gemma-2-9b": {"hf_id": "google/gemma-2-9b", "pe_kind": "rope"},
    "qwen2.5-7b": {"hf_id": "Qwen/Qwen2.5-7B", "pe_kind": "rope"},
    "pythia-1.4b": {"hf_id": "EleutherAI/pythia-1.4b", "pe_kind": "rope"},
    "pythia-410m": {"hf_id": "EleutherAI/pythia-410m", "pe_kind": "rope"},
    "tinyllama-1.1b": {"hf_id": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", "pe_kind": "rope"},
    "tinyllama-nope-1.1b": {"hf_id": "AntNLP/TinyLlama-NoPE-1.1B", "pe_kind": "none"},
    "gpt2-small": {"hf_id": "openai-community/gpt2", "pe_kind": "absolute"},
    "gpt2-medium": {"hf_id": "openai-community/gpt2-medium", "pe_kind": "absolute"},
}

# Fallback when config is unavailable in local cache.
FALLBACK_CONFIG: dict[str, dict[str, Any]] = {
    "llama-3.1-8b": {"head_dim": 128, "rope_theta": 500000.0, "max_position_embeddings": 131072},
    "mistral-7b-v0.1": {"head_dim": 128, "rope_theta": 10000.0, "max_position_embeddings": 32768},
    "olmo-2-7b": {"head_dim": 128, "rope_theta": 500000.0, "max_position_embeddings": 4096},
    "gemma-2-9b": {"head_dim": 224, "rope_theta": 10000.0, "max_position_embeddings": 8192},
    "qwen2.5-7b": {"head_dim": 128, "rope_theta": 1000000.0, "max_position_embeddings": 32768},
    "pythia-1.4b": {"head_dim": 128, "rope_theta": 10000.0, "max_position_embeddings": 2048},
    "pythia-410m": {"head_dim": 64, "rope_theta": 10000.0, "max_position_embeddings": 2048},
    "tinyllama-1.1b": {"head_dim": 64, "rope_theta": 10000.0, "max_position_embeddings": 2048},
    "tinyllama-nope-1.1b": {"head_dim": 64, "rope_theta": 10000.0, "max_position_embeddings": 2048},
    "gpt2-small": {"head_dim": 64, "rope_theta": 10000.0, "max_position_embeddings": 1024},
    "gpt2-medium": {"head_dim": 64, "rope_theta": 10000.0, "max_position_embeddings": 1024},
}


def _load_config(model: str) -> dict[str, Any]:
    meta = MODEL_META.get(model, {})
    pe_kind = str(meta.get("pe_kind", "rope"))
    conf = dict(FALLBACK_CONFIG.get(model, {}))
    conf["model"] = model
    conf["pe_kind"] = pe_kind
    conf["config_source"] = "fallback"

    hf_id = meta.get("hf_id")
    if not hf_id:
        return conf

    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(str(hf_id), local_files_only=True)
    except Exception as local_exc:
        # Best-effort online fallback for public models.
        try:
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(str(hf_id), local_files_only=False)
        except Exception as online_exc:
            conf["config_error"] = (
                f"local={type(local_exc).__name__}: {local_exc}; "
                f"online={type(online_exc).__name__}: {online_exc}"
            )
            return conf

    try:
        n_heads = int(getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 0)) or 0)
        hidden = int(getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 0)) or 0)
        if n_heads > 0 and hidden > 0:
            conf["head_dim"] = int(hidden // n_heads)
        conf["rope_theta"] = float(getattr(cfg, "rope_theta", conf.get("rope_theta", 10000.0)))
        conf["max_position_embeddings"] = int(
            getattr(cfg, "max_position_embeddings", conf.get("max_position_embeddings", 4096))
        )
        conf["rope_scaling"] = getattr(cfg, "rope_scaling", None)
        conf["model_type"] = str(getattr(cfg, "model_type", ""))
        conf["config_source"] = "autoconfig_local_or_online"
    except Exception as exc:
        conf["config_error"] = f"{type(exc).__name__}: {exc}"

    return conf


def _rope_frequencies(head_dim: int, theta_base: float) -> np.ndarray:
    d2 = max(1, int(head_dim) // 2)
    idx = np.arange(d2, dtype=np.float64)
    base = float(theta_base) if float(theta_base) > 0 else 10000.0
    return base ** (-2.0 * idx / float(head_dim))


def _mu1_for_model(meta: dict[str, Any]) -> float:
    pe_kind = str(meta.get("pe_kind", "rope"))
    head_dim = int(meta.get("head_dim", 64))
    if pe_kind == "none":
        return 0.0
    if pe_kind == "absolute":
        # Keep compatibility with Exp7A proxy treatment for learned absolute PE.
        theta = _rope_frequencies(max(64, head_dim), 10000.0)
        return float(abs(np.cos(theta).mean()))

    theta_base = float(meta.get("rope_theta", 10000.0))
    freqs = _rope_frequencies(head_dim, theta_base)
    return float(abs(np.cos(freqs).mean()))


def _welch_mu(head_dim: int, seq_len: int) -> float:
    d = float(max(1, int(head_dim)))
    n = float(max(2, int(seq_len)))
    return float(math.sqrt(max(0.0, (n - d) / (d * (n - 1.0)))))


def _perm_pvalue(x: np.ndarray, y: np.ndarray, *, n_perm: int, seed: int, one_sided_negative: bool = False) -> float:
    rng = np.random.default_rng(int(seed))
    obs = float(np.corrcoef(x, y)[0, 1])
    n = len(x)
    if n < 3:
        return float("nan")
    count = 0
    for _ in range(int(n_perm)):
        yp = y[rng.permutation(n)]
        r = float(np.corrcoef(x, yp)[0, 1])
        if one_sided_negative:
            if r <= obs:
                count += 1
        else:
            if abs(r) >= abs(obs):
                count += 1
    return float((count + 1) / (int(n_perm) + 1))


def run(*, out_root: Path, e31a_table: Path, seq_len_eval: int) -> dict[str, Any]:
    t0 = time.time()
    out_root = ensure_dir(out_root)

    if not e31a_table.exists():
        raise FileNotFoundError(f"[E31B] missing E31A table: {e31a_table}")
    breadth = pd.read_csv(e31a_table)
    if "model" not in breadth.columns or "mean_r2" not in breadth.columns:
        raise RuntimeError("[E31B] E31A table missing required columns: model, mean_r2")

    rows: list[dict[str, Any]] = []
    for model in breadth["model"].astype(str).tolist():
        cfg = _load_config(model)
        mu1 = _mu1_for_model(cfg)
        head_dim = int(cfg.get("head_dim", 64))
        mu_welch = _welch_mu(head_dim=head_dim, seq_len=int(seq_len_eval))
        eta = float(mu1 / mu_welch) if mu_welch > 0 else float("nan")
        rows.append(
            {
                "model": model,
                "pe_kind": str(cfg.get("pe_kind", "rope")),
                "config_source": str(cfg.get("config_source", "unknown")),
                "head_dim": int(head_dim),
                "rope_theta": float(cfg.get("rope_theta", 10000.0)),
                "seq_len_eval": int(seq_len_eval),
                "mu1": float(mu1),
                "mu_welch": float(mu_welch),
                "eta_welch_gap": float(eta),
                "config_error": str(cfg.get("config_error", "")),
            }
        )

    coh_df = pd.DataFrame(rows)
    merged = breadth.merge(coh_df, on="model", how="left")
    merged = merged.sort_values("mean_r2", ascending=False).reset_index(drop=True)

    merged.to_csv(out_root / "coherence_vs_r2_extended.csv", index=False)

    x = merged["eta_welch_gap"].to_numpy(dtype=np.float64)
    y = merged["mean_r2"].to_numpy(dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    pearson = scipy_stats.pearsonr(x, y) if len(x) >= 3 else None
    spearman = scipy_stats.spearmanr(x, y) if len(x) >= 3 else None
    p_perm_two = _perm_pvalue(x, y, n_perm=20000, seed=20260504, one_sided_negative=False)
    p_perm_one_neg = _perm_pvalue(x, y, n_perm=20000, seed=20260505, one_sided_negative=True)

    payload = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "n_models": int(len(merged)),
        "n_valid": int(len(x)),
        "seq_len_eval": int(seq_len_eval),
        "correlation": {
            "pearson_r": float(pearson.statistic) if pearson is not None else float("nan"),
            "pearson_p": float(pearson.pvalue) if pearson is not None else float("nan"),
            "spearman_rho": float(spearman.statistic) if spearman is not None else float("nan"),
            "spearman_p": float(spearman.pvalue) if spearman is not None else float("nan"),
            "perm_p_two_sided": float(p_perm_two),
            "perm_p_one_sided_negative": float(p_perm_one_neg),
        },
        "interpretation": "extended_coherence_correlation_refresh",
        "notes": [
            "Coherence values are proxy computations from PE geometry metadata and should be interpreted as a theoretical lens.",
            "This extends model count relative to Exp7A but remains observational/correlational.",
        ],
    }
    write_json(out_root / "coherence_vs_r2_extended.json", payload)

    pear_r = float(payload["correlation"]["pearson_r"])
    spe_r = float(payload["correlation"]["spearman_rho"])
    if np.isfinite(pear_r):
        if pear_r < -1e-12:
            direction_word = "inversely"
        elif pear_r > 1e-12:
            direction_word = "positively"
        else:
            direction_word = "approximately uncorrelated"
    else:
        direction_word = "not estimable"

    patch = f"""# Coherence Lens Patch (E31B)

Using an expanded model panel (n={len(merged)}), the PE-coherence proxy is {direction_word} correlated with mean SI amplitude.
At seq_len={seq_len_eval}, Pearson r={pear_r:.3f}, Spearman rho={spe_r:.3f},
with permutation p(two-sided)={payload['correlation']['perm_p_two_sided']:.4g}.

Scope: this is a CS-inspired geometric proxy lens, not a causal proof.
"""
    (out_root / "coherence_appendix_patch.md").write_text(patch, encoding="utf-8")

    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "analysis_tier": "hardening",
        "canonical_eligible": True,
        "override_used": False,
        "verdict": {
            "interpretation": "coherence_lens_extended",
            "n_models": int(len(merged)),
            "pearson_r": payload["correlation"]["pearson_r"],
            "spearman_rho": payload["correlation"]["spearman_rho"],
        },
        "elapsed_sec": float(time.time() - t0),
    }

    prereg = {
        "experiment_id": EXPERIMENT_ID,
        "question": "Does the PE coherence-gap lens remain predictive of SI amplitude on an expanded existing-model panel?",
        "primary_endpoint": "correlation between eta_welch_gap and mean_r2",
    }
    p_two = float(payload["correlation"]["perm_p_two_sided"])
    if np.isfinite(pear_r) and pear_r < 0 and np.isfinite(p_two) and p_two < 0.05:
        claim_status = "supported_with_caveat"
        impact = "extended panel remains directionally consistent with inverse coherence-gap interpretation"
    else:
        claim_status = "mixed"
        impact = "expanded panel yields mixed or non-significant direction; retain as exploratory proxy lens only"

    claim_impact = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "claim_status": claim_status,
        "impact": impact,
    }
    data_dict = {
        "coherence_vs_r2_extended.csv": "Per-model breadth stats joined with coherence proxy features.",
        "coherence_vs_r2_extended.json": "Correlation summary (Pearson/Spearman + permutation p-values).",
        "coherence_appendix_patch.md": "Paper-ready wording block for CS-lens integration.",
    }

    emit_core_artifacts(
        experiment_id=EXPERIMENT_ID,
        out_dir=out_root,
        preregistration=prereg,
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dict,
        manifest_extra={
            "inputs": {"e31a_table": str(e31a_table)},
            "seq_len_eval": int(seq_len_eval),
        },
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="E31B coherence-gap correlation refresh")
    p.add_argument("--output-root", default=str(DEFAULT_OUT))
    p.add_argument("--e31a-table", default=str(DEFAULT_E31A))
    p.add_argument("--seq-len", type=int, default=512)
    args = p.parse_args()

    summary = run(out_root=Path(args.output_root), e31a_table=Path(args.e31a_table), seq_len_eval=int(args.seq_len))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
