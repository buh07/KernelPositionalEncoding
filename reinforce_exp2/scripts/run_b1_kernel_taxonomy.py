#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp2.common import RESULTS_ROOT, command_manifest, ensure_dir, timestamp_now, write_json  # noqa: E402
from reinforce_exp2.scripts._shared import (  # noqa: E402
    attach_common_head_features,
    emit_core_artifacts,
    head_dataframe_from_kernels,
    hedges_g,
    holm_adjust_dict,
    ivw_meta,
    parse_models_arg,
    safe_float,
)

try:
    import torch
    from experiment3.theory1_si_circuits import MODELS as _MODELS  # noqa: E402
    from shared.models.loading import load_model, load_tokenizer  # noqa: E402
    _MODEL_INFERENCE_AVAILABLE = True
except ImportError:
    _MODEL_INFERENCE_AVAILABLE = False


def _unit_l2_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    return x / norms


def _zscore_cols(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x, axis=0, keepdims=True)
    sd = np.nanstd(x, axis=0, keepdims=True)
    sd = np.where(sd <= 1e-12, 1.0, sd)
    return (x - mu) / sd


def _spearman_one_sided_gt_with_exact_small_n(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[keep]
    yy = yy[keep]
    n = int(xx.size)
    if n < 3:
        return float("nan"), float("nan")

    rho_raw, p_two_asym = scipy_stats.spearmanr(xx, yy)
    rho = safe_float(rho_raw)
    if not np.isfinite(rho):
        return float("nan"), float("nan")

    if n <= 8:
        idx = list(range(n))
        vals = []
        for perm in itertools.permutations(idx):
            yp = yy[np.asarray(perm, dtype=np.int64)]
            r, _ = scipy_stats.spearmanr(xx, yp)
            r = safe_float(r)
            if np.isfinite(r):
                vals.append(float(r))
        if not vals:
            return rho, float("nan")
        arr = np.asarray(vals, dtype=np.float64)
        p_one = float(np.mean(arr >= rho))
        return rho, p_one

    p_two = safe_float(p_two_asym)
    if np.isfinite(p_two):
        p_one = float(p_two / 2.0) if rho > 0 else float(1.0 - p_two / 2.0)
    else:
        p_one = float("nan")
    return rho, p_one


def _descriptor_from_kernel(vec: np.ndarray) -> dict[str, float]:
    g = np.asarray(vec, dtype=np.float64)
    n = len(g)
    abs_g = np.abs(g)
    idx = int(np.argmax(abs_g))
    peak_abs = float(abs_g[idx])
    signed_peak = float(g[idx])
    half = 0.5 * peak_abs
    width = int(np.sum(abs_g >= half)) if np.isfinite(half) and half > 0 else 0

    lo = max(0, idx - 2)
    hi = min(n, idx + 3)
    local = float(np.sum(abs_g[lo:hi]))
    global_other = float(np.sum(abs_g) - local)
    ratio = float(local / max(global_other, 1e-8))

    freqs = np.fft.rfftfreq(n)
    amps = np.abs(np.fft.rfft(g))
    amp_sum = float(np.sum(amps))
    centroid = float(np.sum(freqs * amps) / max(amp_sum, 1e-12))
    p = amps / max(amp_sum, 1e-12)
    spec_entropy = float(-np.sum(p * np.log(p + 1e-12)))

    return {
        "peak_offset": float(idx),
        "peak_offset_norm": float(idx / max(1, n - 1)),
        "signed_peak_magnitude": signed_peak,
        "peak_abs_magnitude": peak_abs,
        "width_halfmax": float(width),
        "width_halfmax_frac": float(width / max(1, n)),
        "local_global_mass_ratio": ratio,
        "sign_at_zero": float(g[0]) if n else float("nan"),
        "spectral_centroid": centroid,
        "spectral_entropy": spec_entropy,
    }


def _spectral_features(vec: np.ndarray, top_k: int = 5) -> np.ndarray:
    g = np.asarray(vec, dtype=np.float64)
    fft = np.fft.rfft(g)
    amps = np.abs(fft)
    phases = np.angle(fft)
    freqs = np.fft.rfftfreq(len(g))

    # drop DC for top-frequency selection
    work_amps = amps.copy()
    if work_amps.size > 0:
        work_amps[0] = 0.0
    idx = np.argsort(work_amps)[::-1][:top_k]

    feat: list[float] = []
    for i in idx.tolist():
        feat.append(float(amps[i]))
        feat.append(float(phases[i]))
        feat.append(float(freqs[i]))

    amp_sum = float(np.sum(amps))
    centroid = float(np.sum(freqs * amps) / max(amp_sum, 1e-12))
    p = amps / max(amp_sum, 1e-12)
    entropy = float(-np.sum(p * np.log(p + 1e-12)))
    feat.extend([centroid, entropy])
    return np.asarray(feat, dtype=np.float64)


def _build_feature_spaces(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    kernels = np.vstack(df["kernel"].to_numpy())

    # offset_space
    offset = _unit_l2_rows(kernels)

    # spectral_space: zscore per column then unit L2
    spectral = np.vstack([_spectral_features(v) for v in kernels])
    spectral = _unit_l2_rows(_zscore_cols(spectral))

    # descriptor_space: zscore per column (no row unit-L2)
    desc_rows = [_descriptor_from_kernel(v) for v in kernels]
    desc_df = pd.DataFrame(desc_rows)
    desc = _zscore_cols(desc_df.to_numpy(dtype=np.float64))

    return offset, spectral, desc, desc_df


def _fit_kmeans(x: np.ndarray, k: int, seed: int, n_init: int = 8) -> np.ndarray:
    km = KMeans(n_clusters=int(k), random_state=int(seed), n_init=max(1, int(n_init)))
    return km.fit_predict(x)


def _fit_hierarchical(x: np.ndarray, k: int) -> np.ndarray:
    # hierarchical clustering with cosine distance
    if x.shape[0] < 2:
        return np.zeros(x.shape[0], dtype=int)
    z = linkage(pdist(x, metric="cosine"), method="average")
    lab = fcluster(z, t=int(k), criterion="maxclust") - 1
    return lab.astype(int)


def _label_map_first(indices: np.ndarray, labels: np.ndarray) -> dict[int, int]:
    out: dict[int, int] = {}
    for i, l in zip(indices.tolist(), labels.tolist()):
        if i not in out:
            out[int(i)] = int(l)
    return out


def _bootstrap_ari(x: np.ndarray, k: int, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(int(seed))
    n = int(x.shape[0])
    if n < max(4, k + 1):
        return float("nan"), float("nan")

    vals: list[float] = []
    for _ in range(max(10, int(n_boot))):
        idx1 = rng.integers(0, n, size=n)
        idx2 = rng.integers(0, n, size=n)
        x1 = x[idx1]
        x2 = x[idx2]
        try:
            # Bootstrap ARI is the dominant runtime path; keep n_init low here.
            l1 = _fit_kmeans(x1, k=k, seed=int(rng.integers(0, 1_000_000)), n_init=2)
            l2 = _fit_kmeans(x2, k=k, seed=int(rng.integers(0, 1_000_000)), n_init=2)
        except Exception:
            continue
        m1 = _label_map_first(idx1, l1)
        m2 = _label_map_first(idx2, l2)
        common = sorted(set(m1.keys()) & set(m2.keys()))
        if len(common) < max(5, k):
            continue
        y1 = [m1[i] for i in common]
        y2 = [m2[i] for i in common]
        vals.append(float(adjusted_rand_score(y1, y2)))

    if not vals:
        return float("nan"), float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0)


def _select_k(descriptor_space: np.ndarray, n_boot: int, seed: int) -> tuple[int, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for k in range(3, 11):
        if descriptor_space.shape[0] <= k:
            continue
        ari_mean, ari_std = _bootstrap_ari(descriptor_space, k=k, n_boot=n_boot, seed=seed + k * 131)
        labels = _fit_kmeans(descriptor_space, k=k, seed=seed + 17, n_init=8)
        sil = float(silhouette_score(descriptor_space, labels)) if len(np.unique(labels)) > 1 else float("nan")
        rows.append({"k": int(k), "bootstrap_ari_mean": ari_mean, "bootstrap_ari_std": ari_std, "silhouette": sil})
    ks = pd.DataFrame(rows)
    if ks.empty:
        raise RuntimeError("Unable to evaluate candidate k in [3..10]")
    ks = ks.sort_values(["bootstrap_ari_mean", "silhouette", "k"], ascending=[False, False, True]).reset_index(drop=True)
    best_k = int(ks.iloc[0]["k"])
    return best_k, ks


def _reliability_gate_from_shards(
    model,
    device: str,
    calibration_tokens: list[list[int]],
    n_shards: int = 4,
    max_seqs_per_shard: int = 64,
) -> dict[str, Any]:
    """Compute per-shard kernel reliability using actual model attention weights."""
    import torch as _torch
    model.eval()

    n_seqs = len(calibration_tokens)
    shard_size = max(1, n_seqs // n_shards)
    shards: list[list[list[int]]] = []
    for s in range(n_shards):
        start = s * shard_size
        end = start + shard_size if s < n_shards - 1 else n_seqs
        seqs = calibration_tokens[start:end]
        if max_seqs_per_shard > 0:
            seqs = seqs[:max_seqs_per_shard]
        shards.append(seqs)

    # shard_kernels[head_key][shard_idx] = g_h,s as np.ndarray of shape (max_offset,)
    shard_kernels: dict[str, list[np.ndarray]] = {}
    max_seq_len = max(len(s) for ss in shards for s in ss) if any(shards) else 512

    for shard_idx, shard_seqs in enumerate(shards):
        # accumulate attention weights by (layer, head, offset)
        offset_sums: dict[str, np.ndarray] = {}
        offset_counts: dict[str, np.ndarray] = {}

        for seq in shard_seqs:
            toks = _torch.tensor([seq], dtype=_torch.long, device=device)
            with _torch.inference_mode():
                out = model(input_ids=toks, output_attentions=True, use_cache=False)
            attentions = out.attentions  # tuple: (n_layers,) each (1, n_heads, L, L)

            for layer_idx, attn_t in enumerate(attentions):
                attn_np = attn_t[0].cpu().float().numpy()  # (n_heads, L, L)
                n_heads, L, _ = attn_np.shape
                i_idx, j_idx = np.tril_indices(L)
                deltas = i_idx - j_idx  # offset delta = i - j >= 0

                for h in range(n_heads):
                    hk = f"L{layer_idx}H{h}"
                    w = attn_np[h][i_idx, j_idx]
                    if hk not in offset_sums:
                        offset_sums[hk] = np.zeros(L, dtype=np.float64)
                        offset_counts[hk] = np.zeros(L, dtype=np.float64)
                    # only accumulate up to the kernel length we've allocated
                    m = min(L, offset_sums[hk].shape[0])
                    np.add.at(offset_sums[hk][:m], np.clip(deltas, 0, m - 1), w)
                    np.add.at(offset_counts[hk][:m], np.clip(deltas, 0, m - 1), 1.0)

            del toks, out
            if device.startswith("cuda"):
                _torch.cuda.empty_cache()

        for hk in offset_sums:
            g = np.where(offset_counts[hk] > 0, offset_sums[hk] / np.maximum(offset_counts[hk], 1e-12), 0.0)
            if hk not in shard_kernels:
                shard_kernels[hk] = []
            shard_kernels[hk].append(g)

    # compute split-half correlation: combine shards 0+1 vs 2+3
    cors: list[float] = []
    snr_vals: list[float] = []
    for hk, shard_list in shard_kernels.items():
        if len(shard_list) < 4:
            continue
        min_len = min(len(s) for s in shard_list)
        arr = np.stack([s[:min_len] for s in shard_list], axis=0)  # (n_shards, max_offset)
        half1 = np.mean(arr[:2], axis=0)
        half2 = np.mean(arr[2:4], axis=0)
        if np.std(half1) > 1e-9 and np.std(half2) > 1e-9:
            cors.append(float(np.corrcoef(half1, half2)[0, 1]))
        mu_h = np.mean(arr, axis=0)
        p_signal = float(np.var(mu_h))
        # residual: mean over shards of variance of (shard - mu)
        resid_per_shard = np.var(arr - mu_h[None, :], axis=1)
        p_resid = float(np.mean(resid_per_shard))
        snr_vals.append(p_signal / max(p_resid, 1e-8))

    median_split_half = float(np.nanmedian(np.asarray(cors, dtype=np.float64))) if cors else float("nan")
    median_snr = float(np.nanmedian(np.asarray(snr_vals, dtype=np.float64))) if snr_vals else float("nan")
    high_corr_frac = float(np.mean(np.asarray(cors, dtype=np.float64) >= 0.50)) if cors else float(0.0)

    eligible = bool(
        np.isfinite(median_split_half) and median_split_half >= 0.30
        and np.isfinite(median_snr) and median_snr >= 1.20
        and high_corr_frac >= 0.25
    )
    return {
        "median_split_half_corr": median_split_half,
        "median_kernel_snr": median_snr,
        "fraction_corr_ge_0_50": high_corr_frac,
        "reliability_eligible": eligible,
        "n_heads_evaluated": int(len(cors)),
        "gate_method": "shard_wise_attention",
    }


def _sanitize_calibration_tokens_for_vocab(
    calibration_tokens: list[list[int]],
    vocab_size: int,
    min_len: int = 16,
) -> list[list[int]]:
    out: list[list[int]] = []
    vmax = max(1, int(vocab_size))
    for seq in calibration_tokens:
        if not isinstance(seq, list) or not seq:
            continue
        ok = all(isinstance(x, (int, np.integer)) and 0 <= int(x) < vmax for x in seq)
        if not ok:
            continue
        arr = [int(x) for x in seq]
        if len(arr) < int(min_len):
            continue
        out.append(arr)
    return out


def _ensure_output_attentions_supported(model, device: str, probe_tokens: list[int]) -> None:
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ("_attn_implementation", "attn_implementation"):
            if hasattr(cfg, attr):
                try:
                    setattr(cfg, attr, "eager")
                except Exception:
                    pass
    ids = torch.tensor([probe_tokens[: min(32, len(probe_tokens))]], dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model(input_ids=ids, output_attentions=True, use_cache=False)
    atts = getattr(out, "attentions", None)
    if atts is None or len(atts) == 0:
        raise RuntimeError("output_attentions returned empty/None; cannot run shard-wise reliability gate")
    del ids, out
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()


def _reliability_gate(df: pd.DataFrame) -> dict[str, Any]:
    """Fallback proxy gate used when model inference is not available."""
    kernels = np.vstack(df["kernel"].to_numpy())

    # split-half proxy over sequence shards of the pre-computed kernel vectors
    # NOTE: this is a structural-smoothness proxy, not true shard-wise reliability.
    # Use _reliability_gate_from_shards when model inference is available.
    half = kernels.shape[0] // 2
    left = kernels[:half]
    right = kernels[half : half * 2]
    cors = []
    for i in range(min(left.shape[0], right.shape[0])):
        a, b = left[i], right[i]
        if np.std(a) <= 1e-9 or np.std(b) <= 1e-9:
            continue
        cors.append(float(np.corrcoef(a, b)[0, 1]))
    median_split_half = float(np.nanmedian(cors)) if cors else float("nan")

    # SNR proxy based on signal variance vs local residual variance
    snr_vals = []
    for row in kernels:
        signal = float(np.var(row))
        resid = float(np.var(np.diff(row)))
        snr_vals.append(signal / max(resid, 1e-8))
    median_snr = float(np.nanmedian(np.asarray(snr_vals, dtype=np.float64))) if snr_vals else float("nan")

    high_corr_frac = float(np.mean(np.asarray(cors, dtype=np.float64) >= 0.50)) if cors else float(0.0)

    eligible = bool(np.isfinite(median_split_half) and median_split_half >= 0.30 and np.isfinite(median_snr) and median_snr >= 1.20 and high_corr_frac >= 0.25)
    return {
        "median_split_half_corr": median_split_half,
        "median_kernel_snr": median_snr,
        "fraction_corr_ge_0_50": high_corr_frac,
        "reliability_eligible": eligible,
        "gate_method": "proxy_fallback",
    }


def _depth_control_enrichment(model_df: pd.DataFrame, labels: np.ndarray) -> tuple[pd.DataFrame, dict[int, dict[str, Any]]]:
    work = model_df.copy()
    work["cluster"] = labels
    cluster_ids = sorted(int(x) for x in np.unique(labels))

    rows: list[dict[str, Any]] = []
    agg: dict[int, dict[str, Any]] = {}

    for c in cluster_ids:
        strata_eff: list[float] = []
        strata_var: list[float] = []
        strata_n: list[int] = []
        for layer, ldf in work.groupby("layer"):
            in_c = ldf[(ldf["cluster"] == c) & np.isfinite(ldf["boundary_attn_score"])]["boundary_attn_score"].to_numpy(dtype=np.float64)
            out_c = ldf[(ldf["cluster"] != c) & np.isfinite(ldf["boundary_attn_score"])]["boundary_attn_score"].to_numpy(dtype=np.float64)
            n1, n2 = len(in_c), len(out_c)
            if n1 < 3 or n2 < 10:
                continue
            g = hedges_g(in_c, out_c)
            # Approx variance proxy for SMD
            var_g = float((n1 + n2) / max(n1 * n2, 1) + (g * g) / max(2 * (n1 + n2 - 2), 1))
            strata_eff.append(g)
            strata_var.append(var_g)
            strata_n.append(n1 + n2)
            rows.append(
                {
                    "cluster": int(c),
                    "layer": int(layer),
                    "n_cluster": int(n1),
                    "n_noncluster": int(n2),
                    "hedges_g": float(g),
                    "variance": var_g,
                }
            )
        mu, lo, hi = ivw_meta(strata_eff, strata_var)
        if np.isfinite(mu):
            se = float((hi - lo) / (2 * 1.96)) if np.isfinite(lo) and np.isfinite(hi) else float("nan")
            z = float(mu / max(se, 1e-8)) if np.isfinite(se) else float("nan")
            p_two = float(2 * (1.0 - scipy_stats.norm.cdf(abs(z)))) if np.isfinite(z) else float("nan")
        else:
            p_two = float("nan")
        agg[int(c)] = {
            "cluster": int(c),
            "ivw_hedges_g": float(mu),
            "ivw_ci95": [float(lo), float(hi)],
            "p_two": float(p_two),
            "n_valid_layers": int(len(strata_eff)),
        }
    return pd.DataFrame(rows), agg


def _cross_model_prototype_similarity(
    proto_by_model: dict[str, dict[int, np.ndarray]],
    k: int,
    threshold: float = 0.70,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model_pairs = []
    models = sorted(proto_by_model.keys())
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model_pairs.append((models[i], models[j]))

    pair_support = {}
    for a, b in model_pairs:
        pa = proto_by_model[a]
        pb = proto_by_model[b]
        ids_a = sorted(pa.keys())
        ids_b = sorted(pb.keys())
        if not ids_a or not ids_b:
            continue
        sim = np.zeros((len(ids_a), len(ids_b)), dtype=np.float64)
        for i, ca in enumerate(ids_a):
            va = pa[ca]
            na = np.linalg.norm(va)
            for j, cb in enumerate(ids_b):
                vb = pb[cb]
                nb = np.linalg.norm(vb)
                cos = float(np.dot(va, vb) / max(na * nb, 1e-12))
                sim[i, j] = cos

        # greedy matching for simple stable correspondence
        used_r: set[int] = set()
        used_c: set[int] = set()
        flat = [(-sim[r, c], r, c) for r in range(sim.shape[0]) for c in range(sim.shape[1])]
        flat.sort()
        matched = []
        for neg, r, c in flat:
            if r in used_r or c in used_c:
                continue
            used_r.add(r)
            used_c.add(c)
            cos = -neg
            matched.append(cos)
            rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "cluster_a": int(ids_a[r]),
                    "cluster_b": int(ids_b[c]),
                    "shape_only_cosine": float(cos),
                    "passes_0_70": bool(cos >= threshold),
                }
            )

        good = int(sum(1 for x in matched if x >= threshold))
        pair_support[f"{a}__{b}"] = {
            "n_matches": int(len(matched)),
            "n_good": int(good),
            "criterion_min_required": int(min(3, k - 1)),
            "supports_pair": bool(good >= min(3, k - 1)),
        }

    return pd.DataFrame(rows), pair_support


_PARTIAL_SCHEMA_VERSION = 1


def _truthy(raw: str) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_device_map(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in [x.strip() for x in str(raw).split(",") if x.strip()]:
        if ":" in tok:
            m, d = tok.split(":", 1)
            out[m.strip()] = d.strip()
    return out


def _validate_device_map_for_gpu_inference(
    *,
    models: list[str],
    device_map: dict[str, str],
    calibration_tokens: list[list[int]],
) -> None:
    """Fail fast on malformed/missing model->device mapping for shard-wise inference."""
    if not _MODEL_INFERENCE_AVAILABLE:
        return
    if not calibration_tokens:
        return
    inference_models = [m for m in models if m in _MODELS]
    missing = [m for m in inference_models if not str(device_map.get(m, "")).strip()]
    if missing:
        raise ValueError(
            "Missing model-specific --device-map entries for shard-wise B1 inference: "
            f"{missing}. Provide explicit mappings like "
            "'--device-map llama-3.1-8b:cuda:0,olmo-2-7b:cuda:1,mistral-7b-v0.1:cuda:2' "
            "or set an explicit non-empty per-model target (e.g., ':cpu')."
        )


def _load_calibration_tokens(calibration_root: Path) -> list[list[int]]:
    calibration_tokens: list[list[int]] = []
    cal_tok_path = Path(calibration_root) / "calibration_v1_tokens.parquet"
    if cal_tok_path.exists():
        try:
            cal_df = pd.read_parquet(cal_tok_path)
            for row in cal_df.itertuples():
                toks = row.tokens
                if isinstance(toks, (list, np.ndarray)):
                    calibration_tokens.append([int(x) for x in toks])
        except Exception:
            pass
    return calibration_tokens


def _stable_model_index(models: list[str], model: str) -> int:
    if model not in models:
        raise ValueError(f"Model '{model}' not in models list: {models}")
    return int(models.index(model))


def _partial_paths(partial_root: Path, model: str) -> dict[str, Path]:
    safe_model = str(model).replace("/", "_")
    model_root = ensure_dir(partial_root / safe_model)
    return {
        "root": model_root,
        "manifest": model_root / "partial_manifest.json",
        "membership": model_root / "cluster_membership.parquet",
        "stability": model_root / "cluster_stability_report.parquet",
        "enrichment": model_root / "cluster_function_enrichment.parquet",
        "summary": model_root / "model_summary.json",
        "proto": model_root / "prototype_shape.json",
    }


def _partial_config_hash(
    *,
    model: str,
    models: list[str],
    seed: int,
    n_boot: int,
    n_shards: int,
    max_seqs_per_shard: int,
    calibration_root: Path,
) -> str:
    payload = {
        "schema": "b1_partial_v1",
        "model": str(model),
        "models": [str(x) for x in models],
        "seed": int(seed),
        "n_boot": int(n_boot),
        "n_shards": int(n_shards),
        "max_seqs_per_shard": int(max_seqs_per_shard),
        "calibration_root": str(Path(calibration_root).resolve()),
    }
    txt = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def _partial_ready(partial_root: Path, model: str, expected_hash: str) -> tuple[bool, list[str]]:
    paths = _partial_paths(partial_root, model)
    errs: list[str] = []
    for key in ["manifest", "membership", "stability", "enrichment", "summary", "proto"]:
        p = paths[key]
        if not p.exists():
            errs.append(f"missing {key}: {p}")
    if errs:
        return False, errs
    try:
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    except Exception as exc:
        return False, [f"invalid partial manifest: {exc}"]
    if int(manifest.get("schema_version", -1)) != int(_PARTIAL_SCHEMA_VERSION):
        errs.append("schema_version mismatch")
    if str(manifest.get("model", "")) != str(model):
        errs.append("model mismatch")
    if str(manifest.get("config_hash", "")) != str(expected_hash):
        errs.append("config_hash mismatch")
    if str(manifest.get("status", "")) != "ok":
        errs.append("status != ok")
    return (len(errs) == 0), errs


def _write_partial(payload: dict[str, Any], partial_root: Path, config_hash: str) -> None:
    model = str(payload["model"])
    paths = _partial_paths(partial_root, model)

    membership_df = payload["membership_df"].copy()
    stability_df = payload["stability_df"].copy()
    enrichment_df = payload["enrichment_df"].copy()
    model_summary = payload["model_summary"]
    proto_shape = payload["proto_shape"]

    membership_df.to_parquet(paths["membership"], index=False)
    stability_df.to_parquet(paths["stability"], index=False)
    enrichment_df.to_parquet(paths["enrichment"], index=False)
    write_json(paths["summary"], model_summary)

    proto_json = {str(k): [float(x) for x in np.asarray(v, dtype=np.float64).tolist()] for k, v in proto_shape.items()}
    write_json(paths["proto"], proto_json)

    manifest = {
        "timestamp": timestamp_now(),
        "schema_version": int(_PARTIAL_SCHEMA_VERSION),
        "status": "ok",
        "model": model,
        "config_hash": str(config_hash),
        "n_membership_rows": int(membership_df.shape[0]),
        "n_stability_rows": int(stability_df.shape[0]),
        "n_enrichment_rows": int(enrichment_df.shape[0]),
        "paths": {k: str(v) for k, v in paths.items()},
    }
    write_json(paths["manifest"], manifest)


def _load_partial(partial_root: Path, model: str) -> dict[str, Any]:
    paths = _partial_paths(partial_root, model)
    proto_raw = json.loads(paths["proto"].read_text(encoding="utf-8"))
    proto = {int(k): np.asarray(v, dtype=np.float64) for k, v in proto_raw.items()}
    model_summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    gate = model_summary.get("reliability_gate", {})
    eligible = bool(gate.get("reliability_eligible", False))
    concordance = safe_float(model_summary.get("descriptor_vs_spectral_ari"))
    secondary = model_summary.get("secondary_behavioral_direction", {})
    directional_consistent = bool(secondary.get("directionally_consistent", False))
    cluster_enrichment = model_summary.get("cluster_enrichment", [])
    has_depth_support = bool(any(bool(row.get("supports_depth_control_enrichment", False)) for row in cluster_enrichment))
    return {
        "model": str(model),
        "membership_df": pd.read_parquet(paths["membership"]),
        "stability_df": pd.read_parquet(paths["stability"]),
        "enrichment_df": pd.read_parquet(paths["enrichment"]),
        "model_summary": model_summary,
        "proto_shape": proto,
        "eligible": bool(eligible),
        "concordance": float(concordance),
        "depth_support": bool(eligible and has_depth_support),
        "directional_ok": bool(eligible and directional_consistent),
    }


def _compute_model_payload(
    *,
    model: str,
    model_index: int,
    device_map: dict[str, str],
    calibration_tokens: list[list[int]],
    n_boot: int,
    n_shards: int,
    max_seqs_per_shard: int,
    seed: int,
) -> dict[str, Any]:
    mdf = head_dataframe_from_kernels(model)
    mdf = attach_common_head_features(model, mdf)

    device = device_map.get(model, "")
    if _MODEL_INFERENCE_AVAILABLE and device and calibration_tokens and model in _MODELS:
        model_obj = None
        loaded = None
        try:
            tok = load_tokenizer(_MODELS[model])
            vocab_size = int(getattr(tok, "vocab_size", 0) or 0)
            safe_tokens = _sanitize_calibration_tokens_for_vocab(calibration_tokens, vocab_size=vocab_size, min_len=16)
            if len(safe_tokens) < 8:
                raise RuntimeError(
                    f"Insufficient vocab-safe calibration sequences ({len(safe_tokens)}) for shard-wise reliability gate"
                )
            if device == "auto":
                loaded = load_model(_MODELS[model], device_map="auto")
                model_obj = loaded.model
                _input_device = str(next(model_obj.parameters()).device)
            else:
                loaded = load_model(_MODELS[model])
                model_obj = loaded.model.to(device)
                _input_device = device
            model_obj.eval()
            _ensure_output_attentions_supported(model_obj, _input_device, safe_tokens[0])
            gate = _reliability_gate_from_shards(
                model_obj,
                device=_input_device,
                calibration_tokens=safe_tokens,
                n_shards=int(n_shards),
                max_seqs_per_shard=int(max_seqs_per_shard),
            )
        except Exception as exc:
            gate = {
                "median_split_half_corr": float("nan"),
                "median_kernel_snr": float("nan"),
                "fraction_corr_ge_0_50": float("nan"),
                "reliability_eligible": False,
                "n_heads_evaluated": 0,
                "gate_method": "shard_wise_attention_failed",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "failure_traceback_tail": traceback.format_exc(limit=3),
            }
        finally:
            try:
                if model_obj is not None:
                    del model_obj
            except Exception:
                pass
            try:
                if loaded is not None:
                    del loaded
            except Exception:
                pass
            try:
                load_model.cache_clear()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    else:
        gate = _reliability_gate(mdf)

    eligible = bool(gate["reliability_eligible"])
    offset_x, spectral_x, desc_x, desc_df = _build_feature_spaces(mdf)

    model_seed = int(seed) + int(model_index) * 911
    best_k, ks = _select_k(desc_x, n_boot=int(n_boot), seed=model_seed)
    lab_desc_kmeans = _fit_kmeans(desc_x, k=best_k, seed=int(seed) + 101 + int(model_index), n_init=8)
    lab_desc_hier = _fit_hierarchical(desc_x, k=best_k)
    lab_spec_kmeans = _fit_kmeans(spectral_x, k=best_k, seed=int(seed) + 203 + int(model_index), n_init=8)
    lab_off_kmeans = _fit_kmeans(offset_x, k=best_k, seed=int(seed) + 307 + int(model_index), n_init=8)

    concordance = float(adjusted_rand_score(lab_desc_kmeans, lab_spec_kmeans))

    membership_rows: list[dict[str, Any]] = []
    for i, row in mdf.reset_index(drop=True).iterrows():
        membership_rows.append(
            {
                "model": model,
                "layer": int(row["layer"]),
                "head": int(row["head"]),
                "head_key": str(row["head_key"]),
                "cluster_descriptor_kmeans": int(lab_desc_kmeans[i]),
                "cluster_descriptor_hier": int(lab_desc_hier[i]),
                "cluster_spectral_kmeans": int(lab_spec_kmeans[i]),
                "cluster_offset_kmeans": int(lab_off_kmeans[i]),
                "mean_r2": safe_float(row.get("mean_r2")),
                "boundary_attn_score": safe_float(row.get("boundary_attn_score")),
                "is_high_si": bool(row.get("is_high_si", False)),
                "is_low_si": bool(row.get("is_low_si", False)),
            }
        )
    membership_df = pd.DataFrame(membership_rows)

    ks = ks.copy()
    ks["model"] = model
    ks["selected_k"] = int(best_k)
    ks["reliability_eligible"] = bool(eligible)

    strata_df, agg = _depth_control_enrichment(mdf, lab_desc_kmeans)
    if not strata_df.empty:
        strata_df["model"] = model
    pmap = {f"c{c}": safe_float(v.get("p_two")) for c, v in agg.items()}
    p_adj = holm_adjust_dict(pmap)

    cluster_enrichment = []
    has_depth_support = False
    cluster_boundary_effect = {}
    for c, info in agg.items():
        p_h = safe_float(p_adj.get(f"c{c}")) if f"c{c}" in p_adj else float("nan")
        g = safe_float(info.get("ivw_hedges_g"))
        support = bool(np.isfinite(p_h) and p_h < 0.05 and np.isfinite(g) and abs(g) >= 0.20)
        if support:
            has_depth_support = True
        cluster_boundary_effect[int(c)] = g
        cluster_enrichment.append(
            {
                "cluster": int(c),
                "ivw_hedges_g": g,
                "ivw_ci95": info.get("ivw_ci95"),
                "p_two": safe_float(info.get("p_two")),
                "p_two_holm_model": p_h,
                "n_valid_layers": int(info.get("n_valid_layers", 0)),
                "supports_depth_control_enrichment": support,
            }
        )

    cluster_mean_r2 = {}
    for c in sorted(set(lab_desc_kmeans.tolist())):
        vals = mdf.loc[np.asarray(lab_desc_kmeans) == int(c), "mean_r2"].to_numpy(dtype=np.float64)
        cluster_mean_r2[int(c)] = float(np.nanmean(vals)) if vals.size else float("nan")

    common_clusters = sorted(set(cluster_boundary_effect.keys()) & set(cluster_mean_r2.keys()))
    rho = float("nan")
    p_one = float("nan")
    if len(common_clusters) >= 3:
        x = np.asarray([cluster_boundary_effect[c] for c in common_clusters], dtype=np.float64)
        y = np.asarray([cluster_mean_r2[c] for c in common_clusters], dtype=np.float64)
        rho, p_one = _spearman_one_sided_gt_with_exact_small_n(x, y)
    directional_ok = bool(np.isfinite(rho) and rho >= 0.20 and np.isfinite(p_one) and p_one < 0.05)

    desc_work = desc_df.copy()
    desc_work["cluster"] = lab_desc_kmeans
    proto_shape: dict[int, np.ndarray] = {}
    for c, cdf in desc_work.groupby("cluster"):
        fields = ["peak_offset_norm", "width_halfmax_frac", "sign_at_zero", "spectral_centroid", "spectral_entropy"]
        vec = cdf[fields].to_numpy(dtype=np.float64).mean(axis=0)
        nrm = np.linalg.norm(vec)
        if nrm > 1e-12:
            vec = vec / nrm
        proto_shape[int(c)] = vec

    model_summary = {
        "reliability_gate": gate,
        "selected_k": int(best_k),
        "descriptor_vs_spectral_ari": concordance,
        "cluster_enrichment": cluster_enrichment,
        "secondary_behavioral_direction": {
            "rho_cluster_boundary_vs_cluster_mean_r2": rho,
            "p_one_sided": p_one,
            "directionally_consistent": directional_ok,
        },
    }
    return {
        "model": model,
        "membership_df": membership_df,
        "stability_df": ks,
        "enrichment_df": strata_df if not strata_df.empty else pd.DataFrame(columns=["cluster", "layer", "n_cluster", "n_noncluster", "hedges_g", "variance", "model"]),
        "model_summary": model_summary,
        "proto_shape": proto_shape,
        "eligible": bool(eligible),
        "concordance": float(concordance),
        "depth_support": bool(eligible and has_depth_support),
        "directional_ok": bool(eligible and directional_ok),
    }


def _aggregate_and_write(
    *,
    models: list[str],
    payload_by_model: dict[str, dict[str, Any]],
    out_dir: Path,
    n_boot: int,
    seed: int,
    execution_mode: str,
    partial_root: Path,
) -> None:
    membership_df = pd.concat([payload_by_model[m]["membership_df"] for m in models], ignore_index=True) if models else pd.DataFrame()
    stability_df = pd.concat([payload_by_model[m]["stability_df"] for m in models], ignore_index=True) if models else pd.DataFrame()

    enrich_frames = [payload_by_model[m]["enrichment_df"] for m in models if not payload_by_model[m]["enrichment_df"].empty]
    enrichment_df = pd.concat(enrich_frames, ignore_index=True) if enrich_frames else pd.DataFrame()

    b1_model_summary = {m: payload_by_model[m]["model_summary"] for m in models}
    descriptor_proto_shape_only = {m: payload_by_model[m]["proto_shape"] for m in models}
    eligible_models = [m for m in models if bool(payload_by_model[m]["eligible"])]
    concordance_vals = [float(payload_by_model[m]["concordance"]) for m in models if bool(payload_by_model[m]["eligible"])]
    depth_support_models = int(sum(1 for m in models if bool(payload_by_model[m]["depth_support"])))
    directional_consistency_models = int(sum(1 for m in models if bool(payload_by_model[m]["directional_ok"])))

    membership_df.to_parquet(out_dir / "cluster_membership.parquet", index=False)
    stability_df.to_parquet(out_dir / "cluster_stability_report.parquet", index=False)
    if not enrichment_df.empty:
        enrichment_df.to_parquet(out_dir / "cluster_function_enrichment.parquet", index=False)

    selected_k_values = [int(v["selected_k"]) for v in b1_model_summary.values() if "selected_k" in v]
    common_k = min(selected_k_values) if selected_k_values else 3
    proto_df, pair_support = _cross_model_prototype_similarity(descriptor_proto_shape_only, k=common_k, threshold=0.70)
    proto_df.to_parquet(out_dir / "prototype_similarity_matrix.parquet", index=False)

    c1_ok = all(
        bool(b1_model_summary[m]["reliability_gate"]["reliability_eligible"]) and
        bool(
            stability_df[(stability_df["model"] == m) & (stability_df["k"] == int(b1_model_summary[m]["selected_k"]))]["bootstrap_ari_mean"].iloc[0] >= 0.60
        )
        for m in b1_model_summary
        if b1_model_summary[m]["reliability_gate"]["reliability_eligible"]
    ) if eligible_models else False

    min_good = int(min(3, max(common_k - 1, 1)))
    c2_ok = bool(pair_support) and all(bool(v.get("supports_pair", False)) for v in pair_support.values())
    c3_ok = bool(depth_support_models >= 2)
    c4_ok = bool(directional_consistency_models >= 2)
    c5_ok = bool(np.nanmean(np.asarray(concordance_vals, dtype=np.float64)) >= 0.50) if concordance_vals else False
    c6_ok = bool(len(eligible_models) >= 2)
    b1_supported = bool(c1_ok and c2_ok and c3_ok and c4_ok and c5_ok and c6_ok)

    taxonomy_summary = {
        "timestamp": timestamp_now(),
        "experiment_id": "B1_kernel_taxonomy",
        "models": b1_model_summary,
        "eligibility": {
            "eligible_models": eligible_models,
            "n_eligible_models": int(len(eligible_models)),
        },
        "prototype_pair_support": pair_support,
        "criteria": {
            "criterion_1_ari_threshold": bool(c1_ok),
            "criterion_2_prototype_similarity": bool(c2_ok),
            "criterion_3_depth_control_enrichment": bool(c3_ok),
            "criterion_4_secondary_directional": bool(c4_ok),
            "criterion_5_descriptor_spectral_concordance": bool(c5_ok),
            "criterion_6_min_eligible_models": bool(c6_ok),
            "min_required_prototype_matches": min_good,
            "common_k_used_for_pairing": int(common_k),
        },
        "verdict": {
            "B1_supported": bool(b1_supported),
            "claim_status": "supported" if b1_supported else "mixed",
        },
        "limitations": [
            "Kernel reliability gate uses proxy statistics when shard-wise kernel estimates are unavailable.",
            "Secondary behavioral directional endpoint uses cluster mean-R2 as fallback when per-head task-grounded endpoints are unavailable.",
        ],
    }
    write_json(out_dir / "kernel_taxonomy_summary.json", taxonomy_summary)
    write_json(
        out_dir / "cluster_function_enrichment.json",
        {
            "timestamp": timestamp_now(),
            "experiment_id": "B1_kernel_taxonomy",
            "rows": enrichment_df.to_dict(orient="records") if not enrichment_df.empty else [],
        },
    )

    prereg = {
        "experiment_id": "B1_kernel_taxonomy",
        "question": "Do high-SI heads decompose into stable kernel-shape families with interpretable functional enrichment?",
        "primary_hypothesis": "Descriptor-space clustering yields stable, cross-model comparable kernel families with depth-controlled boundary enrichment.",
        "primary_endpoints": [
            "bootstrap ARI (descriptor_space)",
            "cross-model shape_only prototype cosine",
            "depth-controlled boundary enrichment",
            "descriptor-vs-spectral ARI",
        ],
        "secondary_endpoints": ["cluster-level directional behavioral proxy"],
        "model_list": models,
        "dataset_sources": [
            "results/experiment3/theory8_position_ablation/*/estimated_kernels.json",
            "results/experiment3/theory1_si_circuits/*/head_r2_summary.parquet",
            "results/experiment3/theory5b_boundary_detection/*/boundary_attention_scores.parquet",
        ],
        "inclusion_exclusion_rules": [
            "Model must pass reliability gate for confirmatory cross-model interpretation.",
            "Depth-controlled strata require n_cluster>=3 and n_noncluster>=10 per layer.",
        ],
        "sample_size_plan": {"k_candidates": [3, 4, 5, 6, 7, 8, 9, 10], "n_boot": int(n_boot)},
        "seed_plan": {"seed": int(seed)},
        "stopping_rule": "Stop after all models complete clustering and acceptance checks.",
        "multiplicity_family": ["per-model Holm over clusters for depth-controlled enrichment"],
        "acceptance_criteria": [
            "All B1.6 criteria in reinforce_exp2/TODO.md must pass for B1_supported.",
        ],
        "fallback_interpretation_if_null": "Report B1 as exploratory heterogeneity map only.",
    }

    manifest = command_manifest(
        experiment_id="B1_kernel_taxonomy",
        command="run_b1_kernel_taxonomy.py",
        model="+".join(models),
        extras={
            "models": models,
            "n_boot": int(n_boot),
            "seed": int(seed),
            "output_root": str(out_dir),
            "execution_mode": str(execution_mode),
            "partial_root": str(partial_root),
        },
    )
    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": "B1_kernel_taxonomy",
        "analysis_tier": "confirmatory",
        "canonical_eligible": True,
        "override_used": False,
        "verdict": taxonomy_summary["verdict"],
    }
    claim_impact = {
        "timestamp": timestamp_now(),
        "experiment_id": "B1_kernel_taxonomy",
        "claim_status": "supported" if b1_supported else "mixed",
        "supports_main_text": bool(b1_supported),
        "strict_only": True,
        "notes": [
            "If B1_supported is false, taxonomy can remain exploratory but not main mechanistic unit.",
        ],
    }
    data_dictionary = {
        "experiment_id": "B1_kernel_taxonomy",
        "tables": [
            {
                "path": str(out_dir / "cluster_membership.parquet"),
                "description": "Head-level cluster assignments across feature spaces.",
                "columns": [
                    {"name": "model", "dtype": "str", "description": "Model name."},
                    {"name": "layer", "dtype": "int", "description": "Layer index."},
                    {"name": "head", "dtype": "int", "description": "Head index."},
                    {"name": "cluster_descriptor_kmeans", "dtype": "int", "description": "Primary descriptor-space cluster id."},
                    {"name": "mean_r2", "dtype": "float", "description": "Mean SI-R2."},
                    {"name": "boundary_attn_score", "dtype": "float", "description": "Boundary effect score."},
                ],
            },
            {
                "path": str(out_dir / "prototype_similarity_matrix.parquet"),
                "description": "Cross-model prototype cosine matches.",
                "columns": [
                    {"name": "model_a", "dtype": "str", "description": "Left model."},
                    {"name": "model_b", "dtype": "str", "description": "Right model."},
                    {"name": "shape_only_cosine", "dtype": "float", "description": "Prototype cosine similarity."},
                ],
            },
        ],
    }
    emit_core_artifacts(
        experiment_id="B1_kernel_taxonomy",
        out_dir=out_dir,
        preregistration=prereg,
        manifest=manifest,
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )
    write_json(out_dir / "B1_claim_impact.json", claim_impact)


def main() -> None:
    p = argparse.ArgumentParser(description="B1 Kernel Shape Taxonomy", allow_abbrev=False)
    p.add_argument("--models", default="llama-3.1-8b,olmo-2-7b,mistral-7b-v0.1")
    p.add_argument("--output-root", default=str(RESULTS_ROOT / "B1_kernel_taxonomy"))
    p.add_argument("--calibration-root", default=str(RESULTS_ROOT / "calibration_splits"))
    p.add_argument("--device-map", default="")
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--n-shards", type=int, default=4)
    p.add_argument("--max-seqs-per-shard", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260417)
    p.add_argument("--execution-mode", choices=["full", "per_model", "aggregate"], default="full")
    p.add_argument("--partial-root", default="")
    p.add_argument("--resume-partials", default="true")
    p.add_argument("--single-model", default="")
    args = p.parse_args()

    models = parse_models_arg(args.models)
    out_dir = ensure_dir(Path(args.output_root))
    partial_root = ensure_dir(Path(args.partial_root)) if str(args.partial_root).strip() else ensure_dir(out_dir / "partials")
    resume_partials = _truthy(args.resume_partials)
    device_map = _parse_device_map(args.device_map)
    calibration_tokens = _load_calibration_tokens(Path(args.calibration_root))

    execution_mode = str(args.execution_mode).strip().lower()
    if execution_mode in {"full", "per_model"}:
        model_scope = models
        if execution_mode == "per_model":
            target = str(args.single_model).strip()
            if not target:
                raise ValueError("--single-model is required for --execution-mode per_model")
            model_scope = [target]
        _validate_device_map_for_gpu_inference(
            models=model_scope,
            device_map=device_map,
            calibration_tokens=calibration_tokens,
        )

    payload_by_model: dict[str, dict[str, Any]] = {}

    if execution_mode == "per_model":
        model = str(args.single_model).strip()
        if not model:
            raise ValueError("--single-model is required for --execution-mode per_model")
        model_index = _stable_model_index(models, model)
        cfg_hash = _partial_config_hash(
            model=model,
            models=models,
            seed=int(args.seed),
            n_boot=int(args.n_boot),
            n_shards=int(args.n_shards),
            max_seqs_per_shard=int(args.max_seqs_per_shard),
            calibration_root=Path(args.calibration_root),
        )
        ready, errs = _partial_ready(partial_root, model, cfg_hash)
        if resume_partials and ready:
            print(f"[B1 per_model] partial exists and valid for {model}; skipping")
            return
        if resume_partials and (not ready) and errs:
            print(f"[B1 per_model] partial invalid for {model}; recomputing")
        payload = _compute_model_payload(
            model=model,
            model_index=model_index,
            device_map=device_map,
            calibration_tokens=calibration_tokens,
            n_boot=int(args.n_boot),
            n_shards=int(args.n_shards),
            max_seqs_per_shard=int(args.max_seqs_per_shard),
            seed=int(args.seed),
        )
        _write_partial(payload, partial_root, cfg_hash)
        print(f"[B1 per_model] wrote partial for {model} at {partial_root}")
        return

    if execution_mode == "aggregate":
        for model in models:
            cfg_hash = _partial_config_hash(
                model=model,
                models=models,
                seed=int(args.seed),
                n_boot=int(args.n_boot),
                n_shards=int(args.n_shards),
                max_seqs_per_shard=int(args.max_seqs_per_shard),
                calibration_root=Path(args.calibration_root),
            )
            ready, errs = _partial_ready(partial_root, model, cfg_hash)
            if not ready:
                raise RuntimeError(f"Missing/invalid partial for model={model}: {errs}")
            payload_by_model[model] = _load_partial(partial_root, model)
        _aggregate_and_write(
            models=models,
            payload_by_model=payload_by_model,
            out_dir=out_dir,
            n_boot=int(args.n_boot),
            seed=int(args.seed),
            execution_mode=execution_mode,
            partial_root=partial_root,
        )
        print(f"[B1 aggregate] wrote {out_dir}")
        return

    for model in models:
        model_index = _stable_model_index(models, model)
        cfg_hash = _partial_config_hash(
            model=model,
            models=models,
            seed=int(args.seed),
            n_boot=int(args.n_boot),
            n_shards=int(args.n_shards),
            max_seqs_per_shard=int(args.max_seqs_per_shard),
            calibration_root=Path(args.calibration_root),
        )
        ready, _ = _partial_ready(partial_root, model, cfg_hash)
        if resume_partials and ready:
            payload_by_model[model] = _load_partial(partial_root, model)
            continue
        payload = _compute_model_payload(
            model=model,
            model_index=model_index,
            device_map=device_map,
            calibration_tokens=calibration_tokens,
            n_boot=int(args.n_boot),
            n_shards=int(args.n_shards),
            max_seqs_per_shard=int(args.max_seqs_per_shard),
            seed=int(args.seed),
        )
        _write_partial(payload, partial_root, cfg_hash)
        payload_by_model[model] = payload

    _aggregate_and_write(
        models=models,
        payload_by_model=payload_by_model,
        out_dir=out_dir,
        n_boot=int(args.n_boot),
        seed=int(args.seed),
        execution_mode=execution_mode,
        partial_root=partial_root,
    )
    print(f"[B1] wrote {out_dir}")


if __name__ == "__main__":
    main()
