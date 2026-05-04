#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment2.tasks import generate_task_examples  # noqa: E402
from experiment3.theory1_si_circuits import MODELS, HeadID  # noqa: E402
from reinforce_exp2.common import ASSETS_ROOT, load_head_groups, load_kernels, parse_head_key  # noqa: E402

_FASTTEXT_MODEL_CACHE: dict[str, Any] = {}


def sequence_hash(tokens: list[int]) -> str:
    s = ",".join(str(int(x)) for x in tokens)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def hash_string_list(values: list[str]) -> str:
    h = hashlib.sha256()
    for v in sorted(str(x) for x in values):
        h.update(v.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _stable_seed_from_text(text: str) -> int:
    return int(hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def preprocess_token_text(token_str: str) -> str:
    t = str(token_str or "")
    t = t.replace("\u0120", "")
    t = t.replace("\u2581", "")
    t = t.replace("##", "")
    t = t.lower().strip()
    t = re.sub(r"\s+", "", t)
    if not t:
        return ""
    # Prefer lexical core, but retain subword/symbol tokens as fallback text so
    # fastText subword vectors remain available and coverage stays high.
    core = re.sub(r"[^a-z0-9]+", "", t)
    if core:
        return core
    return t[:64]


def _load_fasttext_model() -> Any:
    path = Path(ASSETS_ROOT) / "fasttext" / "cc.en.300.bin"
    key = str(path)
    if key in _FASTTEXT_MODEL_CACHE:
        return _FASTTEXT_MODEL_CACHE[key]
    if not path.exists():
        raise FileNotFoundError(f"Missing fastText model: {path}")
    try:
        import fasttext  # type: ignore
    except Exception as exc:
        raise RuntimeError("fasttext package not installed; run preflight") from exc
    model = fasttext.load_model(str(path))
    _FASTTEXT_MODEL_CACHE[key] = model
    return model


def _kernel_entropy(vec: np.ndarray) -> float:
    v = np.abs(np.asarray(vec, dtype=np.float64))
    p = v / max(float(np.sum(v)), 1e-12)
    h = float(-np.sum(p * np.log(p + 1e-12)))
    return float(h / max(np.log(max(2, len(v))), 1e-12))


def _load_head_metadata(model_name: str) -> pd.DataFrame:
    kernels = load_kernels(model_name)
    rows: list[dict[str, Any]] = []
    for hk, vec in kernels.items():
        try:
            layer, head = parse_head_key(hk)
        except Exception:
            continue
        rows.append(
            {
                "layer": int(layer),
                "head": int(head),
                "head_key": str(hk),
                "kernel": np.asarray(vec, dtype=np.float64),
                "entropy_proxy": _kernel_entropy(np.asarray(vec, dtype=np.float64)),
            }
        )
    if not rows:
        raise RuntimeError(f"No kernels available for model={model_name}")
    df = pd.DataFrame(rows)

    r2_paths = [
        ROOT / "results" / "experiment3" / "theory1_si_circuits" / model_name / "head_r2_summary.parquet",
        ROOT / "results" / "reinforce_exp" / "exp_r3_core_replication" / model_name / "theory1_si_circuits" / model_name / "head_r2_summary.parquet",
    ]
    r2_df = None
    for rp in r2_paths:
        if rp.exists():
            r2_df = pd.read_parquet(rp)
            break
    if r2_df is None:
        raise FileNotFoundError(f"Missing head_r2_summary for model={model_name}")
    if "mean_r2" not in r2_df.columns and "r2" in r2_df.columns:
        r2_df = r2_df.rename(columns={"r2": "mean_r2"})
    if "mean_r2" not in r2_df.columns:
        raise RuntimeError(f"head_r2_summary missing mean_r2 for model={model_name}")

    keep = r2_df[["layer", "head", "mean_r2"]].copy()
    keep["layer"] = keep["layer"].astype(int)
    keep["head"] = keep["head"].astype(int)
    keep["head_key"] = keep.apply(lambda r: f"L{int(r['layer'])}H{int(r['head'])}", axis=1)

    out = df.merge(keep[["layer", "head", "head_key", "mean_r2"]], on=["layer", "head", "head_key"], how="left")
    out["mean_r2"] = out["mean_r2"].astype(float)

    groups = load_head_groups(model_name)
    high = {f"L{int(x['layer'])}H{int(x['head'])}" for x in groups.get("high_si", [])}
    low = {f"L{int(x['layer'])}H{int(x['head'])}" for x in groups.get("low_si", [])}
    out["is_high_si"] = out["head_key"].isin(high)
    out["is_low_si"] = out["head_key"].isin(low)
    return out


def _bin_offsets(delta: np.ndarray, n_bins: int) -> np.ndarray:
    d = np.asarray(delta, dtype=np.float64)
    if d.size == 0:
        return np.zeros(0, dtype=np.int32)
    qs = np.linspace(0.0, 1.0, max(2, int(n_bins) + 1))
    edges = np.unique(np.quantile(d, qs))
    if edges.size <= 2:
        return np.zeros(d.size, dtype=np.int32)
    bins = np.digitize(d, edges[1:-1], right=False)
    return bins.astype(np.int32)


def _demean_by_bins(v: np.ndarray, bins: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64)
    b = np.asarray(bins, dtype=np.int32)
    if x.size == 0:
        return x
    n_bins = int(np.max(b)) + 1 if b.size else 0
    if n_bins <= 0:
        return x - np.nanmean(x)
    sums = np.bincount(b, weights=x, minlength=n_bins)
    cnts = np.bincount(b, minlength=n_bins).astype(np.float64)
    means = sums / np.maximum(cnts, 1.0)
    return x - means[b]


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size < 8:
        return float("nan")
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 1e-12 or sy <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _prepare_content_payloads(
    *,
    tokenizer,
    ft_model,
    sequences: list[list[int]],
    max_pairs_per_sequence: int,
    n_offset_bins: int,
) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    payloads: list[dict[str, np.ndarray]] = []

    total_tokens = 0
    covered_tokens = 0
    usable_sequences = 0
    skipped_sequences = 0

    for seq in sequences:
        ids = [int(x) for x in seq]
        toks = tokenizer.convert_ids_to_tokens(ids)
        words = [preprocess_token_text(t) for t in toks]
        total_tokens += len(words)

        emb = np.zeros((len(words), 300), dtype=np.float64)
        ok = np.zeros(len(words), dtype=bool)
        for i, w in enumerate(words):
            if not w:
                continue
            try:
                v = np.asarray(ft_model.get_word_vector(w), dtype=np.float64)
            except Exception:
                continue
            if v.size != 300:
                continue
            nv = float(np.linalg.norm(v))
            if not np.isfinite(nv) or nv <= 1e-12:
                continue
            emb[i] = v
            ok[i] = True
            covered_tokens += 1

        if int(np.sum(ok)) < 8:
            skipped_sequences += 1
            continue

        norms = np.linalg.norm(emb, axis=1)
        denom = np.maximum(norms[:, None] * norms[None, :], 1e-8)
        sim = (emb @ emb.T) / denom
        ii, jj = np.tril_indices(len(ids), k=-1)
        mask = ok[ii] & ok[jj] & np.isfinite(sim[ii, jj])
        if not np.any(mask):
            skipped_sequences += 1
            continue

        ii = ii[mask]
        jj = jj[mask]
        y = sim[ii, jj].astype(np.float64)
        delta = (ii - jj).astype(np.int32)

        if y.size > int(max_pairs_per_sequence):
            rng = np.random.default_rng(_stable_seed_from_text(sequence_hash(ids)))
            sel = rng.choice(np.arange(y.size, dtype=np.int64), size=int(max_pairs_per_sequence), replace=False)
            y = y[sel]
            delta = delta[sel]

        bins = _bin_offsets(delta.astype(np.float64), n_bins=max(2, int(n_offset_bins)))

        y_rank = scipy_stats.rankdata(y).astype(np.float64)
        y_res = _demean_by_bins(y_rank, bins)
        if np.sum(np.isfinite(y_res)) < 8 or float(np.std(y_res)) <= 1e-12:
            skipped_sequences += 1
            continue

        payloads.append(
            {
                "delta": delta.astype(np.int32),
                "bins": bins.astype(np.int32),
                "y_res": y_res.astype(np.float64),
            }
        )
        usable_sequences += 1

    missing_ratio = 1.0 - (float(covered_tokens) / max(float(total_tokens), 1.0))
    quality = {
        "total_tokens": int(total_tokens),
        "covered_tokens": int(covered_tokens),
        "missing_token_ratio": float(missing_ratio),
        "usable_sequences": int(usable_sequences),
        "skipped_sequences": int(skipped_sequences),
        "n_payloads": int(len(payloads)),
    }
    return payloads, quality


def _compute_alignment_for_heads(
    head_df: pd.DataFrame,
    payloads: list[dict[str, np.ndarray]],
) -> dict[str, float]:
    out: dict[str, float] = {}
    if not payloads:
        return out

    for row in head_df.itertuples():
        kernel = np.abs(np.asarray(row.kernel, dtype=np.float64))
        if kernel.size < 8:
            out[str(row.head_key)] = float("nan")
            continue
        vals = []
        for pl in payloads:
            d = pl["delta"]
            bins = pl["bins"]
            y_res = pl["y_res"]
            idx = np.clip(d.astype(np.int64), 0, kernel.size - 1)
            x = kernel[idx]
            x_rank = scipy_stats.rankdata(x).astype(np.float64)
            x_res = _demean_by_bins(x_rank, bins)
            rho = _safe_corr(x_res, y_res)
            if np.isfinite(rho):
                vals.append(float(rho))
        out[str(row.head_key)] = float(np.nanmedian(np.asarray(vals, dtype=np.float64))) if vals else float("nan")
    return out


def _generate_calibration_sequences(
    *,
    model_name: str,
    seq_len: int,
    seed: int,
    n_sequences: int,
    pools,
) -> list[list[int]]:
    seqs: list[list[int]] = []
    n_per = max(8, int(np.ceil(max(1, n_sequences) / 3.0)))
    specs = [
        ("local_key_match", None),
        ("long_range_retrieval", max(32, int(seq_len // 8))),
        ("copy_offset", max(64, int(seq_len // 4))),
    ]
    for idx, (task_name, span) in enumerate(specs):
        try:
            exs = generate_task_examples(
                task_name=task_name,
                model_name=model_name,
                seq_len=max(128, int(seq_len)),
                seed=int(seed) + idx * 97,
                count=n_per,
                pools=pools,
                span_override=span,
                span_choices=(int(span),) if span is not None else None,
            )
        except Exception:
            exs = generate_task_examples(
                task_name="local_key_match",
                model_name=model_name,
                seq_len=max(128, int(seq_len)),
                seed=int(seed) + idx * 97,
                count=n_per,
                pools=pools,
            )
        seqs.extend([[int(x) for x in ex.tokens] for ex in exs])

    if len(seqs) < n_sequences:
        exs = generate_task_examples(
            task_name="local_key_match",
            model_name=model_name,
            seq_len=max(128, int(seq_len)),
            seed=int(seed) + 707,
            count=max(8, n_sequences - len(seqs)),
            pools=pools,
        )
        seqs.extend([[int(x) for x in ex.tokens] for ex in exs])

    return seqs[: int(n_sequences)]


def build_static_carrier_sets(
    *,
    model_name: str,
    tokenizer,
    pools,
    seq_len: int,
    seed: int,
    eval_sequence_hashes: set[str] | None = None,
    n_calibration: int = 64,
    max_pairs_per_sequence: int = 4000,
    n_offset_bins: int = 8,
    data_quality_max_missing: float = 0.05,
) -> dict[str, Any]:
    if model_name not in MODELS:
        raise ValueError(f"Unsupported model for carrier-set construction: {model_name}")

    head_df = _load_head_metadata(model_name)
    ft_model = _load_fasttext_model()

    calibration_sequences = _generate_calibration_sequences(
        model_name=model_name,
        seq_len=max(128, int(seq_len)),
        seed=int(seed),
        n_sequences=max(24, int(n_calibration)),
        pools=pools,
    )
    cal_hashes = [sequence_hash(seq) for seq in calibration_sequences]
    cal_hash_set = set(cal_hashes)
    eval_hash_set = set(eval_sequence_hashes or set())
    overlap = sorted(cal_hash_set & eval_hash_set)

    payloads, quality = _prepare_content_payloads(
        tokenizer=tokenizer,
        ft_model=ft_model,
        sequences=calibration_sequences,
        max_pairs_per_sequence=max(500, int(max_pairs_per_sequence)),
        n_offset_bins=max(2, int(n_offset_bins)),
    )

    content_similarity_data_quality_failure = bool(float(quality["missing_token_ratio"]) > float(data_quality_max_missing))
    if content_similarity_data_quality_failure:
        raise RuntimeError(
            f"content_similarity_data_quality_failure=true missing_token_ratio={quality['missing_token_ratio']:.4f}"
        )
    if int(quality["n_payloads"]) < 8:
        raise RuntimeError("Insufficient usable calibration sequences for static content alignment")

    model_median_r2 = float(np.nanmedian(head_df["mean_r2"].to_numpy(dtype=np.float64)))
    layer_median_entropy = (
        head_df.groupby("layer", as_index=False)["entropy_proxy"].median().rename(columns={"entropy_proxy": "layer_entropy_median"})
    )
    work = head_df.merge(layer_median_entropy, on="layer", how="left")

    low_pool = work[work["mean_r2"] < model_median_r2].copy()
    align_map = _compute_alignment_for_heads(low_pool, payloads)
    work["content_similarity_alignment"] = work["head_key"].map(align_map).astype(float)

    low_align = work[work["mean_r2"] < model_median_r2]["content_similarity_alignment"].to_numpy(dtype=np.float64)
    low_align = low_align[np.isfinite(low_align)]
    threshold_t = float(np.nanquantile(low_align, 0.75)) if low_align.size else float("nan")

    cond = (
        (work["mean_r2"] < model_median_r2)
        & (work["entropy_proxy"] > work["layer_entropy_median"])
        & np.isfinite(work["content_similarity_alignment"])
        & (work["content_similarity_alignment"] > threshold_t)
    )
    selected = work[cond].copy().sort_values(["content_similarity_alignment", "layer", "head"], ascending=[False, True, True])

    high_si = work[work["is_high_si"]].copy().sort_values(["layer", "head"])
    low_si = work[work["is_low_si"]].copy().sort_values(["layer", "head"])

    content_cond = [HeadID(int(r.layer), int(r.head)) for r in selected.itertuples()]
    if not content_cond and not selected.empty:
        content_cond = [HeadID(int(selected.iloc[0]["layer"]), int(selected.iloc[0]["head"]))]

    diagnostics = {
        "model": model_name,
        "alignment_method": "fasttext_static_embedding_partial_spearman_offset_bins",
        "fasttext_model_path": str(Path(ASSETS_ROOT) / "fasttext" / "cc.en.300.bin"),
        "calibration_sequence_count": int(len(calibration_sequences)),
        "calibration_sequence_hash_sha256": hash_string_list(cal_hashes),
        "calibration_eval_overlap_count": int(len(overlap)),
        "calibration_eval_overlap_ratio": float(len(overlap) / max(len(cal_hash_set), 1)),
        "calibration_eval_disjoint": bool(len(overlap) == 0),
        "data_quality": quality,
        "content_similarity_data_quality_failure": False,
        "model_median_r2": model_median_r2,
        "threshold_T_q75_low_r2": threshold_t,
        "low_r2_pool_size": int(np.sum(work["mean_r2"] < model_median_r2)),
        "content_conditional_size": int(len(content_cond)),
        "set_sizes": {
            "SI": int(high_si.shape[0]),
            "LowSI": int(low_si.shape[0]),
            "ContentCond": int(len(content_cond)),
        },
    }

    return {
        "SI": [HeadID(int(r.layer), int(r.head)) for r in high_si.itertuples()],
        "LowSI": [HeadID(int(r.layer), int(r.head)) for r in low_si.itertuples()],
        "ContentCond": content_cond,
        "diagnostics": diagnostics,
        "alignment_table": work[["layer", "head", "head_key", "mean_r2", "entropy_proxy", "layer_entropy_median", "content_similarity_alignment"]].copy(),
        "calibration_sequence_hashes": cal_hash_set,
    }


def dump_alignment_table(path: Path, alignment_table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    alignment_table.to_parquet(path, index=False)


def dump_carrier_diagnostics(path: Path, diagnostics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
