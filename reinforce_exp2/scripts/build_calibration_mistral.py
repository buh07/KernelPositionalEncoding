#!/usr/bin/env python3
"""
Build a Mistral-specific calibration split from the existing per-model wiki cache.

Unlike build_calibration_v1.py (which requires 3 HF domain caches that don't exist
for Mistral), this script reads the pre-tokenized wiki40b cache directly. All token
IDs in this cache were produced by Mistral's SentencePiece tokenizer and are
guaranteed to be within Mistral's 32k vocabulary — so _sanitize_calibration_tokens_for_vocab
will retain all sequences.

The output schema is identical to build_calibration_v1.py so run_b1_kernel_taxonomy.py
can load it unchanged via --calibration-root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp2.common import RESULTS_ROOT, ensure_dir, file_sha256, timestamp_now, write_json  # noqa: E402


_WIKI_CACHE = ROOT / "data" / "experiment1" / "wiki40b_en_pre2019" / "mistral-7b-v0.1" / "len_1024.jsonl"


def _token_hash(tokens: list[int]) -> str:
    s = ",".join(str(int(x)) for x in tokens)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _load_wiki_cache(path: Path, seq_len: int) -> list[list[int]]:
    rows: list[list[int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            toks = row.get("tokens", row.get("input_ids", []))
            if not isinstance(toks, list) or len(toks) < 128:
                continue
            rows.append([int(x) for x in toks[:seq_len]])
    return rows


def _verify_vocab_range(sequences: list[list[int]], vocab_size: int) -> None:
    """Raise if any token is out of range — catches bugs early."""
    for i, seq in enumerate(sequences):
        bad = [t for t in seq if t < 0 or t >= vocab_size]
        if bad:
            raise ValueError(
                f"Sequence {i} has {len(bad)} out-of-range tokens (vocab_size={vocab_size}): "
                f"first bad = {bad[0]}"
            )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build Mistral-specific calibration split from wiki40b cache",
        allow_abbrev=False,
    )
    p.add_argument(
        "--output-root",
        default=str(RESULTS_ROOT / "calibration_splits" / "mistral-7b-v0.1"),
    )
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=20260417)
    p.add_argument(
        "--wiki-cache",
        default=str(_WIKI_CACHE),
        help="Path to mistral-7b-v0.1/len_1024.jsonl",
    )
    p.add_argument(
        "--mistral-vocab-size",
        type=int,
        default=32000,
        help="Mistral tokenizer vocabulary size (default 32000)",
    )
    args = p.parse_args()

    wiki_path = Path(args.wiki_cache)
    if not wiki_path.exists():
        raise FileNotFoundError(f"Wiki cache not found: {wiki_path}")

    print(f"[calibration_mistral] loading wiki cache from {wiki_path}", flush=True)
    seqs = _load_wiki_cache(wiki_path, int(args.seq_len))
    print(f"[calibration_mistral] loaded {len(seqs)} sequences from cache", flush=True)

    if not seqs:
        raise RuntimeError("No sequences loaded from wiki cache")

    print(f"[calibration_mistral] verifying token IDs within vocab_size={args.mistral_vocab_size}", flush=True)
    _verify_vocab_range(seqs, int(args.mistral_vocab_size))
    print("[calibration_mistral] all tokens in range — verification passed", flush=True)

    rng = np.random.default_rng(int(args.seed))
    n = len(seqs)
    # Use all available sequences; shuffle for reproducibility
    order = rng.permutation(n).tolist()
    seqs = [seqs[i] for i in order]

    ids_rows: list[dict[str, Any]] = []
    tok_rows: list[dict[str, Any]] = []

    for toks in seqs:
        seq_id = _token_hash(toks)
        ids_rows.append({
            "sequence_id": seq_id,
            "domain": "wiki",
            "loader_used": "cache_jsonl",
            "dataset_id": str(wiki_path),
            "config_name": None,
            "split": "cached",
            "token_count": int(len(toks)),
        })
        tok_rows.append({"sequence_id": seq_id, "tokens": toks})

    ids_df = pd.DataFrame(ids_rows).drop_duplicates(subset=["sequence_id"]).reset_index(drop=True)
    tok_df = pd.DataFrame(tok_rows).drop_duplicates(subset=["sequence_id"]).reset_index(drop=True)

    out_dir = ensure_dir(Path(args.output_root))
    ids_path = out_dir / "calibration_v1_ids.parquet"
    tok_path = out_dir / "calibration_v1_tokens.parquet"

    ids_df.to_parquet(ids_path, index=False)
    tok_df.to_parquet(tok_path, index=False)

    ids_sha = file_sha256(ids_path)
    manifest: dict[str, Any] = {
        "timestamp": timestamp_now(),
        "split_id": "calibration_v1_mistral",
        "model_tokenizer": "mistral-7b-v0.1",
        "seed": int(args.seed),
        "target_n": int(n),
        "seq_len": int(args.seq_len),
        "vocab_size": int(args.mistral_vocab_size),
        "token_filter": {"min": 128, "max": int(args.seq_len)},
        "source": "wiki40b_en_pre2019_mistral_cache",
        "per_domain_counts": {"wiki": int(ids_df.shape[0])},
        "loader_used_counts": {"cache_jsonl": int(ids_df.shape[0])},
        "vocab_verified": True,
        "sha256_ids_parquet": ids_sha,
        "ids_path": str(ids_path),
        "tokens_path": str(tok_path),
    }
    write_json(out_dir / "calibration_v1_manifest.json", manifest)

    print(f"[calibration_mistral] wrote {ids_path} ({ids_df.shape[0]} sequences)", flush=True)
    print(f"[calibration_mistral] sha256={ids_sha}", flush=True)
    print(f"[calibration_mistral] token range: min={min(min(s) for s in seqs)}, max={max(max(s) for s in seqs)}", flush=True)


if __name__ == "__main__":
    main()
