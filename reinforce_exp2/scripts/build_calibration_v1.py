#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment3.theory1_si_circuits import MODELS  # noqa: E402
from experiment5.common import build_sequences_from_text_dataset  # noqa: E402
from reinforce_exp2.common import RESULTS_ROOT, ensure_dir, file_sha256, timestamp_now, write_json  # noqa: E402
from shared.models.loading import load_tokenizer  # noqa: E402


def _safe_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(str(x) for x in value if isinstance(x, str))
    if isinstance(value, dict):
        return "\n".join(str(x) for x in value.values() if isinstance(x, str))
    return ""


def _load_cached_sequences(path: Path, *, seq_len: int, max_sequences: int) -> list[list[int]]:
    rows: list[list[int]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            toks = row.get("tokens", row.get("input_ids", []))
            if not isinstance(toks, list):
                continue
            toks = [int(x) for x in toks]
            if len(toks) < 128:
                continue
            toks = toks[:seq_len]
            rows.append(toks)
            if len(rows) >= max_sequences:
                break
    return rows


def _token_hash(tokens: list[int]) -> str:
    import hashlib

    s = ",".join(str(int(x)) for x in tokens)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _pull_domain_sequences(
    *,
    domain: str,
    tokenizer,
    model_name: str,
    target_n: int,
    seq_len: int,
    seed: int,
) -> tuple[list[list[int]], dict[str, Any]]:
    cache_map = {
        "wiki": ROOT / "data" / "experiment1" / "wiki40b_en_pre2019" / model_name / "len_1024.jsonl",
        "code": ROOT / "data" / "experiment1" / "codesearchnet_python_snapshot" / model_name / "len_1024.jsonl",
        "dialogue": ROOT / "data" / "experiment1" / "dialogue_snapshot" / model_name / "len_1024.jsonl",
    }
    cached = _load_cached_sequences(cache_map[domain], seq_len=seq_len, max_sequences=max(target_n, 5000))
    if len(cached) >= target_n:
        return cached[:target_n], {
            "domain": domain,
            "loader_used": "cache_jsonl",
            "dataset_id": str(cache_map[domain]),
            "config_name": None,
            "split": "cached",
            "n_loaded": int(len(cached)),
            "requires_replacement": False,
        }

    fallback_specs: dict[str, list[dict[str, Any]]] = {
        "wiki": [
            {
                "dataset_id": "wikitext",
                "config_name": "wikitext-103-raw-v1",
                "split": "train",
                "text_fields": ("text",),
                "max_rows": 200000,
            },
            {
                "dataset_id": "wiki40b",
                "config_name": "en",
                "split": "train",
                "text_fields": ("text",),
                "max_rows": 200000,
            },
        ],
        "code": [
            {
                "dataset_id": "code_search_net",
                "config_name": "python",
                "split": "train",
                "text_fields": ("whole_func_string", "func_documentation_string", "code", "text"),
                "max_rows": 400000,
            },
            {
                "dataset_id": "mbpp",
                "config_name": None,
                "split": "train",
                "text_fields": ("text", "code"),
                "max_rows": 120000,
            },
            {
                "dataset_id": "codeparrot/github-code",
                "config_name": None,
                "split": "train",
                "text_fields": ("code", "text", "content"),
                "max_rows": 120000,
            },
        ],
        "dialogue": [
            {
                "dataset_id": "daily_dialog",
                "config_name": None,
                "split": "train",
                "text_fields": ("dialog", "text"),
                "max_rows": 200000,
            },
            {
                "dataset_id": "OpenAssistant/oasst1",
                "config_name": None,
                "split": "train",
                "text_fields": ("text",),
                "max_rows": 200000,
            },
        ],
    }

    best = (cached, {
        "domain": domain,
        "loader_used": "cache_jsonl_partial" if cached else "none",
        "dataset_id": str(cache_map[domain]),
        "config_name": None,
        "split": "cached",
        "n_loaded": int(len(cached)),
        "requires_replacement": False,
    })

    for spec in fallback_specs[domain]:
        try:
            seqs = build_sequences_from_text_dataset(
                tokenizer=tokenizer,
                dataset_id=str(spec["dataset_id"]),
                split=str(spec["split"]),
                text_fields=tuple(spec["text_fields"]),
                seq_len=seq_len,
                max_sequences=target_n,
                max_rows=int(spec["max_rows"]),
                seed=seed,
                config_name=spec.get("config_name"),
                streaming=False,
            )
            seqs = [s for s in seqs if 128 <= len(s) <= 512]
            if len(seqs) > len(best[0]):
                best = (
                    seqs,
                    {
                        "domain": domain,
                        "loader_used": "hf_loader_partial",
                        "dataset_id": str(spec["dataset_id"]),
                        "config_name": spec.get("config_name"),
                        "split": str(spec["split"]),
                        "n_loaded": int(len(seqs)),
                        "requires_replacement": False,
                    },
                )
            if len(seqs) >= target_n:
                return seqs[:target_n], {
                    "domain": domain,
                    "loader_used": "hf_loader",
                    "dataset_id": str(spec["dataset_id"]),
                    "config_name": spec.get("config_name"),
                    "split": str(spec["split"]),
                    "n_loaded": int(len(seqs)),
                    "requires_replacement": False,
                }
        except Exception:
            continue

    if len(best[0]) >= max(1, int(target_n * 0.10)):
        prov = dict(best[1])
        prov["requires_replacement"] = True
        return best[0], prov

    raise RuntimeError(f"Unable to source enough calibration data for domain={domain}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build frozen reinforce_exp2 calibration_v1 split", allow_abbrev=False)
    p.add_argument("--model", default="llama-3.1-8b")
    p.add_argument("--output-root", default=str(RESULTS_ROOT / "calibration_splits"))
    p.add_argument("--seed", type=int, default=20260417)
    p.add_argument("--target-n", type=int, default=4096)
    p.add_argument("--seq-len", type=int, default=512)
    args = p.parse_args()

    model_name = str(args.model)
    if model_name not in MODELS:
        raise ValueError(f"Unsupported model for tokenization: {model_name}")

    out_dir = ensure_dir(Path(args.output_root))
    tokenizer = load_tokenizer(MODELS[model_name])
    rng = np.random.default_rng(int(args.seed))

    target_n = int(args.target_n)
    base = target_n // 3
    rem = target_n % 3
    per_domain_targets = {
        "wiki": base + (1 if rem >= 1 else 0),
        "code": base + (1 if rem >= 2 else 0),
        "dialogue": base,
    }

    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []

    for domain in ("wiki", "code", "dialogue"):
        n_req = int(per_domain_targets[domain])
        seqs, prov = _pull_domain_sequences(
            domain=domain,
            tokenizer=tokenizer,
            model_name=model_name,
            target_n=max(n_req + 256, int(round(n_req * 1.25))),
            seq_len=int(args.seq_len),
            seed=int(args.seed) + (1 if domain == "wiki" else 2 if domain == "code" else 3),
        )
        if len(seqs) < n_req and not bool(prov.get("requires_replacement", False)):
            raise RuntimeError(f"Insufficient sequences for domain={domain}: {len(seqs)} < {n_req}")
        replace = bool(prov.get("requires_replacement", False)) and len(seqs) > 0
        picks = rng.choice(np.arange(len(seqs), dtype=np.int64), size=n_req, replace=replace)
        seen_local: dict[str, int] = {}
        for idx in picks.tolist():
            toks = [int(x) for x in seqs[int(idx)]]
            base_id = _token_hash(toks)
            seen_local[base_id] = int(seen_local.get(base_id, 0)) + 1
            if seen_local[base_id] > 1:
                seq_id = f"{base_id}:dup{seen_local[base_id]-1}"
            else:
                seq_id = base_id
            rows.append(
                {
                    "sequence_id": seq_id,
                    "domain": domain,
                    "loader_used": prov["loader_used"],
                    "dataset_id": prov["dataset_id"],
                    "config_name": prov["config_name"],
                    "split": prov["split"],
                    "token_count": int(len(toks)),
                }
            )
            token_rows.append({"sequence_id": seq_id, "tokens": toks})
        provenance.append(prov)

    ids_df = pd.DataFrame(rows).drop_duplicates(subset=["sequence_id"]).reset_index(drop=True)
    tok_df = pd.DataFrame(token_rows).drop_duplicates(subset=["sequence_id"]).reset_index(drop=True)

    if int(ids_df.shape[0]) < target_n:
        raise RuntimeError(f"Calibration split too small after dedup: {ids_df.shape[0]} < {target_n}")
    if int(ids_df.shape[0]) > target_n:
        ids_df = ids_df.sample(n=target_n, random_state=int(args.seed)).reset_index(drop=True)
        keep = set(ids_df["sequence_id"].tolist())
        tok_df = tok_df[tok_df["sequence_id"].isin(keep)].reset_index(drop=True)

    ids_path = out_dir / "calibration_v1_ids.parquet"
    tok_path = out_dir / "calibration_v1_tokens.parquet"
    ids_df.to_parquet(ids_path, index=False)
    tok_df.to_parquet(tok_path, index=False)

    by_domain = ids_df.groupby("domain").size().to_dict()
    by_loader = ids_df.groupby("loader_used").size().to_dict()
    ids_sha = file_sha256(ids_path)
    manifest = {
        "timestamp": timestamp_now(),
        "split_id": "calibration_v1",
        "model_tokenizer": model_name,
        "seed": int(args.seed),
        "target_n": int(target_n),
        "token_filter": {"min": 128, "max": 512},
        "per_domain_counts": {str(k): int(v) for k, v in by_domain.items()},
        "loader_used_counts": {str(k): int(v) for k, v in by_loader.items()},
        "loader_provenance": provenance,
        "sha256_ids_parquet": ids_sha,
        "ids_path": str(ids_path),
        "tokens_path": str(tok_path),
    }
    write_json(out_dir / "calibration_v1_manifest.json", manifest)
    print(f"[calibration_v1] wrote {ids_path} ({ids_df.shape[0]} rows)")
    print(f"[calibration_v1] sha256={ids_sha}")


if __name__ == "__main__":
    main()
