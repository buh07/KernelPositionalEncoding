from __future__ import annotations

import json
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

from experiment2.tasks import TaskExample


def _stable_hash(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_cache_key(
    *,
    generator_hash: str,
    model: str,
    task: str,
    seq_len: int,
    seed: int,
    span: int | None,
    synthetic_count: int,
    fallback_spans: tuple[int, ...],
) -> str:
    payload = {
        "generator_hash": generator_hash,
        "model": model,
        "task": task,
        "seq_len": int(seq_len),
        "seed": int(seed),
        "span": None if span is None else int(span),
        "synthetic_count": int(synthetic_count),
        "fallback_spans": [int(x) for x in fallback_spans],
    }
    return _stable_hash(payload)


def cache_paths(cache_root: Path, key: str) -> tuple[Path, Path]:
    return (
        cache_root / f"{key}.jsonl",
        cache_root / f"{key}.meta.json",
    )


def load_cached_examples(cache_root: Path, key: str) -> list[TaskExample] | None:
    data_path, meta_path = cache_paths(cache_root, key)
    if not data_path.exists() or not meta_path.exists():
        return None
    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [TaskExample(**row) for row in rows]


def write_cached_examples(
    *,
    cache_root: Path,
    key: str,
    examples: list[TaskExample],
    metadata: dict[str, Any],
) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    data_path, meta_path = cache_paths(cache_root, key)
    with data_path.open("w", encoding="utf-8") as handle:
        for ex in examples:
            handle.write(json.dumps(asdict(ex), sort_keys=True) + "\n")
    meta_payload = dict(metadata)
    meta_payload["cache_key"] = key
    meta_path.write_text(json.dumps(meta_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
