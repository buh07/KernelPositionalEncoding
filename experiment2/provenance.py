from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

from experiment2.long_offsets import load_long_offset_bundle


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def generator_hash() -> str:
    from experiment2 import tasks

    source = inspect.getsource(tasks)
    lock_hash = load_long_offset_bundle(strict=False).lock_hash
    payload = {
        "tasks_source_sha256": sha256_text(source),
        "long_offset_lock_hash": lock_hash,
    }
    return sha256_text(json.dumps(payload, sort_keys=True))


def intervention_hash() -> str:
    from experiment2 import interventions

    source = inspect.getsource(interventions)
    return sha256_text(source)


def model_tokenizer_hash(model_spec, model, tokenizer) -> str:
    payload = {
        "model_spec": model_spec.name,
        "model_class": model.__class__.__name__,
        "model_name_or_path": str(getattr(model.config, "_name_or_path", "")),
        "model_hidden_size": int(getattr(model.config, "hidden_size", 0) or 0),
        "model_layers": int(getattr(model.config, "num_hidden_layers", 0) or 0),
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "tokenizer_vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
        "tokenizer_special_ids": sorted(int(x) for x in getattr(tokenizer, "all_special_ids", []) if x is not None),
    }
    return sha256_text(json.dumps(payload, sort_keys=True))


def row_hash(row: dict[str, Any]) -> str:
    return sha256_text(json.dumps(row, sort_keys=True))


def provenance_payload(*, generator: str, intervention: str, model_tokenizer: str) -> dict[str, str]:
    return {
        "generator_hash": generator,
        "intervention_hash": intervention,
        "model_tokenizer_hash": model_tokenizer,
    }


def provenance_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = ("generator_hash", "intervention_hash", "model_tokenizer_hash")
    return all(left.get(k) == right.get(k) for k in keys)
