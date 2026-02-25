from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from shared.config import default_paths
from shared.specs import DatasetSpec, ModelSpec, SequenceLengthSpec

JSONL_SUFFIX = ".jsonl"
MANIFEST_SUFFIX = ".manifest.json"
MANIFEST_VERSION = 1


@dataclass(frozen=True)
class TokenizationRequest:
    model: ModelSpec
    dataset: DatasetSpec
    seq_len: SequenceLengthSpec
    center_count: int = 100
    eval_count: int = 100
    data_root: Path | None = None
    seed: int | None = None

    @property
    def total_required(self) -> int:
        if self.dataset.needs_center_split:
            return self.center_count + self.eval_count
        return self.eval_count

    @property
    def effective_data_root(self) -> Path:
        return self.data_root or default_paths().data_dir


@dataclass(frozen=True)
class SequenceRecord:
    seq_id: int
    split: str
    tokens: list[int]
    length: int
    sha256: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "id": self.seq_id,
                "split": self.split,
                "tokens": self.tokens,
                "length": self.length,
                "sha256": self.sha256,
            }
        )


def tokenize_and_save(request: TokenizationRequest, output_path: Path) -> dict:
    records, manifest = _tokenize(request)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_f:
        for record in records:
            out_f.write(record.to_json())
            out_f.write("\n")
    manifest_path = output_path.with_suffix(MANIFEST_SUFFIX)
    with manifest_path.open("w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)
    return manifest


def _tokenize(request: TokenizationRequest) -> tuple[list[SequenceRecord], dict]:
    seed = _compute_seed(request) if request.seed is None else request.seed
    tokenizer = AutoTokenizer.from_pretrained(request.model.hf_id, use_fast=True)
    sequences = _collect_sequences(request, tokenizer, seed)

    required = request.total_required
    if len(sequences) < required:
        raise RuntimeError(
            f"Only collected {len(sequences)} sequences for {request.dataset.name}; "
            f"need {required}. Consider increasing dataset max or shard count."
        )

    # Assign splits deterministically according to preregistered scheme.
    records: list[SequenceRecord] = []
    center_goal = request.center_count if request.dataset.needs_center_split else 0
    eval_goal = request.eval_count
    center_count = 0
    eval_count = 0
    for idx, tokens in enumerate(sequences[: required]):
        if center_count < center_goal:
            split = "centering"
            center_count += 1
        else:
            split = "eval"
            eval_count += 1
        digest = _hash_tokens(tokens)
        records.append(
            SequenceRecord(
                seq_id=idx,
                split=split,
                tokens=tokens,
                length=request.seq_len.tokens,
                sha256=digest,
            )
        )

    _validate_counts(request, center_count, eval_count, center_goal, eval_goal)

    manifest = {
        "version": MANIFEST_VERSION,
        "model": request.model.name,
        "dataset": request.dataset.name,
        "sequence_length": request.seq_len.tokens,
        "seed": seed,
        "needs_center_split": request.dataset.needs_center_split,
        "center_count": center_count,
        "eval_count": eval_count,
        "expected_center": center_goal,
        "expected_eval": eval_goal,
        "total_records": len(records),
        "total_required": required,
    }
    return records, manifest


def _validate_counts(
    request: TokenizationRequest,
    center_count: int,
    eval_count: int,
    center_goal: int,
    eval_goal: int,
) -> None:
    if request.dataset.needs_center_split:
        if center_count != center_goal or eval_count != eval_goal:
            raise RuntimeError(
                "Tokenization split mismatch for "
                f"{request.dataset.name}/{request.model.name}/len_{request.seq_len.tokens}: "
                f"expected center={center_goal}, eval={eval_goal} but got "
                f"center={center_count}, eval={eval_count}"
            )
    else:
        if center_count != 0:
            raise RuntimeError(
                f"Synthetic dataset {request.dataset.name} should not allocate centering sequences."
            )


def _collect_sequences(request: TokenizationRequest, tokenizer, seed: int) -> list[list[int]]:
    total_required = request.total_required
    if request.dataset.kind == "synthetic":
        return _generate_random_sequences(tokenizer, request.seq_len.tokens, total_required, seed)

    dataset = _load_source_dataset(request, total_required)
    buffer: list[list[int]] = []
    for sample in _iterate_samples(dataset, seed):
        text = _extract_text(sample)
        if not text:
            continue
        token_ids = tokenizer(text)["input_ids"]
        if len(token_ids) < request.seq_len.tokens:
            continue
        buffer.append(token_ids[: request.seq_len.tokens])
        if len(buffer) >= total_required:
            break
    return buffer


def _generate_random_sequences(tokenizer, seq_len: int, count: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    vocab = list(range(tokenizer.vocab_size))
    sequences = [rng.choices(vocab, k=seq_len) for _ in range(count)]
    return sequences


def _load_source_dataset(request: TokenizationRequest, total_required: int) -> Dataset:
    dataset = request.dataset
    max_pool = dataset.max_sequences or max(total_required * 4, total_required + 10)
    if dataset.download_strategy == "snapshot":
        snapshot_dir = request.effective_data_root / "snapshots" / dataset.name
        parquet_files = _resolve_snapshot_files(snapshot_dir, dataset)
        if not parquet_files:
            raise RuntimeError(f"No parquet files available for snapshot dataset {dataset.name}.")
        split = f"train[:{max_pool}]"
        return load_dataset("parquet", data_files=parquet_files, split=split)

    if dataset.hf_id is None:
        raise ValueError(f"Dataset {dataset.name} missing hf_id")
    kwargs = {}
    if dataset.config:
        kwargs["name"] = dataset.config
    cache_dir = request.effective_data_root / "hf_cache" / dataset.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    split = f"{dataset.split}[:{max_pool}]"
    return load_dataset(dataset.hf_id, split=split, cache_dir=str(cache_dir), **kwargs)


def _resolve_snapshot_files(snapshot_dir: Path, dataset: DatasetSpec) -> list[str]:
    if dataset.snapshot_files:
        return [str(snapshot_dir / rel) for rel in dataset.snapshot_files]
    if dataset.snapshot_patterns:
        files: list[str] = []
        for pattern in dataset.snapshot_patterns:
            files.extend(str(path) for path in snapshot_dir.glob(pattern))
        return sorted(files)
    return []


def _iterate_samples(dataset: Dataset, seed: int) -> Iterator[dict]:
    shuffled = dataset.shuffle(seed=seed)  # deterministic order
    for sample in shuffled:
        yield sample


def _extract_text(sample: dict) -> str | None:
    for key in ("text", "content", "code", "document"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _hash_tokens(tokens: Iterable[int]) -> str:
    joined = ",".join(str(tok) for tok in tokens)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _compute_seed(request: TokenizationRequest) -> int:
    base = hash((request.model.name, request.dataset.name, request.seq_len.tokens))
    return base & 0x7FFFFFFF
