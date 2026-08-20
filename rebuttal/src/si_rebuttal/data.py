from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol, TypeAlias, TypeGuard

import numpy as np

from .config import FIXED_COUNTS, FIXED_DATASETS, FIXED_RUNTIME, DatasetConfig
from .provenance import canonical_json_bytes, sha256_bytes, sha256_text

TITLE_PATTERN = re.compile(r"^ = [^=]+ = $")


class DataMaterializationError(ValueError):
    """Raised when the frozen materialization protocol cannot be satisfied."""


@dataclass(frozen=True)
class DocumentRecord:
    domain: str
    document_id: str
    row_indices: tuple[int, ...]
    text: str
    content_sha256: str
    assignment_sha256: str
    partition: str


@dataclass(frozen=True)
class SelectedChunk:
    document_id: str
    partition: str
    chunk_index: int
    source_rows: tuple[int, ...]
    token_count: int
    token_shape: tuple[int, ...]
    token_sha256: str


@dataclass(frozen=True)
class MaterializedDomain:
    schema_version: int
    domain: str
    dataset_repository: str
    dataset_revision: str
    dataset_config: str
    dataset_split: str
    dataset_field: str
    dataset_fingerprint: str
    tokenizer_name: str
    tokenizer_tree_sha256: str
    fit_documents: tuple[DocumentRecord, ...]
    eval_documents: tuple[DocumentRecord, ...]
    fit_chunks: tuple[SelectedChunk, ...]
    eval_chunks: tuple[SelectedChunk, ...]
    manifest_sha256: str


DatasetRow: TypeAlias = Mapping[str, object]


@dataclass(frozen=True)
class LoadedDatasetRows:
    rows: tuple[DatasetRow, ...]
    fingerprint: str
    column_names: tuple[str, ...]


class _LoadedDatasetLike(Protocol):
    _fingerprint: object
    column_names: Sequence[object]

    def __iter__(self) -> Iterator[object]: ...


DatasetLoaderFn: TypeAlias = Callable[..., _LoadedDatasetLike]


class ImmutableHFDatasetLoader:
    """Load only the ADR-frozen Hugging Face dataset snapshot for production materialization."""

    def __init__(self, load_dataset_fn: DatasetLoaderFn) -> None:
        self._load_dataset_fn = load_dataset_fn

    def load(self, dataset: DatasetConfig) -> LoadedDatasetRows:
        _require_frozen_dataset_config(dataset)
        loaded = self._load_dataset_fn(
            dataset.repository,
            name=dataset.config,
            split=dataset.split,
            revision=dataset.revision,
            streaming=False,
        )
        fingerprint = getattr(loaded, "_fingerprint", None)
        if not isinstance(fingerprint, str) or fingerprint == "":
            raise DataMaterializationError("Loaded dataset must expose a non-empty _fingerprint")
        column_names_raw = getattr(loaded, "column_names", None)
        if not _is_column_name_sequence(column_names_raw):
            raise DataMaterializationError("Loaded dataset must expose column_names")
        column_names = _normalize_column_names(column_names_raw)
        required_columns = {dataset.field}
        if dataset.repository_field is not None:
            required_columns.add(dataset.repository_field)
        missing_columns = sorted(required_columns - set(column_names))
        if missing_columns:
            raise DataMaterializationError(
                f"Loaded dataset is missing required columns: {missing_columns}"
            )
        rows = tuple(_freeze_loaded_rows(loaded))
        return LoadedDatasetRows(rows=rows, fingerprint=fingerprint, column_names=column_names)


def _is_loaded_dataset_row(value: object) -> TypeGuard[Mapping[object, object]]:
    return isinstance(value, Mapping)


def _is_column_name_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes)


def _normalize_column_names(names: Sequence[object]) -> tuple[str, ...]:
    return tuple(str(name) for name in names)


def _normalize_loaded_dataset_row(row: Mapping[object, object]) -> DatasetRow:
    return {str(key): value for key, value in row.items()}


def _freeze_loaded_rows(loaded: Iterable[object]) -> Iterable[DatasetRow]:
    for row in loaded:
        if _is_loaded_dataset_row(row):
            yield _normalize_loaded_dataset_row(row)
            continue
        raise DataMaterializationError(
            f"Loaded dataset row must be a mapping, found {type(row).__name__}"
        )


def _require_frozen_dataset_config(dataset: DatasetConfig) -> None:
    expected = FIXED_DATASETS.get(dataset.domain)
    if expected is None:
        raise DataMaterializationError(f"Unsupported frozen dataset domain: {dataset.domain}")
    normalized_expected = {
        "domain": expected["domain"],
        "repository": expected["repository"],
        "revision": expected["revision"],
        "config": expected["config"],
        "split": expected["split"],
        "field": expected["field"],
        "document_title_regex": expected.get("document_title_regex"),
        "repository_field": expected.get("repository_field"),
    }
    actual = {
        "domain": dataset.domain,
        "repository": dataset.repository,
        "revision": dataset.revision,
        "config": dataset.config,
        "split": dataset.split,
        "field": dataset.field,
        "document_title_regex": dataset.document_title_regex,
        "repository_field": dataset.repository_field,
    }
    if actual != normalized_expected:
        raise DataMaterializationError(
            f"Dataset config does not match frozen ADR snapshot for {dataset.domain}"
        )


def _partition_for_document(dataset_revision: str, document_id: str) -> tuple[str, str]:
    digest = sha256_text(f"29039|{dataset_revision}|{document_id}")
    bucket = int.from_bytes(bytes.fromhex(digest)[:32], byteorder="big", signed=False) % 3
    return ("fit" if bucket == 0 else "eval"), digest


def _sorted_partition(
    documents: Iterable[DocumentRecord], partition: str
) -> tuple[DocumentRecord, ...]:
    selected = [document for document in documents if document.partition == partition]
    selected.sort(key=lambda document: (document.assignment_sha256, document.document_id))
    return tuple(selected)


def build_wikipedia_documents(
    rows: Sequence[str], dataset_revision: str
) -> tuple[DocumentRecord, ...]:
    """Build non-empty top-level WikiText documents without crossing title boundaries."""

    documents: list[DocumentRecord] = []
    current_rows: list[int] = []
    current_text: list[str] = []
    start_row: int | None = None

    def flush(end_row: int) -> None:
        nonlocal current_rows, current_text, start_row
        if start_row is None or not current_rows:
            current_rows = []
            current_text = []
            start_row = None
            return
        document_id = f"wiki:{start_row}:{end_row}"
        joined = "\n\n".join(current_text)
        partition, assignment_sha256 = _partition_for_document(dataset_revision, document_id)
        documents.append(
            DocumentRecord(
                domain="wikipedia",
                document_id=document_id,
                row_indices=tuple(current_rows),
                text=joined,
                content_sha256=sha256_text(joined),
                assignment_sha256=assignment_sha256,
                partition=partition,
            )
        )
        current_rows = []
        current_text = []
        start_row = None

    seen_first_title = False
    for row_index, text in enumerate(rows):
        if TITLE_PATTERN.fullmatch(text.rstrip("\r\n")):
            if seen_first_title:
                flush(row_index - 1)
            seen_first_title = True
            start_row = row_index
            current_rows = [row_index]
            current_text = [text]
            continue
        if not seen_first_title or text == "":
            continue
        current_rows.append(row_index)
        current_text.append(text)

    if seen_first_title and start_row is not None:
        flush(len(rows) - 1)
    return tuple(documents)


def build_code_documents(
    rows: Sequence[DatasetRow],
    dataset_revision: str,
    *,
    text_field: str = "whole_func_string",
    repository_field: str = "repository_name",
) -> tuple[DocumentRecord, ...]:
    """Group CodeSearchNet rows by repository name in immutable row order."""

    grouped_rows: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        repository_name = str(row.get(repository_field, "") or "")
        text = str(row.get(text_field, "") or "")
        if repository_name == "" or text == "":
            continue
        grouped_rows[repository_name].append((row_index, text))

    documents: list[DocumentRecord] = []
    for repository_name, items in grouped_rows.items():
        row_indices = tuple(index for index, _ in items)
        joined = "\n\n".join(text for _, text in items)
        document_id = f"code:{repository_name}"
        partition, assignment_sha256 = _partition_for_document(dataset_revision, document_id)
        documents.append(
            DocumentRecord(
                domain="code",
                document_id=document_id,
                row_indices=row_indices,
                text=joined,
                content_sha256=sha256_text(joined),
                assignment_sha256=assignment_sha256,
                partition=partition,
            )
        )
    return tuple(documents)


def token_sha256(tokens: Sequence[int]) -> str:
    array = np.ascontiguousarray(np.asarray(tuple(int(token) for token in tokens), dtype="<i8"))
    shape_bytes = canonical_json_bytes({"shape": list(array.shape)})
    return sha256_bytes(shape_bytes + b"\0" + array.tobytes(order="C"))


def _chunk_document(
    document: DocumentRecord, encode: Callable[[str], Sequence[int]], chunk_length: int
) -> tuple[SelectedChunk, ...]:
    tokens = tuple(int(token) for token in encode(document.text))
    full_chunks = len(tokens) // chunk_length
    chunks: list[SelectedChunk] = []
    for chunk_index in range(full_chunks):
        start = chunk_index * chunk_length
        end = start + chunk_length
        chunk_tokens = tokens[start:end]
        chunks.append(
            SelectedChunk(
                document_id=document.document_id,
                partition=document.partition,
                chunk_index=chunk_index,
                source_rows=document.row_indices,
                token_count=len(chunk_tokens),
                token_shape=(len(chunk_tokens),),
                token_sha256=token_sha256(chunk_tokens),
            )
        )
    return tuple(chunks)


def _select_chunks(
    documents: Sequence[DocumentRecord],
    encode: Callable[[str], Sequence[int]],
    *,
    chunk_length: int,
    required_count: int,
) -> tuple[SelectedChunk, ...]:
    selected: list[SelectedChunk] = []
    for document in documents:
        for chunk in _chunk_document(document, encode, chunk_length):
            selected.append(chunk)
            if len(selected) == required_count:
                return tuple(selected)
    raise DataMaterializationError(
        f"Required {required_count} chunks at length {chunk_length}, found {len(selected)}"
    )


def _documents_referenced_by_chunks(
    documents: Sequence[DocumentRecord], chunks: Sequence[SelectedChunk]
) -> tuple[DocumentRecord, ...]:
    selected_ids = {chunk.document_id for chunk in chunks}
    return tuple(document for document in documents if document.document_id in selected_ids)


def _reject_overlap(
    fit_documents: Sequence[DocumentRecord],
    eval_documents: Sequence[DocumentRecord],
    fit_chunks: Sequence[SelectedChunk],
    eval_chunks: Sequence[SelectedChunk],
) -> None:
    selected_fit_documents = _documents_referenced_by_chunks(fit_documents, fit_chunks)
    selected_eval_documents = _documents_referenced_by_chunks(eval_documents, eval_chunks)

    fit_ids = {document.document_id for document in selected_fit_documents}
    eval_ids = {document.document_id for document in selected_eval_documents}
    overlap_ids = fit_ids & eval_ids
    if overlap_ids:
        raise DataMaterializationError(f"Document overlap across fit/eval: {sorted(overlap_ids)}")

    fit_rows = {row for document in selected_fit_documents for row in document.row_indices}
    eval_rows = {row for document in selected_eval_documents for row in document.row_indices}
    row_overlap = fit_rows & eval_rows
    if row_overlap:
        raise DataMaterializationError(f"Source row overlap across fit/eval: {sorted(row_overlap)}")

    fit_hashes = {document.content_sha256 for document in selected_fit_documents}
    eval_hashes = {document.content_sha256 for document in selected_eval_documents}
    content_overlap = fit_hashes & eval_hashes
    if content_overlap:
        raise DataMaterializationError("Joined-document content hash overlap across fit/eval")

    fit_token_hashes = {chunk.token_sha256 for chunk in fit_chunks}
    eval_token_hashes = {chunk.token_sha256 for chunk in eval_chunks}
    token_overlap = fit_token_hashes & eval_token_hashes
    if token_overlap:
        raise DataMaterializationError("Selected token hash overlap across fit/eval")


def _manifest_identity(payload: Mapping[str, object]) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def _coerce_wikipedia_rows(rows: Sequence[str | DatasetRow], field: str) -> tuple[str, ...]:
    extracted: list[str] = []
    for row in rows:
        if isinstance(row, Mapping):
            extracted.append(str(row.get(field, "") or ""))
            continue
        extracted.append(str(row))
    return tuple(extracted)


def _require_code_rows(
    rows: Sequence[str | DatasetRow],
) -> tuple[DatasetRow, ...]:
    normalized: list[DatasetRow] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise DataMaterializationError(
                f"Code rows must be mappings, found {type(row).__name__}"
            )
        normalized.append(row)
    return tuple(normalized)


def materialize_domain(
    *,
    dataset: DatasetConfig,
    dataset_fingerprint: str,
    rows: Sequence[str | DatasetRow],
    tokenizer_name: str,
    tokenizer_tree_sha256: str,
    encode: Callable[[str], Sequence[int]],
    chunk_length: int,
    fit_count: int,
    eval_count: int,
) -> MaterializedDomain:
    """Pure helper for CPU tests from already-loaded rows."""

    if dataset.domain == "wikipedia":
        documents = build_wikipedia_documents(
            _coerce_wikipedia_rows(rows, dataset.field), dataset.revision
        )
    elif dataset.domain == "code":
        documents = build_code_documents(
            _require_code_rows(rows),
            dataset.revision,
            text_field=dataset.field,
            repository_field=dataset.repository_field or "repository_name",
        )
    else:
        raise DataMaterializationError(f"Unsupported domain {dataset.domain}")

    fit_documents = _sorted_partition(documents, "fit")
    eval_documents = _sorted_partition(documents, "eval")
    fit_chunks = _select_chunks(
        fit_documents, encode, chunk_length=chunk_length, required_count=fit_count
    )
    eval_chunks = _select_chunks(
        eval_documents, encode, chunk_length=chunk_length, required_count=eval_count
    )
    _reject_overlap(fit_documents, eval_documents, fit_chunks, eval_chunks)

    identity_payload = {
        "schema_version": 1,
        "domain": dataset.domain,
        "dataset_repository": dataset.repository,
        "dataset_revision": dataset.revision,
        "dataset_config": dataset.config,
        "dataset_split": dataset.split,
        "dataset_field": dataset.field,
        "dataset_fingerprint": dataset_fingerprint,
        "tokenizer_name": tokenizer_name,
        "tokenizer_tree_sha256": tokenizer_tree_sha256,
        "fit_documents": [asdict(document) for document in fit_documents],
        "eval_documents": [asdict(document) for document in eval_documents],
        "fit_chunks": [asdict(chunk) for chunk in fit_chunks],
        "eval_chunks": [asdict(chunk) for chunk in eval_chunks],
    }
    manifest_sha256 = _manifest_identity(identity_payload)
    return MaterializedDomain(
        schema_version=1,
        domain=dataset.domain,
        dataset_repository=dataset.repository,
        dataset_revision=dataset.revision,
        dataset_config=dataset.config,
        dataset_split=dataset.split,
        dataset_field=dataset.field,
        dataset_fingerprint=dataset_fingerprint,
        tokenizer_name=tokenizer_name,
        tokenizer_tree_sha256=tokenizer_tree_sha256,
        fit_documents=fit_documents,
        eval_documents=eval_documents,
        fit_chunks=fit_chunks,
        eval_chunks=eval_chunks,
        manifest_sha256=manifest_sha256,
    )


def materialize_frozen_domain(
    *,
    dataset: DatasetConfig,
    dataset_loader: ImmutableHFDatasetLoader,
    tokenizer_name: str,
    tokenizer_tree_sha256: str,
    encode: Callable[[str], Sequence[int]],
) -> MaterializedDomain:
    """Production materializer for the exact ADR-frozen dataset snapshot only."""

    _require_frozen_dataset_config(dataset)
    loaded = dataset_loader.load(dataset)
    return materialize_domain(
        dataset=dataset,
        dataset_fingerprint=loaded.fingerprint,
        rows=loaded.rows,
        tokenizer_name=tokenizer_name,
        tokenizer_tree_sha256=tokenizer_tree_sha256,
        encode=encode,
        chunk_length=int(FIXED_RUNTIME["chunk_length"]),
        fit_count=FIXED_COUNTS["fit_sequences"],
        eval_count=FIXED_COUNTS["eval_sequences"],
    )
