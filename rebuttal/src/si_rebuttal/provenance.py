from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, is_dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Protocol, TypeAlias, TypeGuard


class ProvenanceError(RuntimeError):
    """Raised when a provenance artifact cannot be captured deterministically."""


JsonPrimitive: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


class _DataclassInstance(Protocol):
    __dataclass_fields__: dict[str, object]


class _Hasher(Protocol):
    def update(self, data: bytes, /) -> object: ...


def _is_dataclass_instance(value: object) -> TypeGuard[_DataclassInstance]:
    return not isinstance(value, type) and is_dataclass(value)


def _is_object_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)


def _is_object_mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    return isinstance(value, Mapping)


def _normalize_dataclass(obj: _DataclassInstance) -> JsonObject:
    normalized: JsonObject = {}
    for field_name in obj.__dataclass_fields__:
        field_value: object = getattr(obj, field_name)
        normalized[field_name] = _normalize(field_value)
    return normalized


def _normalize_sequence_items(obj: Sequence[object]) -> list[JsonValue]:
    return [_normalize(value) for value in obj]


def _normalize_mapping(obj: Mapping[object, object]) -> JsonObject:
    normalized: JsonObject = {}
    items: list[tuple[str, object]] = []
    for key, value in obj.items():
        items.append((str(key), value))
    for key, value in sorted(items, key=lambda item: item[0]):
        normalized[key] = _normalize(value)
    return normalized


def _normalize(obj: object) -> JsonValue:
    if _is_dataclass_instance(obj):
        return _normalize_dataclass(obj)
    if isinstance(obj, Path):
        return str(obj)
    if _is_object_mapping(obj):
        return _normalize_mapping(obj)
    if _is_object_sequence(obj):
        return _normalize_sequence_items(obj)
    if obj is None or isinstance(obj, bool | int | float | str):
        return obj
    raise TypeError(f"Unsupported provenance payload value: {type(obj).__name__}")


def canonical_json_bytes(payload: object) -> bytes:
    """Encode payload as deterministic compact JSON."""

    normalized = _normalize(payload)
    text = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return text.encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(payload: str) -> str:
    return sha256_bytes(payload.encode("utf-8"))


def compact_json_seed(parts: Sequence[object]) -> int:
    """Return the ADR seed derived from the first eight SHA-256 bytes of compact JSON."""

    digest = hashlib.sha256(canonical_json_bytes(list(parts))).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _stream_file_hash(path: Path, hasher: _Hasher) -> None:
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)


def sha256_tree(root: str | Path) -> str:
    """Hash a file or directory tree without loading the whole tree into memory."""

    root_path = Path(root).resolve()
    if not root_path.exists():
        raise ProvenanceError(f"Missing path for tree hash: {root_path}")
    hasher = hashlib.sha256()
    if root_path.is_file():
        hasher.update(root_path.name.encode("utf-8"))
        _stream_file_hash(root_path, hasher)
        return hasher.hexdigest()
    for path in sorted(candidate for candidate in root_path.rglob("*") if candidate.is_file()):
        rel = path.relative_to(root_path).as_posix()
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\0")
        _stream_file_hash(path, hasher)
        hasher.update(b"\0")
    return hasher.hexdigest()


def atomic_write_json(path: str | Path, payload: object) -> None:
    """Write canonical JSON atomically with durable replace semantics."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(payload) + b"\n"
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
        dir_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if temp_path.exists():
            temp_path.unlink()


@dataclass(frozen=True)
class GitBinding:
    commit_sha: str
    tracked_diff_sha256: str
    tracked_paths: tuple[str, ...]
    relevant_content_sha256: str
    untracked_paths: tuple[str, ...]
    untracked_sha256: str


@dataclass(frozen=True)
class DatasetBinding:
    name: str
    repository: str
    revision: str
    config: str
    split: str
    field: str
    fingerprint: str
    manifest_sha256: str


@dataclass(frozen=True)
class TokenizerBinding:
    model_name: str
    tokenizer_path: Path
    tokenizer_tree_sha256: str


@dataclass(frozen=True)
class ModelBinding:
    model_name: str
    weights_path: Path
    weights_tree_sha256: str


@dataclass(frozen=True)
class RuntimeBinding:
    python_version: str
    platform: str
    package_versions: Mapping[str, str]
    cuda_version: str | None
    driver_version: str | None


@dataclass(frozen=True)
class RunProvenance:
    schema_version: int
    seed_namespace: int
    command: tuple[str, ...]
    config_identity_sha256: str
    sweep_identity_sha256: str
    git: GitBinding
    datasets: tuple[DatasetBinding, ...]
    tokenizers: tuple[TokenizerBinding, ...]
    models: tuple[ModelBinding, ...]
    runtime: RuntimeBinding


def _normalize_repo_paths(repo: Path, paths: Sequence[str | Path]) -> tuple[str, ...]:
    normalized: list[str] = []
    for path in paths:
        candidate = Path(path)
        resolved = candidate.resolve() if candidate.is_absolute() else (repo / candidate).resolve()
        try:
            relative = resolved.relative_to(repo)
        except ValueError as exc:
            raise ProvenanceError(f"Tracked path escapes repository root: {path}") from exc
        normalized.append(relative.as_posix())
    return tuple(dict.fromkeys(normalized))


def _iter_relevant_files(repo: Path, tracked_paths: Sequence[str]) -> tuple[str, ...]:
    files: set[str] = set()
    for relative in tracked_paths:
        candidate = repo / relative
        if candidate.is_file():
            files.add(relative)
            continue
        if candidate.is_dir():
            files.update(
                path.relative_to(repo).as_posix() for path in candidate.rglob("*") if path.is_file()
            )
    return tuple(sorted(files))


def _hash_repo_files(repo: Path, relative_paths: Sequence[str]) -> str:
    hasher = hashlib.sha256()
    for relative in relative_paths:
        path = repo / relative
        if not path.is_file():
            continue
        hasher.update(relative.encode("utf-8"))
        hasher.update(b"\0")
        _stream_file_hash(path, hasher)
        hasher.update(b"\0")
    return hasher.hexdigest()


def _list_standard_untracked(repo: Path, tracked_paths: Sequence[str]) -> tuple[str, ...]:
    listing = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            *tracked_paths,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(path for path in listing.stdout.splitlines() if path)


def capture_git_binding(repo_root: str | Path, tracked_paths: Sequence[str | Path]) -> GitBinding:
    """Capture git commit and diff digest for the tracked rebuttal slice."""

    repo = Path(repo_root).resolve()
    tracked = _normalize_repo_paths(repo, tracked_paths)
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit_sha = head.stdout.strip()
    diff_proc = subprocess.Popen(
        ["git", "-C", str(repo), "diff", "--no-ext-diff", "--binary", "HEAD", "--", *tracked],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout = diff_proc.stdout
    stderr_pipe = diff_proc.stderr
    if stdout is None or stderr_pipe is None:
        raise ProvenanceError("git diff did not expose stdout/stderr pipes")
    diff_hasher = hashlib.sha256()
    for chunk in iter(lambda: stdout.read(1024 * 1024), b""):
        diff_hasher.update(chunk)
    stderr = stderr_pipe.read()
    return_code = diff_proc.wait()
    if return_code != 0:
        raise ProvenanceError(stderr.decode("utf-8", errors="replace").strip() or "git diff failed")
    relevant_paths = _iter_relevant_files(repo, tracked)
    untracked_paths = _list_standard_untracked(repo, tracked)
    return GitBinding(
        commit_sha=commit_sha,
        tracked_diff_sha256=diff_hasher.hexdigest(),
        tracked_paths=tracked,
        relevant_content_sha256=_hash_repo_files(repo, relevant_paths),
        untracked_paths=untracked_paths,
        untracked_sha256=_hash_repo_files(repo, untracked_paths),
    )


def capture_runtime_binding(
    package_names: Sequence[str],
    *,
    cuda_version: str | None = None,
    driver_version: str | None = None,
) -> RuntimeBinding:
    """Capture runtime versions used by the rebuttal environment."""

    versions = {name: version(name) for name in package_names}
    return RuntimeBinding(
        python_version=sys.version,
        platform=platform.platform(),
        package_versions=versions,
        cuda_version=cuda_version,
        driver_version=driver_version,
    )


def build_run_provenance(
    *,
    seed_namespace: int,
    command: Sequence[str],
    config_identity_sha256: str,
    sweep_identity_sha256: str,
    git: GitBinding,
    datasets: Sequence[DatasetBinding],
    tokenizers: Sequence[TokenizerBinding],
    models: Sequence[ModelBinding],
    runtime: RuntimeBinding,
) -> RunProvenance:
    """Build a schema-versioned provenance object for a run manifest."""

    return RunProvenance(
        schema_version=1,
        seed_namespace=seed_namespace,
        command=tuple(command),
        config_identity_sha256=config_identity_sha256,
        sweep_identity_sha256=sweep_identity_sha256,
        git=git,
        datasets=tuple(datasets),
        tokenizers=tuple(tokenizers),
        models=tuple(models),
        runtime=runtime,
    )
