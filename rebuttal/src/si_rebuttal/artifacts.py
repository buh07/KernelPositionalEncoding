from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import Protocol, TypeAlias, TypeGuard

from .provenance import canonical_json_bytes, sha256_bytes


class ArtifactError(RuntimeError):
    """Raised when immutable runner artifacts cannot be created or resumed safely."""


SCHEMA_VERSION = 1
JsonPrimitive: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
JsonMapping: TypeAlias = Mapping[str, JsonValue]


class _DataclassInstance(Protocol):
    __dataclass_fields__: dict[str, object]


def _is_dataclass_instance(value: object) -> TypeGuard[_DataclassInstance]:
    return not isinstance(value, type) and is_dataclass(value)


def _is_json_sequence(value: Sequence[object]) -> TypeGuard[Sequence[JsonValue]]:
    return all(_is_json_value(item) for item in value)


def _is_object_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)


def _is_object_mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    return isinstance(value, Mapping)


def _is_object_dict(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)


def _is_str_keyed_mapping(value: object) -> TypeGuard[Mapping[str, object]]:
    if not _is_object_mapping(value):
        return False
    for key in value.keys():
        if not isinstance(key, str):
            return False
    return True


def _is_json_mapping(value: Mapping[str, object]) -> TypeGuard[JsonMapping]:
    return all(_is_json_value(item) for item in value.values())


def _is_json_object_dict(value: object) -> TypeGuard[dict[str, JsonValue]]:
    if not _is_object_dict(value):
        return False
    for key, item in value.items():
        if not isinstance(key, str) or not _is_json_value(item):
            return False
    return True


def _is_json_value(value: object) -> bool:
    if value is None or isinstance(value, bool | int | float | str):
        return True
    if _is_object_sequence(value):
        return _is_json_sequence(value)
    if _is_str_keyed_mapping(value):
        return _is_json_mapping(value)
    if _is_object_mapping(value):
        return False
    return False


def _require_json_object(value: object, *, label: str) -> JsonObject:
    if not _is_json_object_dict(value):
        if isinstance(value, dict):
            raise ArtifactError(f"{label} contains non-JSON values.")
        raise ArtifactError(f"{label} must decode to a JSON object.")
    return dict(value)


def _normalize_dataclass(value: _DataclassInstance) -> JsonObject:
    normalized: JsonObject = {}
    for field_name in value.__dataclass_fields__:
        field_value: object = getattr(value, field_name)
        normalized[field_name] = _normalize(field_value)
    return normalized


def _normalize_sequence_items(value: Sequence[object]) -> list[JsonValue]:
    return [_normalize(item) for item in value]


def _normalize_object_mapping(value: Mapping[object, object]) -> JsonObject:
    normalized: JsonObject = {}
    items: list[tuple[str, object]] = []
    for key, item in value.items():
        items.append((str(key), item))
    for key, item in sorted(items, key=lambda pair: pair[0]):
        normalized[key] = _normalize(item)
    return normalized


def _normalize(value: object) -> JsonValue:
    if _is_dataclass_instance(value):
        return _normalize_dataclass(value)
    if isinstance(value, Path):
        return str(value)
    if _is_object_mapping(value):
        return _normalize_object_mapping(value)
    if _is_object_sequence(value):
        return _normalize_sequence_items(value)
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"Unsupported artifact payload value: {type(value).__name__}")


def artifact_identity(payload: object) -> str:
    return sha256_bytes(canonical_json_bytes(_normalize(payload)))


def _normalize_json_object(payload: JsonMapping) -> JsonObject:
    return {key: _normalize(item) for key, item in sorted(payload.items())}


def payload_without_identity(payload: JsonMapping) -> JsonObject:
    normalized = dict(_normalize_json_object(payload))
    normalized.pop("identity_sha256", None)
    return normalized


def recompute_payload_identity(payload: JsonMapping) -> str:
    return artifact_identity(payload_without_identity(payload))


def ensure_payload_identity(payload: object) -> JsonValue:
    normalized = _normalize(payload)
    if isinstance(normalized, dict):
        enriched: JsonObject = dict(normalized)
        enriched["identity_sha256"] = recompute_payload_identity(enriched)
        return enriched
    return normalized


def verify_payload_identity(payload: JsonMapping, *, label: str) -> JsonObject:
    normalized = dict(_normalize_json_object(payload))
    expected = recompute_payload_identity(normalized)
    actual = normalized.get("identity_sha256")
    if actual != expected:
        raise ArtifactError(f"{label} identity mismatch: expected {expected}, found {actual!r}.")
    normalized["identity_sha256"] = expected
    return normalized


def atomic_write_bytes_no_clobber(path: str | Path, data: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ArtifactError(f"Refusing to overwrite existing artifact: {destination}") from exc
    dir_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def atomic_write_json_no_clobber(path: str | Path, payload: object) -> None:
    atomic_write_bytes_no_clobber(
        path,
        canonical_json_bytes(ensure_payload_identity(payload)) + b"\n",
    )


def atomic_replace_json(path: str | Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(ensure_payload_identity(payload)) + b"\n"
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
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


def read_json_bytes(path: str | Path) -> bytes:
    return Path(path).read_bytes()


def assert_resume_payload(path: str | Path, expected_identity: str) -> JsonObject:
    source = Path(path)
    if not source.exists():
        raise ArtifactError(f"Resume artifact is missing: {source}")
    payload_text = source.read_text(encoding="ascii")
    identity = sha256_bytes(payload_text.encode("ascii"))
    if identity != expected_identity:
        raise ArtifactError(
            f"Resume artifact identity mismatch for {source}: expected {expected_identity}, "
            f"found {identity}."
        )
    return _require_json_object(json.loads(payload_text), label=f"Resume artifact {source}")


def load_existing_or_write(
    path: str | Path,
    payload: JsonMapping,
) -> tuple[JsonObject, bool]:
    source = Path(path)
    normalized_value = ensure_payload_identity(payload)
    if not isinstance(normalized_value, dict):
        raise ArtifactError("Artifact payload must normalize to a JSON object.")
    normalized = dict(normalized_value)
    rendered = canonical_json_bytes(normalized) + b"\n"
    expected_identity = sha256_bytes(rendered)
    if source.exists():
        existing = source.read_bytes()
        actual_identity = sha256_bytes(existing)
        if actual_identity != expected_identity:
            raise ArtifactError(
                f"Existing immutable artifact conflicts at {source}: expected "
                f"{expected_identity}, found {actual_identity}."
            )
        return _require_json_object(
            json.loads(existing.decode("ascii")),
            label=f"Existing immutable artifact {source}",
        ), False
    atomic_write_bytes_no_clobber(source, rendered)
    return normalized, True


@dataclass(frozen=True)
class RunRoot:
    run_id: str
    root: Path
    batch_id: str | None

    @property
    def manifests_dir(self) -> Path:
        return self.root / "manifests"

    @property
    def shards_dir(self) -> Path:
        return self.root / "shards"

    @property
    def summaries_dir(self) -> Path:
        return self.root / "summaries"

    @property
    def receipts_dir(self) -> Path:
        return self.root / "receipts"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    def ensure(self) -> None:
        for path in (
            self.root,
            self.manifests_dir,
            self.shards_dir,
            self.summaries_dir,
            self.receipts_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ConditionKey:
    model: str
    direction: str
    unit: str
    unit_index: str
    condition: str
    trial: int | None = None

    _KNOWN_MODELS = (
        "llama-3.1-8b",
        "mistral-7b-v0.1",
        "olmo-2-7b",
    )
    _KNOWN_DIRECTIONS = (
        "wikipedia_to_code",
        "code_to_wikipedia",
    )
    _BIN_CONDITIONS = frozenset(
        {
            "source_kernel",
            "target_kernel",
            "offset_permutation",
            "norm",
        }
    )
    _CONTROL_CONDITIONS = frozenset({"offset_permutation", "norm"})
    _CONTROL_TRIALS = frozenset({0, 1, 2})
    _MAX_BIN_INDEX = 19
    _MAX_LAYER_INDEX = 31

    def stem(self) -> str:
        trial_suffix = "" if self.trial is None else f".trial-{self.trial}"
        return (
            f"{self.model}.{self.direction}.{self.unit}.{self.unit_index}."
            f"{self.condition}{trial_suffix}"
        )

    @classmethod
    def parse_stem(cls, stem: str) -> ConditionKey:
        trial: int | None = None
        stem_without_trial = stem
        if ".trial-" in stem:
            stem_without_trial, _, trial_text = stem.rpartition(".trial-")
            if trial_text not in {"0", "1", "2"}:
                raise ArtifactError(f"Unrecognized condition stem: {stem}")
            trial = int(trial_text)

        model_matches = [
            model_name
            for model_name in cls._KNOWN_MODELS
            if stem_without_trial.startswith(f"{model_name}.")
        ]
        if len(model_matches) != 1:
            raise ArtifactError(f"Unrecognized condition stem: {stem}")
        model = model_matches[0]
        remainder = stem_without_trial[len(model) + 1 :]

        direction_matches = [
            direction_name
            for direction_name in cls._KNOWN_DIRECTIONS
            if remainder.startswith(f"{direction_name}.")
        ]
        if len(direction_matches) != 1:
            raise ArtifactError(f"Unrecognized condition stem: {stem}")
        direction = direction_matches[0]
        remainder = remainder[len(direction) + 1 :]

        unit, separator, remainder = remainder.partition(".")
        if separator != "." or unit not in {"global", "bin", "layer"}:
            raise ArtifactError(f"Unrecognized condition stem: {stem}")
        unit_index, separator, condition = remainder.partition(".")
        if separator != "." or not unit_index or not condition or "." in condition:
            raise ArtifactError(f"Unrecognized condition stem: {stem}")

        if unit == "global":
            if unit_index != "baseline" or condition != "baseline" or trial is not None:
                raise ArtifactError(f"Unrecognized condition stem: {stem}")
        elif unit == "bin":
            if not unit_index.isdigit():
                raise ArtifactError(f"Unrecognized condition stem: {stem}")
            bin_index = int(unit_index)
            if bin_index < 0 or bin_index > cls._MAX_BIN_INDEX:
                raise ArtifactError(f"Unrecognized condition stem: {stem}")
            if condition not in cls._BIN_CONDITIONS:
                raise ArtifactError(f"Unrecognized condition stem: {stem}")
            if condition in cls._CONTROL_CONDITIONS:
                if trial is None or trial not in cls._CONTROL_TRIALS:
                    raise ArtifactError(f"Unrecognized condition stem: {stem}")
            elif trial is not None:
                raise ArtifactError(f"Unrecognized condition stem: {stem}")
        else:
            if direction != "wikipedia_to_code" or condition != "depth" or trial is not None:
                raise ArtifactError(f"Unrecognized condition stem: {stem}")
            if not unit_index.isdigit():
                raise ArtifactError(f"Unrecognized condition stem: {stem}")
            layer_index = int(unit_index)
            if layer_index < 0 or layer_index > cls._MAX_LAYER_INDEX:
                raise ArtifactError(f"Unrecognized condition stem: {stem}")

        return cls(
            model=model,
            direction=direction,
            unit=unit,
            unit_index=unit_index,
            condition=condition,
            trial=trial,
        )


def iter_condition_paths(run_root: RunRoot, keys: Iterable[ConditionKey]) -> list[Path]:
    return [run_root.shards_dir / f"{key.stem()}.json" for key in keys]


def build_batch_completion_payload(
    *,
    batch_id: str,
    run_ids: Sequence[str],
    command: Sequence[str],
    config_sha256: str,
    sweep_sha256: str,
) -> JsonObject:
    payload: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "batch_id": batch_id,
        "run_ids": list(run_ids),
        "command": list(command),
        "config_sha256": config_sha256,
        "sweep_sha256": sweep_sha256,
    }
    payload["identity_sha256"] = artifact_identity(payload)
    return payload
