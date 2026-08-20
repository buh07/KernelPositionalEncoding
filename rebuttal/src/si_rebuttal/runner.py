from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import NotRequired, Protocol, TypeAlias, TypedDict, TypeGuard, cast

import numpy as np
import numpy.typing as npt
import torch

from .artifacts import (
    SCHEMA_VERSION,
    ArtifactError,
    ConditionKey,
    RunRoot,
    artifact_identity,
    atomic_write_bytes_no_clobber,
    atomic_write_json_no_clobber,
    ensure_payload_identity,
    load_existing_or_write,
    payload_without_identity,
    verify_payload_identity,
)
from .config import (
    FIXED_COUNTS,
    FIXED_SEED_NAMESPACE,
    FIXED_VALIDATION,
    ModelConfig,
    ResolvedConfig,
    SweepConfig,
    load_base_config,
    load_sweep_config,
)
from .controls import (
    ExpectedAttentionCallCounts,
    compact_ascii_json_seed,
)
from .data import (
    DocumentRecord,
    ImmutableHFDatasetLoader,
    MaterializedDomain,
    SelectedChunk,
    build_code_documents,
    build_wikipedia_documents,
    materialize_frozen_domain,
    token_sha256,
)
from .intervention import (
    ValidationAttentionRegistry,
    ValidationProbeRequest,
    ValidationRegistrySnapshot,
    _coerce_validation_attention_registry,  # pyright: ignore[reportPrivateUsage]
    _raise_grouped_exceptions,  # pyright: ignore[reportPrivateUsage]
    attn_implementation_attr_name,
    derive_supported_model_family_spec_for_capture,
    manual_attention_logits,
    mean_next_token_nll,
)
from .kernels import (
    NUM_HEAD_BINS,
    HeadIndex,
    bin_mean_source_r2,
    estimate_source_r2_for_sequence,
    lower_diagonal_means,
    sort_heads_into_bins,
)
from .models import (
    FrozenModelBinding,
    LoadedModelBundle,
    freeze_model_binding,
    load_local_model_bundle,
)
from .operations import (
    BATCH_ID_ENV,
    BATCH_RECEIPT_ENV,
    GPU_MEMORY_IDLE_THRESHOLD_MIB,
    LAUNCH_RECEIPT_ENV,
    RUN_ROOT_ENV,
    TMUX_GPU0,
    TMUX_GPU1,
    TMUX_SESSION_ENV,
    GpuSnapshot,
    LaunchCommand,
    LaunchPlan,
    available_disk_bytes,
    build_launch_plan,
    default_subprocess_runner,
    gpu_snapshot_payload,
    launch_tmux_plan,
    read_only_status,
    require_tmux_absent,
    require_two_idle_snapshots,
    write_launch_scripts_and_logs,
)
from .provenance import (
    DatasetBinding,
    ModelBinding,
    RuntimeBinding,
    TokenizerBinding,
    build_run_provenance,
    canonical_json_bytes,
    capture_git_binding,
    compact_json_seed,
    sha256_bytes,
)
from .statistics import (
    DEFAULT_SPEARMAN_PERMUTATIONS,
    average_control_trials_per_sequence,
    monte_carlo_spearman_positive,
    paired_percentile_bootstrap,
)


class RunnerError(RuntimeError):
    """Raised when the rebuttal runner cannot satisfy ADR-0001 invariants."""


JsonPrimitive: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
JsonMapping: TypeAlias = Mapping[str, JsonValue]
Int64Array: TypeAlias = npt.NDArray[np.int64]
Float32Array: TypeAlias = npt.NDArray[np.float32]
Float64Array: TypeAlias = npt.NDArray[np.float64]
KernelKey: TypeAlias = tuple[int, int]
KernelValue: TypeAlias = Sequence[float] | Float32Array | Float64Array | torch.Tensor
KernelMap: TypeAlias = Mapping[KernelKey, KernelValue]
HeadKernelMap: TypeAlias = dict[KernelKey, Float64Array]


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


class TokenizerLoader(Protocol):
    def __call__(self, source: str) -> TokenizerLike: ...


class DatasetLoaderFn(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> _LoadedDatasetLike: ...


class RuntimeCommandRunner(Protocol):
    def __call__(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]: ...


class TorchVersionNamespace(Protocol):
    cuda: str | None


class RuntimeCommandPayload(TypedDict):
    resolved_executable: str
    command: list[str]
    returncode: int
    stdout_text: str
    stdout_sha256: str
    stderr_text: str
    stderr_sha256: str


class PackageFreezePayload(RuntimeCommandPayload):
    output_lines: list[str]
    line_count: int
    required_exact_pins: dict[str, str]
    observed_required_pins: dict[str, str]


class GpuInventoryEntry(TypedDict):
    physical_index: int
    uuid: str
    name: str
    total_memory_mib: int
    compute_capability: str
    driver_version: str


class GpuInventoryPayload(RuntimeCommandPayload):
    gpus: list[GpuInventoryEntry]
    line_count: int
    driver_version: str


class TokenSequenceRecord(TypedDict):
    document_id: str
    partition: str
    chunk_index: int
    source_rows: list[int]
    token_count: int
    token_shape: list[int]
    token_sha256: str
    document_content_sha256: str
    document_assignment_sha256: str
    tokens: list[int]


class TokenManifestPayload(TypedDict):
    schema_version: int
    model: str
    domain: str
    partition: str
    count: int
    config_sha256: str
    sweep_sha256: str
    materialized_domain_sha256: str
    sequence_digest_sha256: str
    sequences: list[TokenSequenceRecord]
    identity_sha256: str


class HeadRecordPayload(TypedDict):
    layer: int
    head: int


class DomainEntryPayload(TypedDict):
    model: str
    domain: str
    tokenizer_tree_sha256: str
    materialized_domain_sha256: str
    dataset_repository: str
    dataset_revision: str
    dataset_config: str
    dataset_split: str
    dataset_field: str
    dataset_fingerprint: str
    fit_token_manifest_sha256: str
    eval_token_manifest_sha256: str
    fit_sequence_digest_sha256: str
    eval_sequence_digest_sha256: str
    identity_sha256: NotRequired[str]


class BatchManifestPayload(TypedDict):
    schema_version: int
    materialized_at_utc: str
    command: list[str]
    config_sha256: str
    sweep_sha256: str
    domains: dict[str, str]
    domain_entries: dict[str, DomainEntryPayload]
    tokenizers: dict[str, str]
    models: dict[str, str]
    git_commit_sha: str
    git_tracked_diff_sha256: str
    git_relevant_content_sha256: str
    git_untracked_sha256: str
    runtime: JsonObject
    runtime_capture: JsonObject
    provenance: JsonObject
    environment: dict[str, str]
    identity_sha256: str


@dataclass(frozen=True)
class _ResolvedSelectedDocument:
    document_id: str
    row_indices: tuple[int, ...]
    content_sha256: str


class FitProfileBinPayload(TypedDict):
    bin_index: int
    head_count: int
    mean_source_r2: float
    selected_heads: list[HeadRecordPayload]
    selected_head_digest_sha256: str
    kernel_digest_sha256: str
    identity_sha256: str


class FitProfilePayload(TypedDict):
    schema_version: int
    artifact_kind: str
    model: str
    domain: str
    layers: int
    heads: int
    kernel_length: int
    source_r2_by_layer_head: list[list[float]]
    raw_kernels_by_layer_head: list[list[list[float]]]
    raw_kernel_sha256: str
    bins: list[FitProfileBinPayload]
    config_sha256: str
    sweep_sha256: str
    protocol_identity_sha256: str
    weights_tree_sha256: str
    tokenizer_tree_sha256: str
    materialized_domain_sha256: str
    fit_sequence_digest_sha256: str
    identity_sha256: str


class ShardPayload(TypedDict):
    schema_version: int
    model: str
    direction: str
    unit: str
    unit_index: str
    condition: str
    head_count: int
    sequence_nll: Float64Array
    sequence_digest_sha256: str
    source_domain_manifest_sha256: str
    target_domain_manifest_sha256: str
    weights_tree_sha256: str
    tokenizer_tree_sha256: str
    config_sha256: str
    sweep_sha256: str
    batch_manifest_sha256: str
    identity_sha256: str
    trial: NotRequired[int]
    control_kind: NotRequired[str]
    control_seed: NotRequired[int]
    mean_source_r2: NotRequired[float]
    source_fit_profile_sha256: NotRequired[str]
    target_fit_profile_sha256: NotRequired[str]
    baseline_identity_sha256: NotRequired[str]
    selected_heads: NotRequired[list[HeadRecordPayload]]
    selected_head_digest_sha256: NotRequired[str]
    selected_bin_digest_sha256: NotRequired[str]
    source_kernel_digest_sha256: NotRequired[str]
    target_kernel_digest_sha256: NotRequired[str]
    control_kernel_digest_sha256: NotRequired[str]


class SummaryRowPayload(TypedDict):
    stem: str
    mean_nll: float


class DoseResponseStatisticPayload(TypedDict):
    model: str
    direction: str
    rho: float
    p_one_sided: float
    p_two_sided_scipy: float
    permutation_count: NotRequired[int]
    permutation_exceedance_count: NotRequired[int]
    permutation_seed: NotRequired[int]
    permutation_seed_provenance: NotRequired[JsonObject]


class ContrastIntervalPayload(TypedDict):
    point: float
    ci_low: float
    ci_high: float


class ContrastPayload(TypedDict):
    model: str
    direction: str
    bin: int
    source_delta: ContrastIntervalPayload
    target_delta: ContrastIntervalPayload
    offset_delta: ContrastIntervalPayload
    norm_delta: ContrastIntervalPayload
    source_minus_target: ContrastIntervalPayload
    source_minus_offset: ContrastIntervalPayload
    source_minus_norm: ContrastIntervalPayload


class DepthRowPayload(TypedDict):
    model: str
    direction: str
    layer: int
    mean_source_r2: float
    grouped_loss_delta_per_head: float
    shard_identity_sha256: str
    baseline_identity_sha256: str


class DepthRelationshipPayload(TypedDict):
    model: str
    direction: str
    group_level: bool
    layers: int
    spearman_rho: float
    p_one_sided: float
    p_two_sided_scipy: float
    positive_null_permutation_count: NotRequired[int]
    permutation_seed: int
    permutation_seed_provenance: JsonObject
    layer_indices: list[int]
    mean_source_r2_inputs: list[float]
    grouped_loss_delta_per_head_inputs: list[float]
    depth_row_identity_sha256s: list[str]
    mean_grouped_loss_delta_per_head: float
    min_grouped_loss_delta_per_head: float
    max_grouped_loss_delta_per_head: float
    permutation_count: NotRequired[int]
    permutation_exceedance_count: NotRequired[int]


class _TokenizerFactory(Protocol):
    @staticmethod
    def from_pretrained(source: str, *, local_files_only: bool) -> TokenizerLike: ...


class _ModelOutput(Protocol):
    logits: torch.Tensor


class _CallableModel(Protocol):
    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
    ) -> _ModelOutput: ...


class _LoadedDatasetLike(Protocol):
    _fingerprint: object
    column_names: Sequence[object]

    def __iter__(self) -> Iterator[object]: ...


TOY_SEQUENCE_LENGTH = 512
TOY_LAYER_COUNT = 32
TOY_HEAD_COUNT = 32
TOY_DIRECTIONS = ("wikipedia_to_code", "code_to_wikipedia")
TRACKED_GIT_PATHS = (
    "PLAN.md",
    "rebuttal/src",
    "rebuttal/tests",
    "rebuttal/scripts",
    "rebuttal/configs",
    "rebuttal/README.md",
)
REQUIRED_PACKAGE_PINS: Mapping[str, str] = {
    "torch": "2.7.0",
    "transformers": "5.3.0",
    "datasets": "4.8.2",
    "numpy": "1.26.4",
    "scipy": "1.11.4",
    "pandas": "2.1.4",
    "pyarrow": "23.0.1",
    "pytest": "7.4.4",
    "ruff": "0.12.5",
    "basedpyright": "1.31.1",
}
PACKAGE_NAMES = tuple(REQUIRED_PACKAGE_PINS)
MAX_PROJECTED_BIN_HEAD_COUNT = 52
BENCHMARK_RUNTIME_FORMULA = (
    "projected_total_seconds = sum(component.measured_seconds * "
    "(component.projected_sequence_count / component.measured_sequence_count) "
    "for component in runtime_components)"
)
BENCHMARK_ARTIFACT_FORMULA = (
    "projected_artifact_bytes = immutable_materialized_bytes + "
    "ceil(future_artifact_bytes * future_margin_multiplier)"
)
LANE_HOURS_FORMULA = (
    "gpu0_hours = sum(benchmark_lane_hours[model] for model in placement_gpu0); "
    "gpu1_hours = sum(benchmark_lane_hours[model] for model in placement_gpu1); "
    "admitted = (gpu0_hours <= benchmark_max_lane_hours and gpu1_hours <= benchmark_max_lane_hours)"
)
DISK_HEADROOM_FORMULA = (
    "required_free_bytes = 2 * projected_complete_artifact_bytes + "
    "benchmark_min_free_space_gib * 1024**3; "
    "admitted = free_disk_bytes >= required_free_bytes"
)
_LEGACY_MODEL_HEADERS = (
    ("[models.llama-3.1-8b]", '[models."llama-3.1-8b"]'),
    ("[models.mistral-7b-v0.1]", '[models."mistral-7b-v0.1"]'),
    ("[models.olmo-2-7b]", '[models."olmo-2-7b"]'),
)


@dataclass(frozen=True)
class ToyRunSummary:
    run_id: str
    terminal_summary_path: Path
    shard_count: int
    statistic_count: int


@dataclass(frozen=True)
class BenchmarkReport:
    model_name: str
    fit_sequences: int
    eval_sequences: int
    measured_seconds: Mapping[str, float]
    measured_counts: Mapping[str, int]
    projected_total_seconds: float
    projected_artifact_bytes: int
    lane_hours: float
    runtime_components: JsonMapping
    runtime_formula: str
    projection_components: JsonMapping
    artifact_formula: str
    formula: str
    receipt_identity_sha256: str


@dataclass(frozen=True)
class LaneAdmissionReport:
    gpu0_hours: float
    gpu1_hours: float
    projected_artifact_bytes: int
    free_disk_bytes: int
    required_free_bytes: int
    required_disk_bytes: int
    lane_components: JsonMapping
    lane_formula: str
    disk_components: JsonMapping
    disk_formula: str
    receipt_identity_sha256: str


@dataclass(frozen=True)
class MaterializedBatch:
    manifest_payload: JsonObject
    domains: dict[str, MaterializedDomain]
    run_root: RunRoot


@dataclass(frozen=True)
class DomainSequences:
    fit_tokens: tuple[Int64Array, ...]
    eval_tokens: tuple[Int64Array, ...]


@dataclass(frozen=True)
class DomainFitSummary:
    mean_scores: Float64Array
    raw_kernels: Float64Array


@dataclass(frozen=True)
class FrozenFitProfile:
    payload: FitProfilePayload
    mean_scores: Float64Array
    raw_kernels: Float64Array
    bins: tuple[tuple[HeadIndex, ...], ...]
    bin_means: Float64Array


@dataclass(frozen=True)
class FitCaptureResult:
    layers: int
    heads: int
    seq_len: int
    per_sequence_r2: Float64Array
    mean_scores: Float64Array
    raw_kernel_sums: Float64Array
    sequence_count: int


@dataclass(frozen=True)
class LaunchExecutionContext:
    launch_receipt: JsonObject
    batch_receipt: JsonObject
    logical_device: str


def utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _torch_cuda_version() -> str | None:
    version = cast(TorchVersionNamespace | None, getattr(torch, "version", None))
    return None if version is None else version.cuda


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_json_primitive(value: object) -> TypeGuard[JsonPrimitive]:
    return value is None or isinstance(value, bool | int | float | str)


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _is_object_dict(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)


def _is_object_mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    return isinstance(value, Mapping)


def _is_non_string_object_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)


def _is_json_value(value: object) -> TypeGuard[JsonValue]:
    if _is_json_primitive(value):
        return True
    if _is_object_list(value):
        return all(_is_json_value(item) for item in value)
    if _is_object_dict(value):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False


def _json_value(value: object, *, label: str) -> JsonValue:
    if isinstance(value, Path):
        return str(value)
    if _is_json_value(value):
        return value
    if _is_non_string_object_sequence(value):
        return [_json_value(item, label=label) for item in value]
    if _is_object_mapping(value):
        payload: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RunnerError(f"{label} must use string object keys.")
            payload[key] = _json_value(item, label=label)
        return payload
    raise RunnerError(f"{label} contains non-JSON value {type(value).__name__}.")


def _json_object(value: object, *, label: str) -> JsonObject:
    payload = _json_value(value, label=label)
    if not isinstance(payload, dict):
        raise RunnerError(f"{label} must decode to a JSON object.")
    return payload


def _json_array(value: JsonValue, *, label: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise RunnerError(f"{label} must be a JSON array.")
    return value


def _json_string(value: JsonValue, *, label: str) -> str:
    if not isinstance(value, str):
        raise RunnerError(f"{label} must be a string.")
    return value


def _json_int(value: JsonValue, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise RunnerError(f"{label} must be an integer.")
    return value


def _json_object_item(payload: Mapping[str, object], key: str, *, label: str) -> JsonValue:
    if key not in payload:
        raise RunnerError(f"{label} is missing required key {key!r}.")
    return _json_value(payload[key], label=f"{label}.{key}")


def _json_object_member(payload: Mapping[str, object], key: str, *, label: str) -> JsonObject:
    return _json_object(_json_object_item(payload, key, label=label), label=f"{label}.{key}")


def _json_array_member(payload: Mapping[str, object], key: str, *, label: str) -> list[JsonValue]:
    return _json_array(_json_object_item(payload, key, label=label), label=f"{label}.{key}")


def _json_string_member(payload: Mapping[str, object], key: str, *, label: str) -> str:
    return _json_string(_json_object_item(payload, key, label=label), label=f"{label}.{key}")


def _json_int_member(payload: Mapping[str, object], key: str, *, label: str) -> int:
    return _json_int(_json_object_item(payload, key, label=label), label=f"{label}.{key}")


def _json_float_member(payload: Mapping[str, object], key: str, *, label: str) -> float:
    return _json_value_number(_json_object_item(payload, key, label=label), label=f"{label}.{key}")


def _json_string_mapping(value: JsonValue, *, label: str) -> dict[str, str]:
    payload = _json_object(value, label=label)
    return {key: _json_string(item, label=f"{label}.{key}") for key, item in payload.items()}


def _json_float_mapping(value: JsonValue, *, label: str) -> dict[str, float]:
    payload = _json_object(value, label=label)
    return {key: _json_value_number(item, label=f"{label}.{key}") for key, item in payload.items()}


def _json_int_mapping(value: JsonValue, *, label: str) -> dict[str, int]:
    payload = _json_object(value, label=label)
    return {key: _json_int(item, label=f"{label}.{key}") for key, item in payload.items()}


def _json_float_matrix(
    value: JsonValue, *, label: str, outer: int, inner: int
) -> list[list[float]]:
    rows = _json_array(value, label=label)
    if len(rows) != outer:
        raise RunnerError(f"{label} must contain exactly {outer} rows.")
    out: list[list[float]] = []
    for row_index, row in enumerate(rows):
        items = _json_array(row, label=f"{label}[{row_index}]")
        if len(items) != inner:
            raise RunnerError(f"{label}[{row_index}] must contain exactly {inner} values.")
        out.append(
            [
                _json_value_number(item, label=f"{label}[{row_index}][{index}]")
                for index, item in enumerate(items)
            ]
        )
    return out


def _json_value_number(value: JsonValue, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RunnerError(f"{label} must be numeric.")
    return float(value)


def _json_float_tensor3(
    value: JsonValue, *, label: str, dim0: int, dim1: int, dim2: int
) -> list[list[list[float]]]:
    planes = _json_array(value, label=label)
    if len(planes) != dim0:
        raise RunnerError(f"{label} must contain exactly {dim0} planes.")
    out: list[list[list[float]]] = []
    for plane_index, plane in enumerate(planes):
        rows = _json_array(plane, label=f"{label}[{plane_index}]")
        if len(rows) != dim1:
            raise RunnerError(f"{label}[{plane_index}] must contain exactly {dim1} rows.")
        plane_out: list[list[float]] = []
        for row_index, row in enumerate(rows):
            items = _json_array(row, label=f"{label}[{plane_index}][{row_index}]")
            if len(items) != dim2:
                raise RunnerError(
                    f"{label}[{plane_index}][{row_index}] must contain exactly {dim2} values."
                )
            plane_out.append(
                [
                    _json_value_number(item, label=f"{label}[{plane_index}][{row_index}][{index}]")
                    for index, item in enumerate(items)
                ]
            )
        out.append(plane_out)
    return out


def _head_record_payload(value: JsonValue, *, label: str) -> HeadRecordPayload:
    payload = _json_object(value, label=label)
    return {
        "layer": _json_int_member(payload, "layer", label=label),
        "head": _json_int_member(payload, "head", label=label),
    }


def _domain_entry_payload(value: JsonValue, *, label: str) -> DomainEntryPayload:
    payload = _json_object(value, label=label)
    entry: DomainEntryPayload = {
        "model": _json_string_member(payload, "model", label=label),
        "domain": _json_string_member(payload, "domain", label=label),
        "tokenizer_tree_sha256": _json_string_member(payload, "tokenizer_tree_sha256", label=label),
        "materialized_domain_sha256": _json_string_member(
            payload, "materialized_domain_sha256", label=label
        ),
        "dataset_repository": _json_string_member(payload, "dataset_repository", label=label),
        "dataset_revision": _json_string_member(payload, "dataset_revision", label=label),
        "dataset_config": _json_string_member(payload, "dataset_config", label=label),
        "dataset_split": _json_string_member(payload, "dataset_split", label=label),
        "dataset_field": _json_string_member(payload, "dataset_field", label=label),
        "dataset_fingerprint": _json_string_member(payload, "dataset_fingerprint", label=label),
        "fit_token_manifest_sha256": _json_string_member(
            payload, "fit_token_manifest_sha256", label=label
        ),
        "eval_token_manifest_sha256": _json_string_member(
            payload, "eval_token_manifest_sha256", label=label
        ),
        "fit_sequence_digest_sha256": _json_string_member(
            payload, "fit_sequence_digest_sha256", label=label
        ),
        "eval_sequence_digest_sha256": _json_string_member(
            payload, "eval_sequence_digest_sha256", label=label
        ),
    }
    if "identity_sha256" in payload:
        entry["identity_sha256"] = _json_string_member(payload, "identity_sha256", label=label)
    return entry


def _batch_manifest_payload(value: object, *, label: str) -> BatchManifestPayload:
    payload = _json_object(value, label=label)
    domains_payload = _json_object_member(payload, "domains", label=label)
    domain_entries_payload = _json_object_member(payload, "domain_entries", label=label)
    return {
        "schema_version": _json_int_member(payload, "schema_version", label=label),
        "materialized_at_utc": _json_string_member(payload, "materialized_at_utc", label=label),
        "command": [
            _json_string(item, label=f"{label}.command[{index}]")
            for index, item in enumerate(_json_array_member(payload, "command", label=label))
        ],
        "config_sha256": _json_string_member(payload, "config_sha256", label=label),
        "sweep_sha256": _json_string_member(payload, "sweep_sha256", label=label),
        "domains": {
            key: _json_string(item, label=f"{label}.domains.{key}")
            for key, item in domains_payload.items()
        },
        "domain_entries": {
            key: _domain_entry_payload(item, label=f"{label}.domain_entries.{key}")
            for key, item in domain_entries_payload.items()
        },
        "tokenizers": _json_string_mapping(
            _json_object_item(payload, "tokenizers", label=label), label=f"{label}.tokenizers"
        ),
        "models": _json_string_mapping(
            _json_object_item(payload, "models", label=label), label=f"{label}.models"
        ),
        "git_commit_sha": _json_string_member(payload, "git_commit_sha", label=label),
        "git_tracked_diff_sha256": _json_string_member(
            payload, "git_tracked_diff_sha256", label=label
        ),
        "git_relevant_content_sha256": _json_string_member(
            payload, "git_relevant_content_sha256", label=label
        ),
        "git_untracked_sha256": _json_string_member(payload, "git_untracked_sha256", label=label),
        "runtime": _json_object_member(payload, "runtime", label=label),
        "runtime_capture": _json_object_member(payload, "runtime_capture", label=label),
        "provenance": _json_object_member(payload, "provenance", label=label),
        "environment": _json_string_mapping(
            _json_object_item(payload, "environment", label=label), label=f"{label}.environment"
        ),
        "identity_sha256": _json_string_member(payload, "identity_sha256", label=label),
    }


def _fit_profile_bin_payload_from_json(value: JsonValue, *, label: str) -> FitProfileBinPayload:
    payload = _json_object(value, label=label)
    selected_heads = [
        _head_record_payload(item, label=f"{label}.selected_heads[{index}]")
        for index, item in enumerate(_json_array_member(payload, "selected_heads", label=label))
    ]
    return {
        "bin_index": _json_int_member(payload, "bin_index", label=label),
        "head_count": _json_int_member(payload, "head_count", label=label),
        "mean_source_r2": _json_value_number(
            _json_object_item(payload, "mean_source_r2", label=label),
            label=f"{label}.mean_source_r2",
        ),
        "selected_heads": selected_heads,
        "selected_head_digest_sha256": _json_string_member(
            payload, "selected_head_digest_sha256", label=label
        ),
        "kernel_digest_sha256": _json_string_member(payload, "kernel_digest_sha256", label=label),
        "identity_sha256": _json_string_member(payload, "identity_sha256", label=label),
    }


def _fit_profile_payload(value: object, *, label: str) -> FitProfilePayload:
    payload = _json_object(value, label=label)
    bins = [
        _fit_profile_bin_payload_from_json(item, label=f"{label}.bins[{index}]")
        for index, item in enumerate(_json_array_member(payload, "bins", label=label))
    ]
    return {
        "schema_version": _json_int_member(payload, "schema_version", label=label),
        "artifact_kind": _json_string_member(payload, "artifact_kind", label=label),
        "model": _json_string_member(payload, "model", label=label),
        "domain": _json_string_member(payload, "domain", label=label),
        "layers": _json_int_member(payload, "layers", label=label),
        "heads": _json_int_member(payload, "heads", label=label),
        "kernel_length": _json_int_member(payload, "kernel_length", label=label),
        "source_r2_by_layer_head": _json_float_matrix(
            _json_object_item(payload, "source_r2_by_layer_head", label=label),
            label=f"{label}.source_r2_by_layer_head",
            outer=32,
            inner=32,
        ),
        "raw_kernels_by_layer_head": _json_float_tensor3(
            _json_object_item(payload, "raw_kernels_by_layer_head", label=label),
            label=f"{label}.raw_kernels_by_layer_head",
            dim0=32,
            dim1=32,
            dim2=512,
        ),
        "raw_kernel_sha256": _json_string_member(payload, "raw_kernel_sha256", label=label),
        "bins": bins,
        "config_sha256": _json_string_member(payload, "config_sha256", label=label),
        "sweep_sha256": _json_string_member(payload, "sweep_sha256", label=label),
        "protocol_identity_sha256": _json_string_member(
            payload, "protocol_identity_sha256", label=label
        ),
        "weights_tree_sha256": _json_string_member(payload, "weights_tree_sha256", label=label),
        "tokenizer_tree_sha256": _json_string_member(payload, "tokenizer_tree_sha256", label=label),
        "materialized_domain_sha256": _json_string_member(
            payload, "materialized_domain_sha256", label=label
        ),
        "fit_sequence_digest_sha256": _json_string_member(
            payload, "fit_sequence_digest_sha256", label=label
        ),
        "identity_sha256": _json_string_member(payload, "identity_sha256", label=label),
    }


def _shard_payload(value: object, *, label: str) -> ShardPayload:
    payload = _json_object(value, label=label)
    typed: ShardPayload = {
        "schema_version": _json_int_member(payload, "schema_version", label=label),
        "model": _json_string_member(payload, "model", label=label),
        "direction": _json_string_member(payload, "direction", label=label),
        "unit": _json_string_member(payload, "unit", label=label),
        "unit_index": _json_string_member(payload, "unit_index", label=label),
        "condition": _json_string_member(payload, "condition", label=label),
        "head_count": _json_int_member(payload, "head_count", label=label),
        "sequence_nll": np.asarray(
            [
                _json_value_number(item, label=f"{label}.sequence_nll[{index}]")
                for index, item in enumerate(
                    _json_array_member(payload, "sequence_nll", label=label)
                )
            ],
            dtype=np.float64,
        ),
        "sequence_digest_sha256": _json_string_member(
            payload, "sequence_digest_sha256", label=label
        ),
        "source_domain_manifest_sha256": _json_string_member(
            payload, "source_domain_manifest_sha256", label=label
        ),
        "target_domain_manifest_sha256": _json_string_member(
            payload, "target_domain_manifest_sha256", label=label
        ),
        "weights_tree_sha256": _json_string_member(payload, "weights_tree_sha256", label=label),
        "tokenizer_tree_sha256": _json_string_member(payload, "tokenizer_tree_sha256", label=label),
        "config_sha256": _json_string_member(payload, "config_sha256", label=label),
        "sweep_sha256": _json_string_member(payload, "sweep_sha256", label=label),
        "batch_manifest_sha256": _json_string_member(payload, "batch_manifest_sha256", label=label),
        "identity_sha256": _json_string_member(payload, "identity_sha256", label=label),
    }
    if "trial" in payload:
        typed["trial"] = _json_int_member(payload, "trial", label=label)
    if "control_kind" in payload:
        typed["control_kind"] = _json_string_member(payload, "control_kind", label=label)
    if "control_seed" in payload:
        typed["control_seed"] = _json_int_member(payload, "control_seed", label=label)
    if "mean_source_r2" in payload:
        typed["mean_source_r2"] = _json_value_number(
            _json_object_item(payload, "mean_source_r2", label=label),
            label=f"{label}.mean_source_r2",
        )
    for key in (
        "source_fit_profile_sha256",
        "target_fit_profile_sha256",
        "baseline_identity_sha256",
        "selected_head_digest_sha256",
        "selected_bin_digest_sha256",
        "source_kernel_digest_sha256",
        "target_kernel_digest_sha256",
        "control_kernel_digest_sha256",
    ):
        if key in payload:
            typed[key] = _json_string_member(payload, key, label=label)
    if "selected_heads" in payload:
        typed["selected_heads"] = [
            _head_record_payload(item, label=f"{label}.selected_heads[{index}]")
            for index, item in enumerate(_json_array_member(payload, "selected_heads", label=label))
        ]
    return typed


def _require_shard_str(payload: ShardPayload, key: str, *, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise RunnerError(f"{label} is missing required string field {key!r}.")
    return value


def _require_shard_int(payload: ShardPayload, key: str, *, label: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise RunnerError(f"{label} is missing required integer field {key!r}.")
    return value


def _require_shard_float(payload: ShardPayload, key: str, *, label: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RunnerError(f"{label} is missing required numeric field {key!r}.")
    return float(value)


def _require_shard_heads(payload: ShardPayload, *, label: str) -> list[HeadRecordPayload]:
    heads = payload.get("selected_heads")
    if not isinstance(heads, list):
        raise RunnerError(f"{label} is missing required selected_heads.")
    return heads


def _read_json_object(path: Path, *, label: str) -> JsonObject:
    try:
        raw = json.loads(path.read_text(encoding="ascii"))
    except Exception as exc:
        raise RunnerError(f"{label} could not be parsed from {path.name}.") from exc
    return _json_object(raw, label=label)


def _read_verified_payload(path: Path, *, label: str) -> JsonObject:
    try:
        verified = verify_payload_identity(_read_json_object(path, label=label), label=path.name)
    except Exception as exc:
        raise RunnerError(f"{label} verification failed for {path.name}.") from exc
    return _json_object(verified, label=label)


def _normalized_json_object(payload: object, *, label: str) -> JsonObject:
    return _json_object(ensure_payload_identity(payload), label=label)


def _as_json_object(payload: object, *, label: str) -> JsonObject:
    return _json_object(payload, label=label)


def _load_existing_or_write_payload(
    path: Path, payload: object, *, label: str
) -> tuple[JsonObject, bool]:
    return load_existing_or_write(path, _normalized_json_object(payload, label=label))


def _token_sequence_record(value: JsonValue, *, label: str) -> TokenSequenceRecord:
    payload = _json_object(value, label=label)

    def int_list(name: str) -> list[int]:
        return [
            _json_int(item, label=f"{label}.{name}[]")
            for item in _json_array(payload[name], label=f"{label}.{name}")
        ]

    return {
        "document_id": _json_string(payload["document_id"], label=f"{label}.document_id"),
        "partition": _json_string(payload["partition"], label=f"{label}.partition"),
        "chunk_index": _json_int(payload["chunk_index"], label=f"{label}.chunk_index"),
        "source_rows": int_list("source_rows"),
        "token_count": _json_int(payload["token_count"], label=f"{label}.token_count"),
        "token_shape": int_list("token_shape"),
        "token_sha256": _json_string(payload["token_sha256"], label=f"{label}.token_sha256"),
        "document_content_sha256": _json_string(
            payload["document_content_sha256"], label=f"{label}.document_content_sha256"
        ),
        "document_assignment_sha256": _json_string(
            payload["document_assignment_sha256"], label=f"{label}.document_assignment_sha256"
        ),
        "tokens": int_list("tokens"),
    }


def _token_manifest_payload(value: object, *, label: str) -> TokenManifestPayload:
    payload = _json_object(value, label=label)
    sequences = [
        _token_sequence_record(item, label=f"{label}.sequences[{index}]")
        for index, item in enumerate(_json_array(payload["sequences"], label=f"{label}.sequences"))
    ]
    return {
        "schema_version": _json_int(payload["schema_version"], label=f"{label}.schema_version"),
        "model": _json_string(payload["model"], label=f"{label}.model"),
        "domain": _json_string(payload["domain"], label=f"{label}.domain"),
        "partition": _json_string(payload["partition"], label=f"{label}.partition"),
        "count": _json_int(payload["count"], label=f"{label}.count"),
        "config_sha256": _json_string(payload["config_sha256"], label=f"{label}.config_sha256"),
        "sweep_sha256": _json_string(payload["sweep_sha256"], label=f"{label}.sweep_sha256"),
        "materialized_domain_sha256": _json_string(
            payload["materialized_domain_sha256"], label=f"{label}.materialized_domain_sha256"
        ),
        "sequence_digest_sha256": _json_string(
            payload["sequence_digest_sha256"], label=f"{label}.sequence_digest_sha256"
        ),
        "sequences": sequences,
        "identity_sha256": _json_string(
            payload["identity_sha256"], label=f"{label}.identity_sha256"
        ),
    }


def _resolve_executable(command: Sequence[str]) -> str:
    argv0 = str(command[0])
    if "/" in argv0:
        return str(Path(argv0).resolve())
    return shutil.which(argv0) or argv0


def _validate_loaded_dataset(value: object) -> _LoadedDatasetLike:
    if not hasattr(value, "_fingerprint"):
        raise RunnerError("Loaded dataset must expose a _fingerprint attribute.")
    if not hasattr(value, "column_names"):
        raise RunnerError("Loaded dataset must expose a column_names attribute.")
    if not callable(getattr(value, "__iter__", None)):
        raise RunnerError("Loaded dataset must be iterable.")
    return cast(_LoadedDatasetLike, value)


def _validated_dataset_loader(loader: Callable[..., object]) -> DatasetLoaderFn:
    def _adapter(*args: object, **kwargs: object) -> _LoadedDatasetLike:
        return _validate_loaded_dataset(loader(*args, **kwargs))

    return _adapter


def _default_load_dataset(*args: object, **kwargs: object) -> _LoadedDatasetLike:
    import datasets

    load_dataset = getattr(datasets, "load_dataset", None)
    if not callable(load_dataset):
        raise RunnerError("datasets.load_dataset is unavailable.")
    loader = _validated_dataset_loader(load_dataset)
    return loader(*args, **kwargs)


def _load_transformers_tokenizer_factory() -> _TokenizerFactory:
    from transformers import AutoTokenizer

    return cast(_TokenizerFactory, cast(object, AutoTokenizer))


def _call_model(
    model: object, *, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    if not callable(model):
        raise RunnerError("Loaded model must be callable.")
    output = cast(_CallableModel, model)(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=False
    )
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor):
        raise RunnerError("Loaded model forward pass must return tensor logits.")
    return logits[0].detach().to(torch.float32)


def _fit_capture_attention_registries(
    spec: object,
    capture_attention_forward: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
) -> tuple[tuple[ValidationAttentionRegistry, object], ...]:
    registries: list[tuple[ValidationAttentionRegistry, object]] = []
    seen: set[int] = set()
    family_adapter = getattr(spec, "family_adapter", None)
    attention_modules = getattr(spec, "attention_modules", ())

    def add_registry(candidate: object, installed_value: object) -> None:
        registry = _coerce_validation_attention_registry(candidate)
        if registry is None:
            return
        registry_id = id(registry.registry)
        if registry_id in seen:
            return
        seen.add(registry_id)
        registries.append((registry, installed_value))

    if family_adapter is not None:
        add_registry(
            getattr(family_adapter, "attention_interface", None),
            capture_attention_forward,
        )
        mask_attention_interface = getattr(family_adapter, "mask_attention_interface", None)
        mask_registry = _coerce_validation_attention_registry(mask_attention_interface)
        if mask_attention_interface is None or mask_registry is None:
            raise RunnerError(
                "Fit capture requires a mutable mask attention-interface registry with "
                "register/pop or mapping semantics."
            )
        mask_eager_handler = mask_registry.get("eager")
        if mask_eager_handler is None:
            raise RunnerError("Fit capture requires mask_registry.get('eager').")
        add_registry(mask_registry.registry, mask_eager_handler)
    for module in attention_modules:
        add_registry(getattr(module, "attention_interface", None), capture_attention_forward)
    if not registries:
        raise RunnerError(
            "Fit capture requires an attention-interface registry with register/pop or mapping "
            "semantics."
        )
    return tuple(registries)


def _run_runtime_command(
    command: Sequence[str],
    *,
    command_runner: RuntimeCommandRunner,
    label: str,
) -> RuntimeCommandPayload:
    proc = command_runner(tuple(command))
    stdout = proc.stdout
    stderr = proc.stderr
    if "\x00" in stdout or "\x00" in stderr:
        raise RunnerError(f"{label} emitted NUL bytes.")
    payload: RuntimeCommandPayload = {
        "resolved_executable": _resolve_executable(command),
        "command": list(command),
        "returncode": int(proc.returncode),
        "stdout_text": stdout,
        "stdout_sha256": _text_sha256(stdout),
        "stderr_text": stderr,
        "stderr_sha256": _text_sha256(stderr),
    }
    if proc.returncode != 0:
        raise RunnerError(f"{label} failed with exit code {proc.returncode}.")
    return payload


def _capture_package_freeze(
    command_runner: RuntimeCommandRunner,
) -> PackageFreezePayload:
    command = (sys.executable, "-m", "pip", "freeze", "--all")
    runtime_payload = _run_runtime_command(
        command, command_runner=command_runner, label="pip freeze --all"
    )
    payload: PackageFreezePayload = {
        **runtime_payload,
        "output_lines": [],
        "line_count": 0,
        "required_exact_pins": {},
        "observed_required_pins": {},
    }
    raw_lines = payload["stdout_text"].splitlines()
    lines = [line for line in raw_lines if line]
    if not any(line.strip() for line in lines):
        raise RunnerError("pip freeze --all returned no meaningful package lines.")
    payload["output_lines"] = lines
    payload["line_count"] = len(lines)
    payload["required_exact_pins"] = dict(REQUIRED_PACKAGE_PINS)
    payload["observed_required_pins"] = _selected_package_versions_from_freeze(
        lines, REQUIRED_PACKAGE_PINS
    )
    return payload


def _selected_package_versions_from_freeze(
    lines: Sequence[str], required_package_pins: Mapping[str, str]
) -> dict[str, str]:
    parsed: dict[str, tuple[str, str]] = {}
    relevant_names = {package_name.lower(): package_name for package_name in required_package_pins}
    for line in lines:
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith("#")
            or stripped.startswith("-e ")
            or stripped.startswith("--editable ")
        ):
            continue
        match = re.match(
            r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)\s*(?P<sep>==|@)\s*(?P<value>\S.*)$", stripped
        )
        if match is None:
            continue
        normalized_name = match.group("name").strip().lower()
        if normalized_name not in relevant_names:
            continue
        if normalized_name in parsed:
            raise RunnerError(f"Duplicate pip freeze entry for {normalized_name}.")
        parsed[normalized_name] = (match.group("sep"), match.group("value").strip())
    selected: dict[str, str] = {}
    missing: list[str] = []
    for package_name, expected_version in required_package_pins.items():
        normalized_name = package_name.lower()
        if normalized_name not in parsed:
            missing.append(package_name)
            continue
        separator, observed_value = parsed[normalized_name]
        if separator != "==":
            raise RunnerError(
                f"pip freeze --all must use an exact '==' pin for {package_name}; "
                "direct references are not allowed."
            )
        if observed_value != expected_version:
            raise RunnerError(
                f"pip freeze --all version mismatch for {package_name}: "
                f"expected {expected_version}, found {observed_value}."
            )
        selected[package_name] = observed_value
    if missing:
        raise RunnerError(f"pip freeze --all is missing required packages: {', '.join(missing)}.")
    return selected


def _capture_gpu_inventory(
    command_runner: RuntimeCommandRunner,
) -> GpuInventoryPayload:
    command = (
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    )
    runtime_payload = _run_runtime_command(
        command, command_runner=command_runner, label="nvidia-smi inventory"
    )
    payload: GpuInventoryPayload = {
        **runtime_payload,
        "gpus": [],
        "line_count": 0,
        "driver_version": "",
    }
    rows = [line.strip() for line in payload["stdout_text"].splitlines() if line.strip()]
    if not rows:
        raise RunnerError("nvidia-smi inventory returned no GPU rows.")
    gpus: list[GpuInventoryEntry] = []
    seen_indexes: set[int] = set()
    for row in rows:
        parts = [part.strip() for part in row.split(",")]
        if len(parts) != 6:
            raise RunnerError(f"Malformed nvidia-smi inventory row: {row!r}")
        try:
            physical_index = int(parts[0])
            total_memory_mib = int(parts[3])
        except ValueError as exc:
            raise RunnerError(f"Malformed nvidia-smi numeric field: {row!r}") from exc
        if physical_index in seen_indexes:
            raise RunnerError(f"Duplicate nvidia-smi GPU index {physical_index}.")
        seen_indexes.add(physical_index)
        compute_capability = parts[4]
        if not re.fullmatch(r"\d+\.\d+", compute_capability):
            raise RunnerError(f"Malformed nvidia-smi compute capability: {compute_capability!r}")
        driver_version = parts[5]
        if not driver_version:
            raise RunnerError("nvidia-smi inventory returned an empty driver version.")
        gpus.append(
            {
                "physical_index": physical_index,
                "uuid": parts[1],
                "name": parts[2],
                "total_memory_mib": total_memory_mib,
                "compute_capability": compute_capability,
                "driver_version": driver_version,
            }
        )
    gpus.sort(key=lambda entry: int(entry["physical_index"]))
    payload["gpus"] = gpus
    payload["line_count"] = len(rows)
    payload["driver_version"] = gpus[0]["driver_version"]
    return payload


def _capture_runtime_provenance(
    command_runner: RuntimeCommandRunner,
) -> tuple[RuntimeBinding, JsonObject]:
    package_freeze = _capture_package_freeze(command_runner)
    selected_package_versions = dict(package_freeze["observed_required_pins"])
    gpu_inventory = _capture_gpu_inventory(command_runner)
    runtime = RuntimeBinding(
        python_version=sys.version,
        platform=platform.platform(),
        package_versions=selected_package_versions,
        cuda_version=_torch_cuda_version(),
        driver_version=str(gpu_inventory["driver_version"]),
    )
    torch_cuda: JsonObject = {
        "torch_version": str(torch.__version__),
        "torch_cuda_version": _torch_cuda_version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "visible_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    return runtime, {
        "python_executable": str(Path(sys.executable).resolve()),
        "package_freeze": _json_object(package_freeze, label="runtime_capture.package_freeze"),
        "torch_cuda": torch_cuda,
        "gpu_inventory": _json_object(gpu_inventory, label="runtime_capture.gpu_inventory"),
    }


def _validate_existing_package_freeze_payload(payload: JsonMapping) -> None:
    stdout_text = payload.get("stdout_text")
    stderr_text = payload.get("stderr_text")
    if not isinstance(stdout_text, str) or not isinstance(stderr_text, str):
        raise RunnerError("Batch manifest runtime capture package freeze is not valid UTF-8 text.")
    if "\x00" in stdout_text or "\x00" in stderr_text:
        raise RunnerError("Batch manifest runtime capture package freeze contains NUL bytes.")
    if payload.get("stdout_sha256") != _text_sha256(stdout_text):
        raise RunnerError("Batch manifest runtime capture package freeze stdout digest drifted.")
    if payload.get("stderr_sha256") != _text_sha256(stderr_text):
        raise RunnerError("Batch manifest runtime capture package freeze stderr digest drifted.")
    expected_lines = [line for line in stdout_text.splitlines() if line]
    if not any(line.strip() for line in expected_lines):
        raise RunnerError("Batch manifest runtime capture package freeze lost meaningful content.")
    if payload.get("output_lines") != expected_lines:
        raise RunnerError("Batch manifest runtime capture package freeze output lines drifted.")
    if payload.get("line_count") != len(expected_lines):
        raise RunnerError("Batch manifest runtime capture package freeze line count drifted.")
    if payload.get("required_exact_pins") != dict(REQUIRED_PACKAGE_PINS):
        raise RunnerError("Batch manifest runtime capture package freeze required pin map drifted.")
    observed_required_pins = _selected_package_versions_from_freeze(
        expected_lines, REQUIRED_PACKAGE_PINS
    )
    if payload.get("observed_required_pins") != observed_required_pins:
        raise RunnerError("Batch manifest runtime capture package freeze observed pin map drifted.")


def _runtime_capture_matches_current(
    *,
    stage: str,
    expected_runtime_capture: JsonObject,
    current_runtime_capture: JsonObject,
) -> bool:
    if stage != "run-model":
        return expected_runtime_capture == current_runtime_capture
    if set(expected_runtime_capture) != set(current_runtime_capture):
        return False
    for key, expected_value in expected_runtime_capture.items():
        if key == "torch_cuda":
            continue
        if current_runtime_capture.get(key) != expected_value:
            return False
    expected_torch_cuda = _json_object_member(
        expected_runtime_capture,
        "torch_cuda",
        label="batch_manifest.runtime_capture",
    )
    current_torch_cuda = _json_object_member(
        current_runtime_capture,
        "torch_cuda",
        label="current_runtime_capture",
    )
    if set(expected_torch_cuda) != set(current_torch_cuda):
        return False
    for key, expected_value in expected_torch_cuda.items():
        if key == "visible_device_count":
            continue
        if current_torch_cuda.get(key) != expected_value:
            return False
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible_devices not in {"0", "1"}:
        return False
    return (
        _json_int_member(
            expected_torch_cuda,
            "visible_device_count",
            label="batch_manifest.runtime_capture.torch_cuda",
        )
        == 2
        and _json_int_member(
            current_torch_cuda,
            "visible_device_count",
            label="current_runtime_capture.torch_cuda",
        )
        == 1
    )


def _expected_physical_gpu_for_model(sweep: SweepConfig, *, model_name: str) -> int:
    if model_name in sweep.placement_gpu0:
        return 0
    if model_name in sweep.placement_gpu1:
        return 1
    raise RunnerError(f"Model {model_name} is not assigned to a fixed physical GPU lane.")


def _require_preflight_model_visibility(sweep: SweepConfig, *, model_name: str) -> str:
    expected_physical_gpu = _expected_physical_gpu_for_model(sweep, model_name=model_name)
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible_devices != str(expected_physical_gpu) or "," in cuda_visible_devices:
        raise RunnerError(
            f"{model_name} requires CUDA_VISIBLE_DEVICES={expected_physical_gpu} "
            "so the fixed logical device remains cuda:0."
        )
    return "cuda:0"


def _resolve_cli_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate
    package_root = Path(__file__).resolve().parents[2]
    if candidate.parts and candidate.parts[0] == "rebuttal":
        trimmed = Path(*candidate.parts[1:])
        trimmed_candidate = package_root / trimmed
        if trimmed_candidate.exists():
            return trimmed_candidate
    package_candidate = package_root / candidate
    if package_candidate.exists():
        return package_candidate
    return candidate


def load_configs(
    base_path: str | Path, sweep_path: str | Path
) -> tuple[ResolvedConfig, SweepConfig]:
    return (
        _load_base_config_with_legacy_model_header_normalization(_resolve_cli_path(base_path)),
        load_sweep_config(_resolve_cli_path(sweep_path)),
    )


def _load_base_config_with_legacy_model_header_normalization(path: str | Path) -> ResolvedConfig:
    source_path = Path(path).resolve()
    raw_text = source_path.read_text(encoding="utf-8")
    payload = _json_object(tomllib.loads(raw_text), label=str(source_path))
    paths_payload = payload.get("paths")
    if not isinstance(paths_payload, Mapping):
        raise RunnerError(f"Base config {source_path} is missing [paths].")
    raw_project_root = paths_payload.get("project_root")
    if not isinstance(raw_project_root, str) or not raw_project_root.strip():
        raise RunnerError(f"Base config {source_path} is missing paths.project_root.")
    expected_project_root = (source_path.parent / raw_project_root).resolve()
    original_lines = raw_text.splitlines(keepends=True)
    normalized_lines = list(original_lines)
    for legacy_header, quoted_header in _LEGACY_MODEL_HEADERS:
        legacy_indexes = [
            index for index, line in enumerate(normalized_lines) if line.strip() == legacy_header
        ]
        quoted_indexes = [
            index for index, line in enumerate(normalized_lines) if line.strip() == quoted_header
        ]
        if len(legacy_indexes) + len(quoted_indexes) != 1:
            raise RunnerError(f"Base config legacy header drifted for {legacy_header}.")
        if quoted_indexes:
            continue
        index = legacy_indexes[0]
        line = normalized_lines[index]
        newline = "\n" if line.endswith("\n") else ""
        normalized_lines[index] = f"{quoted_header}{newline}"
    normalized_text = "".join(normalized_lines)
    if normalized_text == raw_text:
        return load_base_config(source_path)
    with tempfile.TemporaryDirectory(
        prefix="si-rebuttal-base-", dir=tempfile.gettempdir()
    ) as tmpdir:
        tmp_path = Path(tmpdir) / source_path.name
        rewritten_project_root = os.path.relpath(expected_project_root, tmp_path.parent)
        if Path(rewritten_project_root).is_absolute():
            raise RunnerError("Normalized base config project_root must remain relative.")
        normalized_lines = _rewrite_project_root_for_temp_file(
            normalized_text, rewritten_project_root
        )
        tmp_path.write_text(normalized_lines, encoding="utf-8")
        resolved = load_base_config(tmp_path)
    if resolved.paths.project_root != expected_project_root:
        raise RunnerError("Normalized base config resolved an unexpected project root.")
    return replace(resolved, source_path=source_path)


def _rewrite_project_root_for_temp_file(toml_text: str, rewritten_project_root: str) -> str:
    lines = toml_text.splitlines(keepends=True)
    in_paths = False
    rewritten = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_paths = stripped == "[paths]"
            continue
        if in_paths and stripped.startswith("project_root"):
            newline = "\n" if line.endswith("\n") else ""
            lines[index] = f'project_root = "{rewritten_project_root}"{newline}'
            rewritten += 1
    if rewritten != 1:
        raise RunnerError("Base config project_root line drifted during normalization.")
    return "".join(lines)


def _manifest_paths(run_root: RunRoot) -> dict[str, Path]:
    return {
        "resolved_config": run_root.manifests_dir / "resolved-config.json",
        "sweep": run_root.manifests_dir / "sweep.json",
        "materialized": run_root.manifests_dir / "materialized-data.json",
        "batch": run_root.manifests_dir / "batch-manifest.json",
    }


def _receipt_path(run_root: RunRoot, stem: str) -> Path:
    return run_root.receipts_dir / f"{stem}.json"


def _summary_v1_path(run_root: RunRoot) -> Path:
    return run_root.summaries_dir / "batch-terminal-summary.json"


def _summary_v2_path(run_root: RunRoot) -> Path:
    return run_root.summaries_dir / "batch-terminal-summary.v2.json"


def _token_manifest_path(
    run_root: RunRoot, *, model_name: str, domain_name: str, partition: str
) -> Path:
    return run_root.manifests_dir / "tokens" / f"{model_name}.{domain_name}.{partition}.json"


def _fit_profile_path(run_root: RunRoot, *, model_name: str, domain_name: str) -> Path:
    return run_root.manifests_dir / "fit-profiles" / f"{model_name}.{domain_name}.json"


def _domain_entry_name(model_name: str, domain_name: str) -> str:
    return f"{model_name}:{domain_name}"


def _batch_domain_entry(
    *,
    model_name: str,
    domain_name: str,
    materialized: MaterializedDomain,
    tokenizer_tree_sha256: str,
    fit_token_manifest_sha256: str,
    eval_token_manifest_sha256: str,
    fit_sequence_digest_sha256: str,
    eval_sequence_digest_sha256: str,
) -> DomainEntryPayload:
    return {
        "model": model_name,
        "domain": domain_name,
        "tokenizer_tree_sha256": tokenizer_tree_sha256,
        "materialized_domain_sha256": materialized.manifest_sha256,
        "dataset_repository": materialized.dataset_repository,
        "dataset_revision": materialized.dataset_revision,
        "dataset_config": materialized.dataset_config,
        "dataset_split": materialized.dataset_split,
        "dataset_field": materialized.dataset_field,
        "dataset_fingerprint": materialized.dataset_fingerprint,
        "fit_token_manifest_sha256": fit_token_manifest_sha256,
        "eval_token_manifest_sha256": eval_token_manifest_sha256,
        "fit_sequence_digest_sha256": fit_sequence_digest_sha256,
        "eval_sequence_digest_sha256": eval_sequence_digest_sha256,
    }


def _load_tokenizer(
    model_config: ModelConfig, tokenizer_loader: TokenizerLoader | None = None
) -> TokenizerLike:
    if tokenizer_loader is None:
        tokenizer_factory = _load_transformers_tokenizer_factory()

        def default_tokenizer_loader(source: str) -> TokenizerLike:
            tokenizer = tokenizer_factory.from_pretrained(source, local_files_only=True)
            encode = getattr(tokenizer, "encode", None)
            if not callable(encode):
                raise RunnerError("Loaded tokenizer must expose an encode(text, ...) method.")
            return tokenizer

        tokenizer_loader = default_tokenizer_loader

    return tokenizer_loader(str(model_config.tokenizer_path))


def _materialize_payload_value(domain: MaterializedDomain) -> JsonObject:
    return _json_object(ensure_payload_identity(domain), label="materialized_domain")


def _collect_model_bindings(resolved_config: ResolvedConfig) -> dict[str, FrozenModelBinding]:
    return {
        name: freeze_model_binding(model_config)
        for name, model_config in resolved_config.models.items()
    }


def _partition_inventory(
    materialized: MaterializedDomain, *, partition: str
) -> Mapping[str, DocumentRecord]:
    documents = materialized.fit_documents if partition == "fit" else materialized.eval_documents
    inventory: dict[str, DocumentRecord] = {}
    for document in documents:
        existing = inventory.get(document.document_id)
        if existing is not None and existing != document:
            raise RunnerError(
                f"{materialized.domain} {partition} provenance inventory drifted for document ID "
                f"{document.document_id!r}."
            )
        inventory[document.document_id] = document
    return inventory


def _resolve_selected_documents(
    materialized: MaterializedDomain, *, partition: str
) -> tuple[_ResolvedSelectedDocument, ...]:
    inventory = _partition_inventory(materialized, partition=partition)
    chunks = materialized.fit_chunks if partition == "fit" else materialized.eval_chunks
    resolved: list[_ResolvedSelectedDocument] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.document_id in seen:
            continue
        document = inventory.get(chunk.document_id)
        if document is None:
            raise RunnerError(
                f"{materialized.domain} {partition} selected document ID {chunk.document_id!r} "
                "is missing from the full provenance inventory."
            )
        resolved.append(
            _ResolvedSelectedDocument(
                document_id=document.document_id,
                row_indices=document.row_indices,
                content_sha256=document.content_sha256,
            )
        )
        seen.add(chunk.document_id)
    return tuple(resolved)


def _accumulate_selected_union(
    union: dict[str, _ResolvedSelectedDocument],
    resolved: Sequence[_ResolvedSelectedDocument],
    *,
    corpus: str,
    partition: str,
) -> None:
    for document in resolved:
        existing = union.get(document.document_id)
        if existing is not None and existing != document:
            raise RunnerError(
                f"{corpus} {partition} selected document ID {document.document_id!r} resolves "
                "inconsistently across tokenizers."
            )
        union[document.document_id] = document


def _validate_selected_chunk_union_overlap(
    *,
    domains: Mapping[str, MaterializedDomain],
    model_names: Sequence[str],
    domain_names: Sequence[str],
) -> None:
    for domain_name in domain_names:
        fit_union: dict[str, _ResolvedSelectedDocument] = {}
        eval_union: dict[str, _ResolvedSelectedDocument] = {}
        for model_name in model_names:
            key = _domain_entry_name(model_name, domain_name)
            materialized = domains[key]
            _accumulate_selected_union(
                fit_union,
                _resolve_selected_documents(materialized, partition="fit"),
                corpus=domain_name,
                partition="fit",
            )
            _accumulate_selected_union(
                eval_union,
                _resolve_selected_documents(materialized, partition="eval"),
                corpus=domain_name,
                partition="eval",
            )
        overlapping_document_ids = sorted(set(fit_union) & set(eval_union))
        fit_rows = {row for document in fit_union.values() for row in document.row_indices}
        eval_rows = {row for document in eval_union.values() for row in document.row_indices}
        overlapping_source_rows = sorted(fit_rows & eval_rows)
        fit_content_hashes = {document.content_sha256 for document in fit_union.values()}
        eval_content_hashes = {document.content_sha256 for document in eval_union.values()}
        overlapping_content_hashes = sorted(fit_content_hashes & eval_content_hashes)
        if overlapping_document_ids or overlapping_source_rows or overlapping_content_hashes:
            details: list[str] = []
            if overlapping_document_ids:
                details.append(f"document IDs {overlapping_document_ids!r}")
            if overlapping_source_rows:
                details.append(f"source rows {overlapping_source_rows!r}")
            if overlapping_content_hashes:
                details.append(f"joined-content hashes {overlapping_content_hashes!r}")
            raise RunnerError(
                f"{domain_name} selected fit/eval union overlap detected: "
                + ", ".join(details)
                + "."
            )


def _write_batch_manifest(
    *,
    run_root: RunRoot,
    resolved_config: ResolvedConfig,
    sweep: SweepConfig,
    domains: Mapping[str, MaterializedDomain],
    token_manifest_entries: Mapping[str, DomainEntryPayload],
    command: Sequence[str],
    runtime_command_runner: RuntimeCommandRunner,
) -> BatchManifestPayload:
    paths = _manifest_paths(run_root)
    bindings = _collect_model_bindings(resolved_config)
    _load_existing_or_write_payload(
        paths["resolved_config"],
        {"schema_version": 1, "config": resolved_config},
        label="resolved_config_manifest",
    )
    _load_existing_or_write_payload(
        paths["sweep"], {"schema_version": 1, "sweep": sweep}, label="sweep_manifest"
    )
    _load_existing_or_write_payload(
        paths["materialized"],
        {
            "schema_version": 1,
            "domains": {key: _materialize_payload_value(value) for key, value in domains.items()},
        },
        label="materialized_manifest",
    )
    git = capture_git_binding(resolved_config.paths.project_root, TRACKED_GIT_PATHS)
    runtime, runtime_capture = _capture_runtime_provenance(runtime_command_runner)
    datasets = [
        DatasetBinding(
            name=key,
            repository=value.dataset_repository,
            revision=value.dataset_revision,
            config=value.dataset_config,
            split=value.dataset_split,
            field=value.dataset_field,
            fingerprint=value.dataset_fingerprint,
            manifest_sha256=value.manifest_sha256,
        )
        for key, value in sorted(domains.items())
    ]
    tokenizers = [
        TokenizerBinding(
            model_name=name,
            tokenizer_path=binding.tokenizer_path,
            tokenizer_tree_sha256=binding.tokenizer_tree_sha256,
        )
        for name, binding in sorted(bindings.items())
    ]
    models = [
        ModelBinding(
            model_name=name,
            weights_path=binding.weights_path,
            weights_tree_sha256=binding.weights_tree_sha256,
        )
        for name, binding in sorted(bindings.items())
    ]
    provenance = build_run_provenance(
        seed_namespace=FIXED_SEED_NAMESPACE,
        command=command,
        config_identity_sha256=resolved_config.identity_sha256,
        sweep_identity_sha256=sweep.identity_sha256,
        git=git,
        datasets=datasets,
        tokenizers=tokenizers,
        models=models,
        runtime=runtime,
    )
    payload: BatchManifestPayload = {
        "schema_version": 1,
        "materialized_at_utc": utc_now(),
        "command": list(command),
        "config_sha256": resolved_config.identity_sha256,
        "sweep_sha256": sweep.identity_sha256,
        "domains": {key: value.manifest_sha256 for key, value in sorted(domains.items())},
        "domain_entries": dict(sorted(token_manifest_entries.items())),
        "tokenizers": {
            name: binding.tokenizer_tree_sha256 for name, binding in sorted(bindings.items())
        },
        "models": {name: binding.weights_tree_sha256 for name, binding in sorted(bindings.items())},
        "git_commit_sha": provenance.git.commit_sha,
        "git_tracked_diff_sha256": provenance.git.tracked_diff_sha256,
        "git_relevant_content_sha256": provenance.git.relevant_content_sha256,
        "git_untracked_sha256": provenance.git.untracked_sha256,
        "runtime": _json_object(asdict(runtime), label="batch_manifest.runtime"),
        "runtime_capture": runtime_capture,
        "provenance": _json_object(asdict(provenance), label="batch_manifest.provenance"),
        "environment": {
            key: os.environ[key]
            for key in sorted(os.environ)
            if key.startswith("SI_REBUTTAL_") or key in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH")
        },
        "identity_sha256": "",
    }
    written, _ = _load_existing_or_write_payload(paths["batch"], payload, label="batch_manifest")
    return _batch_manifest_payload(written, label="batch_manifest")


def _load_batch_manifest(run_root: RunRoot) -> BatchManifestPayload:
    path = _manifest_paths(run_root)["batch"]
    if not path.exists():
        raise RunnerError(f"Missing batch manifest: {path}")
    return _batch_manifest_payload(
        _read_verified_payload(path, label="Batch manifest"), label=path.name
    )


def _load_named_manifest(path: Path, *, label: str) -> JsonObject:
    if not path.exists():
        raise RunnerError(f"Missing {label}: {path}")
    return dict(_read_verified_payload(path, label=label))


def _materialized_domain_payloads(run_root: RunRoot) -> dict[str, JsonObject]:
    path = _manifest_paths(run_root)["materialized"]
    if not path.exists():
        raise RunnerError(f"Missing materialized-data manifest: {path}")
    payload = _read_verified_payload(path, label="Materialized-data manifest")
    domains = _json_object_member(payload, "domains", label=path.name)
    return {
        key: dict(_json_object(value, label=f"{path.name}.domains.{key}"))
        for key, value in domains.items()
    }


def _batch_domain_entries(run_root: RunRoot) -> dict[str, DomainEntryPayload]:
    return dict(_load_batch_manifest(run_root)["domain_entries"])


def _sequence_digest(sequences: Sequence[Int64Array]) -> str:
    payload: list[JsonObject] = []
    for sequence in sequences:
        values = np.asarray(sequence, dtype="<i8")
        payload.append({"shape": list(values.shape), "token_sha256": token_sha256(values.tolist())})
    return artifact_identity(payload)


def sequence_digest(sequences: Sequence[Int64Array]) -> str:
    return _sequence_digest(sequences)


def _is_object_tuple(value: object) -> TypeGuard[tuple[object, ...]]:
    return isinstance(value, tuple)


def _normalize_kernel_digest_key(key: object) -> KernelKey:
    if _is_object_tuple(key):
        if len(key) != 2:
            raise RunnerError(f"Kernel digest key tuple must have length two: {key!r}.")
        layer = key[0]
        head = key[1]
        if (
            isinstance(layer, int)
            and not isinstance(layer, bool)
            and isinstance(head, int)
            and not isinstance(head, bool)
            and layer >= 0
            and head >= 0
        ):
            return layer, head
        raise RunnerError(f"Kernel digest key tuple must be two nonnegative ints: {key!r}.")
    if isinstance(key, str):
        if key != key.strip():
            raise RunnerError(f"Kernel digest key string must be canonical layer:head: {key!r}.")
        match = re.fullmatch(r"(0|[1-9][0-9]*):(0|[1-9][0-9]*)", key)
        if match is None:
            raise RunnerError(f"Kernel digest key string must be canonical layer:head: {key!r}.")
        return int(match.group(1)), int(match.group(2))
    raise RunnerError(f"Kernel digest key has invalid shape: {key!r}.")


def _kernel_digest(kernel_entries: Iterable[tuple[object, KernelValue]]) -> str:
    normalized_items: list[tuple[KernelKey, KernelValue]] = []
    seen_keys: set[KernelKey] = set()
    for key, kernel in kernel_entries:
        normalized_key = _normalize_kernel_digest_key(key)
        if normalized_key in seen_keys:
            raise RunnerError(f"Kernel digest key normalized collision: {key!r}.")
        seen_keys.add(normalized_key)
        normalized_items.append((normalized_key, kernel))
    return artifact_identity(
        {
            f"{layer}:{head}": np.asarray(kernel, dtype=np.float64).tolist()
            for (layer, head), kernel in sorted(normalized_items, key=lambda item: item[0])
        }
    )


def kernel_digest(kernel_map: KernelMap) -> str:
    return _kernel_digest(kernel_map.items())


def _intervention_kernel_map(kernel_map: KernelMap) -> dict[KernelKey, torch.Tensor]:
    return {
        key: torch.as_tensor(value, dtype=torch.float32, device="cpu")
        for key, value in kernel_map.items()
    }


def _selected_head_digest(heads: Sequence[HeadIndex]) -> str:
    return artifact_identity([{"layer": head.layer, "head": head.head} for head in heads])


def _head_records(heads: Sequence[HeadIndex]) -> list[dict[str, int]]:
    return [{"layer": head.layer, "head": head.head} for head in heads]


def _protocol_identity(resolved_config: ResolvedConfig, sweep: SweepConfig) -> str:
    return artifact_identity(
        {
            "runtime": dict(resolved_config.runtime),
            "counts": dict(resolved_config.counts),
            "statistics": dict(resolved_config.statistics),
            "validation": dict(resolved_config.validation),
            "directions": list(sweep.directions),
            "controls": list(sweep.controls),
            "include_depth": bool(sweep.include_depth),
            "depth_direction": sweep.depth_direction,
            "depth_unit": sweep.depth_unit,
        }
    )


def _verify_control_seed(
    *,
    model_name: str,
    direction: str,
    unit: str,
    unit_index: str | int,
    trial: int,
    control_kind: str,
    seed: int,
) -> None:
    if unit != "bin":
        raise RunnerError(f"Unsupported control seed unit: {unit}")
    expected = compact_ascii_json_seed(
        [
            29039,
            model_name,
            direction,
            unit,
            _parse_fixed_unit_index(unit_index, unit=unit),
            trial,
            control_kind,
        ]
    )
    if int(seed) != int(expected):
        raise RunnerError(
            "Control seed drifted for "
            f"{model_name}/{direction}/{unit}/{unit_index}/{control_kind}/{trial}: "
            f"expected {expected}, found {seed}."
        )


def _release_model_bundle(bundle: LoadedModelBundle | None) -> None:
    if bundle is None:
        return
    model = getattr(bundle, "model", None)
    tokenizer = getattr(bundle, "tokenizer", None)
    adapter = getattr(bundle, "adapter", None)
    del model, tokenizer, adapter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _document_lookup(materialized_payload: JsonMapping, partition: str) -> dict[str, JsonObject]:
    documents = _json_array_member(materialized_payload, f"{partition}_documents", label=partition)
    out: dict[str, JsonObject] = {}
    for index, document in enumerate(documents):
        payload = _json_object(document, label=f"{partition}_documents[{index}]")
        document_id = _json_string_member(
            payload, "document_id", label=f"{partition}_documents[{index}]"
        )
        out[document_id] = dict(payload)
    return out


def _chunk_list(materialized_payload: JsonMapping, partition: str) -> list[JsonObject]:
    chunks = _json_array_member(materialized_payload, f"{partition}_chunks", label=partition)
    return [
        dict(_json_object(chunk, label=f"{partition}_chunks[{index}]"))
        for index, chunk in enumerate(chunks)
    ]


def _serialize_token_sequences(
    *,
    sequences: Sequence[Int64Array],
    chunks: Sequence[SelectedChunk],
    materialized: MaterializedDomain,
    manifest_sha256: str,
    config_sha256: str,
    sweep_sha256: str,
    model_name: str,
    domain_name: str,
    partition: str,
) -> TokenManifestPayload:
    if len(sequences) != len(chunks):
        raise RunnerError(
            f"Token sequence/chunk count mismatch for {model_name}:{domain_name}:{partition}."
        )
    document_lookup = {
        document.document_id: document
        for document in (
            materialized.fit_documents if partition == "fit" else materialized.eval_documents
        )
    }
    records: list[TokenSequenceRecord] = []
    for sequence, chunk in zip(sequences, chunks, strict=True):
        values = np.asarray(sequence, dtype=np.int64)
        if values.shape != (512,):
            raise RunnerError(
                f"Expected exact 512-token sequence for {model_name}:{domain_name}:{partition}."
            )
        document = document_lookup[chunk.document_id]
        payload: TokenSequenceRecord = {
            "document_id": chunk.document_id,
            "partition": chunk.partition,
            "chunk_index": int(chunk.chunk_index),
            "source_rows": list(chunk.source_rows),
            "token_count": int(chunk.token_count),
            "token_shape": list(chunk.token_shape),
            "token_sha256": token_sha256(values.tolist()),
            "document_content_sha256": document.content_sha256,
            "document_assignment_sha256": document.assignment_sha256,
            "tokens": values.tolist(),
        }
        records.append(payload)
    manifest_payload = {
        "schema_version": 1,
        "model": model_name,
        "domain": domain_name,
        "partition": partition,
        "count": len(records),
        "config_sha256": config_sha256,
        "sweep_sha256": sweep_sha256,
        "materialized_domain_sha256": manifest_sha256,
        "sequence_digest_sha256": _sequence_digest(sequences),
        "sequences": records,
    }
    return _token_manifest_payload(
        _normalized_json_object(
            manifest_payload, label=f"token_manifest.{model_name}.{domain_name}"
        ),
        label=f"token_manifest.{model_name}.{domain_name}",
    )


def _load_token_sequences(
    *,
    run_root: RunRoot,
    model_name: str,
    domain_name: str,
    partition: str,
    expected_count: int,
    expected_domain_payload: JsonMapping,
    expected_domain_sha256: str,
    expected_config_sha256: str,
    expected_sweep_sha256: str,
    expected_token_manifest_sha256: str | None = None,
    expected_sequence_digest_sha256: str | None = None,
) -> tuple[Int64Array, ...]:
    path = _token_manifest_path(
        run_root, model_name=model_name, domain_name=domain_name, partition=partition
    )
    if not path.exists():
        raise RunnerError(f"Missing token manifest: {path.name}")
    payload = _token_manifest_payload(
        _read_verified_payload(path, label="Token manifest"),
        label=path.name,
    )
    if payload["schema_version"] != SCHEMA_VERSION:
        raise RunnerError(f"Token manifest schema drifted for {path.name}.")
    if (
        payload["model"] != model_name
        or payload["domain"] != domain_name
        or payload["partition"] != partition
    ):
        raise RunnerError(f"Token manifest identity drifted for {path.name}.")
    if payload["count"] != expected_count:
        raise RunnerError(
            f"Token manifest count drifted for {path.name}: expected {expected_count}, "
            f"found {payload['count']}."
        )
    if payload["materialized_domain_sha256"] != expected_domain_sha256:
        raise RunnerError(f"Token manifest domain drifted for {path.name}.")
    if payload["config_sha256"] != expected_config_sha256:
        raise RunnerError(f"Token manifest config drifted for {path.name}.")
    if payload["sweep_sha256"] != expected_sweep_sha256:
        raise RunnerError(f"Token manifest sweep drifted for {path.name}.")
    if (
        expected_token_manifest_sha256 is not None
        and payload["identity_sha256"] != expected_token_manifest_sha256
    ):
        raise RunnerError(f"Token manifest immutable identity drifted for {path.name}.")
    expected_chunks = _chunk_list(expected_domain_payload, partition)
    expected_documents = _document_lookup(expected_domain_payload, partition)
    records = payload["sequences"]
    sequences = tuple(np.asarray(record["tokens"], dtype=np.int64) for record in records)
    if len(sequences) != expected_count:
        raise RunnerError(
            f"Token manifest count drifted for {path.name}: expected {expected_count}, "
            f"found {len(sequences)}."
        )
    if len(expected_chunks) != expected_count:
        raise RunnerError(
            f"Materialized chunk count drifted for {model_name}:{domain_name}:{partition}."
        )
    for sequence, record, expected_chunk in zip(sequences, records, expected_chunks, strict=True):
        if sequence.shape != (512,):
            raise RunnerError(f"Token manifest sequence shape drifted for {path.name}.")
        if record["document_id"] != expected_chunk["document_id"]:
            raise RunnerError(f"Token manifest document drifted for {path.name}.")
        if record["partition"] != expected_chunk["partition"]:
            raise RunnerError(f"Token manifest partition drifted for {path.name}.")
        if record["chunk_index"] != _json_int_member(
            expected_chunk, "chunk_index", label=path.name
        ):
            raise RunnerError(f"Token manifest chunk index drifted for {path.name}.")
        if record["source_rows"] != [
            _json_int(item, label=f"{path.name}.source_rows[]")
            for item in _json_array_member(expected_chunk, "source_rows", label=path.name)
        ]:
            raise RunnerError(f"Token manifest source rows drifted for {path.name}.")
        if record["token_count"] != _json_int_member(
            expected_chunk, "token_count", label=path.name
        ):
            raise RunnerError(f"Token manifest token count drifted for {path.name}.")
        if record["token_shape"] != [
            _json_int(item, label=f"{path.name}.token_shape[]")
            for item in _json_array_member(expected_chunk, "token_shape", label=path.name)
        ]:
            raise RunnerError(f"Token manifest token shape drifted for {path.name}.")
        if record["token_sha256"] != _json_string_member(
            expected_chunk, "token_sha256", label=path.name
        ):
            raise RunnerError(f"Token manifest token digest drifted for {path.name}.")
        document = expected_documents[record["document_id"]]
        if record["document_content_sha256"] != _json_string_member(
            document, "content_sha256", label=path.name
        ):
            raise RunnerError(f"Token manifest content digest drifted for {path.name}.")
        if record["document_assignment_sha256"] != _json_string_member(
            document, "assignment_sha256", label=path.name
        ):
            raise RunnerError(f"Token manifest assignment digest drifted for {path.name}.")
        if token_sha256(sequence.tolist()) != record["token_sha256"]:
            raise RunnerError(f"Token manifest sequence hash drifted for {path.name}.")
    if _sequence_digest(sequences) != payload["sequence_digest_sha256"]:
        raise RunnerError(f"Token manifest sequence digest drifted for {path.name}.")
    if (
        expected_sequence_digest_sha256 is not None
        and payload["sequence_digest_sha256"] != expected_sequence_digest_sha256
    ):
        raise RunnerError(f"Token manifest immutable sequence digest drifted for {path.name}.")
    return sequences


def validate_config(base_path: str | Path, sweep_path: str | Path) -> JsonObject:
    resolved_config, sweep = load_configs(base_path, sweep_path)
    return {
        "schema_version": 1,
        "config_sha256": resolved_config.identity_sha256,
        "sweep_sha256": sweep.identity_sha256,
        "models": list(sweep.models),
        "directions": list(sweep.directions),
        "placement": dict(resolved_config.placement),
    }


def _require_current_batch_manifest_provenance(
    *,
    batch_manifest: BatchManifestPayload,
    resolved_config: ResolvedConfig,
    runtime_command_runner: RuntimeCommandRunner,
    stage: str,
) -> None:
    git = capture_git_binding(resolved_config.paths.project_root, TRACKED_GIT_PATHS)
    if batch_manifest["git_commit_sha"] != git.commit_sha:
        raise RunnerError("Batch manifest git commit drifted.")
    if batch_manifest["git_tracked_diff_sha256"] != git.tracked_diff_sha256:
        raise RunnerError("Batch manifest tracked diff binding drifted.")
    if batch_manifest["git_relevant_content_sha256"] != git.relevant_content_sha256:
        raise RunnerError("Batch manifest relevant content binding drifted.")
    if batch_manifest["git_untracked_sha256"] != git.untracked_sha256:
        raise RunnerError("Batch manifest untracked binding drifted.")
    runtime_capture = _json_object_member(batch_manifest, "runtime_capture", label="batch_manifest")
    _validate_existing_package_freeze_payload(
        _json_object_member(
            runtime_capture, "package_freeze", label="batch_manifest.runtime_capture"
        )
    )
    current_runtime, current_runtime_capture = _capture_runtime_provenance(runtime_command_runner)
    if batch_manifest.get("runtime") != asdict(current_runtime):
        raise RunnerError("Batch manifest runtime binding drifted.")
    if not _runtime_capture_matches_current(
        stage=stage,
        expected_runtime_capture=runtime_capture,
        current_runtime_capture=current_runtime_capture,
    ):
        raise RunnerError("Batch manifest runtime capture drifted.")


def _validate_existing_materialization(
    *,
    root: RunRoot,
    resolved_config: ResolvedConfig,
    sweep: SweepConfig,
    runtime_command_runner: RuntimeCommandRunner,
) -> JsonObject:
    paths = _manifest_paths(root)
    missing = [
        name
        for name in ("resolved_config", "sweep", "materialized", "batch")
        if not paths[name].exists()
    ]
    if missing:
        raise RunnerError(
            f"Existing materialization is incomplete for {root.root}: "
            f"missing {missing[0]} manifest."
        )
    resolved_payload = _load_named_manifest(
        paths["resolved_config"], label="resolved-config manifest"
    )
    sweep_payload = _load_named_manifest(paths["sweep"], label="sweep manifest")
    if (
        _json_string_member(
            _json_object_member(resolved_payload, "config", label="resolved_config"),
            "identity_sha256",
            label="resolved_config.config",
        )
        != resolved_config.identity_sha256
    ):
        raise RunnerError("Resolved config manifest drifted for existing materialization.")
    if (
        _json_string_member(
            _json_object_member(sweep_payload, "sweep", label="sweep_manifest"),
            "identity_sha256",
            label="sweep_manifest.sweep",
        )
        != sweep.identity_sha256
    ):
        raise RunnerError("Sweep manifest drifted for existing materialization.")
    batch_manifest = _load_batch_manifest(root)
    materialized_payloads = _materialized_domain_payloads(root)
    materialized_at_utc = batch_manifest["materialized_at_utc"]
    if not materialized_at_utc:
        raise RunnerError("Batch manifest materialization timestamp drifted.")
    if batch_manifest["config_sha256"] != resolved_config.identity_sha256:
        raise RunnerError("Batch manifest config binding drifted.")
    if batch_manifest["sweep_sha256"] != sweep.identity_sha256:
        raise RunnerError("Batch manifest sweep binding drifted.")
    provenance = _json_object_member(batch_manifest, "provenance", label="batch_manifest")
    if (
        _json_int_member(provenance, "seed_namespace", label="batch_manifest.provenance")
        != FIXED_SEED_NAMESPACE
    ):
        raise RunnerError("Batch manifest seed namespace drifted.")
    if (
        _json_string_member(provenance, "config_identity_sha256", label="batch_manifest.provenance")
        != resolved_config.identity_sha256
    ):
        raise RunnerError("Batch manifest provenance config binding drifted.")
    if (
        _json_string_member(provenance, "sweep_identity_sha256", label="batch_manifest.provenance")
        != sweep.identity_sha256
    ):
        raise RunnerError("Batch manifest provenance sweep binding drifted.")
    _require_current_batch_manifest_provenance(
        batch_manifest=batch_manifest,
        resolved_config=resolved_config,
        runtime_command_runner=runtime_command_runner,
        stage="materialize",
    )
    bindings = _collect_model_bindings(resolved_config)
    expected_keys = {
        _domain_entry_name(model_name, domain_name)
        for model_name in sweep.models
        for domain_name in resolved_config.datasets
    }
    domains_payload = _json_object_member(batch_manifest, "domains", label="batch_manifest")
    if set(domains_payload) != expected_keys:
        raise RunnerError("Batch manifest domain inventory drifted.")
    domain_entries_payload = _json_object_member(
        batch_manifest, "domain_entries", label="batch_manifest"
    )
    if set(domain_entries_payload) != expected_keys:
        raise RunnerError("Batch manifest domain-entry inventory drifted.")
    if set(materialized_payloads) != expected_keys:
        raise RunnerError("Materialized domain inventory drifted.")
    for model_name in sweep.models:
        binding = bindings[model_name]
        models_payload = _json_object_member(batch_manifest, "models", label="batch_manifest")
        if models_payload.get(model_name) != binding.weights_tree_sha256:
            raise RunnerError(f"Batch manifest model binding drifted for {model_name}.")
        tokenizers_payload = _json_object_member(
            batch_manifest, "tokenizers", label="batch_manifest"
        )
        if tokenizers_payload.get(model_name) != binding.tokenizer_tree_sha256:
            raise RunnerError(f"Batch manifest tokenizer binding drifted for {model_name}.")
        for domain_name, dataset in resolved_config.datasets.items():
            key = _domain_entry_name(model_name, domain_name)
            entry = _json_object_member(
                domain_entries_payload, key, label="batch_manifest.domain_entries"
            )
            materialized_payload = dict(materialized_payloads[key])
            materialized_sha256 = _json_string_member(
                materialized_payload, "manifest_sha256", label=key
            )
            if (
                _json_string_member(domains_payload, key, label="batch_manifest.domains")
                != materialized_sha256
            ):
                raise RunnerError(f"Batch manifest materialized-domain binding drifted for {key}.")
            if (
                _json_string_member(entry, "materialized_domain_sha256", label=key)
                != materialized_sha256
            ):
                raise RunnerError(f"Domain entry materialized-domain binding drifted for {key}.")
            if (
                _json_string_member(entry, "tokenizer_tree_sha256", label=key)
                != binding.tokenizer_tree_sha256
            ):
                raise RunnerError(f"Domain entry tokenizer binding drifted for {key}.")
            if _json_string_member(entry, "dataset_repository", label=key) != dataset.repository:
                raise RunnerError(f"Domain entry repository drifted for {key}.")
            if _json_string_member(entry, "dataset_revision", label=key) != dataset.revision:
                raise RunnerError(f"Domain entry revision drifted for {key}.")
            if _json_string_member(entry, "dataset_config", label=key) != dataset.config:
                raise RunnerError(f"Domain entry config drifted for {key}.")
            if _json_string_member(entry, "dataset_split", label=key) != dataset.split:
                raise RunnerError(f"Domain entry split drifted for {key}.")
            if _json_string_member(entry, "dataset_field", label=key) != dataset.field:
                raise RunnerError(f"Domain entry field drifted for {key}.")
            if _json_string_member(entry, "dataset_fingerprint", label=key) != _json_string_member(
                materialized_payload, "dataset_fingerprint", label=key
            ):
                raise RunnerError(f"Domain entry fingerprint drifted for {key}.")
            _load_token_sequences(
                run_root=root,
                model_name=model_name,
                domain_name=domain_name,
                partition="fit",
                expected_count=len(
                    _json_array_member(materialized_payload, "fit_chunks", label=key)
                ),
                expected_domain_payload=materialized_payload,
                expected_domain_sha256=materialized_sha256,
                expected_config_sha256=resolved_config.identity_sha256,
                expected_sweep_sha256=sweep.identity_sha256,
                expected_token_manifest_sha256=_json_string_member(
                    entry, "fit_token_manifest_sha256", label=key
                ),
                expected_sequence_digest_sha256=_json_string_member(
                    entry, "fit_sequence_digest_sha256", label=key
                ),
            )
            _load_token_sequences(
                run_root=root,
                model_name=model_name,
                domain_name=domain_name,
                partition="eval",
                expected_count=len(
                    _json_array_member(materialized_payload, "eval_chunks", label=key)
                ),
                expected_domain_payload=materialized_payload,
                expected_domain_sha256=materialized_sha256,
                expected_config_sha256=resolved_config.identity_sha256,
                expected_sweep_sha256=sweep.identity_sha256,
                expected_token_manifest_sha256=_json_string_member(
                    entry, "eval_token_manifest_sha256", label=key
                ),
                expected_sequence_digest_sha256=_json_string_member(
                    entry, "eval_sequence_digest_sha256", label=key
                ),
            )
    return _json_object(
        {
            "schema_version": 1,
            "run_root": str(root.root),
            "config_sha256": resolved_config.identity_sha256,
            "sweep_sha256": sweep.identity_sha256,
            "materialized_keys": sorted(materialized_payloads),
            "batch_manifest_sha256": str(batch_manifest["identity_sha256"]),
        },
        label="materialized_validation",
    )


def materialize_data(
    base_path: str | Path,
    sweep_path: str | Path,
    *,
    run_root: Path,
    load_dataset_fn: DatasetLoaderFn | None = None,
    tokenizer_loader: TokenizerLoader | None = None,
    runtime_command_runner: RuntimeCommandRunner | None = None,
) -> JsonObject:
    resolved_config, sweep = load_configs(base_path, sweep_path)
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id=None)
    paths = _manifest_paths(root)
    if runtime_command_runner is None:
        runtime_command_runner = default_subprocess_runner
    if paths["batch"].exists() or paths["materialized"].exists():
        return _validate_existing_materialization(
            root=root,
            resolved_config=resolved_config,
            sweep=sweep,
            runtime_command_runner=runtime_command_runner,
        )
    if load_dataset_fn is None:
        load_dataset_fn = _default_load_dataset
    dataset_loader = ImmutableHFDatasetLoader(load_dataset_fn)
    bindings = _collect_model_bindings(resolved_config)
    domains: dict[str, MaterializedDomain] = {}
    for model_name in sweep.models:
        tokenizer = _load_tokenizer(resolved_config.models[model_name], tokenizer_loader)

        def encode(text: str, tok: TokenizerLike = tokenizer) -> list[int]:
            return tok.encode(text, add_special_tokens=False)

        for domain_name, dataset in resolved_config.datasets.items():
            key = f"{model_name}:{domain_name}"
            domains[key] = materialize_frozen_domain(
                dataset=dataset,
                dataset_loader=dataset_loader,
                tokenizer_name=model_name,
                tokenizer_tree_sha256=bindings[model_name].tokenizer_tree_sha256,
                encode=encode,
            )
    _validate_selected_chunk_union_overlap(
        domains=domains,
        model_names=sweep.models,
        domain_names=tuple(resolved_config.datasets),
    )
    token_manifest_entries: dict[str, DomainEntryPayload] = {}
    for model_name in sweep.models:
        for domain_name in resolved_config.datasets:
            key = _domain_entry_name(model_name, domain_name)
            fit, eval_values, materialized = _reconstruct_sequences_for_materialized_domain(
                resolved_config=resolved_config,
                model_name=model_name,
                domain_name=domain_name,
                materialized=domains[key],
                load_dataset_fn=load_dataset_fn,
                tokenizer_loader=tokenizer_loader,
            )
            fit_payload: TokenManifestPayload | None = None
            eval_payload: TokenManifestPayload | None = None
            for partition, sequences, chunks in (
                ("fit", fit, materialized.fit_chunks),
                ("eval", eval_values, materialized.eval_chunks),
            ):
                payload = _serialize_token_sequences(
                    sequences=sequences,
                    chunks=chunks,
                    materialized=materialized,
                    manifest_sha256=materialized.manifest_sha256,
                    config_sha256=resolved_config.identity_sha256,
                    sweep_sha256=sweep.identity_sha256,
                    model_name=model_name,
                    domain_name=domain_name,
                    partition=partition,
                )
                stored_payload, _ = _load_existing_or_write_payload(
                    _token_manifest_path(
                        root, model_name=model_name, domain_name=domain_name, partition=partition
                    ),
                    payload,
                    label=f"token_manifest.{model_name}.{domain_name}.{partition}",
                )
                if partition == "fit":
                    fit_payload = _token_manifest_payload(
                        stored_payload,
                        label=f"token_manifest.{model_name}.{domain_name}.{partition}",
                    )
                else:
                    eval_payload = _token_manifest_payload(
                        stored_payload,
                        label=f"token_manifest.{model_name}.{domain_name}.{partition}",
                    )
            if fit_payload is None or eval_payload is None:
                raise RunnerError(
                    f"Token manifest generation failed for {model_name}:{domain_name}."
                )
            token_manifest_entries[key] = _batch_domain_entry(
                model_name=model_name,
                domain_name=domain_name,
                materialized=materialized,
                tokenizer_tree_sha256=bindings[model_name].tokenizer_tree_sha256,
                fit_token_manifest_sha256=str(fit_payload["identity_sha256"]),
                eval_token_manifest_sha256=str(eval_payload["identity_sha256"]),
                fit_sequence_digest_sha256=str(fit_payload["sequence_digest_sha256"]),
                eval_sequence_digest_sha256=str(eval_payload["sequence_digest_sha256"]),
            )
    manifest = _write_batch_manifest(
        run_root=root,
        resolved_config=resolved_config,
        sweep=sweep,
        domains=domains,
        token_manifest_entries=token_manifest_entries,
        command=sys.argv,
        runtime_command_runner=runtime_command_runner,
    )
    return _json_object(
        {
            "schema_version": 1,
            "run_root": str(run_root),
            "config_sha256": resolved_config.identity_sha256,
            "sweep_sha256": sweep.identity_sha256,
            "materialized_keys": list(sorted(domains)),
            "batch_manifest_sha256": manifest["identity_sha256"],
        },
        label="materialized_result",
    )


def _toy_sequence_losses(*, baseline: float, strength: float, count: int) -> Float64Array:
    idx = np.arange(count, dtype=np.float64)
    return baseline + strength + (idx % 11) * 1e-4


def _terminal_payload_identity(payload: JsonMapping) -> str:
    return artifact_identity(
        payload_without_identity(_json_object(payload, label="terminal_payload"))
    )


def _require_permutation_pvalue_invariant(
    *,
    label: str,
    p_one_sided: float,
    permutation_count: int,
    permutation_exceedance_count: int,
) -> None:
    expected = (1.0 + float(permutation_exceedance_count)) / (1.0 + float(permutation_count))
    if not math.isclose(p_one_sided, expected, rel_tol=0.0, abs_tol=1e-15):
        raise RunnerError(f"Permutation p-value invariant drifted for {label}.")


def _terminal_payload_bytes(payload: JsonMapping) -> bytes:
    normalized = _json_object(payload, label="terminal_payload")
    return canonical_json_bytes(normalized) + b"\n"


def _load_existing_terminal_payload(path: Path, *, label: str) -> JsonObject:
    if not path.exists():
        raise RunnerError(f"Missing {label}: {path}")
    try:
        return verify_payload_identity(json.loads(path.read_text(encoding="ascii")), label=label)
    except (json.JSONDecodeError, UnicodeDecodeError, ArtifactError) as exc:
        raise RunnerError(f"{label} is not a valid immutable terminal summary: {path}") from exc


def _scientific_projection(payload: JsonMapping) -> JsonObject:
    normalized = _json_object(payload, label="terminal_projection")
    dose_response_statistics = [
        _json_object(row, label="dose_response_statistic")
        for row in _json_array_member(
            normalized, "dose_response_statistics", label="terminal_projection"
        )
    ]
    projected_stats: list[JsonObject] = [
        _json_object(
            {
                "model": _json_string_member(row, "model", label="dose_response_statistic"),
                "direction": _json_string_member(row, "direction", label="dose_response_statistic"),
                "rho": _json_float_member(row, "rho", label="dose_response_statistic"),
                "p_one_sided": _json_float_member(
                    row, "p_one_sided", label="dose_response_statistic"
                ),
                "p_two_sided_scipy": _json_float_member(
                    row, "p_two_sided_scipy", label="dose_response_statistic"
                ),
            },
            label="projected_dose_response_statistic",
        )
        for row in dose_response_statistics
    ]
    projected_depth_relationships: list[JsonObject] = []
    for row in _json_array_member(normalized, "depth_relationships", label="terminal_projection"):
        relationship = _json_object(row, label="depth_relationship")
        projected = {
            key: value
            for key, value in relationship.items()
            if key
            not in {
                "positive_null_permutation_count",
                "permutation_count",
                "permutation_exceedance_count",
            }
        }
        projected_depth_relationships.append(
            _json_object(projected, label="projected_depth_relationship")
        )
    summary_hashes = _json_object_member(normalized, "summary_hashes", label="terminal_projection")
    return _json_object(
        {
            "run_id": _json_string_member(normalized, "run_id", label="terminal_projection"),
            "config_sha256": _json_string_member(
                normalized, "config_sha256", label="terminal_projection"
            ),
            "sweep_sha256": _json_string_member(
                normalized, "sweep_sha256", label="terminal_projection"
            ),
            "batch_manifest_sha256": _json_string_member(
                normalized, "batch_manifest_sha256", label="terminal_projection"
            ),
            "shards_verified": _json_int_member(
                normalized, "shards_verified", label="terminal_projection"
            ),
            "summary_rows": _json_array_member(
                normalized, "summary_rows", label="terminal_projection"
            ),
            "dose_response_statistics": projected_stats,
            "contrasts": _json_array_member(normalized, "contrasts", label="terminal_projection"),
            "depth_rows": _json_array_member(normalized, "depth_rows", label="terminal_projection"),
            "depth_relationships": projected_depth_relationships,
            "summary_hashes": _json_object(
                {
                    "shard_inventory_sha256": _json_string_member(
                        summary_hashes, "shard_inventory_sha256", label="summary_hashes"
                    ),
                    "summary_rows_sha256": _json_string_member(
                        summary_hashes, "summary_rows_sha256", label="summary_hashes"
                    ),
                    "contrasts_sha256": _json_string_member(
                        summary_hashes, "contrasts_sha256", label="summary_hashes"
                    ),
                    "depth_rows_sha256": _json_string_member(
                        summary_hashes, "depth_rows_sha256", label="summary_hashes"
                    ),
                },
                label="projected_summary_hashes",
            ),
        },
        label="scientific_projection",
    )


_PRODUCTION_CONFIG_RELATIVE_PATH = Path("rebuttal/configs/base.toml")
_PRODUCTION_SWEEP_RELATIVE_PATH = Path("rebuttal/configs/sweep.toml")
_EXPECTED_PRODUCTION_MANIFEST_FILE_COUNT = 22
_EXPECTED_PRODUCTION_SHARD_COUNT = 1062
_EXPECTED_PRODUCTION_IMMUTABLE_INPUT_ENTRY_COUNT = 1087
_FIXED_PRODUCTION_CONFIG_RAW_BYTE_SHA256 = (
    "5662f9763a6a04660fb959ab989f28808a6945105e8dee76e70e0be8af92c9d0"
)
_FIXED_PRODUCTION_SWEEP_RAW_BYTE_SHA256 = (
    "cb8229585d1fff95a90ac373f748a169bfd3ddd42a23639472484b2af4005922"
)
_FIXED_PRODUCTION_SOURCE_V1_RAW_BYTE_SHA256 = (
    "6e9227ff0b4fd5d54fbf7350ea4cab24f2f3c099c6a24ac49ca6f966ca82ed82"
)
_FIXED_PRODUCTION_IMMUTABLE_INPUT_MANIFEST_CONTENT_SHA256 = (
    "19ab19c5cae268b1fae0709af1b953bf6a324c06b9aff149482d0cb9203be1c9"
)
_FIXED_PRODUCTION_IMMUTABLE_INPUT_MANIFEST_IDENTITY_SHA256 = (
    "445e92a05c9e57e0e5fce002c72c2ea3cac64ad80e398e53acd22304cdd5d915"
)


def _manifest_entry(
    path: Path, *, project_root: Path, expected_byte_sha256: str | None = None
) -> JsonObject:
    file_bytes = path.read_bytes()
    byte_sha256 = sha256_bytes(file_bytes)
    if expected_byte_sha256 is not None and byte_sha256 != expected_byte_sha256:
        raise RunnerError(f"Immutable input file raw byte digest drifted: {path}")
    entry: JsonObject = {
        "path": os.path.relpath(path, project_root),
        "byte_sha256": expected_byte_sha256 or byte_sha256,
    }
    if path.suffix == ".json":
        try:
            payload = verify_payload_identity(
                json.loads(file_bytes.decode("ascii")), label=path.name
            )
        except (json.JSONDecodeError, UnicodeDecodeError, ArtifactError) as exc:
            raise RunnerError(f"Immutable input file is not a valid JSON artifact: {path}") from exc
        entry["identity_sha256"] = _json_string_member(payload, "identity_sha256", label=path.name)
    return entry


def _require_fixed_raw_digest(path: Path, *, expected: str, label: str) -> str:
    digest = sha256_bytes(path.read_bytes())
    if digest != expected:
        raise RunnerError(f"{label} raw byte digest drifted before v2 finalization.")
    return digest


def _require_fixed_production_config_raw_digest(path: Path) -> str:
    return _require_fixed_raw_digest(
        path,
        expected=_FIXED_PRODUCTION_CONFIG_RAW_BYTE_SHA256,
        label="Canonical production config",
    )


def _require_fixed_production_sweep_raw_digest(path: Path) -> str:
    return _require_fixed_raw_digest(
        path,
        expected=_FIXED_PRODUCTION_SWEEP_RAW_BYTE_SHA256,
        label="Canonical production sweep",
    )


def _require_fixed_production_source_v1_raw_digest(path: Path) -> str:
    return _require_fixed_raw_digest(
        path,
        expected=_FIXED_PRODUCTION_SOURCE_V1_RAW_BYTE_SHA256,
        label="Canonical production source-v1",
    )


def _expected_manifest_relative_paths(
    *, resolved_config: ResolvedConfig, sweep: SweepConfig
) -> tuple[Path, ...]:
    expected: list[Path] = [
        Path("batch-manifest.json"),
        Path("materialized-data.json"),
        Path("resolved-config.json"),
        Path("sweep.json"),
    ]
    for model_name in sweep.models:
        for domain_name in sorted(resolved_config.datasets):
            expected.append(Path("fit-profiles") / f"{model_name}.{domain_name}.json")
            for partition in ("eval", "fit"):
                expected.append(Path("tokens") / f"{model_name}.{domain_name}.{partition}.json")
    expected_paths = tuple(sorted(expected))
    if len(expected_paths) != _EXPECTED_PRODUCTION_MANIFEST_FILE_COUNT:
        raise RunnerError("Production manifest inventory shape drifted.")
    return expected_paths


def _require_production_manifest_inventory(
    *, run_root: RunRoot, resolved_config: ResolvedConfig, sweep: SweepConfig
) -> tuple[Path, ...]:
    expected_paths = _expected_manifest_relative_paths(resolved_config=resolved_config, sweep=sweep)
    actual_paths = tuple(
        sorted(
            os.path.relpath(path, run_root.manifests_dir)
            for path in run_root.manifests_dir.rglob("*")
            if stat.S_ISREG(path.lstat().st_mode)
        )
    )
    if len(actual_paths) != _EXPECTED_PRODUCTION_MANIFEST_FILE_COUNT:
        raise RunnerError(
            f"Production manifest inventory count drifted: expected "
            f"{_EXPECTED_PRODUCTION_MANIFEST_FILE_COUNT}, found {len(actual_paths)}."
        )
    expected_strings = tuple(path.as_posix() for path in expected_paths)
    if actual_paths != expected_strings:
        expected_set = set(expected_strings)
        actual_set = set(actual_paths)
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        if missing:
            raise RunnerError(f"Production manifest inventory missing file: {missing[0]}")
        if extra:
            raise RunnerError(f"Production manifest inventory found unexpected file: {extra[0]}")
        raise RunnerError("Production manifest inventory ordering drifted.")
    return tuple(run_root.manifests_dir / relative_path for relative_path in expected_paths)


def _require_production_shard_inventory(
    *, run_root: RunRoot, sweep: SweepConfig
) -> tuple[Path, ...]:
    expected = {
        key.stem()
        for model_name in sweep.models
        for key in _expected_condition_inventory(model_name, include_depth=sweep.include_depth)
    }
    if len(expected) != _EXPECTED_PRODUCTION_SHARD_COUNT:
        raise RunnerError("Production shard inventory shape drifted.")
    actual_paths = {path.stem: path for path in run_root.shards_dir.glob("*.json")}
    missing = sorted(expected - set(actual_paths))
    extra = sorted(set(actual_paths) - expected)
    if missing:
        raise RunnerError(f"Finalizer missing shard: {missing[0]}")
    if extra:
        raise RunnerError(f"Finalizer found unexpected shard: {extra[0]}")
    if len(actual_paths) != _EXPECTED_PRODUCTION_SHARD_COUNT:
        raise RunnerError(
            f"Production shard inventory count drifted: expected "
            f"{_EXPECTED_PRODUCTION_SHARD_COUNT}, found {len(actual_paths)}."
        )
    return tuple(sorted(actual_paths.values()))


def _require_production_manifest_graph(
    *,
    run_root: RunRoot,
    resolved_config: ResolvedConfig,
    sweep: SweepConfig,
    source_v1: JsonObject,
) -> tuple[tuple[Path, ...], BatchManifestPayload, list[str]]:
    manifest_paths = _require_production_manifest_inventory(
        run_root=run_root, resolved_config=resolved_config, sweep=sweep
    )
    source_v1_config_sha256 = _json_string_member(source_v1, "config_sha256", label="source_v1")
    source_v1_sweep_sha256 = _json_string_member(source_v1, "sweep_sha256", label="source_v1")
    source_v1_batch_sha256 = _json_string_member(
        source_v1, "batch_manifest_sha256", label="source_v1"
    )
    if source_v1_config_sha256 != resolved_config.identity_sha256:
        raise RunnerError("Source-v1 config binding drifted.")
    if source_v1_sweep_sha256 != sweep.identity_sha256:
        raise RunnerError("Source-v1 sweep binding drifted.")

    paths = _manifest_paths(run_root)
    resolved_payload = _load_named_manifest(
        paths["resolved_config"], label="resolved-config manifest"
    )
    if (
        _json_int_member(resolved_payload, "schema_version", label=paths["resolved_config"].name)
        != 1
    ):
        raise RunnerError("Resolved-config manifest schema drifted.")
    resolved_manifest_config = _json_object_member(
        resolved_payload, "config", label=paths["resolved_config"].name
    )
    if (
        _json_string_member(
            resolved_manifest_config,
            "identity_sha256",
            label=f"{paths['resolved_config'].name}.config",
        )
        != source_v1_config_sha256
    ):
        raise RunnerError("Resolved-config manifest immutable identity drifted.")

    sweep_payload = _load_named_manifest(paths["sweep"], label="sweep manifest")
    if _json_int_member(sweep_payload, "schema_version", label=paths["sweep"].name) != 1:
        raise RunnerError("Sweep manifest schema drifted.")
    sweep_manifest_payload = _json_object_member(sweep_payload, "sweep", label=paths["sweep"].name)
    if (
        _json_string_member(
            sweep_manifest_payload,
            "identity_sha256",
            label=f"{paths['sweep'].name}.sweep",
        )
        != source_v1_sweep_sha256
    ):
        raise RunnerError("Sweep manifest immutable identity drifted.")

    batch_manifest = _load_batch_manifest(run_root)
    if batch_manifest["identity_sha256"] != source_v1_batch_sha256:
        raise RunnerError("Batch manifest immutable identity drifted.")
    if batch_manifest["config_sha256"] != source_v1_config_sha256:
        raise RunnerError("Batch manifest config binding drifted.")
    if batch_manifest["sweep_sha256"] != source_v1_sweep_sha256:
        raise RunnerError("Batch manifest sweep binding drifted.")

    materialized_payloads = _materialized_domain_payloads(run_root)
    expected_keys = {
        _domain_entry_name(model_name, domain_name)
        for model_name in sweep.models
        for domain_name in resolved_config.datasets
    }
    domains_payload = _json_object_member(batch_manifest, "domains", label="batch_manifest")
    if set(domains_payload) != expected_keys:
        raise RunnerError("Batch manifest domain inventory drifted.")
    domain_entries_payload = _json_object_member(
        batch_manifest, "domain_entries", label="batch_manifest"
    )
    if set(domain_entries_payload) != expected_keys:
        raise RunnerError("Batch manifest domain-entry inventory drifted.")
    if set(materialized_payloads) != expected_keys:
        raise RunnerError("Materialized domain inventory drifted.")

    fit_profile_identity_sha256s: list[str] = []
    bindings = _collect_model_bindings(resolved_config)
    protocol_identity_sha256 = _protocol_identity(resolved_config, sweep)
    for model_name in sweep.models:
        binding = bindings[model_name]
        if batch_manifest["models"].get(model_name) != binding.weights_tree_sha256:
            raise RunnerError(f"Batch manifest model binding drifted for {model_name}.")
        if batch_manifest["tokenizers"].get(model_name) != binding.tokenizer_tree_sha256:
            raise RunnerError(f"Batch manifest tokenizer binding drifted for {model_name}.")
        for domain_name, dataset in resolved_config.datasets.items():
            key = _domain_entry_name(model_name, domain_name)
            entry = batch_manifest["domain_entries"][key]
            materialized_payload = materialized_payloads[key]
            materialized_sha256 = _json_string_member(
                materialized_payload, "manifest_sha256", label=key
            )
            if batch_manifest["domains"].get(key) != materialized_sha256:
                raise RunnerError(f"Batch manifest materialized-domain binding drifted for {key}.")
            if entry["model"] != model_name or entry["domain"] != domain_name:
                raise RunnerError(f"Domain entry identity drifted for {key}.")
            if entry["materialized_domain_sha256"] != materialized_sha256:
                raise RunnerError(f"Domain entry materialized-domain binding drifted for {key}.")
            if entry["tokenizer_tree_sha256"] != binding.tokenizer_tree_sha256:
                raise RunnerError(f"Domain entry tokenizer binding drifted for {key}.")
            if entry["dataset_repository"] != dataset.repository:
                raise RunnerError(f"Domain entry repository drifted for {key}.")
            if entry["dataset_revision"] != dataset.revision:
                raise RunnerError(f"Domain entry revision drifted for {key}.")
            if entry["dataset_config"] != dataset.config:
                raise RunnerError(f"Domain entry config drifted for {key}.")
            if entry["dataset_split"] != dataset.split:
                raise RunnerError(f"Domain entry split drifted for {key}.")
            if entry["dataset_field"] != dataset.field:
                raise RunnerError(f"Domain entry field drifted for {key}.")
            if entry["dataset_fingerprint"] != _json_string_member(
                materialized_payload, "dataset_fingerprint", label=key
            ):
                raise RunnerError(f"Domain entry fingerprint drifted for {key}.")
            fit_sequences = _load_token_sequences(
                run_root=run_root,
                model_name=model_name,
                domain_name=domain_name,
                partition="fit",
                expected_count=len(_chunk_list(materialized_payload, "fit")),
                expected_domain_payload=materialized_payload,
                expected_domain_sha256=materialized_sha256,
                expected_config_sha256=source_v1_config_sha256,
                expected_sweep_sha256=source_v1_sweep_sha256,
                expected_token_manifest_sha256=entry["fit_token_manifest_sha256"],
                expected_sequence_digest_sha256=entry["fit_sequence_digest_sha256"],
            )
            _ = _load_token_sequences(
                run_root=run_root,
                model_name=model_name,
                domain_name=domain_name,
                partition="eval",
                expected_count=len(_chunk_list(materialized_payload, "eval")),
                expected_domain_payload=materialized_payload,
                expected_domain_sha256=materialized_sha256,
                expected_config_sha256=source_v1_config_sha256,
                expected_sweep_sha256=source_v1_sweep_sha256,
                expected_token_manifest_sha256=entry["eval_token_manifest_sha256"],
                expected_sequence_digest_sha256=entry["eval_sequence_digest_sha256"],
            )
            fit_profile = _load_fit_profile(
                run_root=run_root,
                resolved_config=resolved_config,
                sweep=sweep,
                model_name=model_name,
                domain_name=domain_name,
                binding=binding,
                materialized_domain_sha256=materialized_sha256,
                fit_sequence_digest_sha256=entry["fit_sequence_digest_sha256"],
            )
            if fit_profile.payload["protocol_identity_sha256"] != protocol_identity_sha256:
                raise RunnerError(f"Fit profile protocol drifted: {model_name}.{domain_name}.json")
            if fit_profile.payload["fit_sequence_digest_sha256"] != _sequence_digest(fit_sequences):
                raise RunnerError(
                    f"Fit profile fit-sequence digest drifted: {model_name}.{domain_name}.json"
                )
            fit_profile_identity_sha256s.append(str(fit_profile.payload["identity_sha256"]))
    return manifest_paths, batch_manifest, fit_profile_identity_sha256s


def _immutable_input_manifest(
    *,
    run_root: RunRoot,
    resolved_config: ResolvedConfig,
    sweep: SweepConfig,
    project_root: Path,
    config_path: Path,
    sweep_path: Path,
    source_summary_v1_path: Path,
) -> JsonObject:
    manifest_paths = _require_production_manifest_inventory(
        run_root=run_root, resolved_config=resolved_config, sweep=sweep
    )
    shard_paths = _require_production_shard_inventory(run_root=run_root, sweep=sweep)
    if (
        2 + len(manifest_paths) + len(shard_paths) + 1
        != _EXPECTED_PRODUCTION_IMMUTABLE_INPUT_ENTRY_COUNT
    ):
        raise RunnerError("Production immutable-input manifest inventory shape drifted.")
    entries = [
        _manifest_entry(
            path,
            project_root=project_root,
            expected_byte_sha256=(
                _FIXED_PRODUCTION_CONFIG_RAW_BYTE_SHA256
                if path == config_path
                else _FIXED_PRODUCTION_SWEEP_RAW_BYTE_SHA256
                if path == sweep_path
                else _FIXED_PRODUCTION_SOURCE_V1_RAW_BYTE_SHA256
                if path == source_summary_v1_path
                else None
            ),
        )
        for path in sorted(
            (
                config_path,
                sweep_path,
                *manifest_paths,
                *shard_paths,
                source_summary_v1_path,
            ),
            key=lambda value: os.path.relpath(value, project_root),
        )
    ]
    if len(entries) != _EXPECTED_PRODUCTION_IMMUTABLE_INPUT_ENTRY_COUNT:
        raise RunnerError(
            "Production immutable-input manifest entry count drifted: expected "
            f"{_EXPECTED_PRODUCTION_IMMUTABLE_INPUT_ENTRY_COUNT}, found {len(entries)}."
        )
    content = _json_object(
        {
            "entry_count": _EXPECTED_PRODUCTION_IMMUTABLE_INPUT_ENTRY_COUNT,
            "manifest_file_count": len(manifest_paths),
            "shard_count": len(shard_paths),
            "entries": entries,
        },
        label="immutable_input_manifest_content",
    )
    payload = _json_object(
        {
            **content,
            "content_sha256": sha256_bytes(canonical_json_bytes(content)),
        },
        label="immutable_input_manifest",
    )
    payload["identity_sha256"] = artifact_identity(payload_without_identity(payload))
    return payload


def _require_fixed_production_immutable_input_manifest(payload: JsonObject) -> JsonObject:
    label = "immutable_input_manifest"
    if (
        _json_int_member(payload, "entry_count", label=label)
        != _EXPECTED_PRODUCTION_IMMUTABLE_INPUT_ENTRY_COUNT
    ):
        raise RunnerError("Immutable input manifest entry count drifted.")
    if (
        _json_int_member(payload, "manifest_file_count", label=label)
        != _EXPECTED_PRODUCTION_MANIFEST_FILE_COUNT
    ):
        raise RunnerError("Immutable input manifest file count drifted.")
    if _json_int_member(payload, "shard_count", label=label) != _EXPECTED_PRODUCTION_SHARD_COUNT:
        raise RunnerError("Immutable input manifest shard count drifted.")
    if (
        _json_string_member(payload, "content_sha256", label=label)
        != _FIXED_PRODUCTION_IMMUTABLE_INPUT_MANIFEST_CONTENT_SHA256
    ):
        raise RunnerError("Immutable input manifest content digest drifted.")
    if (
        _json_string_member(payload, "identity_sha256", label=label)
        != _FIXED_PRODUCTION_IMMUTABLE_INPUT_MANIFEST_IDENTITY_SHA256
    ):
        raise RunnerError("Immutable input manifest identity drifted.")
    return payload


def _publish_no_clobber(path: Path, payload_bytes: bytes) -> None:
    try:
        atomic_write_bytes_no_clobber(path, payload_bytes)
    except ArtifactError as exc:
        raise RunnerError(f"Failed to publish immutable artifact at {path}.") from exc


def _publish_terminal_summary_v2_no_clobber(
    *,
    path: Path,
    payload: JsonObject,
    payload_bytes: bytes,
    source_v1: JsonObject,
    root: RunRoot,
    resolved_config: ResolvedConfig,
    sweep: SweepConfig,
    project_root: Path,
    config_path: Path,
    sweep_path: Path,
    source_summary_v1_path: Path,
    expected_immutable_manifest: JsonObject,
) -> None:
    _require_fixed_production_config_raw_digest(config_path)
    _require_fixed_production_sweep_raw_digest(sweep_path)
    _require_fixed_production_source_v1_raw_digest(source_summary_v1_path)
    reverified_v1 = finalize_batch(
        config_path,
        sweep_path,
        run_root=root.root,
    )
    if reverified_v1 != source_v1:
        raise RunnerError("Legacy v1 terminal summary drifted before v2 publication.")
    rebuilt_manifest = _immutable_input_manifest(
        run_root=root,
        resolved_config=resolved_config,
        sweep=sweep,
        project_root=project_root,
        config_path=config_path,
        sweep_path=sweep_path,
        source_summary_v1_path=source_summary_v1_path,
    )
    rebuilt_manifest = _require_fixed_production_immutable_input_manifest(rebuilt_manifest)
    if canonical_json_bytes(rebuilt_manifest) != canonical_json_bytes(expected_immutable_manifest):
        raise RunnerError("Immutable input manifest drifted before publication.")
    try:
        atomic_write_bytes_no_clobber(path, payload_bytes)
    except ArtifactError as exc:
        collision_prefix = "Refusing to overwrite existing artifact:"
        if not str(exc).startswith(collision_prefix):
            raise RunnerError(f"Failed to publish immutable artifact at {path}.") from exc
        if not path.exists():
            raise RunnerError(f"Failed to publish immutable artifact at {path}.") from exc
        winner_bytes = path.read_bytes()
        if winner_bytes != payload_bytes:
            raise RunnerError(
                "Finalizer v2 collision produced non-identical terminal summary."
            ) from exc
        existing = _load_existing_terminal_payload(path, label=path.name)
        if existing != payload:
            raise RunnerError("Finalizer v2 collision payload drifted.") from exc


def _require_exact_relative_cli_path(path: str | Path, *, expected: Path, label: str) -> None:
    if Path(path).as_posix() != expected.as_posix():
        raise RunnerError(f"{label} must be the canonical production path {expected.as_posix()}.")


def _require_canonical_production_source_v1_file(path: Path) -> None:
    if not path.is_file():
        raise RunnerError("Canonical production source-v1 path is missing or not a regular file.")


def _production_repo_root_from_project_root(project_root: Path) -> Path:
    canonical_project_root = project_root.resolve()
    repo_root = canonical_project_root.parent
    canonical_config_dir = (canonical_project_root / "configs").resolve()
    expected_config_dir = (repo_root / _PRODUCTION_CONFIG_RELATIVE_PATH.parent).resolve()
    if expected_config_dir != canonical_config_dir:
        raise RunnerError("Resolved project_root is not the canonical rebuttal package root.")
    config_path = (repo_root / _PRODUCTION_CONFIG_RELATIVE_PATH).resolve()
    sweep_path = (repo_root / _PRODUCTION_SWEEP_RELATIVE_PATH).resolve()
    if not config_path.is_file():
        raise RunnerError("Canonical production config path is missing.")
    if not sweep_path.is_file():
        raise RunnerError("Canonical production sweep path is missing.")
    return repo_root


def _finalize_toy_batch_v2(
    base_path: str | Path, sweep_path: str | Path, *, run_root: Path
) -> JsonObject:
    resolved_config, sweep = load_configs(base_path, sweep_path)
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id="toy")
    batch_manifest = _load_batch_manifest(root)
    expected = {
        key.stem(): key
        for model_name in sweep.models
        for key in _expected_condition_inventory(model_name, include_depth=sweep.include_depth)
    }
    actual_paths = {path.stem: path for path in root.shards_dir.glob("*.json")}
    payload = _build_terminal_payload(
        root=root,
        resolved_config=resolved_config,
        sweep=sweep,
        batch_manifest=batch_manifest,
        expected=expected,
        actual_paths=actual_paths,
        schema_version=2,
    )
    summary_path = _summary_v2_path(root)
    if summary_path.exists():
        existing = _load_existing_terminal_payload(summary_path, label=summary_path.name)
        payload["completed_at"] = _json_string_member(
            existing, "completed_at", label=summary_path.name
        )
    else:
        payload["completed_at"] = utc_now()
    payload["identity_sha256"] = _terminal_payload_identity(payload)
    expected_bytes = _terminal_payload_bytes(payload)
    if summary_path.exists():
        if summary_path.read_bytes() != expected_bytes:
            raise RunnerError(f"Immutable artifact conflict at {summary_path}.")
        return _load_existing_terminal_payload(summary_path, label=summary_path.name)
    _publish_no_clobber(summary_path, expected_bytes)
    return _load_existing_terminal_payload(summary_path, label=summary_path.name)


def _parse_fixed_unit_index(unit_index: str | int, *, unit: str) -> int:
    raw = str(unit_index)
    if not raw.isdecimal():
        raise RunnerError(f"{unit} unit_index must be a non-negative decimal: {raw!r}")
    value = int(raw)
    upper_bound = FIXED_COUNTS["head_bins"] if unit == "bin" else 32 if unit == "layer" else None
    if upper_bound is None:
        raise RunnerError(f"Unsupported fixed unit for integer validation: {unit}")
    if value >= upper_bound:
        raise RunnerError(f"{unit} unit_index out of range [0, {upper_bound}): {raw!r}")
    return value


def _expected_condition_inventory(model_name: str, *, include_depth: bool) -> list[ConditionKey]:
    out: list[ConditionKey] = []
    for direction in TOY_DIRECTIONS:
        out.append(
            ConditionKey(
                model=model_name,
                direction=direction,
                unit="global",
                unit_index="baseline",
                condition="baseline",
            )
        )
        for bin_index in range(NUM_HEAD_BINS):
            out.append(
                ConditionKey(
                    model=model_name,
                    direction=direction,
                    unit="bin",
                    unit_index=str(bin_index),
                    condition="source_kernel",
                )
            )
            out.append(
                ConditionKey(
                    model=model_name,
                    direction=direction,
                    unit="bin",
                    unit_index=str(bin_index),
                    condition="target_kernel",
                )
            )
            for trial in range(3):
                out.append(
                    ConditionKey(
                        model=model_name,
                        direction=direction,
                        unit="bin",
                        unit_index=str(bin_index),
                        condition="offset_permutation",
                        trial=trial,
                    )
                )
                out.append(
                    ConditionKey(
                        model=model_name,
                        direction=direction,
                        unit="bin",
                        unit_index=str(bin_index),
                        condition="norm",
                        trial=trial,
                    )
                )
        if include_depth and direction == "wikipedia_to_code":
            for layer_index in range(TOY_LAYER_COUNT):
                out.append(
                    ConditionKey(
                        model=model_name,
                        direction=direction,
                        unit="layer",
                        unit_index=str(layer_index),
                        condition="depth",
                    )
                )
    return out


def toy_smoke(base_path: str | Path, sweep_path: str | Path, *, run_root: Path) -> ToyRunSummary:
    resolved_config, sweep = load_configs(base_path, sweep_path)
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id="toy")
    root.ensure()
    finalizer_ready = _manifest_paths(root)["batch"].exists()
    if finalizer_ready:
        terminal_payload = _finalize_toy_batch_v2(base_path, sweep_path, run_root=run_root)
        return ToyRunSummary(
            run_id=root.run_id,
            terminal_summary_path=_summary_v2_path(root),
            shard_count=_json_int_member(terminal_payload, "shards_verified", label="terminal"),
            statistic_count=len(
                _json_array_member(terminal_payload, "dose_response_statistics", label="terminal")
            ),
        )
    stats: list[JsonObject] = []
    flat_scores = np.linspace(0.0, 1.0, TOY_LAYER_COUNT * TOY_HEAD_COUNT, dtype=np.float64).reshape(
        TOY_LAYER_COUNT, TOY_HEAD_COUNT
    )
    raw_kernels = np.zeros((TOY_LAYER_COUNT, TOY_HEAD_COUNT, TOY_SEQUENCE_LENGTH), dtype=np.float64)
    offsets = np.arange(TOY_SEQUENCE_LENGTH, dtype=np.float64)
    toy_tokenizers = {
        model_name: artifact_identity({"model": model_name, "tokenizer": "toy"})
        for model_name in sweep.models
    }
    toy_models = {
        model_name: artifact_identity({"model": model_name, "weights": "toy"})
        for model_name in sweep.models
    }
    toy_domain_entries: dict[str, DomainEntryPayload] = {
        _domain_entry_name(model_name, domain_name): {
            "model": model_name,
            "domain": domain_name,
            "tokenizer_tree_sha256": toy_tokenizers[model_name],
            "materialized_domain_sha256": artifact_identity(
                {"model": model_name, "domain": domain_name, "materialized": "toy"}
            ),
            "dataset_repository": "toy",
            "dataset_revision": "toy",
            "dataset_config": "toy",
            "dataset_split": "toy",
            "dataset_field": "toy",
            "dataset_fingerprint": "toy",
            "fit_token_manifest_sha256": artifact_identity(
                {"model": model_name, "domain": domain_name, "partition": "fit", "toy": True}
            ),
            "eval_token_manifest_sha256": artifact_identity(
                {"model": model_name, "domain": domain_name, "partition": "eval", "toy": True}
            ),
            "fit_sequence_digest_sha256": artifact_identity(
                {"model": model_name, "domain": domain_name, "partition": "fit", "toy": True}
            ),
            "eval_sequence_digest_sha256": artifact_identity(
                {"model": model_name, "domain": domain_name, "partition": "eval", "toy": True}
            ),
        }
        for model_name in sweep.models
        for domain_name in resolved_config.datasets
    }
    toy_manifest = _batch_manifest_payload(
        _normalized_json_object(
            {
                "schema_version": 1,
                "materialized_at_utc": utc_now(),
                "command": ["toy-smoke"],
                "config_sha256": resolved_config.identity_sha256,
                "sweep_sha256": sweep.identity_sha256,
                "domains": {
                    key: value["materialized_domain_sha256"]
                    for key, value in sorted(toy_domain_entries.items())
                },
                "domain_entries": {
                    key: dict(value) for key, value in sorted(toy_domain_entries.items())
                },
                "tokenizers": toy_tokenizers,
                "models": toy_models,
                "git_commit_sha": "toy",
                "git_tracked_diff_sha256": "toy",
                "git_relevant_content_sha256": "toy",
                "git_untracked_sha256": "toy",
                "runtime": {"toy": True},
                "runtime_capture": {"toy": True},
                "provenance": {"toy": True},
                "environment": {"SI_REBUTTAL_TOY": "1"},
            },
            label="toy_manifest",
        ),
        label="toy_manifest",
    )
    for layer_index in range(TOY_LAYER_COUNT):
        for head_index in range(TOY_HEAD_COUNT):
            raw_kernels[layer_index, head_index] = (
                0.001 * (1 + layer_index) + 0.0001 * (1 + head_index) + offsets / 100000.0
            )
    toy_fit_profiles: dict[tuple[str, str], FrozenFitProfile] = {}
    for model_name in sweep.models:
        binding = freeze_model_binding(resolved_config.models[model_name])
        for domain_name in resolved_config.datasets:
            entry = toy_domain_entries[_domain_entry_name(model_name, domain_name)]
            profile_payload = _serialize_fit_profile(
                resolved_config=resolved_config,
                sweep=sweep,
                model_name=model_name,
                domain_name=domain_name,
                binding=binding,
                materialized_domain_sha256=str(entry["materialized_domain_sha256"]),
                fit_sequence_digest_sha256=str(entry["fit_sequence_digest_sha256"]),
                summary=DomainFitSummary(mean_scores=flat_scores, raw_kernels=raw_kernels),
            )
            load_existing_or_write(
                _fit_profile_path(root, model_name=model_name, domain_name=domain_name),
                profile_payload,
            )
            toy_fit_profiles[(model_name, domain_name)] = _load_fit_profile(
                run_root=root,
                resolved_config=resolved_config,
                sweep=sweep,
                model_name=model_name,
                domain_name=domain_name,
                binding=binding,
                materialized_domain_sha256=str(entry["materialized_domain_sha256"]),
                fit_sequence_digest_sha256=str(entry["fit_sequence_digest_sha256"]),
            )
    toy_depth_baseline_identities: dict[str, str] = {}
    for model_name in sweep.models:
        for direction_index, direction in enumerate(TOY_DIRECTIONS):
            source_domain, target_domain = _direction_domains(direction)
            source_entry = toy_domain_entries[_domain_entry_name(model_name, source_domain)]
            target_entry = toy_domain_entries[_domain_entry_name(model_name, target_domain)]
            source_profile = toy_fit_profiles[(model_name, source_domain)]
            target_profile = toy_fit_profiles[(model_name, target_domain)]
            bins = source_profile.bins
            bin_means = source_profile.bin_means
            baseline = _toy_sequence_losses(
                baseline=2.0 + direction_index * 0.1,
                strength=0.0,
                count=FIXED_COUNTS["eval_sequences"],
            )
            base_key = ConditionKey(
                model=model_name,
                direction=direction,
                unit="global",
                unit_index="baseline",
                condition="baseline",
            )
            baseline_payload = _write_shard(
                root,
                base_key,
                {
                    "schema_version": 1,
                    "model": model_name,
                    "direction": direction,
                    "unit": "global",
                    "unit_index": "baseline",
                    "condition": "baseline",
                    "head_count": 0,
                    "sequence_nll": baseline.tolist(),
                    "sequence_digest_sha256": target_entry["eval_sequence_digest_sha256"],
                    "source_domain_manifest_sha256": source_entry["materialized_domain_sha256"],
                    "target_domain_manifest_sha256": target_entry["materialized_domain_sha256"],
                    "weights_tree_sha256": toy_models[model_name],
                    "tokenizer_tree_sha256": toy_tokenizers[model_name],
                    "config_sha256": resolved_config.identity_sha256,
                    "sweep_sha256": sweep.identity_sha256,
                    "batch_manifest_sha256": toy_manifest["identity_sha256"],
                    "source_fit_profile_sha256": source_profile.payload["identity_sha256"],
                    "target_fit_profile_sha256": target_profile.payload["identity_sha256"],
                },
            )
            direction_baseline_identity = str(baseline_payload["identity_sha256"])
            if direction == "wikipedia_to_code":
                toy_depth_baseline_identities[model_name] = direction_baseline_identity
            responses: list[float] = []
            for bin_index, heads in enumerate(bins):
                source_nll = _toy_sequence_losses(
                    baseline=2.0 + direction_index * 0.1,
                    strength=0.01 + bin_index * 0.002,
                    count=FIXED_COUNTS["eval_sequences"],
                )
                target_nll = _toy_sequence_losses(
                    baseline=2.0 + direction_index * 0.1,
                    strength=0.006 + bin_index * 0.001,
                    count=FIXED_COUNTS["eval_sequences"],
                )
                responses.append(float(np.mean(source_nll - baseline) / len(heads)))
                for condition_name, values in (
                    ("source_kernel", source_nll),
                    ("target_kernel", target_nll),
                ):
                    key = ConditionKey(
                        model=model_name,
                        direction=direction,
                        unit="bin",
                        unit_index=str(bin_index),
                        condition=condition_name,
                    )
                    _write_shard(
                        root,
                        key,
                        {
                            "schema_version": 1,
                            "model": model_name,
                            "direction": direction,
                            "unit": "bin",
                            "unit_index": str(bin_index),
                            "condition": condition_name,
                            "head_count": len(heads),
                            "mean_source_r2": float(bin_means[bin_index]),
                            "sequence_nll": values.tolist(),
                            "sequence_digest_sha256": baseline_payload["sequence_digest_sha256"],
                            "source_domain_manifest_sha256": source_entry[
                                "materialized_domain_sha256"
                            ],
                            "target_domain_manifest_sha256": target_entry[
                                "materialized_domain_sha256"
                            ],
                            "baseline_identity_sha256": baseline_payload["identity_sha256"],
                            "weights_tree_sha256": toy_models[model_name],
                            "tokenizer_tree_sha256": toy_tokenizers[model_name],
                            "config_sha256": resolved_config.identity_sha256,
                            "sweep_sha256": sweep.identity_sha256,
                            "batch_manifest_sha256": toy_manifest["identity_sha256"],
                            "source_fit_profile_sha256": source_profile.payload["identity_sha256"],
                            "target_fit_profile_sha256": target_profile.payload["identity_sha256"],
                            "selected_heads": _head_records(heads),
                            "selected_head_digest_sha256": _selected_head_digest(heads),
                            "selected_bin_digest_sha256": source_profile.payload["bins"][bin_index][
                                "identity_sha256"
                            ],
                            "source_kernel_digest_sha256": _kernel_digest(
                                _kernel_map_for_heads(heads, source_profile.raw_kernels).items()
                            ),
                            "target_kernel_digest_sha256": _kernel_digest(
                                _kernel_map_for_heads(heads, target_profile.raw_kernels).items()
                            ),
                        },
                    )
                for control_kind in ("offset_permutation", "norm"):
                    trials: list[Float64Array] = []
                    for trial in range(3):
                        seed = compact_ascii_json_seed(
                            [29039, model_name, direction, "bin", bin_index, trial, control_kind]
                        )
                        control_map = _streaming_control_maps(
                            heads,
                            _kernel_map_for_heads(heads, source_profile.raw_kernels),
                            seed=seed,
                            control_kind=control_kind,
                        )
                        control_nll = _toy_sequence_losses(
                            baseline=2.0 + direction_index * 0.1,
                            strength=0.004 + bin_index * 0.001 + trial * 1e-4,
                            count=FIXED_COUNTS["eval_sequences"],
                        )
                        trials.append(control_nll)
                        key = ConditionKey(
                            model=model_name,
                            direction=direction,
                            unit="bin",
                            unit_index=str(bin_index),
                            condition=control_kind,
                            trial=trial,
                        )
                        _write_shard(
                            root,
                            key,
                            {
                                "schema_version": 1,
                                "model": model_name,
                                "direction": direction,
                                "unit": "bin",
                                "unit_index": str(bin_index),
                                "condition": control_kind,
                                "trial": trial,
                                "control_kind": control_kind,
                                "control_seed": int(seed),
                                "head_count": len(heads),
                                "mean_source_r2": float(bin_means[bin_index]),
                                "sequence_nll": control_nll.tolist(),
                                "sequence_digest_sha256": baseline_payload[
                                    "sequence_digest_sha256"
                                ],
                                "source_domain_manifest_sha256": source_entry[
                                    "materialized_domain_sha256"
                                ],
                                "target_domain_manifest_sha256": target_entry[
                                    "materialized_domain_sha256"
                                ],
                                "baseline_identity_sha256": baseline_payload["identity_sha256"],
                                "weights_tree_sha256": toy_models[model_name],
                                "tokenizer_tree_sha256": toy_tokenizers[model_name],
                                "config_sha256": resolved_config.identity_sha256,
                                "sweep_sha256": sweep.identity_sha256,
                                "batch_manifest_sha256": toy_manifest["identity_sha256"],
                                "source_fit_profile_sha256": source_profile.payload[
                                    "identity_sha256"
                                ],
                                "target_fit_profile_sha256": target_profile.payload[
                                    "identity_sha256"
                                ],
                                "selected_heads": _head_records(heads),
                                "selected_head_digest_sha256": _selected_head_digest(heads),
                                "selected_bin_digest_sha256": source_profile.payload["bins"][
                                    bin_index
                                ]["identity_sha256"],
                                "source_kernel_digest_sha256": _kernel_digest(
                                    _kernel_map_for_heads(heads, source_profile.raw_kernels).items()
                                ),
                                "control_kernel_digest_sha256": _kernel_digest(control_map.items()),
                            },
                        )
                    avg = average_control_trials_per_sequence(np.stack(trials, axis=0))
                    contrast = paired_percentile_bootstrap(
                        baseline,
                        avg,
                        seed=compact_json_seed(
                            [29039, model_name, direction, "statistic", "all", 0, control_kind]
                        ),
                    )
                    stats.append(
                        _json_object(
                            {
                                "model": model_name,
                                "direction": direction,
                                "condition": control_kind,
                                "bin": bin_index,
                                "point_estimate": contrast.point_estimate,
                            },
                            label=f"toy_stats.{model_name}.{direction}.{control_kind}.{bin_index}",
                        )
                    )
            spearman = monte_carlo_spearman_positive(
                bin_means,
                np.asarray(responses, dtype=np.float64),
                seed=compact_json_seed(
                    [
                        29039,
                        model_name,
                        direction,
                        "statistic",
                        "all",
                        0,
                        "spearman_response_permutation",
                    ]
                ),
            )
            stats.append(
                _json_object(
                    {
                        "model": model_name,
                        "direction": direction,
                        "condition": "spearman",
                        "rho": spearman.rho_observed,
                        "p_one_sided": spearman.p_one_sided,
                    },
                    label=f"toy_stats.{model_name}.{direction}.spearman",
                )
            )
        for layer_index in range(TOY_LAYER_COUNT):
            key = ConditionKey(
                model=model_name,
                direction="wikipedia_to_code",
                unit="layer",
                unit_index=str(layer_index),
                condition="depth",
            )
            _write_shard(
                root,
                key,
                {
                    "schema_version": 1,
                    "model": model_name,
                    "direction": "wikipedia_to_code",
                    "unit": "layer",
                    "unit_index": str(layer_index),
                    "condition": "depth",
                    "head_count": 32,
                    "mean_source_r2": float(
                        np.asarray(flat_scores[layer_index], dtype=np.float64).mean()
                    ),
                    "sequence_nll": _toy_sequence_losses(
                        baseline=2.2,
                        strength=0.004 + layer_index * 2e-4,
                        count=FIXED_COUNTS["eval_sequences"],
                    ).tolist(),
                    "sequence_digest_sha256": toy_domain_entries[
                        _domain_entry_name(model_name, "code")
                    ]["eval_sequence_digest_sha256"],
                    "source_domain_manifest_sha256": toy_domain_entries[
                        _domain_entry_name(model_name, "wikipedia")
                    ]["materialized_domain_sha256"],
                    "target_domain_manifest_sha256": toy_domain_entries[
                        _domain_entry_name(model_name, "code")
                    ]["materialized_domain_sha256"],
                    "baseline_identity_sha256": toy_depth_baseline_identities[model_name],
                    "weights_tree_sha256": toy_models[model_name],
                    "tokenizer_tree_sha256": toy_tokenizers[model_name],
                    "config_sha256": resolved_config.identity_sha256,
                    "sweep_sha256": sweep.identity_sha256,
                    "batch_manifest_sha256": toy_manifest["identity_sha256"],
                    "source_fit_profile_sha256": toy_fit_profiles[
                        (model_name, "wikipedia")
                    ].payload["identity_sha256"],
                    "target_fit_profile_sha256": toy_fit_profiles[(model_name, "code")].payload[
                        "identity_sha256"
                    ],
                    "selected_heads": _head_records(
                        tuple(HeadIndex(layer=layer_index, head=head) for head in range(32))
                    ),
                    "selected_head_digest_sha256": _selected_head_digest(
                        tuple(HeadIndex(layer=layer_index, head=head) for head in range(32))
                    ),
                    "source_kernel_digest_sha256": _kernel_digest(
                        _kernel_map_for_heads(
                            tuple(HeadIndex(layer=layer_index, head=head) for head in range(32)),
                            toy_fit_profiles[(model_name, "wikipedia")].raw_kernels,
                        ).items()
                    ),
                },
            )
    load_existing_or_write(
        _manifest_paths(root)["batch"], _json_object(toy_manifest, label="toy_manifest")
    )
    terminal_payload = _finalize_toy_batch_v2(base_path, sweep_path, run_root=run_root)
    return ToyRunSummary(
        run_id=root.run_id,
        terminal_summary_path=_summary_v2_path(root),
        shard_count=_json_int_member(terminal_payload, "shards_verified", label="terminal_payload"),
        statistic_count=len(
            _json_array_member(
                terminal_payload, "dose_response_statistics", label="terminal_payload"
            )
        ),
    )


def _expected_call_counts(sequence_count: int) -> ExpectedAttentionCallCounts:
    return ExpectedAttentionCallCounts(
        {layer_index: int(sequence_count) for layer_index in range(32)}
    )


def _validation_receipt_payload(
    *,
    model_name: str,
    binding: FrozenModelBinding,
    config_sha256: str,
    sweep_sha256: str,
    batch_manifest_sha256: str,
    validation_domain: str,
    source_domain_manifest_sha256: str,
    fit_token_manifest_sha256: str,
    fit_sequence_digest_sha256: str,
    zero_nll: float,
    baseline_nll: float,
    constant_nll: float,
    nonconstant_nll: float,
) -> JsonObject:
    payload: JsonObject = {
        "schema_version": 1,
        "model": model_name,
        "config_sha256": config_sha256,
        "sweep_sha256": sweep_sha256,
        "batch_manifest_sha256": batch_manifest_sha256,
        "validation_domain": validation_domain,
        "source_domain_manifest_sha256": source_domain_manifest_sha256,
        "fit_token_manifest_sha256": fit_token_manifest_sha256,
        "fit_sequence_digest_sha256": fit_sequence_digest_sha256,
        "weights_tree_sha256": binding.weights_tree_sha256,
        "tokenizer_tree_sha256": binding.tokenizer_tree_sha256,
        "baseline_nll": baseline_nll,
        "zero_nll": zero_nll,
        "constant_nll": constant_nll,
        "nonconstant_nll": nonconstant_nll,
        "signed_by_digests": [
            binding.weights_tree_sha256,
            binding.tokenizer_tree_sha256,
            config_sha256,
            sweep_sha256,
            batch_manifest_sha256,
        ],
    }
    payload["identity_sha256"] = artifact_identity(payload)
    return payload


def validate_model(
    base_path: str | Path,
    sweep_path: str | Path,
    *,
    model_name: str,
    run_root: Path | None = None,
) -> JsonObject:
    resolved_config, sweep = load_configs(base_path, sweep_path)
    if run_root is None:
        raise RunnerError("validate-model requires the already-materialized immutable run root.")
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id=None)
    batch_manifest = _load_batch_manifest(root)
    sequences = _load_sequences_from_run_root(
        root, resolved_config, sweep, model_name, domain_name="wikipedia"
    )
    validation_domain_entry = batch_manifest["domain_entries"][
        _domain_entry_name(model_name, "wikipedia")
    ]
    logical_device = _require_preflight_model_visibility(sweep, model_name=model_name)
    bundle = load_local_model_bundle(
        resolved_config, model_name=model_name, local_files_only=True, logical_device=logical_device
    )
    token_ids = torch.as_tensor(
        sequences.fit_tokens[0][:64], dtype=torch.long, device=bundle.logical_device
    )
    baseline_logits = _call_model(
        bundle.model,
        input_ids=token_ids.unsqueeze(0),
        attention_mask=torch.ones((1, 64), device=bundle.logical_device, dtype=torch.long),
    )
    baseline_nll = float(mean_next_token_nll(baseline_logits, token_ids).item())
    selected_heads = {0: (0,)}
    expected_calls = _expected_call_counts(1)
    zero_kernel = np.zeros(512, dtype=np.float64)
    constant_kernel = np.full(512, 0.25, dtype=np.float64)
    nonconstant_kernel = np.linspace(0.0, 0.5, 512, dtype=np.float64)
    probe = ValidationProbeRequest(layer_index=0, head_index=0)

    zero_intervention = bundle.entry.intervention_factory(
        bundle.model,
        expected_call_counts=expected_calls,
        selected_heads_by_layer=selected_heads,
        kernels_by_layer_head={(0, 0): zero_kernel.tolist()},
        validation_probe=probe,
    )
    with zero_intervention:
        zero_logits = _call_model(
            bundle.model,
            input_ids=token_ids.unsqueeze(0),
            attention_mask=torch.ones((1, 64), device=bundle.logical_device, dtype=torch.long),
        )
    zero_record = zero_intervention.latest_validation_record()
    if zero_record is None:
        raise RunnerError(
            f"Validation probe did not capture the real eager attention record for {model_name}."
        )
    zero_nll = float(mean_next_token_nll(zero_logits, token_ids).item())
    if not torch.allclose(
        baseline_logits,
        zero_logits,
        atol=FIXED_VALIDATION["zero_kernel_atol"],
        rtol=FIXED_VALIDATION["zero_kernel_rtol"],
    ):
        raise RunnerError(f"Zero-kernel token logits drifted for {model_name}.")
    if abs(zero_nll - baseline_nll) > FIXED_VALIDATION["zero_kernel_nll_abs"]:
        raise RunnerError(f"Zero-kernel NLL drifted for {model_name}.")

    constant_intervention = bundle.entry.intervention_factory(
        bundle.model,
        expected_call_counts=expected_calls,
        selected_heads_by_layer=selected_heads,
        kernels_by_layer_head={(0, 0): constant_kernel.tolist()},
        validation_probe=probe,
    )
    with constant_intervention:
        constant_logits = _call_model(
            bundle.model,
            input_ids=token_ids.unsqueeze(0),
            attention_mask=torch.ones((1, 64), device=bundle.logical_device, dtype=torch.long),
        )
    constant_record = constant_intervention.latest_validation_record()
    if constant_record is None:
        raise RunnerError(
            "Constant-offset validation probe did not capture attention probabilities "
            f"for {model_name}."
        )
    constant_nll = float(mean_next_token_nll(constant_logits, token_ids).item())
    if not torch.allclose(
        zero_record.attention_probabilities,
        constant_record.attention_probabilities,
        atol=FIXED_VALIDATION["constant_kernel_atol"],
        rtol=FIXED_VALIDATION["constant_kernel_rtol"],
    ):
        raise RunnerError(f"Constant-offset softmax invariance failed for {model_name}.")
    if abs(constant_nll - baseline_nll) > FIXED_VALIDATION["constant_kernel_nll_abs"]:
        raise RunnerError(f"Constant-offset NLL drifted for {model_name}.")

    nonconstant_intervention = bundle.entry.intervention_factory(
        bundle.model,
        expected_call_counts=expected_calls,
        selected_heads_by_layer=selected_heads,
        kernels_by_layer_head={(0, 0): nonconstant_kernel.tolist()},
        validation_probe=probe,
    )
    with nonconstant_intervention:
        nonconstant_logits = _call_model(
            bundle.model,
            input_ids=token_ids.unsqueeze(0),
            attention_mask=torch.ones((1, 64), device=bundle.logical_device, dtype=torch.long),
        )
    nonconstant_record = nonconstant_intervention.latest_validation_record()
    if nonconstant_record is None:
        raise RunnerError(
            "Nonconstant validation probe did not capture a real attention record "
            f"for {model_name}."
        )
    nonconstant_nll = float(mean_next_token_nll(nonconstant_logits, token_ids).item())
    expected_probs = nonconstant_record.runtime_faithful_attention_probabilities()[
        0, nonconstant_record.head_index
    ].to(dtype=torch.float32)
    if not torch.allclose(
        expected_probs,
        nonconstant_record.attention_probabilities,
        atol=FIXED_VALIDATION["attention_reconstruction_atol"],
        rtol=FIXED_VALIDATION["attention_reconstruction_rtol"],
    ):
        raise RunnerError(f"Manual probability reconstruction failed for {model_name}.")
    actual_delta = nonconstant_record.post_mask - nonconstant_record.pre_mask
    toeplitz = torch.zeros_like(actual_delta)
    for query_index in range(64):
        for key_index in range(query_index + 1):
            toeplitz[query_index, key_index] = -float(nonconstant_kernel[query_index - key_index])
    nonfinite_mask = ~torch.isfinite(nonconstant_record.pre_mask) | ~torch.isfinite(
        nonconstant_record.post_mask
    )
    if nonfinite_mask.any():
        if not torch.all(torch.isneginf(nonconstant_record.pre_mask[nonfinite_mask])):
            raise RunnerError(
                f"Masked baseline entries must remain negative infinity for {model_name}."
            )
        if not torch.all(torch.isneginf(nonconstant_record.post_mask[nonfinite_mask])):
            raise RunnerError(
                f"Masked corrected entries must remain negative infinity for {model_name}."
            )
    finite_mask = torch.isfinite(actual_delta)
    if (
        torch.max(torch.abs(actual_delta[finite_mask] - toeplitz[finite_mask])).item()
        > FIXED_VALIDATION["toeplitz_max_abs"]
    ):
        raise RunnerError(f"Selected-head Toeplitz correction delta failed for {model_name}.")
    full_correction = nonconstant_intervention.build_layer_correction(
        layer_index=0, num_query_heads=32, seq_len=64, device=actual_delta.device
    )
    nonselected = torch.cat((full_correction[:0], full_correction[1:]), dim=0)
    if torch.count_nonzero(nonselected).item() != 0:
        raise RunnerError(
            f"Non-selected query heads received a non-zero correction for {model_name}."
        )
    receipt = _validation_receipt_payload(
        model_name=model_name,
        binding=bundle.binding,
        config_sha256=resolved_config.identity_sha256,
        sweep_sha256=sweep.identity_sha256,
        batch_manifest_sha256=str(batch_manifest["identity_sha256"]),
        validation_domain="wikipedia",
        source_domain_manifest_sha256=str(validation_domain_entry["materialized_domain_sha256"]),
        fit_token_manifest_sha256=str(validation_domain_entry["fit_token_manifest_sha256"]),
        fit_sequence_digest_sha256=str(validation_domain_entry["fit_sequence_digest_sha256"]),
        baseline_nll=baseline_nll,
        zero_nll=zero_nll,
        constant_nll=constant_nll,
        nonconstant_nll=nonconstant_nll,
    )
    load_existing_or_write(_receipt_path(root, f"validation.{model_name}"), receipt)
    return receipt


def _reconstruct_sequences_for_materialized_domain(
    resolved_config: ResolvedConfig,
    model_name: str,
    *,
    domain_name: str,
    materialized: MaterializedDomain,
    load_dataset_fn: DatasetLoaderFn | None = None,
    tokenizer_loader: TokenizerLoader | None = None,
) -> tuple[tuple[Int64Array, ...], tuple[Int64Array, ...], MaterializedDomain]:
    if load_dataset_fn is None:
        load_dataset_fn = _default_load_dataset
    dataset_loader = ImmutableHFDatasetLoader(load_dataset_fn)
    model_config = resolved_config.models[model_name]
    tokenizer = _load_tokenizer(model_config, tokenizer_loader)

    def encode_text(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    dataset = resolved_config.datasets[domain_name]
    loaded = dataset_loader.load(dataset)
    if domain_name == "wikipedia":
        documents = build_wikipedia_documents(
            tuple(str(row.get(dataset.field, "") or "") for row in loaded.rows), dataset.revision
        )
    else:
        documents = build_code_documents(
            loaded.rows,
            dataset.revision,
            text_field=dataset.field,
            repository_field=dataset.repository_field or "repository_name",
        )
    ordered = {
        "fit": [document for document in documents if document.partition == "fit"],
        "eval": [document for document in documents if document.partition == "eval"],
    }
    for partition in ordered:
        ordered[partition].sort(
            key=lambda document: (document.assignment_sha256, document.document_id)
        )

    def collect(target_chunks: Sequence[SelectedChunk], partition: str) -> tuple[Int64Array, ...]:
        values: list[Int64Array] = []
        doc_lookup = {document.document_id: document for document in ordered[partition]}
        for chunk in target_chunks:
            document = doc_lookup[chunk.document_id]
            tokens = np.asarray(
                tuple(int(token) for token in encode_text(document.text)), dtype=np.int64
            )
            start = chunk.chunk_index * 512
            end = start + 512
            piece = tokens[start:end]
            if piece.shape != (512,):
                raise RunnerError(
                    "Failed to reconstruct the exact 512-token chunk for "
                    f"{model_name}:{domain_name}:{partition}."
                )
            values.append(piece)
        return tuple(values)

    fit = collect(materialized.fit_chunks, "fit")
    eval_values = collect(materialized.eval_chunks, "eval")
    return fit, eval_values, materialized


def _load_sequences_from_run_root(
    run_root: RunRoot,
    resolved_config: ResolvedConfig,
    sweep: SweepConfig,
    model_name: str,
    *,
    domain_name: str,
) -> DomainSequences:
    payloads = _materialized_domain_payloads(run_root)
    batch_domain_entries = _batch_domain_entries(run_root)
    key = _domain_entry_name(model_name, domain_name)
    if key not in payloads:
        raise RunnerError(f"Batch manifest is missing materialized domain {key}.")
    if key not in batch_domain_entries:
        raise RunnerError(f"Batch manifest is missing immutable domain entry {key}.")
    expected = payloads[key]
    domain_entry = batch_domain_entries[key]
    if domain_entry["materialized_domain_sha256"] != str(expected["manifest_sha256"]):
        raise RunnerError(f"Batch manifest materialized domain drifted for {key}.")
    fit = _load_token_sequences(
        run_root=run_root,
        model_name=model_name,
        domain_name=domain_name,
        partition="fit",
        expected_count=FIXED_COUNTS["fit_sequences"],
        expected_domain_payload=expected,
        expected_domain_sha256=str(expected["manifest_sha256"]),
        expected_config_sha256=resolved_config.identity_sha256,
        expected_sweep_sha256=sweep.identity_sha256,
        expected_token_manifest_sha256=str(domain_entry["fit_token_manifest_sha256"]),
        expected_sequence_digest_sha256=str(domain_entry["fit_sequence_digest_sha256"]),
    )
    eval_values = _load_token_sequences(
        run_root=run_root,
        model_name=model_name,
        domain_name=domain_name,
        partition="eval",
        expected_count=FIXED_COUNTS["eval_sequences"],
        expected_domain_payload=expected,
        expected_domain_sha256=str(expected["manifest_sha256"]),
        expected_config_sha256=resolved_config.identity_sha256,
        expected_sweep_sha256=sweep.identity_sha256,
        expected_token_manifest_sha256=str(domain_entry["eval_token_manifest_sha256"]),
        expected_sequence_digest_sha256=str(domain_entry["eval_sequence_digest_sha256"]),
    )
    return DomainSequences(fit_tokens=fit, eval_tokens=eval_values)


def _streaming_control_maps(
    heads: Sequence[HeadIndex],
    source_map: HeadKernelMap,
    *,
    seed: int,
    control_kind: str,
) -> HeadKernelMap:
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    control_map: HeadKernelMap = {}
    for head in sorted(heads):
        kernel = np.asarray(source_map[(head.layer, head.head)], dtype=np.float64)
        if control_kind == "offset_permutation":
            control_map[(head.layer, head.head)] = kernel[rng.permutation(kernel.shape[0])]
            continue
        target_norm = float(np.linalg.norm(kernel))
        if target_norm == 0.0:
            raise RunnerError("zero kernel norm is a hard failure.")
        draw = rng.standard_normal(kernel.shape[0]).astype(np.float64, copy=False)
        draw -= draw.mean()
        draw_norm = float(np.linalg.norm(draw))
        if draw_norm == 0.0:
            raise RunnerError("random control draw had zero norm.")
        control_map[(head.layer, head.head)] = draw * (target_norm / draw_norm)
    return control_map


def _stream_post_rope_qk(
    bundle: LoadedModelBundle,
    token_ids: torch.Tensor,
    *,
    consumer: Callable[[int, torch.Tensor], None],
) -> None:
    spec = derive_supported_model_family_spec_for_capture(bundle.model, bundle.entry.name)
    attr_name = attn_implementation_attr_name(bundle.model)
    original = getattr(bundle.model.config, attr_name)
    capture_key = f"si_rebuttal_fit_capture_{id(bundle)}_{int(time.time_ns())}"
    layer_lookup = {id(module): index for index, module in enumerate(spec.attention_modules)}
    seen_layers: set[int] = set()
    attn_implementation_restore_required = False

    def capture_attention_forward(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        dropout: float = 0.0,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_output, attn_weights = spec.family_adapter.eager_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling,
            dropout=dropout,
            **kwargs,
        )
        layer_index = layer_lookup[id(module)]
        logits = manual_attention_logits(
            query.to(torch.float32),
            key.to(torch.float32),
            additive_mask=None,
            num_query_heads=int(query.shape[1]),
        )[0]
        consumer(layer_index, logits.detach().to(torch.float32).cpu())
        seen_layers.add(layer_index)
        return attn_output, attn_weights

    primary_error: BaseException | None = None
    registry_restores: list[tuple[ValidationAttentionRegistry, ValidationRegistrySnapshot]] = []
    try:
        for registry, installed_value in _fit_capture_attention_registries(
            spec, capture_attention_forward
        ):
            snapshot = registry.snapshot(capture_key)
            registry_restores.append((registry, snapshot))
            registry.install(capture_key, installed_value)
        attn_implementation_restore_required = True
        setattr(bundle.model.config, attr_name, capture_key)
        with torch.inference_mode():
            _ = _call_model(
                bundle.model,
                input_ids=token_ids.unsqueeze(0),
                attention_mask=torch.ones(
                    (1, token_ids.shape[0]), device=bundle.logical_device, dtype=torch.long
                ),
            )
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        while registry_restores:
            registry, snapshot = registry_restores.pop()
            try:
                registry.restore(capture_key, snapshot)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if attn_implementation_restore_required:
            try:
                setattr(bundle.model.config, attr_name, original)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            if primary_error is None:
                _raise_grouped_exceptions(
                    "Fit capture cleanup failed after attention interface installation.",
                    cleanup_errors,
                )
            assert primary_error is not None
            _raise_grouped_exceptions(
                "Fit capture failed and cleanup also failed.",
                [primary_error, *cleanup_errors],
            )
    if len(seen_layers) != spec.layer_count:
        raise RunnerError(
            f"Fit capture expected {spec.layer_count} layers, found {len(seen_layers)}."
        )


def _capture_fit_summary(
    bundle: LoadedModelBundle, sequences: Sequence[Int64Array], *, norm_name: str
) -> FitCaptureResult:
    per_sequence = np.zeros((len(sequences), 32 * 32), dtype=np.float64)
    raw_kernel_sums = np.zeros((32 * 32, 512), dtype=np.float64)
    for sequence_index, sequence in enumerate(sequences):
        token_ids = torch.as_tensor(sequence, dtype=torch.long, device=bundle.logical_device)

        def consume(
            layer_index: int, logits: torch.Tensor, sequence_slot: int = sequence_index
        ) -> None:
            if logits.shape != (32, 512, 512):
                raise RunnerError(
                    "Unexpected post-RoPE fit capture shape at layer "
                    f"{layer_index}: {tuple(logits.shape)}."
                )
            logits_np: Float32Array = np.asarray(logits.detach().cpu(), dtype=np.float32)
            for head_index in range(logits_np.shape[0]):
                flat_index = layer_index * 32 + head_index
                r2, _ = estimate_source_r2_for_sequence(logits_np[head_index], norm_name=norm_name)
                per_sequence[sequence_slot, flat_index] = r2
                raw_kernel_sums[flat_index] += np.asarray(
                    lower_diagonal_means(
                        logits_np[head_index], offset_start=0, offset_stop=511
                    ).cpu(),
                    dtype=np.float64,
                )

        _stream_post_rope_qk(bundle, token_ids, consumer=consume)
        del token_ids
    mean_scores = per_sequence.mean(axis=0).reshape(32, 32)
    return FitCaptureResult(
        layers=32,
        heads=32,
        seq_len=512,
        per_sequence_r2=per_sequence,
        mean_scores=mean_scores,
        raw_kernel_sums=raw_kernel_sums.reshape(32, 32, 512),
        sequence_count=len(sequences),
    )


def _eval_nlls(
    bundle: LoadedModelBundle,
    sequences: Sequence[Int64Array],
    *,
    selected_heads_by_layer: Mapping[int, Sequence[int]] | None = None,
    kernels_by_layer_head: KernelMap | None = None,
) -> Float64Array:
    losses: list[float] = []
    context = None
    if selected_heads_by_layer is not None and kernels_by_layer_head is not None:
        context = bundle.entry.intervention_factory(
            bundle.model,
            expected_call_counts=_expected_call_counts(len(sequences)),
            selected_heads_by_layer=selected_heads_by_layer,
            kernels_by_layer_head=_intervention_kernel_map(kernels_by_layer_head),
        )
    manager = context if context is not None else _NullContext()
    with manager:
        with torch.inference_mode():
            for sequence in sequences:
                token_ids = torch.as_tensor(
                    sequence, dtype=torch.long, device=bundle.logical_device
                )
                logits = _call_model(
                    bundle.model,
                    input_ids=token_ids.unsqueeze(0),
                    attention_mask=torch.ones(
                        (1, token_ids.shape[0]), device=bundle.logical_device, dtype=torch.long
                    ),
                )
                losses.append(float(mean_next_token_nll(logits, token_ids).item()))
                del token_ids, logits
    values = np.asarray(losses, dtype=np.float64)
    if values.shape != (len(sequences),) or not np.isfinite(values).all():
        raise RunnerError("Per-sequence NLL evaluation produced invalid values.")
    return values


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        return None


def _kernel_map_for_heads(heads: Sequence[HeadIndex], kernels: Float64Array) -> HeadKernelMap:
    return {
        (head.layer, head.head): np.asarray(kernels[head.layer, head.head], dtype=np.float64)
        for head in heads
    }


def _selected_heads_by_layer(heads: Sequence[HeadIndex]) -> dict[int, tuple[int, ...]]:
    by_layer: dict[int, list[int]] = {}
    for head in heads:
        by_layer.setdefault(head.layer, []).append(head.head)
    return {layer: tuple(values) for layer, values in sorted(by_layer.items())}


def _fit_profile_bin_payload(
    *,
    bin_index: int,
    heads: Sequence[HeadIndex],
    mean_source_r2: float,
    raw_kernels: Float64Array,
) -> JsonObject:
    return _normalized_json_object(
        {
            "bin_index": int(bin_index),
            "head_count": len(heads),
            "mean_source_r2": float(mean_source_r2),
            "selected_heads": _head_records(heads),
            "selected_head_digest_sha256": _selected_head_digest(heads),
            "kernel_digest_sha256": _kernel_digest(
                _kernel_map_for_heads(heads, raw_kernels).items()
            ),
        },
        label=f"fit_profile.bin.{bin_index}",
    )


def _serialize_fit_profile(
    *,
    resolved_config: ResolvedConfig,
    sweep: SweepConfig,
    model_name: str,
    domain_name: str,
    binding: FrozenModelBinding,
    materialized_domain_sha256: str,
    fit_sequence_digest_sha256: str,
    summary: DomainFitSummary,
) -> JsonObject:
    bins = sort_heads_into_bins(summary.mean_scores)
    bin_means = bin_mean_source_r2(summary.mean_scores, bins)
    return _normalized_json_object(
        {
            "schema_version": 1,
            "artifact_kind": "fit_profile",
            "model": model_name,
            "domain": domain_name,
            "layers": int(summary.mean_scores.shape[0]),
            "heads": int(summary.mean_scores.shape[1]),
            "kernel_length": int(summary.raw_kernels.shape[2]),
            "source_r2_by_layer_head": np.asarray(summary.mean_scores, dtype=np.float64).tolist(),
            "raw_kernels_by_layer_head": np.asarray(summary.raw_kernels, dtype=np.float64).tolist(),
            "raw_kernel_sha256": artifact_identity(
                np.asarray(summary.raw_kernels, dtype=np.float64).tolist()
            ),
            "bins": [
                _fit_profile_bin_payload(
                    bin_index=bin_index,
                    heads=heads,
                    mean_source_r2=float(bin_means[bin_index]),
                    raw_kernels=summary.raw_kernels,
                )
                for bin_index, heads in enumerate(bins)
            ],
            "config_sha256": resolved_config.identity_sha256,
            "sweep_sha256": sweep.identity_sha256,
            "protocol_identity_sha256": _protocol_identity(resolved_config, sweep),
            "weights_tree_sha256": binding.weights_tree_sha256,
            "tokenizer_tree_sha256": binding.tokenizer_tree_sha256,
            "materialized_domain_sha256": materialized_domain_sha256,
            "fit_sequence_digest_sha256": fit_sequence_digest_sha256,
        },
        label=f"fit_profile.{model_name}.{domain_name}",
    )


def _load_fit_profile(
    *,
    run_root: RunRoot,
    resolved_config: ResolvedConfig,
    sweep: SweepConfig,
    model_name: str,
    domain_name: str,
    binding: FrozenModelBinding,
    materialized_domain_sha256: str,
    fit_sequence_digest_sha256: str,
) -> FrozenFitProfile:
    path = _fit_profile_path(run_root, model_name=model_name, domain_name=domain_name)
    if not path.exists():
        raise RunnerError(f"Missing fit profile: {path.name}")
    payload = _fit_profile_payload(
        verify_payload_identity(json.loads(path.read_text(encoding="ascii")), label=path.name),
        label=path.name,
    )
    if payload.get("artifact_kind") != "fit_profile":
        raise RunnerError(f"Fit profile kind drifted: {path.name}")
    if payload["model"] != model_name or payload["domain"] != domain_name:
        raise RunnerError(f"Fit profile identity drifted: {path.name}")
    if (
        payload["config_sha256"] != resolved_config.identity_sha256
        or payload["sweep_sha256"] != sweep.identity_sha256
    ):
        raise RunnerError(f"Fit profile config drifted: {path.name}")
    if payload["protocol_identity_sha256"] != _protocol_identity(resolved_config, sweep):
        raise RunnerError(f"Fit profile protocol drifted: {path.name}")
    if (
        payload["weights_tree_sha256"] != binding.weights_tree_sha256
        or payload["tokenizer_tree_sha256"] != binding.tokenizer_tree_sha256
    ):
        raise RunnerError(f"Fit profile model binding drifted: {path.name}")
    if payload["materialized_domain_sha256"] != materialized_domain_sha256:
        raise RunnerError(f"Fit profile domain binding drifted: {path.name}")
    if payload["fit_sequence_digest_sha256"] != fit_sequence_digest_sha256:
        raise RunnerError(f"Fit profile fit-sequence binding drifted: {path.name}")
    mean_scores = np.asarray(payload["source_r2_by_layer_head"], dtype=np.float64)
    raw_kernels = np.asarray(payload["raw_kernels_by_layer_head"], dtype=np.float64)
    if mean_scores.shape != (32, 32) or raw_kernels.shape != (32, 32, 512):
        raise RunnerError(f"Fit profile tensor shape drifted: {path.name}")
    if payload["raw_kernel_sha256"] != artifact_identity(raw_kernels.tolist()):
        raise RunnerError(f"Fit profile raw-kernel digest drifted: {path.name}")
    bins = sort_heads_into_bins(mean_scores)
    bin_means = bin_mean_source_r2(mean_scores, bins)
    stored_bins = payload["bins"]
    if len(stored_bins) != FIXED_COUNTS["head_bins"]:
        raise RunnerError(f"Fit profile bin inventory drifted: {path.name}")
    for bin_index, heads in enumerate(bins):
        expected_bin = _fit_profile_bin_payload(
            bin_index=bin_index,
            heads=heads,
            mean_source_r2=float(bin_means[bin_index]),
            raw_kernels=raw_kernels,
        )
        if stored_bins[bin_index] != expected_bin:
            raise RunnerError(
                f"Fit profile bin membership or digest drifted: {path.name} bin {bin_index}."
            )
    return FrozenFitProfile(
        payload=payload,
        mean_scores=mean_scores,
        raw_kernels=raw_kernels,
        bins=tuple(tuple(heads) for heads in bins),
        bin_means=np.asarray(bin_means, dtype=np.float64),
    )


def _write_shard(run_root: RunRoot, key: ConditionKey, payload: object) -> ShardPayload:
    normalized = _json_object(payload, label=f"shard.{key.stem()}")
    normalized["schema_version"] = SCHEMA_VERSION
    normalized["identity_sha256"] = artifact_identity(
        {k: v for k, v in normalized.items() if k != "identity_sha256"}
    )
    stored, _ = load_existing_or_write(run_root.shards_dir / f"{key.stem()}.json", normalized)
    return _shard_payload(stored, label=f"shard.{key.stem()}")


def _full_projected_condition_count(include_depth: bool) -> int:
    base = len(TOY_DIRECTIONS)
    bins = (
        len(TOY_DIRECTIONS)
        * FIXED_COUNTS["head_bins"]
        * (2 + FIXED_COUNTS["offset_permutation_trials"] + FIXED_COUNTS["norm_trials"])
    )
    depth = 32 if include_depth else 0
    return base + bins + depth


def _benchmark_factor(projected_sequences: int, measured_sequences: int) -> float:
    return projected_sequences / max(1, measured_sequences)


def _required_free_bytes(
    projected_complete_artifact_bytes: int, benchmark_min_free_space_gib: float
) -> int:
    reserve_bytes = int(benchmark_min_free_space_gib * 1024**3)
    return (2 * int(projected_complete_artifact_bytes)) + reserve_bytes


def _project_runtime_seconds(
    measured_seconds: Mapping[str, float],
    measured_counts: Mapping[str, int],
) -> tuple[float, dict[str, dict[str, float | int | str]], str]:
    projected_counts = {
        "fit_capture_kernel_r2": 2 * FIXED_COUNTS["fit_sequences"],
        "baseline_eval": 2 * FIXED_COUNTS["eval_sequences"],
        "source_bin_eval": 2 * FIXED_COUNTS["head_bins"] * FIXED_COUNTS["eval_sequences"],
        "target_bin_eval": 2 * FIXED_COUNTS["head_bins"] * FIXED_COUNTS["eval_sequences"],
        "control_eval": 2
        * FIXED_COUNTS["head_bins"]
        * (FIXED_COUNTS["offset_permutation_trials"] + FIXED_COUNTS["norm_trials"])
        * FIXED_COUNTS["eval_sequences"],
        "depth_eval": 32 * FIXED_COUNTS["eval_sequences"],
    }
    components: dict[str, dict[str, float | int | str]] = {}
    projected_total_seconds = 0.0
    for name, projected_count in projected_counts.items():
        measured_count = int(measured_counts[name])
        factor = float(_benchmark_factor(projected_count, measured_count))
        projected_seconds = float(measured_seconds[name]) * factor
        components[name] = {
            "formula": "measured_seconds * (projected_sequence_count / measured_sequence_count)",
            "measured_seconds": float(measured_seconds[name]),
            "measured_sequence_count": measured_count,
            "projected_sequence_count": int(projected_count),
            "extrapolation_factor": factor,
            "projected_seconds": projected_seconds,
        }
        projected_total_seconds += projected_seconds
    return projected_total_seconds, components, BENCHMARK_RUNTIME_FORMULA


def _immutable_manifest_inventory(
    run_root: RunRoot, batch_manifest: BatchManifestPayload
) -> dict[str, dict[str, int]]:
    manifest_paths = _manifest_paths(run_root)
    token_manifest_paths = [
        _token_manifest_path(
            run_root, model_name=entry["model"], domain_name=entry["domain"], partition=partition
        )
        for entry in batch_manifest["domain_entries"].values()
        for partition in ("fit", "eval")
    ]
    immutable_components = {
        "resolved_config_manifest": [manifest_paths["resolved_config"]],
        "sweep_manifest": [manifest_paths["sweep"]],
        "materialized_data_manifest": [manifest_paths["materialized"]],
        "batch_manifest": [manifest_paths["batch"]],
        "token_manifests": token_manifest_paths,
    }
    inventory: dict[str, dict[str, int]] = {}
    for name, paths in immutable_components.items():
        missing = [path for path in paths if not path.exists()]
        if missing:
            raise RunnerError(f"Immutable manifest inventory is missing {name}: {missing[0]}")
        exact_bytes = sum(path.stat().st_size for path in paths)
        inventory[name] = {
            "count": len(paths),
            "exact_bytes": exact_bytes,
            "projected_bytes": exact_bytes,
        }
    return inventory


def _project_artifact_bytes(
    *,
    run_root: RunRoot,
    resolved_config: ResolvedConfig,
    sweep: SweepConfig,
    model_name: str,
) -> tuple[int, str, JsonObject]:
    del model_name

    def _size(payload: object) -> int:
        return (
            len(
                json.dumps(
                    _as_json_object(payload, label="artifact_size"),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("ascii")
            )
            + 1
        )

    exemplar_model_name = max(sorted(sweep.models), key=lambda name: (len(name), name))
    selected_heads = [
        {"layer": head // 4, "head": head % 32} for head in range(MAX_PROJECTED_BIN_HEAD_COUNT)
    ]
    sequence_nll = [0.0] * FIXED_COUNTS["eval_sequences"]
    kernel = [0.0] * TOY_SEQUENCE_LENGTH
    fit_profile = {
        "schema_version": 1,
        "artifact_kind": "fit_profile",
        "model": exemplar_model_name,
        "domain": "wikipedia",
        "layers": 32,
        "heads": 32,
        "kernel_length": TOY_SEQUENCE_LENGTH,
        "source_r2_by_layer_head": [[0.0] * 32 for _ in range(32)],
        "raw_kernels_by_layer_head": [[kernel for _ in range(32)] for _ in range(32)],
        "raw_kernel_sha256": "x" * 64,
        "bins": [
            {
                "bin_index": 0,
                "head_count": MAX_PROJECTED_BIN_HEAD_COUNT,
                "mean_source_r2": 0.0,
                "selected_heads": selected_heads,
                "selected_head_digest_sha256": "x" * 64,
                "kernel_digest_sha256": "x" * 64,
                "identity_sha256": "x" * 64,
            }
        ]
        * FIXED_COUNTS["head_bins"],
        "config_sha256": "x" * 64,
        "sweep_sha256": "x" * 64,
        "protocol_identity_sha256": "x" * 64,
        "weights_tree_sha256": "x" * 64,
        "tokenizer_tree_sha256": "x" * 64,
        "materialized_domain_sha256": "x" * 64,
        "fit_sequence_digest_sha256": "x" * 64,
        "identity_sha256": "x" * 64,
    }
    baseline_payload = {
        "schema_version": 1,
        "model": exemplar_model_name,
        "direction": "wikipedia_to_code",
        "unit": "global",
        "unit_index": "baseline",
        "condition": "baseline",
        "sequence_nll": sequence_nll,
        "sequence_digest_sha256": "x" * 64,
        "source_domain_manifest_sha256": "x" * 64,
        "target_domain_manifest_sha256": "x" * 64,
        "weights_tree_sha256": "x" * 64,
        "tokenizer_tree_sha256": "x" * 64,
        "config_sha256": "x" * 64,
        "sweep_sha256": "x" * 64,
        "batch_manifest_sha256": "x" * 64,
        "source_fit_profile_sha256": "x" * 64,
        "target_fit_profile_sha256": "x" * 64,
        "identity_sha256": "x" * 64,
    }
    bin_payload = {
        **baseline_payload,
        "unit": "bin",
        "unit_index": "0",
        "condition": "source_kernel",
        "head_count": MAX_PROJECTED_BIN_HEAD_COUNT,
        "mean_source_r2": 0.0,
        "baseline_identity_sha256": "x" * 64,
        "selected_heads": selected_heads,
        "selected_head_digest_sha256": "x" * 64,
        "selected_bin_digest_sha256": "x" * 64,
        "source_kernel_digest_sha256": "x" * 64,
        "target_kernel_digest_sha256": "x" * 64,
    }
    control_payload = {
        **bin_payload,
        "condition": "norm",
        "trial": 2,
        "control_seed": 123,
        "control_kind": "norm",
        "control_kernel_digest_sha256": "x" * 64,
    }
    depth_payload = {
        **baseline_payload,
        "unit": "layer",
        "unit_index": "31",
        "condition": "depth",
        "head_count": 32,
        "mean_source_r2": 0.0,
        "baseline_identity_sha256": "x" * 64,
        "selected_heads": [{"layer": 31, "head": head} for head in range(32)],
        "selected_head_digest_sha256": "x" * 64,
        "source_kernel_digest_sha256": "x" * 64,
    }
    validation_receipt = {
        "schema_version": 1,
        "model": exemplar_model_name,
        "config_sha256": "x" * 64,
        "sweep_sha256": "x" * 64,
        "batch_manifest_sha256": "x" * 64,
        "validation_domain": "wikipedia",
        "source_domain_manifest_sha256": "x" * 64,
        "fit_token_manifest_sha256": "x" * 64,
        "fit_sequence_digest_sha256": "x" * 64,
        "weights_tree_sha256": "x" * 64,
        "tokenizer_tree_sha256": "x" * 64,
        "baseline_nll": 0.0,
        "zero_nll": 0.0,
        "constant_nll": 0.0,
        "nonconstant_nll": 0.0,
        "signed_by_digests": ["x" * 64] * 5,
        "identity_sha256": "x" * 64,
    }
    benchmark_receipt = {
        "schema_version": 1,
        "model": exemplar_model_name,
        "config_sha256": "x" * 64,
        "sweep_sha256": "x" * 64,
        "batch_manifest_sha256": "x" * 64,
        "weights_tree_sha256": "x" * 64,
        "tokenizer_tree_sha256": "x" * 64,
        "validation_receipt_sha256": "x" * 64,
        "fit_sequences": 2,
        "eval_sequences": 4,
        "measured_seconds": {"fit_capture_kernel_r2": 1.0},
        "measured_counts": {"fit_capture_kernel_r2": 2},
        "projected_total_seconds": 1.0,
        "projected_artifact_bytes": 1,
        "lane_hours": 1.0 / 3600.0,
        "runtime_components": {"fit_capture_kernel_r2": {"projected_seconds": 1.0}},
        "runtime_formula": BENCHMARK_RUNTIME_FORMULA,
        "artifact_components": {"inventory_version": 1},
        "projection_components": {"inventory_version": 1},
        "artifact_formula": BENCHMARK_ARTIFACT_FORMULA,
        "formula": BENCHMARK_ARTIFACT_FORMULA,
        "identity_sha256": "x" * 64,
    }
    batch_manifest = {
        "schema_version": 1,
        "materialized_at_utc": "2026-07-27T00:00:00+00:00",
        "command": ["python", "-m", "si_rebuttal", "materialize-data"],
        "config_sha256": "x" * 64,
        "sweep_sha256": "x" * 64,
        "domains": {
            f"{candidate_model}:{domain_name}": "x" * 64
            for candidate_model in sweep.models
            for domain_name in ("wikipedia", "code")
        },
        "domain_entries": {
            f"{candidate_model}:{domain_name}": {
                "model": candidate_model,
                "domain": domain_name,
                "tokenizer_tree_sha256": "x" * 64,
                "materialized_domain_sha256": "x" * 64,
                "dataset_repository": "repo",
                "dataset_revision": "rev",
                "dataset_config": "cfg",
                "dataset_split": "train",
                "dataset_field": "text",
                "dataset_fingerprint": "x" * 64,
                "fit_token_manifest_sha256": "x" * 64,
                "eval_token_manifest_sha256": "x" * 64,
                "fit_sequence_digest_sha256": "x" * 64,
                "eval_sequence_digest_sha256": "x" * 64,
            }
            for candidate_model in sweep.models
            for domain_name in ("wikipedia", "code")
        },
        "models": {candidate_model: "x" * 64 for candidate_model in sweep.models},
        "tokenizers": {candidate_model: "x" * 64 for candidate_model in sweep.models},
        "git_commit_sha": "x" * 40,
        "git_tracked_diff_sha256": "x" * 64,
        "git_relevant_content_sha256": "x" * 64,
        "git_untracked_sha256": "x" * 64,
        "runtime": {
            "python_version": "3.12",
            "platform": "linux",
            "package_versions": {name: "1.0.0" for name in PACKAGE_NAMES},
            "cuda_version": "12.0",
            "driver_version": "570.00",
        },
        "runtime_capture": {
            "python_executable": "/usr/bin/python3",
            "package_freeze": {
                "resolved_executable": "/usr/bin/python3",
                "command": ["python3", "-m", "pip", "freeze", "--all"],
                "returncode": 0,
                "stdout_text": "\n".join(f"{name}==1.0.0" for name in PACKAGE_NAMES),
                "stdout_sha256": "x" * 64,
                "stderr_text": "",
                "stderr_sha256": "x" * 64,
                "output_lines": [f"{name}==1.0.0" for name in PACKAGE_NAMES],
                "line_count": len(PACKAGE_NAMES),
            },
            "torch_cuda": {
                "torch_version": "2.7.0",
                "torch_cuda_version": "12.0",
                "cuda_available": True,
                "visible_device_count": 2,
            },
            "gpu_inventory": {
                "resolved_executable": "/usr/bin/nvidia-smi",
                "command": [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,memory.total,compute_cap,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                "returncode": 0,
                "stdout_text": (
                    "0, GPU-0, L40, 46068, 8.9, 570.00\n1, GPU-1, L40, 46068, 8.9, 570.00"
                ),
                "stdout_sha256": "x" * 64,
                "stderr_text": "",
                "stderr_sha256": "x" * 64,
                "gpus": [
                    {
                        "physical_index": 0,
                        "uuid": "GPU-0",
                        "name": "L40",
                        "total_memory_mib": 46068,
                        "compute_capability": "8.9",
                        "driver_version": "570.00",
                    },
                    {
                        "physical_index": 1,
                        "uuid": "GPU-1",
                        "name": "L40",
                        "total_memory_mib": 46068,
                        "compute_capability": "8.9",
                        "driver_version": "570.00",
                    },
                ],
                "line_count": 2,
                "driver_version": "570.00",
            },
        },
        "environment": {"CUDA_VISIBLE_DEVICES": "0,1"},
        "provenance": {"schema_version": 1},
        "identity_sha256": "x" * 64,
    }
    lane_receipt = {
        "schema_version": 1,
        "admitted": True,
        "config_sha256": "x" * 64,
        "sweep_sha256": "x" * 64,
        "batch_manifest_sha256": "x" * 64,
        "gpu0_hours": 1.0,
        "gpu1_hours": 1.0,
        "projected_complete_artifact_bytes": 1,
        "projected_artifact_bytes": 1,
        "free_disk_bytes": 1,
        "required_free_bytes": _required_free_bytes(
            1, float(resolved_config.benchmark["min_free_space_gib"])
        ),
        "required_disk_bytes": _required_free_bytes(
            1, float(resolved_config.benchmark["min_free_space_gib"])
        ),
        "benchmark_receipts": {candidate_model: "x" * 64 for candidate_model in sweep.models},
        "benchmark_projected_artifact_bytes": {
            candidate_model: 1 for candidate_model in sweep.models
        },
        "benchmark_projected_total_seconds": {
            candidate_model: 1.0 for candidate_model in sweep.models
        },
        "placement_gpu0": list(sweep.placement_gpu0),
        "placement_gpu1": list(sweep.placement_gpu1),
        "benchmark_lane_hours": {candidate_model: 1.0 for candidate_model in sweep.models},
        "lane_components": {},
        "lane_formula": LANE_HOURS_FORMULA,
        "disk_components": {},
        "disk_formula": DISK_HEADROOM_FORMULA,
        "identity_sha256": "x" * 64,
    }
    terminal_summary = {
        "schema_version": 1,
        "run_id": run_root.run_id,
        "config_sha256": "x" * 64,
        "sweep_sha256": "x" * 64,
        "batch_manifest_sha256": "x" * 64,
        "shards_verified": len(sweep.models)
        * _full_projected_condition_count(include_depth=sweep.include_depth),
        "summary_rows": [{"stem": "x", "mean_nll": 0.0}],
        "dose_response_statistics": [
            {
                "model": candidate_model,
                "direction": direction,
                "rho": 0.0,
                "p_one_sided": 1.0,
                "p_two_sided_scipy": 1.0,
            }
            for candidate_model in sweep.models
            for direction in sweep.directions
        ],
        "contrasts": [
            {
                "model": exemplar_model_name,
                "direction": "wikipedia_to_code",
                "bin": 0,
            }
        ],
        "depth_rows": [
            {
                "model": exemplar_model_name,
                "direction": "wikipedia_to_code",
                "layer": 0,
            }
        ],
        "depth_relationships": [
            {
                "model": exemplar_model_name,
                "direction": "wikipedia_to_code",
                "group_level": True,
            }
        ],
        "summary_hashes": {"shard_inventory_sha256": "x" * 64},
        "completed_at": utc_now(),
        "identity_sha256": "x" * 64,
    }
    launch_receipt = {
        "schema_version": 1,
        "batch_id": "batch",
        "tmux_session": "si-rebuttal-gpu0",
        "physical_gpu": 0,
        "model_names": list(sweep.placement_gpu0),
        "command": ["python", "-m", "si_rebuttal", "run-model"],
        "command_shell": "python -m si_rebuttal run-model",
        "launch_script_path": str(run_root.root / "launch-scripts" / "si-rebuttal-gpu0.sh"),
        "launch_script_sha256": "x" * 64,
        "launch_log_path": str(run_root.logs_dir / "si-rebuttal-gpu0.log"),
        "tmux_command": (
            f"bash {run_root.root / 'launch-scripts' / 'si-rebuttal-gpu0.sh'} "
            f">> {run_root.logs_dir / 'si-rebuttal-gpu0.log'} 2>&1"
        ),
        "snapshots": [{"physical_index": 0}, {"physical_index": 0}],
        "identity_sha256": "x" * 64,
    }
    launch_batch_receipt = {
        "schema_version": 1,
        "batch_id": "batch",
        "run_ids": [run_root.run_id],
        "command": ["python", "-m", "si_rebuttal", "launch", "--execute"],
        "config_sha256": "x" * 64,
        "sweep_sha256": "x" * 64,
        "launch_script_paths": {
            "si-rebuttal-gpu0": str(run_root.root / "launch-scripts" / "si-rebuttal-gpu0.sh")
        },
        "launch_script_sha256s": {"si-rebuttal-gpu0": "x" * 64},
        "launch_log_paths": {"si-rebuttal-gpu0": str(run_root.logs_dir / "si-rebuttal-gpu0.log")},
        "identity_sha256": "x" * 64,
    }
    launch_script = {
        "script_path": str(run_root.root / "launch-scripts" / "si-rebuttal-gpu0.sh"),
        "script_sha256": "x" * 64,
        "body": (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "export CUDA_VISIBLE_DEVICES=0\n"
            "exec python -m si_rebuttal run-model --execute\n"
        ),
    }
    launch_log: JsonObject = {}
    typed_batch_manifest = _batch_manifest_payload(batch_manifest, label="projected_batch_manifest")
    immutable_inventory = _immutable_manifest_inventory(run_root, typed_batch_manifest)
    future_inventory: dict[str, dict[str, int]] = {
        "fit_profiles": {
            "count": len(sweep.models) * 2,
            "representative_bytes": _size(fit_profile),
        },
        "baseline_shards": {
            "count": len(sweep.models) * len(TOY_DIRECTIONS),
            "representative_bytes": _size(baseline_payload),
        },
        "source_kernel_shards": {
            "count": len(sweep.models) * len(TOY_DIRECTIONS) * FIXED_COUNTS["head_bins"],
            "representative_bytes": _size(bin_payload),
        },
        "target_kernel_shards": {
            "count": len(sweep.models) * len(TOY_DIRECTIONS) * FIXED_COUNTS["head_bins"],
            "representative_bytes": _size(bin_payload),
        },
        "control_shards": {
            "count": len(sweep.models)
            * len(TOY_DIRECTIONS)
            * FIXED_COUNTS["head_bins"]
            * (FIXED_COUNTS["offset_permutation_trials"] + FIXED_COUNTS["norm_trials"]),
            "representative_bytes": _size(control_payload),
        },
        "depth_shards": {
            "count": len(sweep.models) * (32 if sweep.include_depth else 0),
            "representative_bytes": _size(depth_payload),
        },
        "validation_receipts": {
            "count": len(sweep.models),
            "representative_bytes": _size(validation_receipt),
        },
        "benchmark_receipts": {
            "count": len(sweep.models),
            "representative_bytes": _size(benchmark_receipt),
        },
        "lane_admission_receipt": {"count": 1, "representative_bytes": _size(lane_receipt)},
        "terminal_summary": {"count": 1, "representative_bytes": _size(terminal_summary)},
        "launch_gpu_receipts": {"count": 2, "representative_bytes": _size(launch_receipt)},
        "launch_batch_receipt": {"count": 1, "representative_bytes": _size(launch_batch_receipt)},
        "launch_scripts": {"count": 2, "representative_bytes": _size(launch_script)},
        "launch_logs": {"count": 2, "representative_bytes": _size(launch_log)},
    }
    future_inventory = {
        name: {
            **component,
            "projected_bytes": int(component["count"]) * int(component["representative_bytes"]),
        }
        for name, component in future_inventory.items()
    }
    immutable_total = sum(
        component["projected_bytes"] for component in immutable_inventory.values()
    )
    future_total = sum(component["projected_bytes"] for component in future_inventory.values())
    margin_multiplier = float(resolved_config.benchmark["artifact_margin_multiplier"])
    future_margin_bytes = math.ceil(future_total * margin_multiplier)
    inventory = {**immutable_inventory, **future_inventory}
    components = _json_object(
        {
            "inventory_version": 2,
            "inventory": inventory,
            "artifact_count": sum(int(component["count"]) for component in inventory.values()),
            "inventory_component_count": len(inventory),
            "immutable_materialized_bytes": immutable_total,
            "future_artifact_bytes": future_total,
            "future_margin_multiplier": margin_multiplier,
            "future_margin_bytes": future_margin_bytes,
        },
        label="artifact_projection_components",
    )
    total = immutable_total + future_margin_bytes
    return total, BENCHMARK_ARTIFACT_FORMULA, components


def project_artifact_bytes(
    *,
    run_root: RunRoot,
    resolved_config: ResolvedConfig,
    sweep: SweepConfig,
    model_name: str,
) -> tuple[int, str, JsonObject]:
    return _project_artifact_bytes(
        run_root=run_root,
        resolved_config=resolved_config,
        sweep=sweep,
        model_name=model_name,
    )


def _recompute_lane_hours(benchmark_payload: JsonMapping, *, model_name: str) -> float:
    projected_total_seconds = _json_float_member(
        benchmark_payload, "projected_total_seconds", label=f"benchmark.{model_name}"
    )
    expected_lane_hours = projected_total_seconds / 3600.0
    stored_lane_hours = _json_float_member(
        benchmark_payload, "lane_hours", label=f"benchmark.{model_name}"
    )
    if not math.isclose(stored_lane_hours, expected_lane_hours, rel_tol=0.0, abs_tol=1e-12):
        raise RunnerError(
            f"Benchmark lane_hours drifted for {model_name}: expected {expected_lane_hours:.12f}, "
            f"found {stored_lane_hours:.12f}."
        )
    return expected_lane_hours


def benchmark_model(
    base_path: str | Path,
    sweep_path: str | Path,
    *,
    model_name: str,
    run_root: Path | None = None,
    time_fn: Callable[[], float] | None = None,
) -> BenchmarkReport:
    resolved_config, sweep = load_configs(base_path, sweep_path)
    logical_device = _require_preflight_model_visibility(sweep, model_name=model_name)
    bundle = load_local_model_bundle(
        resolved_config, model_name=model_name, local_files_only=True, logical_device=logical_device
    )
    if run_root is None:
        raise RunnerError("benchmark-model requires the already-materialized immutable run root.")
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id=None)
    batch_manifest = _load_batch_manifest(root)
    validation_path = _receipt_path(root, f"validation.{model_name}")
    if not validation_path.exists():
        raise RunnerError(f"Missing validation receipt for benchmark: {validation_path.name}")
    validation_receipt = verify_payload_identity(
        json.loads(validation_path.read_text(encoding="ascii")), label=validation_path.name
    )
    wikipedia = _load_sequences_from_run_root(
        root, resolved_config, sweep, model_name, domain_name="wikipedia"
    )
    code = _load_sequences_from_run_root(
        root, resolved_config, sweep, model_name, domain_name="code"
    )
    fit_wikipedia = wikipedia.fit_tokens[:1]
    fit_code = code.fit_tokens[:1]
    eval_code = code.eval_tokens[:2]
    eval_wikipedia = wikipedia.eval_tokens[:2]
    fit_count = len(fit_wikipedia) + len(fit_code)
    eval_count = len(eval_code) + len(eval_wikipedia)
    clock = time.perf_counter if time_fn is None else time_fn
    measured: dict[str, float] = {}
    measured_counts: dict[str, int] = {}

    start = clock()
    wikipedia_fit = _capture_fit_summary(
        bundle, fit_wikipedia, norm_name=resolved_config.models[model_name].norm
    )
    code_fit = _capture_fit_summary(
        bundle, fit_code, norm_name=resolved_config.models[model_name].norm
    )
    wikipedia_bins = sort_heads_into_bins(wikipedia_fit.mean_scores)
    code_bins = sort_heads_into_bins(code_fit.mean_scores)
    measured["fit_capture_kernel_r2"] = max(0.0, clock() - start)
    measured_counts["fit_capture_kernel_r2"] = fit_count

    start = clock()
    _ = _eval_nlls(bundle, eval_code)
    _ = _eval_nlls(bundle, eval_wikipedia)
    measured["baseline_eval"] = max(0.0, clock() - start)
    measured_counts["baseline_eval"] = eval_count

    start = clock()
    selected_code = wikipedia_bins[0]
    _ = _eval_nlls(
        bundle,
        eval_code,
        selected_heads_by_layer=_selected_heads_by_layer(selected_code),
        kernels_by_layer_head=_kernel_map_for_heads(
            selected_code, wikipedia_fit.raw_kernel_sums / max(1, wikipedia_fit.sequence_count)
        ),
    )
    selected_wikipedia = code_bins[0]
    _ = _eval_nlls(
        bundle,
        eval_wikipedia,
        selected_heads_by_layer=_selected_heads_by_layer(selected_wikipedia),
        kernels_by_layer_head=_kernel_map_for_heads(
            selected_wikipedia, code_fit.raw_kernel_sums / max(1, code_fit.sequence_count)
        ),
    )
    measured["source_bin_eval"] = max(0.0, clock() - start)
    measured_counts["source_bin_eval"] = eval_count

    start = clock()
    _ = _eval_nlls(
        bundle,
        eval_code,
        selected_heads_by_layer=_selected_heads_by_layer(selected_code),
        kernels_by_layer_head=_kernel_map_for_heads(
            selected_code, code_fit.raw_kernel_sums / max(1, code_fit.sequence_count)
        ),
    )
    _ = _eval_nlls(
        bundle,
        eval_wikipedia,
        selected_heads_by_layer=_selected_heads_by_layer(selected_wikipedia),
        kernels_by_layer_head=_kernel_map_for_heads(
            selected_wikipedia, wikipedia_fit.raw_kernel_sums / max(1, wikipedia_fit.sequence_count)
        ),
    )
    measured["target_bin_eval"] = max(0.0, clock() - start)
    measured_counts["target_bin_eval"] = eval_count

    start = clock()
    for trial in range(3):
        permuted = _streaming_control_maps(
            selected_code,
            _kernel_map_for_heads(
                selected_code, wikipedia_fit.raw_kernel_sums / max(1, wikipedia_fit.sequence_count)
            ),
            seed=compact_ascii_json_seed(
                [29039, model_name, "wikipedia_to_code", "bin", 0, trial, "offset_permutation"]
            ),
            control_kind="offset_permutation",
        )
        _ = _eval_nlls(
            bundle,
            eval_code,
            selected_heads_by_layer=_selected_heads_by_layer(selected_code),
            kernels_by_layer_head=permuted,
        )
    for trial in range(3):
        normed = _streaming_control_maps(
            selected_code,
            _kernel_map_for_heads(
                selected_code, wikipedia_fit.raw_kernel_sums / max(1, wikipedia_fit.sequence_count)
            ),
            seed=compact_ascii_json_seed(
                [29039, model_name, "wikipedia_to_code", "bin", 0, trial, "norm"]
            ),
            control_kind="norm",
        )
        _ = _eval_nlls(
            bundle,
            eval_code,
            selected_heads_by_layer=_selected_heads_by_layer(selected_code),
            kernels_by_layer_head=normed,
        )
    for trial in range(3):
        permuted = _streaming_control_maps(
            selected_wikipedia,
            _kernel_map_for_heads(
                selected_wikipedia, code_fit.raw_kernel_sums / max(1, code_fit.sequence_count)
            ),
            seed=compact_ascii_json_seed(
                [29039, model_name, "code_to_wikipedia", "bin", 0, trial, "offset_permutation"]
            ),
            control_kind="offset_permutation",
        )
        _ = _eval_nlls(
            bundle,
            eval_wikipedia,
            selected_heads_by_layer=_selected_heads_by_layer(selected_wikipedia),
            kernels_by_layer_head=permuted,
        )
    for trial in range(3):
        normed = _streaming_control_maps(
            selected_wikipedia,
            _kernel_map_for_heads(
                selected_wikipedia, code_fit.raw_kernel_sums / max(1, code_fit.sequence_count)
            ),
            seed=compact_ascii_json_seed(
                [29039, model_name, "code_to_wikipedia", "bin", 0, trial, "norm"]
            ),
            control_kind="norm",
        )
        _ = _eval_nlls(
            bundle,
            eval_wikipedia,
            selected_heads_by_layer=_selected_heads_by_layer(selected_wikipedia),
            kernels_by_layer_head=normed,
        )
    measured["control_eval"] = max(0.0, clock() - start)
    measured_counts["control_eval"] = 2 * 6 * len(eval_code)

    start = clock()
    layer_heads = tuple(HeadIndex(layer=0, head=head) for head in range(32))
    _ = _eval_nlls(
        bundle,
        eval_code,
        selected_heads_by_layer=_selected_heads_by_layer(layer_heads),
        kernels_by_layer_head=_kernel_map_for_heads(
            layer_heads, wikipedia_fit.raw_kernel_sums / max(1, wikipedia_fit.sequence_count)
        ),
    )
    measured["depth_eval"] = max(0.0, clock() - start)
    measured_counts["depth_eval"] = len(eval_code)

    projected_total_seconds, runtime_components, runtime_formula = _project_runtime_seconds(
        measured, measured_counts
    )
    projected_artifact_bytes, artifact_formula, projection_components = _project_artifact_bytes(
        run_root=root,
        resolved_config=resolved_config,
        sweep=sweep,
        model_name=model_name,
    )
    lane_hours = projected_total_seconds / 3600.0
    payload = _json_object(
        {
            "schema_version": 1,
            "model": model_name,
            "config_sha256": resolved_config.identity_sha256,
            "sweep_sha256": sweep.identity_sha256,
            "batch_manifest_sha256": batch_manifest["identity_sha256"],
            "weights_tree_sha256": bundle.binding.weights_tree_sha256,
            "tokenizer_tree_sha256": bundle.binding.tokenizer_tree_sha256,
            "validation_receipt_sha256": validation_receipt["identity_sha256"],
            "fit_sequences": fit_count,
            "eval_sequences": eval_count,
            "measured_seconds": measured,
            "measured_counts": measured_counts,
            "projected_total_seconds": projected_total_seconds,
            "projected_artifact_bytes": projected_artifact_bytes,
            "lane_hours": lane_hours,
            "runtime_components": runtime_components,
            "runtime_formula": runtime_formula,
            "artifact_components": projection_components,
            "artifact_formula": artifact_formula,
            "projection_components": projection_components,
            "formula": artifact_formula,
        },
        label=f"benchmark.{model_name}",
    )
    payload["identity_sha256"] = artifact_identity(payload)
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id=None)
    load_existing_or_write(_receipt_path(root, f"benchmark.{model_name}"), payload)
    return BenchmarkReport(
        model_name=model_name,
        fit_sequences=fit_count,
        eval_sequences=eval_count,
        measured_seconds=measured,
        measured_counts=measured_counts,
        projected_total_seconds=projected_total_seconds,
        projected_artifact_bytes=projected_artifact_bytes,
        lane_hours=lane_hours,
        runtime_components=_json_object(
            runtime_components, label=f"benchmark.{model_name}.runtime"
        ),
        runtime_formula=runtime_formula,
        projection_components=_json_object(
            projection_components, label=f"benchmark.{model_name}.projection"
        ),
        artifact_formula=artifact_formula,
        formula=artifact_formula,
        receipt_identity_sha256=payload["identity_sha256"],
    )


def admit_lanes(
    base_path: str | Path, sweep_path: str | Path, *, run_root: Path
) -> LaneAdmissionReport:
    resolved_config, sweep = load_configs(base_path, sweep_path)
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id=None)
    batch_manifest = _load_batch_manifest(root)
    receipts: dict[str, JsonObject] = {}
    for model_name in sweep.models:
        path = _receipt_path(root, f"benchmark.{model_name}")
        if not path.exists():
            raise RunnerError(f"Missing benchmark receipt for lane admission: {path.name}")
        receipt = _read_verified_payload(path, label=path.name)
        receipts[model_name] = receipt
        if (
            _json_string_member(receipt, "config_sha256", label=path.name)
            != resolved_config.identity_sha256
        ):
            raise RunnerError(f"Benchmark config drifted for {model_name}.")
        if _json_string_member(receipt, "sweep_sha256", label=path.name) != sweep.identity_sha256:
            raise RunnerError(f"Benchmark sweep drifted for {model_name}.")
        if (
            _json_string_member(receipt, "weights_tree_sha256", label=path.name)
            != batch_manifest["models"][model_name]
        ):
            raise RunnerError(f"Benchmark model digest drifted for {model_name}.")
        if (
            _json_string_member(receipt, "tokenizer_tree_sha256", label=path.name)
            != batch_manifest["tokenizers"][model_name]
        ):
            raise RunnerError(f"Benchmark tokenizer digest drifted for {model_name}.")
        runtime_total, runtime_components, runtime_formula = _project_runtime_seconds(
            _json_float_mapping(
                _json_object_item(receipt, "measured_seconds", label=path.name),
                label=f"{path.name}.measured_seconds",
            ),
            _json_int_mapping(
                _json_object_item(receipt, "measured_counts", label=path.name),
                label=f"{path.name}.measured_counts",
            ),
        )
        if _json_string_member(receipt, "runtime_formula", label=path.name) != runtime_formula:
            raise RunnerError(f"Benchmark runtime formula drifted for {model_name}.")
        if _json_object_member(receipt, "runtime_components", label=path.name) != _json_object(
            runtime_components, label="runtime_components"
        ):
            raise RunnerError(f"Benchmark runtime components drifted for {model_name}.")
        if not math.isclose(
            _json_value_number(
                _json_object_item(receipt, "projected_total_seconds", label=path.name),
                label=f"{path.name}.projected_total_seconds",
            ),
            runtime_total,
        ):
            raise RunnerError(f"Benchmark runtime total drifted for {model_name}.")
        expected_artifact_bytes, artifact_formula, artifact_components = _project_artifact_bytes(
            run_root=root,
            resolved_config=resolved_config,
            sweep=sweep,
            model_name=model_name,
        )
        if _json_string_member(receipt, "artifact_formula", label=path.name) != artifact_formula:
            raise RunnerError(f"Benchmark artifact formula drifted for {model_name}.")
        if _json_object_member(receipt, "artifact_components", label=path.name) != _json_object(
            artifact_components, label="artifact_components"
        ):
            raise RunnerError(f"Benchmark artifact components drifted for {model_name}.")
        if (
            _json_int_member(receipt, "projected_artifact_bytes", label=path.name)
            != expected_artifact_bytes
        ):
            raise RunnerError(f"Benchmark artifact byte projection drifted for {model_name}.")
    recomputed_lane_hours = {
        name: _recompute_lane_hours(receipts[name], model_name=name) for name in sweep.models
    }
    gpu0_hours = sum(recomputed_lane_hours[name] for name in sweep.placement_gpu0)
    gpu1_hours = sum(recomputed_lane_hours[name] for name in sweep.placement_gpu1)
    if gpu0_hours > float(sweep.benchmark_max_lane_hours):
        raise RunnerError(f"GPU0 lane admission failed: projected {gpu0_hours:.3f}h.")
    if gpu1_hours > float(sweep.benchmark_max_lane_hours):
        raise RunnerError(f"GPU1 lane admission failed: projected {gpu1_hours:.3f}h.")
    representative_model = sweep.models[0]
    representative_receipt = receipts[representative_model]
    representative_projection = _json_object_member(
        representative_receipt, "projection_components", label=representative_model
    )
    representative_formula = _json_string_member(
        representative_receipt, "formula", label=representative_model
    )
    representative_bytes = _json_int_member(
        representative_receipt, "projected_artifact_bytes", label=representative_model
    )
    for model_name in sweep.models[1:]:
        if (
            _json_object_member(receipts[model_name], "projection_components", label=model_name)
            != representative_projection
        ):
            raise RunnerError(f"Benchmark projection inventory drifted for {model_name}.")
        if (
            _json_string_member(receipts[model_name], "formula", label=model_name)
            != representative_formula
        ):
            raise RunnerError(f"Benchmark projection formula drifted for {model_name}.")
        if (
            _json_int_member(receipts[model_name], "projected_artifact_bytes", label=model_name)
            != representative_bytes
        ):
            raise RunnerError(f"Benchmark projected artifact bytes drifted for {model_name}.")
    projected_bytes = representative_bytes
    free_disk = available_disk_bytes(run_root.parent if run_root.parent.exists() else Path.cwd())
    min_headroom_bytes = int(float(sweep.benchmark_min_free_space_gib) * 1024**3)
    required_free = _required_free_bytes(projected_bytes, float(sweep.benchmark_min_free_space_gib))
    if free_disk < required_free:
        raise RunnerError(f"Disk admission failed: need {required_free} bytes, found {free_disk}.")
    lane_components = _json_object(
        {
            "benchmark_lane_hours": {name: recomputed_lane_hours[name] for name in sweep.models},
            "placement_gpu0": list(sweep.placement_gpu0),
            "placement_gpu1": list(sweep.placement_gpu1),
            "benchmark_max_lane_hours": float(sweep.benchmark_max_lane_hours),
            "gpu0_hours": gpu0_hours,
            "gpu1_hours": gpu1_hours,
            "gpu0_admitted": gpu0_hours <= float(sweep.benchmark_max_lane_hours),
            "gpu1_admitted": gpu1_hours <= float(sweep.benchmark_max_lane_hours),
        },
        label="lane_components",
    )
    disk_components = _json_object(
        {
            "projected_complete_artifact_bytes": projected_bytes,
            "benchmark_min_free_space_gib": float(sweep.benchmark_min_free_space_gib),
            "benchmark_min_free_space_bytes": min_headroom_bytes,
            "artifact_copy_factor": 2,
            "required_free_bytes": required_free,
            "required_disk_bytes": required_free,
            "free_disk_bytes": free_disk,
            "disk_admitted": free_disk >= required_free,
        },
        label="disk_components",
    )
    payload = _json_object(
        {
            "schema_version": 1,
            "admitted": True,
            "config_sha256": resolved_config.identity_sha256,
            "sweep_sha256": sweep.identity_sha256,
            "batch_manifest_sha256": batch_manifest["identity_sha256"],
            "gpu0_hours": gpu0_hours,
            "gpu1_hours": gpu1_hours,
            "projected_complete_artifact_bytes": projected_bytes,
            "projected_artifact_bytes": projected_bytes,
            "free_disk_bytes": free_disk,
            "required_free_bytes": required_free,
            "required_disk_bytes": required_free,
            "benchmark_receipts": {
                name: _json_string_member(receipts[name], "identity_sha256", label=name)
                for name in sweep.models
            },
            "benchmark_projected_artifact_bytes": {
                name: _json_int_member(receipts[name], "projected_artifact_bytes", label=name)
                for name in sweep.models
            },
            "benchmark_projected_total_seconds": {
                name: _json_value_number(
                    _json_object_item(receipts[name], "projected_total_seconds", label=name),
                    label=f"{name}.projected_total_seconds",
                )
                for name in sweep.models
            },
            "placement_gpu0": list(sweep.placement_gpu0),
            "placement_gpu1": list(sweep.placement_gpu1),
            "benchmark_lane_hours": {name: recomputed_lane_hours[name] for name in sweep.models},
            "lane_components": lane_components,
            "lane_formula": LANE_HOURS_FORMULA,
            "disk_components": disk_components,
            "disk_formula": DISK_HEADROOM_FORMULA,
        },
        label="lane_admission",
    )
    payload["identity_sha256"] = artifact_identity(payload)
    load_existing_or_write(_receipt_path(root, "lane-admission"), payload)
    return LaneAdmissionReport(
        gpu0_hours=gpu0_hours,
        gpu1_hours=gpu1_hours,
        projected_artifact_bytes=projected_bytes,
        free_disk_bytes=free_disk,
        required_free_bytes=required_free,
        required_disk_bytes=required_free,
        lane_components=lane_components,
        lane_formula=LANE_HOURS_FORMULA,
        disk_components=disk_components,
        disk_formula=DISK_HEADROOM_FORMULA,
        receipt_identity_sha256=payload["identity_sha256"],
    )


def _load_lane_admission_receipt(
    run_root: RunRoot,
    *,
    resolved_config: ResolvedConfig,
    sweep: SweepConfig,
) -> JsonObject:
    lane = _receipt_path(run_root, "lane-admission")
    if not lane.exists():
        raise RunnerError("Lane admission receipt is required before run-model --execute.")
    lane_payload = _read_verified_payload(lane, label=lane.name)
    if _json_object_item(lane_payload, "admitted", label=lane.name) is not True:
        raise RunnerError("Lane admission receipt must record admitted=true.")
    if (
        _json_string_member(lane_payload, "config_sha256", label=lane.name)
        != resolved_config.identity_sha256
        or _json_string_member(lane_payload, "sweep_sha256", label=lane.name)
        != sweep.identity_sha256
    ):
        raise RunnerError("Lane admission config or sweep drifted.")
    return lane_payload


def _resolve_existing_path(path_text: str, *, label: str) -> Path:
    try:
        return Path(path_text).resolve(strict=True)
    except FileNotFoundError as exc:
        raise RunnerError(f"{label} must point at an existing path.") from exc


def _expected_lane_for_physical_gpu(
    sweep: SweepConfig, *, physical_gpu: int
) -> tuple[str, str, tuple[str, ...]]:
    if physical_gpu == 0:
        return TMUX_GPU0, "cuda:0", tuple(sweep.placement_gpu0)
    if physical_gpu == 1:
        return TMUX_GPU1, "cuda:0", tuple(sweep.placement_gpu1)
    raise RunnerError(f"Unsupported physical GPU in launch receipt: {physical_gpu}")


def _require_execute_launch_context(
    *,
    run_root: RunRoot,
    resolved_config: ResolvedConfig,
    sweep: SweepConfig,
    model_names: Sequence[str],
) -> LaunchExecutionContext:
    missing = [
        key
        for key in (
            LAUNCH_RECEIPT_ENV,
            BATCH_RECEIPT_ENV,
            TMUX_SESSION_ENV,
            RUN_ROOT_ENV,
            BATCH_ID_ENV,
            "CUDA_VISIBLE_DEVICES",
        )
        if not os.environ.get(key)
    ]
    if missing:
        raise RunnerError(
            "run-model --execute is production-admitted only from launch --execute "
            "with bound launch receipts."
        )
    expected_run_root = run_root.root.resolve(strict=True)
    env_run_root = _resolve_existing_path(os.environ[RUN_ROOT_ENV], label=RUN_ROOT_ENV)
    if env_run_root != expected_run_root:
        raise RunnerError("Launch context run root does not match the admitted run root.")
    session_name = os.environ[TMUX_SESSION_ENV]
    launch_path = _resolve_existing_path(os.environ[LAUNCH_RECEIPT_ENV], label=LAUNCH_RECEIPT_ENV)
    batch_path = _resolve_existing_path(os.environ[BATCH_RECEIPT_ENV], label=BATCH_RECEIPT_ENV)
    expected_launch_path = _receipt_path(run_root, session_name).resolve(strict=True)
    expected_batch_path = _receipt_path(run_root, "batch").resolve(strict=True)
    if launch_path != expected_launch_path:
        raise RunnerError("Launch receipt path does not match the admitted GPU lane receipt.")
    if batch_path != expected_batch_path:
        raise RunnerError("Batch receipt path does not match the admitted launch batch receipt.")
    launch_receipt = _read_verified_payload(launch_path, label=launch_path.name)
    batch_receipt = _read_verified_payload(batch_path, label=batch_path.name)
    if session_name != _json_string_member(launch_receipt, "tmux_session", label=launch_path.name):
        raise RunnerError("Launch session context drifted.")
    if os.environ[BATCH_ID_ENV] != _json_string_member(
        launch_receipt, "batch_id", label=launch_path.name
    ) or os.environ[BATCH_ID_ENV] != _json_string_member(
        batch_receipt, "batch_id", label=batch_path.name
    ):
        raise RunnerError("Launch batch context drifted.")
    if _json_string_member(
        _json_object_member(batch_receipt, "launch_receipts", label=batch_path.name),
        session_name,
        label=f"{batch_path.name}.launch_receipts",
    ) != _json_string_member(launch_receipt, "identity_sha256", label=launch_path.name):
        raise RunnerError("Launch batch receipt identity binding drifted.")
    if _json_string_member(
        _json_object_member(batch_receipt, "launch_receipt_paths", label=batch_path.name),
        session_name,
        label=f"{batch_path.name}.launch_receipt_paths",
    ) != str(launch_path):
        raise RunnerError("Launch batch path binding drifted.")
    if _json_string_member(launch_receipt, "batch_receipt_path", label=launch_path.name) != str(
        batch_path
    ):
        raise RunnerError("Per-GPU batch path binding drifted.")
    if _json_string_member(launch_receipt, "launch_receipt_path", label=launch_path.name) != str(
        launch_path
    ):
        raise RunnerError("Per-GPU launch receipt path binding drifted.")
    launch_script_path = _resolve_existing_path(
        _json_string_member(launch_receipt, "launch_script_path", label=launch_path.name),
        label="launch_script_path",
    )
    if _json_string_member(
        _json_object_member(batch_receipt, "launch_script_paths", label=batch_path.name),
        session_name,
        label=f"{batch_path.name}.launch_script_paths",
    ) != str(launch_script_path):
        raise RunnerError("Launch batch receipt script path binding drifted.")
    launch_script_sha256 = _json_string_member(
        launch_receipt, "launch_script_sha256", label=launch_path.name
    )
    if (
        _json_string_member(
            _json_object_member(batch_receipt, "launch_script_sha256s", label=batch_path.name),
            session_name,
            label=f"{batch_path.name}.launch_script_sha256s",
        )
        != launch_script_sha256
    ):
        raise RunnerError("Launch batch receipt script digest binding drifted.")
    if hashlib.sha256(launch_script_path.read_bytes()).hexdigest() != launch_script_sha256:
        raise RunnerError("Launch script digest drifted.")
    launch_log_path = _resolve_existing_path(
        _json_string_member(launch_receipt, "launch_log_path", label=launch_path.name),
        label="launch_log_path",
    )
    if _json_string_member(
        _json_object_member(batch_receipt, "launch_log_paths", label=batch_path.name),
        session_name,
        label=f"{batch_path.name}.launch_log_paths",
    ) != str(launch_log_path):
        raise RunnerError("Launch batch receipt log path binding drifted.")
    for payload in (launch_receipt, batch_receipt):
        payload_run_root = _resolve_existing_path(
            _json_string_member(payload, "run_root", label="receipt"),
            label="receipt run_root",
        )
        if payload_run_root != expected_run_root:
            raise RunnerError("Launch receipt run root drifted.")
        if (
            _json_string_member(payload, "config_sha256", label="receipt")
            != resolved_config.identity_sha256
            or _json_string_member(payload, "sweep_sha256", label="receipt")
            != sweep.identity_sha256
        ):
            raise RunnerError("Launch receipt config or sweep drifted.")
    physical_gpu = _json_int_member(launch_receipt, "physical_gpu", label=launch_path.name)
    expected_session, logical_device, expected_models = _expected_lane_for_physical_gpu(
        sweep, physical_gpu=physical_gpu
    )
    if expected_session != session_name:
        raise RunnerError("Launch receipt physical GPU does not match the fixed tmux lane.")
    if tuple(model_names) != expected_models:
        raise RunnerError(
            "run-model --execute must request the exact fixed admitted lane models "
            "and only those models."
        )
    if (
        tuple(
            _json_string(item, label=f"{launch_path.name}.model_names[{index}]")
            for index, item in enumerate(
                _json_array_member(launch_receipt, "model_names", label=launch_path.name)
            )
        )
        != expected_models
    ):
        raise RunnerError("Launch receipt model set drifted from the fixed admitted lane.")
    cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].strip()
    if cuda_visible_devices != str(physical_gpu) or "," in cuda_visible_devices:
        raise RunnerError(
            "CUDA_VISIBLE_DEVICES must name exactly the single admitted physical GPU."
        )
    if (
        _json_string_member(launch_receipt, "cuda_visible_devices", label=launch_path.name)
        != cuda_visible_devices
    ):
        raise RunnerError("Launch receipt CUDA visibility binding drifted.")
    if (
        _json_string_member(launch_receipt, "logical_device", label=launch_path.name)
        != logical_device
    ):
        raise RunnerError("Launch receipt logical device drifted from cuda:0.")
    snapshots = _json_array_member(launch_receipt, "snapshots", label=launch_path.name)
    if len(snapshots) != 2:
        raise RunnerError("Launch receipt must carry two idle GPU snapshots.")
    for index, snapshot_value in enumerate(snapshots):
        snapshot = _json_object(snapshot_value, label=f"{launch_path.name}.snapshots[{index}]")
        if (
            _json_int_member(
                snapshot, "physical_index", label=f"{launch_path.name}.snapshots[{index}]"
            )
            != physical_gpu
        ):
            raise RunnerError("Launch receipt snapshot GPU binding drifted.")
        if (
            _json_float_member(
                snapshot, "memory_used_mib", label=f"{launch_path.name}.snapshots[{index}]"
            )
            > GPU_MEMORY_IDLE_THRESHOLD_MIB
        ):
            raise RunnerError("Launch receipt snapshot is not idle.")
        if (
            _json_float_member(
                snapshot, "utilization_gpu_pct", label=f"{launch_path.name}.snapshots[{index}]"
            )
            > 0.0
        ):
            raise RunnerError("Launch receipt snapshot is not idle.")
        compute_processes = tuple(
            _json_string(
                item, label=f"{launch_path.name}.snapshots[{index}].compute_processes[{item_index}]"
            )
            for item_index, item in enumerate(
                _json_array_member(
                    snapshot, "compute_processes", label=f"{launch_path.name}.snapshots[{index}]"
                )
            )
        )
        if compute_processes != ():
            raise RunnerError("Launch receipt snapshot is not idle.")
    return LaunchExecutionContext(
        launch_receipt=launch_receipt,
        batch_receipt=batch_receipt,
        logical_device=logical_device,
    )


def _direction_domains(direction: str) -> tuple[str, str]:
    if direction == "wikipedia_to_code":
        return ("wikipedia", "code")
    if direction == "code_to_wikipedia":
        return ("code", "wikipedia")
    raise RunnerError(f"Unsupported direction: {direction}")


def _require_receipts(
    run_root: RunRoot,
    resolved_config: ResolvedConfig,
    sweep: SweepConfig,
    model_names: Sequence[str],
) -> None:
    batch_manifest = _load_batch_manifest(run_root)
    _ = batch_manifest["identity_sha256"]
    _ = batch_manifest["git_commit_sha"]
    _ = batch_manifest["git_tracked_diff_sha256"]
    _ = batch_manifest["git_relevant_content_sha256"]
    _ = batch_manifest["git_untracked_sha256"]
    _ = batch_manifest["runtime"]
    _ = batch_manifest["environment"]
    _ = batch_manifest["provenance"]
    lane_payload = _load_lane_admission_receipt(
        run_root, resolved_config=resolved_config, sweep=sweep
    )
    if (
        _json_string_member(lane_payload, "batch_manifest_sha256", label="lane-admission.json")
        != batch_manifest["identity_sha256"]
    ):
        raise RunnerError("Lane admission batch binding drifted.")
    benchmark_receipts = _json_object_member(
        lane_payload, "benchmark_receipts", label="lane-admission.json"
    )
    if set(benchmark_receipts) != set(sweep.models):
        raise RunnerError("Lane admission receipt model set drifted.")
    if [
        _json_string(item, label=f"lane-admission.json.placement_gpu0[{index}]")
        for index, item in enumerate(
            _json_array_member(lane_payload, "placement_gpu0", label="lane-admission.json")
        )
    ] != list(sweep.placement_gpu0) or [
        _json_string(item, label=f"lane-admission.json.placement_gpu1[{index}]")
        for index, item in enumerate(
            _json_array_member(lane_payload, "placement_gpu1", label="lane-admission.json")
        )
    ] != list(sweep.placement_gpu1):
        raise RunnerError("Lane admission placement drifted.")
    if (
        _json_string_member(lane_payload, "lane_formula", label="lane-admission.json")
        != LANE_HOURS_FORMULA
    ):
        raise RunnerError("Lane admission lane formula drifted.")
    if (
        _json_string_member(lane_payload, "disk_formula", label="lane-admission.json")
        != DISK_HEADROOM_FORMULA
    ):
        raise RunnerError("Lane admission disk formula drifted.")
    benchmark_payloads: dict[str, JsonObject] = {}
    for model_name in model_names:
        for stem in (f"validation.{model_name}", f"benchmark.{model_name}"):
            path = _receipt_path(run_root, stem)
            if not path.exists():
                raise RunnerError(f"Missing required receipt for run-model --execute: {path.name}")
            payload = _read_verified_payload(path, label=path.name)
            if (
                _json_string_member(payload, "config_sha256", label=path.name)
                != resolved_config.identity_sha256
            ):
                raise RunnerError(f"Receipt config drifted for {model_name}.")
            if (
                _json_string_member(payload, "sweep_sha256", label=path.name)
                != sweep.identity_sha256
            ):
                raise RunnerError(f"Receipt sweep drifted for {model_name}.")
            if (
                stem.startswith("validation.")
                and _json_string_member(payload, "batch_manifest_sha256", label=path.name)
                != batch_manifest["identity_sha256"]
            ):
                raise RunnerError(f"Validation batch binding drifted for {model_name}.")
        validation_payload = _read_verified_payload(
            _receipt_path(run_root, f"validation.{model_name}"),
            label=f"validation.{model_name}.json",
        )
        benchmark_payload = _read_verified_payload(
            _receipt_path(run_root, f"benchmark.{model_name}"),
            label=f"benchmark.{model_name}.json",
        )
        benchmark_payloads[model_name] = benchmark_payload
        validation_domain_entry = batch_manifest["domain_entries"][
            _domain_entry_name(model_name, "wikipedia")
        ]
        if (
            _json_string_member(
                validation_payload, "validation_domain", label=f"validation.{model_name}.json"
            )
            != "wikipedia"
        ):
            raise RunnerError(f"Validation domain drifted for {model_name}.")
        if (
            _json_string_member(
                validation_payload,
                "source_domain_manifest_sha256",
                label=f"validation.{model_name}.json",
            )
            != validation_domain_entry["materialized_domain_sha256"]
        ):
            raise RunnerError(f"Validation source domain binding drifted for {model_name}.")
        if (
            _json_string_member(
                validation_payload,
                "fit_token_manifest_sha256",
                label=f"validation.{model_name}.json",
            )
            != validation_domain_entry["fit_token_manifest_sha256"]
        ):
            raise RunnerError(f"Validation fit token binding drifted for {model_name}.")
        if (
            _json_string_member(
                validation_payload,
                "fit_sequence_digest_sha256",
                label=f"validation.{model_name}.json",
            )
            != validation_domain_entry["fit_sequence_digest_sha256"]
        ):
            raise RunnerError(f"Validation fit sequence binding drifted for {model_name}.")
        if (
            _json_string_member(
                validation_payload, "weights_tree_sha256", label=f"validation.{model_name}.json"
            )
            != batch_manifest["models"][model_name]
        ):
            raise RunnerError(f"Validation model binding drifted for {model_name}.")
        if (
            _json_string_member(
                validation_payload, "tokenizer_tree_sha256", label=f"validation.{model_name}.json"
            )
            != batch_manifest["tokenizers"][model_name]
        ):
            raise RunnerError(f"Validation tokenizer binding drifted for {model_name}.")
        if (
            _json_string_member(
                benchmark_payload, "validation_receipt_sha256", label=f"benchmark.{model_name}.json"
            )
            != validation_payload["identity_sha256"]
        ):
            raise RunnerError(f"Benchmark validation binding drifted for {model_name}.")
        if (
            _json_string_member(
                benchmark_payload, "batch_manifest_sha256", label=f"benchmark.{model_name}.json"
            )
            != batch_manifest["identity_sha256"]
        ):
            raise RunnerError(f"Benchmark batch binding drifted for {model_name}.")
        if (
            _json_string_member(
                benchmark_payload, "weights_tree_sha256", label=f"benchmark.{model_name}.json"
            )
            != batch_manifest["models"][model_name]
        ):
            raise RunnerError(f"Benchmark model binding drifted for {model_name}.")
        if (
            _json_string_member(
                benchmark_payload, "tokenizer_tree_sha256", label=f"benchmark.{model_name}.json"
            )
            != batch_manifest["tokenizers"][model_name]
        ):
            raise RunnerError(f"Benchmark tokenizer binding drifted for {model_name}.")
        runtime_total, runtime_components, runtime_formula = _project_runtime_seconds(
            _json_float_mapping(
                _json_object_item(
                    benchmark_payload, "measured_seconds", label=f"benchmark.{model_name}.json"
                ),
                label=f"benchmark.{model_name}.json.measured_seconds",
            ),
            _json_int_mapping(
                _json_object_item(
                    benchmark_payload, "measured_counts", label=f"benchmark.{model_name}.json"
                ),
                label=f"benchmark.{model_name}.json.measured_counts",
            ),
        )
        if (
            _json_string_member(
                benchmark_payload, "runtime_formula", label=f"benchmark.{model_name}.json"
            )
            != runtime_formula
        ):
            raise RunnerError(f"Benchmark runtime formula drifted for {model_name}.")
        if _json_object_member(
            benchmark_payload, "runtime_components", label=f"benchmark.{model_name}.json"
        ) != _json_object(runtime_components, label="runtime_components"):
            raise RunnerError(f"Benchmark runtime components drifted for {model_name}.")
        if not math.isclose(
            _json_float_member(
                benchmark_payload, "projected_total_seconds", label=f"benchmark.{model_name}.json"
            ),
            runtime_total,
        ):
            raise RunnerError(f"Benchmark runtime total drifted for {model_name}.")
        if (
            _json_string_member(
                benchmark_receipts, model_name, label="lane-admission.json.benchmark_receipts"
            )
            != benchmark_payload["identity_sha256"]
        ):
            raise RunnerError(f"Lane admission benchmark binding drifted for {model_name}.")
        if model_name not in benchmark_receipts:
            raise RunnerError(f"Lane admission receipt does not include {model_name}.")
    for model_name in sweep.models:
        if model_name not in benchmark_payloads:
            benchmark_payloads[model_name] = _read_verified_payload(
                _receipt_path(run_root, f"benchmark.{model_name}"),
                label=f"benchmark.{model_name}.json",
            )
    expected_lane_hours = {
        model_name: _recompute_lane_hours(benchmark_payloads[model_name], model_name=model_name)
        for model_name in sweep.models
    }
    for model_name in sweep.models:
        stored_lane_hours = _json_float_member(
            _json_object_member(lane_payload, "benchmark_lane_hours", label="lane-admission.json"),
            model_name,
            label="lane-admission.json.benchmark_lane_hours",
        )
        if not math.isclose(
            stored_lane_hours, expected_lane_hours[model_name], rel_tol=0.0, abs_tol=1e-12
        ):
            raise RunnerError(f"Lane admission benchmark lane_hours drifted for {model_name}.")
    expected_gpu0_hours = sum(expected_lane_hours[name] for name in sweep.placement_gpu0)
    expected_gpu1_hours = sum(expected_lane_hours[name] for name in sweep.placement_gpu1)
    if not math.isclose(
        _json_float_member(lane_payload, "gpu0_hours", label="lane-admission.json"),
        expected_gpu0_hours,
    ):
        raise RunnerError("Lane admission GPU0 sum drifted.")
    if not math.isclose(
        _json_float_member(lane_payload, "gpu1_hours", label="lane-admission.json"),
        expected_gpu1_hours,
    ):
        raise RunnerError("Lane admission GPU1 sum drifted.")
    projected_inventory = {
        _json_int_member(
            _json_object_member(
                lane_payload, "benchmark_projected_artifact_bytes", label="lane-admission.json"
            ),
            name,
            label="lane-admission.json.benchmark_projected_artifact_bytes",
        )
        for name in sweep.models
    }
    if len(projected_inventory) != 1:
        raise RunnerError(
            "Lane admission benchmark projected artifact bytes drifted across models."
        )
    expected_projected_bytes = next(iter(projected_inventory))
    if (
        _json_int_member(lane_payload, "projected_artifact_bytes", label="lane-admission.json")
        != expected_projected_bytes
    ):
        raise RunnerError("Lane admission projected artifact bytes drifted.")
    if (
        _json_int_member(
            lane_payload, "projected_complete_artifact_bytes", label="lane-admission.json"
        )
        != expected_projected_bytes
    ):
        raise RunnerError("Lane admission projected complete artifact bytes drifted.")
    min_headroom_bytes = int(float(sweep.benchmark_min_free_space_gib) * 1024**3)
    expected_required_free = _required_free_bytes(
        expected_projected_bytes, float(sweep.benchmark_min_free_space_gib)
    )
    if (
        _json_int_member(lane_payload, "required_free_bytes", label="lane-admission.json")
        != expected_required_free
    ):
        raise RunnerError("Lane admission required free threshold drifted.")
    if (
        _json_int_member(lane_payload, "required_disk_bytes", label="lane-admission.json")
        != expected_required_free
    ):
        raise RunnerError("Lane admission required disk threshold drifted.")
    expected_lane_components = {
        "benchmark_lane_hours": {name: expected_lane_hours[name] for name in sweep.models},
        "placement_gpu0": list(sweep.placement_gpu0),
        "placement_gpu1": list(sweep.placement_gpu1),
        "benchmark_max_lane_hours": float(sweep.benchmark_max_lane_hours),
        "gpu0_hours": expected_gpu0_hours,
        "gpu1_hours": expected_gpu1_hours,
        "gpu0_admitted": expected_gpu0_hours <= float(sweep.benchmark_max_lane_hours),
        "gpu1_admitted": expected_gpu1_hours <= float(sweep.benchmark_max_lane_hours),
    }
    if _json_object_member(
        lane_payload, "lane_components", label="lane-admission.json"
    ) != _json_object(expected_lane_components, label="expected_lane_components"):
        raise RunnerError("Lane admission lane components drifted.")
    expected_disk_components = {
        "projected_complete_artifact_bytes": expected_projected_bytes,
        "benchmark_min_free_space_gib": float(sweep.benchmark_min_free_space_gib),
        "benchmark_min_free_space_bytes": min_headroom_bytes,
        "artifact_copy_factor": 2,
        "required_free_bytes": expected_required_free,
        "required_disk_bytes": expected_required_free,
        "free_disk_bytes": _json_int_member(
            lane_payload, "free_disk_bytes", label="lane-admission.json"
        ),
        "disk_admitted": _json_int_member(
            lane_payload, "free_disk_bytes", label="lane-admission.json"
        )
        >= expected_required_free,
    }
    if _json_object_member(
        lane_payload, "disk_components", label="lane-admission.json"
    ) != _json_object(expected_disk_components, label="expected_disk_components"):
        raise RunnerError("Lane admission disk components drifted.")
    current_free_disk = available_disk_bytes(
        run_root.root.parent if run_root.root.parent.exists() else Path.cwd()
    )
    if current_free_disk < expected_required_free:
        raise RunnerError("Current free disk is below the admitted threshold.")


def _build_launch_batch_receipt(
    *,
    run_root: RunRoot,
    batch_id: str,
    command: Sequence[str],
    config_sha256: str,
    sweep_sha256: str,
    batch_manifest_sha256: str,
    lane_admission_receipt_sha256: str,
    launch_receipts: Mapping[str, str],
    launch_receipt_paths: Mapping[str, str],
    launch_script_paths: Mapping[str, str],
    launch_script_sha256s: Mapping[str, str],
    launch_log_paths: Mapping[str, str],
) -> JsonObject:
    return _normalized_json_object(
        {
            "schema_version": 1,
            "batch_id": batch_id,
            "run_id": run_root.run_id,
            "run_root": str(run_root.root),
            "command": list(command),
            "config_sha256": config_sha256,
            "sweep_sha256": sweep_sha256,
            "batch_manifest_sha256": batch_manifest_sha256,
            "lane_admission_receipt_sha256": lane_admission_receipt_sha256,
            "launch_receipts": dict(launch_receipts),
            "launch_receipt_paths": dict(launch_receipt_paths),
            "launch_script_paths": dict(launch_script_paths),
            "launch_script_sha256s": dict(launch_script_sha256s),
            "launch_log_paths": dict(launch_log_paths),
        },
        label="launch_batch_receipt",
    )


def _build_launch_receipt(
    *,
    run_root: RunRoot,
    launch: LaunchCommand,
    batch_receipt_path: Path,
    config_sha256: str,
    sweep_sha256: str,
    batch_manifest_sha256: str,
    lane_admission_receipt_sha256: str,
    snapshots: tuple[GpuSnapshot, GpuSnapshot],
) -> JsonObject:
    return _normalized_json_object(
        {
            "schema_version": 1,
            "batch_id": run_root.batch_id or "",
            "run_id": run_root.run_id,
            "run_root": str(run_root.root),
            "tmux_session": launch.tmux_session,
            "physical_gpu": launch.physical_gpu,
            "model_names": list(launch.model_names),
            "command": list(launch.command),
            "command_shell": launch.shell_command,
            "config_sha256": config_sha256,
            "sweep_sha256": sweep_sha256,
            "batch_manifest_sha256": batch_manifest_sha256,
            "lane_admission_receipt_sha256": lane_admission_receipt_sha256,
            "launch_receipt_path": str(run_root.receipts_dir / f"{launch.tmux_session}.json"),
            "batch_receipt_path": str(batch_receipt_path),
            "launch_script_path": str(launch.script_path),
            "launch_script_sha256": launch.script_sha256,
            "launch_log_path": str(launch.log_path),
            "cuda_visible_devices": str(launch.physical_gpu),
            "logical_device": "cuda:0",
            "tmux_command": launch.tmux_command,
            "snapshots": [gpu_snapshot_payload(snapshots[0]), gpu_snapshot_payload(snapshots[1])],
        },
        label=f"launch_receipt.{launch.tmux_session}",
    )


def _write_launch_receipts(
    *,
    run_root: RunRoot,
    plan: LaunchPlan,
    snapshots: Mapping[int, tuple[GpuSnapshot, GpuSnapshot]],
    command: Sequence[str],
    config_sha256: str,
    sweep_sha256: str,
    batch_manifest_sha256: str,
    lane_admission_receipt_sha256: str,
) -> None:
    run_root.ensure()
    batch_receipt_path = run_root.receipts_dir / "batch.json"
    launch_receipts: dict[str, str] = {}
    launch_receipt_paths = {
        launch.tmux_session: str(run_root.receipts_dir / f"{launch.tmux_session}.json")
        for launch in plan.commands
    }
    launch_script_paths = {launch.tmux_session: str(launch.script_path) for launch in plan.commands}
    launch_script_sha256s = {launch.tmux_session: launch.script_sha256 for launch in plan.commands}
    launch_log_paths = {launch.tmux_session: str(launch.log_path) for launch in plan.commands}
    receipt_payloads: dict[str, JsonObject] = {}
    for launch in plan.commands:
        receipt = _build_launch_receipt(
            run_root=replace(run_root, batch_id=plan.batch_id),
            launch=launch,
            batch_receipt_path=batch_receipt_path,
            config_sha256=config_sha256,
            sweep_sha256=sweep_sha256,
            batch_manifest_sha256=batch_manifest_sha256,
            lane_admission_receipt_sha256=lane_admission_receipt_sha256,
            snapshots=snapshots[launch.physical_gpu],
        )
        launch_receipts[launch.tmux_session] = _json_string_member(
            receipt, "identity_sha256", label=f"launch_receipt.{launch.tmux_session}"
        )
        receipt_payloads[launch.tmux_session] = receipt
    batch_receipt = _build_launch_batch_receipt(
        run_root=run_root,
        batch_id=plan.batch_id,
        command=command,
        config_sha256=config_sha256,
        sweep_sha256=sweep_sha256,
        batch_manifest_sha256=batch_manifest_sha256,
        lane_admission_receipt_sha256=lane_admission_receipt_sha256,
        launch_receipts=launch_receipts,
        launch_receipt_paths=launch_receipt_paths,
        launch_script_paths=launch_script_paths,
        launch_script_sha256s=launch_script_sha256s,
        launch_log_paths=launch_log_paths,
    )
    atomic_write_json_no_clobber(batch_receipt_path, batch_receipt)
    for launch in plan.commands:
        receipt = receipt_payloads[launch.tmux_session]
        receipt_identity = _json_string_member(
            receipt, "identity_sha256", label=f"launch_receipt.{launch.tmux_session}"
        )
        if launch_receipts[launch.tmux_session] != receipt_identity:
            raise RunnerError(f"Launch receipt identity drifted for {launch.tmux_session}.")
        atomic_write_json_no_clobber(run_root.receipts_dir / f"{launch.tmux_session}.json", receipt)


def _load_existing_shard_if_valid(
    *,
    run_root: RunRoot,
    key: ConditionKey,
    resolved_config: ResolvedConfig,
    sweep: SweepConfig,
    batch_manifest: BatchManifestPayload,
    source_profile: FrozenFitProfile,
    target_profile: FrozenFitProfile,
    expected_baseline_payload: ShardPayload | None,
) -> ShardPayload | None:
    path = run_root.shards_dir / f"{key.stem()}.json"
    if not path.exists():
        return None
    payload = _require_shard_payload(
        _load_shard(path),
        path=path,
        key=key,
        resolved_config=resolved_config,
        sweep=sweep,
        batch_manifest=batch_manifest,
        diagnostic_prefix="Resume",
    )
    if (
        _require_shard_str(payload, "source_fit_profile_sha256", label=path.name)
        != source_profile.payload["identity_sha256"]
    ):
        raise RunnerError(f"Resume source fit-profile binding drifted: {path.name}")
    if (
        _require_shard_str(payload, "target_fit_profile_sha256", label=path.name)
        != target_profile.payload["identity_sha256"]
    ):
        raise RunnerError(f"Resume target fit-profile binding drifted: {path.name}")
    if expected_baseline_payload is not None:
        if (
            _require_shard_str(payload, "baseline_identity_sha256", label=path.name)
            != expected_baseline_payload["identity_sha256"]
        ):
            raise RunnerError(f"Resume baseline binding drifted: {path.name}")
        if payload["sequence_digest_sha256"] != expected_baseline_payload["sequence_digest_sha256"]:
            raise RunnerError(f"Resume target sequence binding drifted: {path.name}")
    if key.condition == "baseline":
        return payload
    if key.unit == "bin":
        bin_index = _parse_fixed_unit_index(key.unit_index, unit="bin")
        heads = source_profile.bins[bin_index]
        if _require_shard_heads(payload, label=path.name) != _head_records(heads):
            raise RunnerError(f"Resume selected-head membership drifted: {path.name}")
        if _require_shard_str(
            payload, "selected_head_digest_sha256", label=path.name
        ) != _selected_head_digest(heads):
            raise RunnerError(f"Resume selected-head digest drifted: {path.name}")
        if (
            _require_shard_str(payload, "selected_bin_digest_sha256", label=path.name)
            != source_profile.payload["bins"][bin_index]["identity_sha256"]
        ):
            raise RunnerError(f"Resume selected-bin digest drifted: {path.name}")
        expected_mean = float(source_profile.bin_means[bin_index])
        if not math.isclose(
            _require_shard_float(payload, "mean_source_r2", label=path.name),
            expected_mean,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RunnerError(f"Resume source-R2 bin mean drifted: {path.name}")
        source_map = _kernel_map_for_heads(heads, source_profile.raw_kernels)
        if _require_shard_str(
            payload, "source_kernel_digest_sha256", label=path.name
        ) != _kernel_digest(source_map.items()):
            raise RunnerError(f"Resume source-kernel digest drifted: {path.name}")
        if key.condition == "target_kernel":
            target_map = _kernel_map_for_heads(heads, target_profile.raw_kernels)
            if _require_shard_str(
                payload, "target_kernel_digest_sha256", label=path.name
            ) != _kernel_digest(target_map.items()):
                raise RunnerError(f"Resume target-kernel digest drifted: {path.name}")
        if key.condition in ("offset_permutation", "norm"):
            control_map = _streaming_control_maps(
                heads,
                source_map,
                seed=_require_shard_int(payload, "control_seed", label=path.name),
                control_kind=_require_shard_str(payload, "control_kind", label=path.name),
            )
            if _require_shard_str(
                payload, "control_kernel_digest_sha256", label=path.name
            ) != _kernel_digest(control_map.items()):
                raise RunnerError(f"Resume control-kernel digest drifted: {path.name}")
        return payload
    if key.unit == "layer":
        layer_index = _parse_fixed_unit_index(key.unit_index, unit="layer")
        heads = tuple(HeadIndex(layer=layer_index, head=head) for head in range(32))
        if _require_shard_heads(payload, label=path.name) != _head_records(heads):
            raise RunnerError(f"Resume depth selected-head membership drifted: {path.name}")
        if _require_shard_str(
            payload, "selected_head_digest_sha256", label=path.name
        ) != _selected_head_digest(heads):
            raise RunnerError(f"Resume depth selected-head digest drifted: {path.name}")
        expected_mean = float(
            np.asarray(source_profile.mean_scores[layer_index], dtype=np.float64).mean()
        )
        if not math.isclose(
            _require_shard_float(payload, "mean_source_r2", label=path.name),
            expected_mean,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RunnerError(f"Resume depth mean source-R2 drifted: {path.name}")
        if _require_shard_str(
            payload, "source_kernel_digest_sha256", label=path.name
        ) != _kernel_digest(_kernel_map_for_heads(heads, source_profile.raw_kernels).items()):
            raise RunnerError(f"Resume depth source-kernel digest drifted: {path.name}")
        return payload
    raise RunnerError(f"Unsupported resume shard unit: {path.name}")


def _direction_shards_complete(
    *,
    run_root: RunRoot,
    resolved_config: ResolvedConfig,
    sweep: SweepConfig,
    batch_manifest: BatchManifestPayload,
    model_name: str,
    direction: str,
    source_profile: FrozenFitProfile,
    target_profile: FrozenFitProfile,
) -> bool:
    baseline_key = ConditionKey(
        model=model_name,
        direction=direction,
        unit="global",
        unit_index="baseline",
        condition="baseline",
    )
    baseline_payload = _load_existing_shard_if_valid(
        run_root=run_root,
        key=baseline_key,
        resolved_config=resolved_config,
        sweep=sweep,
        batch_manifest=batch_manifest,
        source_profile=source_profile,
        target_profile=target_profile,
        expected_baseline_payload=None,
    )
    if baseline_payload is None:
        return False
    for bin_index in range(FIXED_COUNTS["head_bins"]):
        for condition in ("source_kernel", "target_kernel"):
            payload = _load_existing_shard_if_valid(
                run_root=run_root,
                key=ConditionKey(
                    model=model_name,
                    direction=direction,
                    unit="bin",
                    unit_index=str(bin_index),
                    condition=condition,
                ),
                resolved_config=resolved_config,
                sweep=sweep,
                batch_manifest=batch_manifest,
                source_profile=source_profile,
                target_profile=target_profile,
                expected_baseline_payload=baseline_payload,
            )
            if payload is None:
                return False
        for control_kind in ("offset_permutation", "norm"):
            for trial in range(3):
                payload = _load_existing_shard_if_valid(
                    run_root=run_root,
                    key=ConditionKey(
                        model=model_name,
                        direction=direction,
                        unit="bin",
                        unit_index=str(bin_index),
                        condition=control_kind,
                        trial=trial,
                    ),
                    resolved_config=resolved_config,
                    sweep=sweep,
                    batch_manifest=batch_manifest,
                    source_profile=source_profile,
                    target_profile=target_profile,
                    expected_baseline_payload=baseline_payload,
                )
                if payload is None:
                    return False
    if direction == "wikipedia_to_code":
        for layer_index in range(32):
            payload = _load_existing_shard_if_valid(
                run_root=run_root,
                key=ConditionKey(
                    model=model_name,
                    direction=direction,
                    unit="layer",
                    unit_index=str(layer_index),
                    condition="depth",
                ),
                resolved_config=resolved_config,
                sweep=sweep,
                batch_manifest=batch_manifest,
                source_profile=source_profile,
                target_profile=target_profile,
                expected_baseline_payload=baseline_payload,
            )
            if payload is None:
                return False
    return True


def _execute_direction(
    *,
    bundle: LoadedModelBundle,
    run_root: RunRoot,
    resolved_config: ResolvedConfig,
    sweep: SweepConfig,
    model_name: str,
    direction: str,
    source_profile: FrozenFitProfile,
    target_profile: FrozenFitProfile,
    target_eval: Sequence[Int64Array],
    batch_manifest_sha256: str,
) -> None:
    batch_manifest = _load_batch_manifest(run_root)
    domain_entries = batch_manifest["domain_entries"]
    source_domain, target_domain = _direction_domains(direction)
    source_entry = domain_entries[_domain_entry_name(model_name, source_domain)]
    target_entry = domain_entries[_domain_entry_name(model_name, target_domain)]
    source_scores = source_profile.mean_scores
    source_kernels = source_profile.raw_kernels
    target_kernels = target_profile.raw_kernels
    bins = source_profile.bins
    bin_means = source_profile.bin_means

    baseline_key = ConditionKey(
        model=model_name,
        direction=direction,
        unit="global",
        unit_index="baseline",
        condition="baseline",
    )
    baseline_payload = _load_existing_shard_if_valid(
        run_root=run_root,
        key=baseline_key,
        resolved_config=resolved_config,
        sweep=sweep,
        batch_manifest=batch_manifest,
        source_profile=source_profile,
        target_profile=target_profile,
        expected_baseline_payload=None,
    )
    if baseline_payload is None:
        baseline = _eval_nlls(bundle, target_eval)
        baseline_payload = _write_shard(
            run_root,
            baseline_key,
            {
                "model": model_name,
                "direction": direction,
                "unit": "global",
                "unit_index": "baseline",
                "condition": "baseline",
                "head_count": 0,
                "sequence_nll": baseline.tolist(),
                "sequence_digest_sha256": _sequence_digest(target_eval),
                "source_domain_manifest_sha256": source_entry["materialized_domain_sha256"],
                "target_domain_manifest_sha256": target_entry["materialized_domain_sha256"],
                "weights_tree_sha256": bundle.binding.weights_tree_sha256,
                "tokenizer_tree_sha256": bundle.binding.tokenizer_tree_sha256,
                "config_sha256": resolved_config.identity_sha256,
                "sweep_sha256": sweep.identity_sha256,
                "batch_manifest_sha256": batch_manifest_sha256,
                "source_fit_profile_sha256": source_profile.payload["identity_sha256"],
                "target_fit_profile_sha256": target_profile.payload["identity_sha256"],
            },
        )

    for bin_index, heads in enumerate(bins):
        selected_heads = _selected_heads_by_layer(heads)
        source_map = _kernel_map_for_heads(heads, source_kernels)
        target_map = _kernel_map_for_heads(heads, target_kernels)
        for condition_name, kernel_map in (
            ("source_kernel", source_map),
            ("target_kernel", target_map),
        ):
            key = ConditionKey(
                model=model_name,
                direction=direction,
                unit="bin",
                unit_index=str(bin_index),
                condition=condition_name,
            )
            existing = _load_existing_shard_if_valid(
                run_root=run_root,
                key=key,
                resolved_config=resolved_config,
                sweep=sweep,
                batch_manifest=batch_manifest,
                source_profile=source_profile,
                target_profile=target_profile,
                expected_baseline_payload=baseline_payload,
            )
            if existing is not None:
                continue
            values = _eval_nlls(
                bundle,
                target_eval,
                selected_heads_by_layer=selected_heads,
                kernels_by_layer_head=kernel_map,
            )
            _write_shard(
                run_root,
                key,
                {
                    "model": model_name,
                    "direction": direction,
                    "unit": "bin",
                    "unit_index": str(bin_index),
                    "condition": condition_name,
                    "head_count": len(heads),
                    "mean_source_r2": float(bin_means[bin_index]),
                    "sequence_nll": values.tolist(),
                    "sequence_digest_sha256": baseline_payload["sequence_digest_sha256"],
                    "source_domain_manifest_sha256": source_entry["materialized_domain_sha256"],
                    "target_domain_manifest_sha256": target_entry["materialized_domain_sha256"],
                    "baseline_identity_sha256": baseline_payload["identity_sha256"],
                    "weights_tree_sha256": bundle.binding.weights_tree_sha256,
                    "tokenizer_tree_sha256": bundle.binding.tokenizer_tree_sha256,
                    "config_sha256": resolved_config.identity_sha256,
                    "sweep_sha256": sweep.identity_sha256,
                    "batch_manifest_sha256": batch_manifest_sha256,
                    "source_fit_profile_sha256": source_profile.payload["identity_sha256"],
                    "target_fit_profile_sha256": target_profile.payload["identity_sha256"],
                    "selected_heads": _head_records(heads),
                    "selected_head_digest_sha256": _selected_head_digest(heads),
                    "selected_bin_digest_sha256": source_profile.payload["bins"][bin_index][
                        "identity_sha256"
                    ],
                    "source_kernel_digest_sha256": _kernel_digest(source_map.items()),
                    "target_kernel_digest_sha256": _kernel_digest(target_map.items()),
                },
            )
        for control_kind in ("offset_permutation", "norm"):
            for trial in range(3):
                seed = compact_ascii_json_seed(
                    [29039, model_name, direction, "bin", bin_index, trial, control_kind]
                )
                key = ConditionKey(
                    model=model_name,
                    direction=direction,
                    unit="bin",
                    unit_index=str(bin_index),
                    condition=control_kind,
                    trial=trial,
                )
                existing = _load_existing_shard_if_valid(
                    run_root=run_root,
                    key=key,
                    resolved_config=resolved_config,
                    sweep=sweep,
                    batch_manifest=batch_manifest,
                    source_profile=source_profile,
                    target_profile=target_profile,
                    expected_baseline_payload=baseline_payload,
                )
                if existing is not None:
                    continue
                control_map = _streaming_control_maps(
                    heads, source_map, seed=seed, control_kind=control_kind
                )
                values = _eval_nlls(
                    bundle,
                    target_eval,
                    selected_heads_by_layer=selected_heads,
                    kernels_by_layer_head=control_map,
                )
                _write_shard(
                    run_root,
                    key,
                    {
                        "model": model_name,
                        "direction": direction,
                        "unit": "bin",
                        "unit_index": str(bin_index),
                        "condition": control_kind,
                        "trial": trial,
                        "head_count": len(heads),
                        "mean_source_r2": float(bin_means[bin_index]),
                        "sequence_nll": values.tolist(),
                        "sequence_digest_sha256": baseline_payload["sequence_digest_sha256"],
                        "source_domain_manifest_sha256": source_entry["materialized_domain_sha256"],
                        "target_domain_manifest_sha256": target_entry["materialized_domain_sha256"],
                        "weights_tree_sha256": bundle.binding.weights_tree_sha256,
                        "tokenizer_tree_sha256": bundle.binding.tokenizer_tree_sha256,
                        "config_sha256": resolved_config.identity_sha256,
                        "sweep_sha256": sweep.identity_sha256,
                        "batch_manifest_sha256": batch_manifest_sha256,
                        "baseline_identity_sha256": baseline_payload["identity_sha256"],
                        "source_fit_profile_sha256": source_profile.payload["identity_sha256"],
                        "target_fit_profile_sha256": target_profile.payload["identity_sha256"],
                        "selected_heads": _head_records(heads),
                        "selected_head_digest_sha256": _selected_head_digest(heads),
                        "selected_bin_digest_sha256": source_profile.payload["bins"][bin_index][
                            "identity_sha256"
                        ],
                        "source_kernel_digest_sha256": _kernel_digest(source_map.items()),
                        "control_seed": int(seed),
                        "control_kind": control_kind,
                        "control_kernel_digest_sha256": _kernel_digest(control_map.items()),
                    },
                )

    if direction == "wikipedia_to_code":
        for layer_index in range(32):
            heads = tuple(HeadIndex(layer=layer_index, head=head) for head in range(32))
            selected_heads = _selected_heads_by_layer(heads)
            kernel_map = _kernel_map_for_heads(heads, source_kernels)
            key = ConditionKey(
                model=model_name,
                direction=direction,
                unit="layer",
                unit_index=str(layer_index),
                condition="depth",
            )
            existing = _load_existing_shard_if_valid(
                run_root=run_root,
                key=key,
                resolved_config=resolved_config,
                sweep=sweep,
                batch_manifest=batch_manifest,
                source_profile=source_profile,
                target_profile=target_profile,
                expected_baseline_payload=baseline_payload,
            )
            if existing is not None:
                continue
            values = _eval_nlls(
                bundle,
                target_eval,
                selected_heads_by_layer=selected_heads,
                kernels_by_layer_head=kernel_map,
            )
            _write_shard(
                run_root,
                key,
                {
                    "model": model_name,
                    "direction": direction,
                    "unit": "layer",
                    "unit_index": str(layer_index),
                    "condition": "depth",
                    "head_count": 32,
                    "mean_source_r2": float(
                        np.asarray(source_scores[layer_index], dtype=np.float64).mean()
                    ),
                    "sequence_nll": values.tolist(),
                    "sequence_digest_sha256": baseline_payload["sequence_digest_sha256"],
                    "source_domain_manifest_sha256": source_entry["materialized_domain_sha256"],
                    "target_domain_manifest_sha256": target_entry["materialized_domain_sha256"],
                    "weights_tree_sha256": bundle.binding.weights_tree_sha256,
                    "tokenizer_tree_sha256": bundle.binding.tokenizer_tree_sha256,
                    "config_sha256": resolved_config.identity_sha256,
                    "sweep_sha256": sweep.identity_sha256,
                    "batch_manifest_sha256": batch_manifest_sha256,
                    "baseline_identity_sha256": baseline_payload["identity_sha256"],
                    "source_fit_profile_sha256": source_profile.payload["identity_sha256"],
                    "target_fit_profile_sha256": target_profile.payload["identity_sha256"],
                    "selected_heads": _head_records(heads),
                    "selected_head_digest_sha256": _selected_head_digest(heads),
                    "source_kernel_digest_sha256": _kernel_digest(kernel_map.items()),
                },
            )


def run_model(
    base_path: str | Path,
    sweep_path: str | Path,
    *,
    run_root: Path,
    model_names: Sequence[str],
    execute: bool,
    runtime_command_runner: RuntimeCommandRunner = default_subprocess_runner,
) -> JsonObject:
    resolved_config, sweep = load_configs(base_path, sweep_path)
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id=None)
    if not execute:
        return {
            "schema_version": 1,
            "run_root": str(run_root),
            "models": list(model_names),
            "execute": False,
            "planned_condition_count_per_model": _full_projected_condition_count(
                include_depth=sweep.include_depth
            ),
        }
    if not run_root.exists():
        raise RunnerError(
            "run-model --execute requires the already-materialized immutable run root."
        )
    if not run_root.is_dir():
        raise RunnerError("run-model --execute run root must be a directory.")
    launch_context = _require_execute_launch_context(
        run_root=root,
        resolved_config=resolved_config,
        sweep=sweep,
        model_names=model_names,
    )
    batch_manifest = _load_batch_manifest(root)
    _require_current_batch_manifest_provenance(
        batch_manifest=batch_manifest,
        resolved_config=resolved_config,
        runtime_command_runner=runtime_command_runner,
        stage="run-model",
    )
    lane_payload = _load_lane_admission_receipt(root, resolved_config=resolved_config, sweep=sweep)
    if (
        _json_string_member(
            launch_context.launch_receipt, "batch_manifest_sha256", label="launch_receipt"
        )
        != batch_manifest["identity_sha256"]
    ):
        raise RunnerError("Per-GPU launch receipt batch manifest binding drifted.")
    if (
        _json_string_member(
            launch_context.batch_receipt, "batch_manifest_sha256", label="batch_receipt"
        )
        != batch_manifest["identity_sha256"]
    ):
        raise RunnerError("Launch batch receipt batch manifest binding drifted.")
    if (
        _json_string_member(
            launch_context.launch_receipt,
            "lane_admission_receipt_sha256",
            label="launch_receipt",
        )
        != lane_payload["identity_sha256"]
    ):
        raise RunnerError("Per-GPU launch receipt lane admission binding drifted.")
    if (
        _json_string_member(
            launch_context.batch_receipt,
            "lane_admission_receipt_sha256",
            label="batch_receipt",
        )
        != lane_payload["identity_sha256"]
    ):
        raise RunnerError("Launch batch receipt lane admission binding drifted.")
    _require_receipts(root, resolved_config, sweep, model_names)
    for model_name in model_names:
        wikipedia = _load_sequences_from_run_root(
            root, resolved_config, sweep, model_name, domain_name="wikipedia"
        )
        code = _load_sequences_from_run_root(
            root, resolved_config, sweep, model_name, domain_name="code"
        )
        model_binding = freeze_model_binding(resolved_config.models[model_name])
        fit_profiles: dict[str, FrozenFitProfile] = {}
        missing_fit_domains: list[str] = []
        for domain_name in ("wikipedia", "code"):
            domain_entry = batch_manifest["domain_entries"][
                _domain_entry_name(model_name, domain_name)
            ]
            profile_path = _fit_profile_path(root, model_name=model_name, domain_name=domain_name)
            if profile_path.exists():
                fit_profiles[domain_name] = _load_fit_profile(
                    run_root=root,
                    resolved_config=resolved_config,
                    sweep=sweep,
                    model_name=model_name,
                    domain_name=domain_name,
                    binding=model_binding,
                    materialized_domain_sha256=str(domain_entry["materialized_domain_sha256"]),
                    fit_sequence_digest_sha256=str(domain_entry["fit_sequence_digest_sha256"]),
                )
            else:
                missing_fit_domains.append(domain_name)
        direction_needs_compute: dict[str, bool] = {}
        if not missing_fit_domains:
            for direction in sweep.directions:
                source_domain, target_domain = _direction_domains(direction)
                direction_needs_compute[direction] = not _direction_shards_complete(
                    run_root=root,
                    resolved_config=resolved_config,
                    sweep=sweep,
                    batch_manifest=batch_manifest,
                    model_name=model_name,
                    direction=direction,
                    source_profile=fit_profiles[source_domain],
                    target_profile=fit_profiles[target_domain],
                )
            if not any(direction_needs_compute.values()):
                continue
        fit_bundle: LoadedModelBundle | None = None
        try:
            if missing_fit_domains:
                fit_bundle = load_local_model_bundle(
                    resolved_config,
                    model_name=model_name,
                    local_files_only=True,
                    logical_device=launch_context.logical_device,
                )
                for domain_name in missing_fit_domains:
                    sequences = (
                        wikipedia.fit_tokens if domain_name == "wikipedia" else code.fit_tokens
                    )
                    fit_capture = _capture_fit_summary(
                        fit_bundle, sequences, norm_name=resolved_config.models[model_name].norm
                    )
                    summary = DomainFitSummary(
                        mean_scores=fit_capture.mean_scores,
                        raw_kernels=fit_capture.raw_kernel_sums
                        / max(1, fit_capture.sequence_count),
                    )
                    domain_entry = batch_manifest["domain_entries"][
                        _domain_entry_name(model_name, domain_name)
                    ]
                    profile_payload = _serialize_fit_profile(
                        resolved_config=resolved_config,
                        sweep=sweep,
                        model_name=model_name,
                        domain_name=domain_name,
                        binding=model_binding,
                        materialized_domain_sha256=str(domain_entry["materialized_domain_sha256"]),
                        fit_sequence_digest_sha256=str(domain_entry["fit_sequence_digest_sha256"]),
                        summary=summary,
                    )
                    load_existing_or_write(
                        _fit_profile_path(root, model_name=model_name, domain_name=domain_name),
                        profile_payload,
                    )
                    fit_profiles[domain_name] = _load_fit_profile(
                        run_root=root,
                        resolved_config=resolved_config,
                        sweep=sweep,
                        model_name=model_name,
                        domain_name=domain_name,
                        binding=model_binding,
                        materialized_domain_sha256=str(domain_entry["materialized_domain_sha256"]),
                        fit_sequence_digest_sha256=str(domain_entry["fit_sequence_digest_sha256"]),
                    )
                direction_needs_compute = {direction: True for direction in sweep.directions}
            elif not direction_needs_compute:
                direction_needs_compute = {direction: True for direction in sweep.directions}
            if any(direction_needs_compute.values()) and fit_bundle is None:
                fit_bundle = load_local_model_bundle(
                    resolved_config,
                    model_name=model_name,
                    local_files_only=True,
                    logical_device=launch_context.logical_device,
                )
            for direction in sweep.directions:
                if not direction_needs_compute.get(direction, True):
                    continue
                source_domain, target_domain = _direction_domains(direction)
                target_eval = (
                    wikipedia.eval_tokens if target_domain == "wikipedia" else code.eval_tokens
                )
                if fit_bundle is None:
                    raise RunnerError(f"Missing loaded model bundle for {model_name}.")
                _execute_direction(
                    bundle=fit_bundle,
                    run_root=root,
                    resolved_config=resolved_config,
                    sweep=sweep,
                    model_name=model_name,
                    direction=direction,
                    source_profile=fit_profiles[source_domain],
                    target_profile=fit_profiles[target_domain],
                    target_eval=target_eval,
                    batch_manifest_sha256=str(batch_manifest["identity_sha256"]),
                )
        finally:
            _release_model_bundle(fit_bundle)
    return {
        "schema_version": 1,
        "run_root": str(run_root),
        "models": list(model_names),
        "execute": True,
    }


def _load_shard(path: Path) -> ShardPayload:
    return _shard_payload(
        verify_payload_identity(json.loads(path.read_text(encoding="ascii")), label=path.name),
        label=path.name,
    )


def _require_shard_payload(
    payload: ShardPayload,
    *,
    path: Path,
    key: ConditionKey,
    resolved_config: ResolvedConfig,
    sweep: SweepConfig,
    batch_manifest: BatchManifestPayload,
    diagnostic_prefix: str = "Finalizer",
) -> ShardPayload:
    try:
        parsed_key = ConditionKey.parse_stem(path.stem)
        if parsed_key != key:
            raise RunnerError(f"{diagnostic_prefix} filename/payload key drifted: {path.name}")
        data = payload
        if data["schema_version"] != SCHEMA_VERSION:
            raise RunnerError(f"{diagnostic_prefix} schema mismatch: {path.name}")
        if data["model"] != key.model or data["direction"] != key.direction:
            raise RunnerError(f"{diagnostic_prefix} shard identity drifted: {path.name}")
        if (
            data["unit"] != key.unit
            or str(data["unit_index"]) != key.unit_index
            or data["condition"] != key.condition
        ):
            raise RunnerError(f"{diagnostic_prefix} shard protocol drifted: {path.name}")
        if key.trial is None:
            if "trial" in data:
                raise RunnerError(f"{diagnostic_prefix} unexpected trial field: {path.name}")
        elif _require_shard_int(data, "trial", label=path.name) != key.trial:
            raise RunnerError(f"{diagnostic_prefix} trial drifted: {path.name}")
        if (
            data["config_sha256"] != resolved_config.identity_sha256
            or data["sweep_sha256"] != sweep.identity_sha256
        ):
            raise RunnerError(f"{diagnostic_prefix} config/sweep drifted: {path.name}")
        if data["batch_manifest_sha256"] != batch_manifest["identity_sha256"]:
            raise RunnerError(f"{diagnostic_prefix} batch identity drifted: {path.name}")
        if data["weights_tree_sha256"] != batch_manifest["models"][key.model]:
            raise RunnerError(f"{diagnostic_prefix} model digest drifted: {path.name}")
        if data["tokenizer_tree_sha256"] != batch_manifest["tokenizers"][key.model]:
            raise RunnerError(f"{diagnostic_prefix} tokenizer digest drifted: {path.name}")
        source_domain, target_domain = _direction_domains(key.direction)
        source_entry = batch_manifest["domain_entries"][
            _domain_entry_name(key.model, source_domain)
        ]
        target_entry = batch_manifest["domain_entries"][
            _domain_entry_name(key.model, target_domain)
        ]
        if data["source_domain_manifest_sha256"] != source_entry["materialized_domain_sha256"]:
            raise RunnerError(f"{diagnostic_prefix} source domain binding drifted: {path.name}")
        if data["target_domain_manifest_sha256"] != target_entry["materialized_domain_sha256"]:
            raise RunnerError(f"{diagnostic_prefix} target domain binding drifted: {path.name}")
        sequence_nll = data["sequence_nll"]
        if sequence_nll.shape != (FIXED_COUNTS["eval_sequences"],):
            raise RunnerError(
                f"{diagnostic_prefix} requires exact per-sequence length "
                f"{FIXED_COUNTS['eval_sequences']}: {path.name}"
            )
        if not np.isfinite(sequence_nll).all():
            raise RunnerError(f"{diagnostic_prefix} non-finite shard values: {path.name}")
        expected_sequence_digest = target_entry["eval_sequence_digest_sha256"]
        if data["sequence_digest_sha256"] != expected_sequence_digest:
            raise RunnerError(f"{diagnostic_prefix} target sequence binding drifted: {path.name}")
        if key.condition != "baseline":
            if "source_fit_profile_sha256" not in data or "target_fit_profile_sha256" not in data:
                raise RunnerError(f"{diagnostic_prefix} fit-profile linkage missing: {path.name}")
        if key.condition != "baseline":
            if "baseline_identity_sha256" not in data:
                raise RunnerError(f"{diagnostic_prefix} baseline linkage missing: {path.name}")
        if key.unit == "bin":
            if "mean_source_r2" not in data or "selected_heads" not in data:
                raise RunnerError(f"{diagnostic_prefix} bin shard fields missing: {path.name}")
            heads = _require_shard_heads(data, label=path.name)
            if len(heads) != data["head_count"]:
                raise RunnerError(f"{diagnostic_prefix} head inventory drifted: {path.name}")
            if data.get("selected_head_digest_sha256") != artifact_identity(heads):
                raise RunnerError(f"{diagnostic_prefix} selected-head digest drifted: {path.name}")
            if "selected_bin_digest_sha256" not in data:
                raise RunnerError(f"{diagnostic_prefix} selected-bin digest missing: {path.name}")
            if key.condition in ("source_kernel", "target_kernel"):
                if (
                    "source_kernel_digest_sha256" not in data
                    or "target_kernel_digest_sha256" not in data
                ):
                    raise RunnerError(f"{diagnostic_prefix} kernel digests missing: {path.name}")
            if key.condition in ("offset_permutation", "norm"):
                if (
                    "trial" not in data
                    or "control_kind" not in data
                    or "control_seed" not in data
                    or "source_kernel_digest_sha256" not in data
                    or "control_kernel_digest_sha256" not in data
                ):
                    raise RunnerError(
                        f"{diagnostic_prefix} control shard fields missing: {path.name}"
                    )
                _verify_control_seed(
                    model_name=key.model,
                    direction=key.direction,
                    unit="bin",
                    unit_index=key.unit_index,
                    trial=_require_shard_int(data, "trial", label=path.name),
                    control_kind=_require_shard_str(data, "control_kind", label=path.name),
                    seed=_require_shard_int(data, "control_seed", label=path.name),
                )
                if _require_shard_str(data, "control_kind", label=path.name) != key.condition:
                    raise RunnerError(f"{diagnostic_prefix} control kind drifted: {path.name}")
        if key.unit == "layer":
            if data["head_count"] != 32:
                raise RunnerError(f"{diagnostic_prefix} depth head count drifted: {path.name}")
            if (
                "mean_source_r2" not in data
                or "selected_heads" not in data
                or "selected_head_digest_sha256" not in data
                or "source_kernel_digest_sha256" not in data
            ):
                raise RunnerError(f"{diagnostic_prefix} depth shard fields missing: {path.name}")
        return data
    except KeyError as exc:
        raise RunnerError(
            f"{diagnostic_prefix} missing required shard field {exc.args[0]!r}: {path.name}"
        ) from exc


def _build_terminal_payload(
    *,
    root: RunRoot,
    resolved_config: ResolvedConfig,
    sweep: SweepConfig,
    batch_manifest: BatchManifestPayload,
    expected: Mapping[str, ConditionKey],
    actual_paths: Mapping[str, Path],
    schema_version: int = 1,
) -> JsonObject:
    summaries: list[SummaryRowPayload] = []
    shard_payloads: dict[str, ShardPayload] = {}
    shard_identities: dict[str, str] = {}
    fit_profiles: dict[tuple[str, str], FrozenFitProfile] = {}
    depth_baseline_payloads: dict[str, ShardPayload] = {}
    for model_name in sweep.models:
        binding = freeze_model_binding(resolved_config.models[model_name])
        for domain_name in resolved_config.datasets:
            entry = batch_manifest["domain_entries"][_domain_entry_name(model_name, domain_name)]
            fit_profiles[(model_name, domain_name)] = _load_fit_profile(
                run_root=root,
                resolved_config=resolved_config,
                sweep=sweep,
                model_name=model_name,
                domain_name=domain_name,
                binding=binding,
                materialized_domain_sha256=str(entry["materialized_domain_sha256"]),
                fit_sequence_digest_sha256=str(entry["fit_sequence_digest_sha256"]),
            )
    for stem, key in sorted(expected.items()):
        payload = _require_shard_payload(
            _load_shard(actual_paths[stem]),
            path=actual_paths[stem],
            key=key,
            resolved_config=resolved_config,
            sweep=sweep,
            batch_manifest=batch_manifest,
        )
        source_domain, target_domain = _direction_domains(key.direction)
        source_profile = fit_profiles[(key.model, source_domain)]
        target_profile = fit_profiles[(key.model, target_domain)]
        if key.condition != "baseline":
            if (
                _require_shard_str(
                    payload, "source_fit_profile_sha256", label=actual_paths[stem].name
                )
                != source_profile.payload["identity_sha256"]
            ):
                raise RunnerError(
                    f"Finalizer source fit-profile binding drifted: {actual_paths[stem].name}"
                )
            if (
                _require_shard_str(
                    payload, "target_fit_profile_sha256", label=actual_paths[stem].name
                )
                != target_profile.payload["identity_sha256"]
            ):
                raise RunnerError(
                    f"Finalizer target fit-profile binding drifted: {actual_paths[stem].name}"
                )
        if key.unit == "bin":
            bin_index = _parse_fixed_unit_index(key.unit_index, unit="bin")
            heads = source_profile.bins[bin_index]
            selected_heads = _head_records(heads)
            if _require_shard_heads(payload, label=actual_paths[stem].name) != selected_heads:
                raise RunnerError(
                    f"Finalizer selected-head membership drifted: {actual_paths[stem].name}"
                )
            if _require_shard_str(
                payload, "selected_head_digest_sha256", label=actual_paths[stem].name
            ) != _selected_head_digest(heads):
                raise RunnerError(
                    f"Finalizer selected-head digest recompute drifted: {actual_paths[stem].name}"
                )
            if (
                _require_shard_str(
                    payload, "selected_bin_digest_sha256", label=actual_paths[stem].name
                )
                != source_profile.payload["bins"][bin_index]["identity_sha256"]
            ):
                raise RunnerError(
                    f"Finalizer selected-bin digest drifted: {actual_paths[stem].name}"
                )
            expected_mean = float(source_profile.bin_means[bin_index])
            if not math.isclose(
                _require_shard_float(payload, "mean_source_r2", label=actual_paths[stem].name),
                expected_mean,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RunnerError(
                    f"Finalizer source-R2 bin mean drifted: {actual_paths[stem].name}"
                )
            source_map = _kernel_map_for_heads(heads, source_profile.raw_kernels)
            target_map = _kernel_map_for_heads(heads, target_profile.raw_kernels)
            if _require_shard_str(
                payload, "source_kernel_digest_sha256", label=actual_paths[stem].name
            ) != _kernel_digest(source_map.items()):
                raise RunnerError(
                    f"Finalizer source-kernel digest drifted: {actual_paths[stem].name}"
                )
            if key.condition == "target_kernel":
                if _require_shard_str(
                    payload, "target_kernel_digest_sha256", label=actual_paths[stem].name
                ) != _kernel_digest(target_map.items()):
                    raise RunnerError(
                        f"Finalizer target-kernel digest drifted: {actual_paths[stem].name}"
                    )
            if key.condition in ("offset_permutation", "norm"):
                control_map = _streaming_control_maps(
                    heads,
                    source_map,
                    seed=_require_shard_int(payload, "control_seed", label=actual_paths[stem].name),
                    control_kind=_require_shard_str(
                        payload, "control_kind", label=actual_paths[stem].name
                    ),
                )
                if _require_shard_str(
                    payload, "control_kernel_digest_sha256", label=actual_paths[stem].name
                ) != _kernel_digest(control_map.items()):
                    raise RunnerError(
                        f"Finalizer control-kernel digest drifted: {actual_paths[stem].name}"
                    )
        if key.unit == "layer":
            layer_index = _parse_fixed_unit_index(key.unit_index, unit="layer")
            heads = tuple(HeadIndex(layer=layer_index, head=head) for head in range(32))
            expected_heads = _head_records(heads)
            if _require_shard_heads(payload, label=actual_paths[stem].name) != expected_heads:
                raise RunnerError(
                    f"Finalizer depth selected-head membership drifted: {actual_paths[stem].name}"
                )
            if _require_shard_str(
                payload, "selected_head_digest_sha256", label=actual_paths[stem].name
            ) != _selected_head_digest(heads):
                raise RunnerError(
                    f"Finalizer depth selected-head digest drifted: {actual_paths[stem].name}"
                )
            expected_mean = float(
                np.asarray(source_profile.mean_scores[layer_index], dtype=np.float64).mean()
            )
            if not math.isclose(
                _require_shard_float(payload, "mean_source_r2", label=actual_paths[stem].name),
                expected_mean,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RunnerError(
                    f"Finalizer depth mean source-R2 drifted: {actual_paths[stem].name}"
                )
            if _require_shard_str(
                payload, "source_kernel_digest_sha256", label=actual_paths[stem].name
            ) != _kernel_digest(_kernel_map_for_heads(heads, source_profile.raw_kernels).items()):
                raise RunnerError(
                    f"Finalizer depth source-kernel digest drifted: {actual_paths[stem].name}"
                )
        shard_payloads[stem] = payload
        shard_identities[stem] = str(payload["identity_sha256"])
        summaries.append({"stem": stem, "mean_nll": float(np.mean(payload["sequence_nll"]))})
        if (
            key.unit == "global"
            and key.condition == "baseline"
            and key.direction == "wikipedia_to_code"
        ):
            depth_baseline_payloads[key.model] = payload

    baselines: dict[tuple[str, str], Float64Array] = {}
    bin_rows: list[tuple[str, str, float, float]] = []
    depth_rows: list[DepthRowPayload] = []
    for model_name in sweep.models:
        for direction in sweep.directions:
            base_key = ConditionKey(
                model=model_name,
                direction=direction,
                unit="global",
                unit_index="baseline",
                condition="baseline",
            )
            baseline_payload = shard_payloads[base_key.stem()]
            baselines[(model_name, direction)] = baseline_payload["sequence_nll"]
            for bin_index in range(FIXED_COUNTS["head_bins"]):
                source_key = ConditionKey(
                    model=model_name,
                    direction=direction,
                    unit="bin",
                    unit_index=str(bin_index),
                    condition="source_kernel",
                )
                source_payload = shard_payloads[source_key.stem()]
                if (
                    _require_shard_str(
                        source_payload, "baseline_identity_sha256", label=source_key.stem()
                    )
                    != baseline_payload["identity_sha256"]
                ):
                    raise RunnerError(f"Baseline linkage drifted: {source_key.stem()}")
                if (
                    source_payload["sequence_digest_sha256"]
                    != baseline_payload["sequence_digest_sha256"]
                ):
                    raise RunnerError(f"Sequence binding drifted: {source_key.stem()}")
                source_nll = source_payload["sequence_nll"]
                delta = float(
                    np.mean(source_nll - baselines[(model_name, direction)])
                    / int(source_payload["head_count"])
                )
                bin_rows.append(
                    (
                        model_name,
                        direction,
                        _require_shard_float(
                            source_payload, "mean_source_r2", label=source_key.stem()
                        ),
                        delta,
                    )
                )
                target_key = ConditionKey(
                    model=model_name,
                    direction=direction,
                    unit="bin",
                    unit_index=str(bin_index),
                    condition="target_kernel",
                )
                target_payload = shard_payloads[target_key.stem()]
                if (
                    _require_shard_str(
                        target_payload, "baseline_identity_sha256", label=target_key.stem()
                    )
                    != baseline_payload["identity_sha256"]
                ):
                    raise RunnerError(f"Baseline linkage drifted: {target_key.stem()}")
                if (
                    target_payload["sequence_digest_sha256"]
                    != baseline_payload["sequence_digest_sha256"]
                ):
                    raise RunnerError(f"Sequence binding drifted: {target_key.stem()}")
                for control_kind in ("offset_permutation", "norm"):
                    for trial in range(3):
                        control_key = ConditionKey(
                            model=model_name,
                            direction=direction,
                            unit="bin",
                            unit_index=str(bin_index),
                            condition=control_kind,
                            trial=trial,
                        )
                        control_payload = shard_payloads[control_key.stem()]
                        if (
                            _require_shard_str(
                                control_payload,
                                "baseline_identity_sha256",
                                label=control_key.stem(),
                            )
                            != baseline_payload["identity_sha256"]
                        ):
                            raise RunnerError(f"Baseline linkage drifted: {control_key.stem()}")
                        if (
                            control_payload["sequence_digest_sha256"]
                            != baseline_payload["sequence_digest_sha256"]
                        ):
                            raise RunnerError(f"Sequence binding drifted: {control_key.stem()}")
        if sweep.include_depth:
            for layer_index in range(32):
                depth_key = ConditionKey(
                    model=model_name,
                    direction="wikipedia_to_code",
                    unit="layer",
                    unit_index=str(layer_index),
                    condition="depth",
                )
                depth_payload = shard_payloads[depth_key.stem()]
                baseline_payload = depth_baseline_payloads[model_name]
                baseline = baseline_payload["sequence_nll"]
                if (
                    _require_shard_str(
                        depth_payload, "baseline_identity_sha256", label=depth_key.stem()
                    )
                    != baseline_payload["identity_sha256"]
                ):
                    raise RunnerError(f"Baseline linkage drifted: {depth_key.stem()}")
                if (
                    depth_payload["sequence_digest_sha256"]
                    != baseline_payload["sequence_digest_sha256"]
                ):
                    raise RunnerError(f"Sequence binding drifted: {depth_key.stem()}")
                grouped_delta = float(np.mean(depth_payload["sequence_nll"] - baseline) / 32.0)
                depth_rows.append(
                    {
                        "model": model_name,
                        "direction": "wikipedia_to_code",
                        "layer": layer_index,
                        "mean_source_r2": _require_shard_float(
                            depth_payload, "mean_source_r2", label=depth_key.stem()
                        ),
                        "grouped_loss_delta_per_head": grouped_delta,
                        "shard_identity_sha256": depth_payload["identity_sha256"],
                        "baseline_identity_sha256": baseline_payload["identity_sha256"],
                    }
                )

    stats: list[DoseResponseStatisticPayload] = []
    contrasts: list[ContrastPayload] = []
    depth_relationships: list[DepthRelationshipPayload] = []
    for model_name in sweep.models:
        for direction in sweep.directions:
            rows = [
                (r2, delta)
                for model, direct, r2, delta in bin_rows
                if model == model_name and direct == direction
            ]
            seed_parts = [
                29039,
                model_name,
                direction,
                "statistic",
                "all",
                0,
                "spearman_response_permutation",
            ]
            rho = monte_carlo_spearman_positive(
                [row[0] for row in rows],
                [row[1] for row in rows],
                seed=compact_json_seed(seed_parts),
            )
            stat_payload: DoseResponseStatisticPayload = {
                "model": model_name,
                "direction": direction,
                "rho": rho.rho_observed,
                "p_one_sided": rho.p_one_sided,
                "p_two_sided_scipy": rho.p_two_sided_scipy,
            }
            if schema_version == 2:
                stat_payload["permutation_count"] = rho.permutation_count
                stat_payload["permutation_exceedance_count"] = rho.permutation_exceedance_count
                stat_payload["permutation_seed"] = compact_json_seed(seed_parts)
                stat_payload["permutation_seed_provenance"] = _json_object(
                    {
                        "seed_namespace": resolved_config.seed_namespace,
                        "seed_parts": seed_parts,
                        "method": "compact_json_seed",
                    },
                    label=f"dose_response_statistics.{model_name}.{direction}.seed_provenance",
                )
                _require_permutation_pvalue_invariant(
                    label=f"dose_response_statistics.{model_name}.{direction}",
                    p_one_sided=stat_payload["p_one_sided"],
                    permutation_count=stat_payload["permutation_count"],
                    permutation_exceedance_count=stat_payload["permutation_exceedance_count"],
                )
            stats.append(stat_payload)
            baseline = baselines[(model_name, direction)]
            for bin_index in range(FIXED_COUNTS["head_bins"]):
                source_key = ConditionKey(
                    model=model_name,
                    direction=direction,
                    unit="bin",
                    unit_index=str(bin_index),
                    condition="source_kernel",
                )
                target_key = ConditionKey(
                    model=model_name,
                    direction=direction,
                    unit="bin",
                    unit_index=str(bin_index),
                    condition="target_kernel",
                )
                source_values = shard_payloads[source_key.stem()]["sequence_nll"]
                target_values = shard_payloads[target_key.stem()]["sequence_nll"]
                source_delta = paired_percentile_bootstrap(
                    baseline,
                    source_values,
                    seed=compact_json_seed(
                        [29039, model_name, direction, "statistic", bin_index, 0, "source_delta"]
                    ),
                )
                target_delta = paired_percentile_bootstrap(
                    baseline,
                    target_values,
                    seed=compact_json_seed(
                        [29039, model_name, direction, "statistic", bin_index, 0, "target_delta"]
                    ),
                )
                offset_trials: list[Float64Array] = []
                norm_trials: list[Float64Array] = []
                for trial in range(3):
                    offset_key = ConditionKey(
                        model=model_name,
                        direction=direction,
                        unit="bin",
                        unit_index=str(bin_index),
                        condition="offset_permutation",
                        trial=trial,
                    )
                    norm_key = ConditionKey(
                        model=model_name,
                        direction=direction,
                        unit="bin",
                        unit_index=str(bin_index),
                        condition="norm",
                        trial=trial,
                    )
                    offset_trials.append(shard_payloads[offset_key.stem()]["sequence_nll"])
                    norm_trials.append(shard_payloads[norm_key.stem()]["sequence_nll"])
                offset_avg = average_control_trials_per_sequence(np.stack(offset_trials, axis=0))
                norm_avg = average_control_trials_per_sequence(np.stack(norm_trials, axis=0))
                offset_delta = paired_percentile_bootstrap(
                    baseline,
                    offset_avg,
                    seed=compact_json_seed(
                        [29039, model_name, direction, "statistic", bin_index, 0, "offset_delta"]
                    ),
                )
                norm_delta = paired_percentile_bootstrap(
                    baseline,
                    norm_avg,
                    seed=compact_json_seed(
                        [29039, model_name, direction, "statistic", bin_index, 0, "norm_delta"]
                    ),
                )
                source_minus_target = paired_percentile_bootstrap(
                    target_values,
                    source_values,
                    seed=compact_json_seed(
                        [
                            29039,
                            model_name,
                            direction,
                            "statistic",
                            bin_index,
                            0,
                            "source_minus_target",
                        ]
                    ),
                )
                source_minus_offset = paired_percentile_bootstrap(
                    offset_avg,
                    source_values,
                    seed=compact_json_seed(
                        [
                            29039,
                            model_name,
                            direction,
                            "statistic",
                            bin_index,
                            0,
                            "source_minus_offset",
                        ]
                    ),
                )
                source_minus_norm = paired_percentile_bootstrap(
                    norm_avg,
                    source_values,
                    seed=compact_json_seed(
                        [
                            29039,
                            model_name,
                            direction,
                            "statistic",
                            bin_index,
                            0,
                            "source_minus_norm",
                        ]
                    ),
                )
                contrasts.append(
                    {
                        "model": model_name,
                        "direction": direction,
                        "bin": bin_index,
                        "source_delta": {
                            "point": source_delta.point_estimate,
                            "ci_low": source_delta.ci_low,
                            "ci_high": source_delta.ci_high,
                        },
                        "target_delta": {
                            "point": target_delta.point_estimate,
                            "ci_low": target_delta.ci_low,
                            "ci_high": target_delta.ci_high,
                        },
                        "offset_delta": {
                            "point": offset_delta.point_estimate,
                            "ci_low": offset_delta.ci_low,
                            "ci_high": offset_delta.ci_high,
                        },
                        "norm_delta": {
                            "point": norm_delta.point_estimate,
                            "ci_low": norm_delta.ci_low,
                            "ci_high": norm_delta.ci_high,
                        },
                        "source_minus_target": {
                            "point": source_minus_target.point_estimate,
                            "ci_low": source_minus_target.ci_low,
                            "ci_high": source_minus_target.ci_high,
                        },
                        "source_minus_offset": {
                            "point": source_minus_offset.point_estimate,
                            "ci_low": source_minus_offset.ci_low,
                            "ci_high": source_minus_offset.ci_high,
                        },
                        "source_minus_norm": {
                            "point": source_minus_norm.point_estimate,
                            "ci_low": source_minus_norm.ci_low,
                            "ci_high": source_minus_norm.ci_high,
                        },
                    }
                )
        if sweep.include_depth:
            model_depth_rows = [row for row in depth_rows if row["model"] == model_name]
            depth_r2 = [row["mean_source_r2"] for row in model_depth_rows]
            depth_delta = [row["grouped_loss_delta_per_head"] for row in model_depth_rows]
            depth_seed_parts = [
                29039,
                model_name,
                "wikipedia_to_code",
                "statistic",
                "all",
                0,
                "spearman_response_permutation",
            ]
            depth_seed = compact_json_seed(depth_seed_parts)
            rho = monte_carlo_spearman_positive(
                depth_r2,
                depth_delta,
                seed=depth_seed,
            )
            permutation_seed_provenance = _json_object(
                {
                    "seed_namespace": resolved_config.seed_namespace,
                    "seed_parts": depth_seed_parts,
                    "method": "compact_json_seed",
                },
                label=f"depth_relationship.{model_name}.permutation_seed_provenance",
            )
            depth_relationship: DepthRelationshipPayload = {
                "model": model_name,
                "direction": "wikipedia_to_code",
                "group_level": True,
                "layers": 32,
                "spearman_rho": rho.rho_observed,
                "p_one_sided": rho.p_one_sided,
                "p_two_sided_scipy": rho.p_two_sided_scipy,
                "permutation_seed": depth_seed,
                "permutation_seed_provenance": permutation_seed_provenance,
                "layer_indices": [int(row["layer"]) for row in model_depth_rows],
                "mean_source_r2_inputs": depth_r2,
                "grouped_loss_delta_per_head_inputs": depth_delta,
                "depth_row_identity_sha256s": [
                    str(row["shard_identity_sha256"]) for row in model_depth_rows
                ],
                "mean_grouped_loss_delta_per_head": float(
                    np.mean([row["grouped_loss_delta_per_head"] for row in model_depth_rows])
                ),
                "min_grouped_loss_delta_per_head": float(
                    np.min([row["grouped_loss_delta_per_head"] for row in model_depth_rows])
                ),
                "max_grouped_loss_delta_per_head": float(
                    np.max([row["grouped_loss_delta_per_head"] for row in model_depth_rows])
                ),
            }
            if schema_version == 1:
                depth_relationship["positive_null_permutation_count"] = (
                    DEFAULT_SPEARMAN_PERMUTATIONS
                )
            else:
                depth_relationship["permutation_count"] = rho.permutation_count
                depth_relationship["permutation_exceedance_count"] = (
                    rho.permutation_exceedance_count
                )
            depth_relationships.append(depth_relationship)
            if schema_version == 2:
                depth_payload = depth_relationships[-1]
                _require_permutation_pvalue_invariant(
                    label=f"depth_relationships.{model_name}.wikipedia_to_code",
                    p_one_sided=depth_payload["p_one_sided"],
                    permutation_count=_json_int_member(
                        depth_payload,
                        "permutation_count",
                        label=f"depth_relationships.{model_name}",
                    ),
                    permutation_exceedance_count=_json_int_member(
                        depth_payload,
                        "permutation_exceedance_count",
                        label=f"depth_relationships.{model_name}",
                    ),
                )

    return _json_object(
        {
            "schema_version": schema_version,
            "run_id": root.run_id,
            "config_sha256": resolved_config.identity_sha256,
            "sweep_sha256": sweep.identity_sha256,
            "batch_manifest_sha256": batch_manifest["identity_sha256"],
            "shards_verified": len(expected),
            "summary_rows": summaries,
            "dose_response_statistics": stats,
            "contrasts": contrasts,
            "depth_rows": depth_rows,
            "depth_relationships": depth_relationships,
            "summary_hashes": _json_object(
                {
                    "shard_inventory_sha256": artifact_identity(shard_identities),
                    "summary_rows_sha256": artifact_identity(summaries),
                    "dose_response_statistics_sha256": artifact_identity(stats),
                    "contrasts_sha256": artifact_identity(contrasts),
                    "depth_rows_sha256": artifact_identity(depth_rows),
                    "depth_relationships_sha256": artifact_identity(depth_relationships),
                },
                label="terminal_summary.summary_hashes",
            ),
        },
        label="terminal_summary",
    )


def finalize_batch(base_path: str | Path, sweep_path: str | Path, *, run_root: Path) -> JsonObject:
    resolved_config, sweep = load_configs(base_path, sweep_path)
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id=None)
    batch_manifest = _load_batch_manifest(root)
    _ = batch_manifest["models"]
    _ = batch_manifest["tokenizers"]
    _ = batch_manifest["git_commit_sha"]
    _ = batch_manifest["git_tracked_diff_sha256"]
    _ = batch_manifest["git_relevant_content_sha256"]
    _ = batch_manifest["git_untracked_sha256"]
    _ = batch_manifest["runtime"]
    _ = batch_manifest["environment"]
    _ = batch_manifest["provenance"]
    expected = {
        key.stem(): key
        for model_name in sweep.models
        for key in _expected_condition_inventory(model_name, include_depth=sweep.include_depth)
    }
    actual_paths = {path.stem: path for path in root.shards_dir.glob("*.json")}
    missing = sorted(set(expected) - set(actual_paths))
    extra = sorted(set(actual_paths) - set(expected))
    if missing:
        raise RunnerError(f"Finalizer missing shard: {missing[0]}")
    if extra:
        raise RunnerError(f"Finalizer found unexpected shard: {extra[0]}")
    summary_path = _summary_v1_path(root)
    if not summary_path.exists():
        raise RunnerError(
            "Legacy finalize-batch is verify-only and requires an existing v1 summary."
        )
    existing = _load_existing_terminal_payload(summary_path, label=summary_path.name)
    expected_terminal = _build_terminal_payload(
        root=root,
        resolved_config=resolved_config,
        sweep=sweep,
        batch_manifest=batch_manifest,
        expected=expected,
        actual_paths=actual_paths,
        schema_version=1,
    )
    expected_completed_at = _json_string_member(existing, "completed_at", label=summary_path.name)
    if not expected_completed_at:
        raise RunnerError("Finalizer completion timestamp drifted.")
    expected_terminal["completed_at"] = expected_completed_at
    expected_terminal["identity_sha256"] = _terminal_payload_identity(expected_terminal)
    if existing != expected_terminal:
        raise RunnerError("Finalizer completion payload drifted.")
    return _json_object(existing, label=summary_path.name)


def refinalize_batch_v2(
    base_path: str | Path,
    sweep_path: str | Path,
    *,
    run_root: Path,
    source_summary_v1: Path,
) -> JsonObject:
    _require_exact_relative_cli_path(
        base_path, expected=_PRODUCTION_CONFIG_RELATIVE_PATH, label="Config path"
    )
    _require_exact_relative_cli_path(
        sweep_path, expected=_PRODUCTION_SWEEP_RELATIVE_PATH, label="Sweep path"
    )
    resolved_config, sweep = load_configs(base_path, sweep_path)
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id=None)
    project_root = _production_repo_root_from_project_root(resolved_config.paths.project_root)
    config_path = (project_root / _PRODUCTION_CONFIG_RELATIVE_PATH).resolve()
    canonical_sweep_path = (project_root / _PRODUCTION_SWEEP_RELATIVE_PATH).resolve()
    if _resolve_cli_path(base_path).resolve() != config_path:
        raise RunnerError("Config path resolved away from the canonical production source.")
    if _resolve_cli_path(sweep_path).resolve() != canonical_sweep_path:
        raise RunnerError("Sweep path resolved away from the canonical production source.")
    if resolved_config.source_path.resolve() != config_path:
        raise RunnerError("Resolved config source path drifted from the canonical production file.")
    if sweep.source_path.resolve() != canonical_sweep_path:
        raise RunnerError("Resolved sweep source path drifted from the canonical production file.")
    canonical_v1_path = _summary_v1_path(root).resolve()
    if source_summary_v1.resolve() != canonical_v1_path:
        raise RunnerError("refinalize-batch-v2 requires the canonical run-root v1 summary path.")
    _require_canonical_production_source_v1_file(canonical_v1_path)
    config_byte_sha256 = _require_fixed_production_config_raw_digest(config_path)
    sweep_byte_sha256 = _require_fixed_production_sweep_raw_digest(canonical_sweep_path)
    source_v1_byte_sha256 = _require_fixed_production_source_v1_raw_digest(canonical_v1_path)
    source_v1 = finalize_batch(base_path, sweep_path, run_root=run_root)
    (
        manifest_paths,
        batch_manifest,
        fit_profile_identity_sha256s,
    ) = _require_production_manifest_graph(
        run_root=root,
        resolved_config=resolved_config,
        sweep=sweep,
        source_v1=source_v1,
    )
    expected = {
        key.stem(): key
        for model_name in sweep.models
        for key in _expected_condition_inventory(model_name, include_depth=sweep.include_depth)
    }
    if len(expected) != _EXPECTED_PRODUCTION_SHARD_COUNT:
        raise RunnerError("Production shard inventory shape drifted.")
    actual_paths = {path.stem: path for path in root.shards_dir.glob("*.json")}
    missing = sorted(set(expected) - set(actual_paths))
    extra = sorted(set(actual_paths) - set(expected))
    if missing:
        raise RunnerError(f"Finalizer missing shard: {missing[0]}")
    if extra:
        raise RunnerError(f"Finalizer found unexpected shard: {extra[0]}")
    if len(actual_paths) != _EXPECTED_PRODUCTION_SHARD_COUNT:
        raise RunnerError(
            f"Production shard inventory count drifted: expected "
            f"{_EXPECTED_PRODUCTION_SHARD_COUNT}, found {len(actual_paths)}."
        )
    immutable_manifest = _immutable_input_manifest(
        run_root=root,
        resolved_config=resolved_config,
        sweep=sweep,
        project_root=project_root,
        config_path=config_path,
        sweep_path=canonical_sweep_path,
        source_summary_v1_path=canonical_v1_path,
    )
    immutable_manifest = _require_fixed_production_immutable_input_manifest(immutable_manifest)
    v2_payload = _build_terminal_payload(
        root=root,
        resolved_config=resolved_config,
        sweep=sweep,
        batch_manifest=batch_manifest,
        expected=expected,
        actual_paths=actual_paths,
        schema_version=2,
    )
    v1_projection = _scientific_projection(source_v1)
    v2_projection = _scientific_projection(v2_payload)
    if canonical_json_bytes(v1_projection) != canonical_json_bytes(v2_projection):
        raise RunnerError("Scientific projection drifted between v1 and v2 terminal summaries.")
    shard_identity_inventory = {
        path.stem: _json_string_member(_load_shard(path), "identity_sha256", label=path.name)
        for path in sorted(actual_paths.values())
    }
    summary_v2_path = _summary_v2_path(root)
    existing_v2: JsonObject | None = None
    completed_at: str
    if summary_v2_path.exists():
        existing_v2 = _load_existing_terminal_payload(summary_v2_path, label=summary_v2_path.name)
        completed_at = _json_string_member(existing_v2, "completed_at", label=summary_v2_path.name)
    else:
        completed_at = utc_now()
    v2_payload["completed_at"] = completed_at
    v2_payload["immutable_input_manifest"] = immutable_manifest
    v2_payload["correction_provenance"] = _json_object(
        {
            "source_summary_v1": {
                "path": os.path.relpath(canonical_v1_path, project_root),
                "byte_sha256": source_v1_byte_sha256,
                "identity_sha256": _json_string_member(
                    source_v1, "identity_sha256", label="source_summary_v1"
                ),
            },
            "source_config": {
                "path": _PRODUCTION_CONFIG_RELATIVE_PATH.as_posix(),
                "byte_sha256": config_byte_sha256,
                "resolved_identity_sha256": resolved_config.identity_sha256,
            },
            "source_sweep": {
                "path": _PRODUCTION_SWEEP_RELATIVE_PATH.as_posix(),
                "byte_sha256": sweep_byte_sha256,
                "resolved_identity_sha256": sweep.identity_sha256,
            },
            "batch_manifest_identity_sha256": batch_manifest["identity_sha256"],
            "config_identity_sha256": resolved_config.identity_sha256,
            "sweep_identity_sha256": sweep.identity_sha256,
            "manifest_file_count": len(manifest_paths),
            "shard_count": len(actual_paths),
            "fit_profile_identity_sha256s": fit_profile_identity_sha256s,
            "immutable_input_manifest_sha256": _json_string_member(
                immutable_manifest, "content_sha256", label="immutable_input_manifest"
            ),
            "immutable_input_manifest_identity_sha256": _json_string_member(
                immutable_manifest, "identity_sha256", label="immutable_input_manifest"
            ),
            "scientific_projection_sha256": sha256_bytes(canonical_json_bytes(v2_projection)),
            "shard_inventory_sha256": artifact_identity(shard_identity_inventory),
        },
        label="correction_provenance",
    )
    summary_hashes = _json_object_member(v2_payload, "summary_hashes", label="terminal_v2")
    summary_hashes["scientific_projection_sha256"] = sha256_bytes(
        canonical_json_bytes(v2_projection)
    )
    summary_hashes["immutable_input_manifest_sha256"] = _json_string_member(
        immutable_manifest, "content_sha256", label="immutable_input_manifest"
    )
    v2_payload["summary_hashes"] = summary_hashes
    v2_payload["identity_sha256"] = _terminal_payload_identity(v2_payload)
    expected_v2_bytes = _terminal_payload_bytes(v2_payload)
    if summary_v2_path.exists():
        if existing_v2 is None:
            raise RunnerError("Existing finalizer v2 payload was not loaded.")
        if existing_v2 != v2_payload:
            raise RunnerError("Finalizer v2 payload drifted.")
        if summary_v2_path.read_bytes() != expected_v2_bytes:
            raise RunnerError("Finalizer v2 bytes drifted.")
        return _json_object(existing_v2, label=summary_v2_path.name)
    _publish_terminal_summary_v2_no_clobber(
        path=summary_v2_path,
        payload=v2_payload,
        payload_bytes=expected_v2_bytes,
        source_v1=source_v1,
        root=root,
        resolved_config=resolved_config,
        sweep=sweep,
        project_root=project_root,
        config_path=config_path,
        sweep_path=canonical_sweep_path,
        source_summary_v1_path=canonical_v1_path,
        expected_immutable_manifest=immutable_manifest,
    )
    return _json_object(v2_payload, label=summary_v2_path.name)


def launch(
    base_path: str | Path,
    sweep_path: str | Path,
    *,
    run_root: Path,
    batch_id: str,
    execute: bool,
    run_command: RuntimeCommandRunner = default_subprocess_runner,
) -> JsonObject:
    resolved_config, sweep = load_configs(base_path, sweep_path)
    if not run_root.exists():
        raise RunnerError(
            "launch requires the already-existing exact materialized immutable run root."
        )
    if not run_root.is_dir():
        raise RunnerError("launch run root must be a directory.")
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id=batch_id)
    batch_manifest = _load_batch_manifest(root)
    _require_receipts(root, resolved_config, sweep, tuple(sweep.models))
    lane_payload = _load_lane_admission_receipt(root, resolved_config=resolved_config, sweep=sweep)
    if not execute:
        plan = build_launch_plan(
            resolved_config=resolved_config,
            sweep=sweep,
            batch_id=batch_id,
            run_root=run_root,
            execute=False,
            python_executable=sys.executable,
        )
        return _json_object(
            {
                "schema_version": 1,
                "batch_id": batch_id,
                "run_root": str(run_root),
                "batch_manifest_sha256": batch_manifest["identity_sha256"],
                "lane_admission_receipt_sha256": lane_payload["identity_sha256"],
                "commands": [command.shell_command for command in plan.commands],
                "launch_script_paths": {
                    command.tmux_session: str(command.script_path) for command in plan.commands
                },
                "launch_log_paths": {
                    command.tmux_session: str(command.log_path) for command in plan.commands
                },
                "tmux_commands": {
                    command.tmux_session: command.tmux_command for command in plan.commands
                },
            },
            label="launch_dry_run",
        )
    _require_current_batch_manifest_provenance(
        batch_manifest=batch_manifest,
        resolved_config=resolved_config,
        runtime_command_runner=run_command,
        stage="launch",
    )
    for _physical_index, session_name in ((0, "si-rebuttal-gpu0"), (1, "si-rebuttal-gpu1")):
        require_tmux_absent(session_name=session_name, run_command=run_command)
    snapshots = {
        0: require_two_idle_snapshots(physical_index=0, run_command=run_command),
        1: require_two_idle_snapshots(physical_index=1, run_command=run_command),
    }
    for session_name in ("si-rebuttal-gpu0", "si-rebuttal-gpu1"):
        if (_receipt_path(root, session_name).exists()) or (
            root.receipts_dir / f"{session_name}.launch.json"
        ).exists():
            raise RunnerError(f"Launch marker already exists for {session_name}.")
    plan = build_launch_plan(
        resolved_config=resolved_config,
        sweep=sweep,
        batch_id=batch_id,
        run_root=run_root,
        execute=True,
        python_executable=sys.executable,
    )
    write_launch_scripts_and_logs(run_root=root, plan=plan)
    _write_launch_receipts(
        run_root=root,
        plan=plan,
        snapshots=snapshots,
        command=sys.argv,
        config_sha256=resolved_config.identity_sha256,
        sweep_sha256=sweep.identity_sha256,
        batch_manifest_sha256=str(batch_manifest["identity_sha256"]),
        lane_admission_receipt_sha256=str(lane_payload["identity_sha256"]),
    )
    launch_tmux_plan(plan=plan, run_command=run_command)
    return _json_object(
        {
            "schema_version": 1,
            "batch_id": batch_id,
            "launched": True,
            "run_root": str(run_root),
        },
        label="launch_execute",
    )


def _tail_lines(path: Path, *, limit: int = 10) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    lines = text.splitlines()
    return lines[-limit:]


def status(
    *, run_root: Path, run_command: RuntimeCommandRunner = default_subprocess_runner
) -> JsonObject:
    status_payload = _json_object(
        read_only_status(run_root=run_root, run_command=run_command), label="read_only_status"
    )
    receipts_dir = run_root / "receipts"
    if not receipts_dir.exists():
        return status_payload
    launch_receipts: dict[str, JsonObject] = {}
    launch_scripts: dict[str, JsonObject] = {}
    launch_logs: dict[str, JsonObject] = {}
    for session_name in ("si-rebuttal-gpu0", "si-rebuttal-gpu1"):
        receipt_path = receipts_dir / f"{session_name}.json"
        if receipt_path.exists():
            receipt = _read_verified_payload(receipt_path, label=receipt_path.name)
            script_path = Path(
                _json_string_member(receipt, "launch_script_path", label=receipt_path.name)
            )
            log_path = Path(
                _json_string_member(receipt, "launch_log_path", label=receipt_path.name)
            )
            model_names = [
                _json_string(item, label=f"{receipt_path.name}.model_names[{index}]")
                for index, item in enumerate(
                    _json_array_member(receipt, "model_names", label=receipt_path.name)
                )
            ]
            launch_receipts[session_name] = _json_object(
                {
                    "identity_sha256": receipt["identity_sha256"],
                    "batch_manifest_sha256": _json_string_member(
                        receipt, "batch_manifest_sha256", label=receipt_path.name
                    ),
                    "lane_admission_receipt_sha256": _json_string_member(
                        receipt, "lane_admission_receipt_sha256", label=receipt_path.name
                    ),
                    "model_names": model_names,
                    "launch_script_path": str(script_path),
                    "launch_script_sha256": _json_string_member(
                        receipt, "launch_script_sha256", label=receipt_path.name
                    ),
                    "launch_log_path": str(log_path),
                },
                label=f"status.launch_receipt.{session_name}",
            )
            launch_scripts[session_name] = _json_object(
                {
                    "path": str(script_path),
                    "exists": script_path.exists(),
                    "readable": script_path.exists() and os.access(script_path, os.R_OK),
                    "sha256": _json_string_member(
                        receipt, "launch_script_sha256", label=receipt_path.name
                    ),
                },
                label=f"status.launch_script.{session_name}",
            )
            launch_logs[session_name] = _json_object(
                {
                    "path": str(log_path),
                    "exists": log_path.exists(),
                    "readable": log_path.exists() and os.access(log_path, os.R_OK),
                    "tail": _tail_lines(log_path)
                    if log_path.exists() and os.access(log_path, os.R_OK)
                    else [],
                },
                label=f"status.launch_log.{session_name}",
            )
    batch_path = receipts_dir / "batch.json"
    if batch_path.exists():
        batch_receipt = _read_verified_payload(batch_path, label=batch_path.name)
        status_payload["launch_batch_receipt"] = _json_object(
            {
                "identity_sha256": batch_receipt["identity_sha256"],
                "batch_manifest_sha256": _json_string_member(
                    batch_receipt, "batch_manifest_sha256", label=batch_path.name
                ),
                "lane_admission_receipt_sha256": _json_string_member(
                    batch_receipt, "lane_admission_receipt_sha256", label=batch_path.name
                ),
                "launch_receipts": _json_object_member(
                    batch_receipt, "launch_receipts", label=batch_path.name
                ),
                "launch_script_paths": _json_object_member(
                    batch_receipt, "launch_script_paths", label=batch_path.name
                ),
                "launch_script_sha256s": _json_object_member(
                    batch_receipt, "launch_script_sha256s", label=batch_path.name
                ),
                "launch_log_paths": _json_object_member(
                    batch_receipt, "launch_log_paths", label=batch_path.name
                ),
            },
            label="status.launch_batch_receipt",
        )
    status_payload["launch_receipts"] = _json_object(
        launch_receipts, label="status.launch_receipts"
    )
    status_payload["launch_scripts"] = _json_object(launch_scripts, label="status.launch_scripts")
    status_payload["launch_logs"] = _json_object(launch_logs, label="status.launch_logs")
    return status_payload
