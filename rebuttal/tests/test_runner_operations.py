from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import TracebackType
from typing import ClassVar, Protocol, TypeAlias, TypedDict, TypeGuard, TypeVar, cast

import numpy as np
import numpy.typing as npt
import pytest
import torch
from scipy import stats as scipy_stats
from torch.version import cuda as torch_cuda_version

import si_rebuttal.cli as cli_module
import si_rebuttal.runner as runner_module
from si_rebuttal.artifacts import (
    ArtifactError,
    ConditionKey,
    JsonObject,
    JsonValue,
    RunRoot,
    artifact_identity,
    ensure_payload_identity,
    payload_without_identity,
)
from si_rebuttal.artifacts import load_existing_or_write as _artifact_load_existing_or_write
from si_rebuttal.config import (
    ResolvedConfig,
    SweepConfig,
    load_base_config,
)
from si_rebuttal.data import (
    DocumentRecord,
    MaterializedDomain,
    SelectedChunk,
    token_sha256,
)
from si_rebuttal.intervention import ValidationProbeRecord, repeat_kv_after_gqa
from si_rebuttal.models import FrozenModelBinding, LoadedModelBundle
from si_rebuttal.operations import (
    BATCH_ID_ENV,
    BATCH_RECEIPT_ENV,
    GPU_IDLE_SLEEP_SECONDS,
    LAUNCH_RECEIPT_ENV,
    RUN_ROOT_ENV,
    TMUX_SESSION_ENV,
    GpuSnapshot,
    LaunchPlan,
    OperationsError,
    build_launch_plan,
    launch_tmux_plan,
    parse_nvidia_smi_snapshot,
    read_only_status,
    require_two_idle_snapshots,
    shell_join,
    snapshot_gpu,
)
from si_rebuttal.provenance import GitBinding, canonical_json_bytes, compact_json_seed
from si_rebuttal.runner import (
    BENCHMARK_ARTIFACT_FORMULA,
    BENCHMARK_RUNTIME_FORMULA,
    DISK_HEADROOM_FORMULA,
    LANE_HOURS_FORMULA,
    DatasetLoaderFn,
    DomainSequences,
    KernelMap,
    RuntimeCommandRunner,
    TokenizerLike,
    TokenizerLoader,
    _stream_post_rope_qk,  # pyright: ignore[reportPrivateUsage]
    admit_lanes,
    benchmark_model,
    finalize_batch,
    kernel_digest,
    launch,
    load_configs,
    materialize_data,
    project_artifact_bytes,
    run_model,
    sequence_digest,
    status,
    toy_smoke,
    validate_config,
    validate_model,
)
from si_rebuttal.runner import (
    BatchManifestPayload as RunnerBatchManifestPayload,
)
from si_rebuttal.statistics import DEFAULT_SPEARMAN_PERMUTATIONS

Int64Array: TypeAlias = npt.NDArray[np.int64]
Float64Array: TypeAlias = npt.NDArray[np.float64]
T = TypeVar("T")
RunRootMutation: TypeAlias = Callable[[RunRoot], object | None]
_PRODUCTION_RUNTIME_FAITHFUL_ATTENTION_PROBABILITIES = (
    ValidationProbeRecord.runtime_faithful_attention_probabilities
)
_CANONICAL_BASE_RAW_BYTE_SHA256 = "5662f9763a6a04660fb959ab989f28808a6945105e8dee76e70e0be8af92c9d0"
_CANONICAL_SWEEP_RAW_BYTE_SHA256 = (
    "cb8229585d1fff95a90ac373f748a169bfd3ddd42a23639472484b2af4005922"
)


class BatchDomainEntryPayload(TypedDict):
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


class BatchManifestPayload(TypedDict):
    schema_version: int
    materialized_at_utc: str
    command: list[str]
    config_sha256: str
    sweep_sha256: str
    domains: dict[str, str]
    domain_entries: dict[str, BatchDomainEntryPayload]
    models: dict[str, str]
    tokenizers: dict[str, str]
    git_commit_sha: str
    git_tracked_diff_sha256: str
    git_relevant_content_sha256: str
    git_untracked_sha256: str
    runtime: JsonObject
    runtime_capture: JsonObject
    environment: JsonObject
    provenance: JsonObject
    identity_sha256: str


@dataclass(frozen=True)
class FakeHead:
    layer: int
    head: int


@dataclass(frozen=True)
class FakeBinding:
    weights_tree_sha256: str
    tokenizer_tree_sha256: str


@dataclass(frozen=True)
class FakeLoadedBundle:
    binding: FakeBinding
    model: object
    tokenizer: object
    adapter: object | None
    logical_device: torch.device


@dataclass(frozen=True)
class FakeFitProfile:
    payload: JsonObject
    mean_scores: Float64Array
    raw_kernels: Float64Array
    bins: tuple[tuple[FakeHead, ...], ...]
    bin_means: Float64Array


class BuildTerminalPayloadFn(Protocol):
    def __call__(
        self,
        *,
        root: RunRoot,
        resolved_config: ResolvedConfig,
        sweep: SweepConfig,
        batch_manifest: BatchManifestPayload,
        expected: Mapping[str, ConditionKey],
        actual_paths: Mapping[str, Path],
        schema_version: int = 1,
    ) -> object: ...


class RankdataAverageFn(Protocol):
    def __call__(self, values: object, *, method: str = "average") -> npt.NDArray[np.float64]: ...


@dataclass(frozen=True)
class FakeCaptureSummary:
    mean_scores: Float64Array
    raw_kernel_sums: Float64Array
    sequence_count: int


@dataclass(frozen=True)
class FakeToySmokeResult:
    run_id: str
    shard_count: int
    statistic_count: int
    terminal_summary_path: Path


@dataclass(frozen=True)
class FakeForwardOutput:
    logits: torch.Tensor


@dataclass(frozen=True)
class FakeValidationRecord:
    family_name: str
    layer_index: int
    head_index: int
    pre_mask: torch.Tensor
    post_mask: torch.Tensor
    correction: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    attention_probabilities: torch.Tensor
    full_q: torch.Tensor
    full_k: torch.Tensor
    full_post_mask: torch.Tensor
    scaling: float
    runtime_call_log: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], float]]

    def runtime_faithful_attention_probabilities(self) -> torch.Tensor:
        self.runtime_call_log.append(
            (
                tuple(self.full_q.shape),
                tuple(self.full_k.shape),
                tuple(self.full_post_mask.shape),
                self.scaling,
            )
        )
        return _PRODUCTION_RUNTIME_FAITHFUL_ATTENTION_PROBABILITIES(
            cast(ValidationProbeRecord, cast(object, self))
        )


@dataclass(frozen=True)
class FakeBundleEntry:
    name: str
    intervention_factory: Callable[..., object]


@dataclass(frozen=True)
class FakeValidationBundle:
    entry: FakeBundleEntry
    binding: FakeBinding
    model: object
    logical_device: torch.device


class _PinnedRunnerGeneralInterface:
    _global_mapping: ClassVar[dict[str, object]] = {}

    def __init__(
        self,
        *,
        initial_local: Mapping[str, object] | None = None,
    ) -> None:
        self._local_mapping = dict(initial_local or {})

    def snapshot(self) -> tuple[dict[str, object], dict[str, object]]:
        return (dict(type(self)._global_mapping), dict(self._local_mapping))

    def __contains__(self, key: object) -> bool:
        return key in self._local_mapping or key in type(self)._global_mapping

    def get(self, key: str, default: object | None = None) -> object | None:
        if key in self._local_mapping:
            return self._local_mapping[key]
        return type(self)._global_mapping.get(key, default)

    def pop(self, key: str, default: object | None = None) -> object | None:
        return self._local_mapping.pop(key, default)

    def register(self, key: str, value: object) -> None:
        type(self)._global_mapping[key] = value

    def replace_local_mapping(self, mapping: dict[str, object]) -> None:
        self._local_mapping = mapping

    def local_mapping(self) -> dict[str, object]:
        return self._local_mapping

    @classmethod
    def seed_global(cls, key: str, value: object) -> None:
        cls._global_mapping[key] = value


class _PopRaisingDict(dict[str, object]):
    def __init__(self, mapping: Mapping[str, object], *, poisoned_key: str) -> None:
        super().__init__(mapping)
        self._poisoned_key = poisoned_key

    def pop(self, key: str, default: object | None = None) -> object | None:
        if key == self._poisoned_key:
            raise RuntimeError("registry cleanup boom")
        return super().pop(key, default)


@dataclass(init=False)
class _StreamRunnerConfig:
    marker: str
    writes: list[str]

    def __init__(self, *, marker: str, writes: Sequence[str] | None = None) -> None:
        object.__setattr__(self, "marker", marker)
        object.__setattr__(self, "writes", [] if writes is None else list(writes))

    def __setattr__(self, name: str, value: object) -> None:
        if name == "marker":
            writes = cast(list[str] | None, self.__dict__.get("writes"))
            if writes is not None:
                if not isinstance(value, str):
                    raise TypeError("marker must be a string")
                writes.append(value)
        object.__setattr__(self, name, value)


@dataclass(frozen=True)
class _StreamRunnerModel:
    config: _StreamRunnerConfig


class DatasetConfigLike(Protocol):
    repository: str


class LoadedDatasetLike(Protocol):
    _fingerprint: object
    column_names: Sequence[object]

    def __iter__(self) -> Iterator[object]: ...


class TemporaryDirectoryLike(Protocol):
    name: str

    def __enter__(self) -> str: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...


@dataclass(frozen=True)
class ResolvedDigests:
    config_sha256: str
    sweep_sha256: str

    def __getitem__(self, key: str) -> str:
        if key == "config_sha256":
            return self.config_sha256
        if key == "sweep_sha256":
            return self.sweep_sha256
        raise KeyError(key)


@dataclass(frozen=True)
class LaunchReceiptBindings:
    launch_receipt: JsonObject
    batch_receipt: JsonObject


@dataclass(frozen=True)
class FinalizeContrast:
    payload: JsonObject


@dataclass(frozen=True)
class FinalizeDepthRelationship:
    payload: JsonObject


class FakeLoadedDataset:
    _rows: tuple[dict[str, str], ...]
    _fingerprint: object
    column_names: Sequence[object]

    def __init__(
        self,
        *,
        rows: Sequence[dict[str, str]],
        column_names: Sequence[object],
        fingerprint: object = "fp",
    ) -> None:
        self._rows = tuple(rows)
        self._fingerprint = fingerprint
        self.column_names = list(column_names)

    def __iter__(self) -> Iterator[dict[str, str]]:
        return iter(self._rows)


def _is_object_mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    return isinstance(value, Mapping)


def _is_object_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)


def _is_json_object(value: JsonValue) -> TypeGuard[JsonObject]:
    return isinstance(value, dict)


def _json_object(value: object) -> JsonObject:
    json_value = _to_json_value(value)
    if not isinstance(json_value, dict):
        raise AssertionError(f"Expected JSON object, found {type(value).__name__}")
    return json_value


def _to_json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, np.generic):
        return _to_json_value(value.item())
    if _is_object_mapping(value):
        normalized: JsonObject = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise AssertionError("JSON object keys must be strings")
            normalized[key] = _to_json_value(item)
        return normalized
    if _is_object_sequence(value):
        return [_to_json_value(item) for item in value]
    raise AssertionError(f"Expected JSON value, found {type(value).__name__}")


def _json_array(value: object) -> list[JsonValue]:
    json_value = _to_json_value(value)
    if not isinstance(json_value, list):
        raise AssertionError(f"Expected JSON array, found {type(value).__name__}")
    return json_value


def _json_mapping_item(payload: Mapping[str, object], key: str) -> object:
    return payload[key]


def _json_object_item(payload: Mapping[str, object], key: str) -> JsonObject:
    return _json_object(_json_mapping_item(payload, key))


def _json_object_member(payload: JsonObject, key: str) -> JsonObject:
    value = payload[key]
    if not _is_json_object(value):
        raise AssertionError(f"{key} must be a JSON object")
    return value


def _json_string_item(payload: Mapping[str, object], key: str) -> str:
    value = _json_mapping_item(payload, key)
    if not isinstance(value, str):
        raise AssertionError(f"{key} must be a string")
    return value


def _json_int_item(payload: Mapping[str, object], key: str) -> int:
    value = _json_mapping_item(payload, key)
    if not isinstance(value, int):
        raise AssertionError(f"{key} must be an int")
    return value


def _json_float_item(payload: Mapping[str, object], key: str) -> float:
    value = _json_mapping_item(payload, key)
    if not isinstance(value, int | float):
        raise AssertionError(f"{key} must be numeric")
    return float(value)


def _json_list_item(payload: Mapping[str, object], key: str) -> list[JsonValue]:
    return _json_array(_json_mapping_item(payload, key))


def _json_list_member(payload: JsonObject, key: str) -> list[JsonValue]:
    value = payload[key]
    if not isinstance(value, list):
        raise AssertionError(f"{key} must be a JSON array")
    return value


def _json_object_list_member(payload: JsonObject, key: str) -> list[JsonObject]:
    values = _json_list_member(payload, key)
    for index, value in enumerate(values):
        if not _is_json_object(value):
            raise AssertionError(f"{key}[{index}] must be a JSON object")
    return cast(list[JsonObject], values)


def _read_json_object(path: Path) -> JsonObject:
    return _json_object(json.loads(path.read_text(encoding="ascii")))


def _require_batch_manifest(value: Mapping[str, object]) -> BatchManifestPayload:
    return cast(BatchManifestPayload, cast(object, value))


def _runner_batch_manifest(value: RunnerBatchManifestPayload) -> BatchManifestPayload:
    return cast(BatchManifestPayload, cast(object, value))


def _frozen_model_binding(value: FakeBinding) -> FrozenModelBinding:
    return cast(FrozenModelBinding, cast(object, value))


def load_existing_or_write(path: str | Path, payload: object) -> tuple[JsonObject, bool]:
    return _artifact_load_existing_or_write(path, _json_object(payload))


def _resolved_digests(payload: Mapping[str, object]) -> ResolvedDigests:
    return ResolvedDigests(
        config_sha256=_json_string_item(payload, "config_sha256"),
        sweep_sha256=_json_string_item(payload, "sweep_sha256"),
    )


def _tokenizer_loader(tokenizer: TokenizerLike) -> TokenizerLoader:
    def inner(source: str) -> TokenizerLike:
        del source
        return tokenizer

    return inner


def _json_object_list_item(payload: Mapping[str, object], key: str) -> list[JsonObject]:
    return [_json_object(item) for item in _json_list_item(payload, key)]


def _json_identity(payload: Mapping[str, object]) -> str:
    return _json_string_item(payload, "identity_sha256")


def _launch_receipt_bindings(
    payload: Mapping[str, object], session_name: str
) -> LaunchReceiptBindings:
    return LaunchReceiptBindings(
        launch_receipt=_json_object_item(
            _json_object_item(payload, "launch_receipts"), session_name
        ),
        batch_receipt=_json_object_item(payload, "launch_batch_receipt"),
    )


def _finalize_contrast(
    payload: Mapping[str, object], *, model: str, direction: str, bin_index: int
) -> FinalizeContrast:
    contrast = next(
        item
        for item in _json_object_list_item(payload, "contrasts")
        if _json_string_item(item, "model") == model
        and _json_string_item(item, "direction") == direction
        and _json_int_item(item, "bin") == bin_index
    )
    return FinalizeContrast(payload=contrast)


def _finalize_depth_relationship(
    payload: Mapping[str, object], *, model: str
) -> FinalizeDepthRelationship:
    depth = next(
        item
        for item in _json_object_list_item(payload, "depth_relationships")
        if _json_string_item(item, "model") == model
    )
    return FinalizeDepthRelationship(payload=depth)


def _batch_domain_entry(
    batch_manifest: BatchManifestPayload, *, model_name: str, domain_name: str
) -> BatchDomainEntryPayload:
    return batch_manifest["domain_entries"][f"{model_name}:{domain_name}"]


def _with_identity(payload: Mapping[str, object]) -> JsonObject:
    enriched = _json_object(payload)
    enriched["identity_sha256"] = artifact_identity(payload_without_identity(enriched))
    return enriched


def _materialized_payload_snapshot(materialized: MaterializedDomain) -> tuple[JsonObject, bytes]:
    payload = _json_object(ensure_payload_identity(materialized))
    return payload, canonical_json_bytes(payload)


def _noop_kwargs(*args: object, **kwargs: object) -> None:
    del args, kwargs


def _raise_assertion_callable(message: str) -> Callable[..., object]:
    def inner(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError(message)

    return inner


def _return_constant(value: T) -> Callable[..., T]:
    def inner(*args: object, **kwargs: object) -> T:
        del args, kwargs
        return value

    return inner


def _dataset_loader(dataset: LoadedDatasetLike) -> DatasetLoaderFn:
    def inner(*args: object, **kwargs: object) -> LoadedDatasetLike:
        del args, kwargs
        return dataset

    return inner


def _raising_dataset_loader(message: str) -> DatasetLoaderFn:
    def inner(*args: object, **kwargs: object) -> LoadedDatasetLike:
        del args, kwargs
        raise AssertionError(message)

    return inner


def _raising_tokenizer_loader(message: str) -> TokenizerLoader:
    def inner(source: str) -> TokenizerLike:
        del source
        raise AssertionError(message)

    return inner


def _raise_model_load_assertion(
    message: str,
) -> Callable[..., FakeLoadedBundle]:
    def inner(
        resolved_config: object,
        *,
        model_name: str,
        local_files_only: bool = True,
        logical_device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.bfloat16,
        auto_model_loader: object | None = None,
        auto_tokenizer_loader: object | None = None,
    ) -> FakeLoadedBundle:
        del (
            resolved_config,
            model_name,
            local_files_only,
            logical_device,
            torch_dtype,
            auto_model_loader,
            auto_tokenizer_loader,
        )
        raise AssertionError(message)

    return inner


def _kernel_values_for_head(
    kernels_by_layer_head: object, key: tuple[int, int]
) -> Sequence[float] | Float64Array:
    if not _is_object_mapping(kernels_by_layer_head):
        raise AssertionError("kernels_by_layer_head must be provided")
    kernel = kernels_by_layer_head.get(key)
    if kernel is None:
        raise AssertionError(f"Missing kernel for {key!r}")
    if isinstance(kernel, np.ndarray):
        return np.asarray(kernel, dtype=np.float64)
    if _is_object_sequence(kernel):
        normalized: list[float] = []
        for item in kernel:
            if isinstance(item, np.generic):
                normalized.append(float(item.item()))
                continue
            if isinstance(item, int | float):
                normalized.append(float(item))
                continue
            raise AssertionError("kernel values must be numeric sequences")
        return normalized
    raise AssertionError("kernel values must be numeric sequences")


def _repeated_sequences(count: int) -> tuple[Int64Array, ...]:
    return tuple(np.arange(512, dtype=np.int64) for _ in range(count))


@dataclass
class RecordedTokenizerLoader:
    calls: list[str]
    tokenizer: TokenizerLike

    def __call__(self, source: str) -> TokenizerLike:
        self.calls.append(source)
        return self.tokenizer


@dataclass(frozen=True)
class FixedDomainSequenceLoader:
    sequences: DomainSequences

    def __call__(
        self,
        run_root: RunRoot,
        resolved_config: object,
        sweep: SweepConfig,
        model_name: str,
        *,
        domain_name: str,
    ) -> DomainSequences:
        del run_root, resolved_config, sweep, model_name, domain_name
        return self.sequences


@dataclass
class CaptureSummaryRecorder:
    calls: list[int]
    sequence_count_floor: int

    def __call__(
        self, bundle: object, sequences: Sequence[Int64Array], *, norm_name: str
    ) -> FakeCaptureSummary:
        del bundle, norm_name
        self.calls.append(len(sequences))
        return FakeCaptureSummary(
            mean_scores=np.zeros((32, 32), dtype=np.float64),
            raw_kernel_sums=np.ones((32, 32, 512), dtype=np.float64),
            sequence_count=max(self.sequence_count_floor, len(sequences)),
        )


@dataclass
class EvalNllRecorder:
    calls: list[int]

    def __call__(
        self,
        bundle: object,
        sequences: Sequence[Int64Array],
        *,
        selected_heads_by_layer: Mapping[int, Sequence[int]] | None = None,
        kernels_by_layer_head: KernelMap | None = None,
    ) -> Float64Array:
        del bundle, selected_heads_by_layer, kernels_by_layer_head
        self.calls.append(len(sequences))
        return np.ones((len(sequences),), dtype=np.float64)


@dataclass
class LaunchPlanAppender:
    launched: list[Path]

    def __call__(self, *, plan: LaunchPlan, run_command: RuntimeCommandRunner) -> None:
        del run_command
        self.launched.append(plan.run_root)


def _iter_fake_dataset_rows() -> Iterator[dict[str, str]]:
    for index in range(400):
        yield {
            "text": f" = Title {index} = " if index % 2 == 0 else "body",
            "repository_name": f"repo-{index}",
            "whole_func_string": "def f():\n    pass",
        }


def _git_binding_stub(*args: object, **kwargs: object) -> GitBinding:
    del args, kwargs
    return GitBinding(
        commit_sha="git",
        tracked_diff_sha256="tracked",
        tracked_paths=("rebuttal/src",),
        relevant_content_sha256="content",
        untracked_paths=(),
        untracked_sha256="untracked",
    )


def _git_binding_with_commit(commit_sha: str) -> GitBinding:
    return GitBinding(
        commit_sha=commit_sha,
        tracked_diff_sha256="tracked",
        tracked_paths=("rebuttal/src",),
        relevant_content_sha256="content",
        untracked_paths=(),
        untracked_sha256="untracked",
    )


def _git_binding_with_forged_commit(
    repo_root: str | Path,
    tracked_paths: Sequence[str | Path],
) -> GitBinding:
    del repo_root, tracked_paths
    return _git_binding_with_commit("forged")


def _materialized_from_kwargs(kwargs: dict[str, object]) -> MaterializedDomain:
    materialized = kwargs.get("materialized")
    if not isinstance(materialized, MaterializedDomain):
        raise AssertionError("materialized must be provided")
    return materialized


def _dataset_config_from_kwargs(kwargs: dict[str, object]) -> DatasetConfigLike:
    dataset = kwargs.get("dataset")
    if dataset is None or not hasattr(dataset, "repository"):
        raise AssertionError("dataset with repository must be provided")
    return cast(DatasetConfigLike, dataset)


def _string_kwarg(kwargs: dict[str, object], key: str) -> str:
    value = kwargs.get(key)
    if not isinstance(value, str):
        raise AssertionError(f"{key} must be provided as a string")
    return value


def _reconstruct_sequences_from_materialized_stub(
    *args: object, **kwargs: object
) -> tuple[tuple[Int64Array, ...], tuple[Int64Array, ...], MaterializedDomain]:
    del args
    return _fake_reconstructed_sequences(_materialized_from_kwargs(kwargs))


def _empty_reconstruct_sequences_stub(
    materialized: MaterializedDomain,
) -> Callable[..., tuple[tuple[Int64Array, ...], tuple[Int64Array, ...], MaterializedDomain]]:
    return _return_constant(((), (), materialized))


def _fake_materialize_frozen_domain_stub(*args: object, **kwargs: object) -> MaterializedDomain:
    del args
    dataset = _dataset_config_from_kwargs(kwargs)
    domain_name = (
        "wikipedia" if dataset.repository == _dataset_fixture("wikipedia")["repository"] else "code"
    )
    return _fake_materialized_domain(
        domain_name=domain_name,
        model_name=_string_kwarg(kwargs, "tokenizer_name"),
        tokenizer_tree_sha256=_string_kwarg(kwargs, "tokenizer_tree_sha256"),
    )


def _set_json_path_value(payload: JsonObject, path: Sequence[str], value: JsonValue) -> None:
    if not path:
        raise AssertionError("path must not be empty")

    def assign_path(target: JsonObject, path_index: int) -> JsonObject:
        key = path[path_index]
        updated = dict(target)
        if path_index == len(path) - 1:
            updated[key] = value
            return updated
        child = target[key]
        if not _is_json_object(child):
            raise AssertionError(f"{key} must be a JSON object")
        updated[key] = assign_path(child, path_index + 1)
        return updated

    rebuilt = assign_path(payload, 0)
    payload.clear()
    payload.update(rebuilt)


def _idle_gpu_snapshot(physical_index: int) -> GpuSnapshot:
    return GpuSnapshot(
        physical_index=physical_index,
        uuid=f"GPU-{physical_index}",
        name="L40",
        memory_used_mib=0.0,
        utilization_gpu_pct=0.0,
        compute_processes=(),
    )


def _idle_gpu_snapshot_pair(physical_index: int) -> tuple[GpuSnapshot, GpuSnapshot]:
    snapshot = _idle_gpu_snapshot(physical_index)
    return snapshot, snapshot


def _require_two_idle_snapshots_stub(**kwargs: object) -> tuple[GpuSnapshot, GpuSnapshot]:
    physical_index = kwargs.get("physical_index")
    if not isinstance(physical_index, int):
        raise AssertionError("physical_index must be provided as an int")
    return _idle_gpu_snapshot_pair(physical_index)


def _env(tmp_path: Path) -> dict[str, str]:
    model_root = tmp_path / "models"
    model_root.mkdir()
    for name in ("llama", "mistral", "olmo", "tok-llama", "tok-mistral", "tok-olmo"):
        target = model_root / name
        target.mkdir()
        (target / "file.txt").write_text(name, encoding="utf-8")
    return {
        "SI_REBUTTAL_RUNS_ROOT": "runs",
        "SI_REBUTTAL_DATA_ROOT": "data/materialized",
        "SI_REBUTTAL_MODELS_ROOT": str(model_root),
        "SI_REBUTTAL_LOGS_ROOT": "logs",
        "SI_REBUTTAL_CACHE_ROOT": "cache",
        "SI_REBUTTAL_MODEL_LLAMA_3_1_8B": str(model_root / "llama"),
        "SI_REBUTTAL_TOKENIZER_LLAMA_3_1_8B": str(model_root / "tok-llama"),
        "SI_REBUTTAL_MODEL_MISTRAL_7B_V0_1": str(model_root / "mistral"),
        "SI_REBUTTAL_TOKENIZER_MISTRAL_7B_V0_1": str(model_root / "tok-mistral"),
        "SI_REBUTTAL_MODEL_OLMO_2_7B": str(model_root / "olmo"),
        "SI_REBUTTAL_TOKENIZER_OLMO_2_7B": str(model_root / "tok-olmo"),
    }


def _binding_id(*parts: object) -> str:
    return artifact_identity(list(parts))


def _fake_validation_full_q(*, layer_offset: float = 0.0) -> torch.Tensor:
    return (
        torch.arange(1 * 32 * 64 * 2, dtype=torch.float32).reshape(1, 32, 64, 2) / 64.0
        + layer_offset
    ).to(torch.bfloat16)


def _fake_validation_full_k(*, layer_offset: float = 0.0) -> torch.Tensor:
    return (
        torch.arange(1 * 8 * 64 * 2, dtype=torch.float32).reshape(1, 8, 64, 2).flip(-1) / 48.0
        + layer_offset
    ).to(torch.bfloat16)


def _fake_validation_full_mask(
    *, kernel_kind: str, selected_head: int, num_heads: int = 32
) -> torch.Tensor:
    mask = torch.full((64, 64), float("-inf"), dtype=torch.float32)
    for query_index in range(64):
        mask[query_index, : query_index + 1] = 0.0
    full_mask = mask.expand(1, num_heads, 64, 64).clone()
    if kernel_kind == "nonconstant":
        for query_index in range(64):
            for key_index in range(query_index + 1):
                full_mask[0, selected_head, query_index, key_index] = (
                    -(query_index - key_index) / 1022.0
                )
    return full_mask


def _make_fake_validation_record(
    *,
    kernel_kind: str,
    selected_head: int,
) -> FakeValidationRecord:
    full_q = _fake_validation_full_q()
    full_k = _fake_validation_full_k()
    full_post_mask = _fake_validation_full_mask(
        kernel_kind=kernel_kind,
        selected_head=selected_head,
    )
    repeated_k = repeat_kv_after_gqa(full_k, num_query_heads=int(full_q.shape[1]))
    pre_mask = _fake_validation_full_mask(kernel_kind="zero", selected_head=selected_head)[
        0, selected_head
    ].clone()
    scaling = 0.625
    runtime_call_log: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], float]] = []
    proto_record = FakeValidationRecord(
        family_name="mistral-7b-v0.1",
        layer_index=0,
        head_index=selected_head,
        pre_mask=pre_mask,
        post_mask=full_post_mask[0, selected_head].clone(),
        correction=(full_post_mask[0, selected_head] - pre_mask).clone(),
        q=full_q[0, selected_head].to(torch.float32).clone(),
        k=repeated_k[0, selected_head].to(torch.float32).clone(),
        attention_probabilities=torch.zeros((64, 64), dtype=torch.float32),
        full_q=full_q.clone(),
        full_k=full_k.clone(),
        full_post_mask=full_post_mask.clone(),
        scaling=scaling,
        runtime_call_log=runtime_call_log,
    )
    runtime_probs = _PRODUCTION_RUNTIME_FAITHFUL_ATTENTION_PROBABILITIES(
        cast(ValidationProbeRecord, cast(object, proto_record))
    ).to(torch.float32)
    selected_probs = runtime_probs[0, selected_head].clone()
    if kernel_kind == "nonconstant":
        selected_probs[63, 0] = selected_probs[63, 0] + 4e-3
        selected_probs[63, 1] = selected_probs[63, 1] - 4e-3
    runtime_call_log.clear()
    return FakeValidationRecord(
        family_name=proto_record.family_name,
        layer_index=proto_record.layer_index,
        head_index=proto_record.head_index,
        pre_mask=proto_record.pre_mask,
        post_mask=proto_record.post_mask,
        correction=proto_record.correction,
        q=proto_record.q,
        k=proto_record.k,
        attention_probabilities=selected_probs,
        full_q=proto_record.full_q,
        full_k=proto_record.full_k,
        full_post_mask=proto_record.full_post_mask,
        scaling=proto_record.scaling,
        runtime_call_log=runtime_call_log,
    )


def _make_runner_pinned_general_interface(
    *,
    initial_global: Mapping[str, object] | None = None,
    initial_local: Mapping[str, object] | None = None,
) -> _PinnedRunnerGeneralInterface:
    registry_cls = cast(
        type[_PinnedRunnerGeneralInterface],
        type(
            "_PinnedRunnerGeneralInterfaceIsolated",
            (_PinnedRunnerGeneralInterface,),
            {"_global_mapping": dict(initial_global or {})},
        ),
    )
    return registry_cls(initial_local=initial_local)


def _stream_runner_bundle(*, marker: str = "original-marker") -> FakeValidationBundle:
    return FakeValidationBundle(
        entry=FakeBundleEntry(name="mistral-7b-v0.1", intervention_factory=_noop_kwargs),
        binding=FakeBinding("w", "t"),
        model=_StreamRunnerModel(config=_StreamRunnerConfig(marker=marker)),
        logical_device=torch.device("cpu"),
    )


def _capture_key_for_bundle(bundle: FakeValidationBundle, *, time_ns: int) -> str:
    return f"si_rebuttal_fit_capture_{id(bundle)}_{time_ns}"


def _zero_manual_attention_logits(
    query: torch.Tensor,
    key: torch.Tensor,
    additive_mask: torch.Tensor | None,
    num_query_heads: int,
) -> torch.Tensor:
    del additive_mask
    return torch.zeros((1, num_query_heads, query.shape[2], key.shape[2]), dtype=torch.float32)


def _install_stream_post_rope_qk_harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    bundle: FakeValidationBundle,
    attention_registry: _PinnedRunnerGeneralInterface,
    mask_registry: _PinnedRunnerGeneralInterface,
    time_ns: int,
    model_forward_error: str | None = None,
    poison_registry_before_return: bool = False,
    seen_mask_handlers: list[object] | None = None,
    seen_masks: list[torch.Tensor] | None = None,
) -> tuple[torch.nn.Module, ...]:
    modules = (torch.nn.Identity(), torch.nn.Identity())
    registry_poisoned = False

    class _FamilyAdapter:
        def __init__(
            self,
            attention_interface: _PinnedRunnerGeneralInterface,
            mask_attention_interface: _PinnedRunnerGeneralInterface,
        ) -> None:
            self.attention_interface = attention_interface
            self.mask_attention_interface = mask_attention_interface

        def eager_attention_forward(
            self,
            module: torch.nn.Module,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_mask: torch.Tensor | None,
            scaling: float,
            dropout: float = 0.0,
            **kwargs: object,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            del module, query, key, attention_mask, scaling, dropout, kwargs
            return value, None

    @dataclass(frozen=True)
    class _CaptureSpec:
        attention_modules: tuple[torch.nn.Module, ...]
        family_adapter: _FamilyAdapter
        layer_count: int

    spec = _CaptureSpec(
        attention_modules=modules,
        family_adapter=_FamilyAdapter(attention_registry, mask_registry),
        layer_count=len(modules),
    )
    monkeypatch.setattr(
        "si_rebuttal.runner.derive_supported_model_family_spec_for_capture",
        _return_constant(spec),
    )
    monkeypatch.setattr(
        "si_rebuttal.runner.attn_implementation_attr_name",
        _return_constant("marker"),
    )
    monkeypatch.setattr("si_rebuttal.runner.time.time_ns", _return_constant(time_ns))
    monkeypatch.setattr(
        "si_rebuttal.runner.manual_attention_logits",
        _zero_manual_attention_logits,
    )

    def fake_call_model(model: object, **kwargs: object) -> FakeForwardOutput:
        nonlocal registry_poisoned
        del kwargs
        marker = cast(_StreamRunnerModel, model).config.marker
        capture = attention_registry.get(marker)
        if not callable(capture):
            raise RuntimeError(f"registry entry for {marker!r} is not callable")
        mask_builder = mask_registry.get(marker)
        if not callable(mask_builder):
            raise RuntimeError(f"mask registry entry for {marker!r} is not callable")
        if seen_mask_handlers is not None:
            seen_mask_handlers.append(mask_builder)
        for layer_index, module in enumerate(modules):
            query = _fake_validation_full_q(layer_offset=float(layer_index))[:, :2]
            key = _fake_validation_full_k(layer_offset=float(layer_index))[:, :1]
            value = torch.zeros_like(query)
            mask = cast(
                torch.Tensor,
                mask_builder(layer_index=layer_index, query=query, key=key, marker=marker),
            )
            if seen_masks is not None:
                seen_masks.append(mask.clone())
            capture(module, query, key, value, mask, 0.5)
            if poison_registry_before_return and not registry_poisoned:
                attention_registry.replace_local_mapping(
                    _PopRaisingDict(
                        attention_registry.local_mapping(),
                        poisoned_key=marker,
                    )
                )
                mask_registry.replace_local_mapping(
                    _PopRaisingDict(
                        mask_registry.local_mapping(),
                        poisoned_key=marker,
                    )
                )
                registry_poisoned = True
            if model_forward_error is not None and layer_index == 0:
                raise RuntimeError(model_forward_error)
        return FakeForwardOutput(logits=torch.zeros((1, 64, 1), dtype=torch.float32))

    monkeypatch.setattr("si_rebuttal.runner._call_model", fake_call_model)
    return modules


_FREEZE_OUTPUT = (
    "torch==2.7.0\n"
    "transformers==5.3.0\n"
    "datasets==4.8.2\n"
    "numpy==1.26.4\n"
    "scipy==1.11.4\n"
    "pandas==2.1.4\n"
    "pyarrow==23.0.1\n"
    "pytest==7.4.4\n"
    "ruff==0.12.5\n"
    "basedpyright==1.31.1\n"
)
_GPU_INVENTORY_OUTPUT = (
    "0, GPU-0000, NVIDIA L40, 46068, 8.9, 570.00\n1, GPU-1111, NVIDIA L40, 46068, 8.9, 570.00\n"
)


def _dataset_fixture(domain_name: str) -> dict[str, str]:
    resolved_config, _ = load_configs(
        "rebuttal/configs/base.toml",
        "rebuttal/configs/sweep.toml",
    )
    dataset = resolved_config.datasets[domain_name]
    return {
        "repository": dataset.repository,
        "revision": dataset.revision,
        "config": dataset.config,
        "split": dataset.split,
        "field": dataset.field,
    }


def _fake_chunk_tokens(
    *, model_name: str, domain_name: str, partition: str, item_index: int
) -> list[int]:
    seed = int.from_bytes(
        hashlib.sha256(f"{model_name}|{domain_name}|{partition}|{item_index}".encode()).digest()[
            :8
        ],
        "big",
    )
    start = seed % 10_000
    return [int((start + offset) % 32_768) for offset in range(512)]


def _fake_document_record(
    *, domain_name: str, document_id: str, partition: str, row_index: int
) -> DocumentRecord:
    text = f"{domain_name} {partition} document {row_index}"
    return DocumentRecord(
        domain=domain_name,
        document_id=document_id,
        row_indices=(row_index,),
        text=text,
        content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        assignment_sha256=hashlib.sha256(f"{partition}:{document_id}".encode()).hexdigest(),
        partition=partition,
    )


def _chunk_for_document(*, document: DocumentRecord, token_ids: Sequence[int]) -> SelectedChunk:
    return SelectedChunk(
        document_id=document.document_id,
        partition=document.partition,
        chunk_index=0,
        source_rows=document.row_indices,
        token_count=len(token_ids),
        token_shape=(len(token_ids),),
        token_sha256=token_sha256(token_ids),
    )


def _assert_selected_chunk_provenance(materialized: MaterializedDomain) -> None:
    for documents, chunks in (
        (materialized.fit_documents, materialized.fit_chunks),
        (materialized.eval_documents, materialized.eval_chunks),
    ):
        rows_by_document_id = {document.document_id: document.row_indices for document in documents}
        for chunk in chunks:
            assert rows_by_document_id[chunk.document_id] == chunk.source_rows


def _fake_materialized_domain(
    *,
    domain_name: str,
    model_name: str,
    tokenizer_tree_sha256: str,
    dataset_fingerprint: str = "fp",
) -> MaterializedDomain:
    dataset = _dataset_fixture(domain_name)
    fit_documents = tuple(
        _fake_document_record(
            domain_name=domain_name,
            document_id=f"{domain_name}-fit-doc-{item_index}",
            partition="fit",
            row_index=item_index,
        )
        for item_index in range(50)
    )
    eval_documents = tuple(
        _fake_document_record(
            domain_name=domain_name,
            document_id=f"{domain_name}-eval-doc-{item_index}",
            partition="eval",
            row_index=100 + item_index,
        )
        for item_index in range(100)
    )
    fit_chunks = tuple(
        _chunk_for_document(
            document=document,
            token_ids=_fake_chunk_tokens(
                model_name=model_name,
                domain_name=domain_name,
                partition="fit",
                item_index=item_index,
            ),
        )
        for item_index, document in enumerate(fit_documents)
    )
    eval_chunks = tuple(
        _chunk_for_document(
            document=document,
            token_ids=_fake_chunk_tokens(
                model_name=model_name,
                domain_name=domain_name,
                partition="eval",
                item_index=item_index,
            ),
        )
        for item_index, document in enumerate(eval_documents)
    )

    materialized = MaterializedDomain(
        schema_version=1,
        domain=domain_name,
        dataset_repository=dataset["repository"],
        dataset_revision=dataset["revision"],
        dataset_config=dataset["config"],
        dataset_split=dataset["split"],
        dataset_field=dataset["field"],
        dataset_fingerprint=dataset_fingerprint,
        tokenizer_name=model_name,
        tokenizer_tree_sha256=tokenizer_tree_sha256,
        fit_documents=fit_documents,
        eval_documents=eval_documents,
        fit_chunks=fit_chunks,
        eval_chunks=eval_chunks,
        manifest_sha256=_binding_id(
            "materialized",
            model_name,
            domain_name,
            dataset["repository"],
            tokenizer_tree_sha256,
        ),
    )
    _assert_selected_chunk_provenance(materialized)
    return materialized


def _document_record(
    *, domain_name: str, document_id: str, partition: str, row_indices: Sequence[int], text: str
) -> DocumentRecord:
    return DocumentRecord(
        domain=domain_name,
        document_id=document_id,
        row_indices=tuple(row_indices),
        text=text,
        content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        assignment_sha256=hashlib.sha256(f"{partition}:{document_id}".encode()).hexdigest(),
        partition=partition,
    )


def _selected_chunk(
    *,
    model_name: str,
    domain_name: str,
    document: DocumentRecord,
    item_index: int,
) -> SelectedChunk:
    return _chunk_for_document(
        document=document,
        token_ids=_fake_chunk_tokens(
            model_name=model_name,
            domain_name=domain_name,
            partition=document.partition,
            item_index=item_index,
        ),
    )


def _custom_materialized_domain(
    *,
    domain_name: str,
    model_name: str,
    tokenizer_tree_sha256: str,
    fit_documents: Sequence[DocumentRecord],
    eval_documents: Sequence[DocumentRecord],
    fit_chunks: Sequence[SelectedChunk],
    eval_chunks: Sequence[SelectedChunk],
) -> MaterializedDomain:
    base = _fake_materialized_domain(
        domain_name=domain_name,
        model_name=model_name,
        tokenizer_tree_sha256=tokenizer_tree_sha256,
    )
    materialized = replace(
        base,
        fit_documents=tuple(fit_documents),
        eval_documents=tuple(eval_documents),
        fit_chunks=tuple(fit_chunks),
        eval_chunks=tuple(eval_chunks),
    )
    _assert_selected_chunk_provenance(materialized)
    return materialized


def _fake_reconstructed_sequences(
    materialized: MaterializedDomain,
) -> tuple[tuple[Int64Array, ...], tuple[Int64Array, ...], MaterializedDomain]:
    def collect(partition: str, count: int) -> tuple[Int64Array, ...]:
        return tuple(
            np.asarray(
                _fake_chunk_tokens(
                    model_name=materialized.tokenizer_name,
                    domain_name=materialized.domain,
                    partition=partition,
                    item_index=item_index,
                ),
                dtype=np.int64,
            )
            for item_index in range(count)
        )

    return collect("fit", 50), collect("eval", 100), materialized


def _reconstruct_sequences_by_chunk_count(
    *args: object, **kwargs: object
) -> tuple[tuple[Int64Array, ...], tuple[Int64Array, ...], MaterializedDomain]:
    del args
    materialized = _materialized_from_kwargs(kwargs)
    return (
        _repeated_arange_sequences(len(materialized.fit_chunks)),
        _repeated_arange_sequences(len(materialized.eval_chunks)),
        materialized,
    )


def _set_launch_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_root: Path,
    session_name: str,
    batch_id: str,
    physical_gpu: int,
) -> None:
    monkeypatch.setenv(LAUNCH_RECEIPT_ENV, str(run_root / "receipts" / f"{session_name}.json"))
    monkeypatch.setenv(BATCH_RECEIPT_ENV, str(run_root / "receipts" / "batch.json"))
    monkeypatch.setenv(TMUX_SESSION_ENV, session_name)
    monkeypatch.setenv(RUN_ROOT_ENV, str(run_root))
    monkeypatch.setenv(BATCH_ID_ENV, batch_id)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", str(physical_gpu))


def _write_launch_context(
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_root: Path,
    batch_id: str = "batch",
    physical_gpu: int = 0,
) -> None:
    monkeypatch.setattr("si_rebuttal.runner.capture_git_binding", _git_binding_stub)
    monkeypatch.setattr("si_rebuttal.runner.require_tmux_absent", _noop_kwargs)
    monkeypatch.setattr(
        "si_rebuttal.runner.require_two_idle_snapshots",
        _require_two_idle_snapshots_stub,
    )
    monkeypatch.setattr("si_rebuttal.runner.launch_tmux_plan", _noop_kwargs)
    _set_visible_gpu_count(monkeypatch, count=2)
    launch(
        "rebuttal/configs/base.toml",
        "rebuttal/configs/sweep.toml",
        run_root=run_root,
        batch_id=batch_id,
        execute=True,
        run_command=_runtime_command_runner(),
    )
    _set_visible_gpu_count(monkeypatch, count=1)
    _set_launch_env(
        monkeypatch,
        run_root=run_root,
        session_name="si-rebuttal-gpu0" if physical_gpu == 0 else "si-rebuttal-gpu1",
        batch_id=batch_id,
        physical_gpu=physical_gpu,
    )


def _repeated_arange_sequences(count: int) -> tuple[Int64Array, ...]:
    return tuple(np.arange(512, dtype=np.int64) for _ in range(count))


def _runtime_command_runner(
    *,
    freeze_lines: list[str] | None = None,
    gpu_lines: list[str] | None = None,
    failures: dict[tuple[str, ...], subprocess.CompletedProcess[str]] | None = None,
) -> RuntimeCommandRunner:
    freeze_lines = (
        [
            "torch==2.7.0",
            "transformers==5.3.0",
            "datasets==4.8.2",
            "numpy==1.26.4",
            "scipy==1.11.4",
            "pandas==2.1.4",
            "pyarrow==23.0.1",
            "pytest==7.4.4",
            "ruff==0.12.5",
            "basedpyright==1.31.1",
        ]
        if freeze_lines is None
        else list(freeze_lines)
    )
    gpu_lines = (
        [
            "0, GPU-0000, NVIDIA L40, 46068, 8.9, 570.00",
            "1, GPU-1111, NVIDIA L40, 46068, 8.9, 570.00",
        ]
        if gpu_lines is None
        else list(gpu_lines)
    )
    failures = {} if failures is None else dict(failures)
    freeze_command = (sys.executable, "-m", "pip", "freeze", "--all")
    gpu_command = (
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    )

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        resolved_command = tuple(command)
        if resolved_command in failures:
            return failures[resolved_command]
        if resolved_command == freeze_command:
            return subprocess.CompletedProcess(
                list(resolved_command), 0, stdout="\n".join(freeze_lines) + "\n", stderr=""
            )
        if resolved_command == gpu_command:
            return subprocess.CompletedProcess(
                list(resolved_command), 0, stdout="\n".join(gpu_lines) + "\n", stderr=""
            )
        raise AssertionError(f"Unexpected runtime command: {resolved_command!r}")

    return runner


def _editable_freeze_lines() -> list[str]:
    return [
        "# Editable install with no version control (si-rebuttal==0.1.0)",
        "-e /tmp/si_rebuttal",
        "",
        "torch==2.7.0",
        "transformers==5.3.0",
        "datasets==4.8.2",
        "numpy==1.26.4",
        "scipy==1.11.4",
        "pandas==2.1.4",
        "pyarrow==23.0.1",
        "pytest==7.4.4",
        "ruff==0.12.5",
        "basedpyright==1.31.1",
        "custom-local @ file:///tmp/custom-local",
    ]


def _overwrite_json(path: Path, payload: JsonObject) -> None:
    payload = _with_identity(payload)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n",
        encoding="ascii",
    )


def _artifact_bytes(payload: JsonObject) -> bytes:
    payload = _with_identity(payload)
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("ascii")


def _summary_v1_path(root: RunRoot) -> Path:
    return root.summaries_dir / "batch-terminal-summary.json"


def _summary_v2_path(root: RunRoot) -> Path:
    return root.summaries_dir / "batch-terminal-summary.v2.json"


def _assert_canonical_config_and_sweep_raw_bytes() -> tuple[str, str]:
    base_digest = hashlib.sha256(Path("rebuttal/configs/base.toml").read_bytes()).hexdigest()
    sweep_digest = hashlib.sha256(Path("rebuttal/configs/sweep.toml").read_bytes()).hexdigest()
    assert base_digest == _CANONICAL_BASE_RAW_BYTE_SHA256
    assert sweep_digest == _CANONICAL_SWEEP_RAW_BYTE_SHA256
    return base_digest, sweep_digest


def _snapshot_file_tree(root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            snapshot[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def _build_terminal_payload_for_test(
    *,
    root: RunRoot,
    batch_manifest: BatchManifestPayload,
    schema_version: int,
) -> JsonObject:
    resolved_config, sweep = load_configs(
        "rebuttal/configs/base.toml",
        "rebuttal/configs/sweep.toml",
    )
    expected = {
        key.stem(): key
        for model_name in sweep.models
        for key in runner_module._expected_condition_inventory(  # pyright: ignore[reportPrivateUsage]
            model_name,
            include_depth=sweep.include_depth,
        )
    }
    actual_paths = {path.stem: path for path in root.shards_dir.glob("*.json")}
    build_payload = cast(
        BuildTerminalPayloadFn,
        runner_module._build_terminal_payload,  # pyright: ignore[reportPrivateUsage]
    )
    if "schema_version" in inspect.signature(build_payload).parameters:
        payload = _json_object(
            build_payload(
                root=root,
                resolved_config=resolved_config,
                sweep=sweep,
                batch_manifest=batch_manifest,
                expected=expected,
                actual_paths=actual_paths,
                schema_version=schema_version,
            )
        )
    else:
        payload = _json_object(
            build_payload(
                root=root,
                resolved_config=resolved_config,
                sweep=sweep,
                batch_manifest=batch_manifest,
                expected=expected,
                actual_paths=actual_paths,
            )
        )
    payload["schema_version"] = schema_version
    return payload


def _different_hex64(value: str) -> str:
    if len(value) != 64:
        raise AssertionError(f"expected 64 hex chars, got {value!r}")
    replacement = "0" if value[0] != "0" else "1"
    return replacement + value[1:]


_IMMUTABLE_INPUT_MUTATION_CASES: tuple[tuple[str, RunRootMutation, str], ...] = (
    (
        "resolved-config",
        lambda root: _overwrite_json(
            root.manifests_dir / "resolved-config.json",
            {
                **_read_json_object(root.manifests_dir / "resolved-config.json"),
                "config": {
                    **_json_object_item(
                        _read_json_object(root.manifests_dir / "resolved-config.json"), "config"
                    ),
                    "identity_sha256": _different_hex64(
                        _json_string_item(
                            _json_object_item(
                                _read_json_object(root.manifests_dir / "resolved-config.json"),
                                "config",
                            ),
                            "identity_sha256",
                        )
                    ),
                },
            },
        ),
        r"(?i)immutable input manifest content digest drifted\.",
    ),
    (
        "resolved-config-audit-note",
        lambda root: _overwrite_json(
            root.manifests_dir / "resolved-config.json",
            {
                **_read_json_object(root.manifests_dir / "resolved-config.json"),
                "audit_note": "identity-valid unknown extra field",
            },
        ),
        r"(?i)immutable input manifest content digest drifted\.",
    ),
    (
        "sweep",
        lambda root: _overwrite_json(
            root.manifests_dir / "sweep.json",
            {
                **_read_json_object(root.manifests_dir / "sweep.json"),
                "sweep": {
                    **_json_object_item(
                        _read_json_object(root.manifests_dir / "sweep.json"),
                        "sweep",
                    ),
                    "identity_sha256": _different_hex64(
                        _json_string_item(
                            _json_object_item(
                                _read_json_object(root.manifests_dir / "sweep.json"),
                                "sweep",
                            ),
                            "identity_sha256",
                        )
                    ),
                },
            },
        ),
        r"(?i)immutable input manifest content digest drifted\.",
    ),
    (
        "materialized-data",
        lambda root: _overwrite_json(
            root.manifests_dir / "materialized-data.json",
            {
                **_read_json_object(root.manifests_dir / "materialized-data.json"),
                "domains": {
                    **_json_object_item(
                        _read_json_object(root.manifests_dir / "materialized-data.json"), "domains"
                    ),
                    "llama-3.1-8b:wikipedia": {
                        **_json_object_item(
                            _json_object_item(
                                _read_json_object(root.manifests_dir / "materialized-data.json"),
                                "domains",
                            ),
                            "llama-3.1-8b:wikipedia",
                        ),
                        "dataset_fingerprint": (
                            _json_string_item(
                                _json_object_item(
                                    _json_object_item(
                                        _read_json_object(
                                            root.manifests_dir / "materialized-data.json"
                                        ),
                                        "domains",
                                    ),
                                    "llama-3.1-8b:wikipedia",
                                ),
                                "dataset_fingerprint",
                            )
                            + "-drifted"
                        ),
                    },
                },
            },
        ),
        r"(?i)immutable input manifest content digest drifted\.",
    ),
    (
        "token-manifest",
        lambda root: _overwrite_json(
            root.manifests_dir / "tokens" / "llama-3.1-8b.wikipedia.fit.json",
            {
                **_read_json_object(
                    root.manifests_dir / "tokens" / "llama-3.1-8b.wikipedia.fit.json"
                ),
                "sequence_digest_sha256": _different_hex64(
                    _json_string_item(
                        _read_json_object(
                            root.manifests_dir / "tokens" / "llama-3.1-8b.wikipedia.fit.json"
                        ),
                        "sequence_digest_sha256",
                    )
                ),
            },
        ),
        r"(?i)immutable input manifest content digest drifted\.",
    ),
    (
        "fit-profile",
        lambda root: _overwrite_json(
            root.manifests_dir / "fit-profiles" / "llama-3.1-8b.wikipedia.json",
            {
                **_read_json_object(
                    root.manifests_dir / "fit-profiles" / "llama-3.1-8b.wikipedia.json"
                ),
                "domain": "code"
                if _json_string_item(
                    _read_json_object(
                        root.manifests_dir / "fit-profiles" / "llama-3.1-8b.wikipedia.json"
                    ),
                    "domain",
                )
                != "code"
                else "wikipedia",
            },
        ),
        r"^Fit profile identity drifted: llama-3\.1-8b\.wikipedia\.json$",
    ),
    (
        "token-manifest-whitespace",
        lambda root: _rewrite_json_with_whitespace_only_byte_drift(
            root.manifests_dir / "tokens" / "llama-3.1-8b.wikipedia.fit.json"
        ),
        r"(?i)immutable input manifest content digest drifted\.",
    ),
)

_MANIFEST_INPUT_MUTATION_CASES: tuple[tuple[RunRootMutation, str], ...] = (
    (
        lambda root: (root.manifests_dir / "fit-profiles" / "llama-3.1-8b.wikipedia.json").unlink(),
        "missing|manifest",
    ),
    (
        lambda root: (root.manifests_dir / "extra-manifest.json").write_text(
            '{"schema_version":1,"identity_sha256":"forged"}\n',
            encoding="ascii",
        ),
        "unexpected|additional|manifest|identity mismatch",
    ),
    (
        lambda root: (root.manifests_dir / "extra-note.txt").write_text(
            "extra\n",
            encoding="ascii",
        ),
        "unexpected|additional|manifest|non-JSON",
    ),
)


def _write_expected_v1_summary(
    root: RunRoot,
    *,
    batch_manifest: BatchManifestPayload,
) -> JsonObject:
    payload = _build_terminal_payload_for_test(
        root=root,
        batch_manifest=batch_manifest,
        schema_version=1,
    )
    payload["completed_at"] = "2026-07-30T12:00:00+00:00"
    payload["identity_sha256"] = runner_module._terminal_payload_identity(  # pyright: ignore[reportPrivateUsage]
        payload
    )
    _overwrite_json(_summary_v1_path(root), payload)
    return payload


def _patch_refinalize_source_summary_v1_digest(
    monkeypatch: pytest.MonkeyPatch,
    root: RunRoot,
    *,
    digest: str | None = None,
) -> None:
    summary_digest = digest or hashlib.sha256(_summary_v1_path(root).read_bytes()).hexdigest()
    monkeypatch.setattr(
        runner_module,
        "_FIXED_PRODUCTION_SOURCE_V1_RAW_BYTE_SHA256",
        summary_digest,
    )


def _patch_refinalize_fixed_input_digests(
    monkeypatch: pytest.MonkeyPatch,
    root: RunRoot,
    *,
    batch_manifest: BatchManifestPayload,
) -> None:
    _patch_refinalize_source_summary_v1_digest(monkeypatch, root)
    canonical_base_path = Path("rebuttal/configs/base.toml").resolve()
    canonical_sweep_path = Path("rebuttal/configs/sweep.toml").resolve()
    resolved_config, sweep = load_configs(canonical_base_path, canonical_sweep_path)
    manifest_payload = _json_object(
        runner_module._immutable_input_manifest(  # pyright: ignore[reportPrivateUsage]
            run_root=root,
            resolved_config=resolved_config,
            sweep=sweep,
            project_root=Path.cwd().resolve(),
            config_path=canonical_base_path,
            sweep_path=canonical_sweep_path,
            source_summary_v1_path=_summary_v1_path(root),
        )
    )
    for attr_name, attr_value in (
        (
            "_FIXED_PRODUCTION_IMMUTABLE_INPUT_CONTENT_SHA256",
            manifest_payload["content_sha256"],
        ),
        (
            "_FIXED_PRODUCTION_IMMUTABLE_INPUT_MANIFEST_CONTENT_SHA256",
            manifest_payload["content_sha256"],
        ),
        (
            "_FIXED_PRODUCTION_IMMUTABLE_INPUT_MANIFEST_SHA256",
            manifest_payload["content_sha256"],
        ),
        (
            "_FIXED_PRODUCTION_IMMUTABLE_INPUT_IDENTITY_SHA256",
            manifest_payload["identity_sha256"],
        ),
        (
            "_FIXED_PRODUCTION_IMMUTABLE_INPUT_MANIFEST_IDENTITY_SHA256",
            manifest_payload["identity_sha256"],
        ),
    ):
        monkeypatch.setattr(runner_module, attr_name, attr_value, raising=False)


def _write_stub_fit_profile_manifests(
    *,
    root: RunRoot,
    batch_manifest: BatchManifestPayload,
) -> dict[tuple[str, str], FakeFitProfile]:
    resolved_config, sweep = load_configs(
        "rebuttal/configs/base.toml",
        "rebuttal/configs/sweep.toml",
    )
    profiles: dict[tuple[str, str], FakeFitProfile] = {}
    model_order = {model_name: index for index, model_name in enumerate(sweep.models)}
    domain_order = {
        domain_name: index for index, domain_name in enumerate(resolved_config.datasets)
    }
    for model_name in sweep.models:
        binding = FakeBinding(
            weights_tree_sha256=batch_manifest["models"][model_name],
            tokenizer_tree_sha256=batch_manifest["tokenizers"][model_name],
        )
        for domain_name in resolved_config.datasets:
            # Match the finalizer's cached fit-profile layer mean exactly.
            mean_scores = np.asarray(
                [
                    [
                        0.1
                        + (0.01 * layer_index)
                        + (0.001 * model_order[model_name])
                        + (0.0001 * domain_order[domain_name])
                        + (head_index * 1e-6)
                        for head_index in range(32)
                    ]
                    for layer_index in range(32)
                ],
                dtype=np.float64,
            )
            raw_kernels = np.full(
                (32, 32, 512),
                fill_value=float((10 * model_order[model_name]) + domain_order[domain_name] + 1),
                dtype=np.float64,
            )
            domain_entry = _batch_domain_entry(
                batch_manifest,
                model_name=model_name,
                domain_name=domain_name,
            )
            payload = _json_object(
                runner_module._serialize_fit_profile(  # pyright: ignore[reportPrivateUsage]
                    resolved_config=resolved_config,
                    sweep=sweep,
                    model_name=model_name,
                    domain_name=domain_name,
                    binding=_frozen_model_binding(binding),
                    materialized_domain_sha256=domain_entry["materialized_domain_sha256"],
                    fit_sequence_digest_sha256=domain_entry["fit_sequence_digest_sha256"],
                    summary=runner_module.DomainFitSummary(
                        mean_scores=mean_scores,
                        raw_kernels=raw_kernels,
                    ),
                )
            )
            load_existing_or_write(
                root.manifests_dir / "fit-profiles" / f"{model_name}.{domain_name}.json",
                payload,
            )
            bins = runner_module.sort_heads_into_bins(mean_scores)
            profiles[(model_name, domain_name)] = FakeFitProfile(
                payload=payload,
                mean_scores=mean_scores,
                raw_kernels=raw_kernels,
                bins=tuple(
                    tuple(FakeHead(layer=head.layer, head=head.head) for head in heads)
                    for heads in bins
                ),
                bin_means=np.asarray(
                    runner_module.bin_mean_source_r2(mean_scores, bins),
                    dtype=np.float64,
                ),
            )
    return profiles


def _install_refinalize_manifest_graph_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_require_inventory = runner_module._require_production_manifest_inventory  # pyright: ignore[reportPrivateUsage]

    def require_manifest_graph_stub(
        *,
        run_root: RunRoot,
        resolved_config: ResolvedConfig,
        sweep: SweepConfig,
        source_v1: JsonObject,
    ) -> tuple[tuple[Path, ...], RunnerBatchManifestPayload, list[str]]:
        manifest_paths = real_require_inventory(
            run_root=run_root,
            resolved_config=resolved_config,
            sweep=sweep,
        )
        source_v1_config_sha256 = _json_string_item(source_v1, "config_sha256")
        source_v1_sweep_sha256 = _json_string_item(source_v1, "sweep_sha256")
        source_v1_batch_sha256 = _json_string_item(source_v1, "batch_manifest_sha256")
        if source_v1_config_sha256 != resolved_config.identity_sha256:
            raise runner_module.RunnerError("Source-v1 config binding drifted.")
        if source_v1_sweep_sha256 != sweep.identity_sha256:
            raise runner_module.RunnerError("Source-v1 sweep binding drifted.")

        batch_manifest = runner_module._load_batch_manifest(  # pyright: ignore[reportPrivateUsage]
            run_root
        )
        if batch_manifest["identity_sha256"] != source_v1_batch_sha256:
            raise runner_module.RunnerError("Batch manifest immutable identity drifted.")
        if batch_manifest["config_sha256"] != source_v1_config_sha256:
            raise runner_module.RunnerError("Batch manifest config binding drifted.")
        if batch_manifest["sweep_sha256"] != source_v1_sweep_sha256:
            raise runner_module.RunnerError("Batch manifest sweep binding drifted.")

        protocol_identity_sha256 = runner_module._protocol_identity(  # pyright: ignore[reportPrivateUsage]
            resolved_config,
            sweep,
        )
        local_batch_manifest = _runner_batch_manifest(batch_manifest)
        fit_profile_identity_sha256s: list[str] = []
        for model_name in sweep.models:
            for domain_name in resolved_config.datasets:
                domain_entry = _batch_domain_entry(
                    local_batch_manifest,
                    model_name=model_name,
                    domain_name=domain_name,
                )
                path = run_root.manifests_dir / "fit-profiles" / f"{model_name}.{domain_name}.json"
                payload = runner_module._fit_profile_payload(  # pyright: ignore[reportPrivateUsage]
                    runner_module.verify_payload_identity(
                        json.loads(path.read_text(encoding="ascii")),
                        label=path.name,
                    ),
                    label=path.name,
                )
                if payload["model"] != model_name or payload["domain"] != domain_name:
                    raise runner_module.RunnerError(f"Fit profile identity drifted: {path.name}")
                if payload["config_sha256"] != resolved_config.identity_sha256:
                    raise runner_module.RunnerError(f"Fit profile config drifted: {path.name}")
                if payload["sweep_sha256"] != sweep.identity_sha256:
                    raise runner_module.RunnerError(f"Fit profile sweep drifted: {path.name}")
                if payload["protocol_identity_sha256"] != protocol_identity_sha256:
                    raise runner_module.RunnerError(f"Fit profile protocol drifted: {path.name}")
                if payload["weights_tree_sha256"] != batch_manifest["models"][model_name]:
                    raise runner_module.RunnerError(
                        f"Fit profile model binding drifted: {path.name}"
                    )
                if payload["tokenizer_tree_sha256"] != batch_manifest["tokenizers"][model_name]:
                    raise runner_module.RunnerError(
                        f"Fit profile tokenizer binding drifted: {path.name}"
                    )
                if (
                    payload["materialized_domain_sha256"]
                    != domain_entry["materialized_domain_sha256"]
                ):
                    raise runner_module.RunnerError(
                        f"Fit profile domain binding drifted: {path.name}"
                    )
                if (
                    payload["fit_sequence_digest_sha256"]
                    != domain_entry["fit_sequence_digest_sha256"]
                ):
                    raise runner_module.RunnerError(
                        f"Fit profile fit-sequence binding drifted: {path.name}"
                    )
                fit_profile_identity_sha256s.append(payload["identity_sha256"])
        return manifest_paths, batch_manifest, fit_profile_identity_sha256s

    monkeypatch.setattr(
        runner_module,
        "_require_production_manifest_graph",
        require_manifest_graph_stub,
    )


def _prepare_finalizer_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    run_name: str = "run",
    write_v1_summary: bool,
) -> tuple[ResolvedDigests, BatchManifestPayload, Path, RunRoot]:
    _assert_canonical_config_and_sweep_raw_bytes()
    resolved = _resolved_digests(
        validate_config("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")
    )
    batch_manifest = _batch_manifest_payload()
    run_root = tmp_path / run_name
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id=None)
    root.ensure()
    load_existing_or_write(root.manifests_dir / "batch-manifest.json", batch_manifest)
    _write_batch_supporting_manifests(
        root=root,
        batch_manifest=batch_manifest,
        config_sha256=resolved.config_sha256,
        sweep_sha256=resolved.sweep_sha256,
    )
    _install_refinalize_manifest_graph_patch(
        monkeypatch,
    )
    _populate_finalize_inventory(
        monkeypatch=monkeypatch,
        root=root,
        resolved=resolved,
        batch_manifest=batch_manifest,
    )
    if write_v1_summary:
        _write_expected_v1_summary(root, batch_manifest=batch_manifest)
        _patch_refinalize_fixed_input_digests(
            monkeypatch,
            root,
            batch_manifest=batch_manifest,
        )
    return resolved, batch_manifest, run_root, root


def _rewrite_json_with_whitespace_only_byte_drift(path: Path) -> None:
    payload = _read_json_object(path)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="ascii",
    )


def _refinalize_batch_v2(
    *,
    run_root: Path,
    source_summary_v1: Path,
    base_path: str | Path = "rebuttal/configs/base.toml",
    sweep_path: str | Path = "rebuttal/configs/sweep.toml",
) -> JsonObject:
    refinalize = getattr(runner_module, "refinalize_batch_v2", None)
    if not callable(refinalize):
        raise AssertionError("runner.refinalize_batch_v2 must exist")
    return _json_object(
        refinalize(
            base_path,
            sweep_path,
            run_root=run_root,
            source_summary_v1=source_summary_v1,
        )
    )


def _dose_stat(payload: Mapping[str, object], *, model: str, direction: str) -> JsonObject:
    return next(
        item
        for item in _json_object_list_item(payload, "dose_response_statistics")
        if _json_string_item(item, "model") == model
        and _json_string_item(item, "direction") == direction
    )


def _synthetic_terminal_row_keys() -> list[tuple[str, str]]:
    return [
        ("llama-3.1-8b", "wikipedia_to_code"),
        ("llama-3.1-8b", "code_to_wikipedia"),
        ("mistral-7b-v0.1", "wikipedia_to_code"),
        ("mistral-7b-v0.1", "code_to_wikipedia"),
        ("olmo-2-7b", "wikipedia_to_code"),
        ("olmo-2-7b", "code_to_wikipedia"),
        ("llama-3.1-8b", "wikipedia_to_code"),
        ("mistral-7b-v0.1", "wikipedia_to_code"),
        ("olmo-2-7b", "wikipedia_to_code"),
    ]


def _synthetic_spearman_rho(
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> tuple[float, Float64Array]:
    rankdata_average = cast(RankdataAverageFn, scipy_stats.rankdata)
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    x_ranks = np.asarray(rankdata_average(x, method="average"), dtype=np.float64)
    y_ranks = np.asarray(rankdata_average(y, method="average"), dtype=np.float64)
    rho_observed = float(np.corrcoef(x_ranks, y_ranks)[0, 1])
    return rho_observed, x_ranks


def _synthetic_spearman_exceedance_count(
    x_values: Sequence[float],
    y_values: Sequence[float],
    *,
    seed: int,
) -> int:
    rankdata_average = cast(RankdataAverageFn, scipy_stats.rankdata)
    y = np.asarray(y_values, dtype=np.float64)
    rho_observed, x_ranks = _synthetic_spearman_rho(x_values, y_values)
    x_centered = x_ranks - x_ranks.mean()
    x_norm = float(np.linalg.norm(x_centered))
    if x_norm == 0.0:
        raise AssertionError("synthetic Spearman x ranks must have non-zero norm")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    exceedances = 0
    remaining = DEFAULT_SPEARMAN_PERMUTATIONS
    while remaining > 0:
        batch = min(5_000, remaining)
        permuted_ranks = np.stack(
            [
                np.asarray(
                    rankdata_average(rng.permutation(y), method="average"),
                    dtype=np.float64,
                )
                for _ in range(batch)
            ],
            axis=0,
        )
        centered = permuted_ranks - permuted_ranks.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1)
        if np.any(norms == 0.0):
            raise AssertionError("synthetic Spearman permutation produced zero norm")
        rho_batch = (centered @ x_centered) / (x_norm * norms)
        exceedances += int(np.count_nonzero(rho_batch >= rho_observed))
        remaining -= batch
    return exceedances


def _synthetic_expected_dose_exceedances(
    root: RunRoot,
) -> dict[tuple[str, str], int]:
    expected: dict[tuple[str, str], int] = {}
    for model_name, direction in _synthetic_terminal_row_keys()[:6]:
        baseline_payload = _read_json_object(
            root.shards_dir / f"{model_name}.{direction}.global.baseline.baseline.json"
        )
        baseline = np.asarray(
            _json_list_item(baseline_payload, "sequence_nll"),
            dtype=np.float64,
        )
        r2_values: list[float] = []
        delta_values: list[float] = []
        for bin_index in range(20):
            source_payload = _read_json_object(
                root.shards_dir / f"{model_name}.{direction}.bin.{bin_index}.source_kernel.json"
            )
            source_nll = np.asarray(
                _json_list_item(source_payload, "sequence_nll"),
                dtype=np.float64,
            )
            r2_values.append(_json_float_item(source_payload, "mean_source_r2"))
            delta_values.append(
                float(np.mean(source_nll - baseline) / _json_int_item(source_payload, "head_count"))
            )
        seed_parts = [
            29039,
            model_name,
            direction,
            "statistic",
            "all",
            0,
            "spearman_response_permutation",
        ]
        expected[(model_name, direction)] = _synthetic_spearman_exceedance_count(
            r2_values,
            delta_values,
            seed=compact_json_seed(seed_parts),
        )
    return expected


def _synthetic_expected_depth_exceedances(root: RunRoot) -> dict[str, int]:
    expected: dict[str, int] = {}
    for model_name, direction in _synthetic_terminal_row_keys()[6:]:
        baseline_payload = _read_json_object(
            root.shards_dir / f"{model_name}.{direction}.global.baseline.baseline.json"
        )
        baseline = np.asarray(
            _json_list_item(baseline_payload, "sequence_nll"),
            dtype=np.float64,
        )
        r2_values: list[float] = []
        delta_values: list[float] = []
        for layer_index in range(32):
            depth_payload = _read_json_object(
                root.shards_dir / f"{model_name}.{direction}.layer.{layer_index}.depth.json"
            )
            depth_nll = np.asarray(
                _json_list_item(depth_payload, "sequence_nll"),
                dtype=np.float64,
            )
            r2_values.append(_json_float_item(depth_payload, "mean_source_r2"))
            delta_values.append(float(np.mean(depth_nll - baseline) / 32.0))
        seed_parts = [
            29039,
            model_name,
            direction,
            "statistic",
            "all",
            0,
            "spearman_response_permutation",
        ]
        expected[model_name] = _synthetic_spearman_exceedance_count(
            r2_values,
            delta_values,
            seed=compact_json_seed(seed_parts),
        )
    return expected


def _set_visible_gpu_count(monkeypatch: pytest.MonkeyPatch, *, count: int) -> None:
    monkeypatch.setattr("si_rebuttal.runner.torch.cuda.is_available", lambda: count > 0)
    monkeypatch.setattr("si_rebuttal.runner.torch.cuda.device_count", lambda: count)


def _runtime_binding_payload() -> JsonObject:
    return _json_object(
        {
            "python_version": sys.version,
            "platform": platform.platform(),
            "package_versions": {
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
            },
            "cuda_version": torch_cuda_version,
            "driver_version": "570.00",
        }
    )


def _runtime_capture_payload(*, visible_device_count: int) -> JsonObject:
    return _json_object(
        {
            "python_executable": str(Path(sys.executable).resolve()),
            "package_freeze": {
                "resolved_executable": str(Path(sys.executable).resolve()),
                "command": [sys.executable, "-m", "pip", "freeze", "--all"],
                "returncode": 0,
                "stdout_text": _FREEZE_OUTPUT,
                "stdout_sha256": hashlib.sha256(_FREEZE_OUTPUT.encode("ascii")).hexdigest(),
                "stderr_text": "",
                "stderr_sha256": hashlib.sha256(b"").hexdigest(),
                "output_lines": [
                    "torch==2.7.0",
                    "transformers==5.3.0",
                    "datasets==4.8.2",
                    "numpy==1.26.4",
                    "scipy==1.11.4",
                    "pandas==2.1.4",
                    "pyarrow==23.0.1",
                    "pytest==7.4.4",
                    "ruff==0.12.5",
                    "basedpyright==1.31.1",
                ],
                "line_count": 10,
                "required_exact_pins": {
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
                },
                "observed_required_pins": {
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
                },
            },
            "torch_cuda": {
                "torch_version": str(torch.__version__),
                "torch_cuda_version": torch_cuda_version,
                "cuda_available": visible_device_count > 0,
                "visible_device_count": visible_device_count,
            },
            "gpu_inventory": {
                "resolved_executable": shutil.which("nvidia-smi") or "nvidia-smi",
                "command": [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,memory.total,compute_cap,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                "returncode": 0,
                "stdout_text": _GPU_INVENTORY_OUTPUT,
                "stdout_sha256": hashlib.sha256(_GPU_INVENTORY_OUTPUT.encode("ascii")).hexdigest(),
                "stderr_text": "",
                "stderr_sha256": hashlib.sha256(b"").hexdigest(),
                "gpus": [
                    {
                        "physical_index": 0,
                        "uuid": "GPU-0000",
                        "name": "NVIDIA L40",
                        "total_memory_mib": 46068,
                        "compute_capability": "8.9",
                        "driver_version": "570.00",
                    },
                    {
                        "physical_index": 1,
                        "uuid": "GPU-1111",
                        "name": "NVIDIA L40",
                        "total_memory_mib": 46068,
                        "compute_capability": "8.9",
                        "driver_version": "570.00",
                    },
                ],
                "line_count": 2,
                "driver_version": "570.00",
            },
        }
    )


def _loaded_bundle(*, batch_manifest: BatchManifestPayload, model_name: str) -> FakeLoadedBundle:
    return FakeLoadedBundle(
        binding=FakeBinding(
            weights_tree_sha256=batch_manifest["models"][model_name],
            tokenizer_tree_sha256=batch_manifest["tokenizers"][model_name],
        ),
        model=object(),
        tokenizer=object(),
        adapter=None,
        logical_device=torch.device("cpu"),
    )


def _batch_manifest_payload(
    *,
    fit_sequences_by_key: dict[str, tuple[Int64Array, ...]] | None = None,
    eval_sequences_by_key: dict[str, tuple[Int64Array, ...]] | None = None,
) -> BatchManifestPayload:
    resolved, sweep = load_configs("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")
    fit_sequences_by_key = {} if fit_sequences_by_key is None else dict(fit_sequences_by_key)
    eval_sequences_by_key = {} if eval_sequences_by_key is None else dict(eval_sequences_by_key)
    domains: dict[str, str] = {}
    domain_entries: dict[str, BatchDomainEntryPayload] = {}
    for model_name in sweep.models:
        for domain_name, dataset in resolved.datasets.items():
            key = f"{model_name}:{domain_name}"
            materialized_sha = _binding_id("materialized", key)
            fit_manifest_sha = _binding_id("tokens", key, "fit")
            eval_manifest_sha = _binding_id("tokens", key, "eval")
            fit_sequence_sha = sequence_digest(
                fit_sequences_by_key.get(key, _repeated_arange_sequences(50))
            )
            eval_sequence_sha = sequence_digest(
                eval_sequences_by_key.get(key, _repeated_arange_sequences(100))
            )
            domains[key] = materialized_sha
            domain_entries[key] = {
                "model": model_name,
                "domain": domain_name,
                "tokenizer_tree_sha256": f"t-{model_name}",
                "materialized_domain_sha256": materialized_sha,
                "dataset_repository": dataset.repository,
                "dataset_revision": dataset.revision,
                "dataset_config": dataset.config,
                "dataset_split": dataset.split,
                "dataset_field": dataset.field,
                "dataset_fingerprint": _binding_id("dataset", key),
                "fit_token_manifest_sha256": fit_manifest_sha,
                "eval_token_manifest_sha256": eval_manifest_sha,
                "fit_sequence_digest_sha256": fit_sequence_sha,
                "eval_sequence_digest_sha256": eval_sequence_sha,
            }
    payload: BatchManifestPayload = {
        "schema_version": 1,
        "materialized_at_utc": "2026-07-27T00:00:00+00:00",
        "command": ["pytest"],
        "config_sha256": resolved.identity_sha256,
        "sweep_sha256": sweep.identity_sha256,
        "domains": domains,
        "domain_entries": domain_entries,
        "models": {model_name: f"w-{model_name}" for model_name in sweep.models},
        "tokenizers": {model_name: f"t-{model_name}" for model_name in sweep.models},
        "git_commit_sha": "git",
        "git_tracked_diff_sha256": "tracked",
        "git_relevant_content_sha256": "content",
        "git_untracked_sha256": "untracked",
        "runtime": _runtime_binding_payload(),
        "runtime_capture": _runtime_capture_payload(visible_device_count=2),
        "environment": _json_object({}),
        "provenance": _json_object(
            {
                "seed_namespace": resolved.seed_namespace,
                "config_identity_sha256": resolved.identity_sha256,
                "sweep_identity_sha256": sweep.identity_sha256,
                "git": {
                    "commit_sha": "git",
                    "tracked_diff_sha256": "tracked",
                    "relevant_content_sha256": "content",
                    "untracked_sha256": "untracked",
                },
            }
        ),
        "identity_sha256": "",
    }
    payload["identity_sha256"] = artifact_identity(payload_without_identity(_json_object(payload)))
    return payload


def _write_batch_supporting_manifests(
    *,
    root: RunRoot,
    batch_manifest: BatchManifestPayload,
    config_sha256: str,
    sweep_sha256: str,
) -> None:
    load_existing_or_write(
        root.manifests_dir / "resolved-config.json",
        _with_identity({"schema_version": 1, "config": {"identity_sha256": config_sha256}}),
    )
    load_existing_or_write(
        root.manifests_dir / "sweep.json",
        _with_identity({"schema_version": 1, "sweep": {"identity_sha256": sweep_sha256}}),
    )
    load_existing_or_write(
        root.manifests_dir / "materialized-data.json",
        _with_identity(
            {
                "schema_version": 1,
                "domains": {
                    key: {
                        "manifest_sha256": entry["materialized_domain_sha256"],
                        "dataset_repository": entry["dataset_repository"],
                        "dataset_revision": entry["dataset_revision"],
                        "dataset_config": entry["dataset_config"],
                        "dataset_split": entry["dataset_split"],
                        "dataset_field": entry["dataset_field"],
                        "dataset_fingerprint": entry["dataset_fingerprint"],
                    }
                    for key, entry in batch_manifest["domain_entries"].items()
                },
            }
        ),
    )
    for entry in batch_manifest["domain_entries"].values():
        for partition in ("fit", "eval"):
            count = 50 if partition == "fit" else 100
            sequence_digest_sha256 = (
                entry["fit_sequence_digest_sha256"]
                if partition == "fit"
                else entry["eval_sequence_digest_sha256"]
            )
            load_existing_or_write(
                root.manifests_dir
                / "tokens"
                / f"{entry['model']}.{entry['domain']}.{partition}.json",
                _with_identity(
                    {
                        "schema_version": 1,
                        "model": entry["model"],
                        "domain": entry["domain"],
                        "partition": partition,
                        "count": count,
                        "config_sha256": config_sha256,
                        "sweep_sha256": sweep_sha256,
                        "materialized_domain_sha256": entry["materialized_domain_sha256"],
                        "sequence_digest_sha256": sequence_digest_sha256,
                        "sequences": [],
                    }
                ),
            )


def _populate_finalize_inventory(
    *,
    monkeypatch: pytest.MonkeyPatch,
    root: RunRoot,
    resolved: ResolvedDigests,
    batch_manifest: BatchManifestPayload,
    invalid_depth: tuple[str, int, int] | None = None,
) -> None:
    from si_rebuttal.controls import compact_ascii_json_seed

    profile_cache = _write_stub_fit_profile_manifests(
        root=root,
        batch_manifest=batch_manifest,
    )

    def fake_profile(model_name: str, domain_name: str) -> FakeFitProfile:
        return profile_cache[(model_name, domain_name)]

    def fake_load_fit_profile(
        *,
        run_root: RunRoot,
        resolved_config: ResolvedConfig,
        sweep: SweepConfig,
        model_name: str,
        domain_name: str,
        binding: object,
        materialized_domain_sha256: str,
        fit_sequence_digest_sha256: str,
    ) -> FakeFitProfile:
        del (
            run_root,
            resolved_config,
            sweep,
            binding,
            materialized_domain_sha256,
            fit_sequence_digest_sha256,
        )
        return profile_cache[(model_name, domain_name)]

    def fake_kernel_map_for_heads(
        heads: Sequence[FakeHead],
        raw_kernels: Float64Array,
    ) -> KernelMap:
        return {
            (head.layer, head.head): [
                float(raw_kernels[0, 0, 0]),
                float(head.layer),
                float(head.head),
            ]
            for head in heads
        }

    def fake_streaming_control_maps(
        heads: Sequence[FakeHead],
        source_map: KernelMap,
        *,
        seed: int,
        control_kind: str,
    ) -> KernelMap:
        del source_map
        return {
            (head.layer, head.head): [
                float(int(seed)),
                float(len(control_kind)),
                float(head.layer + head.head),
            ]
            for head in heads
        }

    monkeypatch.setattr(
        "si_rebuttal.runner._kernel_map_for_heads",
        fake_kernel_map_for_heads,
    )
    monkeypatch.setattr(
        "si_rebuttal.runner._streaming_control_maps",
        fake_streaming_control_maps,
    )
    monkeypatch.setattr(runner_module, "_load_fit_profile", fake_load_fit_profile)

    baseline = [0.0] * 100
    wikipedia_code_baselines: dict[str, str] = {}
    for model in ("llama-3.1-8b", "mistral-7b-v0.1", "olmo-2-7b"):
        for direction in ("wikipedia_to_code", "code_to_wikipedia"):
            source_domain = "wikipedia" if direction == "wikipedia_to_code" else "code"
            target_domain = "code" if direction == "wikipedia_to_code" else "wikipedia"
            source_entry = _batch_domain_entry(
                batch_manifest, model_name=model, domain_name=source_domain
            )
            target_entry = _batch_domain_entry(
                batch_manifest, model_name=model, domain_name=target_domain
            )
            source_profile = fake_profile(model, source_domain)
            target_profile = fake_profile(model, target_domain)
            base = ConditionKey(
                model=model,
                direction=direction,
                unit="global",
                unit_index="baseline",
                condition="baseline",
            )
            baseline_payload = _with_identity(
                {
                    "schema_version": 1,
                    "model": model,
                    "direction": direction,
                    "unit": "global",
                    "unit_index": "baseline",
                    "condition": "baseline",
                    "head_count": 0,
                    "sequence_nll": baseline,
                    "sequence_digest_sha256": target_entry["eval_sequence_digest_sha256"],
                    "weights_tree_sha256": batch_manifest["models"][model],
                    "tokenizer_tree_sha256": batch_manifest["tokenizers"][model],
                    "config_sha256": resolved["config_sha256"],
                    "sweep_sha256": resolved["sweep_sha256"],
                    "batch_manifest_sha256": batch_manifest["identity_sha256"],
                    "source_domain_manifest_sha256": source_entry["materialized_domain_sha256"],
                    "target_domain_manifest_sha256": target_entry["materialized_domain_sha256"],
                }
            )
            load_existing_or_write(root.shards_dir / f"{base.stem()}.json", baseline_payload)
            if direction == "wikipedia_to_code":
                wikipedia_code_baselines[model] = str(baseline_payload["identity_sha256"])
            for bin_index in range(20):
                heads = source_profile.bins[bin_index]
                selected_heads = [{"layer": head.layer, "head": head.head} for head in heads]
                source_map = fake_kernel_map_for_heads(heads, source_profile.raw_kernels)
                target_map = fake_kernel_map_for_heads(heads, target_profile.raw_kernels)
                for condition in ("source_kernel", "target_kernel"):
                    model_offset = {
                        "llama-3.1-8b": 0.0,
                        "mistral-7b-v0.1": 0.02,
                        "olmo-2-7b": 0.04,
                    }[model]
                    direction_offset = 0.0 if direction == "wikipedia_to_code" else 0.01
                    if (model, direction, bin_index, condition) == (
                        "llama-3.1-8b",
                        "wikipedia_to_code",
                        0,
                        "source_kernel",
                    ):
                        values = [6.0] * 100
                    elif condition == "source_kernel":
                        values = [1.0 + model_offset + direction_offset + (0.1 * bin_index)] * 100
                    else:
                        values = [0.5 + model_offset + direction_offset + (0.05 * bin_index)] * 100
                    key = ConditionKey(
                        model=model,
                        direction=direction,
                        unit="bin",
                        unit_index=str(bin_index),
                        condition=condition,
                    )
                    load_existing_or_write(
                        root.shards_dir / f"{key.stem()}.json",
                        _with_identity(
                            {
                                "schema_version": 1,
                                "model": model,
                                "direction": direction,
                                "unit": "bin",
                                "unit_index": str(bin_index),
                                "condition": condition,
                                "head_count": len(heads),
                                "mean_source_r2": float(source_profile.bin_means[bin_index]),
                                "sequence_nll": values,
                                "sequence_digest_sha256": target_entry[
                                    "eval_sequence_digest_sha256"
                                ],
                                "weights_tree_sha256": batch_manifest["models"][model],
                                "tokenizer_tree_sha256": batch_manifest["tokenizers"][model],
                                "config_sha256": resolved["config_sha256"],
                                "sweep_sha256": resolved["sweep_sha256"],
                                "batch_manifest_sha256": batch_manifest["identity_sha256"],
                                "baseline_identity_sha256": baseline_payload["identity_sha256"],
                                "source_domain_manifest_sha256": source_entry[
                                    "materialized_domain_sha256"
                                ],
                                "target_domain_manifest_sha256": target_entry[
                                    "materialized_domain_sha256"
                                ],
                                "source_fit_profile_sha256": source_profile.payload[
                                    "identity_sha256"
                                ],
                                "target_fit_profile_sha256": target_profile.payload[
                                    "identity_sha256"
                                ],
                                "selected_heads": selected_heads,
                                "selected_head_digest_sha256": artifact_identity(selected_heads),
                                "selected_bin_digest_sha256": _json_string_item(
                                    _json_object(
                                        _json_array(source_profile.payload["bins"])[bin_index]
                                    ),
                                    "identity_sha256",
                                ),
                                "source_kernel_digest_sha256": kernel_digest(source_map),
                                "target_kernel_digest_sha256": kernel_digest(target_map),
                            }
                        ),
                    )
                for control in ("offset_permutation", "norm"):
                    for trial in range(3):
                        key = ConditionKey(
                            model=model,
                            direction=direction,
                            unit="bin",
                            unit_index=str(bin_index),
                            condition=control,
                            trial=trial,
                        )
                        if (model, direction, bin_index, control) == (
                            "llama-3.1-8b",
                            "wikipedia_to_code",
                            0,
                            "offset_permutation",
                        ):
                            trial_values = [[1.0] * 100, [4.0] * 100, [7.0] * 100][trial]
                        elif (model, direction, bin_index, control) == (
                            "llama-3.1-8b",
                            "wikipedia_to_code",
                            0,
                            "norm",
                        ):
                            trial_values = [[2.0] * 100, [5.0] * 100, [8.0] * 100][trial]
                        else:
                            trial_values = [1.0] * 100
                        control_seed = compact_ascii_json_seed(
                            [29039, model, direction, "bin", bin_index, trial, control]
                        )
                        control_map = fake_streaming_control_maps(
                            heads, source_map, seed=control_seed, control_kind=control
                        )
                        load_existing_or_write(
                            root.shards_dir / f"{key.stem()}.json",
                            _with_identity(
                                {
                                    "schema_version": 1,
                                    "model": model,
                                    "direction": direction,
                                    "unit": "bin",
                                    "unit_index": str(bin_index),
                                    "condition": control,
                                    "trial": trial,
                                    "control_kind": control,
                                    "control_seed": control_seed,
                                    "head_count": len(heads),
                                    "mean_source_r2": float(source_profile.bin_means[bin_index]),
                                    "sequence_nll": trial_values,
                                    "sequence_digest_sha256": target_entry[
                                        "eval_sequence_digest_sha256"
                                    ],
                                    "weights_tree_sha256": batch_manifest["models"][model],
                                    "tokenizer_tree_sha256": batch_manifest["tokenizers"][model],
                                    "config_sha256": resolved["config_sha256"],
                                    "sweep_sha256": resolved["sweep_sha256"],
                                    "batch_manifest_sha256": batch_manifest["identity_sha256"],
                                    "baseline_identity_sha256": baseline_payload["identity_sha256"],
                                    "source_domain_manifest_sha256": source_entry[
                                        "materialized_domain_sha256"
                                    ],
                                    "target_domain_manifest_sha256": target_entry[
                                        "materialized_domain_sha256"
                                    ],
                                    "source_fit_profile_sha256": source_profile.payload[
                                        "identity_sha256"
                                    ],
                                    "target_fit_profile_sha256": target_profile.payload[
                                        "identity_sha256"
                                    ],
                                    "selected_heads": selected_heads,
                                    "selected_head_digest_sha256": artifact_identity(
                                        selected_heads
                                    ),
                                    "selected_bin_digest_sha256": _json_string_item(
                                        _json_object(
                                            _json_array(source_profile.payload["bins"])[bin_index]
                                        ),
                                        "identity_sha256",
                                    ),
                                    "source_kernel_digest_sha256": kernel_digest(source_map),
                                    "control_kernel_digest_sha256": kernel_digest(control_map),
                                }
                            ),
                        )
        for layer_index in range(32):
            key = ConditionKey(
                model=model,
                direction="wikipedia_to_code",
                unit="layer",
                unit_index=str(layer_index),
                condition="depth",
            )
            source_profile = fake_profile(model, "wikipedia")
            target_profile = fake_profile(model, "code")
            depth_heads = tuple(FakeHead(layer=layer_index, head=head) for head in range(32))
            selected_heads = [{"layer": head.layer, "head": head.head} for head in depth_heads]
            source_map = fake_kernel_map_for_heads(depth_heads, source_profile.raw_kernels)
            sequence_length = 100
            if invalid_depth is not None and (model, layer_index) == (
                invalid_depth[0],
                invalid_depth[1],
            ):
                sequence_length = invalid_depth[2]
            depth_sequence = [
                1.0 + (0.05 * layer_index) + (0.0001 * position)
                for position in range(sequence_length)
            ]
            load_existing_or_write(
                root.shards_dir / f"{key.stem()}.json",
                _with_identity(
                    {
                        "schema_version": 1,
                        "model": model,
                        "direction": "wikipedia_to_code",
                        "unit": "layer",
                        "unit_index": str(layer_index),
                        "condition": "depth",
                        "head_count": len(depth_heads),
                        "mean_source_r2": float(
                            np.asarray(
                                source_profile.mean_scores[layer_index], dtype=np.float64
                            ).mean()
                        ),
                        "sequence_nll": depth_sequence,
                        "sequence_digest_sha256": _batch_domain_entry(
                            batch_manifest, model_name=model, domain_name="code"
                        )["eval_sequence_digest_sha256"],
                        "weights_tree_sha256": batch_manifest["models"][model],
                        "tokenizer_tree_sha256": batch_manifest["tokenizers"][model],
                        "config_sha256": resolved["config_sha256"],
                        "sweep_sha256": resolved["sweep_sha256"],
                        "batch_manifest_sha256": batch_manifest["identity_sha256"],
                        "baseline_identity_sha256": wikipedia_code_baselines[model],
                        "source_domain_manifest_sha256": _batch_domain_entry(
                            batch_manifest, model_name=model, domain_name="wikipedia"
                        )["materialized_domain_sha256"],
                        "target_domain_manifest_sha256": _batch_domain_entry(
                            batch_manifest, model_name=model, domain_name="code"
                        )["materialized_domain_sha256"],
                        "source_fit_profile_sha256": source_profile.payload["identity_sha256"],
                        "target_fit_profile_sha256": target_profile.payload["identity_sha256"],
                        "selected_heads": selected_heads,
                        "selected_head_digest_sha256": artifact_identity(selected_heads),
                        "source_kernel_digest_sha256": kernel_digest(source_map),
                    }
                ),
            )


def _artifact_projection_for_test_root(run_root: Path, model_name: str) -> tuple[int, JsonObject]:
    resolved_config, sweep = load_configs(
        "rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml"
    )
    projected_artifact_bytes, _, artifact_components = project_artifact_bytes(
        run_root=RunRoot(run_id=run_root.name, root=run_root, batch_id=None),
        resolved_config=resolved_config,
        sweep=sweep,
        model_name=model_name,
    )
    return projected_artifact_bytes, artifact_components


def _validation_receipt_payload(
    *,
    model_name: str,
    batch_manifest: BatchManifestPayload,
    config_sha256: str,
    sweep_sha256: str,
) -> JsonObject:
    domain_entry = _batch_domain_entry(
        batch_manifest, model_name=model_name, domain_name="wikipedia"
    )
    return _with_identity(
        {
            "schema_version": 1,
            "model": model_name,
            "config_sha256": config_sha256,
            "sweep_sha256": sweep_sha256,
            "batch_manifest_sha256": batch_manifest["identity_sha256"],
            "validation_domain": "wikipedia",
            "source_domain_manifest_sha256": domain_entry["materialized_domain_sha256"],
            "fit_token_manifest_sha256": domain_entry["fit_token_manifest_sha256"],
            "fit_sequence_digest_sha256": domain_entry["fit_sequence_digest_sha256"],
            "weights_tree_sha256": batch_manifest["models"][model_name],
            "tokenizer_tree_sha256": batch_manifest["tokenizers"][model_name],
            "baseline_nll": 1.0,
            "zero_nll": 1.0,
            "constant_nll": 1.0,
            "nonconstant_nll": 1.1,
            "signed_by_digests": [
                batch_manifest["models"][model_name],
                batch_manifest["tokenizers"][model_name],
                config_sha256,
                sweep_sha256,
                batch_manifest["identity_sha256"],
            ],
        }
    )


def _benchmark_receipt_payload(
    *,
    model_name: str,
    batch_manifest: BatchManifestPayload,
    config_sha256: str,
    sweep_sha256: str,
    validation_receipt_sha256: str,
    lane_hours: float = 1.0,
    projected_total_seconds: float | None = None,
    stored_lane_hours: float | None = None,
    projected_artifact_bytes: int = 100,
    artifact_components: JsonObject | None = None,
) -> JsonObject:
    projected_total_seconds = (
        lane_hours * 3600.0 if projected_total_seconds is None else float(projected_total_seconds)
    )
    stored_lane_hours = lane_hours if stored_lane_hours is None else float(stored_lane_hours)
    runtime_components = {
        "fit_capture_kernel_r2": {
            "formula": "measured_seconds * (projected_sequence_count / measured_sequence_count)",
            "measured_seconds": projected_total_seconds,
            "measured_sequence_count": 100,
            "projected_sequence_count": 100,
            "extrapolation_factor": 1.0,
            "projected_seconds": projected_total_seconds,
        },
        "baseline_eval": {
            "formula": "measured_seconds * (projected_sequence_count / measured_sequence_count)",
            "measured_seconds": 0.0,
            "measured_sequence_count": 200,
            "projected_sequence_count": 200,
            "extrapolation_factor": 1.0,
            "projected_seconds": 0.0,
        },
        "source_bin_eval": {
            "formula": "measured_seconds * (projected_sequence_count / measured_sequence_count)",
            "measured_seconds": 0.0,
            "measured_sequence_count": 4000,
            "projected_sequence_count": 4000,
            "extrapolation_factor": 1.0,
            "projected_seconds": 0.0,
        },
        "target_bin_eval": {
            "formula": "measured_seconds * (projected_sequence_count / measured_sequence_count)",
            "measured_seconds": 0.0,
            "measured_sequence_count": 4000,
            "projected_sequence_count": 4000,
            "extrapolation_factor": 1.0,
            "projected_seconds": 0.0,
        },
        "control_eval": {
            "formula": "measured_seconds * (projected_sequence_count / measured_sequence_count)",
            "measured_seconds": 0.0,
            "measured_sequence_count": 24000,
            "projected_sequence_count": 24000,
            "extrapolation_factor": 1.0,
            "projected_seconds": 0.0,
        },
        "depth_eval": {
            "formula": "measured_seconds * (projected_sequence_count / measured_sequence_count)",
            "measured_seconds": 0.0,
            "measured_sequence_count": 3200,
            "projected_sequence_count": 3200,
            "extrapolation_factor": 1.0,
            "projected_seconds": 0.0,
        },
    }
    artifact_components = (
        {
            "inventory_version": 1,
            "inventory": {
                "token_manifests": {"count": 12, "representative_bytes": 1, "projected_bytes": 12}
            },
            "artifact_count": 12,
            "inventory_component_count": 1,
            "immutable_materialized_bytes": 0,
            "future_artifact_bytes": projected_artifact_bytes,
            "future_margin_multiplier": 1.0,
            "future_margin_bytes": projected_artifact_bytes,
        }
        if artifact_components is None
        else dict(artifact_components)
    )
    return _with_identity(
        {
            "schema_version": 1,
            "model": model_name,
            "config_sha256": config_sha256,
            "sweep_sha256": sweep_sha256,
            "batch_manifest_sha256": batch_manifest["identity_sha256"],
            "weights_tree_sha256": batch_manifest["models"][model_name],
            "tokenizer_tree_sha256": batch_manifest["tokenizers"][model_name],
            "validation_receipt_sha256": validation_receipt_sha256,
            "fit_sequences": 50,
            "eval_sequences": 100,
            "measured_seconds": {
                "fit_capture_kernel_r2": projected_total_seconds,
                "baseline_eval": 0.0,
                "source_bin_eval": 0.0,
                "target_bin_eval": 0.0,
                "control_eval": 0.0,
                "depth_eval": 0.0,
            },
            "measured_counts": {
                "fit_capture_kernel_r2": 100,
                "baseline_eval": 200,
                "source_bin_eval": 4000,
                "target_bin_eval": 4000,
                "control_eval": 24000,
                "depth_eval": 3200,
            },
            "projected_total_seconds": projected_total_seconds,
            "projected_artifact_bytes": projected_artifact_bytes,
            "lane_hours": stored_lane_hours,
            "runtime_components": runtime_components,
            "runtime_formula": BENCHMARK_RUNTIME_FORMULA,
            "artifact_components": artifact_components,
            "projection_components": artifact_components,
            "artifact_formula": BENCHMARK_ARTIFACT_FORMULA,
            "formula": BENCHMARK_ARTIFACT_FORMULA,
        }
    )


def _lane_receipt_payload(
    *,
    batch_manifest: BatchManifestPayload,
    config_sha256: str,
    sweep_sha256: str,
    benchmark_receipts: dict[str, JsonObject],
) -> JsonObject:
    gpu0 = ["llama-3.1-8b", "olmo-2-7b"]
    gpu1 = ["mistral-7b-v0.1"]
    first_receipt = next(iter(benchmark_receipts.values()))
    projected_artifact_bytes = _json_int_item(first_receipt, "projected_artifact_bytes")
    required_free_bytes = (2 * projected_artifact_bytes) + int(20.0 * 1024**3)
    lane_hours = {
        name: _json_float_item(payload, "projected_total_seconds") / 3600.0
        for name, payload in benchmark_receipts.items()
    }
    gpu0_hours = sum(lane_hours[name] for name in gpu0)
    gpu1_hours = sum(lane_hours[name] for name in gpu1)
    return _with_identity(
        {
            "schema_version": 1,
            "admitted": True,
            "config_sha256": config_sha256,
            "sweep_sha256": sweep_sha256,
            "batch_manifest_sha256": batch_manifest["identity_sha256"],
            "benchmark_receipts": {
                name: _json_string_item(payload, "identity_sha256")
                for name, payload in benchmark_receipts.items()
            },
            "benchmark_projected_artifact_bytes": {
                name: _json_int_item(payload, "projected_artifact_bytes")
                for name, payload in benchmark_receipts.items()
            },
            "benchmark_projected_total_seconds": {
                name: _json_float_item(payload, "projected_total_seconds")
                for name, payload in benchmark_receipts.items()
            },
            "benchmark_lane_hours": lane_hours,
            "placement_gpu0": gpu0,
            "placement_gpu1": gpu1,
            "gpu0_hours": gpu0_hours,
            "gpu1_hours": gpu1_hours,
            "projected_complete_artifact_bytes": projected_artifact_bytes,
            "projected_artifact_bytes": projected_artifact_bytes,
            "required_free_bytes": required_free_bytes,
            "required_disk_bytes": required_free_bytes,
            "free_disk_bytes": required_free_bytes + 1,
            "lane_components": {
                "benchmark_lane_hours": lane_hours,
                "placement_gpu0": gpu0,
                "placement_gpu1": gpu1,
                "benchmark_max_lane_hours": 120.0,
                "gpu0_hours": gpu0_hours,
                "gpu1_hours": gpu1_hours,
                "gpu0_admitted": True,
                "gpu1_admitted": True,
            },
            "lane_formula": LANE_HOURS_FORMULA,
            "disk_components": {
                "projected_complete_artifact_bytes": projected_artifact_bytes,
                "benchmark_min_free_space_gib": 20.0,
                "benchmark_min_free_space_bytes": int(20.0 * 1024**3),
                "artifact_copy_factor": 2,
                "required_free_bytes": required_free_bytes,
                "required_disk_bytes": required_free_bytes,
                "free_disk_bytes": required_free_bytes + 1,
                "disk_admitted": True,
            },
            "disk_formula": DISK_HEADROOM_FORMULA,
        }
    )


@pytest.fixture(name="_set_env", autouse=True)
def set_env_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for key, value in _env(tmp_path).items():
        monkeypatch.setenv(key, value)


def test_validate_config_surface() -> None:
    payload = validate_config("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")
    assert payload["schema_version"] == 1
    assert payload["models"] == ["llama-3.1-8b", "mistral-7b-v0.1", "olmo-2-7b"]
    resolved, _ = load_configs("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")
    assert resolved.paths.project_root.name == "rebuttal"


def test_load_configs_normalizes_legacy_model_headers_and_cleans_temp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[Path] = []
    real_tempdir = tempfile.TemporaryDirectory

    class TrackingTemporaryDirectory:
        def __init__(
            self,
            suffix: str | None = None,
            prefix: str | None = None,
            dir: str | os.PathLike[str] | None = None,
            ignore_cleanup_errors: bool = False,
        ) -> None:
            self._inner = cast(
                TemporaryDirectoryLike,
                cast(
                    object,
                    real_tempdir(
                        suffix=suffix,
                        prefix=prefix,
                        dir=dir,
                        ignore_cleanup_errors=ignore_cleanup_errors,
                    ),
                ),
            )
            created.append(Path(self._inner.name))

        def __enter__(self) -> str:
            return self._inner.__enter__()

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            self._inner.__exit__(exc_type, exc, tb)

    monkeypatch.setattr(
        "si_rebuttal.runner.tempfile.TemporaryDirectory", TrackingTemporaryDirectory
    )

    resolved, _ = load_configs("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")

    assert list(resolved.models) == ["llama-3.1-8b", "mistral-7b-v0.1", "olmo-2-7b"]
    assert resolved.paths.project_root == Path("rebuttal").resolve()
    assert resolved.source_path == Path("rebuttal/configs/base.toml").resolve()
    assert not created


def test_load_configs_rewrites_legacy_model_headers_and_cleans_temp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    created: list[Path] = []
    real_tempdir = tempfile.TemporaryDirectory

    class TrackingTemporaryDirectory:
        def __init__(
            self,
            suffix: str | None = None,
            prefix: str | None = None,
            dir: str | os.PathLike[str] | None = None,
            ignore_cleanup_errors: bool = False,
        ) -> None:
            self._inner = cast(
                TemporaryDirectoryLike,
                cast(
                    object,
                    real_tempdir(
                        suffix=suffix,
                        prefix=prefix,
                        dir=dir,
                        ignore_cleanup_errors=ignore_cleanup_errors,
                    ),
                ),
            )
            created.append(Path(self._inner.name))

        def __enter__(self) -> str:
            return self._inner.__enter__()

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            self._inner.__exit__(exc_type, exc, tb)

    monkeypatch.setattr(
        "si_rebuttal.runner.tempfile.TemporaryDirectory", TrackingTemporaryDirectory
    )

    base_text = Path("rebuttal/configs/base.toml").read_text(encoding="utf-8")
    legacy_path = tmp_path / "base-legacy.toml"
    legacy_path.write_text(
        base_text.replace(
            'project_root = ".."',
            f'project_root = "{os.path.relpath(Path("rebuttal").resolve(), tmp_path)}"',
        )
        .replace('[models."llama-3.1-8b"]', "[models.llama-3.1-8b]")
        .replace('[models."mistral-7b-v0.1"]', "[models.mistral-7b-v0.1]")
        .replace('[models."olmo-2-7b"]', "[models.olmo-2-7b]"),
        encoding="utf-8",
    )

    resolved, _ = load_configs(legacy_path, "rebuttal/configs/sweep.toml")

    assert list(resolved.models) == ["llama-3.1-8b", "mistral-7b-v0.1", "olmo-2-7b"]
    assert resolved.paths.project_root == Path("rebuttal").resolve()
    assert resolved.source_path == legacy_path.resolve()
    assert created
    assert all(not path.exists() for path in created)


def test_materialize_data_uses_offline_tokenizer_and_real_digests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return list(range(512)) if text else []

    fake_loader = RecordedTokenizerLoader(calls=calls, tokenizer=FakeTokenizer())

    monkeypatch.setattr(
        "si_rebuttal.runner.materialize_frozen_domain", _fake_materialize_frozen_domain_stub
    )
    monkeypatch.setattr("si_rebuttal.runner.capture_git_binding", _git_binding_stub)
    monkeypatch.setattr(
        "si_rebuttal.runner._reconstruct_sequences_for_materialized_domain",
        _reconstruct_sequences_from_materialized_stub,
    )
    load_dataset = _dataset_loader(
        FakeLoadedDataset(
            rows=tuple(_iter_fake_dataset_rows()),
            column_names=("text", "repository_name", "whole_func_string"),
        )
    )

    payload = materialize_data(
        "rebuttal/configs/base.toml",
        "rebuttal/configs/sweep.toml",
        run_root=tmp_path / "materialize",
        load_dataset_fn=load_dataset,
        tokenizer_loader=fake_loader,
        runtime_command_runner=_runtime_command_runner(),
    )
    assert payload["schema_version"] == 1
    assert payload["batch_manifest_sha256"]
    assert len(calls) >= 3
    manifest = _require_batch_manifest(
        _read_json_object(tmp_path / "materialize" / "manifests" / "batch-manifest.json")
    )
    assert _json_string_item(manifest, "materialized_at_utc").endswith("+00:00")
    assert _json_string_item(_json_object_item(manifest, "runtime"), "driver_version") == "570.00"
    package_freeze = _json_object_item(
        _json_object_item(manifest, "runtime_capture"), "package_freeze"
    )
    output_lines = _json_list_item(package_freeze, "output_lines")
    assert output_lines[0] == "torch==2.7.0"
    gpu_inventory = _json_object_item(
        _json_object_item(manifest, "runtime_capture"), "gpu_inventory"
    )
    first_gpu = _json_object(_json_list_item(gpu_inventory, "gpus")[0])
    assert _json_string_item(first_gpu, "compute_capability") == "8.9"
    assert all(value != "pending" for value in _json_object_item(manifest, "models").values())
    assert all(value != "pending" for value in _json_object_item(manifest, "tokenizers").values())


def test_materialize_data_preserves_full_pip_freeze_output_and_reuses_exact_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return list(range(512)) if text else []

    monkeypatch.setattr(
        "si_rebuttal.runner.materialize_frozen_domain", _fake_materialize_frozen_domain_stub
    )
    monkeypatch.setattr("si_rebuttal.runner.capture_git_binding", _git_binding_stub)
    monkeypatch.setattr(
        "si_rebuttal.runner._reconstruct_sequences_for_materialized_domain",
        _reconstruct_sequences_from_materialized_stub,
    )
    load_dataset = _dataset_loader(
        FakeLoadedDataset(
            rows=tuple(_iter_fake_dataset_rows()),
            column_names=("text", "repository_name", "whole_func_string"),
        )
    )
    load_tokenizer = _tokenizer_loader(FakeTokenizer())

    freeze_lines = _editable_freeze_lines()
    run_root = tmp_path / "materialize"
    payload = materialize_data(
        "rebuttal/configs/base.toml",
        "rebuttal/configs/sweep.toml",
        run_root=run_root,
        load_dataset_fn=load_dataset,
        tokenizer_loader=load_tokenizer,
        runtime_command_runner=_runtime_command_runner(freeze_lines=freeze_lines),
    )
    manifest = _read_json_object(run_root / "manifests" / "batch-manifest.json")
    package_freeze = _json_object_item(
        _json_object_item(manifest, "runtime_capture"), "package_freeze"
    )
    expected_stdout = "\n".join(freeze_lines) + "\n"
    assert _json_string_item(package_freeze, "stdout_text") == expected_stdout
    assert (
        _json_string_item(package_freeze, "stdout_sha256")
        == hashlib.sha256(expected_stdout.encode("utf-8")).hexdigest()
    )
    filtered_freeze_lines = [line for line in freeze_lines if line]
    assert _json_list_item(package_freeze, "output_lines") == filtered_freeze_lines
    assert _json_int_item(package_freeze, "line_count") == len(filtered_freeze_lines)
    assert _json_list_item(package_freeze, "output_lines")[:2] == freeze_lines[:2]
    assert (
        _json_string_item(_json_object_item(package_freeze, "required_exact_pins"), "transformers")
        == "5.3.0"
    )
    assert (
        _json_string_item(
            _json_object_item(package_freeze, "observed_required_pins"), "basedpyright"
        )
        == "1.31.1"
    )
    assert _json_object_item(_json_object_item(manifest, "runtime"), "package_versions") == {
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

    repeated = materialize_data(
        "rebuttal/configs/base.toml",
        "rebuttal/configs/sweep.toml",
        run_root=run_root,
        load_dataset_fn=_raising_dataset_loader("loader must not run on repeat"),
        tokenizer_loader=_raising_tokenizer_loader("tokenizer must not run on repeat"),
        runtime_command_runner=_runtime_command_runner(freeze_lines=freeze_lines),
    )
    assert repeated == payload


def test_materialize_data_reuses_existing_manifest_without_loader_and_rejects_tamper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return list(range(512)) if text else []

    monkeypatch.setattr(
        "si_rebuttal.runner.materialize_frozen_domain", _fake_materialize_frozen_domain_stub
    )
    monkeypatch.setattr("si_rebuttal.runner.capture_git_binding", _git_binding_stub)
    monkeypatch.setattr(
        "si_rebuttal.runner._reconstruct_sequences_for_materialized_domain",
        _reconstruct_sequences_from_materialized_stub,
    )
    load_dataset = _dataset_loader(
        FakeLoadedDataset(
            rows=tuple(_iter_fake_dataset_rows()),
            column_names=("text", "repository_name", "whole_func_string"),
        )
    )
    load_tokenizer = _tokenizer_loader(FakeTokenizer())
    run_root = tmp_path / "materialize"
    payload = materialize_data(
        "rebuttal/configs/base.toml",
        "rebuttal/configs/sweep.toml",
        run_root=run_root,
        load_dataset_fn=load_dataset,
        tokenizer_loader=load_tokenizer,
        runtime_command_runner=_runtime_command_runner(),
    )
    first_manifest = _read_json_object(run_root / "manifests" / "batch-manifest.json")
    repeated = materialize_data(
        "rebuttal/configs/base.toml",
        "rebuttal/configs/sweep.toml",
        run_root=run_root,
        load_dataset_fn=_raising_dataset_loader("loader must not run on repeat"),
        tokenizer_loader=_raising_tokenizer_loader("tokenizer must not run on repeat"),
        runtime_command_runner=_runtime_command_runner(),
    )
    assert repeated == payload
    repeated_manifest = _read_json_object(run_root / "manifests" / "batch-manifest.json")
    assert _json_string_item(repeated_manifest, "materialized_at_utc") == _json_string_item(
        first_manifest, "materialized_at_utc"
    )
    token_manifest = next((run_root / "manifests" / "tokens").glob("*.json"))
    token_payload = _read_json_object(token_manifest)
    token_payload["sequence_digest_sha256"] = "tampered"
    token_manifest.write_text(
        json.dumps(token_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    with pytest.raises(
        Exception,
        match=rf"Token manifest verification failed for {re.escape(token_manifest.name)}\.",
    ):
        materialize_data(
            "rebuttal/configs/base.toml",
            "rebuttal/configs/sweep.toml",
            run_root=run_root,
            load_dataset_fn=_raising_dataset_loader("loader must not run after tamper"),
            tokenizer_loader=_raising_tokenizer_loader("tokenizer must not run after tamper"),
            runtime_command_runner=_runtime_command_runner(),
        )


def test_materialize_data_rejects_cross_tokenizer_selected_union_overlap_before_manifest_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return list(range(512)) if text else []

    def custom_materialize_domain(*args: object, **kwargs: object) -> MaterializedDomain:
        del args
        dataset = _dataset_config_from_kwargs(kwargs)
        model_name = _string_kwarg(kwargs, "tokenizer_name")
        tokenizer_tree_sha256 = _string_kwarg(kwargs, "tokenizer_tree_sha256")
        domain_name = (
            "wikipedia"
            if dataset.repository == _dataset_fixture("wikipedia")["repository"]
            else "code"
        )
        if domain_name == "code":
            return _fake_materialized_domain(
                domain_name=domain_name,
                model_name=model_name,
                tokenizer_tree_sha256=tokenizer_tree_sha256,
            )
        fit_documents = (
            _document_record(
                domain_name=domain_name,
                document_id=f"{model_name}-fit",
                partition="fit",
                row_indices=(10,),
                text=f"{model_name} fit clean",
            ),
        )
        eval_row = 7 if model_name == "mistral-7b-v0.1" else 20
        eval_text = "shared overlap content" if model_name == "mistral-7b-v0.1" else "eval clean"
        eval_documents = (
            _document_record(
                domain_name=domain_name,
                document_id=f"{model_name}-eval",
                partition="eval",
                row_indices=(eval_row,),
                text=eval_text,
            ),
        )
        if model_name == "llama-3.1-8b":
            fit_documents = (
                _document_record(
                    domain_name=domain_name,
                    document_id="llama-fit-overlap",
                    partition="fit",
                    row_indices=(7,),
                    text="shared overlap content",
                ),
            )
        return _custom_materialized_domain(
            domain_name=domain_name,
            model_name=model_name,
            tokenizer_tree_sha256=tokenizer_tree_sha256,
            fit_documents=fit_documents,
            eval_documents=eval_documents,
            fit_chunks=(
                _selected_chunk(
                    model_name=model_name,
                    domain_name=domain_name,
                    document=fit_documents[0],
                    item_index=0,
                ),
            ),
            eval_chunks=(
                _selected_chunk(
                    model_name=model_name,
                    domain_name=domain_name,
                    document=eval_documents[0],
                    item_index=0,
                ),
            ),
        )

    writes: list[Path] = []

    def tracking_write(path: Path, payload: object, *, label: str) -> tuple[JsonObject, bool]:
        del payload, label
        writes.append(path)
        raise AssertionError("manifest write must not run before selected-union validation")

    monkeypatch.setattr("si_rebuttal.runner.materialize_frozen_domain", custom_materialize_domain)
    monkeypatch.setattr("si_rebuttal.runner.capture_git_binding", _git_binding_stub)
    monkeypatch.setattr(
        "si_rebuttal.runner._reconstruct_sequences_for_materialized_domain",
        _raise_assertion_callable("reconstruct must not run after selected-union overlap"),
    )
    monkeypatch.setattr("si_rebuttal.runner._load_existing_or_write_payload", tracking_write)

    load_dataset = _dataset_loader(
        FakeLoadedDataset(
            rows=tuple(_iter_fake_dataset_rows()),
            column_names=("text", "repository_name", "whole_func_string"),
        )
    )
    load_tokenizer = _tokenizer_loader(FakeTokenizer())

    with pytest.raises(
        Exception,
        match=(
            r"wikipedia selected fit/eval union overlap detected: "
            r".*source rows \[7\].*joined-content hashes"
        ),
    ):
        materialize_data(
            "rebuttal/configs/base.toml",
            "rebuttal/configs/sweep.toml",
            run_root=tmp_path / "materialize",
            load_dataset_fn=load_dataset,
            tokenizer_loader=load_tokenizer,
            runtime_command_runner=_runtime_command_runner(),
        )

    assert writes == []
    manifests_dir = tmp_path / "materialize" / "manifests"
    assert not manifests_dir.exists()


def test_materialize_data_preserves_payloads_and_selection_for_clean_selected_unions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected_materialized_payloads: dict[str, JsonObject] = {}
    expected_materialized_bytes: dict[str, bytes] = {}

    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return list(range(512)) if text else []

    def custom_materialize_domain(*args: object, **kwargs: object) -> MaterializedDomain:
        del args
        dataset = _dataset_config_from_kwargs(kwargs)
        model_name = _string_kwarg(kwargs, "tokenizer_name")
        tokenizer_tree_sha256 = _string_kwarg(kwargs, "tokenizer_tree_sha256")
        domain_name = (
            "wikipedia"
            if dataset.repository == _dataset_fixture("wikipedia")["repository"]
            else "code"
        )
        if domain_name == "code":
            materialized = _fake_materialized_domain(
                domain_name=domain_name,
                model_name=model_name,
                tokenizer_tree_sha256=tokenizer_tree_sha256,
            )
        else:
            index = {"llama-3.1-8b": 0, "mistral-7b-v0.1": 1, "olmo-2-7b": 2}[model_name]
            shared_unused_text = "unused duplicate inventory shared across partitions"
            fit_documents = (
                _document_record(
                    domain_name=domain_name,
                    document_id=f"{model_name}-fit",
                    partition="fit",
                    row_indices=(index,),
                    text=f"{model_name} fit unique",
                ),
                _document_record(
                    domain_name=domain_name,
                    document_id=f"{model_name}-fit-unused",
                    partition="fit",
                    row_indices=(index + 100,),
                    text=shared_unused_text,
                ),
            )
            eval_documents = (
                _document_record(
                    domain_name=domain_name,
                    document_id=f"{model_name}-eval",
                    partition="eval",
                    row_indices=(index + 10,),
                    text=f"{model_name} eval unique",
                ),
                _document_record(
                    domain_name=domain_name,
                    document_id=f"{model_name}-eval-unused",
                    partition="eval",
                    row_indices=(index + 200,),
                    text=shared_unused_text,
                ),
            )
            materialized = _custom_materialized_domain(
                domain_name=domain_name,
                model_name=model_name,
                tokenizer_tree_sha256=tokenizer_tree_sha256,
                fit_documents=fit_documents,
                eval_documents=eval_documents,
                fit_chunks=(
                    _selected_chunk(
                        model_name=model_name,
                        domain_name=domain_name,
                        document=fit_documents[0],
                        item_index=index,
                    ),
                ),
                eval_chunks=(
                    _selected_chunk(
                        model_name=model_name,
                        domain_name=domain_name,
                        document=eval_documents[0],
                        item_index=index,
                    ),
                ),
            )
        key = f"{model_name}:{domain_name}"
        expected_payload, expected_bytes = _materialized_payload_snapshot(materialized)
        expected_materialized_payloads[key] = expected_payload
        expected_materialized_bytes[key] = expected_bytes
        return materialized

    monkeypatch.setattr("si_rebuttal.runner.materialize_frozen_domain", custom_materialize_domain)
    monkeypatch.setattr("si_rebuttal.runner.capture_git_binding", _git_binding_stub)
    monkeypatch.setattr(
        "si_rebuttal.runner._reconstruct_sequences_for_materialized_domain",
        _reconstruct_sequences_by_chunk_count,
    )

    run_root = tmp_path / "materialize"
    payload = materialize_data(
        "rebuttal/configs/base.toml",
        "rebuttal/configs/sweep.toml",
        run_root=run_root,
        load_dataset_fn=_dataset_loader(
            FakeLoadedDataset(
                rows=tuple(_iter_fake_dataset_rows()),
                column_names=("text", "repository_name", "whole_func_string"),
            )
        ),
        tokenizer_loader=_tokenizer_loader(FakeTokenizer()),
        runtime_command_runner=_runtime_command_runner(),
    )

    assert payload["materialized_keys"] == [
        "llama-3.1-8b:code",
        "llama-3.1-8b:wikipedia",
        "mistral-7b-v0.1:code",
        "mistral-7b-v0.1:wikipedia",
        "olmo-2-7b:code",
        "olmo-2-7b:wikipedia",
    ]
    materialized_manifest = _read_json_object(run_root / "manifests" / "materialized-data.json")
    domains_payload = _json_object_item(materialized_manifest, "domains")
    wikipedia_payload = _json_object_item(domains_payload, "llama-3.1-8b:wikipedia")
    assert set(domains_payload) == set(expected_materialized_payloads)
    assert list(wikipedia_payload) == [
        "dataset_config",
        "dataset_field",
        "dataset_fingerprint",
        "dataset_repository",
        "dataset_revision",
        "dataset_split",
        "domain",
        "eval_chunks",
        "eval_documents",
        "fit_chunks",
        "fit_documents",
        "identity_sha256",
        "manifest_sha256",
        "schema_version",
        "tokenizer_name",
        "tokenizer_tree_sha256",
    ]
    assert _json_string_item(wikipedia_payload, "domain") == "wikipedia"
    fit_documents_payload = _json_object_list_item(wikipedia_payload, "fit_documents")
    eval_documents_payload = _json_object_list_item(wikipedia_payload, "eval_documents")
    fit_chunks_payload = _json_object_list_item(wikipedia_payload, "fit_chunks")
    eval_chunks_payload = _json_object_list_item(wikipedia_payload, "eval_chunks")
    assert len(fit_documents_payload) == 2
    assert len(eval_documents_payload) == 2
    assert _json_string_item(fit_documents_payload[1], "content_sha256") == _json_string_item(
        eval_documents_payload[1], "content_sha256"
    )
    assert _json_string_item(fit_chunks_payload[0], "document_id") == _json_string_item(
        fit_documents_payload[0], "document_id"
    )
    assert _json_string_item(eval_chunks_payload[0], "document_id") == _json_string_item(
        eval_documents_payload[0], "document_id"
    )
    assert _json_list_item(fit_chunks_payload[0], "source_rows") == [0]
    assert _json_list_item(eval_chunks_payload[0], "source_rows") == [10]
    for key, expected_payload in expected_materialized_payloads.items():
        stored_payload = _json_object_item(domains_payload, key)
        assert stored_payload == expected_payload
        assert canonical_json_bytes(stored_payload) == expected_materialized_bytes[key]
    fit_manifest = _read_json_object(
        run_root / "manifests" / "tokens" / "llama-3.1-8b.wikipedia.fit.json"
    )
    eval_manifest = _read_json_object(
        run_root / "manifests" / "tokens" / "llama-3.1-8b.wikipedia.eval.json"
    )
    assert _json_int_item(fit_manifest, "count") == 1
    assert _json_int_item(eval_manifest, "count") == 1
    assert len(_json_object_list_item(fit_manifest, "sequences")) == 1
    assert len(_json_object_list_item(eval_manifest, "sequences")) == 1


@pytest.mark.parametrize(
    ("runtime_runner", "match"),
    (
        (
            _runtime_command_runner(
                failures={
                    (sys.executable, "-m", "pip", "freeze", "--all"): subprocess.CompletedProcess(
                        [sys.executable, "-m", "pip", "freeze", "--all"],
                        1,
                        stdout="",
                        stderr="freeze failed",
                    )
                }
            ),
            "pip freeze --all failed",
        ),
        (
            _runtime_command_runner(freeze_lines=["", ""]),
            "pip freeze --all returned no meaningful package lines",
        ),
        (
            _runtime_command_runner(freeze_lines=["torch==2.7.0\x00", "transformers==5.3.0"]),
            "pip freeze --all emitted NUL bytes",
        ),
        (
            _runtime_command_runner(
                freeze_lines=[
                    "torch==2.7.0",
                    "transformers==5.2.9",
                    "datasets==4.8.2",
                    "numpy==1.26.4",
                    "scipy==1.11.4",
                    "pandas==2.1.4",
                    "pyarrow==23.0.1",
                    "pytest==7.4.4",
                    "ruff==0.12.5",
                    "basedpyright==1.31.1",
                ]
            ),
            "version mismatch for transformers",
        ),
        (
            _runtime_command_runner(
                freeze_lines=[
                    "torch==2.7.0",
                    "transformers @ file:///wheelhouse/transformers-5.3.0.whl",
                    "datasets==4.8.2",
                    "numpy==1.26.4",
                    "scipy==1.11.4",
                    "pandas==2.1.4",
                    "pyarrow==23.0.1",
                    "pytest==7.4.4",
                    "ruff==0.12.5",
                    "basedpyright==1.31.1",
                ]
            ),
            "must use an exact '==' pin for transformers",
        ),
        (
            _runtime_command_runner(
                freeze_lines=[
                    "torch==2.7.0",
                    "transformers==5.3.0",
                    "numpy==1.26.4",
                    "scipy==1.11.4",
                    "pandas==2.1.4",
                    "pyarrow==23.0.1",
                    "pytest==7.4.4",
                    "ruff==0.12.5",
                    "basedpyright==1.31.1",
                ]
            ),
            "missing required packages: datasets",
        ),
        (
            _runtime_command_runner(gpu_lines=["0, GPU-0000, NVIDIA L40, malformed"]),
            "Malformed nvidia-smi inventory row",
        ),
    ),
)
def test_materialize_data_fails_closed_on_runtime_command_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    runtime_runner: RuntimeCommandRunner,
    match: str,
) -> None:
    dataset = _dataset_fixture("wikipedia")

    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return list(range(512)) if text else []

    fake_domain = MaterializedDomain(
        schema_version=1,
        domain="wikipedia",
        dataset_repository=dataset["repository"],
        dataset_revision=dataset["revision"],
        dataset_config=dataset["config"],
        dataset_split=dataset["split"],
        dataset_field=dataset["field"],
        dataset_fingerprint="fp",
        tokenizer_name="llama-3.1-8b",
        tokenizer_tree_sha256="tok",
        fit_documents=(),
        eval_documents=(),
        fit_chunks=(),
        eval_chunks=(),
        manifest_sha256="manifest",
    )
    monkeypatch.setattr(
        "si_rebuttal.runner.materialize_frozen_domain", _return_constant(fake_domain)
    )
    monkeypatch.setattr("si_rebuttal.runner.capture_git_binding", _git_binding_stub)
    monkeypatch.setattr(
        "si_rebuttal.runner._reconstruct_sequences_for_materialized_domain",
        _empty_reconstruct_sequences_stub(fake_domain),
    )

    load_dataset = _dataset_loader(FakeLoadedDataset(rows=(), column_names=("text",)))
    load_tokenizer = _tokenizer_loader(FakeTokenizer())

    with pytest.raises(Exception, match=match):
        materialize_data(
            "rebuttal/configs/base.toml",
            "rebuttal/configs/sweep.toml",
            run_root=tmp_path / "materialize",
            load_dataset_fn=load_dataset,
            tokenizer_loader=load_tokenizer,
            runtime_command_runner=runtime_runner,
        )


@pytest.mark.parametrize(
    ("tamper_path", "tamper_value", "match"),
    (
        (
            ("runtime_capture", "package_freeze", "output_lines"),
            ["torch==9.9.9"],
            "runtime capture drifted|package freeze output lines drifted",
        ),
        (
            ("runtime_capture", "gpu_inventory", "gpus"),
            [
                {
                    "physical_index": 0,
                    "uuid": "GPU-0000",
                    "name": "NVIDIA L40",
                    "total_memory_mib": 46068,
                    "compute_capability": "9.9",
                    "driver_version": "570.00",
                }
            ],
            "runtime capture drifted",
        ),
    ),
)
def test_materialize_data_reuse_rejects_tampered_runtime_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tamper_path: tuple[str, ...],
    tamper_value: object,
    match: str,
) -> None:
    dataset = _dataset_fixture("wikipedia")

    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return list(range(512)) if text else []

    def fake_document(document_id: str, partition: str, row_index: int) -> DocumentRecord:
        text = f"{partition} document {row_index}"
        return DocumentRecord(
            domain="wikipedia",
            document_id=document_id,
            row_indices=(row_index,),
            text=text,
            content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            assignment_sha256=hashlib.sha256(f"{partition}:{document_id}".encode()).hexdigest(),
            partition=partition,
        )

    fake_domain = MaterializedDomain(
        schema_version=1,
        domain="wikipedia",
        dataset_repository=dataset["repository"],
        dataset_revision=dataset["revision"],
        dataset_config=dataset["config"],
        dataset_split=dataset["split"],
        dataset_field=dataset["field"],
        dataset_fingerprint="fp",
        tokenizer_name="llama-3.1-8b",
        tokenizer_tree_sha256="tok",
        fit_documents=tuple(fake_document(f"doc-fit-{i}", "fit", i) for i in range(50)),
        eval_documents=tuple(fake_document(f"doc-eval-{i}", "eval", 100 + i) for i in range(100)),
        fit_chunks=tuple(
            SelectedChunk(
                document_id=f"doc-fit-{i}",
                partition="fit",
                chunk_index=0,
                source_rows=(i,),
                token_count=512,
                token_shape=(512,),
                token_sha256=token_sha256(list(range(512))),
            )
            for i in range(50)
        ),
        eval_chunks=tuple(
            SelectedChunk(
                document_id=f"doc-eval-{i}",
                partition="eval",
                chunk_index=0,
                source_rows=(100 + i,),
                token_count=512,
                token_shape=(512,),
                token_sha256=token_sha256(list(range(512))),
            )
            for i in range(100)
        ),
        manifest_sha256="manifest",
    )
    _assert_selected_chunk_provenance(fake_domain)
    monkeypatch.setattr(
        "si_rebuttal.runner.materialize_frozen_domain", _return_constant(fake_domain)
    )
    monkeypatch.setattr("si_rebuttal.runner.capture_git_binding", _git_binding_stub)

    def reconstruct_sequences(
        *args: object, **kwargs: object
    ) -> tuple[tuple[Int64Array, ...], tuple[Int64Array, ...], MaterializedDomain]:
        del args, kwargs
        return (_repeated_sequences(50), _repeated_sequences(100), fake_domain)

    monkeypatch.setattr(
        "si_rebuttal.runner._reconstruct_sequences_for_materialized_domain",
        reconstruct_sequences,
    )
    load_dataset = _dataset_loader(
        FakeLoadedDataset(
            rows=tuple(_iter_fake_dataset_rows()),
            column_names=("text", "repository_name", "whole_func_string"),
        )
    )
    load_tokenizer = _tokenizer_loader(FakeTokenizer())

    run_root = tmp_path / "materialize"
    runtime_runner = _runtime_command_runner()
    materialize_data(
        "rebuttal/configs/base.toml",
        "rebuttal/configs/sweep.toml",
        run_root=run_root,
        load_dataset_fn=load_dataset,
        tokenizer_loader=load_tokenizer,
        runtime_command_runner=runtime_runner,
    )
    batch_path = run_root / "manifests" / "batch-manifest.json"
    tampered = _read_json_object(batch_path)
    _set_json_path_value(tampered, tamper_path, cast(JsonValue, tamper_value))
    _overwrite_json(batch_path, tampered)

    with pytest.raises(Exception, match=match):
        materialize_data(
            "rebuttal/configs/base.toml",
            "rebuttal/configs/sweep.toml",
            run_root=run_root,
            load_dataset_fn=_raising_dataset_loader("loader must not run after tamper"),
            tokenizer_loader=_raising_tokenizer_loader("tokenizer must not run after tamper"),
            runtime_command_runner=runtime_runner,
        )


def test_toy_smoke_writes_terminal_outputs(tmp_path: Path) -> None:
    run_root = tmp_path / "toy-run"
    first = toy_smoke(
        "rebuttal/configs/base.toml",
        "rebuttal/configs/sweep.toml",
        run_root=run_root,
    )
    first_payload = _read_json_object(first.terminal_summary_path)
    second = toy_smoke(
        "rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml", run_root=run_root
    )
    second_payload = _read_json_object(second.terminal_summary_path)

    assert first.terminal_summary_path.name == "batch-terminal-summary.v2.json"
    assert second.terminal_summary_path == first.terminal_summary_path
    assert first_payload["schema_version"] == 2
    assert second_payload == first_payload
    assert first_payload["shards_verified"] == first.shard_count == second.shard_count


def test_validate_model_requires_real_intervention_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    records_by_kernel: dict[str, FakeValidationRecord] = {}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    run_root = tmp_path / "run"
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id=None)
    root.ensure()
    validate_config("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")
    batch_manifest = _batch_manifest_payload()
    load_existing_or_write(root.manifests_dir / "batch-manifest.json", batch_manifest)

    class FakeModel:
        def __call__(self, **kwargs: object) -> FakeForwardOutput:
            del kwargs
            calls.append("forward")
            return FakeForwardOutput(logits=torch.zeros((1, 64, 1024), dtype=torch.float32))

    class FakeIntervention:
        def __init__(self, kernel_kind: str) -> None:
            self.kernel_kind = kernel_kind
            self.record = _make_fake_validation_record(
                kernel_kind=kernel_kind,
                selected_head=0,
            )
            records_by_kernel[kernel_kind] = self.record

        def __enter__(self) -> FakeIntervention:
            calls.append(f"enter:{self.kernel_kind}")
            return self

        def __exit__(self, exc_type: object, exc: object, exc_tb: object) -> None:
            del exc_type, exc, exc_tb
            return None

        def latest_validation_record(self) -> FakeValidationRecord:
            return self.record

        def build_layer_correction(self, **kwargs: object) -> torch.Tensor:
            del kwargs
            return torch.zeros((32, 64, 64), dtype=torch.float32)

    def factory(model: object, **kwargs: object) -> FakeIntervention:
        del model
        kernel = _kernel_values_for_head(kwargs.get("kernels_by_layer_head"), (0, 0))
        if np.allclose(kernel, 0.0):
            kind = "zero"
        elif np.allclose(kernel, kernel[0]):
            kind = "constant"
        else:
            kind = "nonconstant"
        calls.append(f"factory:{kind}")
        return FakeIntervention(kind)

    bundle = FakeValidationBundle(
        entry=FakeBundleEntry(name="llama-3.1-8b", intervention_factory=factory),
        binding=FakeBinding(
            weights_tree_sha256=batch_manifest["models"]["llama-3.1-8b"],
            tokenizer_tree_sha256=batch_manifest["tokenizers"]["llama-3.1-8b"],
        ),
        model=FakeModel(),
        logical_device=torch.device("cpu"),
    )
    monkeypatch.setattr("si_rebuttal.runner.load_local_model_bundle", _return_constant(bundle))
    monkeypatch.setattr(
        "si_rebuttal.runner._load_sequences_from_run_root",
        _return_constant(
            DomainSequences(
                fit_tokens=(np.arange(512, dtype=np.int64),),
                eval_tokens=(np.arange(512, dtype=np.int64),),
            )
        ),
    )
    monkeypatch.setattr(
        "si_rebuttal.runner.materialize_frozen_domain",
        _raise_assertion_callable("frozen loader should not run after materialization"),
    )
    payload = validate_model(
        "rebuttal/configs/base.toml",
        "rebuttal/configs/sweep.toml",
        model_name="llama-3.1-8b",
        run_root=run_root,
    )
    assert payload["identity_sha256"]
    assert payload["batch_manifest_sha256"] == batch_manifest["identity_sha256"]
    assert payload["weights_tree_sha256"] == batch_manifest["models"]["llama-3.1-8b"]
    assert payload["tokenizer_tree_sha256"] == batch_manifest["tokenizers"]["llama-3.1-8b"]
    assert calls[:3] == ["forward", "factory:zero", "enter:zero"]
    assert "factory:nonconstant" in calls
    assert records_by_kernel["zero"].runtime_call_log == []
    assert records_by_kernel["constant"].runtime_call_log == []
    assert records_by_kernel["nonconstant"].runtime_call_log == [
        ((1, 32, 64, 2), (1, 8, 64, 2), (1, 32, 64, 64), 0.625)
    ]


def test_benchmark_model_measures_real_components(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    run_root = tmp_path / "run"
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id=None)
    root.ensure()
    resolved = _resolved_digests(
        validate_config("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")
    )
    batch_manifest = _batch_manifest_payload()
    load_existing_or_write(root.manifests_dir / "batch-manifest.json", batch_manifest)
    _write_batch_supporting_manifests(
        root=root,
        batch_manifest=batch_manifest,
        config_sha256=resolved["config_sha256"],
        sweep_sha256=resolved["sweep_sha256"],
    )
    load_existing_or_write(
        root.receipts_dir / "validation.llama-3.1-8b.json",
        _validation_receipt_payload(
            model_name="llama-3.1-8b",
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
        ),
    )
    monkeypatch.setattr(
        "si_rebuttal.runner.load_local_model_bundle",
        _return_constant(
            FakeLoadedBundle(
                binding=FakeBinding("w", "t"),
                model=object(),
                tokenizer=object(),
                adapter=None,
                logical_device=torch.device("cpu"),
            )
        ),
    )
    monkeypatch.setattr(
        "si_rebuttal.runner._load_sequences_from_run_root",
        _return_constant(
            DomainSequences(
                fit_tokens=tuple(np.arange(512, dtype=np.int64) for _ in range(2)),
                eval_tokens=tuple(np.arange(512, dtype=np.int64) for _ in range(2)),
            )
        ),
    )
    monkeypatch.setattr(
        "si_rebuttal.runner._capture_fit_summary",
        _return_constant(
            FakeCaptureSummary(
                mean_scores=np.zeros((32, 32), dtype=np.float64),
                raw_kernel_sums=np.ones((32, 32, 512), dtype=np.float64),
                sequence_count=1,
            )
        ),
    )
    monkeypatch.setattr(
        "si_rebuttal.runner.sort_heads_into_bins",
        _return_constant([[FakeHead(layer=0, head=0)] for _ in range(20)]),
    )
    eval_calls: list[int] = []
    capture_calls: list[int] = []

    monkeypatch.setattr(
        "si_rebuttal.runner._capture_fit_summary",
        CaptureSummaryRecorder(calls=capture_calls, sequence_count_floor=1),
    )
    monkeypatch.setattr("si_rebuttal.runner._eval_nlls", EvalNllRecorder(calls=eval_calls))
    monkeypatch.setattr(
        "si_rebuttal.runner.materialize_frozen_domain",
        _raise_assertion_callable("benchmark must reuse frozen manifests"),
    )

    def next_tick() -> float:
        return next(tick)

    tick = iter([0.0, 1.0, 1.5, 2.0, 2.5, 3.5, 4.0, 5.0, 5.5, 6.5, 7.0, 8.0])
    report = benchmark_model(
        "rebuttal/configs/base.toml",
        "rebuttal/configs/sweep.toml",
        model_name="llama-3.1-8b",
        run_root=run_root,
        time_fn=next_tick,
    )
    assert report.projected_total_seconds > 0.0
    assert set(report.measured_seconds) == {
        "fit_capture_kernel_r2",
        "baseline_eval",
        "source_bin_eval",
        "target_bin_eval",
        "control_eval",
        "depth_eval",
    }
    assert report.measured_counts == {
        "fit_capture_kernel_r2": 2,
        "baseline_eval": 4,
        "source_bin_eval": 4,
        "target_bin_eval": 4,
        "control_eval": 24,
        "depth_eval": 2,
    }
    assert capture_calls == [1, 1]
    assert eval_calls == [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    projection_components = _json_object(report.projection_components)
    inventory = _json_object_item(projection_components, "inventory")
    assert _json_int_item(_json_object_item(inventory, "token_manifests"), "count") == 12
    assert _json_int_item(_json_object_item(inventory, "token_manifests"), "exact_bytes") > 0
    assert _json_int_item(_json_object_item(inventory, "fit_profiles"), "count") == 6
    assert _json_int_item(_json_object_item(inventory, "launch_gpu_receipts"), "count") == 2
    assert _json_int_item(_json_object_item(inventory, "launch_logs"), "count") == 2
    assert report.runtime_formula == BENCHMARK_RUNTIME_FORMULA
    assert report.formula == BENCHMARK_ARTIFACT_FORMULA


@pytest.mark.parametrize(
    ("model_name", "cuda_visible_devices"),
    (
        ("llama-3.1-8b", "0"),
        ("olmo-2-7b", "0"),
        ("mistral-7b-v0.1", "1"),
    ),
)
def test_validate_model_requires_fixed_single_gpu_lane_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_name: str,
    cuda_visible_devices: str,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cuda_visible_devices)
    run_root = tmp_path / "run"
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id=None)
    root.ensure()
    batch_manifest = _batch_manifest_payload()
    load_existing_or_write(root.manifests_dir / "batch-manifest.json", batch_manifest)
    monkeypatch.setattr(
        "si_rebuttal.runner._load_sequences_from_run_root",
        FixedDomainSequenceLoader(
            DomainSequences(
                fit_tokens=(np.arange(512, dtype=np.int64),),
                eval_tokens=(np.arange(512, dtype=np.int64),),
            )
        ),
    )

    def fake_load_local_model_bundle(
        resolved_config: object,
        *,
        model_name: str,
        local_files_only: bool = True,
        logical_device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.bfloat16,
        auto_model_loader: object | None = None,
        auto_tokenizer_loader: object | None = None,
    ) -> FakeLoadedBundle:
        del (
            resolved_config,
            model_name,
            local_files_only,
            torch_dtype,
            auto_model_loader,
            auto_tokenizer_loader,
        )
        assert logical_device == "cuda:0"
        raise RuntimeError("stop after placement check")

    monkeypatch.setattr("si_rebuttal.runner.load_local_model_bundle", fake_load_local_model_bundle)
    with pytest.raises(RuntimeError, match="stop after placement check"):
        validate_model(
            "rebuttal/configs/base.toml",
            "rebuttal/configs/sweep.toml",
            model_name=model_name,
            run_root=run_root,
        )


@pytest.mark.parametrize(
    ("model_name", "cuda_visible_devices"),
    (
        ("llama-3.1-8b", "1"),
        ("olmo-2-7b", ""),
        ("mistral-7b-v0.1", "0"),
    ),
)
def test_validate_model_rejects_wrong_fixed_gpu_lane_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_name: str,
    cuda_visible_devices: str,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cuda_visible_devices)
    run_root = tmp_path / "run"
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id=None)
    root.ensure()
    batch_manifest = _batch_manifest_payload()
    load_existing_or_write(root.manifests_dir / "batch-manifest.json", batch_manifest)
    monkeypatch.setattr(
        "si_rebuttal.runner._load_sequences_from_run_root",
        FixedDomainSequenceLoader(
            DomainSequences(
                fit_tokens=(np.arange(512, dtype=np.int64),),
                eval_tokens=(np.arange(512, dtype=np.int64),),
            )
        ),
    )
    monkeypatch.setattr(
        "si_rebuttal.runner.load_local_model_bundle",
        _raise_model_load_assertion("model load must not run"),
    )
    with pytest.raises(Exception, match=rf"{model_name} requires CUDA_VISIBLE_DEVICES="):
        validate_model(
            "rebuttal/configs/base.toml",
            "rebuttal/configs/sweep.toml",
            model_name=model_name,
            run_root=run_root,
        )


@pytest.mark.parametrize("preseed_scope", [None, "global", "local"])
def test_stream_post_rope_qk_restores_pinned_registry_and_marker_on_success(
    monkeypatch: pytest.MonkeyPatch,
    preseed_scope: str | None,
) -> None:
    bundle = _stream_runner_bundle()
    capture_time_ns = 123_456
    capture_key = _capture_key_for_bundle(bundle, time_ns=capture_time_ns)
    previous_value = object()
    attention_registry = _make_runner_pinned_general_interface(
        initial_global={capture_key: previous_value} if preseed_scope == "global" else None,
        initial_local={capture_key: previous_value} if preseed_scope == "local" else None,
    )
    eager_mask_calls: list[tuple[int, tuple[int, ...]]] = []

    def eager_mask_handler(*, layer_index: int, query: torch.Tensor, **_: object) -> torch.Tensor:
        eager_mask_calls.append((layer_index, tuple(query.shape)))
        return _fake_validation_full_mask(
            kernel_kind="zero", selected_head=0, num_heads=int(query.shape[1])
        )

    mask_registry = _make_runner_pinned_general_interface(
        initial_global=(
            {"eager": eager_mask_handler, capture_key: previous_value}
            if preseed_scope == "global"
            else {"eager": eager_mask_handler}
        ),
        initial_local={capture_key: previous_value} if preseed_scope == "local" else None,
    )
    attention_before = attention_registry.snapshot()
    mask_before = mask_registry.snapshot()
    seen_masks: list[torch.Tensor] = []
    seen_mask_handlers: list[object] = []
    modules = _install_stream_post_rope_qk_harness(
        monkeypatch,
        bundle=bundle,
        attention_registry=attention_registry,
        mask_registry=mask_registry,
        time_ns=capture_time_ns,
        seen_mask_handlers=seen_mask_handlers,
        seen_masks=seen_masks,
    )
    seen: list[int] = []

    _stream_post_rope_qk(
        cast(LoadedModelBundle, cast(object, bundle)),
        torch.arange(64, dtype=torch.long),
        consumer=lambda layer_index, logits: seen.append(layer_index),
    )

    assert seen == [0, 1]
    assert attention_registry.snapshot() == attention_before
    assert mask_registry.snapshot() == mask_before
    assert cast(_StreamRunnerModel, bundle.model).config.marker == "original-marker"
    assert cast(_StreamRunnerModel, bundle.model).config.writes == [
        capture_key,
        "original-marker",
    ]
    assert eager_mask_calls == [
        (0, (1, 2, 64, 2)),
        (1, (1, 2, 64, 2)),
    ]
    assert seen_mask_handlers == [eager_mask_handler]
    assert len(seen_masks) == 2
    assert all(mask.ndim == 4 and mask.shape == (1, 2, 64, 64) for mask in seen_masks)
    assert torch.equal(
        seen_masks[0],
        _fake_validation_full_mask(kernel_kind="zero", selected_head=0, num_heads=2),
    )
    assert len(modules) == 2


@pytest.mark.parametrize(
    ("error_message", "error_origin"),
    (
        ("model forward boom", "model"),
        ("consumer boom", "consumer"),
    ),
)
@pytest.mark.parametrize("preseed_scope", [None, "global", "local"])
def test_stream_post_rope_qk_restores_pinned_registry_and_marker_on_errors(
    monkeypatch: pytest.MonkeyPatch,
    preseed_scope: str | None,
    error_message: str,
    error_origin: str,
) -> None:
    bundle = _stream_runner_bundle()
    capture_time_ns = 789_012
    capture_key = _capture_key_for_bundle(bundle, time_ns=capture_time_ns)
    previous_value = object()
    attention_registry = _make_runner_pinned_general_interface(
        initial_global={capture_key: previous_value} if preseed_scope == "global" else None,
        initial_local={capture_key: previous_value} if preseed_scope == "local" else None,
    )

    def eager_mask_handler(*, query: torch.Tensor, **_: object) -> torch.Tensor:
        return _fake_validation_full_mask(
            kernel_kind="zero", selected_head=0, num_heads=int(query.shape[1])
        )

    mask_registry = _make_runner_pinned_general_interface(
        initial_global=(
            {"eager": eager_mask_handler, capture_key: previous_value}
            if preseed_scope == "global"
            else {"eager": eager_mask_handler}
        ),
        initial_local={capture_key: previous_value} if preseed_scope == "local" else None,
    )
    attention_before = attention_registry.snapshot()
    mask_before = mask_registry.snapshot()
    _install_stream_post_rope_qk_harness(
        monkeypatch,
        bundle=bundle,
        attention_registry=attention_registry,
        mask_registry=mask_registry,
        time_ns=capture_time_ns,
        model_forward_error=error_message if error_origin == "model" else None,
    )

    def consumer(layer_index: int, logits: torch.Tensor) -> None:
        del logits
        if error_origin == "consumer" and layer_index == 0:
            raise RuntimeError(error_message)

    with pytest.raises(RuntimeError, match=error_message):
        _stream_post_rope_qk(
            cast(LoadedModelBundle, cast(object, bundle)),
            torch.arange(64, dtype=torch.long),
            consumer=consumer,
        )

    assert attention_registry.snapshot() == attention_before
    assert mask_registry.snapshot() == mask_before
    assert cast(_StreamRunnerModel, bundle.model).config.marker == "original-marker"
    assert cast(_StreamRunnerModel, bundle.model).config.writes == [
        capture_key,
        "original-marker",
    ]


def test_stream_post_rope_qk_restores_marker_when_registry_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _stream_runner_bundle()
    capture_time_ns = 246_810
    capture_key = _capture_key_for_bundle(bundle, time_ns=capture_time_ns)
    attention_registry = _make_runner_pinned_general_interface()

    def eager_mask_handler(*, query: torch.Tensor, **_: object) -> torch.Tensor:
        return _fake_validation_full_mask(
            kernel_kind="zero", selected_head=0, num_heads=int(query.shape[1])
        )

    mask_registry = _make_runner_pinned_general_interface(
        initial_global={"eager": eager_mask_handler}
    )
    _install_stream_post_rope_qk_harness(
        monkeypatch,
        bundle=bundle,
        attention_registry=attention_registry,
        mask_registry=mask_registry,
        time_ns=capture_time_ns,
        poison_registry_before_return=True,
    )

    with pytest.raises(
        ExceptionGroup,
        match=r"Fit capture cleanup failed after attention interface installation\.",
    ) as excinfo:
        _stream_post_rope_qk(
            cast(LoadedModelBundle, cast(object, bundle)),
            torch.arange(64, dtype=torch.long),
            consumer=lambda layer_index, logits: None,
        )

    cleanup_messages = [str(error) for error in excinfo.value.exceptions]
    assert cleanup_messages == ["registry cleanup boom", "registry cleanup boom"]
    assert cast(_StreamRunnerModel, bundle.model).config.marker == "original-marker"
    assert cast(_StreamRunnerModel, bundle.model).config.writes == [
        capture_key,
        "original-marker",
    ]


def test_stream_post_rope_qk_preserves_primary_failure_when_cleanup_also_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _stream_runner_bundle()
    capture_time_ns = 975_318
    attention_registry = _make_runner_pinned_general_interface()

    def eager_mask_handler(*, query: torch.Tensor, **_: object) -> torch.Tensor:
        return _fake_validation_full_mask(
            kernel_kind="zero", selected_head=0, num_heads=int(query.shape[1])
        )

    mask_registry = _make_runner_pinned_general_interface(
        initial_global={"eager": eager_mask_handler}
    )
    _install_stream_post_rope_qk_harness(
        monkeypatch,
        bundle=bundle,
        attention_registry=attention_registry,
        mask_registry=mask_registry,
        time_ns=capture_time_ns,
        model_forward_error="model forward boom",
        poison_registry_before_return=True,
    )

    with pytest.raises(
        ExceptionGroup, match=r"Fit capture failed and cleanup also failed\."
    ) as excinfo:
        _stream_post_rope_qk(
            cast(LoadedModelBundle, cast(object, bundle)),
            torch.arange(64, dtype=torch.long),
            consumer=lambda layer_index, logits: None,
        )

    messages = [str(error) for error in excinfo.value.exceptions]
    assert messages == [
        "model forward boom",
        "registry cleanup boom",
        "registry cleanup boom",
    ]
    assert cast(_StreamRunnerModel, bundle.model).config.marker == "original-marker"


@pytest.mark.parametrize(
    ("model_name", "cuda_visible_devices"),
    (
        ("llama-3.1-8b", "0"),
        ("olmo-2-7b", "0"),
        ("mistral-7b-v0.1", "1"),
    ),
)
def test_benchmark_model_requires_fixed_single_gpu_lane_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_name: str,
    cuda_visible_devices: str,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cuda_visible_devices)
    run_root = tmp_path / "run"
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id=None)
    root.ensure()
    batch_manifest = _batch_manifest_payload()
    load_existing_or_write(root.manifests_dir / "batch-manifest.json", batch_manifest)
    load_existing_or_write(
        root.receipts_dir / f"validation.{model_name}.json",
        _validation_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=batch_manifest["config_sha256"],
            sweep_sha256=batch_manifest["sweep_sha256"],
        ),
    )

    def fake_load_local_model_bundle(
        resolved_config: object,
        *,
        model_name: str,
        local_files_only: bool = True,
        logical_device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.bfloat16,
        auto_model_loader: object | None = None,
        auto_tokenizer_loader: object | None = None,
    ) -> FakeLoadedBundle:
        del (
            resolved_config,
            model_name,
            local_files_only,
            torch_dtype,
            auto_model_loader,
            auto_tokenizer_loader,
        )
        assert logical_device == "cuda:0"
        raise RuntimeError("stop after placement check")

    monkeypatch.setattr("si_rebuttal.runner.load_local_model_bundle", fake_load_local_model_bundle)
    with pytest.raises(RuntimeError, match="stop after placement check"):
        benchmark_model(
            "rebuttal/configs/base.toml",
            "rebuttal/configs/sweep.toml",
            model_name=model_name,
            run_root=run_root,
        )


@pytest.mark.parametrize(
    ("model_name", "cuda_visible_devices"),
    (
        ("llama-3.1-8b", "1"),
        ("olmo-2-7b", "0,1"),
        ("mistral-7b-v0.1", ""),
    ),
)
def test_benchmark_model_rejects_wrong_fixed_gpu_lane_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_name: str,
    cuda_visible_devices: str,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cuda_visible_devices)
    run_root = tmp_path / "run"
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id=None)
    root.ensure()
    batch_manifest = _batch_manifest_payload()
    load_existing_or_write(root.manifests_dir / "batch-manifest.json", batch_manifest)
    load_existing_or_write(
        root.receipts_dir / f"validation.{model_name}.json",
        _validation_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=batch_manifest["config_sha256"],
            sweep_sha256=batch_manifest["sweep_sha256"],
        ),
    )
    monkeypatch.setattr(
        "si_rebuttal.runner.load_local_model_bundle",
        _raise_model_load_assertion("model load must not run"),
    )
    with pytest.raises(Exception, match=rf"{model_name} requires CUDA_VISIBLE_DEVICES="):
        benchmark_model(
            "rebuttal/configs/base.toml",
            "rebuttal/configs/sweep.toml",
            model_name=model_name,
            run_root=run_root,
        )


def test_lane_admission_sums_gpu0_and_gpu1_and_checks_disk(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id=None)
    root.ensure()
    resolved = _resolved_digests(
        validate_config("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")
    )
    batch_manifest = _batch_manifest_payload()
    load_existing_or_write(root.manifests_dir / "batch-manifest.json", batch_manifest)
    _write_batch_supporting_manifests(
        root=root,
        batch_manifest=batch_manifest,
        config_sha256=resolved["config_sha256"],
        sweep_sha256=resolved["sweep_sha256"],
    )
    projected_artifact_bytes, artifact_components = _artifact_projection_for_test_root(
        run_root, "llama-3.1-8b"
    )
    validation_receipts = {
        model_name: _validation_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
        )
        for model_name in ("llama-3.1-8b", "mistral-7b-v0.1", "olmo-2-7b")
    }
    benchmark_receipts = {
        "llama-3.1-8b": _benchmark_receipt_payload(
            model_name="llama-3.1-8b",
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            validation_receipt_sha256=_json_identity(validation_receipts["llama-3.1-8b"]),
            lane_hours=40.0,
            projected_artifact_bytes=projected_artifact_bytes,
            artifact_components=artifact_components,
        ),
        "olmo-2-7b": _benchmark_receipt_payload(
            model_name="olmo-2-7b",
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            validation_receipt_sha256=_json_identity(validation_receipts["olmo-2-7b"]),
            lane_hours=50.0,
            projected_artifact_bytes=projected_artifact_bytes,
            artifact_components=artifact_components,
        ),
        "mistral-7b-v0.1": _benchmark_receipt_payload(
            model_name="mistral-7b-v0.1",
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            validation_receipt_sha256=_json_identity(validation_receipts["mistral-7b-v0.1"]),
            lane_hours=60.0,
            projected_artifact_bytes=projected_artifact_bytes,
            artifact_components=artifact_components,
        ),
    }
    for model_name, payload in validation_receipts.items():
        load_existing_or_write(root.receipts_dir / f"validation.{model_name}.json", payload)
    for model_name, payload in benchmark_receipts.items():
        load_existing_or_write(root.receipts_dir / f"benchmark.{model_name}.json", payload)
    report = admit_lanes(
        "rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml", run_root=run_root
    )
    assert report.gpu0_hours == 90.0
    assert report.gpu1_hours == 60.0


def test_artifact_projection_inventory_is_model_independent(tmp_path: Path) -> None:
    run_root = tmp_path / "projection-root"
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id=None)
    root.ensure()
    resolved = _resolved_digests(
        validate_config("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")
    )
    batch_manifest = _batch_manifest_payload()
    load_existing_or_write(root.manifests_dir / "batch-manifest.json", batch_manifest)
    _write_batch_supporting_manifests(
        root=root,
        batch_manifest=batch_manifest,
        config_sha256=resolved["config_sha256"],
        sweep_sha256=resolved["sweep_sha256"],
    )
    first_bytes, first_components = _artifact_projection_for_test_root(run_root, "llama-3.1-8b")
    for model_name in ("mistral-7b-v0.1", "olmo-2-7b"):
        projected_bytes, components = _artifact_projection_for_test_root(run_root, model_name)
        assert projected_bytes == first_bytes
        assert components == first_components


def test_lane_admission_rejects_forged_low_lane_hours(tmp_path: Path) -> None:
    run_root = tmp_path / "lane-hours-root"
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id=None)
    root.ensure()
    resolved = _resolved_digests(
        validate_config("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")
    )
    batch_manifest = _batch_manifest_payload()
    load_existing_or_write(root.manifests_dir / "batch-manifest.json", batch_manifest)
    _write_batch_supporting_manifests(
        root=root,
        batch_manifest=batch_manifest,
        config_sha256=resolved["config_sha256"],
        sweep_sha256=resolved["sweep_sha256"],
    )
    projected_artifact_bytes, artifact_components = _artifact_projection_for_test_root(
        run_root, "llama-3.1-8b"
    )
    validation_receipts: dict[str, JsonObject] = {}
    benchmark_receipts: dict[str, JsonObject] = {}
    for model_name, lane_hours in (
        ("llama-3.1-8b", 40.0),
        ("olmo-2-7b", 50.0),
        ("mistral-7b-v0.1", 60.0),
    ):
        validation_payload = _validation_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
        )
        validation_receipts[model_name] = validation_payload
        benchmark_receipts[model_name] = _benchmark_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            validation_receipt_sha256=_json_identity(validation_payload),
            lane_hours=lane_hours,
            projected_artifact_bytes=projected_artifact_bytes,
            artifact_components=artifact_components,
        )
    benchmark_receipts["llama-3.1-8b"] = _benchmark_receipt_payload(
        model_name="llama-3.1-8b",
        batch_manifest=batch_manifest,
        config_sha256=resolved["config_sha256"],
        sweep_sha256=resolved["sweep_sha256"],
        validation_receipt_sha256=_json_identity(validation_receipts["llama-3.1-8b"]),
        lane_hours=40.0,
        projected_total_seconds=130.0 * 3600.0,
        stored_lane_hours=40.0,
        projected_artifact_bytes=projected_artifact_bytes,
        artifact_components=artifact_components,
    )
    for model_name, payload in validation_receipts.items():
        load_existing_or_write(root.receipts_dir / f"validation.{model_name}.json", payload)
    for model_name, payload in benchmark_receipts.items():
        load_existing_or_write(root.receipts_dir / f"benchmark.{model_name}.json", payload)
    with pytest.raises(Exception, match=r"Benchmark lane_hours drifted for llama-3\.1-8b"):
        admit_lanes("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml", run_root=run_root)


def test_atomic_no_clobber_and_resume_conflict(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    payload = {"schema_version": 1, "value": 1}
    _, written = load_existing_or_write(target, payload)
    assert written is True
    _, written_again = load_existing_or_write(target, payload)
    assert written_again is False
    with pytest.raises(ArtifactError, match="conflicts"):
        load_existing_or_write(target, {"schema_version": 1, "value": 2})


def test_finalize_batch_is_verify_only_for_existing_v1_and_rejects_missing_v1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, batch_manifest, run_root, root = _prepare_finalizer_run(
        monkeypatch, tmp_path, run_name="verify-only", write_v1_summary=True
    )
    expected_v1 = _write_expected_v1_summary(root, batch_manifest=batch_manifest)
    assert (
        finalize_batch(
            "rebuttal/configs/base.toml",
            "rebuttal/configs/sweep.toml",
            run_root=run_root,
        )
        == expected_v1
    )
    _summary_v1_path(root).unlink()
    with pytest.raises(
        Exception,
        match=r"(?i)legacy finalize-batch is verify-only and requires an existing v1 summary\.",
    ):
        finalize_batch(
            "rebuttal/configs/base.toml",
            "rebuttal/configs/sweep.toml",
            run_root=run_root,
        )


def test_refinalize_batch_v2_adds_only_v2_and_preserves_preexisting_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, _, run_root, root = _prepare_finalizer_run(
        monkeypatch, tmp_path, run_name="publish-v2", write_v1_summary=True
    )
    _patch_refinalize_source_summary_v1_digest(monkeypatch, root)
    before = _snapshot_file_tree(run_root)
    v1_bytes = _summary_v1_path(root).read_bytes()

    payload = _refinalize_batch_v2(run_root=run_root, source_summary_v1=_summary_v1_path(root))

    after = _snapshot_file_tree(run_root)
    assert payload["schema_version"] == 2
    assert _summary_v2_path(root).exists()
    assert _summary_v1_path(root).read_bytes() == v1_bytes
    assert set(after) - set(before) == {"summaries/batch-terminal-summary.v2.json"}
    assert {path: digest for path, digest in after.items() if path in before} == before


def test_refinalize_batch_v2_uses_frozen_inputs_without_current_git_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, _, run_root, root = _prepare_finalizer_run(
        monkeypatch, tmp_path, run_name="frozen-git-binding", write_v1_summary=True
    )
    _patch_refinalize_source_summary_v1_digest(monkeypatch, root)
    monkeypatch.setattr(
        "si_rebuttal.runner.capture_git_binding",
        _raise_assertion_callable("refinalize must not read the current source git binding"),
    )

    payload = _refinalize_batch_v2(run_root=run_root, source_summary_v1=_summary_v1_path(root))

    assert payload["schema_version"] == 2
    assert _summary_v2_path(root).exists()


def test_refinalize_batch_v2_immutable_manifest_covers_exact_22_manifests_and_1062_shards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, _, run_root, root = _prepare_finalizer_run(
        monkeypatch, tmp_path, run_name="coverage", write_v1_summary=True
    )
    _patch_refinalize_source_summary_v1_digest(monkeypatch, root)
    payload = _refinalize_batch_v2(run_root=run_root, source_summary_v1=_summary_v1_path(root))

    manifest_entries = sorted(path for path in root.manifests_dir.rglob("*") if path.is_file())
    shard_entries = sorted(path for path in root.shards_dir.glob("*.json"))
    assert len(manifest_entries) == 22
    assert len(shard_entries) == 1062
    assert (
        _json_int_item(_json_object_item(payload, "immutable_input_manifest"), "entry_count")
        == 1087
    )


def test_refinalize_batch_v2_rejects_noncanonical_source_summary_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, _, run_root, root = _prepare_finalizer_run(
        monkeypatch, tmp_path, run_name="canonical", write_v1_summary=True
    )
    _patch_refinalize_source_summary_v1_digest(monkeypatch, root)
    aliased_path = tmp_path / "aliased-v1.json"
    aliased_path.write_bytes(_summary_v1_path(root).read_bytes())

    with pytest.raises(Exception, match="canonical run-root v1 summary path"):
        _refinalize_batch_v2(run_root=run_root, source_summary_v1=aliased_path)


def test_refinalize_batch_v2_rejects_config_and_sweep_path_substitution_before_first_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, _, run_root, root = _prepare_finalizer_run(
        monkeypatch, tmp_path, run_name="path-substitution", write_v1_summary=True
    )
    _patch_refinalize_source_summary_v1_digest(monkeypatch, root)
    base_path = Path("rebuttal/configs/base.toml")
    sweep_path = Path("rebuttal/configs/sweep.toml")
    substituted_config = tmp_path / "base-substitution.toml"
    substituted_sweep = tmp_path / "sweep-substitution.toml"
    substituted_config.write_text(base_path.read_text(encoding="utf-8"), encoding="utf-8")
    substituted_sweep.write_text(sweep_path.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(
        Exception,
        match=(
            r"Config path must be the canonical production path "
            r"rebuttal/configs/base\.toml\."
        ),
    ):
        _refinalize_batch_v2(
            run_root=run_root,
            source_summary_v1=_summary_v1_path(root),
            base_path=substituted_config,
        )
    with pytest.raises(
        Exception,
        match=(
            r"Sweep path must be the canonical production path "
            r"rebuttal/configs/sweep\.toml\."
        ),
    ):
        _refinalize_batch_v2(
            run_root=run_root,
            source_summary_v1=_summary_v1_path(root),
            sweep_path=substituted_sweep,
        )
    assert not _summary_v2_path(root).exists()


def test_refinalize_batch_v2_rejects_same_path_config_byte_substitution_before_first_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, _, run_root, root = _prepare_finalizer_run(
        monkeypatch, tmp_path, run_name="same-path-substitution", write_v1_summary=True
    )
    _patch_refinalize_source_summary_v1_digest(monkeypatch, root)
    canonical_config = Path("rebuttal/configs/base.toml").resolve()
    original_read_bytes = Path.read_bytes
    publish_calls = 0

    def drifted_read_bytes(self: Path) -> bytes:
        data = original_read_bytes(self)
        if self.resolve() == canonical_config:
            return data.replace(
                b'\ntorch_dtype = "bfloat16"\n',
                b'\ntorch_dtype = "bfloat16"  # drifted\n',
                1,
            )
        return data

    def fail_if_publish_attempted(path: str | Path, payload_bytes: bytes) -> None:
        nonlocal publish_calls
        publish_calls += 1
        assert path == _summary_v2_path(root)
        assert isinstance(payload_bytes, bytes)
        raise AssertionError("summary v2 publication must not start after same-path config drift")

    monkeypatch.setattr(Path, "read_bytes", drifted_read_bytes)
    monkeypatch.setattr(
        runner_module, "atomic_write_bytes_no_clobber", fail_if_publish_attempted, raising=False
    )
    with pytest.raises(
        Exception,
        match=r"Canonical production config raw byte digest drifted before v2 finalization\.",
    ):
        _refinalize_batch_v2(run_root=run_root, source_summary_v1=_summary_v1_path(root))
    assert publish_calls == 0
    assert not _summary_v2_path(root).exists()


@pytest.mark.parametrize(
    ("label", "mutate_file", "match"),
    _IMMUTABLE_INPUT_MUTATION_CASES,
)
def test_refinalize_batch_v2_rejects_identity_valid_immutable_input_mutations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    label: str,
    mutate_file: Callable[[RunRoot], object],
    match: str,
) -> None:
    _, _, run_root, root = _prepare_finalizer_run(
        monkeypatch,
        tmp_path,
        run_name=f"immutable-input-{label}",
        write_v1_summary=True,
    )
    _patch_refinalize_source_summary_v1_digest(monkeypatch, root)
    publish_calls = 0

    def fail_if_publish_attempted(path: str | Path, payload_bytes: bytes) -> None:
        nonlocal publish_calls
        publish_calls += 1
        assert path == _summary_v2_path(root)
        assert isinstance(payload_bytes, bytes)
        raise AssertionError(f"summary v2 publication must not start after {label} drift")

    mutate_file(root)
    monkeypatch.setattr(
        runner_module, "atomic_write_bytes_no_clobber", fail_if_publish_attempted, raising=False
    )
    with pytest.raises(Exception, match=match):
        _refinalize_batch_v2(run_root=run_root, source_summary_v1=_summary_v1_path(root))
    assert publish_calls == 0
    assert not _summary_v2_path(root).exists()


@pytest.mark.parametrize(
    ("mutate", "match"),
    _MANIFEST_INPUT_MUTATION_CASES,
)
def test_refinalize_batch_v2_rejects_missing_additional_and_non_json_manifest_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutate: Callable[[RunRoot], object],
    match: str,
) -> None:
    _, _, run_root, root = _prepare_finalizer_run(
        monkeypatch, tmp_path, run_name="manifest-errors", write_v1_summary=True
    )
    _patch_refinalize_source_summary_v1_digest(monkeypatch, root)
    mutate(root)
    with pytest.raises(Exception, match=match):
        _refinalize_batch_v2(run_root=run_root, source_summary_v1=_summary_v1_path(root))
    assert not _summary_v2_path(root).exists()


@pytest.mark.parametrize("seam_target", ("manifest", "shard", "config", "sweep", "v1"))
def test_refinalize_batch_v2_rechecks_semantic_inputs_in_last_prepublish_window_and_writes_no_v2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, seam_target: str
) -> None:
    _, _, run_root, root = _prepare_finalizer_run(
        monkeypatch,
        tmp_path,
        run_name=f"last-window-{seam_target}",
        write_v1_summary=True,
    )
    _patch_refinalize_source_summary_v1_digest(monkeypatch, root)
    original_immutable_manifest = (
        runner_module._immutable_input_manifest  # pyright: ignore[reportPrivateUsage]
    )
    immutable_manifest_calls = 0
    publish_calls = 0
    canonical_target = {
        "config": Path("rebuttal/configs/base.toml").resolve(),
        "sweep": Path("rebuttal/configs/sweep.toml").resolve(),
        "v1": _summary_v1_path(root).resolve(),
    }

    def rebuild_with_late_input(*args: object, **kwargs: object) -> object:
        nonlocal immutable_manifest_calls
        immutable_manifest_calls += 1
        if immutable_manifest_calls == 2:
            if seam_target == "manifest":
                _overwrite_json(
                    root.manifests_dir / "extra-manifest.json",
                    _json_object({"schema_version": 1, "marker": "late-manifest"}),
                )
            elif seam_target == "shard":
                _overwrite_json(
                    root.shards_dir / "late.extra.bin.0.baseline.trial-0.json",
                    _json_object({"schema_version": 1, "marker": "late-shard"}),
                )
            else:
                original_read_bytes = Path.read_bytes
                target_path = canonical_target[seam_target]

                def reread_with_late_bytes(self: Path) -> bytes:
                    data = original_read_bytes(self)
                    if self.resolve() != target_path:
                        return data
                    if seam_target == "v1":
                        return data + b"\n"
                    return data + b"\n# late-drift\n"

                with pytest.MonkeyPatch.context() as local_patch:
                    local_patch.setattr(Path, "read_bytes", reread_with_late_bytes)
                    return original_immutable_manifest(*args, **kwargs)
        return original_immutable_manifest(*args, **kwargs)

    def fail_if_publish_attempted(path: str | Path, payload_bytes: bytes) -> None:
        nonlocal publish_calls
        publish_calls += 1
        if seam_target == "manifest":
            assert path == _summary_v2_path(root)
        assert isinstance(payload_bytes, bytes)
        raise AssertionError("summary v2 publish must not start after fresh inventory drift")

    monkeypatch.setattr(runner_module, "_immutable_input_manifest", rebuild_with_late_input)
    monkeypatch.setattr(
        runner_module, "atomic_write_bytes_no_clobber", fail_if_publish_attempted, raising=False
    )
    expected_match = {
        "manifest": r"^Production manifest inventory count drifted: expected 22, found 23\.$",
        "shard": r"^Finalizer found unexpected shard: late\.extra\.bin\.0\.baseline\.trial-0$",
        "config": (
            rf"^Immutable input file raw byte digest drifted: "
            rf"{re.escape(str(canonical_target['config']))}$"
        ),
        "sweep": (
            rf"^Immutable input file raw byte digest drifted: "
            rf"{re.escape(str(canonical_target['sweep']))}$"
        ),
        "v1": (
            rf"^Immutable input file raw byte digest drifted: "
            rf"{re.escape(str(canonical_target['v1']))}$"
        ),
    }[seam_target]
    with pytest.raises(
        Exception,
        match=expected_match,
    ):
        _refinalize_batch_v2(run_root=run_root, source_summary_v1=_summary_v1_path(root))
    assert immutable_manifest_calls == 2
    assert publish_calls == 0
    assert not _summary_v2_path(root).exists()


def test_refinalize_batch_v2_binds_all_nine_permutation_streams_and_plus_one_formula_exactly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, _, run_root, root = _prepare_finalizer_run(
        monkeypatch, tmp_path, run_name="counts", write_v1_summary=True
    )
    _patch_refinalize_source_summary_v1_digest(monkeypatch, root)
    payload = _refinalize_batch_v2(run_root=run_root, source_summary_v1=_summary_v1_path(root))
    expected_dose_exceedances = _synthetic_expected_dose_exceedances(root)
    expected_depth_exceedances = _synthetic_expected_depth_exceedances(root)

    dose_stats = _json_object_list_item(payload, "dose_response_statistics")
    depth_relationships = _json_object_list_item(payload, "depth_relationships")
    assert len(dose_stats) == 6
    assert len(depth_relationships) == 3

    for model_name, direction in _synthetic_terminal_row_keys()[:6]:
        row = _dose_stat(payload, model=model_name, direction=direction)
        permutation_count = _json_int_item(row, "permutation_count")
        permutation_seed = _json_int_item(row, "permutation_seed")
        expected_exceed = expected_dose_exceedances[(model_name, direction)]
        expected_seed_parts = [
            29039,
            model_name,
            direction,
            "statistic",
            "all",
            0,
            "spearman_response_permutation",
        ]
        provenance = _json_object_item(row, "permutation_seed_provenance")
        assert permutation_count == DEFAULT_SPEARMAN_PERMUTATIONS
        assert _json_int_item(row, "permutation_exceedance_count") == expected_exceed
        assert _json_int_item(provenance, "seed_namespace") == 29039
        assert _json_list_item(provenance, "seed_parts") == expected_seed_parts
        assert _json_string_item(provenance, "method") == "compact_json_seed"
        assert permutation_seed == compact_json_seed(expected_seed_parts)
        assert np.isclose(
            _json_float_item(row, "p_one_sided"),
            (1 + expected_exceed) / (permutation_count + 1),
        )

    for model_name, direction in _synthetic_terminal_row_keys()[6:]:
        row = _finalize_depth_relationship(payload, model=model_name).payload
        permutation_count = _json_int_item(row, "permutation_count")
        permutation_seed = _json_int_item(row, "permutation_seed")
        expected_exceed = expected_depth_exceedances[model_name]
        expected_seed_parts = [
            29039,
            model_name,
            "wikipedia_to_code",
            "statistic",
            "all",
            0,
            "spearman_response_permutation",
        ]
        provenance = _json_object_item(row, "permutation_seed_provenance")
        assert _json_string_item(row, "direction") == direction
        assert permutation_count == DEFAULT_SPEARMAN_PERMUTATIONS
        assert _json_int_item(row, "permutation_exceedance_count") == expected_exceed
        assert _json_int_item(provenance, "seed_namespace") == 29039
        assert _json_list_item(provenance, "seed_parts") == expected_seed_parts
        assert _json_string_item(provenance, "method") == "compact_json_seed"
        assert permutation_seed == compact_json_seed(expected_seed_parts)
        assert np.isclose(
            _json_float_item(row, "p_one_sided"),
            (1 + expected_exceed) / (permutation_count + 1),
        )


def test_refinalize_batch_v2_forbids_model_data_validation_launch_and_run_model_reachability(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, _, run_root, root = _prepare_finalizer_run(
        monkeypatch, tmp_path, run_name="forbidden", write_v1_summary=True
    )
    _patch_refinalize_source_summary_v1_digest(monkeypatch, root)
    monkeypatch.setattr(
        "si_rebuttal.runner.materialize_data",
        _raise_assertion_callable("materialize-data must not run"),
    )
    monkeypatch.setattr(
        "si_rebuttal.runner.validate_model",
        _raise_assertion_callable("validate-model must not run"),
    )
    monkeypatch.setattr(
        "si_rebuttal.runner.benchmark_model",
        _raise_assertion_callable("benchmark-model must not run"),
    )
    monkeypatch.setattr(
        "si_rebuttal.runner.launch",
        _raise_assertion_callable("launch must not run"),
    )
    monkeypatch.setattr(
        "si_rebuttal.runner.run_model",
        _raise_assertion_callable("run-model must not run"),
    )

    payload = _refinalize_batch_v2(run_root=run_root, source_summary_v1=_summary_v1_path(root))
    assert payload["schema_version"] == 2


def test_refinalize_batch_v2_rejects_mutated_existing_v2_projection_and_manifest_surfaces(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, _, run_root, root = _prepare_finalizer_run(
        monkeypatch, tmp_path, run_name="mutations", write_v1_summary=True
    )
    _patch_refinalize_source_summary_v1_digest(monkeypatch, root)
    _ = _refinalize_batch_v2(run_root=run_root, source_summary_v1=_summary_v1_path(root))
    v2_path = _summary_v2_path(root)

    mutation_cases: tuple[tuple[Callable[[JsonObject], None], str], ...] = (
        (
            lambda payload: _json_list_member(payload, "summary_rows").reverse(),
            r"(?:Finalizer v2 payload drifted\.|Finalizer v2 bytes drifted\.)",
        ),
        (
            lambda payload: _json_object_list_member(payload, "dose_response_statistics")[
                0
            ].__setitem__(
                "permutation_seed",
                _json_int_item(
                    _json_object_list_member(payload, "dose_response_statistics")[0],
                    "permutation_seed",
                )
                + 1,
            ),
            r"(?:Finalizer v2 payload drifted\.|Finalizer v2 bytes drifted\.)",
        ),
        (
            lambda payload: _json_object_list_member(payload, "depth_relationships")[0].__setitem__(
                "permutation_exceedance_count",
                _json_int_item(
                    _json_object_list_member(payload, "depth_relationships")[0],
                    "permutation_exceedance_count",
                )
                + 1,
            ),
            r"(?:Finalizer v2 payload drifted\.|Finalizer v2 bytes drifted\.)",
        ),
        (
            lambda payload: _json_object_list_member(payload, "contrasts")[0].__setitem__(
                "bin",
                _json_int_item(_json_object_list_member(payload, "contrasts")[0], "bin") + 1,
            ),
            r"(?:Finalizer v2 payload drifted\.|Finalizer v2 bytes drifted\.)",
        ),
        (
            lambda payload: _json_object_list_member(payload, "depth_rows")[0].__setitem__(
                "layer",
                (_json_int_item(_json_object_list_member(payload, "depth_rows")[0], "layer") + 1)
                % 32,
            ),
            r"(?:Finalizer v2 payload drifted\.|Finalizer v2 bytes drifted\.)",
        ),
        (
            lambda payload: _json_object_member(payload, "summary_hashes").__setitem__(
                "shard_inventory_sha256",
                _different_hex64(
                    _json_string_item(
                        _json_object_member(payload, "summary_hashes"),
                        "shard_inventory_sha256",
                    )
                ),
            ),
            r"(?:Finalizer v2 payload drifted\.|Finalizer v2 bytes drifted\.)",
        ),
        (
            lambda payload: _json_object_member(payload, "summary_hashes").__setitem__(
                "summary_rows_sha256",
                _different_hex64(
                    _json_string_item(
                        _json_object_member(payload, "summary_hashes"),
                        "summary_rows_sha256",
                    )
                ),
            ),
            r"(?:Finalizer v2 payload drifted\.|Finalizer v2 bytes drifted\.)",
        ),
        (
            lambda payload: _json_object_member(payload, "summary_hashes").__setitem__(
                "contrasts_sha256",
                _different_hex64(
                    _json_string_item(
                        _json_object_member(payload, "summary_hashes"),
                        "contrasts_sha256",
                    )
                ),
            ),
            r"(?:Finalizer v2 payload drifted\.|Finalizer v2 bytes drifted\.)",
        ),
        (
            lambda payload: _json_object_member(payload, "summary_hashes").__setitem__(
                "depth_rows_sha256",
                _different_hex64(
                    _json_string_item(
                        _json_object_member(payload, "summary_hashes"),
                        "depth_rows_sha256",
                    )
                ),
            ),
            r"(?:Finalizer v2 payload drifted\.|Finalizer v2 bytes drifted\.)",
        ),
        (
            lambda payload: _json_object_member(payload, "summary_hashes").__setitem__(
                "scientific_projection_sha256",
                _different_hex64(
                    _json_string_item(
                        _json_object_member(payload, "summary_hashes"),
                        "scientific_projection_sha256",
                    )
                ),
            ),
            r"(?:Finalizer v2 payload drifted\.|Finalizer v2 bytes drifted\.)",
        ),
        (
            lambda payload: _json_object_member(payload, "correction_provenance").__setitem__(
                "immutable_input_manifest_sha256",
                _different_hex64(
                    _json_string_item(
                        _json_object_member(payload, "correction_provenance"),
                        "immutable_input_manifest_sha256",
                    )
                ),
            ),
            r"(?:Finalizer v2 payload drifted\.|Finalizer v2 bytes drifted\.)",
        ),
    )

    for mutate, match in mutation_cases:
        payload = _read_json_object(v2_path)
        mutate(payload)
        _overwrite_json(v2_path, payload)
        with pytest.raises(Exception, match=match):
            _refinalize_batch_v2(run_root=run_root, source_summary_v1=_summary_v1_path(root))
        v2_path.unlink()
        _ = _refinalize_batch_v2(run_root=run_root, source_summary_v1=_summary_v1_path(root))


def test_refinalize_batch_v2_handles_idempotent_repeat_corrupt_v2_and_publication_races(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, _, run_root, root = _prepare_finalizer_run(
        monkeypatch, tmp_path, run_name="race", write_v1_summary=True
    )
    _patch_refinalize_source_summary_v1_digest(monkeypatch, root)
    first = _refinalize_batch_v2(run_root=run_root, source_summary_v1=_summary_v1_path(root))
    first_bytes = _summary_v2_path(root).read_bytes()
    second = _refinalize_batch_v2(run_root=run_root, source_summary_v1=_summary_v1_path(root))
    assert second == first
    assert _summary_v2_path(root).read_bytes() == first_bytes

    _summary_v2_path(root).write_text("{bad json}\n", encoding="ascii")
    with pytest.raises(Exception, match="valid immutable terminal summary"):
        _refinalize_batch_v2(run_root=run_root, source_summary_v1=_summary_v1_path(root))

    _summary_v2_path(root).unlink()
    _ = _refinalize_batch_v2(run_root=run_root, source_summary_v1=_summary_v1_path(root))
    _summary_v2_path(root).unlink()
    publish_calls = 0

    def identical_winner(path: str | Path, payload_bytes: bytes) -> None:
        nonlocal publish_calls
        publish_calls += 1
        assert isinstance(payload_bytes, bytes)
        Path(path).write_bytes(payload_bytes)
        raise ArtifactError(f"Refusing to overwrite existing artifact: {path}")

    v2_path = _summary_v2_path(root)
    monkeypatch.setattr(
        runner_module,
        "atomic_write_bytes_no_clobber",
        identical_winner,
        raising=False,
    )
    raced = _refinalize_batch_v2(run_root=run_root, source_summary_v1=_summary_v1_path(root))
    assert publish_calls == 1
    assert raced == _read_json_object(v2_path)

    race_failures: tuple[tuple[str, bytes], ...] = (
        (
            "conflict",
            _artifact_bytes(
                _json_object({"schema_version": 2, "completed_at": "2026-07-30T12:00:00+00:00"})
            ),
        ),
        ("corrupt", b"{bad json}\n"),
        ("partial", v2_path.read_bytes()[:32]),
    )
    for _label, winner_bytes in race_failures:
        v2_path.unlink()

        def nonidentical_winner(
            path: str | Path, payload_bytes: bytes, *, winner: bytes = winner_bytes
        ) -> None:
            assert isinstance(payload_bytes, bytes)
            Path(path).write_bytes(winner)
            raise ArtifactError(f"Refusing to overwrite existing artifact: {path}")

        monkeypatch.setattr(
            runner_module, "atomic_write_bytes_no_clobber", nonidentical_winner, raising=False
        )
        with pytest.raises(
            Exception,
            match=(
                r"(?:Finalizer v2 collision produced non-identical terminal summary\."
                r"|batch-terminal-summary\.v2\.json is not a valid immutable terminal "
                r"summary: .*"
                r"|Finalizer v2 payload drifted\."
                r"|Finalizer v2 bytes drifted\.)"
            ),
        ):
            _refinalize_batch_v2(run_root=run_root, source_summary_v1=_summary_v1_path(root))
        assert v2_path.read_bytes() == winner_bytes


def test_refinalize_batch_v2_rejects_fixed_production_v1_digest_mismatch_before_v2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, _, run_root, root = _prepare_finalizer_run(
        monkeypatch, tmp_path, run_name="fixed-v1-digest-mismatch", write_v1_summary=True
    )
    _patch_refinalize_source_summary_v1_digest(
        monkeypatch,
        root,
        digest=_different_hex64(hashlib.sha256(_summary_v1_path(root).read_bytes()).hexdigest()),
    )
    with pytest.raises(
        Exception,
        match=r"Canonical production source-v1 raw byte digest drifted before v2 finalization\.",
    ):
        _refinalize_batch_v2(run_root=run_root, source_summary_v1=_summary_v1_path(root))
    assert not _summary_v2_path(root).exists()


def test_condition_key_parse_stem_enforces_canonical_control_trials() -> None:
    valid = ConditionKey.parse_stem("llama-3.1-8b.wikipedia_to_code.bin.0.norm.trial-2")
    assert valid == ConditionKey(
        model="llama-3.1-8b",
        direction="wikipedia_to_code",
        unit="bin",
        unit_index="0",
        condition="norm",
        trial=2,
    )
    assert ConditionKey.parse_stem(valid.stem()) == valid

    invalid_stems = (
        "llama-3.1-8b.wikipedia_to_code.bin.0.norm",
        "llama-3.1-8b.wikipedia_to_code.bin.0.source_kernel.trial-0",
        "llama-3.1-8b.wikipedia_to_code.bin.0.norm.trial-00",
        "llama-3.1-8b.wikipedia_to_code.bin.0.norm.trial-01",
        "llama-3.1-8b.wikipedia_to_code.bin.0.norm.trial-3",
        "llama-3.1-8b.wikipedia_to_code.bin.0.offset_permutation.trial-999",
    )
    for stem in invalid_stems:
        with pytest.raises(ArtifactError, match="Unrecognized condition stem"):
            ConditionKey.parse_stem(stem)


def test_finalizer_rejects_missing_and_exact_length(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resolved = _resolved_digests(
        validate_config("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")
    )
    batch_manifest = _batch_manifest_payload()
    with pytest.raises(Exception, match="missing shard"):
        run_root = tmp_path / "run"
        root = RunRoot(run_id=run_root.name, root=run_root, batch_id=None)
        root.ensure()
        load_existing_or_write(root.manifests_dir / "batch-manifest.json", batch_manifest)
        finalize_batch(
            "rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml", run_root=run_root
        )

    run_root = tmp_path / "filled"
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id=None)
    root.ensure()
    load_existing_or_write(root.manifests_dir / "batch-manifest.json", batch_manifest)
    _populate_finalize_inventory(
        monkeypatch=monkeypatch,
        root=root,
        resolved=resolved,
        batch_manifest=batch_manifest,
    )
    _write_expected_v1_summary(root, batch_manifest=batch_manifest)
    invalid_depth_path = root.shards_dir / "llama-3.1-8b.wikipedia_to_code.layer.31.depth.json"
    invalid_depth_payload = _read_json_object(invalid_depth_path)
    invalid_depth_payload["sequence_nll"] = _json_list_item(
        invalid_depth_payload,
        "sequence_nll",
    )[:99]
    invalid_depth_payload["identity_sha256"] = artifact_identity(
        payload_without_identity(_json_object(invalid_depth_payload))
    )
    _overwrite_json(invalid_depth_path, invalid_depth_payload)
    with pytest.raises(Exception, match=r"exact per-sequence length"):
        finalize_batch(
            "rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml", run_root=run_root
        )


GPU_UUID = "GPU-11111111-2222-3333-4444-555555555555"
FOREIGN_GPU_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
NVIDIA_SMI_INVENTORY_COMMAND = (
    "nvidia-smi",
    "--id=1",
    "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits",
)
NVIDIA_SMI_COMPUTE_APPS_COMMAND = (
    "nvidia-smi",
    "--id=1",
    "--query-compute-apps=gpu_uuid,pid",
    "--format=csv,noheader,nounits",
)
OVERLONG_ASCII_INT = "1" * 5000


def test_parse_nvidia_smi_snapshot_accepts_keyword_only_default_and_explicit_empty_apps() -> None:
    parameter = inspect.signature(parse_nvidia_smi_snapshot).parameters["compute_apps_csv_text"]

    omitted = parse_nvidia_smi_snapshot(
        f"1, {GPU_UUID}, NVIDIA L40, 16, 0\n",
        physical_index=1,
    )
    explicit_empty = parse_nvidia_smi_snapshot(
        f"1, {GPU_UUID}, NVIDIA L40, 16, 0\n",
        physical_index=1,
        compute_apps_csv_text="",
    )

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == ""
    assert omitted.compute_processes == ()
    assert explicit_empty.compute_processes == ()
    assert omitted.is_idle is True
    assert explicit_empty.is_idle is True


def test_parse_nvidia_smi_snapshot_thresholds_and_matching_compute_app_pid() -> None:
    matching_pid = parse_nvidia_smi_snapshot(
        f"1, {GPU_UUID}, NVIDIA L40, 0, 0\n",
        physical_index=1,
        compute_apps_csv_text=f"{GPU_UUID}, 4242\n",
    )
    busy_memory = parse_nvidia_smi_snapshot(
        f"1, {GPU_UUID}, NVIDIA L40, 16.1, 0\n",
        physical_index=1,
        compute_apps_csv_text="",
    )
    busy_utilization = parse_nvidia_smi_snapshot(
        f"1, {GPU_UUID}, NVIDIA L40, 16, 0.1\n",
        physical_index=1,
        compute_apps_csv_text="",
    )

    assert matching_pid.compute_processes == ("4242",)
    assert matching_pid.is_idle is False
    assert busy_memory.is_idle is False
    assert busy_utilization.is_idle is False


def test_parse_nvidia_smi_snapshot_rejects_foreign_compute_app_uuid() -> None:
    with pytest.raises(OperationsError, match=r"(?i)(invalid|malformed|uuid|gpu|compute|match)"):
        parse_nvidia_smi_snapshot(
            f"1, {GPU_UUID}, NVIDIA L40, 0, 0\n",
            physical_index=1,
            compute_apps_csv_text=f"{FOREIGN_GPU_UUID}, 4242\n",
        )


@pytest.mark.parametrize(
    ("inventory_csv_text", "compute_apps_csv_text", "match"),
    (
        ("1, GPU-1111, NVIDIA L40, 0\n", "", r"(?i)(malformed|invalid|csv|column)"),
        ('1,"unterminated,NVIDIA L40,0,0\n', "", r"(?i)(malformed|invalid|csv|quote)"),
        (
            f"{OVERLONG_ASCII_INT}, {GPU_UUID}, NVIDIA L40, 0, 0\n",
            "",
            r"(?i)(malformed|invalid|index|gpu|number|numeric)",
        ),
        (
            f"1, {GPU_UUID}, NVIDIA L40, abc, 0\n",
            "",
            r"(?i)(malformed|invalid|number|numeric|memory)",
        ),
        (
            f"1, {GPU_UUID}, NVIDIA L40, 1_0, 0\n",
            "",
            r"(?i)(invalid|malformed|number|numeric|memory)",
        ),
        (
            f"1, {GPU_UUID}, NVIDIA L40, 1e1, 0\n",
            "",
            r"(?i)(invalid|malformed|number|numeric|memory)",
        ),
        (
            f"1, {GPU_UUID}, NVIDIA L40, nan, 0\n",
            "",
            r"(?i)(malformed|invalid|finite|number|numeric|memory)",
        ),
        (
            f"1, {GPU_UUID}, NVIDIA L40, inf, 0\n",
            "",
            r"(?i)(malformed|invalid|finite|number|numeric|memory)",
        ),
        (
            f"1, {GPU_UUID}, NVIDIA L40, -1, 0\n",
            "",
            r"(?i)(malformed|invalid|negative|number|numeric|memory)",
        ),
        (
            f"1, {GPU_UUID}, NVIDIA L40, 0, abc\n",
            "",
            r"(?i)(malformed|invalid|number|numeric|utilization)",
        ),
        (
            f"1, {GPU_UUID}, NVIDIA L40, 0, 1_0\n",
            "",
            r"(?i)(invalid|malformed|number|numeric|utilization)",
        ),
        (
            f"1, {GPU_UUID}, NVIDIA L40, 0, 1e1\n",
            "",
            r"(?i)(invalid|malformed|number|numeric|utilization)",
        ),
        (
            f"1, {GPU_UUID}, NVIDIA L40, 0, nan\n",
            "",
            r"(?i)(malformed|invalid|finite|number|numeric|utilization)",
        ),
        (
            f"1, {GPU_UUID}, NVIDIA L40, 0, inf\n",
            "",
            r"(?i)(malformed|invalid|finite|number|numeric|utilization)",
        ),
        (
            f"1, {GPU_UUID}, NVIDIA L40, 0, -0.1\n",
            "",
            r"(?i)(malformed|invalid|negative|number|numeric|utilization)",
        ),
        (
            f"1, {GPU_UUID}, NVIDIA L40, 0, 100.1\n",
            "",
            r"(?i)(malformed|invalid|100|range|utilization)",
        ),
        (
            f"0, {GPU_UUID}, NVIDIA L40, 0, 0\n",
            "",
            r"(?i)(invalid|malformed|physical gpu|index|contain|match)",
        ),
        (
            f"1, {GPU_UUID}, NVIDIA L40, 0, 0\n1, {GPU_UUID}, NVIDIA L40, 0, 0\n",
            "",
            r"(?i)(invalid|malformed|duplicate|multiple|index|gpu)",
        ),
        ("1, , NVIDIA L40, 0, 0\n", "", r"(?i)(invalid|malformed|uuid|blank)"),
        (
            "1, 11111111-2222-3333-4444-555555555555, NVIDIA L40, 0, 0\n",
            "",
            r"(?i)(invalid|malformed|uuid|canonical|gpu)",
        ),
        (
            "1, GPU-1111, NVIDIA L40, 0, 0\n",
            "",
            r"(?i)(invalid|malformed|uuid|canonical|gpu)",
        ),
        (f"1, {GPU_UUID}, , 0, 0\n", "", r"(?i)(invalid|malformed|name|blank)"),
        (
            f"1, {GPU_UUID}, NVIDIA L40, 0, 0\n",
            '"unterminated,4242\n',
            r"(?i)(invalid|malformed|csv|quote|pid)",
        ),
        (
            f"1, {GPU_UUID}, NVIDIA L40, 0, 0\n",
            f"{GPU_UUID}\n",
            r"(?i)(invalid|malformed|csv|column|pid)",
        ),
        (
            f"1, {GPU_UUID}, NVIDIA L40, 0, 0\n",
            f"{GPU_UUID}, \n",
            r"(?i)(invalid|malformed|pid|blank)",
        ),
        (
            f"1, {GPU_UUID}, NVIDIA L40, 0, 0\n",
            f"{GPU_UUID}, 0\n",
            r"(?i)(invalid|malformed|pid|zero)",
        ),
        (
            f"1, {GPU_UUID}, NVIDIA L40, 0, 0\n",
            f"{GPU_UUID}, +42\n",
            r"(?i)(invalid|malformed|pid|ascii|digit)",
        ),
        (
            f"1, {GPU_UUID}, NVIDIA L40, 0, 0\n",
            f"{GPU_UUID}, -42\n",
            r"(?i)(invalid|malformed|pid|ascii|digit)",
        ),
        (
            f"1, {GPU_UUID}, NVIDIA L40, 0, 0\n",
            f"{GPU_UUID}, abc\n",
            r"(?i)(invalid|malformed|pid|digit)",
        ),
        (
            f"1, {GPU_UUID}, NVIDIA L40, 0, 0\n",
            f"{GPU_UUID}, \u0664\u0662\n",
            r"(?i)(invalid|malformed|pid|ascii|digit)",
        ),
    ),
)
def test_parse_nvidia_smi_snapshot_fails_closed_on_malformed_inputs(
    inventory_csv_text: str,
    compute_apps_csv_text: str,
    match: str,
) -> None:
    with pytest.raises(OperationsError, match=match):
        parse_nvidia_smi_snapshot(
            inventory_csv_text,
            physical_index=1,
            compute_apps_csv_text=compute_apps_csv_text,
        )


def test_snapshot_gpu_uses_inventory_and_compute_app_queries_in_order() -> None:
    calls: list[tuple[str, ...]] = []
    outputs = iter(
        (
            subprocess.CompletedProcess(
                list(NVIDIA_SMI_INVENTORY_COMMAND),
                0,
                stdout=f"1, {GPU_UUID}, NVIDIA L40, 0, 0\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                list(NVIDIA_SMI_COMPUTE_APPS_COMMAND),
                0,
                stdout=f"{GPU_UUID}, 4242\n",
                stderr="",
            ),
        )
    )

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(command))
        return next(outputs)

    snapshot = snapshot_gpu(physical_index=1, run_command=runner)

    assert calls == [NVIDIA_SMI_INVENTORY_COMMAND, NVIDIA_SMI_COMPUTE_APPS_COMMAND]
    assert snapshot.compute_processes == ("4242",)
    assert snapshot.is_idle is False


def test_snapshot_gpu_accepts_empty_compute_app_output() -> None:
    outputs = iter(
        (
            subprocess.CompletedProcess(
                list(NVIDIA_SMI_INVENTORY_COMMAND),
                0,
                stdout=f"1, {GPU_UUID}, NVIDIA L40, 0, 0\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                list(NVIDIA_SMI_COMPUTE_APPS_COMMAND),
                0,
                stdout="",
                stderr="",
            ),
        )
    )

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        del command
        return next(outputs)

    snapshot = snapshot_gpu(physical_index=1, run_command=runner)

    assert snapshot.compute_processes == ()
    assert snapshot.is_idle is True


def test_snapshot_gpu_raises_operations_error_on_nonzero_inventory_completion() -> None:
    calls: list[tuple[str, ...]] = []

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(command))
        return subprocess.CompletedProcess(list(command), 9, stdout="", stderr="inventory failed")

    with pytest.raises(OperationsError, match=r"(?i)(nvidia-smi|inventory|failed|gpu|command)"):
        snapshot_gpu(physical_index=1, run_command=runner)

    assert calls == [NVIDIA_SMI_INVENTORY_COMMAND]


def test_snapshot_gpu_raises_operations_error_on_nonzero_compute_app_completion() -> None:
    calls: list[tuple[str, ...]] = []
    outputs = iter(
        (
            subprocess.CompletedProcess(
                list(NVIDIA_SMI_INVENTORY_COMMAND),
                0,
                stdout=f"1, {GPU_UUID}, NVIDIA L40, 0, 0\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                list(NVIDIA_SMI_COMPUTE_APPS_COMMAND),
                7,
                stdout="",
                stderr="compute apps failed",
            ),
        )
    )

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(command))
        return next(outputs)

    with pytest.raises(OperationsError, match=r"(?i)(nvidia-smi|compute|failed|gpu|command)"):
        snapshot_gpu(physical_index=1, run_command=runner)

    assert calls == [NVIDIA_SMI_INVENTORY_COMMAND, NVIDIA_SMI_COMPUTE_APPS_COMMAND]


def test_snapshot_gpu_wraps_inventory_runtime_error_without_later_call() -> None:
    calls: list[tuple[str, ...]] = []

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(command))
        raise RuntimeError("inventory runtime failure")

    with pytest.raises(OperationsError, match=r"(?i)(nvidia-smi|runtime|inventory|failed|error)"):
        snapshot_gpu(physical_index=1, run_command=runner)

    assert calls == [NVIDIA_SMI_INVENTORY_COMMAND]


def test_snapshot_gpu_wraps_compute_app_runtime_error_without_unintended_later_call() -> None:
    calls: list[tuple[str, ...]] = []

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(command))
        resolved_command = tuple(command)
        if resolved_command == NVIDIA_SMI_INVENTORY_COMMAND:
            return subprocess.CompletedProcess(
                list(command),
                0,
                stdout=f"1, {GPU_UUID}, NVIDIA L40, 0, 0\n",
                stderr="",
            )
        if resolved_command == NVIDIA_SMI_COMPUTE_APPS_COMMAND:
            raise RuntimeError("compute apps runtime failure")
        raise AssertionError(resolved_command)

    with pytest.raises(OperationsError, match=r"(?i)(nvidia-smi|runtime|compute|failed|error)"):
        snapshot_gpu(physical_index=1, run_command=runner)

    assert calls == [NVIDIA_SMI_INVENTORY_COMMAND, NVIDIA_SMI_COMPUTE_APPS_COMMAND]


def test_require_two_idle_snapshots_uses_exact_four_calls_and_one_five_second_sleep() -> None:
    calls: list[tuple[str, ...]] = []
    outputs = iter(
        (
            subprocess.CompletedProcess(
                list(NVIDIA_SMI_INVENTORY_COMMAND),
                0,
                stdout=f"1, {GPU_UUID}, NVIDIA L40, 0, 0\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                list(NVIDIA_SMI_COMPUTE_APPS_COMMAND),
                0,
                stdout="",
                stderr="",
            ),
            subprocess.CompletedProcess(
                list(NVIDIA_SMI_INVENTORY_COMMAND),
                0,
                stdout=f"1, {GPU_UUID}, NVIDIA L40, 16, 0\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                list(NVIDIA_SMI_COMPUTE_APPS_COMMAND),
                0,
                stdout="",
                stderr="",
            ),
        )
    )
    slept: list[float] = []

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(command))
        return next(outputs)

    def record_sleep(seconds: float) -> None:
        slept.append(seconds)

    first, second = require_two_idle_snapshots(
        physical_index=1,
        run_command=runner,
        sleep_fn=record_sleep,
    )

    assert first.is_idle is True
    assert second.is_idle is True
    assert calls == [
        NVIDIA_SMI_INVENTORY_COMMAND,
        NVIDIA_SMI_COMPUTE_APPS_COMMAND,
        NVIDIA_SMI_INVENTORY_COMMAND,
        NVIDIA_SMI_COMPUTE_APPS_COMMAND,
    ]
    assert slept == [GPU_IDLE_SLEEP_SECONDS]


def test_require_two_idle_snapshots_rejects_busy_first_snapshot_without_sleep() -> None:
    calls: list[tuple[str, ...]] = []
    outputs = iter(
        (
            subprocess.CompletedProcess(
                list(NVIDIA_SMI_INVENTORY_COMMAND),
                0,
                stdout=f"1, {GPU_UUID}, NVIDIA L40, 16.1, 0\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                list(NVIDIA_SMI_COMPUTE_APPS_COMMAND),
                0,
                stdout="",
                stderr="",
            ),
        )
    )
    slept: list[float] = []

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(command))
        return next(outputs)

    def record_sleep(seconds: float) -> None:
        slept.append(seconds)

    with pytest.raises(OperationsError, match=r"(?i)(gpu|idle|first|snapshot)"):
        require_two_idle_snapshots(
            physical_index=1,
            run_command=runner,
            sleep_fn=record_sleep,
        )

    assert calls == [NVIDIA_SMI_INVENTORY_COMMAND, NVIDIA_SMI_COMPUTE_APPS_COMMAND]
    assert slept == []


def test_require_two_idle_snapshots_rejects_busy_second_snapshot_after_one_sleep() -> None:
    calls: list[tuple[str, ...]] = []
    outputs = iter(
        (
            subprocess.CompletedProcess(
                list(NVIDIA_SMI_INVENTORY_COMMAND),
                0,
                stdout=f"1, {GPU_UUID}, NVIDIA L40, 0, 0\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                list(NVIDIA_SMI_COMPUTE_APPS_COMMAND),
                0,
                stdout="",
                stderr="",
            ),
            subprocess.CompletedProcess(
                list(NVIDIA_SMI_INVENTORY_COMMAND),
                0,
                stdout=f"1, {GPU_UUID}, NVIDIA L40, 16, 0\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                list(NVIDIA_SMI_COMPUTE_APPS_COMMAND),
                0,
                stdout=f"{GPU_UUID}, 4242\n",
                stderr="",
            ),
        )
    )
    slept: list[float] = []

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(command))
        return next(outputs)

    def record_sleep(seconds: float) -> None:
        slept.append(seconds)

    with pytest.raises(OperationsError, match=r"(?i)(gpu|idle|second|snapshot)"):
        require_two_idle_snapshots(
            physical_index=1,
            run_command=runner,
            sleep_fn=record_sleep,
        )

    assert calls == [
        NVIDIA_SMI_INVENTORY_COMMAND,
        NVIDIA_SMI_COMPUTE_APPS_COMMAND,
        NVIDIA_SMI_INVENTORY_COMMAND,
        NVIDIA_SMI_COMPUTE_APPS_COMMAND,
    ]
    assert slept == [GPU_IDLE_SLEEP_SECONDS]


def test_launch_plan_and_shell_quoting_is_safe_and_dry_run_nonmutating(tmp_path: Path) -> None:
    from si_rebuttal.runner import load_configs

    resolved, sweep = load_configs("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")
    plan = build_launch_plan(
        resolved_config=resolved,
        sweep=sweep,
        batch_id="batch 01",
        run_root=tmp_path / "run root",
        execute=False,
        python_executable="/usr/bin/python3",
    )
    assert not (tmp_path / "run root").exists()
    assert "export CUDA_VISIBLE_DEVICES=0" in plan.commands[0].script_body
    expected_launch_receipt = tmp_path / "run root" / "receipts" / "si-rebuttal-gpu0.json"
    assert (
        f"export {LAUNCH_RECEIPT_ENV}='{expected_launch_receipt}'" in plan.commands[0].script_body
    )
    assert (
        f"export {BATCH_RECEIPT_ENV}='{tmp_path / 'run root' / 'receipts' / 'batch.json'}'"
        in plan.commands[0].script_body
    )
    assert plan.commands[0].tmux_command.endswith(
        f">> '{tmp_path / 'run root' / 'logs' / 'si-rebuttal-gpu0.log'}' 2>&1"
    )
    assert str(plan.commands[0].script_path).endswith("launch-scripts/si-rebuttal-gpu0.sh")
    assert shell_join(("/bin/echo", "hello world", "a'b")) == "/bin/echo 'hello world' 'a'\"'\"'b'"


def test_launch_plan_commands_preserve_self_contained_config_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from si_rebuttal.runner import load_configs

    base_env = {key: value for key, value in os.environ.items() if key.startswith("SI_REBUTTAL_")}
    monkeypatch.setenv("SI_REBUTTAL_AMBIENT_TEST", "ambient")

    expected = load_base_config("rebuttal/configs/base.toml", env=base_env)
    resolved, sweep = load_configs("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")
    assert resolved.identity_sha256 == expected.identity_sha256
    assert resolved.paths == expected.paths
    for model_name, model_config in expected.models.items():
        assert resolved.models[model_name].weights_path == model_config.weights_path
        assert resolved.models[model_name].tokenizer_path == model_config.tokenizer_path

    root_allowlist = {
        "SI_REBUTTAL_RUNS_ROOT": "runs",
        "SI_REBUTTAL_DATA_ROOT": "data/materialized",
        "SI_REBUTTAL_LOGS_ROOT": "logs",
        "SI_REBUTTAL_CACHE_ROOT": "cache",
        "SI_REBUTTAL_MODELS_ROOT": str((tmp_path / "models").resolve()),
    }
    assert {key: base_env[key] for key in root_allowlist} == root_allowlist
    model_env_names = tuple(
        model_env
        for model_config in expected.models.values()
        for model_env in (model_config.weight_env, model_config.tokenizer_env)
    )
    assert all(Path(base_env[name]).is_absolute() for name in model_env_names)

    plan = build_launch_plan(
        resolved_config=resolved,
        sweep=sweep,
        batch_id="batch 01",
        run_root=tmp_path / "run root",
        execute=False,
        python_executable="/usr/bin/python3",
    )
    for command in plan.commands:
        command_env = dict(command.environment)
        assert "SI_REBUTTAL_AMBIENT_TEST" not in command_env
        assert {key: command_env.get(key) for key in root_allowlist} == root_allowlist
        assert all(Path(command_env[name]).is_absolute() for name in model_env_names)

        reloaded = load_base_config("rebuttal/configs/base.toml", env=command_env)
        assert reloaded.identity_sha256 == expected.identity_sha256
        assert reloaded.paths == expected.paths
        for model_name, model_config in expected.models.items():
            assert reloaded.models[model_name].weights_path == model_config.weights_path
            assert reloaded.models[model_name].tokenizer_path == model_config.tokenizer_path


def test_build_launch_plan_rejects_model_environment_name_conflicting_with_lane_identity(
    tmp_path: Path,
) -> None:
    from si_rebuttal.runner import load_configs

    resolved, sweep = load_configs("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")
    target_model_name = sweep.placement_gpu0[0]
    target_model = resolved.models[target_model_name]
    conflicted_model = replace(
        target_model,
        weight_env="CUDA_VISIBLE_DEVICES",
        weights_path=target_model.weights_path,
    )
    conflicted_models = dict(resolved.models)
    conflicted_models[target_model_name] = conflicted_model
    conflicted_resolved = replace(resolved, models=conflicted_models)

    with pytest.raises(
        OperationsError,
        match=r"(?i)(CUDA_VISIBLE_DEVICES|duplicate|conflict|environment)",
    ):
        build_launch_plan(
            resolved_config=conflicted_resolved,
            sweep=sweep,
            batch_id="batch 01",
            run_root=tmp_path / "run root",
            execute=False,
            python_executable="/usr/bin/python3",
        )


def test_launch_tmux_plan_rolls_back_only_created_sessions(tmp_path: Path) -> None:
    resolved, sweep = load_configs("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")
    run_root = tmp_path / "launch-root"
    run_root.mkdir()
    plan = build_launch_plan(
        resolved_config=resolved,
        sweep=sweep,
        batch_id="batch",
        run_root=run_root,
        execute=True,
        python_executable=sys.executable,
    )
    commands: list[tuple[str, ...]] = []

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        resolved_command = tuple(command)
        commands.append(resolved_command)
        if resolved_command[:4] == ("tmux", "new-session", "-d", "-s"):
            if resolved_command[4] == "si-rebuttal-gpu0":
                return subprocess.CompletedProcess(list(command), 0, stdout="", stderr="")
            if resolved_command[4] == "si-rebuttal-gpu1":
                return subprocess.CompletedProcess(
                    list(command), 1, stdout="", stderr="second failed"
                )
        if resolved_command == ("tmux", "kill-session", "-t", "si-rebuttal-gpu0"):
            return subprocess.CompletedProcess(list(command), 0, stdout="", stderr="")
        raise AssertionError(resolved_command)

    with pytest.raises(Exception, match="second failed"):
        launch_tmux_plan(plan=plan, run_command=runner)
    assert commands == [
        ("tmux", "new-session", "-d", "-s", "si-rebuttal-gpu0", plan.commands[0].tmux_command),
        ("tmux", "new-session", "-d", "-s", "si-rebuttal-gpu1", plan.commands[1].tmux_command),
        ("tmux", "kill-session", "-t", "si-rebuttal-gpu0"),
    ]


def test_launch_tmux_plan_reports_cleanup_failure_after_primary_failure(tmp_path: Path) -> None:
    resolved, sweep = load_configs("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")
    run_root = tmp_path / "launch-root"
    run_root.mkdir()
    plan = build_launch_plan(
        resolved_config=resolved,
        sweep=sweep,
        batch_id="batch",
        run_root=run_root,
        execute=True,
        python_executable=sys.executable,
    )

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        resolved_command = tuple(command)
        if resolved_command[:4] == ("tmux", "new-session", "-d", "-s"):
            if resolved_command[4] == "si-rebuttal-gpu0":
                return subprocess.CompletedProcess(list(command), 0, stdout="", stderr="")
            if resolved_command[4] == "si-rebuttal-gpu1":
                return subprocess.CompletedProcess(
                    list(command), 1, stdout="", stderr="second failed"
                )
        if resolved_command == ("tmux", "kill-session", "-t", "si-rebuttal-gpu0"):
            return subprocess.CompletedProcess(list(command), 1, stdout="", stderr="cleanup failed")
        raise AssertionError(resolved_command)

    with pytest.raises(Exception, match=r"second failed.*Rollback also failed: cleanup failed"):
        launch_tmux_plan(plan=plan, run_command=runner)


def test_launch_dry_run_requires_admitted_root_and_does_not_call_tmux_or_nvidia(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called: list[tuple[str, ...]] = []

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        called.append(tuple(command))
        raise AssertionError("dry-run should not call subprocesses")

    with pytest.raises(
        Exception,
        match="launch requires the already-existing exact materialized immutable run root",
    ):
        launch(
            "rebuttal/configs/base.toml",
            "rebuttal/configs/sweep.toml",
            run_root=tmp_path / "launch",
            batch_id="b",
            execute=False,
            run_command=runner,
        )
    run_root = tmp_path / "launch-root"
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id="b")
    root.ensure()
    resolved = _resolved_digests(
        validate_config("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")
    )
    batch_manifest = _batch_manifest_payload()
    load_existing_or_write(root.manifests_dir / "batch-manifest.json", batch_manifest)
    with pytest.raises(
        Exception, match="Lane admission receipt is required before run-model --execute"
    ):
        launch(
            "rebuttal/configs/base.toml",
            "rebuttal/configs/sweep.toml",
            run_root=run_root,
            batch_id="b",
            execute=False,
            run_command=runner,
        )
    benchmark_receipts: dict[str, JsonObject] = {}
    for model_name in ("llama-3.1-8b", "mistral-7b-v0.1", "olmo-2-7b"):
        validation_payload = _validation_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
        )
        benchmark_payload = _benchmark_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            validation_receipt_sha256=_json_identity(validation_payload),
        )
        benchmark_receipts[model_name] = benchmark_payload
        load_existing_or_write(
            root.receipts_dir / f"validation.{model_name}.json", validation_payload
        )
        load_existing_or_write(
            root.receipts_dir / f"benchmark.{model_name}.json", benchmark_payload
        )
    load_existing_or_write(
        root.receipts_dir / "lane-admission.json",
        _lane_receipt_payload(
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            benchmark_receipts=benchmark_receipts,
        ),
    )
    before = sorted(path.relative_to(run_root).as_posix() for path in run_root.rglob("*"))
    payload = launch(
        "rebuttal/configs/base.toml",
        "rebuttal/configs/sweep.toml",
        run_root=run_root,
        batch_id="b",
        execute=False,
        run_command=runner,
    )
    after = sorted(path.relative_to(run_root).as_posix() for path in run_root.rglob("*"))
    assert _json_string_item(payload, "batch_id") == "b"
    assert _json_string_item(payload, "batch_manifest_sha256") == batch_manifest["identity_sha256"]
    assert _json_string_item(payload, "lane_admission_receipt_sha256")
    assert _json_string_item(
        _json_object_item(payload, "launch_script_paths"), "si-rebuttal-gpu0"
    ).endswith("launch-root/launch-scripts/si-rebuttal-gpu0.sh")
    assert _json_string_item(
        _json_object_item(payload, "launch_log_paths"), "si-rebuttal-gpu0"
    ).endswith("launch-root/logs/si-rebuttal-gpu0.log")
    assert called == []
    assert before == after


def test_launch_execute_reuses_existing_admitted_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_root = tmp_path / "launch-root"
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id="batch")
    root.ensure()
    resolved = _resolved_digests(
        validate_config("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")
    )
    batch_manifest = _batch_manifest_payload()
    load_existing_or_write(root.manifests_dir / "batch-manifest.json", batch_manifest)
    validation_receipts: dict[str, JsonObject] = {}
    benchmark_receipts: dict[str, JsonObject] = {}
    for model_name in ("llama-3.1-8b", "mistral-7b-v0.1", "olmo-2-7b"):
        validation_payload = _validation_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
        )
        benchmark_payload = _benchmark_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            validation_receipt_sha256=_json_identity(validation_payload),
        )
        validation_receipts[model_name] = validation_payload
        benchmark_receipts[model_name] = benchmark_payload
        load_existing_or_write(
            root.receipts_dir / f"validation.{model_name}.json", validation_payload
        )
        load_existing_or_write(
            root.receipts_dir / f"benchmark.{model_name}.json", benchmark_payload
        )
    load_existing_or_write(
        root.receipts_dir / "lane-admission.json",
        _lane_receipt_payload(
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            benchmark_receipts=benchmark_receipts,
        ),
    )
    monkeypatch.setattr("si_rebuttal.runner.capture_git_binding", _git_binding_stub)
    launched: list[Path] = []
    monkeypatch.setattr("si_rebuttal.runner.require_tmux_absent", _noop_kwargs)
    monkeypatch.setattr(
        "si_rebuttal.runner.require_two_idle_snapshots", _require_two_idle_snapshots_stub
    )
    monkeypatch.setattr(
        "si_rebuttal.runner.launch_tmux_plan",
        LaunchPlanAppender(launched=launched),
    )
    _set_visible_gpu_count(monkeypatch, count=2)
    payload = launch(
        "rebuttal/configs/base.toml",
        "rebuttal/configs/sweep.toml",
        run_root=run_root,
        batch_id="batch",
        execute=True,
        run_command=_runtime_command_runner(),
    )
    assert payload["launched"] is True
    assert launched == [run_root]
    assert (run_root / "launch-scripts" / "si-rebuttal-gpu0.sh").exists()
    assert (run_root / "logs" / "si-rebuttal-gpu0.log").exists()
    launch_receipt = _read_json_object(root.receipts_dir / "si-rebuttal-gpu0.json")
    lane_receipt = _read_json_object(root.receipts_dir / "lane-admission.json")
    batch_receipt = _read_json_object(root.receipts_dir / "batch.json")
    assert (
        _json_string_item(launch_receipt, "batch_manifest_sha256")
        == batch_manifest["identity_sha256"]
    )
    assert _json_string_item(launch_receipt, "lane_admission_receipt_sha256") == _json_string_item(
        lane_receipt, "identity_sha256"
    )
    assert _json_string_item(launch_receipt, "launch_receipt_path") == str(
        root.receipts_dir / "si-rebuttal-gpu0.json"
    )
    assert _json_string_item(launch_receipt, "batch_receipt_path") == str(
        root.receipts_dir / "batch.json"
    )
    assert _json_string_item(launch_receipt, "launch_script_path") == str(
        run_root / "launch-scripts" / "si-rebuttal-gpu0.sh"
    )
    assert _json_string_item(launch_receipt, "launch_log_path") == str(
        run_root / "logs" / "si-rebuttal-gpu0.log"
    )
    expected_tmux = (
        f"bash {run_root / 'launch-scripts' / 'si-rebuttal-gpu0.sh'} "
        f">> {run_root / 'logs' / 'si-rebuttal-gpu0.log'} 2>&1"
    )
    assert _json_string_item(launch_receipt, "tmux_command") == expected_tmux
    assert _json_string_item(launch_receipt, "cuda_visible_devices") == "0"
    assert _json_string_item(launch_receipt, "logical_device") == "cuda:0"
    assert (
        _json_string_item(batch_receipt, "batch_manifest_sha256")
        == batch_manifest["identity_sha256"]
    )
    assert _json_string_item(batch_receipt, "lane_admission_receipt_sha256") == _json_string_item(
        lane_receipt, "identity_sha256"
    )
    assert _json_string_item(
        _json_object_item(batch_receipt, "launch_receipts"), "si-rebuttal-gpu0"
    ) == _json_string_item(launch_receipt, "identity_sha256")
    assert _json_string_item(
        _json_object_item(batch_receipt, "launch_receipt_paths"), "si-rebuttal-gpu0"
    ) == str(root.receipts_dir / "si-rebuttal-gpu0.json")
    assert _json_string_item(
        _json_object_item(batch_receipt, "launch_script_paths"), "si-rebuttal-gpu0"
    ) == str(run_root / "launch-scripts" / "si-rebuttal-gpu0.sh")
    assert _json_string_item(
        _json_object_item(batch_receipt, "launch_log_paths"), "si-rebuttal-gpu0"
    ) == str(run_root / "logs" / "si-rebuttal-gpu0.log")


def test_launch_execute_rejects_forged_lane_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_root = tmp_path / "launch-root"
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id="batch")
    root.ensure()
    resolved = _resolved_digests(
        validate_config("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")
    )
    batch_manifest = _batch_manifest_payload()
    load_existing_or_write(root.manifests_dir / "batch-manifest.json", batch_manifest)
    benchmark_receipts: dict[str, JsonObject] = {}
    for model_name in ("llama-3.1-8b", "mistral-7b-v0.1", "olmo-2-7b"):
        validation_payload = _validation_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
        )
        benchmark_payload = _benchmark_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            validation_receipt_sha256=_json_identity(validation_payload),
        )
        benchmark_receipts[model_name] = benchmark_payload
        load_existing_or_write(
            root.receipts_dir / f"validation.{model_name}.json", validation_payload
        )
        load_existing_or_write(
            root.receipts_dir / f"benchmark.{model_name}.json", benchmark_payload
        )
    forged = _lane_receipt_payload(
        batch_manifest=batch_manifest,
        config_sha256=resolved["config_sha256"],
        sweep_sha256=resolved["sweep_sha256"],
        benchmark_receipts=benchmark_receipts,
    )
    _json_object_member(forged, "benchmark_receipts")["llama-3.1-8b"] = "forged"
    forged = _with_identity(forged)
    load_existing_or_write(root.receipts_dir / "lane-admission.json", forged)
    monkeypatch.setattr("si_rebuttal.runner.capture_git_binding", _git_binding_stub)
    monkeypatch.setattr("si_rebuttal.runner.require_tmux_absent", _noop_kwargs)
    monkeypatch.setattr(
        "si_rebuttal.runner.require_two_idle_snapshots", _require_two_idle_snapshots_stub
    )
    with pytest.raises(Exception, match="Lane admission benchmark binding drifted"):
        launch(
            "rebuttal/configs/base.toml",
            "rebuttal/configs/sweep.toml",
            run_root=run_root,
            batch_id="batch",
            execute=True,
            run_command=_runtime_command_runner(),
        )


def test_launch_execute_rejects_forged_low_benchmark_lane_hours(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_root = tmp_path / "launch-root"
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id="batch")
    root.ensure()
    resolved = _resolved_digests(
        validate_config("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")
    )
    batch_manifest = _batch_manifest_payload()
    load_existing_or_write(root.manifests_dir / "batch-manifest.json", batch_manifest)
    benchmark_receipts: dict[str, JsonObject] = {}
    for model_name, lane_hours in (
        ("llama-3.1-8b", 40.0),
        ("mistral-7b-v0.1", 60.0),
        ("olmo-2-7b", 50.0),
    ):
        validation_payload = _validation_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
        )
        benchmark_payload = _benchmark_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            validation_receipt_sha256=_json_identity(validation_payload),
            lane_hours=lane_hours,
        )
        if model_name == "llama-3.1-8b":
            benchmark_payload = _benchmark_receipt_payload(
                model_name=model_name,
                batch_manifest=batch_manifest,
                config_sha256=resolved["config_sha256"],
                sweep_sha256=resolved["sweep_sha256"],
                validation_receipt_sha256=_json_identity(validation_payload),
                lane_hours=40.0,
                projected_total_seconds=130.0 * 3600.0,
                stored_lane_hours=40.0,
            )
        benchmark_receipts[model_name] = benchmark_payload
        load_existing_or_write(
            root.receipts_dir / f"validation.{model_name}.json", validation_payload
        )
        load_existing_or_write(
            root.receipts_dir / f"benchmark.{model_name}.json", benchmark_payload
        )
    load_existing_or_write(
        root.receipts_dir / "lane-admission.json",
        _lane_receipt_payload(
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            benchmark_receipts=benchmark_receipts,
        ),
    )
    monkeypatch.setattr("si_rebuttal.runner.capture_git_binding", _git_binding_stub)
    monkeypatch.setattr("si_rebuttal.runner.require_tmux_absent", _noop_kwargs)
    monkeypatch.setattr(
        "si_rebuttal.runner.require_two_idle_snapshots", _require_two_idle_snapshots_stub
    )
    with pytest.raises(Exception, match=r"Benchmark lane_hours drifted for llama-3\.1-8b"):
        launch(
            "rebuttal/configs/base.toml",
            "rebuttal/configs/sweep.toml",
            run_root=run_root,
            batch_id="batch",
            execute=True,
            run_command=_runtime_command_runner(),
        )


def test_launch_execute_rejects_git_drift_before_side_effects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_root = tmp_path / "launch-root"
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id="batch")
    root.ensure()
    resolved = _resolved_digests(
        validate_config("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")
    )
    batch_manifest = _batch_manifest_payload()
    load_existing_or_write(root.manifests_dir / "batch-manifest.json", batch_manifest)
    benchmark_receipts: dict[str, JsonObject] = {}
    for model_name in ("llama-3.1-8b", "mistral-7b-v0.1", "olmo-2-7b"):
        validation_payload = _validation_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
        )
        benchmark_payload = _benchmark_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            validation_receipt_sha256=_json_identity(validation_payload),
        )
        benchmark_receipts[model_name] = benchmark_payload
        load_existing_or_write(
            root.receipts_dir / f"validation.{model_name}.json", validation_payload
        )
        load_existing_or_write(
            root.receipts_dir / f"benchmark.{model_name}.json", benchmark_payload
        )
    load_existing_or_write(
        root.receipts_dir / "lane-admission.json",
        _lane_receipt_payload(
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            benchmark_receipts=benchmark_receipts,
        ),
    )
    monkeypatch.setattr(
        "si_rebuttal.runner.capture_git_binding",
        _git_binding_with_forged_commit,
    )
    monkeypatch.setattr(
        "si_rebuttal.runner.require_tmux_absent",
        _raise_assertion_callable("tmux check must not run after provenance drift"),
    )
    monkeypatch.setattr(
        "si_rebuttal.runner.require_two_idle_snapshots",
        _raise_assertion_callable("gpu snapshot must not run after provenance drift"),
    )
    monkeypatch.setattr(
        "si_rebuttal.runner.launch_tmux_plan",
        _raise_assertion_callable("tmux launch must not run after provenance drift"),
    )
    with pytest.raises(Exception, match="Batch manifest git commit drifted"):
        launch(
            "rebuttal/configs/base.toml",
            "rebuttal/configs/sweep.toml",
            run_root=run_root,
            batch_id="batch",
            execute=True,
            run_command=_runtime_command_runner(),
        )
    assert not (run_root / "receipts" / "si-rebuttal-gpu0.json").exists()
    assert not (run_root / "receipts" / "batch.json").exists()
    assert not (run_root / "launch-scripts" / "si-rebuttal-gpu0.sh").exists()
    assert not (run_root / "logs" / "si-rebuttal-gpu0.log").exists()


def test_launch_execute_rejects_runtime_drift_before_side_effects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_root = tmp_path / "launch-root"
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id="batch")
    root.ensure()
    resolved = _resolved_digests(
        validate_config("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")
    )
    batch_manifest = _batch_manifest_payload()
    load_existing_or_write(root.manifests_dir / "batch-manifest.json", batch_manifest)
    benchmark_receipts: dict[str, JsonObject] = {}
    for model_name in ("llama-3.1-8b", "mistral-7b-v0.1", "olmo-2-7b"):
        validation_payload = _validation_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
        )
        benchmark_payload = _benchmark_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            validation_receipt_sha256=_json_identity(validation_payload),
        )
        benchmark_receipts[model_name] = benchmark_payload
        load_existing_or_write(
            root.receipts_dir / f"validation.{model_name}.json", validation_payload
        )
        load_existing_or_write(
            root.receipts_dir / f"benchmark.{model_name}.json", benchmark_payload
        )
    load_existing_or_write(
        root.receipts_dir / "lane-admission.json",
        _lane_receipt_payload(
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            benchmark_receipts=benchmark_receipts,
        ),
    )
    monkeypatch.setattr("si_rebuttal.runner.capture_git_binding", _git_binding_stub)
    monkeypatch.setattr("si_rebuttal.runner.platform.platform", lambda: "drifted-platform")
    monkeypatch.setattr(
        "si_rebuttal.runner.require_tmux_absent",
        _raise_assertion_callable("tmux check must not run after provenance drift"),
    )
    monkeypatch.setattr(
        "si_rebuttal.runner.require_two_idle_snapshots",
        _raise_assertion_callable("gpu snapshot must not run after provenance drift"),
    )
    monkeypatch.setattr(
        "si_rebuttal.runner.launch_tmux_plan",
        _raise_assertion_callable("tmux launch must not run after provenance drift"),
    )
    _set_visible_gpu_count(monkeypatch, count=2)
    with pytest.raises(Exception, match="Batch manifest runtime binding drifted"):
        launch(
            "rebuttal/configs/base.toml",
            "rebuttal/configs/sweep.toml",
            run_root=run_root,
            batch_id="batch",
            execute=True,
            run_command=_runtime_command_runner(),
        )
    assert not (run_root / "receipts" / "si-rebuttal-gpu0.json").exists()
    assert not (run_root / "receipts" / "batch.json").exists()
    assert not (run_root / "launch-scripts" / "si-rebuttal-gpu0.sh").exists()
    assert not (run_root / "logs" / "si-rebuttal-gpu0.log").exists()


def test_status_is_read_only() -> None:
    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if tuple(command[:2]) == ("tmux", "has-session"):
            return subprocess.CompletedProcess(list(command), 1, stdout="", stderr="")
        raise AssertionError(command)

    payload = read_only_status(run_root=Path("/tmp/does-not-exist"), run_command=runner)
    assert payload["exists"] is False
    assert payload["tmux_sessions"]["si-rebuttal-gpu0"] is False


def test_status_reports_launch_receipt_bindings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_root = tmp_path / "status-root"
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id="batch")
    root.ensure()
    load_existing_or_write(
        root.receipts_dir / "si-rebuttal-gpu0.json",
        {
            "schema_version": 1,
            "batch_id": "batch",
            "run_id": root.run_id,
            "run_root": str(root.root),
            "tmux_session": "si-rebuttal-gpu0",
            "physical_gpu": 0,
            "model_names": ["llama-3.1-8b", "olmo-2-7b"],
            "command": ["python"],
            "command_shell": "python",
            "config_sha256": "c" * 64,
            "sweep_sha256": "s" * 64,
            "batch_manifest_sha256": "b" * 64,
            "lane_admission_receipt_sha256": "l" * 64,
            "launch_script_path": str(run_root / "launch-scripts" / "si-rebuttal-gpu0.sh"),
            "launch_script_sha256": "p" * 64,
            "launch_log_path": str(run_root / "logs" / "si-rebuttal-gpu0.log"),
            "snapshots": [{"physical_index": 0}, {"physical_index": 0}],
        },
    )
    load_existing_or_write(
        root.receipts_dir / "batch.json",
        {
            "schema_version": 1,
            "batch_id": "batch",
            "run_id": root.run_id,
            "run_root": str(root.root),
            "command": ["python"],
            "config_sha256": "c" * 64,
            "sweep_sha256": "s" * 64,
            "batch_manifest_sha256": "b" * 64,
            "lane_admission_receipt_sha256": "l" * 64,
            "launch_receipts": {"si-rebuttal-gpu0": "r" * 64},
            "launch_script_paths": {
                "si-rebuttal-gpu0": str(run_root / "launch-scripts" / "si-rebuttal-gpu0.sh")
            },
            "launch_script_sha256s": {"si-rebuttal-gpu0": "p" * 64},
            "launch_log_paths": {
                "si-rebuttal-gpu0": str(run_root / "logs" / "si-rebuttal-gpu0.log")
            },
        },
    )

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if tuple(command[:2]) == ("tmux", "has-session"):
            return subprocess.CompletedProcess(list(command), 1, stdout="", stderr="")
        raise AssertionError(command)

    payload = status(run_root=run_root, run_command=runner)
    bindings = _launch_receipt_bindings(payload, "si-rebuttal-gpu0")
    assert _json_string_item(bindings.launch_receipt, "batch_manifest_sha256") == "b" * 64
    assert _json_string_item(bindings.launch_receipt, "lane_admission_receipt_sha256") == "l" * 64
    assert _json_string_item(bindings.batch_receipt, "lane_admission_receipt_sha256") == "l" * 64
    assert _json_string_item(
        _json_object_item(_json_object_item(payload, "launch_scripts"), "si-rebuttal-gpu0"),
        "path",
    ).endswith("si-rebuttal-gpu0.sh")
    assert _json_string_item(
        _json_object_item(_json_object_item(payload, "launch_logs"), "si-rebuttal-gpu0"),
        "path",
    ).endswith("si-rebuttal-gpu0.log")


def test_run_model_dry_run_does_not_emit_benchmark_metadata(tmp_path: Path) -> None:
    payload = run_model(
        "rebuttal/configs/base.toml",
        "rebuttal/configs/sweep.toml",
        run_root=tmp_path / "run",
        model_names=("llama-3.1-8b",),
        execute=False,
    )
    assert payload["execute"] is False
    assert "benchmarks" not in payload


def test_run_model_execute_passes_declared_execute_direction_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id=None)
    root.ensure()
    resolved = _resolved_digests(
        validate_config("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")
    )
    batch_manifest = _batch_manifest_payload()
    load_existing_or_write(root.manifests_dir / "batch-manifest.json", batch_manifest)
    validation_receipts: dict[str, JsonObject] = {}
    benchmark_receipts: dict[str, JsonObject] = {}
    for model_name in ("llama-3.1-8b", "mistral-7b-v0.1", "olmo-2-7b"):
        validation_payload = _validation_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
        )
        benchmark_payload = _benchmark_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            validation_receipt_sha256=_json_identity(validation_payload),
        )
        validation_receipts[model_name] = validation_payload
        benchmark_receipts[model_name] = benchmark_payload
        load_existing_or_write(
            root.receipts_dir / f"validation.{model_name}.json", validation_payload
        )
        load_existing_or_write(
            root.receipts_dir / f"benchmark.{model_name}.json", benchmark_payload
        )
    load_existing_or_write(
        root.receipts_dir / "lane-admission.json",
        _lane_receipt_payload(
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            benchmark_receipts=benchmark_receipts,
        ),
    )
    _write_launch_context(monkeypatch, run_root=run_root, batch_id="batch", physical_gpu=1)
    monkeypatch.setattr(
        "si_rebuttal.runner._load_sequences_from_run_root",
        FixedDomainSequenceLoader(
            DomainSequences(fit_tokens=_repeated_sequences(2), eval_tokens=_repeated_sequences(2))
        ),
    )

    def fake_load_local_model_bundle(
        resolved_config: object,
        *,
        model_name: str,
        local_files_only: bool = True,
        logical_device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.bfloat16,
        auto_model_loader: object | None = None,
        auto_tokenizer_loader: object | None = None,
    ) -> FakeLoadedBundle:
        del (
            resolved_config,
            model_name,
            local_files_only,
            torch_dtype,
            auto_model_loader,
            auto_tokenizer_loader,
        )
        assert logical_device == "cuda:0"
        return FakeLoadedBundle(
            binding=FakeBinding(
                weights_tree_sha256="w-mistral-7b-v0.1",
                tokenizer_tree_sha256="t-mistral-7b-v0.1",
            ),
            model=object(),
            tokenizer=object(),
            adapter=None,
            logical_device=torch.device("cpu"),
        )

    monkeypatch.setattr("si_rebuttal.runner.load_local_model_bundle", fake_load_local_model_bundle)
    monkeypatch.setattr(
        "si_rebuttal.runner._capture_fit_summary",
        _return_constant(
            FakeCaptureSummary(
                mean_scores=np.zeros((32, 32), dtype=np.float64),
                raw_kernel_sums=np.ones((32, 32, 512), dtype=np.float64),
                sequence_count=2,
            )
        ),
    )
    calls: list[tuple[str, tuple[str, ...], str]] = []

    def fake_execute_direction(
        *,
        bundle: object,
        run_root: RunRoot,
        resolved_config: object,
        sweep: SweepConfig,
        model_name: str,
        direction: str,
        source_profile: object,
        target_profile: object,
        target_eval: Sequence[Int64Array],
        batch_manifest_sha256: str,
    ) -> None:
        del resolved_config, source_profile, target_profile
        assert bundle is not None
        calls.append((model_name, tuple(sweep.directions), direction))
        assert run_root.root == root.root
        assert batch_manifest_sha256 == batch_manifest["identity_sha256"]
        assert len(target_eval) == 2

    monkeypatch.setattr("si_rebuttal.runner._execute_direction", fake_execute_direction)
    monkeypatch.setattr(
        "si_rebuttal.runner.materialize_frozen_domain",
        _raise_assertion_callable("run-model must reuse frozen manifests"),
    )
    payload = run_model(
        "rebuttal/configs/base.toml",
        "rebuttal/configs/sweep.toml",
        run_root=run_root,
        model_names=("mistral-7b-v0.1",),
        execute=True,
        runtime_command_runner=_runtime_command_runner(),
    )
    assert payload["execute"] is True
    assert calls == [
        ("mistral-7b-v0.1", ("wikipedia_to_code", "code_to_wikipedia"), "wikipedia_to_code"),
        ("mistral-7b-v0.1", ("wikipedia_to_code", "code_to_wikipedia"), "code_to_wikipedia"),
    ]


def test_run_model_execute_rejects_runtime_drift_before_token_or_model_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id=None)
    root.ensure()
    resolved = _resolved_digests(
        validate_config("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")
    )
    batch_manifest = _batch_manifest_payload()
    load_existing_or_write(root.manifests_dir / "batch-manifest.json", batch_manifest)
    benchmark_receipts: dict[str, JsonObject] = {}
    for model_name in ("llama-3.1-8b", "mistral-7b-v0.1", "olmo-2-7b"):
        validation_payload = _validation_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
        )
        benchmark_payload = _benchmark_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            validation_receipt_sha256=_json_identity(validation_payload),
        )
        benchmark_receipts[model_name] = benchmark_payload
        load_existing_or_write(
            root.receipts_dir / f"validation.{model_name}.json", validation_payload
        )
        load_existing_or_write(
            root.receipts_dir / f"benchmark.{model_name}.json", benchmark_payload
        )
    load_existing_or_write(
        root.receipts_dir / "lane-admission.json",
        _lane_receipt_payload(
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            benchmark_receipts=benchmark_receipts,
        ),
    )
    _write_launch_context(monkeypatch, run_root=run_root, batch_id="batch", physical_gpu=1)
    monkeypatch.setattr("si_rebuttal.runner.sys.version", "0.0")
    monkeypatch.setattr(
        "si_rebuttal.runner._load_sequences_from_run_root",
        _raise_assertion_callable("token load must not run after provenance drift"),
    )
    monkeypatch.setattr(
        "si_rebuttal.runner.load_local_model_bundle",
        _raise_model_load_assertion("model load must not run"),
    )
    with pytest.raises(Exception, match="Batch manifest runtime binding drifted"):
        run_model(
            "rebuttal/configs/base.toml",
            "rebuttal/configs/sweep.toml",
            run_root=run_root,
            model_names=("mistral-7b-v0.1",),
            execute=True,
            runtime_command_runner=_runtime_command_runner(),
        )


def test_run_model_execute_allows_expected_single_visible_device_lane_before_token_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id=None)
    root.ensure()
    resolved = _resolved_digests(
        validate_config("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")
    )
    batch_manifest = _batch_manifest_payload()
    load_existing_or_write(root.manifests_dir / "batch-manifest.json", batch_manifest)
    benchmark_receipts: dict[str, JsonObject] = {}
    for model_name in ("llama-3.1-8b", "mistral-7b-v0.1", "olmo-2-7b"):
        validation_payload = _validation_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
        )
        benchmark_payload = _benchmark_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            validation_receipt_sha256=_json_identity(validation_payload),
        )
        benchmark_receipts[model_name] = benchmark_payload
        load_existing_or_write(
            root.receipts_dir / f"validation.{model_name}.json", validation_payload
        )
        load_existing_or_write(
            root.receipts_dir / f"benchmark.{model_name}.json", benchmark_payload
        )
    load_existing_or_write(
        root.receipts_dir / "lane-admission.json",
        _lane_receipt_payload(
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            benchmark_receipts=benchmark_receipts,
        ),
    )
    _write_launch_context(monkeypatch, run_root=run_root, batch_id="batch", physical_gpu=1)
    monkeypatch.setattr(
        "si_rebuttal.runner._load_sequences_from_run_root",
        _raise_assertion_callable("token load reached after one-device runtime guard"),
    )
    monkeypatch.setattr(
        "si_rebuttal.runner.load_local_model_bundle",
        _raise_model_load_assertion("model load must not run before token load"),
    )
    with pytest.raises(AssertionError, match="token load reached after one-device runtime guard"):
        run_model(
            "rebuttal/configs/base.toml",
            "rebuttal/configs/sweep.toml",
            run_root=run_root,
            model_names=("mistral-7b-v0.1",),
            execute=True,
            runtime_command_runner=_runtime_command_runner(),
        )


def test_run_model_execute_rejects_unexpected_visible_device_count_before_token_or_model_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id=None)
    root.ensure()
    resolved = _resolved_digests(
        validate_config("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")
    )
    batch_manifest = _batch_manifest_payload()
    load_existing_or_write(root.manifests_dir / "batch-manifest.json", batch_manifest)
    benchmark_receipts: dict[str, JsonObject] = {}
    for model_name in ("llama-3.1-8b", "mistral-7b-v0.1", "olmo-2-7b"):
        validation_payload = _validation_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
        )
        benchmark_payload = _benchmark_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            validation_receipt_sha256=_json_identity(validation_payload),
        )
        benchmark_receipts[model_name] = benchmark_payload
        load_existing_or_write(
            root.receipts_dir / f"validation.{model_name}.json", validation_payload
        )
        load_existing_or_write(
            root.receipts_dir / f"benchmark.{model_name}.json", benchmark_payload
        )
    load_existing_or_write(
        root.receipts_dir / "lane-admission.json",
        _lane_receipt_payload(
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            benchmark_receipts=benchmark_receipts,
        ),
    )
    _write_launch_context(monkeypatch, run_root=run_root, batch_id="batch", physical_gpu=1)
    _set_visible_gpu_count(monkeypatch, count=2)
    monkeypatch.setattr(
        "si_rebuttal.runner._load_sequences_from_run_root",
        _raise_assertion_callable("token load must not run after runtime capture drift"),
    )
    monkeypatch.setattr(
        "si_rebuttal.runner.load_local_model_bundle",
        _raise_model_load_assertion("model load must not run"),
    )
    with pytest.raises(Exception, match="Batch manifest runtime capture drifted"):
        run_model(
            "rebuttal/configs/base.toml",
            "rebuttal/configs/sweep.toml",
            run_root=run_root,
            model_names=("mistral-7b-v0.1",),
            execute=True,
            runtime_command_runner=_runtime_command_runner(),
        )


def test_run_model_execute_rejects_git_drift_before_token_or_model_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id=None)
    root.ensure()
    resolved = _resolved_digests(
        validate_config("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")
    )
    batch_manifest = _batch_manifest_payload()
    load_existing_or_write(root.manifests_dir / "batch-manifest.json", batch_manifest)
    benchmark_receipts: dict[str, JsonObject] = {}
    for model_name in ("llama-3.1-8b", "mistral-7b-v0.1", "olmo-2-7b"):
        validation_payload = _validation_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
        )
        benchmark_payload = _benchmark_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            validation_receipt_sha256=_json_identity(validation_payload),
        )
        benchmark_receipts[model_name] = benchmark_payload
        load_existing_or_write(
            root.receipts_dir / f"validation.{model_name}.json", validation_payload
        )
        load_existing_or_write(
            root.receipts_dir / f"benchmark.{model_name}.json", benchmark_payload
        )
    load_existing_or_write(
        root.receipts_dir / "lane-admission.json",
        _lane_receipt_payload(
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            benchmark_receipts=benchmark_receipts,
        ),
    )
    _write_launch_context(monkeypatch, run_root=run_root, batch_id="batch", physical_gpu=1)
    monkeypatch.setattr(
        "si_rebuttal.runner.capture_git_binding",
        _git_binding_with_forged_commit,
    )
    monkeypatch.setattr(
        "si_rebuttal.runner._load_sequences_from_run_root",
        _raise_assertion_callable("token load must not run after provenance drift"),
    )
    monkeypatch.setattr(
        "si_rebuttal.runner.load_local_model_bundle",
        _raise_model_load_assertion("model load must not run"),
    )
    with pytest.raises(Exception, match="Batch manifest git commit drifted"):
        run_model(
            "rebuttal/configs/base.toml",
            "rebuttal/configs/sweep.toml",
            run_root=run_root,
            model_names=("mistral-7b-v0.1",),
            execute=True,
            runtime_command_runner=_runtime_command_runner(),
        )


def test_run_model_execute_rejects_direct_invocation_before_model_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id=None)
    root.ensure()
    resolved = _resolved_digests(
        validate_config("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")
    )
    batch_manifest = _batch_manifest_payload()
    load_existing_or_write(root.manifests_dir / "batch-manifest.json", batch_manifest)
    benchmark_receipts: dict[str, JsonObject] = {}
    for model_name in ("llama-3.1-8b", "mistral-7b-v0.1", "olmo-2-7b"):
        validation_payload = _validation_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
        )
        benchmark_payload = _benchmark_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            validation_receipt_sha256=_json_identity(validation_payload),
        )
        benchmark_receipts[model_name] = benchmark_payload
        load_existing_or_write(
            root.receipts_dir / f"validation.{model_name}.json", validation_payload
        )
        load_existing_or_write(
            root.receipts_dir / f"benchmark.{model_name}.json", benchmark_payload
        )
    load_existing_or_write(
        root.receipts_dir / "lane-admission.json",
        _lane_receipt_payload(
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            benchmark_receipts=benchmark_receipts,
        ),
    )
    monkeypatch.setattr(
        "si_rebuttal.runner.load_local_model_bundle",
        _raise_model_load_assertion("model load must not run"),
    )
    with pytest.raises(Exception, match="production-admitted only from launch --execute"):
        run_model(
            "rebuttal/configs/base.toml",
            "rebuttal/configs/sweep.toml",
            run_root=run_root,
            model_names=("llama-3.1-8b", "olmo-2-7b"),
            execute=True,
        )


def test_run_model_execute_rejects_wrong_gpu_before_model_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id=None)
    root.ensure()
    resolved = _resolved_digests(
        validate_config("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")
    )
    batch_manifest = _batch_manifest_payload()
    load_existing_or_write(root.manifests_dir / "batch-manifest.json", batch_manifest)
    benchmark_receipts: dict[str, JsonObject] = {}
    for model_name in ("llama-3.1-8b", "mistral-7b-v0.1", "olmo-2-7b"):
        validation_payload = _validation_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
        )
        benchmark_payload = _benchmark_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            validation_receipt_sha256=_json_identity(validation_payload),
        )
        benchmark_receipts[model_name] = benchmark_payload
        load_existing_or_write(
            root.receipts_dir / f"validation.{model_name}.json", validation_payload
        )
        load_existing_or_write(
            root.receipts_dir / f"benchmark.{model_name}.json", benchmark_payload
        )
    load_existing_or_write(
        root.receipts_dir / "lane-admission.json",
        _lane_receipt_payload(
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            benchmark_receipts=benchmark_receipts,
        ),
    )
    _write_launch_context(monkeypatch, run_root=run_root, batch_id="batch", physical_gpu=0)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(
        "si_rebuttal.runner.load_local_model_bundle",
        _raise_model_load_assertion("model load must not run"),
    )
    with pytest.raises(
        Exception, match="CUDA_VISIBLE_DEVICES must name exactly the single admitted physical GPU"
    ):
        run_model(
            "rebuttal/configs/base.toml",
            "rebuttal/configs/sweep.toml",
            run_root=run_root,
            model_names=("llama-3.1-8b", "olmo-2-7b"),
            execute=True,
            runtime_command_runner=_runtime_command_runner(),
        )


def test_run_model_execute_rejects_wrong_model_lane_before_model_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id=None)
    root.ensure()
    resolved = _resolved_digests(
        validate_config("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")
    )
    batch_manifest = _batch_manifest_payload()
    load_existing_or_write(root.manifests_dir / "batch-manifest.json", batch_manifest)
    benchmark_receipts: dict[str, JsonObject] = {}
    for model_name in ("llama-3.1-8b", "mistral-7b-v0.1", "olmo-2-7b"):
        validation_payload = _validation_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
        )
        benchmark_payload = _benchmark_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            validation_receipt_sha256=_json_identity(validation_payload),
        )
        benchmark_receipts[model_name] = benchmark_payload
        load_existing_or_write(
            root.receipts_dir / f"validation.{model_name}.json", validation_payload
        )
        load_existing_or_write(
            root.receipts_dir / f"benchmark.{model_name}.json", benchmark_payload
        )
    load_existing_or_write(
        root.receipts_dir / "lane-admission.json",
        _lane_receipt_payload(
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            benchmark_receipts=benchmark_receipts,
        ),
    )
    _write_launch_context(monkeypatch, run_root=run_root, batch_id="batch", physical_gpu=0)
    monkeypatch.setattr(
        "si_rebuttal.runner.load_local_model_bundle",
        _raise_model_load_assertion("model load must not run"),
    )
    with pytest.raises(Exception, match="exact fixed admitted lane models"):
        run_model(
            "rebuttal/configs/base.toml",
            "rebuttal/configs/sweep.toml",
            run_root=run_root,
            model_names=("mistral-7b-v0.1",),
            execute=True,
            runtime_command_runner=_runtime_command_runner(),
        )


def test_run_model_execute_rejects_forged_launch_context_before_model_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id=None)
    root.ensure()
    resolved = _resolved_digests(
        validate_config("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")
    )
    batch_manifest = _batch_manifest_payload()
    load_existing_or_write(root.manifests_dir / "batch-manifest.json", batch_manifest)
    benchmark_receipts: dict[str, JsonObject] = {}
    for model_name in ("llama-3.1-8b", "mistral-7b-v0.1", "olmo-2-7b"):
        validation_payload = _validation_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
        )
        benchmark_payload = _benchmark_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            validation_receipt_sha256=_json_identity(validation_payload),
        )
        benchmark_receipts[model_name] = benchmark_payload
        load_existing_or_write(
            root.receipts_dir / f"validation.{model_name}.json", validation_payload
        )
        load_existing_or_write(
            root.receipts_dir / f"benchmark.{model_name}.json", benchmark_payload
        )
    load_existing_or_write(
        root.receipts_dir / "lane-admission.json",
        _lane_receipt_payload(
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            benchmark_receipts=benchmark_receipts,
        ),
    )
    _write_launch_context(monkeypatch, run_root=run_root, batch_id="batch", physical_gpu=0)
    batch_receipt_path = root.receipts_dir / "batch.json"
    forged = _read_json_object(batch_receipt_path)
    launch_receipt_paths = dict(_json_object_item(forged, "launch_receipt_paths"))
    launch_receipt_paths["si-rebuttal-gpu0"] = str(root.receipts_dir / "forged-launch.json")
    forged["launch_receipt_paths"] = _json_object(launch_receipt_paths)
    forged = _with_identity(forged)
    batch_receipt_path.write_text(
        json.dumps(forged, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    monkeypatch.setattr(
        "si_rebuttal.runner.load_local_model_bundle",
        _raise_model_load_assertion("model load must not run"),
    )
    with pytest.raises(Exception, match="batch path binding drifted"):
        run_model(
            "rebuttal/configs/base.toml",
            "rebuttal/configs/sweep.toml",
            run_root=run_root,
            model_names=("llama-3.1-8b", "olmo-2-7b"),
            execute=True,
            runtime_command_runner=_runtime_command_runner(),
        )


def test_run_model_resume_reuses_valid_profiles_and_shards_without_compute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_root = tmp_path / "resume-run"
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id=None)
    root.ensure()
    resolved = _resolved_digests(
        validate_config("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")
    )
    batch_manifest = _batch_manifest_payload()
    load_existing_or_write(root.manifests_dir / "batch-manifest.json", batch_manifest)
    benchmark_receipts: dict[str, JsonObject] = {}
    for model_name in ("llama-3.1-8b", "mistral-7b-v0.1", "olmo-2-7b"):
        validation_payload = _validation_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
        )
        benchmark_payload = _benchmark_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            validation_receipt_sha256=_json_identity(validation_payload),
        )
        benchmark_receipts[model_name] = benchmark_payload
        load_existing_or_write(
            root.receipts_dir / f"validation.{model_name}.json", validation_payload
        )
        load_existing_or_write(
            root.receipts_dir / f"benchmark.{model_name}.json", benchmark_payload
        )
    load_existing_or_write(
        root.receipts_dir / "lane-admission.json",
        _lane_receipt_payload(
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            benchmark_receipts=benchmark_receipts,
        ),
    )
    _write_launch_context(monkeypatch, run_root=run_root, batch_id="batch", physical_gpu=1)
    launch_receipt_path = root.receipts_dir / "si-rebuttal-gpu1.json"
    launch_receipt = _read_json_object(launch_receipt_path)
    batch_receipt = _read_json_object(root.receipts_dir / "batch.json")
    assert os.environ[LAUNCH_RECEIPT_ENV] == str(launch_receipt_path)
    assert os.environ[BATCH_RECEIPT_ENV] == str(root.receipts_dir / "batch.json")
    assert os.environ[TMUX_SESSION_ENV] == "si-rebuttal-gpu1"
    assert os.environ[RUN_ROOT_ENV] == str(run_root)
    assert os.environ[BATCH_ID_ENV] == "batch"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    assert _json_string_item(launch_receipt, "tmux_session") == "si-rebuttal-gpu1"
    assert _json_int_item(launch_receipt, "physical_gpu") == 1
    assert _json_list_item(launch_receipt, "model_names") == ["mistral-7b-v0.1"]
    assert _json_string_item(launch_receipt, "launch_receipt_path") == str(launch_receipt_path)
    assert _json_string_item(launch_receipt, "launch_script_path") == str(
        run_root / "launch-scripts" / "si-rebuttal-gpu1.sh"
    )
    assert _json_string_item(launch_receipt, "launch_log_path") == str(
        run_root / "logs" / "si-rebuttal-gpu1.log"
    )
    assert _json_string_item(
        _json_object_item(batch_receipt, "launch_receipt_paths"), "si-rebuttal-gpu1"
    ) == str(launch_receipt_path)
    assert _json_string_item(
        _json_object_item(batch_receipt, "launch_script_paths"), "si-rebuttal-gpu1"
    ) == str(run_root / "launch-scripts" / "si-rebuttal-gpu1.sh")
    assert _json_string_item(
        _json_object_item(batch_receipt, "launch_log_paths"), "si-rebuttal-gpu1"
    ) == str(run_root / "logs" / "si-rebuttal-gpu1.log")
    monkeypatch.setattr(
        "si_rebuttal.runner._load_sequences_from_run_root",
        FixedDomainSequenceLoader(
            DomainSequences(
                fit_tokens=_repeated_sequences(50), eval_tokens=_repeated_sequences(100)
            )
        ),
    )
    monkeypatch.setattr(
        "si_rebuttal.runner.load_local_model_bundle",
        _return_constant(
            _loaded_bundle(batch_manifest=batch_manifest, model_name="mistral-7b-v0.1")
        ),
    )
    monkeypatch.setattr(
        "si_rebuttal.runner._capture_fit_summary",
        _return_constant(
            FakeCaptureSummary(
                mean_scores=np.zeros((32, 32), dtype=np.float64),
                raw_kernel_sums=np.ones((32, 32, 512), dtype=np.float64),
                sequence_count=50,
            )
        ),
    )
    monkeypatch.setattr("si_rebuttal.runner._eval_nlls", EvalNllRecorder(calls=[]))
    run_model(
        "rebuttal/configs/base.toml",
        "rebuttal/configs/sweep.toml",
        run_root=run_root,
        model_names=("mistral-7b-v0.1",),
        execute=True,
        runtime_command_runner=_runtime_command_runner(),
    )
    monkeypatch.setattr(
        "si_rebuttal.runner.load_local_model_bundle",
        _raise_model_load_assertion("model load must not run on valid resume"),
    )
    monkeypatch.setattr(
        "si_rebuttal.runner._capture_fit_summary",
        _raise_assertion_callable("fit capture must not run on valid resume"),
    )
    monkeypatch.setattr(
        "si_rebuttal.runner._eval_nlls",
        _raise_assertion_callable("eval must not run on valid resume"),
    )
    payload = run_model(
        "rebuttal/configs/base.toml",
        "rebuttal/configs/sweep.toml",
        run_root=run_root,
        model_names=("mistral-7b-v0.1",),
        execute=True,
        runtime_command_runner=_runtime_command_runner(),
    )
    assert payload["execute"] is True


def test_run_model_resume_rejects_forged_fit_profile_and_shard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id=None)
    root.ensure()
    resolved = _resolved_digests(
        validate_config("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")
    )
    batch_manifest = _batch_manifest_payload()
    load_existing_or_write(root.manifests_dir / "batch-manifest.json", batch_manifest)
    benchmark_receipts: dict[str, JsonObject] = {}
    for model_name in ("llama-3.1-8b", "mistral-7b-v0.1", "olmo-2-7b"):
        validation_payload = _validation_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
        )
        benchmark_payload = _benchmark_receipt_payload(
            model_name=model_name,
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            validation_receipt_sha256=_json_identity(validation_payload),
        )
        benchmark_receipts[model_name] = benchmark_payload
        load_existing_or_write(
            root.receipts_dir / f"validation.{model_name}.json", validation_payload
        )
        load_existing_or_write(
            root.receipts_dir / f"benchmark.{model_name}.json", benchmark_payload
        )
    load_existing_or_write(
        root.receipts_dir / "lane-admission.json",
        _lane_receipt_payload(
            batch_manifest=batch_manifest,
            config_sha256=resolved["config_sha256"],
            sweep_sha256=resolved["sweep_sha256"],
            benchmark_receipts=benchmark_receipts,
        ),
    )
    _write_launch_context(monkeypatch, run_root=run_root, batch_id="batch", physical_gpu=1)
    monkeypatch.setattr(
        "si_rebuttal.runner._load_sequences_from_run_root",
        FixedDomainSequenceLoader(
            DomainSequences(
                fit_tokens=_repeated_sequences(50), eval_tokens=_repeated_sequences(100)
            )
        ),
    )
    monkeypatch.setattr(
        "si_rebuttal.runner.load_local_model_bundle",
        _return_constant(
            _loaded_bundle(batch_manifest=batch_manifest, model_name="mistral-7b-v0.1")
        ),
    )
    monkeypatch.setattr(
        "si_rebuttal.runner._capture_fit_summary",
        _return_constant(
            FakeCaptureSummary(
                mean_scores=np.zeros((32, 32), dtype=np.float64),
                raw_kernel_sums=np.ones((32, 32, 512), dtype=np.float64),
                sequence_count=50,
            )
        ),
    )
    monkeypatch.setattr("si_rebuttal.runner._eval_nlls", EvalNllRecorder(calls=[]))
    run_model(
        "rebuttal/configs/base.toml",
        "rebuttal/configs/sweep.toml",
        run_root=run_root,
        model_names=("mistral-7b-v0.1",),
        execute=True,
        runtime_command_runner=_runtime_command_runner(),
    )
    profile_path = root.manifests_dir / "fit-profiles" / "mistral-7b-v0.1.wikipedia.json"
    original_profile_payload = json.loads(profile_path.read_text(encoding="ascii"))
    profile_payload = dict(original_profile_payload)
    profile_payload["materialized_domain_sha256"] = "forged"
    _overwrite_json(profile_path, profile_payload)
    with pytest.raises(Exception, match="Fit profile domain binding drifted"):
        run_model(
            "rebuttal/configs/base.toml",
            "rebuttal/configs/sweep.toml",
            run_root=run_root,
            model_names=("mistral-7b-v0.1",),
            execute=True,
            runtime_command_runner=_runtime_command_runner(),
        )
    _overwrite_json(profile_path, original_profile_payload)
    profile_payload = dict(original_profile_payload)
    profile_payload["materialized_domain_sha256"] = "forged"
    profile_path.write_text(
        json.dumps(profile_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n",
        encoding="ascii",
    )
    with pytest.raises(ArtifactError, match="identity mismatch"):
        run_model(
            "rebuttal/configs/base.toml",
            "rebuttal/configs/sweep.toml",
            run_root=run_root,
            model_names=("mistral-7b-v0.1",),
            execute=True,
            runtime_command_runner=_runtime_command_runner(),
        )
    _overwrite_json(profile_path, original_profile_payload)
    shard_path = root.shards_dir / "mistral-7b-v0.1.wikipedia_to_code.bin.0.source_kernel.json"
    shard_payload = json.loads(shard_path.read_text(encoding="ascii"))
    shard_payload["selected_head_digest_sha256"] = "forged"
    _overwrite_json(shard_path, shard_payload)
    with pytest.raises(Exception, match="Resume selected-head digest drifted"):
        run_model(
            "rebuttal/configs/base.toml",
            "rebuttal/configs/sweep.toml",
            run_root=run_root,
            model_names=("mistral-7b-v0.1",),
            execute=True,
            runtime_command_runner=_runtime_command_runner(),
        )


def test_finalize_batch_uses_all_three_control_trials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resolved = _resolved_digests(
        validate_config("rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml")
    )
    batch_manifest = _batch_manifest_payload()
    run_root = tmp_path / "filled"
    root = RunRoot(run_id=run_root.name, root=run_root, batch_id=None)
    root.ensure()
    load_existing_or_write(root.manifests_dir / "batch-manifest.json", batch_manifest)
    _populate_finalize_inventory(
        monkeypatch=monkeypatch, root=root, resolved=resolved, batch_manifest=batch_manifest
    )
    _write_expected_v1_summary(root, batch_manifest=batch_manifest)
    payload = finalize_batch(
        "rebuttal/configs/base.toml", "rebuttal/configs/sweep.toml", run_root=run_root
    )
    contrast = _finalize_contrast(
        payload, model="llama-3.1-8b", direction="wikipedia_to_code", bin_index=0
    ).payload
    assert (
        abs(_json_float_item(_json_object_item(contrast, "source_minus_offset"), "point") - 2.0)
        < 1e-12
    )
    assert (
        abs(_json_float_item(_json_object_item(contrast, "source_minus_norm"), "point") - 1.0)
        < 1e-12
    )
    depth = _finalize_depth_relationship(payload, model="llama-3.1-8b").payload
    assert np.isfinite(_json_float_item(depth, "p_two_sided_scipy"))
    assert _json_int_item(depth, "positive_null_permutation_count") == 200000
    assert _json_list_item(
        _json_object_item(depth, "permutation_seed_provenance"), "seed_parts"
    ) == [
        29039,
        "llama-3.1-8b",
        "wikipedia_to_code",
        "statistic",
        "all",
        0,
        "spearman_response_permutation",
    ]
    assert len(_json_list_item(depth, "depth_row_identity_sha256s")) == 32


def test_cli_maps_subcommands_and_new_receipt_flow(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli_module.main(
        [
            "validate-config",
            "--config",
            "rebuttal/configs/base.toml",
            "--sweep",
            "rebuttal/configs/sweep.toml",
        ]
    )
    assert exit_code == 0
    payload = _json_object(json.loads(capsys.readouterr().out))
    assert payload["schema_version"] == 1

    parser = cli_module.build_parser()
    parsed = parser.parse_args(
        ["validate-model", "--run-root", str(tmp_path / "run"), "--model", "llama-3.1-8b"]
    )
    assert parsed.run_root


def test_cli_finalize_batch_verify_only_boundary_preserves_v2_and_surfaces_v1_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, batch_manifest, run_root, root = _prepare_finalizer_run(
        monkeypatch, tmp_path, run_name="verify-only-cli", write_v1_summary=True
    )
    v1_path = _summary_v1_path(root)
    v2_path = _summary_v2_path(root)
    v2_bytes = b'{"preserved":true,"schema_version":2}\n'
    v2_path.write_bytes(v2_bytes)
    expected_payload = _write_expected_v1_summary(root, batch_manifest=batch_manifest)
    v1_path.unlink()

    with pytest.raises(SystemExit) as missing_exc:
        cli_module.main(
            [
                "finalize-batch",
                "--config",
                "rebuttal/configs/base.toml",
                "--sweep",
                "rebuttal/configs/sweep.toml",
                "--run-root",
                str(run_root),
            ]
        )
    assert missing_exc.value.code == 2
    assert re.search(
        r"(?i)legacy finalize-batch is verify-only and requires an existing v1 summary\.",
        capsys.readouterr().err,
    )
    assert v2_path.read_bytes() == v2_bytes

    v1_path.write_text("{bad json}\n", encoding="ascii")
    with pytest.raises(SystemExit) as corrupt_exc:
        cli_module.main(
            [
                "finalize-batch",
                "--config",
                "rebuttal/configs/base.toml",
                "--sweep",
                "rebuttal/configs/sweep.toml",
                "--run-root",
                str(run_root),
            ]
        )
    assert corrupt_exc.value.code == 2
    assert "valid immutable terminal summary" in capsys.readouterr().err
    assert v2_path.read_bytes() == v2_bytes

    _overwrite_json(v1_path, expected_payload)
    exit_code = cli_module.main(
        [
            "finalize-batch",
            "--config",
            "rebuttal/configs/base.toml",
            "--sweep",
            "rebuttal/configs/sweep.toml",
            "--run-root",
            str(run_root),
        ]
    )
    assert exit_code == 0
    assert _json_object(json.loads(capsys.readouterr().out)) == expected_payload
    assert v2_path.read_bytes() == v2_bytes

    exit_code = cli_module.main(
        [
            "finalize-batch",
            "--config",
            "rebuttal/configs/base.toml",
            "--sweep",
            "rebuttal/configs/sweep.toml",
            "--run-root",
            str(run_root),
        ]
    )
    assert exit_code == 0
    assert _json_object(json.loads(capsys.readouterr().out)) == expected_payload
    assert v2_path.read_bytes() == v2_bytes


def test_cli_refinalize_batch_v2_correction_flow_uses_actual_command_and_preserves_existing_v2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, batch_manifest, run_root, root = _prepare_finalizer_run(
        monkeypatch, tmp_path, run_name="refinalize-batch-v2-cli", write_v1_summary=True
    )
    v1_path = _summary_v1_path(root)
    v2_path = _summary_v2_path(root)
    expected_v1 = _write_expected_v1_summary(root, batch_manifest=batch_manifest)
    _patch_refinalize_source_summary_v1_digest(monkeypatch, root)

    exit_code = cli_module.main(
        [
            "refinalize-batch-v2",
            "--config",
            "rebuttal/configs/base.toml",
            "--sweep",
            "rebuttal/configs/sweep.toml",
            "--run-root",
            str(run_root),
            "--source-summary-v1",
            str(v1_path),
        ]
    )
    assert exit_code == 0
    first_stdout = capsys.readouterr().out
    first_payload = _json_object(json.loads(first_stdout))
    first_bytes = v2_path.read_bytes()
    assert first_payload["schema_version"] == 2

    exit_code = cli_module.main(
        [
            "refinalize-batch-v2",
            "--config",
            "rebuttal/configs/base.toml",
            "--sweep",
            "rebuttal/configs/sweep.toml",
            "--run-root",
            str(run_root),
            "--source-summary-v1",
            str(v1_path),
        ]
    )
    assert exit_code == 0
    assert _json_object(json.loads(capsys.readouterr().out)) == first_payload
    assert v2_path.read_bytes() == first_bytes

    v2_path.unlink()
    v1_path.unlink()
    with pytest.raises(SystemExit) as missing_v1_exc:
        cli_module.main(
            [
                "refinalize-batch-v2",
                "--config",
                "rebuttal/configs/base.toml",
                "--sweep",
                "rebuttal/configs/sweep.toml",
                "--run-root",
                str(run_root),
                "--source-summary-v1",
                str(v1_path),
            ]
        )
    assert missing_v1_exc.value.code == 2
    assert re.search(
        r"Canonical production source-v1 path is missing or not a regular file\.",
        capsys.readouterr().err,
    )
    assert not v2_path.exists()

    _overwrite_json(v1_path, expected_v1)
    v2_path.write_text("{bad json}\n", encoding="ascii")
    with pytest.raises(SystemExit) as corrupt_v2_exc:
        cli_module.main(
            [
                "refinalize-batch-v2",
                "--config",
                "rebuttal/configs/base.toml",
                "--sweep",
                "rebuttal/configs/sweep.toml",
                "--run-root",
                str(run_root),
                "--source-summary-v1",
                str(v1_path),
            ]
        )
    assert corrupt_v2_exc.value.code == 2
    assert "valid immutable terminal summary" in capsys.readouterr().err
    assert v2_path.read_text(encoding="ascii") == "{bad json}\n"

    conflicting_v2_bytes = _artifact_bytes(
        _json_object({"schema_version": 2, "completed_at": "2026-07-30T12:00:00+00:00"})
    )
    v2_path.write_bytes(conflicting_v2_bytes)
    with pytest.raises(SystemExit) as conflicting_v2_exc:
        cli_module.main(
            [
                "refinalize-batch-v2",
                "--config",
                "rebuttal/configs/base.toml",
                "--sweep",
                "rebuttal/configs/sweep.toml",
                "--run-root",
                str(run_root),
                "--source-summary-v1",
                str(v1_path),
            ]
        )
    assert conflicting_v2_exc.value.code == 2
    assert capsys.readouterr().err == "Finalizer v2 payload drifted.\n"
    assert v2_path.read_bytes() == conflicting_v2_bytes

    v2_path.unlink()
    v1_path.write_text("{bad json}\n", encoding="ascii")
    with pytest.raises(SystemExit) as corrupt_v1_exc:
        cli_module.main(
            [
                "refinalize-batch-v2",
                "--config",
                "rebuttal/configs/base.toml",
                "--sweep",
                "rebuttal/configs/sweep.toml",
                "--run-root",
                str(run_root),
                "--source-summary-v1",
                str(v1_path),
            ]
        )
    assert corrupt_v1_exc.value.code == 2
    assert (
        capsys.readouterr().err
        == "Canonical production source-v1 raw byte digest drifted before v2 finalization.\n"
    )
    assert not v2_path.exists()


def test_cli_toy_smoke_emits_json_with_string_terminal_summary_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    terminal_summary_path = tmp_path / "toy-run" / "terminal-summary.json"

    monkeypatch.setattr(
        "si_rebuttal.cli.toy_smoke",
        _return_constant(
            FakeToySmokeResult(
                run_id="toy-run-id",
                shard_count=3,
                statistic_count=7,
                terminal_summary_path=terminal_summary_path,
            )
        ),
    )

    exit_code = cli_module.main(["toy-smoke", "--run-root", str(tmp_path / "toy-run")])

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert (
        stdout
        == '{"run_id":"toy-run-id","shard_count":3,"statistic_count":7,"terminal_summary_path":"'
        f"{terminal_summary_path}"
        '"}\n'
    )
    payload = _json_object(json.loads(stdout))
    assert _json_string_item(payload, "run_id") == "toy-run-id"
    assert _json_int_item(payload, "shard_count") == 3
    assert _json_int_item(payload, "statistic_count") == 7
    assert _json_string_item(payload, "terminal_summary_path") == str(terminal_summary_path)
