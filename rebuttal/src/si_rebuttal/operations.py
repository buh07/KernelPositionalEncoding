from __future__ import annotations

import csv
import hashlib
import math
import re
import shlex
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, TypedDict

from .artifacts import (
    JsonObject,
    RunRoot,
    atomic_write_bytes_no_clobber,
    atomic_write_json_no_clobber,
    build_batch_completion_payload,
)
from .config import ResolvedConfig, SweepConfig


class OperationsError(RuntimeError):
    """Raised when GPU/tmux placement or launch invariants fail."""


TMUX_GPU0 = "si-rebuttal-gpu0"
TMUX_GPU1 = "si-rebuttal-gpu1"
GPU_IDLE_SLEEP_SECONDS = 5.0
GPU_MEMORY_IDLE_THRESHOLD_MIB = 16.0
LAUNCH_RECEIPT_ENV = "SI_REBUTTAL_LAUNCH_RECEIPT"
BATCH_RECEIPT_ENV = "SI_REBUTTAL_BATCH_RECEIPT"
TMUX_SESSION_ENV = "SI_REBUTTAL_TMUX_SESSION"
RUN_ROOT_ENV = "SI_REBUTTAL_RUN_ROOT"
BATCH_ID_ENV = "SI_REBUTTAL_BATCH_ID"
RUNS_ROOT_ENV = "SI_REBUTTAL_RUNS_ROOT"
DATA_ROOT_ENV = "SI_REBUTTAL_DATA_ROOT"
MODELS_ROOT_ENV = "SI_REBUTTAL_MODELS_ROOT"
LOGS_ROOT_ENV = "SI_REBUTTAL_LOGS_ROOT"
CACHE_ROOT_ENV = "SI_REBUTTAL_CACHE_ROOT"
CompletedTextProcessRunner: TypeAlias = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
_CANONICAL_GPU_UUID_RE = re.compile(
    r"^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$"
)
_ASCII_PID_RE = re.compile(r"^[1-9][0-9]*$")
_ASCII_INT_RE = re.compile(r"^[0-9]+$")
_ASCII_DECIMAL_RE = re.compile(r"^(0|[1-9][0-9]*)(\.[0-9]+)?$")


@dataclass(frozen=True)
class GpuSnapshot:
    physical_index: int
    uuid: str
    name: str
    memory_used_mib: float
    utilization_gpu_pct: float
    compute_processes: tuple[str, ...]

    @property
    def is_idle(self) -> bool:
        return (
            self.memory_used_mib <= GPU_MEMORY_IDLE_THRESHOLD_MIB
            and self.utilization_gpu_pct == 0.0
            and not self.compute_processes
        )


class GpuSnapshotPayload(TypedDict):
    physical_index: int
    uuid: str
    name: str
    memory_used_mib: float
    utilization_gpu_pct: float
    compute_processes: list[str]


@dataclass(frozen=True)
class LaunchCommand:
    tmux_session: str
    physical_gpu: int
    model_names: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    command: tuple[str, ...]
    script_path: Path
    log_path: Path

    @property
    def shell_command(self) -> str:
        return shell_join(self.command)

    @property
    def script_body(self) -> str:
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
        ]
        for key, value in self.environment:
            lines.append(f"export {key}={shlex.quote(value)}")
        lines.append(f"exec {self.shell_command}")
        return "\n".join(lines) + "\n"

    @property
    def script_sha256(self) -> str:
        return hashlib.sha256(self.script_body.encode("ascii")).hexdigest()

    @property
    def tmux_command(self) -> str:
        return (
            f"bash {shlex.quote(str(self.script_path))} >> {shlex.quote(str(self.log_path))} 2>&1"
        )


@dataclass(frozen=True)
class LaunchPlan:
    batch_id: str
    run_root: Path
    commands: tuple[LaunchCommand, ...]


class ReadOnlyStatusPayload(TypedDict):
    run_root: str
    exists: bool
    tmux_sessions: dict[str, bool]
    receipts: list[str]


def gpu_snapshot_payload(snapshot: GpuSnapshot) -> GpuSnapshotPayload:
    return {
        "physical_index": snapshot.physical_index,
        "uuid": snapshot.uuid,
        "name": snapshot.name,
        "memory_used_mib": snapshot.memory_used_mib,
        "utilization_gpu_pct": snapshot.utilization_gpu_pct,
        "compute_processes": list(snapshot.compute_processes),
    }


def _parse_nonblank_csv_rows(csv_text: str, *, field_count: int, row_kind: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw_line in csv_text.splitlines():
        if not raw_line.strip():
            continue
        try:
            parsed_rows = list(csv.reader([raw_line], strict=True))
        except csv.Error as exc:
            raise OperationsError(f"Malformed {row_kind} CSV row: {raw_line!r}") from exc
        if len(parsed_rows) != 1:
            raise OperationsError(f"Malformed {row_kind} CSV row: {raw_line!r}")
        row = [field.strip() for field in parsed_rows[0]]
        if len(row) != field_count:
            raise OperationsError(f"Malformed {row_kind} CSV row: {raw_line!r}")
        rows.append(row)
    return rows


def _parse_nonnegative_int(text: str, *, field_name: str) -> int:
    if not _ASCII_INT_RE.fullmatch(text):
        raise OperationsError(f"Malformed {field_name}: {text!r}")
    try:
        return int(text)
    except ValueError as exc:
        raise OperationsError(f"Malformed {field_name}: {text!r}") from exc


def _parse_nonnegative_decimal(text: str, *, field_name: str) -> float:
    if not _ASCII_DECIMAL_RE.fullmatch(text):
        raise OperationsError(f"Malformed {field_name}: {text!r}")
    try:
        value = float(text)
    except ValueError as exc:
        raise OperationsError(f"Malformed {field_name}: {text!r}") from exc
    if not math.isfinite(value) or value < 0.0:
        raise OperationsError(f"Invalid {field_name}: {text!r}")
    return value


def _parse_utilization_pct(text: str) -> float:
    value = _parse_nonnegative_decimal(text, field_name="GPU utilization")
    if value > 100.0:
        raise OperationsError(f"Invalid GPU utilization: {text!r}")
    return value


def _parse_canonical_gpu_uuid(text: str) -> str:
    if not _CANONICAL_GPU_UUID_RE.fullmatch(text):
        raise OperationsError(f"Malformed GPU UUID: {text!r}")
    return text


def parse_nvidia_smi_snapshot(
    inventory_csv_text: str, *, physical_index: int, compute_apps_csv_text: str = ""
) -> GpuSnapshot:
    if physical_index < 0:
        raise OperationsError(f"Invalid physical GPU index: {physical_index}")

    inventory_rows = _parse_nonblank_csv_rows(
        inventory_csv_text, field_count=5, row_kind="nvidia-smi inventory"
    )
    matched_inventory_row: list[str] | None = None
    for row in inventory_rows:
        row_index = _parse_nonnegative_int(row[0], field_name="GPU index")
        if row_index != physical_index:
            raise OperationsError(
                f"Invalid nvidia-smi inventory GPU index: expected "
                f"{physical_index}, got {row_index}"
            )
        if matched_inventory_row is not None:
            raise OperationsError(
                f"Invalid nvidia-smi inventory rows: duplicate physical GPU {physical_index}"
            )
        matched_inventory_row = row
    if matched_inventory_row is None:
        raise OperationsError(
            f"Invalid nvidia-smi inventory rows: missing physical GPU {physical_index}"
        )

    uuid = _parse_canonical_gpu_uuid(matched_inventory_row[1])
    name = matched_inventory_row[2]
    if not name:
        raise OperationsError("Malformed GPU name: blank")
    memory_used_mib = _parse_nonnegative_decimal(
        matched_inventory_row[3], field_name="GPU memory used"
    )
    utilization_gpu_pct = _parse_utilization_pct(matched_inventory_row[4])

    compute_processes: list[str] = []
    if compute_apps_csv_text.strip():
        compute_rows = _parse_nonblank_csv_rows(
            compute_apps_csv_text, field_count=2, row_kind="nvidia-smi compute-apps"
        )
        for row in compute_rows:
            row_uuid = _parse_canonical_gpu_uuid(row[0])
            if row_uuid != uuid:
                raise OperationsError(
                    f"Invalid nvidia-smi compute-apps GPU UUID: expected {uuid!r}, got {row_uuid!r}"
                )
            pid = row[1]
            if not _ASCII_PID_RE.fullmatch(pid):
                raise OperationsError(f"Malformed compute PID: {pid!r}")
            compute_processes.append(pid)

    return GpuSnapshot(
        physical_index=physical_index,
        uuid=uuid,
        name=name,
        memory_used_mib=memory_used_mib,
        utilization_gpu_pct=utilization_gpu_pct,
        compute_processes=tuple(compute_processes),
    )


def _run_nvidia_smi_command(
    run_command: CompletedTextProcessRunner, command: Sequence[str]
) -> subprocess.CompletedProcess[str]:
    try:
        proc = run_command(command)
    except Exception as exc:
        raise OperationsError(f"nvidia-smi invocation failed: {exc}") from exc
    if proc.returncode != 0:
        raise OperationsError(proc.stderr.strip() or "nvidia-smi failed.")
    return proc


def snapshot_gpu(*, physical_index: int, run_command: CompletedTextProcessRunner) -> GpuSnapshot:
    inventory_proc = _run_nvidia_smi_command(
        run_command,
        (
            "nvidia-smi",
            f"--id={physical_index}",
            "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ),
    )
    compute_apps_proc = _run_nvidia_smi_command(
        run_command,
        (
            "nvidia-smi",
            f"--id={physical_index}",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ),
    )
    return parse_nvidia_smi_snapshot(
        inventory_proc.stdout,
        physical_index=physical_index,
        compute_apps_csv_text=compute_apps_proc.stdout,
    )


def require_two_idle_snapshots(
    *,
    physical_index: int,
    run_command: CompletedTextProcessRunner,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[GpuSnapshot, GpuSnapshot]:
    first = snapshot_gpu(physical_index=physical_index, run_command=run_command)
    if not first.is_idle:
        raise OperationsError(f"GPU {physical_index} is not idle on first snapshot.")
    sleep_fn(GPU_IDLE_SLEEP_SECONDS)
    second = snapshot_gpu(physical_index=physical_index, run_command=run_command)
    if not second.is_idle:
        raise OperationsError(f"GPU {physical_index} is not idle on second snapshot.")
    return first, second


def require_tmux_absent(*, session_name: str, run_command: CompletedTextProcessRunner) -> None:
    proc = run_command(("tmux", "has-session", "-t", session_name))
    if proc.returncode == 0:
        raise OperationsError(f"tmux session already exists: {session_name}")


def shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _project_relative_root(*, env_name: str, root_path: Path, project_root: Path) -> str:
    try:
        return str(root_path.resolve().relative_to(project_root.resolve()))
    except ValueError as exc:
        raise OperationsError(f"{env_name} must remain within project_root: {root_path}") from exc


def _append_environment_entry(
    entries: list[tuple[str, str]], seen: dict[str, str], *, key: str, value: str
) -> None:
    existing = seen.get(key)
    if existing is not None:
        if existing != value:
            raise OperationsError(f"Conflicting launch environment binding for {key}")
        raise OperationsError(f"Duplicate launch environment binding for {key}")
    seen[key] = value
    entries.append((key, value))


def _resolved_config_launch_environment(
    resolved_config: ResolvedConfig,
) -> tuple[tuple[str, str], ...]:
    paths = resolved_config.paths
    entries: list[tuple[str, str]] = []
    seen: dict[str, str] = {}
    for key, value in (
        (
            RUNS_ROOT_ENV,
            _project_relative_root(
                env_name=RUNS_ROOT_ENV,
                root_path=paths.runs_root,
                project_root=paths.project_root,
            ),
        ),
        (
            DATA_ROOT_ENV,
            _project_relative_root(
                env_name=DATA_ROOT_ENV,
                root_path=paths.data_root,
                project_root=paths.project_root,
            ),
        ),
        (MODELS_ROOT_ENV, str(paths.models_root.resolve())),
        (
            LOGS_ROOT_ENV,
            _project_relative_root(
                env_name=LOGS_ROOT_ENV,
                root_path=paths.logs_root,
                project_root=paths.project_root,
            ),
        ),
        (
            CACHE_ROOT_ENV,
            _project_relative_root(
                env_name=CACHE_ROOT_ENV,
                root_path=paths.cache_root,
                project_root=paths.project_root,
            ),
        ),
    ):
        _append_environment_entry(entries, seen, key=key, value=value)
    for model in resolved_config.models.values():
        _append_environment_entry(
            entries,
            seen,
            key=model.weight_env,
            value=str(model.weights_path.resolve()),
        )
        _append_environment_entry(
            entries,
            seen,
            key=model.tokenizer_env,
            value=str(model.tokenizer_path.resolve()),
        )
    return tuple(entries)


def _launch_environment(
    *,
    physical_gpu: int,
    resolved_config: ResolvedConfig,
    receipts_dir: Path,
    batch_receipt_path: Path,
    tmux_session: str,
    run_root: Path,
    batch_id: str,
) -> tuple[tuple[str, str], ...]:
    entries: list[tuple[str, str]] = []
    seen: dict[str, str] = {}
    for key, value in (
        ("CUDA_VISIBLE_DEVICES", str(physical_gpu)),
        *_resolved_config_launch_environment(resolved_config),
        (LAUNCH_RECEIPT_ENV, str(receipts_dir / f"{tmux_session}.json")),
        (BATCH_RECEIPT_ENV, str(batch_receipt_path)),
        (TMUX_SESSION_ENV, tmux_session),
        (RUN_ROOT_ENV, str(run_root)),
        (BATCH_ID_ENV, batch_id),
    ):
        _append_environment_entry(entries, seen, key=key, value=value)
    return tuple(entries)


def build_launch_plan(
    *,
    resolved_config: ResolvedConfig,
    sweep: SweepConfig,
    batch_id: str,
    run_root: Path,
    execute: bool,
    python_executable: str,
) -> LaunchPlan:
    receipts_dir = run_root / "receipts"
    scripts_dir = run_root / "launch-scripts"
    logs_dir = run_root / "logs"
    batch_receipt_path = receipts_dir / "batch.json"
    gpu0 = LaunchCommand(
        tmux_session=TMUX_GPU0,
        physical_gpu=0,
        model_names=tuple(sweep.placement_gpu0),
        environment=_launch_environment(
            physical_gpu=0,
            resolved_config=resolved_config,
            receipts_dir=receipts_dir,
            batch_receipt_path=batch_receipt_path,
            tmux_session=TMUX_GPU0,
            run_root=run_root,
            batch_id=batch_id,
        ),
        command=(
            python_executable,
            "-m",
            "si_rebuttal",
            "run-model",
            "--config",
            str(resolved_config.source_path),
            "--sweep",
            str(sweep.source_path),
            "--run-root",
            str(run_root),
            "--models",
            ",".join(sweep.placement_gpu0),
            *(("--execute",) if execute else ("--dry-run",)),
        ),
        script_path=scripts_dir / f"{TMUX_GPU0}.sh",
        log_path=logs_dir / f"{TMUX_GPU0}.log",
    )
    gpu1 = LaunchCommand(
        tmux_session=TMUX_GPU1,
        physical_gpu=1,
        model_names=tuple(sweep.placement_gpu1),
        environment=_launch_environment(
            physical_gpu=1,
            resolved_config=resolved_config,
            receipts_dir=receipts_dir,
            batch_receipt_path=batch_receipt_path,
            tmux_session=TMUX_GPU1,
            run_root=run_root,
            batch_id=batch_id,
        ),
        command=(
            python_executable,
            "-m",
            "si_rebuttal",
            "run-model",
            "--config",
            str(resolved_config.source_path),
            "--sweep",
            str(sweep.source_path),
            "--run-root",
            str(run_root),
            "--models",
            ",".join(sweep.placement_gpu1),
            *(("--execute",) if execute else ("--dry-run",)),
        ),
        script_path=scripts_dir / f"{TMUX_GPU1}.sh",
        log_path=logs_dir / f"{TMUX_GPU1}.log",
    )
    return LaunchPlan(batch_id=batch_id, run_root=run_root, commands=(gpu0, gpu1))


def write_launch_scripts_and_logs(*, run_root: RunRoot, plan: LaunchPlan) -> None:
    run_root.ensure()
    for launch in plan.commands:
        if launch.script_path.exists():
            raise OperationsError(f"Launch script already exists: {launch.script_path}")
        if launch.log_path.exists():
            raise OperationsError(f"Launch log already exists: {launch.log_path}")
    for launch in plan.commands:
        atomic_write_bytes_no_clobber(launch.script_path, launch.script_body.encode("ascii"))
        atomic_write_bytes_no_clobber(launch.log_path, b"")


def write_launch_receipts(
    *,
    run_root: RunRoot,
    plan: LaunchPlan,
    snapshots: Mapping[int, tuple[GpuSnapshot, GpuSnapshot]],
    command: Sequence[str],
    config_sha256: str,
    sweep_sha256: str,
) -> None:
    run_root.ensure()
    for launch in plan.commands:
        receipt = {
            "schema_version": 1,
            "batch_id": plan.batch_id,
            "tmux_session": launch.tmux_session,
            "physical_gpu": launch.physical_gpu,
            "model_names": list(launch.model_names),
            "command": list(launch.command),
            "command_shell": launch.shell_command,
            "snapshots": [
                snapshots[launch.physical_gpu][0].__dict__,
                snapshots[launch.physical_gpu][1].__dict__,
            ],
        }
        atomic_write_json_no_clobber(run_root.receipts_dir / f"{launch.tmux_session}.json", receipt)
    completion: JsonObject = build_batch_completion_payload(
        batch_id=plan.batch_id,
        run_ids=(run_root.run_id,),
        command=command,
        config_sha256=config_sha256,
        sweep_sha256=sweep_sha256,
    )
    atomic_write_json_no_clobber(run_root.receipts_dir / "batch.json", completion)


def launch_tmux_plan(*, plan: LaunchPlan, run_command: CompletedTextProcessRunner) -> None:
    if not plan.run_root.exists():
        raise OperationsError(f"Launch run root must already exist: {plan.run_root}")
    if not plan.run_root.is_dir():
        raise OperationsError(f"Launch run root is not a directory: {plan.run_root}")
    created_sessions: list[str] = []
    for launch in plan.commands:
        create = run_command(
            ("tmux", "new-session", "-d", "-s", launch.tmux_session, launch.tmux_command)
        )
        if create.returncode != 0:
            launch_failure = (
                create.stderr.strip() or f"tmux launch failed for {launch.tmux_session}."
            )
            cleanup_failures: list[str] = []
            for session_name in reversed(created_sessions):
                cleanup = run_command(("tmux", "kill-session", "-t", session_name))
                if cleanup.returncode != 0:
                    cleanup_failures.append(
                        cleanup.stderr.strip() or f"tmux rollback failed for {session_name}."
                    )
            if cleanup_failures:
                raise OperationsError(
                    f"{launch_failure} Rollback also failed: {cleanup_failures[0]}"
                )
            raise OperationsError(launch_failure)
        created_sessions.append(launch.tmux_session)


def read_only_status(
    *, run_root: Path, run_command: CompletedTextProcessRunner
) -> ReadOnlyStatusPayload:
    payload: ReadOnlyStatusPayload = {
        "run_root": str(run_root),
        "exists": run_root.exists(),
        "tmux_sessions": {},
        "receipts": (
            sorted(path.name for path in (run_root / "receipts").glob("*.json"))
            if run_root.exists()
            else []
        ),
    }
    for session_name in (TMUX_GPU0, TMUX_GPU1):
        proc = run_command(("tmux", "has-session", "-t", session_name))
        payload["tmux_sessions"][session_name] = proc.returncode == 0
    return payload


def default_subprocess_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def available_disk_bytes(path: str | Path) -> int:
    return int(shutil.disk_usage(Path(path)).free)
