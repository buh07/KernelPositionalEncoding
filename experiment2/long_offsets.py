from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LEGACY_LONG_OFFSETS = (128, 256, 384, 512)
LEGACY_FALLBACK_SPANS = (128, 192, 256)
LEGACY_ACTIVE_LONG_TASKS = ("delayed_copy", "long_range_retrieval")
LOCK_FILENAME = "long_offset_lock.json"
LOCK_PATH = Path(__file__).resolve().parent / LOCK_FILENAME


@dataclass(frozen=True)
class LongOffsetBundle:
    status: str
    long_offsets: tuple[int, ...]
    fallback_spans: tuple[int, ...]
    active_long_tasks: tuple[str, ...]
    long_task_policy: str
    lock_resolution_mode: str
    source_run_id: str | None
    threshold: float | None
    generated_at: str | None
    lock_hash: str
    lock_exists: bool

    @property
    def valid_for_confirmatory(self) -> bool:
        return (
            self.status == "ok"
            and len(self.long_offsets) >= 2
            and len(self.fallback_spans) >= 1
            and len(self.active_long_tasks) >= 1
        )


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_int_tuple(values: Any) -> tuple[int, ...]:
    out: list[int] = []
    if isinstance(values, (list, tuple)):
        for val in values:
            try:
                out.append(int(val))
            except Exception:
                continue
    return tuple(sorted(set(v for v in out if v > 0)))


def _normalize_task_tuple(values: Any) -> tuple[str, ...]:
    out: list[str] = []
    if isinstance(values, (list, tuple)):
        for val in values:
            token = str(val).strip()
            if not token:
                continue
            if token not in out:
                out.append(token)
    return tuple(out)


def _default_bundle() -> LongOffsetBundle:
    payload = {
        "status": "legacy_fallback",
        "selected_long_offsets": list(LEGACY_LONG_OFFSETS),
        "fallback_spans": list(LEGACY_FALLBACK_SPANS),
        "active_long_tasks": list(LEGACY_ACTIVE_LONG_TASKS),
        "long_task_policy": "strict_two_task",
        "lock_resolution_mode": "legacy_fallback",
    }
    return LongOffsetBundle(
        status="legacy_fallback",
        long_offsets=tuple(LEGACY_LONG_OFFSETS),
        fallback_spans=tuple(LEGACY_FALLBACK_SPANS),
        active_long_tasks=tuple(LEGACY_ACTIVE_LONG_TASKS),
        long_task_policy="strict_two_task",
        lock_resolution_mode="legacy_fallback",
        source_run_id=None,
        threshold=None,
        generated_at=None,
        lock_hash=_sha256(_canonical_json(payload)),
        lock_exists=False,
    )


def load_long_offset_bundle(*, strict: bool = False) -> LongOffsetBundle:
    if not LOCK_PATH.exists():
        if strict:
            raise RuntimeError(
                f"Missing long-offset lock artifact: {LOCK_PATH}. "
                "Run feasibility-sweep and lock-long-offsets --apply first."
            )
        return _default_bundle()

    payload = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    status = str(payload.get("status", "unknown"))
    long_offsets = _normalize_int_tuple(payload.get("selected_long_offsets", []))
    fallback_spans = _normalize_int_tuple(payload.get("fallback_spans", []))
    active_long_tasks = _normalize_task_tuple(payload.get("active_long_tasks", []))
    if not active_long_tasks:
        active_long_tasks = tuple(LEGACY_ACTIVE_LONG_TASKS)
    long_task_policy = str(payload.get("long_task_policy", "strict_two_task"))
    lock_resolution_mode = str(payload.get("lock_resolution_mode", "unknown"))
    bundle = LongOffsetBundle(
        status=status,
        long_offsets=long_offsets,
        fallback_spans=fallback_spans,
        active_long_tasks=active_long_tasks,
        long_task_policy=long_task_policy,
        lock_resolution_mode=lock_resolution_mode,
        source_run_id=(None if payload.get("source_run_id") is None else str(payload.get("source_run_id"))),
        threshold=(None if payload.get("threshold") is None else float(payload.get("threshold"))),
        generated_at=(None if payload.get("generated_at") is None else str(payload.get("generated_at"))),
        lock_hash=_sha256(_canonical_json(payload)),
        lock_exists=True,
    )
    if strict and not bundle.valid_for_confirmatory:
        raise RuntimeError(
            f"Invalid long-offset lock artifact at {LOCK_PATH}: "
            f"status={bundle.status} offsets={bundle.long_offsets} fallback={bundle.fallback_spans}. "
            "Run lock-long-offsets --apply with a passing sweep first."
        )
    if not bundle.long_offsets:
        if strict:
            raise RuntimeError(
                f"Long-offset lock artifact has no selected offsets: {LOCK_PATH}"
            )
        return _default_bundle()
    return bundle


def active_long_offsets(*, strict: bool = False) -> tuple[int, ...]:
    return load_long_offset_bundle(strict=strict).long_offsets


def active_fallback_spans(*, strict: bool = False) -> tuple[int, ...]:
    return load_long_offset_bundle(strict=strict).fallback_spans


def active_long_tasks(*, strict: bool = False) -> tuple[str, ...]:
    return load_long_offset_bundle(strict=strict).active_long_tasks
