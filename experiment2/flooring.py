from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from experiment2.long_offsets import active_fallback_spans

FLOOR_VERSION = "v2_lock_driven_fallback_spans"
FLOOR_THRESHOLD = 0.15


@dataclass(frozen=True)
class FloorDecision:
    key: str
    baseline_accuracy: float
    fallback_accuracy: float | None
    fallback_applied: bool
    floor_limited: bool
    fallback_spans: tuple[int, ...]
    floor_baseline_scope_policy: str = "global_task_feasibility"
    scoped_noop_baseline_accuracy: float | None = None
    scoped_noop_delta_vs_global: float | None = None
    scoped_noop_consistent: bool | None = None
    threshold: float = FLOOR_THRESHOLD
    rule_version: str = FLOOR_VERSION

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fallback_spans"] = list(self.fallback_spans)
        return payload


def floor_key(row: dict[str, Any]) -> str:
    parts = [
        str(row.get("phase")),
        str(row.get("split")),
        str(row.get("model")),
        str(row.get("task")),
        str(row.get("seq_len")),
        str(row.get("seed")),
        str(row.get("span")),
    ]
    return "|".join(parts)


def intervention_rank(name: str) -> int:
    # Enforce baseline-first ordering in execution.
    order = {
        "none": 0,
        "ablate_high_medium": 1,
        "ablate_low_medium": 2,
        "ablate_high_strong": 3,
        "ablate_low_strong": 4,
        "random_medium": 5,
        "random_strong": 6,
    }
    return order.get(name, 99)


def supports_fallback(task: str) -> bool:
    return task in {"delayed_copy", "long_range_retrieval"}


def fallback_spans_for_task(task: str, seq_len: int) -> tuple[int, ...]:
    if not supports_fallback(task):
        return tuple()
    return tuple(s for s in active_fallback_spans(strict=False) if 0 < s < seq_len)
