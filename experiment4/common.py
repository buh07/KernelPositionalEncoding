from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class HeadIndex:
    layer: int
    head: int


def now_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_list()
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    return str(obj)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_parquet(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    ensure_dir(path.parent)
    if isinstance(rows, pd.DataFrame):
        df = rows
    else:
        df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def format_eta(seconds: float) -> str:
    s = max(0, int(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def mean_ci95(values: Iterable[float]) -> tuple[float, tuple[float, float]]:
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, (mean, mean)
    se = float(arr.std(ddof=1) / math.sqrt(arr.size))
    half = 1.96 * se
    return mean, (mean - half, mean + half)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return float("nan")
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = math.sqrt(max(1e-12, (((x.size - 1) * vx) + ((y.size - 1) * vy)) / max(1, x.size + y.size - 2)))
    return float((x.mean() - y.mean()) / pooled)


def clear_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
