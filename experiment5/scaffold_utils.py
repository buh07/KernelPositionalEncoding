from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def now_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

