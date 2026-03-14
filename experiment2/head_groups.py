from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

NATURAL_DATASETS = ("wiki40b_en_pre2019", "codesearchnet_python_snapshot")
TRACKB_GROUP_PREFERENCE = (
    "track_b_shared_mean_gpuopt_v1",
    "track_b_shared_mean_v1",
    "track_b_bucketed_mean_v1",
    "track_b_canonical_perpos_v1",
    "track_b",
)


def _source_group(results_root: Path) -> str:
    for group in TRACKB_GROUP_PREFERENCE:
        if (results_root / group).exists():
            return group
    raise FileNotFoundError("No Track B summary namespace found for head-group freezing.")


def _collect_track_a_r2(results_root: Path, model: str, group: str) -> pd.DataFrame:
    files: list[Path] = []
    for ds in NATURAL_DATASETS:
        p = results_root / group / model / ds / "len_1024" / "summary.parquet"
        if p.exists():
            files.append(p)
    if not files:
        raise FileNotFoundError(f"No natural len_1024 summaries found for model={model} group={group}")
    frames = [pd.read_parquet(p, columns=["layer", "head", "track_a_mean_r2"]) for p in files]
    df = pd.concat(frames, ignore_index=True)
    return df.groupby(["layer", "head"], as_index=False)["track_a_mean_r2"].mean()


def freeze_phase2d_head_groups(*, results_root: Path, run_id: str, models: tuple[str, ...] = ("olmo-1b", "llama-3.2-1b")) -> Path:
    source_group = _source_group(results_root)
    payload: dict[str, Any] = {
        "source_group": source_group,
        "natural_datasets": list(NATURAL_DATASETS),
        "selection": "top_bottom_quartile_by_track_a_mean_r2",
        "models": {},
    }
    for model in models:
        agg = _collect_track_a_r2(results_root, model, source_group)
        n_layers = int(agg["layer"].max()) + 1
        early_layers = [l for l in (0, 1) if l < n_layers]
        deep_start = int(math.floor(0.75 * n_layers))
        deep_layers = list(range(deep_start, n_layers))

        model_payload: dict[str, Any] = {}
        for scope, layers in (("early-only", early_layers), ("deep-only", deep_layers)):
            scoped = agg[agg["layer"].isin(layers)].copy()
            if scoped.empty:
                model_payload[scope] = {"high-kernel": [], "low-kernel": []}
                continue
            n = len(scoped)
            k = max(1, int(math.floor(0.25 * n)))
            high = scoped.sort_values("track_a_mean_r2", ascending=False).head(k)
            low = scoped.sort_values("track_a_mean_r2", ascending=True).head(k)
            model_payload[scope] = {
                "high-kernel": [
                    {"layer": int(r.layer), "head": int(r.head), "track_a_mean_r2": float(r.track_a_mean_r2)}
                    for r in high.itertuples(index=False)
                ],
                "low-kernel": [
                    {"layer": int(r.layer), "head": int(r.head), "track_a_mean_r2": float(r.track_a_mean_r2)}
                    for r in low.itertuples(index=False)
                ],
            }
        payload["models"][model] = {
            "num_layers": n_layers,
            "early_layers": early_layers,
            "deep_layers": deep_layers,
            "groups": model_payload,
        }

    out = results_root / "experiment2" / "phase2d" / run_id / "frozen_head_groups.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def load_target_heads(*, results_root: Path, run_id: str, model: str, scope: str, head_group: str) -> dict[int, set[int]]:
    path = results_root / "experiment2" / "phase2d" / run_id / "frozen_head_groups.json"
    if not path.exists():
        path = freeze_phase2d_head_groups(results_root=results_root, run_id=run_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    model_info = payload.get("models", {}).get(model)
    if not model_info:
        raise KeyError(f"Missing frozen head groups for model={model}")
    rows = model_info.get("groups", {}).get(scope, {}).get(head_group, [])
    mapping: dict[int, set[int]] = {}
    for row in rows:
        mapping.setdefault(int(row["layer"]), set()).add(int(row["head"]))
    return mapping
