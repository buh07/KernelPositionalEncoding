from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment2.interventions import build_rope_intervention_plan


def test_pair_interventions_are_supported_and_deterministic() -> None:
    plan = build_rope_intervention_plan(head_dim=128, intervention="ablate_pair_8", seed=7)
    assert plan.removed_indices == (8,)
    assert plan.target == "single_pair"

    rand_a = build_rope_intervention_plan(head_dim=128, intervention="random_pair", seed=7, random_draw=1)
    rand_b = build_rope_intervention_plan(head_dim=128, intervention="random_pair", seed=7, random_draw=1)
    rand_c = build_rope_intervention_plan(head_dim=128, intervention="random_pair", seed=7, random_draw=2)
    assert len(rand_a.removed_indices) == 1
    assert rand_a.removed_indices == rand_b.removed_indices
    assert rand_a.removed_indices != rand_c.removed_indices


def test_quick_manifest_builder_forces_spans_and_expands_random_draws(tmp_path: Path) -> None:
    base_manifest = tmp_path / "base.jsonl"
    rows = [
        {
            "phase": "phase2b",
            "split": "synthetic",
            "model": "llama-3.1-8b",
            "task": "long_range_retrieval",
            "seq_len": 1024,
            "intervention": "none",
            "seed": 0,
            "span": None,
            "random_draw": None,
            "run_id": "base",
        },
        {
            "phase": "phase2b",
            "split": "synthetic",
            "model": "llama-3.1-8b",
            "task": "long_range_retrieval",
            "seq_len": 1024,
            "intervention": "none",
            "seed": 1,
            "span": None,
            "random_draw": None,
            "run_id": "base",
        },
    ]
    base_manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    out_manifest = tmp_path / "quick.jsonl"
    cmd = [
        sys.executable,
        "scripts/experiment2_quick_manifest_build.py",
        "--input-manifest",
        str(base_manifest),
        "--output-manifest",
        str(out_manifest),
        "--run-id",
        "quick_run",
        "--phase",
        "quick_slope",
        "--model",
        "llama-3.1-8b",
        "--task",
        "long_range_retrieval",
        "--seeds",
        "0,1",
        "--force-spans",
        "32,64",
        "--source-interventions",
        "none",
        "--expand-interventions",
        "none,random_strong",
        "--random-draws",
        "3",
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])

    built = [
        json.loads(line)
        for line in out_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # rows = seeds(2) * spans(2) * conditions(1 none + 3 random draws)
    assert len(built) == 16
    assert all(r["phase"] == "quick_slope" for r in built)
    assert all(r["run_id"] == "quick_run" for r in built)
    assert sorted({int(r["span"]) for r in built}) == [32, 64]
    random_rows = [r for r in built if r["intervention"] == "random_strong"]
    assert sorted({int(r["random_draw"]) for r in random_rows}) == [0, 1, 2]
