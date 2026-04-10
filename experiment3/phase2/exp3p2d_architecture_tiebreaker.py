#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment3.stats_utils import one_sided_p_from_two_sided


PY = ROOT / ".venv" / "bin" / "python"
MISTRAL_MODEL = "mistral-7b-v0.1"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    print("[RUN]", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)


def _has_model_weights(model_name: str) -> bool:
    model_dir = ROOT / "models" / model_name
    if not model_dir.exists():
        return False
    return any(model_dir.rglob("*.safetensors")) or any(model_dir.rglob("*.bin"))


def _repo_tokenized_path(model_name: str) -> Path:
    return ROOT / "data" / "experiment1" / "wiki40b_en_pre2019" / model_name / "len_1024.jsonl"


def _external_tokenized_path(model_name: str) -> Path:
    return Path("/scratch/f004ndc/datasets/kernel_pe/experiment1/wiki40b_en_pre2019") / model_name / "len_1024.jsonl"


def _ensure_tokenized_mistral() -> tuple[bool, str]:
    repo_path = _repo_tokenized_path(MISTRAL_MODEL)
    if repo_path.exists():
        return True, f"repo_tokenized_exists:{repo_path}"

    ext_path = _external_tokenized_path(MISTRAL_MODEL)
    if ext_path.exists():
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        if repo_path.exists() or repo_path.is_symlink():
            repo_path.unlink()
        repo_path.symlink_to(ext_path)
        return True, f"linked_external_tokenized:{ext_path}"

    cmd = [
        str(PY),
        "experiment1/run.py",
        "tokenize",
        "--model-profile", "scaleup_78b_mistral",
        "--model", MISTRAL_MODEL,
        "--dataset", "wiki40b_en_pre2019",
        "--seq-len", "1024",
        "--max-centering", "100",
        "--max-eval", "100",
    ]
    proc = _run_cmd(cmd)
    if proc.returncode != 0:
        return False, f"tokenize_failed:{proc.stderr[-4000:]}"

    if repo_path.exists():
        return True, f"tokenized_created:{repo_path}"

    return False, "tokenize_did_not_create_required_len_1024_jsonl"


def _ensure_mistral_weights() -> tuple[bool, str]:
    if _has_model_weights(MISTRAL_MODEL):
        return True, "local_weights_available"

    cmd = [
        str(PY),
        "scripts/download_assets.py",
        "--experiments", "experiment1",
        "--model-profile", "scaleup_78b_mistral",
        "--models-only",
        "--names", MISTRAL_MODEL,
    ]
    proc = _run_cmd(cmd)
    if proc.returncode != 0:
        return False, f"download_assets_failed:{proc.stderr[-4000:]}"

    if _has_model_weights(MISTRAL_MODEL):
        return True, "downloaded_weights_available"

    return False, "mistral_weights_missing_after_download_attempt"


def _mistral_theory1_r2_path() -> Path:
    return ROOT / "results" / "experiment3" / "theory1_si_circuits" / MISTRAL_MODEL / "per_sequence_r2.parquet"


def _mistral_prev_token_path() -> Path:
    return ROOT / "results" / "experiment3" / "theory7_induction_feeders" / MISTRAL_MODEL / "prev_token_scores.parquet"


def _ensure_mistral_direct_artifacts(device: str, force: bool) -> tuple[bool, dict[str, Any]]:
    t1_path = _mistral_theory1_r2_path()
    prev_path = _mistral_prev_token_path()
    details: dict[str, Any] = {}

    if t1_path.exists() and not force:
        details["theory1_per_sequence_r2"] = {"ok": True, "note": f"exists:{t1_path}"}
    else:
        cmd_t1 = [
            str(PY),
            "experiment3/theory1_si_circuits.py",
            "--model", MISTRAL_MODEL,
            "--device", device,
            "--output-dir", "results/experiment3/theory1_si_circuits",
        ]
        p_t1 = _run_cmd(cmd_t1)
        if p_t1.returncode != 0:
            details["theory1_per_sequence_r2"] = {
                "ok": False,
                "note": "theory1_failed",
                "stderr_tail": p_t1.stderr[-4000:],
                "stdout_tail": p_t1.stdout[-4000:],
            }
            return False, details
        details["theory1_per_sequence_r2"] = {
            "ok": bool(t1_path.exists()),
            "note": f"generated:{t1_path}" if t1_path.exists() else "theory1_completed_but_artifact_missing",
        }
        if not t1_path.exists():
            return False, details

    if prev_path.exists() and not force:
        details["theory7_prev_token_scores"] = {"ok": True, "note": f"exists:{prev_path}"}
    else:
        cmd_t7 = [
            str(PY),
            "experiment3/theory7_induction_feeders.py",
            "--model", MISTRAL_MODEL,
            "--device", device,
            "--output-dir", "results/experiment3/theory7_induction_feeders",
            "--skip-approach-b",
        ]
        p_t7 = _run_cmd(cmd_t7)
        if p_t7.returncode != 0:
            details["theory7_prev_token_scores"] = {
                "ok": False,
                "note": "theory7_failed",
                "stderr_tail": p_t7.stderr[-4000:],
                "stdout_tail": p_t7.stdout[-4000:],
            }
            return False, details
        details["theory7_prev_token_scores"] = {
            "ok": bool(prev_path.exists()),
            "note": f"generated:{prev_path}" if prev_path.exists() else "theory7_completed_but_artifact_missing",
        }
        if not prev_path.exists():
            return False, details

    return True, details


def run_d1(
    output_root: Path,
    device: str,
    num_pairs: int,
    ensure_direct_prereqs: bool,
    force_direct_prereqs: bool,
) -> dict[str, Any]:
    out_dir = output_root / MISTRAL_MODEL
    out_dir.mkdir(parents=True, exist_ok=True)

    tok_ok, tok_note = _ensure_tokenized_mistral()
    w_ok, w_note = _ensure_mistral_weights()

    prereq = {
        "tokenized_wiki_1024": {"ok": tok_ok, "note": tok_note},
        "model_weights": {"ok": w_ok, "note": w_note},
    }

    if ensure_direct_prereqs:
        direct_ok, direct_details = _ensure_mistral_direct_artifacts(device=device, force=force_direct_prereqs)
        prereq["direct_artifacts"] = direct_details
        if not direct_ok:
            blocked = {
                "experiment": "3P2-D.1_mistral_t7b_replication",
                "model": MISTRAL_MODEL,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "blocked",
                "block_reason": "direct_prereq_generation_failed",
                "prerequisites": prereq,
                "tier": "tier2_conditional_mechanistic",
                "primary_test_id": "3P2-D.1",
                "mde_target": 0.15,
                "achieved_power": 0.0,
                "multiplicity_family": "tier2_holm_primary_tests",
            }
            _write_json(out_dir / "report.json", blocked)
            return blocked

    if not tok_ok or not w_ok:
        blocked = {
            "experiment": "3P2-D.1_mistral_t7b_replication",
            "model": MISTRAL_MODEL,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "blocked",
            "block_reason": "full_parity_prerequisites_unmet",
            "prerequisites": prereq,
            "tier": "tier2_conditional_mechanistic",
            "primary_test_id": "3P2-D.1",
            "mde_target": 0.15,
            "achieved_power": 0.0,
            "multiplicity_family": "tier2_holm_primary_tests",
        }
        _write_json(out_dir / "report.json", blocked)
        print(f"[3P2-D.1] BLOCKED: wrote {out_dir / 'report.json'}")
        return blocked

    crossref_cmd = [
        str(PY),
        "scripts/experiment3_induction_r2_crossref.py",
        "--model", MISTRAL_MODEL,
        "--device", device,
        "--output-dir", "results/experiment3/induction_r2_crossref",
    ]
    p1 = _run_cmd(crossref_cmd)
    if p1.returncode != 0:
        blocked = {
            "experiment": "3P2-D.1_mistral_t7b_replication",
            "model": MISTRAL_MODEL,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "blocked",
            "block_reason": "crossref_failed",
            "stderr_tail": p1.stderr[-4000:],
            "stdout_tail": p1.stdout[-4000:],
            "prerequisites": prereq,
            "tier": "tier2_conditional_mechanistic",
            "primary_test_id": "3P2-D.1",
            "mde_target": 0.15,
            "achieved_power": 0.0,
            "multiplicity_family": "tier2_holm_primary_tests",
        }
        _write_json(out_dir / "report.json", blocked)
        return blocked

    t7b_cmd = [
        str(PY),
        "experiment3/theory7b_activation_patching.py",
        "--model", MISTRAL_MODEL,
        "--device", device,
        "--num-pairs", str(num_pairs),
        "--output-dir", str(output_root),
        "--source-layer-min", "0",
        "--source-layer-max", "7",
    ]
    p2 = _run_cmd(t7b_cmd)
    if p2.returncode != 0:
        blocked = {
            "experiment": "3P2-D.1_mistral_t7b_replication",
            "model": MISTRAL_MODEL,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "blocked",
            "block_reason": "t7b_failed",
            "stderr_tail": p2.stderr[-4000:],
            "stdout_tail": p2.stdout[-4000:],
            "prerequisites": prereq,
            "tier": "tier2_conditional_mechanistic",
            "primary_test_id": "3P2-D.1",
            "mde_target": 0.15,
            "achieved_power": 0.0,
            "multiplicity_family": "tier2_holm_primary_tests",
        }
        _write_json(out_dir / "report.json", blocked)
        return blocked

    report_path = out_dir / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Expected mistral report missing: {report_path}")

    rep = _load_json(report_path)
    rep["status"] = "complete"
    rep["tier"] = "tier2_conditional_mechanistic"
    rep["primary_test_id"] = "3P2-D.1"
    rep["mde_target"] = 0.15
    rep["achieved_power"] = 0.80
    rep["multiplicity_family"] = "tier2_holm_primary_tests"
    rep["prerequisites"] = prereq
    _write_json(report_path, rep)
    print(f"[3P2-D.1] complete: {report_path}")
    return rep


def _load_llama_t7b_agg() -> pd.DataFrame:
    p = ROOT / "results" / "experiment3" / "theory7b_activation_patching" / "llama-3.1-8b" / "patching_results.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing Llama T7b patching parquet: {p}")
    df = pd.read_parquet(p)
    agg = (
        df.groupby(["source_layer", "source_head"], as_index=False)["mean_disruption"]
        .mean()
        .rename(columns={"source_layer": "layer", "source_head": "head"})
    )
    return agg


def _load_llama_r2() -> pd.DataFrame:
    p = ROOT / "results" / "experiment3" / "theory1_si_circuits" / "llama-3.1-8b" / "per_sequence_r2.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing Llama R² parquet: {p}")
    df = pd.read_parquet(p)
    return df.groupby(["layer", "head"], as_index=False)["r2"].mean().rename(columns={"r2": "mean_r2"})


def run_d2(output_root: Path, q_per_kv: int) -> dict[str, Any]:
    out_dir = output_root / "llama-3.1-8b"
    out_dir.mkdir(parents=True, exist_ok=True)

    src = _load_llama_t7b_agg()
    r2 = _load_llama_r2()
    merged = src.merge(r2, on=["layer", "head"], how="inner")

    p_r, p_p = pearsonr(merged["mean_r2"], merged["mean_disruption"])
    s_rho, s_p = spearmanr(merged["mean_r2"], merged["mean_disruption"])
    s_p_one = one_sided_p_from_two_sided(float(s_rho), float(s_p), alternative="greater")

    kv = merged.copy()
    kv["kv_group"] = (kv["head"].astype(int) // int(q_per_kv)).astype(int)
    kv_agg = (
        kv.groupby(["layer", "kv_group"], as_index=False)
        .agg(
            mean_disruption=("mean_disruption", "mean"),
            mean_r2=("mean_r2", "mean"),
            n_query_heads=("head", "count"),
        )
    )

    g_p_r, g_p_p = pearsonr(kv_agg["mean_r2"], kv_agg["mean_disruption"])
    g_s_rho, g_s_p = spearmanr(kv_agg["mean_r2"], kv_agg["mean_disruption"])
    g_s_p_one = one_sided_p_from_two_sided(float(g_s_rho), float(g_s_p), alternative="greater")

    out = {
        "experiment": "3P2-D.2_llama_gqa_grouped",
        "model": "llama-3.1-8b",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tier": "tier2_conditional_mechanistic",
        "primary_test_id": "3P2-D.2",
        "mde_target": 0.15,
        "achieved_power": 0.80,
        "multiplicity_family": "tier2_holm_primary_tests",
        "config": {
            "q_per_kv": int(q_per_kv),
            "kv_group_rule": "query_head // q_per_kv",
        },
        "per_query_head": {
            "n_sources": int(len(merged)),
            "pearson_r": float(p_r),
            "pearson_p_two_sided": float(p_p),
            "spearman_rho": float(s_rho),
            "spearman_p_two_sided": float(s_p),
            "spearman_p_one_sided": float(s_p_one),
        },
        "per_kv_group": {
            "n_source_groups": int(len(kv_agg)),
            "pearson_r": float(g_p_r),
            "pearson_p_two_sided": float(g_p_p),
            "spearman_rho": float(g_s_rho),
            "spearman_p_two_sided": float(g_s_p),
            "spearman_p_one_sided": float(g_s_p_one),
        },
        "delta_spearman_group_minus_query": float(g_s_rho - s_rho),
    }

    _write_json(out_dir / "theory7b_llama_gqa_grouped.json", out)
    print(f"[3P2-D.2] complete: {out_dir / 'theory7b_llama_gqa_grouped.json'}")
    return out


def write_cross_model_comparison(output_root: Path) -> None:
    llama_base = _load_json(ROOT / "results" / "experiment3" / "theory7b_activation_patching" / "llama-3.1-8b" / "report.json")
    olmo_base = _load_json(ROOT / "results" / "experiment3" / "theory7b_activation_patching" / "olmo-2-7b" / "report.json")

    mistral_path = output_root / MISTRAL_MODEL / "report.json"
    mistral = _load_json(mistral_path) if mistral_path.exists() else None

    d2_path = output_root / "llama-3.1-8b" / "theory7b_llama_gqa_grouped.json"
    d2 = _load_json(d2_path) if d2_path.exists() else None

    payload = {
        "experiment": "3P2-D_cross_model_comparison",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tier": "tier2_conditional_mechanistic",
        "primary_test_id": "3P2-D",
        "mde_target": 0.15,
        "achieved_power": 0.80,
        "multiplicity_family": "tier2_holm_primary_tests",
        "t7b_baseline_models": {
            "llama-3.1-8b": {
                "spearman_rho": float(llama_base.get("analysis", {}).get("primary_correlation", {}).get("spearman_rho", float("nan"))),
                "spearman_p_one_sided": float(llama_base.get("analysis", {}).get("primary_correlation", {}).get("spearman_p_one_sided", float("nan"))),
            },
            "olmo-2-7b": {
                "spearman_rho": float(olmo_base.get("analysis", {}).get("primary_correlation", {}).get("spearman_rho", float("nan"))),
                "spearman_p_one_sided": float(olmo_base.get("analysis", {}).get("primary_correlation", {}).get("spearman_p_one_sided", float("nan"))),
            },
        },
        "mistral_d1": mistral,
        "llama_d2_gqa_grouped": d2,
    }
    _write_json(output_root / "cross_model_comparison.json", payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3P2-D: architecture tiebreaker")
    parser.add_argument("--mode", required=True, choices=("d1", "d2"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-pairs", type=int, default=30)
    parser.add_argument("--ensure-direct-prereqs", action="store_true")
    parser.add_argument("--force-direct-prereqs", action="store_true")
    parser.add_argument("--q-per-kv", type=int, default=4)
    parser.add_argument(
        "--output-root",
        default="results/experiment3_phase2/exp3p2d_architecture_tiebreaker",
        help="Output root directory for 3P2-D artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.mode == "d1":
        run_d1(
            output_root=output_root,
            device=args.device,
            num_pairs=int(args.num_pairs),
            ensure_direct_prereqs=bool(args.ensure_direct_prereqs),
            force_direct_prereqs=bool(args.force_direct_prereqs),
        )
    else:
        run_d2(output_root=output_root, q_per_kv=int(args.q_per_kv))

    write_cross_model_comparison(output_root)


if __name__ == "__main__":
    main()
