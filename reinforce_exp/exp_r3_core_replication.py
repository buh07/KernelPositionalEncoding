#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp.common import RESULTS_ROOT, command_manifest, ensure_dir, read_json, safe_float, timestamp_now, write_json  # noqa: E402

DEFAULT_OUT = RESULTS_ROOT / "exp_r3_core_replication"


def _run_cmd(cmd: list[str], log_path: Path) -> None:
    ensure_dir(log_path.parent)
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            logf.write(line)
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"Command failed ({code}): {' '.join(cmd)}")


def _stage_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    try:
        data = read_json(path)
    except Exception as exc:
        return {"status": "invalid_json", "path": str(path), "error": f"{type(exc).__name__}: {exc}"}
    return {"status": "ok", "path": str(path), "data": data}


def run_core_replication(
    *,
    model: str,
    device: str,
    output_root: Path,
    num_sequences_b: int,
    synthetic_b: int,
    num_sequences_j: int,
) -> dict[str, Any]:
    py = str(ROOT / ".venv" / "bin" / "python")
    out_root = output_root / model
    ensure_dir(out_root)
    log_root = ROOT / "logs" / "reinforce_exp" / "exp_r3_core_replication" / model / time.strftime("%Y%m%d_%H%M%S")
    ensure_dir(log_root)

    stages: list[dict[str, Any]] = []

    # Stage 1: Theory 8 (kernel ablation)
    cmd = [
        py,
        "-u",
        "experiment3/theory8_position_ablation.py",
        "--model",
        model,
        "--device",
        device,
        "--output-dir",
        str(out_root / "theory8_position_ablation"),
        "--estimation-seqs",
        "50",
        "--eval-seqs",
        "100",
        "--seq-len",
        "512",
    ]
    t0 = time.time()
    _run_cmd(cmd, log_root / "stage1_t8.log")
    stages.append({"stage": "theory8", "elapsed_sec": float(time.time() - t0), "cmd": cmd})

    # Stage 2: 3P2-B strict full (generic runner that does not require
    # legacy model-restricted Theory5b prerequisites).
    cmd = [
        py,
        "-u",
        "reinforce_exp/exp_r3_b_strict_generic.py",
        "--model",
        model,
        "--device",
        device,
        "--num-sequences",
        str(int(num_sequences_b)),
        "--seq-len",
        "512",
        "--top-k-dims",
        "16",
        "--synthetic-target-per-cell",
        str(int(synthetic_b)),
        "--seed",
        "0",
        "--output-root",
        str(out_root / "exp3p2b_strict_generic"),
    ]
    t0 = time.time()
    _run_cmd(cmd, log_root / "stage2_b.log")
    stages.append({"stage": "exp3p2b", "elapsed_sec": float(time.time() - t0), "cmd": cmd})

    # Stage 3: 3P2-C.1
    cmd = [
        py,
        "-u",
        "experiment3/phase2/exp3p2c_redundancy_quantification.py",
        "--model",
        model,
        "--device",
        device,
        "--output-root",
        str(out_root / "exp3p2c_redundancy_quantification"),
        "--num-seeds",
        "3",
        "--synthetic-count",
        "100",
        "--batch-size-synth",
        "8",
        "--ntp-count-per-seed",
        "100",
        "--ntp-seq-len",
        "512",
        "--batch-size-ntp",
        "4",
    ]
    t0 = time.time()
    _run_cmd(cmd, log_root / "stage3_c1.log")
    stages.append({"stage": "exp3p2c1", "elapsed_sec": float(time.time() - t0), "cmd": cmd})

    # Stage 4: 3P2-C.2 extended
    cmd = [
        py,
        "-u",
        "reinforce_exp/exp_r3_c2_extended.py",
        "--model",
        model,
        "--device",
        device,
        "--fractions",
        "0,25,50,75",
        "--output-root",
        str(out_root / "exp3p2c2_extended"),
        "--num-seeds",
        "3",
        "--synthetic-count",
        "100",
        "--batch-size-synth",
        "8",
        "--ntp-count-per-seed",
        "100",
        "--ntp-seq-len",
        "512",
        "--batch-size-ntp",
        "4",
    ]
    t0 = time.time()
    _run_cmd(cmd, log_root / "stage4_c2.log")
    stages.append({"stage": "exp3p2c2", "elapsed_sec": float(time.time() - t0), "cmd": cmd})

    # Stage 5: Generic conditional regime analysis (J-like)
    cmd = [
        py,
        "-u",
        "reinforce_exp/exp_r3_conditional_regimes_generic.py",
        "--model",
        model,
        "--device",
        device,
        "--num-sequences",
        str(int(num_sequences_j)),
        "--seq-len",
        "512",
        "--batch-size",
        "4",
        "--num-seed-shards",
        "16",
        "--long-span-min-distance",
        "64",
        "--min-regime-sample-count",
        "256",
        "--output-root",
        str(out_root / "conditional_regimes_generic"),
    ]
    t0 = time.time()
    _run_cmd(cmd, log_root / "stage5_jgeneric.log")
    stages.append({"stage": "exp3p2j_generic", "elapsed_sec": float(time.time() - t0), "cmd": cmd})

    # Collect outputs.
    t8_summary = _stage_result(out_root / "theory8_position_ablation" / model / "report.json")
    b_summary = _stage_result(out_root / "exp3p2b_strict_generic" / model / "synthetic_boundary_results.json")
    c1_summary = _stage_result(out_root / "exp3p2c_redundancy_quantification" / model / "curve_fit_comparison.json")
    c2_summary = _stage_result(out_root / "exp3p2c2_extended" / f"{model}_nonlinearity_summary.json")
    j_summary = _stage_result(out_root / "conditional_regimes_generic" / model / "regime_summary.json")

    b_prefix_flag = None
    if b_summary["status"] == "ok":
        b_prefix_flag = bool(b_summary["data"].get("prefix_following_artifact_flag", False))

    c1_supported = None
    if c1_summary["status"] == "ok":
        c1_supported = bool(c1_summary["data"].get("summary", {}).get("supports_redundancy_threshold_pattern", False))

    report = {
        "timestamp": timestamp_now(),
        "experiment": "EXP-R3",
        "model": model,
        "device": device,
        "stages": stages,
        "artifacts": {
            "theory8": t8_summary,
            "exp3p2b": b_summary,
            "exp3p2c1": c1_summary,
            "exp3p2c2": c2_summary,
            "exp3p2j_generic": j_summary,
        },
        "headline_verdict": {
            "kernel_load_bearing": bool(
                t8_summary["status"] == "ok"
                and safe_float(t8_summary["data"].get("analysis", {}).get("comparisons", {}).get("subtract_kernel_high_si", {}).get("mean_loss_increase")) > 0
            ),
            "boundary_strict_prefix_flag": b_prefix_flag,
            "c1_threshold_supported": c1_supported,
            "claim_impact": "strengthens_main_claim" if bool(c1_supported) else "no_change",
        },
        "log_root": str(log_root),
    }

    write_json(out_root / "core_replication_report.json", report)
    write_json(
        out_root / "manifest.json",
        command_manifest(
            experiment_id="EXP-R3",
            command="core_replication",
            model=model,
            extras={
                "device": device,
                "num_sequences_b": int(num_sequences_b),
                "synthetic_b": int(synthetic_b),
                "num_sequences_j": int(num_sequences_j),
                "output_root": str(out_root),
                "log_root": str(log_root),
            },
        ),
    )
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXP-R3: third primary-scale model core replication")
    p.add_argument("--model", default="mistral-7b-v0.1")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-sequences-b", type=int, default=32)
    p.add_argument("--synthetic-b", type=int, default=600)
    p.add_argument("--num-sequences-j", type=int, default=160)
    p.add_argument("--output-root", default=str(DEFAULT_OUT))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rep = run_core_replication(
        model=str(args.model),
        device=str(args.device),
        output_root=Path(args.output_root),
        num_sequences_b=max(8, int(args.num_sequences_b)),
        synthetic_b=max(64, int(args.synthetic_b)),
        num_sequences_j=max(16, int(args.num_sequences_j)),
    )
    print(f"[EXP-R3] wrote {Path(args.output_root) / args.model / 'core_replication_report.json'}")
    print(f"[EXP-R3] headline_verdict={rep['headline_verdict']}")


if __name__ == "__main__":
    main()
