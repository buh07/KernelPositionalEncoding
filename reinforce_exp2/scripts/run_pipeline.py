#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp2.common import LOGS_ROOT, RESULTS_ROOT, ensure_dir, file_sha256, timestamp_now, write_json  # noqa: E402
from reinforce_exp2.scripts._shared import parse_models_arg, patch_schema_validation_flag, validate_core_artifacts  # noqa: E402


def _script_path(name: str) -> Path:
    return ROOT / "reinforce_exp2" / "scripts" / name


def _truthy(raw: str) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_output_root_map(raw: str) -> dict[str, str]:
    if not str(raw).strip():
        return {}
    txt = str(raw).strip()
    if txt.startswith("{"):
        obj = json.loads(txt)
        return {str(k): str(v) for k, v in obj.items()}
    out: dict[str, str] = {}
    for tok in [x.strip() for x in txt.split(",") if x.strip()]:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        out[str(k).strip()] = str(v).strip()
    return out


def _parse_device_map(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in [x.strip() for x in str(raw).split(",") if x.strip()]:
        if ":" not in tok:
            continue
        model, device = tok.split(":", 1)
        out[str(model).strip()] = str(device).strip()
    return out


def _default_output_roots(mode: str) -> dict[str, Path]:
    suffix = "" if mode == "full" else "_smoke"
    roots: dict[str, Path] = {
        "preflight": RESULTS_ROOT / "preflight",
        "calibration_v1": RESULTS_ROOT / "calibration_splits",
        "A0_claim_hygiene": RESULTS_ROOT / "A0_claim_hygiene",
        "A1_evidence_consolidation": RESULTS_ROOT / "A1_evidence_consolidation",
        "A2_prediction_layer": RESULTS_ROOT / "A2_prediction_layer",
        "B1_kernel_taxonomy": RESULTS_ROOT / f"B1_kernel_taxonomy{suffix}",
        "B2a_head_alignment": RESULTS_ROOT / f"B2a_head_alignment{suffix}",
        "B2b_cluster_alignment": RESULTS_ROOT / f"B2b_cluster_alignment{suffix}",
        "B3_cluster_ablation": RESULTS_ROOT / f"B3_cluster_ablation{suffix}",
        "B4_confound_isolation": RESULTS_ROOT / f"B4_confound_isolation{suffix}",
        "C1_olmo_causal_trace": RESULTS_ROOT / f"C1_olmo_causal_trace{suffix}",
        "C2_carrier_class_predictions": RESULTS_ROOT / f"C2_carrier_class_predictions{suffix}",
    }
    return roots


def _required_primary_files(experiment_id: str, out_dir: Path) -> list[Path]:
    common = [
        out_dir / "preregistration.json",
        out_dir / "manifest.json",
        out_dir / "summary.json",
        out_dir / "claim_impact.json",
        out_dir / "data_dictionary.json",
    ]
    extra: list[Path] = []
    if experiment_id == "preflight":
        extra = [out_dir / "summary.json", out_dir / "fasttext_asset_manifest.json"]
        common = []
    elif experiment_id == "calibration_v1":
        extra = [
            out_dir / "calibration_v1_ids.parquet",
            out_dir / "calibration_v1_tokens.parquet",
            out_dir / "calibration_v1_manifest.json",
        ]
        common = []
    elif experiment_id == "A0_claim_hygiene":
        extra = [out_dir / "claim_matrix.csv", out_dir / "claim_matrix.json", out_dir / "wording_changes.md"]
    elif experiment_id == "A1_evidence_consolidation":
        extra = [out_dir / "evidence_registry.parquet", out_dir / "evidence_registry.json", out_dir / "discrepancy_log.md"]
    elif experiment_id == "A2_prediction_layer":
        extra = [out_dir / "predictions_preregistered.json"]
    elif experiment_id == "B1_kernel_taxonomy":
        extra = [
            out_dir / "kernel_taxonomy_summary.json",
            out_dir / "cluster_membership.parquet",
            out_dir / "prototype_similarity_matrix.parquet",
        ]
    elif experiment_id == "B2a_head_alignment":
        extra = [out_dir / "boundary_offset_profiles.parquet", out_dir / "head_alignment_scores.parquet"]
    elif experiment_id == "B2b_cluster_alignment":
        extra = [out_dir / "cluster_alignment_report.json", out_dir / "cluster_alignment_depth_control.json"]
    elif experiment_id == "B3_cluster_ablation":
        extra = [
            out_dir / "cluster_ablation_curve.parquet",
            out_dir / "curve_fit_by_condition.json",
            out_dir / "collapse_point_comparison.json",
            out_dir / "mechanism_disambiguation_verdict.json",
            out_dir / "matching_diagnostics.json",
            out_dir / "cluster_contribution_share.json",
        ]
    elif experiment_id == "B4_confound_isolation":
        extra = [out_dir / "within_layer_contrast.parquet", out_dir / "entropy_matched_contrast.parquet"]
    elif experiment_id == "C1_olmo_causal_trace":
        extra = [out_dir / "causal_trace_matrix.parquet", out_dir / "restoration_ratio_summary.json", out_dir / "head_set_interaction_model.json"]
    elif experiment_id == "C2_carrier_class_predictions":
        extra = [out_dir / "position_shuffle_results.parquet", out_dir / "long_context_results.parquet", out_dir / "carrier_class_prediction_tests.json"]
    return common + extra


def _artifact_ready(experiment_id: str, out_dir: Path) -> tuple[bool, list[str], bool]:
    required_files = _required_primary_files(experiment_id, out_dir)
    errors: list[str] = []
    for p in required_files:
        if not p.exists():
            errors.append(f"missing required artifact: {p}")
    schema_ok, schema_errors = validate_core_artifacts(out_dir) if experiment_id not in {"preflight", "calibration_v1"} else (True, [])
    if not schema_ok:
        errors.extend(schema_errors)
    return len(errors) == 0, errors, schema_ok


def _run_with_log(
    cmd: list[str],
    *,
    cwd: Path,
    log_path: Path,
    env_extra: dict[str, str] | None = None,
    dry_run: bool,
) -> int:
    ensure_dir(log_path.parent)
    if dry_run:
        print("[dry-run]", " ".join(cmd), ">", log_path, flush=True)
        return 0
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
        return int(proc.returncode)


def _run_split_parallel(
    *,
    task: dict[str, Any],
    cwd: Path,
    log_path: Path,
    env_extra: dict[str, str] | None,
    dry_run: bool,
) -> tuple[int, list[dict[str, Any]]]:
    ensure_dir(log_path.parent)
    stamp = log_path.stem
    split_name = str(task.get("split_name", "split")).strip().lower()
    substeps: list[dict[str, Any]] = []
    per_model_cmds = list(task.get("per_model_cmds", []))
    aggregate_cmd = list(task.get("aggregate_cmd", []))

    if dry_run:
        for item in per_model_cmds:
            model = str(item.get("model", "unknown"))
            model_log = log_path.parent / f"{stamp}__{split_name}_per_model_{model.replace('/', '_')}.log"
            substeps.append(
                {
                    "name": f"{split_name}_per_model::{model}",
                    "status": "dry_run",
                    "return_code": 0,
                    "start_ts": timestamp_now(),
                    "end_ts": timestamp_now(),
                    "elapsed_s": 0.0,
                    "log_path": str(model_log),
                    "command": item.get("cmd", []),
                    "resume_decision": "internal_partial_resume",
                }
            )
        agg_log = log_path.parent / f"{stamp}__{split_name}_aggregate.log"
        substeps.append(
            {
                "name": f"{split_name}_aggregate",
                "status": "dry_run",
                "return_code": 0,
                "start_ts": timestamp_now(),
                "end_ts": timestamp_now(),
                "elapsed_s": 0.0,
                "log_path": str(agg_log),
                "command": aggregate_cmd,
                "resume_decision": "run",
            }
        )
        with log_path.open("w", encoding="utf-8") as logf:
            logf.write(f"[dry-run] split_parallel {split_name}\n")
        return 0, substeps

    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)

    proc_rows: list[dict[str, Any]] = []
    failures = 0
    with log_path.open("w", encoding="utf-8") as orchestrator_log:
        orchestrator_log.write(f"[{split_name.upper()} split_parallel] starting per-model workers\n")
        for item in per_model_cmds:
            model = str(item.get("model", "unknown"))
            cmd = list(item.get("cmd", []))
            model_log = log_path.parent / f"{stamp}__{split_name}_per_model_{model.replace('/', '_')}.log"
            ensure_dir(model_log.parent)
            step_env = env.copy()
            step_env["REINFORCE_EXP2_SPLIT_SUBSTEP"] = f"{split_name}_per_model::{model}"
            step_env["REINFORCE_EXP2_LOG_PATH"] = str(model_log)
            start_ts = timestamp_now()
            start_t = time.time()
            fh = model_log.open("w", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=str(cwd), env=step_env, stdout=fh, stderr=subprocess.STDOUT, text=True)
            proc_rows.append(
                {
                    "name": f"{split_name}_per_model::{model}",
                    "command": cmd,
                    "log_path": str(model_log),
                    "start_ts": start_ts,
                    "start_t": start_t,
                    "proc": proc,
                    "fh": fh,
                    "resume_decision": "internal_partial_resume",
                }
            )
            orchestrator_log.write(f"[start] {model} log={model_log}\n")
            orchestrator_log.flush()

        for row in proc_rows:
            proc: subprocess.Popen[str] = row["proc"]  # type: ignore[assignment]
            rc = int(proc.wait())
            elapsed = float(time.time() - float(row["start_t"]))
            row["fh"].close()
            status = "ok" if rc == 0 else "failed"
            if rc != 0:
                failures += 1
            substeps.append(
                {
                    "name": row["name"],
                    "status": status,
                    "return_code": rc,
                    "start_ts": row["start_ts"],
                    "end_ts": timestamp_now(),
                    "elapsed_s": elapsed,
                    "log_path": row["log_path"],
                    "command": row["command"],
                    "resume_decision": row["resume_decision"],
                }
            )
            orchestrator_log.write(f"[end] {row['name']} rc={rc} elapsed_s={elapsed:.2f}\n")
            orchestrator_log.flush()

        if failures > 0:
            orchestrator_log.write(f"[{split_name.upper()} split_parallel] aborting aggregate due to {failures} per-model failures\n")
            orchestrator_log.flush()
            return 1, substeps

        agg_log = log_path.parent / f"{stamp}__{split_name}_aggregate.log"
        agg_env = env.copy()
        agg_env["REINFORCE_EXP2_SPLIT_SUBSTEP"] = f"{split_name}_aggregate"
        agg_env["REINFORCE_EXP2_LOG_PATH"] = str(agg_log)
        start_ts = timestamp_now()
        start_t = time.time()
        orchestrator_log.write(f"[start] aggregate log={agg_log}\n")
        orchestrator_log.flush()
        with agg_log.open("w", encoding="utf-8") as agg_fh:
            proc = subprocess.run(aggregate_cmd, cwd=str(cwd), env=agg_env, stdout=agg_fh, stderr=subprocess.STDOUT, text=True)
        rc = int(proc.returncode)
        elapsed = float(time.time() - start_t)
        status = "ok" if rc == 0 else "failed"
        substeps.append(
            {
                "name": f"{split_name}_aggregate",
                "status": status,
                "return_code": rc,
                "start_ts": start_ts,
                "end_ts": timestamp_now(),
                "elapsed_s": elapsed,
                "log_path": str(agg_log),
                "command": aggregate_cmd,
                "resume_decision": "run",
            }
        )
        orchestrator_log.write(f"[end] {split_name}_aggregate rc={rc} elapsed_s={elapsed:.2f}\n")
        orchestrator_log.flush()
        return rc, substeps


def _build_tasks(
    *,
    phase: str,
    mode: str,
    models_arg: str,
    device_map: str,
    python_bin: str,
    output_roots: dict[str, Path],
    strict_todo: bool,
    b1_execution_mode: str,
    b1_partial_root: str,
    b1_resume_partials: bool,
    b3_execution_mode: str,
    b3_partial_root: str,
    b3_resume_partials: bool,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    run_a = phase in {"A", "ALL"}
    run_b = phase in {"B", "ALL"}
    run_c = phase in {"C", "ALL"}

    if phase == "ALL":
        tasks.append(
            {
                "id": "preflight",
                "script": "run_preflight.py",
                "output_root": output_roots["preflight"],
                "cmd": [python_bin, str(_script_path("run_preflight.py")), "--output-root", str(output_roots["preflight"]), "--python-bin", python_bin],
                "depends_on": [],
            }
        )
        tasks.append(
            {
                "id": "calibration_v1",
                "script": "build_calibration_v1.py",
                "output_root": output_roots["calibration_v1"],
                "cmd": [python_bin, str(_script_path("build_calibration_v1.py")), "--output-root", str(output_roots["calibration_v1"]), "--model", "llama-3.1-8b"],
                "depends_on": ["preflight"],
            }
        )

    if run_a:
        tasks.extend(
            [
                {
                    "id": "A0_claim_hygiene",
                    "script": "run_a0_claim_hygiene.py",
                    "output_root": output_roots["A0_claim_hygiene"],
                    "cmd": [python_bin, str(_script_path("run_a0_claim_hygiene.py")), "--output-root", str(output_roots["A0_claim_hygiene"])],
                    "depends_on": ["preflight"] if phase == "ALL" else [],
                },
                {
                    "id": "A1_evidence_consolidation",
                    "script": "run_a1_evidence_consolidation.py",
                    "output_root": output_roots["A1_evidence_consolidation"],
                    "cmd": [python_bin, str(_script_path("run_a1_evidence_consolidation.py")), "--output-root", str(output_roots["A1_evidence_consolidation"])],
                    "depends_on": ["A0_claim_hygiene"],
                },
                {
                    "id": "A2_prediction_layer",
                    "script": "run_a2_prediction_layer.py",
                    "output_root": output_roots["A2_prediction_layer"],
                    "cmd": [python_bin, str(_script_path("run_a2_prediction_layer.py")), "--output-root", str(output_roots["A2_prediction_layer"])],
                    "depends_on": ["A1_evidence_consolidation"],
                },
            ]
        )

    if run_b:
        b_dep_prefix = ["calibration_v1"] if phase == "ALL" else []
        parsed_models = parse_models_arg(models_arg)
        parsed_device_map = _parse_device_map(device_map)
        b1_partial = str(Path(b1_partial_root)) if str(b1_partial_root).strip() else str(output_roots["B1_kernel_taxonomy"] / "partials")
        b3_partial = str(Path(b3_partial_root)) if str(b3_partial_root).strip() else str(output_roots["B3_cluster_ablation"] / "partials")
        tasks.extend(
            [
                {
                    "id": "B2a_head_alignment",
                    "script": "run_b2a_head_alignment.py",
                    "output_root": output_roots["B2a_head_alignment"],
                    "cmd": [python_bin, str(_script_path("run_b2a_head_alignment.py")), "--models", models_arg, "--output-root", str(output_roots["B2a_head_alignment"])],
                    "depends_on": b_dep_prefix,
                },
            ]
        )
        if str(b1_execution_mode).strip().lower() == "split_parallel":
            per_model_cmds: list[dict[str, Any]] = []
            for model in parsed_models:
                model_device = parsed_device_map.get(model, "")
                cmd = [
                    python_bin,
                    str(_script_path("run_b1_kernel_taxonomy.py")),
                    "--execution-mode",
                    "per_model",
                    "--single-model",
                    str(model),
                    "--models",
                    models_arg,
                    "--output-root",
                    str(output_roots["B1_kernel_taxonomy"]),
                    "--partial-root",
                    str(b1_partial),
                    "--resume-partials",
                    "true" if bool(b1_resume_partials) else "false",
                    "--calibration-root",
                    str(output_roots.get("calibration_v1", RESULTS_ROOT / "calibration_splits")),
                ]
                if model_device:
                    cmd.extend(["--device-map", f"{model}:{model_device}"])
                per_model_cmds.append({"model": model, "cmd": cmd})
            agg_cmd = [
                python_bin,
                str(_script_path("run_b1_kernel_taxonomy.py")),
                "--execution-mode",
                "aggregate",
                "--models",
                models_arg,
                "--output-root",
                str(output_roots["B1_kernel_taxonomy"]),
                "--partial-root",
                str(b1_partial),
                "--resume-partials",
                "true" if bool(b1_resume_partials) else "false",
                "--calibration-root",
                str(output_roots.get("calibration_v1", RESULTS_ROOT / "calibration_splits")),
            ]
            tasks.append(
                {
                    "id": "B1_kernel_taxonomy",
                    "script": "run_b1_kernel_taxonomy.py",
                    "output_root": output_roots["B1_kernel_taxonomy"],
                    "depends_on": b_dep_prefix,
                    "split_parallel": True,
                    "split_name": "b1",
                    "b1_partial_root": str(b1_partial),
                    "b1_resume_partials": bool(b1_resume_partials),
                    "per_model_cmds": per_model_cmds,
                    "aggregate_cmd": agg_cmd,
                }
            )
        else:
            tasks.append(
                {
                    "id": "B1_kernel_taxonomy",
                    "script": "run_b1_kernel_taxonomy.py",
                    "output_root": output_roots["B1_kernel_taxonomy"],
                    "cmd": [
                        python_bin,
                        str(_script_path("run_b1_kernel_taxonomy.py")),
                        "--execution-mode",
                        "full",
                        "--models",
                        models_arg,
                        "--output-root",
                        str(output_roots["B1_kernel_taxonomy"]),
                        "--partial-root",
                        str(b1_partial),
                        "--resume-partials",
                        "true" if bool(b1_resume_partials) else "false",
                        "--calibration-root",
                        str(output_roots.get("calibration_v1", RESULTS_ROOT / "calibration_splits")),
                        "--device-map",
                        device_map,
                    ],
                    "depends_on": b_dep_prefix,
                }
            )
        tasks.extend(
            [
                {
                    "id": "B2b_cluster_alignment",
                    "script": "run_b2b_cluster_alignment.py",
                    "output_root": output_roots["B2b_cluster_alignment"],
                    "cmd": [
                        python_bin,
                        str(_script_path("run_b2b_cluster_alignment.py")),
                        "--models",
                        models_arg,
                        "--output-root",
                        str(output_roots["B2b_cluster_alignment"]),
                        "--b1-root",
                        str(output_roots["B1_kernel_taxonomy"]),
                        "--b2a-root",
                        str(output_roots["B2a_head_alignment"]),
                    ],
                    "depends_on": ["B1_kernel_taxonomy", "B2a_head_alignment"],
                },
                {
                    "id": "B4_confound_isolation",
                    "script": "run_b4_confound_isolation.py",
                    "output_root": output_roots["B4_confound_isolation"],
                    "cmd": [
                        python_bin,
                        str(_script_path("run_b4_confound_isolation.py")),
                        "--models",
                        models_arg,
                        "--output-root",
                        str(output_roots["B4_confound_isolation"]),
                        "--b1-root",
                        str(output_roots["B1_kernel_taxonomy"]),
                        "--calibration-root",
                        str(output_roots.get("calibration_v1", RESULTS_ROOT / "calibration_splits")),
                        "--device-map",
                        device_map,
                    ],
                    "depends_on": ["B1_kernel_taxonomy"],
                },
                {
                    "id": "B3_cluster_ablation",
                    "script": "run_b3_cluster_ablation.py",
                    "output_root": output_roots["B3_cluster_ablation"],
                    "depends_on": ["B1_kernel_taxonomy", "B2a_head_alignment"],
                },
            ]
        )
        if str(b3_execution_mode).strip().lower() == "split_parallel":
            per_model_cmds_b3: list[dict[str, Any]] = []
            for model in parsed_models:
                model_device = parsed_device_map.get(model, "")
                cmd = [
                    python_bin,
                    str(_script_path("run_b3_cluster_ablation.py")),
                    "--execution-mode",
                    "per_model",
                    "--single-model",
                    str(model),
                    "--models",
                    models_arg,
                    "--output-root",
                    str(output_roots["B3_cluster_ablation"]),
                    "--partial-root",
                    str(b3_partial),
                    "--resume-partials",
                    "true" if bool(b3_resume_partials) else "false",
                    "--b1-root",
                    str(output_roots["B1_kernel_taxonomy"]),
                    "--b2a-root",
                    str(output_roots["B2a_head_alignment"]),
                    "--calibration-root",
                    str(output_roots.get("calibration_v1", RESULTS_ROOT / "calibration_splits")),
                ]
                if model_device:
                    cmd.extend(["--device-map", f"{model}:{model_device}"])
                per_model_cmds_b3.append({"model": model, "cmd": cmd})
            agg_cmd_b3 = [
                python_bin,
                str(_script_path("run_b3_cluster_ablation.py")),
                "--execution-mode",
                "aggregate",
                "--models",
                models_arg,
                "--output-root",
                str(output_roots["B3_cluster_ablation"]),
                "--partial-root",
                str(b3_partial),
                "--resume-partials",
                "true" if bool(b3_resume_partials) else "false",
                "--b1-root",
                str(output_roots["B1_kernel_taxonomy"]),
                "--b2a-root",
                str(output_roots["B2a_head_alignment"]),
                "--calibration-root",
                str(output_roots.get("calibration_v1", RESULTS_ROOT / "calibration_splits")),
            ]
            tasks[-1].update(
                {
                    "split_parallel": True,
                    "split_name": "b3",
                    "b3_partial_root": str(b3_partial),
                    "b3_resume_partials": bool(b3_resume_partials),
                    "per_model_cmds": per_model_cmds_b3,
                    "aggregate_cmd": agg_cmd_b3,
                }
            )
        else:
            tasks[-1]["cmd"] = [
                python_bin,
                str(_script_path("run_b3_cluster_ablation.py")),
                "--execution-mode",
                "full",
                "--models",
                models_arg,
                "--output-root",
                str(output_roots["B3_cluster_ablation"]),
                "--partial-root",
                str(b3_partial),
                "--resume-partials",
                "true" if bool(b3_resume_partials) else "false",
                "--b1-root",
                str(output_roots["B1_kernel_taxonomy"]),
                "--b2a-root",
                str(output_roots["B2a_head_alignment"]),
                "--calibration-root",
                str(output_roots.get("calibration_v1", RESULTS_ROOT / "calibration_splits")),
                "--device-map",
                device_map,
            ]

    if run_c:
        c_models = parse_models_arg(models_arg)
        c1_models = ",".join([m for m in c_models if m in {"llama-3.1-8b", "olmo-2-7b"}] or ["llama-3.1-8b", "olmo-2-7b"])
        tasks.extend(
            [
                {
                    "id": "C1_olmo_causal_trace",
                    "script": "run_c1_olmo_causal_trace.py",
                    "output_root": output_roots["C1_olmo_causal_trace"],
                    "cmd": [
                        python_bin,
                        str(_script_path("run_c1_olmo_causal_trace.py")),
                        "--models",
                        c1_models,
                        "--device-map",
                        device_map,
                        "--output-root",
                        str(output_roots["C1_olmo_causal_trace"]),
                    ],
                    "depends_on": ["B3_cluster_ablation", "B2b_cluster_alignment", "B4_confound_isolation"] if strict_todo else [],
                },
                {
                    "id": "C2_carrier_class_predictions",
                    "script": "run_c2_carrier_class_predictions.py",
                    "output_root": output_roots["C2_carrier_class_predictions"],
                    "cmd": [
                        python_bin,
                        str(_script_path("run_c2_carrier_class_predictions.py")),
                        "--models",
                        models_arg,
                        "--device-map",
                        device_map,
                        "--output-root",
                        str(output_roots["C2_carrier_class_predictions"]),
                    ],
                    "depends_on": ["B3_cluster_ablation", "B2b_cluster_alignment", "B4_confound_isolation"] if strict_todo else [],
                },
            ]
        )
    return tasks


def main() -> None:
    p = argparse.ArgumentParser(description="reinforce_exp2 strict pipeline launcher", allow_abbrev=False)
    p.add_argument("--phase", choices=["A", "B", "C", "ALL"], default="ALL")
    p.add_argument("--mode", choices=["full", "smoke"], default="full")
    p.add_argument("--models", default="llama-3.1-8b,olmo-2-7b,mistral-7b-v0.1")
    p.add_argument("--device-map", default="llama-3.1-8b:cuda:0,olmo-2-7b:cuda:1,mistral-7b-v0.1:cuda:2")
    p.add_argument("--python", default=str(ROOT / ".venv" / "bin" / "python"))
    p.add_argument("--resume-policy", choices=["strict_artifact_check", "none"], default="strict_artifact_check")
    p.add_argument("--output-root-map", default="")
    p.add_argument("--log-root", default=str(LOGS_ROOT))
    p.add_argument("--strict-todo", default="true")
    p.add_argument("--b1-execution-mode", choices=["split_parallel", "legacy"], default="split_parallel")
    p.add_argument("--b1-partial-root", default="")
    p.add_argument("--b1-resume-partials", default="true")
    p.add_argument("--b3-execution-mode", choices=["split_parallel", "legacy"], default="split_parallel")
    p.add_argument("--b3-partial-root", default="")
    p.add_argument("--b3-resume-partials", default="true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    strict_todo = _truthy(args.strict_todo)
    phase = str(args.phase).upper()
    mode = str(args.mode).lower()
    py = str(args.python)
    log_root = ensure_dir(Path(args.log_root))

    output_roots = _default_output_roots(mode)
    override = _parse_output_root_map(args.output_root_map)
    for k, v in override.items():
        output_roots[k] = Path(v)
    for pth in output_roots.values():
        ensure_dir(pth)

    tasks = _build_tasks(
        phase=phase,
        mode=mode,
        models_arg=str(args.models),
        device_map=str(args.device_map),
        python_bin=py,
        output_roots=output_roots,
        strict_todo=strict_todo,
        b1_execution_mode=str(args.b1_execution_mode),
        b1_partial_root=str(args.b1_partial_root),
        b1_resume_partials=_truthy(args.b1_resume_partials),
        b3_execution_mode=str(args.b3_execution_mode),
        b3_partial_root=str(args.b3_partial_root),
        b3_resume_partials=_truthy(args.b3_resume_partials),
    )

    marker_dir = ensure_dir(RESULTS_ROOT / "pipeline_runs")
    b_complete_marker = marker_dir / f"B_COMPLETE_{mode}.marker"
    if phase == "C" and strict_todo and not b_complete_marker.exists() and not args.dry_run:
        raise SystemExit(
            f"Strict C phase requires B completion marker: {b_complete_marker}. "
            "Run B phase first (or ALL) under matching mode."
        )

    status_by_id: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    failures = 0

    calib_manifest = output_roots["calibration_v1"] / "calibration_v1_manifest.json"
    calibration_sha = None
    if calib_manifest.exists():
        try:
            calibration_sha = json.loads(calib_manifest.read_text(encoding="utf-8")).get("sha256_ids_parquet")
        except Exception:
            calibration_sha = None

    for task in tasks:
        exp_id = str(task["id"])
        out_dir = Path(task["output_root"])
        deps = list(task.get("depends_on", []))

        dep_fail = [
            d
            for d in deps
            if (d not in status_by_id) or (status_by_id.get(d) not in {"ok", "skipped_existing", "dry_run"})
        ]
        if dep_fail:
            status_by_id[exp_id] = "skipped_dependency_failed"
            ts_now = timestamp_now()
            rows.append(
                {
                    "experiment_id": exp_id,
                    "status": "skipped_dependency_failed",
                    "return_code": 0,
                    "timestamp": ts_now,
                    "started_at": ts_now,
                    "ended_at": ts_now,
                    "elapsed_seconds": 0.0,
                    "depends_on": deps,
                    "dependency_failures": dep_fail,
                    "command": task.get("cmd", task.get("aggregate_cmd", [])),
                }
            )
            continue

        resume_decision = "run"
        if args.resume_policy == "strict_artifact_check":
            ready, errs, schema_ok = _artifact_ready(exp_id, out_dir)
            if ready:
                resume_decision = "skip_existing_valid"
                status_by_id[exp_id] = "skipped_existing"
                ts_now = timestamp_now()
                rows.append(
                    {
                        "experiment_id": exp_id,
                        "status": "skipped_existing",
                        "return_code": 0,
                        "timestamp": ts_now,
                        "started_at": ts_now,
                        "ended_at": ts_now,
                        "elapsed_seconds": 0.0,
                        "depends_on": deps,
                        "resume_decision": resume_decision,
                        "schema_validation_passed": schema_ok,
                        "command": task.get("cmd", task.get("aggregate_cmd", [])),
                    }
                )
                continue
            resume_decision = "rerun_missing_or_invalid"

        stamp = timestamp_now().replace(":", "-").replace(" ", "_")
        log_path = log_root / exp_id / f"{stamp}.log"
        dep_inputs = {"depends_on": deps, "output_root": str(out_dir)}
        env_extra = {
            "REINFORCE_EXP2_RUN_MODE": mode,
            "REINFORCE_EXP2_DEPENDENCY_INPUTS_JSON": json.dumps(dep_inputs),
            "REINFORCE_EXP2_RESUME_DECISION": resume_decision,
            "REINFORCE_EXP2_LOG_PATH": str(log_path),
        }
        if calibration_sha:
            env_extra["REINFORCE_EXP2_CALIBRATION_SPLIT_SHA256"] = str(calibration_sha)

        print(f"[pipeline] running {exp_id}", flush=True)
        task_started_at = timestamp_now()
        task_start_t = time.time()
        substeps: list[dict[str, Any]] = []
        if bool(task.get("split_parallel", False)):
            rc, substeps = _run_split_parallel(
                task=task,
                cwd=ROOT,
                log_path=log_path,
                env_extra=env_extra,
                dry_run=bool(args.dry_run),
            )
        else:
            rc = _run_with_log(task["cmd"], cwd=ROOT, log_path=log_path, env_extra=env_extra, dry_run=bool(args.dry_run))
        task_elapsed = float(time.time() - task_start_t)
        task_ended_at = timestamp_now()

        schema_ok_after = False
        if rc == 0 and (not args.dry_run) and exp_id not in {"preflight", "calibration_v1"}:
            ready_after, errs_after, schema_ok_after = _artifact_ready(exp_id, out_dir)
            if not ready_after:
                rc = 2
                with log_path.open("a", encoding="utf-8") as logf:
                    logf.write("\n[post-run validation] artifact contract failed:\n")
                    for e in errs_after:
                        logf.write(f"- {e}\n")
                patch_schema_validation_flag(out_dir, False)
            else:
                patch_schema_validation_flag(out_dir, bool(schema_ok_after))

        status = "dry_run" if (rc == 0 and args.dry_run) else ("ok" if rc == 0 else "failed")
        status_by_id[exp_id] = status
        if rc != 0:
            failures += 1
        rows.append(
            {
                "experiment_id": exp_id,
                "status": status,
                "return_code": int(rc),
                "timestamp": task_ended_at,
                "started_at": task_started_at,
                "ended_at": task_ended_at,
                "elapsed_seconds": task_elapsed,
                "depends_on": deps,
                "resume_decision": resume_decision,
                "schema_validation_passed": bool(schema_ok_after),
                "log_path": str(log_path),
                "command": task.get("cmd", task.get("aggregate_cmd", [])),
                "substeps": substeps,
                "substep_elapsed_seconds_total": float(sum(float(x.get("elapsed_s", 0.0)) for x in substeps)) if substeps else 0.0,
                "substep_elapsed_seconds_max": float(max((float(x.get("elapsed_s", 0.0)) for x in substeps), default=0.0)) if substeps else 0.0,
            }
        )

    out_dir = ensure_dir(RESULTS_ROOT / "pipeline_runs")
    payload = {
        "timestamp": timestamp_now(),
        "phase": phase,
        "mode": mode,
        "models": parse_models_arg(args.models),
        "device_map": str(args.device_map),
        "resume_policy": str(args.resume_policy),
        "strict_todo": bool(strict_todo),
        "b1_execution_mode": str(args.b1_execution_mode),
        "b1_partial_root": str(args.b1_partial_root),
        "b1_resume_partials": bool(_truthy(args.b1_resume_partials)),
        "b3_execution_mode": str(args.b3_execution_mode),
        "b3_partial_root": str(args.b3_partial_root),
        "b3_resume_partials": bool(_truthy(args.b3_resume_partials)),
        "log_root": str(log_root),
        "output_root_map": {k: str(v) for k, v in output_roots.items()},
        "n_tasks": int(len(tasks)),
        "n_failures": int(failures),
        "rows": rows,
    }
    out_path = out_dir / f"pipeline_{phase.lower()}_{mode}_{timestamp_now().replace(':', '-').replace(' ', '_')}.json"
    write_json(out_path, payload)
    print(f"[pipeline] wrote {out_path}")

    if failures == 0 and phase in {"B", "ALL"} and not args.dry_run:
        write_json(
            b_complete_marker,
            {
                "timestamp": timestamp_now(),
                "phase": phase,
                "mode": mode,
                "n_tasks": int(len(tasks)),
                "n_failures": int(failures),
                "source_pipeline_run": str(out_path),
            },
        )
        print(f"[pipeline] wrote B completion marker {b_complete_marker}")

    if failures > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
