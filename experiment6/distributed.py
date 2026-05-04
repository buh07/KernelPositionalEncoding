#!/usr/bin/env python3
"""Distributed orchestration utilities for Experiment 6.

Public entrypoints:
  python -m experiment6.distributed run
  python -m experiment6.distributed merge
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from experiment4.common import now_timestamp, write_json
from experiment6.config import TARGET_MODELS
from experiment6.pipeline import _build_comparison_summary


EXPERIMENT_ORDER: tuple[str, ...] = ("6a", "6b", "6c", "6d", "6e")

EXPERIMENT_SPECS: dict[str, dict[str, Any]] = {
    "6a": {
        "module": "experiment6.exp6a_gradient_routing",
        "output_dir": "exp6a_gradient_routing",
        "runs_file": "exp6a_runs.parquet",
        "comparison_file": "exp6a_comparison.json",
        "condition_x": "f_gradient_routed",
        "condition_y": "d_full_qlora_baseline",
        "uses_seeds": True,
    },
    "6b": {
        "module": "experiment6.exp6b_anti_localization",
        "output_dir": "exp6b_anti_localization",
        "runs_file": "exp6b_runs.parquet",
        "comparison_file": "exp6b_comparison.json",
        "condition_x": "g_anti_localization",
        "condition_y": "d_full_qlora_baseline",
        "uses_seeds": True,
    },
    "6c": {
        "module": "experiment6.exp6c_distillation",
        "output_dir": "exp6c_distillation",
        "runs_file": "exp6c_runs.parquet",
        "comparison_file": "exp6c_comparison.json",
        "condition_x": "h_si_student",
        "condition_y": "h_teacher_baseline",
        "uses_seeds": True,
    },
    "6d": {
        "module": "experiment6.exp6d_contrastive",
        "output_dir": "exp6d_contrastive",
        "runs_file": "exp6d_runs.parquet",
        "comparison_file": "exp6d_comparison.json",
        "condition_x": "i_contrastive_channel",
        "condition_y": "d_full_qlora_baseline",
        "uses_seeds": True,
    },
    "6e": {
        "module": "experiment6.exp6e_tokenizer",
        "output_dir": "exp6e_tokenizer",
        "summary_file": "exp6e_tokenizer_comparison.json",
        "uses_seeds": False,
    },
}


@dataclass(frozen=True)
class Job:
    experiment: str
    model: str
    seed: int

    @property
    def job_id(self) -> str:
        return f"{self.experiment}:{self.model}:seed{self.seed}"


def _parse_csv(raw: str) -> list[str]:
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


def _parse_int_csv(raw: str) -> list[int]:
    return [int(tok.strip()) for tok in str(raw).split(",") if tok.strip()]


def _ordered_unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _normalize_models(raw: str) -> list[str]:
    items = _parse_csv(raw)
    if len(items) == 1 and items[0].lower() == "all":
        return list(TARGET_MODELS)
    models = _ordered_unique(items)
    invalid = [m for m in models if m not in TARGET_MODELS]
    if invalid:
        raise ValueError(f"Unsupported model(s): {invalid}. Valid: {list(TARGET_MODELS)}")
    return models


def _normalize_experiments(raw: str) -> list[str]:
    items = _parse_csv(raw)
    if len(items) == 1 and items[0].lower() == "all":
        return list(EXPERIMENT_ORDER)
    exps = _ordered_unique(items)
    invalid = [e for e in exps if e not in EXPERIMENT_SPECS]
    if invalid:
        raise ValueError(f"Unsupported experiment(s): {invalid}. Valid: {list(EXPERIMENT_ORDER)}")
    return exps


def _build_jobs(models: list[str], experiments: list[str], seeds: list[int]) -> list[Job]:
    jobs: list[Job] = []
    exp_rank = {exp: idx for idx, exp in enumerate(EXPERIMENT_ORDER)}
    model_rank = {model: idx for idx, model in enumerate(models)}
    for model in models:
        for exp in experiments:
            spec = EXPERIMENT_SPECS[exp]
            if bool(spec["uses_seeds"]):
                for seed in seeds:
                    jobs.append(Job(experiment=exp, model=model, seed=int(seed)))
            else:
                jobs.append(Job(experiment=exp, model=model, seed=0))
    jobs.sort(key=lambda j: (model_rank.get(j.model, 999), exp_rank.get(j.experiment, 999), int(j.seed)))
    return jobs


def _job_output_root(shard_root: Path, job: Job) -> Path:
    spec = EXPERIMENT_SPECS[job.experiment]
    return shard_root / str(spec["output_dir"]) / f"{job.model}__seed{int(job.seed)}"


def _required_shard_artifacts(shard_root: Path, job: Job) -> list[Path]:
    spec = EXPERIMENT_SPECS[job.experiment]
    job_root = _job_output_root(shard_root, job) / job.model
    required = [job_root / "run_manifest.json"]
    if job.experiment in ("6a", "6b", "6c", "6d"):
        required.append(job_root / str(spec["runs_file"]))
        required.append(job_root / str(spec["comparison_file"]))
    else:
        required.append(job_root / str(spec["summary_file"]))
    return required


def _required_canonical_paths(canonical_root: Path, job: Job) -> dict[str, Path]:
    spec = EXPERIMENT_SPECS[job.experiment]
    base = canonical_root / str(spec["output_dir"]) / job.model
    out: dict[str, Path] = {"manifest": base / "run_manifest.json"}
    if job.experiment in ("6a", "6b", "6c", "6d"):
        out["runs"] = base / str(spec["runs_file"])
        out["comparison"] = base / str(spec["comparison_file"])
    else:
        out["summary"] = base / str(spec["summary_file"])
    return out


def _seed_complete_in_canonical(canonical_root: Path, job: Job) -> bool:
    paths = _required_canonical_paths(canonical_root, job)
    if job.experiment in ("6a", "6b", "6c", "6d"):
        runs_path = paths["runs"]
        if not runs_path.exists():
            return False
        try:
            df = pd.read_parquet(runs_path)
        except Exception:
            return False
        if "seed" not in df.columns or "condition" not in df.columns:
            return False
        spec = EXPERIMENT_SPECS[job.experiment]
        seed_df = df[df["seed"] == int(job.seed)]
        if seed_df.empty:
            return False
        conds = set(seed_df["condition"].astype(str).tolist())
        return str(spec["condition_x"]) in conds and str(spec["condition_y"]) in conds

    summary_path = paths["summary"]
    if not summary_path.exists():
        return False
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    summary_seed = payload.get("seed")
    if summary_seed is None:
        return True
    try:
        return int(summary_seed) == int(job.seed)
    except Exception:
        return False


def _job_complete(shard_root: Path, canonical_root: Path, job: Job) -> bool:
    shard_done = all(path.exists() for path in _required_shard_artifacts(shard_root, job))
    if shard_done:
        return True
    return _seed_complete_in_canonical(canonical_root, job)


def _build_subprocess_cmd(job: Job, shard_root: Path) -> list[str]:
    spec = EXPERIMENT_SPECS[job.experiment]
    output_root = _job_output_root(shard_root, job)
    output_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        str(spec["module"]),
        "--model",
        job.model,
        "--device",
        "cuda:0",
        "--output-root",
        str(output_root),
    ]
    if bool(spec["uses_seeds"]):
        cmd.extend(["--seeds", str(job.seed)])
    else:
        cmd.extend(["--seed", str(job.seed)])
    return cmd


def _run_jobs(
    *,
    jobs: list[Job],
    shard_root: Path,
    canonical_root: Path,
    gpu: int,
    retry_failed_once: bool,
    dry_run: bool,
) -> dict[str, Any]:
    started = time.time()
    shard_root.mkdir(parents=True, exist_ok=True)
    completed = 0
    skipped = 0
    failed = 0
    records: list[dict[str, Any]] = []

    if dry_run:
        for idx, job in enumerate(jobs, start=1):
            done = _job_complete(shard_root, canonical_root, job)
            records.append(
                {
                    "idx": idx,
                    "job_id": job.job_id,
                    "status": "skip_complete" if done else "would_run",
                    "gpu": int(gpu),
                    "cmd": _build_subprocess_cmd(job, shard_root),
                }
            )
        return {
            "timestamp": now_timestamp(),
            "gpu": int(gpu),
            "dry_run": True,
            "total_jobs": len(jobs),
            "records": records,
        }

    for idx, job in enumerate(jobs, start=1):
        if _job_complete(shard_root, canonical_root, job):
            skipped += 1
            print(f"[exp6-dist][gpu={gpu}] {idx}/{len(jobs)} SKIP complete {job.job_id}", flush=True)
            records.append({"idx": idx, "job_id": job.job_id, "status": "skip_complete", "gpu": int(gpu)})
            continue

        cmd = _build_subprocess_cmd(job, shard_root)
        print(f"[exp6-dist][gpu={gpu}] {idx}/{len(jobs)} RUN {job.job_id}", flush=True)
        rc = subprocess.call(cmd)
        if rc != 0 and retry_failed_once:
            print(f"[exp6-dist][gpu={gpu}] retrying once: {job.job_id}", flush=True)
            time.sleep(3)
            rc = subprocess.call(cmd)
        if rc == 0:
            completed += 1
            status = "completed"
        else:
            failed += 1
            status = f"failed_rc_{rc}"
        records.append({"idx": idx, "job_id": job.job_id, "status": status, "gpu": int(gpu), "cmd": cmd})
        if rc != 0:
            print(f"[exp6-dist][gpu={gpu}] FAIL {job.job_id} rc={rc}", flush=True)

    return {
        "timestamp": now_timestamp(),
        "gpu": int(gpu),
        "dry_run": False,
        "total_jobs": len(jobs),
        "completed": int(completed),
        "skipped": int(skipped),
        "failed": int(failed),
        "runtime_seconds": float(time.time() - started),
        "records": records,
    }


def _collect_runs_frames(
    *,
    experiment: str,
    model: str,
    shard_root: Path,
    canonical_root: Path,
) -> list[tuple[str, pd.DataFrame]]:
    spec = EXPERIMENT_SPECS[experiment]
    frames: list[tuple[str, pd.DataFrame]] = []

    canonical_runs = canonical_root / str(spec["output_dir"]) / model / str(spec["runs_file"])
    if canonical_runs.exists():
        try:
            frames.append(("canonical", pd.read_parquet(canonical_runs)))
        except Exception:
            pass

    shard_base = shard_root / str(spec["output_dir"])
    for path in sorted(shard_base.glob(f"*/{model}/{spec['runs_file']}")):
        try:
            frames.append((str(path), pd.read_parquet(path)))
        except Exception:
            continue
    return frames


def _collect_6e_records(
    *,
    model: str,
    shard_root: Path,
    canonical_root: Path,
) -> list[tuple[str, dict[str, Any]]]:
    spec = EXPERIMENT_SPECS["6e"]
    recs: list[tuple[str, dict[str, Any]]] = []

    canonical_summary = canonical_root / str(spec["output_dir"]) / model / str(spec["summary_file"])
    if canonical_summary.exists():
        try:
            recs.append(("canonical", json.loads(canonical_summary.read_text(encoding="utf-8"))))
        except Exception:
            pass

    shard_base = shard_root / str(spec["output_dir"])
    for path in sorted(shard_base.glob(f"*/{model}/{spec['summary_file']}")):
        try:
            recs.append((str(path), json.loads(path.read_text(encoding="utf-8"))))
        except Exception:
            continue
    return recs


def _merge_one(
    *,
    experiment: str,
    model: str,
    shard_root: Path,
    canonical_root: Path,
    seeds_filter: set[int] | None,
) -> dict[str, Any]:
    spec = EXPERIMENT_SPECS[experiment]
    out_dir = canonical_root / str(spec["output_dir"]) / model
    out_dir.mkdir(parents=True, exist_ok=True)

    if experiment in ("6a", "6b", "6c", "6d"):
        frames = _collect_runs_frames(
            experiment=experiment,
            model=model,
            shard_root=shard_root,
            canonical_root=canonical_root,
        )
        if not frames:
            return {"experiment": experiment, "model": model, "status": "no_data"}

        tagged: list[pd.DataFrame] = []
        for src, frame in frames:
            tmp = frame.copy()
            tmp["_source_priority"] = 0 if src == "canonical" else 1
            tagged.append(tmp)
        merged = pd.concat(tagged, ignore_index=True, sort=False)
        if "seed" not in merged.columns or "condition" not in merged.columns:
            return {"experiment": experiment, "model": model, "status": "invalid_frame"}
        merged = merged.sort_values(["seed", "condition", "_source_priority"]).drop_duplicates(
            subset=["seed", "condition"], keep="last"
        )
        merged = merged.drop(columns=["_source_priority"], errors="ignore")
        if seeds_filter:
            merged = merged[merged["seed"].astype(int).isin(seeds_filter)].copy()
        merged = merged.sort_values(["seed", "condition"]).reset_index(drop=True)
        if merged.empty:
            return {"experiment": experiment, "model": model, "status": "empty_after_filter"}

        runs_path = out_dir / str(spec["runs_file"])
        merged.to_parquet(runs_path, index=False)

        seeds = sorted(int(x) for x in merged["seed"].astype(int).unique().tolist())
        summary = _build_comparison_summary(
            merged,
            str(spec["condition_x"]),
            str(spec["condition_y"]),
            seeds,
        )
        summary["experiment"] = experiment.upper()
        summary["model"] = model
        summary["merged_timestamp"] = now_timestamp()
        summary["source_count"] = int(len(frames))
        comparison_path = out_dir / str(spec["comparison_file"])
        write_json(comparison_path, summary)

        manifest = {
            "timestamp": now_timestamp(),
            "experiment": experiment.upper(),
            "status": "merged",
            "model": model,
            "seeds": seeds,
            "n_runs": int(len(merged)),
            "artifacts": {
                "runs": str(runs_path),
                "comparison": str(comparison_path),
            },
        }
        write_json(out_dir / "run_manifest.json", manifest)
        return {"experiment": experiment, "model": model, "status": "merged", "rows": int(len(merged))}

    records = _collect_6e_records(model=model, shard_root=shard_root, canonical_root=canonical_root)
    if not records:
        return {"experiment": experiment, "model": model, "status": "no_data"}

    by_seed: dict[int, dict[str, Any]] = {}
    for _, rec in records:
        try:
            seed = int(rec.get("seed", 0))
        except Exception:
            seed = 0
        if seeds_filter and seed not in seeds_filter:
            continue
        by_seed[seed] = rec
    if not by_seed:
        return {"experiment": experiment, "model": model, "status": "empty_after_filter"}

    preferred_seed = min(by_seed.keys())
    summary_path = out_dir / str(spec["summary_file"])
    write_json(summary_path, by_seed[preferred_seed])
    if len(by_seed) > 1:
        write_json(out_dir / "exp6e_multi_seed_summary.json", {"records_by_seed": by_seed})

    manifest = {
        "timestamp": now_timestamp(),
        "experiment": "6E",
        "status": "merged",
        "model": model,
        "seeds": sorted(by_seed.keys()),
        "artifacts": {"summary": str(summary_path)},
    }
    write_json(out_dir / "run_manifest.json", manifest)
    return {"experiment": experiment, "model": model, "status": "merged", "rows": int(len(by_seed))}


def _write_worker_report(report: dict[str, Any], worker_tag: str) -> None:
    out_dir = Path("logs/experiment6/distributed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{worker_tag}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    write_json(out_path, report)
    print(f"[exp6-dist] report: {out_path}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 6 distributed orchestrator")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run distributed Exp6 jobs for a shard worker")
    run_p.add_argument("--models", default="all", help="Comma list or 'all'")
    run_p.add_argument("--experiments", default="all", help="Comma list among 6a,6b,6c,6d,6e or 'all'")
    run_p.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds used by 6a-6d")
    run_p.add_argument("--gpu", type=int, default=0, help="Physical GPU index used by this worker (for logs)")
    run_p.add_argument("--worker-index", type=int, default=0)
    run_p.add_argument("--num-workers", type=int, default=1)
    run_p.add_argument("--retry-failed-once", action="store_true")
    run_p.add_argument("--dry-run", action="store_true")
    run_p.add_argument("--shard-root", default="results/experiment6_shards")
    run_p.add_argument("--canonical-root", default="results/experiment6")
    run_p.add_argument("--worker-tag", default="")

    merge_p = sub.add_parser("merge", help="Merge shard outputs into canonical Exp6 results")
    merge_p.add_argument("--models", default="all", help="Comma list or 'all'")
    merge_p.add_argument("--experiments", default="all", help="Comma list among 6a,6b,6c,6d,6e or 'all'")
    merge_p.add_argument("--seeds", default="0,1,2", help="Comma-separated seed filter")
    merge_p.add_argument("--shard-root", default="results/experiment6_shards")
    merge_p.add_argument("--canonical-root", default="results/experiment6")

    return parser.parse_args()


def _run_command(args: argparse.Namespace) -> int:
    models = _normalize_models(args.models)
    experiments = _normalize_experiments(args.experiments)
    seeds = _parse_int_csv(args.seeds)
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be >= 1")
    if args.worker_index < 0 or args.worker_index >= args.num_workers:
        raise ValueError("--worker-index must satisfy 0 <= worker-index < num-workers")

    all_jobs = _build_jobs(models=models, experiments=experiments, seeds=seeds)
    jobs = [job for idx, job in enumerate(all_jobs) if idx % int(args.num_workers) == int(args.worker_index)]

    print(
        f"[exp6-dist] worker={args.worker_index}/{args.num_workers} gpu={args.gpu} "
        f"models={models} exps={experiments} seeds={seeds} selected_jobs={len(jobs)}",
        flush=True,
    )
    report = _run_jobs(
        jobs=jobs,
        shard_root=Path(args.shard_root),
        canonical_root=Path(args.canonical_root),
        gpu=int(args.gpu),
        retry_failed_once=bool(args.retry_failed_once),
        dry_run=bool(args.dry_run),
    )
    worker_tag = args.worker_tag or f"gpu{int(args.gpu)}_w{int(args.worker_index)}"
    _write_worker_report(report, worker_tag=worker_tag)
    if not bool(args.dry_run) and int(report.get("failed", 0)) > 0:
        return 1
    return 0


def _merge_command(args: argparse.Namespace) -> int:
    models = _normalize_models(args.models)
    experiments = _normalize_experiments(args.experiments)
    seeds = set(_parse_int_csv(args.seeds))

    shard_root = Path(args.shard_root)
    canonical_root = Path(args.canonical_root)
    canonical_root.mkdir(parents=True, exist_ok=True)

    merged_rows: list[dict[str, Any]] = []
    for model in models:
        for exp in experiments:
            rep = _merge_one(
                experiment=exp,
                model=model,
                shard_root=shard_root,
                canonical_root=canonical_root,
                seeds_filter=seeds,
            )
            merged_rows.append(rep)
            print(f"[exp6-dist][merge] {exp} {model}: {rep.get('status')}", flush=True)

    summary = {
        "timestamp": now_timestamp(),
        "models": models,
        "experiments": experiments,
        "seeds_filter": sorted(seeds),
        "rows": merged_rows,
    }
    out_dir = Path("logs/experiment6/distributed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"merge_{time.strftime('%Y%m%d_%H%M%S')}.json"
    write_json(out_path, summary)
    print(f"[exp6-dist][merge] summary: {out_path}", flush=True)
    return 0


def main() -> None:
    args = _parse_args()
    if args.command == "run":
        rc = _run_command(args)
    else:
        rc = _merge_command(args)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
