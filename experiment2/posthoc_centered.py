from __future__ import annotations

import fcntl
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from experiment2.config import DEFAULT_SYNTHETIC_COUNT, DEFAULT_TIER1_COUNT, MODEL_BY_NAME
from experiment2.execution import (
    _batch_size_for_row,
    _build_intervention,
    _iter_example_batches,
    _load_or_build_examples,
    _prepare_batch_tensors,
)
from experiment2.kernels import KernelMetricsAccumulator
from experiment2.provenance import generator_hash
from experiment2.tasks import build_token_pools
from shared.attention import get_adapter
from shared.models.loading import load_model, load_tokenizer

LOGGER = logging.getLogger("kernel_pe.experiment2.posthoc_centered")


@dataclass
class PosthocCenteredConfig:
    output_root: Path
    phase: str
    run_id: str
    device: str
    force: bool = False
    synthetic_count: int = DEFAULT_SYNTHETIC_COUNT
    tier1_count: int = DEFAULT_TIER1_COUNT
    batch_size_synth: int = 8
    batch_size_tier1: int = 8
    strict_posthoc: bool = False
    model_filter: frozenset[str] | None = None
    split_filter: frozenset[str] | None = None
    seed_start: int | None = None
    seed_end: int | None = None
    enable_example_cache: bool = True


@dataclass
class PosthocQueueWorkerConfig:
    queue_file: Path
    output_root: Path
    phase: str
    device: str
    batch_size_synth: int = 24
    batch_size_tier1: int = 24
    strict_posthoc: bool = True
    force: bool = False
    enable_example_cache: bool = True
    preferred_model: str | None = None


@dataclass
class _LoadedResources:
    model_name: str
    model_spec: Any
    model: Any
    tokenizer: Any
    pools: Any
    adapter: Any


_DEVICE_RESOURCE_CACHE: dict[str, _LoadedResources] = {}


def _condition_dir(output_root: Path, row: dict[str, Any]) -> Path:
    run_id = row.get("run_id") or "unknown_run"
    parts = [output_root, row["phase"], run_id, row["model"]]
    if row.get("dataset"):
        parts.append(row["dataset"])
    parts.extend([row["task"], f"len_{row['seq_len']}"])
    if row.get("span") is not None:
        parts.append(f"span_{int(row['span'])}")
    parts.append(row["intervention"])
    if row.get("random_draw") is not None:
        parts.append(f"draw_{int(row['random_draw'])}")
    parts.append(f"seed_{row['seed']}")
    out = Path(parts[0])
    for p in parts[1:]:
        out = out / str(p)
    return out


def _run_config_paths(phase_root: Path) -> list[Path]:
    return sorted(phase_root.glob("**/run_config.json"))


def _posthoc_batch_keys(row: dict[str, Any]) -> tuple[tuple[str, str, str, int], tuple[str, str, int]]:
    model = str(row.get("model", ""))
    split = str(row.get("split", ""))
    task = str(row.get("task", ""))
    seq_len = int(row.get("seq_len", 0))
    return (model, split, task, seq_len), (model, split, seq_len)


def _initial_posthoc_batch_size(
    row: dict[str, Any],
    cfg: Any,
    *,
    batch_size_memory: dict[tuple[Any, ...], int],
) -> int:
    task_key, split_key = _posthoc_batch_keys(row)
    if task_key in batch_size_memory:
        return max(1, int(batch_size_memory[task_key]))
    if split_key in batch_size_memory:
        return max(1, int(batch_size_memory[split_key]))

    base = int(_batch_size_for_row(row, cfg))
    split = str(row.get("split"))
    seq_len = int(row.get("seq_len", 0))
    task = str(row.get("task"))
    model = str(row.get("model"))

    # Preserve large configured starts for easy rows while avoiding repeated
    # OOM retry cascades on known-heavy 1024-token synthetic tasks.
    if split in {"synthetic", "span_bridge", "mechanistic"} and seq_len >= 1024:
        if task in {"local_copy_offset", "local_key_match"}:
            base = min(base, 6)
        if model == "tinyllama-1.1b":
            base = min(base, 3)
    return max(1, int(base))


def _remember_posthoc_batch_size(
    row: dict[str, Any],
    *,
    batch_size: int,
    batch_size_memory: dict[tuple[Any, ...], int],
) -> None:
    task_key, split_key = _posthoc_batch_keys(row)
    value = max(1, int(batch_size))
    batch_size_memory[task_key] = value
    prev_split = batch_size_memory.get(split_key)
    if prev_split is None:
        batch_size_memory[split_key] = value
    else:
        batch_size_memory[split_key] = min(int(prev_split), value)


def _release_cached_resources(device_key: str) -> None:
    cached = _DEVICE_RESOURCE_CACHE.pop(device_key, None)
    if cached is None:
        return
    try:
        del cached.model
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_or_load_resources(*, model_name: str, device: str) -> _LoadedResources:
    device_key = str(device)
    cached = _DEVICE_RESOURCE_CACHE.get(device_key)
    if cached is not None and cached.model_name == model_name:
        return cached
    if cached is not None and cached.model_name != model_name:
        _release_cached_resources(device_key)

    model_spec = MODEL_BY_NAME[model_name]
    loader = load_model(model_spec)
    model = loader.model.to(device)
    model.eval()
    tokenizer = load_tokenizer(model_spec)
    pools = build_token_pools(
        model_name=model_spec.name,
        vocab_size=tokenizer.vocab_size,
        special_ids=getattr(tokenizer, "all_special_ids", []),
    )
    adapter = get_adapter(model_spec)
    loaded = _LoadedResources(
        model_name=model_name,
        model_spec=model_spec,
        model=model,
        tokenizer=tokenizer,
        pools=pools,
        adapter=adapter,
    )
    _DEVICE_RESOURCE_CACHE[device_key] = loaded
    return loaded


def run_posthoc_centered(config: PosthocCenteredConfig) -> dict[str, Any]:
    phase_root = config.output_root / config.phase / config.run_id
    if not phase_root.exists():
        raise FileNotFoundError(f"Phase root does not exist: {phase_root}")

    run_cfg_paths = _run_config_paths(phase_root)
    rows: list[tuple[dict[str, Any], Path]] = []
    for cfg_path in run_cfg_paths:
        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
        row = dict(payload.get("row", {}))
        if not row:
            continue
        if config.model_filter and str(row.get("model")) not in config.model_filter:
            continue
        if config.split_filter and str(row.get("split")) not in config.split_filter:
            continue
        seed = row.get("seed")
        if (config.seed_start is not None) and (seed is not None) and (int(seed) < int(config.seed_start)):
            continue
        if (config.seed_end is not None) and (seed is not None) and (int(seed) > int(config.seed_end)):
            continue
        row.setdefault("run_id", config.run_id)
        out_dir = _condition_dir(config.output_root, row)
        centered_path = out_dir / "kernel_track_b_centered_summary.parquet"
        pending = bool(payload.get("centered_pending", False))
        if not config.force and (not pending) and centered_path.exists():
            continue
        rows.append((row, cfg_path))

    if not rows:
        return {
            "status": "no_rows",
            "phase": config.phase,
            "run_id": config.run_id,
            "processed": 0,
        }

    rows.sort(key=lambda x: (x[0]["model"], x[0]["split"], x[0]["task"], int(x[0]["seed"])))
    processed = 0
    skipped = 0
    current_model: str | None = None
    resources: _LoadedResources | None = None
    generator_hash_value = generator_hash()
    intervention_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    batch_size_memory: dict[tuple[Any, ...], int] = {}

    for row, cfg_path in rows:
        out_dir = _condition_dir(config.output_root, row)
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if row["model"] != current_model:
            current_model = row["model"]
            resources = _get_or_load_resources(model_name=current_model, device=config.device)

        if resources is None:
            skipped += 1
            continue

        floor = cfg.get("floor", {})
        fallback_applied = bool(floor.get("fallback_applied", False))
        fallback_spans = tuple(int(x) for x in floor.get("fallback_spans", [])) if fallback_applied else tuple()
        syn_used = cfg.get("synthetic_count_used")
        tier1_used = cfg.get("tier1_count_used")
        if config.strict_posthoc and (syn_used is None or tier1_used is None):
            raise RuntimeError(
                f"Strict posthoc requires persisted execution counts, missing for {cfg_path}"
            )

        class _LocalExecCfg:
            output_root = config.output_root
            synthetic_count = int(syn_used) if syn_used is not None else int(config.synthetic_count)
            tier1_count = int(tier1_used) if tier1_used is not None else int(config.tier1_count)
            batch_size_synth = int(config.batch_size_synth)
            batch_size_tier1 = int(config.batch_size_tier1)
            enable_example_cache = bool(config.enable_example_cache)

        examples = _load_or_build_examples(
            row=row,
            config=_LocalExecCfg,
            pools=resources.pools,
            span_choices=fallback_spans,
            generator_hash_value=generator_hash_value,
        )
        plan, intervention_cm = _build_intervention(
            resources.model,
            resources.model_spec,
            row,
            results_root=config.output_root.parent,
            run_id=str(row.get("run_id", config.run_id)),
            cache=intervention_cache,
        )

        centered_mode = str(cfg.get("kernel_centered_mode", "shared_mean"))
        kernel_engine = str(cfg.get("kernel_engine", "optimized"))
        adapter = resources.adapter
        batch_size = _initial_posthoc_batch_size(
            row,
            _LocalExecCfg,
            batch_size_memory=batch_size_memory,
        )
        while True:
            kernel_accum = KernelMetricsAccumulator(resources.model_spec, engine=kernel_engine, centered_mode=centered_mode)
            try:
                with intervention_cm:
                    adapter.register(resources.model)
                    try:
                        batches = _iter_example_batches(examples, batch_size)
                        if centered_mode == "shared_mean":
                            for batch in batches:
                                input_ids, attention_mask = _prepare_batch_tensors(batch, config.device)
                                capture = adapter.capture(
                                    resources.model,
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    include_logits=False,
                                    return_token_logits=False,
                                    output_device=config.device,
                                )
                                kernel_accum.accumulate_shared_means(capture)
                                del capture
                            shared_q, shared_k = kernel_accum.finalize_shared_means()
                            for batch in batches:
                                input_ids, attention_mask = _prepare_batch_tensors(batch, config.device)
                                capture = adapter.capture(
                                    resources.model,
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    include_logits=False,
                                    return_token_logits=False,
                                    output_device=config.device,
                                )
                                kernel_accum.update_centered(capture, shared_q_mean=shared_q, shared_k_mean=shared_k)
                                del capture
                        else:
                            for batch in batches:
                                input_ids, attention_mask = _prepare_batch_tensors(batch, config.device)
                                capture = adapter.capture(
                                    resources.model,
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    include_logits=False,
                                    return_token_logits=False,
                                    output_device=config.device,
                                )
                                kernel_accum.update_centered(capture)
                                del capture
                    finally:
                        adapter.cleanup()
                break
            except RuntimeError as exc:
                msg = str(exc).lower()
                is_oom = ("out of memory" in msg) or ("cuda out of memory" in msg)
                if (not is_oom) or (batch_size <= 1):
                    raise
                batch_size = max(1, batch_size // 2)
                LOGGER.warning(
                    "OOM in posthoc-centered row; retrying with reduced batch_size=%d row=%s",
                    batch_size,
                    row,
                )
                _remember_posthoc_batch_size(row, batch_size=batch_size, batch_size_memory=batch_size_memory)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        _remember_posthoc_batch_size(row, batch_size=batch_size, batch_size_memory=batch_size_memory)

        _, _, centered_df = kernel_accum.to_dataframes(
            model=row["model"],
            phase=row["phase"],
            split=row["split"],
            dataset=row.get("dataset"),
            task=row["task"],
            span=row.get("span"),
            intervention=row["intervention"],
            random_draw=row.get("random_draw"),
            seed=row["seed"],
            seq_len=row["seq_len"],
        )
        centered_df.to_parquet(out_dir / "kernel_track_b_centered_summary.parquet", engine="pyarrow", index=False)

        cfg["centered_pending"] = False
        cfg["centered_posthoc_at"] = datetime.now().isoformat(timespec="seconds")
        cfg["centered_posthoc_plan"] = plan.to_json_dict()
        cfg["centered_posthoc_batch_size_used"] = int(batch_size)
        cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        processed += 1

    return {
        "status": "ok",
        "phase": config.phase,
        "run_id": config.run_id,
        "model_filter": sorted(config.model_filter) if config.model_filter else None,
        "split_filter": sorted(config.split_filter) if config.split_filter else None,
        "seed_start": config.seed_start,
        "seed_end": config.seed_end,
        "processed": processed,
        "skipped": skipped,
    }


def _queue_paths(queue_file: Path) -> tuple[Path, Path, Path, Path]:
    pending = queue_file.with_suffix(".pending.tsv")
    done = queue_file.with_suffix(".done.tsv")
    fail = queue_file.with_suffix(".failures.tsv")
    lock = queue_file.with_suffix(".lock")
    return pending, done, fail, lock


def _parse_job_line(line: str) -> dict[str, Any]:
    parts = [p.strip() for p in line.rstrip("\n").split("\t")]
    if len(parts) == 6:
        run_id, job_id, model, seed_start, seed_end, expected_rows = parts
        split = ""
    elif len(parts) >= 7:
        run_id, job_id, model, split, seed_start, seed_end, expected_rows = parts[:7]
    else:
        raise RuntimeError(f"Malformed posthoc queue line: {line!r}")
    return {
        "run_id": run_id,
        "job_id": job_id,
        "model": model,
        "split": split,
        "seed_start": int(seed_start),
        "seed_end": int(seed_end),
        "expected_rows": int(expected_rows),
    }


def _claim_next_job(
    *,
    pending_file: Path,
    lock_file: Path,
    preferred_model: str | None,
    current_model: str | None,
) -> dict[str, Any] | None:
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file.touch(exist_ok=True)
    with lock_file.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            if not pending_file.exists():
                return None
            lines = pending_file.read_text(encoding="utf-8").splitlines()
            if not lines:
                return None
            idx = None
            if current_model:
                for i, line in enumerate(lines):
                    if _parse_job_line(line)["model"] == current_model:
                        idx = i
                        break
            if idx is None and preferred_model:
                for i, line in enumerate(lines):
                    if _parse_job_line(line)["model"] == preferred_model:
                        idx = i
                        break
            if idx is None:
                idx = 0
            selected = lines[idx]
            remaining = lines[:idx] + lines[idx + 1 :]
            pending_file.write_text("\n".join(remaining) + ("\n" if remaining else ""), encoding="utf-8")
            return _parse_job_line(selected)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _append_tsv(path: Path, row: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(row)


def run_posthoc_queue_worker(config: PosthocQueueWorkerConfig) -> dict[str, Any]:
    queue_file = config.queue_file
    if not queue_file.exists():
        raise FileNotFoundError(f"Posthoc queue file not found: {queue_file}")
    pending_file, done_file, fail_file, lock_file = _queue_paths(queue_file)
    done_file.parent.mkdir(parents=True, exist_ok=True)
    done_file.touch(exist_ok=True)
    fail_file.touch(exist_ok=True)
    if not pending_file.exists():
        pending_file.write_text(queue_file.read_text(encoding="utf-8"), encoding="utf-8")

    done_count = 0
    fail_count = 0
    current_model: str | None = None
    started_at = datetime.now().isoformat(timespec="seconds")

    while True:
        job = _claim_next_job(
            pending_file=pending_file,
            lock_file=lock_file,
            preferred_model=config.preferred_model,
            current_model=current_model,
        )
        if job is None:
            break

        current_model = str(job["model"])
        split_value = str(job.get("split") or "").strip()
        split_filter = frozenset([split_value]) if split_value else None
        row_cfg = PosthocCenteredConfig(
            output_root=config.output_root,
            phase=config.phase,
            run_id=str(job["run_id"]),
            device=config.device,
            force=bool(config.force),
            batch_size_synth=max(1, int(config.batch_size_synth)),
            batch_size_tier1=max(1, int(config.batch_size_tier1)),
            strict_posthoc=bool(config.strict_posthoc),
            model_filter=frozenset([current_model]),
            split_filter=split_filter,
            seed_start=int(job["seed_start"]),
            seed_end=int(job["seed_end"]),
            enable_example_cache=bool(config.enable_example_cache),
        )

        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
        try:
            result = run_posthoc_centered(row_cfg)
            done_count += 1
            _append_tsv(
                done_file,
                (
                    f"{stamp}\t{job['run_id']}\t{job['job_id']}\t{job['model']}\t{split_value or '*'}"
                    f"\t{job['seed_start']}\t{job['seed_end']}\tok\t{result.get('processed', 0)}\n"
                ),
            )
        except Exception as exc:
            fail_count += 1
            _append_tsv(
                fail_file,
                (
                    f"{stamp}\t{job['run_id']}\t{job['job_id']}\t{job['model']}\t{split_value or '*'}"
                    f"\t{job['seed_start']}\t{job['seed_end']}\tfailed\t{type(exc).__name__}:{exc}\n"
                ),
            )
            LOGGER.exception("Posthoc queue worker failed job=%s", job)

    pending_left = 0
    if pending_file.exists():
        pending_left = len([x for x in pending_file.read_text(encoding="utf-8").splitlines() if x.strip()])
    return {
        "status": "ok" if fail_count == 0 else "partial_failure",
        "queue_file": str(queue_file),
        "started_at": started_at,
        "ended_at": datetime.now().isoformat(timespec="seconds"),
        "done_jobs": int(done_count),
        "failed_jobs": int(fail_count),
        "pending_jobs": int(pending_left),
        "preferred_model": config.preferred_model,
    }
