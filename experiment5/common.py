from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from experiment5.config import MODELS
from shared.models.loading import load_model, load_tokenizer


def now_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


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


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _force_eager_attention(model) -> None:
    cfg = getattr(model, "config", None)
    if cfg is not None:
        if hasattr(cfg, "_attn_implementation"):
            setattr(cfg, "_attn_implementation", "eager")
        if hasattr(cfg, "attn_implementation"):
            setattr(cfg, "attn_implementation", "eager")
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None and hasattr(gen_cfg, "attn_implementation"):
        setattr(gen_cfg, "attn_implementation", "eager")


def load_model_bundle(model_name: str, device: str) -> tuple[Any | None, Any | None, str | None]:
    spec = MODELS[model_name]
    try:
        try:
            loaded = load_model(spec, attn_implementation="eager")
        except Exception:
            loaded = load_model(spec)
        model = loaded.model.to(device)
        _force_eager_attention(model)
        model.eval()
        tokenizer = load_tokenizer(spec)
        return model, tokenizer, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def _load_jsonl_sequences(path: Path, seq_len: int, max_sequences: int) -> list[list[int]]:
    rows: list[list[int]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            toks = row.get("tokens", row.get("input_ids", []))
            if len(toks) < seq_len:
                continue
            rows.append([int(x) for x in toks[:seq_len]])
            if len(rows) >= max_sequences:
                break
    return rows


def load_cached_sequences(
    *,
    model_name: str,
    dataset_name: str,
    seq_len: int,
    max_sequences: int,
) -> list[list[int]]:
    path = Path("data/experiment1") / dataset_name / model_name / f"len_{seq_len}.jsonl"
    return _load_jsonl_sequences(path, seq_len=seq_len, max_sequences=max_sequences)


def _load_dataset_hf(dataset_id: str, *, split: str, config_name: str | None = None, streaming: bool = False, max_rows: int | None = None):
    """Import and call HuggingFace ``datasets.load_dataset`` robustly.

    In this workspace there are top-level directories named ``datasets`` that can
    shadow the external library in non-venv interpreters. We retry with the
    project venv site-packages injected when needed.
    """
    kwargs: dict[str, Any] = {}
    if config_name:
        kwargs["name"] = config_name

    def _try_import_load_dataset():
        import importlib

        mod = importlib.import_module("datasets")
        fn = getattr(mod, "load_dataset", None)
        if callable(fn):
            return fn
        return None

    fn = None
    try:
        fn = _try_import_load_dataset()
    except Exception:
        fn = None

    if fn is None:
        # Retry by preferring the project's venv site-packages.
        project_root = Path(__file__).resolve().parents[1]
        venv_lib = project_root / ".venv" / "lib"
        candidates = sorted(venv_lib.glob("python*/site-packages"))
        for sp in candidates:
            sp_s = str(sp)
            if sp_s not in sys.path:
                sys.path.insert(0, sp_s)
        try:
            if "datasets" in sys.modules:
                del sys.modules["datasets"]
        except Exception:
            pass
        fn = _try_import_load_dataset()

    if fn is None:
        raise RuntimeError(
            "Could not import HuggingFace datasets.load_dataset. "
            "Use the project venv (e.g., `.venv/bin/python -m ...`) or install the `datasets` package."
        )

    if streaming:
        return fn(dataset_id, split=split, streaming=True, **kwargs)
    if max_rows is None:
        return fn(dataset_id, split=split, **kwargs)
    return fn(dataset_id, split=f"{split}[:{max_rows}]", **kwargs)


def build_sequences_from_text_dataset(
    *,
    tokenizer,
    dataset_id: str,
    split: str,
    text_fields: tuple[str, ...],
    seq_len: int,
    max_sequences: int,
    max_rows: int,
    seed: int,
    config_name: str | None = None,
    streaming: bool = False,
) -> list[list[int]]:
    if streaming:
        ds = _load_dataset_hf(
            dataset_id,
            split=split,
            config_name=config_name,
            streaming=True,
        )
        try:
            ds = ds.shuffle(seed=seed, buffer_size=min(20000, max(1000, max_rows * 2)))
        except Exception:
            pass
    else:
        ds = _load_dataset_hf(
            dataset_id,
            split=split,
            config_name=config_name,
            streaming=False,
            max_rows=max_rows,
        )
        ds = ds.shuffle(seed=seed)

    sep = tokenizer.encode("\n", add_special_tokens=False)
    sep = sep[:1] if sep else []

    buffer: list[int] = []
    out: list[list[int]] = []
    for row_idx, row in enumerate(ds):
        if row_idx >= max_rows:
            break
        txt = None
        for key in text_fields:
            v = row.get(key)
            if isinstance(v, str) and v.strip():
                txt = v.strip()
                break
            if isinstance(v, (list, tuple)):
                parts = [str(x).strip() for x in v if isinstance(x, str) and str(x).strip()]
                if parts:
                    txt = "\n".join(parts)
                    break
            if isinstance(v, dict):
                parts = [str(x).strip() for x in v.values() if isinstance(x, str) and str(x).strip()]
                if parts:
                    txt = "\n".join(parts)
                    break
        if not txt:
            continue
        ids = tokenizer.encode(txt, add_special_tokens=False)
        if not ids:
            continue
        buffer.extend(ids)
        if sep:
            buffer.extend(sep)
        while len(buffer) >= seq_len and len(out) < max_sequences:
            out.append(buffer[:seq_len])
            buffer = buffer[seq_len:]
        if len(out) >= max_sequences:
            break
    return out


def build_adversarial_bpe_sequences(*, tokenizer, base_sequences: list[list[int]], max_sequences: int, seq_len: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    out: list[list[int]] = []
    fragments = [
        "xqzv", "@@@", "123123123", "\\u4f60\\u597d", "####", "zzzzzz", "---", "_{_}_", "!!??!!",
        "aAaAaA", "tokenization", "segmentation",
    ]

    for seq in base_sequences:
        txt = tokenizer.decode(seq, skip_special_tokens=True)
        injected = txt
        for _ in range(6):
            frag = rng.choice(fragments)
            pos = rng.randint(0, max(0, len(injected)))
            injected = injected[:pos] + " " + frag + " " + injected[pos:]
        ids = tokenizer.encode(injected, add_special_tokens=False)
        if len(ids) < seq_len:
            ids = ids + seq[: (seq_len - len(ids))]
        out.append([int(x) for x in ids[:seq_len]])
        if len(out) >= max_sequences:
            break
    return out
