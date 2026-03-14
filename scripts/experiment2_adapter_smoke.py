#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from experiment2.config import MODEL_BY_NAME, model_names_for_profile
from shared.attention import get_adapter
from shared.models.loading import load_model, load_tokenizer


def _parse_models(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(model_names_for_profile("scaleup_78b"))
    out: list[str] = []
    for part in raw.split(","):
        name = part.strip()
        if not name:
            continue
        out.append(name)
    if not out:
        raise SystemExit("--models produced an empty model list.")
    return out


def _shape(value: Any) -> list[int] | None:
    if value is None:
        return None
    return [int(x) for x in value.shape]


def run_smoke(*, model_name: str, device: str, prompt: str, include_logits: bool) -> dict[str, Any]:
    if model_name not in MODEL_BY_NAME:
        raise SystemExit(f"Unknown model name '{model_name}'.")
    spec = MODEL_BY_NAME[model_name]

    loaded = load_model(spec)
    model = loaded.model.to(device)
    tokenizer = load_tokenizer(spec)
    adapter = get_adapter(spec)

    try:
        adapter.register(model)
        tokenized = tokenizer(prompt, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        record = adapter.capture(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            include_logits=include_logits,
            return_token_logits=True,
            output_device="cpu",
        )
        q_shape = _shape(record.q)
        k_shape = _shape(record.k)
        logits_shape = _shape(record.logits)
        token_logits_shape = _shape(record.token_logits)
        q_heads = int(q_shape[1]) if q_shape and len(q_shape) >= 2 else None
        k_heads = int(k_shape[1]) if k_shape and len(k_shape) >= 2 else None
        return {
            "model": model_name,
            "hf_id": spec.hf_id,
            "device": device,
            "ok": bool(q_shape and k_shape),
            "q_shape": q_shape,
            "k_shape": k_shape,
            "logits_shape": logits_shape,
            "token_logits_shape": token_logits_shape,
            "num_hidden_layers_expected": int(getattr(model.config, "num_hidden_layers", -1)),
            "num_layers_captured": int(q_shape[0]) if q_shape else 0,
            "num_q_heads_captured": q_heads,
            "num_k_heads_captured": k_heads,
            "gqa_head_alignment_ok": (None if q_heads is None or k_heads is None else bool(q_heads == k_heads)),
        }
    finally:
        try:
            adapter.cleanup()
        except Exception:
            pass
        del model
        gc.collect()
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        # Avoid retaining heavyweight objects in lru caches across model smokes.
        load_model.cache_clear()
        load_tokenizer.cache_clear()


def run_smoke_safe(*, model_name: str, device: str, prompt: str, include_logits: bool) -> dict[str, Any]:
    try:
        return run_smoke(
            model_name=model_name,
            device=device,
            prompt=prompt,
            include_logits=include_logits,
        )
    except Exception as exc:
        return {
            "model": model_name,
            "device": device,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test adapter capture for Experiment 2 models.")
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model names or 'all' (default uses scaleup_78b profile).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Execution device (default cuda:0 when available, else cpu).",
    )
    parser.add_argument(
        "--prompt",
        default="The quick brown fox jumps over the lazy dog.",
        help="Prompt text for the capture smoke pass.",
    )
    parser.add_argument(
        "--include-logits",
        action="store_true",
        help="Capture per-layer attention logits in addition to Q/K tensors.",
    )
    args = parser.parse_args()

    models = _parse_models(str(args.models))
    results = [
        run_smoke_safe(
            model_name=name,
            device=str(args.device),
            prompt=str(args.prompt),
            include_logits=bool(args.include_logits),
        )
        for name in models
    ]
    print(json.dumps({"status": "ok", "results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
