#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str in sys.path:
    sys.path.remove(project_root_str)
sys.path.insert(0, project_root_str)

from experiment1.paths import tokenized_path
from experiment2.config import DATASET_BY_NAME, MODEL_BY_NAME
from experiment2.interventions import AbsPEDCTInterventionContext, RopeQKInterventionContext, build_abspe_dct_plan, build_rope_intervention_plan
from shared.models.loading import load_model, load_tokenizer
from shared.specs import SequenceLengthSpec


@dataclass(frozen=True)
class ProbeExample:
    seq_id: int
    tokens: list[int]
    target_pos: int
    target_token: int
    span: int
    seed: int


def _stable_seed(*parts: object) -> int:
    key = "|".join(str(x) for x in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _parse_int_csv(raw: str) -> list[int]:
    vals: list[int] = []
    seen: set[int] = set()
    for part in str(raw).split(","):
        tok = part.strip()
        if not tok:
            continue
        v = int(tok)
        if v not in seen:
            vals.append(v)
            seen.add(v)
    return vals


def _parse_str_csv(raw: str) -> list[str]:
    vals: list[str] = []
    seen: set[str] = set()
    for part in str(raw).split(","):
        tok = part.strip()
        if not tok:
            continue
        if tok not in seen:
            vals.append(tok)
            seen.add(tok)
    return vals


def _expand_interventions(interventions: list[str], *, random_draws: int) -> list[tuple[str, int | None]]:
    out: list[tuple[str, int | None]] = []
    for name in interventions:
        if name.startswith("random_"):
            for draw in range(max(1, int(random_draws))):
                out.append((name, draw))
        else:
            out.append((name, None))
    return out


def _load_eval_sequences(*, model_name: str, dataset_name: str, seq_len: int, data_root: Path) -> list[tuple[int, list[int]]]:
    model_spec = MODEL_BY_NAME[model_name]
    dataset_spec = DATASET_BY_NAME[dataset_name]
    path = tokenized_path(data_root, model_spec, dataset_spec, SequenceLengthSpec(tokens=seq_len))
    if not path.exists():
        raise FileNotFoundError(f"Missing tokenized source: {path}")
    out: list[tuple[int, list[int]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") != "eval":
                continue
            out.append((int(row["id"]), list(int(t) for t in row["tokens"])))
    if not out:
        raise RuntimeError(f"No eval rows found in {path}")
    return out


def _collect_examples_for_span(
    *,
    sequences: list[tuple[int, list[int]]],
    span: int,
    seed: int,
    count: int,
    special_ids: set[int],
) -> list[ProbeExample]:
    rng = random.Random(_stable_seed("semi_natural_span", span, seed, count))
    order = list(range(len(sequences)))
    rng.shuffle(order)
    out: list[ProbeExample] = []
    for idx in order:
        seq_id, tokens = sequences[idx]
        valid_positions: list[int] = []
        for pos in range(span, len(tokens)):
            tok = int(tokens[pos])
            if tok in special_ids:
                continue
            if int(tokens[pos - span]) == tok:
                valid_positions.append(pos)
        if not valid_positions:
            continue
        rng.shuffle(valid_positions)
        chosen = int(valid_positions[0])
        out.append(
            ProbeExample(
                seq_id=int(seq_id),
                tokens=tokens,
                target_pos=chosen,
                target_token=int(tokens[chosen]),
                span=int(span),
                seed=int(seed),
            )
        )
        if len(out) >= count:
            break
    if len(out) < count:
        raise RuntimeError(
            f"Insufficient semi-natural probes for span={span} seed={seed}: "
            f"found={len(out)} requested={count}"
        )
    return out


def _build_candidates(example: ProbeExample, *, candidate_size: int, draw: int) -> list[int]:
    uniq = []
    seen: set[int] = set()
    for tok in example.tokens:
        itok = int(tok)
        if itok == int(example.target_token):
            continue
        if itok in seen:
            continue
        seen.add(itok)
        uniq.append(itok)
    rng = random.Random(_stable_seed("semi_nat_candidates", example.seq_id, example.target_pos, draw, candidate_size))
    rng.shuffle(uniq)
    candidates = [int(example.target_token)]
    for tok in uniq:
        if tok == int(example.target_token):
            continue
        candidates.append(int(tok))
        if len(candidates) >= candidate_size:
            break
    while len(candidates) < candidate_size:
        filler = int(rng.randrange(0, 32000))
        if filler not in candidates:
            candidates.append(filler)
    return candidates


def _iter_batches(examples: list[ProbeExample], batch_size: int):
    for i in range(0, len(examples), batch_size):
        yield examples[i : i + batch_size]


def main() -> None:
    ap = argparse.ArgumentParser(description="Semi-natural wiki long-gap repeated-token retrieval probe.")
    ap.add_argument("--run-tag", type=str, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--logs-root", type=Path, required=True)
    ap.add_argument("--model", type=str, default="llama-3.1-8b")
    ap.add_argument("--dataset", type=str, default="wiki40b_en_pre2019")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--spans", type=str, default="32,48,64")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument(
        "--interventions",
        type=str,
        default="none,ablate_high_strong,ablate_low_strong,random_strong",
    )
    ap.add_argument("--random-draws", type=int, default=3)
    ap.add_argument("--synthetic-count", type=int, default=100)
    ap.add_argument("--candidate-size", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--seed-end", type=int, default=2)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    args = ap.parse_args()

    if args.model not in MODEL_BY_NAME:
        raise RuntimeError(f"Unknown model name: {args.model}")

    spans = _parse_int_csv(args.spans)
    seeds = [s for s in _parse_int_csv(args.seeds) if int(args.seed_start) <= s <= int(args.seed_end)]
    interventions = _parse_str_csv(args.interventions)
    expanded = _expand_interventions(interventions, random_draws=max(1, int(args.random_draws)))

    out_dir = args.output_root / args.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    args.logs_root.mkdir(parents=True, exist_ok=True)

    model_spec = MODEL_BY_NAME[args.model]
    tokenizer = load_tokenizer(model_spec)
    loader = load_model(model_spec)
    model = loader.model.to(args.device)

    sequences = _load_eval_sequences(
        model_name=args.model,
        dataset_name=args.dataset,
        seq_len=int(args.seq_len),
        data_root=args.data_root,
    )
    special_ids = set(int(x) for x in getattr(tokenizer, "all_special_ids", []) if x is not None)

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for span in spans:
            examples = _collect_examples_for_span(
                sequences=sequences,
                span=int(span),
                seed=int(seed),
                count=int(args.synthetic_count),
                special_ids=special_ids,
            )
            for intervention, random_draw in expanded:
                cm = nullcontext()  # build context after spec-name normalization below.
                if intervention != "none":
                    if model_spec.pe_scheme == "RoPE":
                        head_dim = model.config.hidden_size // model.config.num_attention_heads
                        plan = build_rope_intervention_plan(
                            head_dim=head_dim,
                            intervention=intervention,
                            seed=int(seed),
                            random_draw=random_draw,
                        )
                        cm = RopeQKInterventionContext(model, plan)
                    elif model_spec.pe_scheme == "learned-absolute":
                        n_positions = int(getattr(model.transformer.wpe.weight, "shape")[0])
                        plan = build_abspe_dct_plan(
                            n_positions=n_positions,
                            intervention=intervention,
                            seed=int(seed),
                            random_draw=random_draw,
                        )
                        cm = AbsPEDCTInterventionContext(model, plan)
                    else:
                        raise RuntimeError(f"Unsupported PE scheme for semi-natural probe: {model_spec.pe_scheme}")

                full_correct = 0
                restricted_correct = 0
                full_nll_sum = 0.0
                restricted_nll_sum = 0.0
                total = 0
                with cm:
                    for batch in _iter_batches(examples, int(args.batch_size)):
                        input_ids = torch.tensor([ex.tokens for ex in batch], dtype=torch.long, device=args.device)
                        attn_mask = torch.ones_like(input_ids)
                        with torch.no_grad():
                            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
                        batch_logits = outputs.logits
                        if batch_logits is None or batch_logits.ndim != 3:
                            raise RuntimeError(f"Unexpected logits shape in semi-natural probe: {tuple(batch_logits.shape)}")
                        for i, ex in enumerate(batch):
                            token_logits = batch_logits[i]
                            pred_vec = token_logits[ex.target_pos - 1]
                            full_pred = int(torch.argmax(pred_vec).item())
                            full_correct += int(full_pred == ex.target_token)
                            full_nll_sum += float(-torch.log_softmax(pred_vec, dim=0)[int(ex.target_token)].item())

                            candidates = _build_candidates(ex, candidate_size=int(args.candidate_size), draw=(random_draw or 0))
                            cand_tensor = torch.tensor(candidates, dtype=torch.long, device=pred_vec.device)
                            cand_logits = pred_vec.index_select(0, cand_tensor)
                            restricted_pred = int(candidates[int(torch.argmax(cand_logits).item())])
                            restricted_correct += int(restricted_pred == ex.target_token)
                            target_idx = int(candidates.index(ex.target_token))
                            restricted_nll_sum += float(-torch.log_softmax(cand_logits, dim=0)[target_idx].item())
                            total += 1

                if total <= 0:
                    raise RuntimeError("Semi-natural probe evaluation produced zero targets.")
                rows.append(
                    {
                        "phase": "quick_natural",
                        "split": "semi_natural_wiki",
                        "model": args.model,
                        "dataset": args.dataset,
                        "task": "wiki_repeated_token_retrieval",
                        "span": int(span),
                        "seq_len": int(args.seq_len),
                        "intervention": intervention,
                        "random_draw": (None if random_draw is None else int(random_draw)),
                        "seed": int(seed),
                        "num_sequences": int(len(examples)),
                        "num_targets": int(total),
                        "mean_accuracy": float(restricted_correct / total),
                        "mean_nll": float(restricted_nll_sum / total),
                        "mean_accuracy_full_vocab": float(full_correct / total),
                        "mean_nll_full_vocab": float(full_nll_sum / total),
                        "mean_accuracy_restricted": float(restricted_correct / total),
                        "mean_nll_restricted": float(restricted_nll_sum / total),
                        "eval_mode": "restricted",
                        "candidate_count": int(args.candidate_size),
                        "chance_accuracy": float(1.0 / max(1, int(args.candidate_size))),
                        "candidate_policy_version": "semi_natural_restricted_v1_sequence_values",
                        "exploratory_only": True,
                    }
                )

    task_df = pd.DataFrame(rows).sort_values(["span", "seed", "intervention", "random_draw"])
    task_df.to_parquet(out_dir / "aggregate_task_metrics.parquet", engine="pyarrow", index=False)

    summary = (
        task_df.groupby(["span", "intervention"], as_index=False)
        .agg(
            n=("seed", "count"),
            mean_accuracy_restricted=("mean_accuracy_restricted", "mean"),
            mean_accuracy_full_vocab=("mean_accuracy_full_vocab", "mean"),
            mean_nll_restricted=("mean_nll_restricted", "mean"),
            mean_nll_full_vocab=("mean_nll_full_vocab", "mean"),
        )
        .sort_values(["intervention", "span"])
    )
    summary_payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "run_tag": str(args.run_tag),
        "model": str(args.model),
        "dataset": str(args.dataset),
        "seq_len": int(args.seq_len),
        "spans": [int(x) for x in spans],
        "seeds": [int(x) for x in seeds],
        "interventions": [x for x, _ in expanded],
        "random_draws": int(args.random_draws),
        "synthetic_count": int(args.synthetic_count),
        "candidate_size": int(args.candidate_size),
        "exploratory_only": True,
        "summary_rows": summary.to_dict(orient="records"),
        "notes": [
            "Probe examples are sourced from tokenized real wiki eval rows.",
            "Full-vocab and restricted metrics are computed from the same examples.",
            "This panel is exploratory and external-validity-oriented only.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    memo = [
        "# Semi-Natural Bridge Decision Memo",
        "",
        f"Run: `{args.run_tag}`",
        f"Model: `{args.model}`",
        "Panel type: exploratory-only (external validity).",
        "",
        "## Summary table",
    ]
    for row in summary.to_dict(orient="records"):
        memo.append(
            f"- span={int(row['span'])}, intervention={row['intervention']}: "
            f"restricted_acc={float(row['mean_accuracy_restricted']):.4f}, "
            f"full_vocab_acc={float(row['mean_accuracy_full_vocab']):.4f}"
        )
    (out_dir / "decision_memo.md").write_text("\n".join(memo) + "\n", encoding="utf-8")
    print(json.dumps(summary_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
