#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp.common import RESULTS_ROOT, command_manifest, ensure_dir, safe_float, timestamp_now, write_json  # noqa: E402
from experiment3.theory1_si_circuits import MODELS, compute_per_head_r2  # noqa: E402
from experiment5.pipeline import _load_code_sequences, _load_dialogue_sequences, _load_wiki_sequences  # noqa: E402
from shared.attention.adapters import get_adapter  # noqa: E402
from shared.models.loading import load_model, load_tokenizer  # noqa: E402

TARGET_MODELS = ("llama-3.1-8b", "olmo-2-7b")
DOMAINS = ("wiki", "code", "dialogue")
DEFAULT_OUT = RESULTS_ROOT / "exp_new_r15_head_stability"


def _load_domain_sequences(
    *,
    domain: str,
    model_name: str,
    tokenizer,
    seq_len: int,
    max_sequences: int,
    seed: int,
) -> list[list[int]]:
    if domain == "wiki":
        seqs = _load_wiki_sequences(model_name, tokenizer, seq_len=seq_len, max_sequences=max_sequences, seed=seed)
    elif domain == "code":
        seqs = _load_code_sequences(model_name, tokenizer, seq_len=seq_len, max_sequences=max_sequences, seed=seed + 17)
    elif domain == "dialogue":
        seqs = _load_dialogue_sequences(tokenizer, seq_len=seq_len, max_sequences=max_sequences, seed=seed + 31)
    else:
        raise ValueError(f"Unsupported domain={domain}")
    return seqs[:max_sequences]


def _top_set(mean_r2: pd.DataFrame, quantile: float) -> set[tuple[int, int]]:
    work = mean_r2.sort_values("mean_r2", ascending=False).reset_index(drop=True)
    n = len(work)
    n_sel = max(1, int(math.floor(float(n) * float(quantile))))
    top = work.head(n_sel)
    return {(int(r.layer), int(r.head)) for r in top.itertuples()}


def _jaccard(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> float:
    u = a | b
    if not u:
        return float("nan")
    return float(len(a & b) / len(u))


def _bootstrap_jaccard_ci(
    *,
    a: set[tuple[int, int]],
    b: set[tuple[int, int]],
    universe: list[tuple[int, int]],
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    obs = _jaccard(a, b)
    if not universe:
        return obs, float("nan"), float("nan")
    idx = {h: i for i, h in enumerate(universe)}
    avec = np.zeros(len(universe), dtype=np.int8)
    bvec = np.zeros(len(universe), dtype=np.int8)
    for h in a:
        if h in idx:
            avec[idx[h]] = 1
    for h in b:
        if h in idx:
            bvec[idx[h]] = 1
    rng = np.random.default_rng(seed)
    boot = np.empty(max(2000, int(n_boot)), dtype=np.float64)
    n = len(universe)
    for i in range(len(boot)):
        sidx = rng.integers(0, n, size=n)
        aa = avec[sidx]
        bb = bvec[sidx]
        inter = int(np.sum((aa == 1) & (bb == 1)))
        union = int(np.sum((aa == 1) | (bb == 1)))
        boot[i] = float(inter / max(1, union))
    return obs, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _expected_random_jaccard(n_universe: int, k_set: int) -> float:
    # E[|A∩B|] = k^2 / N for random sets of size k from universe N.
    if n_universe <= 0 or k_set <= 0:
        return float("nan")
    e_inter = (float(k_set) * float(k_set)) / float(n_universe)
    e_union = float(2 * k_set) - e_inter
    return float(e_inter / max(1e-8, e_union))


def _spearman_pair(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> float:
    l = left[["layer", "head", "mean_r2"]].rename(columns={"mean_r2": "r2_left"})
    r = right[["layer", "head", "mean_r2"]].rename(columns={"mean_r2": "r2_right"})
    m = l.merge(r, on=["layer", "head"], how="inner")
    if len(m) < 3:
        return float("nan")
    rho, _ = scipy_stats.spearmanr(m["r2_left"].astype(float).values, m["r2_right"].astype(float).values)
    return safe_float(rho)


def run_model(
    *,
    model_name: str,
    device: str,
    output_root: Path,
    seq_len: int,
    num_sequences: int,
    quantile: float,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    t0 = time.time()
    out_dir = output_root / model_name
    ensure_dir(out_dir)

    model_spec = MODELS[model_name]
    loaded = load_model(model_spec)
    model = loaded.model.to(device)
    model.eval()
    tokenizer = load_tokenizer(model_spec)
    adapter = get_adapter(model_spec)
    adapter.register(model)

    domain_r2: dict[str, pd.DataFrame] = {}
    domain_top: dict[str, set[tuple[int, int]]] = {}
    domain_meta: dict[str, Any] = {}
    all_heads_set: set[tuple[int, int]] = set()

    for idx, domain in enumerate(DOMAINS):
        seqs = _load_domain_sequences(
            domain=domain,
            model_name=model_name,
            tokenizer=tokenizer,
            seq_len=max(64, int(seq_len)),
            max_sequences=max(8, int(num_sequences)),
            seed=int(seed) + idx * 101,
        )
        if len(seqs) < 4:
            raise RuntimeError(f"[NEW-R15] domain={domain} has only {len(seqs)} sequences for model={model_name}")

        t_dom = time.time()
        per_seq = compute_per_head_r2(
            model=model,
            adapter=adapter,
            tokenizer=tokenizer,
            model_spec=model_spec,
            device=device,
            sequences=seqs,
        )
        if per_seq.empty:
            raise RuntimeError(f"[NEW-R15] empty per-sequence R2 for domain={domain} model={model_name}")
        mean_r2 = (
            per_seq.groupby(["layer", "head"], as_index=False)["r2"]
            .mean()
            .rename(columns={"r2": "mean_r2"})
            .sort_values(["layer", "head"])
            .reset_index(drop=True)
        )
        top = _top_set(mean_r2, float(quantile))
        domain_r2[domain] = mean_r2
        domain_top[domain] = top
        all_heads_set |= set((int(r.layer), int(r.head)) for r in mean_r2.itertuples())

        per_seq.to_parquet(out_dir / f"per_sequence_r2_{domain}.parquet", index=False)
        mean_r2.to_parquet(out_dir / f"head_r2_summary_{domain}.parquet", index=False)
        domain_meta[domain] = {
            "n_sequences": int(len(seqs)),
            "n_heads": int(len(mean_r2)),
            "n_top_quartile": int(len(top)),
            "elapsed_sec": float(time.time() - t_dom),
        }
        print(
            f"[NEW-R15][{model_name}] domain={domain} n_seq={len(seqs)} "
            f"n_heads={len(mean_r2)} n_top={len(top)} elapsed={domain_meta[domain]['elapsed_sec']:.1f}s",
            flush=True,
        )

    universe = sorted(all_heads_set)
    n_univ = len(universe)
    k = len(next(iter(domain_top.values()))) if domain_top else 0
    expected_j = _expected_random_jaccard(n_universe=n_univ, k_set=k)

    pair_names = [("wiki", "code"), ("wiki", "dialogue"), ("code", "dialogue")]
    jaccard_rows: list[dict[str, Any]] = []
    spearman_rows: list[dict[str, Any]] = []
    pair_jaccards: list[float] = []
    pair_rhos: list[float] = []
    for pidx, (a, b) in enumerate(pair_names):
        obs, ci_lo, ci_hi = _bootstrap_jaccard_ci(
            a=domain_top[a],
            b=domain_top[b],
            universe=universe,
            n_boot=max(2000, int(n_boot)),
            seed=int(seed) + pidx * 313,
        )
        rho = _spearman_pair(domain_r2[a], domain_r2[b])
        pair_jaccards.append(obs)
        pair_rhos.append(rho)
        jaccard_rows.append(
            {
                "pair": f"{a}__{b}",
                "jaccard": safe_float(obs),
                "ci95": [safe_float(ci_lo), safe_float(ci_hi)],
                "ci_excludes_random_expected": bool(np.isfinite(ci_lo) and np.isfinite(expected_j) and ci_lo > expected_j),
            }
        )
        spearman_rows.append({"pair": f"{a}__{b}", "rho": safe_float(rho)})

    inter_all = domain_top["wiki"] & domain_top["code"] & domain_top["dialogue"]
    union_all = domain_top["wiki"] | domain_top["code"] | domain_top["dialogue"]
    j_three = float(len(inter_all) / max(1, len(union_all)))

    head_counts: list[dict[str, Any]] = []
    for layer, head in universe:
        present = sum(1 for d in DOMAINS if (layer, head) in domain_top[d])
        row = {"layer": int(layer), "head": int(head), "top_quartile_presence_count": int(present)}
        for d in DOMAINS:
            m = domain_r2[d]
            v = m[(m["layer"] == int(layer)) & (m["head"] == int(head))]["mean_r2"]
            row[f"mean_r2_{d}"] = safe_float(v.iloc[0]) if len(v) else float("nan")
        head_counts.append(row)

    head_counts = sorted(
        head_counts,
        key=lambda r: (
            -int(r["top_quartile_presence_count"]),
            -safe_float(np.nanmean([r.get("mean_r2_wiki"), r.get("mean_r2_code"), r.get("mean_r2_dialogue")])),
            int(r["layer"]),
            int(r["head"]),
        ),
    )

    pair_j = np.asarray([x for x in pair_jaccards if np.isfinite(x)], dtype=np.float64)
    pair_r = np.asarray([x for x in pair_rhos if np.isfinite(x)], dtype=np.float64)
    stable = bool(pair_j.size == 3 and np.all(pair_j >= 0.65) and pair_r.size == 3 and np.all(pair_r >= 0.70))
    unstable = bool(pair_j.size == 3 and np.any(pair_j <= 0.45))
    if stable:
        status = "stable_high_confidence"
    elif unstable:
        status = "unstable"
    else:
        status = "partially_stable"

    summary = {
        "timestamp": timestamp_now(),
        "experiment": "NEW-R15",
        "model": model_name,
        "settings": {
            "seq_len": int(seq_len),
            "num_sequences_per_domain": int(num_sequences),
            "quantile": float(quantile),
            "n_boot": int(n_boot),
            "seed": int(seed),
        },
        "domain_meta": domain_meta,
        "universe_size": int(n_univ),
        "top_set_size": int(k),
        "hypergeometric_expected_jaccard": safe_float(expected_j),
        "pairwise_jaccard": jaccard_rows,
        "pairwise_spearman": spearman_rows,
        "three_way": {
            "intersection_size": int(len(inter_all)),
            "union_size": int(len(union_all)),
            "jaccard": safe_float(j_three),
        },
        "verdict": {
            "status": status,
            "stable_high_confidence": bool(status == "stable_high_confidence"),
            "partially_stable": bool(status == "partially_stable"),
            "unstable": bool(status == "unstable"),
        },
        "runtime_sec": float(time.time() - t0),
    }

    write_json(out_dir / "stability_summary.json", summary)
    write_json(out_dir / "per_head_consistency_scores.json", {"rows": head_counts})
    write_json(
        out_dir / "stable_core_set.json",
        {
            "model": model_name,
            "n_heads": int(len(inter_all)),
            "heads": [{"layer": int(l), "head": int(h)} for (l, h) in sorted(inter_all)],
        },
    )
    return summary


def _parse_device_map(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in [x.strip() for x in str(raw).split(",") if x.strip()]:
        model, dev = tok.split(":", 1)
        out[model.strip()] = dev.strip()
    return out


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.output_root)
    ensure_dir(out_root)
    reports: dict[str, Any] = {}
    dmap = _parse_device_map(args.device_map)
    for model in TARGET_MODELS:
        dev = dmap.get(model, "cuda:0")
        cmd = [
            str(ROOT / ".venv" / "bin" / "python"),
            "-u",
            "reinforce_exp/exp_new_r15_head_stability.py",
            "--model",
            model,
            "--device",
            dev,
            "--output-root",
            str(out_root),
            "--seq-len",
            str(args.seq_len),
            "--num-sequences",
            str(args.num_sequences),
            "--quantile",
            str(args.quantile),
            "--n-boot",
            str(args.n_boot),
            "--seed",
            str(args.seed),
        ]
        print("[NEW-R15] exec:", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(ROOT), check=True)
        reports[model] = json.loads((out_root / model / "stability_summary.json").read_text(encoding="utf-8"))

    payload = {"timestamp": timestamp_now(), "experiment": "NEW-R15", "models": reports}
    write_json(out_root / "aggregate_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NEW-R15: high-SI head identity stability across corpora")
    p.add_argument("--model", choices=["all", *TARGET_MODELS], default="all")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device-map", default="llama-3.1-8b:cuda:0,olmo-2-7b:cuda:1")
    p.add_argument("--output-root", default=str(DEFAULT_OUT))
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--num-sequences", type=int, default=500)
    p.add_argument("--quantile", type=float, default=0.25)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=17)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    ensure_dir(out_root)

    if args.model == "all":
        agg = run_all(args)
        write_json(
            out_root / "manifest.json",
            command_manifest(
                experiment_id="NEW-R15",
                command="all_models",
                model="+".join(TARGET_MODELS),
                extras={
                    "seq_len": int(args.seq_len),
                    "num_sequences": int(args.num_sequences),
                    "quantile": float(args.quantile),
                    "output_root": str(out_root),
                },
            ),
        )
        print(f"[NEW-R15] wrote {out_root / 'aggregate_summary.json'}")
        print(f"[NEW-R15] models={list(agg['models'].keys())}")
        return

    rep = run_model(
        model_name=str(args.model),
        device=str(args.device),
        output_root=out_root,
        seq_len=max(64, int(args.seq_len)),
        num_sequences=max(8, int(args.num_sequences)),
        quantile=max(0.05, min(0.5, float(args.quantile))),
        n_boot=max(1000, int(args.n_boot)),
        seed=int(args.seed),
    )
    write_json(
        out_root / str(args.model) / "manifest.json",
        command_manifest(
            experiment_id="NEW-R15",
            command="single_model",
            model=str(args.model),
            extras={
                "device": str(args.device),
                "seq_len": int(args.seq_len),
                "num_sequences": int(args.num_sequences),
                "quantile": float(args.quantile),
                "output_root": str(out_root),
            },
        ),
    )
    print(f"[NEW-R15] wrote {out_root / str(args.model) / 'stability_summary.json'}")
    print(f"[NEW-R15] status={rep['verdict']['status']}")


if __name__ == "__main__":
    main()
