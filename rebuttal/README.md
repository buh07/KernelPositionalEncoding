# SI Rebuttal Runner

Install the rebuttal package in its own environment from `rebuttal/`:

```bash
cd rebuttal
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

For a verification-only repository smoke entrypoint, invoke `bin/research-smoke`. That
repository-root shim must point at `../rebuttal/scripts/research-smoke.py`; the implementation
stays under `rebuttal/scripts/`. The smoke command keeps all mutable output in a temporary root,
hides every GPU from its subprocesses, and disables Python bytecode/cache writes inside the
repository.

Set the frozen mutable roots under `rebuttal/` plus the three local model/tokenizer trees:

```bash
export SI_REBUTTAL_RUNS_ROOT=runs
export SI_REBUTTAL_DATA_ROOT=data/materialized
export SI_REBUTTAL_LOGS_ROOT=logs
export SI_REBUTTAL_CACHE_ROOT=cache
export SI_REBUTTAL_MODELS_ROOT=/absolute/models-root
export SI_REBUTTAL_MODEL_LLAMA_3_1_8B=/absolute/llama-3.1-8b
export SI_REBUTTAL_TOKENIZER_LLAMA_3_1_8B=/absolute/llama-3.1-8b
export SI_REBUTTAL_MODEL_MISTRAL_7B_V0_1=/absolute/mistral-7b-v0.1
export SI_REBUTTAL_TOKENIZER_MISTRAL_7B_V0_1=/absolute/mistral-7b-v0.1
export SI_REBUTTAL_MODEL_OLMO_2_7B=/absolute/olmo-2-7b
export SI_REBUTTAL_TOKENIZER_OLMO_2_7B=/absolute/olmo-2-7b
```

The exact CLI order is fixed. `materialize-data` freezes the batch manifest first, including the
original UTC materialization timestamp plus the full `pip freeze --all` and `nvidia-smi`
command/output snapshot; every later command reuses that immutable root and must match the stored
environment snapshot exactly.

```bash
RUN_ROOT=rebuttal/runs/si-r2

PYTHONPATH=rebuttal/src python -m si_rebuttal validate-config
PYTHONPATH=rebuttal/src python -m si_rebuttal toy-smoke --run-root rebuttal/runs/toy-smoke
PYTHONPATH=rebuttal/src python -m si_rebuttal materialize-data --run-root "${RUN_ROOT}"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=rebuttal/src python -m si_rebuttal validate-model --run-root "${RUN_ROOT}" --model llama-3.1-8b
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=rebuttal/src python -m si_rebuttal validate-model --run-root "${RUN_ROOT}" --model mistral-7b-v0.1
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=rebuttal/src python -m si_rebuttal validate-model --run-root "${RUN_ROOT}" --model olmo-2-7b

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=rebuttal/src python -m si_rebuttal benchmark-model --run-root "${RUN_ROOT}" --model llama-3.1-8b
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=rebuttal/src python -m si_rebuttal benchmark-model --run-root "${RUN_ROOT}" --model mistral-7b-v0.1
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=rebuttal/src python -m si_rebuttal benchmark-model --run-root "${RUN_ROOT}" --model olmo-2-7b

PYTHONPATH=rebuttal/src python -m si_rebuttal admit-lanes --run-root "${RUN_ROOT}"
```

`benchmark-model` persists two exact projection receipts per model:
`projected_total_seconds = sum(component.measured_seconds * (component.projected_sequence_count / component.measured_sequence_count) for component in runtime_components)`
`projected_artifact_bytes = immutable_materialized_bytes + ceil(future_artifact_bytes * future_margin_multiplier)`

`admit-lanes` recomputes those benchmark operands, then persists exact lane and disk formulas:
`gpu0_hours = sum(benchmark_lane_hours[model] for model in placement_gpu0); gpu1_hours = sum(benchmark_lane_hours[model] for model in placement_gpu1); admitted = (gpu0_hours <= 120 and gpu1_hours <= 120)`
`required_free_bytes = 2 * projected_complete_artifact_bytes + 20 * 1024**3; admitted = free_disk_bytes >= required_free_bytes`

Dry-run planning and launch status are read-only and must point at the same immutable run root:

```bash
PYTHONPATH=rebuttal/src python -m si_rebuttal launch --batch-id si-r2 --run-root "${RUN_ROOT}" --dry-run
PYTHONPATH=rebuttal/src python -m si_rebuttal status --run-root "${RUN_ROOT}"
```

The dry-run payload must echo the exact admitted `batch_manifest_sha256` and
`lane_admission_receipt_sha256`, plus the planned per-session script paths, log paths, and tmux
commands. `status` must report the same bindings from any persisted launch receipts, along with
script readability and log tails; cross-run or forged identities are protocol violations.

Real launch is the sole production entry. After `materialize-data`, all three `validate-model`
commands, all three `benchmark-model` commands, and `admit-lanes`, only `launch --execute` may
start compute. Direct `run-model --execute` is prohibited and fails closed unless the exact
launcher-written receipt context is present.

Real launch reuses the already materialized/admitted root and rejects only tmux-session or
launch-marker collisions:

```bash
PYTHONPATH=rebuttal/src python -m si_rebuttal launch --batch-id si-r2 --run-root "${RUN_ROOT}" --execute
PYTHONPATH=rebuttal/src python -m si_rebuttal finalize-batch --run-root "${RUN_ROOT}"
```

Before tmux startup, launch writes one deterministic shell script and one log file per exact tmux
session under the admitted run root, then writes per-session receipts and a batch receipt that all
embed the exact admitted `batch_manifest_sha256`, exact lane-admission receipt identity, exact
receipt paths, exact script paths and script digests, exact log paths, the single physical
`CUDA_VISIBLE_DEVICES`, and the fixed logical device `cuda:0`. Startup is valid only if those
bindings match the immutable run root already admitted by `admit-lanes`.

The fixed preflight placement is not advisory: Llama and OLMo must run with `CUDA_VISIBLE_DEVICES=0`,
Mistral must run with `CUDA_VISIBLE_DEVICES=1`, and each command then loads the model on logical
`cuda:0` inside that single-device namespace.

Claim boundary: the runner writes immutable manifests, validation receipts, benchmark receipts, lane-admission receipts, per-condition shards, summaries, and launch receipts. Completion is only the successful `finalize-batch` terminal artifact; it is not itself a scientific claim.
