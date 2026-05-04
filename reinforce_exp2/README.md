# reinforce_exp2

Implementation package for the A/B/C pipeline in `reinforce_exp2/TODO.md`.

## Scope
This folder provides executable entrypoints for:

1. `A0` claim hygiene
2. `A1` evidence consolidation
3. `A2` preregistered prediction layer
4. `B1` kernel-shape taxonomy
5. `B2a` head-level kernel-boundary alignment
6. `B2b` cluster-level alignment enrichment
7. `B3` cluster-wise ablation disambiguation
8. `B4` confound-isolated SI-vs-low comparison
9. `C1` OLMo causal tracing carrier-class test
10. `C2` carrier-class conditional prediction tests

The implementation emphasizes strict/exploratory provenance, reproducibility manifests, and machine-readable artifacts under `results/reinforce_exp2/`.

## Layout

- `schemas/`: JSON schema contracts for artifacts.
- `scripts/`: per-experiment runnable scripts and pipeline orchestrator.
- `common.py`: shared paths/utilities.

## Basic usage

From repo root (`/jumbo/lisp/f004ndc/Kernel PE`):

```bash
.venv/bin/python reinforce_exp2/scripts/run_pipeline.py --phase A
.venv/bin/python reinforce_exp2/scripts/run_pipeline.py --phase B --models llama-3.1-8b,olmo-2-7b,mistral-7b-v0.1
.venv/bin/python reinforce_exp2/scripts/run_pipeline.py --phase C --models llama-3.1-8b,olmo-2-7b
```

Or run individual experiments:

```bash
.venv/bin/python reinforce_exp2/scripts/run_b1_kernel_taxonomy.py --models llama-3.1-8b,olmo-2-7b,mistral-7b-v0.1
.venv/bin/python reinforce_exp2/scripts/run_b2a_head_alignment.py --models llama-3.1-8b,olmo-2-7b,mistral-7b-v0.1
.venv/bin/python reinforce_exp2/scripts/run_b3_cluster_ablation.py --models llama-3.1-8b,olmo-2-7b,mistral-7b-v0.1 --device-map llama-3.1-8b:cuda:2,olmo-2-7b:cuda:3,mistral-7b-v0.1:cuda:4
```

## Smoke-first recommendation

Start with low-cost validation before full runs:

```bash
.venv/bin/python reinforce_exp2/scripts/run_a0_claim_hygiene.py
.venv/bin/python reinforce_exp2/scripts/run_a1_evidence_consolidation.py
.venv/bin/python reinforce_exp2/scripts/run_a2_prediction_layer.py

# B-path smoke (single model / low bootstrap)
.venv/bin/python reinforce_exp2/scripts/run_b1_kernel_taxonomy.py --models llama-3.1-8b --n-boot 5 --output-root results/reinforce_exp2/B1_kernel_taxonomy_smoke
.venv/bin/python reinforce_exp2/scripts/run_b2a_head_alignment.py --models llama-3.1-8b --num-sequences 8 --n-null 16 --seq-len 256 --output-root results/reinforce_exp2/B2a_head_alignment_smoke
.venv/bin/python reinforce_exp2/scripts/run_b2b_cluster_alignment.py --models llama-3.1-8b --b1-root results/reinforce_exp2/B1_kernel_taxonomy_smoke --b2a-root results/reinforce_exp2/B2a_head_alignment_smoke --output-root results/reinforce_exp2/B2b_cluster_alignment_smoke
.venv/bin/python reinforce_exp2/scripts/run_b3_cluster_ablation.py --models llama-3.1-8b --output-root results/reinforce_exp2/B3_cluster_ablation_smoke
.venv/bin/python reinforce_exp2/scripts/run_b4_confound_isolation.py --models llama-3.1-8b --b1-root results/reinforce_exp2/B1_kernel_taxonomy_smoke --output-root results/reinforce_exp2/B4_confound_isolation_smoke
```

Notes:
1. `B1` runtime scales steeply with `--n-boot` and model count; smoke first, then increase.
2. `C1` and `C2` are GPU-heavy and should be launched only after B-path smoke passes.

## Artifact policy
Each script writes:

1. `preregistration.json`
2. `manifest.json`
3. `summary.json`
4. `claim_impact.json`
5. `data_dictionary.json`

under `results/reinforce_exp2/<experiment_id>/...`.

## Notes

- Existing mature reinforcement scripts in `reinforce_exp/` are reused where possible for computationally heavy subroutines.
- `B3` and `C1` include explicit smoke modes for preflight validation before full runs.
