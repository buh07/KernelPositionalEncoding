# Experiment 5: Systematic Tokenizer Effect Audit

## Document Status

Authoring timestamp: **2026-04-09 (America/New_York)**.

This TODO defines a staged tokenizer audit program:
- `5A` cross-tokenizer SI profiling (mandatory first),
- `5B` same-family tokenizer variation,
- `5C` perturbation battery and `5D` shift fragility (parallel after `5A`).

All outputs are additive and non-retroactive to Experiment 3/Phase 2 claims.

---

## Merit and Course-of-Action Assessment

### Agreement with planned direction

The experiment has strong merit:
1. It directly tests whether the Llama prefix-following issue is tokenizer-family-driven or model-specific.
2. It is relatively cheap in GPU budget for high interpretability value.
3. It complements existing `3P2-B / 3P2-I / 3P2-K` evidence with systematic controls.

### Critical methodological guardrails

1. Keep tokenizer and model-family claims clearly separated; do not over-claim causal attribution from observational comparisons alone.
2. Use consistent SI metrics and boundary-control definitions across all model families.
3. Maintain strong provenance for perturbation and adversarial data generation.

Decision: proceed, with strict `5A` first-gate before expanding to `5B/5C/5D`.

---

## Stage Gates

### Gate E5-G1 (`5A -> 5B/5C/5D`)

Proceed only if `5A` delivers:
1. New Pythia SI profiles and tokenizer-control outputs.
2. A complete tokenizer-entanglement matrix (>=6 models x 4 features).
3. A clear directional read on tokenizer-family vs model-family signal.

If `5A` is inconclusive, prioritize improving `5A` measurement quality before scaling to `5C/5D`.

---

## Shared Defaults

### Core tokenizer features
- `space_prefix`
- `capitalization_marker`
- `punctuation_adjacency`
- `token_length_bucket`

### Shared metrics
1. Per-head `R²` distribution.
2. SI head density and layer concentration.
3. Post-control boundary effect `d`.
4. Prefix-following artifact status.

---

## 5A: Cross-Tokenizer SI Profiling (Pythia + Existing Baselines)

### Goal
Establish cross-family tokenizer entanglement patterns using Pythia, GPT-2, Llama 3.1, and OLMo.

### Models
- `pythia-410m`, `pythia-1.4b`
- `gpt2-small`, `gpt2-medium`
- `llama-3.1-8b`, `olmo-2-7b`

### Protocol
1. Run Experiment 1 Track A (`R²`) for Pythia models.
2. Reuse existing SI/boundary artifacts where available for GPT-2/Llama/OLMo.
3. Run `3P2-B`-style tokenizer controls for Pythia models.
4. Build model x feature entanglement matrix.

### Acceptance criteria
1. Pythia `R²` profiles complete.
2. Pythia tokenizer controls complete.
3. Matrix emitted with >=6 models and all 4 feature channels.

### Artifact contract
- `results/experiment5/exp5a_cross_tokenizer_si_profiling/run_manifest.json`
- `results/experiment5/exp5a_cross_tokenizer_si_profiling/cross_tokenizer_r2_profiles.parquet`
- `results/experiment5/exp5a_cross_tokenizer_si_profiling/tokenizer_entanglement_matrix.json`

### Budget
- Estimated 6–10 GPU-hours.

---

## 5B: Same-Family Tokenizer Variation (Llama 2 vs Llama 3.1)

### Goal
Test SI head identity and boundary artifact stability within Llama family under tokenizer change.

### Models
- `meta-llama/Llama-2-7b-hf`
- `meta-llama/Meta-Llama-3.1-8B`

### Protocol
1. Run `R²` profiling for Llama-2-7b.
2. Compare SI head identity:
   - top-quartile Jaccard overlap,
   - per-head `R²` rank correlation.
3. Run `3P2-B` controls on Llama-2-7b.
4. Run abbreviated English-only invariance check for Llama-2-7b.

### Acceptance criteria
1. Llama-2 profile + 3P2-B outputs complete.
2. Head-overlap metrics complete.
3. Prefix-following status comparison reported.

### Artifact contract
- `results/experiment5/exp5b_same_family_tokenizer_variation/run_manifest.json`
- `results/experiment5/exp5b_same_family_tokenizer_variation/llama_tokenizer_comparison.json`
- `results/experiment5/exp5b_same_family_tokenizer_variation/head_identity_overlap.json`

### Budget
- Estimated 8–12 GPU-hours.

---

## 5C: Synthetic Tokenizer Perturbation Battery

### Goal
Map SI sensitivity across controlled perturbation classes beyond current 3P2-B.2 controls.

### Models
- `llama-3.1-8b`, `olmo-2-7b` (extend to others if `5A` schedule allows)

### Perturbations
1. `fake_boundary_with_prefix`
2. `real_boundary_no_prefix`
3. `morpheme_internal_break`
4. `character_decompose`
5. `merge_across_boundary`
6. `random_resegment`

### Protocol
1. >=200 positions per perturbation per model.
2. Measure:
   - high-SI boundary attention delta,
   - local span `R²` delta,
   - prefix-following status change.
3. Build sensitivity matrix with CIs.

### Acceptance criteria
1. Coverage threshold met for each perturbation.
2. Effect size + CI emitted for each matrix cell.

### Artifact contract
- `results/experiment5/exp5c_synthetic_tokenizer_perturbation/run_manifest.json`
- `results/experiment5/exp5c_synthetic_tokenizer_perturbation/perturbation_sensitivity_matrix.json`
- `results/experiment5/exp5c_synthetic_tokenizer_perturbation/perturbation_results.parquet`

### Budget
- Estimated 16–24 GPU-hours (two-model execution).

---

## 5D: Distribution-Shift Fragility Test

### Goal
Test whether tokenizer-entangled SI circuits are less robust to distribution shift.

### Models
- `llama-3.1-8b`, `olmo-2-7b`

### Shift conditions
1. Code (Python)
2. Dialogue
3. Technical/scientific
4. Adversarial BPE

### Protocol
1. >=100 sequences per condition per model.
2. Run SI profiling + boundary detection under each condition.
3. Report deltas from Wikipedia baseline.
4. Run model x condition interaction test.

### Acceptance criteria
1. Coverage threshold met.
2. Delta tables + CIs complete.
3. Interaction test result emitted.

### Artifact contract
- `results/experiment5/exp5d_distribution_shift_fragility/run_manifest.json`
- `results/experiment5/exp5d_distribution_shift_fragility/distribution_shift_fragility.json`
- `results/experiment5/exp5d_distribution_shift_fragility/shift_results.parquet`

### Budget
- Estimated 6–10 GPU-hours.

---

## Execution Order

1. `5A` mandatory first.
2. `5B` second.
3. `5C` and `5D` parallel after `5A` completion (and preferably after initial `5B` signal check).

---

## Scaffolding Commands

### List entrypoints
```bash
python -m experiment5.run --list
```

### Emit `5A` manifest
```bash
python -m experiment5.run 5a --device cuda:0
```

### Emit `5B` manifest
```bash
python -m experiment5.run 5b --device cuda:0
```

### Emit `5C` manifest
```bash
python -m experiment5.run 5c --device cuda:0
```

### Emit `5D` manifest
```bash
python -m experiment5.run 5d --device cuda:0
```

### Emit all manifests in priority order
```bash
python -m experiment5.run all --device cuda:0
```

---

## Open Implementation TODOs

1. Add tokenizer-family metadata normalization for all compared models.
2. Build unified SI profile loader across Exp1/Exp3 namespaces.
3. Implement `3P2-B`-compatible tokenizer-control runner for non-Phase-2 model families.
4. Implement perturbation generator with deterministic seed and auditable text-level edits.
5. Implement shift-benchmark data builders (dialogue, technical, adversarial BPE).

