# Experiment 4: SI-Guided Fine-Tuning for Mathematical Reasoning

## Document Status

Authoring timestamp: **2026-04-09 (America/New_York)**.

This TODO defines a staged Experiment 4 program:
- `4C` diagnostic (mandatory first gate),
- `4A` SI-aware LoRA comparison,
- `4B` SI-loss-augmented extension (conditional on `4A` signal).

All outputs are additive and non-retroactive to Experiment 3/Phase 2 claims.

---

## Merit and Course-of-Action Assessment

### Agreement with planned direction

The proposed direction has merit and is technically coherent:
1. It correctly distinguishes **load-bearing SI infrastructure** from **inference-time gain scaling**.
2. It tests a plausible mechanism: training can re-route computation through existing SI channels without brute-force amplification.
3. It introduces a cheap high-value gate (`4C`) before expensive intervention sweeps.

### Important cautions

1. `4A` should be interpreted primarily as **targeted adaptation** evidence, not direct mechanistic proof.
2. `4B` introduces optimization confounds (aux-loss tuning); keep strict dev/test separation.
3. Keep generalization claims bounded to tested models/tasks unless scale replication is added.

Decision: proceed, with strict gating (`4C -> 4A -> 4B`) and prereg-style contracts below.

---

## Stage Gates

### Gate E4-G1 (`4C -> 4A`)

`4A` is authorized only after `4C` completes both models and produces:
1. Checkpoint SI trajectories at >=5 checkpoints.
2. Stable `R²` + boundary-`d` audit outputs.
3. A clear qualitative diagnosis: SI expands, contracts, or remains stable during standard math FT.

### Gate E4-G2 (`4A -> 4B`)

`4B` is authorized only if at least one of:
1. `4A(a)` vs `4A(b)` shows directional benefit for SI-protecting routing on math accuracy.
2. `4A` shows meaningful post-FT SI-structure change that motivates preservation/routing losses.

If neither condition holds, keep `4B` deferred.

---

## Shared Defaults

### Models
- `llama-3.1-8b`
- `olmo-2-7b`

### Math training mix
- Digit arithmetic (2–6 digits, carries)
- Counting tasks with traces
- Modular arithmetic
- Sequence continuation (arithmetic/geometric/Fibonacci-style)
- + 20% control general-text mix (wiki/instruction)

### SI head definition
- Reuse Experiment 3 T1 per-head `R²` ranking.
- SI heads = top quartile by `R²` per model.

### Core evaluation channels (all stages)
1. Held-out math accuracy.
2. Held-out wiki perplexity (forgetting check).
3. Post-FT `R²` profiling.
4. Post-FT abbreviated 3P2-B boundary detection.
5. Post-FT abbreviated 3P2-C.1 cumulative ablation curve.

---

## 4C: Diagnostic SI Trajectory During Standard FT

### Goal
Establish baseline SI-structure dynamics under standard math QLoRA before SI-targeted intervention design.

### Protocol
1. Standard math fine-tune per model.
2. Audit checkpoints: `0, 100, 500, 1000, 2000, final`.
3. At each checkpoint:
   - full `R²` profiling,
   - abbreviated boundary `d` measurement,
   - top-quartile SI head-set overlap vs step 0 baseline.
4. Produce trajectory plots/tables for SI expansion/shrinkage/shift.

### Acceptance criteria
1. >=5 checkpoint audit points completed per model.
2. Trajectory outputs emitted for both models.

### Artifact contract
- `results/experiment4/exp4c_si_trajectory_during_ft/<model>/run_manifest.json`
- `results/experiment4/exp4c_si_trajectory_during_ft/<model>/si_trajectory_during_ft.parquet`
- `results/experiment4/exp4c_si_trajectory_during_ft/<model>/si_trajectory_summary.json`

### Budget
- Estimated 8–12 GPU-hours total.

---

## 4A: SI-Aware LoRA Fine-Tuning

### Goal
Test whether SI-protecting adaptation outperforms uniform adaptation for math while preserving SI structure.

### Conditions (5 total)
1. `a_si_protecting_lora`:
   - LoRA on non-SI heads + MLPs.
   - SI projections frozen or rank-1 LoRA only.
2. `b_uniform_lora`:
   - Uniform rank on attention + MLP projections only.
3. `c_si_only_lora`:
   - LoRA only on SI heads.
4. `d_full_qlora_baseline`:
   - Full baseline rank across all `nn.Linear` modules (broader than `b`).
5. `e_si_amplified_lora`:
   - SI heads receive higher rank while non-SI heads receive reduced rank.

### Protocol
1. Use same math mix and control mix for all conditions.
2. Minimum 3 seeds per condition (paired comparisons by seed).
3. Post-training: math eval + forgetting eval + SI audit + abbreviated C.1.
4. Primary comparison: condition `(a)` vs `(b)`.

### Acceptance criteria
1. >=3 seeds per condition per model.
2. Paired effect + CI for `(a)` vs `(b)` on math accuracy.
3. At least one condition shows measurable post-FT SI-structure movement (distribution shift and/or boundary signal change).

### Artifact contract
- `results/experiment4/exp4a_si_aware_lora/<model>/run_manifest.json`
- `results/experiment4/exp4a_si_aware_lora/<model>/si_lora_comparison.json`
- `results/experiment4/exp4a_si_aware_lora/<model>/post_ft_si_audit.json`
- `results/experiment4/exp4a_si_aware_lora/<model>/post_ft_ablation_curve.parquet`

### Budget
- Estimated 24–40 GPU-hours total.

---

## 4B: SI-Loss-Augmented Training

### Goal
Test whether explicit SI-preservation/routing losses improve math outcomes and SI retention.

### Losses
1. SI-preservation loss:
   - penalize degradation of top-quartile SI-head `R²` vs baseline.
2. SI-routing loss:
   - encourage SI-head attention to math-critical token positions.
3. Combined loss:
   - use best lambdas from individual sweeps.

### Conditions
1. `a_standard_ft` (no aux loss).
2. `b_si_preserve` with `lambda in {0.01, 0.1, 1.0}`.
3. `c_si_routing` with `lambda in {0.01, 0.1, 1.0}`.
4. `d_combined` using best lambda selections from b/c.

### Protocol
1. Use 4A infra/eval harness unchanged.
2. Run lambda sweeps on dev split only.
3. Compare best-lambda condition(s) against standard FT baseline.

### Acceptance criteria
1. Full lambda sweeps complete.
2. Best-lambda paired comparison vs baseline produced.
3. SI-preservation metrics improve in `(b)` and/or `(d)` vs `(a)` without unacceptable forgetting.

### Artifact contract
- `results/experiment4/exp4b_si_loss_augmented/<model>/run_manifest.json`
- `results/experiment4/exp4b_si_loss_augmented/<model>/si_loss_sweep.json`
- `results/experiment4/exp4b_si_loss_augmented/<model>/si_routing_results.json`
- `results/experiment4/exp4b_si_loss_augmented/<model>/post_ft_si_audit.json`

### Budget
- Estimated 40–60 GPU-hours total.

---

## Execution Order

1. `4C` (mandatory diagnostic gate).
2. `4A` (core SI-aware LoRA test).
3. `4B` (conditional extension after `4A` signal).

---

## Scaffolding Commands

### List entrypoints
```bash
python -m experiment4.run --list
```

### Emit `4C` manifests
```bash
python -m experiment4.run 4c --model all --device-map llama-3.1-8b:cuda:0,olmo-2-7b:cuda:1
```

### Emit `4A` manifests
```bash
python -m experiment4.run 4a --model all --device-map llama-3.1-8b:cuda:0,olmo-2-7b:cuda:1
```

### Emit `4B` manifests
```bash
python -m experiment4.run 4b --model all --device-map llama-3.1-8b:cuda:0,olmo-2-7b:cuda:1
```

### Emit all in gate order
```bash
python -m experiment4.run all --model all --device-map llama-3.1-8b:cuda:0,olmo-2-7b:cuda:1
```

---

## Open Implementation TODOs

1. Implement SI-head extraction from Experiment 3 T1 artifacts in FT dataloaders.
2. Implement math-mix dataset builders with deterministic splits and schema checks.
3. Implement QLoRA training loops and budget-matched LoRA adapter policies.
4. Implement post-FT audit harness reusing Experiment 3 metrics (`R²`, 3P2-B, abbreviated C.1).
5. Implement paired-statistics report generator with Holm correction where needed.
