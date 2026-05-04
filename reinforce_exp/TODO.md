# Reinforcement Experiment Plan (Post-Submission Strengthening)

This document defines high-value follow-up experiments that directly strengthen the current paper's weakest points under a NeurIPS review standard.  
The goal is not breadth; it is to harden the central claims with targeted, decision-oriented evidence.

## Guiding principle

Each experiment below is designed to reduce one specific reviewer-critical uncertainty:

1. Llama strict-gate ambiguity.
2. Dependence-aware inference robustness.
3. Cross-model sample size for mechanism claims.
4. RoPE-vs-non-RoPE confounding.
5. Conditional-specialization validity beyond corpus-proxy regimes.

---

## EXP-R1: Llama Strict-Gate Final Adjudication

## Why it is necessary

The current paper's boundary mechanism claim is strongest in OLMo but ambiguous in Llama under strict controls.  
This is currently the single highest-impact caveat in main-text inference.

Without a stronger adjudication, reviewers can argue that Llama boundary effects are largely tokenizer-trivial.

## Hypothesis

Llama's strict-gate result is either:

1. stably artifact-positive (tokenizer-entangled), or
2. stably artifact-negative (clean), or
3. genuinely unstable (ambiguous).

The current 7-seed result suggests (3), but with pooled directional evidence. We need higher precision and stronger decision boundaries.

## Implementation

1. Run a 12-seed strict rerun of `3P2-B` for Llama only.
2. Fix sample sizes at:
   - `num_sequences >= 48`
   - `synthetic_target_per_cell >= 1000`
3. Keep strict controls identical to current canonical path:
   - no override,
   - same prefix-artifact decision rule,
   - same embedding-ablation protocol.
4. Add three text domains with matched token counts:
   - wiki (baseline),
   - dialogue-style corpus,
   - code-style corpus.
5. Pre-register pooled decision rule:
   - seed-level flag definition unchanged,
   - weighted Stouffer pooled one-sided test,
   - practical-effect threshold on fake-minus-real with-prefix delta,
   - explicit ambiguity zone.
6. Emit:
   - `multiseed_gate_summary_v2.json`
   - `multidomain_gate_summary_v1.json`
   - `gate_decision_manifest.json`

## Acceptance criteria

1. Stable clean: <= 2/12 flagged, pooled one-sided p > 0.2, pooled delta <= 0.
2. Stable entangled: >= 9/12 flagged, pooled one-sided p < 0.01, pooled delta >= practical threshold.
3. Otherwise: final canonical status remains `ambiguous` and should be locked as such.

## Expected impact

This experiment directly determines whether Llama can be promoted from caveated to clean in strict boundary claims, or whether ambiguity should be formalized as a substantive model property.

---

## EXP-R2: Dependence-Aware Statistical Reanalysis

## Why it is necessary

Current p-values in several analyses are based on large counts (heads/tokens/positions), which invites pseudoreplication criticism.  
Even when effect sizes are large, reviewers will expect dependence-aware confirmation.

## Hypothesis

Core conclusions (T8 causal relevance, 3P2-B boundary non-triviality, 3P2-C threshold preference, 3P2-J interactions) remain directionally and practically robust under clustered/mixed-effects inference.

## Implementation

1. Build a shared analysis script:
   - `scripts/reinforce_exp/r2_dependence_reanalysis.py`
2. Recompute key tests with hierarchical structure:
   - random intercepts for sequence, head, and seed where applicable,
   - cluster-robust SEs by sequence/seed blocks,
   - permutation tests at sequence level for sensitivity.
3. For each core claim, report:
   - original test output,
   - dependence-aware estimate,
   - practical effect size,
   - conclusion stability flag.
4. Emit:
   - `results/reinforce_exp/dependence_reanalysis/claim_stability_table.json`
   - `.../model_summaries/*.json`

## Acceptance criteria

1. Direction unchanged for all four central claims.
2. Practical effect remains above pre-specified MDE even if p-values weaken.
3. Any claim failing this test must be downgraded in main text.

## Expected impact

Substantially increases statistical credibility and preempts common NeurIPS reviewer objections on independence assumptions.

---

## EXP-R3: Third Primary-Scale Model Core Replication

## Why it is necessary

Main claims are currently grounded in two primary models.  
A third model with the same core battery materially improves confidence in the phrase “cross-model pattern.”

## Hypothesis

At least one additional 7–8B model reproduces the core E2 pattern (threshold-style redundancy), with potentially model-conditional boundary and regime overlays.

## Implementation

1. Select one additional 7–8B model not yet used as a partial tiebreaker in weakened form.
2. Run only the core load-bearing subset:
   - T8 kernel ablation,
   - 3P2-B strict,
   - 3P2-C.1/C.2,
   - 3P2-J powered regimes.
3. Keep exact analysis scripts and thresholds matched to current canonical pipeline.
4. Emit:
   - `results/reinforce_exp/core_replication/<model>/core_replication_report.json`
   - `.../cross_model_core_table.json`

## Acceptance criteria

1. If E2 pattern reproduces: strengthen “cross-model” claim to “across 3 primary models.”
2. If E2 fails: narrow claim to “two-model result” and discuss heterogeneity explicitly.

## Expected impact

This is the most compute-efficient way to increase generalizability credibility without running the full historical experiment program.

---

## EXP-R4: PE-Scheme Contrast Extension (RoPE vs Non-RoPE)

## Why it is necessary

Current non-RoPE evidence uses a single GPT-2 anchor.  
Reviewers may still argue PE-scheme confounding is unresolved.

## Hypothesis

SI structure and non-trivial boundary effects appear under at least one additional non-RoPE/alternative-PE model, supporting PE-scheme generality.

## Implementation

1. Add at least one additional non-RoPE or alternative PE model.
2. Run reduced robustness battery:
   - SI profiling (`R^2` distribution),
   - strict boundary non-triviality check,
   - minimal kernel ablation sanity check.
3. Keep compute bounded; this is a robustness extension, not a full mechanistic battery.
4. Emit:
   - `results/reinforce_exp/pe_scheme_control/pe_scheme_comparison.json`

## Acceptance criteria

1. Positive replication in at least one additional non-RoPE/alt-PE model:
   - SI structure detected,
   - non-trivial boundary effect after strict control.
2. If negative: claim must be narrowed to RoPE-facilitated SI organization.

## Expected impact

Converts the current “RoPE confound weakened” statement into a stronger cross-scheme robustness claim.

---

## EXP-R5: Task-Grounded Conditional Specialization Validation

## Why it is necessary

Current 3P2-J regimes are corpus-proxy definitions.  
Even with powered repair, reviewers can ask whether interaction effects generalize to explicit task-grounded conditions.

## Hypothesis

The SI-group × regime interaction remains significant under controlled synthetic/task-grounded long-span and uncertainty manipulations.

## Implementation

1. Construct synthetic evaluation sets for:
   - long-span retrieval with fixed distance bins,
   - high-uncertainty contexts with controlled entropy manipulations.
2. Reuse 3P2-J interaction analysis code with regime source switched from corpus-proxy to synthetic labels.
3. Match sample sizes to current powered run (`n_pairs >= 1200` per regime).
4. Emit:
   - `results/reinforce_exp/j_task_grounded/regime_summary.json`
   - `.../interaction_comparison_proxy_vs_task.json`

## Acceptance criteria

1. Interaction remains significant after multiplicity correction.
2. At least 3 of 4 regime-direction signs match the current proxy regime profile per model.
3. If mismatch: downgrade J interpretation to proxy-specific.

## Expected impact

Upgrades conditional-specialization evidence from “strong proxy evidence” to “task-grounded mechanistic evidence.”

---

## Suggested execution order

1. EXP-R2 (fastest credibility gain with existing data).
2. EXP-R1 (highest-impact caveat resolution).
3. EXP-R5 (strengthens one of the main claims).
4. EXP-R4 (robustness extension).
5. EXP-R3 (compute-heavier but highest generalization payoff).

---

## Reporting requirements for all reinforcement experiments

Every reinforcement experiment must output:

1. JSON manifest with exact command, model, seed set, and timestamp.
2. Explicit claim impact field:
   - `strengthens_main_claim`,
   - `requires_claim_downgrade`,
   - `no_change`.
3. Multiplicity family metadata and corrected p-values where inferential claims are made.
4. A one-page markdown summary in `results/reports/` suitable for direct paper updates.

