# Reinforcement Experiment Plan v2 (NeurIPS 2026 Hardening)

Date: 2026-04-17  
Scope: Post-revision strengthening focused on reviewer-critical weaknesses that remain after integrating R1/R2/R4/R5.

---

## ACTION-0: Paper update required from existing completed data (no new experiments)

**EXP-R3 (Mistral-7B core replication) is fully complete as of 2026-04-17.**

All five subtests have results in `results/reinforce_exp/exp_r3_core_replication/mistral-7b-v0.1/`:

| Subtest | Result | Key numbers |
|---------|--------|-------------|
| T8 kernel ablation | ✓ Replicated | +3.43 nats (+185%), d=14.6, p<10⁻¹¹⁷; low-SI control +0.017 nats |
| 3P2-B strict boundary | ✓ Clean | boundary\_non\_trivial=true, prefix\_artifact=false |
| 3P2-C.1 threshold preference | ✓ Replicated | 6/6 threshold votes (all criteria, all tasks) |
| 3P2-C.2 superlinearity | ✓ Not supported | 1/3 votes; pooled test supports\_nonlinearity=false |
| 3P2-J regime interaction | ✓ Replicated | F=1229.36, p=1.11×10⁻¹⁶, η²=0.744; all 4 regimes high>low |

**Required paper updates before any new experiments run:**

1. Remove "three-model core replication: pending" language from Table 2 (Claim Robustness Matrix) and Limits section — replace with "three-model replication complete."
2. Promote T8 kernel ablation to **three-model claim**: Llama (+4.15 nats, d=11.6), OLMo (+0.058 nats, d=0.81), Mistral (+3.43 nats, d=14.6).
3. Promote threshold preference (C1) to **three-model claim**: 6/6 votes in all three models.
4. Update boundary claim: Mistral is clean (joins OLMo). Boundary non-triviality is now clean in 2/3 models (OLMo and Mistral); Llama ambiguous. Boundary claim can be stated as "clean in two of three tested models; model-conditional in Llama."
5. Promote regime interaction (J) to **three-model claim**: Llama η²=0.481, OLMo η²=0.358, Mistral η²=0.744 — all highly significant. Note Mistral's full positive sign profile (all 4 regimes high>low) contrasts with OLMo's partial profile.
6. Promote superlinearity absence (C2) to **three-model claim**: 0/3 support in Llama, 0/3 in OLMo, 1/3 in Mistral (pooled: not supported).
7. Update the Robustness and Limits table and any "pending" flags in text.

These updates require zero additional GPU time and substantially upgrade the paper's cross-model generalization claims.

---

## 0) Current claim state (for planning)

Main-text status is currently:

1. SI causal relevance: `supported` (two-model; can upgrade to three-model with ACTION-0).
2. Boundary non-triviality: `supported with caveat` (OLMo clean, Llama ambiguous; Mistral clean pending ACTION-0).
3. Distributed redundancy: `supported` (two-model; can upgrade to three-model with ACTION-0).
4. Conditional specialization semantics: `proxy-specific` (two-model; can upgrade to three-model with ACTION-0, though proxy-specific qualifier remains).
5. Three-model core replication: `complete` (all five subtests done; paper still says pending — fix in ACTION-0).
6. RoPE confound: `narrowed` (not eliminated).

This plan targets the gaps that still limit acceptance confidence.

---

## 1) Prioritized critical path

**Pre-experiment (zero GPU cost):** Complete ACTION-0 paper updates from existing Mistral R3 data.

If compute is limited, run in this order:

1. ~~`EXP-R3F`~~ **COMPLETE** — see ACTION-0.
2. `EXP-NEW-R12` Random-order ablation null control (directly defends core invariant).
3. `EXP-NEW-R15` High-SI head identity stability (defends construct validity of all claims).
4. `EXP-R2B` Power reinforcement of OLMo strict boundary non-triviality.
5. `EXP-NEW-R14` SI kernel permutation specificity (defends causal interpretation of T8).
6. `EXP-R5B` Regime-transfer replication with one-to-one proxy↔task mapping.
7. `EXP-R6` Split-sample SI head selection robustness (anti-circularity test).
8. `EXP-NEW-R13` Layer-stratified causal relevance (mechanistic depth).
9. `EXP-R7` SI-threshold sensitivity analysis.
10. `EXP-R9` Extended trivial-feature confound battery.
11. `EXP-R8` Additional non-RoPE/alt-PE anchors.
12. `EXP-R10` Minimal >8B transfer probe.
13. `EXP-R11` Long-context stress extension.

**Rationale for reordering:**
- NEW-R12 is highest priority because it directly protects the paper's single strongest cross-model invariant (6/6 threshold preference) from a ranking-artifact critique that an informed reviewer is likely to raise.
- NEW-R15 is low GPU cost and defends construct validity of all four results simultaneously.
- R2B is still required because the OLMo boundary non-triviality directional exception is an explicit caveat in the main text.
- NEW-R14 is the causal depth check for T8; moderate cost, high mechanistic value.
- R5B remains required to potentially lift the proxy-specific qualifier from regime semantics.

---

## 2) Experiments

## EXP-R3F: Third-model core replication — **COMPLETE** (see ACTION-0)

All five subtests finished as of 2026-04-17. Results are in
`results/reinforce_exp/exp_r3_core_replication/mistral-7b-v0.1/`.

| Subtest | Status | Result |
|---------|--------|--------|
| T8 kernel ablation | replicated | +3.43 nats, d=14.6 |
| 3P2-B strict boundary | replicated (clean) | non-trivial=true, artifact=false |
| 3P2-C.1 threshold | replicated | 6/6 threshold votes |
| 3P2-C.2 superlinearity | replicated | not supported (1/3 votes, pooled false) |
| 3P2-J regime interaction | replicated | F=1229.36, p=1.11×10⁻¹⁶, η²=0.744 |

**Remaining action:** Paper text updates only (ACTION-0). No further GPU runs required.

---

## EXP-R2B: OLMo strict boundary non-triviality power reinforcement (required)

### Why this is necessary

R2 left one unsupported directional cell: OLMo `B_boundary_nontriviality`.  
This is the most direct statistical caveat in the current claim matrix.

### Hypothesis

Under dependence-aware sequence-block inference and larger sample support, OLMo strict boundary non-triviality remains practically positive (artifact-negative and non-trivial effect retained).

### Implementation

1. Add:
   - `reinforce_exp/exp_r2b_olmo_boundary_power.py`.
2. Re-run strict 3P2-B OLMo path with:
   - `num_sequences >= 64` per seed,
   - `n_seeds >= 12`,
   - same strict confound controls as canonical run.
3. Evaluate across three domains (wiki/code/dialogue) to mirror R1 adjudication structure.
4. Emit:
   - `results/reinforce_exp/exp_r2b_olmo_boundary_power/olmo-2-7b/`,
   - `cluster_inference_summary.json`,
   - `domain_summary.json`,
   - `claim_impact.json`.

### Statistical plan

1. Primary test: sequence-cluster bootstrap CI for fake-minus-real-with-prefix.
2. Secondary: permutation test at sequence block level.
3. Report post-ablation effect size (`d`) and artifact flag jointly.
4. Multiplicity: Holm correction across domain-level checks.

### Acceptance criteria

1. Practical-positive criterion:
   - cluster CI excludes 0 in expected direction OR effect exceeds prereg practical threshold.
2. Artifact criterion:
   - no positive artifact flag under strict rule.
3. If failed, boundary claim for OLMo should be downgraded from “supported with caveat” to “descriptive/caveated”.

### Estimated cost

~6–10 GPU-hours.

---

## EXP-R5B: One-to-one regime-transfer replication (required)

### Why this is necessary

R5 interaction replicated but proxy→task sign transfer failed (Llama 1/4, OLMo 2/4).  
Current regimes are not one-to-one mapped to proxy regime definitions, limiting interpretability.

### Hypothesis

With construct-matched task regimes and adequate seed power, sign transfer improves; otherwise regime semantics should be permanently labeled proxy-specific.

### Implementation

1. Add:
   - `reinforce_exp/exp_r5b_regime_alignment.py`.
2. Build task-grounded regimes that directly mirror proxy definitions:
   - boundary-dense,
   - high-uncertainty,
   - long-span retrieval,
   - rare-token context.
3. Keep each regime matched on nuisance variables:
   - token length distribution,
   - lexical frequency bins,
   - answer-space size,
   - prompt format.
4. Use at least:
   - `n_seeds >= 12`,
   - `n_pairs >= 1200` per regime per model.
5. Emit:
   - `results/reinforce_exp/exp_r5b_regime_alignment/interaction_transfer_report.json`,
   - `regime_construct_audit.json`,
   - `proxy_vs_task_sign_matrix.json`.

### Statistical plan

1. Primary: SI-group × regime interaction significance (Holm-corrected).
2. Transfer criterion: per-model sign concordance `>= 3/4`.
3. Secondary robustness:
   - mixed-effects model with seed and template random intercepts.

### Acceptance criteria

1. Interaction must remain significant in both models.
2. Sign concordance:
   - `>= 3/4` per model => promote semantics from proxy-specific to task-grounded.
   - else keep proxy-specific permanently.

### Estimated cost

~8–14 GPU-hours.

---

## EXP-R6: Split-sample SI head selection robustness (anti-circularity) (high value)

### Why this is necessary

A likely reviewer criticism is head-selection circularity:
SI heads are selected using data that may overlap with evaluation contexts.

### Hypothesis

Core effects (T8, B, C1) persist when SI head ranking is computed on disjoint corpora/splits from those used for intervention evaluation.

### Implementation

1. Add:
   - `reinforce_exp/exp_r6_split_selection.py`.
2. Construct split protocol:
   - Split A for SI ranking,
   - Split B for intervention evaluation,
   - rotate A/B and average (cross-fit).
3. Run for Llama and OLMo:
   - T8,
   - strict B,
   - C1.
4. Emit:
   - `split_selection_robustness.json`,
   - cross-fit fold summaries.

### Statistical plan

1. Compare original vs split-selection effect sizes and directions.
2. Predefine tolerance bands (e.g., effect retains sign and at least 70% magnitude).

### Acceptance criteria

1. Direction retained for all three tests in both models.
2. No claim-level reversal under split-selection.

### Estimated cost

~6–10 GPU-hours.

---

## EXP-R7: SI threshold sensitivity analysis (high value)

### Why this is necessary

Current SI definition is top quartile by R².  
A reviewer can argue conclusions are threshold-dependent.

### Hypothesis

Core claim directions are stable across reasonable SI cutoffs.

### Implementation

1. Add:
   - `reinforce_exp/exp_r7_threshold_sensitivity.py`.
2. Evaluate SI cutoffs:
   - top 10%, 20%, 25%, 33%, 40%.
3. Recompute key claim outputs:
   - T8 effect,
   - strict B effect and artifact flag,
   - C1 threshold-vs-linear preference.
4. Emit:
   - `threshold_sensitivity_summary.json`,
   - `claim_status_by_cutoff.csv`.

### Statistical plan

1. Plot effect-size trajectories across cutoffs.
2. Use monotonic trend checks and sign-stability checks.

### Acceptance criteria

1. No sign flips on T8/C1 across cutoffs in either model.
2. If B artifact status flips by cutoff in Llama, retain conservative ambiguity language.

### Estimated cost

~4–8 GPU-hours.

---

## EXP-R9: Extended trivial-feature confound battery (high value)

### Why this is necessary

Current strict control emphasizes space-prefix confounds.  
Reviewers may ask whether subtler lexical/tokenizer artifacts remain.

### Hypothesis

For OLMo, boundary effect remains non-trivial under a broader confound battery; for Llama, confound dependence remains stronger and model-conditional.

### Implementation

1. Add:
   - `reinforce_exp/exp_r9_confound_battery.py`.
2. Feature families:
   - space-prefix,
   - capitalization marker,
   - punctuation adjacency,
   - token-length bucket,
   - continuation-marker surrogates.
3. For each feature family:
   - train feature classifier on embeddings,
   - ablate top-k predictive dims,
   - rerun boundary and SI-linked disruption metrics.
4. Emit:
   - `tokenizer_audit_report_v2.json`,
   - per-feature dependence indices.

### Statistical plan

1. Holm correction within feature-family tests.
2. Report practical deltas, not just significance.
3. Include per-domain robustness slice (wiki/code/dialogue).

### Acceptance criteria

1. OLMo: no single feature family yields practical collapse + artifact-positive pattern.
2. Llama: if one family dominates dependence, report mechanistic entanglement explicitly (not as failure).

### Estimated cost

~8–12 GPU-hours.

---

## EXP-R8: Non-RoPE/alt-PE extension beyond GPT-2 family (moderate value)

### Why this is necessary

R4 used two GPT-2 anchors from one family.  
A reviewer may still call this family-specific rather than PE-scheme-general.

### Hypothesis

At least one additional non-RoPE/alternative-PE family shows SI structure + strict non-trivial boundary signal.

### Implementation

1. Add:
   - `reinforce_exp/exp_r8_alt_pe_extension.py`.
2. Candidate models (choose based on availability/memory):
   - one ALiBi model,
   - one learned-absolute model outside GPT-2 family.
3. Run reduced battery:
   - SI profiling,
   - strict B,
   - lightweight T8 sanity.
4. Emit:
   - `alt_pe_comparison_summary.json`.

### Statistical plan

1. Same strict criteria as R4 for comparability.
2. Report family-level qualitative replication matrix.

### Acceptance criteria

1. At least one additional family positive on SI + strict non-trivial boundary.
2. If not, narrow claim from “general across PE schemes” to “supported in RoPE + GPT-2 absolute PE”.

### Estimated cost

~6–10 GPU-hours.

---

## EXP-R10: Minimal >8B transfer probe (optional but high reviewer value)

### Why this is necessary

Current core claims are at 7–8B scale for primary models.
Scale-generalization concern (>8B) remains explicit.

### Hypothesis

Reduced core battery on one >8B model reproduces at least SI causal relevance + E2 threshold signature.

### Implementation

1. Add:
   - `reinforce_exp/exp_r10_scale_probe.py`.
2. Run reduced battery on one >8B model:
   - T8,
   - strict B,
   - C1.
3. Keep dataset size moderate to control cost.
4. Emit:
   - `scale_probe_report.json`.

### Statistical plan

1. Same directionality criteria as primary models.
2. No over-interpretation of single-model scale probe.

### Acceptance criteria

1. If positive: add “external scale probe is consistent” statement.
2. If mixed/negative: keep >8B as explicit limitation.

### Estimated cost

~10–20 GPU-hours (model-dependent).

---

## EXP-R11: Long-context stress extension (optional)

### Why this is necessary

Current task-grounded long-span settings are relatively short (`64/128` bins in R5).  
A positional-mechanism paper can be challenged on long-context relevance.

### Hypothesis

Core SI effects remain directionally consistent at larger spans (256/512/1024 where feasible).

### Implementation

1. Add:
   - `reinforce_exp/exp_r11_long_context_stress.py`.
2. Extend long-span task bins:
   - 64, 128, 256, 512 (1024 if memory permits).
3. Evaluate interaction and failure mode transitions.
4. Emit:
   - `long_context_stress_summary.json`.

### Statistical plan

1. Mixed-effects analysis with span bin as ordered factor.
2. Check monotonic degradation and SI-group differential.

### Acceptance criteria

1. No claim-level reversals at larger spans.
2. If reversals occur, document explicit context-length boundary for claim scope.

### Estimated cost

~8–16 GPU-hours.

---

---

## EXP-NEW-R12: Ablation Ordering Null Control — Ranked vs. Random (required)

### Why this is necessary

The paper's single strongest cross-model invariant is the unanimous threshold-model
preference in cumulative ablation (6/6 votes in each of three models).
However, all current ablation orderings are non-random: heads are removed in
high-to-low or low-to-high R² order.

A well-informed reviewer can raise the following critique:
*When units are ranked by a property (R²) and removed in that order,
the removal sequence is internally correlated — adjacent steps remove similar
units, and marginal damage accumulates non-linearly by construction.*
This would produce apparent threshold-style degradation even in a system with
no genuine collective redundancy, because the most-similar heads are removed
as a block before dissimilar heads are introduced.

A random-order ablation is the definitive control.
If threshold preference persists across random orderings, the capacity structure
is genuinely threshold-shaped regardless of removal order, confirming true redundancy.
If only ranked ablation shows threshold preference, the result is a ranking artifact,
which would require substantially weakening the redundancy claim.

### Hypothesis

The threshold-piecewise model is preferred over linear degradation under
multiple random head-ablation orderings in both Llama-3.1-8B and OLMo-2-7B,
with vote fraction ≥ 4/6 across random orderings.

### Implementation

1. Add:
   - `reinforce_exp/exp_new_r12_ordering_control.py`
2. Generate 10 independent random head orderings per model (seeded for reproducibility).
3. For each random ordering, run the full cumulative ablation evaluation
   at the same fractions used in the canonical run
   (0, 1, 2, 5, 10, 15, 20, 25, 50%).
4. For each ordering × task × ablation fraction, fit threshold-piecewise vs. linear
   using the same BIC criterion as the canonical run.
5. Emit per-ordering vote counts and an aggregate summary.
6. Required outputs in `results/reinforce_exp/exp_new_r12_ordering_control/{model}/`:
   - `ordering_vote_summary.json` (per ordering: threshold votes, linear votes),
   - `aggregate_robustness.json` (fraction of orderings where threshold wins majority),
   - `canonical_vs_random_comparison.json`.

### Statistical plan

1. Primary: fraction of random orderings where threshold wins majority of
   task × ablation-criterion comparisons (preregistered threshold: ≥ 7/10 orderings).
2. Secondary: compare mean BIC delta (threshold − linear) across random orderings
   to the canonical ranked ordering. Report whether ranked ordering is an outlier.
3. Paired test: for each random ordering, test whether threshold model BIC
   is lower than linear BIC across tasks. Report sign-consistency.

### Acceptance criteria

1. **Strong replication:** ≥ 8/10 random orderings show threshold majority
   → retain "distributed redundancy is the strongest cross-model invariant" language unchanged.
2. **Partial replication:** 5–7/10 orderings → add ordering-robustness caveat;
   strengthen discussion of redundancy as approximate rather than strict.
3. **Failure:** ≤ 4/10 orderings → downgrade redundancy claim to
   "observed under ranked ablation; not confirmed under random ordering;
   ordering-dependent capacity structure."

### Estimated cost

~6–10 GPU-hours per model (10 random orderings × evaluation cost ≈ canonical run × 5).

---

## EXP-NEW-R13: Layer-Stratified Causal Relevance (high mechanistic value)

### Why this is necessary

The direct kernel ablation result (T8) establishes that removing SI kernels from
all high-SI heads simultaneously increases loss.
However, this is a whole-model, all-layer effect.

Two plausible mechanistic accounts make different layer-specificity predictions:

1. *Early-layer account:* SI infrastructure is concentrated in early layers,
   where position is established relative to nearby tokens and boundary structure
   is computed. Ablating early layers should produce the dominant loss effect.
2. *Late-layer account:* SI structure in later layers implements higher-level
   positional context integration. Late-layer ablation dominates.
3. *Distributed account:* SI contribution is spread across layers; no stratum
   dominates and partial ablation at each stratum produces proportional degradation.

Layer stratification distinguishes these accounts and provides mechanistic depth
that is currently absent from the paper. It also addresses a likely reviewer question:
"Is SI causal relevance a property of the positional encoding computation
(early layers) or of higher-order positional integration (late layers)?"

### Hypothesis

Based on RoPE's geometric design, SI causal relevance is concentrated in early-to-middle
layers (bottom third of the network), consistent with the view that positional
structure is established early and refined over depth.

### Implementation

1. Add:
   - `reinforce_exp/exp_new_r13_layer_stratified_ablation.py`
2. Divide layers into three equal strata:
   - Early: layers 0–10 (Llama-3.1-8B has 32 layers, so 0–10/11–21/22–31).
   - Middle: layers 11–21.
   - Late: layers 22–31.
   - Apply same strata proportionally for OLMo-2-7B.
3. For each stratum, ablate ONLY the high-SI heads in that stratum
   (leaving high-SI heads in other strata intact).
4. Measure mean loss increase vs. no-ablation baseline (same evaluation as T8).
5. Also run full-network ablation (replicating T8) for comparison.
6. Emit results in `results/reinforce_exp/exp_new_r13_layer_stratified/{model}/`:
   - `stratum_ablation_summary.json` (mean loss delta, d, CI per stratum),
   - `stratum_vs_full_decomposition.json` (sum of stratum effects vs. joint effect,
     to test additivity),
   - `layer_effect_profile.json` (per-layer ablation for finer resolution).

### Statistical plan

1. Primary: test whether early-stratum ablation loss delta is larger than
   middle- and late-stratum deltas (Holm-corrected pairwise comparisons).
2. Secondary: compare sum of stratum-specific loss deltas to joint T8 effect
   to assess interaction/redundancy across layers.
3. Effect sizes (Cohen's d) per stratum reported as primary evidence;
   p-values as diagnostics.

### Acceptance criteria

1. Whichever stratum dominates, report as a mechanistic finding, not a
   hypothesis confirmation — the account is exploratory regardless of outcome.
2. If strata are additive (sum ≈ joint), report as a clean decomposition.
   If subadditive, report cross-layer redundancy as an additional finding.
3. The result strengthens the paper regardless of direction by adding
   mechanistic resolution to an otherwise aggregate finding.

### Estimated cost

~8–14 GPU-hours per model (4 ablation conditions × evaluation cost).

---

## EXP-NEW-R14: SI Kernel Permutation Specificity Control (required for causal claim)

### Why this is necessary

The T8 result shows that removing the estimated shift-invariant kernel g_h(Δ)
from high-SI heads increases language-model loss.
The interpretation is that *shift-invariant positional structure* is causally relevant.

However, a critical reviewer can raise:
*The loss increase might reflect a generic disruption to attention magnitude
rather than anything specific about shift-invariance.
Subtracting g_h(Δ) reduces the mean value of attention logits across positions,
which would increase loss for reasons unrelated to positional structure.*

A permutation control isolates whether the CONTENT of g_h (its offset structure,
which encodes shift-invariant positional preferences) is what matters,
vs. merely the magnitude of the removed component.

**Design:** Compute g_h(Δ) for each high-SI head. Construct a permuted control
g_π(Δ) by randomly permuting the mapping Δ → g_h(Δ) (so the set of values is
identical but the offset-to-value assignment is scrambled).
Subtract g_π instead of g from logits. A permuted kernel has exactly the same
total magnitude as the original but is no longer shift-invariant in the
structured sense — it applies arbitrary magnitude distortions to each offset.

If the loss increase under permuted subtraction equals that under true subtraction,
the effect is attributable to overall magnitude reduction, not SI content.
If true subtraction produces a *larger* loss increase than permuted subtraction,
the structured offset map of g_h carries information that the permuted version
destroys — confirming that shift-invariant structure specifically is what matters.

### Hypothesis

True-kernel subtraction produces a larger loss increase than offset-permuted
subtraction (paired by head and sequence), in both primary models.

### Implementation

1. Add:
   - `reinforce_exp/exp_new_r14_kernel_permutation.py`
2. Load estimated kernels from the T8 experiment outputs:
   - `results/reinforce_exp/exp_r3_core_replication/{model}/theory8_position_ablation/{model}/estimated_kernels.json`
   - For Llama and OLMo, load from the original T8 experiment outputs in `results/experiment3/`.
3. For each high-SI head, generate 5 independent random permutations of Δ → g_h(Δ).
4. Run forward passes with:
   - true-kernel subtraction (replicates T8),
   - each permuted-kernel subtraction,
   - no subtraction (baseline).
5. Compute mean loss delta per condition.
6. Emit in `results/reinforce_exp/exp_new_r14_kernel_permutation/{model}/`:
   - `permutation_vs_true_comparison.json` (mean loss delta: true vs. permuted, 5 seeds),
   - `specificity_test.json` (paired bootstrap test: true_delta > permuted_delta),
   - `per_head_specificity.json` (head-level breakdown).

### Statistical plan

1. Primary: paired bootstrap test of (true subtraction loss delta) minus
   (mean permuted subtraction loss delta) > 0.
2. Report specificity ratio: true_delta / mean_permuted_delta.
   Ratio > 1.0 supports SI content specificity; ratio ≈ 1.0 is a null result.
3. Secondary: check whether high-R² heads show higher specificity ratios
   than heads at the R² quartile boundary (within the high-SI group).

### Acceptance criteria

1. **SI content confirmed:** Specificity ratio > 1.5 in both models
   and paired bootstrap CI excludes 0
   → add one sentence to T8 interpretation: "the loss increase is substantially
   larger under true-kernel subtraction than under magnitude-matched random
   permuted subtraction, confirming SI offset structure rather than overall
   attention magnitude is what is causally load-bearing."
2. **Weak specificity:** Ratio 1.1–1.5 → add a cautionary note that the effect
   is partly but not exclusively explained by magnitude.
3. **Null specificity:** Ratio ≈ 1.0 → revise the T8 causal interpretation
   from "SI structure is causally relevant" to "high-SI head attention components
   are causally relevant; the role of shift-invariance per se requires further
   isolation."

### Estimated cost

~4–8 GPU-hours per model (re-uses estimated kernels, only new forward passes needed).

---

## EXP-NEW-R15: High-SI Head Identity Stability Across Corpora (required for construct validity)

### Why this is necessary

All four main-text results — boundary detection, cumulative ablation, kernel ablation,
and regime interaction — rest on a fixed set of "high-SI heads" defined as the
top quartile of per-head R² scores.
A reviewer can ask: *Is the set of high-SI heads a stable property of the model,
or is it corpus-dependent? If the same heads are not consistently classified as
high-SI across different text domains, the downstream analyses may be measuring
domain-specific rather than model-intrinsic properties.*

This test is a prerequisite for any claim of the form "high-SI heads implement X."
If head identity is unstable (Jaccard overlap near chance), the classification
is measuring a noisy or context-dependent quantity, and the entire mechanistic
account requires reframing.

This is also unusually cheap relative to its impact: it requires measurement
runs only (no intervention), reuses existing infrastructure, and can be
completed without new model loading if R² values per corpus are logged.

### Hypothesis

Jaccard overlap of the top-quartile head set across disjoint corpora
(wiki, code, dialogue) is substantially above chance (hypergeometric baseline)
in both primary models, indicating that high-SI head identity is a
model-intrinsic property rather than a corpus artifact.

Preregistered minimum: Jaccard ≥ 0.60 for all pairwise corpus comparisons,
well above the hypergeometric null (~0.25 for top-quartile sets of size 256
out of 1024 total).

### Implementation

1. Add:
   - `reinforce_exp/exp_new_r15_head_stability.py`
2. Run the SI measurement (R² computation) independently on three disjoint corpora:
   - Domain A: Wikipedia (500 sequences × 512 tokens).
   - Domain B: Code (500 sequences × 512 tokens, from The Stack or similar).
   - Domain C: Dialogue (500 sequences × 512 tokens).
3. For each model, compute per-head R² independently for each corpus.
4. Define top-quartile head set for each corpus (256 heads of 1024).
5. Compute:
   - Pairwise Jaccard overlap (wiki∩code/wiki∪code, wiki∩dialogue, code∩dialogue).
   - Three-way intersection Jaccard (all three corpora).
   - Rank-correlation of per-head R² values across corpus pairs (Spearman ρ).
6. Compute hypergeometric baseline:
   expected Jaccard for random same-size sets drawn from 1024 heads.
7. Emit in `results/reinforce_exp/exp_new_r15_head_stability/{model}/`:
   - `stability_summary.json` (Jaccard matrix, Spearman correlations, hypergeometric null),
   - `per_head_consistency_scores.json` (how many corpora each head is in top-quartile),
   - `stable_core_set.json` (heads in top-quartile across all three corpora).

### Statistical plan

1. Primary: bootstrap CI for each pairwise Jaccard. Report whether CI excludes
   the hypergeometric null.
2. Secondary: Spearman ρ of per-head R² rankings across corpus pairs.
   ρ > 0.7 indicates stable relative ranking even if absolute values shift.
3. Tertiary: identify the "stable core" — heads that appear in top-quartile
   across all three corpora. Report this as the most conservative high-SI set
   and note whether main-text findings replicate when restricted to this set.

### Acceptance criteria

1. **Stable (high confidence):** All pairwise Jaccard ≥ 0.65 and Spearman ρ ≥ 0.70
   → add one sentence to Setup: "High-SI head identity is stable across
   text domains (pairwise Jaccard = X±Y across wiki/code/dialogue)."
2. **Partially stable:** Jaccard 0.45–0.65 → note domain heterogeneity as a scope
   condition; report findings for the stable core set as a robustness check.
3. **Unstable:** Jaccard ≤ 0.45 → reframe all four results as
   "corpus-conditional SI head profiles" rather than model-intrinsic properties;
   add substantial caveat to Setup and all Result sections.

### Estimated cost

~2–4 GPU-hours per model (measurement only, no interventions).
Likely reusable from existing R² computation code in `experiment3/theory1_si_circuits.py`.

---

## 3) Reporting requirements for every new run

For each experiment above, require:

1. `manifest.json` with:
   - command,
   - model,
   - seed set,
   - commit hash,
   - timestamp,
   - dataset version hash.
2. `claim_impact` enum:
   - `strengthens_main_claim`,
   - `requires_claim_downgrade`,
   - `no_change`.
3. Corrected p-values and family definitions.
4. One-page markdown summary in:
   - `results/reinforce_exp/reports/<exp_id>.md`.
5. Paper patch checklist:
   - exact sections/tables requiring updates if acceptance criteria trigger claim-scope changes.

---

## 4) Decision policy after TODOv2

### R3F (complete — ACTION-0 required)
Update paper to three-model wording for T8, C1, C2, B (Mistral+OLMo clean), J.
No decision needed; update is unambiguously warranted.

### After new experiments:

1. **If `NEW-R12` (random ordering) passes** (≥ 8/10 orderings threshold):
   - Retain "distributed redundancy is the strongest cross-model invariant"
     with added sentence about robustness to random ablation ordering.

2. **If `NEW-R12` fails** (≤ 4/10):
   - Downgrade to "observed under ranked ablation; ordering-dependent capacity
     structure cannot be ruled out." Distributed redundancy claim is substantially
     weakened; paper's main claim shifts to T8 causal relevance + boundary claim.

3. **If `NEW-R15` (head stability) passes** (Jaccard ≥ 0.65):
   - Add stability sentence to Setup; all four results gain construct-validity support.
   - Identify stable-core set for Appendix.

4. **If `NEW-R15` fails** (Jaccard ≤ 0.45):
   - Reframe all results as corpus-conditional; substantial rewrite required.
   - This is a worst-case scenario; run early to avoid late-stage surprises.

5. **If `R2B` passes**:
   - Upgrade OLMo boundary claim from "supported with caveat" to "supported."

6. **If `R2B` fails**:
   - Downgrade OLMo boundary strength claim; focus paper on T8 + distributed
     redundancy + regime interaction.

7. **If `NEW-R14` (permutation specificity) passes** (ratio > 1.5):
   - Strengthen T8 causal interpretation with one sentence on offset-structure specificity.

8. **If `NEW-R14` fails** (ratio ≈ 1.0):
   - Add nuance: high-SI head attention magnitude is causally relevant; SI structure
     per se requires further isolation.

9. **If `R5B` passes** (concordance ≥ 3/4 per model):
   - Promote regime semantics from proxy-specific to task-grounded.

10. **If `R5B` fails**:
    - Keep proxy-specific qualifier; add note on power and construct-match quality.

11. **If optional experiments (`R8/R10/R11/NEW-R13`) fail**:
    - Treat as scope boundaries, not paper blockers.

