# Experiment 3 Phase 2: Testing Competing Mechanistic Explanations

## Document Status

Evidence snapshot: **2026-04-04** (post-refresh + Stage 2 updates).
Phase 2 is in progress. Stage 1 is completed; Stage 2a-2c execution is completed with 3P2-D.1 now complete. Stage 2d (3P2-C.1) is running with a rerun after bug fixes.
Refresh reruns are complete for T1/T3/T5/T5b/T7b/T10, and all metrics below are from post-refresh artifacts.
Stage 1 completion metadata:
- `llama-3.1-8b_20260401_144845`
- `olmo-2-7b_20260401_144845`
- Gate G1 emitted at `2026-04-01 14:49:58` with `continue_to_stage2=true`.
Stage 2 run metadata:
- `llama-3.1-8b_20260401_153909` (`3P2-G/H` + `3P2-D.2` complete)
- `olmo-2-7b_20260402_135540` (`3P2-G/H` complete; rerun replaced failed `olmo-2-7b_20260401_153909`)
- `mistral-7b-v0.1_20260404_175041` (`3P2-D.1` complete; hypothesis_supported=false, spearman_rho=0.0962, p_one=0.0624)
- `llama-3.1-8b_c1_20260404_200917` (`3P2-C.1` rerun in progress)
- `olmo-2-7b_c1_20260404_200917` (`3P2-C.1` rerun in progress)

This document specifies follow-up experiments to adjudicate between competing explanations of the Experiment 3 results. All Phase 2 outputs are additive and non-retroactive to the Experiment 3 refresh conclusions.

Evidence confidence note:
- T7 is retained as legacy-supportive evidence, but it is lower confidence for causal claims because its knockout-style intervention has a known residual-stream confound.
- For causal adjudication in Phase 2, prioritize T7b and T10 over T7 when conclusions diverge.

Stage 1 status snapshot:

| Experiment | Llama-3.1-8B | OLMo-2-7B | Gate-impact |
|---|---|---|---|
| 3P2-E | complete (`consistent_with_boundary_primary`) | complete (`consistent_with_boundary_primary`) | none |
| 3P2-F | complete (`median delta_r2=0.00334`, retained by current rule) | complete (`median delta_r2=0.00318`, retained by current rule) | proxy-collapse not triggered |
| 3P2-B | complete (`d=1.085`, no prefix-following artifact) | complete (`d=0.543`, no prefix-following artifact) | boundary-artifact not triggered |

---

## Background: What Experiment 3 Established

### Supported (both models)

| Theory | Finding | Key metric |
|---|---|---|
| T5b | High-SI heads detect word boundaries | Llama: A=Y, B=Y, C=Y (d=1.085). OLMo: A=Y, B=N, C=Y |
| T7 | High-SI heads feed induction circuits | Pearson r=0.441, Spearman rho=0.578 (Llama global) |
| T8 | Positional kernel g(Delta) is functionally important | Kernel removal significantly increases loss |

### Not Supported (both models)

| Theory | Finding | Key metric |
|---|---|---|
| T1 | High-SI ablation does not selectively impair positional tasks | Llama: Delta_drop=-0.0149, p=0.999993 |
| T5 | High-SI ablation does not selectively hurt continuation tokens | Llama interaction=-1.96, OLMo=-1.44 (wrong sign in aggregate) |
| T10 | High-SI feeder effect is not induction-specific | Llama: rho_ind=+0.064 ~ rho_rand=-0.012 |

### Mixed / Descriptive

| Theory | Finding | Key metric |
|---|---|---|
| T7b | Activation patching supports feeder role in OLMo only | OLMo: rho=+0.207, p=0.0009. Llama: rho=+0.064, p=0.305 |
| T3 | Cross-term correlation is descriptive | Llama Spearman=-0.110, OLMo=+0.050 |
| T9 | High-freq RoPE component correlates with feeder effect | rho_hf=+0.209 to +0.438, not causal |

---

## Competing Explanations

Phase 2 tests ten competing (partially overlapping) explanations of this result pattern. They are ordered for continuity and practical test design rather than as mutually exclusive alternatives.
These explanations are not mutually exclusive; Phase 2 adjudicates relative explanatory power and interaction effects rather than forcing a single-winner narrative.

### E1: General-Purpose Positional Infrastructure (MOST LIKELY)

**Claim:** High-SI heads are a shared positional service layer consumed broadly by many downstream circuits. They are not wired into any single circuit specifically.

**What it explains:**
- T5b supported: boundary detection is a universally useful positional primitive
- T7 supported + T10 not supported: they feed induction circuits as one of many consumers; rho_ind ~ rho_rand because all downstream circuits benefit equally
- T1 not supported: ablation damage is diffuse across all tasks, not selective for positional ones
- T8 supported: the kernel carries the positional signal that the broadcast depends on

**What it struggles with:**
- Unfalsifiability risk: "does everything a little" is hard to distinguish from "does nothing important"
- T5b boundary result (d=1.085) suggests some specialization, not pure generality
- T5 raw numbers show differential damage by position type (word-initial loss increase 4.106 > continuation 2.799), inconsistent with purely uniform broadcast

**Distinguishing prediction:** Multi-task patching causes *uniform* degradation across task types proportional to task difficulty, not selective degradation on positional tasks.

### E2: Distributed Redundancy

**Claim:** High-SI heads do carry specialized positional signal, but redundant encoding across many heads masks the ablation effect. Low-SI heads carry enough backup positional information to compensate when high-SI heads are removed.

**What it explains:**
- T1 not supported: compensation by backup heads masks the true causal role
- T8 supported: removing the kernel from *all* heads simultaneously overwhelms backup pathways
- Consistent with Michel et al. (2019) and Lieberum et al. (2023) on head redundancy at scale

**What it struggles with:**
- Does not explain T10's equal rho values: under pure redundancy, the *correlation* between source R-squared and target disruption should still be stronger for induction targets than random targets. Instead rho_ind ~ rho_rand.
- OLMo mean R-squared=0.058 (much lower than Llama's 0.380). With less SI signal to be redundant, OLMo should show *more* selective deficits. Instead both models show the same null pattern.
- Cumulative ablation (Experiment 3P2-C) would reveal a threshold collapse if redundancy is real.

**Distinguishing prediction:** Ablating progressively more high-SI heads produces a *threshold* collapse in positional task performance (not linear degradation).

### E3: Non-Specific Feeder Broadcasting

**Claim:** High-SI heads broadcast positional information into the residual stream. Any downstream head that needs positional context reads from this broadcast. The feeding is a consequence of residual stream architecture, not circuit-specific wiring.

**Relationship to E1:** E3 is a specific mechanistic instantiation of E1 focused on the information flow pathway. E1 is the functional characterization; E3 is the circuit-level mechanism.

**What it explains:**
- T7 supported: induction circuits are one consumer of the broadcast
- T10 not supported: the broadcast is read by induction, random mid-layer, and low-SI late targets equally
- Consistent with Wang et al. (2023, IOI) finding that heads participate in multiple circuits

**Distinguishing prediction:** Residual stream probes for relative position, word boundary, and syntactic role all degrade equally when high-SI head outputs are zeroed (the positional subspace in the residual stream is disrupted broadly).

### E4: Architecture-Dependent Mechanism (for T7b divergence)

**Claim:** The Llama/OLMo T7b divergence reflects architectural differences (RMSNorm vs LayerNorm, or GQA head-sharing patterns) rather than a fundamental mechanistic difference.

**What it explains:**
- T7b supported in OLMo (standard MHA, LayerNorm) but not Llama (GQA, RMSNorm)
- GQA shares KV heads across query heads, potentially diluting per-head patching signal

**What it struggles with:**
- n=2 models: cannot distinguish architecture from training data / hyperparameter / random seed effects
- GQA dilution is speculative; no published evidence that GQA specifically degrades activation patching sensitivity
- T7 (the non-patching version) is supported in *both* models, so the divergence may be methodological noise

**Distinguishing prediction:** A third GQA model (Mistral-7B) shows Llama-like null T7b results; a third standard-MHA model shows OLMo-like positive results.

### E5: Boundary Detection as Primary Function

**Claim:** The main functional contribution of high-SI heads is marking subword/word boundaries. This is broadly useful across tasks without being task-specific, which explains T1's null result.

**What it explains:**
- T5b supported with large effect sizes
- T1 not supported: boundary information helps all tasks, so no selective positional deficit
- T7 supported: boundary-marked residual stream positions are especially useful for induction (which needs to identify where repeated patterns start)

**What it struggles with:**
- Trivial feature confound: BPE tokenizers prefix word-initial tokens with Gbar/underscore. Any head attending to t-1 will see boundary features for free. The "detection" may be incidental, not learned.
- T5b Approach B (4-way position taxonomy) failed for OLMo (B=N), weakening cross-model robustness.
- T8 shows the kernel carries information beyond boundaries (distance-dependent profiles matter for loss).
- Circular reasoning risk: "heads that attend at fixed offsets see features of adjacent tokens" is trivially true.

**Distinguishing prediction:** After controlling for trivial BPE features, high-SI heads still show elevated boundary attention compared to other fixed-offset heads.

### E6: Metric/Proxy Confound (R-squared is not the causal variable)

**Claim:** Mean head R-squared is an imperfect proxy that tracks attention geometry and head-level nuisance factors (entropy, dominant offset, norm, depth) rather than the causal positional computation itself.

**What it explains:**
- T1/T10 nulls despite strong T5b/T8 signals: interventions keyed to R-squared can miss causally important heads if R-squared is only partially aligned with mechanism.
- Llama/OLMo differences: model-specific mapping between R-squared and true mechanism can weaken cross-model consistency.
- Mixed T7b outcomes: correlation magnitudes can shift with proxy quality rather than mechanistic absence.

**What it struggles with:**
- If R-squared is mostly proxy noise, repeated positive associations with boundary metrics still require an explanation.
- Does not by itself explain why T5b remains strongly positive in both models.
- Needs explicit multivariate controls to avoid becoming a post-hoc catch-all.

**Distinguishing prediction:** Once controlling for entropy, offset preference, head norm, and layer, the unique contribution of R-squared to disruption/boundary effects collapses or becomes negligible.

### E7: Intervention Saturation / Floor Effects

**Claim:** Zero-ablation is too strong and drives outputs toward behavioral floors, masking selective deficits that would be visible under graded attenuation.

**What it explains:**
- T1 null selectivity: both high-SI and low-SI interventions can appear similarly bad under saturated regimes.
- T5 wrong-sign aggregates: floor compression can distort relative damage signs across token types.
- Strong T8 effect: global kernel removal is expected to overwhelm compensation pathways.

**What it struggles with:**
- Requires clear nonlinearity in dose-response; if curves are linear, this explanation weakens.
- Must explain why some contrasts (e.g., T5b A/C) remain detectable despite potential saturation.
- Saturation alone does not explain model divergence in T7b without additional assumptions.

**Distinguishing prediction:** Under attenuation/dose-response interventions, selective high-SI effects emerge at moderate strengths before floor collapse at extreme strengths.

### E8: Source/Target Set Misspecification in T7b/T10

**Claim:** Current source-layer windows and target group definitions under-sample the true feeder pathways, causing attenuated or misleading specificity signals.

**What it explains:**
- T10 nulls: random-mid targets may contain substantial positional consumers, flattening induction-vs-random contrasts.
- Llama T7b weakness: relevant feeder sources may exist outside currently emphasized source windows or grouping choices.
- Apparent inconsistencies between T7, T7b, and T10.

**What it struggles with:**
- Becomes less plausible if expanded source/target sweeps preserve existing null patterns.
- Needs pre-registered grouping rules to avoid researcher degrees of freedom.
- Does not directly explain T5b/T8 outcomes.

**Distinguishing prediction:** Correlation/specificity patterns change materially when source layers and target definitions are expanded, stratified, and pre-registered.

### E9: Tokenizer/Corpus-Specific Boundary Artifact

**Claim:** Part of the boundary signal is specific to tokenizer markers and corpus composition, so observed boundary effects may not generalize as a mechanism-level property.

**What it explains:**
- Strong boundary effects under the current tokenizer/corpus setup.
- OLMo Approach B instability (B=N) as a robustness warning.
- Potential mismatch between boundary-heavy tasks and broader positional conclusions.

**What it struggles with:**
- Must reconcile cross-model T5b support under different training pipelines.
- Cannot explain T8 without additional mechanism.
- If boundary effects persist under marker controls and corpus shifts, this explanation weakens sharply.

**Distinguishing prediction:** Boundary effect size shifts substantially across tokenizer perturbations and corpus slices, with partial or full attenuation under prefix-feature controls.

### E10: Context-Conditional Specialization

**Claim:** High-SI heads are selectively important only in specific regimes (long spans, high uncertainty, boundary-dense or noisy contexts), so aggregate averages obscure true specialization.

**What it explains:**
- Global nulls in T1/T5/T10 despite meaningful effects in some sub-analyses.
- Mixed T7b outcomes if each model sees different regime prevalence.
- Coexistence of strong T5b and weak task-aggregate selectivity.

**What it struggles with:**
- Requires preregistered regime definitions to avoid cherry-picking.
- If conditional splits fail to produce stronger effects, explanation collapses.
- Needs enough sample size per regime for stable inference.

**Distinguishing prediction:** Pre-registered hard-regime subsets show large selective high-SI effects even when full-sample aggregates remain weak.

---

## Phase 2 Governance (Stop/Go, Power, Multiplicity)

### Stage-Gating (required)

Phase 2 is executed in stages, not as a single full sweep.

Stage 1 (mandatory): 3P2-E -> 3P2-F -> 3P2-B.

Gate G1 is evaluated immediately after Stage 1:
Gate G1 decision artifact is emitted immediately after 3P2-B completes, before any Stage 2 experiment begins.
1. **G1-Proxy collapse check (from 3P2-F):**
   - Collapse condition: median `delta_r2 < 0.02` and 95% CI includes 0 for >= 2 of 3 outcomes in both models.
   - If true: pause 3P2-G, 3P2-H, and 3P2-J; re-scope intervention-driven claims to proxy-independent definitions before additional GPU runs.
2. **G1-Boundary validity check (from 3P2-B):**
   - Artifact condition: post-control boundary effect `d < 0.20` or synthetic controls show prefix-following (fake-boundary effect >= real-boundary effect).
   - If true: mark E5 low-confidence and skip 3P2-I (external-validity robustness is moot when the base mechanism fails controls).
3. **G1-Continue condition:**
   - Continue to Stage 2 only if 3P2-F does not collapse by the rule above and 3P2-B does not trigger artifact condition.

Stage 2 (conditional mechanistic tests): 3P2-G, 3P2-H, 3P2-D, 3P2-C, 3P2-A, 3P2-J (subject to each experiment's own gate).

Stage 3 (publication-readiness robustness): 3P2-I, only if 3P2-B passes and core mechanism claims remain active after Stage 2.

### Power and MDE Requirements (required)

Null outcomes are never interpreted as support unless confidence intervals exclude practically meaningful effects.

Global defaults:
1. Power target: >= 0.80 for primary confirmatory test(s) per experiment.
2. Each experiment must pre-register MDE(s) before execution and report achieved power.
3. If achieved power is below target, result is labeled `inconclusive` regardless of p-value direction.

Default MDE anchors (unless a stricter experiment-specific MDE is justified):
1. Correlations (3P2-D/3P2-H): `|rho| >= 0.15`.
2. Mean contrasts (3P2-B/3P2-E/3P2-G/3P2-J): `|d| >= 0.35`.
3. Interaction tests (3P2-A/3P2-J): partial `eta^2 >= 0.02`.

### Unified Multiplicity Policy (required)

Phase 2 claims are partitioned into tiers with explicit correction:
1. **Tier 1 (confirmatory core):** 3P2-E, 3P2-F, 3P2-B.
   - Family-wise correction: Holm at alpha=0.05 across Tier 1 primary tests.
2. **Tier 2 (conditional mechanistic):** 3P2-G, 3P2-H, 3P2-D, 3P2-C, 3P2-A, 3P2-J.
   - Run only after Gate G1 continuation.
   - Family-wise correction: Holm at alpha=0.05 across executed Tier 2 primary tests.
3. **Tier 3 (external validity / publication readiness):** 3P2-I.
   - Report with FDR q-values; exploratory by default unless explicitly promoted in a later preregistered phase.

Every artifact carrying inferential claims must include: `primary_test_id`, `tier`, `mde_target`, `achieved_power`, `multiplicity_family`, and corrected significance values.

---

## Planned Experiments

### Experiment 3P2-A: Positional Broadcast Test

**Tests:** E1 (general infrastructure) vs. "specialized positional operator" alternative.

**Models:** Llama-3.1-8B, OLMo-2-7B.
**Data:** Wikipedia (wiki40b_en_pre2019), 100 sequences, len=1024.

**Sub-experiment 3P2-A.1: Multi-task simultaneous patching**

Protocol:
1. Define a task battery spanning positional and non-positional demands:
   - Positional: retrieval (spans 32, 48, 64), local_key_match, copy_offset
   - Boundary-sensitive: next-token prediction at word boundaries vs mid-word
   - Syntactic: subject-verb agreement across relative clauses (if feasible at 7-8B scale; fallback: gendered pronoun resolution)
   - Factual: simple factual recall (entity completion from Wikipedia contexts)
2. Run baseline performance on all tasks.
3. Ablation conditions (same as T1): none, ablate_high_si, ablate_low_si, ablate_random x3.
4. Compute per-task normalized degradation: delta_task = (baseline - ablated) / (baseline - floor).
5. Primary test: one-way ANOVA of delta_task across task categories (positional / boundary / syntactic / factual).
   - E1 predicts: no significant main effect of task category (F ~ 1, p > 0.05).
   - Specialized-operator predicts: positional delta_task >> other categories.
6. Secondary: interaction test (ablation_group x task_category). Under E1, the interaction is null; under specialization, high-SI ablation x positional category has the largest cell.

Seeds: 5.
Statistical unit: per-seed mean performance per (task, ablation_condition).

Acceptance:
1. Task battery has >= 4 categories with >= 2 tasks each.
2. All tasks exceed floor threshold (0.15 accuracy or perplexity ratio > 1.05 for NTP tasks).
3. ANOVA and interaction test computed with Holm correction across task families.
4. Report artifacts: `task_battery_results.parquet`, `anova_summary.json`, `interaction_test.json`.

**Sub-experiment 3P2-A.2: Residual stream probing**

Protocol:
1. Train linear probes on the residual stream at layers {0, 4, 8, 12, 16, 20, 24, 28, 31} for three target features:
   - (a) Relative position: predict binned offset to nearest word boundary (bins: 0, 1, 2, 3, 4+)
   - (b) Word boundary: binary classification (is this position a word boundary?)
   - (c) Part-of-speech tag: coarse POS (noun, verb, adj, other) from spaCy or stanza
2. Train probes on 500 sequences (len=512), evaluate on held-out 100 sequences.
3. Measure probe accuracy under 3 conditions:
   - Intact residual stream
   - High-SI head outputs zeroed (same zeroing as T1)
   - Random head outputs zeroed (matched count, 3 draws)
4. Compute delta_probe = (intact_accuracy - ablated_accuracy) for each (layer, feature, condition).
5. Primary test: paired comparison of delta_probe(high_SI) across the three features.
   - E1 predicts: delta_probe is similar magnitude across features (a), (b), (c).
   - Specialized-operator predicts: delta_probe is largest for features (a) and (b), smallest for (c).

Acceptance:
1. Probe baselines (intact) exceed chance by > 10 percentage points for all three features at layers >= 4.
2. Delta_probe reported per (layer, feature, condition) with bootstrap 95% CIs.
3. Artifacts: `probe_weights/`, `probe_accuracy.parquet`, `delta_probe_comparison.json`.

**Sub-experiment 3P2-A.3: Cross-task activation transfer**

Protocol:
1. Cache high-SI head o_proj outputs from 50 retrieval-task sequences (the "donor" context).
2. Run 50 word-boundary-detection sequences (the "recipient" context), patching in the donor high-SI head activations at corresponding sequence positions.
3. Measure recipient task performance relative to (a) intact baseline, (b) random-donor patching (activations from unrelated Wikipedia sequences).
4. E1 predicts: donor activations from retrieval still support boundary detection (because the positional content is generic). Performance under retrieval-donor patching ~ intact baseline >> random-donor patching.
5. Specialized-operator predicts: retrieval-donor activations carry task-specific positional signal unsuitable for boundary detection. Performance degrades relative to intact.

Acceptance:
1. At least 50 sequence pairs per condition.
2. Performance difference between donor-patched and intact reported with paired t-test and effect size.
3. Artifacts: `cross_task_transfer.json`, `transfer_results.parquet`.

Depends on: T1 head_groups.json (existing).
Estimated GPU-hours: ~8-12 total (probing is lightweight; task battery is the bottleneck).

---

### Experiment 3P2-B: Trivial Feature Control for Boundary Detection

**Tests:** E5 (boundary detection as primary function) — specifically whether the T5b result reflects a learned specialization or a trivial artifact of BPE space-prefix features.

**Models:** Llama-3.1-8B, OLMo-2-7B.
**Data:** Wikipedia (wiki40b_en_pre2019), 50 sequences, len=512.

**Sub-experiment 3P2-B.1: Space-prefix feature ablation**

Protocol:
1. Identify the embedding dimensions most correlated with the BPE space prefix (Gbar / underscore_):
   - Collect embeddings for all word-initial tokens (with space prefix) and all continuation tokens (without).
   - Fit a logistic regression classifier; extract top-k dimensions by coefficient magnitude (k=16).
2. Mean-ablate these k dimensions in the token embedding layer (replace with global mean for those dimensions).
3. Re-run T5b Approach A on the ablated model.
4. If high-SI heads still show elevated boundary attention after space-prefix information is removed, the detection is non-trivial.

Acceptance:
1. Space-prefix classifier achieves >= 90% accuracy (confirming the feature is identifiable and ablatable).
2. After ablation, report T5b Approach A metrics (attention to previous last token, Cohen's d) with 95% CIs.
3. Decision rule: if Cohen's d(high-SI vs low-SI boundary attention) remains > 0.5 after ablation, boundary detection is non-trivial. If d < 0.2, it is likely a trivial artifact.
4. Artifacts: `space_prefix_classifier.json`, `post_ablation_t5b_a.json`.

**Sub-experiment 3P2-B.2: Synthetic boundary insertion**

Protocol:
1. Construct adversarial sequences where BPE space-prefix tokens appear mid-word (by inserting space characters before continuation tokens in the raw text, then re-tokenizing).
2. Construct control sequences where genuine word boundaries lack the space prefix (by stripping spaces and concatenating words, then re-tokenizing — this produces long tokens spanning word boundaries).
3. Run T5b Approach A on both adversarial sets.
4. If high-SI heads follow the space prefix (attend to fake boundaries, ignore real boundaries without prefix), the detection is trivially feature-based. If they attend to actual linguistic boundaries regardless of prefix, it is learned.

Acceptance:
1. At least 200 adversarial boundary positions and 200 control positions.
2. Report boundary attention scores for high-SI heads at (real boundary with prefix, fake boundary with prefix, real boundary without prefix, fake boundary without prefix) — a 2x2 design.
3. Artifacts: `synthetic_boundary_results.json`, `adversarial_sequences.parquet`.

**Sub-experiment 3P2-B.3: Fixed-offset head comparison**

Protocol:
1. Identify heads that attend primarily to offsets other than t-1. Specifically:
   - t-2 heads: heads where argmax of mean attention profile is at offset 2
   - t-3 heads: heads where argmax is at offset 3
   - Diffuse heads: heads with high entropy in their attention profile (no strong offset preference)
2. Compute T5b Approach C boundary score for each group.
3. If boundary detection is a property of t-1 offset specifically (because boundary information is at the adjacent position), t-2/t-3/diffuse heads should show near-zero boundary scores.
4. If boundary detection is a broader property of SI heads (regardless of offset), all SI heads should show elevated boundary scores.

Acceptance:
1. At least 20 heads per offset group.
2. Report boundary score distributions per group with KS tests.
3. Artifacts: `offset_group_boundary_scores.json`.

Depends on: T1 head_groups.json, T5b artifacts (existing).
Estimated GPU-hours: ~4-6 total.

---

### Experiment 3P2-C: Redundancy Quantification

**Tests:** E2 (distributed redundancy) vs. E1 (genuine non-specificity).

**Models:** Llama-3.1-8B, OLMo-2-7B.
**Data:** Wikipedia (wiki40b_en_pre2019) + synthetic retrieval/local_key_match tasks.

**Sub-experiment 3P2-C.1: Cumulative ablation curve**

Protocol:
1. Sort all heads by R-squared (descending).
2. Ablate heads one at a time in this order, measuring task performance at each step.
   - Tasks: retrieval (span 48), local_key_match, next-token perplexity on 100 Wikipedia sequences.
   - Record performance at ablation fractions: 0%, 1%, 2%, 5%, 10%, 15%, 20%, 25%, 50%.
   - Implementation note (2026-04-04): if available wiki NTP sequences are fewer than requested (`num_seeds * ntp_count_per_seed`), the runner now auto-downshifts to the largest equal per-seed quota and logs the applied cap.
3. Fit two models to the degradation curve:
   - Linear: performance = a - b * fraction_ablated
   - Threshold: performance = a if fraction < threshold, else a - c * (fraction - threshold)
4. Compare model fits via BIC.
   - E2 (redundancy) predicts: threshold model fits better; performance holds until a critical fraction, then collapses.
   - E1 (non-specificity) predicts: linear model fits better; modest, gradual degradation with no threshold.

Seeds: 3 (for random ablation order as control).
Also run the same curve with heads sorted by *ascending* R-squared (ablate low-SI first) as a comparison.

Acceptance:
1. At least 9 ablation fraction steps per (model, task, sort_order) condition.
2. BIC comparison reported for both models per condition.
3. Artifacts: `cumulative_ablation_curve.parquet`, `curve_fit_comparison.json`.

**Sub-experiment 3P2-C.2: All-high-SI simultaneous ablation**

Protocol:
1. Ablate ALL high-SI heads (top 25%, ~256 heads) simultaneously.
2. Measure performance on the full task battery from Experiment 3P2-A.
3. Compare delta_task distribution under all-high-SI ablation vs. single-group ablation from T1.
   - E2 predicts: all-high-SI ablation causes dramatically larger, potentially threshold-crossing degradation compared to the 25% group ablation in T1 (because backup is overwhelmed).
   - E1 predicts: degradation is proportional to fraction removed, roughly 4x the single-group effect.

Note: T1 already tests this condition (ablate_high_si zeroes the top 25%). Sub-experiment 3P2-C.2 extends to 50% and 75% to test for nonlinearity.

Acceptance:
1. Test at 25%, 50%, 75% ablation of high-SI heads.
2. Nonlinearity test: is the 75% degradation significantly more than 3x the 25% degradation? (paired bootstrap comparison).
3. Artifacts: `simultaneous_ablation_results.parquet`, `nonlinearity_test.json`.

**Sub-experiment 3P2-C.3: Low-SI contribution probe (non-destructive)**

Protocol:
1. Keep model weights and active heads intact (no 75% structural ablation).
2. Decompose residual-stream contributions by head group (high-SI, low-SI, random matched).
3. Train matched linear probes to decode positional features from each group contribution stream:
   - nearest-boundary distance bin
   - boundary indicator
4. Compare probe performance and variance explained across groups.
5. Optional stress test: attenuate high-SI heads to 0.5 (not 0.0) and re-measure low-SI probe performance for compensation evidence.

Acceptance:
1. Probe baselines exceed chance by > 10 percentage points for all reported targets.
2. Report groupwise decode accuracy, delta-to-chance, and bootstrap CIs.
3. Interpretive rule pre-registered: low-SI redundancy support requires low-SI decode performance within 10% relative of high-SI for at least one positional target.
4. Artifacts: `low_si_contribution_probe.json`, `low_si_probe_results.parquet`.

Depends on: T1 head_groups.json (existing).
Estimated GPU-hours: ~6-10 total (cumulative curve is the bottleneck).

---

### Experiment 3P2-D: Architecture Tiebreaker for T7b

**Tests:** E4 (architecture-dependent mechanism).

**Models:** Mistral-7B-v0.1 (GQA, RoPE, RMSNorm — same architecture family as Llama), plus optionally one standard-MHA model if available at 7B scale.
**Data:** Synthetic repeated random sequences (period=32, len=128), 30 pairs.

**Sub-experiment 3P2-D.1: Mistral T7b replication**

Protocol:
1. Run the full T7b protocol (activation patching) on Mistral-7B:
   - R-squared profiling (50 wiki sequences, len=512)
   - Induction scoring (20 synthetic sequences)
   - Resample patching (30 pairs, source heads L0-L7)
2. If Mistral (GQA like Llama) shows a null result (rho < 0.10, p > 0.05), the divergence is GQA-related.
3. If Mistral shows a positive result (rho > 0.15, p < 0.01), the divergence is model-specific, not architectural.

Acceptance:
1. Full T7b artifact set produced for Mistral-7B.
2. Comparison table: Llama / OLMo / Mistral T7b rho and p-values.
3. Artifacts: `results/experiment3_phase2/exp3p2d_architecture_tiebreaker/mistral-7b-v0.1/report.json`.

**Sub-experiment 3P2-D.2: GQA-aware patching for Llama**

Protocol:
1. Re-run T7b on Llama, but aggregate disruption scores per KV-head group (average across all query heads sharing the same KV head) before correlating with R-squared.
2. Rationale: GQA means the same KV projection is shared across multiple query heads. Per-query-head patching may dilute the signal. Grouping by KV head tests whether the null result is a GQA artifact.
3. If the grouped analysis yields rho > 0.15 and p < 0.05, the Llama null was a methodological artifact of GQA dilution. If it remains null, the divergence is genuine.

Acceptance:
1. Report both per-query-head and per-KV-group rho/p for Llama.
2. Artifacts: `theory7b_llama_gqa_grouped.json`.

Depends on: Existing T7b infrastructure. Mistral-7B already in Experiment 2 model set.
Estimated GPU-hours: ~4-6 total.

---

### Experiment 3P2-E: T5 vs T5b Reconciliation

**Tests:** Internal consistency between T5 (not supported) and T5b (supported).

This experiment requires no new data collection — it is a re-analysis of existing T5 and T5b artifacts.

**Sub-experiment 3P2-E.1: Disaggregated T5 position-type analysis**

Protocol:
1. Load T5 raw data (per-token loss under each ablation condition).
2. Disaggregate the interaction effect into four position types from T5b Approach B:
   - mid_continuation, last_subword, word_initial_after_multi, word_initial_after_single
3. Compute ablation loss increase per (position_type, ablation_condition).
4. The T5 "wrong sign" arose because word-initial loss increase (4.106) > continuation loss increase (2.799) for high-SI ablation. But this is actually *consistent* with boundary detection (T5b): high-SI heads are most important at boundary-adjacent positions (word_initial > continuation).
5. Test whether the T5 interaction reverses after controlling for baseline loss difficulty (word-initial tokens have higher baseline loss: 1.412 vs 1.071).

Acceptance:
1. 4-way position-type breakdown of T5 interaction for both models.
2. Baseline-loss-controlled analysis (ANCOVA or matched-sample comparison).
3. Narrative reconciliation document explaining whether T5 and T5b are genuinely contradictory or reflect the same underlying mechanism measured differently.
4. Artifacts: `t5_t5b_reconciliation.json`, `position_type_breakdown.parquet`.

**Sub-experiment 3P2-E.2: Boundary-mediated continuation effect**

Protocol:
1. For continuation tokens, measure whether the loss increase from high-SI ablation is larger when the continuation token is *near* a word boundary (position 2 or 3 within a multi-subword word) vs. *far* from a boundary (position 4+ within a long word).
2. If the continuation damage is mediated by boundary detection, near-boundary continuations should show more damage than far-boundary continuations.
3. This tests whether boundary detection (T5b) and subword assembly (T5) are the same mechanism or distinct ones.

Acceptance:
1. At least 100 near-boundary and 100 far-boundary continuation positions.
2. Paired comparison with effect size and CI.
3. Artifacts: `boundary_mediation_analysis.json`.

Depends on: T5 and T5b existing artifacts.
Estimated GPU-hours: ~0 (CPU-only re-analysis).

---

### Experiment 3P2-F: Proxy Decomposition for R-squared

**Tests:** E6 (metric/proxy confound).

**Models:** Llama-3.1-8B, OLMo-2-7B.
**Data:** Existing T1/T5b/T7b/T10 artifacts + 100 wiki sequences (len=512) for auxiliary per-head features.

Protocol:
1. Build a unified per-head table containing:
   - mean R-squared, layer, head index, head norm, attention entropy, dominant offset, previous-token score
   - downstream outcomes: T7b disruption, T10 group disruptions, T5b boundary score
2. Fit nested models for each outcome:
   - Base: layer + head geometry proxies
   - Full: base + R-squared
3. Report partial effects and variance contribution of R-squared (delta R2, partial correlation, bootstrap CIs).
4. Run permutation sanity checks for R-squared within layer to test robustness against layer confounding.
5. Classify whether R-squared effect is retained or collapsed after proxy controls.

Acceptance:
1. Unified per-head table produced for both models with no missing key predictors.
2. For each outcome, report base/full model fit, delta R2, and confidence intervals.
3. Pre-registered retained/collapsed rule documented in output (for example, retained if median delta R2 > 0.02 with CI excluding 0).
4. Artifacts: `proxy_decomposition.json`, `per_head_features.parquet`.

Depends on: Existing Experiment 3 artifacts (T1/T5b/T7b/T10).
Estimated GPU-hours: ~1-2 total (mostly CPU analysis).

---

### Experiment 3P2-G: Dose-Response and Saturation Test

**Tests:** E7 (intervention saturation / floor effects).

**Models:** Llama-3.1-8B, OLMo-2-7B.
**Data:** Retrieval/local_key_match tasks + next-token perplexity set (100 wiki sequences).

Protocol:
1. Replace binary zero-ablation with attenuation scales for selected head sets:
   - high-SI and low-SI groups at scales {1.00, 0.75, 0.50, 0.25, 0.10, 0.00}
2. Evaluate tasks at each scale with matched seeds and examples.
3. Fit linear vs piecewise-saturation curves for degradation vs attenuation strength.
4. Test for early selective slopes (high-SI steeper than low-SI before floor region).
5. Quantify floor proximity to flag saturated regions where selectivity is uninterpretable.

Acceptance:
1. Dose-response curves reported per (model, task, group) with >= 6 attenuation levels.
2. Saturation model comparison (BIC/AIC) produced for each curve.
3. Selective-slope test reported in pre-floor region with corrected p-values.
4. Artifacts: `dose_response_curve.parquet`, `saturation_fit.json`.

Depends on: T1 ablation infrastructure and head groups.
Estimated GPU-hours: ~8-12 total.

---

### Experiment 3P2-H: Source/Target Specification Sweep

**Tests:** E8 (source/target set misspecification in T7b/T10).
**Execution gate:** Run only if Gate G1 continues and a pre-registered trigger for off-window source relevance is met.

**Models:** Llama-3.1-8B, OLMo-2-7B.
**Data:** Synthetic repeated-random sequences (same base protocol as T7b/T10), 20-30 pairs.

Protocol:
1. Keep L0-L7 as the primary confirmatory source window (original feeder prior).
2. Add exactly one secondary source window per model, selected *before* running 3P2-H using this trigger rule:
   - trigger is true if >= 25% of candidate feeder heads (top decile prev-token score intersect top quartile R-squared) lie outside L0-L7 in that model.
   - secondary window is the contiguous 8-layer band with the highest density of those triggered heads.
3. Recompute disruption-rho with two pre-registered target taxonomies only:
   - taxonomy A: original induction/random/low-SI groups.
   - taxonomy B: matched-control groups stratified by baseline induction sensitivity decile.
4. Run sensitivity analysis only for one pre-registered inclusion variant (`r2_top_quartile AND prev_token_top_decile`) to avoid a broad fishing sweep.
5. Compare specificity conclusions under each specification with tier-corrected multiplicity policy and report stability vs recovery.

Acceptance:
1. `source_target_preregister.json` committed before execution with trigger status, selected secondary window, and frozen taxonomy definitions.
2. Exactly 2 source windows x 2 target taxonomies evaluated per model (no unplanned expansions).
3. All correlation/specificity outputs include CIs, corrected p-values, and MDE/power fields.
4. Stability verdict (stable null vs recovered specificity) reported explicitly.
5. Artifacts: `source_target_preregister.json`, `layer_target_sweep.json`, `sweep_results.parquet`.

Depends on: Gate G1 continuation, T7b/T10 patching infrastructure, and T1 feature tables.
Estimated GPU-hours: ~4-7 total.

---

### Experiment 3P2-I: Tokenizer/Corpus Invariance Test (Publication-Readiness)

**Tests:** E9 (tokenizer/corpus-specific boundary artifact).
**Execution gate:** Run only if 3P2-B confirms a non-trivial boundary signal (`d > 0.50` post-controls and no prefix-following artifact).

**Models:** Llama-3.1-8B, OLMo-2-7B.
**Data:** Wikipedia + at least one additional corpus slice (for example code/news) with matched sequence counts.

Protocol:
1. Re-run T5b Approach A/C boundary metrics across corpora with matched length and token-count controls.
2. Apply tokenizer-marker perturbation controls from 3P2-B (space-prefix ablation / adversarial boundary variants).
3. Compare effect-size stability across (model, corpus, perturbation) cells.
4. Compute invariance score: relative effect retention after perturbation and corpus shift.
5. Determine whether boundary effects are robust, partially sensitive, or collapse under invariance checks.

Acceptance:
1. At least 2 corpora and 2 marker-control conditions per model.
2. Boundary effect sizes with CIs reported for all cells and pooled mixed-effects summary.
3. Explicit invariance verdict generated (invariant / partially sensitive / non-invariant), with tier label set to exploratory unless promoted.
4. Artifacts: `invariance_report.json`, `corpus_breakdown.parquet`.

Depends on: 3P2-B pass gate and existing T5b metric code.
Estimated GPU-hours: ~4-8 total.

---

### Experiment 3P2-J: Context-Conditional Specialization

**Tests:** E10 (context-conditional specialization).
**Execution gate:** Run only after 3P2-A task definitions are frozen and Gate G1 continues.

**Models:** Llama-3.1-8B, OLMo-2-7B.
**Data:** Task batteries from 3P2-A plus stratified subsets from wiki/synthetic contexts.

Protocol:
1. Pre-register regime definitions *quantitatively* before looking at intervention outcomes:
   - long-span retrieval: retrieval tasks with span >= 64
   - high-uncertainty tokens: baseline surprisal >= model-specific P90 on held-out wiki
   - boundary-dense contexts: >= 4 boundary tokens within prior 16-token window
   - noisy/rare-token contexts: token unigram frequency <= model-specific P10 in reference corpus
2. Freeze definitions in `regime_definition_manifest.json` with timestamp and dataset hash.
3. Recompute high-vs-low SI intervention effects within each regime and in the aggregate.
4. Fit interaction model: effect ~ group * regime with corrected multiple comparisons.
5. Compare regime-specific effects to aggregate effect to quantify dilution.
6. Produce regime-level interpretation matrix (strong conditional effects vs globally weak effects).

Acceptance:
1. Each regime has minimum sample count (>= 200 evaluated positions or equivalent task units).
2. Group x regime interaction reported with confidence intervals, corrected p-values, and MDE/power fields.
3. Regime-to-aggregate effect lift documented for both models.
4. Artifacts: `regime_definition_manifest.json`, `conditional_effects.parquet`, `regime_summary.json`.

Depends on: Gate G1 continuation, 3P2-A task definitions, and T1/T5/T10 intervention outputs.
Estimated GPU-hours: ~5-9 total.

---

## Phase 2 Artifact Contract (New Additions)

| Experiment | Required file | Mandatory fields (minimum) |
|---|---|---|
| G1 | `phase2_governance/gate_g1_decision.json` | `timestamp`, `tier1_results`, `proxy_collapse_rule`, `boundary_artifact_rule`, `continue_to_stage2`, `deferred_experiments` |
| 3P2-F | `exp3p2f_proxy_decomposition/<model>/proxy_decomposition.json` | `model`, `timestamp`, `tier`, `primary_test_id`, `outcomes`, `base_model_spec`, `full_model_spec`, `delta_r2`, `partial_effect_r2`, `mde_target`, `achieved_power`, `multiplicity_family`, `retained_vs_collapsed_verdict` |
| 3P2-G | `exp3p2g_dose_response/<model>/dose_response_curve.parquet` | `model`, `task`, `seed`, `group`, `attenuation_scale`, `metric_name`, `metric_value`, `floor_proximity`, `tier`, `primary_test_id`, `mde_target`, `achieved_power`, `multiplicity_family` |
| 3P2-H | `exp3p2h_source_target_spec/<model>/layer_target_sweep.json` | `model`, `timestamp`, `tier`, `primary_test_id`, `source_target_preregister`, `source_bins`, `target_taxonomies`, `rho_results`, `specificity_tests`, `mde_target`, `achieved_power`, `multiplicity_family`, `stability_verdict` |
| 3P2-I | `exp3p2i_tokenizer_corpus/<model>/invariance_report.json` | `model`, `timestamp`, `tier`, `execution_gate`, `corpora`, `perturbation_conditions`, `effect_sizes`, `invariance_score`, `invariance_verdict`, `multiplicity_family` |
| 3P2-J | `exp3p2j_conditional_regimes/<model>/conditional_effects.parquet` | `model`, `task`, `regime`, `group`, `seed`, `effect_metric`, `effect_value`, `sample_count`, `tier`, `primary_test_id`, `regime_definition_manifest`, `mde_target`, `achieved_power`, `multiplicity_family` |

## Execution Priority

Execution is staged and gated:

1. **Stage 1 Core:** 3P2-E -> 3P2-F -> 3P2-B (mandatory, low-cost, high-information).
2. **Gate G1 decision:** emit `phase2_governance/gate_g1_decision.json` and stop if continuation criteria fail.
3. **Stage 2a (if G1 continues):** 3P2-G (saturation check).
4. **Stage 2b (if G1 continues and H-trigger is true):** 3P2-H (narrow source/target check).
5. **Stage 2c (if G1 continues):** 3P2-D (architecture tiebreaker).
6. **Stage 2d (if G1 continues):** 3P2-C (redundancy quantification).
7. **Stage 2e (if G1 continues):** 3P2-A (broad integration test).
8. **Stage 2f (if G1 continues and 3P2-A definitions are frozen):** 3P2-J (conditional specialization).
9. **Stage 3 Publication-Readiness:** 3P2-I only if 3P2-B passed non-triviality controls and mechanism claims remain active.

Stage 2a-2c tmux runbook:

```bash
# Preflight
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

# Launch
tmux new-session -d -s exp3p2_stage2_llama "cd '/scratch/f004ndc/Kernel PE' && bash scripts/experiment3_phase2_stage2_gpu0.sh"
tmux new-session -d -s exp3p2_stage2_olmo "cd '/scratch/f004ndc/Kernel PE' && bash scripts/experiment3_phase2_stage2_gpu1.sh"
tmux new-session -d -s exp3p2_stage2_mistral "cd '/scratch/f004ndc/Kernel PE' && bash scripts/experiment3_phase2_stage2_gpu2.sh"

# Monitor
tmux ls
tmux capture-pane -p -t exp3p2_stage2_llama
tmux capture-pane -p -t exp3p2_stage2_olmo
tmux capture-pane -p -t exp3p2_stage2_mistral

# Finalize (only after all three sessions complete)
bash scripts/experiment3_phase2_stage2_finalize.sh
```

## Output Structure

All Phase 2 artifacts will be written to `results/experiment3_phase2/<experiment_name>/<model_name>/`:

```
results/experiment3_phase2/
  phase2_governance/
    gate_g1_decision.json
  exp3p2a_positional_broadcast/
    llama-3.1-8b/
    olmo-2-7b/
  exp3p2b_trivial_feature_control/
    llama-3.1-8b/
    olmo-2-7b/
  exp3p2c_redundancy_quantification/
    llama-3.1-8b/
    olmo-2-7b/
  exp3p2d_architecture_tiebreaker/
    mistral-7b-v0.1/
    llama-3.1-8b/  (GQA-grouped reanalysis)
  exp3p2e_t5_t5b_reconciliation/
    llama-3.1-8b/
    olmo-2-7b/
  exp3p2f_proxy_decomposition/
    llama-3.1-8b/
    olmo-2-7b/
  exp3p2g_dose_response/
    llama-3.1-8b/
    olmo-2-7b/
  exp3p2h_source_target_spec/
    llama-3.1-8b/
    olmo-2-7b/
  exp3p2i_tokenizer_corpus/
    llama-3.1-8b/  (Stage 3 only)
    olmo-2-7b/      (Stage 3 only)
  exp3p2j_conditional_regimes/
    llama-3.1-8b/
    olmo-2-7b/
```

## Decision Logic

After Phase 2 completion, adjudicate explanations as follows:

| Outcome | Implication |
|---|---|
| Gate G1 fails proxy-collapse rule (3P2-F) | Pause 3P2-G/3P2-H/3P2-J and re-scope SI-head intervention claims before further GPU experiments |
| Gate G1 fails boundary-validity rule (3P2-B artifact condition) | Mark E5 low-confidence and skip 3P2-I (publication-readiness robustness deferred) |
| Any primary test has achieved power < 0.80 or CI overlaps pre-registered MDE | Mark result `inconclusive`; do not update explanation support status |
| 3P2-A: ANOVA null (uniform degradation across task types) | E1 (general infrastructure) supported |
| 3P2-A: ANOVA significant (positional tasks selectively degraded) | E1 rejected; specialization hypothesis revived |
| 3P2-B: boundary d > 0.5 after space-prefix ablation | T5b is non-trivial; E5 (boundary primary) strengthened |
| 3P2-B: boundary d < 0.2 after space-prefix ablation | T5b is a trivial artifact; E5 rejected |
| 3P2-C: threshold model fits better than linear | E2 (redundancy) supported; revisit T1 interpretation |
| 3P2-C: linear model fits better | E2 rejected; T1 null is genuine non-specificity |
| 3P2-C.3: low-SI contribution probes recover positional features near high-SI level | E2 strengthened (evidence for distributed backup signal) |
| 3P2-C.3: low-SI contribution probes remain weak | E2 weakened (backup signal likely limited) |
| 3P2-D: Mistral null (like Llama) | T7b divergence is GQA-related (E4 supported) |
| 3P2-D: Mistral positive (like OLMo) | T7b divergence is model-specific, not architectural |
| 3P2-E: controlled T5 interaction reverses sign | T5 and T5b are consistent; boundary mechanism is primary |
| 3P2-E: controlled T5 interaction remains wrong-signed | T5 and T5b measure different phenomena; dual mechanism |
| 3P2-F: proxy-adjusted R-squared effect retained | E6 weakened; R-squared remains mechanistically informative |
| 3P2-F: proxy-adjusted R-squared effect collapses | E6 supported; reinterpret R-squared as mixed proxy |
| 3P2-G: early selective slope before floor-collapse | E7 supported; prior nulls likely saturation-masked |
| 3P2-G: only floor-collapse behavior with no selective slope | E7 weakened; saturation not the primary explanation |
| 3P2-H: recovered specificity after source/target revision | E8 supported; prior T7b/T10 settings were misspecified |
| 3P2-H: stable null under expanded specifications | E8 weakened; null conclusions are robust |
| 3P2-I (Stage 3 only): strong invariance across tokenizer/corpus shifts | E9 weakened; boundary mechanism appears general |
| 3P2-I (Stage 3 only): substantial attenuation or direction shifts | E9 supported; boundary signal is setup-dependent |
| 3P2-J: strong conditional effects in preregistered hard regimes | E10 supported; aggregate nulls hide regime-specific specialization |
| 3P2-J: weak effects across all regimes | E10 weakened; little evidence for conditional specialization |

## Stage 2 Post-Run Snapshot

<!-- STAGE2_POSTRUN_BEGIN -->
Snapshot timestamp: `2026-04-04 20:10:11`

Run IDs:
- `llama-3.1-8b`: `llama-3.1-8b_20260401_153909`
- `olmo-2-7b`: `olmo-2-7b_20260402_135540`
- `mistral-7b-v0.1`: `mistral-7b-v0.1_20260404_175041`
- `llama-3.1-8b (3P2-C.1)`: `llama-3.1-8b_c1_20260404_200917`
- `olmo-2-7b (3P2-C.1)`: `olmo-2-7b_c1_20260404_200917`

| Stage 2 item | Status | Notes |
|---|---|---|
| 3P2-G (Llama/OLMo) | complete | rows=180/180; early-selective=False/True |
| 3P2-H (Llama/OLMo) | complete | trigger=True/True; verdict=recovered_specificity/stable_null |
| 3P2-D.2 (Llama GQA grouped) | complete | delta_spearman=0.009676622444847433 |
| 3P2-D.1 (Mistral replication) | complete | spearman_rho=0.0962; p_one=0.0624; hypothesis_supported=false |
| 3P2-C.1 (Llama/OLMo cumulative ablation) | running | rerun launched after NTP shortage + bf16 conversion fixes in `exp3p2c_redundancy_quantification.py` |
| Governance artifact | updated | `results/experiment3_phase2/phase2_governance/stage2_status.json` |
<!-- STAGE2_POSTRUN_END -->
