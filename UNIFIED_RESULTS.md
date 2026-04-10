# Unified Results: Experiments 1, 2, 3, and 3 Phase 2

Evidence base locked from:
- `experiment1/experiment1overview.md`
- `experiment1/experiment1results.md`
- `experiment2/experiment2overview.md`
- `experiment3/experiment3overview.md`
- `experiment3/experiment3results.md`
- `experiment3/phase2/RESULTS.md`

In-flight snapshot timestamp (America/New_York): **2026-04-09 20:26:37 EDT**.

---

## Executive Summary

This program progresses from descriptive measurement to causal intervention to mechanistic adjudication:

1. **Experiment 1** established that shift-invariant positional structure is present but heterogeneous across model families and analysis choices. The primary hypothesis is **partially supported**.
2. **Experiment 2** tested causal frequency-band interventions. `H1` (high-frequency channels matter more for short-range tasks) was broadly supported, while `H2` (low-frequency channels selectively drive long-range tasks) was repeatedly reversed. The main conclusion is that RoPE-frequency usage is real but **model-conditional**, not universal.
3. **Experiment 3 (core)** tested mechanism-level theories. The strongest convergent positives were boundary-related behavior (`T5b`) and functional relevance of positional kernels (`T8`), while some feeder-specific claims (`T10`) were not supported.
4. **Experiment 3 Phase 2** adjudicated competing explanations (`E1`-`E10`). The cleanest current read is:
   - **E2 (distributed redundancy)** is the strongest cross-model invariant.
   - Boundary signal is real but model-dependent in cleanliness (strict in OLMo, confounded by prefix-following in Llama).
   - Cross-model mechanism realization is conditional (Llama vs OLMo differences are substantive).

Current in-flight work is mostly on strict Llama gate adjudication (3P2-B multiseed) and long-span completion parity for 3P2-J.

---

## Program Scope and Core Research Questions

### Overall scope
The full program asks whether transformer attention contains a causal, shift-invariant positional kernel; whether that kernel can be causally manipulated in frequency space; and what mechanistic role high shift-invariance (high-SI) heads play across models.

### Program-level questions
1. Is attention logit structure approximately a function of relative offset `Delta` (shift-invariant), and where (layer/head/model) does this hold?  
2. Are specific RoPE frequency bands causally responsible for short- vs long-range behavior?  
3. What do high-SI heads do mechanistically: generic infrastructure, boundary detection, feeder pathways, conditional specialization, or model-specific hybrids?

---

## Global Hypothesis Map (E1-E10 + prior hypothesis families)

| Family | Question | Tested primarily in | Current status | Key evidence pointers |
|---|---|---|---|---|
| Exp1 primary | Do RoPE+LN/RMSNorm models show shift-invariant kernels? | Experiment 1 | **Partially supported** | `experiment1/experiment1results.md` (Pre-Registered Criteria Outcomes, Primary Hypothesis) |
| Exp1 secondary | Do recovered kernels align spectrally with PE frequencies? | Experiment 1 | **Confirmatory gate not passed; exploratory positive signals** | `experiment1/experiment1results.md` (Secondary Hypothesis, spectral sections) |
| Exp2 `H1` | Do high-frequency ablations hurt short-range more? | Experiment 2 | **Broadly supported** (with caveats on pooled 1B confirmatory) | `experiment2/experiment2overview.md` (Final Results Summary, Interpretation) |
| Exp2 `H2` | Do low-frequency ablations selectively hurt long-range more? | Experiment 2 | **Not supported / reversed** | `experiment2/experiment2overview.md` (Final Results Summary, Interpretation) |
| E1 | General-purpose positional infrastructure | 3P2-A, 3P2-C, 3P2-J | **Mixed/model-conditional** | `results/experiment3_phase2/exp3p2a_positional_broadcast/*`; `experiment3/phase2/RESULTS.md` |
| E2 | Distributed redundancy | 3P2-C.1/C.2/C.3 | **Strongest cross-model support** | `results/experiment3_phase2/exp3p2c_redundancy_quantification/*`; `experiment3/phase2/RESULTS.md` |
| E3 | Non-specific feeder broadcasting | 3P2-A.3 and related | **Weakened as pure account** (A.3 transfer negative in both models) | `results/experiment3_phase2/exp3p2a_positional_broadcast/*/cross_task_transfer.json` |
| E4 | Architecture-dependent explanation for divergence | 3P2-D (incl. Mistral) | **Under-resolved/weak unique-SI evidence** | `results/experiment3_phase2/exp3p2d_architecture_tiebreaker_np60/*` |
| E5 | Boundary detection as primary function | 3P2-B, 3P2-I | **Supported in OLMo; Llama caveated by strict gate failure** | `results/experiment3_phase2/exp3p2b_trivial_feature_control/*`; `exp3p2i_tokenizer_corpus*` |
| E6 | R-squared proxy confound | 3P2-F | **Weakened, but limited power** | `results/experiment3_phase2/exp3p2f_proxy_decomposition/*` |
| E7 | Saturation/floor masking | 3P2-G | **Mixed** (OLMo positive, Llama null) | `results/experiment3_phase2/exp3p2g_dose_response/*` |
| E8 | Source/target misspecification | 3P2-H | **Mixed** (Llama recovery, OLMo stable null) | `results/experiment3_phase2/exp3p2h_source_target_spec/*` |
| E9 | Tokenizer/corpus artifact | 3P2-B, 3P2-I (+ pending tokenizer audit) | **Model-conditional concern remains** | `exp3p2b_trivial_feature_control/*`; `exp3p2i_tokenizer_corpus*` |
| E10 | Context-conditional specialization | 3P2-J | **Supported with model-conditional regime shape** | `results/experiment3_phase2/exp3p2j_conditional_regimes/*` and longspan repair namespace |

---

## Experiment 1 (Measurement Foundation)

### Setup and hypotheses
Source docs: `experiment1/experiment1overview.md`, `experiment1/experiment1results.md`.

- **Primary hypothesis:** attention logits are approximated by a head-specific shift-invariant function of `Delta` with stronger early-layer signal.
- **Secondary spectral hypothesis:** recovered kernels align with PE frequency structure.
- Dual measurement tracks:
  - **Track A (primary):** raw pre-softmax logit shift-invariance.
  - **Track B (secondary):** content-removed/centered Gram formulation.

### Key results
- Primary hypothesis is **partially supported**.
- NoPE baseline remained low and consistent with PE-specific expectation.
- Track-B centering methodology issues were explicitly surfaced; shared-mean centering substantially changed interpretation vs per-position centering.
- Historical confirmatory spectral gate did not pass; exploratory ungated spectral analyses were informative but non-confirmatory.

### Limitations/caveats carried forward
- Strong sensitivity of centered diagnostics to centering methodology.
- Spectral claim remained exploratory due gate provenance.
- Cross-family tokenizer differences limit naive natural-text comparability.

---

## Transition: Experiment 1 -> Experiment 2

### What was learned
- Shift-invariant structure exists but is heterogeneous and not adequately explained by static R2 maps alone.
- Frequency content is a plausible causal axis for intervention.

### What remained unresolved
- Whether the measured SI structure is causally load-bearing for task behavior.
- Whether different frequency regions of RoPE drive different dependency regimes.

### Why Experiment 2 was the minimal next step
- Frequency-targeted ablations/attenuations directly test causal contribution while reusing Exp1 kernel framing.
- This is the narrowest intervention that turns Exp1 from descriptive to causal evidence.

---

## Experiment 2 (Causal Frequency Interventions)

### Setup and hypotheses
Source doc: `experiment2/experiment2overview.md`.

- Confirmatory core vs exploratory extension explicitly separated.
- Main hypotheses:
  - `H1`: high-frequency ablation hurts short-range more.
  - `H2`: low-frequency ablation hurts long-range more.
  - `H3`: targeted effects exceed matched random controls (secondary).
  - `H4/H5`: exploratory norm-family and transfer effects.

### Key results (artifact-locked summary)
- `H1`: broadly supported across many branches.
- `H2`: consistently reversed (including attenuation branches), not rescued by hard-zero artifact checks.
- Pair-level analyses show model-conditional profiles (Llama stronger structured effect; OLMo non-monotonic; Mistral retrieval often non-evaluable in that branch).
- Interpretation: kernel frequency structure is useful but utilization is model-dependent; universal monotonic low-frequency-long-range narrative is not supported.

### Confirmatory vs exploratory boundary
- The document maintains strict boundary: exploratory pair findings are non-retroactive to confirmatory adjudication.

### Limitations/caveats carried forward
- Floor/headroom effects materially influence interpretability.
- Model-specific feasibility constraints reduce direct cross-model comparability in some exploratory branches.

---

## Transition: Experiment 2 -> Experiment 3

### What was learned
- Frequency interventions are behaviorally meaningful but insufficient to identify exact circuit roles.
- Coarse/global contrasts can hide per-head mechanism heterogeneity.

### What remained unresolved
- What high-SI heads compute mechanistically (boundary, feeder, general infrastructure, etc.).
- Whether frequency-related causal effects correspond to specific interpretable circuits.

### Why Experiment 3 was the minimal next step
- Head-level mechanistic tests (ablation, patching, boundary analyses) are required to disambiguate competing circuit explanations.

---

## Experiment 3 Core (Mechanistic Theory Battery T1-T10)

### Setup and question
Source docs: `experiment3/experiment3overview.md`, `experiment3/experiment3results.md`.

- Models: `llama-3.1-8b`, `olmo-2-7b`.
- Goal: adjudicate mechanistic theories of high-SI head function using targeted interventions and cross-references.

### Core theory outcomes (from generated `experiment3results.md`)

| Theory | llama-3.1-8b | olmo-2-7b | Interpretation shorthand |
|---|---|---|---|
| T1 | NOT SUPPORTED | NOT SUPPORTED | Aggregate SI-vs-task selectivity null in this formulation |
| T3 | DESCRIPTIVE | DESCRIPTIVE | Correlation-only support channel |
| T5 | NOT SUPPORTED | NOT SUPPORTED | Continuation-vs-initial formulation not supported |
| T5b | SUPPORTED | SUPPORTED | Boundary-related signal robust |
| T7 | SUPPORTED | SUPPORTED | Feeder-oriented evidence positive |
| T7b | NOT SUPPORTED | SUPPORTED | Cross-model split appears |
| T8 | SUPPORTED | SUPPORTED | Positional kernel is functionally relevant |
| T9 | DESCRIPTIVE | DESCRIPTIVE | Frequency-feeder analysis remains descriptive |
| T10 | NOT SUPPORTED | NOT SUPPORTED | Feeder specificity not supported in this setup |

### Implication entering Phase 2
Core results produced a tension: strong boundary/functional relevance signals coexisted with aggregate nulls and cross-model divergence, motivating explicit competing-explanation testing.

---

## Transition: Experiment 3 Core -> Experiment 3 Phase 2

### What was learned
- Some high-value claims were robust (`T5b`, `T8`), but multiple alternatives remained plausible (`E1`-`E10`).
- Cross-model asymmetries (e.g., `T7b`) required focused adjudication rather than one-pass interpretation.

### What remained unresolved
- Redundancy vs non-specificity (`E2` vs `E1`), architecture dependence (`E4`), tokenizer artifact risk (`E9`), and conditional specialization (`E10`).

### Why Phase 2 was the minimal next step
- Phase 2 is explicitly designed as an adjudication layer with gated, preregistered decision criteria across competing explanations.

---

## Experiment 3 Phase 2 (Adjudication Layer)

Primary source: `experiment3/phase2/RESULTS.md`.

### Scope status from documented Phase 2 results
- Completed in doc window: `3P2-A/B/C.1/C.2/C.3/D/E/F/G/H/I/J`, plus Idea 2/4/6 integrations (with caveats).
- Not executed in doc window: Idea 1/3/5.

### Per-experiment result summary

| Phase2 experiment | Main documented conclusion | Strictness/caveat |
|---|---|---|
| 3P2-E | T5/T5b mismatch reconciles toward boundary-primary interpretation | Re-analysis heavy |
| 3P2-F | Proxy-adjusted R2 retained (E6 weakened) | Power below target (`0.6715`) |
| 3P2-B | Boundary signal non-trivial in both models; Llama strict gate fails on prefix-following | Foundational strict gate |
| 3P2-G | OLMo early selective slope, Llama null | Mixed E7 support |
| 3P2-H | Llama recovered specificity, OLMo stable null | Mixed E8 |
| 3P2-D | Mistral marginal univariate positive at `np60`, but weak unique SI signal after prev-token control | E4 under-resolved |
| 3P2-C.1 | Threshold model unanimously preferred over linear in both models | Strong E2 support |
| 3P2-C.2 | No superlinear nonlinearity support | Weakens explosive-collapse variant |
| 3P2-C.3 | Directional low-SI backup support | Strict >10pp baseline rule not fully met |
| 3P2-I | OLMo strict invariance passes; Llama strict path blocked | Cross-model strict invariance unresolved |
| 3P2-A | Llama specialization ANOVA; OLMo near-uniform degradation | Model-conditional integration |
| 3P2-J | Conditional specialization supported; regime profile differs by model | Model-conditional E10 shape |
| Idea 4 | Cleaner after balanced/consensus reruns; internal quality gates pass | External human norming still absent |
| Idea 6 | SI channel amplification did not improve math in this setup | Better read as fragility mapping |

### Phase 2 cross-experiment synthesis (documented)
- Strongest invariant: **E2 distributed redundancy**.
- Strong but model-conditional boundary story: OLMo strict-clean vs Llama strict-blocked by tokenizer-linked artifact.
- Architecture-level single-factor claim remains weak.

---

## Discrepancy Notes (Documented Claims vs Latest Artifact State)

These are intentionally short and explicit, per strict reporting convention.

1. `experiment3/phase2/RESULTS.md` evidence window is through **2026-04-09 14:45 EDT**, but new artifacts/runs exist afterward.
2. Non-RoPE anchor (`3P2-K`) is now complete in artifacts:
   - `results/experiment3_phase2/exp3p2k_non_rope_control/non_rope_control_summary.json`
   - verdict includes `rope_confound_weakened_for_anchor=true`.
3. Long-span repair (`3P2-J`) has partial post-window completion:
   - OLMo coverage complete in `exp3p2j_conditional_regimes_longspan_repair/olmo-2-7b/*`.
   - Llama counterpart still pending.
4. Llama multiseed strict gate adjudication run is active under `exp3p2b_trivial_feature_control_multiseed/*`; pooled summary is not yet finalized.
5. Existing `results/experiment3_phase2/exp3p2b_trivial_feature_control/llama-3.1-8b/multiseed_gate_summary.json` is an earlier placeholder-style summary and should not be treated as the active multiseed run output.

---

## Current Running / Pending Experiments (Provisional Snapshot)

As of **2026-04-09 20:26:37 EDT**.

### Live tmux sessions
From `tmux ls`:
- `exp3p2_b_multi_g0`
- `exp3p2_b_multi_g1`
- `exp3p2_b_multi_finalize`
- `exp3p2_phase2_chain`

### 3P2-B multiseed strict adjudication (Llama)
Source checks:
- `scripts/experiment3_phase2_b_multiseed_status.sh`
- `results/experiment3_phase2/exp3p2b_trivial_feature_control_multiseed/*`

Provisional state:
- Seed artifacts complete: `seed0`, `seed1`, `seed2`, `seed3`.
- Pending/missing final artifacts: `seed4`, `seed5`, `seed6`.
- Pooled multiseed decision: **pending** (not finalized).

### 3P2-J long-span repair
Source checks:
- `scripts/experiment3_phase2_j_longspan_status.sh`
- `results/experiment3_phase2/exp3p2j_conditional_regimes_longspan_repair/*`

Provisional state:
- `olmo-2-7b`: coverage complete (`regime_coverage_complete=true` with long-span regime present and threshold-satisfying sample counts).
- `llama-3.1-8b`: pending in longspan-repair namespace.
- No active tmux session currently dedicated to this job.

### 3P2-K non-RoPE anchor
Source checks:
- `scripts/experiment3_phase2_nonrope_status.sh`
- `results/experiment3_phase2/exp3p2k_non_rope_control/non_rope_control_summary.json`

State:
- Complete on `gpt2-medium`.
- Summary verdict flags:
  - `si_structure_detected=true`
  - `boundary_non_trivial_after_control=true`
  - `rope_confound_weakened_for_anchor=true`

**Important:** this section is provisional runtime state, not final adjudication text.

---

## End-to-End Synthesis and What It Means

### Stable findings across the full program
1. Shift-invariant positional structure exists and is causally relevant.
2. Redundancy is a major organizational property of SI-related capacity.
3. Boundary-related computation is real but not identically realized across models.

### Model-conditional findings (now central, not peripheral)
1. Llama and OLMo differ in strict boundary artifact behavior and in conditional regime profiles.
2. Architecture-level single-cause interpretation remains weaker than a broader model-specific realization interpretation.

### Unresolved questions that still matter
1. Final strict adjudication of Llama multiseed gate status (`3P2-B` in-flight).
2. Full long-span parity for `3P2-J` in Llama repair branch.
3. How far non-RoPE anchor findings should narrow or reframe global RoPE-confound claims in the final manuscript.

### Immediate decision points after current in-flight runs finish
1. Finalize and lock pooled multiseed Llama gate decision.
2. Complete Llama long-span repair or mark regime as insufficient/pending for confirmatory claims.
3. Update `experiment3/phase2/RESULTS.md` evidence window and cross-experiment summary to incorporate post-14:45 artifacts before paper-freeze packaging.

---

## Traceability Index (Primary Pointers)

### Experiment 1
- `experiment1/experiment1overview.md`
- `experiment1/experiment1results.md`

### Experiment 2
- `experiment2/experiment2overview.md`
- `results/experiment2/quick/quick_pair_7seed_mistral_20260315_0205/reports/pair_expanded/story_snapshot.json`

### Experiment 3 (core)
- `experiment3/experiment3overview.md`
- `experiment3/experiment3results.md`
- `results/experiment3/theory*/<model>/*.json`

### Experiment 3 Phase 2
- `experiment3/phase2/RESULTS.md`
- `results/experiment3_phase2/exp3p2a_positional_broadcast/*`
- `results/experiment3_phase2/exp3p2b_trivial_feature_control/*`
- `results/experiment3_phase2/exp3p2c_redundancy_quantification/*`
- `results/experiment3_phase2/exp3p2d_architecture_tiebreaker*/*`
- `results/experiment3_phase2/exp3p2i_tokenizer_corpus*/*`
- `results/experiment3_phase2/exp3p2j_conditional_regimes*/*`
- `results/experiment3_phase2/idea4_structural_ambiguity*/*`
- `results/experiment3_phase2/idea6_math_si_channels/*`
- `results/experiment3_phase2/exp3p2k_non_rope_control/*`

