# Experiment 3 Phase 2: Combined Experiment Catalog

Evidence snapshots used while merging:
- `TODO.md`: 2026-04-04 (core Phase 2 protocol and governance)
- `IDEAS.md`: 2026-04-02 (Phase 2+ extension ideas)
- Merge generated: 2026-04-07

This file combines the experiment definitions from:
- `experiment3/phase2/TODO.md`
- `experiment3/phase2/IDEAS.md`

The goal is a single document listing all Phase 2 experiments in one place:
- Core preregistered protocol: `3P2-A` through `3P2-J`
- Phase 2+ extensions: `Idea 1` through `Idea 6`

This is a consolidation document. For original long-form rationale and exact wording, keep `TODO.md` and `IDEAS.md` as source references.

---

## 1) Unified Index

| ID | Name | Origin | Primary explanations tested | Gate / dependency | Retraining required? |
|---|---|---|---|---|---|
| 3P2-A | Positional Broadcast Test | TODO | E1 vs specialization alternatives | Stage 2e (after G1 continue) | No |
| 3P2-B | Trivial Feature Control for Boundary Detection | TODO | E5 validity vs trivial artifact | Stage 1 (mandatory) | No |
| 3P2-C | Redundancy Quantification (`C.1/C.2/C.3`) | TODO | E2 | Stage 2d (after G1 continue) | No |
| 3P2-D | Architecture Tiebreaker for T7b (`D.1/D.2`) | TODO | E4 | Stage 2c (after G1 continue) | No |
| 3P2-E | T5 vs T5b Reconciliation (`E.1/E.2`) | TODO | Internal consistency (T5/T5b), E5 narrative | Stage 1 (mandatory) | No |
| 3P2-F | Proxy Decomposition for R-squared | TODO | E6 | Stage 1 (mandatory) | No |
| 3P2-G | Dose-Response and Saturation Test | TODO | E7 | Stage 2a (after G1 continue) | No |
| 3P2-H | Source/Target Specification Sweep | TODO | E8 | Stage 2b trigger + G1 continue | No |
| 3P2-I | Tokenizer/Corpus Invariance (Publication Stage) | TODO | E9 | Stage 3; run only if 3P2-B gate passes | No |
| 3P2-J | Context-Conditional Specialization | TODO | E10 | Stage 2f, requires 3P2-A task freeze + G1 continue | No |
| Idea 1 | Word-Level Retokenization and Retraining | IDEAS | E5, E9 | Phase 2+ extension | Yes |
| Idea 2 | Cross-Lingual Tokenization | IDEAS | E5, E9, E1 | Phase 2+ extension (can fold into 3P2-I) | No |
| Idea 3 | N-Gram Tokenization | IDEAS | E1, E5, E9 | Phase 2+ extension | Yes |
| Idea 4 | Structural Ambiguity and Parse Invariance | IDEAS | E1, E5, E10 | Phase 2+ extension | No |
| Idea 5 | Superposition Within Frequency Bands | IDEAS | E6 (and E7 indirectly) | Phase 2+ extension | No |
| Idea 6 | Shift-Invariant Channels for Math Reasoning | IDEAS | E1, E10 | Phase 2+ extension | Intervention: No; architecture variant: Yes |

---

## 2) Core Protocol (3P2-A ... 3P2-J)

## 3P2-A: Positional Broadcast Test

**Purpose:** Distinguish broad positional infrastructure (E1/E3) from narrower specialization.

**Sub-experiments**
- `3P2-A.1` Multi-task simultaneous patching.
- `3P2-A.2` Residual stream probing.
- `3P2-A.3` Cross-task activation transfer.

**Key acceptance outputs**
- Taskwise degradation matrix, interaction statistics, and cross-task transfer analysis.

---

## 3P2-B: Trivial Feature Control for Boundary Detection

**Purpose:** Validate that T5b boundary signal is non-trivial (not a tokenizer-prefix artifact).

**Sub-experiments**
- `3P2-B.1` Space-prefix feature ablation.
- `3P2-B.2` Synthetic boundary insertion.
- `3P2-B.3` Fixed-offset head comparison.

**Execution role**
- Mandatory Stage 1 experiment.
- Feeds Stage 3 gate for `3P2-I`.

**Gate criterion for 3P2-I**
- Continue only if post-control `d > 0.50` and no prefix-following artifact.

---

## 3P2-C: Redundancy Quantification

**Purpose:** Test E2 (distributed redundancy) versus smooth non-specific degradation.

**Sub-experiments**
- `3P2-C.1` Cumulative ablation curve with linear vs threshold fit.
- `3P2-C.2` Simultaneous high-SI ablation stress test (25/50/75%).
- `3P2-C.3` Low-SI contribution probe (non-destructive decomposition/probing).

**Key acceptance outputs**
- `cumulative_ablation_curve.parquet`
- `curve_fit_comparison.json`
- `simultaneous_ablation_results.parquet`
- `nonlinearity_test.json`
- `low_si_contribution_probe.json`
- `low_si_probe_results.parquet`

---

## 3P2-D: Architecture Tiebreaker for T7b

**Purpose:** Resolve whether T7b divergence is architecture family effect (E4) or model-specific.

**Sub-experiments**
- `3P2-D.1` Mistral T7b replication (GQA family test).
- `3P2-D.2` Llama GQA-aware grouped patching.

**Key acceptance outputs**
- Mistral replication report.
- Llama per-query vs per-KV-group comparison.
- Cross-model comparison table.

---

## 3P2-E: T5 vs T5b Reconciliation

**Purpose:** Resolve apparent contradiction between T5 (not supported) and T5b (supported).

**Sub-experiments**
- `3P2-E.1` Disaggregated T5 position-type analysis.
- `3P2-E.2` Boundary-mediated continuation effect.

**Execution role**
- Stage 1 mandatory, mostly re-analysis.

---

## 3P2-F: Proxy Decomposition for R-squared

**Purpose:** Test whether R-squared is mechanistically informative after controlling for head-geometry proxies (E6).

**Execution role**
- Stage 1 mandatory.
- Feeds Gate G1 proxy-collapse check.

**Key acceptance outputs**
- Unified per-head feature table.
- Nested-model comparisons and retained/collapsed verdict.

---

## 3P2-G: Dose-Response and Saturation

**Purpose:** Test whether binary ablations masked selective effects via floor/saturation (E7).

**Execution role**
- Stage 2a after G1 continuation.

**Key acceptance outputs**
- Dose-response curves.
- Piecewise vs linear saturation fits.
- Early selective slope analysis.

---

## 3P2-H: Source/Target Specification Sweep

**Purpose:** Test whether T7b/T10 nulls are source/target set misspecification artifacts (E8).

**Execution role**
- Stage 2b after G1 continuation and preregistered trigger.

**Key acceptance outputs**
- Frozen preregistration manifest.
- Limited 2x2 source-window/target-taxonomy sweep.
- Stability verdict (`stable_null` or `recovered_specificity`).

---

## 3P2-I: Tokenizer/Corpus Invariance (Publication Readiness)

**Purpose:** Test external robustness of boundary signal across corpus/perturbation settings (E9).

**Execution gate**
- Run only if 3P2-B confirms non-triviality (`d > 0.50`) and no prefix-following artifact.

**Execution role**
- Stage 3 publication-readiness experiment.

**Key acceptance outputs**
- Invariance score and explicit verdict (`invariant`, `partially_sensitive`, `non_invariant`).
- Corpus-by-perturbation effect table.

---

## 3P2-J: Context-Conditional Specialization

**Purpose:** Test whether aggregate nulls hide strong effects in preregistered hard regimes (E10).

**Execution gate**
- Requires G1 continuation and frozen `3P2-A` task definitions.

**Execution role**
- Stage 2f.

**Key acceptance outputs**
- Regime definition manifest.
- Group x regime interaction tests.
- Regime-to-aggregate lift analysis.

---

## 3) Phase 2+ Extensions (Ideas 1 ... 6)

## Idea 1: Word-Level Retokenization and Retraining

**Core question**
- Is boundary detection a deep positional mechanism or BPE-fragmentation-dependent?

**Sketch**
- Train matched small model with strict word-level tokenizer.
- Re-run SI/T5b analyses under word-level tokenization.

**Interpretive use**
- Strong test of tokenizer dependence beyond marker controls.

**Priority in IDEAS**
- High.

---

## Idea 2: Cross-Lingual Tokenization

**Core question**
- Does boundary mechanism generalize across typologically different languages?

**Sketch**
- Re-run T5b-style metrics on agglutinative/isolating/fusional language slices.
- Compare R²-boundary coupling across languages.

**Interpretive use**
- Distinguishes universal boundary mechanism vs tokenizer/language-fragmentation dependence.

**Priority in IDEAS**
- Medium.

---

## Idea 3: N-Gram Tokenization

**Core question**
- Do SI heads track linguistically meaningful boundaries, or arbitrary token edges?

**Sketch**
- Retrain matched models with fixed n-gram tokenizers.
- Re-profile SI structure and boundary effects.

**Interpretive use**
- Complement to Idea 1 from the opposite direction (arbitrary segmentation).

**Priority in IDEAS**
- Medium-low.

---

## Idea 4: Structural Ambiguity and Positional Invariance in Semantic Parsing

**Core question**
- Do SI heads affect parse ambiguity resolution (beyond local token-boundary marking)?

**Sketch**
- Build ambiguity-focused stimuli (PP-attachment, clause attachment, coordination).
- Compare intact vs high-SI ablated vs low-SI ablated parse preference/consistency.

**Interpretive use**
- Mechanistic depth test connecting E1/E5/E10 in harder regimes than T1-style tasks.

**Priority in IDEAS**
- High.

---

## Idea 5: Superposition Within Frequency Bands

**Core question**
- Are high-frequency RoPE components single-purpose or multiplexed superposed channels?

**Sketch**
- Decompose per-pair contributions in high-SI heads.
- Apply factorization/dictionary methods and validate with selective pair attenuation.

**Interpretive use**
- Converts descriptive T9-style frequency observations into causal component tests.

**Priority in IDEAS**
- Medium.

---

## Idea 6: Shift-Invariant Channels for Mathematical Reasoning

**Core question**
- Does channeling computation through SI pathways help positional math tasks?

**Sketch**
- Intervention mode: amplify SI-head contribution, attenuate non-SI during math evaluations.
- Optional architecture mode: train Toeplitz-constrained SI head variants.

**Interpretive use**
- Tests whether SI is general infrastructure or selectively beneficial in math-like regimes.

**Priority in IDEAS**
- Medium.

---

## 4) Unified Stage and Dependency View

## Core staged execution (from TODO)

1. Stage 1 mandatory:
- `3P2-E -> 3P2-F -> 3P2-B`
2. Gate G1 decision.
3. Stage 2 conditional:
- `3P2-G`, `3P2-H`, `3P2-D`, `3P2-C`, `3P2-A`, `3P2-J` (with each experiment's own gate).
4. Stage 3 publication-readiness:
- `3P2-I` only if 3P2-B gate passes and mechanism claims remain active.

## Phase 2+ extension queue (from IDEAS)

- Lowest barrier extensions:
  - Idea 2 (cross-lingual, can fold into `3P2-I`)
  - Idea 4 (structural ambiguity, no retraining)
  - Idea 6 intervention mode
- Retraining-heavy extensions:
  - Idea 1 and Idea 3
- Analytical deepening:
  - Idea 5

---

## 5) Crosswalk: Ideas vs Core 3P2 Experiments

| Idea | Most related core experiment(s) | How they interact |
|---|---|---|
| Idea 1 | `3P2-B`, `3P2-I` | Stronger tokenizer-causality test than marker controls or corpus invariance alone |
| Idea 2 | `3P2-I` | Natural cross-lingual extension of invariance stage |
| Idea 3 | `3P2-B`, `3P2-E`, `3P2-I` | Tests whether "boundary" means linguistic segmentation or generic token-edge signal |
| Idea 4 | `3P2-A`, `3P2-J` | Hard-regime syntactic ambiguity test of conditional specialization and mechanistic depth |
| Idea 5 | `3P2-F`, `3P2-G`, T9 background | Decomposes what R²/frequency effects may represent at sub-band level |
| Idea 6 | `3P2-G`, `3P2-A` | Applies SI intervention logic to math-specific positional demands |

---

## 6) Artifact Guidance

Use TODO artifact contract as canonical schema for `3P2-*` runs. For `Idea *` runs, define artifact contracts before launch in the same style:
- fixed output root
- required fields
- tier and multiplicity family
- MDE/power fields
- preregistered acceptance criteria

---

## 7) Notes on Canonical Sources

- `TODO.md` remains canonical for preregistered core protocol (`3P2-A` ... `3P2-J`), gates, multiplicity policy, and artifact contract.
- `IDEAS.md` remains canonical for exploratory/extension motivation and priority arguments for `Idea 1` ... `Idea 6`.
- This combined document is a planning/navigation layer to avoid switching between two files during execution.
