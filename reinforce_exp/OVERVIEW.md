# Kernel PE Program Overview (Experiments 1-8 + `reinforce_exp` Consolidation)

Last updated: 2026-04-20  
Primary scope: explain the full experimental arc, why each stage existed, and what the reinforcement wave changed for submission readiness.

---

## 1) Program Motivation and Scientific Thread

The Kernel PE project started from one core mechanistic question:

1. Do transformer attention heads implement a **shift-invariant positional kernel** that is functionally real (not just a descriptive fit)?
2. If yes, is that kernel load-bearing for behavior, robust across model families, and interpretable as a reusable mechanism?

The full program was intentionally sequenced from:

1. **Measurement** (can we estimate the object cleanly?),
2. **Intervention** (does changing PE-frequency structure causally change behavior?),
3. **Mechanistic adjudication** (which concrete claims survive strict controls?),
4. **Application/extension** (can SI structure guide adaptation, tokenizer analysis, and theory unification?),
5. **Reinforcement** (can we harden weak points to NeurIPS-level reviewer scrutiny?).

`reinforce_exp` is the consolidation layer for step 5.

---

## 2) Narrative Arc Across Experiments 1 to 8

## Experiment 1: Measurement Foundation

**Why it was done**
- Establish whether a shift-invariant kernel can be recovered from QK geometry at all.
- Build reproducible profiling pipelines and preregistered gates.

**What it contributed**
- Strong evidence that SI structure exists, but with heterogeneity across models/layers.
- The primary hypothesis was only partially supported; spectral confirmatory gates were not fully passed.

**What remained unresolved**
- Measurement alone could not prove causality.
- Reviewers could still argue this was descriptive geometry, not functionally load-bearing structure.

---

## Experiment 2: Causal Frequency Interventions

**Why it was done**
- Move from descriptive SI profiling to causal perturbation on PE-frequency components.
- Test whether expected locality/globality hypotheses hold under controlled interventions.

**What it contributed**
- Core high-frequency hypothesis (`H1`) broadly supported.
- Low-frequency long-range selectivity (`H2`) was not supported (often reversed).
- Forced refinement of simplistic “frequency band = function” interpretations.

**What remained unresolved**
- Needed clearer mechanistic decomposition beyond frequency-band narratives.
- Needed model-level robustness and stricter controls against confounding.

---

## Experiment 3 (Core + Phase 2): Mechanistic Battery and Adjudication

**Why it was done**
- Convert E1/E2 causal hints into a multi-test mechanistic case (T1-T10, then 3P2 adjudication).
- Separate robust claims from attractive but fragile interpretations.

**What it contributed**
- Strong support for kernel load-bearing behavior and threshold-style redundancy in core channels.
- Boundary and specialization claims became more precise but model-conditional.
- Phase 2 resolved many ambiguities and formalized conservative wording for remaining caveats.

**What remained unresolved entering reinforcement**
- Llama strict boundary status was still ambiguous under conservative gating.
- Reviewer-facing dependence/pseudoreplication concerns still needed a dedicated statistical hardening pass.
- Cross-model breadth needed explicit reinforcement for headline claims.

---

## Experiment 4: SI-Guided Fine-Tuning for Math

**Why it was done**
- Test whether SI channels are not just explanatory but also actionable for adaptation.
- Probe whether SI-aware LoRA routing can improve math adaptation while preserving structure.

**What current artifacts show**
- `4C` trajectory outputs exist for Llama/OLMo and indicate very small SI drift during standard FT (near-stable SI structure over checkpoints).
- `4A` SI-aware LoRA comparisons exist, but primary `a_vs_b` gains are not convincing (effects are small and statistically weak in current runs).
- `4B` is not in the visible completed artifact set and appears deferred.

**Interpretation**
- SI structure looks stable under short FT; “easy win” adaptation advantages are not strongly established yet.

---

## Experiment 5: Tokenizer Effect Audit

**Why it was done**
- Stress-test whether boundary effects are tokenizer entanglement or broader SI mechanism.
- Explain why Llama behaves differently from OLMo in strict boundary controls.

**What current artifacts show**
- `5A` completed model/feature coverage but is labeled `status: partial` due acceptance constraints on finite dependence-index cells.
- `5B` same-family tokenizer variation is highly informative:
  - top-quartile SI-head Jaccard `0.177`,
  - rank Spearman `0.0477` (non-significant),
  - Llama-2 prefix artifact flag `false` vs Llama-3.1 `true`.
- `5C` perturbation battery completed with strong tokenizer perturbation sensitivity signals (notably fake-boundary and character-decompose effects).
- `5D` distribution-shift fragility completed with condition-level interaction deltas emitted.

**Interpretation**
- Tokenizer-sensitive SI identity is real and likely central for explaining model-conditional boundary behavior.

---

## Experiment 6: SI-Guided Math Knowledge Localization

**Why it was done**
- Test multiple training-time routing/localization recipes (6A-6E) for directing math capability into SI channels.

**What current artifacts show**
- Full artifact families exist for 6A-6E across TinyLlama/Llama/OLMo.
- Pattern across current comparisons is mostly mixed or unfavorable:
  - 6A gradient routing tends to reduce math accuracy (especially Llama) while often helping perplexity,
  - 6B/6D mostly small, non-robust differences,
  - 6C SI-student distillation is near-neutral/slightly mixed on math and worsens perplexity,
  - 6E SI-optimized tokenizer formatting hurts accuracy for Llama/OLMo in current runs.

**Interpretation**
- SI-aware training/control is nontrivial; naive routing/formatting is not sufficient for reliable gains.

---

## Experiment 7: CS-Theory Link (Welch/Phase-Transition)

**Why it was done**
- Bridge empirical SI results to a compressed-sensing style theory with quantitative predictions.

**What current artifacts show**
- 7A completed and produced strong correlations, but directionality is not straightforward (`prediction_vs_observed` Pearson is negative while `mu1_vs_observed` is positive).
- 7B completed, but predicted-vs-observed threshold relation is strongly negative in current fit.

**Interpretation**
- Theory-connection work is active but not yet in a clean confirmatory state for headline claims.

---

## Experiment 8: Tokenizer Analysis via SI Structure

**Why it was done**
- Convert tokenizer discussion into quantifiable SI-linked metrics and practical intervention tests.

**What current artifacts show**
- 8A runs exist for GPT-2, TinyLlama, Llama, OLMo.
- High-vs-low SI contrast is significant in all listed models, but original SIBAS-style threshold hypotheses are not globally supported.
- 8B pruning is model-dependent:
  - Llama: SI-matched mask beats several baselines on math and shows matched>mismatched math signal,
  - OLMo: pattern is mixed, with some metrics favoring mismatched/random baselines.
- 8C cross-lingual extension appears planned rather than completed in current artifacts.

**Interpretation**
- SI-tokenizer metrics are promising but not uniformly monotone across models; practical gains remain conditional.

---

## 3) Why `reinforce_exp` Was Necessary

After Exp1-Exp8, the project had strong signal but still exposed reviewer-critical weaknesses:

1. **Strict boundary ambiguity (especially Llama)**.
2. **Dependence-aware inference concerns** (pseudoreplication criticism risk).
3. **Need for third-model strengthening on core claims**.
4. **PE-scheme confound skepticism**.
5. **Proxy-regime semantics vs truly task-grounded specialization**.
6. **Potential ordering artifacts and specificity objections**.
7. **Construct validity concern: are SI head sets stable enough across domains?**

`reinforce_exp` targets exactly those objections with focused, decision-oriented experiments.

---

## 4) `reinforce_exp` Results in Context

Below is the consolidation status based on current artifacts under `results/reinforce_exp`.

### R1: Llama strict multidomain adjudication
- Artifact: `exp_r1_multidomain_gate/multidomain_gate_summary_v1.json`
- Result: **ambiguous overall** (`assessment=ambiguous`), with one clean domain and two ambiguous domains.
- Impact: preserves conservative wording; does not support unconditional cleanup of Llama boundary caveat.

### R2: Dependence-aware reanalysis
- Artifact: `exp_r2_dependence_reanalysis/claim_stability_table.json`
- Result: `7/8` directionally supported (`fraction_supported=0.875`), `claim_impact=strengthens_main_claim`.
- Impact: major statistical credibility upgrade; one caveated cell remains.

### R3: Third-model core replication (Mistral)
- Artifact: `exp_r3_core_replication/mistral-7b-v0.1/core_replication_report.json`
- Result headline: `kernel_load_bearing=true`, `boundary_strict_prefix_flag=false`, `c1_threshold_supported=true`.
- Impact: strengthens cross-model core claims and reduces two-model fragility.

### R4: PE-scheme contrast extension
- Artifact: `exp_r4_pe_scheme_contrast/pe_scheme_comparison.json`
- Result: `pe_scheme_generality_supported=true` with 2/2 positive non-RoPE-anchor replications.
- Impact: weakens “RoPE-only artifact” criticism.

### R5: Task-grounded conditional specialization
- Artifact: `exp_r5_task_grounded/interaction_comparison_proxy_vs_task.json`
- Result: interaction remains significant in both models, but proxy-task sign matches are weak (Llama `1/4`, OLMo `2/4`).
- Impact: supports interaction existence, not semantic transfer generalization.

### R2B: OLMo boundary power reinforcement
- Artifact: `exp_r2b_olmo_boundary_power/olmo-2-7b/claim_impact.json`
- Result headline: `does_not_reinforce_boundary_nontriviality`, `boundary_claim_status=descriptive_caveated`, `n_domains_supported=0`.
- Impact: forces conservative OLMo boundary wording despite prior strengths elsewhere.

### R5B: one-to-one regime alignment reinforcement
- Artifact: `exp_r5b_regime_alignment/interaction_transfer_report.json`
- Result: interaction significant in both models, but promotion gate fails (`all_models_pass=false`; sign concordance criterion fails).
- Impact: keeps “proxy-specific semantics” interpretation.

### NEW-R12: random-ordering null control for C1
- Artifacts: `exp_new_r12_ordering_control/<model>/summary.json`
- Result: both primary models have `10/10` threshold-majority votes (`strong_replication`).
- Impact: materially de-risks ordering-artifact criticism for redundancy-threshold claims.

### NEW-R14: kernel permutation specificity
- Artifact: `exp_new_r14_kernel_permutation/olmo-2-7b/summary.json`
- Result headline: true effect >> permuted effect (`specificity_ratio ~ 4.57`, `supports_specificity=true`).
- Impact: strengthens causal specificity interpretation of kernel subtraction effects.

### NEW-R15: SI head stability across domains
- Artifacts: `exp_new_r15_head_stability/<model>/stability_summary.json`
- Result: both models `partially_stable`.
  - Llama pairwise Jaccard roughly `0.636-0.829`.
  - OLMo pairwise Jaccard roughly `0.510-0.631`.
- Caveat: code-domain sample count is small in these runs.
- Impact: supports construct validity, but only as partial stability.

---

## 5) What Is Now Strong vs Still Caveated

## Stronger after consolidation

1. **Kernel load-bearing and threshold-style redundancy** are now among the strongest program-level claims.
2. **Cross-model credibility** improved via Mistral core replication.
3. **Statistical defensibility** improved via dependence-aware reanalysis.
4. **Specificity defenses** improved via ordering control (R12) and permutation specificity (R14).

## Still intentionally conservative

1. **Boundary mechanism wording** must remain model-conditional and caveated (Llama ambiguity + OLMo R2B directional caveat).
2. **Conditional specialization semantics** remain proxy-specific despite robust interaction effects.
3. **SI head-set stability** is supportive but partial, not absolute invariance.
4. **Exp6/Exp7/Exp8 extension claims** should be framed as active or mixed rather than definitive headline proof.

---

## 6) Why This Matters for a NeurIPS Narrative

For submission quality, the key shift from `reinforce_exp` is not “everything became positive.” It is that the claim set became:

1. **Sharper** (what is and is not supported is now explicit),
2. **More defensible** (dependence-aware and control-rich),
3. **Less fragile to obvious reviewer attacks** (ordering, specificity, cross-model scope),
4. **More honest about heterogeneity** (boundary and semantics caveats are now principled, not hand-wavy).

This is exactly what a robust consolidation phase should do.

---

## 7) Suggested Claim Framing (Program-Level)

Use the following framing style in downstream docs/manuscript updates:

1. **Core SI mechanism**: supported and replicated across three primary-scale models for kernel load-bearing and threshold-style redundancy.
2. **Boundary non-triviality**: supported in part, but model-conditional and conservatively caveated under strict controls.
3. **Conditional specialization**: interaction effects are robust; semantic transfer from proxy regimes is not established.
4. **Tokenizer relation**: tokenizer strongly influences SI head identity and some boundary behaviors, but applied gains are model-dependent.

---

## 8) Traceability Pointers

Primary reinforcement roots:

1. `results/reinforce_exp/exp_r1_multidomain_gate/`
2. `results/reinforce_exp/exp_r2_dependence_reanalysis/`
3. `results/reinforce_exp/exp_r3_core_replication/`
4. `results/reinforce_exp/exp_r4_pe_scheme_contrast/`
5. `results/reinforce_exp/exp_r5_task_grounded/`
6. `results/reinforce_exp/exp_r2b_olmo_boundary_power/`
7. `results/reinforce_exp/exp_r5b_regime_alignment/`
8. `results/reinforce_exp/exp_new_r12_ordering_control/`
9. `results/reinforce_exp/exp_new_r14_kernel_permutation/`
10. `results/reinforce_exp/exp_new_r15_head_stability/`

Upstream context docs:

1. `UNIFIED_RESULTS.md`
2. `experiment1/experiment1overview.md`, `experiment1/experiment1results.md`
3. `experiment2/experiment2overview.md`
4. `experiment3/experiment3overview.md`, `experiment3/experiment3results.md`, `experiment3/phase2/RESULTS.md`
5. `experiment4/TODO.md`
6. `experiment5/TODO.md`
7. `experiment6/pipeline.py`
8. `experiment7/TODO.md`
9. `experiment8/tokenizer_analysis_via_si_structure.md`

