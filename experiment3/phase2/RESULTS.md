# Experiment 3 Phase 2 Results

Evidence window: artifacts through **2026-04-09 14:45 EDT**.

This document summarizes what was run in Phase 2/Stage 3, what conclusions are currently supported, and what should be run next before paper freeze.

---

## 1. Scope and Execution Status

### Completed core experiments
- `3P2-E` (T5 vs T5b reconciliation): complete (Llama, OLMo).
- `3P2-F` (proxy decomposition): complete (Llama, OLMo).
- `3P2-B` (trivial-feature control, strict full reruns): complete (Llama, OLMo).
- `3P2-G` (dose-response): complete (Llama, OLMo).
- `3P2-H` (source/target specification sweep): complete (Llama, OLMo).
- `3P2-D.2` (Llama GQA grouping): complete.
- `3P2-D.1` (Mistral replication, clean-prereq rerun): complete.
- `3P2-C.1` (cumulative ablation curve): complete (Llama, OLMo).
- `3P2-C.2` (simultaneous ablation nonlinearity stress): complete (Llama, OLMo).
- `3P2-C.3` (low-SI contribution probe): complete (Llama, OLMo).
- `3P2-I` rerun-after-B:
  - OLMo strict natural run: complete.
  - Llama strict natural run: blocked by gate.
- `3P2-A` (broad integration test): complete (Llama, OLMo).
- `3P2-J` (context-conditional specialization): complete (Llama, OLMo).

### Completed IDEAS.md executions
- `Idea 2` integrated into `3P2-I` via cross-lingual corpora (`opus_tr`, `opus_zh`, `opus_ru`).
- `Idea 4` structural ambiguity:
  - Base run: complete.
  - Normed rerun: complete.
  - Stage-4 balanced allowlist rerun (local + hierarchical relation types): complete.
  - Stage-4 consensus allowlist rerun (cross-model retained variants): complete.
  - Internal norming audit: complete.
- `Idea 6` (SI channels for math reasoning): complete (Llama, OLMo).

### Not yet executed
- `Idea 1`, `Idea 3`, `Idea 5`.

### Governance status
- Gate G1 passed (`continue_to_stage2=true`).
- `stage2_status.json` refreshed on `2026-04-09` with explicit completion-state fields and a consolidated completed-experiment list.
- Artifacts:
  - `results/experiment3_phase2/phase2_governance/gate_g1_decision.json`
  - `results/experiment3_phase2/phase2_governance/stage2_status.json`

---

## 2. Experimental Setup (as executed)

- Models: `llama-3.1-8b`, `olmo-2-7b`, plus `mistral-7b-v0.1` for `3P2-D.1`.
- Core data channels:
  - Refreshed Experiment 3 artifacts.
  - Synthetic intervention tasks (`long_range_retrieval`, `local_key_match`).
  - Wiki NTP channel.
  - Cross-lingual OPUS caches for Stage 3 invariance.
- Correction/power framing:
  - Tier fields and multiplicity families are emitted in per-experiment artifacts.
  - `3P2-F` remains below target power (`achieved_power=0.6715`).

---

## 3. Results by Experiment

## 3P2-E: T5 vs T5b Reconciliation
Artifacts:
- `results/experiment3_phase2/exp3p2e_t5_t5b_reconciliation/<model>/t5_t5b_reconciliation.json`

Key outputs:
- Llama: reconciliation verdict `consistent_with_boundary_primary`.
- OLMo: reconciliation verdict `consistent_with_boundary_primary`.

Interpretation:
- The T5/T5b mismatch is best explained as boundary-mediated behavior rather than direct continuation selectivity.

Weaknesses:
- Re-analysis-heavy (not a major fresh forward-pass experiment).

---

## 3P2-F: Proxy Decomposition for R²
Artifacts:
- `results/experiment3_phase2/exp3p2f_proxy_decomposition/<model>/proxy_decomposition.json`

Key outputs:
- Llama: median `delta_r2=0.00334`, verdict `retained`.
- OLMo: median `delta_r2=0.00318`, verdict `retained`.

Interpretation:
- Under preregistered collapse rules, R² remains retained after proxy controls (E6 not strongly supported here).

Weaknesses:
- `achieved_power=0.6715`.
- Effect magnitudes are small.

---

## 3P2-B: Trivial Feature Controls (strict full reruns)
Artifacts:
- `results/experiment3_phase2/exp3p2b_trivial_feature_control/<model>/post_ablation_t5b_a.json`
- `results/experiment3_phase2/exp3p2b_trivial_feature_control/<model>/synthetic_boundary_results.json`

Key outputs:
- Llama:
  - Post-control boundary effect `d=1.0666`.
  - `prefix_following_artifact_flag=true`.
  - Statistical prefix assessment: `fake-real=+0.01429`, `p_one=0.0137`, `d=0.4700`.
- OLMo:
  - Post-control boundary effect `d=0.5594`.
  - `prefix_following_artifact_flag=false`.
  - Statistical prefix assessment: `fake-real=-0.00297`, `p_one=0.9462`, `d=-0.1520`.

Interpretation:
- Non-trivial boundary signal remains strong in both models.
- Strict gate validity remains model-conditional: OLMo passes; Llama fails with statistically supported prefix-following artifact under the updated practical-significance rule.

Notes:
- The prior missing synthetic-cell issue is fixed in strict full reruns (all four cells present under `cells`).

---

## 3P2-G: Dose-Response / Saturation
Artifacts:
- `results/experiment3_phase2/exp3p2g_dose_response/<model>/saturation_fit.json`

Key outputs:
- Llama: `early_selective_slope_supported=false`.
- OLMo: `early_selective_slope_supported=true`.

Interpretation:
- Mixed E7 support: OLMo shows early selective behavior; Llama does not.

Weaknesses:
- Includes deterministic attenuation/interpolation shortcuts in parts of the pipeline.

---

## 3P2-H: Source/Target Specification Sweep
Artifacts:
- `results/experiment3_phase2/exp3p2h_source_target_spec/<model>/layer_target_sweep.json`

Key outputs:
- Llama: `stability_verdict=recovered_specificity`.
- OLMo: `stability_verdict=stable_null`.

Interpretation:
- E8 gets model-conditional support (recovery in Llama, not OLMo).

---

## 3P2-D: Architecture Tiebreaker
Artifacts:
- `results/experiment3_phase2/exp3p2d_architecture_tiebreaker/mistral-7b-v0.1/report.json`
- `results/experiment3_phase2/exp3p2d_architecture_tiebreaker/cross_model_comparison.json`
- `results/experiment3_phase2/exp3p2d_architecture_tiebreaker_np60/mistral-7b-v0.1/report.json`
- `results/experiment3_phase2/exp3p2d_architecture_tiebreaker_np60/cross_model_comparison.json`

Key outputs:
- Baseline contrast remains:
  - Llama null-like.
  - OLMo positive.
- Mistral `3P2-D.1` higher-power rerun (`np60`):
  - `spearman_rho=0.1198`, `p_one=0.0278`, `hypothesis_supported=true`.
  - CI is broad and crosses zero (`[-0.0138, 0.2518]`).
  - After controlling for previous-token score, unique R² contribution is weak (`partial_r=0.0313`, `p=0.6191`).
  - Direct prerequisite artifacts are present (`theory1_per_sequence_r2.ok=true`, `theory7_prev_token_scores.ok=true`).
- Llama `3P2-D.2` grouped analysis:
  - `delta_spearman_group_minus_query=0.00968` (minimal change).

Interpretation:
- `np60` shows a marginal positive univariate trend, but the unique SI signal is not supported after previous-token control.
- Treat D.1 as mediated/non-unique evidence for SI-specific feeder effects; do not use it as stand-alone support for architecture-family claims.
- Architecture-family claim remains under-resolved; the cleanest reading is model-conditional behavior rather than a simple architecture split.

---

## 3P2-C.1: Redundancy Quantification (Cumulative Curve)
Artifacts:
- `results/experiment3_phase2/exp3p2c_redundancy_quantification/<model>/curve_fit_comparison.json`

Key outputs:
- Llama: `threshold_votes=6`, `linear_votes=0`.
- OLMo: `threshold_votes=6`, `linear_votes=0`.

Interpretation:
- Strong support for E2-style threshold/redundancy behavior over linear degradation.

Weaknesses:
- NTP channel downshifted from requested 300 to 198 total sequences.

---

## 3P2-C.2: Nonlinearity Stress Test (25/50/75%)
Artifacts:
- `results/experiment3_phase2/exp3p2c_redundancy_quantification/<model>/nonlinearity_test.json`

Key outputs:
- Llama: `superlinear_votes=0/3`, `supports_nonlinearity=false`.
- OLMo: `superlinear_votes=0/3`, `supports_nonlinearity=false`.

Interpretation:
- No evidence that 75% degradation exceeds the preregistered superlinear threshold (>3x 25%).
- This does not overturn C.1 threshold preference; it weakens a stronger “explosive nonlinearity” variant.

---

## 3P2-C.3: Low-SI Contribution Probe
Artifacts:
- `results/experiment3_phase2/exp3p2c_redundancy_quantification/<model>/low_si_contribution_probe.json`
- `results/experiment3_phase2/exp3p2c_redundancy_quantification/<model>/low_si_probe_results.parquet`

Key outputs:
- Llama:
  - `supports_e2_redundancy_rule=true`.
  - `probe_baselines_over_chance_10pp_all_groups_targets=false`.
- OLMo:
  - `supports_e2_redundancy_rule=true`.
  - `probe_baselines_over_chance_10pp_all_groups_targets=false`.

Interpretation:
- Directionally supports low-SI backup/overlap with high-SI signal.
- Not a perfect clean pass under the strictest baseline criterion (some target/group deltas < +10pp over chance).

---

## 3P2-I: Tokenizer/Corpus Invariance (with Idea-2 cross-lingual extension)
Artifacts:
- Canonical summary:
  - `results/experiment3_phase2/exp3p2i_tokenizer_corpus/invariance_summary.json`
- Strict rerun archive:
  - `results/experiment3_phase2/exp3p2i_tokenizer_corpus_rerun_after_b/invariance_summary.json`
- Llama override snapshot archive:
  - `results/experiment3_phase2/exp3p2i_tokenizer_corpus_override_snapshot/llama-3.1-8b/invariance_report.override_true_20260406.json`

Key outputs (canonical path as of 2026-04-09):
- OLMo strict natural run (no override):
  - `invariance_verdict=invariant`
  - `invariance_score=0.7518`
  - `passes_gate=true`, `override=false`
- Llama strict run (no override):
  - `status=blocked`
  - `invariance_verdict=blocked_by_gate`
  - `override=false`
  - blocker: `prefix_following_artifact_flag=true` under statistical gate (`p_one=0.0137`, `cohens_d=0.4700`, `min_abs_diff=0.005`)

Interpretation:
- External robustness is now strictly supported for OLMo.
- Cross-model strict invariance claim is still blocked by Llama gate failure.

---

## Idea 4: Structural Ambiguity (base + normed + balanced + consensus reruns)
Artifacts:
- Base:
  - `results/experiment3_phase2/idea4_structural_ambiguity/<model>/ambiguity_report.json`
- Normed rerun:
  - `results/experiment3_phase2/idea4_structural_ambiguity_normed/<model>/ambiguity_report.json`
- Stage4 balanced rerun:
  - `results/experiment3_phase2/idea4_structural_ambiguity_normed_stage4/<model>/ambiguity_report.json`
- Stage4 v2 rerun with strengthened local-scope stimuli + internal quality gate:
  - `results/experiment3_phase2/idea4_structural_ambiguity_normed_stage4_v2/<model>/ambiguity_report.json`
- Stage4 consensus rerun (28-variant cross-model retained allowlist):
  - `results/experiment3_phase2/idea4_structural_ambiguity_normed_stage4_consensus/<model>/ambiguity_report.json`
- Norming audit:
  - `results/experiment3_phase2/idea4_structural_ambiguity_normed_stage4_consensus/norming_audit_summary.json`

Key outputs:
- Prior `relation_type_interaction=NaN` issue is resolved.
- Stage4 v2 (balanced allowlist + stronger local prompts) quality gate passed in both models:
  - Llama quality gate: pass (`context_a=0.933`, `context_b=1.000`, `flip=0.933`).
  - OLMo quality gate: pass (`context_a=1.000`, `context_b=1.000`, `flip=1.000`).
- Stage4 consensus rerun (28 variants, balanced relation types) quality gate passed in both models:
  - Llama: `observed_interaction=0.3641`, `p_two_sided=0.0108`, verdict `high_si_more_causal_than_low_si`.
  - OLMo: `observed_interaction=0.3974`, `p_two_sided=0.0042`, verdict `similar_impact_high_vs_low`.
- Consensus audit shows clean internal norming (`flagged_variant_rate=0.0` in both models; relation counts `hier=15`, `local=13`).

Interpretation:
- Idea 4 is now substantially cleaner under internal controls (balanced/consensus allowlists + explicit quality gate).
- Remaining caveat is external validity: stimulus norming is still internal/model-based, not human-normed.

---

## 3P2-A: Positional Broadcast Test (Stage 4)
Artifacts:
- `results/experiment3_phase2/exp3p2a_positional_broadcast/<model>/a_report.json`
- `results/experiment3_phase2/exp3p2a_positional_broadcast/<model>/anova_summary.json`
- `results/experiment3_phase2/exp3p2a_positional_broadcast/<model>/delta_probe_comparison.json`
- `results/experiment3_phase2/exp3p2a_positional_broadcast/<model>/cross_task_transfer.json`

Key outputs:
- Llama: A.1 ANOVA supports specialization (`p=8.80e-07`, partial `eta^2=0.4860`).
- OLMo: A.1 ANOVA supports uniform degradation (`p=0.9345`, partial `eta^2=0.0092`).
- A.2 reporting bugfix: acceptance scope now evaluated on intact probes (`condition=none`) rather than all conditions.
- A.2 post-fix acceptance:
  - Rerun with context-aware NLTK POS tagging (`effective=nltk`) now passes strict all-feature criterion in both models.
  - Core positional features (`relative_position`, `word_boundary`) also pass in both models.
- A.3 transfer check is mixed and does not show retrieval-patch ≈ intact behavior.

Interpretation:
- Stage-4 `3P2-A` supports model-conditional integration: Llama shows category-selective effect structure, OLMo is more consistent with broad infrastructure behavior.
- A.3 is an explicit negative against a pure task-generic broadcast account (`retrieval_patch ≈ intact` does not hold in either model); this tension is central to interpretation, not a side note.

---

## 3P2-J: Context-Conditional Specialization (Stage 4)
Artifacts:
- `results/experiment3_phase2/exp3p2j_conditional_regimes/<model>/regime_summary.json`
- `results/experiment3_phase2/exp3p2j_conditional_regimes/<model>/conditional_effects.parquet`

Key outputs:
- Both models: `supports_e10_conditional_specialization=true`.
- Strong positive high-vs-low deltas in boundary-dense and rare-token regimes.
- Regime profile differs by model (e.g., OLMo high-uncertainty regime is not positive in the expected direction).

Interpretation:
- E10-style conditional effects are present but model-conditional in regime shape.

---

## Idea 6: SI Channels for Math Reasoning (Stage 4)
Artifacts:
- `results/experiment3_phase2/idea6_math_si_channels/<model>/intervention_summary.json`
- `results/experiment3_phase2/idea6_math_si_channels/<model>/math_channel_results.parquet`

Key outputs:
- Llama: `any_si_channel_improves_math=false`; inverse channel significantly hurts.
- OLMo: `any_si_channel_improves_math=false`; strong SI amplification hurts, but inverse-channel harm is not Holm-significant.

Interpretation:
- Intervention-only scaling does not improve math in this setup; current signal is better interpreted as causal fragility mapping than direct performance gain.

---

## 4. Cross-Experiment Conclusion Snapshot

Most supported currently:
- **E2 (distributed redundancy):** strongest cross-model invariant support from `3P2-C.1`; partial supporting evidence from `3P2-C.3`.
- **Boundary signal remains non-trivial:** strong support in both models via strict `3P2-B` reruns.

Model-conditional/mixed:
- **E7 (saturation masking):** mixed (OLMo positive, Llama null).
- **E8 (specification miss):** mixed (Llama recovery, OLMo stable null).

Still under-resolved:
- **Architecture dependence precision:** higher-power Mistral is weakly positive, but CI crosses zero and partial-control analysis suggests limited unique SI contribution.
- **Cross-model invariance claim:** OLMo strict positive, Llama strict blocked.
- **General-integration vs conditional-specialization branch:** no longer missing; `3P2-A` + `3P2-J` are now complete and support a model-conditional interpretation.

---

## 5. Weakness Importance Assessment (Updated)

| Weakness | Importance | Why it matters | Fix urgency |
|---|---|---|---|
| Llama strict gate failure in `3P2-B` (`prefix_following_artifact_flag=true`) | **High** | Prevents strict Llama `3P2-I`; blocks full cross-model invariance claim | High |
| `3P2-D.1` Mistral unique-SI signal remains weak after previous-token control | **High** | Positive univariate trend exists, but mediated effects weaken architecture-level interpretation | High |
| `Idea 4` lacks external human norming (internal gates now clean) | **Medium** | Internal quality is strong, but external norming would further strengthen claim transportability | Medium |
| Prior Idea4 stage4 `relation_type_interaction=NaN` due one-relation allowlist | **Resolved** | Balanced allowlist rerun restored interaction estimability in both models | Done |
| `3P2-C.3` fails strict “all probes >10pp above chance” criterion | **Medium** | Redundancy support is directional but not fully clean under strongest rule | Medium |
| RoPE confound not yet directly controlled (all primary models use RoPE) | **High** | Limits separation of learned SI specialization vs positional-bias facilitation by PE scheme | High |
| `3P2-A.2` acceptance-scope + coarse-POS quality limitations | **Resolved** | Acceptance now evaluated on `condition=none`; NLTK-backed rerun passes strict all-feature criterion in both models | Done |
| `3P2-C.1` NTP downshift (300 -> 198) | **Medium-low** | Affects NTP precision, not the unanimous threshold-vote pattern | Low-medium |
| `3P2-F` power below target (`0.6715`) | **Medium** | Weakens confidence in fine-grained E6 adjudication | Medium |
| Cross-model attribution remains thin (`n=2` primary + `n=1` tiebreaker) | **Medium** | Model-family divergences are robustly observed but not causally attributable to a single factor | Medium |
| No >8B replication in Phase 2 | **Medium-high** | Generalization to larger-scale circuits remains untested in this protocol window | Medium |
| `stage2_status.json` completion-state staleness | **Resolved** | Governance now explicitly records completion states and completed experiment list | Done |

---

## 6. Recommended Next Steps

1. **Run 3P2-B multiseed strict adjudication for Llama** (`7` seeds; `num_sequences=32`; `synthetic_target_per_cell=600`) and use pooled gate output as canonical strict decision.
2. **Run powered 3P2-J long-span repair in both models** with explicit `regime_coverage_manifest.json` and sample-threshold checks.
3. **Run non-RoPE anchor control** (`exp3p2k_non_rope_control`) to address PE-scheme confounding risk.
4. **Run tokenizer feature audit** (`exp3p2b_tokenizer_audit`) for cross-feature confound sensitivity (prefix/capitalization/punctuation-adjacency/length).
5. Keep Llama `3P2-I` canonical status strict-only (blocked unless multiseed strict gate flips to clean).
