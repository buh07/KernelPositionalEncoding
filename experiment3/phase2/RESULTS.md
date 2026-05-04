# Experiment 3 Phase 2 Results

Evidence window: artifacts through **2026-04-19 11:32 EDT**.

This document summarizes what was run in Phase 2/Stage 3+, which conclusions are supported, and what remains caveated.

---

## 1. Scope and Execution Status

### Completed core experiments
- `3P2-E` (T5 vs T5b reconciliation): complete (Llama, OLMo).
- `3P2-F` (proxy decomposition): complete (Llama, OLMo).
- `3P2-B` strict full rerun: complete (Llama, OLMo).
- `3P2-B` multiseed strict adjudication (Llama, 7 seeds): complete.
- Reinforcement multidomain strict adjudication (Llama, 12 seeds/domain over wiki+code+dialogue): complete (`R1`).
- `3P2-B` tokenizer audit extension (Llama, OLMo): complete.
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
- `3P2-J` long-span repair (`n_pairs>=256` target): complete (Llama, OLMo).
- `3P2-K` non-RoPE anchor control: complete (`gpt2-medium`).

### Completed IDEAS.md executions
- `Idea 2` integrated into `3P2-I` via cross-lingual corpora (`opus_tr`, `opus_zh`, `opus_ru`).
- `Idea 4` structural ambiguity:
  - Base run: complete.
  - Normed rerun: complete.
  - Stage-4 balanced allowlist rerun: complete.
  - Stage-4 consensus allowlist rerun: complete.
  - Internal norming audit: complete.
- `Idea 6` (SI channels for math reasoning): complete (Llama, OLMo).

### Not yet executed
- `Idea 1`, `Idea 3`, `Idea 5`.

### Runtime status snapshot
- `tmux ls`: no active sessions.
- All currently planned reruns in this wave are complete; remaining work is interpretation/packaging plus optional follow-up runs.

### Governance status
- Gate G1 passed (`continue_to_stage2=true`).
- `stage2_status.json` has been refreshed and now includes post-window additions (`3P2-B` multiseed, `3P2-J` long-span repair, `3P2-K`, tokenizer-audit extension).

---

## 2. Experimental Setup (as executed)

- Models:
  - Primary: `llama-3.1-8b`, `olmo-2-7b`.
  - Architecture tiebreaker: `mistral-7b-v0.1` (`3P2-D.1`).
  - Non-RoPE anchor: `gpt2-medium` (`3P2-K`; learned absolute positional embeddings).
- Core data channels:
  - Refreshed Experiment 3 artifacts.
  - Synthetic intervention tasks (`long_range_retrieval`, `local_key_match`).
  - Wiki NTP channel.
  - Cross-lingual OPUS caches for Stage 3 invariance (`tr`, `zh`, `ru`).
- Power/correction framing:
  - Tier and multiplicity fields are emitted in per-experiment artifacts.
  - `3P2-F` remains below target power (`achieved_power=0.6715`).
- Additional adjudication settings:
  - `3P2-B` multiseed: 7 seeds, fixed synthetic target per cell (`600`), pooled weighted Stouffer one-sided meta-test.
  - `3P2-J` long-span repair: explicit coverage manifest with powered long-span regime in both models.

---

## 3. Results by Experiment

## 3P2-E: T5 vs T5b Reconciliation
Artifacts:
- `results/experiment3_phase2/exp3p2e_t5_t5b_reconciliation/<model>/t5_t5b_reconciliation.json`

Key outputs:
- Llama: `consistent_with_boundary_primary`.
- OLMo: `consistent_with_boundary_primary`.

Interpretation:
- The T5/T5b mismatch is best explained as boundary-mediated behavior rather than direct continuation selectivity.

---

## 3P2-F: Proxy Decomposition for R²
Artifacts:
- `results/experiment3_phase2/exp3p2f_proxy_decomposition/<model>/proxy_decomposition.json`

Key outputs:
- Llama: `delta_r2.median=0.00334`, verdict `retained`.
- OLMo: `delta_r2.median=0.00318`, verdict `retained`.
- `achieved_power=0.6715`.

Interpretation:
- Under preregistered collapse rules, R² remains retained after proxy controls (E6 not supported as a collapse account here).

Weaknesses:
- Underpowered for fine-grained null adjudication.
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
  - Prefix assessment: `fake-real=+0.01429`, `p_one=0.0137`, `d=0.4700`.
- OLMo:
  - Post-control boundary effect `d=0.5594`.
  - `prefix_following_artifact_flag=false`.
  - Prefix assessment: `fake-real=-0.00297`, `p_one=0.9462`, `d=-0.1520`.
- Synthetic 2x2 cell coverage is complete (including `fake_boundary_with_prefix`) in both models.

Interpretation:
- Non-trivial boundary signal remains in both models.
- Strict gate remains model-conditional: OLMo clean, Llama flagged.

---

## 3P2-B (Llama): Multiseed Strict Adjudication
Artifacts:
- `results/experiment3_phase2/exp3p2b_trivial_feature_control_multiseed/llama-3.1-8b/multiseed_gate_summary.json`
- Canonical mirror:
  - `results/experiment3_phase2/exp3p2b_trivial_feature_control/llama-3.1-8b/multiseed_gate_summary.json`

Key outputs:
- `seeds_completed=[0..6]` (7/7 complete).
- Pooled summary:
  - `n_flagged=3/7`
  - `meta_p_one=4.21e-07`
  - `pooled_diff_fake_minus_real_with_prefix=0.00604`
  - `pooled_cohens_d=0.2446`
- Overall adjudication: `assessment=ambiguous`.
- Canonical recommendation emitted by artifact:
  - `strict_gate_passes=false`
  - `strict_gate_blocked=false`

Interpretation:
- Prefix-following evidence is directionally present in pooled statistics but not stable enough across seeds to classify as cleanly blocked under the preregistered stability rule.
- Canonical strict status is unresolved/ambiguous rather than clean-pass.

---

## 3P2-B Extension: Tokenizer Feature Audit
Artifacts:
- Per-model reports:
  - `results/experiment3_phase2/exp3p2b_tokenizer_audit/llama-3.1-8b/tokenizer_audit_report.json`
  - `results/experiment3_phase2/exp3p2b_tokenizer_audit/olmo-2-7b/tokenizer_audit_report.json`
- Aggregate report:
  - `results/experiment3_phase2/exp3p2b_tokenizer_audit/tokenizer_audit_report.json`

Key outputs:
- Llama:
  - Baseline `prefix_following_artifact_flag=true`.
  - Tested feature ablations (`space_prefix`, capitalization, punctuation adjacency, token-length bucket) do not collapse high-vs-low boundary effect (`collapse_flag=false` for all).
- OLMo:
  - Baseline `prefix_following_artifact_flag=false`.
  - No tested feature causes collapse; robustness is consistent across feature battery.
- Aggregate report caveat:
  - Top-level aggregate verdict fields are currently inconsistent with per-model reports; use per-model reports as source of truth.

Interpretation:
- OLMo remains robust to this confound battery.
- Llama remains prefix-flagged at baseline, but this extension did not isolate a single embedding-dimension feature ablation that removes SI-linked behavior.

---

## 3P2-G: Dose-Response / Saturation
Artifacts:
- `results/experiment3_phase2/exp3p2g_dose_response/<model>/saturation_fit.json`

Key outputs:
- Llama: `early_selective_slope_supported=false`.
- OLMo: `early_selective_slope_supported=true`.

Interpretation:
- Mixed E7 support: OLMo shows early selective slope; Llama does not.

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
- `results/experiment3_phase2/exp3p2d_architecture_tiebreaker_np60/mistral-7b-v0.1/report.json`
- `results/experiment3_phase2/exp3p2d_architecture_tiebreaker_np60/cross_model_comparison.json`
- `results/experiment3_phase2/exp3p2d_architecture_tiebreaker/llama-3.1-8b/theory7b_llama_gqa_grouped.json`

Key outputs:
- Mistral `np60`:
  - Spearman `rho=0.1198`, `p_one=0.0278`, `hypothesis_supported=true` (univariate).
  - Spearman CI crosses zero (`[-0.0138, 0.2518]`).
  - Pearson `r=0.0770`, `p=0.2193`.
  - Previous-token mediation check:
    - `spearman_prev_token_vs_disruption=0.1539`, `p=0.0137`.
    - `partial_r2_vs_disruption_ctrl_prev_token=0.0313`, `p=0.6191`.
  - Direct prerequisite artifacts are present (`theory1_per_sequence_r2.ok=true`, `theory7_prev_token_scores.ok=true`).
- Llama `3P2-D.2` grouped analysis: `delta_spearman_group_minus_query=0.00968` (minimal).

Interpretation:
- D.1 provides only weak marginal univariate support and does not show unique SI contribution after previous-token control.
- Architecture-family claim is not cleanly supported; model-conditional interpretation remains stronger.

---

## 3P2-C.1: Redundancy Quantification (Cumulative Curve)
Artifacts:
- `results/experiment3_phase2/exp3p2c_redundancy_quantification/<model>/curve_fit_comparison.json`

Key outputs:
- Llama: `threshold_votes=6`, `linear_votes=0`.
- OLMo: `threshold_votes=6`, `linear_votes=0`.

Interpretation:
- Strong cross-model support for E2-style threshold redundancy.

Weaknesses:
- NTP channel downshifted from requested `300` to `198` sequences.

---

## 3P2-C.2: Nonlinearity Stress Test (25/50/75%)
Artifacts:
- `results/experiment3_phase2/exp3p2c_redundancy_quantification/<model>/nonlinearity_test.json`

Key outputs:
- Llama: `superlinear_votes=0/3`, `supports_nonlinearity=false`.
- OLMo: `superlinear_votes=0/3`, `supports_nonlinearity=false`.

Interpretation:
- No support for an explosive superlinear collapse variant.

---

## 3P2-C.3: Low-SI Contribution Probe
Artifacts:
- `results/experiment3_phase2/exp3p2c_redundancy_quantification/<model>/low_si_contribution_probe.json`

Key outputs:
- Llama: `supports_e2_redundancy_rule=true`, strict baseline criterion not fully met.
- OLMo: `supports_e2_redundancy_rule=true`, strict baseline criterion not fully met.

Interpretation:
- Directional support for low-SI backup/overlap; not a perfect pass under the strictest baseline criterion.

---

## 3P2-I: Tokenizer/Corpus Invariance (with Idea-2 cross-lingual extension)
Artifacts:
- `results/experiment3_phase2/exp3p2i_tokenizer_corpus/invariance_summary.json`

Key outputs:
- OLMo strict natural run (`override=false`):
  - `invariance_verdict=invariant`
  - `invariance_score=0.7518`
- Llama strict natural run (`override=false`):
  - `status=blocked`
  - `invariance_verdict=blocked_by_gate`

Interpretation:
- Strict invariance is supported for OLMo.
- Strict cross-model invariance remains unresolved because Llama path is blocked in canonical strict mode.

---

## 3P2-K: Non-RoPE Anchor Control
Artifacts:
- `results/experiment3_phase2/exp3p2k_non_rope_control/non_rope_control_summary.json`
- `results/experiment3_phase2/exp3p2k_non_rope_control/gpt2-medium/non_rope_control_summary.json`

Key outputs (`gpt2-medium`):
- `si_structure_detected=true`.
- `boundary_non_trivial_after_control=true`.
- `rope_confound_weakened_for_anchor=true`.

Interpretation:
- The RoPE-only confound is weakened by a positive non-RoPE anchor showing SI-structured heads plus non-trivial boundary signal.
- This does not fully eliminate PE-scheme confounding; it narrows it.

---

## Idea 4: Structural Ambiguity (consensus rerun)
Artifacts:
- `results/experiment3_phase2/idea4_structural_ambiguity_normed_stage4_consensus/<model>/ambiguity_report.json`
- `results/experiment3_phase2/idea4_structural_ambiguity_normed_stage4_consensus/norming_audit_summary.json`

Key outputs:
- Llama:
  - `observed_interaction=0.3641`, `p_two_sided=0.0108`.
  - Verdict: `high_si_more_causal_than_low_si`.
- OLMo:
  - `observed_interaction=0.3974`, `p_two_sided=0.0042`.
  - Verdict: `similar_impact_high_vs_low`.
- Internal norming audit: `flagged_variant_rate=0.0` for both models; balanced relation counts retained (`local=13`, `hierarchical=15`).

Interpretation:
- The prior estimability issue is resolved.
- Result is model-split and internally clean, but still externally exploratory (no human norming).

---

## 3P2-A: Positional Broadcast Test (Stage 4)
Artifacts:
- `results/experiment3_phase2/exp3p2a_positional_broadcast/<model>/a_report.json`
- `results/experiment3_phase2/exp3p2a_positional_broadcast/<model>/delta_probe_comparison.json`
- `results/experiment3_phase2/exp3p2a_positional_broadcast/<model>/cross_task_transfer.json`

Key outputs:
- A.1 ANOVA:
  - Llama supports specialization (`p=8.80e-07`, partial `eta^2=0.4860`).
  - OLMo supports near-uniform degradation (`p=0.9345`, partial `eta^2=0.0092`).
- A.2 probe acceptance (post-fix, `condition=none`, NLTK backend):
  - all-feature criterion pass in both models.
  - core positional features (`relative_position`, `word_boundary`) pass in both models.
- A.3 cross-task transfer:
  - Llama `retrieval_patch_vs_intact` Cohen's `d=-0.6481` (significantly worse than intact).
  - OLMo `retrieval_patch_vs_intact` Cohen's `d=-0.8424` (significantly worse than intact).

Interpretation:
- A.1 supports model-conditional integration (specialized in Llama, broader in OLMo).
- A.3 is explicit negative evidence against pure task-generic broadcast (`retrieval_patch ≈ intact` does not hold).

---

## 3P2-J: Context-Conditional Specialization (Stage 4 + Long-Span Repair)
Artifacts:
- Baseline: `results/experiment3_phase2/exp3p2j_conditional_regimes/<model>/regime_summary.json`
- Repair: `results/experiment3_phase2/exp3p2j_conditional_regimes_longspan_repair/<model>/regime_summary.json`
- Coverage manifests: `results/experiment3_phase2/exp3p2j_conditional_regimes_longspan_repair/<model>/regime_coverage_manifest.json`

Key outputs (repair namespace):
- Coverage: `regime_coverage_complete=true` for both models.
- Long-span powered in both models (`n_pairs=1600` each).
- Interaction model supports conditional specialization in both models (`p_value_holm=1.11e-16`).
- Regime profile remains model-conditional:
  - Llama: all four regimes positive high>low.
  - OLMo: high-uncertainty regime is negative while other three regimes are positive.

Interpretation:
- E10 is supported cross-model with model-specific regime shape.
- Prior long-span underpower concern is resolved in this repair run.

---

## Idea 6: SI Channels for Math Reasoning
Artifacts:
- `results/experiment3_phase2/idea6_math_si_channels/<model>/intervention_summary.json`

Key outputs:
- Llama:
  - `any_si_channel_improves_math=false`.
  - `inverse_channel_hurts_math=true` (`mean_delta=-0.6467`, Holm-adjusted one-sided `p=8.68e-09`).
- OLMo:
  - `any_si_channel_improves_math=false`.
  - `inverse_channel_hurts_math=false` (inverse-channel Holm-adjusted one-sided `p=0.0797`).

Interpretation:
- Simple SI-channel scaling does not improve math in this setup.
- SI channels appear load-bearing/fragility-linked, but not a direct performance bottleneck to be solved by gain-only intervention.

---

## 4. Cross-Experiment Conclusion Snapshot

Most supported currently:
- **E2 (distributed redundancy):** strongest cross-model invariant support from `3P2-C.1`, with directional backup support from `3P2-C.3`.
- **Boundary signal is non-trivial:** strict `3P2-B` supports non-triviality in both models.
- **Conditional specialization (E10):** supported in both models after long-span repair, with model-specific regime profile.
- **Three-model core battery:** reinforcement `R3` replicates T8 load-bearing behavior, strict-B clean artifact status, and C.1 threshold preference in Mistral.
- **Ordering robustness of E2:** `NEW-R12` shows threshold-majority in 10/10 random ablation orderings for both primary models.
- **Kernel-specificity control:** `NEW-R14` supports true-kernel specificity over permuted-kernel perturbations in OLMo.

Model-conditional/mixed:
- **E7 (saturation masking):** OLMo positive, Llama null.
- **E8 (specification miss):** Llama recovery, OLMo stable null.
- **Boundary strictness:** OLMo clean; Llama strict status remains ambiguous/flagged depending on rule layer.

Under-resolved or caveated:
- **Architecture dependence (E4):** Mistral univariate trend is weak and mediated by previous-token signal.
- **Strict cross-model invariance:** still unresolved due Llama strict gating state.
- **Tokenizer-extension interpretation:** aggregate report is now synchronized with per-model outputs; Llama remains baseline prefix-flagged while OLMo remains robust.
- **OLMo boundary directional status under dependence-aware framing:** `R2` and `R2B` keep OLMo boundary non-triviality as descriptive/caveated.
- **Head identity stability:** `NEW-R15` is partially stable in both models, but does not meet high-confidence stability gates.

---

## 5. Weakness Importance Assessment (Updated)

| Weakness | Importance | Why it matters | Current state |
|---|---|---|---|
| Llama strict gate status (`3P2-B`) is ambiguous after multiseed | **High** | Canonical strict cross-model invariance claim depends on this | 7-seed phase2 adjudication and 12-seed/domain reinforcement adjudication both converge on ambiguous (not clean-pass, not stable-blocked) |
| Mistral D.1 unique SI signal weak after prev-token control | **High** | Prevents strong SI-specific architecture claim | Transparently caveated; no further rerun required for current claim set |
| `3P2-I` strict Llama path blocked | **High** | Blocks strict cross-model invariance statement | Still blocked in canonical strict summary |
| Tokenizer-audit aggregate report consistency | **Low** | Prevents stale top-level readouts from diverging from per-model outputs | Refreshed; aggregate includes both models; use per-model reports for primary interpretation |
| `Idea 4` external norming absent | **Medium** | Limits strength of syntactic-generalization claim | Internal quality now strong; keep exploratory label |
| `3P2-F` power below target (`0.6715`) | **Medium** | Weakens confidence in E6 null-like interpretation | Disclosed caveat |
| `3P2-C.3` strict baseline rule not fully met | **Medium** | Keeps redundancy backup claim directional rather than fully clean | Disclosed caveat |
| `3P2-C.1` NTP downshift (`300 -> 198`) | **Medium-low** | Precision reduction on one channel | Core threshold-vote pattern unchanged |
| RoPE confound | **Medium** | Needed PE-scheme separation | Reduced by positive non-RoPE anchor (`3P2-K`), not fully eliminated |
| Cross-model attribution thin (`n=2` primary + `n=1` tiebreaker) | **Medium** | Limits causal attribution of model differences | Must remain a scope limitation |
| No >8B replication in this protocol window | **Medium-high** | Limits scale generalization | Scope limitation; future-work target |
| `stage2_status.json` drift risk (future waves) | **Low** | Packaging/bookkeeping can lag if later reruns are not re-indexed | Refreshed for current wave; re-run governance refresh after any new post-window additions |

---

## 6. Recommended Next Steps

1. **Documentation/package lock:** keep governance summaries in sync for any future reruns; current wave has been refreshed (including `3P2-B` multiseed, `3P2-J` long-span repair, tokenizer audit extension, and `3P2-K`).
2. **Gate policy decision:** formally decide whether Llama strict status remains blocked under conservative canonical policy when multiseed is ambiguous.
3. **Tokenizer-audit maintenance:** preserve aggregate/per-model consistency in future reruns (single-model reruns should not overwrite the aggregate to a partial state).
4. **Manuscript framing:** lock claims to “E2 invariant + model-conditional realization,” with explicit caveats on Llama strict invariance and Mistral mediation.
5. **Optional high-value follow-up:** external human norming for Idea 4 and/or >8B replication, depending on pre-submission budget.

---

## 7. Reinforcement Update (R1-R5 + Follow-ups)

Artifacts:
- `results/reinforce_exp/exp_r1_multidomain_gate/multidomain_gate_summary_v1.json`
- `results/reinforce_exp/exp_r2_dependence_reanalysis/claim_stability_table.json`
- `results/reinforce_exp/exp_r3_core_replication/mistral-7b-v0.1/core_replication_report.json`
- `results/reinforce_exp/exp_r4_pe_scheme_contrast/pe_scheme_comparison.json`
- `results/reinforce_exp/exp_r5_task_grounded/interaction_comparison_proxy_vs_task.json`
- `results/reinforce_exp/exp_r2b_olmo_boundary_power/olmo-2-7b/claim_impact.json`
- `results/reinforce_exp/exp_r5b_regime_alignment/interaction_transfer_report.json`
- `results/reinforce_exp/exp_new_r12_ordering_control/<model>/summary.json`
- `results/reinforce_exp/exp_new_r14_kernel_permutation/olmo-2-7b/specificity_test.json`
- `results/reinforce_exp/exp_new_r15_head_stability/<model>/stability_summary.json`

Key outputs:
- **R1:** Llama strict adjudication remains aggregate-ambiguous; `strict_gate_passes=false`, `strict_gate_blocked=false`.
- **R2:** directional support in 7/8 audited claim cells; unsupported cell is OLMo boundary non-triviality.
- **R3:** Mistral core replication headline is positive (`kernel_load_bearing=true`, `boundary_strict_prefix_flag=false`, `c1_threshold_supported=true`).
- **R4:** both non-RoPE anchors are positive for SI structure + strict boundary non-triviality.
- **R5:** interaction remains significant; transfer fails (`1/4` Llama, `2/4` OLMo).
- **R2B:** OLMo boundary reinforcement does not directionally reinforce boundary non-triviality (`n_domains_supported=0`), preserving caveated wording.
- **R5B:** alignment reinforcement keeps interaction significant but transfer still fails (`1/2` in each model).
- **NEW-R12:** threshold preference is robust to random-order ablation (10/10 threshold-majority in each primary model).
- **NEW-R14:** OLMo true-vs-permuted kernel specificity is positive (mean diff `0.0427`, CI excludes 0, `p_one=2.0e-4`).
- **NEW-R15:** head identity is partially stable (Llama three-way Jaccard `0.594`, OLMo `0.427`) with small code-domain shard (`n=41`) as caveat.

Interpretation:
- Reinforcement evidence strengthens the E2-centered mechanism and T8 specificity channel.
- Boundary claims remain intentionally conservative and model-conditional.
- Interaction-level conditional specialization is robust, but regime semantics remain proxy-specific under transfer gates.
