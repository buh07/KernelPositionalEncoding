# reinforce_exp2 TODO: Full A/B/C Narrative Execution Plan (NeurIPS 2026)

Date: 2026-04-17  
Owner: f004ndc  
Purpose: Convert the A/B/C narrative options into an execution-grade, preregistered, reviewer-hardened pipeline with explicit requirements for evidentiary rigor.

This document is intentionally more operational and more stringent than `paper/story.md`.  
It defines what must be run, what each experiment must prove, and what standards are required before claims can appear in the main paper.

---

## 0) Mission and Success Criteria

### 0.1 Mission
Deliver a NeurIPS-ready submission that:
1. Avoids claim-evidence mismatch.
2. Preserves your strongest empirical findings.
3. Adds at least one mechanism-level result beyond descriptive characterization.
4. Contains explicit, falsifiable predictions.

### 0.2 Global Success Criteria (paper-level)
A submission is considered "ready" only if all of the following are true:
1. Main claims map to confirmatory artifacts with frozen decision rules.
2. All strict-vs-exploratory boundaries are explicit in figures, tables, and prose.
3. At least one mechanism bridge is demonstrated (kernel-shape or causal trace level).
4. Cross-model statements are phrased according to evidence class (invariant vs model-conditional vs unresolved).
5. Residual caveats are exposed in main text (not hidden in appendix only).

---

## 1) Non-Negotiable Rigor Standards (Apply to A, B, C)

## 1.0 Model-Set Definitions (Frozen)
1. Primary model set for confirmatory cross-model claims:
- `Llama-3.1-8B`
- `OLMo-2-7B`
- `Mistral-7B` (project-standard checkpoint; exact tag frozen in prereg artifact)
2. `reliability-eligible primary models` = primary models that pass the relevant reliability gate for that analysis family.
3. A model that fails a reliability gate remains part of the primary set but is marked `canonical_eligible=false` for that family.
4. Any acceptance rule requiring `>=2 reliability-eligible primary models` automatically fails confirmatory support if fewer than 2 are eligible.
5. Failures caused by eligibility shortfall must be labeled `mixed` or `deferred`, not silently dropped.

## 1.1 Preregistration Package (required per experiment)
Before launching any experiment in this plan, create a prereg JSON with:
1. `experiment_id`
2. `question`
3. `primary_hypothesis`
4. `primary_endpoints`
5. `secondary_endpoints`
6. `model_list`
7. `dataset_sources`
8. `inclusion_exclusion_rules`
9. `sample_size_plan`
10. `seed_plan`
11. `stopping_rule`
12. `multiplicity_family`
13. `acceptance_criteria`
14. `fallback_interpretation_if_null`

Required path pattern:
- `results/reinforce_exp2/<experiment_id>/preregistration.json`

## 1.2 Confirmatory vs Exploratory Labeling
Each experiment output must include:
- `analysis_tier`: `confirmatory` or `exploratory`
- `canonical_eligible`: boolean
- `override_used`: boolean

Any run with `override_used=true` is automatically exploratory and cannot feed canonical claim tables.

## 1.3 Data and Split Hygiene
1. Any head-selection procedure and evaluation procedure must use disjoint data where possible.
2. If overlap is unavoidable, it must be declared with a quantified overlap ratio.
3. For domain comparisons, sequence IDs must be stored so no duplicate examples are silently reused across conditions.

## 1.3a Shared Calibration Split Definition (`calibration_v1`)
1. `calibration_v1` is a frozen sequence-ID set used by B3 (`m_h`) and B4 (`E_h`) unless prereg explicitly overrides.
2. Construction rule:
- source pool = union of three explicitly defined domain loaders used in reinforcement runs:
  - `wiki`: cached `data/experiment1/wiki40b_en_pre2019/{model}/len_1024.jsonl`; if missing, fallback loader chain `wikitext/wikitext-103-raw-v1 -> wiki40b/en`.
  - `code`: cached `data/experiment1/codesearchnet_python_snapshot/{model}/len_1024.jsonl`; if missing, fallback loader chain `mbpp -> codeparrot/github-code`.
  - `dialogue`: loader chain `daily_dialog -> OpenAssistant/oasst1`.
- sequence-construction API must be identical to the multidomain reinforcement pipeline (`build_sequences_from_text_dataset`) before token-length filtering `128 <= token_count <= 512`.
- sample exactly 4096 sequences from this pool with fixed seed `20260417`.
- target domain composition for `calibration_v1`: `wiki/code/dialogue = 1/3 each` (tolerance ±2% absolute).
- keep domain stratification metadata and token-length metadata for reproducibility.
3. Required artifact:
- `results/reinforce_exp2/calibration_splits/calibration_v1_ids.parquet` with required columns:
  - `sequence_id`, `domain`, `loader_used`, `dataset_id`, `config_name`, `split`, `token_count`.
- `results/reinforce_exp2/calibration_splits/calibration_v1_manifest.json` with:
  - per-domain counts,
  - per-domain loader provenance summary (`loader_used` frequency),
  - seed, sampling timestamp, token-length filter parameters, and `sha256_ids_parquet`.
4. Confirmatory reproducibility lock:
- confirmatory B3/B4 runs must consume an existing frozen `calibration_v1_ids.parquet` and record its `sha256`.
- if `calibration_v1` must be regenerated (missing/corrupt artifact), that run is exploratory-only until the regenerated split is frozen and checksum-pinned in prereg.
5. Any override split must declare:
- source pool, sample size, seed, overlap ratio vs `calibration_v1`, and rationale.

## 1.4 Statistical Requirements
1. Report effect size and uncertainty, not only p-values.
2. Multiplicity correction required for every family of related tests.
3. One-sided tests allowed only when preregistered and direction justified.
4. Sequence-level or cluster-level inference required where dependence is present.
5. Required report bundle for each primary endpoint:
- raw estimate
- CI
- corrected p-value
- practical threshold check
- direction check

## 1.5 Reproducibility Requirements
1. All runs emit `manifest.json` including command, git commit hash, model checksum, seeds, and runtime.
2. Deterministic seeds for all random operations.
3. Any retries/fallbacks must be logged with rationale.
4. All summary JSONs must be schema-validated.

## 1.6 Artifact Contract (minimum)
Each experiment folder must include:
1. `manifest.json`
2. `preregistration.json`
3. `summary.json`
4. `claim_impact.json`
5. `data_dictionary.json`
6. machine-readable table(s) for all primary metrics

---

## 2) Shared Claim Taxonomy and Allowed Language

Use these statuses in all docs and paper text:
1. `supported`
2. `supported_with_caveat`
3. `proxy_specific`
4. `mixed`
5. `pending`
6. `deferred`

Language constraints:
1. Do not use "causally load-bearing" unless structural causal criteria are met.
2. Use "disruption-sensitive" / "functionally associated" when intervention is post-hoc component subtraction.
3. Use "threshold-capacity organization" instead of "distributed redundancy" unless mechanism alternatives are ruled out.

---

## 3) Narrative A Pipeline: Honest Characterization (Conservative, Fast)

Narrative A is mandatory baseline hygiene even if B/C are pursued.

### A0: Claim Hygiene Rewrite (No GPU)
Goal: eliminate overclaim and align wording with evidence class.

Required edits:
1. Demote T8 phrasing from mechanism-established to disruption-sensitive evidence.
2. Elevate cross-model heterogeneity (especially OLMo/Llama contrast) to central finding.
3. Re-label redundancy language as threshold-capacity unless B3 resolves alternatives.
4. Explicitly mark regime semantics as proxy-specific if transfer criterion fails.

Acceptance criteria:
1. Every headline claim in main text has a status tag.
2. No main claim depends on override-only artifact.
3. Contradiction audit passes (numbers and wording consistent across abstract/results/discussion).

Artifacts:
1. `results/reinforce_exp2/A0_claim_hygiene/claim_matrix.csv`
2. `results/reinforce_exp2/A0_claim_hygiene/claim_matrix.json`
3. `results/reinforce_exp2/A0_claim_hygiene/wording_changes.md`

### A1: Cross-Experiment Evidence Consolidation (No GPU)
Goal: produce a strict evidence table with line-of-evidence provenance.

Requirements:
1. For each major claim, list all supporting and conflicting artifacts.
2. Include sample size, model set, and caveat fields.
3. Add discrepancy field when old docs and latest artifacts diverge.

Acceptance criteria:
1. Zero unresolved claim-artifact mismatches.
2. All caveats have an explicit location in main text or appendix.

Artifacts:
1. `results/reinforce_exp2/A1_evidence_consolidation/evidence_registry.json`
2. `results/reinforce_exp2/A1_evidence_consolidation/discrepancy_log.md`

### A2: Conservative Prediction Layer (No GPU)
Goal: add at least one falsifiable prediction from current framework.

Required prediction:
1. Higher mean SI R² predicts greater degradation under position-shuffling intervention.

Acceptance criteria:
1. Prediction is written before running test.
2. Prediction includes explicit failure interpretation.

Artifacts:
1. `results/reinforce_exp2/A2_prediction_layer/predictions_preregistered.json`

### Narrative A Completion Gate
Narrative A is complete when A0, A1, A2 all pass.

Expected paper posture after A:
1. Strongly defensible characterization paper.
2. Lower novelty ceiling but high internal consistency.

---

## 4) Narrative B Pipeline: Kernels as Mechanisms (Primary Recommended Path)

Narrative B adds mechanistic depth while staying computationally moderate.

## B1: Kernel Shape Taxonomy (Required)

### B1.1 Question
Do high-SI heads decompose into stable kernel-shape families with interpretable functional roles?

### B1.2 Inputs
1. `estimated_kernels.json` from Llama, OLMo, Mistral.
2. Head metadata: layer, mean R², boundary metrics, T8 impact if available.

### B1.3 Preprocessing Requirements
1. Kernel normalization (z-score and unit-L2 variants; run both).
2. Optional denoising limited to preregistered smoothing kernel.
3. Preserve raw kernels for audit.
4. Build three preregistered feature spaces before clustering:
- `offset_space`: raw normalized `g_h(Δ)` vectors.
- `spectral_space`: FFT-derived features (dominant frequency amplitudes, phase summaries, spectral centroid, spectral entropy).
- `descriptor_space`: interpretable summary stats (peak offset(s), signed peak magnitude, width/sharpness, local-vs-global mass ratio, sign at Δ=0).
5. Freeze descriptor definitions before feature extraction:
- define `Delta_peak = argmax_Delta |g_h(Delta)|`.
- define local window `W_local = {Delta: |Delta - Delta_peak| <= 2}`.
- define local-vs-global mass ratio:
  - `R_local_global = sum_{Delta in W_local} |g_h(Delta)| / max(sum_{Delta notin W_local} |g_h(Delta)|, 1e-8)`.
6. Store both raw and derived descriptor values per head for audit.
7. Freeze feature extraction code before model-wise analysis.

### B1.4 Clustering Protocol
Run at least two methods:
1. hierarchical clustering (cosine distance)
2. k-means (euclidean on normalized vectors)

Feature-space normalization for k-means (frozen):
1. `offset_space`: per-head unit-L2 normalization (no per-dimension z-score).
2. `spectral_space`: per-dimension z-score across heads, then per-head unit-L2 normalization.
3. `descriptor_space`: per-dimension z-score across heads; no per-head unit-L2 normalization.

Model-selection protocol:
1. Candidate `k` in [3..10].
2. Choose `k` by quantitative criterion only (no subjective interpretability term):
- maximize mean bootstrap ARI in primary space (`descriptor_space`)
- tie-break 1: higher mean silhouette
- tie-break 2: smaller `k` (parsimony)
3. Report interpretability diagnostics separately; they cannot change selected `k`.
4. Run clustering in each feature space independently; do not treat one space as a post-hoc fallback.
5. Primary clustering space is `descriptor_space`; `spectral_space` and `offset_space` are robustness analyses.
6. Cross-method distance non-equivalence note:
- in `descriptor_space`, hierarchical (cosine) and k-means (euclidean on z-scored features) are intentionally non-equivalent analyses.
- confirmatory stability claims are method-internal (e.g., bootstrap ARI within chosen method), while cross-method agreement is descriptive-only.

Stability requirements:
1. Bootstrap resampling across heads (>=1000 replicates).
2. Adjusted Rand Index across replicates.
3. Split-half reproducibility by corpus split (disjoint halves where possible).

### B1.5 Primary Endpoints
1. Cluster stability (`ARI_bootstrap`)
2. Cross-model prototype similarity (cosine), with frozen normalization:
- primary `shape_only` comparison: descriptor vectors with all magnitude-carrying fields removed, then per-prototype unit-L2 normalization.
- secondary `magnitude_aware` comparison: full descriptor vectors with within-model z-scoring by descriptor dimension.
- report both; only `shape_only` is used for B1 acceptance criterion.
3. Cluster-level functional enrichment on out-of-clustering behavioral endpoints only:
- depth-controlled boundary effect (primary)
- T8 per-head disruption effect (secondary)
- regime-differential response (secondary)
4. Representation-concordance endpoint:
- agreement of cluster assignments between `descriptor_space` and `spectral_space`

Depth-control protocol for B1 enrichment (frozen):
1. For each cluster, compute boundary-effect enrichment within each layer separately (cluster heads vs non-cluster heads in same layer).
2. Minimum stratum size for confirmatory layer inclusion:
- cluster heads in layer >= 3
- non-cluster heads in layer >= 10
3. Layers failing minimum stratum size are excluded from confirmatory aggregation and logged.
4. Aggregate layer-wise effects using inverse-variance weighted meta-analysis.
5. Use Holm correction across clusters for confirmatory inference.
6. Regression-only depth controls may be reported as sensitivity analyses, but cannot replace the stratified primary protocol.
7. Practical effect threshold for B1 depth-controlled enrichment:
- absolute Hedges g >= 0.20 at the aggregated (IVW) level.

### B1.5a Model-Specific Reliability Gate
Before confirmatory interpretation, compute kernel reliability by model:
1. split-half kernel correlation across disjoint sequence shards
2. per-head kernel SNR proxy (frozen formula):
- partition calibration data into >=4 disjoint shards.
- estimate shard-wise kernels `g_h,s(Delta)` for each head `h` and shard `s`.
- mean kernel `mu_h(Delta) = mean_s g_h,s(Delta)`.
- signal power `P_signal(h) = Var_Delta(mu_h(Delta))`.
- residual power `P_resid(h) = mean_Delta Var_s(g_h,s(Delta) - mu_h(Delta))`.
- `SNR_h = P_signal(h) / max(P_resid(h), 1e-8)`.
- model-level SNR summary = median `SNR_h` over analyzed heads.
3. Freeze reliability minima before running model-wise clustering:
- model-level median split-half correlation >= 0.30
- model-level median kernel SNR >= 1.20
- at least 25% of analyzed heads with split-half correlation >= 0.50

Rationale:
1. Keep ARI acceptance threshold constant across eligible models (avoid moving-goalpost thresholding).
2. Handle low-SNR models via eligibility gating rather than weaker clustering criteria.

If reliability fails preregistered minimum for a model, that model is marked:
1. `canonical_eligible=false` for B1 mechanistic claims
2. `analysis_tier=exploratory` for cluster interpretation in that model
3. `reliability_failure_reason` must be emitted in summary artifact

### B1.6 Acceptance Criteria
`B1_supported` if all hold:
1. ARI stability threshold:
- each reliability-eligible primary model must have mean bootstrap ARI >= 0.60 for selected `k` in primary space (`descriptor_space`).
- pooled cross-model mean ARI is reported descriptively but does not replace per-model thresholding.
2. At least `min(3, k-1)` clusters have cross-model prototype cosine (`shape_only`) >= 0.70 across reliability-eligible model pairs.
3. Depth-controlled boundary enrichment survives Holm correction with practical effect threshold:
- in >=2 reliability-eligible primary models, at least one cluster has:
  - Holm-corrected `p < 0.05`, and
  - aggregated absolute Hedges g >= 0.20.
  - B1 depth-control Holm family is frozen as:
    - per-model family: `{clusters}` for depth-controlled boundary enrichment tests.
    - model-level support is evaluated per model after per-model Holm correction; cross-model pooling is descriptive-only.
4. At least one secondary behavioral enrichment endpoint (T8 or regime-differential) is directionally consistent.
  - directional consistency is frozen as:
    - for each available endpoint and model, compute Spearman correlation between cluster boundary-enrichment score and cluster behavioral endpoint score.
    - endpoint-model cell is directionally consistent iff `rho >= 0.20` and one-sided `p < 0.05`.
    - criterion passes if >=2 reliability-eligible primary models have at least one directionally consistent endpoint-model cell.
  - if `T8` is unavailable for a model, criterion 4 may be satisfied by regime-differential endpoint alone for that model, but must be labeled `behavioral_validation_tier=reduced_regime_only` in `B1_claim_impact.json`.
5. Descriptor-vs-spectral cluster concordance threshold:
- ARI between descriptor-space and spectral-space assignments >= 0.50 (averaged across reliability-eligible primary models).
6. Cross-model eligibility edge case:
- if fewer than 2 reliability-eligible primary models are available, B1 cannot be confirmatorily supported on cross-model prototype grounds and is automatically exploratory for cross-model claims.

If not met:
- Label B1 as exploratory heterogeneity map, not mechanistic taxonomy.

### B1.7 Required Outputs
1. `kernel_taxonomy_summary.json`
2. `cluster_stability_report.json`
3. `prototype_similarity_matrix.parquet`
4. `cluster_membership.parquet`
5. `cluster_function_enrichment.json`
6. `B1_claim_impact.json`

## B2a: Head-Level Kernel-Offset Alignment (Required, No B1 Dependency)

### B2a.1 Question
Do learned kernel profiles align with empirical boundary-offset statistics in tokenized corpora at the head level?

### B2a.2 Data Requirements
For each model:
1. empirical boundary profile by offset in a fixed reference frame.
2. kernel vectors `g_h(Δ)`.
3. disjoint corpora for:
- kernel estimation (train shard)
- boundary profiling (held-out shard)

### B2a.3 Reference-Frame and Sign Conventions (Must Be Frozen)
1. Primary boundary profile:
- `P(key_at_i_minus_delta_is_word_initial | query=i, Δ)` (key-relative framing).
2. Secondary sensitivity check:
- query-relative boundary profile.
3. Signed alignment metric and absolute-magnitude alignment metric must both be reported.
4. Positive and negative alignment are both interpretable:
- positive = boundary-seeking tendency
- negative = boundary-suppressing tendency

### B2a.4 Controls
1. Smoothness-preserving null via circular-shift or phase-randomized controls.
2. Token-frequency matched null.
3. Corpus-swap robustness (if multiple held-out shards available).

### B2a.5 Primary Endpoints
1. Head-level signed alignment score.
2. Head-level absolute alignment score.
3. Model-level aggregate alignment direction and magnitude.

### B2a.6 Acceptance Criteria
`B2a_supported` if all hold:
1. At least one preregistered alignment metric (signed or absolute) is estimable (non-missing with valid variance estimate) in >=2 reliability-eligible primary models.
  - valid variance estimate is defined as:
    - contributing head count `n_heads >= 10` for the model-level aggregate,
    - standard error is finite and strictly positive (`0 < SE < +inf`),
    - variance estimator is not rank-deficient (no singular covariance warning in the fitted aggregate model).
2. Corrected p < 0.05 for primary alignment family.
3. Practical effect threshold met (non-directional primary claim):
- absolute-alignment metric: Hedges g (observed vs matched-null) >= 0.20 in >=2 reliability-eligible primary models.
4. Directional subclaim eligibility (for boundary-seeking/suppressing language) requires:
- signed-alignment metric: absolute Hedges g (observed vs matched-null) >= 0.20 in >=2 reliability-eligible primary models,
- and signed direction is consistent across >=2 reliability-eligible primary models.
5. Null-control robustness (frozen; calibrated to prior reinforcement directional baseline):
- evaluate all preregistered null controls per reliability-eligible primary model.
- directional retention requirement: at least 75% of model-control cells retain the primary effect direction (matches prior 7/8 directional-support benchmark).
- magnitude retention requirement: in >=2 reliability-eligible primary models, median attenuation ratio across controls
  - `median_controls( g_control / max(g_primary, 1e-8) ) >= 0.50`.
- instability guard: no reliability-eligible primary model may show sign reversal in >1 null-control variant.
6. Any model failing kernel reliability gate is reported as exploratory-only for B2a claims.

Multiplicity family for B2a corrected tests (frozen):
1. Primary confirmatory Holm family:
- `{alignment_metric in [signed, absolute]} x {reliability-eligible primary models}`.
2. Exploratory robustness checks (e.g., corpus-swap sensitivities) are excluded from this family.

### B2a.7 Required Outputs
1. `boundary_offset_profiles.parquet`
2. `head_alignment_scores.parquet`
3. `alignment_sign_convention_report.json`
4. `alignment_null_tests.json`
5. `B2a_claim_impact.json`

## B2b: Cluster-Level Alignment Enrichment (Required, Depends on B1)

### B2b.1 Question
Given B1 cluster assignments, are specific kernel clusters enriched for boundary-aligned behavior?

### B2b.2 Primary Endpoints
1. cluster-level signed alignment enrichment.
2. cluster-level absolute alignment enrichment.
3. depth-controlled cluster enrichment.

Confirmatory cluster-family size control (frozen):
1. Confirmatory-eligible clusters must have >=5 heads.
2. If confirmatory-eligible cluster count > 4, retain top 4 by descending cluster size (tie-break by stable cluster label ordering).
3. Remaining clusters are exploratory-only for B2b confirmatory inference.

Depth-control protocol for B2b enrichment (frozen):
1. For each cluster, compute alignment enrichment within each layer separately (cluster heads vs non-cluster heads in same layer).
2. Minimum stratum size for confirmatory layer inclusion:
- cluster heads in layer >= 3
- non-cluster heads in layer >= 10
3. Layers failing minimum stratum size are excluded from confirmatory aggregation and logged.
4. Aggregate layer-wise enrichment via inverse-variance weighted meta-analysis.
5. Holm correction is applied across tested clusters in the confirmatory family.
6. Confirmatory Holm family for B2b:
- `{tested_clusters_confirmatory} x {reliability-eligible primary models}` for the depth-controlled enrichment tests.

### B2b.3 Acceptance Criteria
`B2b_supported` if:
1. at least one confirmatory-family cluster enrichment survives Holm correction in >=2 reliability-eligible primary models.
2. depth-control retention is satisfied:
- for at least one Holm-significant confirmatory-family cluster in >=2 reliability-eligible primary models:
  - depth-controlled effect sign matches the corresponding unadjusted cluster effect sign, and
  - attenuation ratio `|g_depth_controlled| / max(|g_unadjusted|, 1e-8) >= 0.50`.
3. effect exceeds practical threshold:
- aggregated (depth-controlled) absolute Hedges g >= 0.20 for at least one Holm-significant cluster.

### B2b.4 Required Outputs
1. `cluster_alignment_report.json`
2. `cluster_alignment_depth_control.json`
3. `B2b_claim_impact.json`

## B3: Cluster-Wise Ablation Mechanism Disambiguation (Required)

### B3.1 Question
Is threshold behavior driven by true distributed redundancy or by cluster-structured removal artifacts?

### B3.2 Conditions
At minimum, for each model:
1. within-cluster concentrated ablation
2. across-cluster mixed ablation (matched head count)
3. random matched ablation baseline
4. ranked-R² reference curve (existing baseline)

Matching constraints:
1. Match by head count.
2. Match by layer-depth distribution exactly for the primary analysis.
3. Match by aggregate perturbation magnitude proxy, frozen as:
- per-head perturbation magnitude `m_h = E_s,i[||z_h(s,i)||_2]` on held-out calibration corpus
- set-level magnitude `M(S) = sum_{h in S} m_h`
- primary matched sets must satisfy `|M(S_a)-M(S_b)| / mean(M(S_a),M(S_b)) <= 0.05`
4. `m_h` is computed once on calibration split and cannot be recomputed after outcome inspection.
5. Any unmatched analysis is sensitivity-only and cannot drive primary verdict.
6. Precompute and freeze eligible matched head-set pools before evaluating outcomes.

Primary comparison structure (frozen):
1. Build matched triplets of ablation sets (`within`, `mixed`, `random`) so the 5% magnitude tolerance is satisfied pairwise for all three set pairs in the triplet.
2. Primary inference uses only triplet-matched sets.
3. `ranked-R²` reference curve is exempt from strict magnitude matching and is interpretation-only (not part of primary matched inference family).

Matched-budget definition for isolated-cluster contribution analysis (frozen):
1. Budget target axis is perturbation magnitude, not raw head count.
2. For each ablation fraction `f`, define target budget:
- `B(f) = median_t M(S_within,t,f)` across confirmatory matched triplets `t` (equivalently matched to mixed/random by design).
3. For each cluster `c` and fraction `f`, construct isolated set `S_c,f subseteq c` minimizing `|M(S_c,f)-B(f)|`.
  - deterministic construction rule:
    - enumerate admissible subsets via monotone greedy prefixes over heads sorted by descending `m_h` (tie-break `layer`, then `head` ascending).
    - if two subsets have equal `|M-B|`, choose subset with head count closest to target count at fraction `f`; remaining ties broken by lexical head-ID order.
4. Confirmatory feasibility at `(c,f)` requires:
- `|M(S_c,f)-B(f)| / max(B(f), 1e-8) <= 0.05`.
5. If no subset meets 5% tolerance:
- use nearest subset for sensitivity-only output, and mark `(c,f)` non-confirmatory.
6. Cluster `c` is contribution-confirmatory-eligible only if >=80% of confirmatory fraction points are budget-feasible.
7. Contribution AUC for confirmatory criteria uses only the common fraction grid shared by all contribution-confirmatory-eligible clusters; if this common grid has <8 fraction points, contribution analysis is non-informative.

Calibration/evaluation split rule:
1. Calibration corpus used to estimate `m_h` must be sequence-disjoint from evaluation corpus used for degradation curves.
2. Any overlap >0 by sequence ID forces `analysis_tier=exploratory` for that run and must be disclosed in `matching_diagnostics.json`.
3. Calibration coordination with B4:
- default to shared calibration split (`calibration_v1`) for `m_h` (B3) and `E_h` (B4) to maximize reproducibility.
- if separate calibration splits are used, prereg must record both split IDs and overlap ratio.
- regardless of sharing/separation, each calibration split must remain sequence-disjoint from its experiment's evaluation corpus.

### B3.3 Endpoints
1. Degradation curve shape per condition.
2. Threshold vs linear fit preference per condition.
3. Relative collapse point and slope changes.
4. Layer-match quality diagnostics (must pass for primary analysis).

Layer-match diagnostic pass rule (frozen):
1. For each matched triplet (`within`, `mixed`, `random`), compare layer distributions pairwise.
2. Primary pass criteria:
- total variation distance between layer histograms <= 0.10 for each pair, and
- chi-squared test of distribution difference has `p > 0.10` for each pair.
3. Small-count fallback:
- if expected counts violate chi-squared assumptions, use a permutation-based distribution-equality test and require `p > 0.10`.
4. If either criterion fails, that triplet is exploratory-only for B3 primary inference.

Model specification and fit protocol for B3 (frozen):
1. Full confirmatory B3 run must use >=11 ablation fractions spanning `[0.0, 1.0]`.
2. Linear model:
- `y(f) = a + b * f`.
3. Threshold model (continuous piecewise linear / hockey-stick):
- `y(f) = a + b1 * f + b2 * max(0, f - tau)`.
- breakpoint `tau` constrained continuously to interior fraction range (`0.1 <= tau <= 0.9`).
4. Fitting method:
- hybrid optimization:
  - coarse grid initialization over tested interior fractions to find stable seeds,
  - bounded continuous refinement of `tau` on `[0.1, 0.9]` (holding OLS solve for `(a, b1, b2)` at each optimizer step),
  - retain global minimum-loss solution.
5. Model selection metric:
- use `AICc` (not AIC) with parameter counts `k_linear=2`, `k_threshold=4` (`tau` counted as fitted parameter).
6. Collapse-signal definition:
- primary `AUC_drop` is normalized area-under-drop over `[0.0, 1.0]`.
- secondary diagnostic `AUC_drop_early` reported over `[0.0, 0.5]`.
7. Window-sufficiency diagnostic:
- if fitted `f_c > 0.45`, mark `early_window_may_miss_collapse=true` and require reporting both full-range and early-range AUC summaries.
8. If confirmatory ablation-fraction count is <11, B3 model-selection results are exploratory-only.

Within-vs-mixed inferential test for B3 group-structure criterion (frozen):
1. For each reliability-eligible primary model, compute matched-triplet paired differences:
- `d_t = AUC_drop(within, t) - AUC_drop(mixed, t)` for each exact layer-matched triplet `t`.
2. Primary test statistic:
- one-sided paired permutation sign-flip test on `{d_t}` for `H0: mean(d_t) <= 0`, alternative `mean(d_t) > 0`.
3. Monte Carlo settings:
- default `n_perm = 10000`; if unique sign-flip states are fewer, enumerate exactly.
4. Reported uncertainty:
- paired bootstrap BCa 95% CI for `mean(d_t)` with `n_boot = 5000`.
5. Model-level p-values from this test are the inputs to the B3 Holm family.

### B3.4 Acceptance Criteria
`B3_redundancy_supported` if:
1. Threshold preference persists across random and mixed controls:
- threshold model beats linear model with `DeltaAICc >= 6` in both controls in >=2 reliability-eligible primary models.
2. No single cluster explains most collapse signal:
- define per-cluster contribution via isolated-cluster ablations:
  - only clusters with >=5 heads and contribution-confirmatory eligibility (budget-feasible fraction coverage >=80%) are confirmatory-eligible for isolated-cluster contribution analysis.
  - clusters below 5 heads are excluded from contribution-share denominator and must be logged in `cluster_contribution_share.json`.
  - for each cluster `c`, run cluster-isolated ablation curve (only heads from `c` removed at matched budgets) and compute `AUC_drop(c)` over `[0.0,1.0]`.
  - contribution share `S_c = max(AUC_drop(c), 0) / max(sum_j max(AUC_drop(j),0), 1e-8)`.
  - if `sum_j max(AUC_drop(j),0) < 0.05` or common confirmatory fraction grid has <8 points, mark contribution analysis non-informative and route B3 to `B3_inconclusive`.
  - largest share `max_c S_c < 0.50`.
3. Concentrated vs mixed difference is not sufficient to account for threshold phenomenon:
- `|AUC_drop(within) - AUC_drop(mixed)| < 0.15`, and
- collapse-fraction shift `|f_c(within) - f_c(mixed)| <= 0.10`, where `f_c` is fitted collapse fraction from threshold model.
4. Primary verdict is based on exact layer-matched analyses only.

`B3_group_structure_supported` if:
1. Concentrated within-cluster ablation causes significantly sharper collapse than mixed matched ablation:
- computed on exact layer-matched triplets only,
- Holm-corrected `p < 0.05` from the preregistered paired sign-flip test,
- and `AUC_drop(within) - AUC_drop(mixed) >= 0.15` in >=2 reliability-eligible primary models.
2. Collapse-location shift criterion:
- `f_c(mixed) - f_c(within) >= 0.10` in >=2 reliability-eligible primary models, using exact layer-matched triplets.
3. Layer-matching primacy is mandatory:
- if either criterion (1) or (2) is computed from unmatched sets, that estimate is sensitivity-only and cannot support `B3_group_structure_supported`.

Holm family for B3_group_structure inferential tests (frozen):
1. Holm correction is applied jointly across per-model `within vs mixed` comparisons in reliability-eligible primary models.

`B3_inconclusive` if:
1. neither `B3_redundancy_supported` nor `B3_group_structure_supported` is true, or
2. confirmatory preconditions for model-selection are not met (e.g., <11 ablation fractions), or
3. fewer than 2 reliability-eligible primary models are available for B3 confirmatory criteria, or
4. fewer than 2 clusters are confirmatory-eligible for contribution-share analysis, or
5. per-model mechanism verdicts are heterogeneous across reliability-eligible primary models:
- at least one model-level verdict supports redundancy and at least one supports group-structure.
- in this case set `B3_inconclusive_reason = model_conditional_mechanism`.

Per-model B3 verdict derivation (frozen; required for heterogeneity reporting):
1. `model_level_redundancy_supported=true` iff all hold within that model:
- threshold beats linear with `DeltaAICc >= 6` in both mixed and random controls,
- `max_c S_c < 0.50` (informative contribution analysis required),
- `|AUC_drop(within)-AUC_drop(mixed)| < 0.15`,
- `|f_c(within)-f_c(mixed)| <= 0.10`,
- all metrics computed on exact layer-matched confirmatory sets.
2. `model_level_group_structure_supported=true` iff all hold within that model:
- one-sided paired sign-flip test (`within > mixed`) Holm-corrected `p < 0.05` using the B3 group-structure family,
- `AUC_drop(within)-AUC_drop(mixed) >= 0.15`,
- `f_c(mixed)-f_c(within) >= 0.10`,
- all metrics computed on exact layer-matched confirmatory sets.
3. Model-level verdict assignment:
- `redundancy` if redundancy=true and group_structure=false,
- `group_structure` if group_structure=true and redundancy=false,
- `neither` otherwise.
4. Cross-model B3 support criteria remain unchanged and continue to use preregistered Holm-corrected thresholds where specified.

### B3.5 Required Outputs
1. `cluster_ablation_curve.parquet`
2. `curve_fit_by_condition.json`
3. `collapse_point_comparison.json`
4. `mechanism_disambiguation_verdict.json`
5. `B3_claim_impact.json`
6. `matching_diagnostics.json`
7. `cluster_contribution_share.json`
8. `b3_model_spec_and_fit_report.json`

Verdict schema requirement:
1. `mechanism_disambiguation_verdict.json` must contain exactly one of:
- `B3_redundancy_supported`
- `B3_group_structure_supported`
- `B3_inconclusive`
2. Required metadata fields:
- `B3_inconclusive_reason` (one of: `insufficient_eligible_models`, `insufficient_clusters`, `noninformative_contribution`, `model_conditional_mechanism`, `other`)
- `per_model_mechanism_breakdown` with one row per reliability-eligible model:
  - `model_name`
  - `model_level_redundancy_supported` (bool)
  - `model_level_group_structure_supported` (bool)
  - `model_level_verdict` (`redundancy` / `group_structure` / `neither`)
3. If `B3_inconclusive_reason == model_conditional_mechanism`, paper/report must include a per-model mechanism breakdown table in the main results section.

## B4: Confound-Isolated High-vs-Low Comparison (Strongly Recommended)

### B4.1 Question
Do high-vs-low SI effects persist after controlling layer depth and attention entropy?

### B4.2 Required Comparisons
1. within-layer high-vs-low SI differential
2. entropy-matched high-vs-low differential
3. (optional) function-class stratified SI differential

Entropy definition for matching (frozen):
1. For each head/query, compute normalized Shannon attention entropy:
- `H(h,s,i) = -sum_{j<=i} a_{h,s,i,j} log(a_{h,s,i,j}) / log(i)` for `i>=2`
2. Head-level entropy score:
- `E_h = median_{s,i}(H(h,s,i))` on held-out calibration corpus
3. Entropy-matched comparison requires:
- exact layer match
- nearest-neighbor matching on `E_h` with tolerance `|delta E_h| <= 0.02`
- one-to-one matching without replacement (many-to-one disallowed)
- deterministic tie-break order: smallest `|delta E_h|`, then low-SI head ID lexical order
- unmatched pairs are excluded and exclusion count must be reported.
4. Minimum matched-pair count for confirmatory entropy-matched analysis:
- a model is entropy-confirmatory-eligible only if matched pairs >= 5.
- models below this threshold are exploratory-only for entropy-matched inference.

### B4.3 Acceptance Criteria
`B4_supported` if all hold:
1. Direction retention:
- within-layer contrasts must have the same sign as the unadjusted high-vs-low contrast in each reliability-eligible primary model.
- entropy-matched sign retention applies only to entropy-confirmatory-eligible models.
2. Statistical retention:
- Holm-corrected `p < 0.05` for within-layer and entropy-matched contrasts in at least 2 reliability-eligible primary models.
- Holm family is defined jointly across all confirmatory B4 tests:
  - `{contrast_type in [within-layer, entropy-matched]} x {reliability-eligible primary models}`.
3. Practical magnitude retention:
- absolute Cohen's d >= 0.20 in both confound-controlled contrasts, and
- each confound-controlled effect is at least 50% of the corresponding unadjusted effect magnitude.
4. Entropy-eligibility edge case:
- if fewer than 2 reliability-eligible primary models are entropy-confirmatory-eligible, full `B4_supported` cannot be asserted and B4 is `mixed` (within-layer confirmatory, entropy component exploratory/caveated).

Outputs:
1. `within_layer_contrast.parquet`
2. `entropy_matched_contrast.parquet`
3. `B4_claim_impact.json`

## Narrative B Completion Gate
Narrative B has two preregistered completion tiers:
1. `B_full` (strong mechanistic claim): B1, B2a, B2b supported + B3 clear disambiguation verdict.
2. `B_partial` (restricted mechanistic claim): B1 and B2a supported, B2b null/mixed allowed, and B3 reported.

Claim-scope rules:
1. `clear disambiguation verdict` means exactly one of `B3_redundancy_supported` or `B3_group_structure_supported` is true.
2. `B3_inconclusive` (or both booleans true due unexpected overlap) disqualifies `B_full` and routes to `B_partial`.
3. `B_full`: main text may claim kernel families plus cluster-level enrichment and mechanism disambiguation.
4. `B_partial`: main text may claim stable kernel families and head-level kernel-boundary alignment only; cluster-level enrichment claims are disallowed.
5. If B4 is run and fails, SI-vs-low comparative language must be downgraded to caveated.

Expected paper posture after B:
1. Mechanism-level improvement over pure characterization.
2. Stronger defense against core reviewer critiques.

---

## 5) Narrative C Pipeline: Two-Carrier-Class Theory (High Risk, High Reward)

Narrative C should only start after B1/B2 are at least partially positive.

## C0: Preconditions
Proceed only if:
1. B1 shows interpretable kernel families or clear model-conditional kernel structure.
2. B2a shows non-trivial head-level boundary alignment signal.
3. B2b is optional for entering C-path, but if B2b is null/mixed then C claims cannot include cluster-level boundary enrichment.

If prerequisites fail, defer C and submit B/A narrative.

## C1: OLMo Positional Mechanism Characterization via Causal Tracing (Required for C)

### C1.1 Question
If OLMo has weak SI signal, which circuits carry position-sensitive computation?

### C1.2 Dataset Requirements
Construct position-diagnostic suites where answer changes if and only if position changes, holding token content fixed:
1. copy-offset tasks (`token at t-N` retrieval).
2. indexed retrieval tasks (`what is token at position k`).
3. counting/ordinal tasks over fixed token sets (exploratory-only in C1; excluded from confirmatory C1 composite endpoint).
4. controlled permutation templates (`same tokens, different order, fixed queried index`).

Naturalistic linguistic suites (agreement/pronoun/order) may be included as secondary exploratory tests only.

Minimum coverage target:
1. Confirmatory-eligible phenomena only (`copy-offset`, `indexed retrieval`, `controlled permutation templates`) must each have >=2000 examples per model for confirmatory pass.
2. Any deviation below 2000 per confirmatory phenomenon forces `analysis_tier=exploratory` for C1 confirmatory claims.
3. Held-out evaluation set for final reporting.

### C1.3 Intervention Protocol
1. Clean run, corrupted run, patch run.
2. Corruption family is frozen and position-targeted:
- primary confirmatory corruption: context-order permutation corruption.
- for each example, keep token multiset fixed, permute pre-query token order with a seeded permutation, and keep query text unchanged.
- corruption must preserve token identities and counts; only token-to-position assignment changes.
3. Confirmatory eligibility by task type:
- include only task instances where permutation corruption changes the ground-truth answer.
- counting/ordinal tasks are excluded from confirmatory C1 endpoints under permutation corruption.
4. Counting/ordinal auxiliary corruption (exploratory only):
- token-identity substitution corruption preserving length/format while changing target-relevant token counts.
- cannot contribute to `C1_two_carrier_class_supported`.
5. Optional secondary corruption (exploratory only): direct positional-encoding perturbation where supported by model implementation.
6. Embedding-noise corruption is disallowed for confirmatory C1 endpoints.
7. Patch at candidate head outputs and/or attention patterns.
8. Compute restoration metrics relative to clean-corrupted gap.

### C1.4 Candidate Head Sets
At minimum compare:
1. high-SI heads
2. low-SI heads
3. content-conditional candidate sets (predefined, not post-hoc)

Content-conditional candidate set must be frozen before C1 runs. Required operational definition:
1. `mean_r2 < model_median_r2`
2. `attention_entropy > layer_median_entropy`
3. `content_similarity_alignment > preregistered threshold T`, where:
- token-content similarity matrix `C_s(i,j)` is defined from frozen external lexical embeddings:
  - embedding source: fastText `cc.en.300` (or model-language-equivalent frozen static embedding), never model-internal activations
  - token text preprocessing: strip tokenizer boundary markers, lowercase, keep alphanumeric core
  - token embedding `e(t_i)` = fastText vector of preprocessed token text (subword fallback handled by fastText)
  - `C_s(i,j) = cosine(e(t_i), e(t_j))`, diagonal set to 0
- head attention matrix for alignment uses raw causal attention weights `A_h,s(i,j)` with `j<=i`
- vectorization domain is causal off-diagonal entries only (`j<i`), excluding BOS/pad positions
- compute partial Spearman correlation between `vec(A_h,s)` and `vec(C_s)` controlling for relative-offset bins
- `content_similarity_alignment(h)` = median partial correlation across calibration sequences
- `T` = 75th percentile of `content_similarity_alignment(h)` among low-R² heads in that model on calibration split
4. set construction cannot use C1 restoration outcomes
5. if required external embeddings are missing for >5% tokens after preprocessing, C1 must halt and emit `content_similarity_data_quality_failure=true`.

### C1.5 Endpoints
1. Restoration ratio by head-set and model.
2. Cross-model interaction of head-set utility.
3. Layer profile of restoration.
4. Separate confirmatory endpoints for position-diagnostic suite and exploratory endpoints for naturalistic suite.

Restoration metric definitions (frozen):
1. For each head-set `S`, `RR(S) = (A_patch(S) - A_corrupt) / max(A_clean - A_corrupt, 1e-6)`.
2. `RR(S)` is clipped to [-1, 2] before aggregation.
3. SI concentration share:
- `share_SI = max(RR(SI), 0) / sum_{K in {SI, LowSI, ContentCond}} max(RR(K), 0)`.
4. Primary per-model summary uses median `RR(S)` across confirmatory task instances.
5. Division-by-zero handling:
- if `sum_{K} max(RR(K),0) == 0`, set `share_SI = NaN`, set `all_nonpositive_rr_flag=true`, and treat SI-concentration criteria as failed for confirmatory support.
6. Minimum corruption-gap eligibility for confirmatory RR:
- include only task instances with `(A_clean - A_corrupt) >= 0.05` in confirmatory RR aggregation.
- instances below this threshold are logged and treated as exploratory-only for C1 confirmatory endpoints.
7. Uncertainty quantification:
- compute bootstrap 95% BCa confidence intervals (>=2000 bootstrap resamples over confirmatory task instances) for `RR(SI)_llama`, `RR(LowSI)_llama`, `share_SI_llama`, and required OLMo contrasts.

### C1.6 Acceptance Criteria
`C1_two_carrier_class_supported` if all hold:
1. Llama SI concentration:
- `RR(SI)_llama >= 0.30`
- `RR(SI)_llama / max(RR(LowSI)_llama, 0.05) >= 1.50`
- `share_SI_llama >= 0.50`
- uncertainty guardrail:
  - lower 95% BCa CI bound for `RR(SI)_llama` >= 0.20
  - lower 95% BCa CI bound for `RR(SI)_llama / max(RR(LowSI)_llama, 0.05)` >= 1.20
  - lower 95% BCa CI bound for `share_SI_llama` >= 0.40
2. OLMo non-SI concentration with alternative recovery:
- non-SI concentration guard (piecewise):
  - if `RR(LowSI)_olmo >= 0.05`: require `RR(SI)_olmo / RR(LowSI)_olmo <= 1.20`
  - if `RR(LowSI)_olmo < 0.05`: require `RR(SI)_olmo - RR(LowSI)_olmo <= 0.20`
- `RR(ContentCond)_olmo >= 0.20`
- alternative-recovery guard (piecewise):
  - if `RR(LowSI)_olmo >= 0.05`: require `RR(ContentCond)_olmo / RR(LowSI)_olmo >= 1.50`
  - if `RR(LowSI)_olmo < 0.05`: require `RR(ContentCond)_olmo - RR(LowSI)_olmo >= 0.10`
- Holm-corrected `p < 0.05` for `ContentCond > LowSI` in OLMo.
3. Cross-model interaction:
- model x head-set interaction Holm-corrected `p < 0.05`
- practical threshold `|DeltaDeltaRR| >= 0.10`, where
  - `DeltaDeltaRR = (RR(SI)-RR(ContentCond))_llama - (RR(SI)-RR(ContentCond))_olmo`.

Multiplicity family for C1 corrected tests (frozen):
1. All confirmatory C1 inferential tests share one Holm family:
- OLMo `ContentCond > LowSI`
- model x head-set interaction term
- any additional preregistered confirmatory head-set contrasts used for acceptance
2. Exploratory contrasts are excluded from this confirmatory Holm family.

If ambiguous:
- Narrative C is downgraded; retain B/A.

### C1.7 Required Outputs
1. `causal_trace_matrix.parquet`
2. `restoration_ratio_summary.json`
3. `head_set_interaction_model.json`
4. `C1_claim_impact.json`

## C2: Carrier-Class-Conditional Prediction Test (Required for C)

### C2.1 Prediction 1: Position-Shuffle Robustness
Hypothesis:
- Higher SI-R² models are more sensitive to order-shuffling perturbations.

Requirements:
1. Matched content with shuffled positions (multiple perturbation strengths).
2. Model-count target:
- target `n_models >= 4`.
- minimum executable set is `n_models = 3`; in that case cross-model conclusions are restricted per the `n_models < 5` scope rules below.
- if `n_models < 3`, C2.1 is `deferred` for confirmatory claims.
3. Correlate degradation with mean SI-R².
4. Confound-aware analysis plan (frozen):
- primary confirmatory analysis: within-model shuffle sensitivity slopes, meta-analyzed across models.
- secondary cross-model analysis: association between mean SI-R² and shuffle sensitivity.
- if `n_models >= 5`, secondary analysis must include covariate adjustment for log parameter count and architecture family.
- if `n_models < 5`, covariate-adjusted cross-model regression is exploratory-only and cannot support confirmatory claims.

Shuffle slope definition (frozen):
1. Shuffle strength `s in [0,1]` is the fraction of pre-query tokens permuted.
2. Normalized degradation at each strength:
- `D(s) = (A_clean - A_shuffle(s)) / max(A_clean, 1e-6)`.
3. Within-model slope `beta_shuffle` is the OLS coefficient from:
- `D(s) = alpha + beta_shuffle * s` across preregistered shuffle strengths (including `s=0` baseline).
4. Meta-analysis uses random-effects pooling on `beta_shuffle` across models.
5. Nonlinearity sensitivity check:
- report normalized AUC of `D(s)` curve; confirmatory verdict remains tied to `beta_shuffle`.

Acceptance:
1. Directional consistency and practical slope threshold:
- if `n_models >= 5`: pooled `beta_shuffle > 0` with correction-adjusted `p < 0.05` and pooled `beta_shuffle >= 0.10` are required confirmatory criteria.
- if `n_models < 5`: pooled criterion is descriptive/exploratory; confirmatory decision uses the within-model rule in item 3.
  - unadjusted within-model Holm family (frozen): `{beta_shuffle > 0 tests across reliability-eligible primary models}`.
2. Robustness to nuisance controls (frozen):
- nuisance-adjusted within-model slope model:
  - `D(s) = alpha + beta_shuffle_adj * s + gamma_len * length_z + gamma_diff * difficulty_z`.
- where `length_z` is z-scored sequence length and `difficulty_z` is z-scored clean-run difficulty proxy.
- primary difficulty proxy is `1 - A_clean`; if `A_clean` is unavailable at per-instance resolution, use z-scored clean-run negative log-likelihood (or equivalent monotone confidence surrogate) and record `difficulty_proxy_used`.
- confirmatory robustness pass requires:
  - at least 2 reliability-eligible primary models with `beta_shuffle_adj > 0` and Holm-corrected within-model `p < 0.05` (Holm family: adjusted-slope tests across reliability-eligible primary models), and
  - pooled adjusted slope retains >=70% of pooled unadjusted magnitude (`beta_pooled_adj / max(beta_pooled_unadj,1e-8) >= 0.70`).
3. Confirmatory claim scope:
- with `n_models < 5`, restrict confirmatory wording to within-model directional consistency, defined as:
  - at least `ceil(2/3 * n_reliability_eligible_primary_models)` models with `beta_shuffle > 0`,
  - and at least 2 reliability-eligible primary models with `beta_shuffle >= 0.10` and Holm-corrected within-model `p < 0.05`.
- with `n_models >= 5`, require covariate-adjusted SI term to retain directional/practical support for cross-model wording, defined as:
  - adjusted SI coefficient `beta_SI_adj > 0`,
  - Holm-corrected `p < 0.05` for SI term in adjusted model,
  - and semi-partial `R^2` for SI term >= 0.05.

### C2.2 Prediction 2: Long-Context Generalization
Hypothesis:
- SI-dominant carrier class generalizes better to unseen larger offsets.

Requirements:
1. controlled long-context battery beyond training-like lengths using model-specific thresholds:
- define `L_trainlike(model)` as published pretraining context length when available; otherwise freeze a documented surrogate in prereg.
- test context grid as multiples of `L_trainlike`: `{0.5x, 1.0x, 1.25x}` (capped at model maximum context).
- add exploratory stress point `2.0x` when model maximum context allows; otherwise log as not feasible.
- a model is confirmatory-eligible for C2.2 only if at least one tested point is `> 1.0x L_trainlike`.
- OLMo surrogate rule (frozen):
  - for `OLMo-2-7B`, set `L_trainlike = 2048` tokens for C2.2 scaling.
- confirmatory interaction eligibility:
  - C2.2 confirmatory interaction analysis uses only confirmatory-eligible models.
  - if fewer than 2 models are confirmatory-eligible, C2.2 is `deferred` for confirmatory interaction claims.
2. matched token/content distributions across context windows
3. preregister auxiliary diagnostics to disambiguate failure modes:
- SI-support diagnostic: fraction of kernel mass within tested long offsets
- content-degradation diagnostic: content-similarity alignment vs context length slope

Acceptance:
1. model-by-context interaction consistent with preregistered direction:
- confirmatory interaction model:
  - per-model regression at instance level:
    - `D = alpha + beta_ctx * context_mult + beta_class * carrier_class + beta_int * context_mult*carrier_class`.
  - pooled confirmatory test uses random-effects meta-analysis over model-level `beta_int`.
  - interaction test family (frozen) has two confirmatory tests:
    - pooled `beta_int` directional test,
    - endpoint carrier-class contrast test at target long context point.
  - apply Holm across these two tests; criterion 1 passes if pooled `beta_int` remains Holm-corrected `p < 0.05` in preregistered direction.
2. practical effect threshold met:
- absolute difference in normalized long-context drop between carrier classes at `1.25x` point >= 0.05 (or, if `1.25x` is not feasible for a model due context cap, use the largest tested point `>1.0x`), or
- standardized interaction effect size (partial eta squared) >= 0.01.
3. Failure interpretation must be assigned to one of:
- `carrier_class_inconsistent`
- `kernel_support_limited`
- `content_alignment_degrades`
- `ambiguous_failure`

Failure-mode assignment rules (frozen):
1. Compute diagnostics at the longest confirmatory-tested context point (>1.0x):
- `K_support` = SI-support diagnostic (fraction of kernel mass within tested long offsets).
- `S_content` = slope of content-similarity alignment vs context length.
2. Assign `carrier_class_inconsistent` if:
- interaction direction is opposite preregistered expectation with correction-adjusted `p < 0.05`.
3. Else assign `kernel_support_limited` if:
- `K_support < 0.20` and `S_content > -0.02`.
4. Else assign `content_alignment_degrades` if:
- `K_support >= 0.20` and `S_content <= -0.05`.
5. Else assign `ambiguous_failure`.

### C2.3 Required Outputs
1. `position_shuffle_results.parquet`
2. `long_context_results.parquet`
3. `carrier_class_prediction_tests.json`
4. `C2_claim_impact.json`

## C3: Optional Strong External Validity Probe (>8B Reduced Battery)

Purpose:
- Stress-test carrier-class framing at larger scale.

Recommended reduced battery:
1. T8-like disruption-sensitive test
2. strict boundary non-triviality test
3. C1-style threshold-capacity sanity test

Caveat:
- Treat as external validity probe only, not central evidence unless clean.

## Narrative C Completion Gate
Narrative C is eligible only if:
1. C1 and C2 both pass.
2. Model carrier-class distinction is explicit and robust.
3. C claims are framed as "evidence for distinct position-information carrier classes across models" and not causal attribution of why models differ.

Expected paper posture after C:
1. Highest novelty and strongest NeurIPS upside.
2. Highest implementation and interpretation risk.

---

## 6) Cross-Path Decision Logic (Operational)

Decision ladder:
1. Always run A0/A1/A2 first.
2. Run B2a head-level analysis first (no B1 dependency).
3. Run B1 kernel taxonomy.
4. Run B2b cluster-level enrichment (depends on B1).
5. If either B1 or B2a is weak -> submit A narrative.
6. If B1 and B2a are strong -> run B3 and finalize B tier (`B_full` vs `B_partial`) using B3 verdict rules; B2b may still be null/mixed in `B_partial`.
7. Only if B1/B2a/B3 coherent and time remains -> attempt C1/C2.

Hard stop rules:
1. If B3 fails to disambiguate mechanism, do not force mechanistic overclaim.
2. If `B3_inconclusive`, use threshold-capacity language and route to `B_partial`.
3. If C1 is diffuse/ambiguous, do not force two-carrier-class headline.

---

## 7) Compute Budget and Scheduling

Approximate incremental costs (beyond already-running reinforcement work):
1. A-path tasks: 0 GPU-hours
2. B1: 4-8 GPU-hours
3. B2a: 0 GPU-hours (analysis)
4. B2b: 0 GPU-hours (analysis)
5. B3: 14-24 GPU-hours (includes isolated-cluster contribution curves)
6. B4: 6-10 GPU-hours
7. C1: 16-24 GPU-hours
8. C2: 4-8 GPU-hours
9. C3 optional: 10-20 GPU-hours

Budgeted plans:
1. Minimal robust plan (recommended): A + B2a + B1 + B2b + B3 = ~18-32 GPU-hours
2. Strong plan: A + B2a + B1 + B2b + B3 + B4 = ~24-42 GPU-hours
3. Maximal plan: Strong plan + C1 + C2 = ~44-74 GPU-hours

---

## 8) File/Artifact Layout for reinforce_exp2

Create this layout:

1. `reinforce_exp2/TODO.md` (this file)
2. `reinforce_exp2/README.md` (execution quickstart)
3. `reinforce_exp2/schemas/` (JSON schema files)
4. `reinforce_exp2/scripts/` (launch and validation scripts)
5. `results/reinforce_exp2/A0_claim_hygiene/...`
6. `results/reinforce_exp2/B1_kernel_taxonomy/...`
7. `results/reinforce_exp2/B2a_head_alignment/...`
8. `results/reinforce_exp2/B2b_cluster_alignment/...`
9. `results/reinforce_exp2/B3_cluster_ablation/...`
10. `results/reinforce_exp2/B4_confound_isolation/...`
11. `results/reinforce_exp2/C1_olmo_causal_trace/...`
12. `results/reinforce_exp2/C2_carrier_class_predictions/...`

---

## 9) Experimental Rigor Checklist (Must Pass Before Claim Promotion)

For every experiment:
1. prereg exists and matches executed command.
2. seed plan executed exactly.
3. exclusions logged and justified.
4. multiplicity correction applied as preregistered.
5. practical-effect threshold checked.
6. strict/exploratory flag present.
7. claim-impact file generated.
8. negative findings documented (not suppressed).

For main-text eligibility:
1. result is confirmatory.
2. correction-adjusted significance passes.
3. practical threshold passes.
4. cross-check control does not invalidate interpretation.

## 9.1 Mandatory Smoke Tests for High-Cost New Pipelines

### B3 Smoke Test (required before full B3)
Run:
1. 1 model
2. 3 conditions (`within_cluster`, `mixed_cluster`, `random_cluster`)
3. 5 ablation fractions
4. 1 seed
5. compute `m_h` on a small calibration split and run degradation curves on a sequence-disjoint evaluation split.
6. run a mini model-fit smoke branch with >=11 fractions on a tiny sample to validate AICc threshold-vs-linear code path.
7. run at least 1 confirmatory-eligible isolated-cluster ablation curve using matched-budget logic.
8. run a contribution-confirmatory branch with >=8 ablation fractions to validate the common-grid confirmatory contribution pathway.

Validate:
1. selected head sets exactly satisfy matching constraints.
2. curve outputs are produced for all three conditions.
3. matching diagnostics file is non-empty and schema-valid.
4. `m_h` artifact exists and is derived only from calibration split.
5. calibration/evaluation disjointness check passes by sequence ID.
6. AICc model-fit report is produced and includes both linear and threshold fits without runtime errors.
7. isolated-cluster pipeline emits `cluster_contribution_share.json` with non-empty budget-feasibility diagnostics.
8. confirmatory contribution branch proves common-grid logic with >=8 valid fraction points (or emits explicit non-informative route with correct flagging).

### C1 Smoke Test (required before full C1)
Run:
1. 1 model (`OLMo-2-7B`, the target model for C1)
2. 50 examples on position-diagnostic suite
3. all preregistered head sets

Validate:
1. clean/corrupt/patch pipeline runs end-to-end.
2. restoration ratio is computed and bounded.
3. head-set labeling and provenance fields are correct.
4. content-conditional set construction pipeline (C1.4) executes successfully, including fastText availability and calibration split reads.

No full B3 or C1 launch is allowed before smoke tests pass and artifacts are reviewed.

---

## 10) Paper Integration Rules by Narrative

## 10.1 If only A is complete
Main paper should present:
1. disciplined characterization claims
2. explicit caveats on mechanism and generalization
3. prediction section as future test

## 10.2 If B is complete
Main paper should present:
1. kernel taxonomy as primary mechanistic unit
2. boundary linkage through offset alignment
3. redundancy interpretation bounded by B3 verdict

## 10.3 If C is complete
Main paper should present:
1. two operational carrier-class framing with explicit evidence boundaries
2. carrier-class-conditional prediction tests
3. conservative attribution language (empirical heterogeneity, no matched-training causal attribution)

---

## 11) Risk Register and Mitigations

Risk 1: Kernel clusters are unstable.
1. Mitigation: run multi-method clustering; downgrade to exploratory map if unstable.

Risk 2: Alignment effect in B2 is weak or null.
1. Mitigation: keep characterization framing; avoid mechanism headline.

Risk 3: B3 inconclusive between redundancy and cluster alternatives.
1. Mitigation: use "threshold-capacity organization" language only.

Risk 4: C1 tracing is diffuse and non-localizable.
1. Mitigation: stop C-path and avoid two-carrier-class claim.

Risk 5: Multiple comparisons inflate nominal significance.
1. Mitigation: strict family definitions and corrected reporting tables only.

Risk 6: Overrun in GPU budget/time.
1. Mitigation: preserve order A -> B2a -> B1 -> B2b -> B3; C only if runway remains.

Risk 7: C-path language overreaches causal attribution across unmatched models.
1. Mitigation: enforce carrier-class heterogeneity wording and forbid causal-attribution language without matched-training evidence.

Risk 8: Expensive runs fail due untested new code paths.
1. Mitigation: mandatory B3 and C1 smoke tests before full runs.

Risk 9: B2b confirmatory Holm family is too large, reducing power.
1. Mitigation: cap confirmatory cluster family to at most 4 preregistered clusters (size-ranked), keep remaining clusters exploratory, and report confirmatory vs exploratory family sizes explicitly.

---

## 12) Definition of Done

A pipeline is done only if:
1. all required experiments in that pipeline meet acceptance criteria or are explicitly failed with documented scope reduction.
2. claim matrix updated with status labels.
3. paper text updated to match claim matrix.
4. reproducibility and contradiction checks pass.

Project-level done for submission:
1. At least A complete.
2. Preferably B complete for stronger odds.
3. C attempted only if evidence is clean and schedule allows.

---

## 13) Immediate Execution Order (Recommended)

1. Complete A0/A1/A2 documentation pass now.
2. Launch B2a head-level alignment from existing kernels immediately (no GPU).
3. Launch B1 kernel taxonomy.
4. Run B2b cluster enrichment after B1 outputs exist.
5. Implement and pass B3 smoke test, then run full B3.
6. Re-evaluate submission narrative.
7. If entering C-path: implement and pass C1 smoke test before full C1/C2.

---

## 14) Minimum Required Tables for Final Paper (regardless of path)

1. Claim robustness matrix with status labels and caveat column.
2. Cross-model summary table (Llama/OLMo/Mistral) for core batteries.
3. Strict-vs-exploratory provenance table.
4. Open limitations table with unresolved weaknesses and planned future tests.

---

## 15) Notes on Interpretation Discipline

1. A statistically significant disruption after post-hoc component subtraction is not, by itself, mechanism proof.
2. Correlation between kernel shape and boundary profile is evidence for linkage, not complete causal mechanism.
3. Threshold curves can admit multiple mechanisms; language must reflect what was actually ruled out.
4. Model differences should be reported as empirical heterogeneity unless matched-training causal tests exist.
