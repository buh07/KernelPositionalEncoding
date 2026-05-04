# reinforce_exp3 — Experiment TODO and Design Specifications

**Created:** April 29, 2026  
**Purpose:** Enumerate, justify, and fully specify all remaining experiments needed before NeurIPS 2026 submission. Each entry explains the weakness being addressed, the precise experimental design, pre-specified success/failure criteria, and expected artifacts.

Experiments are ordered by priority tier. The first three (**E0, E3, E6**) are must-haves that address vulnerabilities a reviewer is likely to identify immediately. The next four (**E2, E8, E9, E10**) are quick additions that meaningfully strengthen existing claims. The final five (**E4, E5, E7, E11, E12**) are valuable but time-intensive and should be pursued if the submission timeline allows.

All experiments should produce a `claim_impact.json` and `manifest.json` at their result root, consistent with the reinforce_exp and reinforce_exp2 artifact schema.

---

## Table of Contents

| ID | Name | Priority | Estimated effort |
|----|------|----------|-----------------|
| E0 | Cluster-Sequential vs. Interleaved Ablation | Must Have | Medium (3–5 days) |
| E3 | Continuous-Metric Variants for Synthetic Tasks | Must Have | Medium (2–4 days) |
| E6 | Llama Strict-Boundary Control Expansion | Must Have | Medium (3–5 days) |
| E2 | Random-Order Sensitivity Beyond R12 Grid | Helpful / Quick | Short (1–2 days) |
| E8 | Tokenizer-Overlap SI Comparison | Helpful / Quick | Short (1–2 days) |
| E9 | Layer-Head SI Heatmaps | Helpful / Quick | Very short (< 1 day) |
| E10 | SI vs. Retrieval-Head Overlap | Helpful / Quick | Short (1–2 days) |
| E4 | Checkpoint SI-Emergence Trajectory | Helpful / Long | Long (1–2 weeks) |
| E5 | Additional 7–9B Model Family | Helpful / Long | Long (2–3 weeks) |
| E7 | Non-English Boundary Domain | Helpful / Long | Medium-Long (1 week) |
| E11 | High-SI RoPE-Frequency Decomposition | Helpful / Long | Long (1 week+) |
| E12 | ICL Sensitivity Under SI-Kernel Subtraction | Helpful / Long | Long (1 week+) |

---

---

# TIER 1 — MUST HAVE

---

## E0 — Cluster-Sequential vs. Interleaved Ablation

**Result addressed:** Result III (Distributed Redundancy)  
**Weakness addressed:** Critique Weakness 3 — "Distributed Redundancy Is Consistent with Multiple Incompatible Mechanisms"  
**Artifact root:** `results/reinforce_exp3/E0_cluster_sequential_ablation/`

### Background and Motivation

Result III's central claim — that SI heads form a collectively redundant positional resource — rests on the unanimous 0/18 linear BIC vote and the 26/30 random-order threshold-majority from NEW-R12. These results establish that cumulative ablation produces threshold-style, not linear, degradation. However, the experiments as designed cannot distinguish between three fundamentally different mechanisms that all predict identical threshold-piecewise degradation curves:

1. **Genuine distributed redundancy:** Each head carries a small, interchangeable slice of positional information. The collective pool has uniform load-bearing capacity. Degradation is flat until enough heads are removed that the remaining pool cannot cover the full positional range, causing collapse.

2. **Cluster-boundary depletion:** SI heads decompose into 3–4 distinct functional groups (clusters), each implementing a different positional computation (e.g., previous-token tracking, induction-lag attention, BOS-anchored attention). Because cumulative ablation is ranked by R², heads within the highest-R² cluster tend to be removed together. Performance remains stable while cluster members are partially depleted because surviving members compensate within the cluster. Collapse occurs abruptly when an entire cluster is exhausted. The "threshold" marks a cluster boundary, not a redundancy depletion point. This mechanism is **incompatible** with genuine redundancy but predicts **identical** BIC votes.

3. **Loss-landscape tolerance:** Removing any large coherent subset of heads perturbs the model off a flat region of the loss surface. The threshold marks when the total perturbation magnitude — not the depletion of any specific computation — exceeds a generic tolerance. Under this mechanism, any ranked removal order would show similar threshold behavior regardless of what property is used to rank heads.

The B3 cluster-ablation experiment in reinforce_exp2 was designed to test mechanism 2 but was **inconclusive** due to underpowering. The B3 design required exact layer-distribution and magnitude-matched triplets (`layer_hist_tvd_max ≤ 0.10`, `magnitude_relative_tolerance ≤ 0.05`), which produced only 9 valid triplets in Llama and OLMo and 15 in Mistral — far below the n needed to detect realistic effect sizes. Additionally, because Llama's cluster structure is heavily imbalanced (cluster 1 contains 198/256 = 77% of all high-SI heads), the "within-cluster" and "mixed-cluster" conditions in B3 were nearly identical in practice, making the test uninterpretable regardless of power.

The B3 cluster membership data (from `results/reinforce_exp2/B1_kernel_taxonomy/cluster_membership.parquet`) confirms the imbalance:

| Model | Cluster 0 high-SI | Cluster 1 high-SI | Cluster 2 high-SI | Cluster 3 high-SI |
|-------|-------------------|-------------------|-------------------|-------------------|
| Llama-3.1-8B | 44 | 198 | 14 | — |
| OLMo-2-7B | 126 | 87 | 43 | — |
| Mistral-7B-v0.1 | 13 | 50 | 192 | 1 |

OLMo has the most balanced cluster distribution among primary models (126/87/43) and is therefore the model where this experiment has the highest power.

The present experiment redesigns the cluster-structure test to avoid the matching constraints that killed B3, use a directly interpretable ordering comparison, and generate a degradation curve that can be fit with the same BIC infrastructure used in 3P2-C.

### Experimental Design

The core prediction of the cluster-boundary depletion mechanism (mechanism 2) is: **exhausting one cluster before touching another should produce a sharper, earlier collapse than interleaving ablations evenly across clusters.** If genuine redundancy is the mechanism, ordering heads by cluster (sequential) versus spreading them across clusters (interleaved) should produce identical degradation curves, because what matters is the total count removed, not the cluster composition of the removed set.

**Step 1 — Load cluster assignments.** Load `cluster_membership.parquet` for each model. Use the `cluster_descriptor_kmeans` column as the cluster label. Restrict to high-SI heads (`is_high_si == True`). This gives 256 high-SI heads per model.

**Step 2 — Construct three ablation orderings.**

- **Sequential ordering:** Remove all heads in the smallest cluster first, then the next-smallest, then the largest. Within each cluster, rank heads by R² (descending). For OLMo: remove all 43 heads in cluster 2 first, then 87 heads in cluster 1, then 126 heads in cluster 0. The intent is to exhaust complete clusters in sequence.

- **Interleaved ordering:** At each ablation step, remove one head from each cluster in round-robin fashion (rotating across clusters, ranked by R² within cluster at each step). This guarantees that all clusters are depleted proportionally at every ablation fraction. At any given fraction f, the number of removed heads from each cluster is approximately f × (cluster size).

- **R²-ranked ordering (reference):** Standard descending-R² order, identical to 3P2-C and R12. This is the comparison baseline.

**Step 3 — Evaluate degradation curves.** At ablation fractions {0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50} (finer grid near expected threshold), measure:
- LM loss on Wikipedia held-out sequences (same corpus as 3P2-C)
- Local key-match accuracy
- Long-range retrieval accuracy

Use 5 seeds per fraction per ordering. This gives ~135 evaluation runs per model (9 fractions × 3 orderings × 5 seeds).

**Step 4 — Fit degradation models.** For each ordering × task combination, fit linear and threshold-piecewise models (same BIC infrastructure as 3P2-C, `model_spec` from `B3_model_spec_and_fit_report.json`). Compare threshold locations (tau) and AUC drops across orderings.

**Step 5 — Compute the discriminating statistic.** The primary test is: does the sequential ordering produce a statistically significantly **earlier** collapse fraction (lower tau) than the interleaved ordering?

Formally: for each model, compute `tau_sequential - tau_interleaved` across task × seed cells. One-sided test: `H_A: tau_sequential < tau_interleaved` (sequential collapses earlier → cluster structure drives threshold). The null is: tau values are exchangeable across orderings. Use permutation of ordering labels within each task × model cell to construct the null distribution.

### Pre-Specified Success / Failure Criteria

**Evidence for cluster-boundary mechanism (bad for Result III):**  
`tau_sequential < tau_interleaved` with one-sided p < 0.05 (Holm-corrected across models) in ≥ 2/3 primary models, AND the AUC drop under sequential ordering is ≥ 20% larger than under interleaved ordering in the same models.

**Evidence for genuine redundancy (good for Result III):**  
`|tau_sequential - tau_interleaved| < 0.05` and one-sided p > 0.20 in all three models. This would show that cluster composition of the removed set does not predict collapse timing — only the count matters.

**Inconclusive:**  
Mixed outcomes across models or one-sided p in 0.05–0.20 range.

**What to claim in the paper:**
- If redundancy-consistent: add a sentence to Result III robustness noting that sequential vs interleaved ordering produces statistically indistinguishable collapse fractions, directly ruling out cluster-boundary depletion as the primary mechanism.
- If inconclusive or cluster-structure-consistent: demote Result III's claim from "distributed redundancy" to "threshold-style collective capacity collapse of unknown internal organization" and note this explicitly as an open question.

### Why B3 Cannot Serve This Purpose

B3 used layer/magnitude-matched triplets. The matching constraints were so strict that only 9 valid triplets per model were generated, and the within-cluster vs mixed-cluster conditions in Llama were nearly degenerate (cluster 1 has 77% of high-SI heads, so any random mixed selection is already mostly cluster 1). The redesign in E0 avoids matching entirely: the test is about ordering, not composition matching. The signal is whether exhausting clusters one at a time shifts the collapse fraction — a direct comparison of degradation curve shape, not a matched-sample comparison of AUC values.

### Expected Artifacts

```
results/reinforce_exp3/E0_cluster_sequential_ablation/
├── manifest.json
├── preregistration.json
├── claim_impact.json
├── <model>/
│   ├── sequential_degradation_curve.parquet
│   ├── interleaved_degradation_curve.parquet
│   ├── r2ranked_degradation_curve.parquet    # reference, from 3P2-C
│   ├── tau_comparison.json                   # tau_seq, tau_interl, tau_r2, test p-values
│   ├── ordering_permutation_null.parquet
│   └── curve_fit_by_ordering.json
└── cross_model_summary.json
```

---

## E3 — Continuous-Metric Variants for Synthetic Tasks

**Result addressed:** Result III (Distributed Redundancy)  
**Weakness addressed:** "Are Emergent Abilities a Mirage?" (Schaeffer et al., NeurIPS 2023) — the metric-artifact objection  
**Artifact root:** `results/reinforce_exp3/E3_continuous_metric_variants/`

### Background and Motivation

Result III's threshold-piecewise degradation is the single strongest claim in the paper. However, Schaeffer et al. (NeurIPS 2023) — a well-known and frequently cited result — showed that many apparent emergence and threshold phenomena in large language models vanish when accuracy-style metrics are replaced by continuous probability-mass or log-likelihood metrics. The argument is that accuracy metrics introduce a threshold artificially because they discretize a smooth underlying performance function: a model can be improving continuously in probability assignment while appearing flat in accuracy until the probability crosses 0.5.

The current paper includes LM loss (a continuous metric) as one of three tasks in 3P2-C, and the threshold preference holds for LM loss in all three models. This is a meaningful defense. However, the two synthetic tasks — local key-match accuracy and long-range retrieval accuracy — are binary accuracy metrics. A reviewer familiar with Schaeffer et al. will note that the 12/18 threshold BIC votes include these two binary tasks, and may argue that the threshold preference in those cells is metric-induced.

The fix is to design continuous-score versions of both synthetic tasks and show that threshold-piecewise preference holds in those continuous versions as well. This preempts the objection before it is raised.

### Experimental Design

**Local key-match continuous metric:** The current task tests whether the model correctly identifies a repeated key-value token at a specific local offset. Replace the binary correct/incorrect outcome with the log-probability assigned to the correct token at the target position. This is the model's confidence in the correct answer on a continuous scale, not a threshold at p=0.5.

Formally: given a prompt of the form `[key_A] [val_A] ... [key_A]`, the task is to predict `[val_A]`. The current metric is `1(argmax == val_A)`. The continuous metric is `log P(val_A | context)` normalized by the model's baseline entropy for that token type. This normalization accounts for token frequency and is computed as `log P(val_A | context) - log P_prior(val_A)` where `P_prior` is estimated from a held-out unconditional sample.

**Long-range retrieval continuous metric:** The current task tests whether the model correctly retrieves a value from a long-context key-value pair. Replace the binary outcome with the rank-based continuous score: `1 - (rank_of_correct_token - 1) / vocab_size`, which is the fraction of vocabulary tokens scored lower than the correct token. This is equivalent to area under the ROC curve for the single-item retrieval task and is fully continuous.

Alternative (simpler): use the raw log-probability of the correct retrieval token, same approach as the local key-match continuous metric.

**Evaluation protocol:** Run both continuous metrics at the same ablation fractions as 3P2-C ({0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50}). Use the same R²-ranked ablation order and the same 5-seed design. Fit linear vs threshold-piecewise vs logistic using the standard BIC infrastructure. Report BIC vote tally in the same format as the existing 18-cell table.

**Models:** All three primary models (Llama, Mistral, OLMo). 6 new BIC cells per model (2 continuous tasks × 3 models), 18 new cells total. Add to the existing 18 cells to produce a 36-cell table with a split between binary and continuous tasks.

**Expected outcome:** If the threshold preference holds in both binary and continuous metrics, the 36-cell table (0/36 linear) is a decisive refutation of the metric-artifact objection. If threshold preference drops in continuous metrics, this is a critical finding that requires downgrading Result III's claim.

### Pre-Specified Success / Failure Criteria

**Success (metric-artifact objection refuted):**  
Continuous-metric BIC votes show linear: 0/12, threshold+logistic: 12/12 across new cells. The updated 36-cell table has 0/36 linear votes across binary and continuous metrics.

**Partial success:**  
Continuous metrics show threshold preference in ≥ 8/12 cells. Retain Result III with a note that threshold preference extends to continuous metrics in the majority of test cells.

**Failure (metric artifact is real):**  
Continuous metrics show linear preference in ≥ 6/12 cells. Would require substantial rewriting of Result III to remove the synthetic tasks from the claim and restrict the threshold finding to LM loss only.

**Note on LM loss:** LM loss was already continuous and already shows threshold preference in 6/6 cells across three models. Even in the failure case, the LM loss result stands. But the argument becomes weaker if the synthetic tasks drop out.

### Expected Artifacts

```
results/reinforce_exp3/E3_continuous_metric_variants/
├── manifest.json
├── claim_impact.json
├── <model>/
│   ├── local_keymatch_logprob_curve.parquet
│   ├── longrange_logprob_curve.parquet
│   ├── bic_votes_continuous.json
│   └── combined_bic_table.json           # merge with existing 3P2-C cells
└── cross_model_bic_summary.json          # 36-cell combined table
```

---

## E6 — Llama Strict-Boundary Control Expansion

**Result addressed:** Result II (Boundary Computation Is Real but Model-Conditional)  
**Weakness addressed:** The primary model (Llama, mean R² = 0.380) is excluded from the confirmatory boundary claim while secondary models are included. This inversion invites reviewer scrutiny.  
**Artifact root:** `results/reinforce_exp3/E6_llama_boundary_expansion/`

### Background and Motivation

Result II's confirmatory claim states that high-SI heads preferentially attend to word boundaries in OLMo and Mistral under strict tokenizer-feature controls, but Llama is *excluded* from the confirmatory claim and classified as "ambiguous." This is the current status from the 7-seed multidomain adjudication in `reinforce_exp2/B3_cluster_ablation` and the 3P2-B multiseed analysis.

The inversion is unusual and demands an explanation: the model with the strongest SI signal (Llama, mean R² = 0.380, boundary d = 1.13 pre-control) cannot make the boundary claim, while models with weaker SI (OLMo d = 0.55, Mistral) can. The paper's current explanation — that "concentrated positional computation is more entangled with tokenizer structure" — is plausible but speculative. A reviewer will press on this because it looks like the main model failed the hardest test.

The ambiguity arises from the prefix-following artifact gate: in Llama, 3/7 seeds were flagged under the strict `d ≥ 0.2` prefix-delta rule. This is neither a clean pass (all seeds clean) nor a stable block (all seeds flagged). The 7-seed design has insufficient power to classify Llama definitively.

The expansion here takes a two-pronged approach:
1. **Increase seed count** to 24 (matching the R2B 24-seed design for OLMo) to get a stable multiseed verdict.
2. **Expand the feature ablation battery** to include a broader set of potential tokenizer artifacts beyond the current space-prefix predictor (top-16 embedding dimensions from a 99.7%-accuracy prefix classifier). Additional potential artifacts include: BPE merge frequency features, token length (number of characters in the subword), subword position within word (word-initial, word-medial, word-final), and capitalization. The current gate may be failing because Llama has a richer set of boundary-correlated surface features beyond just the space-prefix signal.

### Experimental Design

**Phase 1 — 24-seed replication of current strict-gate design.**  
Re-run the existing 3P2-B strict-control protocol on Llama-3.1-8B with 24 seeds across three domains (wiki, code, dialogue) — 8 seeds per domain. Apply the existing strict gate: boundary effect must be present (d > threshold) and prefix-following artifact must be absent (fake-minus-real delta d < 0.2). Report domain-level outcomes using the same `stable_clean / ambiguous / stable_blocked` classification. Primary question: does increasing from 7 to 24 seeds resolve the ambiguous verdict in one direction?

If 24-seed result is `stable_clean` in ≥ 2/3 domains: Llama joins OLMo and Mistral in the confirmatory claim. Update Result II to read "artifact-negative in all three primary models."

If 24-seed result is `stable_blocked` in ≥ 2/3 domains: Llama is confirmed to fail the strict gate. Result II remains OLMo+Mistral only, but the ambiguity is resolved to a clear exclusion rather than an open question.

If still ambiguous after 24 seeds: accept that the test is genuinely inconclusive for Llama and report this explicitly with a quantified uncertainty (e.g., posterior probability of clean-pass given multiseed data).

**Phase 2 — Expanded feature ablation battery.**  
Construct expanded tokenizer feature classifiers:
- **Space-prefix (existing):** binary indicator for whether the token begins with a space character.
- **BPE merge rank:** the position of this subword in the BPE merge priority list (lower = more frequent merge). Normalized to [0,1].
- **Subword length:** number of characters in the decoded subword. This captures the intuition that boundary-initial tokens are often longer (full words) vs. boundary-internal tokens (suffixes, morphemes).
- **Intra-word position:** categorical label — word-initial, word-medial, word-final, whole-word — derived by matching against the whitespace-delimited original text.
- **Capitalization flag:** binary indicator for whether the first character is uppercase.

Train separate probe classifiers for each feature using the same embedding-dimension ablation approach (identify top-k predictive dimensions via linear probe, then null those dimensions). For each feature, compute the fake-minus-real boundary delta after ablating feature-predictive dimensions. A feature "passes" the artifact gate if the boundary effect persists after ablation (d > 0.2 on boundary effect, d < 0.2 on artifact delta).

The expanded gate requires the boundary effect to survive all five feature ablations simultaneously. This is a strictly harder test than the existing single-feature (space-prefix) gate.

**Why this matters for the paper:** If Llama passes the 24-seed design or the expanded feature battery, Result II becomes a clean three-model result. If Llama consistently fails the expanded battery, the paper gains a more specific mechanistic insight: Llama's boundary sensitivity is entangled with *which specific feature* (e.g., intra-word position rather than space-prefix), which is itself a finding about how SI heads encode boundary information differently in different models.

### Pre-Specified Success / Failure Criteria

**Criterion 1 (24-seed verdict):** Stable verdict in ≥ 2/3 domains (either stable_clean or stable_blocked). Ambiguous in all 3 domains = inconclusive, report with posterior.

**Criterion 2 (expanded battery):** Boundary effect survives ≥ 3/5 feature ablations with d > 0.2 AND artifact flag negative for those same features = "conditionally clean" (partial support for boundary claim with enumerated confounds).

**Minimum acceptable outcome:** A quantified statement of what Llama's boundary status is, with a specific confidence level. The current "ambiguous" verdict is the one outcome that must be avoided — it is not a defensible resting state in the final submission.

### Expected Artifacts

```
results/reinforce_exp3/E6_llama_boundary_expansion/
├── manifest.json
├── preregistration.json
├── claim_impact.json
├── phase1_24seed/
│   ├── llama-3.1-8b/
│   │   ├── multiseed_gate_summary.json     # updated n=24 version
│   │   ├── domain_outcomes.json            # wiki/code/dialogue verdicts
│   │   └── pooled_adjudication.json
├── phase2_expanded_battery/
│   ├── feature_classifiers/
│   │   ├── space_prefix_probe.json
│   │   ├── bpe_merge_rank_probe.json
│   │   ├── subword_length_probe.json
│   │   ├── intraword_position_probe.json
│   │   └── capitalization_probe.json
│   ├── boundary_per_feature_ablation.json  # d values before/after each ablation
│   └── expanded_gate_summary.json
└── combined_verdict.json                   # final Llama boundary status
```

---

---

# TIER 2 — HELPFUL AND QUICK

---

## E2 — Random-Order Sensitivity Beyond R12 Grid

**Result addressed:** Result III (Distributed Redundancy)  
**Weakness addressed:** The 26/30 figure from NEW-R12 replaced the incorrect 30/30 claim in the paper draft; Llama's 7/10 result is classified as `partial_replication`. A reviewer may ask whether the threshold preference under random orderings is sensitive to the coarseness of the ablation schedule or the number of fractions evaluated.  
**Artifact root:** `results/reinforce_exp3/E2_ordering_sensitivity/`

### Background

NEW-R12 used a fixed ablation fraction grid of {0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50} across 10 random orderings per model. The 26/30 threshold-majority result is robust but not unanimous: Llama achieves only 7/10. The Llama partial result may reflect the coarseness of the fraction grid near Llama's threshold (~20%) where the grid has a single point (0.20) before jumping to 0.25 and 0.50. A finer grid near the threshold may stabilize the fit and increase Llama's threshold-majority count.

Additionally, the R12 design used a fixed batch size of 1 head per ablation step (removing one head at each fraction point). Different batch sizes (e.g., removing 5 heads at a time vs. 1 head at a time) could in principle shift the apparent threshold location.

### Experimental Design

**Variation 1 — Fine-grid schedule near threshold:**  
For all three models, re-run the 10 existing random orderings from R12 with a denser fraction grid: {0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.40, 0.50}. This adds 6 evaluation points near Llama's estimated threshold (0.10–0.25) without running new orderings. The re-fit BIC on the denser grid should give more stable threshold estimates.

**Variation 2 — Batch-size sensitivity:**  
For one model per priority (OLMo, since it has 10/10 strong replication), run 5 new random orderings with batch size 5 (remove 5 heads per step instead of 1). If threshold preference holds under batch-5 removal, the finding is not grid-discretization-dependent.

**Primary output:** Update the R12 table with fine-grid BIC votes. If Llama improves from 7/10 to 8/10 or better with the finer grid, update the paper to `partial_replication → strong_replication` for Llama and report 28–30/30 aggregate. If Llama stays at 7/10 with the finer grid, the partial-replication label is stable and the grid coarseness is not the cause — which is itself a useful diagnostic.

### Pre-Specified Success / Failure Criteria

- **Fine-grid stabilizes Llama:** Llama threshold-majority increases to ≥ 8/10. Report updated aggregate (28–30/30).
- **Fine-grid does not help Llama:** Llama stays at 7/10. Report this explicitly; coarseness is ruled out as the cause of Llama's partial result.
- **Batch sensitivity passes:** OLMo batch-5 shows threshold majority in ≥ 4/5 orderings. Add as robustness note.

### Expected Artifacts

```
results/reinforce_exp3/E2_ordering_sensitivity/
├── manifest.json
├── claim_impact.json
├── <model>/
│   ├── finegrid_bic_votes.json
│   └── batch5_bic_votes.json          # OLMo only
└── updated_r12_table.json
```

---

## E8 — Tokenizer-Overlap SI Comparison

**Result addressed:** Result I (SI structure exists; strength is model-conditional)  
**Weakness addressed:** Cross-model SI strength variation (Llama 0.380 vs. OLMo 0.058) is currently attributed to "model-conditional" differences without identifying which factor (tokenizer, training data, normalization, architecture) drives the variation. The tokenizer is the most tractable potential explanation.  
**Artifact root:** `results/reinforce_exp3/E8_tokenizer_overlap_si/`

### Background

The three primary models use different tokenizers: Llama-3.1-8B uses a Llama-3 BPE tokenizer with 128k vocabulary; Mistral-7B-v0.1 uses a 32k BPE tokenizer; OLMo-2-7B uses a GPT-NeoX tokenizer with 50k vocabulary. The tokenizers differ in vocabulary size, merge priority, and space-prefix encoding (Llama/Mistral use `▁` space-prefix; OLMo uses byte-level encoding). These differences could plausibly affect SI structure because SI is defined over token offsets, and the statistical regularity of offset-based patterns in logit space depends on how consistently tokens at given offsets share structural properties (boundary status, subword position, etc.).

The specific question: does tokenizer vocabulary overlap between model pairs correlate with SI score similarity? If models with more similar tokenizers have more similar R² distributions, tokenizer is a plausible driver. If not, other factors dominate.

### Experimental Design

**Step 1 — Compute pairwise tokenizer overlap.** For each pair of primary models (Llama–Mistral, Llama–OLMo, Mistral–OLMo), compute:
- Vocabulary overlap: |vocab_A ∩ vocab_B| / |vocab_A ∪ vocab_B| (Jaccard)
- Merge-order similarity: Spearman correlation of BPE merge ranks for shared tokens
- Boundary token fraction: fraction of tokens in each vocabulary that represent word-initial positions (space-prefixed or byte-level equivalent)

**Step 2 — Compute SI distribution similarity.** For each pair, compute:
- KL divergence of per-head R² distributions (using kernel density estimate)
- Spearman correlation of per-head R² rankings (do high-SI heads in one model correspond to high-SI heads in the other, by layer-head position?)
- Mean R² gap: |mean_R²_A - mean_R²_B|

**Step 3 — Test tokenizer-SI correlation.** With 3 model pairs and 3 similarity metrics, compute a 3×3 correlation matrix and report whether tokenizer overlap correlates with SI distribution similarity more than expected under a null model (model-pair permutation null).

**Interpretation boundary:** With n=3 pairs, no statistical inference is meaningful in isolation. The result is informative only if the correlation is monotone and the effect is large (r > 0.8 or r < −0.8). If the pattern is mixed, report as inconclusive and retain the "model-conditional" framing without attribution.

### Expected Artifacts

```
results/reinforce_exp3/E8_tokenizer_overlap_si/
├── manifest.json
├── claim_impact.json
├── tokenizer_overlap_matrix.json
├── si_distribution_similarity_matrix.json
└── correlation_summary.json
```

---

## E9 — Layer-Head SI Heatmaps

**Result addressed:** All results (communication/interpretability)  
**Weakness addressed:** The paper currently describes SI structure distributions in prose. Reviewers expect a figure showing where high-SI heads live in layer-head space. The heatmap is the standard visualization for this type of analysis and is expected in any mechanistic interpretability paper.  
**Artifact root:** `results/reinforce_exp3/E9_si_heatmaps/`

### Background

All per-head R² values are already computed and stored in `results/experiment3/theory1_si_circuits/<model>/head_r2_summary.parquet` and `results/reinforce_exp2/B1_kernel_taxonomy/cluster_membership.parquet`. No new model runs are needed — this is purely a visualization and data-export task.

### Experimental Design

**Figure 1: R² heatmap (layer × head, per model)**  
For each of the three primary models, produce a 32-layer × 32-head heatmap with R² values encoded as color intensity. Use a perceptually uniform colormap (e.g., viridis). Mark the top-quartile threshold with a contour or secondary annotation. The three heatmaps should be panels in a single figure (3-panel layout) to allow direct cross-model comparison.

**Figure 2: Cluster membership overlay**  
Using the `cluster_descriptor_kmeans` labels from `cluster_membership.parquet`, produce a second version of each heatmap where cells are colored by cluster ID rather than continuous R². Use a discrete colormap with 3–4 colors (matching cluster count per model). Overlay the high-SI contour from Figure 1 as a boundary line.

**Figure 3: Boundary attention score heatmap**  
Using the `boundary_attn_score` column from `cluster_membership.parquet`, produce a third heatmap showing boundary attention score in layer-head space. This directly visualizes Result II and shows which heads attend to boundaries, complementing the R² heatmap.

These three figure panels should be generated as high-DPI PDFs suitable for inclusion in `paper/neurips2026/figures/`. Also export the underlying data tables as parquet for reproducibility.

### Expected Artifacts

```
results/reinforce_exp3/E9_si_heatmaps/
├── manifest.json
├── figures/
│   ├── fig_r2_heatmap_all_models.pdf
│   ├── fig_cluster_heatmap_all_models.pdf
│   └── fig_boundary_heatmap_all_models.pdf
└── data/
    ├── r2_heatmap_data.parquet
    ├── cluster_heatmap_data.parquet
    └── boundary_heatmap_data.parquet
```

---

## E10 — SI vs. Retrieval-Head Overlap

**Result addressed:** Result III (Distributed Redundancy); broader positioning  
**Weakness addressed:** The DuoAttention / retrieval-head literature (Wu et al. 2024, "Retrieval Heads") is cited in the paper. Reviewers familiar with that line of work will ask: are your "distributed SI infrastructure" heads the same population as retrieval heads? If they overlap substantially, the finding may not be novel. If they are disjoint, this is an informative distinction that strengthens the paper's positioning.  
**Artifact root:** `results/reinforce_exp3/E10_si_retrieval_overlap/`

### Background

Wu et al. (2024) identify retrieval heads as a sparse subset of heads that copy information from long context. Their definition is based on information-copy behavior under needle-in-a-haystack type tasks — heads that consistently attend to the location of a key fact when that fact is required for the answer. This is a fundamentally different criterion than R²: retrieval heads are defined by *content-conditional* long-range attention, while high-SI heads are defined by *offset-structured*, content-independent attention. By definition, these should be mostly disjoint populations (content-conditional vs. content-independent). Testing this directly provides an important triangulation.

### Experimental Design

**Step 1 — Identify retrieval heads.** Implement the Wu et al. (2024) "copy score" definition: for each head, compute the average attention weight placed on the position of a key token in a needle-in-a-haystack evaluation, minus the mean attention across all positions. A head's retrieval score is its mean copy score across many (key, needle position, distractor context) triples.

Use 200 evaluation examples per model, with needle positions sampled uniformly from the sequence length. Run on the same three primary models. Rank all 1024 heads by retrieval score per model.

**Step 2 — Compute overlap statistics.**  
- **Jaccard overlap:** |top-quartile_SI ∩ top-quartile_retrieval| / |top-quartile_SI ∪ top-quartile_retrieval| for each model.
- **Rank correlation:** Spearman ρ between R² rank and retrieval-score rank across all 1024 heads per model.
- **Cluster enrichment:** For each cluster (from B1 taxonomy), compute mean retrieval score. Test whether any cluster is significantly enriched for retrieval-head behavior (Kruskal-Wallis across clusters, Holm-corrected).

**Step 3 — Interpret.** If Jaccard < 0.10 and rank correlation |ρ| < 0.20 in all three models: SI heads and retrieval heads are empirically disjoint populations. Add a sentence to Result III or Discussion noting this and citing Wu et al. as the reference point.

If overlap is substantial (Jaccard > 0.25 or |ρ| > 0.40): the two head types co-occur and the paper needs to distinguish them mechanistically, or acknowledge that "SI infrastructure" and "retrieval capacity" may be operationally linked.

### Expected Artifacts

```
results/reinforce_exp3/E10_si_retrieval_overlap/
├── manifest.json
├── claim_impact.json
├── <model>/
│   ├── retrieval_scores.parquet           # per-head copy scores
│   ├── si_retrieval_jaccard.json
│   └── cluster_retrieval_enrichment.json
└── cross_model_overlap_summary.json
```

---

---

# TIER 3 — HELPFUL BUT TIME-INTENSIVE

---

## E4 — Checkpoint SI-Emergence Trajectory

**Result addressed:** Result I (SI structure is present and functionally load-bearing; "learned strength")  
**Weakness addressed:** Critique Weakness 2 — in RoPE models, some degree of shift-invariant structure in logits is expected by construction, because the RoPE inner product depends on relative offset by design. The paper claims SI strength is "learned" and "model-conditional," but it does not demonstrate that R² increases during training vs. being present from initialization.  
**Artifact root:** `results/reinforce_exp3/E4_checkpoint_trajectory/`

### Background

The paper's framing distinguishes between the *availability* of positional structure (all RoPE models have it by construction) and the *degree to which individual heads exploit it* (varies from R² = 0.058 in OLMo to 0.380 in Llama). The "learned" framing assumes that training shapes which heads develop high R² — that it is not simply the RoPE initialization doing all the work. This assumption is plausible but untested.

A trained head with R² = 0.9 could arise because: (a) training specifically optimized that head to implement offset-based attention, which would show R² increasing from a random-initialization baseline; or (b) the RoPE geometry makes offset-structured logits the path of least resistance, and training simply didn't penalize this structure, leaving it at a high but architecturally-natural level.

The test: compute R² distributions at multiple training checkpoints for a single model, from initialization (step 0) through pretraining. If R² for what-become-high-SI heads starts near zero and increases substantially during training, the "learned" framing is correct. If R² starts high at initialization and stays roughly flat, SI structure is largely architectural and the "learned" framing is misleading.

### Experimental Design

Use a single model for which intermediate training checkpoints are accessible. OLMo-2-7B is the most suitable candidate because OLMo releases public training checkpoints at regular intervals. Check availability of OLMo-2 intermediate checkpoints via the OLMo-2 training run documentation; if checkpoints are available at step intervals of 10k–100k steps, this is feasible.

**Step 1 — Load checkpoints.** Identify 6–10 checkpoints spanning training from ~step 1k (early) to step ~final. Include step 0 (random initialization) if available.

**Step 2 — Compute per-head R² at each checkpoint.** Use the same evaluation corpus and same g_h estimation procedure as Experiment 1. This yields a per-head R² time series of shape (n_checkpoints × 1024 heads).

**Step 3 — Track high-SI head emergence.** For the final-checkpoint top-quartile heads (the 256 heads that end up high-SI), plot their mean R² over training. For the final-checkpoint bottom-quartile heads, plot their mean R² over training. The question is whether the two groups diverge during training (learned specialization) or are separated from the start (architectural separation).

**Step 4 — Fit trajectory model.** For each head, fit a monotone growth model (logistic or piecewise linear) to R² vs training step. Cluster heads by trajectory shape: heads that start high and stay high (architectural), heads that start low and grow (learned), heads that start high and decrease (suppressed). The mixture of trajectory types provides a nuanced answer to the learned-vs-architectural question.

**Output for paper:** A training trajectory figure showing R² evolution for the top-quartile and bottom-quartile head groups. If the top group starts low and grows, add to Result I: "SI strength emerges during training rather than being present at initialization, supporting the learned-property interpretation." If it starts high, add to Discussion: "Much of OLMo's SI structure appears geometrically induced by RoPE rather than actively learned; this may explain OLMo's weak-SI regime behavior."

### Effort and Dependencies

Requires: (1) access to OLMo-2-7B intermediate checkpoints (check HuggingFace or OLMo release), (2) compute to run the R² estimation pipeline on each checkpoint (roughly 1 GPU-hour per checkpoint × 8 checkpoints = ~8 GPU-hours), (3) modification to the Experiment 1 pipeline to accept arbitrary checkpoint paths. Estimated total: 1–2 weeks including checkpoint acquisition and pipeline modification.

### Expected Artifacts

```
results/reinforce_exp3/E4_checkpoint_trajectory/
├── manifest.json
├── claim_impact.json
├── olmo-2-7b/
│   ├── r2_per_checkpoint.parquet     # shape: (n_checkpoints, n_heads)
│   ├── trajectory_clustering.parquet
│   ├── trajectory_fit_summary.json
│   └── checkpoint_metadata.json
└── figures/
    └── fig_r2_training_trajectory.pdf
```

---

## E5 — Additional 7–9B Model Family

**Result addressed:** All four results  
**Weakness addressed:** N=3 models is the current cross-model evidence base. All three are RoPE-based, open-weight, English-trained models from 2023–2024. Adding a fourth model from a different training lineage (e.g., Gemma-7B, Qwen-7B, or Falcon-7B) would meaningfully extend external validity.  
**Artifact root:** `results/reinforce_exp3/E5_fourth_model/`

### Experimental Design

Run the full core battery on a fourth 7–9B RoPE model:
- T8 kernel ablation (Result I)
- Strict-gate boundary test (Result II)
- Cumulative ablation with BIC voting (Result III)
- Regime interaction (Result IV, if computational budget allows)

Model selection priority: **Gemma-2-9B** (Google, 2024) is the most different from existing models in training pipeline (Google-internal data, GQA architecture, different normalization). Qwen-2.5-7B is an alternative (different tokenizer family, stronger multilingual training). Falcon-7B is less preferred (older, weaker baseline performance).

The full battery requires approximately 2–3 weeks of compute plus code adaptation for the new model's tokenizer and architecture. Results would strengthen every cross-model claim from "n=3" to "n=4" and provide a meaningful check on whether the threshold framing holds outside the Llama/Mistral/OLMo training lineage.

### Priority note

If E0, E3, and E6 are completed first and the submission is tight on time, E5 is best left as future work with an explicit note: "extending to additional model families is a planned follow-up." The n=3 evidence base is sufficient for a NeurIPS submission; n=4 would be stronger but is not blocking.

---

## E7 — Non-English Boundary Domain

**Result addressed:** Result II (Boundary computation is real but model-conditional)  
**Weakness addressed:** All three primary models are predominantly English-trained. The boundary attention result could be specific to English whitespace-based tokenization rather than a general property of subword boundary processing. Non-English evaluation would distinguish "whitespace-boundary detection" from "tokenizer-structural-boundary detection."  
**Artifact root:** `results/reinforce_exp3/E7_nonenglish_boundary/`

### Background and Design

The existing `data/experiment3_phase2_multilingual/` directory contains data for Russian (opus100_ru), Turkish (opus100_tr), and Chinese (opus100_zh). Chinese is the most informative choice: Chinese text has no whitespace between words, so "word boundaries" in Chinese correspond to character-level morphological boundaries rather than whitespace markers. If high-SI heads show boundary-linked attention in Chinese text despite the absence of whitespace, this rules out the "heads are detecting whitespace characters" explanation and supports a more abstract "structural boundary detection" interpretation.

Turkish is also valuable because Turkish morphology produces very long words with multiple meaningful suffixes, so word-initial BPE tokens are more clearly marked by capitalization/frequency than by whitespace.

**Design:** Apply the same boundary attention analysis from Result II to Chinese and Turkish evaluation text, using the Llama and OLMo tokenizers. Note that Llama-3's BPE tokenizer handles Chinese by decomposing to byte-level sequences; the definition of "word boundary" must be adapted to character-level or morpheme-level segmentation using an external tokenizer (jieba for Chinese, snowball/MorphAnalyzer for Turkish).

**Primary question:** Does the high-SI vs. low-SI boundary attention differential survive in Chinese and Turkish text? If yes, the boundary result is not specific to English whitespace. If no, the finding is explicitly scoped to whitespace-tokenized text.

---

## E11 — High-SI RoPE-Frequency Decomposition

**Result addressed:** All results (mechanistic depth)  
**Weakness addressed:** Critique Weakness 5 and 6 — the paper lacks a mechanistic account of *what* high-SI heads compute. The shape of g_h(Δ) (the learned offset-response kernel) is the most tractable route to understanding what SI heads actually do, and its connection to RoPE frequency bands provides the link between the positional encoding design and the learned SI structure.  
**Artifact root:** `results/reinforce_exp3/E11_rope_freq_decomposition/`

### Background and Design

Barbero et al. (2024) "Round and Round We Go: What Makes RoPE Useful?" (arXiv:2410.06205) shows that in Gemma-7B, high-frequency RoPE dimensions produce robust positional/diagonal attention patterns while low-frequency dimensions carry semantic content. This frequency-band division of labor is likely present in Llama and Mistral and would explain how SI structure arises.

The decomposition: for each high-SI head h, the learned kernel g_h(Δ) is a function of relative offset. Under RoPE, the attention logit can be written as a sum over frequency bands:

```
q_i^T k_j = Σ_{d=0}^{D/2} (Re(q_i^(d)) Re(k_j^(d)) + Im(q_i^(d)) Im(k_j^(d))) cos((i-j)θ_d)
           + cross terms
```

where θ_d = base^{-2d/D} are the RoPE frequencies. For heads with strong SI structure (high R²), the cross terms should be small, meaning the logit is dominated by the cosine terms — and the dominant frequency bands for each head determine what offsets that head is sensitive to.

**Design:** For each high-SI head, compute the power spectrum of g_h(Δ) via DFT. Identify the peak frequency. Cluster heads by their peak frequency (low-frequency heads = long-range preference; high-frequency heads = short-range/local preference). Cross-reference with boundary attention score (Result II) and cluster membership (B1). The hypothesis: high-frequency SI heads are the boundary-sensitive ones (because word boundaries are at offset Δ=1, which is a high-frequency pattern), while low-frequency SI heads handle long-range structure.

This analysis requires no new model runs — g_h(Δ) is already estimated in the Experiment 1 pipeline. It is a reanalysis of existing artifacts.

---

## E12 — ICL Sensitivity Under SI-Kernel Subtraction

**Result addressed:** Result I (SI structure is functionally load-bearing)  
**Weakness addressed:** The paper's causal claim (SI structure is "functionally necessary") currently rests entirely on perplexity increase under kernel ablation. The specific *task* that SI infrastructure serves is unspecified. Induction heads — the closest prior art to SI heads — are defined by their role in in-context learning. Testing whether ICL performance is disproportionately sensitive to SI-kernel ablation would directly link SI infrastructure to the induction head literature.  
**Artifact root:** `results/reinforce_exp3/E12_icl_sensitivity/`

### Background and Design

Olsson et al. (2022) established that induction heads are causally necessary for in-context learning: ablating induction heads causes a sharp drop in ICL performance. High-SI heads include the induction-head class (induction heads attend at a fixed offset corresponding to where the pattern was seen, making them high-R²). If SI infrastructure overlaps substantially with induction heads, we would expect ICL performance to degrade preferentially (relative to non-ICL tasks) under SI-kernel subtraction.

**ICL task design:** Use a simple few-shot classification task: present 4 examples of (token X → token Y) mapping followed by a query (token X → ?). Measure accuracy and log-probability of the correct output Y. Compare SI-kernel-ablated models vs. intact models on this ICL task vs. a matched non-ICL task (same token pairs, but presented in a random order that does not support pattern inference).

**Primary question:** Is the ratio `ICL accuracy loss / non-ICL accuracy loss` significantly greater than 1 under SI-kernel ablation? If yes, SI heads are preferentially serving ICL-relevant computation, linking them to the induction head mechanism. If no, the causal cost of SI-kernel ablation is task-general rather than ICL-specific.

**Complication:** The E10 SI vs. retrieval-head overlap experiment should run first, since if retrieval heads and SI heads overlap substantially, the ICL test conflates two distinct populations. E12 is most interpretable when E10 has confirmed that SI heads and retrieval heads are distinct.

---

---

# Implementation Notes and Execution Order

## Recommended execution order

1. **E9** (heatmaps) — do this first; zero compute, enables paper figure work in parallel with experiments
2. **E3** (continuous metrics) — must-have; can run while E0 is being designed
3. **E0** (cluster-sequential ablation) — must-have; uses existing infrastructure from 3P2-C and B1
4. **E6** (Llama boundary expansion) — must-have; Phase 1 (24-seed) can start immediately
5. **E2** (fine-grid R12) — quick; uses existing R12 infrastructure with denser grid
6. **E10** (SI vs retrieval overlap) — quick; requires implementing Wu et al. copy score
7. **E8** (tokenizer overlap) — quick; primarily reanalysis
8. **E4, E5, E7, E11, E12** — in parallel or sequentially as time allows

## Artifact schema requirements

Every experiment must produce:
- `manifest.json`: run metadata (timestamp, git SHA, seed list, model IDs, data paths, sample sizes)
- `claim_impact.json`: structured verdict with `claim_status` (supported/mixed/not_supported), `supports_main_text` (bool), and `notes` list
- All data outputs as `.parquet` (not `.csv`) with a `data_dictionary.json` schema descriptor

## Paper update notes

- **E9** output goes directly into `paper/neurips2026/figures/` — coordinate with paper editing
- **E3** result updates the abstract's "0/18 linear" claim to "0/36 linear across binary and continuous metrics" if successful
- **E0** result either confirms or demotes the "distributed redundancy" interpretation in Result III Discussion
- **E6** result either expands or formally resolves the Llama boundary exclusion in Result II
- **E2** fine-grid result may update the R12 aggregate from 26/30 upward

## Connection to paper stale-number fixes (pre-submission, not experiments)

These are editorial fixes to `main.tex` required regardless of which experiments are run:
- Line 39 (abstract): `30/30` → `26/30`; add `(Llama 7/10, Mistral 9/10, OLMo 10/10)`
- Line 343 (Result III interpretation): `30/30` → `26/30`
- Lines 437–438 (robustness): Update to match UNIFIED_RESULTS.md language; Llama is `partial_replication`
- Lines 445–446 (NEW-R14 paragraph): Remove "not yet run for Llama and Mistral" — R14 is complete for all three models
- Line 454 (R2B): `12-seed` → `24-seed`
- Abstract lines 43–44 (R14): Update to cite all three models with their specificity ratios (4.57 / 11.83 / 4.57)
