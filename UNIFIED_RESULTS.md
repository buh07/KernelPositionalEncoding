# Unified Results: Kernel PE (Paper-Facing Snapshot)

**Last updated:** May 4, 2026 (America/New_York)  
**Primary manuscript target:** `paper/Formatting_Instructions_For_NeurIPS_2026/neurips_2026.tex`

This file is the canonical, paper-facing consolidation of finalized evidence used in the NeurIPS 2026 submission posture.

Thesis alignment (paper-facing): RoPE exposes an understudied SI channel; trained models use that channel in load-bearing but heterogeneous ways; this motivates a practical pretraining-design direction rather than a closed causal claim.

Reproducibility status (submission): an anonymized GitHub repository is included in supplementary material with code, execution scripts, environment specification, and artifact-mapped reproduction commands for the finalized evidence stack.

## 1) Finalized headline posture

### Result I (Promoted)
**Permutation-specific SI disruption cost is real, load-bearing, and now strongly hardened at head level.**

- Core LM-loss deltas under SI-kernel subtraction:
  - Llama-3.1-8B: `+4.15` nats (`+206%`, `d=11.6`)
  - Mistral-7B-v0.1: `+3.43` nats (`+185%`, `d=14.6`)
  - OLMo-2-7B: `+0.058` nats (`+2.8%`, `d=0.81`)
- NEW-R14 permutation specificity remains positive in all three models (`p=2.0e-4` each).
- E17 norm-matched control does **not** support SI-exclusivity (true > permuted, but true < norm-matched random).
- E28A/E28B/E28D/E29A/E30B hardening outcomes:
  - E28A: 20-bin exact dose-response supported in all three models.
  - E28B: SI-ranked intervention > importance-matched non-SI control in all three models.
  - E28D: non-trivial excess over local-bias null supported in 2/3 models (Llama, Mistral; not OLMo).
  - E29A: kernel-transplant specificity supported in all three models (`low_transplant_normmatched - low_self > 0` with positive CI and one-sided permutation support).
  - E30B: richer local-bias-family decomposition also supports non-trivial excess tracking in 2/3 models (Llama, Mistral; not OLMo).
- E31D naturalistic add-on (Llama only, natural-text passkey contexts):
  - SI-targeted ablation exceeds importance-matched non-SI ablation:
    - mean delta `+3.636`
    - 95% CI `[2.133, 5.028]`
    - one-sided `p=3.33e-05`
  - SI-targeted ablation also exceeds permuted control:
    - mean delta `+3.080`
    - 95% CI `[1.707, 4.368]`
    - one-sided `p=3.33e-05`

### Result II (Supporting organizational observation)
**Ranked cumulative ablation robustly rejects linear depletion, but this is not SI-unique at collective level.**

- SI-ranked cells: `0/18` linear votes (3-way BIC); random-order corroboration `26/30`.
- Matched non-SI baseline (E24): non-SI cells also mostly non-linear (`1/9` linear; `8/9` non-linear).
- Direct SI-vs-non-SI enrichment quantification (E24b): mixed / non-decisive (CI spans zero).

**Interpretation:** strong ranked-ablation nonlinearity regularity; limited SI-specificity at collective level.

### Result III (Supporting functional probe finding)
**SI intervention preferentially disrupts controlled offset-repetition retrieval probes, with bounded transfer.**

- E12 controlled 9-token probe: directional preferential disruption in all three primary models.
- E18 strict format-confound adjudication: mixed/model-conditional (`1/3` survives).
- E28C broader probe-diversity transfer (7 families): **not supported** at model-pass level (`0/3` pass).
- E29B natural-text long-context probe: supported with caveat (`2/3` model directional pass; Llama/Mistral positive, OLMo negative).
- E30A boundary 2x2 grid (short/long × high/low regularity): supported with caveat (`2/3` model directional pass; Llama/Mistral pass, OLMo non-pass), with non-universal factor-direction effects.
- E29C Qwen2.5-7B quick anchor: no directional corroboration in this quickcheck (`not_supported`, non-primary-model status).

**Interpretation:** valid controlled mechanistic probe result; not a naturalistic broad ICL generalization claim.

### Theoretical Lens (Supporting with caveat)
**A CS-inspired framework provides preliminary predictive structure for SI heterogeneity and ablation criticality.**

- Appendix H / Exp 7A (corrected audit): model-level proxy coherence relation is mixed/non-promotional:
  - model-level (`n=6`): Pearson `r=+0.731`, `p=0.099`; Spearman `rho=+0.516`, `p=0.295`; permutation two-sided `p=0.141`
  - 36-row table is exploratory/pseudo-replicated (row-level Pearson `r=+0.609`, `p=8.11e-05`)
  - geometry source is analytic proxy metadata, not learned PE-matrix extraction
- Appendix H / Exp 7B (corrected calibration): sparsity-critical-point analogue `m* ∝ s_eff log(N/s_eff)` remains strongly directional:
  - canonical single-condition calibration: `C=7.057` (Llama-3.1-8B, seq-len 256)
  - all six conditions: Pearson `r=-0.959`, `p=0.00254`; Spearman `rho=-0.878`, `p=0.0213`
  - out-of-sample (`n=5`, excluding calibrated condition): Pearson `r=-0.986`, `p=0.00192`; Spearman `rho=-0.866`, `p=0.0577`
  - global-fit reference `C=7.834` reported separately as sensitivity context
- E31B expanded proxy coherence refresh (11-model panel): mixed/non-promotional
  - `Pearson r=+0.577`, `p=0.063`
  - `Spearman rho=+0.500`, `p=0.117`
  - permutation two-sided `p=0.070`
- Scope: treated as a **potential theoretical framework**, not a formal proof or confirmatory causal mechanism (effective `n` remains small).

### Breadth Extension (Supporting with caveat)
**SI heterogeneity remains large on an expanded model panel.**

- E31A canonical breadth consolidation over 11 models and 7 families:
  - spread ratio max/min mean R² = `140.12x`
  - top means: Gemma-2-9B `0.868`, Qwen2.5-7B `0.605`, Llama-3.1-8B `0.380`
  - low means: OLMo-2-7B `0.058`, Pythia-1.4B `0.0065`, Pythia-410m `0.0062`
- Interpretation: supports broad heterogeneity as a descriptive empirical pattern; not causal attribution.

### Functional Coupling Reanalysis (Supporting with caveat)
**Higher model-level SI amplitude aligns with larger preferential functional disruption on tested probe suites.**

- E31C (using existing E12/E29B/E30A outputs):
  - `Pearson r=0.956`
  - `Spearman rho=0.949`
  - grouped one-sided exact permutation `p=0.00461`
- Scope: directional coupling in existing model/task set; not an external-benchmark generalization claim.

---

## 2) Completed control stack (status)

| Control | Final status | Claim impact |
|---|---|---|
| E17 norm-matched specificity | `not_supported` | removes SI-exclusivity language in Result I |
| E18 strict format-confound | `mixed / model_conditional` | keeps Result III directional/model-conditional |
| E19 SI-score robustness | `supported / robust_cross_model` | moves SI-measurement caveat from unresolved to bounded-supported |
| E20 variance decomposition | `model_family_dominant` | heterogeneity not seed-noise in matched proxy setup |
| E21 paired RoPE-vs-NoPE proxy | `rope_gt_nope_proxy` | strengthens capacity anchor at 1.1B proxy scale |
| E24 matched non-SI cumulative baseline | `supported_with_caveat` | shows collective nonlinearity is mostly generic |
| E24b SI-vs-non-SI contrast | `mixed` | SI-specific collective enrichment non-decisive |
| E25 comparability panel | `supported_with_caveat` | bounded cross-model comparability; control-normalized axis mixed |
| E27 absolute-threshold robustness | `supported` | fixed-threshold per-head cross-model gradient consistent |
| E28A 20-bin exact hardening | `supported` | replaces fragile 5-bin approximate-p reporting |
| E28B importance-matched control | `supported` | direct SI-over-generic-importance separation |
| E28D local-bias decomposition | `supported_with_caveat` | non-trivial excess evidence in 2/3 models |
| E28C probe-diversity transfer | `not_supported` | bounds Result III scope beyond short aligned probe |
| E29A kernel-transplant specificity | `supported` | strengthens anti-tautology specificity for Result I |
| E29B natural-text long-context probe | `supported_with_caveat` | adds semi-naturalistic scope with explicit 2/3 boundary |
| E30A boundary 2x2 transfer grid | `supported_with_caveat` | reproduces 2/3 split but no universal factor-direction law |
| E30B richer local-bias-family decomposition | `supported_with_caveat` | confirms 2/3 non-trivial excess tracking beyond multi-family local-bias nulls |
| E29C Qwen2.5-7B quick anchor | `not_supported` | directional-only non-primary anchor did not corroborate |
| E31A breadth consolidation | `supported_with_caveat` | extends heterogeneity beyond trio (11 models / 7 families; 140x spread) |
| E31B coherence refresh | `mixed` | expanded proxy coherence relation is non-decisive; keeps CS lens framework-level |
| E31C functional-coupling reanalysis | `supported_with_caveat` | directional R²-to-functional-gap coupling on existing probe suites |
| E31D naturalistic SI vs importance (Llama) | `supported` | strengthens bounded practical relevance of SI-targeted intervention |

---

## 3) Key finalized numerical updates

### E20 (completed)
- `between_model_var_si = 0.04074`
- `within_model_seed_var_si = 0.000223`
- ratio `between/within = 182.73`
- interpretation: `model_family_dominant`

### E21 (completed)
- paired seeds `n=3`
- `delta si_mean_r2 (RoPE - NoPE) = 0.4665`
- `95% CI = [0.4553, 0.4872]`
- interpretation: `rope_gt_nope_proxy`

### E28A (completed)
- interpretation: `dose_response_exact_supported`
- Llama: `rho=0.9594`, one-tailed permutation `p=4.999975e-06`
- Mistral: `rho=0.9639`, one-tailed permutation `p=4.999975e-06`
- OLMo: `rho=0.6842`, one-tailed permutation `p=7.04996475e-04`

### E28B (completed)
- interpretation: `si_over_importance_supported`
- cross-model pass: `3/3`
- pooled mean delta (SI - importance): `+2.2800`
- per-model delta (95% CI):
  - Llama: `+3.6048` `[3.5142, 3.7005]`
  - Mistral: `+3.2086` `[3.1397, 3.2778]`
  - OLMo: `+0.02669` `[0.01672, 0.03813]`

### E28D (completed)
- interpretation: `r2_excess_tracks_disruption_supported_with_caveat`
- pass count: `2/3`
- supports non-trivial excess tracking:
  - Llama: yes (`rho_excess=0.8932`, `p=1.16e-07`)
  - Mistral: yes (`rho_excess=0.8120`, `p=1.38e-05`)
  - OLMo: no (`rho_excess=0.2064`, `p=0.3825`)

### E28C (completed)
- interpretation: `no_generalization_beyond_short_probe`
- model pass count: `0/3`
- LP family passes (`required >=6/7`):
  - Llama: `4/7`
  - Mistral: `4/7`
  - OLMo: `2/7`

### E29A (completed)
- interpretation: `kernel_transplant_specificity_supported_all_models`
- model pass count: `3/3`
- delta (`low_transplant_normmatched - low_self`), mean [95% CI]:
  - Llama: `+0.9472` `[0.9023, 0.9989]`
  - Mistral: `+0.3738` `[0.3501, 0.3978]`
  - OLMo: `+0.01736` `[0.01503, 0.02021]`

### E29B (completed)
- interpretation: `naturaltext_longcontext_preferential_gap_supported_with_caveat`
- model directional pass count: `2/3`
- pooled true preferential LP gap [95% CI]:
  - Llama: `+3.844` `[3.099, 4.576]` (pass)
  - Mistral: `+3.647` `[3.178, 4.130]` (pass)
  - OLMo: `-0.904` `[-1.112, -0.694]` (fail)

### E29C (completed)
- interpretation: `qwen_anchor_no_directional_corroboration`
- model directional pass count: `0/1`
- Qwen2.5-7B quickcheck:
  - `si_mean_r2 = 0.6051`
  - `delta_true_minus_perm = -0.1671` (no directional corroboration)

### E30A (completed)
- interpretation: `boundary_grid_supported_with_caveat`
- model directional pass count: `2/3`
- pooled true preferential LP gap [95% CI]:
  - Llama: `+5.086` `[4.245, 5.939]` (pass)
  - Mistral: `+3.455` `[2.821, 4.076]` (pass)
  - OLMo: `-0.744` `[-1.026, -0.468]` (non-pass)
- boundary-factor contrasts are not universal:
  - regularity high-minus-low is negative in all three models
  - length long-minus-short is positive in Llama/Mistral and near-zero negative in OLMo

### E30B (completed)
- interpretation: `localbias_family_excess_tracks_disruption_supported_with_caveat`
- model pass count: `2/3`
- supports non-trivial excess tracking over best local-bias family fit:
  - Llama: yes (`rho_excess_family=0.9008`, `p=6.16e-08`)
  - Mistral: yes (`rho_excess_family=0.8030`, `p=2.02e-05`)
  - OLMo: no (`rho_excess_family=0.2273`, `p=0.3352`)

### E31A (completed)
- interpretation: `breadth_panel_consolidated`
- panel size: `n_models=11`, `n_families=7`
- spread ratio max/min mean R²: `140.121`

### E31B (completed)
- interpretation: `coherence_lens_extended`
- panel size: `n_models=11`
- correlation:
  - Pearson `r=+0.577`, `p=0.063`
  - Spearman `rho=+0.500`, `p=0.117`
  - permutation two-sided `p=0.070`
- one-sided negative-direction permutation test is non-supportive (`p=0.973`)

### E31C (completed)
- interpretation: `directional_functional_coupling_supported_with_caveat`
- rows: `n=9` (3 models x E12/E29B/E30A)
- Pearson `r=0.956`, one-sided grouped exact `p=0.00461`

### E31D (completed)
- interpretation: `naturalistic_si_vs_importance_supported_single_model`
- model: Llama-3.1-8B
- prompts: `n=60` (3 families x 20)
- mean delta (SI - importance): `+3.636` (95% CI `[2.133, 5.028]`, `p=3.33e-05`)
- mean delta (SI - permuted): `+3.080` (95% CI `[1.707, 4.368]`, `p=3.33e-05`)

---

## 4) What this means for the paper

1. **Result I is the strongest SI-specific evidence stack** once read as bounded disruption cost (not SI-exclusivity).
2. **Result II is supporting context:** a collective regularity result, not a unique SI mechanism result.
3. **Result III is supporting and intentionally narrow:** controlled probe sensitivity, non-passing broad synthetic-family transfer, and mixed long-context transfer (2/3).
4. **Performance-improvement notion is hypothesis-only:** broader semantic SI exploitation may help, but this is not a finding in the current evidence stack.
5. **Capacity framing is strengthened but still proxy-bounded:** E21 is 1.1B matched proxy evidence; 7–8B matched PE-family causal training contrast remains open.
6. **Cross-model heterogeneity is now variance-decomposed in proxy setting (E20),** but causal driver attribution across tokenizer/corpus/optimization differences is still open.
7. **Expanded breadth and functional coupling are now directionally stronger (E31A/C),** but extra-family causal/theoretical attribution remains unresolved (E31B mixed).

---

## 5) Residual limitations (explicit)

- No matched 7–8B RoPE-vs-NoPE/ALiBi full training counterfactual.
- No Llama/Mistral checkpoint trajectory equivalent to OLMo-only E4.
- Result II SI-specificity at collective level remains mixed after E24/E24b.
- Result III transfer beyond short aligned probe families is non-passing (E28C), and natural-text long-context transfer is model-conditional (E29B 2/3).
- E31D naturalistic SI-vs-importance result is Llama-only (60 prompts): useful bounded pilot evidence, not broad transfer proof.
- Boundary-grid transfer corroborates the same 2/3 split but does not identify a universal context/regularity factor law (E30A).
- Intervention-mechanism decomposition is incomplete: no full softmax-redistribution audit yet (token-level entropy/peakiness shifts and constant-offset perturbation controls), so Result I remains a load-bearing sensitivity claim under this perturbation class.
- Extra-model quick anchor (Qwen2.5-7B) did not provide directional corroboration in quickcheck mode (E29C).
- Cross-model comparability remains bounded; some normalized rankings remain mixed (E25).

---

## 6) Canonical artifact roots

- `results/reinforce_exp3/E28a_e26_20bin_exact/`
- `results/reinforce_exp3/E28b_importance_matched_control/`
- `results/reinforce_exp3/E28d_r2_localbias_decomposition/`
- `results/reinforce_exp3/E28c_probe_diversity_transfer/`
- `results/reinforce_exp3/E29a_kernel_transplant_specificity/`
- `results/reinforce_exp3/E29b_naturaltext_longcontext_probe/`
- `results/reinforce_exp3/E29c_qwen_anchor_quickcheck/`
- `results/reinforce_exp3/E30a_probe_boundary_grid/`
- `results/reinforce_exp3/E30b_localbias_null_family/`
- `results/reinforce_exp3/E31a_breadth_consolidation/`
- `results/reinforce_exp3/E31b_coherence_refresh/`
- `results/reinforce_exp3/E31c_functional_coupling_reanalysis/`
- `results/reinforce_exp3/E31d_llama_naturalistic_importance_control/`
- `results/reinforce_exp3/E24_non_si_matched_baseline/`
- `results/reinforce_exp3/E24b_si_vs_non_si_contrast/`
- `results/reinforce_exp3/E25_cross_model_comparability/`
- `results/reinforce_exp3/E27_absolute_threshold_robustness/`
- `results/reinforce_exp3/runs/rexp3_fix_20260429_192038/`
