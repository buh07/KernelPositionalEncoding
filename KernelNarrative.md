# Kernel PE Narrative (Finalized Posture)

**Version:** May 4, 2026  
**Scope:** Narrative guidance synced to finalized controls and E28/E29/E30/E31 hardening bundle (E28A/B/C/D, E29A/B/C, E30A/B, E31A/B/C/D).

**Reproducibility note:** The submission package includes an anonymized GitHub repository with the code, scripts, environment specification, and artifact-mapped commands used to reproduce claim-bearing results.

## 1) One-sentence thesis

RoPE exposes an understudied shift-invariant channel, trained models use that channel in load-bearing but heterogeneous ways, and the resulting evidence supports a bounded case-study narrative: strong head-level specificity (Result I), non-SI-unique collective context (Result II), bounded functional probe sensitivity (Result III), broader heterogeneity beyond the primary trio (E31A), and a CS-inspired framework lens for future prediction (still mixed in expanded refresh, E31B).

## 2) Claim hierarchy (what is strongest)

1. **Primary claim (Result I):** permutation-specific SI-kernel subtraction disruption is real and load-bearing; E28A/B/D plus E29A and E30B strengthen head-level specificity and anti-tautology evidence.
2. **Supporting organizational observation (Result II):** ranked cumulative ablation robustly rejects linear depletion, but matched non-SI controls show this collective nonlinearity is mostly generic.
3. **Supporting scoped functional finding (Result III):** controlled offset-repetition probe sensitivity is directionally positive in all primary models; strict confound control is model-conditional, broader probe-diversity transfer is non-passing, natural-text long-context transfer is 2/3 model-conditional, and boundary-grid transfer is also 2/3 with non-universal factor directionality.
4. **Supporting breadth/coupling extension (E31A/C):** expanded model breadth preserves large SI heterogeneity; model-level SI amplitude directionally tracks preferential functional disruption on tested probe suites.
5. **Supporting theoretical lens (Appendix H + E31B):** corrected Exp7A is mixed at model level (proxy geometry, small effective n), corrected Exp7B remains strongly directional under canonical single-condition calibration, and E31B expanded refresh is mixed; treat the CS stack as framework-level, not mechanism-proof.

## 3) Post-E28 interpretation anchors

### E28A
- 20-bin exact one-tailed permutation testing removes the old small-n fragility.
- Use exact values directly (Llama/Mistral `p=5.0e-06`, OLMo `p=7.05e-04`).

### E28B
- SI-ranked intervention exceeding importance-matched non-SI control in all 3 models is the cleanest anti-"generic important heads" evidence.

### E28D
- `R^2_excess` tracking supports non-trivial SI beyond local-decay null in Llama/Mistral, but not OLMo.
- Frame as **model-conditional strength**, not universal.

### E29A
- Kernel-transplant specificity supports SI structure-level relevance beyond low-head self-kernel subtraction.
- Use as direct anti-tautology hardening: transplanted high-SI kernels on low heads are more disruptive than low-head self subtraction in all three primary models.

### E29B
- Semi-naturalistic long-context passkey families (wiki/code, 256--512 contexts) support preferential SI disruption in Llama and Mistral, not OLMo (2/3 pass).
- Use as bounded functional extension, not universal downstream proof.

### E29C
- Qwen2.5-7B quick anchor is non-primary and non-corroborating in this run.
- Keep as directional-only context with no claim promotion.

### E30A
- Boundary-grid extension (short/long × high/low regularity) reproduces the same model split as E29B: Llama/Mistral pass, OLMo non-pass (2/3).
- Do not claim a universal factor law: regularity and length contrasts are not directionally consistent across models.

### E30B
- Local-bias null-family decomposition (exponential/power-law/window) retains non-trivial excess tracking support in 2/3 models.
- Use as a stronger metric-validity control that complements E28D.

### E31A
- Breadth consolidation over 11 models / 7 families shows large SI-amplitude heterogeneity (max/min mean R² ratio `140x`).
- Use as descriptive breadth extension, not causal attribution.

### E31B
- Expanded proxy coherence-gap refresh is mixed (`r=+0.577`, `rho=+0.500`, non-significant at conventional thresholds).
- Keep CS lens as potential framework; do not promote as confirmatory explanatory law.

### Exp7A / Exp7B (corrected CS audit posture)
- Exp7A is explicitly proxy-based (analytic PE metadata descriptors, not learned PE-matrix extraction) and should be read at model-level effective n:
  - model-level (`n=6`): Pearson `r=+0.731` (`p=0.099`), Spearman `rho=+0.516` (`p=0.295`), two-sided permutation `p=0.141`
  - 36-row table remains exploratory/pseudo-replicated context only
- Exp7B now matches manuscript-calibration wording in canonical outputs:
  - single-condition calibration `C=7.057` on Llama-3.1-8B (seq-len 256)
  - six-condition fit: Pearson `r=-0.959`, Spearman `rho=-0.878`
  - out-of-sample (`n=5`, excluding calibrated condition): Pearson remains strong (`r=-0.986`), Spearman directional/marginal (`rho=-0.866`)
- Global-fit reference (`C=7.834`) is retained as sensitivity context, not the canonical calibration mode.

### E31C
- Reanalysis across existing probe suites (E12/E29B/E30A) shows strong directional coupling between model mean R² and preferential functional gap.
- Use as bounded “so-what” linkage, explicitly within tested probes/models.

### E31D
- Llama naturalistic add-on: SI-targeted ablation > importance-matched non-SI and > permuted controls across 60 prompts.
- Use as bounded practical extension for Result III (single-model add-on, not universal transfer).

### E28C
- Broader 7-family transfer non-passing (`0/3`) is a scope bound, not a failure of all SI functionality.
- Keep Result III explicitly “controlled probe” rather than naturalistic ICL generalization.

## 4) Tokenizer story (how to state it)

Use tokenizer mediation as a **model-conditional implementation story**, not as the causal explanation of cross-model SI amplitude heterogeneity.

- Supported: Llama boundary-linked SI behavior is tokenizer-mediated in strict diagnostics.
- Not supported: tokenizer overlap alone explains the 6.5x SI amplitude spread.
- Practical phrasing: tokenizer structure is a routing substrate for SI usage in some models; it does not by itself explain why different models amplify SI to different levels.

## 5) Representational commitment (status)

Treat representational commitment as a **falsifiable synthesis hypothesis**, not an established mechanism.

Operational statement:
- Current evidence is consistent with SI channels being stably allocated toward sparse positional/surface routing under current training trajectories.
- Short-horizon adaptation has limited retargeting signal.
- Broader semantic SI use likely requires pretraining-era shaping pressure.
- Any resulting downstream performance improvement is a future-work hypothesis, not a finding of this paper.

Do **not** claim this hypothesis is causally proven for 7–8B production models.

## 6) What to avoid

- Avoid “SI-specific collective organization is established.”
  - E24/E24b prevent this.
- Avoid broad “SI drives ICL generally.”
  - E28C bounds transfer; keep “controlled offset-repetition probe sensitivity.”
- Avoid “RoPE vs NoPE solved at 7–8B.”
  - E21 is matched **1.1B proxy** evidence.
- Avoid claiming full intervention propagation mechanism is established.
  - Current evidence is load-bearing sensitivity under position-dependent subtraction; full softmax-redistribution diagnostics are still pending.
- Avoid implying Llama and Mistral are independent architecture replications.
  - They are close neighbors; OLMo is a weak-SI divergent anchor.

## 7) Recommended narrative arc in paper text

1. **Question:** RoPE provides SI capacity; what do trained models do with it?
2. **Empirical surprise:** 6.5x amplitude spread, not seed noise (E20).
3. **Mechanistic intervention evidence:** SI-kernel disruption is real and hardens under E28A/B/D.
4. **Specificity hardening:** kernel-transplant specificity (E29A) shows SI-structured perturbation relevance beyond generic low-head self perturbation.
5. **Metric-validity hardening:** richer local-bias-family nulls keep 2/3 support for non-trivial excess tracking (E30B), with OLMo as bounded non-pass.
6. **Organization caveat:** ranked-ablation nonlinearity is robust but not SI-unique (E24/E24b).
7. **Functional scope:** controlled probe sensitivity is positive; strict control and transfer results bound scope (E18/E28C/E29B/E30A), with a single-model naturalistic add-on strengthening bounded relevance (E31D).
8. **Breadth extension:** heterogeneity extends across a wider multi-family panel (E31A); functional coupling is directionally positive in tested suites (E31C).
9. **Synthesis:** representational commitment as forward hypothesis and prediction target, with CS lens kept framework-level due mixed expanded coherence refresh (E31B).

## 8) Canonical phrasing snippets

- **Result I:** “Permutation-specific disruption cost is load-bearing under intervention, with strengthened head-level specificity from E28A/B/D plus kernel-transplant and richer local-bias controls (E29A/E30B).”
- **Result II:** “Ranked-ablation linear rejection is robust, but matched non-SI controls indicate this collective nonlinearity is not uniquely SI.”
- **Result III:** “SI intervention preferentially disrupts controlled offset-repetition retrieval probes; strict confound survival is model-conditional and broader probe-family transfer is non-passing.”
- **Result III extension:** “Natural-text long-context and boundary-grid transfer are partially supported (2/3), reinforcing bounded functional relevance rather than universal downstream claims.”
- **Capacity caveat:** “Matched proxy-scale TinyLlama evidence supports RoPE>NoPE SI separation; matched 7–8B causal training contrasts remain open.”

## 9) Non-promoted context (keep brief)

- **E22/E23** remain context/future-work; do not use as claim-bearing evidence for main Results I–III in this cycle.
- If mentioned, explicitly mark as exploratory and scale-limited.
