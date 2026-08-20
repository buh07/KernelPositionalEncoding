# Kernel PE Rebuttal Playbook

This document is a working rebuttal memo for the current NeurIPS submission:

- Paper title: `The Missing Positional Story in LLMs: A Case Study of Shift-Invariant Attention`
- Main manuscript: `paper/Formatting_Instructions_For_NeurIPS_2026/neurips_2026.tex`
- Narrative posture note: `KernelNarrative.md`

The goal is not to defend every speculative idea equally. The goal is to defend the strongest claim hierarchy that the current paper actually supports, answer likely reviewer objections directly, and concede the right things without collapsing the main contribution.

## 1. Core rebuttal posture

The most important move in rebuttal is to keep the claim hierarchy stable.

**Primary claim to defend aggressively**

- In 7-8B primary models, head-level shift-invariant (SI) structure is load-bearing under intervention, and per-head SI amplitude is a strong predictor of disruption cost under a converging specificity-control stack.

**Secondary claims to defend carefully**

- SI-targeted ablation has bounded functional relevance on offset-structured retrieval probes.
- SI exploitation amplitude is heterogeneous across models and appears to be a trainable property rather than a pure architectural given.

**Claims to defend only as hypotheses or framework lenses**

- The frequency-domain interpretation is an organizing lens, not a complete causal mechanism.
- The representational-commitment story is a falsifiable hypothesis, not an established explanation.
- The compressed-sensing-inspired appendix is suggestive framework evidence, not confirmatory proof.

**Things to concede early if challenged**

- The paper does not provide matched-scale 7-8B RoPE-vs-NoPE causal training contrasts.
- The paper does not identify causal drivers of the cross-model SI-amplitude spread.
- Functional transfer is bounded and model-conditional, not universal.
- Collective ablation nonlinearity is not SI-unique.
- The intervention reveals structured sensitivity, not a full mediation decomposition.
- OLMo is supportive on the core intervention stack, but ambiguous on the stronger "non-trivial SI beyond local-bias proxy" controls.

## 2. Non-negotiable rebuttal rules

Use these rules consistently across all reviewer responses.

1. Always describe the main result as a **bounded empirical result**.
2. Separate **load-bearing under intervention** from **exclusive mechanism identification**.
3. Separate **7-8B primary evidence** from **1.1B proxy controls**.
4. Use **model-conditional** language whenever discussing OLMo, strict transfer, naturalistic extension, or local-bias alternatives.
5. Do not try to "rescue" negative or mixed controls. Use them as evidence that the paper is careful and calibrated.
6. Do not imply that SI is the same as general semantic invariance. The paper's own question is that richer semantic SI may be under-realized.
7. Do not claim that RoPE alone explains the effect. One of the strongest supporting observations is that GPT-2 still exhibits measurable SI structure.
8. Distinguish **core within-model intervention evidence** from **extra-model quick anchors** such as Qwen.
9. Treat negative results like H2 reversal and T1 failure as informative constraints, not embarrassments to be hidden.

## 3. One-paragraph global response if needed

If a meta-review or broad review asks what the paper contributes in one paragraph, a good default is:

> The paper's main contribution is a bounded intervention result at the head level. We define a direct, offset-only SI score on pre-softmax logits, intervene by subtracting the estimated SI kernel, and show that disruption scales strongly with SI amplitude under a converging control stack: permutation controls, importance-matched non-SI controls, kernel-transplant tests, and local-bias alternatives. We intentionally do not overclaim beyond that core: broader transfer is mixed and explicitly bounded, collective ablation is not SI-unique, and causal drivers of cross-model heterogeneity remain open. We view the paper as establishing that SI structure is real, measurable, and load-bearing under intervention, while carefully separating that result from stronger claims that the current evidence does not support.

## 4. Likely reviewer concerns and rebuttal strategies

Each section below is written in a practical format:

- **Concern**: what the reviewer may say
- **Best response**: the core answer
- **Evidence to cite**: where to point in the paper
- **What to concede**: what not to fight
- **Avoid saying**: phrases that overclaim or weaken credibility

### 4.1 "This is obvious from RoPE. RoPE already gives relative-position structure."

**Best response**

- The paper does not claim to discover that RoPE mathematically permits offset dependence.
- The paper asks what trained models actually do with that architectural route: whether they exploit it, how load-bearing it is, and how heterogeneous that exploitation is across models.
- The main novelty is empirical and intervention-based: direct SI measurement at the head level plus a converging control stack showing that higher measured SI predicts larger disruption when the SI kernel is removed.

**Evidence to cite**

- Introduction and Setup: the paper explicitly distinguishes capacity from learned exploitation.
- Section 4: the main dose-response result.
- Appendix L, especially RC12-RC15: higher-resolution dose-response, matched non-SI control, local-bias alternative, and kernel-transplant specificity.

**What to concede**

- RoPE makes the existence of an SI channel architecturally plausible by construction.
- The paper's novelty is not in proving RoPE's geometry.

**Avoid saying**

- "We prove a new property of RoPE."
- "RoPE models are shift-invariant."

### 4.2 "Your SI score is just measuring local-distance bias or a trivial offset heuristic."

**Best response**

- The paper already treats this as a serious alternative and does not use `R_h^2` as a standalone mechanism identifier.
- The metric is framed as an offset-conditioned structure score, not a pure SI-mechanism proof.
- Local-bias alternatives were explicitly tested, and the disruption effect tracks non-trivial SI excess beyond local-decay nulls in Llama and Mistral.

**Evidence to cite**

- Setup, "What `R^2` does and does not identify."
- Section 4 and Table of model-by-model specificity.
- Appendix L:
  - RC14: disruption tracks `R^2_excess` over local-decay null in 2/3 models
  - RC19: stronger local-bias family nulls still support 2/3 models

**What to concede**

- OLMo is a non-pass on the stronger local-bias-excess controls.
- OLMo can still be used as a weak directional anchor on the core intervention stack, but not as equally clean evidence that the measured SI effect exceeds simple local-bias proxies.
- The ambiguity is narrowed, not fully eliminated, which the paper already says.

**Avoid saying**

- "We rule out local bias entirely."

### 4.3 "Your intervention is arbitrary. Subtracting an estimated kernel is just a perturbation."

**Best response**

- Yes, it is a perturbation. The paper does not deny this.
- The question is whether this specific perturbation class, targeted to the measured SI component, produces structured and predictable disruption.
- The key evidence is not merely that performance drops, but that drop size scales with measured SI, exceeds permutation and matched non-SI controls, and survives transplant tests.

**Evidence to cite**

- Setup, Intervention subsection: the paper explicitly calls these "intervention sensitivity signatures" rather than a full causal decomposition.
- Section 4:
  - dose-response
  - permutation control
  - matched non-SI control
  - kernel-transplant specificity
- Appendix L / RC4:
  - corpus-transfer and offset-reweighting analyses preserve cross-model ordering with bounded effect-size drift, which narrows the concern that the measured kernel is merely corpus-specific offset structure

**What to concede**

- The intervention does not identify the full downstream redistribution pathway.
- It is not a mediation proof.

**Avoid saying**

- "Subtracting the SI kernel removes the SI computation and nothing else."

### 4.4 "Norm-matched random perturbations are stronger than SI subtraction, so SI is not special."

**Best response**

- This is exactly why the paper downgrades the strongest possible exclusivity interpretation.
- The relevant comparison for the main claim is not "SI subtraction is the largest perturbation possible"; it is "correct-offset SI subtraction is more disruptive than matched permutation and matched non-SI alternatives."
- The norm-matched random result bounds the claim upward and shows the paper is not overselling exclusivity.
- It also emphasizes that the paper is studying a targeted structured perturbation rather than relying on arbitrary large-magnitude noise to produce an effect.

**Evidence to cite**

- Main text Section 4: norm-matched perturbation controls place an upper bound on the effect.
- Appendix L:
  - RC1: true subtraction exceeds permuted control but is weaker than norm-matched random perturbation

**What to concede**

- SI subtraction is not the uniquely strongest way to damage the model under equal-norm perturbation.
- The claim is load-bearing specificity under intervention, not perturbation supremacy.

**Avoid saying**

- "This proves SI is the uniquely causal structure."

### 4.5 "Head selection by top quartile is arbitrary."

**Best response**

- The paper does not rely solely on a single within-model quartile threshold.
- Absolute-threshold robustness checks show the result replicates across tested cutoffs, and fixed-threshold per-head disruption tracks model-level mean SI.

**Evidence to cite**

- Setup section, end of SI operationalization subsection.
- Appendix L:
  - RC11: all 18 threshold x model conditions show true > permuted; fixed-threshold per-head disruption tracks model-level mean `R^2`

**What to concede**

- Any thresholding scheme has some arbitrariness.
- The reason the rebuttal should work is that the qualitative result is threshold-robust.

**Avoid saying**

- "Quartile choice is obviously correct."

### 4.6 "The binning/statistics seem fragile or post-hoc."

**Best response**

- Earlier fragility concerns were already addressed by the higher-resolution 20-bin hardening and exact one-tailed permutation tests.
- The paper explicitly moved away from weaker 5-bin wording and now relies on the stronger RC12 result.

**Evidence to cite**

- Appendix L:
  - RC12: 20-bin dose-response positive in all three models with exact one-tailed permutation testing

**What to concede**

- Earlier lower-resolution binning would have been weaker.
- The paper's current version already adopts the strengthened analysis.

**Avoid saying**

- "The original 5-bin version was already enough."

### 4.7 "Why are the p-values one-tailed?"

**Best response**

- Because the confirmatory question in the dose-response and specificity analyses is directional: higher SI should predict larger disruption, and true SI subtraction should exceed the designated controls.
- That said, the magnitude of the observed effects is large enough in the strongest results that the paper's interpretation does not hang on a marginal sign choice.

**Evidence to cite**

- Section 4 effect sizes and Appendix L RC12-RC15.

**What to concede**

- Some reviewers may prefer two-tailed reporting on principle.
- If space permits in the actual rebuttal, offer to include two-tailed equivalents or note that the large observed positive gaps make the substantive conclusion unchanged.

**Avoid saying**

- "One-tailed is always the correct choice."

### 4.8 "You are double-dipping by using the same measured SI structure to define and perturb heads."

**Best response**

- The paper does not claim an independent supervised prediction benchmark.
- The logic is mechanistic and intervention-based: estimate a structured component, perturb that component, and ask whether its measured amplitude predicts disruption relative to explicit alternative perturbations.
- The crucial anti-circularity evidence is that alternate perturbation schemes do not produce the same pattern: permutation and matched non-SI controls are weaker, and kernel transplants show structure-level specificity.

**Evidence to cite**

- Section 4 core controls.
- Appendix L RC13 and RC15.

**What to concede**

- The measured object and the perturbation are related by design.
- The paper's answer to this is the control stack, not a claim of total measurement independence.

**Avoid saying**

- "There is no circularity concern at all."

### 4.9 "The effect may be due to softmax redistribution or attention sinks, not SI specifically."

**Best response**

- The paper agrees that softmax redistribution is part of the mechanism after perturbation.
- That does not weaken the core result; it clarifies it.
- The relevant question is whether removing the measured offset-only component triggers structured redistribution in a way that scales with SI amplitude and exceeds alternatives. The answer is yes.

**Evidence to cite**

- Setup, Intervention subsection: the paper already limits interpretation to intervention sensitivity signatures.
- Entropy analysis in Setup: true SI subtraction concentrates attention, permuted subtraction does not in Llama/Mistral.

**What to concede**

- The paper does not fully decompose the redistribution pathway.
- Sink-like redistribution is an interpretation consistent with the data, not a complete proof or a uniquely identified pathway.

**Avoid saying**

- "We identify the exact downstream causal chain."

### 4.10 "This only works in Llama and Mistral. OLMo weakens the claim."

**Best response**

- OLMo weakens universality, not the main bounded claim.
- The main paper already frames OLMo as a weak-SI anchor and uses it to show heterogeneity in exploitation amplitude.
- Importantly, OLMo is not a pure contradiction: dose-response, permutation specificity, matched non-SI superiority, and transplant specificity remain directionally positive, though compressed and weaker on some controls.
- The cleanest way to describe OLMo is: supportive on the core intervention stack, but ambiguous on the stronger "non-trivial SI beyond local-bias proxy" controls.

**Evidence to cite**

- Section 4 model table:
  - OLMo still has positive dose-response, positive SI-minus-non-SI, positive transplant-minus-self
- Scope and Robustness Boundaries:
  - explicit model-conditional framing

**What to concede**

- OLMo fails or weakens several stronger supporting controls.
- OLMo should not be used as equally strong evidence for the metric-validity story that the effect exceeds simple local-bias alternatives.
- This is why the paper does not claim universality.

**Avoid saying**

- "OLMo supports everything."

### 4.11 "Llama and Mistral are not independent replications."

**Best response**

- That is fair, and the paper should not present them as fully independent architecture replications.
- The right claim is that the strongest positive evidence is concentrated in two high-SI modern RoPE models, while OLMo serves as a divergent weak-SI anchor.
- The paper also adds a broader 11-model descriptive panel to show that heterogeneity is not reducible to a single pair.
- One architecture difference worth explicitly acknowledging is grouped-query attention (GQA): Llama and Mistral use `32/8` attention/KV heads, while OLMo uses `32/32`. This is one of several uncontrolled cross-model differences and is a reason to emphasize that the core head-level result is within-model and rank-based rather than a clean cross-model causal comparison.

**Evidence to cite**

- KernelNarrative posture and main text discussion of model breadth.
- Section 4 plus Appendix L provenance and breadth controls.

**What to concede**

- Llama and Mistral are close neighbors in the model ecosystem.
- They also share GQA, unlike OLMo, so architecture-level independence is limited.
- Breadth evidence is descriptive, not matched-scale causal replication.

**Avoid saying**

- "Llama and Mistral are fully independent confirmations."

### 4.12 "The 11-model breadth panel is confounded and mixed-scale."

**Best response**

- Correct. The paper already labels it that way.
- The 11-model panel is used for descriptive breadth and heterogeneity, not for matched-scale causal attribution.
- The headline causal/intervention claims remain anchored to the three 7-8B primary models.

**Evidence to cite**

- Main text Section 4 and Scope.
- Appendix L provenance map:
  - breadth extension marked explicitly as mixed-scale descriptive breadth

**What to concede**

- The breadth panel mixes architecture family, scale, tokenizer, training corpus, and optimization.
- It cannot identify drivers of heterogeneity.

**Avoid saying**

- "The 11-model panel isolates the cause of the SI spread."

### 4.13 "Your seed-variance decomposition is only proxy scale."

**Best response**

- Yes, and the paper says so.
- The result is still useful because it addresses one specific concern: whether the observed spread could plausibly be dismissed as mere training-run noise under matched proxy conditions.
- It strengthens, but does not complete, the heterogeneity story.

**Evidence to cite**

- Appendix L:
  - RC5: between-model variance dominates within-model seed variance in the tested proxy setup
- Residual scope note immediately below the control tables

**What to concede**

- This is not a direct 7-8B seed-variance estimate.

**Avoid saying**

- "We have proven seed variance is negligible at 7-8B."

### 4.14 "You still do not have a matched-scale RoPE-vs-NoPE causal contrast."

**Best response**

- Correct. This is an open item, not a hidden weakness.
- The current paper uses proxy-scale RoPE-vs-NoPE and GPT-2 absolute-PE anchors to bound the interpretation:
  - RoPE increases accessible SI capacity in matched proxy settings
  - measurable SI can still be learned without RoPE
- That combination motivates the paper's capacity-versus-exploitation framing, but does not replace a matched-scale causal contrast.

**Evidence to cite**

- Appendix L:
  - RC6: matched proxy RoPE > NoPE
- Main text and Scope:
  - GPT-2 shows non-trivial SI under absolute PE

**What to concede**

- A matched-scale 7-8B causal PE-family comparison remains future work.

**Avoid saying**

- "The proxy result solves the matched-scale causal question."

### 4.15 "GPT-2 also showing SI seems to undermine the RoPE framing."

**Best response**

- It undermines the strongest RoPE-only artifact story, which is good for the paper.
- The intended interpretation is not "only RoPE can produce SI," but "RoPE supplies an architectural route while training can also learn SI-like structure in non-RoPE architectures."
- That is why the paper frames SI exploitation as trainable rather than architecturally guaranteed in magnitude.

**Evidence to cite**

- Introduction and Section 4 discussion of GPT-2 anchors.

**What to concede**

- GPT-2's SI should not be overinterpreted as equivalent to RoPE-mediated SI.
- The paper uses GPT-2 as a learned-computation anchor, not as matched-mechanism evidence.

**Avoid saying**

- "GPT-2 confirms the exact same mechanism."

### 4.16 "The functional result is weak because broader transfer is non-passing."

**Best response**

- Broader transfer non-passing is already incorporated into the claim boundary.
- The paper's functional claim is intentionally narrow: SI-targeted ablation preferentially disrupts controlled offset-structured retrieval, with explicit transfer boundaries.
- This is still useful because it shows the head-level SI result is not purely abstract; it has some behavioral footprint under aligned tasks.

**Evidence to cite**

- Section 5:
  - controlled offset-repetition probe is 3/3 positive
  - stricter format control is mixed
  - broader synthetic family transfer is 0/3
  - naturalistic long-context extension is 2/3
- Scope and Robustness Boundaries:
  - the paper explicitly treats these as bounds

**What to concede**

- The functional result is not a universal downstream generalization law.
- It is strongest on aligned offset-structured retrieval tasks.
- The core controlled probe is 3/3 positive, but strict format survival is only 1/3, so we cannot rule out format sensitivity in Mistral and OLMo.

**Avoid saying**

- "SI ablation broadly disrupts real tasks."

### 4.17 "Strict format controls are mixed, so maybe the functional result is just a formatting artifact."

**Best response**

- Strict format sensitivity is one reason the paper narrows the functional claim.
- The cleanest statement is: the core controlled probe is 3/3 positive, but strict format survival is only 1/3, so we cannot rule out demonstration-format sensitivity in Mistral and OLMo.
- That said, the result is not reducible to formatting alone, because two independent long-context extensions remain positive in the two strongest SI models.

**Evidence to cite**

- Section 5 and Appendix L:
  - RC2: strict format control mixed/model-conditional
  - RC16 and RC18: naturalistic and boundary-grid extensions positive in Llama/Mistral

**What to concede**

- Demonstration format matters in some settings.
- Format sensitivity is a real limitation for the broader functional interpretation outside the controlled probe setting.

**Avoid saying**

- "The functional result is fully format-robust."

### 4.18 "Collective ablation is not SI-specific, so why discuss it?"

**Best response**

- The paper already downgrades it from primary evidence to contextual organizational evidence.
- It is useful because it shows that ordered head removal produces non-linear disruption accumulation, but the matched non-SI controls prevent us from treating that pattern as SI-unique.

**Evidence to cite**

- Scope and Robustness Boundaries, collective-ablation paragraph.
- Appendix L:
  - RC7 and RC8

**What to concede**

- This is not primary mechanism evidence.

**Avoid saying**

- "Collective ablation proves SI-specific organization."

### 4.19 "The frequency-domain story is speculative."

**Best response**

- It is interpretive, but not free-floating.
- The paper ties it to a measurable empirical regularity: richer DFT content in learned SI kernels predicts larger disruption cost.
- The right level of claim is "organizing explanatory lens with predictive value," not "fully identified mechanism."
- A fair caveat is that richer spectral content also means a more structured subtraction target, so the frequency-domain story should be framed as a bounded explanatory lens rather than a closed mechanism account.

**Evidence to cite**

- Discussion section frequency-domain paragraph.
- Appendix figure with kernel examples and spectra.

**What to concede**

- The frequency-domain lens is not a complete mechanistic derivation of all downstream effects.
- Spectral richness may partly track how structured the perturbation itself is; the reason this does not collapse the argument is that the main specificity result is still anchored by permutation, matched non-SI, and transplant controls.

**Avoid saying**

- "The frequency-domain account is proven."

### 4.20 "The compressed-sensing-inspired appendix is weak or out of place."

**Best response**

- This is the easiest thing to concede without harming the paper.
- The CS-inspired appendix is explicitly framework-level and non-confirmatory.
- The main paper does not depend on it for the head-level intervention result.

**Evidence to cite**

- Discussion: CS-inspired lens called supplemental and predictive, not confirmed.
- Appendix H itself: mixed proxy relation, preliminary threshold analogue, small effective `n`

**What to concede**

- If a reviewer dislikes the CS lens, the main empirical claim stands without it.

**Avoid saying**

- "Appendix H is central evidence."

### 4.21 "The representational-commitment hypothesis overreaches."

**Best response**

- The current paper already labels it as a hypothesis and derives falsifiable predictions from it.
- It is included to organize future causal tests, not to elevate a speculation into a result.

**Evidence to cite**

- Discussion section: "not an established mechanism"
- Conclusion: future-work prediction language

**What to concede**

- The commitment story is not established by the current data.

**Avoid saying**

- "We show models have committed SI channels to surface routing."

### 4.22A "Qwen2.5-7B has high SI but the quickcheck does not corroborate the intervention story."

**Best response**

- This is a real and fair question, and the right answer is to narrow what the Qwen result is allowed to mean.
- E29C is explicitly a non-primary quick anchor, not a fourth full intervention replication.
- Concretely, the Qwen script is wiki-only, uses a lightweight per-head SI profile, estimates kernels from a small profile subset, and runs one grouped high-SI subtraction check with a permuted control. It is a directional corroboration check, not a full replication pipeline.
- That is why the paper treats the Qwen result as bounded extra-model context rather than evidence that can overrule the primary trio.

**Evidence to cite**

- Appendix L:
  - RC17: "Not supported" and no claim promotion
- Main text Scope/Open limitations:
  - "the Qwen quickcheck does not directionally corroborate"
- The actual E29C setup:
  - non-primary, wiki-only, lightweight quick anchor with one grouped high-SI subtraction check and lightweight kernel estimation

**What to concede**

- A careful reviewer is right that Qwen is the most salient extra-model anomaly in the current package.
- Because E29C is not a matched full replication, it should not be used to argue either for or against the primary intervention result as strongly as the three primary models.

**Avoid saying**

- "Qwen doesn't matter."
- "Qwen is just noise."

### 4.22B "Experiment 2 H2 is reversed. Doesn't that undermine the spectral/frequency story?"

**Best response**

- H2 reversal is a real negative result and should be treated as informative, not hidden.
- The clean interpretation is that trained models do not exhibit the simple "low-frequency channels are functionally specialized for long-range dependencies" story.
- This weakens a naive spectral-specialization view, but it is still consistent with the paper's broader framing that current SI allocation does not match the clean theoretical decomposition one might have expected from RoPE geometry alone.

**Evidence to cite**

- Appendix Experiment 2:
  - H2 is reversed across confirmatory, feasibility, and attenuation branches
  - manuscript text explicitly states "Low-frequency channels serve all dependency ranges"

**What to concede**

- The simplest low-frequency-equals-long-range interpretation is not supported.
- The frequency-domain lens should therefore be used as an organizing lens on learned structure, not as a strong functional specialization law.

**Avoid saying**

- "Experiment 2 confirms the spectral specialization story."

### 4.22C "T1 fails while T8 succeeds. Doesn't that weaken the mechanistic interpretation?"

**Best response**

- T1 failure is not ideal, but it is already explained by the paper's redundancy interpretation.
- The paper's mechanistic claim is collective and intervention-based: removing the measured SI component from many high-SI heads is costly.
- T1 asks a different question: whether a single head's SI score predicts task performance on its own. Under distributed redundancy, that correlation can be near zero even when collective ablation is costly.

**Evidence to cite**

- Appendix Experiment 3:
  - T1 not supported, T8 supported
  - explicit note on T1 vs T8 says this pattern is structurally consistent with distributed redundancy
- Appendix Experiment 3 Phase 2 / C.1:
  - linear depletion rejected, supporting a collective non-linear redundancy story

**What to concede**

- T1 failure means the paper should not claim that SI score is a strong per-head task-performance predictor.
- The strongest claim is about collective structured sensitivity under intervention, not per-head task predictiveness.

**Avoid saying**

- "T1 is irrelevant."

### 4.22D "GQA makes the cross-model comparison hard to interpret."

**Best response**

- This is a legitimate cross-model comparability caveat and should be acknowledged directly.
- Llama and Mistral use grouped-query attention with `32/8` attention/KV heads, while OLMo uses `32/32`.
- That means GQA is one more uncontrolled architecture difference in the primary trio, alongside corpus, tokenizer, normalization, and optimization.
- The reason the core result still stands is that the main evidence is within-model and rank-based: per-head SI amplitude predicts disruption within each model under matched controls.

**Evidence to cite**

- Local model configs:
  - Llama-3.1-8B `num_attention_heads=32`, `num_key_value_heads=8`
  - Mistral-7B-v0.1 `32/8`
  - OLMo-2-7B `32/32`
- Main text Scope:
  - cross-model causal attribution remains open because many factors vary jointly

**What to concede**

- GQA limits how strongly one should interpret absolute cross-model SI-amplitude differences causally.
- It does not undercut the within-model dose-response and control-stack evidence.

**Avoid saying**

- "GQA has no bearing on the comparison."

### 4.22 "The paper is too broad and tries to do too many things."

**Best response**

- That concern is understandable because the project includes measurement, intervention, functional probes, breadth checks, and interpretive appendices.
- The clearest response is to re-anchor the paper around one result: head-level SI specificity under intervention.
- Everything else should be described as scope-bounding, breadth-extending, or framework-organizing rather than equally central.

**Evidence to cite**

- Contribution list already prioritizes the head-level result first.
- Scope and Robustness Boundaries section already narrows secondary claims.

**What to concede**

- The paper has many appendices.
- The rebuttal should help reviewers separate primary from supporting evidence.

**Avoid saying**

- "Every section is equally central."

### 4.23 "What is actually novel here relative to prior head-importance or pruning work?"

**Best response**

- The novelty is not generic head importance.
- It is the pairing of:
  - a direct offset-only SI measurement on logits,
  - SI-targeted kernel subtraction,
  - dose-response between measured SI and disruption,
  - and a control stack aimed specifically at ruling out generic importance and trivial offset artifacts.
- Prior head-pruning work does not usually isolate a learned offset-only kernel and test whether its amplitude predicts targeted intervention damage in this way.

**Evidence to cite**

- Setup and Section 4.
- Appendix L RC13-RC15.

**What to concede**

- The paper builds on standard intervention logic from mechanistic interpretability.

**Avoid saying**

- "No prior work studies head importance or position-conditioned behavior."

### 4.24 "The paper's theoretical framing does not match its actual evidence."

**Best response**

- The paper intentionally distinguishes between direct empirical evidence and framework-level interpretation.
- The evidence for the main claim is intervention-based and does not require the full theoretical stack to be decisive.
- Where the theory is only suggestive, the manuscript already says so.

**Evidence to cite**

- Discussion: empirical findings separated from interpretive framework.
- Scope and Open limitations.

**What to concede**

- Some appendices are more about interpretation and future prediction than about evidentiary closure.

**Avoid saying**

- "The theory is fully validated."

### 4.25 "How does this relate to representation-learning methods like SAEs or MSAEs?"

**Best response**

- The current paper studies an operator-level object: a head-specific offset-only component of pre-softmax logits.
- Direct kernel extraction is therefore lower-assumption for the main claim than training a representation model and then inferring shift-invariance from latent structure.
- An SAE/MSAE-style decomposition could be a valuable follow-up for a different question: where SI structure lives in the representation and how it entangles with content. But it would still need a direct SI metric like the current one to verify actual shift-invariance rather than just position sensitivity.

**Evidence to cite**

- Setup equation `A_h(i,j) = g_h(i-j) + r_h(i,j)` and the intervention setup.

**What to concede**

- Representation-learning tools could complement this paper in future work.
- They are not a replacement for directly measuring the pairwise kernel object that this paper's primary claim is about.

**Avoid saying**

- "SAE/MSAE methods are irrelevant."

### 4.26 "The paper needs a cleaner statement of what it does not show."

**Best response**

- This is a good-faith concern, and the paper already contains much of that language.
- In rebuttal, it helps to restate the three most important non-claims:
  - we do not prove universal SI transfer,
  - we do not identify the causal driver of cross-model heterogeneity,
  - we do not prove that SI is the unique or complete mechanism of the affected behaviors.

**Evidence to cite**

- Scope and Robustness Boundaries.
- Conclusion.

**What to concede**

- Clarity can always be improved; reviewers may be reacting to breadth rather than actual overclaiming.

**Avoid saying**

- "The manuscript already makes this perfectly obvious."

## 5. Reviewer-type specific guidance

### 5.1 If a reviewer is positive on novelty but worried about soundness

Lead with:

- the direct object of measurement,
- the control stack,
- the downgraded interpretation of random perturbation,
- the explicit non-claims.

Do not lead with:

- the representational-commitment story,
- Appendix H,
- broad future-work aspirations.

### 5.2 If a reviewer is positive on soundness but worried about significance

Lead with:

- this is one of the few papers to directly quantify and intervene on an offset-only component at the head level,
- the result is not merely descriptive because intervention damage scales with measured SI,
- the model heterogeneity result raises a concrete pretraining-design question.

### 5.3 If a reviewer is skeptical because of mixed transfer results

Lead with:

- the functional result is explicitly bounded,
- the head-level specificity result does not depend on universal behavioral transfer,
- negative controls and bounded transfer are signs of calibration, not collapse.

### 5.4 If a reviewer is skeptical because of breadth/confounding

Lead with:

- the breadth panel is descriptive,
- the causal/intervention claims are anchored to the 7-8B trio,
- the proxy controls are labeled as proxy controls,
- and GQA vs non-GQA is one of several uncontrolled architecture differences that limit causal cross-model interpretation.

## 6. Fast rebuttal snippets for common objections

These are intentionally short and can be adapted directly into the actual rebuttal.

### 6.1 "RoPE already implies this."

RoPE implies SI capacity, not the degree to which trained models exploit that capacity. Our main result is empirical and intervention-based: per-head SI amplitude predicts disruption under SI-kernel subtraction, and that relation survives permutation, matched non-SI, transplant, and local-bias controls.

### 6.2 "Your metric is just local bias."

We do not treat `R_h^2` as a pure mechanism identifier. The relevant test is whether disruption tracks non-trivial SI excess beyond simple local-decay alternatives. That is supported in Llama and Mistral under both the basic and richer local-bias families, and explicitly non-universal in OLMo.

### 6.3 "This is just a generic perturbation."

The paper agrees that SI subtraction is a perturbation. The key point is specificity: correct-offset subtraction is more disruptive than offset permutation and importance-matched non-SI controls, and transplanting high-SI kernels onto low-SI heads is more disruptive than low-head self subtraction.

### 6.4 "Transfer is weak."

The transfer claim is intentionally narrow. We support bounded functional sensitivity on offset-structured retrieval probes and explicitly report non-passing broader synthetic transfer plus mixed strict-format controls. Those negative results bound the claim; they do not overturn the head-level specificity result.

### 6.5 "The cross-model story is too confounded."

We agree that causal attribution of the SI-amplitude spread remains open. That is why the paper separates 7-8B intervention evidence from mixed-scale breadth description and labels the seed and RoPE-vs-NoPE controls as proxy scale.

### 6.6 "What about the Qwen anomaly?"

Qwen is a real extra-model anomaly, but it is not a fourth full replication. E29C is a non-primary quick anchor: wiki-only, lightweight SI profiling, lightweight kernel estimation, and one grouped high-SI subtraction check with a permuted control. That is why the paper treats it as bounded extra-model context rather than evidence that can overrule the primary trio.

### 6.7 "H2 is reversed, so doesn't that break the spectral story?"

H2 reversal is a real negative result. The clean takeaway is that trained models do not exhibit the simple low-frequency-equals-long-range specialization story. That weakens a naive spectral-specialization view, but it is still consistent with the paper's broader claim that current SI allocation does not match the clean theoretical decomposition one might have expected from RoPE geometry alone.

### 6.8 "T1 fails while T8 succeeds."

Those results address different levels of organization. T1 asks whether single-head SI scores predict task performance on their own; T8 asks whether collectively removing the measured SI component from many high-SI heads is costly. Under distributed redundancy, T1 can fail while T8 still succeeds, which is exactly how the paper interprets this pattern.

### 6.9 "GQA makes the cross-model comparison hard to interpret."

That is a fair caveat. Llama and Mistral use `32/8` attention/KV heads while OLMo uses `32/32`, so GQA is one more uncontrolled architecture difference in the trio. The reason the core claim still stands is that the main evidence is within-model and rank-based: per-head SI amplitude predicts disruption within each model under matched controls.

## 7. Things the rebuttal should proactively say before reviewers force them

These sentences increase trust because they show the paper is not hiding its weak points.

- "We agree that the strongest supported claim is the head-level intervention result, not a universal behavioral generalization claim."
- "We agree that the 11-model panel is descriptive breadth, not matched-scale causal attribution."
- "We agree that the proxy-scale seed and RoPE-vs-NoPE controls do not replace a matched 7-8B causal contrast."
- "We agree that the random-perturbation upper-bound result narrows the exclusivity interpretation; our claim is load-bearing specificity under intervention, not perturbation maximality."
- "We agree that OLMo bounds generality and is best read as a compressed directional anchor rather than a full replication, especially on the stronger local-bias-excess controls."
- "We agree that the core controlled probe is 3/3 positive, but strict format survival is only 1/3, so broader functional interpretation must remain tightly bounded."
- "We agree that Qwen is a non-primary quick anchor and not a fourth full intervention replication."
- "We agree that H2 reversal is a real negative result and should be interpreted as a constraint on the spectral story, not hidden."

## 8. Things the rebuttal should not say

- "We prove that modern LLMs are shift-invariant."
- "We isolate the causal mechanism of SI exploitation."
- "We rule out local bias."
- "We establish broad downstream utility."
- "We identify why Llama and Mistral exploit SI more than OLMo."
- "The compressed-sensing appendix confirms the mechanism."
- "RoPE is the reason for all observed SI structure."
- "GPT-2 demonstrates the same mechanism as RoPE."

## 9. If there is room for one additional analysis after reviews arrive

If time permits a small follow-up analysis, prioritize something that strengthens the main result rather than expanding scope.

Best options:

1. Report two-tailed versions of the key RC12-RC15 statistics if a reviewer objects to one-tailed reporting.
2. Add a compact leave-one-dataset-out or equal-count-offset robustness summary for `R_h^2` if measurement validity becomes a central reviewer concern.
3. Add a concise cross-model fixed-threshold summary sentence from RC11 in the rebuttal if a reviewer attacks quartile selection.
4. Add a concise per-layer SI concentration summary for the three primary models if a reviewer presses on where the high-SI heads live or whether the result is driven by a narrow layer band.
5. If compute permits, run a stronger non-primary intervention replication on Gemma-2-9B or a fuller Qwen follow-up that is more than a quickcheck; this is much higher value than spending rebuttal budget on Appendix H.

Do not spend rebuttal budget trying to rescue:

- universal transfer,
- matched-scale PE-family causality,
- or collective SI-unique organization.
- The CS appendix should also stay low priority unless a reviewer makes it central; the main empirical claim does not depend on it.

Those are not where the paper is strongest.

## 10. Optional future-work bridge to MSAE-style analysis

If a reviewer asks how to make the mechanism story sharper, one good answer is:

- a follow-up representation-level analysis could train an MSAE-style position/content decomposition on `Q`, `K`, or head outputs;
- reconstruct branchwise logit terms such as position-position vs mixed terms;
- and test whether the measured SI kernel is carried by a clean positional branch or by entangled mixed structure.

This is worth mentioning only as future work. It should not replace the paper's main defense, because the current paper's main object is a directly measured pairwise kernel on logits rather than a learned latent representation.

## 11. Final recommended rebuttal tone

The best tone for this paper is:

- calm,
- precise,
- selective,
- and transparent about scope.

The paper gets stronger when it says:

- "Here is the exact thing we show,"
- "Here is the exact thing we do not show,"
- and "Here is why the thing we do show is still meaningful."

That is the tone most likely to preserve credibility with skeptical reviewers while keeping the primary result intact.
