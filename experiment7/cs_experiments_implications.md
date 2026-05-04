# CS-Theoretic Experiments and Practical Implications

## Overview

This document proposes two concrete experiments that directly reinforce the compressed sensing interpretation of shift-invariant positional kernel structure, and then develops the practical implications for PE design, model architecture, and mechanistic interpretability tooling. The experiments are designed to be **low overhead** — they reuse the existing measurement infrastructure from the six-experiment program — while generating **high-leverage theoretical confirmation** of the CS framework.

---

## Experiment A: The Welch Bound Predicts the R² Ceiling

### Motivation

The companion theory document establishes that the maximum achievable shift-invariant R² is bounded by the mutual coherence of the positional encoding matrix. This is a falsifiable quantitative prediction, not just a qualitative narrative. Experiment A tests it directly.

### Design

**Step 1: Compute coherence profiles analytically.** For each model in the existing study (Llama-3.2-1B, OLMo-1B, TinyLlama, GPT-2 small/medium), extract the RoPE or sinusoidal frequencies $\{\omega_i\}$ and compute the Gram matrix $\mathbf{G} \in \mathbb{R}^{N \times N}$ with entries:

$$G_{ij} = \frac{1}{D} \sum_{k=1}^D \cos(\omega_k (i - j))$$

Report the **coherence profile** $\mu(\Delta) = |G_{i,i+\Delta}|$ as a function of offset $\Delta$. This is purely analytical — no model inference required. For GPT-2's absolute sinusoidal PE, the same formula applies. For the NoPE control, the coherence profile is trivially zero (serving as the ground-truth baseline).

**Step 2: Compute the Welch bound gap.** For each model with $N$ positions and $d_{\text{head}}$ dimensions, the Welch bound is:

$$\mu_{\text{Welch}} = \sqrt{\frac{N - d_{\text{head}}}{d_{\text{head}}(N-1)}}$$

Report the **Welch gap ratio** $\eta = \mu(1) / \mu_{\text{Welch}}$ — the ratio of adjacent-position coherence to the theoretical minimum. This single number quantifies how far the PE geometry is from CS-optimality.

**Step 3: Test the coherence-to-R² prediction.** From the existing Experiment 1 data, compute the content-to-position variance ratio $\sigma^2_c / \sigma^2_p$ per model (estimable from the residual variance of the $g(\Delta)$ fit). Apply the theoretical bound:

$$R^2_{\text{predicted}} = 1 - \frac{\sigma^2_c / \sigma^2_p}{1 + \sigma^2_c / \sigma^2_p} \cdot \mu^2(1)$$

Compare $R^2_{\text{predicted}}$ against the observed early-layer R² for each of the 36 model × dataset × length combinations. A Pearson correlation $r > 0.80$ would constitute strong confirmation.

**Step 4: Design a CS-optimal PE and measure its coherence profile.** Construct a comparison PE matrix by selecting $D$ frequencies to **minimize maximum pairwise coherence** subject to the BOS constraint (K = 1). The optimization problem is:

$$\min_{\{\omega_k\}_{k=1}^D} \max_{\Delta \in \{1, \ldots, N-1\}} \left| \frac{1}{D} \sum_k \cos(\omega_k \Delta) \right|$$

This can be solved approximately via greedy frequency selection (Tropp & Gilbert 2007) or by gradient descent on the coherence objective. Compute the coherence profile of this optimized PE and show that $\eta_{\text{opt}} < \eta_{\text{RoPE}}$ — the optimized PE approaches the Welch bound more closely.

### Hypotheses

- **H_A1:** $R^2_{\text{predicted}}$ from the coherence bound correlates with observed R² across all 36 experimental combinations ($r > 0.75$, one-sided).
- **H_A2:** Adjacent-position coherence $\mu(1)$ is negatively correlated with early-layer R² across models (more coherent PEs produce lower SI ceiling). Expected: Llama-3.2-1B has the highest R² and lowest adjacent coherence among RoPE models.
- **H_A3:** The NoPE control shows coherence profiles consistent with random embeddings (low $\mu$ for all $\Delta$), but *also* shows the lowest R² — confirming that the causal mechanism is PE structure, not coherence per se.

### Expected Outcome and Significance

The coherence bound converts the "strong-form hypothesis not met" finding into a **confirmed quantitative prediction**: the R² ceiling is not a failure of the model to learn positional structure, but a mathematical consequence of the PE geometry. This is the kind of result that moves a finding from descriptive ("R² is surprisingly low") to explanatory ("R² is exactly as low as the PE's Welch gap predicts"), substantially elevating the paper's theoretical contribution.

A secondary output is the coherence profiles themselves, which give a **per-model fingerprint** of positional distinguishability as a function of offset — a novel visualization connecting PE design to empirical model behavior that could appear as a main-body figure.

### Computational Cost

This experiment is almost entirely analytical. Computing the $N \times N$ Gram matrix for $N = 1024$ is O(N² D) floating-point operations — on the order of milliseconds on a CPU. The greedy frequency optimization is $O(D \cdot N \cdot T_{\text{iter}})$ with $T_{\text{iter}} \approx 500$ iterations. Total compute: **less than 1 GPU-hour**. The only non-trivial computation is the correlation analysis against the 36 existing experimental combinations, which requires re-processing saved logit tensors from Experiment 1.

---

## Experiment B: The CS Phase Transition Predicts the Threshold Capacity Boundary

### Motivation

Experiment 3P2-C established that the threshold capacity model is preferred over linear degradation (6/6 criterion votes, both models). The CS phase transition gives a *quantitative* prediction for where the threshold lies and how it scales with context length and attention sparsity — predictions that the original experiment did not test.

### Design

**Step 1: Characterize baseline effective sparsity $s$.** For each of Llama-3.1-8B and OLMo-2-7B, compute the **effective attention sparsity** of each head: the average number of positions receiving more than $\epsilon = 0.01$ of the total softmax probability mass. Call this $s_{\text{eff}}(h)$ for head $h$. This is computable from existing activation data — no new inference runs required.

**Step 2: Predict the critical threshold $m^*$.** Under the CS phase transition, the minimum number of high-SI heads needed for reliable positional recovery scales as:

$$m^* \approx C \cdot s_{\text{eff}} \cdot \log(N / s_{\text{eff}})$$

where $C \approx 2$–$4$ (a CS-universal constant). Compute the predicted $m^*$ for each model at both $N = 256$ and $N = 1024$ (the two sequence lengths in Experiment 1), giving four predictions in total.

**Step 3: Run threshold ablation at multiple context lengths.** Extend the progressive ablation from Experiment 3P2-C across three context lengths: $N \in \{256, 512, 1024\}$. At each length, ablate high-SI heads in order (highest R² first) from 0 to 80% of the total, recording language modeling loss at each step. Fit both a threshold model and a linear model to each curve.

The CS prediction is:
1. The transition point (critical threshold $m^*$) shifts **upward with $N$** — longer contexts require more heads for reliable positional recovery.
2. The scaling should be **log-linear**: plotting $m^*$ vs. $\log(N / s_{\text{eff}})$ should produce a straight line.
3. The sharpness of the transition (width of the degradation cliff) should be **narrower at longer contexts**, because the phase transition becomes sharper as the problem dimension grows.

**Step 4: Compare predicted vs. observed transition points.** For each (model, context length) combination, read off the empirically observed $m^*$ as the head count at the inflection point of the loss-vs.-ablation curve. Compute Spearman correlation between predicted $m^*$ (from $s_{\text{eff}}$ and $N$) and observed $m^*$ across all conditions.

### Extended variant: measuring the full phase diagram

The Donoho-Tanner phase diagram plots recovery probability as a function of both sparsity fraction $\rho = s/N$ and measurement fraction $\delta = m/N$. A neural analogue can be constructed by:

1. Varying context length $N \in \{128, 256, 512, 1024, 2048\}$ (varying $\delta = m/N$ for fixed $m$)
2. Varying the number of ablated high-SI heads $m \in \{0.1H, 0.2H, \ldots, 0.9H\}$ (varying $\delta$)
3. Measuring per-position prediction accuracy on the synthetic key-match task (which has known ground-truth $s_{\text{eff}}$)

This produces a 2D heatmap over $(\delta, \rho)$ space where the phase boundary — the curve separating good and poor positional recovery — should resemble the Donoho-Tanner curve. If it does, the CS framework is not merely analogous but **quantitatively predictive** of transformer positional processing.

### Hypotheses

- **H_B1:** The observed transition point $m^*$ increases with $N$ (log-linear scaling), consistent with CS measurement requirements.
- **H_B2:** The transition sharpens with $N$ — the coefficient of variation of the loss at the transition point decreases as context length increases.
- **H_B3:** $m^*$ predicted from $s_{\text{eff}} \cdot \log(N/s_{\text{eff}})$ correlates with observed $m^*$ across (model, context length) pairs ($\rho > 0.7$).
- **H_B4 (exploratory):** The 2D phase diagram, if constructed, shows a phase boundary with a shape consistent with the Donoho-Tanner curve.

### Expected Outcome and Significance

If confirmed, this experiment establishes that the **CS measurement redundancy formula is a quantitative theory of positional head capacity** in transformers. The threshold finding (3P2-C) is currently described as an organizational principle without a first-principles explanation. The CS phase transition provides that explanation, converts it into a scaling law, and makes it predictive across model and context-length variations.

This also directly motivates a design principle: to support a context window of $N$ tokens with attention effective sparsity $s$, a model needs at least $C \cdot s \cdot \log(N/s)$ heads dedicated to high-quality positional sensing. This is a concrete architectural guideline derivable from CS theory.

### Computational Cost

Experiment B requires new model inference runs for the ablation sweeps at three context lengths. Estimated cost:

- Effective sparsity computation: reuse existing activation data. ~0 new GPU-hours.
- Progressive ablation at $N \in \{256, 512, 1024\}$, 9 ablation steps, 2 models: approximately 18 eval runs. ~**6 GPU-hours** on A100.
- Extended phase diagram (optional): ~20 additional conditions per model. ~**15 GPU-hours**.

Total required: 6 GPU-hours (core), 21 GPU-hours (with phase diagram). Both are within the scope of a paper revision.

---

## Practical Implications

### 1. A Principled Framework for Positional Encoding Design

The current state of PE design is almost entirely empirical: RoPE, ALiBi, YaRN, and their extensions were developed through intuition and ablation, without a principled objective. The CS framework provides the missing design objective: **minimize the mutual coherence of the PE matrix** subject to the BOS constraint.

This suggests three immediately actionable design alternatives:

**Coherence-minimizing frequency selection.** Instead of geometric RoPE frequencies ($\theta_i = 10000^{-2(i-1)/d}$), select frequencies to minimize adjacent-position coherence. The optimization is fast (< 1 GPU-hour), frequencies are fixed at initialization, and the rest of the architecture is unchanged. The prediction is that models trained with coherence-minimizing frequencies will achieve higher SI R², cleaner spectral utilization profiles, and better length generalization — because the PE sensing matrix will be closer to CS-optimal.

**Welch-bound-normalized PE scaling.** A direct consequence of the Welch bound is that the PE dimension $d$ must scale as $\Omega(s \log N)$ for reliable positional recovery over $N$ positions with $s$-sparse attention. Current practice fixes $d$ irrespective of $N$ — which is why all existing PE methods fail at sufficiently long contexts. A CS-derived scaling rule $d_{\text{min}}(N) = C \cdot s_{\text{eff}} \cdot \log N$ could serve as a design principle for long-context models, motivating adaptive or growing PE dimensions.

**Tight-frame PE initialization.** Initializing the PE matrix as (an approximation to) a tight frame — $\mathbf{P}^\top \mathbf{P} = (N/d)\mathbf{I}$ — ensures equal measurement capacity across all positional directions, directly analogous to the Parseval Networks regularizer for convolutional weights. This is implementable as a regularization loss during pretraining at minimal compute cost.

### 2. Explaining and Fixing Length Generalization Failure

The CS framework provides the first **causal account** of why length generalization fails. When a model trained on context length $N_{\text{train}}$ is applied to $N_{\text{test}} > N_{\text{train}}$:

1. The PE matrix coherence $\mu(\Delta)$ for offsets $\Delta > N_{\text{train}}$ was never observed during training. The model has learned measurement vectors $\mathbf{w}$ calibrated for the coherence structure of $\mathbf{P}$ up to $N_{\text{train}}$ positions.
2. Beyond $N_{\text{train}}$, the coherence profile changes (especially for geometric frequency spacing, where low-frequency components wrap around), creating **mismatched measurement conditions** — the CS analogue of distribution shift in the sensing matrix.
3. Position interpolation methods (PI, NTK-aware interpolation) can be reinterpreted as **sensing matrix rescaling**: they modify the effective coherence profile to match what the learned measurement vectors expect. NTK-aware interpolation outperforms uniform PI because it preserves high-frequency coherence structure (local discrimination) while rescaling only the low-frequency components where wrapping occurs.

This interpretation makes a concrete prediction: **position interpolation methods that minimize the change in coherence profile** (rather than minimizing the change in absolute frequencies) should outperform those that minimize frequency change. This is testable by computing the coherence profile before and after interpolation for each PI method and correlating the change in coherence profile with downstream perplexity.

### 3. A Diagnostic Toolkit for Positional Processing

The R² metric and coherence analysis together define a practical diagnostic toolkit for understanding positional processing in any transformer model, with the following components:

**Head-level positional quality scores.** R² per head quantifies how cleanly each head implements a CS measurement of position. Models with universally low R² have poor positional processing (consistent with NoPE results). Models with bimodal R² distributions have specialized heads (Llama pattern). Models with uniform moderate R² have distributed positional processing (OLMo pattern).

**Spectral utilization profiles.** Pair-level ablation profiles visualize which RoPE frequency bands each model treats as informative. A CS-optimal model should show flat profiles; deviation from flatness diagnoses suboptimal frequency allocation. This diagnostic is directly usable during pretraining monitoring: a flat profile is a signal of healthy positional learning.

**Coherence-R² consistency check.** For any new model, computing the Gram matrix and predicting the R² ceiling takes under 1 GPU-minute. If the observed R² substantially exceeds the predicted ceiling, something unexpected is occurring (e.g., the model is using positional information through a non-shift-invariant mechanism). If it substantially falls below, the model has failed to fully exploit the positional signal available in its PE basis.

**Phase transition capacity estimation.** Given $s_{\text{eff}}$ and $N$, the predicted minimum number of high-SI heads $m^* = C \cdot s_{\text{eff}} \cdot \log(N/s_{\text{eff}})$ provides an estimate of how much positional head capacity a model needs for reliable processing. Models with fewer than $m^*$ high-SI heads are in the "insufficient measurement" regime and will degrade sharply under ablation or out-of-distribution inputs.

### 4. Connections to Mechanistic Interpretability at Scale

The CS framework suggests a **quantitative theory of when mechanistic findings generalize**. The model-conditional realization finding — that Llama and OLMo realize SI structure differently — is not just an architectural quirk but a prediction of CS theory: different sensing matrix designs (different PE frequency choices, different head dimensions, different normalization) produce different coherence structures, which in turn produce different optimal measurement strategies. Mechanistic findings about specific heads or circuits are therefore **coherence-structure-conditional**: they should be expected to generalize across models that share similar coherence profiles, and to differ across models that do not.

This provides a principled answer to a central challenge in mechanistic interpretability — why circuits found in one model fail to transfer to others. Rather than attributing differences to arbitrary training variation, the CS framework attributes them to differences in the underlying sensing matrix geometry. Cross-model generalizability of a mechanistic finding can be predicted in advance by comparing the coherence profiles of the source and target models.

### 5. Implications for MoE and Sparse Architecture Design

The threshold capacity result has direct implications for **Mixture of Experts (MoE)** architectures, where routing decisions are position-sensitive. The CS phase transition predicts that MoE models need at least $m^* = C \cdot s_{\text{eff}} \cdot \log(N/s_{\text{eff}})$ high-SI routing heads to maintain reliable positional processing. Below this threshold, routing decisions become positionally unreliable — heads cannot distinguish which expert should process a given position — producing the "lost in the middle" failure mode as a special case of insufficient CS measurements.

For sparse models more broadly, the measurement redundancy principle from CS suggests that **distributed positional sensing** (many heads each contributing partial positional signal) is more robust than concentrated sensing (few heads each contributing strong positional signal), because distributed measurements are more resilient to individual head dropout, activation sparsity, and quantization error. This connects directly to the distributed redundancy finding and suggests a training objective: regularize toward distributed high-R² heads rather than concentrated ones.

---

## Summary

| Experiment | Core Test | Key Prediction | Cost |
|---|---|---|---|
| **A: Welch Bound & R² Ceiling** | Coherence profile predicts R² ceiling across 36 experimental conditions | $r > 0.75$ between predicted and observed R² | < 1 GPU-hour |
| **B: CS Phase Transition** | Threshold capacity scales as $m^* \sim s \cdot \log(N/s)$ | $m^*$ increases log-linearly with context length | 6–21 GPU-hours |

| Practical Direction | CS Principle | Actionable Step |
|---|---|---|
| PE frequency design | Minimize Welch-gap coherence | Coherence-minimizing frequency optimizer |
| Length generalization | Sensing matrix rescaling | Coherence-profile-preserving interpolation |
| Model diagnostics | R² + coherence profile | Diagnostic toolkit for pretraining monitoring |
| MoE routing robustness | Phase transition threshold | Minimum high-SI head budget for reliable routing |
| Mechanistic transferability | Coherence-structure-conditional circuits | Cross-model coherence comparison before transfer |
