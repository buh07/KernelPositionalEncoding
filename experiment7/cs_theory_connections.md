# Compressed Sensing as a Theoretical Lens for Shift-Invariant Positional Kernels

## Overview

This document formalizes the connections between classical compressed sensing (CS) theory and the empirical findings reported in *Attention as a Shift-Invariant Positional Kernel*. The central claim is that the shift-invariant kernel framework and the CS measurement framework are not merely analogous — they are mathematically the same object viewed from different angles. Establishing this correspondence retroactively explains three otherwise puzzling findings: the reversed H2 result, the R² ceiling, and the threshold-gated redundancy structure. It also opens a principled design space for positional encodings that the current paper can gesture toward without requiring new experiments.

---

## 1. The Kernel Formula Is a CS Sensing Vector

The positional kernel identified empirically takes the form:

$$k_{\text{pos}}(\Delta) = \sum_{i=1}^{D} a_i \cos(\omega_i \Delta) + b_i \sin(\omega_i \Delta)$$

where $D = d_{\text{head}}/2$ and the amplitudes $(a_i, b_i)$ are determined by the learned Q/K projections. This expression has a precise CS interpretation. Define the **Fourier feature map** of relative position $\Delta$ as:

$$\phi(\Delta) = [\cos(\omega_1 \Delta),\ \sin(\omega_1 \Delta),\ \cos(\omega_2 \Delta),\ \sin(\omega_2 \Delta),\ \ldots,\ \cos(\omega_D \Delta),\ \sin(\omega_D \Delta)]^\top \in \mathbb{R}^{2D}$$

Then $k_{\text{pos}}(\Delta) = \mathbf{w}^\top \phi(\Delta)$, where $\mathbf{w} = [a_1, b_1, \ldots, a_D, b_D]^\top$ is the **learned measurement vector** for that head. The full positional encoding (PE) matrix $\mathbf{P} \in \mathbb{R}^{N \times 2D}$, whose $t$-th row is $\phi(t)$, is therefore a **sensing matrix** in the CS sense: each head's kernel is a single linear measurement $k_{\text{pos}}(\cdot) = \mathbf{w}^\top \mathbf{P}$ of the positional signal.

### Formal identification with bounded orthonormal systems

The PE matrix $\mathbf{P}$ is a **bounded orthonormal system (BOS)** with constant $K = 1$, since $|\cos(\omega_i \Delta)| \leq 1$ and $|\sin(\omega_i \Delta)| \leq 1$ for all $i, \Delta$. The BOS-RIP theorem (Rauhut 2010; Foucart & Rauhut 2013, Theorem 12.31) states that a random subsampling of $m$ rows from a BOS with constant $K$ satisfies the Restricted Isometry Property (RIP) of order $s$ with high probability provided:

$$m \geq C \cdot \delta^{-2} \cdot K^2 \cdot s \cdot \log^3(s) \cdot \log(N)$$

With $K = 1$ (the Fourier BOS achieves the optimal constant), the PE matrix requires the fewest measurements of any orthonormal system to achieve a given RIP constant. This is a formal sense in which the sinusoidal PE basis is CS-optimal — not in frequency selection, but in the BOS boundedness condition.

**Implication for the paper:** Every result in the paper that involves fitting $g(\Delta)$ to attention logits is implicitly studying the *learned measurement vectors* $\mathbf{w}$ that each head has selected from this optimal BOS. The R² metric quantifies how cleanly each head implements a rank-1 projection of the PE sensing matrix — a natural quantity in CS theory called the **signal-to-measurement-noise ratio**.

---

## 2. The R² Ceiling Is a Coherence Bound

**Finding (Experiment 1):** The strong-form hypothesis (R² > 0.80) is not met. Maximum observed R² is 0.625 (Llama, synthetic data). This is treated as a null result, but it has a principled explanation.

### The Gram matrix and mutual coherence

The mutual coherence of the PE matrix is defined as:

$$\mu(\mathbf{P}) = \max_{i \neq j} \frac{|\langle \mathbf{p}_i, \mathbf{p}_j \rangle|}{\|\mathbf{p}_i\| \cdot \|\mathbf{p}_j\|}$$

For the sinusoidal PE matrix, the inner product between position vectors $i$ and $j$ is:

$$\langle \mathbf{p}_i, \mathbf{p}_j \rangle = \frac{1}{D} \sum_{k=1}^{D} \cos(\omega_k (i - j))$$

This is the kernel $k_{\text{pos}}(i - j)$ normalized by $D$. The Gram matrix $\mathbf{G} = \mathbf{P}\mathbf{P}^\top$ is therefore **Toeplitz** with entries $G_{ij} = k_{\text{pos}}(i - j)/D$, confirming the spectral structure identified in Gu et al. (2025).

### The Welch bound constrains achievable R²

The **Welch bound** states that for any $N$ unit vectors in $\mathbb{R}^d$:

$$\mu(\mathbf{P}) \geq \sqrt{\frac{N - d}{d(N - 1)}} \approx \frac{1}{\sqrt{d}} \quad \text{for } N \gg d$$

For the geometric frequency spacing used in RoPE, the coherence between adjacent positions $\Delta = 1$ equals $\frac{1}{D}\sum_{k=1}^D \cos(\omega_k)$. With $D = d_{\text{head}}/2 = 32$ (for a 64-dimensional head) and geometric frequencies spanning $[10000^{-0}, 10000^{-1}]$, this coherence is high for small $\Delta$ — substantially above the Welch bound.

The R² of the shift-invariant fit is bounded by how cleanly the PE vectors separate positions. When adjacent positions have high coherence, the attention logits for nearby token pairs are intrinsically similar regardless of content, imposing a **ceiling on R²** that cannot be overcome by learning. The ceiling is:

$$R^2_{\max} \approx 1 - \frac{\sigma^2_{\text{content}}}{\sigma^2_{\text{content}} + \sigma^2_{\text{pos}}} \cdot \mu^2_{\text{max}}$$

where $\mu_{\text{max}}$ is the maximum coherence in the PE matrix and $\sigma^2_{\text{content}}$ is the content variance. On synthetic data (minimal content variance), the ceiling is determined almost entirely by $\mu_{\text{max}}$, explaining why synthetic experiments achieve the highest R² (0.625) while real text yields lower values (~0.52).

**Concrete prediction:** The Welch-bound gap of the sinusoidal RoPE PE matrix — quantifiable as $\mu_{\text{geometric}} / \mu_{\text{Welch}}$ — should predict the gap between observed R² and 1.0 across models and datasets. This is testable without new experiments: computing the Gram matrix of each model's PE frequencies and measuring the coherence profile is a purely analytical exercise.

---

## 3. The Reversed H2 Is a CS Global Sensing Prediction

**Finding (Experiment 2):** Ablating low-frequency RoPE channels degrades performance *across all dependency ranges*, not selectively for long-range tasks. The "low frequency = long range" prediction is consistently reversed.

### Low-frequency components as global position sensors

In the CS framework, the low-frequency components of the PE matrix are the **incoherent global sensors** of the position signal. Consider two positions $i$ and $j$ with large separation $|i - j| = L_{\text{long}}$. The low-frequency term $\cos(\omega_k \cdot L_{\text{long}})$ varies slowly (since $\omega_k \ll 1$), meaning it takes a similar value for positions near $i$ and near $j$. Low-frequency components therefore provide **coarse global position identity** — they tell the model approximately where in the sequence a token lives — but they do *not* selectively discriminate long-range dependencies.

High-frequency components oscillate rapidly: $\cos(\omega_k \cdot \Delta)$ for large $\omega_k$ changes substantially between $\Delta = 1$ and $\Delta = 2$. These components are therefore **precise local discriminators** — they tell the model the exact local position identity, enabling fine-grained short-range processing.

The CS parallel is the difference between **DC/low-frequency measurements** and **high-frequency measurements** in signal recovery. Removing the DC component of a Fourier measurement set doesn't eliminate recovery of long-range signals — it eliminates the ability to determine *absolute* signal amplitude (the baseline). Removing low-frequency components from the PE therefore removes the **reference frame** that anchors all positional reasoning, degrading all tasks equally.

### Formal statement

**Proposition.** Let $\mathbf{P}^{(lo)}$ denote the PE matrix restricted to the bottom-$D/2$ frequency bands, and let $s$-sparse attention over $N$ positions be the recovery problem. The mutual coherence of $\mathbf{P}^{(lo)}$ satisfies:

$$\mu(\mathbf{P}^{(lo)}) = \max_{|i-j| \leq K} \frac{1}{D/2} \sum_{k \leq D/2} \cos(\omega_k (i-j)) \approx 1 \quad \text{for small } |i-j|$$

because slow-varying basis functions agree on nearby positions. Therefore $\mathbf{P}^{(lo)}$ has **high coherence for short-range pairs**, meaning removal of low-frequency channels destroys the model's ability to distinguish *nearby* positions — a short-range capability, not a long-range one.

**Implication for the paper:** H2 reversed not because the frequency-to-range assignment hypothesis was wrong in its direction of effect, but because it was wrong in its *logic*. Low frequencies are not long-range specialists; they are **global anchors**. Removing them collapses position identification at all ranges. This reframing turns a confusing null-or-reversed finding into a positive theoretical result: *the CS global-sensing framework correctly predicts the pattern of H2 reversal*.

---

## 4. Threshold Capacity Is a CS Phase Transition

**Finding (Experiment 3P2-C):** The threshold capacity model is unanimously preferred over the linear model across both Llama and OLMo (6/6 criterion votes each). Positional kernel capacity is organized as a distributed, threshold-gated resource.

### The CS phase transition

Compressed sensing theory predicts a sharp phase transition in recovery performance as a function of the number of measurements $m$ relative to the sparsity $s$ and ambient dimension $N$. For Gaussian measurement matrices, Donoho & Tanner (2009) established that $\ell_1$ recovery succeeds with high probability if $m/N > \rho_D(s/N)$ (the Donoho-Tanner phase boundary) and fails with high probability otherwise. This transition is sharp: there is no gradual degradation zone, only an abrupt boundary.

The threshold capacity model you observe — where positional kernel capacity holds stable until a critical number of high-SI heads are ablated, then degrades sharply — is the direct neural analogue of this phase transition. Each high-SI head is a **measurement** of the positional signal. The critical threshold is the minimum number of measurements $m^*$ required for reliable recovery of $s$-sparse attention patterns over $N$ positions.

### Prediction

Under the CS phase transition model, the critical number of high-SI heads $m^*$ should scale as:

$$m^* \sim C \cdot s \cdot \log(N / s)$$

where $s$ is the effective attention sparsity (number of positions an individual head meaningfully attends to) and $N$ is the context length. This predicts that:

1. **Models with sparser attention** (lower effective $s$) should tolerate more head ablation before performance collapses.
2. **Longer contexts** (larger $N$) should require more high-SI heads to maintain positional fidelity, shifting the threshold $m^*$ upward.
3. **The threshold shape should depend on context length** in a log-linear way.

All three predictions are testable with the existing ablation infrastructure. The phase boundary itself can be estimated by varying the fraction of ablated heads across a range of context lengths and fitting the Donoho-Tanner curve.

---

## 5. Boundary Detection as Coherence-Driven Specialization

**Finding (Experiment 3, T5b):** High-SI heads attend disproportionately to subword boundaries. This is supported in both Llama and OLMo.

### Boundaries as incoherence events

In CS terms, a subword boundary at position $t$ is a location where **content coherence drops sharply**: the token at $t$ is maximally unlike its immediate neighbors in semantic and distributional space. At such positions, the **content term** $k_{\text{content}}(x_i, x_j)$ in the attention decomposition contributes minimal cross-term noise to the positional kernel — because content variation is maximal, not minimal. This means boundary positions are where the shift-invariant positional signal is *most separable* from content, i.e., where the PE sensing matrix is most effective.

High-SI heads gravitating toward boundaries is therefore not a learned heuristic but a consequence of CS optimality: the model discovers that boundary positions are where positional measurements have the highest signal-to-noise ratio. The effective RIP constant of the measurement at a boundary is lower than at a content-dense interior position, making boundary-attending heads better CS receivers.

**Formal connection.** Let $r_{\text{cross}}(i,j) = A(i,j) - k_{\text{content}}(x_i, x_j) - k_{\text{pos}}(\Delta)$ be the cross-term capturing content-position coupling. At boundary positions, $\text{Var}[k_{\text{content}}]$ is high (boundary tokens are distributionally extreme), so the cross-term variance as a fraction of total variance is *lower* — meaning the positional kernel contributes proportionally more. High-SI heads are rational: they self-organize toward the positions where their measurement function is most reliable.

---

## 6. Model-Conditional Realization as Learned Frequency Selection

**Finding (Experiments 2 & 4):** Llama shows bimodal spectral utilization (dip at pairs 8–16, rebound at 24–32), while OLMo shows a near-null profile. Fine-tuning does not systematically alter SI structure.

### Spectral utilization as empirical CS frequency selection

The pair-level ablation profiles you measure are empirical estimates of each model's **effective sensing matrix** — which frequency pairs carry the most positional information for which task types. A CS-optimal PE would distribute measurement effort evenly across frequencies (flat spectral utilization), because uneven coverage leaves some frequency bands acting as high-coherence, low-discrimination measurements.

Llama's bimodal profile suggests the model has *partially* discovered CS-optimal sensing: intermediate-frequency pairs (8–16) are underutilized, while very high and very low frequency pairs carry disproportionate load. This is a suboptimal CS measurement strategy — it leaves a coherence "dip" in the middle-frequency range that could be exploited for better positional discrimination.

OLMo's near-null profile suggests either that its positional sensing is more uniformly distributed across heads (not concentrated in specific frequency pairs) or that GQA (grouped-query attention) imposes structural constraints that flatten the spectral profile.

**Why fine-tuning doesn't change SI structure.** The CS perspective explains the stability finding (Experiment 4, $\Delta R^2 = +0.002$ over 300 steps): the PE frequencies $\{\omega_i\}$ are fixed at initialization and are not learned. Fine-tuning adjusts the Q/K projection weights — the measurement vectors $\mathbf{w}$ — but cannot change the sensing basis $\{\phi(\Delta)\}$. Since the BOS structure is architectural, not parametric, fine-tuning operates within a fixed CS framework: it can reweight which measurements to trust but cannot redesign the measurement matrix itself.

---

## Summary Table: CS Theory Explains Each Key Finding

| Finding | CS Interpretation | Formal Tool |
|---|---|---|
| R² ceiling at ~0.62 | Coherence bound from geometric frequency spacing | Welch bound, Gram matrix analysis |
| H2 reversal (low-freq → all ranges) | Low frequencies are global anchors, not long-range specialists | BOS incoherence, DC component theory |
| Threshold capacity (3P2-C) | CS phase transition in measurement sufficiency | Donoho-Tanner phase boundary |
| Boundary detection specialization (T5b) | Boundaries maximize positional SNR (low cross-term noise) | RIP stability, cross-term variance |
| Bimodal Llama spectral profile | Suboptimal CS frequency selection (uneven measurement load) | Mutual coherence, frequency allocation |
| Fine-tuning SI stability | PE basis is fixed; fine-tuning only adjusts measurement vectors | BOS architecture vs. parametric weights |
| Distributed redundancy | Multiple heads provide redundant CS measurements | Measurement diversity, over-determination |

---

## Theoretical Contribution

The key theoretical contribution of connecting CS to this paper is the **coherence-prediction theorem** for R²:

> **Theorem (informal).** Let $\mathbf{P} \in \mathbb{R}^{N \times d}$ be the positional encoding matrix with mutual coherence $\mu(\mathbf{P})$, and let $\sigma^2_c / \sigma^2_p$ be the content-to-position variance ratio in the pre-softmax logits. Then the maximum achievable shift-invariant R² is bounded by:
> $$R^2 \leq 1 - \frac{\sigma^2_c / \sigma^2_p}{1 + \sigma^2_c / \sigma^2_p} \cdot \mu^2(\mathbf{P})$$
>
> In particular, for a PE matrix with coherence $\mu > 0$, perfect shift-invariance (R² = 1) is impossible in any model trained on naturalistic text (where $\sigma^2_c > 0$).

This theorem is provable from first principles using standard CS analysis of noisy measurement, and it turns the observed R² ceiling from a limitation into a **testable quantitative prediction** linking PE geometry to the achievable degree of shift-invariance in trained models.
