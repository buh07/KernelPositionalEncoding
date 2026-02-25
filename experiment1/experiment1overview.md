# Experiment 1: Estimating the Positional Kernel From Queries and Keys

## Hypothesis and Exploratory Framing

This is the first experiment in the series. While a specific hypothesis motivates the measurement design, the experiment is **primarily exploratory**: the goal is to characterize the full empirical distribution of R² across models, layers, heads, datasets, and sequence lengths, not to confirm a single point prediction. The pre-registered thresholds below define what constitutes strong support or strong falsification, but intermediate R² values (the expected common case) are themselves informative and will be preserved in full for downstream analysis.

**Primary hypothesis:** In LayerNorm/RMSNorm + RoPE-based transformers, each attention head implements a head-specific, shift-invariant positional kernel that is stable across content. Concretely, the pre-softmax attention logits — both measured directly on raw sequences and estimated via an averaged, content-removed Gram matrix — are well-approximated by a function of the relative offset Δ = i − j only, with R² that is higher in early layers and decays toward later layers.

**Secondary hypothesis (spectral):** The shift-invariant kernel recovered from queries and keys has a spectrum whose peaks align with the frequencies defined by the positional encoding scheme (RoPE rotation rates, or sinusoidal PE frequencies). This spectral alignment is specific to structured PEs — a no-PE baseline will not exhibit it.

These hypotheses are derived from Proposition 2 (Approximate Shift-Invariance) in the theoretical framework: under RoPE, the positional component of the attention logit is exactly shift-invariant by construction (`⟨R(i)q, R(j)k⟩ = ⟨q, R(i−j)k⟩`), and under LayerNorm the representations are constrained to a centered sphere so that direction — not norm — carries information, making inner products between query/key vectors a clean measure of the shift-invariant positional kernel.

**Interpretation of intermediate R² values:** R² falling between the falsification threshold (< 0.40) and the confirmation threshold (> 0.80) does not constitute a null result. Intermediate values (0.40–0.80) indicate a partial or attenuated shift-invariant structure and are the primary motivation for follow-on experiments: they identify which heads, layers, and model families are most promising for deeper analysis, guide the product-kernel decomposition in Experiment 2, and provide a quantitative baseline for comparing architectural variants. All per-head, per-layer R² values — not just aggregate summaries — are retained in full for this purpose.

---

## Expected Results and Motivation

### What we expect to find

The ranges below are theoretical predictions, not hard success criteria. Any R² values outside these ranges are informative, not failures — they sharpen the picture of where and how strongly shift-invariance holds.

| Layer depth | Expected R² (RoPE + LN) | Intuition |
|---|---|---|
| Layer 0–1 | High (> 0.80) | Content and position are independently initialized; cross-terms are small |
| Layer 2–3 | Moderate (0.50–0.80) | Residual stream accumulates content-position entanglement |
| Layer 4+ | Lower (likely < 0.50, but head-dependent) | MLPs and residuals create structured but non-separable cross-terms; some heads may retain high R² |

All per-head R² values are stored regardless of where they fall. The pattern of which heads maintain high R² at depth is itself a finding — it may reveal functionally specialized heads (e.g., induction heads or positional heads) that warrant targeted analysis in later experiments.

- **Claim (a) — shift-invariance:** R² from fitting `g(Δ)` to both the raw per-sequence logit matrices and the averaged Gram matrix is expected to be higher at early layers and to decay toward later layers, reflecting content-position entanglement accumulating through residual connections and MLPs. The decay need not be monotonic: individual heads may buck the trend, and such outliers are as scientifically interesting as the aggregate decay. The raw-logit measurement is the primary, direct test; the Gram matrix measurement is the secondary, noise-reduced view. Both measure the same claim and should agree at early layers; divergence at depth indicates content is dominating the logits.
- **Claim (b) — spectral alignment (contingent):** Only if Claim (a) yields meaningful R² (> 0.60 on average across early-layer heads), proceed to spectral analysis. The DFT of `g(Δ)` should have peaks at, or closely aligned with, the RoPE rotation frequencies (fixed at `θ_i = 10000^{−2i/d}`). If Claim (a) fails, spectral analysis is deferred. Note: the spectral claim is the harder, more specific prediction — partial or ambiguous R² at early layers may still yield clear spectral structure, which would itself be a notable finding.
- **No-PE baseline:** A model with no positional encodings should show low R² (close to the null expectation) and no structured spectral peaks, anchoring the scale of the measurement. Unexpectedly high NoPE R² would indicate implicit position leakage through causal masking — a notable finding that takes priority over the primary hypothesis.
- **Random-token control:** The synthetic dataset (random tokens) isolates whether shift-invariant structure is driven by content statistics or purely by architecture/PE scheme. We expect R² to be comparable to natural text at early layers if the structure is architectural, and lower if it is learned from data co-occurrence patterns. A large gap between random-token and Wikipedia R² is a finding worth reporting regardless of which direction it falls.
- **Length sensitivity (256 vs. 1024):** RoPE's theoretical shift-invariance is length-independent, so R² should not degrade significantly at longer sequences. A meaningful drop would suggest that the approximation is sensitive to the training-length distribution or that aliasing effects emerge at longer contexts (consistent with YaRN findings on frequency aliasing beyond training lengths). Length-dependent R² is retained in full — not summarized — so that aliasing patterns can be inspected at the per-head level.

### Literature grounding

- **RoPE exact shift-invariance** (Su et al., 2021, RoFormer): By construction, `⟨R(i)q, R(j)k⟩` depends only on `i − j`. This is the theoretical upper bound — empirically, content entanglement will reduce R² below 1.0 even at layer 0 because `W_Q`, `W_K` are trained jointly on content and position.
- **Content-position entanglement at depth** (the "entanglement" prediction from Sections 5–6 of the framework): Residual connections accumulate cross-terms across layers. MLPs apply per-token nonlinearities that mix content and positional features. R² decay is therefore a predicted architectural consequence, not a failure of the framework.
- **LayerNorm constrains representations to a centered sphere** (Proposition 1): Norm is fixed at `√d`, so inner products are dominated by direction. This is what makes the Gram matrix a clean estimate of the kernel — without LN, norms vary and the Gram matrix mixes kernel values with scale information.
- **Primacy bias / boundary artifacts** (Haviv et al., 2022; Yin et al., 2024): Early sequence positions in autoregressive models show anomalous attention due to the causal masking boundary effect (Proposition 4). This motivates reporting R² separately for early-sequence vs. interior tokens.

---

## Scope

### Models

Models are chosen to span a 2×3 design crossing **norm type** (LayerNorm vs. RMSNorm) with **PE scheme** (RoPE, learned absolute PE, and no PE). This design ensures that differences in R² can be attributed to PE scheme or norm type independently, not to their combination. Both GPT-2 small and GPT-2 medium are included to give a size-matched absolute-PE baseline against the ~1B RoPE models; they share the same PE scheme and norm, so the GPT-2 size comparison is a secondary within-cell axis. Gemma-2B is excluded: with LLaMA-3.2-1B and TinyLlama-1.1B already occupying the RoPE + RMSNorm cell, a third model there adds cost without new architectural information.

#### Comparison Grid

```
                    ┌─────────────────────────────────────────────────────────────────────────────┐
                    │                              PE Scheme                                      │
                    ├────────────────────┬─────────────────────────────┬──────────────────────────┤
                    │  RoPE (relative)   │    Learned absolute PE      │      No PE (NoPE)        │
   ┌────────────────┼────────────────────┼─────────────────────────────┼──────────────────────────┤
   │  LayerNorm     │  OLMo-1B (1.0B)    │  GPT-2 small (117M)         │          —               │
   │  (centered)    │                    │  GPT-2 medium (345M)        │  (no public model found; │
   │                │                    │                             │   see Design notes)      │
   ├────────────────┼────────────────────┼─────────────────────────────┼──────────────────────────┤
   │  RMSNorm       │  LLaMA-3.2-1B      │          —                  │  TinyLlama-NoPE-1.1B     │
   │  (uncentered)  │  TinyLlama-1.1B    │  (no public model found;    │  (paired with            │
   │                │                    │   see Design notes)         │   TinyLlama-1.1B)        │
   └────────────────┴────────────────────┴─────────────────────────────┴──────────────────────────┘

  Axes of comparison derived from this grid:
    (1) PE scheme effect      — read across columns, within a norm row
    (2) Norm type effect      — read down rows, within a PE column (RoPE column has both rows)
    (3) NoPE control          — TinyLlama-NoPE vs. TinyLlama-RoPE (architecture-matched pair)
    (4) Size scaling          — GPT-2 small (117M) vs. GPT-2 medium (345M) within the abs-PE cell;
                                OLMo-1B (1.0B) vs. LLaMA (1B) / TinyLlama (1.1B) across norm rows
    (5) Dataset sensitivity   — Wikipedia vs. code vs. random tokens (within every model)
    (6) Length sensitivity    — 256 vs. 1024 tokens (within every model)
    (7) Track A vs. Track B   — raw logits vs. content-removed Gram matrix (within every model)
```

**Design notes:**
- The RoPE column is the only one with both LayerNorm (OLMo-1B) and RMSNorm (LLaMA, TinyLlama) entries. This is the critical column for isolating the norm-type effect on R².
- **NoPE + LayerNorm cell is empty.** A systematic search found no publicly available pretrained decoder-only model with NoPE + standard LayerNorm. The only public NoPE model at scale (`McGill-NLP/codellm_1b_nope`) uses T5LayerNorm, which is RMSNorm-style (no mean centering), placing it in the RMSNorm row — the same cell as TinyLlama-NoPE. The NoPE + LayerNorm combination was not a research priority when GPT-2-style (LayerNorm) architectures were dominant, and the NoPE literature emerged after the field had shifted to RMSNorm-based bases.
- **Absolute PE + RMSNorm cell is empty.** Absolute PE was the standard before ~2022, when LayerNorm was universal; RMSNorm became standard alongside RoPE after ~2022. No mainstream pretrained model combines the two. This is a genuine historical gap in the ecosystem, not an oversight.
- Both empty cells are documented here to make the design's limitations explicit. If models filling these cells are identified before data collection begins, they should be added.
+
| Model | PE scheme | Norm | Params | HuggingFace ID | Notes |
|---|---|---|---|---|---|
| GPT-2 small | Learned absolute PE | LayerNorm | 117M | `openai-community/gpt2` | Absolute PE + LayerNorm baseline; smallest size point |
| GPT-2 medium | Learned absolute PE | LayerNorm | 345M | `openai-community/gpt2-medium` | Same cell as GPT-2 small; provides size scaling within the absolute PE column and a parameter count closer to the ~1B RoPE models |
| OLMo-1B | RoPE | LayerNorm | 1.0B | `allenai/OLMo-1B-hf` | RoPE + LayerNorm; the only model filling this cell; breaks the PE/norm confound; fully open (Apache 2.0), no gating |
| LLaMA-3.2-1B | RoPE | RMSNorm | 1B | `meta-llama/Llama-3.2-1B` | Modern RoPE + RMSNorm; paired with OLMo-1B for norm-type comparison |
| TinyLlama-1.1B | RoPE | RMSNorm | 1.1B | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` | RoPE + RMSNorm; architecture-matched RoPE counterpart to TinyLlama-NoPE |
| TinyLlama-NoPE-1.1B | None | RMSNorm | 1.1B | `AntNLP/TinyLlama-NoPE-1.1B` | NoPE control; identical architecture to TinyLlama-1.1B with RoPE patched out |

**Note on the TinyLlama paired comparison:**
- `AntNLP/TinyLlama-NoPE-1.1B` patches out the rotary embeddings from `TinyLlama/TinyLlama-1.1B` while preserving everything else (RMSNorm, 22 layers, 32 heads, same pretraining corpus). This is the strongest possible NoPE control: any R² difference between the two is attributable purely to the presence or absence of RoPE.
- If NoPE R² is high, this would indicate either (a) implicit position leakage through causal masking, or (b) the Gram matrix centering procedure is creating artifactual stationarity — both important findings warranting investigation before proceeding to later experiments.
- Verify the NoPE patch is correctly applied before running at scale: pass the same sequence twice with different position IDs; outputs should be identical if RoPE is truly disabled.

**Note on OLMo-1B:**
- `allenai/OLMo-1B-hf` uses full RoPE (all head dimensions rotated; no `rotary_pct` or `partial_rotary_factor`) and non-parametric LayerNorm (standard mean-centering + variance normalization, but without learnable gamma/beta affine parameters). The absence of learnable affine parameters does not affect the experiment: Proposition 1 in the theoretical framework requires only that the norm operation centers and projects onto a sphere — the learnable scale/shift are irrelevant to the geometry of the centering constraint. Training data is Dolma v1 (Common Crawl, C4, Wikipedia, books, code, scientific papers) — normal pretraining data with no synthetic content. Fully open, Apache 2.0 license, no HuggingFace gating. Explicitly designed for language model science and mechanistic interpretability. Architecture: 16 layers, 16 heads, d_model=2048, d_head=128.

**Note on LLaMA-3.2-1B Grouped Query Attention (GQA):** LLaMA-3.2-1B uses GQA with 32 query heads but only 8 key/value heads. Multiple query heads share the same key vector. The hook extraction utility must account for this: when capturing K vectors per head, each of the 32 query heads is paired with one of 8 KV heads (heads 0–3 share KV head 0, heads 4–7 share KV head 1, etc.). The logit matrix `A[h] = Q[h] @ K[kv_head(h)].T / √d_head` is still a T×T matrix for each query head `h`, and R² is computed per query head as usual — but the key vector has only 8 distinct values, not 32. This means the 32 per-head R² values for LLaMA are not all independent: heads sharing the same KV head will have correlated R² values. Report this explicitly and, where relevant, group the 32 heads by their shared KV head.

All models are used **pretrained and frozen** (inference only). This tests whether shift-invariant structure exists in trained models, not just at initialization.

### Datasets

Three datasets, all preprocessed to be **older than the oldest model's training cutoff** (pre-2019 for GPT-2):

1. **Wikipedia (English):** Natural prose. Tests whether shift-invariant structure emerges under realistic content distributions.
2. **The Pile — GitHub/code subset (or CodeSearchNet):** Structured, repetitive token distributions. Tests whether structured content statistics inflate apparent shift-invariance.
3. **Synthetic random tokens:** Uniformly sampled token IDs from the model's vocabulary, no linguistic structure. Control for architecture-only effects.

For each dataset, sample **200 sequences** at each of the two target lengths (see below). These 200 sequences serve a dual role: the **first 100 (indices 0–99) are the centering set** used exclusively in Track B to estimate and subtract the per-position content mean from Q and K; the **last 100 (indices 100–199) are the evaluation set** used in both Track A and Track B to compute R². This split is fixed before any data is collected — it is not random and is not re-sampled. The centering and evaluation sets must never overlap. For the synthetic random-token sequences no centering set is needed (there is no content mean to remove), so all 200 sequences serve as evaluation sequences in both tracks. The saved file format must encode the split explicitly: files are saved with 200 rows, with a `split` column taking values `centering` (rows 0–99) or `eval` (rows 100–199), so the split cannot be accidentally reconstructed differently across runs.

**Centering sample size note:** The centering mean `E_{centering}[q_{ℓ,h}^(x)(t)]` is estimated from 100 sequences. For deep layers where Q/K vectors are highly context-dependent, 100 sequences may give a noisy mean estimate, which adds noise to the centered Q/K vectors in Track B and inflates the residual variance, pushing R²_{gram} downward. This is a known limitation: the centering procedure trades off between removing position-specific content statistics (the goal) and introducing estimation noise (an artifact). 100 sequences is a practical compromise — the effect of this noise can be estimated by comparing R²_{gram} computed with centering means from 50 vs. 100 centering sequences; if R²_{gram} is stable, the mean estimate is well-converged. Report whether this stability check was performed.

**Tokenizer note:** GPT-2 (BPE, vocabulary size 50,257) and the RoPE models (OLMo-1B uses a SentencePiece/BPE hybrid with vocabulary size 50,280; LLaMA-3.2-1B uses a byte-level BPE tokenizer with vocabulary size 128,256; TinyLlama uses a LLaMA-1 SentencePiece tokenizer with vocabulary size 32,000) tokenize the same Wikipedia text into different token sequences. "Wikipedia" is therefore a fixed text corpus but **not** a fixed token stimulus across model families — sequence lengths in tokens will differ for the same text span. This is expected and not a confound: the experiment measures per-model R² on that model's own token sequence, not a matched stimulus. However, this means cross-model R² differences confound PE scheme with tokenization granularity. Report this limitation explicitly and, when comparing GPT-2 vs. RoPE models on natural text, note that any differences may partly reflect vocabulary size and subword granularity rather than PE scheme alone. The random-token control sidesteps this issue entirely (tokens are drawn from each model's vocabulary independently).

### Sequence lengths

- **256 tokens** (short context)
- **1024 tokens** (longer context, within typical training lengths)

Use exactly these lengths (pad or truncate as needed) so that positional indices are directly comparable across sequences.

### What is measured

The primary metric is **R²** from fitting a 1D shift-invariant function `g(Δ)` to the pre-softmax attention logit matrix, measured two ways: (Track A) directly on raw per-sequence logit matrices, and (Track B) on an averaged, content-removed Gram matrix. Both are restricted to off-diagonal entries only.

---

## Methods of Evaluation

There are two measurement tracks that run in parallel throughout. **Track A (raw logits)** is the primary measurement: it fits `g(Δ)` directly to real pre-softmax attention logits on natural sequences, making no assumptions about content-position separability. **Track B (averaged Gram matrix)** is the secondary measurement: it constructs a content-removed, averaged Gram matrix before fitting `g(Δ)`, which is a cleaner but more indirect estimate of the positional kernel. Both tracks produce R²_{shift} and are compared side-by-side.

### Step 1: Extract queries, keys, and raw logits

For each model, layer `ℓ`, and attention head `h`, register forward hooks to capture:

```
q_{ℓ,h}(t),  k_{ℓ,h}(t),   t = 1, ..., T          [shape: head_dim]
A_{ℓ,h}(t, s) = ⟨q(t), k(s)⟩ / √d_head            [shape: T × T, pre-softmax, pre-mask]
```

Do this for all 200 sequences in each dataset × length condition. Save `A_{ℓ,h}` per sequence for Track A; save `q` and `k` for Track B.

Sanity check: verify that `Q @ K.T / sqrt(head_dim)` matches the model's own pre-softmax logit tensor for a single sequence before running at scale.

---

### Track A: Raw Logit Measurement (Primary)

#### Step A1: Per-sequence shift-invariance R²

For each sequence `x` and each (layer, head):

1. Take the off-diagonal entries of `A_{ℓ,h}^(x)(t, s)` — exclude the diagonal (`t = s`) entirely.
2. **Causal masking restriction:** Because attention is causally masked, future-position logits are set to −∞ before softmax and carry no signal. Only the lower triangle (source position `s < t`, i.e., positive lags `Δ = t − s > 0`) contains real logit values. Pool only the **strictly lower-triangular** entries: `(Δ, A^(x)(t, t+Δ))` pairs for `t = 1, ..., T−1` and `Δ = 1, ..., t`. This means `g(Δ)` is estimated for `Δ ∈ {1, ..., T−1}` only (one-sided). **Structural limitation:** the experiment tests one-sided shift-invariance — whether `A(t, s)` depends only on `Δ = t − s > 0` — which is a strictly weaker claim than the full two-sided shift-invariance in the theoretical framework. This asymmetry is inherent to causal LMs and cannot be remediated.
3. Estimate `g^(x)(Δ) = mean_{t > Δ} A^(x)(t, t−Δ)` for each `Δ ∈ {1, ..., T−1}` (positive lags only; mean is over all `t` such that `t−Δ ≥ 0`).
4. Compute the **per-sequence R²**:

```
R²_{shift}^(x) = 1 − Var_{(t,Δ), t>Δ}[A^(x)(t, t−Δ) − g^(x)(Δ)] / Var_{(t,s), s<t}[A^(x)(t, s)]
```

Both variance terms are restricted to strictly lower-triangular entries (`s < t`, i.e., positive lags only), consistent with the causal masking restriction above.

5. Record `R²_{shift}^(x)` for every sequence. This gives a **distribution** of R² values across 100 evaluation sequences, directly measuring "is the kernel stable across content?"

#### Step A2: Aggregate across sequences

- Report mean ± std of `R²_{shift}^(x)` across sequences, per (layer, head).
- A high mean with low std confirms that shift-invariance is a stable, content-independent property. High mean with high std suggests it holds on average but not per-instance.
- Also compute a **pooled R²** by pooling all `(Δ, A^(x)(t, t−Δ))` pairs (lower-triangular entries, positive lags) across all evaluation sequences before fitting `g(Δ)`. This is the single-number summary for the comparison table.

---

### Track B: Averaged Gram Matrix (Secondary)

#### Step B1: Content removal

Split the 200 sequences into 100 centering sequences and 100 evaluation sequences (fixed split, set before running).

Subtract the per-position mean estimated on the centering set:

```
q̃_{ℓ,h}(t) = q_{ℓ,h}(t) − E_{centering}[q_{ℓ,h}^(x)(t)]
k̃_{ℓ,h}(t) = k_{ℓ,h}(t) − E_{centering}[k_{ℓ,h}^(x)(t)]
```

Run the full Track B pipeline **both with and without** this centering step and report both R² values. If stationarity appears only after centering, flag it as potentially artifactual (see mini-experiment below).

Optionally apply whitening after centering. If whitening substantially changes R², report this explicitly as a diagnostic.

**Mini-experiment — centering artifact check (required):** The per-position centering in Track B removes the mean Q and K vectors at each position, which also removes any position-specific content statistics that happen to be stationary across sequences (e.g., if position 1 is always a sentence-start token, its mean Q/K vector encodes that structural fact, not a purely positional feature). Subtracting these means can artificially inflate R² by removing variation that was content-driven. To detect this, compare Track B R² on random tokens (no centering) against Track B R² on natural text (with centering).

**Implementation note:** Random-token sequences have no position-specific content statistics by construction (each token is drawn i.i.d. from the vocabulary), so centering random-token Q/K vectors would subtract the per-position mean of random embeddings — a quantity with no structural meaning. Therefore, for the random-token dataset, run Track B **without centering**: set `q̃ = q` and `k̃ = k` (identity). This gives the uncentered Gram matrix `Ḡ_{QK}(t,s) = E_{eval}[⟨q^(x)(t), k^(x)(s)⟩]`. The resulting Track B R² measures whether averaging alone (without content removal) produces Toeplitz structure. For natural text, Track B is always run **with centering** as described in Step B1.

The comparison logic:

- **If Track B R² (no centering, random tokens) ≈ Track B R² (with centering, Wikipedia):** the Toeplitz structure is not produced by centering; it is architectural. Both tracks measure the same positional kernel.
- **If Track B R² (no centering, random tokens) << Track B R² (with centering, Wikipedia):** centering is removing position-specific content statistics from natural text (e.g., sentence-start structure) and inflating R². The stationarity in natural text is partly or wholly artifactual — the centering step is doing the work, not the positional kernel.
- **If Track B R² (no centering, random tokens) is itself high:** the averaging step alone is creating apparent Toeplitz structure even for random Q/K vectors — a methodological confound that would require investigating whether the finite-sample Gram matrix estimator is biased toward Toeplitz structure.

Report Track B R² on random tokens (no centering) alongside Track B R² on natural text (with centering) for all models, all layers, all heads. This is a required diagnostic, not an optional sanity check.

#### Step B2: Form the averaged Gram matrix

Compute and average over the 100 evaluation sequences:

```
Ḡ_{QK}(t, s) = E_{eval}[⟨q̃^(x)(t), k̃^(x)(s)⟩]
```

Visualize this `T × T` matrix as a heatmap. It should appear approximately Toeplitz (constant along diagonals) if the kernel is shift-invariant. Verify by eye before computing R².

#### Step B3: Fit g(Δ) and compute R²

Pool all strictly lower-triangular entries: `(Δ, Ḡ_{QK}(t, t−Δ))` for `Δ ∈ {1, ..., T−1}` (positive lags only; same causal restriction as Track A). Estimate:

```
g(Δ) = mean_{t > Δ} Ḡ_{QK}(t, t−Δ)
```

Compute the **Gram-matrix R²**:

```
R²_{gram} = 1 − Var_{(t,Δ), t>Δ}[Ḡ_{QK}(t, t−Δ) − g(Δ)] / Var_{(t,s), s<t}[Ḡ_{QK}(t, s)]
```

Both variance terms are restricted to strictly lower-triangular entries (`s < t`, positive lags only), consistent with the causal masking restriction. This is a cleaner estimate than Track A because content variation has been averaged out, but it cannot give per-sequence statistics.

---

### Step 2: Spectral analysis of g(Δ) — Claim (b) [Contingent]

**Gate condition:** Only proceed to spectral analysis if the mean Track A pooled R² across early-layer heads is > 0.60. If this threshold is not met, spectral analysis is deferred to a follow-up experiment and this step is skipped.

If the gate condition is met, for each (layer, head):

1. Compute the DFT of `g(Δ)` over the range of observed Δ. The lag sequence has length `N_lag = 2T − 1` (from `Δ = -(T-1)` to `Δ = T-1`), but in practice only the causal lags `Δ = 1, ..., T-1` are populated (autoregressive masking); use only these. Zero-pad to length `N_fft = 4T` before taking the FFT to improve frequency resolution:

```
ĝ(ω_m) = FFT(zero_pad(g[1..T-1], N_fft))[m],   m = 0, ..., N_fft/2
ω_m    = 2π · m / N_fft                          [radians per token, range 0..π]
```

2. Identify the top-5 peaks of `|ĝ(ω)|` using a local-maximum finder with minimum peak separation of `2π / T` (one unpadded DFT bin width) to suppress sidelobes.

3. **Frequency-normalized matching for RoPE models.** The RoPE frequencies `θ_i = 10000^{−2i/d_head}` span roughly six decades (from ~1.0 rad/token at `i=0` down to ~10⁻⁴ rad/token at `i = d_head/2 − 1`). A fixed ±1 bin tolerance is far too coarse for low frequencies (where bins are widely spaced) and far too tight for high frequencies (where many RoPE frequencies crowd into a small number of bins). Instead, use a **frequency-normalized tolerance**:

   ```
   A DFT peak at ω_m matches a RoPE frequency θ_i if:
       |ω_m − θ_i| / θ_i  <  τ_rel               [relative tolerance]
       OR
       |ω_m − θ_i|         <  δ_abs               [absolute floor]
   ```

   Set `τ_rel = 0.10` (10% relative tolerance) and `δ_abs = 2π / N_fft` (one padded DFT bin, as an absolute floor for low-frequency peaks that are close together). This criterion is tighter than ±1 unpadded bin at high frequencies (where RoPE frequencies are well-separated) and looser at low frequencies (where the DFT grid is coarse relative to frequency spacing). **Rationale:** at `i = 0`, `θ_0 ≈ 1.0` rad/token and adjacent RoPE frequencies differ by ~0.1 rad/token — ±1 unpadded bin at T=256 is `2π/255 ≈ 0.025` rad, which is too tight relative to the aliasing expected from finite T. At `i = d_head/2 − 1`, `θ_{d/2-1} ≈ 10^{-4}` and adjacent frequencies are extremely close — ±1 bin at T=256 is many multiples of the spacing, making the criterion trivially satisfied. The relative tolerance avoids both failure modes.

   For each top-5 DFT peak, record:
   - The peak frequency `ω_m` in rad/token
   - The nearest RoPE frequency `θ_i` (by relative distance)
   - The relative error `|ω_m − θ_i| / θ_i`
   - Whether it falls within the tolerance (matched: yes/no)

4. **Spectral analysis for GPT-2 (descriptive, not confirmatory).** For GPT-2, no fixed reference spectrum is available — the learned PE matrix `wpe` (shape `[max_len, d_model]`) encodes whatever frequency content was learned during training, which is arbitrary. Matching empirical attention logit peaks against `wpe`'s own DFT (via head-projected PE vectors) is nearly tautological: by construction, the PE contributes to the logits, so its spectral fingerprint will appear in the logits to some degree. Instead, use GPT-2's spectral analysis only descriptively: (a) compute the DFT of `g(Δ)` and identify its top-5 peaks; (b) compute the head-projected PE inner-product spectrum `S_PE(Δ) = ⟨W_Q·wpe[t], W_K·wpe[t−Δ]⟩` averaged over `t`, take its DFT, and overlay the two spectra; (c) report the Pearson correlation between the two power spectra. The goal is not to confirm a pre-specified hypothesis (the RoPE frequencies are known in advance; GPT-2's PE frequencies are not) but to characterize how much of the observed g(Δ) periodicity can be attributed to the learned PE structure. A high Pearson correlation is expected and provides confirmation that the measurement is capturing PE structure; a low correlation would suggest the logit periodicity arises from content statistics, not the PE.

5. Compute Pearson correlation between `|ĝ(ω)|²` (the empirical power spectrum) and the expected PE power spectrum (evaluated on the same `N_fft`-point grid). This provides a global alignment score independent of peak-counting.

6. Report per (layer, head): (a) number of top-5 peaks matched under the normalized criterion, (b) the relative errors for matched peaks, (c) the Pearson correlation. Flag any matched peaks with relative error > 0.05 as "marginal" to distinguish tight from loose matches.

**Why ±1 DFT bin is wrong for RoPE:** The `d_head/2` RoPE frequencies range from `θ_0 ≈ 1` down to `θ_{d/2-1} ≈ 10^{-4}` rad/token — a range of ~10,000×. At T=256, one unpadded DFT bin spans `2π/255 ≈ 0.025` rad, which is coarser than the spacing between adjacent RoPE frequencies at the low end (`Δθ ≈ 10^{-4}` rad) and finer than the spacing at the high end (`Δθ ≈ 0.1` rad). A fixed bin tolerance is simultaneously too lenient (matches spurious peaks at low frequencies) and too strict (misses genuine matches at high frequencies where the DFT grid is coarse).

For **NoPE models**, the expected spectrum is flat (no peaks). Any structured spectral peaks in NoPE models would indicate implicit positional encoding leakage — a notable finding worth flagging. The same peak-finding procedure is applied, but there is no matching step; instead, report whether the top-5 peaks rise significantly above the median spectral power (threshold: > 3× median).

### Step 3: Aggregate across heads and layers

For each model, report:

1. **R²_{shift} per head per layer (Track A, pooled):** Heatmap of shape `[num_layers × num_heads]`. One heatmap per model. This is the primary result figure.
2. **Mean R²_{shift} per layer** (averaged over heads, both tracks): Line plot vs. layer depth. Show Track A (pooled) and Track B (gram) as separate lines on the same axes to confirm they agree. Fit a linear regression of R² vs. layer index; report the slope as the quantitative depth-decay measure.
3. **Per-sequence R² distribution (Track A only):** For early-layer heads, plot a violin or box plot of the `R²_{shift}^(x)` distribution across 100 sequences. A tight distribution confirms content-stability; a wide distribution signals that shift-invariance is highly content-dependent.
4. **Spectral alignment score per layer** (averaged over heads, contingent on gate): Parallel plot to the R² vs. depth curve.
5. **Variance across data subsets:** Report mean ± std of Track A pooled R² across 5 random 80/20 splits of the 100 evaluation sequences.

### Step 4: Report comparisons

For each metric, report results broken down by:

- **PE scheme:** RoPE (OLMo-1B, LLaMA, TinyLlama) vs. learned absolute (GPT-2 small, GPT-2 medium) vs. none (TinyLlama-NoPE) — directly tests the PE-specificity of shift-invariance
- **Norm type:** LayerNorm (GPT-2, OLMo-1B) vs. RMSNorm (LLaMA, TinyLlama) — isolated cleanly within the RoPE column (OLMo-1B vs. LLaMA/TinyLlama); tests whether norm centering constraint matters
- **NoPE paired control:** TinyLlama-NoPE-1.1B vs. TinyLlama-1.1B — architecture-matched; any R² difference is attributable to RoPE alone
- **Model size:** GPT-2 small (117M) vs. GPT-2 medium (345M) within the absolute PE column; OLMo-1B (1.0B) vs. LLaMA/TinyLlama (~1B) within the RoPE column — tests size scaling while holding PE and norm fixed (OLMo-1B and LLaMA-3.2-1B are the same parameter count, so this is primarily a norm-type comparison at matched size)
- **Dataset:** Wikipedia vs. code vs. random tokens — tests whether effect is data-dependent
- **Sequence length:** 256 vs. 1024 — tests length sensitivity
- **Track A vs. Track B** — tests whether the centering procedure in Track B is creating artifactual stationarity
- **Early sequence tokens vs. interior tokens** — compute R² restricted to positions 1–20 vs. positions 50+ to test causal masking boundary effects

The key comparison table has rows = (model, dataset, length) and columns = (Track A early-layer R² mean, Track A late-layer R² mean, Track A depth-decay slope, Track B early-layer R² mean, Track A vs. B agreement at early layers, spectral alignment score [if computed]).

---

## Pre-Registration Summary

Before running any experiments, the following criteria are fixed. Thresholds define **strong support** and **strong falsification** only — intermediate outcomes are not failures but are the primary substrate for follow-on experiments. All raw per-head, per-layer R² values are stored in full regardless of where they fall relative to any threshold.

| Criterion | Track | Threshold | Interpretation |
|---|---|---|---|
| Early-layer pooled R² (layers 0–1), RoPE models | A | > 0.80 → strong support; 0.40–0.80 → partial / exploratory; < 0.40 → falsified | Full distribution retained; intermediate values ground Experiment 2 |
| Falsification threshold, RoPE models | A | < 0.40 even after centering (Track B) | Directly falsifies hypothesis; triggers investigation of centering artifact vs. genuine absence |
| Track A vs. Track B agreement, early layers | Both | \|R²_A − R²_B\| < 0.10 | Confirms centering is not creating artifactual stationarity |
| Depth-decay slope, RoPE models only | A | Negative on average; non-monotone heads flagged individually | Supports entanglement prediction; per-head outliers stored for Experiment 2 targeting. For GPT-2 (absolute PE, injected once at input), depth-decay is not predicted — PE signal may plateau or have non-monotone profile; report descriptively without a pre-registered direction |
| Gate condition for spectral analysis | A | Mean early-layer R² > 0.60 | If not met, spectral analysis deferred |
| Spectral alignment (top-K peaks), if gated | B | ≥ 50% of top-5 peaks within normalized tolerance (see Step 2) | Supports spectral claim; see Step 2 for frequency-normalized matching criterion |
| NoPE model R², early layers | A | < 0.40 | Confirms shift-invariance is PE-specific; if NoPE R² is high, investigate implicit position leakage |
| Random-token R² vs. Wikipedia R², early layers | A | Difference < 0.10 | Supports architectural (not data-driven) origin; large gap in either direction is reported as a finding |
| Per-sequence R² std, early-layer RoPE heads | A | < 0.15 → content-stable; > 0.15 → content-dependent | Wide std motivates per-content-type breakdown in follow-on analysis |

**Calibration note on thresholds:** The numeric thresholds above (> 0.80, < 0.40, < 0.15, etc.) are **hypotheses derived from theoretical reasoning**, not empirically calibrated values. There is no pre-computed null distribution for R²_{shift} under random model weights or random logits. In particular:

- R²_{shift} is not scale-invariant: it is a ratio of variances, and its null expectation depends on the signal-to-noise structure of the logit matrix, which varies across models, layers, and sequence lengths in ways that are not analytically tractable without running the experiment. A model with low-variance logits can exhibit artificially high R² from noise alone if the denominator variance is small.
- The > 0.80 "strong support" threshold is motivated by the fact that RoPE is *exactly* shift-invariant at the positional level (Prop. 2), so early-layer heads should have near-unity R² if content entanglement is small — but "small" has not been quantified.
- The < 0.40 falsification threshold is motivated by a rough intuition that R² < 0.40 indicates the variance explained by shift-invariance is smaller than the residual variance — but this intuition has not been validated on held-out data or against a known null.

These thresholds should be treated as **provisional working hypotheses** that will be revised in light of the data. The first-pass analysis (Phase 6) should compute the empirical R² distribution across all NoPE heads (which theoretically should have near-zero positional structure) and use this as an empirical null — the NoPE distribution anchors the scale and provides a data-driven baseline for what "low R²" means in this measurement framework. The per-sequence R² std threshold of < 0.15 is similarly uncalibrated and should be revised after inspecting the empirical distribution.

---

## TODO: Concrete Steps to Complete This Experiment

### Phase 0: Setup

- [ ] Create a project directory structure: `data/`, `models/`, `results/`, `figures/`, `scripts/`
- [ ] Set up a Python environment with: `torch`, `transformers` (HuggingFace), `numpy`, `scipy`, `matplotlib`, `seaborn`, `datasets` (HuggingFace), `scikit-learn`
- [ ] Verify GPU access; plan to run one model at a time and save intermediate activations to disk rather than holding all layers in memory simultaneously

### Phase 1: Data Preparation

- [ ] Download **Wikipedia (English)** via HuggingFace `datasets` (`wikimedia/wikipedia`, `20231101.en`), fixed random seed
- [ ] Download a **code dataset**: `codeparrot/github-code` (Python subset) or CodeSearchNet from HuggingFace
- [ ] Write a single tokenization script that accepts a model name as input (so tokenization uses that model's tokenizer) and:
  - Tokenizes raw text chunks
  - Truncates/pads to exactly 256 or 1024 tokens
  - Discards sequences with fewer than 256 tokens of natural content (no padding at the 256-length condition)
  - Samples exactly 200 sequences per (dataset, length) condition, saved as `data/{model}_{dataset}_{length}_seqs.pt`
  - Uses the first 100 as the centering set and the last 100 as the evaluation set (fixed split, not random)
- [ ] Generate **synthetic random-token sequences**: sample token IDs uniformly from the model vocabulary (excluding special tokens), 200 sequences at each length — these need no centering split since there is no "content mean" to remove

### Phase 2: Model Loading and Hook Setup

- [ ] Load `openai-community/gpt2` and `openai-community/gpt2-medium` in eval mode, `torch.no_grad()` (both publicly available, no gating)
- [ ] Load `allenai/OLMo-1B-hf` in eval mode (publicly available, Apache 2.0, no gating; bfloat16 recommended; verify `config.json` has `"layer_norm_type": "default"` and no `rotary_pct`/`partial_rotary_factor` before running — these confirm full RoPE and non-parametric LayerNorm)
- [ ] Load `meta-llama/Llama-3.2-1B` in eval mode (requires HuggingFace token)
- [ ] Load `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` in eval mode (the base pretrained RoPE model; publicly available without gating)
- [ ] Load `AntNLP/TinyLlama-NoPE-1.1B` in eval mode — note this patches out the RoPE embedding; verify the patch is applied correctly by checking that position IDs do not affect outputs (run the same sequence twice with different position IDs; outputs should be identical)
- [ ] Write a **model-agnostic hook extraction utility**:
  - Registers forward hooks on all attention modules
  - Captures Q and K **after** the head split and **after** any rotary embedding is applied (so for RoPE models, Q and K already encode the rotary rotation; logits are `Q @ K.T / √d`)
  - Captures the pre-softmax, pre-mask logit matrix `A = Q @ K.T / √d_head` directly
  - Output per sequence: `q` shape `[num_layers, num_heads, T, d_head]`, `k` same, `A` shape `[num_layers, num_heads, T, T]`
- [ ] Sanity check: verify `Q[l,h] @ K[l,h].T / sqrt(d_head) == A[l,h]` for one sequence across all layers and heads before proceeding

### Phase 3: Track A — Raw Logit Measurement

**Storage strategy (Track A):**
- **Both T=256 and T=1024:** For all 100 evaluation sequences, compute R² statistics on-the-fly and save only: (a) the per-sequence scalar R²_{shift}^(x), (b) the compact 1D array `g^(x)(Δ)` of length T−1 for spectral analysis. Do **not** save full `A_{ℓ,h}^(x)` tensors for all sequences — saving full T×T matrices for all 100 sequences is impractical even at T=256 (e.g., TinyLlama: 100 seq × 22 layers × 32 heads × 256² × 4 bytes ≈ **14 GB per dataset**, × 3 datasets = 42 GB for one model alone).
- **Visualization subset at T=256:** For **10 representative evaluation sequences** (indices 0, 11, 22, ..., 99 — evenly spaced to cover the evaluation range) from the **Wikipedia dataset only**, save the full `A_{ℓ,h}^(x)` tensors to disk. These are used for Gram matrix heatmap visualization only. Storage for 10 sequences at T=256: TinyLlama (worst case) = 10×22×32×256²×4 bytes ≈ **1.8 GB**; across all 6 models ≈ **7 GB total**.

- [ ] For each (model, dataset, length):
  - Forward-pass all 100 evaluation sequences
  - **If (T=256 AND dataset=Wikipedia AND sequence index in visualization subset):** additionally save `A_{ℓ,h}^(x)` to `results/{model}_wikipedia_256_A_vis/seq_{x}/`
  - For each sequence `x`, layer `ℓ`, head `h`:
    - Extract the **strictly lower-triangular** entries only (`s < t`; causal mask): pool `(Δ, A^(x)(t, t−Δ))` pairs for `Δ = 1, ..., T−1`
    - Estimate `g^(x)(Δ) = mean_{t > Δ} A^(x)(t, t−Δ)` for each positive lag Δ
    - Compute per-sequence R²:
      ```
      numerator   = Var_{(t,Δ), t>Δ} [ A^(x)(t, t−Δ) − g^(x)(Δ) ]
      denominator = Var_{(t,s), s<t} [ A^(x)(t, s) ]
      R²_A^(x)   = 1 − numerator / denominator
      ```
    - Save `R²_A^(x)` to a results dataframe; also save `g^(x)(Δ)` as a compact 1D array (length T−1) for later spectral analysis
  - Compute the **pooled Track A R²**: pool all lower-triangular `(Δ, A^(x)(t, t−Δ))` pairs across all 100 evaluation sequences before fitting `g(Δ)`, then compute R² on the pooled set
  - Save results as `results/{model}_{dataset}_{length}_trackA.csv`, columns: `[model, dataset, length, layer, head, seq_id, R2_per_seq, R2_pooled]`

### Phase 4: Track B — Averaged Gram Matrix

**Storage strategy (Track B):**
- **Centering means:** Save `mean_q` and `mean_k` for all (layer, head, position, d_head) for every (model, dataset, length) condition. These are the most important artifacts to retain — they cannot be reconstructed without re-running the centering pass. Total centering means storage (all 6 models × 3 datasets × 2 lengths, float32): ~18 GB (see Runtime Estimate section for per-model breakdown).
- **T=256:** Save full centered Gram matrices `Ḡ_{QK}` (shape `[num_layers, num_heads, 256, 256]`) to disk, for all 3 datasets. These are used for Toeplitz-structure heatmap visualization. Estimated size: TinyLlama (worst case) = 22×32×256²×4 bytes ≈ **185 MB per (model, dataset)** → 6 models × 3 datasets ≈ **3.3 GB total**.
- **T=1024:** Do **not** save full Gram matrices to disk. Instead, accumulate `Ḡ_{QK}` in-memory as a running average over the 100 evaluation sequences (peak memory during accumulation: L×H×1024×1024×4 bytes ≈ 2.9 GB for TinyLlama — fits in GPU memory), extract R² statistics and `g(Δ)` immediately, then discard the full matrix.

- [ ] For each (model, dataset, length):
  - Forward-pass the 100 centering sequences; compute and save `mean_q[layer, head, pos, :]` and `mean_k[layer, head, pos, :]` as `results/{model}_{dataset}_{length}_centering_means.pt`
  - Forward-pass the 100 evaluation sequences; subtract the centering mean to get `q̃` and `k̃`
  - Accumulate `Ḡ_{QK}[t, s] = mean_x( q̃^(x)(t) · k̃^(x)(s) )` for each (layer, head) as a running average in memory
  - **If T=256:** save the full centered Gram matrix as `results/{model}_{dataset}_256_gram_centered.pt`, shape `[num_layers, num_heads, 256, 256]`; also save the uncentered version as `_gram_raw.pt`
  - **If T=1024:** extract R² statistics and `g(Δ)` from the in-memory Gram matrix immediately after accumulation; do not write full 1024×1024 matrices to disk
  - For each (layer, head):
    - Extract **strictly lower-triangular** entries only (`s < t`; positive lags Δ=1,...,T−1)
    - Estimate `g(Δ) = mean_{t > Δ} Ḡ_{QK}(t, t−Δ)` over causal entries
    - Compute R²_gram:
      ```
      numerator   = Var_{(t,Δ), t>Δ} [ Ḡ_{QK}(t, t−Δ) − g(Δ) ]
      denominator = Var_{(t,s), s<t} [ Ḡ_{QK}(t, s) ]
      R²_gram     = 1 − numerator / denominator
      ```
    - Compute again for the uncentered gram matrix as `R²_gram_raw`
    - Save `g(Δ)` as a compact 1D array for later spectral analysis (Phase 5)
  - Save to `results/{model}_{dataset}_{length}_trackB.csv`, columns: `[model, dataset, length, layer, head, R2_gram, R2_gram_raw]`
- [ ] Visualize 4–6 centered Gram matrices as heatmaps (one per layer for one representative RoPE head) to visually confirm Toeplitz structure — use T=256 saved tensors for this

### Phase 5: Spectral Analysis [Contingent on Gate]

- [ ] **Gate check:** Compute mean Track A pooled R² over all early-layer heads (layers 0–1) for the RoPE models. If this mean is > 0.60, proceed. Otherwise, stop here, note the gate was not met, and skip Phases 5–6.
- [ ] For each (model, dataset, length, layer, head) where Track A pooled R² > 0.40:
  - Take the `g(Δ)` estimated from Track B (centered) on causal lags `Δ = 1, ..., T-1`
  - Zero-pad to `N_fft = 4T` and compute `ĝ(ω) = FFT(zero_pad(g, N_fft))` giving frequencies `ω_m = 2π·m/N_fft` for `m = 0, ..., N_fft/2`
  - Find the top-5 peaks of `|ĝ(ω)|` using a local-maximum finder with minimum separation `2π/T` to suppress sidelobes
  - For RoPE models (OLMo-1B, LLaMA, TinyLlama-RoPE): compute expected frequencies `θ_i = 10000^{−2i/d_head}` for each head dimension pair; match each top-5 peak using the **frequency-normalized criterion**: matched if `|ω_m − θ_i| / θ_i < 0.10` OR `|ω_m − θ_i| < 2π/N_fft` (whichever is more permissive); record the relative error for each peak. Note: OLMo-1B has d_head=128 (64 frequency pairs), while LLaMA-3.2-1B has d_head=64 (32 frequency pairs) — OLMo's expected spectrum is denser; report the number of matched peaks as a fraction of the total number of expected frequencies, not as a raw count
  - For GPT-2: extract `model.transformer.wpe.weight` (shape `[max_len, d_model]`), project through head-specific `W_Q`/`W_K`, compute FFT along `max_len`; match peaks using the same relative-tolerance criterion
  - Record per peak: matched (yes/no), relative error `|ω_m − θ_i|/θ_i`, flag as "marginal" if relative error > 0.05
  - Compute Pearson correlation between `|ĝ(ω)|²` and the expected PE power spectrum (evaluated on the same `N_fft`-point grid)
- [ ] Save spectral scores to `results/{model}_{dataset}_{length}_spectral.csv`

### Phase 6: Aggregation and Visualization

- [ ] **Figure 1:** For each model, produce a heatmap of **Track A pooled R²** with axes [layer × head]. One heatmap per (model, dataset, length). This is the primary result figure.
- [ ] **Figure 2:** Line plot of mean R² (averaged over heads) vs. layer depth for each model. Show Track A (pooled), Track B (centered gram), and Track B (raw gram) as three separate lines on the same axes. Overlay the linear regression fit on Track A and report the slope. This figure directly shows whether the two tracks agree.
- [ ] **Figure 3:** For 3–5 representative early-layer heads (chosen as the heads with highest Track A R²), plot violin plots of the **per-sequence R²** distribution across the 100 evaluation sequences. Separate panels for: RoPE+RMSNorm (LLaMA, TinyLlama-RoPE), RoPE+LayerNorm (OLMo-1B), NoPE+RMSNorm (TinyLlama-NoPE), and absolute PE+LayerNorm (GPT-2 small, GPT-2 medium) — these four panels directly visualize the populated cells of the 2×3 comparison grid.
- [ ] **Figure 4 [if spectral gate met]:** For one representative high-R² head from an early layer of a RoPE model: (a) centered Gram matrix heatmap showing Toeplitz structure, (b) `g(Δ)` curve, (c) `|ĝ(ω)|²` power spectrum (zero-padded to `N_fft = 4T`) with RoPE frequencies `θ_i` marked as vertical dashed lines and the ±10% relative tolerance bands shaded. Label each marked frequency with its relative matching error.
- [ ] **Table 1:** Summary table with rows = (model, dataset, length) and columns = (Track A early-layer R² mean±std, Track A late-layer R² mean, Track A depth-decay slope, Track B centered R² at early layer, |Track A − Track B| agreement, spectral alignment score [or "not computed" if gated out]). This is the comparison table for the pre-registered criteria.

### Phase 7: Write-Up

- [ ] Write a results section covering: (a) does shift-invariance hold at early layers in Track A (raw logits) across RoPE models? (b) do Track A and Track B agree, and is the centering step creating artificial stationarity? (c) do NoPE models show low R², confirming PE-specificity? (d) is per-sequence R² std low enough to claim content-stability? (e) does R² decay with depth as predicted? (f) [if gated] do spectral peaks align with PE frequencies?
- [ ] Explicitly address each pre-registered falsification criterion: for each row in the Pre-Registration Summary table, state whether the threshold was met, partially met, or failed
- [ ] Note any anomalies at early sequence positions (positions 1–20) consistent with the causal masking boundary effect
- [ ] Note any NoPE model heads where Track A R² is unexpectedly high — this would indicate implicit position leakage through causal masking, a finding relevant to future experiments

---

## Runtime Estimate (2× RTX 6000 Ada, 48 GB VRAM each)

These estimates assume: code is written and debugged, models are already downloaded, activations are saved to disk per-model rather than held in memory, and the two GPUs run different models in parallel where possible. They do not include debugging time, re-runs, or figure polishing.

### Per-model inference cost

The bottleneck is forward-passing 200 sequences × 2 lengths per dataset × 3 datasets = 1,200 total forward passes per model, saving full Q, K, and A tensors across all layers and heads.

| Model | Layers | Heads | A tensor size (1024-length) | Forward pass time (est.) | Total inference (1,200 passes) |
|---|---|---|---|---|---|
| GPT-2 small | 12 | 12 | 12 × 12 × 1024² × 4B ≈ 576 MB | ~0.05s | ~1 min |
| GPT-2 medium | 24 | 16 | 24 × 16 × 1024² × 4B ≈ 1.6 GB | ~0.1s | ~2 min |
| OLMo-1B | 16 | 16 | 16 × 16 × 1024² × 4B ≈ 1.1 GB | ~0.15s | ~3 min |
| LLaMA-3.2-1B | 16 | 32 | 16 × 32 × 1024² × 4B ≈ 2.1 GB | ~0.15s | ~3 min |
| TinyLlama-1.1B (RoPE) | 22 | 32 | 22 × 32 × 1024² × 4B ≈ 2.9 GB | ~0.2s | ~4 min |
| TinyLlama-NoPE-1.1B | 22 | 32 | 22 × 32 × 1024² × 4B ≈ 2.9 GB | ~0.2s | ~4 min |

**Note on storage:** A tiered storage strategy is used to keep total disk usage manageable (see Phase 3 and Phase 4 for full details). The naive approach — saving full A tensors for all 100 evaluation sequences, all layers, all heads — is impractical at any length. For example, TinyLlama-1.1B at T=1024 alone would require 1,200 sequences × 22 layers × 32 heads × 1024² × 4 bytes ≈ **3.5 TB**. Instead:

**What is always saved (both T=256 and T=1024):**
- Track A: Per-sequence R²_{shift} as a scalar (one per sequence × layer × head), and the diagonal statistics `g^(x)(Δ)` as a compact 1D array of length T−1. Storage: negligible (< 1 GB across all models and conditions).
- Track B: Centering means `mean_q` and `mean_k` for all (layer, head, position, d_head), stored as float32. These cannot be reconstructed without re-running the centering pass. Arithmetic: the six models have between 12 and 22 layers, 12–32 heads, and d_head of 64 or 128. For 3 datasets × 2 lengths × 2 (Q+K), total centering means storage is approximately:
  - GPT-2 small (12 L × 12 H × d_head=64): 12×12×(256+1024)×64×4×2×3 ≈ **0.9 GB**
  - GPT-2 medium (24 L × 16 H × d_head=64): 24×16×(256+1024)×64×4×2×3 ≈ **2.4 GB**
  - OLMo-1B (16 L × 16 H × d_head=128): 16×16×(256+1024)×128×4×2×3 ≈ **3.1 GB**
  - LLaMA-3.2-1B (16 L × 32 H × d_head=64): 16×32×(256+1024)×64×4×2×3 ≈ **3.1 GB**
  - TinyLlama-RoPE (22 L × 32 H × d_head=64): 22×32×(256+1024)×64×4×2×3 ≈ **4.3 GB**
  - TinyLlama-NoPE (22 L × 32 H × d_head=64): same ≈ **4.3 GB**
  - **Total centering means: ~18 GB**

**What is saved only at T=256 for visualization:**
- Track A: Full A tensors for **10 representative sequences** (not all 100) for a single dataset (Wikipedia only). These are used for Gram matrix heatmap visualization. Storage per model: 10 × L × H × 256² × 4 bytes. Largest case (TinyLlama: 22×32): 10×22×32×256²×4 ≈ **1.8 GB per model**, × 6 models ≈ **11 GB**.
- Track B: Full centered Gram matrices `Ḡ_{QK}` (shape [L × H × T × T]) at T=256 for all 3 datasets. Largest case (TinyLlama: 22×32): 22×32×256²×4 ≈ **185 MB per (model, dataset)** → 6 models × 3 datasets = **3.3 GB**.

**What is never saved (computed on-the-fly and discarded):**
- Full A tensors at T=1024 (would be TB-scale)
- Full per-sequence A tensors for the remaining 90 evaluation sequences at T=256
- Full in-memory Gram matrices at T=1024 (accumulated as running average, R²/g(Δ) extracted, then discarded)

**Total storage budget: ~35 GB** (18 GB centering means + 11 GB visualization A tensors at T=256 + 3.3 GB Gram matrices + ~2 GB compact R²/g(Δ) statistics across all conditions). TinyLlama's 22 layers × 32 heads is now the dominant contributor (OLMo-1B's reduction from Phi-1.5's 32×32 to 16×16 cuts that model's storage contribution by 4×).

### Phase-by-phase wall-clock estimates

| Phase | Task | Estimated time |
|---|---|---|
| 0 | Environment and directory setup | 30 min |
| 1 | Data download and tokenization | 1–2 hours (Wikipedia download is ~20 GB) |
| 2 | Model loading and hook verification | 1–2 hours (writing + debugging hook utility across 5 model families; add 15–30 min for TinyLlama-NoPE position-ID invariance check) |
| 3 | Track A inference + per-seq R² (all models) | **2–3 hours** (parallelized across 2 GPUs: GPT-2 small+medium+OLMo-1B on one, LLaMA+TinyLlama-RoPE+TinyLlama-NoPE on the other; TinyLlama at 22 layers is the bottleneck) |
| 4 | Track B centering pass + Gram matrix + R² | **1–2 hours** (same forward passes re-used from Phase 3 if activations cached; otherwise another 2–3 hours) |
| 5 | Spectral analysis (if gate met) | 45 min (pure NumPy/SciPy, CPU-only; zero-padding to 4T adds minor overhead) |
| 6 | Figure generation | 1–2 hours |
| 7 | Write-up | not estimated |

**Total compute time from a complete, working codebase: approximately 6–10 hours of wall-clock time on 2× RTX 6000 Ada**, assuming activations are streamed and not fully cached. If activations are cached to disk for reuse across Track A and Track B, add 1–2 hours of I/O but save the Track B inference pass entirely.

LLaMA requires a HuggingFace gated access token — factor in 5–10 minutes for token setup if not already done. GPT-2, GPT-2 medium, OLMo-1B, and TinyLlama variants are all publicly available without gating. The TinyLlama-NoPE model patches out RoPE at load time; budget an extra 15–30 minutes to verify the patch is correctly applied (position-ID invariance check) before running at scale.
