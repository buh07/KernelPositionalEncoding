# Tokenizer Analysis via Shift-Invariant Structure

## Overview

This document proposes three experiments that use shift-invariant (SI) head analysis as a lens for understanding, evaluating, and improving tokenizer design. The core insight is that SI heads function as the model's internal readout of tokenizer segmentation quality — they detect token boundaries, and their behavior changes causally when tokenization changes. This makes SI analysis uniquely suited for tokenizer diagnostics: it measures not what the tokenizer *does* to text, but how the model *responds* to the tokenizer's decisions.

The three directions are:

- **Experiment 8A: SI Boundary Alignment Score** — a quantitative tokenizer quality metric derived from SI attention profiles
- **Experiment 8B: Tokenizer-Aware Pruning** — using tokenizer-specific SI masks for model compression, demonstrating that SI classification enables something generic metrics cannot
- **Experiment 8C: Cross-Lingual Tokenizer Diagnostics** — measuring whether SI boundary detection quality degrades for languages with poor tokenizer coverage

Each experiment builds on existing infrastructure from Experiments 3 and 5, requiring minimal new code.

---

## Empirical Foundation

The following established findings motivate this work:

| Finding | Source | Key Number |
|---|---|---|
| SI heads detect token boundaries | 3P2-B (T5b) | Llama: d=1.085, p=7.3e-28; OLMo: d=0.543, p=3.0e-9 |
| Boundary detection survives trivial-feature controls | 3P2-B | Space-prefix, capitalization, punctuation, token-length all controlled |
| SI head identity is tokenizer-determined | 5B | Llama-2 vs 3.1 top-quartile Jaccard = 0.177, rank ρ = 0.048 (NS) |
| Tokenizer perturbations causally shift SI attention | 5C | Fake boundary: Δ=+0.017, p=8.8e-22; char decompose: Δ=+0.017, p=4.5e-35 |
| SI boundary detection is cross-corpus invariant | 3P2-I | Invariance score = 0.752; boundary_d range 0.382–0.581 across corpora |
| SI-only LoRA matches full LoRA | 4A (condition c) | Llama: 0.887 vs 0.896, NS; OLMo: 0.647 vs 0.635, NS |
| SI structure is immutable during fine-tuning | 4C | R² delta < 0.002 over 300 FT steps |

The critical observation is that findings 5B and 5C together imply a causal chain: **tokenizer → boundary positions → SI head identity → positional processing quality**. If this chain is real, then tokenizer quality should be measurable through SI head behavior, and tokenizer-informed interventions should outperform tokenizer-agnostic ones.

---

## Experiment 8A: SI Boundary Discrimination Metrics (BDI / BAC / KOA)

### Motivation

Current tokenizer evaluation metrics focus on compression ratio (tokens per word), vocabulary coverage (fraction of text encoded without `<unk>`), and fertility (tokens per morpheme). None of these measure how the model internally processes the boundaries the tokenizer creates. We fill this gap with three complementary metrics that quantify how SI heads respond to tokenizer-created boundaries, each testing a different facet of the tokenizer–SI alignment.

> **Note (v2 redesign).** The original SIBAS metric (ratio of boundary attention mass to uniform expectation) failed empirically: a three-way boundary classification (word_boundary / subword_boundary / non_boundary) assigned 99.8% of positions to the first two classes, making the uniform-attention denominator vacuous. The redesigned metrics below use the same boundary definition as Experiment 3P2-B (`word_ids[t] ≠ word_ids[t-1]`), which yields a ~55–65% boundary / 35–45% continuation split and produces meaningful contrasts.

### Definitions

**Boundary definition.** Following 3P2-B, a position $t$ ($t \geq 1$) is a **boundary** if `word_ids[t] ≠ word_ids[t-1]` (first token of a new word or the first subword after a word transition), and a **continuation** if `word_ids[t] == word_ids[t-1]` (subsequent subword within the same word). Position 0 and special tokens are excluded.

**Head classification.** Using existing R² profiling, heads are split into quartiles by mean R². The top quartile is the SI set $\mathcal{H}_{\text{SI}}$; the bottom quartile is the non-SI set $\mathcal{H}_{\text{lo}}$.

#### Metric 1: Boundary Discrimination Index (BDI)

For head $h$ on a sequence, the BDI measures whether $h$ attends more strongly to the immediately preceding token at boundary positions than at continuation positions:

$$\text{BDI}(h) = \overline{A_h(t, t{-}1)}_{\,t \in \mathcal{B}} \;-\; \overline{A_h(t, t{-}1)}_{\,t \in \mathcal{C}}$$

where $\mathcal{B}$ and $\mathcal{C}$ are the boundary and continuation position sets. BDI > 0 means the head increases its "look-back-one" attention at word transitions — consistent with the 3P2-B boundary detection finding.

The aggregate BDI for the SI set is $\overline{\text{BDI}}_{\text{SI}} = \text{mean}_{h \in \mathcal{H}_{\text{SI}}} \text{BDI}(h)$, and similarly for the non-SI set.

#### Metric 2: Boundary Attention Contrast (BAC)

$$\text{BAC} = \frac{\overline{\text{BDI}}_{\text{SI}}}{\overline{\text{BDI}}_{\text{lo}}}$$

BAC > 1 means SI heads discriminate boundaries more than non-SI heads. BAC < 0 (when the denominator is negative) means SI and non-SI heads have *opposite* boundary behavior — an even stronger specialization signal.

#### Metric 3: Kernel Offset Alignment (KOA)

For each SI head $h$, estimate its **dominant offset** $\Delta^*(h)$ as the mode of $\arg\max_{j < t} A_h(t, j)$ across query positions (excluding $\Delta=0$). Then KOA measures whether inter-boundary distances are multiples of $\Delta^*$:

$$\text{KOA}(h) = \frac{|\{(b_i, b_{i+1}) : |b_{i+1} - b_i| \bmod \Delta^* \leq 1\}|}{|\text{consecutive boundary pairs}|}$$

KOA ≈ 1.0 means the tokenizer's boundary spacing is quantized at the SI head's preferred kernel wavelength. KOA at chance depends on $\Delta^*$ and sequence length and is estimated via a permutation baseline.

### Design

**Models.** Compute all three metrics for:

| Model | Tokenizer | PE Scheme | Notes |
|---|---|---|---|
| GPT-2 small | GPT-2 BPE (50k vocab) | Learned absolute | Non-RoPE control |
| TinyLlama-1.1B | Llama-2 BPE (32k vocab) | RoPE | Scale control |
| Llama-3.1-8B | Llama-3 BPE (128k vocab) | RoPE | Primary target |
| OLMo-2-7B | OLMo BPE | RoPE | Different architecture |

**Data.** 24 sequences of length 512 from the same Wikipedia corpus used in R² profiling.

**Statistical tests.** For each model, across the 24 sequences:
- **H1:** $\overline{\text{BDI}}_{\text{SI}} > 0$ (one-sample $t$-test)
- **H2:** $\overline{\text{BDI}}_{\text{SI}} > \overline{\text{BDI}}_{\text{lo}}$ (paired $t$-test across sequences)
- **H3:** BAC > 1.0 (one-sample $t$-test)
- **H4:** KOA > chance (one-sample $t$-test against permutation baseline)
- **Correlation:** Pearson $r$ between per-head BDI and per-head R² (tests whether boundary discrimination strength scales with SI-ness)

### Hypotheses

- **H_8A1:** BDI_SI > 0 for all models — SI heads preferentially attend to the previous token at word boundaries. Sanity check, expected from 3P2-B.
- **H_8A2:** BDI_SI > BDI_nonSI — SI heads show stronger boundary discrimination than non-SI heads. Core specialization claim.
- **H_8A3:** BAC > 1.0 (or BAC < 0 with BDI_lo < 0) — SI heads are disproportionately responsible for boundary detection. If BAC < 0, the interpretation is even stronger: SI and non-SI heads have *opposing* boundary policies.
- **H_8A4:** KOA > chance — boundary spacing aligns with SI kernel peak offsets, suggesting the tokenizer's segmentation and the model's kernel wavelength co-adapt.
- **H_8A5:** Per-head BDI × R² correlation is positive and significant — the more shift-invariant a head, the stronger its boundary discrimination.

### Expected Outcome

If H_8A1–8A2 hold, boundary discrimination is confirmed as an SI-specific property (extending 3P2-B from a binary test to a continuous measure). If H_8A3 and the BDI × R² correlation hold, BDI provides a **continuous, per-head tokenizer alignment metric** — useful for head selection, pruning, and diagnostics. KOA (H_8A4) adds a structural alignment dimension: not just "do SI heads detect boundaries" but "do boundary spacings match kernel wavelengths."

Together, BDI/BAC/KOA replace the original SIBAS metric with a richer, more robust suite that avoids the boundary-fraction degeneracy of SIBAS and tests three distinct aspects of tokenizer–SI alignment.

### Computational Cost

- R² profiling: reuse existing profiles. 0 GPU-hours.
- Attention extraction: 24 forward passes per model. ~0.5 GPU-hours per model.
- Total for 4 models: ~**2 GPU-hours**.

---

## Experiment 8B: Tokenizer-Aware Pruning via SI Masks

### Motivation

Experiment 4A showed that SI-only LoRA (condition c) matches full LoRA with ~25% of the parameters. This suggests SI classification can guide parameter-efficient methods. But a NeurIPS reviewer would ask: is SI classification *better* than simpler head selection criteria (attention entropy, activation magnitude, random)?

This experiment answers that question directly — and adds a twist that only SI analysis can provide: **tokenizer-specific pruning masks.** Since SI head identity is tokenizer-determined (5B: Jaccard=0.177), the optimal pruning mask should differ across tokenizers even for the same architecture. Generic metrics like attention entropy don't capture this tokenizer dependence.

### Design

**Step 1: Construct pruning masks.** For Llama-3.1-8B, construct four types of head pruning masks:

| Mask Type | Selection Criterion | Tokenizer-Specific? |
|---|---|---|
| SI-matched | Retain top-quartile SI heads (by R²) from Llama-3.1 profile | Yes |
| SI-mismatched | Retain top-quartile SI heads from Llama-2 profile applied to Llama-3.1 model | Yes (wrong tokenizer) |
| Entropy-based | Retain heads with lowest attention entropy (most peaked attention) | No |
| Activation-based | Retain heads with highest mean activation magnitude | No |
| Random | Retain a random 25% of heads | No |

The critical comparison is **SI-matched vs SI-mismatched**: same number of retained heads, same architecture, but different tokenizer-informed masks. If the matched mask outperforms the mismatched mask, it proves SI classification carries tokenizer-specific information that matters for downstream performance.

**Step 2: Progressive ablation.** For each mask type, ablate heads progressively from 0% to 75% (retaining 100% down to 25% of heads). At each ablation level, evaluate:
- Wiki perplexity (general LM quality)
- Math accuracy (using the existing eval battery)
- Boundary detection d (does the boundary signal survive pruning?)

This produces 5 degradation curves, one per mask type.

**Step 3: Compute area-under-degradation-curve (AUDC).** The AUDC summarizes each mask's robustness across all ablation levels. Lower AUDC = more graceful degradation = better mask.

**Step 4: Cross-tokenizer transfer test.** As a stronger test, repeat the experiment on Llama-2-7B:
- SI-matched mask: from Llama-2's own R² profile
- SI-mismatched mask: from Llama-3.1's R² profile applied to Llama-2
- Generic baselines (entropy, activation, random)

If the matched mask wins in both directions (Llama-3.1→Llama-3.1 and Llama-2→Llama-2), and the mismatched mask performs worse than the matched mask in both cases, the tokenizer-specificity claim is established with a clean crossover design.

### Hypotheses

- **H_8B1:** SI-matched mask has lower AUDC than random and activation-based masks on wiki perplexity ($p < 0.05$, paired across ablation levels).
- **H_8B2:** SI-matched mask has lower AUDC than entropy-based mask. This is the harder test — entropy-based selection also captures some positional structure. Expected: SI-matched wins because it captures tokenizer-specific boundary alignment, not just generic attention sharpness.
- **H_8B3:** SI-matched outperforms SI-mismatched ($p < 0.05$). This is the key finding: the same number of heads retained, but using the correct tokenizer's mask matters. The performance gap quantifies the "tokenizer-mask mismatch penalty."
- **H_8B4:** The crossover replicates: Llama-2 matched > Llama-2 mismatched, and Llama-3.1 matched > Llama-3.1 mismatched. Directionality is consistent.
- **H_8B5:** Boundary detection d degrades faster under SI-matched pruning than under random pruning (SI-matched mask specifically retains boundary-processing heads, so ablating the *non-retained* heads — which are the SI heads — should destroy boundary detection more efficiently). This is a mechanistic validation that the mask captures the right heads.

### Expected Outcome

If H_8B1–8B3 hold, the conclusion is: **SI classification enables tokenizer-aware model compression that generic head-importance metrics cannot replicate.** This is the "SI analysis enables something new" result that elevates the paper from characterization to methodology.

The mismatch penalty (H_8B3) is especially important. It means that when you change a model's tokenizer (e.g., continued pretraining with a new tokenizer), the optimal pruning strategy changes — and SI analysis tells you how. This is directly relevant to practitioners doing tokenizer adaptation or multilingual extension of existing models.

### Computational Cost

- R² profiling: reuse existing Llama-3.1 and Llama-2 profiles. 0 GPU-hours.
- Attention entropy / activation magnitude computation: one forward pass per model. ~0.5 GPU-hours.
- Progressive ablation: 5 mask types × ~8 ablation levels × 3 eval tasks × 2 models = 240 eval runs. Each eval run is a forward pass on ~1000 sequences. ~**8 GPU-hours** total.
- Total: ~**9 GPU-hours**.

---

## Experiment 8C: Cross-Lingual Tokenizer Diagnostics

### Motivation

Multilingual models notoriously underperform on languages with poor tokenizer coverage. The standard explanation is vocabulary-level: low-resource languages get fragmented into many subword tokens, increasing sequence length and computational cost. But this doesn't explain *why* fragmentation hurts beyond the obvious length penalty — a 2x longer sequence should be 2x slower but not necessarily 2x less accurate.

SI analysis provides a deeper explanation: poor tokenization creates boundaries that don't align with the model's SI infrastructure, degrading positional processing quality. If this is true, then the SI boundary alignment score (from 8A) should predict cross-lingual performance gaps better than compression ratio alone.

### Design

**Step 1: Select evaluation languages.** Choose languages spanning the tokenizer quality spectrum:

| Language | Expected Tokenizer Quality | Rationale |
|---|---|---|
| English | High | Primary training language for all models |
| German | Medium-high | Related to English, moderate subword fragmentation |
| Chinese | Medium | Character-level tokenization, fundamentally different boundary structure |
| Turkish | Medium-low | Agglutinative morphology, heavy subword splitting |
| Arabic | Low-medium | Right-to-left, complex morphology, often poorly tokenized |
| Amharic or Tigrinya | Low | Ge'ez script, minimal representation in training data |

**Step 2: Compute per-language SI diagnostics.** For each model and language, compute:

1. **BDI / BAC** (from 8A): SI boundary discrimination metrics
2. **Boundary detection d**: using the existing boundary analysis pipeline on text in each language
3. **Compression ratio**: tokens per whitespace-delimited word
4. **Effective R²**: mean SI head R² on text in each language (does the SI kernel fit degrade on non-English text?)
5. **SI head activation variance**: do SI heads activate less consistently on poorly-tokenized languages?

**Step 3: Test the positional processing degradation hypothesis.** The central prediction is:

$$\text{BDI}_{\text{English}} > \text{BDI}_{\text{German}} > \text{BDI}_{\text{Chinese}} > \text{BDI}_{\text{Turkish}} > \text{BDI}_{\text{Amharic}}$$

following the expected tokenizer quality gradient. Furthermore, this ordering should correlate with per-language perplexity *after controlling for compression ratio*.

**Step 4: Decompose the performance gap.** For each language pair (English vs language X), decompose the perplexity gap into:
- **Vocabulary component**: attributable to compression ratio differences (longer sequences)
- **Positional component**: attributable to BDI differences (SI boundary misalignment)
- **Residual**: unexplained by either

If the positional component is non-trivial (>10% of the total gap), it establishes that tokenizer-driven SI misalignment is a mechanistically distinct contributor to cross-lingual performance degradation.

### Hypotheses

- **H_8C1:** BDI decreases monotonically with language distance from English ($\rho > 0.7$ Spearman between language distance rank and BDI rank).
- **H_8C2:** Boundary detection d (Cohen's d for SI heads attending to boundaries vs non-boundaries) degrades for languages with poor tokenizer coverage. Expected: d drops below 0.5 for the lowest-resource languages.
- **H_8C3:** BDI explains per-language perplexity variance beyond compression ratio (incremental $R^2 > 0.10$ in a regression of perplexity on compression ratio + BDI vs compression ratio alone).
- **H_8C4:** Mean SI head R² is relatively stable across languages (delta < 0.05). This would confirm that the SI *mechanism* generalizes cross-linguistically even if the boundary *alignment* degrades — the heads are still doing positional processing, just less effectively because the boundaries are in the wrong places.

### Expected Outcome

If H_8C1–8C3 hold, the conclusion is: **multilingual performance gaps are partly attributable to SI boundary misalignment, not just vocabulary coverage.** This provides a mechanistic target for improving multilingual models: tokenizers should be evaluated not just on compression ratio but on BDI/BAC across target languages.

If H_8C4 holds simultaneously, it means the model's positional processing *infrastructure* is language-universal, but its *effectiveness* is tokenizer-dependent. This is consistent with the Exp 5 finding that SI structure is universal while SI head identity is tokenizer-specific — and extends it to the multilingual setting.

A negative result on H_8C3 (BDI doesn't predict perplexity beyond compression ratio) would also be informative: it would mean tokenizer quality matters for LM performance primarily through the vocabulary channel, not the positional processing channel.

### Computational Cost

- Multilingual text preparation: sourcing ~1000 sequences per language from Wikipedia or OPUS. Minimal compute.
- R² profiling on multilingual text: reuse existing pipeline, ~24 sequences per language per model. ~0.5 GPU-hours per model per language.
- BDI/BAC computation: attention extraction, ~0.1 GPU-hours per model per language.
- Total for 3 models × 6 languages: ~**12 GPU-hours**.

### Scope Note

This experiment is the most ambitious of the three and is likely out of scope for a first submission. It is included here as a natural extension that leverages the same infrastructure. For a NeurIPS paper, a reduced version (English + Chinese + Turkish on Llama-3.1 only) would suffice to establish the principle, with the full language sweep reserved for follow-up work.

---

## How These Experiments Relate

The three experiments form a progression:

```
8A (Metric)     →  Define BDI/BAC/KOA as tokenizer-SI alignment measures
                     ↓
8B (Application) →  Show SI classification enables tokenizer-aware pruning
                     that generic metrics can't replicate
                     ↓
8C (Extension)   →  Apply the framework to explain cross-lingual
                     performance gaps
```

**8A** establishes the measurement tools (BDI/BAC/KOA). **8B** demonstrates practical utility via tokenizer-aware pruning (the "SI analysis enables something new" result). **8C** extends the framework to cross-lingual settings where it makes novel predictions.

For a NeurIPS submission, 8A + 8B are sufficient. 8A is cheap (~3 GPU-hours) and validates the metric. 8B is the practical contribution (~9 GPU-hours) and directly addresses the "so what" question. 8C is a compelling follow-up but can be scoped down or deferred.

---

## Combined Experimental Budget

| Experiment | Core Cost | Extended Cost | Priority |
|---|---|---|---|
| 8A: BDI / BAC / KOA | 2 GPU-hours | — | High (validates the metrics) |
| 8B: Tokenizer-Aware Pruning | 9 GPU-hours | — | High (practical contribution) |
| 8C: Cross-Lingual Diagnostics | 4 GPU-hours (reduced) | 12 GPU-hours (full) | Medium (extension) |
| **Total** | **15 GPU-hours** | **23 GPU-hours** | |

All experiments reuse existing R² profiles and evaluation infrastructure. No new model training is required. The primary new code is the BDI/BAC/KOA computation (attention extraction + boundary identification + scoring), the mask construction for 8B (straightforward given existing ablation code), and multilingual text preparation for 8C.

---

## Connections to the CS Framework (Experiment 7)

The tokenizer analysis experiments connect to the compressed sensing framework from Experiment 7 in a specific way. Under the CS interpretation, token boundaries are positions where the **content-position cross-term noise** is minimized (Section 5 of `cs_theory_connections.md`). BDI is therefore an empirical estimate of the **effective measurement SNR** of the PE sensing matrix at boundary positions, while KOA measures whether the sensing matrix's preferred sampling grid aligns with boundary spacing.

This predicts a quantitative relationship: BDI should correlate with the **coherence gap** $\eta = \mu(1) / \mu_{\text{Welch}}$ from Experiment 7A, modulated by boundary density. Models with lower coherence gaps (better CS geometry) should show higher BDI — because their PE matrices distinguish positions more cleanly, making boundary detection easier. If this cross-experiment correlation holds, it links the theoretical (7A) and practical (8A) contributions into a single coherent framework.

Similarly, the tokenizer-aware pruning experiment (8B) connects to the phase transition prediction (7B): the critical threshold $m^*$ should be **tokenizer-dependent** because different tokenizers produce different effective sparsity $s_{\text{eff}}$ at boundary positions. A tokenizer with cleaner boundaries (higher BDI) should have lower effective sparsity, requiring fewer SI heads for reliable positional recovery — making the model more robust to pruning. This is testable by comparing the 8B degradation curves against the 7B predictions.
