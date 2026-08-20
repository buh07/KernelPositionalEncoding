# Per-Layer SI Summary

This note summarizes a lightweight follow-up analysis of the existing per-head SI artifacts for the three primary 7-8B models. The goal is narrow: if a reviewer asks whether the measured SI effect is confined to a tiny early-layer slice, this gives a concise answer grounded in already-generated artifacts.

## Source artifacts

All numbers below were computed from:

- `results/experiment3/theory1_si_circuits/llama-3.1-8b/head_r2_summary.parquet`
- `results/experiment3/theory1_si_circuits/mistral-7b-v0.1/head_r2_summary.parquet`
- `results/experiment3/theory1_si_circuits/olmo-2-7b/head_r2_summary.parquet`

Each file contains `layer`, `head`, and `mean_r2`.

## Method

- For each model, compute the global 75th percentile of `mean_r2`.
- Mark heads above that threshold as "top-quartile SI heads."
- Summarize by layer:
  - layerwise mean `R^2`
  - number of heads in the top quartile
  - share of heads in the top quartile

This is a descriptive follow-up only. It does not change any main-paper statistic or claim boundary.

## High-level takeaway

The SI effect is **not confined to a single tiny early-layer band** in any of the three primary models, but the layerwise concentration patterns differ sharply:

- `Llama-3.1-8B`: strongest concentration is in early-to-lower-mid layers, with some later reuse.
- `Mistral-7B-v0.1`: strongest concentration is even more front-loaded, especially around layer 1, but still not single-layer only.
- `OLMo-2-7B`: low-amplitude and more diffuse; top-quartile SI heads are spread across multiple layers rather than sharply concentrated.

This is most consistent with the main-paper framing that:

- Llama and Mistral show stronger, more structured SI exploitation.
- OLMo shows weaker and more diffuse SI exploitation.
- The primary intervention result is not reducible to a single pathological layer.

## Model summaries

### Llama-3.1-8B

- Global mean per-head `R^2`: `0.3803`
- Top-quartile threshold (`q75`): `0.5040`

Top layers by top-quartile SI share:

| layer | layer mean R² | top-quartile heads | share |
|---|---:|---:|---:|
| 4 | 0.4956 | 20 / 32 | 0.6250 |
| 3 | 0.5037 | 18 / 32 | 0.5625 |
| 1 | 0.5118 | 16 / 32 | 0.5000 |
| 5 | 0.4607 | 16 / 32 | 0.5000 |
| 0 | 0.4584 | 14 / 32 | 0.4375 |
| 2 | 0.4315 | 14 / 32 | 0.4375 |
| 7 | 0.4574 | 13 / 32 | 0.4063 |
| 31 | 0.4199 | 13 / 32 | 0.4063 |

Interpretation:

- Llama's strongest SI concentration is clearly front-loaded in layers `1-5`.
- But the pattern is not purely "early only": later reuse exists, including at layer `31`.

### Mistral-7B-v0.1

- Global mean per-head `R^2`: `0.2658`
- Top-quartile threshold (`q75`): `0.3688`

Top layers by top-quartile SI share:

| layer | layer mean R² | top-quartile heads | share |
|---|---:|---:|---:|
| 1 | 0.4646 | 23 / 32 | 0.7188 |
| 2 | 0.3366 | 18 / 32 | 0.5625 |
| 6 | 0.3322 | 17 / 32 | 0.5313 |
| 4 | 0.3445 | 16 / 32 | 0.5000 |
| 3 | 0.3728 | 15 / 32 | 0.4688 |
| 5 | 0.3334 | 15 / 32 | 0.4688 |
| 17 | 0.3251 | 15 / 32 | 0.4688 |
| 8 | 0.3340 | 13 / 32 | 0.4063 |

Interpretation:

- Mistral is also strongly front-loaded, with an especially strong layer-1 concentration.
- Like Llama, it is not literally single-layer or strictly early-only: there is meaningful later-layer presence as well.

### OLMo-2-7B

- Global mean per-head `R^2`: `0.0576`
- Top-quartile threshold (`q75`): `0.0814`

Top layers by top-quartile SI share:

| layer | layer mean R² | top-quartile heads | share |
|---|---:|---:|---:|
| 7 | 0.0817 | 15 / 32 | 0.4688 |
| 22 | 0.0741 | 13 / 32 | 0.4063 |
| 20 | 0.0895 | 12 / 32 | 0.3750 |
| 14 | 0.0675 | 12 / 32 | 0.3750 |
| 25 | 0.0651 | 11 / 32 | 0.3438 |
| 17 | 0.0678 | 10 / 32 | 0.3125 |
| 16 | 0.0638 | 10 / 32 | 0.3125 |
| 8 | 0.0622 | 10 / 32 | 0.3125 |

Interpretation:

- OLMo has much lower overall SI amplitude than Llama or Mistral.
- Its top-quartile SI heads are more diffuse and less cleanly front-loaded.
- This is consistent with the paper's current "weak-SI directional anchor" framing.

## Suggested rebuttal use

If asked whether the measured SI effect is "just an early-layer artifact," a careful answer is:

> A lightweight layerwise follow-up on the existing per-head SI artifacts shows that Llama and Mistral do have stronger early/lower-mid layer concentration of top-quartile SI heads, but the effect is not confined to a single tiny early-layer slice. OLMo is lower-amplitude and more diffuse. This is consistent with the paper's current interpretation that SI exploitation is structured and model-heterogeneous rather than a one-layer pathology.

If asked whether this proves an early-layer theory from the appendix, the answer should be:

> No. This is a descriptive follow-up only and should not be over-interpreted as a full validation of the theoretical layerwise entanglement story.
