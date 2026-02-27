# TODO

## Post-Track-B: Run Spectral Analysis (All Models)

Run spectral analysis only after the canonical Track B rerun finishes (`results/track_b_canonical_perpos_v1` reaches `36/36` summaries).

### Pre-checks

- Confirm canonical Track B rerun is complete:
  - `find results/track_b_canonical_perpos_v1 -name summary.parquet | wc -l` -> `36`
- Keep spectral outputs separate from prior runs.

### Recommended spectral run (all models, ungated exploratory pass)

This runs spectral on all combos using the **current canonical Track B rerun outputs** and writes to a new spectral namespace:

```bash
.venv/bin/python experiment1/run.py spectral \
  --model all \
  --dataset all \
  --seq-len all \
  --spectral-gate-threshold 0.0 \
  --spectral-track-b-group track_b_canonical_perpos_v1 \
  --spectral-output-group spectral_canonical_perpos_v1
```

### Optional confirmatory-style pass (keep gate)

If we want a stricter gated run for comparison (likely sparse/empty depending on threshold):

```bash
.venv/bin/python experiment1/run.py spectral \
  --model all \
  --dataset all \
  --seq-len all \
  --spectral-gate-threshold 0.60 \
  --spectral-track-b-group track_b_canonical_perpos_v1 \
  --spectral-output-group spectral_canonical_perpos_v1_gated060
```

## Original Track B vs Current Canonical Track B (Important for Spectral)

### Original Track B (`results/track_b`)

- Uses the legacy centered Track B path.
- For RoPE models, centered Q/K were computed from **per-position means** on **post-RoPE captured Q/K**.
- This is the centered-result path with the known RoPE-centered methodology confound.
- Spectral (as implemented) analyzes `g_centered`, so pointing spectral here means spectral is analyzing the **legacy centered Track B** kernels.

### Current Canonical Track B Rerun (`results/track_b_canonical_perpos_v1`)

- Uses the `canonical_per_position` centered Track B mode (diagnostic equivalence check).
- For RoPE models, Q/K are unrotated to a canonical frame, centered per-position there, then rotated back before Gram fitting.
- This still uses **per-position centering** (so it is not the final “fix”), but it removes the specific pre/post-RoPE basis dependence concern.
- Spectral pointed at this namespace analyzes `g_centered` from the canonical rerun and is the preferred spectral input for comparing against the current rerun campaign.

## Interpretation Notes for Spectral

- Spectral gating uses **Track A** early-layer `mean_r2` (not Track B), so changing Track B namespaces does **not** change gate values.
- Changing `--spectral-track-b-group` changes which `g_centered` series are analyzed after the gate passes.
- Use separate spectral output namespaces so legacy vs canonical spectral outputs can be compared directly.

## Nice-to-Have Follow-up

- Run both legacy and canonical spectral passes (same gate threshold) and compare:
  - matched peak counts
  - mean relative error
  - top frequency overlap
