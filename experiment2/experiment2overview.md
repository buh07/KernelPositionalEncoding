# Experiment 2: Causal Frequency Interventions and Kernel Sensitivity

## Document Status and Snapshot Context
Snapshot captured: **2026-03-03 22:23:46 EST**.

This protocol is written from a mixed project state:
1. Experiment 1 core analyses are complete across finalized namespaces (`track_a`, `track_b`, `track_b_canonical_perpos_v1`, `track_b_shared_mean_v1`, `track_b_bucketed_mean_v1`, `boundary`, and spectral historical/exploratory namespaces).
2. Optional fine-grained granularity sweep namespaces (`b8`, `b16`, `b64`) are not fully complete, but branch selection is no longer gated on them.

Running sweep status at snapshot:
1. `results/track_b_bucketed_mean_b8_v1`: **in progress** (`32/36` summaries).
2. `results/track_b_bucketed_mean_b16_v1`: **queued/not started** (`0/36`).
3. `results/track_b_bucketed_mean_b64_v1`: **queued/not started** (`0/36`).

Scope caveat:
1. Confirmatory synthetic spans target `<=16` (short) and a feasibility-calibrated long-offset set locked from a baseline sweep (`experiment2/long_offset_lock.json`), with a dedicated medium-range transition mini-panel (`spans={32,64,96}`) in Phase 2B for transition-region sensitivity checks.

Branch-selection rule:
1. **Branch B (heterogeneous recovery) is selected** based on completed Experiment 1 evidence (`per_position`, `canonical_per_position`, `bucketed_b32`, `shared_mean`) and norm/family heterogeneity in `results/reports/track_b_granularity_norm_v1/summary.md`.
2. Branch outcome only changes the default centered-diagnostic configuration in Experiment 2.
3. Branch outcome does not change the raw-first primary adjudication policy (`Track A` + `Track B raw` remain primary kernel readouts).
4. The incomplete optional `b8/b16/b64` sweep is retained as a refinement check, not a branch-selection gate.
5. Execution manifests are generated with explicit device assignment via `experiment2/run.py`; default operational target is `cuda` (GPU), not CPU.

## Why Experiment 2 Follows Experiment 1
Experiment 2 is constrained by measured behavior from Experiment 1 and must not ignore known methodology sensitivity.

Empirical anchors (from `results/reports/track_b_granularity_norm_v1/summary.md` and `experiment1/experiment1results.md`):
1. Legacy per-position and canonical per-position Track B are numerically near-identical (`mean |delta centered|` near zero, `max` around `1e-5`), so frame placement is not the dominant issue.
2. Centering granularity strongly affects centered diagnostics on natural data; per-position -> bucketed -> shared shows broad recovery trends.
3. Synthetic centered~=raw remains effectively exact across variants.
4. Raw Track B remains invariant across centering variants and is the stable comparator to Track A.
5. In Experiment 2 implementation, `shared_mean` centering is computed within each manifest cell (condition-local pooled mean across that cell’s sequences), not pooled across different cells/seeds.
6. Recovery magnitude is heterogeneous across norm/model families, motivating explicit norm-aware decomposition.

Theory anchors (from `overview.tex` references):
1. RoPE relative-position geometry and frequency structure (`su2021roformer`).
2. Context-length/frequency behavior (`peng2024yarn`, `kazemnejad2024lengthgen`).
3. Attention-as-kernel and Fourier-feature framing (`tsai2019transformer`, `rahimi2007random`).
4. Norm geometry effects (`layernorm2024geometry`, `layernorm2024reintro`).
5. Head specialization and positional entanglement caveats (`unpacking2025spectral`, `clark2019analyzing`, `voita2019headpruning`).

## Scope Split: Confirmatory Core vs Exploratory Extension
This preregistration uses a strict inferential split:
1. **Confirmatory core (Phase 2A/2B)**:
   - hypothesis-adjudicating,
   - fixed thresholds/tests,
   - fixed seeds,
   - bounded compute,
   - primary publication claims.
2. **Exploratory extension (Phase 2C/2D)**:
   - mechanism and transfer expansion,
   - fully specified runs,
   - secondary inference only.

Global policies:
1. No adaptive or sequential seed expansion.
2. Matched `random_medium` and `random_strong` controls are required.
3. H1/H2 are primary; H3 is secondary confirmatory; H4/H5 are exploratory.
4. Centered Track B remains a diagnostic channel, not a sole primary adjudicator.

## Core Causal Hypotheses (Preregistered)
Definitions:
1. Short-range tasks require effective dependencies `<=16` tokens.
2. Long-range tasks require effective dependencies from the globally locked long-offset set selected by the feasibility sweep (default candidate pool `{16,32,64,128}` before lock).
3. High-frequency ablation suppresses highest-rotation RoPE channels.
4. Low-frequency ablation suppresses lowest-rotation RoPE channels.
5. Random controls suppress matched counts of channels without frequency targeting.

Hypotheses:
1. **H1 (primary):** high-frequency ablation produces larger degradation on short-range than long-range tasks.
2. **H2 (primary):** low-frequency ablation produces larger degradation on long-range than short-range tasks.
3. **H3 (secondary confirmatory):** targeted ablations produce stronger kernel shifts than matched random controls and co-occur with task degradation in the same direction.
4. **H4 (exploratory):** effect sizes differ between LayerNorm and RMSNorm families.
5. **H5 (exploratory transfer):** directional synthetic effects partially transfer to stratified natural-language perplexity.

Directional pattern expectations (used in interpretation labels):
1. H1-targeted pattern: short-range degradation should be substantial while long-range degradation remains limited.
2. H2-targeted pattern: long-range degradation should be substantial while short-range degradation remains limited.
3. If both classes degrade strongly but with the predicted ordering, result is labeled directional-but-nonselective (weaker support).

Nulls:
1. H1/H2 null: no directional class contrast or reversed direction.
2. H3 null: targeted and random effects are not meaningfully distinguishable on predefined specificity metrics.
3. H4/H5 null: no robust stratified effects after predefined thresholds/controls.

## Baseline Intervention Mechanism (Deterministic and Auditable)
### RoPE frequency indexing
Let `d_head` be head dimension and `D = d_head / 2` rotary channel-pair count.

For each pair index `i in [0, D-1]`, define rotation rate `omega_i` from the model's implemented RoPE schedule.

Band definitions:
1. High-frequency band `H`: top half by `omega_i` (largest `omega_i`).
2. Low-frequency band `L`: bottom half by `omega_i` (smallest `omega_i`).
3. If `D` is odd, the middle index is excluded for symmetric partitioning.

Equivalent index form for standard monotone schedules:
1. `H = {0, ..., floor(D/2)-1}`.
2. `L = {ceil(D/2), ..., D-1}`.

### Severity and channel counts
Severity fractions apply to targeted-band size, not all channels:
1. Medium: remove `40%` of targeted-band channels.
2. Strong: remove `60%` of targeted-band channels.

Count rules:
1. `m_medium = floor(0.4 * |band|)`.
2. `m_strong = floor(0.6 * |band|)`.
3. Enforce `m_medium >= 1` and `m_strong > m_medium`.

### Removal order and nestedness
Deterministic removal order within each targeted band:
1. High-targeted ablation: remove highest-frequency-first (`omega_i` descending).
2. Low-targeted ablation: remove lowest-frequency-first (`omega_i` ascending).
3. Ties are broken by lower channel index.

Nestedness guarantee:
1. Medium removal set is a strict subset of strong removal set.
2. Nestedness is audited per layer/head and serialized in run metadata.

### Matched random controls
1. `random_medium` removes exactly `m_medium` channel pairs.
2. `random_strong` removes exactly `m_strong` channel pairs.
3. Sampling is without replacement from all `D` channel pairs.
4. Random indices are seeded and recorded in run metadata.
5. For each random run, report overlap with corresponding targeted set:
   - overlap count,
   - overlap fraction (`|R ∩ T| / |T|`),
   - Jaccard index (`|R ∩ T| / |R ∪ T|`).
6. H3 interpretation must include overlap diagnostics; unusually high overlap rows are flagged in summaries.

### Norm-preserving rescaling and drift audit
1. After masking, Q/K vectors are rescaled to match pre-mask L2 norms per token/layer/head.
2. Drift metric: `median(| ||x_post||/||x_pre|| - 1 |)`.
3. Hard tolerance: drift must be `<=0.05`.
4. Runs failing tolerance are invalid and must be rerun/fixed before analysis.

## Task Suite (Fully Reproducible Synthetic Generators)
Confirmatory synthetic tasks:
1. Local copy-offset (short).
2. Local key-match (short).
3. Delayed copy (long).
4. Long-range retrieval (long).

Task-family caveat:
1. `copy-offset` and `delayed-copy` share a homologous copy mechanism across span scales.
2. Confirmatory H1/H2 interpretation therefore requires reporting content-gated tasks (`key-match` and `retrieval`) separately in addition to class-aggregated metrics.

### Shared generator controls
1. Token domain excludes model special/reserved IDs (`BOS`, `EOS`, `PAD`, `UNK`, and tokenizer-specific reserved IDs).
2. Key pool and value pool are disjoint per model and fixed from `seed=0` pool construction.
3. Sample size is fixed: `200` sequences per `(model, task, intervention, length, seed)`.
4. Sequence IDs are reused across interventions within seed for paired contrasts.
5. Post-tokenization validation is mandatory for every labeled target.
6. Invalid instances are deterministically resampled using `(global_seed, sequence_id, retry_idx)`.

### Restricted-candidate evaluation policy (synthetic/bridge/mechanistic only)
1. Splits `synthetic`, `span_bridge`, and `mechanistic` use restricted-candidate scoring by default.
2. Candidate set size is fixed to `10` (`1` correct token + `9` distractors), with unique-token enforcement.
3. Candidate sets are deterministic per `(example_id, target_position, candidate_policy_version)` and therefore invariant across interventions for the same seeded examples.
4. Distractor policy is task-structured first, then deterministic filler fallback:
   - copy-family tasks: structured distractors from same-sequence copy sources/other targets/local context, then filler fallback;
   - key-match: other sequence values + `NO_MATCH` token, then filler fallback;
   - retrieval-family tasks: other sequence values (including distractor-linked values when present), then filler fallback.
5. Restricted `mean_accuracy` is the primary synthetic/bridge/mechanistic task metric used by floor gates and H1/H2 contrasts.
6. Full-vocab `accuracy/NLL` are retained as diagnostic metadata fields for transparency.
7. Tier-1 split (`tier1_ppl`) remains full-vocab and unchanged.

### Long-offset feasibility lock policy (confirmatory amendment)
1. Confirmatory long offsets are no longer hardcoded; they are locked from a one-off baseline-only feasibility sweep.
2. Sweep mode (`run.py --mode feasibility-sweep`) evaluates:
   - models: `llama-3.2-1b`, `olmo-1b`, `tinyllama-1.1b`
   - tasks: `delayed_copy`, `long_range_retrieval`
   - offsets: `{8,12,16,24,32,48,64,96,128}`
   - seeds: `3`
   - intervention: `none`
   - eval mode: restricted (`candidate_size=10`)
3. Lock mode (`run.py --mode lock-long-offsets --sweep-run-id <id> --apply`) selects offsets by rule:
   - per `(offset, model, task)`, compute seed pass-rate at floor threshold,
   - per `(offset, task)`, compute fraction of models with pass-rate `>= 2/3`,
   - offset is feasible if each long task has model-support fraction `>= 2/3`,
   - candidate confirmatory offsets are feasible offsets in `{8,12,16,24,32,48,64,96,128}`,
   - choose contiguous set from `8` up to max feasible,
   - require at least two locked long offsets; otherwise lock fails.
4. Fallback spans are the smallest locked long offset.
5. Confirmatory manifest build for Phase 2A/2B is blocked unless `experiment2/long_offset_lock.json` is valid.
6. Run metadata records `long_offset_lock_hash`, `long_offsets_used`, and `fallback_spans_used` per cell.

### Scale-up span-overlap remediation amendment (Option C then Option B)
1. The previously observed scale-up lock `selected_long_offsets=[8,12,16,24,32]` overlaps the short regime (`<=16`) and is treated as interpretation-confounded for primary H2 adjudication.
2. Remediation proceeds in two branches on GPUs `0/1/2`:
   - **Option C (exploratory):** single-model llama branch with deep long offsets `32,48,64`, retrieval-fallback lock policy, and explicit exploratory-only claim status.
   - **Option B (cross-model primary remediation):** scale-up three-model branch with non-overlap long lock candidates `24,32`, retrieval-fallback lock policy, and standard multi-model gating semantics.
3. `run.py --model-allowlist` is used to restrict build/feasibility/lock matrices for Option C while preserving default behavior when omitted.
4. Lock artifacts now record the effective model set used for lock resolution (`model_allowlist`, `effective_models`) in addition to selected offsets and active long tasks.
5. For single-model Phase 2B runs, analysis artifacts include:
   - `confirmatory_applicability=false`
   - `exploratory_reason=single_model_design`
   This prevents confirmatory-style promotion for single-model designs while preserving full diagnostic outputs.

### Scale-up runtime efficiency amendment (no semantic drift)
1. Phase 2A pilot may run via `--mode floor-prepass` (baseline-only floor decisions) instead of full execute.
2. `calibrate-floor` accepts two equivalent evidence sources:
   - `aggregate_task_metrics.parquet` (standard execute + reanalyze path), or
   - `floor_decisions/*.json` baseline accuracies (pilot prepass path).
3. Floor-threshold selection logic is unchanged:
   - keep `0.15` if pass-rate `>= 0.70`,
   - else use `0.13` if pass-rate `>= 0.70`,
   - else stop before full Phase 2A/2B.
4. Quick baseline scoring keeps token logits on-device to avoid CPU transfer bottlenecks; scoring semantics are unchanged.
5. Branch policy:
   - Option C (single-model exploratory) may skip posthoc-centered in Phase 2B with explicit centered-pending reanalysis allowance;
   - Option B (cross-model remediation branch) keeps posthoc-centered mandatory before final Phase 2B interpretation.

### Quick hypothesis stress-tests amendment (exploratory, non-retroactive)
1. A two-stage exploratory stress-test program is authorized under `results/experiment2/quick/<run_tag>/` and does not retroactively change confirmatory adjudication for authoritative runs.
2. Stage 1 (immediate):
   - fixed-span retrieval slope test and band-dose curve on `llama-3.1-8b`,
   - task restricted to `long_range_retrieval`,
   - spans forced to `32,48,64` (no mixed-span sampling),
   - seeds `0,1,2`,
   - `synthetic_count=100`,
   - runtime flags fixed to `track_a_enabled=false`, `centered_compute=defer`, `candidate_size=10`.
3. Stage 2 (next):
   - sparse single-pair RoPE sweep with additive exploratory intervention names:
     - `ablate_pair_<idx>`
     - `random_pair` (deterministic by seed/draw),
   - semi-natural wiki bridge panel sourced from real tokenized eval rows with long-gap repeated-token retrieval probes.
4. Reporting requirements for these quick tests:
   - per-span effects must include:
     - absolute drop (`none - ablated`),
     - baseline-normalized drop,
     - headroom-normalized drop,
   - monotonic checks over `32->48->64` are required for designated interventions.
5. Endpoint-policy metadata for quick tests is fixed to `h12_endpoint_policy=co_primary_raw_headroom` for interpretive consistency, but this remains non-gating for confirmatory promotion.
6. Current completed quick-test snapshots:
   - Stage 1 slope+dose run: `quick_hstress_20260313_220254`
     - `results/experiment2/quick/quick_hstress_20260313_220254/reports/slope/summary.json`
     - `results/experiment2/quick/quick_hstress_20260313_220254/reports/dose/summary.json`
   - Stage 2 pair+semi-natural run: `quick_hstress_stage2_20260313_223708`
     - `results/experiment2/quick/quick_hstress_stage2_20260313_223708/reports/pair_sweep/summary.json`
     - `results/experiment2/quick/quick_hstress_stage2_20260313_223708/reports/natural/summary.json`

### Cross-model pair-sweep expansion amendment (exploratory follow-on)
1. This expansion is explicitly exploratory and runs only after formal cross-model Phase 2B completion.
2. GPU policy for this pass is fixed to `0/1` with model-pinned tmux workers.
3. Pre-specified matrix:
   - retrieval task:
     - `olmo-2-7b` spans `24,32`
     - `gemma-7b` spans `16,24`
   - mirror short-task:
     - `local_key_match` for `llama-3.1-8b`, `olmo-2-7b`, `gemma-7b`
   - shared interventions:
     - `none`
     - `ablate_pair_<idx>` for `idx in {0,8,16,24,32,40,48,56}`
     - `random_pair` with draws `{0,1,2}`
   - seeds `{0,1,2}`, `synthetic_count=100`.
4. Inference/reporting policy:
   - co-primary descriptive metrics are always emitted (raw + headroom-normalized effects),
   - direction-agreement checks between raw and headroom-normalized effects are required,
   - multiple-comparison correction uses Holm within each pre-specified family (`model x task x span`),
   - floor-limited exclusions are reported per `model/task/span` (`none` pass-rates and excluded rows).
5. Claim boundary:
   - pair-sweep findings are exploratory and do not retroactively alter confirmatory adjudication of authoritative runs.

### 1) Local copy-offset
1. Sequence length `L` is fixed by cell.
2. Offset `k` is sampled from `{1,2,4,8,16}`.
3. Valid targets are positions `t` where `t-k >= 0`.
4. Ground truth at `t` is token at `t-k`.
5. Per-sequence target positions are sampled uniformly from valid positions with `n_targets=32`.

### 2) Delayed copy
1. Sequence length `L` is fixed by cell.
2. Offset `k` is sampled from the globally locked long-offset set in `experiment2/long_offset_lock.json` (subject to `k < L`).
3. Valid targets are positions `t` where `t-k >= 0`.
4. Ground truth at `t` is token at `t-k`.
5. Per-sequence target positions are sampled uniformly from valid positions with `n_targets=16`.

### 3) Local key-match
1. Fixed pair count `K=24` per sequence.
2. Key positions are placed with minimum spacing `>=2` tokens.
3. Keys and values are sampled from disjoint pools and mapped one-to-one within sequence.
4. Query targets are positions `t` that inspect trailing window `[t-16, t-1]`.
5. Matching rule:
   - if one or more keys appear in trailing window, nearest preceding key wins;
   - tie by distance is broken by lower key position index.
6. If no key appears in trailing window, target is sentinel `NO_MATCH` token from a fixed reserve set disjoint from key/value pools.
7. No-match rate is recorded per seed/cell.

### 4) Long-range retrieval
1. Fixed pair count `K=16` key-value associations per sequence.
2. Each association enforces minimum key-query gap from the same globally locked long-offset set used by delayed-copy (subject to sequence-length feasibility).
3. Query format uses dedicated query marker followed by key token.
4. Retrieval target is the associated value token.
5. Distractor handling:
   - each query includes one distractor key from same key pool but mapped to different value.
6. Ambiguity handling:
   - if multiple valid antecedents exist, nearest valid antecedent wins;
   - tie is broken by lower position index.
7. Ambiguity audit:
   - if ambiguity rate exceeds `5%` in any seed batch, batch is rejected and deterministically resampled.

### Fixed JSONL schema
Each row must include:
1. `id`
2. `task_name`
3. `tokens`
4. `target_positions`
5. `target_tokens`
6. `dependency_span`
7. `task_class`
8. `seed`
9. `model`
10. `length`
11. `task_params`
12. `pair_count`
13. `query_key`
14. `distractor_key`
15. `match_rule`
16. `has_no_match`

## Confirmatory Core (Primary Claims)
### Phase 2A: RoPE single-model proof of concept
Configuration:
1. Model: `llama-3.2-1b`.
2. Tasks: 3 synthetic tasks (1 short, 2 long):
   - short: `local_key_match`
   - long: `delayed_copy`
   - long: `long_range_retrieval`
3. Length: `1024`.
4. Interventions with random-draw expansion:
   - `none`
   - `ablate_high_medium`
   - `ablate_high_strong`
   - `ablate_low_medium`
   - `ablate_low_strong`
   - `random_medium_draw{0,1,2}`
   - `random_strong_draw{0,1,2}`
5. Synthetic seeds: fixed `7`.
6. Kernel extraction: all intervention conditions.

Run count:
1. Full Phase 2A: `1 x 3 x 1 x 11 x 7 = 231` settings.
2. Baseline-only Phase 2A pilot: `1 x 3 x 1 x 1 x 7 = 21` settings.

### Baseline performance floor (required)
For each `(model, task)` seed-batch:
1. Run `none` baseline first.
2. Phase 2A pilot recalibration rule (restricted-candidate metric):
   - if `>=70%` of Phase 2A `none` cells pass at `0.15`, keep threshold `0.15`;
   - else if `>=70%` pass at `0.13`, lock threshold `0.13`;
   - else stop before Phase 2B and mark design still floor-limited.
3. Apply the locked threshold for the clean 2A+2B rerun.
4. If below threshold, apply deterministic fallback using the lock artifact fallback set (`fallback_spans` in `experiment2/long_offset_lock.json`).
5. If still below threshold, mark cell floor-limited and exclude from H1/H2 confirmatory adjudication (descriptive reporting only).

### Phase 2A gate criteria (numeric)
All required for automatic progression:
1. Intervention audit passes (`norm_drift <= 0.05`).
2. At least one strong targeted condition shows either:
   - absolute accuracy drop `>=0.05`, or
   - relative NLL increase `>=0.10` vs `none`.
3. Directional class contrast threshold met:
   - `|drop_short - drop_long| >= 0.05` in predicted direction.
4. Specificity threshold:
   - targeted class-contrast exceeds matched random by `>=0.02`.
5. Kernel shift threshold:
   - `|delta_R2_trackA| >= 0.03` or `|delta_R2_trackB_raw| >= 0.03`.
6. Precision rule:
   - threshold must hold and 95% BCa CI lower bound must exceed `0` for at least one primary contrast;
   - if threshold holds but CI overlaps `0`, mark as **imprecise pass** (manual review, no auto-advance).

### Phase 2B: RoPE replication + Tier-1 transfer
Configuration:
1. Models: `olmo-1b`, `llama-3.2-1b`, `tinyllama-1.1b`.
2. Tasks: all 4 synthetic tasks.
3. Length: `1024`.
4. Interventions: same expanded confirmatory intervention set as Phase 2A (`11` conditions with random-draw axis).
5. Synthetic seeds: fixed `7`.
6. Tier-1 PPL interventions: `none`, `ablate_high_strong`, `ablate_low_strong`, `random_strong_draw{0,1,2}`.
7. Tier-1 seeds: `7`.
8. Medium-range transition mini-panel (confirmatory sensitivity channel):
   - tasks: `copy_offset_bridge`, `retrieval_bridge`
   - spans: `{32,64,96}`
   - interventions: `none`, `ablate_high_strong`, `ablate_low_strong`, `random_strong_draw{0,1,2}`
   - seeds: `7`
   - models: all 3 RoPE models
9. Long-task generation in Phase 2B uses the same global lock artifact as Phase 2A for delayed-copy/retrieval span sampling and floor fallback.

Run counts:
1. Synthetic: `3 x 4 x 1 x 11 x 7 = 924`.
2. Tier-1 PPL: `3 x 2 datasets x 1 x 6 x 7 = 252`.
3. Medium-range transition mini-panel: `3 x 2 tasks x 3 spans x 6 x 7 = 756`.
4. Phase 2B total: `1932`.

### Confirmatory total
1. `231 + 1932 = 2163` settings.

### Phase 2B gate criteria (explicit replication gate)
Criteria 1-5 are required for confirmatory success labeling:
1. Replication criterion:
   - at least `2/3` RoPE models satisfy H1 or H2 strong-severity threshold.
2. Cross-task agreement criterion:
   - both short tasks agree in sign and both long tasks agree in sign in at least `2/3` models.
3. Content-gated safeguard:
   - directional contrast must also hold for `key-match` (short) vs `retrieval` (long) in at least `2/3` models with contrast magnitude `>=0.03`.
   - this prevents copy-family tasks from solely driving class-level support labels.
4. Specificity criterion:
   - targeted-minus-random class-contrast advantage `>=0.02` in at least `2/3` models.
5. Kernel specificity criterion:
   - H3 metrics pass in at least `50%` of `(model x task-class)` groups.
6. Transfer criterion (exploratory):
   - H5 threshold met in at least `1/3` models.

Advancement rule:
1. If criteria 1-5 pass, proceed to Phase 2C regardless of criterion 6.
2. If any of criteria 1-5 fail, stop confirmatory claims and continue only with exploratory labels.
3. Execution policy note: phases may still be precomputed in parallel for throughput; gate outcomes control interpretation/promotion labels, not scheduler hard-stops.

### Confirmatory operational flow (amended)
1. Run `--mode feasibility-sweep` and collect baseline-only restricted-eval feasibility outputs.
2. Run `--mode lock-long-offsets --sweep-run-id <id> --apply`; build for Phase 2A/2B is blocked until this lock is valid.
3. Run Phase 2A baseline-only pilot (`21` cells).
4. Run `--mode calibrate-floor` and lock floor threshold (`0.15` keep, `0.13` fallback, else stop).
5. Run clean authoritative Phase 2A+2B manifests under the locked long-offset and floor-threshold settings.
6. Run `posthoc-centered`, then `reanalyze`, before any confirmatory interpretation/promotion.

## Tier-1 Stratified Perplexity (Hardened Definition)
Datasets:
1. `wiki40b_en_pre2019`.
2. `codesearchnet_python_snapshot`.

Per `(model, dataset, length)` cell:
1. Sequence count is fixed at `100`.
2. Under `none`, compute position-level local-concentration score from early layers (0-1): trailing 16-token attention mass fraction averaged over heads.
3. Dependency-type bins are frozen after baseline:
   - `local-dominant`: top quartile of local-concentration score.
   - `distal-dominant`: bottom quartile of local-concentration score.
4. Baseline-difficulty control:
   - compute baseline position-level NLL quartiles under `none`.
5. Report a `2 x 4` table per cell:
   - dependency type (`local`, `distal`) x baseline-NLL quartile (`Q1..Q4`).
6. Intervention deltas are computed within each stratum, then aggregated equally across quartiles.

### Binning Validity Check
1. Compute mean local-concentration for frozen `local-dominant` vs `distal-dominant` bins under baseline.
2. Validity threshold: mean separation gap must be `>=0.15`.
3. If gap `<0.15`, mark H5 as `uninterpretable for that cell` and report descriptively only.

H5 threshold:
1. `|delta_PPL_local - delta_PPL_distal| >= 0.5` under strong targeted intervention.
2. Targeted differential must exceed matched random by `>=0.2`.
3. H5 support requires persistence after quartile conditioning and binning validity pass.

H5 inferential scope note:
1. Tier-1 confirmatory cells now use multi-seed execution (7), enabling seed-level variance estimates.
2. H5 remains exploratory in claim hierarchy, even with improved variance estimation.

## Statistical Inference and Decision Rules
### Primary-family correction
1. Primary family: H1 and H2.
2. Error control: Holm-Bonferroni, familywise `alpha=0.05`.

### Effect estimation
1. All intervention comparisons use paired within-seed differences.
2. 95% BCa bootstrap CIs are computed from seed-level effects (`B=20,000` resamples).
3. Report effect size first, then CI and adjusted p-value.
4. BCa reproducibility policy:
   - primary CI is deterministic (`seed=0`),
   - secondary seed-sensitivity diagnostics use fixed seeds `{0,1,2,3,4}` with reduced bootstrap count for stability reporting.

### P-value method (locked)
1. Use two-sided paired sign-flip permutation test on seed-level paired deltas (`100,000` flips or all exact flips if smaller).
2. With `n=7`, minimum exact two-sided p-value is `2/128 = 0.015625`, which can pass Holm-adjusted primary-family thresholds.
3. Apply Holm adjustment to H1/H2 p-values.

### Quantitative thresholds
1. Meaningful synthetic task effect:
   - absolute accuracy drop `>=0.05`, or
   - relative NLL increase `>=0.10`.
2. H1/H2 class-contrast threshold:
   - `>=0.05` in predicted direction.
3. Kernel-shift threshold for co-occurrence:
   - `|delta_R2| >= 0.03` on Track A or Track B raw.

### H3 specificity metrics and stability
1. `S_ratio = |delta_R2_targeted| / (|delta_R2_random| + 0.01)`.
2. `S_diff = |delta_R2_targeted| - |delta_R2_random|`.
3. Confirmatory H3 uses per-seed expected random baseline (mean across random draws per seed) for targeted-vs-random pairing; per-draw diagnostics are reported separately.
4. If `|delta_R2_random| < 0.01`, ratio is flagged unstable and `S_diff` is primary.
5. Strong H3 support requires:
   - `S_diff > 0`.
   - `S_ratio >= 1.5` when ratio is stable.
   - directional consistency at both medium and strong severities.
6. If only strong passes, label H3 as partial co-occurrence, not mediation-strength support.
7. Reporting terminology:
   - `h3_rate_point` is the confirmatory H3 pass-rate used by the gate.
   - `h3_rate_ci` / `h3_ci_gated_pass_rate` are CI-gated diagnostic pass-rates; they are not confidence intervals over the pass-rate statistic itself.

### Precision labeling
For every hypothesis call:
1. **Pass**: threshold met and CI lower bound > 0.
2. **Imprecise pass**: threshold met but CI overlaps 0.
3. **Fail**: threshold not met or direction reversed.

### Directional selectivity labels (H1/H2)
1. **Strong directional support**:
   - class contrast threshold met, and
   - non-target class drop is limited (`<=0.03` absolute or `<=50%` of target-class drop).
2. **Directional-but-nonselective**:
   - class contrast threshold met, but non-target class drop exceeds the selectivity bound.
3. **No directional support**:
   - class contrast threshold not met or reversed.
4. **Baseline-conditioned selectivity sensitivity check**:
   - compute headroom-normalized drop per class: `normalized_drop = drop / max(baseline_accuracy, 0.15)`.
   - if raw and normalized selectivity labels disagree, mark the result `selectivity-floor-sensitive` and downgrade interpretation strength by one level.

## Power and Detectability (Pre-Declared)
1. Confirmatory thresholds are effect-size anchored and remain fixed.
2. After Phase 2A, run a pre-declared pilot-variance check using observed seed-level variance.
3. Compute minimum detectable effect (MDE) for Phase 2B contrasts at 80% power under current seed count and alpha control.
4. Interpretation rule:
   - if MDE exceeds the target threshold for a hypothesis family, mark that family as `underpowered` and downgrade claim strength.
5. This section affects confidence labeling only; it does not permit post-hoc threshold changes.

## Phase 2A -> Phase 2B Reuse Policy
For overlapping cells (`llama-3.2-1b`, same task/length/intervention/seed):
1. Phase 2A outputs may be reused in Phase 2B.
2. Reuse requires exact hash match on:
   - generator version,
   - intervention audit implementation,
   - tokenizer/model revision.
3. If any hash mismatches, overlapping cells are rerun and replacement is logged.
4. Reuse/rerun decisions are written to `reuse_decisions.json`.

Cross-model seed-correlation note:
1. The same numeric seed IDs are reused across models for paired design convenience, but tokenized sequence realizations differ by tokenizer and vocabulary.
2. The `2/3` model replication criterion is interpreted as cross-architecture robustness, not three statistically independent stochastic draws.
3. Report cross-model baseline difficulty correlation (per-seed aggregate baseline difficulty by task); if correlation exceeds `0.6`, label replication evidence as `correlation-sensitive`.

Cluster sensitivity evaluability note:
1. Cluster sensitivity uses an OR evaluability policy: if either H1 or H2 has enough model-level support for an exact sign-flip test, the cluster block is evaluable.
2. Non-evaluable hypothesis p-values in that block may still be shown for completeness, but they are not confirmatory decision criteria.
3. Confirmatory gating remains driven by pooled fixed-effects primary inference + preregistered threshold criteria; cluster results are robustness diagnostics.

Training-variance claim-scope note:
1. Inference scope is limited to task-sampling variability unless retraining variance is explicitly measured.
2. Machine-readable claim guards must block training-run generalization language while this limitation holds.

## Exploratory Extension (Fully Preregistered)
### Phase 2C: Family controls + broader coverage (exploratory)
Scope:
1. RoPE len-256 sensitivity (3 RoPE models).
2. AbsPE analog controls (`gpt2-small`, `gpt2-medium`).
3. NoPE baseline (`tinyllama-nope-1.1b`).
4. Tier-1 PPL strong-only subset.

Interventions:
1. RoPE models: 7-condition set (same as confirmatory).
2. GPT-2 AbsPE analog: 7-condition set via deterministic DCT masking of positional embeddings.
3. NoPE: `none` only.

### GPT-2 AbsPE analog intervention (deterministic specification)
1. Transform positional embedding table with orthonormal DCT-II along the position axis.
2. Define frequency indices by DCT component order.
3. Split low/high bands symmetrically by index.
4. Apply medium/strong severities as 40%/60% of targeted-band components.
5. Removal order:
   - high-targeted: highest-frequency-first.
   - low-targeted: lowest-frequency-first.
   - medium set must be nested in strong set.
6. Reconstruct with orthonormal DCT-III.
7. Apply same norm-drift audit policy (`<=0.05`) as RoPE runs.
8. Mark all such results as `AbsPE analog` and non-equivalent to direct RoPE channel interventions.

Phase 2C base run matrix:
1. RoPE len-256 synthetic:
   - `3 models x 4 tasks x 1 len x 7 interventions x 5 seeds = 420`.
2. GPT-2 synthetic:
   - `2 models x 4 tasks x 1 len x 7 interventions x 5 seeds = 280`.
3. NoPE synthetic:
   - `1 model x 4 tasks x 1 len x 1 intervention x 5 seeds = 20`.
4. Tier-1 PPL:
   - RoPE strong-only: `3 x 2 x 1 x 4 x 1 = 24`.
   - GPT-2 strong-only: `2 x 2 x 1 x 4 x 1 = 16`.
   - NoPE baseline: `1 x 2 x 1 x 1 x 1 = 2`.
5. Base Phase 2C total: `420 + 280 + 20 + 24 + 16 + 2 = 762`.

### Medium-Range Span Bridge (Exploratory)
1. Purpose: directly probe medium-range dependency regime (`17-127`) not covered by confirmatory spans.
2. Spans: `{32, 64, 96}`.
3. Task subset:
   - one copy-style task (`copy-offset-bridge`).
   - one retrieval-style task (`retrieval-bridge`).
4. Interventions: `none`, `ablate_high_strong`, `ablate_low_strong`, `random_strong`.
5. Seeds: `5`.
6. Matrix count:
   - `3 RoPE models x 2 tasks x 3 spans x 4 interventions x 5 seeds = 360`.
7. Interpretation rule:
   - span-bridge outcomes are exploratory and cannot overturn confirmatory labels.

### Phase 2C total (including bridge)
1. `762 + 360 = 1122` settings.

Phase 2C reporting rules:
1. Exploratory labels only; no change to confirmatory H1/H2 adjudication.
2. Cross-family comparisons must explicitly include comparability class (`RoPE direct`, `AbsPE analog`, `NoPE baseline`).
3. Exploratory power caveat:
   - Phase 2C and span-bridge use 5 seeds, so power is lower than confirmatory phases.
4. Promotion rule:
   - if a Phase 2C/bridge result is proposed as a headline narrative claim, the exact slice must be rerun at 7 seeds before confirmatory-style phrasing is allowed.

### Phase 2D: Scope-aligned head/layer mechanism probing (optional exploratory)
Trigger:
1. Run only if Phase 2B gate criteria 1-5 pass.

Models:
1. `olmo-1b`.
2. `llama-3.2-1b`.

Scope-aligned head-group definitions (frozen from Experiment 1 snapshots):
1. `early-head groups`:
   - top/bottom 25% heads ranked by early-slice Track A mean R2 (layers 0-1, natural datasets, len 1024).
2. `deep-head groups`:
   - top/bottom 25% heads ranked by deep-slice Track A mean R2 (final quartile layers, natural datasets, len 1024).
3. Group membership is frozen and serialized before Phase 2D.

Scopes:
1. `early-only` scope uses early-head groups.
2. `deep-only` scope uses deep-head groups.

Interventions:
1. `ablate_high_medium`
2. `ablate_high_strong`
3. `ablate_low_medium`
4. `ablate_low_strong`
5. `random_medium`
6. `random_strong`

Tasks/length/seeds:
1. 4 synthetic tasks.
2. length `1024`.
3. seeds `5`.

Phase 2D matrix:
1. `2 models x 4 tasks x 2 scopes x 2 head_groups x 6 interventions x 5 seeds = 960`.

Baselines:
1. `none` baselines are reused from Phase 2B hashes when available; otherwise fallback runs are added:
   - `2 models x 4 tasks x 1 len x 1 intervention x 5 seeds = 40` max fallback runs.

Phase 2D total:
1. `960` targeted/random runs (+ up to `40` fallback baselines).

Phase 2D reporting rules:
1. Exploratory mechanism claims only.
2. Report per-group effect sizes and uncertainty; do not overwrite confirmatory labels.
3. Phase 2D uses 5 seeds per cell, so inferential precision is lower than confirmatory phases.
4. If a Phase 2D result is proposed as a headline narrative claim, rerun the exact slice at 7 seeds before confirmatory-style language is allowed.

### H4 norm interaction handling
1. H4 remains exploratory across all phases.
2. Norm interaction is reported via stratified estimates and confidence intervals.
3. No primary confirmatory claim is made on H4 in this protocol version.
4. There is no dedicated confirmatory interaction test path in this version; interaction interpretation relies on exploratory stratified decomposition outputs.
5. Dedicated exploratory interaction output (`h4_interaction_exploratory.parquet`) reports task-class contrasts and `track_b_raw` targeted-vs-random kernel differentials by norm family.

## Conditional Branch Logic (Selected Branch + Refinement Override Policy)
Branch selection is currently locked to Branch B from completed Experiment 1 evidence.

Branch outputs affect only centered-diagnostic defaults:
1. Branch A (strong monotonic recovery): use coarser centering default for centered diagnostics.
2. Branch B (heterogeneous recovery): use family/norm-specific centered diagnostic defaults.
3. Branch C (weak recovery): center diagnostics downgraded; rely on Track A + Track B raw for primary kernel interpretation.

Refinement override policy:
1. If full `b8/b16/b64` completion later shows a clear cross-family monotonic trend that invalidates Branch B assumptions, branch selection may be revised in a dated protocol update.
2. Any branch revision must leave the raw-first confirmatory adjudication invariant.

Primary adjudication invariant:
1. Branch choice never changes the raw-first confirmatory policy.

## Risks, Failure Modes, and Mitigations
1. **Generic perturbation fragility**:
   - mitigated by matched random controls at medium and strong severities.
2. **Centering-method confound**:
   - mitigated by raw-first primary kernel adjudication; centered remains diagnostic.
3. **Rescaling caveat**:
   - matched random controls stabilize specificity contrasts, but absolute magnitudes may still be inflated by intervention-time rescaling not seen in training.
4. **Baseline floor limitations**:
   - explicit floor gate + deterministic fallback + floor-limited exclusion policy.
5. **Seed-independence caveat**:
   - seed variation measures task-sampling variability, not model-training variability.
6. **Model-specificity risk**:
   - cross-model replication reduces but does not eliminate architecture-specific confounding.
7. **AbsPE analog non-equivalence**:
   - GPT-2 frequency intervention is exploratory analog, not direct RoPE-equivalent evidence.
8. **Fine-grained sweep incompleteness**:
   - `b8/b16/b64` are currently incomplete; they may refine centered-diagnostic defaults but are not used to redefine primary confirmatory readouts.
9. **Random-control overlap sensitivity**:
   - confirmatory random controls use three draws per seed; overlap diagnostics are mandatory and unusually high-overlap rows are explicitly flagged in H3 reporting.
10. **Task-family overlap caveat**:
   - copy-offset and delayed-copy are mechanism-homologous, so confirmatory interpretation explicitly checks content-gated task contrast to avoid copy-only dilution artifacts.
11. **Cross-model seed-correlation caveat**:
   - shared numeric seeds across models can induce correlated difficulty patterns; replication is treated as robustness evidence and flagged when measured correlation is high.
12. **Phase 2D low-power caveat**:
   - Phase 2D cell-level estimates use 5 seeds; 7-seed reruns are required before headline-level claims.

## Artifact Schema and Output Layout
Output root:
1. `results/experiment2/<phase>/<run_id>/...`.

### GPU Execution Policy (Operational Default)
1. Experiment 2 run manifests must be generated with explicit GPU device assignment (`--device cuda` or `--device cuda:<id>`).
2. Default command:
   - `.venv/bin/python experiment2/run.py --phase all --device cuda --run-id <run_id> --output-root results/experiment2 --print-summary`
3. Convenience script:
   - `scripts/experiment2_build_manifests_gpu.sh <run_id> <phase|all> <device>`
4. CPU is allowed only as explicit fallback for planning/validation (`--allow-cpu-fallback`); confirmatory/exploratory production runs are GPU-targeted.
5. Device assignment is serialized per cell (`device` field in `manifest.jsonl`) and audited in phase summaries.

Per-condition required files:
1. `run_config.json`
2. `task_metrics.parquet`
3. `kernel_track_a_summary.parquet`
4. `kernel_track_b_raw_summary.parquet`
5. `kernel_track_b_centered_summary.parquet`
6. `intervention_audit.json`

Required metadata fields:
1. Removed channel/component indices.
2. Frequency-band definition and ordering mode.
3. `norm_drift_median_abs_ratio` and pass/fail flag.
4. Baseline-floor status and fallback-applied flag.
5. Generator version hash and intervention-audit hash.
6. Tier-1 frozen dependency bins and baseline-difficulty quartile bins.

Phase-level files:
1. `aggregate_task_metrics.parquet`
2. `aggregate_kernel_metrics.parquet`
3. `specificity_metrics.parquet` (`S_ratio`, `S_diff`, ratio stability flag)
4. `gate_evaluation.json`
5. `reuse_decisions.json`
6. `promotion_guard.json`
7. `claim_guard.json`

Experiment-level files:
1. `decision_summary.json` (H1-H5 labels, CI/precision flags, Holm-adjusted p-values)
2. `norm_family_decomposition.parquet`
3. `h4_interaction_exploratory.parquet`
4. `protocol_revision_log.json`

## Runtime and Compute Estimate (Recomputed)
Assumptions:
1. Single-GPU baseline.
2. Task-only settings: about 1-2 minutes.
3. Task+kernel settings: about 6-10 minutes.
4. No repeated OOM/failed-run loops.

### Run-count summary
1. Phase 2A: `231`.
2. Phase 2B: `1932`.
3. Confirmatory core: `2163`.
4. Phase 2C exploratory (including span bridge): `1122`.
5. Phase 2D exploratory: `960` (+ up to `40` fallback baselines).
6. Full nominal total: `4245` settings (up to `4285` with fallback baselines).

### Wall-clock estimates (1 GPU)
| scope | low | likely | high |
| --- | --- | --- | --- |
| Confirmatory core (2A+2B) | ~150h | ~202h | ~271h |
| Full pipeline (2A-2D) | ~272h | ~394h | ~545h |

### Wall-clock estimates (2 balanced GPUs)
| scope | low | likely | high |
| --- | --- | --- | --- |
| Confirmatory core (2A+2B) | ~75h | ~101h | ~136h |
| Full pipeline (2A-2D) | ~136h | ~197h | ~273h |

Interpretation:
1. Confirmatory core is substantially larger under random-draw + Tier-1 seed hardening and is best run on at least 2 GPUs.
2. Full exploratory expansion remains materially larger and should be scheduled after confirmatory gate evaluation.

## Protocol Revision Audit Trail
| issue_id | change_made | reason | impact_on_compute | impact_on_inference_strength | deferred_or_resolved |
| --- | --- | --- | --- | --- | --- |
| C1 | Added explicit quantitative thresholds for all gates/hypotheses | Prevent post-hoc threshold drift | None | High improvement | Resolved |
| C2 | Added paired sign-flip p-values + BCa CI + Holm | Inferential reproducibility | Low | High improvement | Resolved |
| C3 | Added deterministic generator specs for all 4 tasks | Cross-lab reproducibility | None | High improvement | Resolved |
| C4 | Added exact frequency bands/severity/removal order/nestedness | Intervention reproducibility | None | High improvement | Resolved |
| C5 | Increased confirmatory synthetic seeds to 7 | Remove exact-test p-value ceiling under two-sided sign-flip + Holm | Moderate increase | High improvement | Resolved |
| S5 | Added `random_medium` controls | Matched specificity at medium severity | Moderate increase | Medium improvement | Resolved |
| S6 | Added baseline-floor gate + fallback + exclusion | Avoid floor-driven false negatives | Low | Medium improvement | Resolved |
| S7 | Kept H4 exploratory with explicit power caveat | Avoid overclaim with norm imbalance | None | Medium improvement | Resolved |
| S8 | Required H3 medium+strong directional consistency | Reduce mediation overclaim risk | Low | Medium improvement | Resolved |
| S9 | Hardened Tier-1 stratification, counts, difficulty conditioning, and binning validity check | Reproducible transfer readout with diagnostic validity | Low | Medium improvement | Resolved |
| S10 | Added medium-range transition coverage (`spans={32,64,96}`) as a secondary confirmatory sensitivity channel in Phase 2B, with broader exploratory bridge retained in Phase 2C | Address 17-127 coverage gap while preserving primary short-vs-long confirmatory labels | Moderate increase | Medium improvement | Resolved |
| M8 | Added 2A->2B hash-gated reuse policy | Reproducibility + compute efficiency | Decrease | Medium improvement | Resolved |
| M9 | Added L2-rescaling caveat in risk register | Interpretation transparency | None | Medium improvement | Resolved |
| M10 | Added seed-independence caveat | Generalization transparency | None | Medium improvement | Resolved |
| M11 | Fully specified 2C/2D matrices, triggers, and totals | Remove under-specification | High increase | Medium improvement | Resolved |
| M12 | Fully specified DCT-based AbsPE analog intervention | Remove asymmetry vs RoPE intervention spec | Low | Medium improvement | Resolved |
| M13 | Added power/MDE section with underpowered downgrade rule | Improve inferential transparency under finite seeds | Low | Medium improvement | Resolved |
| M14 | Replaced early-only head ranking with scope-aligned early/deep head rankings | Remove deep-layer interpretation confound in Phase 2D | None | Medium improvement | Resolved |
| M15 | Added random-target overlap diagnostics for random controls | Reduce sensitivity to adversarial single-draw random masks | None | Medium improvement | Resolved |
| M16 | Added directional-selectivity labels for H1/H2 | Distinguish selective effects from broad degradation with ordering only | None | Medium improvement | Resolved |
| M17 | Added explicit H5 single-seed inferential limitation | Prevent misuse of seed-level inferential machinery on Tier-1 | None | Medium improvement | Resolved |
| M18 | Added Phase 2C 5-seed caveat + 7-seed promotion rule | Prevent overclaiming from lower-power exploratory slices | Low | Medium improvement | Resolved |
| M19 | Added content-gated safeguard in Phase 2B gate | Prevent copy-family tasks from solely driving class-level support labels | None | Medium improvement | Resolved |
| M20 | Added baseline-conditioned selectivity sensitivity check | Reduce bias in selectivity labeling when non-target baseline headroom is limited | None | Medium improvement | Resolved |
| M21 | Added cross-model seed-correlation acknowledgement and reporting rule | Clarify replication-independence limits under shared numeric seeds | None | Medium improvement | Resolved |
| M22 | Added Phase 2D low-power caveat + 7-seed promotion rule | Prevent overclaiming from sparse per-cell exploratory mechanism estimates | Low | Medium improvement | Resolved |
| B1 | Selected Branch B from completed Experiment 1 evidence; moved `b8/b16/b64` to refinement-only status | Remove stale pending status and align protocol with observed heterogeneity | None | Medium improvement | Resolved |
| M23 | Added long-offset feasibility sweep + auto-lock artifact (`long_offset_lock.json`) for confirmatory long-task generation/fallback | Replace floor-limited hardcoded long spans with data-calibrated globally fixed long-range regime | Moderate increase | High improvement | Resolved |
| M24 | Expanded Phase 2A to include `long_range_retrieval` (3-task PoC) and updated confirmatory counts | Improve long-task mechanism coverage in confirmatory entry gate | Moderate increase | Medium improvement | Resolved |

## Scale-Up Amendment (7–8B Profile, Preparation Stage)
Status:
1. 1B confirmatory program is closed as finalized negative-result evidence (`H1/H2/H3 = fail`) with headroom/feasibility diagnostics preserved.
2. Scale-up execution is prepared as a separate profile and does not retroactively alter 1B adjudication.

Scale-up model profile (`scaleup_78b`):
1. `olmo-2-7b` (`allenai/OLMo-2-1124-7B`)
2. `llama-3.1-8b` (`meta-llama/Meta-Llama-3.1-8B`)
3. `gemma-7b` (`google/gemma-7b`)

Operational policy:
1. Default profile remains `legacy_1b`.
2. Scale-up is opt-in via `--model-profile scaleup_78b`.
3. Long-offset lock policy is amended for scale-up only:
   - `--long-task-feasibility-policy strict_two_task` keeps legacy behavior (both long tasks required).
   - `--long-task-feasibility-policy retrieval_fallback` resolves lock as:
     1. use strict offsets if strict yields `>=2` contiguous offsets,
     2. else use `long_range_retrieval`-only feasible offsets if that yields `>=2`,
     3. else fail lock as before.
   - lock payload records strict and fallback diagnostics (`strict_selected_long_offsets`, `fallback_selected_long_offsets`, `active_long_tasks`, `lock_resolution_mode`).
4. H1/H2 endpoint policy for scale-up runs is pre-registered via run metadata:
   - `--h12-endpoint-policy co_primary_raw_headroom`
   - gate/decision artifacts carry `h12_endpoint_policy` and `h12_endpoint_adjudication`.
   - legacy runs remain `raw_primary` unless explicitly overridden.
5. Efficiency remediation policy for scale-up execution (rigor-safe):
   - GPUs constrained to `0/1/2` only.
   - Experiment 1 continuity is reduced to diagnostic minimum (`wiki40b_en_pre2019`, `seq_len=1024`, `Track A + one Track B pass`).
   - Feasibility execution is seed-sharded (`seed=0,1,2`) across GPUs `0/1/2` and finalized from shared artifacts.
   - Experiment 2 pipeline defaults use `BATCH_SIZE_SYNTH=24` and `BATCH_SIZE_TIER1=24`; deterministic OOM-halving fallback remains authoritative.
   - Phase 2A-only capture pruning is enabled for scale-up runs:
     1. for `phase2a` non-baseline rows, adapter capture runs with `include_logits=false` and `capture_attention=false`,
     2. kernel accumulation (Track A/raw/centered) is baseline-only in Phase 2A,
     3. baseline rows remain unchanged for floor and baseline diagnostics.
   - Phase 2A pilot/full use `synthetic_count=50` (speed remediation); Phase 2B confirmatory collection is unchanged.
   - Run metadata fields:
     - `phase2a_efficiency_profile = capture_pruned_nonbaseline_count50`
     - `phase2b_confirmatory_collection_unchanged = true`.
6. Phase 2B cross-task-sign criterion is adaptive to active long-task count:
   - two long tasks: preserve existing cross-task sign agreement rule,
   - one long task (retrieval fallback): apply single-long directional consistency rule; criterion remains gating.
7. Experiment 1 continuity OOM recovery policy:
   - if scale-up Track B at `seq_len=1024` fails mechanically (OOM), rerun only failed models at `seq_len=512`,
   - keep successful model artifacts intact,
   - apply allocator guard `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

### GQA interpretation note (important for interventions)
1. For grouped-query attention models (e.g., Llama 3.1 8B), K/V heads are shared across multiple Q heads.
2. RoPE channel-pair ablation in K projections therefore propagates to all Q heads mapped to the same KV head.
3. This is consistent with the spectral interpretation (removing/reweighting positional frequency components), but audit fields must report:
   - `num_query_heads`
   - `num_key_value_heads`
   - `gqa_repeat_factor`
   - `kv_sharing_effect_note`

### Ready command templates (do not launch in this document pass)
Adapter capture smoke (real weights, profile-safe loader path):
```bash
.venv/bin/python scripts/experiment2_adapter_smoke.py --models all --device cuda:0 --include-logits
```

Lightweight Experiment 1 continuity check (Track A + single Track B pass):
```bash
.venv/bin/python experiment1/run.py tokenize --model-profile scaleup_78b --model all --dataset wiki40b_en_pre2019 --seq-len 1024
.venv/bin/python experiment1/run.py track-a --model-profile scaleup_78b --model all --dataset wiki40b_en_pre2019 --seq-len 1024 --device cuda
.venv/bin/python experiment1/run.py track-b --model-profile scaleup_78b --model all --dataset wiki40b_en_pre2019 --seq-len 1024 --device cuda --track-b-centering-mode legacy_per_position --track-b-output-group track_b_scaleup78b_rawcheck
```

Track B OOM recovery (failed scale-up models only, seq_len=512):
```bash
bash scripts/scaleup78b_trackb_512_recovery_launch.sh <run_tag>
```

Experiment 2 feasibility sweep at scale (seed-sharded workflow):
```bash
.venv/bin/python experiment2/run.py --mode feasibility-build --model-profile scaleup_78b --h12-endpoint-policy co_primary_raw_headroom --run-id exp2_feas_scaleup_<timestamp> --output-root results/experiment2 --device cuda --feasibility-offsets 8,12,16,24,32,48,64,96,128 --lock-candidate-offsets 8,12,16,24,32,48,64,96,128 --floor-threshold 0.15 --print-summary

# run three seed shards in parallel (seed=0,1,2) with --mode execute --feasibility-task-only
.venv/bin/python experiment2/run.py --mode execute --manifest results/experiment2/feasibility/exp2_feas_scaleup_<timestamp>/manifest.jsonl --device cuda --output-root results/experiment2 --feasibility-task-only --synthetic-eval-mode restricted --candidate-size 10 --seed-start 0 --seed-end 0 --print-summary
.venv/bin/python experiment2/run.py --mode execute --manifest results/experiment2/feasibility/exp2_feas_scaleup_<timestamp>/manifest.jsonl --device cuda --output-root results/experiment2 --feasibility-task-only --synthetic-eval-mode restricted --candidate-size 10 --seed-start 1 --seed-end 1 --print-summary
.venv/bin/python experiment2/run.py --mode execute --manifest results/experiment2/feasibility/exp2_feas_scaleup_<timestamp>/manifest.jsonl --device cuda --output-root results/experiment2 --feasibility-task-only --synthetic-eval-mode restricted --candidate-size 10 --seed-start 2 --seed-end 2 --print-summary

.venv/bin/python experiment2/run.py --mode feasibility-finalize --model-profile scaleup_78b --run-id exp2_feas_scaleup_<timestamp> --output-root results/experiment2 --feasibility-offsets 8,12,16,24,32,48,64,96,128 --lock-candidate-offsets 8,12,16,24,32,48,64,96,128 --long-task-feasibility-policy retrieval_fallback --floor-threshold 0.15 --print-summary
.venv/bin/python experiment2/run.py --mode lock-long-offsets --model-profile scaleup_78b --sweep-run-id exp2_feas_scaleup_<timestamp> --output-root results/experiment2 --feasibility-offsets 8,12,16,24,32,48,64,96,128 --lock-candidate-offsets 8,12,16,24,32,48,64,96,128 --long-task-feasibility-policy retrieval_fallback --floor-threshold 0.15 --apply --print-summary
```

## Mapping to Existing Documents
1. `experiment1/experiment1overview.md`:
   - carries forward Track A/Track B role definitions and caveat framing.
2. `experiment1/experiment1results.md`:
   - grounds granularity sensitivity, raw invariance, and norm/family heterogeneity constraints.
3. `results/reports/track_b_granularity_norm_v1/summary.md`:
   - quantitative source for Branch B selection and centered-diagnostic sensitivity framing.
4. `overview.tex`:
   - theoretical links among normalization geometry, positional representation, attention kernels, and frequency-domain reasoning.
