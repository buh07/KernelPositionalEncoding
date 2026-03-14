# TODO

## Formal Phase2B + Cross-Model Pair Sweep Expansion (Active)

Policy:
- Execution order is locked:
  1. finish formal cross-model Option B Phase2B (`execute -> posthoc-centered -> reanalyze`);
  2. run expanded exploratory pair sweeps.
- GPU scope is locked to `0/1` for this pass.
- Pair sweeps remain exploratory and non-retroactive to confirmatory promotion.

- [ ] Git push hygiene pass (code/docs/tests only)
  Acceptance:
  1. `.gitignore` excludes local HF cache/home, LaTeX build products, and ad-hoc probe/result folders.
  2. staged commit excludes `data/experiment1/**` manifest churn and other generated artifacts.
  3. staged-file size gate (20MB) and HF token diff-scan pass before push.
  4. commit is pushed to `origin/main`.

- [ ] GPU 0/1-only orchestration hardening
  Acceptance:
  1. scale-up pipeline scripts propagate a configurable GPU list across feasibility, phase2a pilot/full, phase2b execute, and posthoc.
  2. no `*_g2` tmux sessions are spawned when GPU list is `0 1`.
  3. phase2b posthoc autoscale inherits the same GPU-list policy.

- [ ] Formal cross-model Phase2B completion (primary verdict path)
  Acceptance:
  1. Option B runs under retrieval-fallback lock with non-overlap lock candidates `24,32`.
  2. run metadata records `h12_endpoint_policy=co_primary_raw_headroom`.
  3. final artifacts exist: `gate_evaluation.json`, `decision_summary.json`, `promotion_guard.json`.
  4. centered backlog is zero before final promotion logic.

- [ ] Exploratory pair-sweep expansion (after formal Phase2B)
  Pre-specified matrix:
  1. retrieval pair sweep:
     - `olmo-2-7b` spans `24,32`
     - `gemma-7b` spans `16,24`
  2. mirror `local_key_match` pair sweep:
     - models `llama-3.1-8b`, `olmo-2-7b`, `gemma-7b`
  3. shared settings:
     - pair indices `0,8,16,24,32,40,48,56`
     - seeds `0,1,2`
     - interventions `none`, `ablate_pair_<idx>`, `random_pair(draw=0,1,2)`
     - `synthetic_count=100`
  Reporting requirements:
  1. per-pair effects: raw, baseline-normalized, headroom-normalized.
  2. direction agreement flags between raw and headroom-normalized effects.
  3. multiple-comparison correction: Holm within each pre-specified family (`model x task x span`).
  4. floor-exclusion diagnostics: baseline pass-rate and excluded row counts per `model/task/span`.
  5. exploratory-only claim boundary in report memo.

## Quick Hypothesis Stress-Tests (Active)

Policy:
- Exploratory-only stress tests under `results/experiment2/quick/<run_tag>/`.
- Non-retroactive: these runs do not supersede authoritative confirmatory runs by default.
- Endpoint metadata is fixed to `h12_endpoint_policy=co_primary_raw_headroom` (raw + headroom-normalized), non-gating.

- [x] Stage 1A: fixed-span retrieval slope test (immediate)
  Locked matrix:
  1. model: `llama-3.1-8b`
  2. task: `long_range_retrieval` only
  3. spans: `32,48,64` (forced; no mixed-span rows)
  4. seeds: `0,1,2`
  5. interventions: `none`, `ablate_high_strong`, `ablate_low_strong`, `random_strong` with `draw=0,1,2`
  6. runtime flags: `synthetic_count=100`, `track_a_enabled=false`, `centered_compute=defer`, `candidate_size=10`
  tmux sessions:
  1. `exp2_qslope_g0` (GPU 0; seeds 0-1)
  2. `exp2_qslope_g1` (GPU 1; seed 2)
  Acceptance:
  1. manifest rows are synthetic-only retrieval rows with forced spans.
  2. per-span report emits absolute drop, baseline-normalized drop, headroom-normalized drop.
  3. monotonic check block exists for `32->48->64`.
  Result:
  1. Completed as `quick_hstress_20260313_220254` (synthetic retrieval-only forced spans `32,48,64`).
  2. Artifacts:
     - `results/experiment2/quick/quick_hstress_20260313_220254/reports/slope/summary.json`
     - `results/experiment2/quick/quick_hstress_20260313_220254/reports/slope/decision_memo.md`

- [x] Stage 1B: band-dose curve
  Locked matrix:
  1. model/task/spans/seeds same as Stage 1A
  2. interventions: `none`, `ablate_high_medium`, `ablate_high_strong`, `ablate_low_medium`, `ablate_low_strong`, `random_strong` with `draw=0,1,2`
  tmux sessions:
  1. `exp2_qdose_g0` (GPU 0; seeds 0-1)
  2. `exp2_qdose_g1` (GPU 1; seed 2)
  Acceptance:
  1. medium vs strong dose effects are reported per span.
  2. summary artifacts are written to `results/experiment2/quick/<run_tag>/reports/`.
  Result:
  1. Completed as `quick_hstress_20260313_220254`.
  2. Artifacts:
     - `results/experiment2/quick/quick_hstress_20260313_220254/reports/dose/summary.json`
     - `results/experiment2/quick/quick_hstress_20260313_220254/reports/dose/decision_memo.md`

- [x] Stage 2A: sparse single-pair spectral sweep
  Locked matrix:
  1. pair indices: `0,8,16,24,32,40,48,56`
  2. spans: `32,48,64`
  3. seeds: `0,1,2`
  4. interventions per run: `none`, `ablate_pair_<idx>`, `random_pair` (`draw=0,1,2`)
  5. runtime flags: `synthetic_count=100`, `track_a_enabled=false`, `centered_compute=defer`, `candidate_size=10`
  tmux sessions:
  1. `exp2_qpair_g0` (GPU 0; seeds 0-1)
  2. `exp2_qpair_g1` (GPU 1; seed 2)
  Acceptance:
  1. intervention audit confirms single-pair removal for `ablate_pair_<idx>`.
  2. random-pair control is deterministic for fixed `(seed,draw)`.
  3. pair-sweep summary exists (`summary.json`, `decision_memo.md`).
  Result:
  1. Completed as `quick_hstress_stage2_20260313_223708` (pair sweep phase).
  2. Artifacts:
     - `results/experiment2/quick/quick_hstress_stage2_20260313_223708/reports/pair_sweep/summary.json`
     - `results/experiment2/quick/quick_hstress_stage2_20260313_223708/reports/pair_sweep/decision_memo.md`

- [x] Stage 2B: semi-natural wiki bridge panel
  Locked matrix:
  1. source: tokenized `wiki40b_en_pre2019` eval rows
  2. probe: long-gap repeated-token retrieval targets at spans `32,48,64`
  3. seeds: `0,1,2`
  4. interventions: `none`, `ablate_high_strong`, `ablate_low_strong`, `random_strong` (`draw=0,1,2`)
  5. reporting: emit full-vocab and restricted metrics from identical probe sets
  tmux sessions:
  1. `exp2_qnat_g0` (GPU 0; seeds 0-1)
  2. `exp2_qnat_g1` (GPU 1; seed 2)
  Acceptance:
  1. probe artifacts are explicitly tagged exploratory-only.
  2. merged natural-panel summary exists (`summary.json`, `decision_memo.md`).
  Result:
  1. Completed as `quick_hstress_stage2_20260313_223708` (semi-natural merged panel).
  2. Artifacts:
     - `results/experiment2/quick/quick_hstress_stage2_20260313_223708/reports/natural/summary.json`
     - `results/experiment2/quick/quick_hstress_stage2_20260313_223708/reports/natural/decision_memo.md`

## Methodological Limitations (Active)

- [ ] Distractor difficulty asymmetry across task families
  Acceptance: `gate_evaluation.json` includes explicit advisory diagnostics for task-family headroom and class-level baseline/headroom summaries so copy-vs-retrieval difficulty differences are audit-visible.
- [ ] Long-task headroom compression confound for H2
  Acceptance: non-gating baseline-normalized and headroom-normalized H1/H2 contrasts (effect, CI, p, Holm, n, direction) are emitted in phase gate payloads.
- [ ] Raw-vs-normalized contrast divergence risk
  Acceptance: gate payload includes `headroom_confound_risk` and per-hypothesis raw-vs-normalized direction agreement fields; confirmatory raw criteria remain unchanged.
- [ ] Short/long span-overlap interpretation caveat
  Acceptance: gate payload includes a span-overlap diagnostic from `experiment2/long_offset_lock.json` and a required interpretation note when long offsets overlap short regime.
- [ ] Operational backfill for existing authoritative runs
  Acceptance: run `reanalyze` for active authoritative phase2a/phase2b runs and update report text from new gate fields (no execute/posthoc rerun for this patch).

## Option A: Extended Long-Range Feasibility Sweep (Active)

- [x] Run fresh feasibility sweep for offsets `{16,24,32,48,64}`
  Acceptance: complete `--mode feasibility-sweep` run with `--feasibility-task-only`, `--floor-threshold 0.15`, and artifacts written under `results/experiment2/feasibility/<run_id>/`.
  Result: completed as `exp2_feas_optA_20260310_204817`.
- [x] Compute per `(model, task, offset)` seed pass-rates at floor `0.15`
  Acceptance: `offset_pass_rates.parquet` is parsed and summarized with pass criterion `pass_rate >= 2/3`.
  Result: `results/experiment2/feasibility/exp2_feas_optA_20260310_204817/offset_pass_rates.parquet`.
- [x] Compute max feasible offset by model/task for `llama-3.2-1b` and `olmo-1b`
  Acceptance: `optionA_feasibility_summary.json` includes max feasible offset fields for both models and long tasks.
  Result: `results/experiment2/feasibility/exp2_feas_optA_20260310_204817/optionA_feasibility_summary.json`.
- [x] Decide whether exploratory rerun at `>=32` is viable
  Acceptance: summary includes boolean `viable_ge_32` and short decision memo text.
  Result: `viable_ge_32=true`; memo at `results/experiment2/feasibility/exp2_feas_optA_20260310_204817/optionA_decision_memo.txt`.
- [x] Keep lock unchanged pending follow-up decision
  Acceptance: no write to `experiment2/long_offset_lock.json` in this pass.
  Result: lock artifact unchanged; no `--mode lock-long-offsets --apply` run executed.

## Scale-up 7–8B Program (Active)

Program transition note:
- The 1B confirmatory campaign is closed as a completed negative-result program (`H1/H2/H3 = fail`) with documented headroom/feasibility diagnostics.
- Follow-on work moves to a new scale profile instead of extending 1B execution.

- [x] Add profile-plumbed model selection (`legacy_1b` default, `scaleup_78b` opt-in)
  Acceptance: `experiment1/run.py`, `experiment2/run.py`, and `scripts/download_assets.py` all support `--model-profile`; legacy behavior unchanged by default.
- [x] Extend adapter coverage for `olmo-2-7b`, `llama-3.1-8b`, `gemma-7b`
  Acceptance: `get_adapter()` resolves all three names and capture smoke succeeds without changing legacy adapters.
- [x] Add GQA audit transparency fields
  Acceptance: intervention audit outputs include `num_query_heads`, `num_key_value_heads`, `gqa_repeat_factor`, and explicit KV-sharing note.
- [x] Add scale-up continuity command templates (no launch in this pass)
  Acceptance: docs include ready commands for lightweight Exp1 continuity and Exp2 feasibility sweep under `scaleup_78b`.
- [x] Adapter capture integrity guardrails
  Acceptance: projection adapters are covered by layer-count unit tests for Llama/OLMo2/Gemma paths and no duplicate-layer capture occurs.
- [x] Register H1/H2 endpoint policy for scale-up runs
  Acceptance: `--h12-endpoint-policy` is available and gate/decision payloads include `h12_endpoint_policy` metadata (`raw_primary` or `co_primary_raw_headroom`).
- [ ] Execute scale-up readiness gate (operations)
  Acceptance:
  1. download `scaleup_78b` checkpoints,
  2. run real-weight adapter smoke for all 3 models,
  3. run lightweight Exp1 continuity,
  4. run Exp2 scale feasibility sweep,
  5. decide lock/apply path for full scale Exp2 launch.
  Current snapshot:
  - Step 1 complete: `olmo-2-7b`, `llama-3.1-8b`, and `gemma-7b` weights downloaded.
  - Step 2 complete: real-weight adapter smoke passes for all 3 models (`scripts/experiment2_adapter_smoke.py --models all --include-logits`).
  - Steps 3-5 are being relaunched under the efficiency-remediated pipeline below.

- [ ] Apply scale-up efficiency remediation (GPUs 0/1/2 only, rigor-safe)
  Acceptance:
  1. Stop current scale-up sessions and mark run superseded for audit.
  2. Relaunch reduced Exp1 continuity scope (`wiki40b_en_pre2019`, `seq_len=1024`, `Track A + single Track B pass`).
  3. Run feasibility as seed-sharded parallel execution (`seed 0/1/2` on GPUs `0/1/2`) using split modes (`feasibility-build` -> sharded `execute` -> `feasibility-finalize`).
  4. Continue lock apply -> Phase 2A pilot (`execute + reanalyze + calibrate-floor`, no posthoc-centered) -> Phase 2A full -> Phase 2B pipeline.
  5. Use `BATCH_SIZE_SYNTH=24` and `BATCH_SIZE_TIER1=24` for Exp2 execute/posthoc defaults with unchanged OOM halving.
  Current policy notes:
  - Exp1 continuity remains diagnostic and non-gating for Exp2 confirmatory adjudication.
  - Feasibility sharding is operational parallelism only; inferential semantics unchanged.

- [ ] Complete scale-up full Experiment 2 chain (operations)
  Acceptance:
  1. `lock-long-offsets --apply` succeeds from scale-up feasibility sweep,
  2. Phase 2A pilot executes, recalibrates floor, and proceeds,
  3. Phase 2A full execute + posthoc-centered + reanalyze complete,
  4. Phase 2B execute + posthoc-centered + reanalyze complete,
  5. final gate/decision artifacts produced under scale-up run IDs with `h12_endpoint_policy=co_primary_raw_headroom`.
  Current snapshot:
  - Pipeline entrypoint: `scripts/scaleup78b_exp2_full_pipeline.sh`.
  - Active orchestrator session: `scaleup78b_exp2_pipe`.

### Span-Overlap Remediation Restart (Active: Option C -> Option B)

Status note:
- The prior scale-up lock (`selected_long_offsets=[8,12,16,24,32]`) is treated as overlap-confounded for primary H2 interpretation because `8/12/16` intersect the short regime (`<=16`).
- Current overlap-confounded runs are superseded for primary-claim adjudication once this restart begins.

- [ ] Option C exploratory deep-span branch (llama-only)
  Acceptance:
  1. Run with `--model-allowlist llama-3.1-8b`.
  2. Feasibility/lock candidates fixed to `32,48,64`.
  3. Lock resolves under `retrieval_fallback` with `active_long_tasks=["long_range_retrieval"]`.
  4. Phase 2A/2B execute -> posthoc-centered -> reanalyze complete.
  5. Gate/decision payload explicitly reports `confirmatory_applicability=false` and `exploratory_reason=single_model_design`.

- [ ] Option B cross-model non-overlap branch (primary remediation path)
  Acceptance:
  1. Use full `scaleup_78b` profile (llama/olmo/gemma).
  2. Feasibility/lock candidates fixed to `24,32` (no short-regime overlap).
  3. Lock resolves under `retrieval_fallback` with non-overlap long offsets.
  4. Phase 2A/2B execute -> posthoc-centered -> reanalyze complete.
  5. Updated gate artifacts show `span_overlap_diagnostic.overlaps_short_regime=false` for active lock.

- [ ] Orchestration on GPUs 0/1/2 (sequential branch policy)
  Acceptance:
  1. Launch Option C then Option B sequentially via tmux.
  2. Preserve strict claim boundaries (Option C exploratory-only; Option B used for cross-model interpretation).
  3. Maintain restricted evaluation (`candidate_size=10`), floor policy, and statistical thresholds unchanged.

### Pilot/Runtime Efficiency Amendment (Active, Rigor-Safe)

- [ ] Use `floor-prepass` for Phase 2A pilot instead of full execute
  Acceptance:
  1. Pilot baseline rows are processed via `run.py --mode floor-prepass` seed-sharded on GPUs `0/1/2`.
  2. Pilot no longer requires attention/kernel capture or centered-path work.
  3. Floor calibration runs successfully from pilot outputs without requiring pilot `aggregate_task_metrics.parquet`.
- [ ] Calibrate floor from `floor_decisions` when pilot aggregate metrics are absent
  Acceptance:
  1. `calibrate-floor` supports `phase2a/<run_id>/floor_decisions/*.json` as a valid source.
  2. Output includes `calibration_source` to distinguish `aggregate_task_metrics` vs `floor_decisions_baseline_accuracy`.
  3. Threshold rule remains unchanged (`0.15` / `0.13` / stop).
- [ ] Remove CPU transfer bottleneck in quick baseline scoring
  Acceptance:
  1. `_quick_baseline_accuracy` evaluates with token logits on the active device (no forced `output_device=cpu` copy).
  2. Metric semantics remain unchanged (same restricted/full-vocab accuracy definitions).
- [ ] Option C posthoc policy: skip centered fill; Option B remains strict
  Acceptance:
  1. Option C Phase 2B can skip posthoc-centered and still run reanalysis with explicit centered-pending allowance.
  2. Option B Phase 2B keeps posthoc-centered mandatory before final reanalysis/promotion.

### Phase2A Speed Remediation (Active)

- [x] Apply Phase2A non-baseline capture pruning (runtime)
  Acceptance:
  1. In `phase2a` with `intervention != none`, execute path calls adapter capture with `include_logits=false` and `capture_attention=false`.
  2. `return_token_logits=true` remains enabled so task metrics are unchanged in definition.
  3. Baseline (`intervention=none`) behavior remains unchanged for floor/baseline diagnostics.
- [x] Gate Phase2A kernel accumulation to baseline rows only
  Acceptance:
  1. `phase2a` non-baseline rows skip Track A/raw/centered kernel accumulation.
  2. `phase2b` behavior is unchanged (kernels still collected for confirmatory H3 paths).
- [x] Force scale-up Phase2A `seq_len>=1024` synthetic-like rows to start at batch size 1
  Acceptance:
  1. `_batch_size_for_row` returns `1` for `phase2a` rows on `{olmo-2-7b,llama-3.1-8b,gemma-7b}` at `seq_len>=1024`.
  2. Existing deterministic OOM-halving fallback remains unchanged.
- [x] Run Phase2A pilot/full with `synthetic_count=50` only
  Acceptance:
  1. Scale-up pipeline passes `--synthetic-count 50` to Phase2A pilot/full execute shards.
  2. Phase2B execute/posthoc counts remain unchanged.
  3. Run metadata/doc notes include:
     - `phase2a_efficiency_profile=capture_pruned_nonbaseline_count50`
     - `phase2b_confirmatory_collection_unchanged=true`.
- [x] Restart and validate runtime recovery
  Acceptance:
  1. Stop `scaleup78b_p2a_exec_g0/g1/g2` and `scaleup78b_exp2_pipe`.
  2. Mark stalled run superseded in scale-up audit metadata.
  3. Relaunch from existing lock state and observe `run_config.jsonl` growth within 10–15 minutes (no repeated OOM-only loops).
  Result:
  - supersede metadata: `logs/scaleup/scaleup78b_remed_20260311_092924/superseded_by_phase2a_efficiency_restart.json`
  - relaunch tag: `scaleup78b_p2aeff_20260311_154844`
  - pilot run configs observed (`seed_0/2/4`) with no OOM retry loops in pilot worker logs.

### Scale-up Remediation (Policy + Runtime) — Active

- [x] Add scale-up-only long-task feasibility policy mode (`retrieval_fallback`)
  Acceptance:
  1. `experiment2/run.py` supports `--long-task-feasibility-policy {strict_two_task,retrieval_fallback}`.
  2. For `--model-profile scaleup_78b`, lock computation persists strict + retrieval diagnostics and can resolve to retrieval-only when strict fails.
  3. For legacy profiles, requested fallback policy is automatically downgraded to strict semantics.
- [x] Wire confirmatory manifests to lock artifact `active_long_tasks`
  Acceptance:
  1. Phase 2A/2B builders use `active_long_tasks` from `experiment2/long_offset_lock.json` for scale-up runs.
  2. Under retrieval-only fallback, `delayed_copy` is excluded from confirmatory synthetic manifests while short tasks and Tier-1 remain unchanged.
- [x] Implement adaptive Phase 2B cross-task-sign criterion for single-long-task batteries
  Acceptance:
  1. `experiment2/analysis.py` keeps original criterion when two long tasks are present.
  2. When one long task is present, cross-task-sign uses the single-long directional consistency rule and remains gating.
  3. Gate payload includes `cross_task_sign_policy`, `active_long_tasks`, `cross_task_sign_rule_version`.
- [x] Add Exp1 Track B OOM recovery path at `seq_len=512` for failed scale-up models only
  Acceptance:
  1. `experiment1` scale-up profile supports `seq_len=512`.
  2. Recovery scripts exist for rerunning only `llama-3.1-8b` and `olmo-2-7b` Track B at 512 with allocator guard `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  3. Existing successful artifacts (e.g., gemma) are not recomputed.
- [ ] Resume path from existing scale-up feasibility outputs (operations)
  Acceptance:
  1. Recompute/apply lock from `exp2_feas_scaleup_scaleup78b_eff_20260311_000436` with `retrieval_fallback`.
  2. Launch fresh authoritative Phase 2A pilot -> floor calibration -> Phase 2A full -> Phase 2B pipeline.
  3. Keep 1B artifacts frozen and non-retroactive.
  Current snapshot:
  - Step 1 complete: lock applied with `status=ok`, `lock_resolution_mode=retrieval_only_fallback`, `active_long_tasks=["long_range_retrieval"]`, `selected_long_offsets=[8,12,16,24,32]`.

## Experiment 2 Restricted-Candidate Remediation (New Authoritative Path)

Superseded run:
- `exp2_gpu345_p2ab_fixv5_20260304_164135` is marked non-authoritative for confirmatory interpretation.
- Audit snapshot: `results/experiment2/audit/20260305_150050_supersede_exp2_gpu345_p2ab_fixv5_20260304_164135.md`

### 1) Build fresh Phase 2A pilot manifests

```bash
RUN_ID="exp2_gpu345_restrictedpilot_$(date +%Y%m%d_%H%M%S)"
for phase in phase2a; do
  .venv/bin/python experiment2/run.py \
    --mode build \
    --phase "$phase" \
    --device cuda \
    --run-id "$RUN_ID" \
    --output-root results/experiment2 \
    --random-draws-confirmatory 3 \
    --tier1-seeds-phase2b 7 \
    --print-summary
done
.venv/bin/python scripts/experiment2_shard_manifests_by_model.py \
  --run-id "$RUN_ID" \
  --phases phase2a \
  --root results/experiment2
```

### 2) Run Phase 2A pilot with restricted 10-way scoring

```bash
SYNTHETIC_EVAL_MODE=restricted CANDIDATE_SIZE=10 FLOOR_THRESHOLD=0.15 \
CENTERED_COMPUTE=defer BATCH_SIZE_SYNTH=8 BATCH_SIZE_TIER1=8 \
bash scripts/experiment2_modelpin_execute.sh 3 \
  "results/experiment2/phase2a/${RUN_ID}/model_shards/llama-3.2-1b.jsonl" \
  |& tee "logs/experiment2/${RUN_ID}_g3_phase2a_pilot.log"
```

### 3) Reanalyze + calibrate floor threshold from pilot

```bash
.venv/bin/python experiment2/run.py \
  --mode reanalyze \
  --phase phase2a \
  --run-id "$RUN_ID" \
  --output-root results/experiment2 \
  --print-summary

.venv/bin/python experiment2/run.py \
  --mode calibrate-floor \
  --run-id "$RUN_ID" \
  --output-root results/experiment2 \
  --phase2a-floor-min-pass-rate 0.70 \
  --print-summary
```

### 4) Launch clean Phase 2A+2B rerun with locked threshold

- Use `selected_floor_threshold` from `results/experiment2/phase2a/<RUN_ID>/floor_recalibration.json`.
- Launch on GPUs 3/4/5 with model-pinned shards:

```bash
FLOOR_THRESHOLD="<0.15_or_0.13>" SYNTHETIC_EVAL_MODE=restricted CANDIDATE_SIZE=10 \
bash scripts/experiment2_modelpin_prepare_p2ab.sh "<NEW_RUN_ID>" results/experiment2
FLOOR_THRESHOLD="<0.15_or_0.13>" SYNTHETIC_EVAL_MODE=restricted CANDIDATE_SIZE=10 \
bash scripts/experiment2_modelpin_launch_p2ab.sh "<NEW_RUN_ID>"
```

### 5) Promotion sequence (required order)

1. Execute shards
2. `posthoc-centered`
3. `reanalyze`

## Experiment 2 Operational Follow-ups (Historical Run Notes)

Run ID: `exp2_gpu345_p2ab_fixv5_20260304_164135`

Status note (historical context):
- This section documents legacy execution flow for `exp2_gpu345_p2ab_fixv5_20260304_164135`.
- Authoritative current Phase 2B run is already non-blocked and finalized; these blocked-status notes are not current-state guidance.

### 1) Wait for execute workers to finish

- Verify tmux workers complete:
  - `tmux ls`
  - Look for `all assigned manifests complete` in:
    - `logs/experiment2/exp2_gpu345_p2ab_fixv5_20260304_164135_g3_p2ab.log`
    - `logs/experiment2/exp2_gpu345_p2ab_fixv5_20260304_164135_g4_p2ab.log`
    - `logs/experiment2/exp2_gpu345_p2ab_fixv5_20260304_164135_g5_p2ab.log`

### 2) Run posthoc centered (required before confirmatory analysis)

```bash
.venv/bin/python experiment2/run.py \
  --mode posthoc-centered \
  --phase phase2a \
  --run-id exp2_gpu345_p2ab_fixv5_20260304_164135 \
  --device cuda \
  --output-root results/experiment2 \
  --strict-posthoc \
  --print-summary

.venv/bin/python experiment2/run.py \
  --mode posthoc-centered \
  --phase phase2b \
  --run-id exp2_gpu345_p2ab_fixv5_20260304_164135 \
  --device cuda \
  --output-root results/experiment2 \
  --strict-posthoc \
  --print-summary
```

### 3) Reanalyze with patched gates/statistics

```bash
.venv/bin/python experiment2/run.py \
  --mode reanalyze \
  --phase phase2a \
  --run-id exp2_gpu345_p2ab_fixv5_20260304_164135 \
  --output-root results/experiment2 \
  --print-summary

.venv/bin/python experiment2/run.py \
  --mode reanalyze \
  --phase phase2b \
  --run-id exp2_gpu345_p2ab_fixv5_20260304_164135 \
  --output-root results/experiment2 \
  --print-summary
```

### 4) Verify confirmatory artifacts are no longer blocked

- Check `gate_evaluation.json` contains real `h1/h2` blocks and not `status=blocked_centered_pending`:
  - `results/experiment2/phase2a/exp2_gpu345_p2ab_fixv5_20260304_164135/gate_evaluation.json`
  - `results/experiment2/phase2b/exp2_gpu345_p2ab_fixv5_20260304_164135/gate_evaluation.json`
- Check `phase2b` includes:
  - `h3_confirmatory_track_b_raw_per_draw_diagnostic`
  - pooled H1/H2 `ci`, `p`, `p_holm`, `label`

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

## Exploratory: Feasibility-Conditioned Long-Range Subset Analysis

**Status**: Ready to implement. No GPU needed — uses existing parquet data from the completed authoritative run.

**Motivation**: The primary pooled H1/H2 inference includes tinyllama-1.1b, which is floor-limited on `delayed_copy` at 85.7% of rows (Table D). Its near-floor long-range baselines contribute noisy contrasts. This exploratory analysis restricts to models that pass the feasibility criterion for long-range tasks (llama-3.2-1b and olmo-1b). Model selection is driven by the pre-existing feasibility lock — not by inspecting H1/H2 outcomes.

**Reporting requirements**:
- Primary all-model pooled result remains the preregistered confirmatory analysis (unchanged, already finalized as H1=fail, H2=fail, H3=fail)
- Feasibility-conditioned subset is clearly labeled exploratory/diagnostic
- Note reduced N (2 models × 7 seeds = 14 contrasts instead of 3 × 7 = 21) and corresponding power reduction
- Document that subset selection criterion (feasibility lock) was determined before H1/H2 results were examined

### Step 1: Add `_phase2b_gate_subset()` to `analysis.py`

Add a wrapper function that:
1. Loads `aggregate_task_metrics.parquet` and `aggregate_kernel_metrics.parquet` from `phase_root`
2. Filters both DataFrames to `model.isin(subset_models)` before computing effects
3. Calls the same `_task_effects()` → `_eligible_task_rows()` → `_class_contrast()` pipeline
4. Computes pooled H1/H2 effects, CIs, p-values, Cohen's d on the filtered data
5. Does NOT apply Holm correction (single exploratory analysis, not a confirmatory family)
6. Returns a dict with keys: `subset_models`, `subset_rationale`, `h1_effect`, `h1_ci`, `h1_p`, `h2_effect`, `h2_ci`, `h2_p`, `n_contrasts`, `per_model_effects`

Key implementation detail: filter `task_eff` by model BEFORE passing to `_class_contrast()`. The existing `_class_contrast()` function groups by `(model, seed)`, so filtering models upstream is clean and correct.

```python
def _phase2b_feasibility_subset(
    task_eff: pd.DataFrame,
    kernel_delta: pd.DataFrame,
    *,
    subset_models: list[str],
    rationale: str,
) -> dict[str, Any]:
    eff = _eligible_task_rows(task_eff)
    eff = eff[eff["model"].isin(subset_models)].copy()
    # ... same contrast/CI/p-value logic as _phase2b_gate but simpler
    # No Holm correction (exploratory, not confirmatory family)
    # No H3/H5/transition panel (not needed for this diagnostic)
```

### Step 2: Wire into `evaluate_phase()` for Phase 2B

In `evaluate_phase()` (analysis.py ~line 1365), after the main `_phase2b_gate()` call, add:

```python
if phase == "phase2b":
    subset_result = _phase2b_feasibility_subset(
        task_eff, kernel_delta,
        subset_models=["llama-3.2-1b", "olmo-1b"],
        rationale="feasibility_lock_delayed_copy_floor_exclusion",
    )
    gate["feasibility_conditioned_subset"] = subset_result
```

This writes the subset results into the existing `gate_evaluation.json` as a nested block — no new files needed.

### Step 3: Run reanalysis (no GPU, ~10 seconds)

```bash
source .venv/bin/activate

.venv/bin/python experiment2/run.py \
  --mode reanalyze \
  --phase phase2b \
  --run-id exp2_phase2b_fastq_v8_20260307_170106 \
  --output-root results/experiment2 \
  --print-summary
```

### Step 4: Inspect subset results

```bash
python -c "
import json
gate = json.load(open('results/experiment2/phase2b/exp2_phase2b_fastq_v8_20260307_170106/gate_evaluation.json'))
sub = gate.get('feasibility_conditioned_subset', {})
print(json.dumps(sub, indent=2))
"
```

Expected output includes:
- `subset_models: ["llama-3.2-1b", "olmo-1b"]`
- `h1_effect`, `h1_ci`, `h1_p` — does H1 pass threshold (0.05) without tinyllama dragging it down?
- `h2_effect`, `h2_ci`, `h2_p` — is H2 still reversed even in feasibility-passing models?
- `n_contrasts` — should be 14 (2 models × 7 seeds)
- `per_model_effects` — individual model H1/H2 effects for interpretation

### Step 5: Update `overview_revised.tex` with exploratory subset results

Add the subset results to the H1/H2 sections as exploratory diagnostics. Key interpretive questions:
- If H2 remains reversed in the subset → strong evidence the reversal is real, not a TinyLlama artifact
- If H2 flips to near-zero or positive → TinyLlama was driving the reversal; interpretation changes
- If H1 crosses the 0.05 threshold in the subset → the near-miss was caused by TinyLlama dilution

### Step 6: Add test coverage

Add a test in `tests/test_experiment2_protocol_remediation.py`:

```python
def test_feasibility_conditioned_subset_filters_models():
    # Verify subset function only uses specified models
    # Verify n_contrasts matches expected (models × seeds)
    # Verify no Holm correction applied (exploratory)
```

## Nice-to-Have Follow-up

- Run both legacy and canonical spectral passes (same gate threshold) and compare:
  - matched peak counts
  - mean relative error
  - top frequency overlap
- Experiment 2: keep `_task_structured_distractors` memoization deferred unless profiling shows candidate construction is a material bottleneck; current bottleneck is model forward passes.
