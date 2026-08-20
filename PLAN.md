# PLAN — Shift-Invariant Attention rebuttal experiments

> Written for the 2026-07-27 rebuttal run. Attacked by `/adversarial PLAN.md` before
> implementation. The parent workspace topology task remains paused and unchanged; this plan governs
> only the independent `Kernel PE` repository.

## Goal

Create a self-contained, reproducible `rebuttal/` experiment package, verify it, and launch the
reviewer-requested cross-corpus SI intervention and depth analyses in tmux on physical GPUs 0 and 1.

## Non-goals

- Do not retain or claim the strong “SI is learned, not architectural” causal headline.
- Do not start matched 1.5B–3B RoPE-versus-NoPE pretraining: `SIRebuttal.md` makes it conditional on
  retaining that headline, while the chosen rebuttal narrows the claim.
- Do not change existing experiment source, prior results, manuscripts, `per_layer_si_summary.md`,
  or the pre-existing untracked `rebuttal.md`.
- Do not treat a successful launch, smoke run, or partial output as an experimental result.
- Do not commit, push, preempt another process, or use GPUs other than physical 0 and 1.

## Constraints

- Research-full mode: every run is bound to config, seed, data identity, Git SHA, and dirty-tree
  digest; code review and claim review remain separate.
- Experimental evaluation and artifact format are one-way doors. Follow
  [`rebuttal/decisions/ADR-0001-cross-corpus-si-intervention.md`](rebuttal/decisions/ADR-0001-cross-corpus-si-intervention.md)
  and obtain an independent plan `SHIP` before implementation.
- All new experiment implementation, configs, tests, launch scripts, logs, and results live under
  `rebuttal/`. Root `PLAN.md`, `HANDOFF.md`, and `.workspace/multi-agent.toml` are workflow state,
  not experiment implementation.
- Use the existing local Llama-3.1-8B, Mistral-7B-v0.1, and OLMo-2-7B weights without modifying
  them. Force eager attention because kernel subtraction operates on pre-softmax logits.
- Full runs use fixed sequence counts and never adaptively stop based on observed effects.
- The exact corpus revisions, document split, estimator, raw intervention kernel, head-bin rule,
  stochastic seed derivation, controls, statistics, inference dtype, and numerical validation
  tolerances are frozen in ADR-0001; configuration validation must reject any drift.
- Physical placement is fixed: GPU0 runs Llama then OLMo sequentially; GPU1 runs Mistral. Inside each
  `CUDA_VISIBLE_DEVICES` namespace the process uses `cuda:0`.
- Launch only after fresh `nvidia-smi` checks show both requested GPUs have no compute process and
  after the launch scripts pass a two-snapshot idle preflight.

## Design-bank clarification

- `bin/design query` returned no entry for rebuttal scope, matched RoPE/NoPE training, cross-corpus
  intervention design, or artifact format.
- Scope is therefore taken from the supplied reviews and `SIRebuttal.md`: narrow the architectural
  claim; run the major cross-corpus intervention in both directions; add the lower-priority
  depth-stratified analysis using the same frozen inputs.
- Low-risk implementation assumptions are recorded in ADR-0001. No new public API or manuscript
  claim is selected here.

## Approach

Add a small Python package under `rebuttal/src/si_rebuttal` with a tracked TOML base config and sweep
definition. It will:

1. materialize exact-revision Wikipedia and Python-code source-row/document manifests, then
   deterministic model-specific token snapshots with document-disjoint fit/evaluation partitions;
2. freeze source documents, row IDs, content hashes, tokenizer digests, and exactly 50 fit/100
   evaluation sequence IDs before model intervention;
3. estimate `g_h(Δ)` and source-corpus per-head `R²`, freeze 20 source-ranked head bins, and evaluate
   the transferred kernel on the other corpus;
4. compare transferred subtraction with an in-domain kernel reference, offset-permuted controls,
   and norm-matched random controls on identical target sequences;
5. repeat Wikipedia→code and code→Wikipedia for all three primary 7–8B models;
6. on Wikipedia→code, intervene on one whole layer at a time to relate layer-mean source `R²` to
   grouped per-head disruption; and
7. write immutable resolved configs, dataset/token digests, per-sequence losses, summaries,
   environment/GPU metadata, and completion markers below ignored `rebuttal/runs/`.

ADR-0001 fixes the paired 10,000-resample bootstrap, 200,000-permutation directional Spearman null,
compact-JSON SHA-256 seed derivation, and per-sequence averaging order. No outcome-dependent
threshold, tolerance, test, sample count, or model exclusion is allowed.

**Alternative considered and rejected:** Reusing E19 alone is insufficient because E19 transfers
`R²` scoring but does not apply a source-fitted kernel as a causal-sensitivity intervention on a
different corpus.

**Simpler option:** A single Wikipedia→code direction would answer part of the reviewer question,
but the supplied rebuttal explicitly asks for the reverse direction if compute permits. Two idle
L40 GPUs are available, so both directions are fixed before results are seen.

**Relevant ADRs:** `rebuttal/decisions/ADR-0001-cross-corpus-si-intervention.md`.

## Milestones

- [ ] **M1 — Frozen protocol, environment, and data/provenance slice.** Add `rebuttal/.gitignore`,
  a rebuttal-scoped package and exact direct dependency pins, base/sweep configs, strict validation,
  immutable-revision corpus loading, document-level splitting, row/document/token hash manifests,
  environment capture, and unit tests. Install the missing pinned PyArrow, Ruff, and basedpyright
  before any smoke or production preflight.  
  **Acceptance:** a CPU-only smoke materialization twice yields identical resolved-config and split
  digests; selected fit/evaluation document IDs, source rows, document-content hashes, and token
  hashes are disjoint; `git check-ignore rebuttal/runs/example` succeeds; imports and the pinned
  tool versions resolve from the project venv.
- [ ] **M2 — Cross-corpus intervention slice.** Add kernel/R² estimation, frozen source head bins,
  transferred and in-domain interventions, permutation/norm controls, paired uncertainty, resumable
  atomic shard outputs, and synthetic CPU tests for the exact ADR math and failure paths.  
  **Acceptance:** toy tensors prove the double-centered/smoothed source R² estimator remains
  distinct from the raw unsmoothed subtraction kernel; deterministic sorting yields exactly 20
  non-empty bins; control seeds/arrays and statistics match frozen fixtures; a toy smoke run
  exercises both directions and emits schema-valid finite summaries without production weights.
- [ ] **M3 — Depth and operations slice.** Add one-layer-at-a-time depth analysis, tmux launch/status
  scripts, exact GPU mapping, fail-closed two-snapshot idle checks, benchmark/admission report,
  manifest/finalizer, and runbook.  
  **Acceptance:** dry-run commands name only physical GPUs 0/1, use unique run roots, refuse
  collisions, and finalization rejects missing/mismatched model shards; the fixed benchmark
  extrapolation stops above 120 hours/lane or below the disk margin without changing the protocol.
- [ ] **M4 — Audited integration and coordinator verification.** Integrate only independently
  audited multi-agent patches, install missing environment tools without changing model/data
  bytes, and run the research verification gate plus reproducibility checks.  
  **Acceptance:** coordinator verification passes on the exact source/config snapshot intended for
  launch; the diff receives an independent `SHIP`.
- [ ] **M4a — Audited WikiText raw-row boundary correction.** Apply the frozen top-level-title
  regex to `text.rstrip("\r\n")` for detection only, while preserving every original row
  byte-for-byte in joined document text, source-row provenance, and content hashes. This is the
  owner-approved minimal correction for the production materialization failure on 2026-07-28:
  immutable WikiText title rows such as `" = Valkyria Chronicles III = \n"` otherwise produce zero
  documents. Implement the correction through an independently audited multi-agent patch, then run
  coordinator verification, fresh diff review, and fresh claim review in that order. No replacement
  batch root, production data materialization, GPU model preflight, benchmark, admission receipt, or
  tmux launch may begin until all M4a gates pass.  
  **Acceptance:** focused fixtures prove detection for `LF` and `CRLF`, reject non-title section
  headings, and assert exact expected joined UTF-8 text, source-row indices, document boundaries,
  and SHA-256 values constructed from the unmodified raw rows. The test suite must also prove
  fit/evaluation chunk sufficiency. The full coordinator verification and fresh diff/claim reviews
  pass on the exact correction bytes before a new immutable batch root is materialized.
- [ ] **M4b — Audited real-driver GPU snapshot correction.** Replace the invalid production
  `nvidia-smi --query-gpu=...,compute_apps.pid` invocation with separate, valid GPU-inventory and
  compute-application queries while preserving the fail-closed idle definition (`<=16 MiB`, `0%`
  utilization, and no compute PID), the two snapshots five seconds apart, and physical GPU 0/1
  placement. Implement the correction through an independently audited multi-agent patch, then
  rerun coordinator verification, fresh diff review, and fresh claim review. No replacement batch
  materialization, real-model preflight, benchmark, admission receipt, or tmux launch may begin
  until M4b passes.
  **Acceptance:** a live read-only probe on this machine shows the old field is rejected and both
  replacement commands succeed; focused tests bind the exact command shapes, associate PIDs by GPU
  UUID, accept an empty compute-app response, fail closed on a foreign UUID or malformed row,
  reject command failures, and retain the existing two-snapshot timing and idle thresholds. Full
  promotion and fresh reviews pass on the exact integrated bytes.
- [ ] **M4c — Audited selected-artifact overlap correction.** Scope the document-ID, source-row,
  and joined-document-content overlap guard to the documents referenced by the already-selected 50
  fit and 100 evaluation chunks, matching the M1 experimental-artifact criterion. Retain the current
  `schema_version = 1` materialized-domain payload exactly: `fit_documents` and `eval_documents`
  remain full partition inventories for provenance, including unused duplicates, and every field,
  ordering rule, and serialization remains unchanged. Keep the frozen document construction, IDs,
  document-ID partition assignment, tokenizer inputs, chunk selection, and selected token-overlap
  guard unchanged; do not deduplicate, drop, or repartition corpus documents. Before writing any
  materialized-domain or batch manifest, additionally compare the per-corpus union of documents
  referenced by all three tokenizers' selected fit chunks with the corresponding evaluation union;
  document ID, source row, or joined-content overlap remains a hard failure. Selected token hashes
  remain a per-tokenizer guard. Implement the correction through an independently audited
  multi-agent patch, then rerun coordinator verification, fresh diff review, and fresh claim review.
  No replacement batch materialization, real-model preflight, benchmark, admission receipt, or tmux
  launch may begin until M4c passes.  
  **Acceptance:** focused fixtures prove unused cross-partition duplicate content may remain in the
  unchanged full-inventory payload, while a selected overlap within one tokenizer or only across two
  tokenizers still fails closed before manifest write. Byte-level fixtures prove the existing
  materialized-domain payload shape and full document inventories are unchanged. A tracked versioned
  diagnostic JSON records the dataset identity/fingerprint, all three tokenizer-tree digests, corpus
  total/unique/cross-partition counts, per-tokenizer selected document IDs, and per-corpus selected-
  union overlap counts. Full coordinator verification and fresh diff/claim reviews pass on the exact
  integrated bytes.
- [ ] **M4d — Audited pinned-Transformers attention-marker compatibility correction.** Repair the
  real-model preflight failure from immutable batch `si-r2-20260728T124918Z`: Transformers 5.3
  exposes the resolved implementation through `_attn_implementation` but the current resolver
  unconditionally reads the absent public `attn_implementation` attribute after model load. The
  minimal correction must read each known marker defensively, accept a present string-valued private
  `eager` marker when the public marker is absent, retain support for a public-only marker, reject
  non-string values or disagreement when both markers exist, and retain the final hard failure unless
  the resolved value is exactly the frozen `eager` implementation. Do not add another attention
  backend, change dtype/device/model placement, alter the intervention or evaluation protocol, or
  reuse the failed batch root. Implement source and focused regression coverage through an audited
  multi-agent wave, then run full coordinator verification, reproducibility/secret checks, and fresh
  diff/claim reviews on identical bytes before creating a fresh immutable batch root.  
  **Acceptance:** focused tests cover private-only, public-only, matching dual-marker, conflicting,
  invalid-type, and missing-marker configurations; a read-only pinned-Transformers diagnostic shows
  all three local config classes lack the public marker and expose the private marker; full promotion
  passes; then all three real-model probes and benchmarks run on the fixed GPU0/GPU1 placement before
  lane admission and tmux production launch.
- [ ] **M4e — Audited pinned-Transformers validation-interface/mask registration correction.** The
  fresh post-M4d batch `si-r2-20260728T205146Z` reached both requested GPUs and loaded the real Llama
  and Mistral models, but their bounded validation forwards failed before receipts because the
  temporary custom attention implementation key was registered only in the attention-function
  interface. Transformers 5.3 mask construction checks its global mask-interface mapping and returns
  `None` for unknown custom keys, while the frozen validation hook correctly requires the actual
  rank-4 additive causal mask. Register the same temporary key with the pinned eager mask function
  for the validation context, retain the strict non-`None` mask requirement, and restore both
  registries/config state exactly on success or failure. Repair cleanup for pinned `GeneralInterface`
  by distinguishing class-level `register` storage from instance-local mapping deletion rather than
  swallowing `KeyError`. Do not synthesize a replacement mask in the hook, weaken mask validation,
  change eager attention math, or alter dtype/device/model/corpus/split/intervention/evaluation/schema.
  Implement source and focused offline/pinned-interface coverage through an audited multi-agent wave,
  then rerun full coordinator verification, reproducibility/secret checks, identical-byte diff/claim
  reviews, and all three real-model probes/benchmarks from a new immutable batch root.  
  **Acceptance:** tests prove the temporary key is present in both attention and mask global mappings
  during validation, the mask handler is exactly the pinned eager handler, cleanup removes an absent
  key or restores a prior value without error, the validation forward receives an actual rank-4
  causal mask, partial-install/forward failures restore all state, and a still-missing/non-tensor mask
  fails closed. Batch `si-r2-20260728T205146Z` remains terminal and is never reused.
- [ ] **M4f — Shape-faithful BF16 validation reconstruction and fit-capture cleanup.** Fresh batch
  `si-r2-20260729T002042Z` materialized on the promoted M4e bytes. Its Mistral bounded validation
  passed, but Llama validation failed the unchanged manual-reconstruction tolerance because the
  verifier reduced the real full 32-head BF16 attention matmul to a one-head matmul. A read-only
  diagnostic on GPU0 proved the full `[1,32,64,128]` query and repeated-key matmul, original BF16
  dtype, runtime scaling, full rank-4 mask, and eager softmax/cast order reproduce the returned head
  probabilities exactly, while the one-head shape changes the CUDA BF16 kernel and has maximum
  absolute error `0.015625` (above the frozen `0.005` tolerance). The one-way validation decision is
  explicitly amended in
  [`ADR-0001`](rebuttal/decisions/ADR-0001-cross-corpus-si-intervention.md#decision-amendment--2026-07-29-runtime-faithful-validation-reconstruction):
  retain the complete raw validation query/key tensors plus runtime scaling only for the bounded
  probe; reconstruct at the original
  full-head shape and select the preregistered probe head afterward. Do not loosen tolerances, cast
  the eager path to FP32, or change intervention/estimator math. The same batch exposed another
  pinned `GeneralInterface` cleanup site in `_stream_post_rope_qk`: capture-key `register` writes the
  class global mapping, while `pop` deletes the instance local mapping. Reuse the already-audited
  exact registry snapshot/install/restore semantics for this benchmark capture, including absent and
  preexisting global/local keys and cleanup on forward/consumer error. Implement source and focused
  tests through an audited multi-agent wave, then rerun full coordinator verification,
  reproducibility/secret checks, identical-byte diff/claim reviews, and all three real-model probes
  and benchmarks from another new immutable batch root.  
  **Acceptance:** a shape-sensitive regression proves reconstruction is performed once at the real
  full-head BF16 shape and only then sliced to the selected head, and fails if dtype, runtime scale,
  rank-4 mask, softmax dtype, or final cast order changes; the fixed tolerance remains unchanged;
  item 4 and source-estimator math remain float32; benchmark-capture tests faithfully model
  class-global and instance-local state and prove exact restoration of absent, preexisting global,
  and preexisting local-override state on success, forward failure, and consumer failure. Batch
  `si-r2-20260729T002042Z` is terminal and is
  never reused.
- [ ] **M4g — Self-contained tmux launch environment correction.** Batch
  `si-r2-20260729T083455Z` passed all three real-model validations, benchmarks, and lane admission,
  but both production sessions exited before model loading because the generated launch scripts
  relied on the pre-existing tmux server to propagate config path/model environment variables. The
  tmux server predates the queue script's exports, so its environment correctly lacked
  `SI_REBUTTAL_RUNS_ROOT`. Extend each generated launch script with the complete, non-secret
  allowlist needed to reload the already-resolved frozen config: `SI_REBUTTAL_RUNS_ROOT`,
  `SI_REBUTTAL_DATA_ROOT`, `SI_REBUTTAL_MODELS_ROOT`, `SI_REBUTTAL_LOGS_ROOT`,
  `SI_REBUTTAL_CACHE_ROOT`, and every configured model/tokenizer variable, in addition to the
  existing placement and receipt identity variables. Derive values from `ResolvedConfig`, preserve
  relative mutable roots and absolute model paths,
  and do not copy arbitrary ambient variables, credentials, tokens, or change scientific config.
  Add a regression that reloads the base config using only the generated allowlisted environment.
  Treat the failed batch as terminal, then bind full coordinator `/verify`, reproducibility and
  secret scans, independent diff review, and independent no-result claim review to the identical
  integrated bytes before materializing and executing a new unique batch through all real-model
  preflights.
  **Acceptance:** a generated GPU0 or GPU1 environment alone can reload the identical config
  identity with all resolved paths/model paths unchanged; launch-script tests assert the complete
  allowlist and absence of arbitrary ambient variables; full verification passes; and the new
  production tmux sessions remain alive with their processes resident only on physical GPUs 0/1
  after startup. Batch `si-r2-20260729T083455Z` is never reused.
- [ ] **M5 — Production launch.** Capture two fresh GPU snapshots, materialize one batch manifest,
  launch `si-rebuttal-gpu0` and `si-rebuttal-gpu1`, and inspect both sessions after startup. Every
  M5 substep depends on completed M4a, M4b, M4c, M4d, M4e, M4f, and M4g evidence bound to the
  exact integrated bytes; it must use only the new batch materialized after M4g promotion and must
  not reuse any failed batch or materialization root, including `si-r2-20260728T124918Z`,
  `si-r2-20260728T205146Z`, `si-r2-20260729T002042Z`, and `si-r2-20260729T083455Z`.  
  **Acceptance:** all three real-model 64-token intervention probes and fixed runtime benchmarks
  pass first; both tmux sessions are alive, each process is resident only on its assigned physical
  GPU, logs and resolved configs exist, and no result claim is made.

- [ ] **M6 — Corrected terminal-summary schema and shard-only re-finalization.** Correct the
  adversarially identified reporting defect without rerunning valid model forwards. Add the explicit
  v2-only CLI `refinalize-batch-v2 --config <path> --sweep <path> --run-root <path>
  --source-summary-v1 <path>`. The source must resolve to the run root's canonical existing
  `summaries/batch-terminal-summary.json`; the legacy `finalize-batch` path remains a schema-v1
  verify-only compatibility path when v1 exists and is never used to create or replace v2.
  `refinalize-batch-v2` alone may publish the corrected production artifact
  `summaries/batch-terminal-summary.v2.json`, using one fsynced atomic exclusive-create/no-clobber
  operation. Every public and internal legacy-v1 finalization path, including `finalize_batch`, must
  reject an absent v1 and may only reconstruct and verify existing v1 bytes; it can never create,
  replace, upgrade, or select v2. Redirect both `toy_smoke` call sites to an explicitly toy-only v2
  builder/publisher that writes `batch-terminal-summary.v2.json` and never creates v1; toy payloads
  are not correction provenance or production results. Tests exercise all absent/present/valid/
  corrupt v1×v2 collision combinations for the legacy CLI, correction CLI, and both toy-smoke call
  sites. Missing, corrupt, noncanonical, identity-invalid, or recomputation-drifted production v1
  fails before any write. A valid existing production v2 is recomputed with its frozen
  `completed_at`, verified byte-for-byte, and returned unchanged; a conflicting/corrupt/partial v2
  fails without overwrite; concurrent creators permit at most one identical winner and every loser
  verifies and returns the winner or fails closed. Temporary files are removed after failure and no
  pre-existing artifact is mutated.

  Before publishing, snapshot a canonical immutable-input manifest embedded in v2. In sorted order
  it records repository-relative path, byte SHA-256, and recomputed JSON identity where present for
  the two explicit canonical config/sweep byte inputs (which are currently Git-untracked), all 22
  files under the run's `manifests/` tree (including six fit profiles,
  twelve token manifests, materialized data, resolved config, sweep, and batch manifest), all 1,062
  shard JSON files, and v1. Record the manifest's canonical-content SHA-256 and identity in
  `correction_provenance`, together with v1 path/byte SHA/identity, batch-manifest identity,
  config/sweep identities, shard-inventory identity, and ordered fit-profile identities. Re-read and
  verify these bytes immediately before the exclusive write. Tests snapshot the complete pre-existing
  run-root file inventory and hashes, make every model/data-materialization/validation/benchmark/
  launch/run-model entry point raise if reached, and prove the command adds only v2. Missing or
  mutated inputs, invalid identities, duplicate/extra/missing shards, and a publish race all fail
  closed without a GPU/model forward or data construction.

  Extend the Spearman result object to retain the actual positive-tail exceedance numerator and total
  Monte Carlo trial count. For each of the six primary dose-response and three depth rows emit exactly
  `permutation_count`, `permutation_exceedance_count`, `permutation_seed`, and
  `permutation_seed_provenance={"seed_namespace":...,"seed_parts":...,"method":"compact_json_seed"}`;
  retain the fixed 200,000-draw PCG64 stream and remove the misleading
  `positive_null_permutation_count`. The v2 root has `schema_version=2`; v1 remains readable,
  byte-identical, explicitly superseded provenance rather than an error-free canonical result.

  Define and test a JSON-pointer-level version-invariant scientific projection. It retains, in
  original order, `/run_id`, `/config_sha256`, `/sweep_sha256`, `/batch_manifest_sha256`,
  `/shards_verified`, complete `/summary_rows`, `/contrasts`, and all 96 `/depth_rows`, plus stable
  `/summary_hashes/{shard_inventory_sha256,summary_rows_sha256,contrasts_sha256,depth_rows_sha256}`.
  It projects all six `/dose_response_statistics/*` objects to every v1 field by dropping only v2's
  four added audit fields. It projects all three `/depth_relationships/*` objects to every v1 field
  except v1's known-bad `/positive_null_permutation_count`: from v2 it drops only the two newly added
  count fields while retaining and exactly comparing the already-existing depth seed/provenance,
  layer inputs, shard linkages, aggregates, rho, and p-values. Canonical projection bytes and their
  SHA-256 must match exactly across v1 and v2.

  The complete allowed JSON-pointer diff is: `/schema_version`; `/completed_at`;
  `/correction_provenance`; `/immutable_input_manifest` and its digest/identity pointers; the four
  added fields under `/dose_response_statistics/*`; removal of
  `/depth_relationships/*/positive_null_permutation_count` plus addition of only
  `/depth_relationships/*/{permutation_count,permutation_exceedance_count}`; dependent
  `/summary_hashes/{dose_response_statistics_sha256,depth_relationships_sha256}` plus newly added
  projection/manifest hash pointers; and root `/identity_sha256`. No other pointer, value, array
  order, source hash, scientific input, rho, p-value, contrast, or linkage may change. Negative
  mutation tests alter one value/order/linkage in each of `/summary_rows`,
  `/dose_response_statistics`, `/contrasts`, `/depth_rows`, `/depth_relationships`, and the stable
  summary-hash pointers and must be rejected; substitution of either explicit config/sweep file,
  even at the same path, must also be rejected.

  This artifact-format one-way door is governed by the 2026-07-29 amendment to
  [`rebuttal/decisions/ADR-0001-cross-corpus-si-intervention.md`](rebuttal/decisions/ADR-0001-cross-corpus-si-intervention.md).
  The owner explicitly selected re-finalization from the existing verified shards rather than GPU
  reruns; the design-bank query returned no matching canon. Implement source and regression tests
  through audited `/multi-agent`, then run coordinator `/verify rebuttal`, reproducibility and secret
  scans, and a fresh adversarial diff review on the identical bytes before executing the v2-only
  finalizer. No model-forward process or tmux/GPU session is launched.

  **Acceptance:** source-v1 byte SHA-256 remains
  `6e9227ff0b4fd5d54fbf7350ea4cab24f2f3c099c6a24ac49ca6f966ca82ed82` and its `identity_sha256` field
  remains `56b0439e4c20487ce78c1778cdb144607cb49599a9ae53418bbfe84b9c090c30`;
  v2 verifies and binds all immutable inputs and exactly 1,062 shards; primary exceedances are
  `0,0,0,16,0,0` and depth exceedances are `555,52,0`; all nine rows record
  `permutation_count=200000` and satisfy
  `p_one_sided=(1+permutation_exceedance_count)/(permutation_count+1)` exactly. The independent
  post-write audit recomputes all nine streams and binds its evidence to the exact v2 byte SHA, v1
  byte SHA, embedded immutable-input-manifest SHA, and scientific-projection SHA. A fresh research
  claim review is recorded against those same identities. Repeating the v2 command is byte-idempotent;
  corrupt-v1, corrupt-v2, input mutation, write failure, and race tests establish no-clobber behavior.

## Definition of done

- [ ] Every new experiment source/config/test/script/result path is below `rebuttal/`; mutable runs
  are ignored and tracked configs remain immutable.
- [ ] Both cross-corpus directions use frozen kernels, frozen source-based head selection, disjoint
  fit/eval sequences, identical target examples across conditions, three primary models, 20 bins,
  and preregistered controls/uncertainty.
- [ ] Every checkpoint passes the production-path zero-kernel, constant-offset, exact
  selected-head-mask/logit injection, manual-attention reconstruction, hook-count, and GQA mapping
  tests at the fixed ADR tolerances before any production shard is written.
- [ ] Depth output states the intervention unit (all query heads in one layer) and reports the
  per-layer `R²`/disruption relationship without presenting it as individual-head causality.
- [ ] Run metadata records config/sweep hashes, seed, dataset identity/fingerprint and token-tree
  digest, model weight-tree digest, Git SHA plus dirty diff digest, Python/package/CUDA/GPU details,
  exact command, physical/logical device mapping, and timestamps.
- [ ] CPU smoke, lint, type checks, unit tests, config validation, and production preflight pass on
  the coordinator; no full training is run by verification.
- [ ] Generated tmux scripts carry the complete non-secret config environment needed to reload
  the identical resolved config without relying on ambient tmux-server state.
- [ ] `si-rebuttal-gpu0` runs Llama then OLMo on physical GPU0 and `si-rebuttal-gpu1` runs Mistral on
  physical GPU1, with startup evidence captured under the batch run root.
- [ ] No improvement, robustness, or causal conclusion is stated until terminal artifacts are
  compared with the in-domain/control baselines and pass `research-claim-review`.

## Verification plan

- Plan: `AGENT_WORKSPACE_HOST_ROOT="$PWD" /jumbo/lisp/f004ndc/.agent-workspace/bin/check-plan
  --path PLAN.md`, followed by independent `/adversarial PLAN.md`.
- Types/lint: attached-repo research `/verify rebuttal`, using basedpyright and ruff on
  `rebuttal/src` and `rebuttal/tests`.
- Tests: pytest covers config rejection, deterministic splits, no leakage, kernel/control
  construction, paired bootstrap/permutation determinism, atomic resume, shard finalization, and
  tmux dry-run/device mapping.
- Smoke: CPU toy run of both directions and depth finalization; production-weight preflight loads
  each model one at a time only after audited integration and validates the actual production eager
  hook against captured Q/K, attention masks, and returned attention probabilities.
- Reproducibility: run `repro-guard` checks for seeds, deterministic settings, paths, secrets,
  environment pins, and data versions.
- Launch: `nvidia-smi`, tmux session inspection, process/GPU mapping, log tail, and artifact
  existence checks. Full experimental claims are outside `/verify`.

## Risks & one-way doors ⚠️

- **Evaluation protocol:** corpus identity, split, head selection, control construction, sequence
  counts, and statistics affect the rebuttal conclusion. ADR-0001 freezes them before coding; any
  material change requires a new adversarial review and a deviation entry.
- **Artifact format:** downstream aggregation depends on the resolved-config/manifest/shard schema.
  Version it as `schema_version = 1` and fail closed on mismatches.
- **Corpus availability:** canonical Experiment 1 JSONL files are currently absent. The only
  production inputs are the exact immutable Hugging Face revisions/fields in ADR-0001. Download
  failures, insufficient document-disjoint chunks, or overlap checks block launch; no fallback is
  permitted. Terminal `CR`/`LF` removal is allowed only for matching the already-frozen
  top-level-title regex; the original row text remains the experiment input and provenance payload.
- **Dependency drift:** the existing venv lacks pyarrow, ruff, and basedpyright and has no lockfile.
  Add a rebuttal-scoped pinned environment file and record actual versions before launch.
- **Compute/runtime:** eager 7–8B interventions may run for many hours. The fixed two-fit/four-eval
  per-model benchmark must project at most 120 hours per physical-GPU lane and satisfy the fixed
  disk margin. Otherwise stop rather than reduce conditions. Atomic per-condition shards,
  deterministic resume, tmux logs, and completion markers prevent partial output from masquerading
  as a terminal result.
- **GPU collision/OOM:** fail closed if either GPU is occupied; start with a measured batch-size
  preflight and never silently change scientific sample counts or condition definitions. Query
  GPU inventory and compute applications separately because `compute_apps.pid` is not a valid
  `--query-gpu` field on the installed NVIDIA driver.

## Open questions

- Terminal runtime is unknown until the audited production-weight preflight. Batch size starts at
  one; it may not change after the benchmark batch manifest is frozen. An OOM blocks launch and
  requires a revised, independently reviewed plan rather than an automatic fallback.

## Deviations log (fill during implementation)

- **2026-07-28 — owner-approved WikiText title-detection correction.** The first guarded
  `materialize-data` attempt loaded the frozen WikiText snapshot (fingerprint
  `7dabb830ac9ebb0d`) but selected zero documents because every observed title row retained a
  terminal newline while the full-match regex was defined over the visible title. The owner
  approved the minimal correction: remove only terminal `CR`/`LF` characters for title
  classification, preserve raw row text everywhere else, add focused regression coverage, and
  rerun the complete promotion gates. The failed attempt wrote no batch manifest or materialized
  result. This clarification does not change the repository, revision, split, row order, title
  regex, document-ID rule, partition function, tokenization rule, counts, or statistics. Because
  the failed attempt produced no manifest, the corrected run must create and record fresh
  document, partition, and token digests rather than claim identity with nonexistent prior
  artifacts.
- **2026-07-28 — real-driver GPU snapshot command correction.** A coordinator read-only probe
  found that the installed `nvidia-smi` exits 2 for
  `--query-gpu=index,uuid,name,memory.used,utilization.gpu,compute_apps.pid`, so the production idle
  gate cannot reach either snapshot. The same driver accepts
  `--query-gpu=index,uuid,name,memory.used,utilization.gpu` and
  `--query-compute-apps=gpu_uuid,pid`, including `--id=<physical-index>`. The minimal correction
  separates those queries and binds application PIDs to the inventory UUID; it does not weaken the
  idle thresholds, remove either snapshot, change placement, or alter the experimental protocol.

- **2026-07-28 — selected-artifact overlap-guard correction.** The first post-M4b replacement
  materialization reached the immutable WikiText snapshot but found exact joined-document duplicates
  assigned to both corpus partitions. A read-only diagnostic found 29,443 constructed documents,
  29,001 unique joined-content hashes, and 146 hashes crossing the frozen document-ID partitions.
  The actual first 50 fit and 100 evaluation chunks for each of the three frozen tokenizers remained
  disjoint, and their per-corpus union remained disjoint: zero selected document-ID, selected source-
  row, selected joined-content-hash, or per-tokenizer selected token-hash overlap. The minimal
  correction evaluates document, row, and joined-content overlap over both each tokenizer's selected
  documents and the cross-tokenizer selected-document union, while retaining the per-tokenizer token-
  hash guard. Existing full partition document inventories remain serialized unchanged as provenance.
  The correction does not deduplicate or repartition the corpus and changes no selected example or
  scientific computation. Both failed attempts wrote no batch manifest and their roots must not be
  reused.

- **2026-07-28 — pinned Transformers attention-marker compatibility correction.** After both
  requested GPUs became idle, the guarded queue began real-model validation for Llama and Mistral,
  but both lanes failed before writing validation or benchmark receipts because Transformers 5.3
  `LlamaConfig` and `MistralConfig` do not define public `attn_implementation`; the installed pinned
  implementation stores the loader-selected value behind `_attn_implementation`. A read-only local
  diagnostic confirms the same public-marker absence for `Olmo2Config`. The minimal correction is a
  compatibility read of the two already-recognized markers with strict type/conflict checks and the
  unchanged final requirement that the resolved implementation equal `eager`. It changes no model
  request, attention backend, dtype, device, corpus, split, selected sequence, intervention,
  statistic, artifact schema, or scientific claim. Batch `si-r2-20260728T124918Z` is terminally
  failed and must not be reused; because source provenance changes, promotion must finish before a
  new immutable batch is materialized.

- **2026-07-28 — pinned Transformers validation mask-interface and registry cleanup correction.**
  Fresh batch `si-r2-20260728T205146Z` materialized successfully and both requested GPUs were idle;
  its tmux queue loaded Llama on GPU0 and Mistral on GPU1, then both validation forwards failed before
  validation/benchmark/admission/production receipts. The temporary validation attention key was
  absent from Transformers 5.3 `ALL_MASK_ATTENTION_FUNCTIONS._global_mapping`, so
  `create_causal_mask` classified it as a custom backend without mask construction and forwarded
  `None`; the unchanged intervention guard rejected that missing actual mask. Cleanup then exposed a
  second pinned-interface mismatch: `GeneralInterface.register` writes the class `_global_mapping`,
  while inherited `MutableMapping.pop` delegates deletion to the instance `_local_mapping`, producing
  `KeyError`. The minimal correction registers the same temporary key with the exact pinned eager
  mask handler and restores class-level registered state explicitly; it retains the hook's strict
  rank-4 additive-mask requirement and does not synthesize a substitute mask or change attention
  mathematics. This is a compatibility repair that realizes the already-frozen eager causal
  intervention rather than an evaluation-protocol change. The failed batch is terminal and must not
  be reused.

- **2026-07-29 — full-head BF16 reconstruction and benchmark registry cleanup correction.** Fresh
  promoted batch `si-r2-20260729T002042Z` passed materialization and Mistral validation, then exposed
  two further pinned-runtime mismatches before admission or production. Llama's returned eager
  probabilities come from a full batched 32-head BF16 matmul; reconstructing only the selected head
  selects a different CUDA BF16 kernel and exceeds the frozen tolerance. Diagnostic hash
  `502c247708eec3e55bd3d577bc0369144d53ba4b4f4f0d6cebbc500a90e7f942` proves full-shape reconstruction
  is bit-exact while selected-shape reconstruction differs by `0.015625`. The minimal correction
  retains bounded full-shape raw query/key and runtime scaling for validation only, performs the
  same eager dtype/shape/operation order, then selects the frozen head. Separately, benchmark fit
  capture repeats the already-diagnosed class-global `register`/instance-local `pop` mismatch; it
  must use the same exact registry snapshot/install/restore abstraction already audited for
  validation. Neither correction changes tolerances, estimator inputs, attention/intervention math,
  datasets, splits, placement, statistics, or artifact schema. This batch is terminal and must not
  be reused.

- **2026-07-29 — self-contained tmux launch environment correction.** Fresh batch
  `si-r2-20260729T083455Z` passed all three production-weight validation probes, fixed runtime
  benchmarks, and lane admission, then both generated production scripts exited before model load
  with `Environment variable SI_REBUTTAL_RUNS_ROOT is required`. The queue exported the required
  values inside a tmux session, but the tmux server existed before those exports and did not
  propagate them to newly created sessions. The minimal correction makes generated launch scripts
  carry only the complete non-secret resolved config environment (five path roots plus all model and
  tokenizer paths), while retaining all frozen config/protocol/placement/receipt checks. It does not
  copy arbitrary ambient variables or credentials and changes no dataset, split, model, intervention,
  statistic, artifact schema, or claim. The failed batch wrote zero shards/summaries, is terminal,
  and must not be reused; promotion and a new immutable batch precede another production launch.
