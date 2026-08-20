# ADR-0001 — Cross-corpus SI intervention protocol

- **Status:** proposed
- **Date:** 2026-07-27
- **Decision owners:** project owner and rebuttal experiment coordinator
- **Review gate:** must receive independent `SHIP` with `PLAN.md` before implementation

## Context

Reviewer C3Mi asks whether a kernel fitted on one corpus produces the dose-response when disruption
is evaluated on a different corpus, with kernel and head selection frozen. The supplied
`SIRebuttal.md` also lists the reverse direction if compute permits and a lower-priority
depth-stratified analysis. Existing E19 transfers an SI score but does not perform the requested
cross-corpus intervention.

The rebuttal removes the broad “learned, not architectural” headline. Matched 1.5B–3B
RoPE-versus-NoPE training is therefore conditional rather than part of this rebuttal run.

## Frozen inputs

### Models

Use the local, read-only weight trees for Llama-3.1-8B, Mistral-7B-v0.1, and OLMo-2-7B. Record the
model config and a streamed SHA-256 tree digest before running. Load every model in `bfloat16`,
evaluation mode, with `attn_implementation="eager"`; hard-fail if the resolved attention
implementation or device differs. Model/tokenizer files may not be downloaded or modified after
the batch manifest is materialized.

### Corpora and documents

Use only these Hugging Face revisions and fields:

| Domain | Repository and immutable revision | Config / split / field |
| --- | --- | --- |
| Wikipedia | `Salesforce/wikitext@b08601e04326c79dfdd32d625aee71d232d685c3` | `wikitext-103-raw-v1` / `train` / `text` |
| Python code | `code-search-net/code_search_net@bd0cf261e357a3eb5c8fba490d23ec1a1cd59555` | `python` / `train` / `whole_func_string` |

Dataset row order is the immutable Arrow row order. Wikipedia documents are the non-empty rows from
one top-level title through the row before the next title; a top-level title matches the full regex
`^ = [^=]+ = $` after removing only terminal `CR` and `LF` characters from a detection-only view of
that row. Preserve the original row text, including any terminal line ending, when joining and
hashing document content. Do not trim spaces, normalize Unicode, or otherwise rewrite a row. Ignore
rows before the first title. Its document ID is
`wiki:<start-row>:<end-row>`. Code documents are repositories: require non-empty
`repository_name`, group all non-empty `whole_func_string` rows by that exact value, preserve row
order, and use `code:<repository_name>` as the document ID. Join rows inside a document with the
literal UTF-8 string `"\n\n"`. Do not cross document boundaries.

For each domain, assign a document to fit when the unsigned big-endian value of
`SHA256("29039|" + dataset-revision + "|" + document-id)` modulo 3 equals zero; assign it to
evaluation otherwise. This document assignment is shared by all models and frozen before
tokenization. Within each partition, order documents by that SHA-256 and document ID. Tokenize each
document independently with its model tokenizer, `add_special_tokens=False`; form non-overlapping
512-token chunks, discard each document's remainder, and select the first 50 fit chunks and first
100 evaluation chunks in `(document order, chunk index)` order. Hard-fail if either count is short.

The materializer saves the dataset `_fingerprint`, repository/revision/config/split/field, every
selected source row index, document ID, SHA-256 of the exact joined UTF-8 document, tokenizer tree
digest, and SHA-256 of each selected `int64` token array. The existing `schema_version = 1` domain
payload also retains the full fit/evaluation partition document inventories as provenance. Among
documents referenced by selected chunks, it hard-fails if a document ID, source row, or joined-
document content hash crosses fit/evaluation, both within each tokenizer and in the per-corpus union
across all frozen tokenizers; it also hard-fails if a selected token hash is duplicated across the
partitions for one tokenizer. Exact corpus duplicates that are not referenced by any selected chunk
remain in the full-inventory provenance payload but neither enter the experiment nor invalidate
materialization. Production has no local-cache, alternate-field, or synthetic fallback.

## Frozen estimator and intervention

### Logit tensor and source score

For each fit sequence, capture each layer's scaled, post-RoPE `QKᵀ/sqrt(d)` immediately before mask
addition and softmax, shaped by **query heads after GQA KV repetition**. Compute in float32 after
capture.

For RMSNorm models (Llama and Mistral), double-center each complete 512×512 head matrix as
`L - row_mean(L) - column_mean(L) + global_mean(L)`; OLMo's LayerNorm matrices are unchanged. For
each head and each sequence, use only causal offsets `Δ = i-j = 1..511` and:

1. compute the arithmetic mean of every lower diagonal;
2. take an `rfft` of the 511 means;
3. retain DC and the eight largest-magnitude non-DC components (`torch.topk` index behavior is the
   frozen tie rule), zero all others, and apply `irfft(n=511)`;
4. compute pair-weighted `SSE` over every retained causal logit and
   `SST` around their single global mean; and
5. set `R²=max(0, 1-SSE/SST)`, or `1` when `SST==0`.

The source score for a head is the arithmetic mean of its 50 per-sequence `R²` values. There is no
other centering, offset weighting, sequence weighting, smoothing, or missing-sequence exclusion.

### Subtracted kernel

Intervention kernels are estimated separately from the same 50 fit captures using the **raw**
float32 scaled post-RoPE logits: no RMS double-centering and no FFT smoothing. For each query head,
take each lower-diagonal mean for `Δ=0..511` in each sequence, then the arithmetic mean across all
50 sequences. A missing head/sequence or non-finite value is a hard failure.

For a selected query head, add the exact causal Toeplitz correction
`C[i,j] = -g(i-j)` for `j<=i` to its additive attention mask before softmax. Non-selected query
heads receive exactly zero correction. The 20-bin intervention unit is all query heads in that bin
simultaneously; the depth unit is all 32 query heads in one layer simultaneously. Loss is the
per-sequence arithmetic mean next-token NLL for targets at positions `1..511`; report the arithmetic
mean over the same 100 target evaluation sequences. The descriptive per-head group proxy is the
group loss delta divided by the exact number of simultaneously intervened heads; it is explicitly
interaction-sensitive and not an individual-head causal effect.

### Bins, conditions, and controls

Sort all query heads ascending by `(mean_source_R², layer_index, head_index)` and partition that list
with `numpy.array_split(..., 20)` into exactly 20 non-empty, near-equal-count bins. This is the only
bin/tie rule; do not use quantile boundaries or drop heads.

For both Wikipedia→code and code→Wikipedia, evaluate baseline once and these conditions on identical
target sequences for every source bin:

1. subtract the frozen source kernel;
2. subtract the target in-domain kernel on the same source-selected heads;
3. three offset-permutation controls, independently permuting all 512 `Δ=0..511` source-kernel
   entries for each selected head; and
4. three norm controls, drawing 512 iid standard normals per head, subtracting their vector mean,
   and rescaling their L2 norm to that head's source-kernel L2 norm.

Use NumPy `Generator(PCG64(seed))`; permutation means its `permutation(512)` result. A zero kernel
norm is a hard failure. Every stochastic seed is the unsigned big-endian first eight bytes of the
SHA-256 of compact ASCII JSON for the list
`[29039,model,direction,unit,bin_or_layer,trial,control_kind]`. Trial indices are `0,1,2`.
The all-zero and constant-offset kernels are validation probes only, not scientific conditions.

The depth follow-up is Wikipedia→code only. It subtracts the Wikipedia kernel from all query heads
in one layer at a time and relates layer-mean Wikipedia `R²` to grouped loss delta divided by 32.

## Frozen statistics

For each model/direction, the primary dose-response statistic is Spearman's rho across exactly 20
pairs `(bin mean source R², source-kernel group loss delta / bin size)`. Its one-tailed positive
Monte Carlo null keeps the 20 R² values fixed and independently permutes the 20 response values
200,000 times with a seed from the rule above using unit `statistic`, bin/layer `all`, trial `0`,
and control kind `spearman_response_permutation`. Report
`p=(1 + count(rho_permuted >= rho_observed))/200001` and SciPy's two-sided approximation. A
non-finite statistic is a hard failure.

For each aggregate delta and transferred-minus-reference/control contrast, draw exactly 10,000
paired resamples of the 100 target sequence indices with replacement using the same seed rule and
report the 2.5/97.5 percentiles with NumPy's `method="linear"`. Average the three control trials at
the per-sequence level before forming a control contrast. Preserve all per-sequence values. No test,
threshold, exclusion, direction, or sample count may change after observing results.

## Intervention validation gate

Before production, every real checkpoint must pass one fixed 64-token eager-attention probe:

1. Assert expected layers/query heads/KV heads/repetition factor (`32/32/8/4` for Llama and Mistral,
   `32/32/32/1` for OLMo), hook call count, logical layer, and device.
2. A zero correction must match baseline token logits at `atol=2e-2, rtol=2e-2` and mean NLL within
   absolute `1e-3`.
3. A constant correction on one query head must produce attention probabilities matching baseline
   at `atol=5e-3, rtol=5e-3` and mean NLL within absolute `1e-3`, demonstrating softmax invariance.
4. For a fixed nonconstant kernel, capture the mask before and after the production hook. In
   float32, `after-before` must equal the expected selected-head Toeplitz correction within maximum
   absolute `1e-6`, while every non-selected logical query head is exactly zero.
5. Capture the complete raw post-RoPE query tensor `[1,32,64,d]`, raw pre-repeat key tensor
   `[1,kv_heads,64,d]`, runtime `scaling` scalar, and exact forwarded corrected additive mask
   `[1,32,64,64]` from the same eager call. Q/K must retain the checkpoint's frozen `bfloat16`
   inference dtype and original full-head shapes. Repeat K with the production GQA mapping, execute
   exactly one full-head batched `matmul` in `bfloat16`, multiply by the captured runtime scaling
   without an intervening cast, add the captured float32 mask, apply softmax with `dtype=float32`,
   cast the probabilities back to the query dtype as the pinned eager implementation does, and only
   then select the preregistered probe head. Compare that selected output to the returned attention
   probabilities at the unchanged `atol=5e-3, rtol=5e-3`; non-selected heads must match their
   baseline. Slicing to one head before matmul or reconstructing the eager BF16 kernel with an FP32
   matmul is forbidden because CUDA matmul rounding depends on the full batched shape. This probe
   explicitly verifies the query-head-to-repeated-KV mapping. Item 4 mask-delta validation and the
   source-kernel/R² estimator remain float32 and are unchanged.

Any failure blocks production; tolerances may not be tuned to observed failures.

## Operations and artifacts

Physical placement is fixed: GPU0 runs Llama then OLMo, and GPU1 runs Mistral; there is no fallback.
Result schemas are versioned `1`. Resolved configs, dataset/token manifests, per-condition atomic
shards, summaries, provenance, logs, and completion markers are immutable beneath a unique ignored
`rebuttal/runs/` root.

Before launch, benchmark each model with exactly two fit and four evaluation sequences at length
512, including capture/kernel estimation, baseline, one source-bin condition, all six controls, and
one depth condition. Extrapolate by the fixed count ratio in the runner and save the formula and
timings. Launch only if every model benchmark completes, projected total wall time for the GPU0
lane (Llama+OLMo) and GPU1 lane (Mistral) is at most 120 hours each, and free space is at least
`2 × projected artifact bytes + 20 GiB`. Otherwise stop and report the block; do not change the
protocol. Production runs until all frozen conditions complete, with no deadline truncation.

The rebuttal environment installs exact direct pins before smoke/preflight:
`torch==2.7.0`, `transformers==5.3.0`, `datasets==4.8.2`, `numpy==1.26.4`,
`scipy==1.11.4`, `pandas==2.1.4`, `pyarrow==23.0.1`, `pytest==7.4.4`,
`ruff==0.12.5`, and `basedpyright==1.31.1`; it records `pip freeze --all`, Python, CUDA, and driver
versions in the batch manifest.

## Consequences

- The design directly answers the major cross-corpus reviewer question in both directions.
- In-domain and randomized references distinguish weakened transfer from generally weak target
  effects while preserving identical examples.
- Whole-bin and whole-layer interventions remain group-level, interaction-sensitive diagnostics.
- Material changes to corpora, splits, estimator, intervention, controls, statistics, validation,
  or artifact schema require a revised ADR and fresh adversarial review before launch.
- No architectural-learning claim follows from this experiment.

## 2026-07-28 protocol clarification

The first production materialization attempt loaded the immutable WikiText snapshot with dataset
fingerprint `7dabb830ac9ebb0d`, but the stored title rows include terminal newlines (for example,
`" = Valkyria Chronicles III = \n"`). Full-matching the frozen visible-title regex against the raw
row therefore produced no documents and failed before writing a batch manifest.

The project owner approved the minimal correction recorded above: remove only terminal `CR`/`LF`
characters for title detection and preserve the raw row for all scientific content and provenance.
This correction changes no dataset identity, source-row ordering, visible-title regex, document-ID
rule, partition function, tokenizer-input construction beyond enabling the intended documents,
sample count, estimator, intervention, statistic, or artifact schema. The failed attempt produced
no manifest, so the corrected run creates and records fresh document, partition, and token digests;
it does not claim identity with a prior materialization. It requires a fresh plan review, audited
implementation, coordinator verification, diff review, and claim review before production
materialization resumes.

The next guarded replacement attempt exposed a separate implementation mismatch: the immutable
WikiText train split contains exact duplicate joined documents, and 146 content hashes cross the
frozen document-ID partitions even though none of those crossings appears among the selected fit and
evaluation artifacts for any frozen tokenizer. The overlap invariant above therefore applies to
documents referenced by selected chunks, which are the only documents used by the experiment. Full
partition document inventories remain serialized unchanged as provenance in every model-specific
domain payload. The correction must not deduplicate, drop, or repartition corpus documents and must
not change document construction, partition order, tokenizer input, selected chunks, selected
digests, or any `schema_version = 1` payload field. The materializer must also reject a document-ID,
row, or content crossing visible only in the per-corpus union across model tokenizers; token-hash
checks remain tokenizer-local. The failed attempt wrote no batch manifest. Fresh plan review, audited
implementation, coordinator verification, diff review, and claim review are required before another
materialization.

## Rejected alternatives

- **E19-only reanalysis:** no causal-sensitivity intervention on a held-out corpus.
- **One direction only:** leaves a preregistered, feasible asymmetry check unanswered.
- **Matched pretraining now:** unnecessary after narrowing the headline and infeasible as a quick
  rebuttal experiment without a separate training-data/schedule decision.
- **Reuse existing Wikipedia head groups:** violates the request that source-specific selection be
  explicit and frozen for each direction.

## Decision amendment — 2026-07-29 runtime-faithful validation reconstruction

This amendment supersedes only the numerical reconstruction rule in intervention-gate item 5.
Fresh promoted batch `si-r2-20260729T002042Z` revealed that the earlier prose's selected-head FP32
reconstruction was not a faithful recomputation of the pinned eager BF16 implementation. A bounded
read-only Llama diagnostic captured the real full tensors and showed:

- original full-head BF16 shape/order reproduces returned probabilities bit-for-bit;
- slicing to one head before BF16 matmul changes CUDA kernel rounding and has maximum absolute error
  `0.015625`; and
- selected-head FP32 reconstruction also exceeds the existing `0.005` tolerance.

The diagnostic transcript SHA-256 is
`502c247708eec3e55bd3d577bc0369144d53ba4b4f4f0d6cebbc500a90e7f942`. The acceptance threshold is
not tuned: `atol=5e-3, rtol=5e-3` remains frozen. Instead, the validation recomputation is bound to
the already-frozen eager implementation's actual dtype, full shape, runtime scale, mask, softmax,
and cast order. The full raw tensors are bounded to the 64-token preflight probe and are not result
artifacts. No source estimator, intervention, dataset, split, model, result schema, statistic, or
production condition changes. This post-observation validation-protocol amendment requires a fresh
independent `SHIP` on the amended PLAN and ADR before implementation; failed batch
`si-r2-20260729T002042Z` remains terminal and must never be reused.


## 2026-07-29 terminal-summary schema correction

A final independent adversarial audit recomputed all six dose-response and three depth Spearman
statistics from the immutable production inputs. Every rho and one-sided p-value matches exactly,
but schema-v1 depth rows mislabeled the fixed 200,000 permutation budget as
`positive_null_permutation_count`. Actual positive-tail exceedance numerators are 555, 52, and 0 for
Llama, Mistral, and OLMo; primary numerators are 0, 0, 0, 16, 0, and 0 in frozen model/direction
order. Primary rows also omitted seed/count audit provenance. This is a terminal-summary reporting
and artifact-schema defect, not a model-forward, shard, estimator, tail-direction, seed-stream, or
p-value-computation defect.

The owner selected the minimal correction: retain all 1,062 schema-v1 shards and six fit profiles,
do not rerun GPU forwards, and rerun only finalization. The existing
`summaries/batch-terminal-summary.json` is immutable and remains byte-for-byte untouched and readable
as superseded provenance. Corrected finalization is a distinct v2-only interface,
`refinalize-batch-v2`, requiring explicit config, sweep, run root, and the canonical v1 source path.
Every public or internal legacy-v1 finalization path, including `finalize_batch`, remains
verify-only for an existing v1 and rejects absent v1. It cannot create, replace, upgrade, or select
v2. Both `toy_smoke` call sites are redirected to a toy-only v2 builder/publisher that writes no v1
and is never production correction provenance. The production correction interface writes only
`summaries/batch-terminal-summary.v2.json` with root `schema_version=2`, using an fsynced
exclusive-create/no-clobber publish. It verifies v1 identity and exact reconstruction before
writing. Valid existing v2 bytes are verified and returned unchanged; missing/corrupt/drifted
inputs, conflicting/corrupt v2, write failure, or publication races fail closed and never overwrite
an artifact. A race loser may return only after verifying the identical winner. Tests cover all
v1/v2 presence, validity, conflict, idempotency, and race combinations across public and internal
callers.

V2 embeds a canonical sorted immutable-input manifest over the explicit canonical config/sweep
byte inputs (currently Git-untracked, and therefore bound by repository-relative path, byte digest,
and recomputed identity rather than a tracked-file claim), every file under the run's manifests tree
(including resolved config, sweep, batch manifest, six fit
profiles, twelve token manifests, and materialized data), all 1,062 shards, and v1. Each entry binds
relative path, byte SHA-256, and JSON artifact identity when present. Correction provenance binds the
manifest digest/identity, v1 path/byte digest/identity, batch-manifest identity, config/sweep
identities, shard-inventory identity, and ordered fit-profile identities. The finalizer verifies
those bytes again immediately before the only publish. Tests prohibit reachability of model,
dataset materialization, validation, benchmark, launch, and run-model paths and prove that only the
new v2 file is added to the existing run root.

Every primary dose-response and depth statistic records `permutation_count`,
`permutation_exceedance_count`, `permutation_seed`, and complete compact-JSON seed provenance with
`seed_namespace`, ordered `seed_parts`, and method `compact_json_seed`; the misleading field is
absent. The invariant
`p_one_sided = (1 + permutation_exceedance_count) / (permutation_count + 1)` is checked for all nine
statistics. All fixed 200,000-draw PCG64 streams are unchanged.

A canonical JSON-pointer-level version-invariant scientific projection is the compatibility
contract. It preserves `/run_id`, `/config_sha256`, `/sweep_sha256`,
`/batch_manifest_sha256`, `/shards_verified`, complete ordered `/summary_rows`, `/contrasts`, all 96
`/depth_rows`, and stable shard/summary/contrast/depth-row summary hashes. All six
`/dose_response_statistics/*` objects project to every v1 field by dropping their four v2 audit
fields. All three `/depth_relationships/*` objects project to every v1 field except the known-bad
v1 `positive_null_permutation_count`; only the two added v2 count fields are dropped, while existing
depth seed/provenance, input arrays, shard linkages, aggregates, rho, and p-values compare exactly.
Canonical projection bytes and digest must match.

The only permitted JSON-pointer differences are root schema/completion/correction/input-manifest
metadata; four audit fields under each primary statistic; removal of the bad depth field and addition
of the two depth count fields; the two dependent dose/depth-relationship summary hashes and new
projection/manifest hashes; and root identity. Negative mutation tests cover each scientific
container, stable summary hashes, array ordering/linkage, and config/sweep substitutions. No other
scientific input, source hash, linkage, ordering, rho, p-value, contrast, or depth value may differ.

Schema-v2 is the canonical terminal summary for this production batch; any future non-toy production finalization interface must also emit v2 and requires its own reviewed caller contract. Any further
terminal-summary field or filename change is a new artifact-format one-way door requiring plan,
adversarial, verification, and claim-review gates. The post-write independent audit and claim review
must bind to exact v2 and v1 byte digests, the immutable-input-manifest digest, the scientific
projection digest, and independent recomputation of all nine permutation streams.
