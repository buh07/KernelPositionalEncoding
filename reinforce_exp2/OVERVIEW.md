# reinforce_exp2 Overview

Last updated: 2026-04-20 (America/New_York)

## Purpose

`reinforce_exp2` is the second consolidation layer of the Kernel PE program. It is not a new standalone hypothesis family. It is an execution and evidence-hardening framework that turns a long sequence of experiments (1-8 plus `reinforce_exp`) into a submission-grade, preregistered, dependency-aware A/B/C narrative with strict artifact contracts.

In short:

1. Experiments 1-8 discovered and stress-tested the science.
2. `reinforce_exp` hardened key weak points and clarified claim boundaries.
3. `reinforce_exp2` operationalizes final claim governance: what is confirmatory, what is exploratory, what is promoted, what is demoted, and why.

---

## Program Continuity: How We Got Here

## Experiments 1-3 established the mechanistic core

1. **Experiment 1** measured shift-invariant (SI) positional structure from Q/K geometry and logits.
   - Outcome: SI structure exists, but strength and stability are heterogeneous.
   - Gap left open: descriptive fit alone cannot establish causal role.

2. **Experiment 2** introduced frequency-targeted causal interventions.
   - Outcome: H1-style effects broadly supported; H2 repeatedly weak/reversed in coarse bands.
   - Gap left open: frequency-band stories were too coarse and model-conditional.

3. **Experiment 3 + Phase 2** executed mechanism adjudication (T1-T10 and 3P2 family).
   - Outcome: strongest invariants converged on kernel load-bearing behavior and threshold-like organization, with model-conditional boundary/semantic nuances.
   - Gap left open: reviewer-facing concerns remained around strict boundary interpretation, dependence-aware inference, construct stability, and over-interpretation risk.

## Experiments 4-8 broadened adaptation, tokenizer, and theory lenses

4. **Experiment 4** tested SI-guided fine-tuning ideas (trajectory, SI-aware LoRA, optional SI-loss extension).
   - Value: checks whether SI structure is actionable during adaptation.
   - Typical finding direction: SI structure tends to be stable; adaptation gains are not automatic.

5. **Experiment 5** audited tokenizer effects.
   - Value: tests whether SI boundary behavior is tokenizer/model-family entangled.
   - Key direction: SI head identity is tokenizer-sensitive; perturbations causally shift SI-linked behavior.

6. **Experiment 6** tested SI-guided knowledge localization/training variants.
   - Value: aggressive training-time intervention space (routing, anti-localization, distillation, contrastive, tokenizer formatting).
   - Typical finding direction: mixed; many naive SI-guided recipes are not robust wins.

7. **Experiment 7** connected SI findings to compressed-sensing style theory (Welch/coherence and phase-transition scaling).
   - Value: attempts quantitative theory bridge.
   - Typical finding direction: promising but not yet cleanly confirmatory as headline evidence.

8. **Experiment 8** reframed tokenizer quality through SI structure and pruning.
   - Value: practical utility tests for SI-derived tokenizer metrics and masks.
   - Typical finding direction: model-conditional utility; some strong effects, but not universal monotonicity.

## `reinforce_exp` tightened claims for review robustness

`reinforce_exp` ran targeted reinforcements (R1/R2/R3/R4/R5 + R2B/R5B + NEW-R12/R14/R15) to address concrete reviewer attack surfaces:

1. strict boundary ambiguity,
2. dependence-aware statistics,
3. third-model replication breadth,
4. PE confound skepticism,
5. proxy vs task semantic transfer,
6. ordering/specificity artifacts,
7. head-set stability.

That wave improved defensibility materially, but still left one final need: convert mixed historical assets into a strict, reproducible, submission-ready execution contract with explicit promotion rules. That final need is exactly `reinforce_exp2`.

---

## Why reinforce_exp2 Exists

`reinforce_exp2` is designed to solve four endgame problems:

1. **Claim-evidence mismatch risk**
   - Every promoted claim must map to a specific confirmatory artifact bundle with schema validation.

2. **Pipeline ambiguity risk**
   - Historical runs mixed smoke/proxy/full behavior. `reinforce_exp2` introduces explicit phase dependencies, strict prerequisites, and resumable artifact validation.

3. **Interpretation drift risk**
   - Allowed language is tied to evidence status (`supported`, `supported_with_caveat`, `proxy_specific`, `mixed`, `pending`, `deferred`).

4. **Reproducibility and governance risk**
   - Shared calibration split (`calibration_v1`), checksum pinning, per-run manifests, and schema-validated contracts are first-class requirements.

---

## Design Philosophy in reinforce_exp2

The folder is intentionally more operational than narrative docs. It enforces:

1. **Frozen model-set logic** with reliability eligibility.
2. **Prereg package per experiment**.
3. **Confirmatory vs exploratory tiering** with `canonical_eligible` and `override_used` tags.
4. **Data split hygiene**, especially calibration/eval disjointness where required.
5. **Multiplicity-aware statistics** and practical-threshold checks.
6. **Artifact contract minimum** per experiment:
   - `preregistration.json`
   - `manifest.json`
   - `summary.json`
   - `claim_impact.json`
   - `data_dictionary.json`

The implementation supports these through `reinforce_exp2/scripts/_shared.py` (core artifact emission, schema validation, pipeline metadata injection).

---

## The A/B/C Narrative Structure

`reinforce_exp2/TODO.md` defines three paths. They are not alternatives in rigor; they are alternatives in ambition.

## A Path: Honest Characterization (mandatory baseline hygiene)

A path is fast and conservative. It ensures the paper is internally consistent before any further mechanistic escalation.

### A0: Claim hygiene
- Builds explicit claim matrix and wording changes.
- Goal: remove overclaim and enforce status-tagged language.

### A1: Evidence consolidation
- Builds cross-experiment evidence registry plus discrepancy log.
- Goal: ensure no unresolved claim/artifact mismatch.

### A2: Prediction layer
- Freezes at least one falsifiable prediction.
- Goal: avoid purely retrospective storytelling.

This path is mandatory because it constrains every later claim promotion.

## B Path: Kernel-as-mechanism deepening (primary recommended path)

B path is the main mechanistic bridge for submission novelty while staying computationally bounded.

### B1: Kernel shape taxonomy
- Clusters heads in descriptor/spectral/offset spaces.
- Adds stability, prototype similarity, depth-controlled enrichment, and reliability gates.
- Purpose: test whether high-SI behavior decomposes into stable, interpretable kernel families.

### B2a: Head-level kernel-offset alignment
- Quantifies signed/absolute alignment between kernel offset structure and boundary-linked signals.
- Purpose: establish per-head directional alignment with controls.

### B2b: Cluster-level enrichment (depends on B1+B2a)
- Tests whether B1 clusters are enriched for B2a alignment signals under depth controls.
- Purpose: move from head-wise noise to cluster-level mechanism evidence.

### B3: Cluster-wise ablation disambiguation
- Full matched triplets (`within`, `mixed`, `random`) with strict layer/magnitude matching.
- Fits linear vs threshold-piecewise curves (AICc), compares collapse points, and computes contribution-share diagnostics.
- Purpose: disambiguate threshold behavior as distributed redundancy vs group-structure artifact.

### B4: Confound-isolated high-vs-low SI comparison
- Within-layer and entropy-matched contrasts with retention criteria.
- Purpose: ensure high-vs-low SI differences are not confound-driven artifacts.

## C Path: Two-carrier-class theory (high risk, high reward)

C path attempts a stronger, more specific mechanistic story: SI and non-SI content-conditional carriers interact differently across models.

### C1: Causal tracing with strict carrier sets
- Builds static carrier sets (`SI`, `LowSI`, `ContentCond`) using fastText-based content-similarity alignment with offset-bin partial control.
- Runs corruption/restoration ratio analyses plus interaction tests.
- Enforces quality and disjointness gates.

### C2: Carrier-class conditional predictions
- C2.1: position-shuffle sensitivity slopes.
- C2.2: long-context carrier-class interaction with preregistered failure-mode assignment.
- Purpose: convert C1 structure into falsifiable prediction tests.

C path can yield high-impact mechanism claims, but only if strict criteria pass.

---

## What Is Implemented in Code

Core executables in `reinforce_exp2/scripts/`:

1. `run_preflight.py`
   - Ensures schemas exist.
   - Installs `fasttext` if missing.
   - Downloads and checksum-pins `cc.en.300.bin` under `shared_storage/reinforce_exp2_assets/fasttext/`.

2. `build_calibration_v1.py`
   - Builds frozen `calibration_v1` split (target 4096 sequences, approximately 1/3 wiki, 1/3 code, 1/3 dialogue).
   - Emits IDs parquet, tokens parquet, and manifest with SHA256.

3. `run_a0_claim_hygiene.py`, `run_a1_evidence_consolidation.py`, `run_a2_prediction_layer.py`
   - Implement A path contracts and emit schema-bound core artifacts.

4. `run_b1_kernel_taxonomy.py`, `run_b2a_head_alignment.py`, `run_b2b_cluster_alignment.py`, `run_b3_cluster_ablation.py`, `run_b4_confound_isolation.py`
   - Implement B path end to end, including B3 matched-set mechanism disambiguation.

5. `run_c1_olmo_causal_trace.py`, `run_c2_carrier_class_predictions.py`, `_carrier_sets.py`
   - Implement C path carrier construction, causal trace metrics, and prediction tests.

6. `run_pipeline.py`
   - Provides phase orchestration with strict dependencies, `--mode full|smoke`, resumable strict artifact checks, per-task logs, and run records.

---

## Execution and Orchestration Model

There are two orchestration layers:

1. **Python pipeline** (`run_pipeline.py`)
   - Handles dependency graph, resume behavior, and per-task log capture.
   - Requires core artifacts plus experiment-specific primary tables before marking an experiment complete.

2. **tmux operator scripts** (`launch_tmux_full.sh`, `status_tmux.sh`, `stop_tmux.sh`)
   - Launches staged windows:
     - `preflight`
     - `phase_a`
     - `phase_b_core`
     - `phase_b_adv`
     - `phase_c`
     - `monitor`
   - Uses marker files and event logs for dependency gating and run tracking.

This separation allows both single-command reproducibility and operational monitoring on multi-GPU systems.

---

## Current Artifact Snapshot (As of 2026-04-20)

This snapshot is from `results/reinforce_exp2/` and associated pipeline records.

## Preflight and calibration

1. `preflight` is present and passes strict readiness checks.
2. `calibration_v1` is present:
   - `target_n = 4096`
   - domain counts: wiki 1366, code 1365, dialogue 1365
   - `sha256_ids_parquet = ee30db624a8668077ecca634655856a019fce245c180ff2b06988009926e90c8`

## A path status

1. **A0** complete (`n_claims=5`; statuses include pending/proxy_specific/supported_with_caveat/mixed).
2. **A1** complete (`n_discrepancies=4`, high-severity=0).
3. **A2** complete (`n_predictions=1`).

## B path status

1. **B1** complete artifact bundle, verdict `B1_supported=false` (`claim_status=mixed`).
   - Reported issue: reliability eligibility shortfall (`n_eligible_models=0` in taxonomy summary).

2. **B2a** verdict `B2a_supported=false`.
   - Directional criterion passes, corrected significance/practical criteria fail.

3. **B2b** verdict `B2b_supported=true`.
   - Holm/depth/practical criteria all pass.

4. **B3** available under `B3_cluster_ablation_test/` with full disambiguation artifacts.
   - Current summary there is exploratory with `B3_inconclusive` (reason: `insufficient_eligible_models`).
   - Canonical root `B3_cluster_ablation/` is currently empty in this snapshot.

5. **B4** verdict `B4_supported=false` (`mixed`).
   - Direction/statistical/practical retention criteria currently fail.

## C path status

1. Code for C1/C2 is implemented.
2. In this filesystem snapshot, canonical output roots
   - `results/reinforce_exp2/C1_olmo_causal_trace/`
   - `results/reinforce_exp2/C2_carrier_class_predictions/`
   are currently empty.
3. A smoke artifact exists in `C2_carrier_class_predictions_smoke/position_shuffle_results.parquet`.

This means C-path claim adjudication is not yet represented by complete canonical artifact bundles in the current tree snapshot.

---

## How to Read reinforce_exp2 Scientifically

`reinforce_exp2` should be read as a claim-governance framework, not as a single experiment.

Its core scientific function is to convert prior heterogeneous findings into a disciplined decision process:

1. **A path** guarantees narrative honesty.
2. **B path** tests whether kernel-level organization is mechanistically informative beyond descriptive SI scores.
3. **C path** attempts a stronger dual-carrier theory with explicit failure assignment.

If B and/or C fail strict criteria, the design still preserves publication value by forcing conservative but defensible interpretation rather than optimistic overreach.

---

## Why This Is Important for NeurIPS Readiness

From a reviewer perspective, `reinforce_exp2` directly addresses common rejection vectors:

1. unclear prereg/confirmatory boundary,
2. post-hoc storytelling,
3. missing controls or dependence-aware inference,
4. hidden pipeline ambiguity,
5. unclear provenance of promoted claims.

By construction, `reinforce_exp2` makes these failure modes explicit and auditable.

---

## Practical Reading Order for This Folder

If you want to understand the folder quickly and correctly:

1. Read `reinforce_exp2/TODO.md` for scientific contract and acceptance logic.
2. Read `reinforce_exp2/README.md` for runnable scope and artifact policy.
3. Read `reinforce_exp2/scripts/run_pipeline.py` for dependency/resume semantics.
4. Read `run_b3_cluster_ablation.py` and `_carrier_sets.py` for the highest-leverage method details.
5. Inspect `results/reinforce_exp2/*/summary.json` and `claim_impact.json` for claim-level status.

---

## Bottom Line

`reinforce_exp2` is the final consolidation scaffold that turns the Kernel PE research arc into a rigorous submission package. It preserves the strongest empirical core from experiments 1-8 and `reinforce_exp`, enforces strict claim discipline, and provides a clear path for either:

1. promoting mechanism claims when criteria pass, or
2. maintaining conservative characterization when they do not.

That is exactly the role this phase should play before NeurIPS submission.
