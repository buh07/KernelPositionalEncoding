# HANDOFF — Shift-Invariant Attention rebuttal experiments

**Updated:** 2026-07-29 21:45 EDT · **Mode:** research-full · **Project HEAD:** `fd531e6`

## Goal

Implement, launch, complete, and review the frozen cross-corpus SI intervention and depth analyses
under `rebuttal/` on physical GPUs 0 and 1.

## Status

Production batch `si-r3-20260729T202806Z` is terminal and finalized. Both tmux sessions exited
cleanly, GPUs 0/1 are idle, all 1,062 preregistered shards and six fit profiles exist, and
`summaries/batch-terminal-summary.json` verifies all shards. Terminal identity:
`56b0439e4c20487ce78c1778cdb144607cb49599a9ae53418bbfe84b9c090c30`.
No runtime/OOM/environment error appears in either production log.

The final independent claim review returned `REVISE`: the narrow six-endpoint cross-corpus
dose-response claim is supported, but broader causal, superiority, pooled/population, or performance
claims are not.

## Results

- All six model×direction dose responses are positive: Spearman rho `0.764–0.986`, frozen one-sided
  permutation `p <= 8.5e-5`.
- Source-kernel loss disruption exceeds offset-permuted controls in 102/120 bin contrasts by paired
  95% CI; 11 overlap zero and 7 favor the offset control.
- Source-versus-in-domain-target contrasts are mixed (59/120 favor source, 16 favor target, 45
  overlap zero), so source-kernel superiority is not supported.
- Norm-matched random controls are more disruptive than source kernels in 119/120 bins; this control
  does not support a simple source-superiority claim.
- Wikipedia→code depth relationships are positive at the group/layer level for all models: rho
  `0.483`, `0.587`, and `0.805`, one-sided p `0.00278`, `0.000265`, and `0.000005`.

## Next

Use only the narrow reviewed wording in any rebuttal draft. If the manuscript/result report is
edited, obtain a fresh claim review of the exact bytes before presenting the conclusion.

**Exact next command:**

```bash
cd "/jumbo/lisp/f004ndc/experiments/submitted/NeurIPS-2026/Kernel PE" &&
python -m json.tool rebuttal/runs/si-r3-20260729T202806Z/summaries/batch-terminal-summary.json >/dev/null
```

## Do not

- Do not claim that SI is learned rather than architectural, individual-head causality, transferred
  source-kernel superiority, population-level generalization, or performance improvement.
- Do not pool the six endpoints or reinterpret the norm controls after seeing outcomes.
- Do not alter the frozen data split, metrics, protocol, `per_layer_si_summary.md`, or `rebuttal.md`.
- Do not reuse failed batches, commit, or push without explicit human instruction.

## Files in flight

- `rebuttal/src/si_rebuttal/operations.py`
- `rebuttal/tests/test_launch.py`
- `rebuttal/runs/si-r3-20260729T202806Z/**` (ignored terminal artifacts)

## 2026-07-29 adversarial result audit

Fresh independent `/adversarial` returned `BLOCK`. All six primary rho/p values independently
recompute exactly; their exceedance counts are `0,0,0,16,0,0`, so five `4.999975e-6` values are the
Monte Carlo floor `(1+0)/(200000+1)`. The blocker is result-schema reporting: depth field
`positive_null_permutation_count` is hard-coded to the 200,000 trial budget rather than the actual
positive-tail exceedance counts `555,52,0`. Primary dose rows also omit seed/count provenance.
Do not overwrite the immutable terminal summary. Before claiming the artifact is error-free, plan
and review a schema/source/test correction and emit a new corrected terminal artifact; the raw
1,062 shards and independently recomputed p-values remain intact.
