# reinforce_exp3 Rigorous Repair + Relaunch (2026-04-29)

## Run root
- `results/reinforce_exp3/runs/rexp3_fix_20260429_192038`

## Code repairs applied
- Corrected permutation test implementation in `reinforce_exp3/scripts/_shared.py` to use one permutation split per draw.
- Added deterministic RNG stream helpers + coverage contract enforcement in shared utilities.
- Patched head ablation in `experiment3/theory1_si_circuits.py`:
  - runtime head-dim inference,
  - divisibility guard,
  - explicit incompatible-layout errors,
  - model-structure guards.
- E0: deterministic stream-seeded permutation tests + Holm integration + hard fail on low tau-fit coverage.
- E2: deterministic ordering provenance + hard fail coverage contracts.
- E3: fixed stale criterion wording (`0/(2*n_models)`), added shard-safe finalize-only / no-finalize flow with required-model coverage.
- E5: removed unconditional support flag, aligned claims to actually run components, added architecture guards, hard-fail component coverage, shard finalize flow.
- E6: replaced skip/insufficient paths with fail-hard on missing domain/seed/feature coverage; finalize-only / no-finalize flow.
- E7: removed alternating-token Chinese fallback; invalid boundary construction now fail-hard; added coverage contracts + shard finalize flow.
- E10: fixed overlap semantics (negative rho is separation, not overlap), fail-hard on missing SI data/non-finite stats, shard finalize flow.
- E11: disabled synthetic kernel fallback in confirmatory mode (unless explicitly allowed), coverage contracts, shard finalize flow.
- E12: denominator-guarded preferential degradation criteria, explicit indeterminate state, valid-model-only cross-model interpretation, shard finalize flow.

## Preflight checks completed
- `py_compile` passed for all modified scripts + dependency patches.
- CLI help checks passed for all modified CLIs.
- Determinism smoke: permutation test repeated with same seed gives identical `(observed_diff, p)`.
- Ablation-hook smoke:
  - Llama real forward with head ablation passes and yields finite logits.
  - Guard-path smoke confirms clear runtime error on incompatible hidden/head layout.
- Fail-hard smokes:
  - E11 aborts on missing real kernels in confirmatory mode.
  - E6 aborts on missing required domain sequences.
- Logic smokes:
  - E10 synthetic negative-rho input classified as separation/disjoint.
  - E12 near-zero denominator scenario classified as `insufficient_valid_models`.

## Relaunch orchestration
- tmux sessions:
  - `rexp3fix_rexp3_fix_20260429_192038_gpu0`
  - `rexp3fix_rexp3_fix_20260429_192038_gpu1`
  - `rexp3fix_rexp3_fix_20260429_192038_gpu1fb` (fallback queue)
  - `rexp3fix_rexp3_fix_20260429_192038_gpu2`
  - `rexp3fix_rexp3_fix_20260429_192038_gpu3`
  - `rexp3fix_rexp3_fix_20260429_192038_gpu4`
  - `rexp3fix_rexp3_fix_20260429_192038_gpu5`
  - `rexp3fix_rexp3_fix_20260429_192038_cpu`
- Logs: `results/reinforce_exp3/runs/rexp3_fix_20260429_192038/logs/`

## Operational notes
- GPU free-only policy is active (workers wait for busy GPUs instead of preempting).
- GPU0 currently occupied by an external notebook PID and is waiting.
- GPU2/3/4/5 currently occupied by running non-rexp3 jobs; workers are waiting.
- GPU1 is actively running E2 at time of this note.
- CPU watcher has completed E9 and E8 and is waiting for shard artifacts for finalize steps.

## Known external constraint
- `google/gemma-2-9b` access is gated in this environment; E5 may fail if the model cannot be loaded from authorized cache.
