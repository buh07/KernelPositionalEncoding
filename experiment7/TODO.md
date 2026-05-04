# Experiment 7 TODO (A+B Core)

## Scope
- Implement and run **7A** (Welch/coherence vs R² ceiling) and **7B** (phase-transition scaling core).
- Exclude the optional 2D Donoho-Tanner phase diagram in this pass.

## 7A Protocol
1. Parse Track-A combo table from `experiment1/experiment1results.md` (expects 36 rows).
2. Build analytical coherence profiles for each model and sequence length.
3. Compute Welch gap metrics: `mu1`, `mu_max`, `mu_welch`, `eta_welch_gap`.
4. Fit `predicted_r2 = 1 - alpha * mu1^2` against observed `early_mean_r2`.
5. Report Pearson/Spearman correlations and bootstrap CI (95%, 2000 samples, seed 42).

Artifacts:
- `results/experiment7/exp7a_welch_r2/coherence_profiles.parquet`
- `results/experiment7/exp7a_welch_r2/welch_gap_summary.parquet`
- `results/experiment7/exp7a_welch_r2/r2_prediction_results.json`
- `results/experiment7/exp7a_welch_r2/run_manifest.json`

## 7B Protocol
1. For each model (`llama-3.1-8b`, `olmo-2-7b`) and `N in {256,512,1024}`:
   - Run `exp3p2c_redundancy_quantification` (high-to-low curve) with `--ntp-seq-len N`.
2. Compute effective sparsity `s_eff` for high-SI heads using attention probabilities (`epsilon=0.01`).
3. Extract observed threshold `m*` from curve-fit threshold fraction (`high_to_low::wiki_ntp::loss`).
4. Fit `m*_pred = C * s_eff * log(N / s_eff)` across model×length cells.
5. Report Pearson/Spearman agreement between predicted vs observed thresholds.

Artifacts:
- `results/experiment7/exp7b_phase_transition/effective_sparsity.parquet`
- `results/experiment7/exp7b_phase_transition/mstar_predictions.parquet`
- `results/experiment7/exp7b_phase_transition/phase_transition_scaling.json`
- `results/experiment7/exp7b_phase_transition/run_manifest.json`

## Run Commands
- List entrypoints:
  - `python -m experiment7.run --list`
- Run all:
  - `python -m experiment7.run all --device cuda:0`
- Run 7A only:
  - `python -m experiment7.run 7a`
- Run 7B only:
  - `python -m experiment7.run 7b --device cuda:0`

## Acceptance Checks
1. 7A parses exactly **36** Track-A rows.
2. 7A outputs finite Welch-gap values for non-NoPE models.
3. 7B writes per-length C1 outputs under `exp7b_phase_transition/c1_runs/`.
4. 7B prediction table has 6 rows (2 models × 3 lengths).
5. No `Traceback`/`ERROR` in runtime logs.

