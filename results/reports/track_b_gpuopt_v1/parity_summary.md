# Track B GPU-Optimized Parity Report

## Thresholds
- `MAE(r2_raw) <= 1e-10`
- `MAE(r2_centered) <= 1e-07`
- `max_abs_delta(any r2 field) <= 1e-05`

## Coverage
| group | expected_summaries | present_summaries | complete |
| --- | --- | --- | --- |
| track_b_bucketed_mean_b16_gpuopt_v1 | 36 | 0 | False |
| track_b_bucketed_mean_b32_gpuopt_v1 | 36 | 0 | False |
| track_b_bucketed_mean_b64_gpuopt_v1 | 36 | 0 | False |
| track_b_bucketed_mean_b8_gpuopt_v1 | 36 | 0 | False |
| track_b_canonical_perpos_gpuopt_v1 | 36 | 0 | False |
| track_b_gpuopt_v1 | 36 | 2 | False |
| track_b_shared_mean_gpuopt_v1 | 36 | 0 | False |

## Variant-Matched Parity
| gpuopt_group | baseline_group | overlap_rows | mae_r2_raw | mae_r2_centered | max_abs_r2_raw | max_abs_r2_centered | max_abs_any_r2 | pass_mae_raw | pass_mae_centered | pass_max_abs_any | pass_all |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| track_b_bucketed_mean_b16_gpuopt_v1 | track_b_bucketed_mean_b16_v1 | 0 |  |  |  |  |  | False | False | False | False |
| track_b_bucketed_mean_b32_gpuopt_v1 | track_b_bucketed_mean_v1 | 0 |  |  |  |  |  | False | False | False | False |
| track_b_bucketed_mean_b64_gpuopt_v1 | track_b_bucketed_mean_b64_v1 | 0 |  |  |  |  |  | False | False | False | False |
| track_b_bucketed_mean_b8_gpuopt_v1 | track_b_bucketed_mean_b8_v1 | 0 |  |  |  |  |  | False | False | False | False |
| track_b_canonical_perpos_gpuopt_v1 | track_b_canonical_perpos_v1 | 0 |  |  |  |  |  | False | False | False | False |
| track_b_gpuopt_v1 | track_b | 1408 | 6.54463856846e-17 | 5.43283852249e-17 | 4.4408920985e-16 | 4.4408920985e-16 | 4.4408920985e-16 | True | True | True | True |
| track_b_shared_mean_gpuopt_v1 | track_b_shared_mean_v1 | 0 |  |  |  |  |  | False | False | False | False |

## Raw Invariance Across GPU-Optimized Variants
| rows_with_2plus_variants | mean_spread_r2_raw | max_spread_r2_raw |
| --- | --- | --- |
| 0 |  |  |

## Synthetic Centered vs Raw Parity
| group | rows | mae_abs_centered_minus_raw | max_abs_centered_minus_raw |
| --- | --- | --- | --- |
| track_b_bucketed_mean_b16_gpuopt_v1 | 0 |  |  |
| track_b_bucketed_mean_b32_gpuopt_v1 | 0 |  |  |
| track_b_bucketed_mean_b64_gpuopt_v1 | 0 |  |  |
| track_b_bucketed_mean_b8_gpuopt_v1 | 0 |  |  |
| track_b_canonical_perpos_gpuopt_v1 | 0 |  |  |
| track_b_gpuopt_v1 | 0 |  |  |
| track_b_shared_mean_gpuopt_v1 | 0 |  |  |

## Runtime Log Summary
_No rows._

## Gate Status
- Overall strict parity pass: `False`
