# Experiment 3 Phase 2 tmux Runbook (2026-04-06)

## Scope launched now
- `3P2-B` rerun with **full controls** (fresh forward passes + full 2x2 synthetic cells):
  - `llama-3.1-8b`
  - `olmo-2-7b`
- `3P2-D.1` clean rerun for `mistral-7b-v0.1` with direct-prereq enforcement.

## Sessions
- `exp3p2_b_llama`
- `exp3p2_b_olmo`
- `exp3p2_d1_mistral`

## Launch
```bash
cd "/jumbo/lisp/f004ndc/Kernel PE"
bash scripts/experiment3_phase2_rerun_launch_tmux.sh
```

## Monitor
```bash
cd "/jumbo/lisp/f004ndc/Kernel PE"
bash scripts/experiment3_phase2_rerun_status.sh
```

Per-session tails:
```bash
tmux capture-pane -p -t exp3p2_b_llama | tail -n 120
tmux capture-pane -p -t exp3p2_b_olmo | tail -n 120
tmux capture-pane -p -t exp3p2_d1_mistral | tail -n 120
```

## Session scripts
- `scripts/experiment3_phase2_rerun_b_full_gpu0.sh`
- `scripts/experiment3_phase2_rerun_b_full_gpu1.sh`
- `scripts/experiment3_phase2_rerun_d1_clean_gpu2.sh`

## Output artifacts to check
- `results/experiment3_phase2/exp3p2b_trivial_feature_control/<model>/space_prefix_classifier.json`
- `results/experiment3_phase2/exp3p2b_trivial_feature_control/<model>/post_ablation_t5b_a.json`
- `results/experiment3_phase2/exp3p2b_trivial_feature_control/<model>/synthetic_boundary_results.json`
- `results/experiment3_phase2/exp3p2b_trivial_feature_control/<model>/adversarial_sequences.parquet`
- `results/experiment3_phase2/exp3p2d_architecture_tiebreaker/mistral-7b-v0.1/report.json`
- `results/experiment3_phase2/exp3p2d_architecture_tiebreaker/cross_model_comparison.json`

## Failure scan
```bash
cd "/jumbo/lisp/f004ndc/Kernel PE"
rg -n "Traceback|RuntimeError|CUDA out of memory|error:" logs/experiment3_phase2_rerun -S
```

## Stage 3 add-on (launched separately)
Scope:
- `3P2-I + Idea 2` cross-lingual invariance (`llama-3.1-8b`, `olmo-2-7b`)
- `Idea 4` structural ambiguity intervention test (`llama-3.1-8b`, `olmo-2-7b`)
- Optional `3P2-C.2` simultaneous ablation stress test (`llama-3.1-8b`, `olmo-2-7b`)

Launch:
```bash
cd "/jumbo/lisp/f004ndc/Kernel PE"
bash scripts/experiment3_phase2_stage3_launch_tmux.sh
```

Monitor:
```bash
cd "/jumbo/lisp/f004ndc/Kernel PE"
bash scripts/experiment3_phase2_stage3_status.sh
```

Sessions:
- `exp3p2_stage3_llama`
- `exp3p2_stage3_olmo`

Stage-3 session scripts:
- `scripts/experiment3_phase2_stage3_gpu0.sh`
- `scripts/experiment3_phase2_stage3_gpu1.sh`
- `scripts/experiment3_phase2_stage3_launch_tmux.sh`
- `scripts/experiment3_phase2_stage3_status.sh`

Primary stage-3 artifacts:
- `results/experiment3_phase2/exp3p2i_tokenizer_corpus/<model>/invariance_report.json`
- `results/experiment3_phase2/exp3p2i_tokenizer_corpus/<model>/corpus_breakdown.parquet`
- `results/experiment3_phase2/idea4_structural_ambiguity/<model>/stimulus_manifest.json`
- `results/experiment3_phase2/idea4_structural_ambiguity/<model>/per_item_scores.parquet`
- `results/experiment3_phase2/idea4_structural_ambiguity/<model>/ambiguity_report.json`
- `results/experiment3_phase2/exp3p2c_redundancy_quantification/<model>/simultaneous_ablation_results.parquet`
- `results/experiment3_phase2/exp3p2c_redundancy_quantification/<model>/nonlinearity_test.json`

## Notes
- `exp3p2b_trivial_feature_control.py` now supports `--mode full`.
- `exp3p2d_architecture_tiebreaker.py` now supports `--ensure-direct-prereqs`.
- `theory1_si_circuits.py` and `theory7_induction_feeders.py` now include `mistral-7b-v0.1` in `--model` choices for direct artifact generation.
- `llama-3.1-8b` currently fails the strict 3P2-I gate (`prefix_following_artifact_flag=true`), so stage-3 runner uses `--force-gate-override` for that model and records override status in `execution_gate`.
- `theory7_induction_feeders.py` skip behavior was patched on 2026-04-07: `--skip-approach-b` now always skips knockout execution (even when cached knockout rows are absent).
- Relaunch session for cleaned D.1 path: `exp3p2_d1_mistral_fix`.

---

## 2026-04-09 Generalization Push (New)

### Phase 2: Llama strict-gate multiseed adjudication (7 seeds)
Launch:
```bash
cd "/jumbo/lisp/f004ndc/Kernel PE"
bash scripts/experiment3_phase2_b_multiseed_launch_tmux.sh
```

Monitor:
```bash
cd "/jumbo/lisp/f004ndc/Kernel PE"
bash scripts/experiment3_phase2_b_multiseed_status.sh
tmux capture-pane -p -t exp3p2_b_multi_g0 | tail -n 120
tmux capture-pane -p -t exp3p2_b_multi_g1 | tail -n 120
tmux capture-pane -p -t exp3p2_b_multi_finalize | tail -n 120
```

Sessions:
- `exp3p2_b_multi_g0` (seeds `0,2,4,6`)
- `exp3p2_b_multi_g1` (seeds `1,3,5`)
- `exp3p2_b_multi_finalize` (waiter + pooled adjudication + optional strict Llama 3P2-I relaunch if clean)

Key artifacts:
- `results/experiment3_phase2/exp3p2b_trivial_feature_control_multiseed/seed*/llama-3.1-8b/*`
- `results/experiment3_phase2/exp3p2b_trivial_feature_control/llama-3.1-8b/multiseed_gate_summary.json`

### Phase 4: 3P2-J long-span repair
Launch:
```bash
cd "/jumbo/lisp/f004ndc/Kernel PE"
bash scripts/experiment3_phase2_j_longspan_launch_tmux.sh
```

Monitor:
```bash
cd "/jumbo/lisp/f004ndc/Kernel PE"
bash scripts/experiment3_phase2_j_longspan_status.sh
```

Sessions:
- `exp3p2_j_longspan_llama`
- `exp3p2_j_longspan_olmo`

Key artifacts:
- `results/experiment3_phase2/exp3p2j_conditional_regimes_longspan_repair/<model>/regime_summary.json`
- `results/experiment3_phase2/exp3p2j_conditional_regimes_longspan_repair/<model>/regime_coverage_manifest.json`

### Phase 5: non-RoPE anchor control
Launch:
```bash
cd "/jumbo/lisp/f004ndc/Kernel PE"
bash scripts/experiment3_phase2_nonrope_launch_tmux.sh 2 auto 24 300
```

Monitor:
```bash
cd "/jumbo/lisp/f004ndc/Kernel PE"
bash scripts/experiment3_phase2_nonrope_status.sh
tmux capture-pane -p -t exp3p2_nonrope | tail -n 120
```

Key artifacts:
- `results/experiment3_phase2/exp3p2k_non_rope_control/non_rope_control_summary.json`
- `results/experiment3_phase2/exp3p2k_non_rope_control/<selected_model>/non_rope_control_summary.json`

### Phase 3: tokenizer-entanglement audit
Launch (after Phase 2/4 critical runs):
```bash
cd "/jumbo/lisp/f004ndc/Kernel PE"
bash scripts/experiment3_phase2_tokenizer_audit_launch_tmux.sh
```

Monitor:
```bash
cd "/jumbo/lisp/f004ndc/Kernel PE"
bash scripts/experiment3_phase2_tokenizer_audit_status.sh
```

Key artifacts:
- `results/experiment3_phase2/exp3p2b_tokenizer_audit/tokenizer_audit_report.json`
- `results/experiment3_phase2/exp3p2b_tokenizer_audit/<model>/tokenizer_audit_report.json`
