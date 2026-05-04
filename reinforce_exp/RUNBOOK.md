# reinforce_exp Runbook

## Implemented Experiments

- `EXP-R1` strict-gate multidomain adjudication:
  - Script: `reinforce_exp/exp_r1_multidomain_gate.py`
  - Outputs: `results/reinforce_exp/exp_r1_multidomain_gate/`
  - Artifacts:
    - `multiseed_gate_summary_v2` equivalent per domain via canonical summaries
    - `multidomain_gate_summary_v1.json`

- `EXP-R2` dependence-aware reanalysis:
  - Script: `reinforce_exp/exp_r2_dependence_reanalysis.py`
  - Outputs: `results/reinforce_exp/exp_r2_dependence_reanalysis/`
  - Artifacts:
    - `claim_stability_table.json`
    - `model_summaries/*.json`

- `EXP-R3` third-model core replication (default `mistral-7b-v0.1`):
  - Orchestrator: `reinforce_exp/exp_r3_core_replication.py`
  - Helpers:
    - `reinforce_exp/exp_r3_c2_extended.py`
    - `reinforce_exp/exp_r3_conditional_regimes_generic.py`
  - Outputs: `results/reinforce_exp/exp_r3_core_replication/<model>/`

- `EXP-R4` PE-scheme contrast extension:
  - Script: `reinforce_exp/exp_r4_pe_scheme_contrast.py`
  - Outputs: `results/reinforce_exp/exp_r4_pe_scheme_contrast/`
  - Artifact: `pe_scheme_comparison.json`

- `EXP-R5` task-grounded conditional specialization:
  - Script: `reinforce_exp/exp_r5_task_grounded_specialization.py`
  - Outputs: `results/reinforce_exp/exp_r5_task_grounded/`
  - Artifacts:
    - `*/regime_summary.json`
    - `interaction_comparison_proxy_vs_task.json`

## tmux Orchestration

- Launch all experiments:

```bash
bash reinforce_exp/launch_tmux.sh
```

- Monitor status:

```bash
bash reinforce_exp/status_tmux.sh
```

## Session Names

- `reinforce_r1_g2`
- `reinforce_r1_g3`
- `reinforce_r1_g4`
- `reinforce_r1_finalize`
- `reinforce_r2`
- `reinforce_r3`
- `reinforce_r4`
- `reinforce_r5`

## Notes

- Launcher uses GPUs `2..7` only.
- `EXP-R5` supports `--count-scale` for smoke vs full runs.
- All scripts emit JSON manifests for traceability.
