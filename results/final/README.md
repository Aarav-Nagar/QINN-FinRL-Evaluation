# Final Ten-Seed Evaluation

This directory contains the guarded fixed-split evaluation for the selected
20,000-step PPO budget and MPS bond dimension 2. It includes all ten matched
seeds for Base FinRL, ANN signal, and QINN-MPS signal; no seed was excluded.

## Primary result

- ANN mean Sharpe: 0.799767.
- MPS mean Sharpe: 0.761550.
- Mean paired MPS-minus-ANN Sharpe: -0.038217.
- Paired-seed bootstrap 95% interval: [-0.089151, 0.015716].
- MPS-minus-ANN Sharpe was positive in 3 of 10 seeds.
- Exact two-sided sign-test p-value: 0.34375.

The interval describes variation across ten matched policy-training seeds on
one historical split. It is not a population-level or causal interval. The
result does not establish an MPS advantage; it also does not establish general
inferiority beyond this configuration and period.

## Artifacts

- `run_manifest.json`: configuration, runtime, environment, input checksums,
  and recovery disclosure.
- `ppo_backtest_metrics.csv`: every condition/seed endpoint plus the unseeded
  equal-weight reference.
- `equity_curves.csv`: daily out-of-sample account values.
- `signal_metrics.csv`: validation and test prediction metrics.
- `encoder_training_history.csv`: encoder fit histories.
- `condition_summary.csv`: condition-level seed summaries.
- `paired_seed_effects.csv`: matched MPS-minus-ANN differences for every seed.
- `primary_inference.json`: deterministic paired-seed bootstrap and exact sign
  test.
- `final_manifest.json`: SHA-256 provenance for source and generated files.

The training process completed all tabular outputs and then failed during plot
writing because a Windows path exceeded the default path limit. Recovery
validated the configured condition/seed set, regenerated plots with safe
filenames, preserved the training commit, and recorded the finalization commit.
Encoder timing was not persisted before the failure and is explicitly
unavailable rather than estimated.

## Reproduce the guarded summaries

```powershell
python scripts/summarize_final_evaluation.py results/final `
  --condition-output results/final/condition_summary.csv `
  --paired-output results/final/paired_seed_effects.csv `
  --inference-output results/final/primary_inference.json `
  --manifest-output results/final/final_manifest.json

python scripts/plot_final_effect.py `
  results/final/paired_seed_effects.csv `
  results/final/primary_inference.json `
  --png-output results/figures/final_paired_effect.png `
  --pdf-output results/figures/final_paired_effect.pdf
```
