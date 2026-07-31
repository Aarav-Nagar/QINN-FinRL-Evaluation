# Corrected Final Ten-Seed Evaluation

This directory contains the boundary-corrected primary evaluation for the
selected 20,000-step PPO budget and MPS bond dimension 2. It includes matched
seeds 0--9 for Base FinRL, ANN signal, and QINN-MPS signal; no seed was
excluded. Encoder-fit and validation rows are retained only when their next-day
target remains inside the corresponding period.

## Primary result

- ANN mean Sharpe: 0.821005.
- MPS mean Sharpe: 0.784397.
- Mean paired MPS-minus-ANN Sharpe: -0.036607.
- Paired-seed bootstrap 95% interval: [-0.112857, 0.044800].
- MPS-minus-ANN Sharpe was positive in 3 of 10 seeds.
- Exact two-sided sign-test p-value: 0.34375.
- Annualized MPS-minus-ANN return difference: -0.81 percentage points.
- Twenty-day moving-block-bootstrap return interval: [-3.00, 1.37]
  percentage points.

The intervals describe variation within this fixed historical experiment. They
are not population-level or causal intervals. The corrected result does not
establish an MPS advantage and does not establish general inferiority beyond
this configuration and period.

## Artifacts

- `run_manifest.json` and `run_status.json`: exact configuration, dates,
  boundary controls, runtime, environment, source commit, and checksums.
- `ppo_backtest_metrics.csv`: every condition/seed endpoint plus the separate
  equal-weight reference.
- `equity_curves.csv`: daily out-of-sample account values and trading records.
- `signal_metrics.csv` and `encoder_training_history.csv`: predictive evidence.
- `condition_seed_summary.csv`, `annual_period_metrics.csv`,
  `annual_period_seed_summary.csv`, and `ann_vs_mps_block_bootstrap.csv`:
  runner-generated descriptive and time-series analyses.
- `condition_summary.csv`, `paired_seed_effects.csv`, and
  `primary_inference.json`: repository-generated guarded ten-seed summaries.
- `final_manifest.json`: SHA-256 provenance for source and summary files.

## Regenerate guarded summaries

```powershell
python scripts/summarize_final_evaluation.py results/final `
  --condition-output results/final/condition_summary.csv `
  --paired-output results/final/paired_seed_effects.csv `
  --inference-output results/final/primary_inference.json `
  --manifest-output results/final/final_manifest.json `
  --artifact-name final_ten_seed_evaluation

python scripts/plot_final_effect.py `
  results/final/paired_seed_effects.csv `
  results/final/primary_inference.json `
  --png-output results/figures/final_paired_effect.png `
  --pdf-output results/figures/final_paired_effect.pdf
```