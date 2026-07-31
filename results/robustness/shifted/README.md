# Shifted-Window Ten-Seed Robustness Evaluation

This directory contains the prespecified secondary evaluation with encoder
fitting through 2015, encoder validation and PPO fitting through 2016, and
portfolio evaluation during 2017--2018. It retains the primary model settings,
20,000-step PPO budget, costs, bond dimension 2, and matched seeds 0--9. Both
encoder fitting and validation apply the next-day target boundary guard.

## Window-specific result

- Base FinRL mean Sharpe: 0.652458.
- ANN mean Sharpe: 0.627464.
- MPS mean Sharpe: 0.713709.
- Mean paired MPS-minus-ANN Sharpe: +0.086245.
- Paired-seed bootstrap 95% interval: [0.001119, 0.194500].
- MPS Sharpe was higher in 9 of 10 seeds.
- Exact two-sided sign-test p-value: 0.021484.
- Annualized MPS-minus-ANN return difference: +2.34 percentage points.
- Twenty-day moving-block-bootstrap return interval: [-0.83, 5.98]
  percentage points.

This result is secondary, window-specific evidence. The seed directions favor
MPS in 2017--2018, but the return interval includes zero and the corrected
2019--2023 primary comparison has the opposite mean sign. The supported
conclusion is evaluation-window sensitivity, not stable MPS superiority.

## Provenance

`run_manifest.json` and `run_status.json` record the exact dates, completed
status, CPU-only environment, seeds, source commit, and runtime. The guarded
repository summaries are `condition_summary.csv`, `paired_seed_effects.csv`,
`robustness_inference.json`, and `robustness_manifest.json`. Runner-generated
annual, block-bootstrap, signal, and curve artifacts are preserved alongside
them. Generated diagnostic figures are under `figures/`.

## Regenerate guarded summaries

```powershell
python scripts/summarize_final_evaluation.py results/robustness/shifted `
  --condition-output results/robustness/shifted/condition_summary.csv `
  --paired-output results/robustness/shifted/paired_seed_effects.csv `
  --inference-output results/robustness/shifted/robustness_inference.json `
  --manifest-output results/robustness/shifted/robustness_manifest.json `
  --artifact-name shifted_window_ten_seed_evaluation

python scripts/plot_final_effect.py `
  results/robustness/shifted/paired_seed_effects.csv `
  results/robustness/shifted/robustness_inference.json `
  --png-output results/figures/shifted_paired_effect.png `
  --pdf-output results/figures/shifted_paired_effect.pdf
```