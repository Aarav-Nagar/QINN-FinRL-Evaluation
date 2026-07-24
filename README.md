# Testing ANN and MPS signals in FinRL

This repository contains my follow-up experiment based on Professor Xiao-Yang
Liu's suggestion to test a quantum-inspired representation inside a FinRL
trading agent.

I compared three PPO agents:

1. the regular FinRL market state;
2. the same state plus an ANN prediction for each stock; and
3. the same state plus a matrix-product-state (MPS) prediction for each stock.

The ANN and MPS models use the same inputs and each has 369 trainable
parameters. The MPS is a classical tensor-network model. It does not use
quantum hardware.

## Main result

The MPS had slightly better prediction MSE and directional accuracy, but it
did not produce better trading performance.

| Agent | Mean Sharpe | Total return | Max drawdown | Annualized turnover |
|---|---:|---:|---:|---:|
| Base FinRL | 0.559 | 73.24% | -40.46% | 26.43% |
| ANN signal | 0.735 | 104.66% | -27.55% | 27.32% |
| MPS signal | 0.694 | 94.27% | -26.79% | 24.14% |

These are means across PPO seeds 0, 1, and 2 on the 2019-2023 test period.
MPS trailed ANN in all three paired seeds. A 20-day block bootstrap gave a
95% interval of -4.63 to +2.63 percentage points for the MPS-minus-ANN
annualized return difference.

This is a negative result under one fixed setup, not a general conclusion
about tensor networks.

## Experiment dates

- Encoder training: 2013-2017
- Encoder validation: 2018
- PPO training: 2013-2018
- Out-of-sample test: 2019-2023

All agents used the same 15 stocks, PPO settings, 5,000-step training budget,
random seeds, and 0.10% transaction cost.

## Run the experiment

```powershell
python -m pip install -r requirements.txt
python run_experiment.py `
  --data-dir .cache\data `
  --finrl-dir .cache\FinRL `
  --output-dir results `
  --timesteps 5000 `
  --seeds 0 1 2
python -m pytest -q test_experiment.py
```

The reference results are already included in `results/`. The run manifest
records the dataset checksum, FinRL commit, settings, and limitations.

## Files

- `research_report.md`: short explanation of the setup and results
- `technical_report.pdf` and `technical_report.docx`: formatted technical note
- `run_experiment.py`: full experiment pipeline
- `test_experiment.py`: integrity tests
- `results/`: per-seed metrics, yearly metrics, equity curves, and bootstrap
  output

This repository is for research and education, not financial advice.
