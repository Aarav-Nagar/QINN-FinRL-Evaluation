# Saved Results

This directory contains the outputs from the reference run using PPO seeds 0,
1, and 2.

## Primary result files

| File | Contents |
|---|---|
| `run_manifest.json` | Full configuration, dates, state construction, checksums, feature lists, and limitations |
| `signal_metrics.csv` | ANN and MPS prediction metrics on the 2019-2023 test period |
| `ppo_backtest_metrics.csv` | Full-period portfolio metrics for every condition and seed |
| `ann_vs_mps_block_bootstrap.csv` | Paired MPS-minus-ANN return difference and 95% block-bootstrap interval |
| `annual_period_metrics.csv` | Calendar-year metrics for every condition and seed |
| `condition_seed_summary.csv` | Full-period means and exploratory seed-level intervals |
| `annual_period_seed_summary.csv` | Calendar-year summaries across seeds |
| `equity_curves.csv` | Daily account value, returns, executed notional, turnover, and costs |
| `encoder_training_history.csv` | ANN and MPS training and validation loss by epoch |

## Figures

- `figures/equity_curves.png`: mean out-of-sample equity curves and
  seed-to-seed variation.
- `figures/sharpe_by_condition.png`: Sharpe ratios by condition and PPO seed.
- `figures/encoder_validation_loss.png`: prediction-model validation loss.

## Key values

| Condition | Mean Sharpe | Total return | Maximum drawdown | Annualized turnover | Total cost |
|---|---:|---:|---:|---:|---:|
| Base FinRL | 0.559 | 73.24% | -40.46% | 26.43% | $1,375.38 |
| ANN signal | 0.735 | 104.66% | -27.55% | 27.32% | $1,429.65 |
| MPS signal | 0.694 | 94.27% | -26.79% | 24.14% | $1,253.61 |
| Equal-weight benchmark | 1.115 | 218.25% | -28.54% | 20.06% | $1,000.00 |

The paired annualized mean-return difference, MPS minus ANN, was -0.92
percentage points. The moving 20-day block-bootstrap 95% interval was -4.63
to +2.63 percentage points.

All PPO summary values are means across seeds 0, 1, and 2. The equal-weight
benchmark has no PPO seed.
