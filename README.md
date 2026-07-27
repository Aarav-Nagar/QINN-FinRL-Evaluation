# ANN and MPS Signals in a FinRL PPO Agent

This repository evaluates whether a quantum-inspired matrix-product-state
(MPS) prediction signal improves sequential trading decisions relative to a
parameter-matched artificial neural network (ANN) signal.

The experiment was developed as a follow-up to Professor Xiao-Yang Liu's
suggestion to test quantum-inspired representations inside a FinRL benchmark,
rather than evaluating them only as prediction models.

## Development history

The initial commit imports an experiment developed before this standalone
repository was published. It includes the first complete research snapshot:
the ANN and MPS models, FinRL integration, saved backtest outputs, integrity
tests, and an initial report. Subsequent commits and pull requests document
repository preparation, reviewer-requested analysis, reproducibility
improvements, and continuing experimental development.

The work progressed through these stages:

1. Compare ANN and quantum-inspired models as stock-return predictors.
2. Add the ANN and MPS predictions separately to the FinRL PPO state.
3. Evaluate the agents out of sample with matched seeds and transaction costs.
4. Report drawdown, turnover, costs, calendar-period results, and uncertainty.
5. Package the experiment as a reproducible repository and technical report.

Planned follow-up work is tracked publicly:

- [Expand the PPO evaluation to 10-20
  seeds](https://github.com/Aarav-Nagar/QINN-FinRL-Evaluation/issues/3)
- [Run an MPS bond-dimension sensitivity
  sweep](https://github.com/Aarav-Nagar/QINN-FinRL-Evaluation/issues/4)
- [Add rolling or expanding-window
  evaluation](https://github.com/Aarav-Nagar/QINN-FinRL-Evaluation/issues/5)

The expanded SecureFinAI short-paper work is governed by the
[deliverables tracker](docs/PAPER_DELIVERABLES.md) and the prespecified
[experiment protocol](docs/EXPERIMENT_PROTOCOL.md). Daily milestones,
verification, and any protocol deviations are retained in the
[research log](docs/DAILY_LOG.md). Post-pilot choices are preserved in the
[experiment decision log](docs/DECISIONS.md).

The first expanded-study result is the matched
[PPO training-budget pilot](results/pilots/2026-07-27/), which selected 20,000
steps for the final ten-seed evaluation while retaining the absence of
established convergence as a limitation.

## Review guide

For a concise review of the project:

1. Download the
   [reviewer release](https://github.com/Aarav-Nagar/QINN-FinRL-Evaluation/releases/tag/v1.0-atl-evaluation),
   which contains the final PDF and editable Word report.
2. Read the
   [technical report (PDF)](docs/Aarav_Nagar_ANN_MPS_FinRL_Technical_Report.pdf).
3. Review the exact configuration in
   [`results/run_manifest.json`](results/run_manifest.json).
4. See the saved-output guide in [`results/README.md`](results/README.md).
5. Inspect [`run_experiment.py`](run_experiment.py) and
   [`test_experiment.py`](test_experiment.py) for the implementation and
   integrity checks.

An editable
[Word version of the report](docs/Aarav_Nagar_ANN_MPS_FinRL_Technical_Report.docx)
is also included.

## Research question

When the data, prediction target, parameter count, PPO configuration,
transaction costs, and random seeds are held constant, does an MPS signal
improve out-of-sample FinRL trading performance or robustness relative to an
ANN signal?

## Experimental protocol

Three PPO conditions were compared:

| Condition | State supplied to PPO | State dimension |
|---|---|---:|
| Base FinRL | Cash, prices, holdings, and 10 indicators per stock | 181 |
| ANN signal | Base state plus one frozen ANN prediction per stock | 196 |
| MPS signal | Base state plus one frozen MPS prediction per stock | 196 |

For the two signal agents, the encoder generated one standardized next-day
return prediction for each of 15 stocks. The resulting 15-value vector was
appended as the final indicator block in the PPO state. The encoder remained
frozen during PPO training.

The ANN and MPS models:

- used the same 13 market inputs and next-day return target;
- contained exactly 369 trainable parameters each;
- were fit on 2013-2017 data and validated on 2018; and
- were evaluated using PPO policies trained on 2013-2018 and tested on the
  unseen 2019-2023 period.

Each PPO condition used seeds 0, 1, and 2, a 5,000-step training budget, a
$1,000,000 initial portfolio, and a 0.10% transaction cost on every executed
buy and sell.

## Main results

The MPS model achieved slightly lower prediction error and higher directional
accuracy, but this did not produce better trading performance.

| Condition | Mean Sharpe | Total return | Maximum drawdown | Annualized turnover |
|---|---:|---:|---:|---:|
| Base FinRL | 0.559 | 73.24% | -40.46% | 26.43% |
| ANN signal | 0.735 | 104.66% | -27.55% | 27.32% |
| MPS signal | 0.694 | 94.27% | -26.79% | 24.14% |
| Equal-weight benchmark | 1.115 | 218.25% | -28.54% | 20.06% |

MPS trailed ANN in all three paired PPO seeds. The annualized mean-return
difference, MPS minus ANN, was -0.92 percentage points. A moving
20-trading-day block bootstrap produced a 95% interval from -4.63 to +2.63
percentage points, which includes zero.

The supported conclusion is limited: under this configuration and historical
split, the MPS signal did not improve trading performance relative to the ANN
signal. This is not a general conclusion about tensor-network methods.

## Reproduce the reference run

Use Python 3.12 and install the exact artifact-verification environment:

```powershell
python -m pip install -r requirements-lock.txt
python run_experiment.py `
  --data-dir .cache\data `
  --finrl-dir .cache\FinRL `
  --output-dir results `
  --timesteps 5000 `
  --seeds 0 1 2 `
  --bond-dimension 4 `
  --encoder-device auto
python -m pytest -q test_experiment.py test_smoke_matrix.py
```

The pipeline downloads and checksum-verifies the processed Nasdaq data, checks
out the recorded FinRL commit, trains the ANN and MPS encoders, runs the PPO
conditions, and regenerates the metrics and figures.

Additional reproduction details are documented in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Repository structure

```text
.
|-- README.md
|-- REPRODUCIBILITY.md
|-- run_experiment.py
|-- test_experiment.py
|-- requirements.txt
|-- requirements-lock.txt
|-- docs/
|   |-- Aarav_Nagar_ANN_MPS_FinRL_Technical_Report.pdf
|   |-- Aarav_Nagar_ANN_MPS_FinRL_Technical_Report.docx
|   |-- technical_report.md
|   `-- source_notes.md
`-- results/
    |-- README.md
    |-- run_manifest.json
    |-- signal_metrics.csv
    |-- ppo_backtest_metrics.csv
    |-- annual_period_metrics.csv
    |-- ann_vs_mps_block_bootstrap.csv
    `-- figures/
```

## Limitations

- The MPS is a classical tensor-network simulation; no quantum hardware was
  used.
- The experiment uses one historical split and a 15-stock Nasdaq subset.
- Three PPO seeds reveal seed sensitivity but do not support broad statistical
  generalization.
- The 5,000-step PPO budget may leave policies undertrained.
- The cost model includes a fixed transaction fee but not spread, slippage,
  market impact, taxes, liquidity limits, or borrow costs.
- Neither signal agent beat the equal-weight benchmark over the full test
  period.

This repository is for research and education and is not financial advice.
