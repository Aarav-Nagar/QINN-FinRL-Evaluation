# ANN vs. Quantum-Inspired MPS Signals in a FinRL PPO Agent

**A reproducible evaluation of the prediction-decision gap in financial AI**

**Author:** Aarav Nagar

**Evaluation date:** July 24, 2026

**Scope:** Classical simulation on historical data; no quantum hardware; not financial advice

## Executive summary

This study tests whether a quantum-inspired matrix-product-state (MPS)
predictor improves sequential trading when its output is added to the state of
a FinRL Proximal Policy Optimization (PPO) agent.

The result is negative but useful: the MPS encoder produced slightly better
prediction error and directional accuracy than a parameter-matched artificial
neural network (ANN), yet the MPS trading agent did not outperform the ANN
agent.

- Mean out-of-sample Sharpe across seeds 0, 1, and 2 was **0.694 for
  QINN-MPS** and **0.735 for ANN**.
- QINN-MPS had lower Sharpe than ANN in **all three paired seeds**.
- The annualized mean-return difference, MPS minus ANN, was **-0.92 percentage
  points**. A paired 20-trading-day block bootstrap gave a **95% interval of
  [-4.63, +2.63] percentage points**, so the difference is not statistically
  established.
- Mean annualized gross turnover was **24.14% for QINN-MPS**, **27.32% for
  ANN**, and **26.43% for Base FinRL**.
- Mean transaction costs over the five-year test were **$1,253.61 for
  QINN-MPS**, **$1,429.65 for ANN**, and **$1,375.38 for Base FinRL**, from a
  $1,000,000 initial portfolio and a 0.10% cost on every buy and sell.
- Neither learned agent beat the equal-weight buy-and-hold benchmark, whose
  Sharpe was **1.115**.

The evidence supports the prediction-decision gap raised by Professor
Xiao-Yang Liu: a small improvement in predictive metrics does not necessarily
produce a better autonomous trading policy.

## Research question

> When all other inputs, dates, model sizes, transaction costs, PPO settings,
> and random seeds are held constant, does a quantum-inspired MPS prediction
> signal improve FinRL trading performance or robustness relative to an ANN
> signal?

## Experimental protocol

### Conditions

| Condition | State supplied to PPO | Dimension |
|---|---|---:|
| Base FinRL | Cash, prices, holdings, and 10 market indicators per asset | 181 |
| ANN signal | Base state plus one frozen ANN prediction per asset | 196 |
| QINN-MPS signal | Base state plus one frozen MPS prediction per asset | 196 |

Every PPO condition used the same 15 stocks, data, policy architecture,
training budget, costs, and seeds.

### Data and chronology

The experiment uses a checksum-pinned processed Nasdaq dataset aligned with
the FinRL Contest 2025 stock-trading workflow. The selected universe is:

`AAPL, AMD, AMGN, AMZN, COST, FANG, GILD, HON, INTC, MSFT, NFLX, NVDA, PEP,
SBUX, XEL`.

The chronology is fixed before evaluation:

- **2013-2017:** fit the ANN and MPS encoders.
- **2018:** early stopping and encoder validation.
- **2013-2018:** train the PPO policies.
- **2019-2023:** locked out-of-sample trading evaluation.

No 2019-2023 return target is used to fit either encoder or PPO policy. The
downloaded files are checksum-pinned, and the environment is pinned to FinRL
commit `2334a5fe6d30629157f13c3b0319e1637e15e123`.

### Parameter-matched signal models

Both encoders receive the same 13 current-market inputs and predict the same
standardized next-day return:

- 1-, 5-, and 20-day returns;
- price relative to 30- and 60-day moving averages;
- scaled MACD;
- Bollinger position and width;
- scaled RSI, CCI, and directional index;
- five-day log-volume change;
- scaled VIX.

The ANN uses `13 -> 16 -> 8 -> 1` fully connected layers with tanh
activations. The MPS uses a trigonometric local feature map and bond dimension
4. Each model has exactly **369 trainable parameters**. The MPS is a classical
tensor-network simulation; it does not use qubits and does not demonstrate
quantum advantage.

### Exact signal entry into the PPO state

For 15 assets and 10 standard indicators, the Base FinRL state is:

`1 cash + 15 prices + 15 holdings + (10 indicators x 15 assets) = 181 values`.

For either signal condition, the frozen encoder produces one standardized,
clipped next-day-return prediction for each asset. That 15-value vector is
appended as the final per-asset indicator block:

`1 cash + 15 prices + 15 holdings + (11 indicators x 15 assets) = 196 values`.

The encoder is not updated during PPO training. The ANN and MPS variants differ
only in this final signal block.

### PPO and trading assumptions

- Algorithm: Stable-Baselines3 PPO with a two-layer 64-unit policy and value
  network.
- Training budget: 5,000 environment steps and three optimization epochs per
  update.
- Seeds: 0, 1, and 2 for each PPO condition.
- Initial portfolio: $1,000,000.
- Transaction cost: 0.10% of executed notional on every buy and sell.
- Maximum order: 100 shares per asset per step.
- Reward scaling: `1e-4`.
- Test inference: deterministic policy action.

The 5,000-step budget is a scoped CPU evaluation, not evidence that every
policy converged.

### Turnover definition

Turnover is calculated from executed, not requested, trades:

`daily turnover = sum(|executed shares| x pre-trade price) / beginning-of-step portfolio value`.

This is gross one-way turnover: buys and sells both contribute to traded
notional. Cumulative turnover is the sum of daily turnover, and annualized
turnover is mean daily turnover multiplied by 252.

The environment also records total executed notional. With a constant 0.10%
fee, the reported transaction cost reconciles exactly to:

`total cost = gross executed notional x 0.001`.

The backtest does not model bid-ask spread, slippage, market impact, taxes, or
borrow costs.

## Results

### Prediction before reinforcement learning

Metrics use untouched 2019-2023 next-day targets. MSE and MAE are measured
against the standardized return target.

| Encoder | Parameters | MSE | MAE | Directional accuracy | Information coefficient |
|---|---:|---:|---:|---:|---:|
| ANN | 369 | 1.5101 | 0.7994 | 50.25% | +0.0119 |
| QINN-MPS | 369 | **1.5034** | **0.7967** | **51.06%** | -0.0060 |

The MPS result is mixed: error and directional accuracy are slightly better,
but its information coefficient is slightly negative.

### Portfolio performance

PPO values are means across three seeds. The passive benchmark has no seed.

| Condition | Total return | Annual return | Sharpe | Max drawdown | Annualized turnover | Total cost |
|---|---:|---:|---:|---:|---:|---:|
| Base FinRL | 73.24% | 11.48% | 0.559 | -40.46% | 26.43% | $1,375.38 |
| ANN signal | **104.66%** | **15.26%** | **0.735** | -27.55% | 27.32% | $1,429.65 |
| QINN-MPS signal | 94.27% | 14.14% | 0.694 | **-26.79%** | **24.14%** | **$1,253.61** |
| Equal-weight buy-and-hold | **218.25%** | **26.15%** | **1.115** | -28.54% | 20.06% | $1,000.00 |

QINN-MPS traded less and had a slightly shallower mean drawdown than ANN, but
it also produced lower return and risk-adjusted performance. Lower turnover is
not enough to establish superior decision quality.

### Seed-level Sharpe

| Seed | Base FinRL | ANN signal | QINN-MPS signal | MPS minus ANN |
|---:|---:|---:|---:|---:|
| 0 | 0.436 | **0.790** | 0.759 | -0.032 |
| 1 | 0.633 | **0.812** | 0.726 | -0.086 |
| 2 | **0.608** | 0.602 | 0.596 | -0.006 |
| Mean | 0.559 | **0.735** | 0.694 | -0.041 |

QINN-MPS did not beat ANN in any paired seed. Both signal agents beat the Base
agent in seeds 0 and 1 and trailed it slightly in seed 2.

### Confidence intervals across seeds

These are descriptive two-sided 95% Student-t intervals across only three
seeds. They are intentionally labeled exploratory because `n = 3` produces
wide intervals.

| Condition | Mean Sharpe | Exploratory 95% interval | Mean annualized turnover | Exploratory 95% interval |
|---|---:|---:|---:|---:|
| Base FinRL | 0.559 | [0.292, 0.826] | 26.43% | [4.43%, 48.43%] |
| ANN signal | 0.735 | [0.449, 1.021] | 27.32% | [16.52%, 38.12%] |
| QINN-MPS signal | 0.694 | [0.480, 0.907] | 24.14% | [20.88%, 27.40%] |

The more relevant paired uncertainty analysis averages daily returns across
seeds and applies a moving 20-day block bootstrap with 2,000 samples:

- Annualized mean-return difference, MPS minus ANN: **-0.92 percentage points**
- 95% bootstrap interval: **[-4.63, +2.63] percentage points**
- Bootstrap probability that MPS exceeds ANN: **31.05%**

The interval includes zero, so this run does not establish a reliable
difference between the signal agents.

### Calendar-year market periods

Calendar years were declared as the period definition to avoid choosing
favorable regimes after observing results. Values below are mean returns across
the three PPO seeds.

| Year | Base FinRL | ANN signal | QINN-MPS signal | Equal-weight |
|---:|---:|---:|---:|---:|
| 2019 | 8.00% | **15.69%** | 14.85% | 43.52% |
| 2020 | **35.48%** | 25.85% | 26.28% | 36.13% |
| 2021 | 18.36% | **20.48%** | 18.70% | 34.88% |
| 2022 | -27.72% | **-9.24%** | -15.79% | -17.47% |
| 2023 | **38.40%** | 30.72% | 34.55% | 46.34% |

MPS beat ANN in 2020 and 2023, while ANN beat MPS in 2019, 2021, and 2022.
The year-by-year result therefore does not show a stable MPS advantage.

Trading activity was concentrated early in the test. Mean annualized turnover
fell from approximately 103% for ANN and 100% for MPS in 2019 to less than 1%
for both by 2023. This behavior should be retested under longer PPO training
and rolling retraining.

## Interpretation

This experiment supports a narrow conclusion:

> Under this fixed dataset, parameter budget, PPO configuration, three seeds,
> and historical split, a quantum-inspired MPS signal did not improve FinRL
> trading performance over an equally sized ANN signal.

The result does not show that all tensor-network methods are ineffective. It
does show why prediction and decision quality must be evaluated separately.
MPS achieved slightly better prediction MSE and directional accuracy, but ANN
produced higher Sharpe in all three paired trading runs.

The passive benchmark is also important. None of the PPO agents beat
equal-weight buy-and-hold over this Nasdaq-heavy period, so the study does not
claim market-beating alpha.

## Limitations

- Only one 2013-2018/2019-2023 chronological split was tested.
- Three PPO seeds reveal policy instability but are insufficient for broad
  statistical generalization.
- The 5,000-step training budget may leave policies undertrained.
- The universe is a 15-stock Nasdaq subset and is not sector-neutral.
- The MPS is a classical quantum-inspired model, not a quantum computation.
- The ANN and MPS are parameter-matched but have different inductive biases.
- The cost model includes a fixed 0.10% fee but excludes spread, slippage,
  impact, taxes, liquidity limits, and borrow constraints.
- Calendar years provide a transparent robustness check but do not replace
  rolling or expanding-window evaluation.
- The declining turnover suggests that some policies became close to static
  allocations later in the test.

## Recommended next contribution

Before making a strong claim about tensor-network policies, the most useful ATL
or FinRL continuation would be:

1. Increase PPO training to at least 50,000 steps and use 10 or more seeds.
2. Add rolling or expanding-window retraining.
3. Predeclare and compare several MPS bond dimensions.
4. Add exposure, concentration, slippage, and crisis-window diagnostics.
5. Only then compare a tensor-network policy head against the external-signal
   design used here.

This current experiment is suitable as a reproducible ATL evaluation case
because it includes a controlled ablation, a negative result, seed-level
variation, confidence intervals, turnover, transaction costs, drawdown, exact
state construction, and auditable limitations.

## Reproducibility artifacts

- `run_experiment.py`: complete data, encoder, FinRL PPO, and evaluation
  pipeline.
- `test_experiment.py`: integrity tests for parameter matching, MPS feature
  normalization, state dimensions, turnover, yearly metrics, and confidence
  summaries.
- `results/signal_metrics.csv`: encoder prediction metrics.
- `results/ppo_backtest_metrics.csv`: full-period per-seed portfolio metrics.
- `results/equity_curves.csv`: daily account values, returns, traded notional,
  turnover, and costs.
- `results/annual_period_metrics.csv`: calendar-year metrics by condition and
  seed.
- `results/condition_seed_summary.csv`: full-period means and exploratory
  seed-level confidence intervals.
- `results/annual_period_seed_summary.csv`: calendar-year seed summaries.
- `results/ann_vs_mps_block_bootstrap.csv`: paired uncertainty analysis.
- `results/run_manifest.json`: checksums, pinned commit, configuration, state
  design, turnover definition, and limitations.

## References

1. Biamonte, J. et al. [Quantum Machine Learning](https://www.nature.com/articles/nature23474).
   *Nature* 549, 195-202 (2017).
2. Liu, X.-Y. and Fang, Y. [Quantum Tensor Networks for Variational
   Reinforcement Learning](https://tensorworkshop.github.io/NeurIPS2020/accepted_papers/NIPS_2020_Workshop_Yiming%20%281%29.pdf).
   NeurIPS 2020 Quantum Tensor Networks in Machine Learning Workshop.
3. [NeurIPS 2020 Quantum Tensor Networks in Machine Learning Workshop:
   Accepted Papers](https://tensorworkshop.github.io/NeurIPS2020/accepted_papers.html).
4. [NeurIPS 2021 Quantum Tensor Networks in Machine Learning Workshop:
   Accepted Papers](https://tensorworkshop.github.io/NeurIPS2021/accepted_papers.html).
5. [FinRL Contest 2025 Task 1: FinRL-DeepSeek for Stock
   Trading](https://finrl-contest.readthedocs.io/en/latest/finrl2025/task1.html).
6. Wang, K. et al. [FinRL Contests: Benchmarking Data-driven Financial
   Reinforcement Learning Agents](https://arxiv.org/abs/2504.02281) (2025).
7. [FinRL](https://github.com/AI4Finance-Foundation/FinRL), pinned in this
   experiment to commit `2334a5fe6d30629157f13c3b0319e1637e15e123`.
