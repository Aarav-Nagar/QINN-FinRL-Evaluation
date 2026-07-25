# Evaluating ANN and Matrix-Product-State Signals in a FinRL PPO Agent

**Aarav Nagar | July 2026**

## Abstract

This study examines whether a quantum-inspired matrix-product-state (MPS)
model provides a more useful trading signal than a conventional artificial
neural network (ANN). The two models were matched in parameter count and were
trained on the same inputs and next-day return target. Their predictions were
then added separately to the state of a FinRL Proximal Policy Optimization
(PPO) trading agent. The MPS produced slightly lower prediction error and
higher directional accuracy, but the ANN-based agent achieved a higher mean
out-of-sample Sharpe ratio (0.735 compared with 0.694). MPS trailed ANN in all
three paired PPO seeds. A block-bootstrap confidence interval for the
annualized return difference included zero. Under this experimental setup,
the MPS signal did not improve trading performance. The result illustrates
that a small improvement in prediction metrics does not necessarily lead to a
better sequential trading policy.

## 1. Research Question

My earlier project compared quantum-inspired neural networks with hybrid
ensemble ANNs for stock-market prediction. One question remained after that
work: if one model predicts returns slightly better, will a trading agent
actually make better decisions when transaction costs and changing market
conditions are included?

Following Professor Xiao-Yang Liu's suggestion, I tested the prediction models
as state signals inside a FinRL benchmark. The question for this experiment
was:

*When the data, parameter count, PPO configuration, transaction costs, and
random seeds are held constant, does an MPS signal improve out-of-sample
trading performance relative to an ANN signal?*

## 2. Experimental Design

### 2.1 Data and time split

The experiment used 15 stocks from the processed Nasdaq dataset associated
with the FinRL Contest 2025 stock-trading workflow:

`AAPL, AMD, AMGN, AMZN, COST, FANG, GILD, HON, INTC, MSFT, NFLX, NVDA, PEP,
SBUX, XEL`.

The time split was fixed before evaluating the trading results:

- 2013-2017: fit the ANN and MPS prediction models;
- 2018: validate the prediction models and apply early stopping;
- 2013-2018: train the PPO policies; and
- 2019-2023: evaluate the policies out of sample.

No 2019-2023 return target was used to fit either prediction model or PPO
policy. The processed dataset was checksum-pinned, and the environment used
FinRL commit `2334a5fe6d30629157f13c3b0319e1637e15e123`.

### 2.2 Prediction models

Both models used the same 13 market features and predicted the same
standardized next-day return. The inputs included short- and medium-horizon
returns, moving-average ratios, MACD, Bollinger-band measures, RSI, CCI,
directional index, volume change, and VIX.

The ANN used fully connected layers of `13-16-8-1`. The MPS used a
trigonometric local feature map and bond dimension 4. Each model contained
exactly 369 trainable parameters. The MPS was simulated entirely on classical
hardware; this experiment did not use qubits or test quantum advantage.

### 2.3 PPO state construction

Three PPO conditions were compared.

**Table 1. State construction for the three PPO conditions**

| Condition | State supplied to PPO | Dimension |
|---|---|---:|
| Base FinRL | Cash, prices, holdings, and 10 indicators per stock | 181 |
| ANN signal | Base state plus one frozen ANN prediction per stock | 196 |
| MPS signal | Base state plus one frozen MPS prediction per stock | 196 |

For 15 stocks, the base state contained 1 cash value, 15 prices, 15 holdings,
and 10 indicator values for each stock:

`1 + 15 + 15 + (10 x 15) = 181 values`.

For the ANN and MPS conditions, the model produced one prediction per stock.
The resulting 15-value vector was appended as an additional indicator block:

`1 + 15 + 15 + (11 x 15) = 196 values`.

The prediction model was frozen during PPO training. The ANN and MPS agents
therefore differed only in the final 15 state values.

### 2.4 Trading setup

Each PPO condition used the same two-layer 64-unit policy and value network,
5,000 environment steps, three optimization epochs per update, and random
seeds 0, 1, and 2. The starting portfolio was $1,000,000. A transaction cost
of 0.10% was applied to every executed buy and sell.

Turnover was calculated from executed trades:

`daily turnover = gross executed notional / beginning-of-step portfolio value`.

The backtest did not model bid-ask spread, slippage, market impact, taxes,
liquidity limits, or borrow costs.

## 3. Results

### 3.1 Prediction results

The MPS achieved slightly lower mean-squared error and slightly higher
directional accuracy. Its information coefficient was slightly negative,
however, so the prediction evidence was mixed.

**Table 2. Out-of-sample prediction results, 2019-2023**

| Model | Parameters | MSE | Directional accuracy | Information coefficient |
|---|---:|---:|---:|---:|
| ANN | 369 | 1.5101 | 50.25% | +0.0119 |
| MPS | 369 | 1.5034 | 51.06% | -0.0060 |

### 3.2 Trading results

The small improvement in two prediction metrics did not carry over to the PPO
backtest.

**Table 3. Mean out-of-sample portfolio results**

| Condition | Total return | Sharpe | Maximum drawdown | Annualized turnover | Total cost |
|---|---:|---:|---:|---:|---:|
| Base FinRL | 73.24% | 0.559 | -40.46% | 26.43% | $1,375.38 |
| ANN signal | 104.66% | 0.735 | -27.55% | 27.32% | $1,429.65 |
| MPS signal | 94.27% | 0.694 | -26.79% | 24.14% | $1,253.61 |
| Equal-weight buy-and-hold | 218.25% | 1.115 | -28.54% | 20.06% | $1,000.00 |

The PPO values are means across seeds 0, 1, and 2. MPS produced a lower Sharpe
than ANN in every paired seed: 0.759 compared with 0.790 for seed 0, 0.726
compared with 0.812 for seed 1, and 0.596 compared with 0.602 for seed 2.
MPS traded less and had a slightly smaller mean drawdown, but it also produced
lower return and risk-adjusted performance.

![Mean equity curves for the three PPO conditions and the equal-weight
benchmark](results/figures/equity_curves.png)

**Figure 1. Mean out-of-sample equity curves. Shaded regions show variation
across PPO seeds.**

### 3.3 Uncertainty and performance across years

I averaged daily returns across the three seeds and applied a moving
20-trading-day block bootstrap with 2,000 samples. The annualized return
difference, MPS minus ANN, was -0.92 percentage points. The 95% interval was
-4.63 to +2.63 percentage points, and the bootstrap probability that MPS
exceeded ANN was 31.05%. Because the interval includes zero, the experiment
does not establish a reliable difference between the two signal agents.

The results also varied across calendar years. MPS outperformed ANN in 2020
and 2023, while ANN outperformed MPS in 2019, 2021, and 2022. Neither signal
agent beat the equal-weight benchmark over the full test period.

## 4. Discussion

The evidence supports a narrow conclusion: under this dataset, time split,
parameter budget, PPO configuration, and set of random seeds, the MPS signal
did not improve trading performance relative to the ANN signal.

The prediction and trading results should be considered separately. The MPS
had slightly better MSE and directional accuracy, but the ANN agent produced a
higher Sharpe ratio in each paired PPO run. This is consistent with the
prediction-decision gap that motivated the experiment. An agent acts on a
sequence of noisy predictions while also responding to its portfolio state,
costs, and earlier actions; a small improvement in one-step prediction is not
enough by itself to guarantee a better policy.

The passive benchmark is also important. None of the PPO agents beat the
equal-weight buy-and-hold portfolio over this Nasdaq-heavy test period. The
study therefore does not claim market-beating performance.

## 5. Limitations and Next Step

The main limitations are the use of a single historical split, only three PPO
seeds, one MPS bond dimension, and a 5,000-step PPO training budget. The
15-stock universe is small and not sector-neutral. In addition, trading
activity became very low later in the test period, suggesting that some
policies settled into nearly static positions.

Before testing a tensor-network policy architecture, I would strengthen this
baseline with more random seeds, a longer PPO training budget, and rolling or
expanding-window retraining. A policy-level comparison would then be easier to
interpret.

## 6. Reproducibility

The repository contains the experiment pipeline, integrity tests, daily equity
curves, per-seed metrics, calendar-year results, bootstrap output, and a run
manifest with the dataset checksum, FinRL commit, configuration, and known
limitations.

Repository: https://github.com/Aarav-Nagar/QINN-FinRL-Evaluation

## References

1. Biamonte, J., et al. “Quantum Machine Learning.” *Nature*, vol. 549, 2017,
   pp. 195-202. https://www.nature.com/articles/nature23474
2. Liu, X.-Y., and Y. Fang. “Quantum Tensor Networks for Variational
   Reinforcement Learning.” *NeurIPS 2020 Quantum Tensor Networks in Machine
   Learning Workshop*.
3. NeurIPS 2020 Quantum Tensor Networks in Machine Learning Workshop.
   https://tensorworkshop.github.io/NeurIPS2020/accepted_papers.html
4. NeurIPS 2021 Quantum Tensor Networks in Machine Learning Workshop.
   https://tensorworkshop.github.io/NeurIPS2021/accepted_papers.html
5. FinRL Contest 2025. “Task 1: FinRL-DeepSeek for Stock Trading.”
   https://finrl-contest.readthedocs.io/en/latest/finrl2025/task1.html
6. AI4Finance Foundation. *FinRL*. Commit
   `2334a5fe6d30629157f13c3b0319e1637e15e123`.
