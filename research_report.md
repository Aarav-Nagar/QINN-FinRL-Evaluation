# Testing ANN and MPS Signals in FinRL

**Short technical note**  
**Aarav Nagar | July 2026**

## Question

My earlier project found that a quantum-inspired model could perform slightly
better than an ANN on some stock-prediction metrics. Professor Liu suggested
testing whether that difference still mattered when the prediction became
part of an autonomous trading agent.

I therefore asked: if the data, model size, PPO setup, transaction costs, and
random seeds are held constant, does an MPS prediction signal improve FinRL
trading performance compared with an ANN signal?

## What I did

I used 15 stocks from the processed Nasdaq dataset used in the FinRL Contest
2025 workflow. I fit the two prediction models on 2013-2017 data, used 2018
for validation, trained PPO on 2013-2018, and tested the trading agents on the
unseen 2019-2023 period.

The three trading conditions were:

| Condition | State given to PPO | State size |
|---|---|---:|
| Base FinRL | Cash, prices, holdings, and 10 indicators per stock | 181 |
| ANN signal | Base state plus one ANN prediction per stock | 196 |
| MPS signal | Base state plus one MPS prediction per stock | 196 |

The base state contains 1 cash value, 15 prices, 15 holdings, and 10 indicators
for each of 15 stocks. For the ANN and MPS conditions, I appended the 15 frozen
next-day return predictions as one additional indicator block. The signal
model was not updated while PPO was training.

Both prediction models used the same 13 inputs and had exactly 369 trainable
parameters. The ANN used 13-16-8-1 fully connected layers. The MPS used a
trigonometric feature map with bond dimension 4. This MPS is a classical
tensor-network model; no quantum hardware was used.

For each PPO condition, I used seeds 0, 1, and 2, a 5,000-step training budget,
a $1,000,000 starting portfolio, and a 0.10% transaction cost on every buy and
sell. Turnover was calculated from the trades that were actually executed:

`daily turnover = traded notional / portfolio value at the start of the step`

The backtest does not include spread, slippage, market impact, or taxes.

## Results

On the prediction task, the MPS had slightly lower error and slightly higher
directional accuracy. Its information coefficient, however, was slightly
negative.

| Model | MSE | Directional accuracy | Information coefficient |
|---|---:|---:|---:|
| ANN | 1.5101 | 50.25% | +0.0119 |
| MPS | 1.5034 | 51.06% | -0.0060 |

The small prediction improvement did not carry over to trading.

| Agent | Total return | Sharpe | Max drawdown | Annualized turnover | Total cost |
|---|---:|---:|---:|---:|---:|
| Base FinRL | 73.24% | 0.559 | -40.46% | 26.43% | $1,375.38 |
| ANN signal | 104.66% | 0.735 | -27.55% | 27.32% | $1,429.65 |
| MPS signal | 94.27% | 0.694 | -26.79% | 24.14% | $1,253.61 |
| Equal-weight | 218.25% | 1.115 | -28.54% | 20.06% | $1,000.00 |

These PPO values are averages across the three seeds. MPS had a lower Sharpe
than ANN in every paired seed: 0.759 versus 0.790 for seed 0, 0.726 versus
0.812 for seed 1, and 0.596 versus 0.602 for seed 2.

I also averaged the daily returns across seeds and used a 20-trading-day block
bootstrap with 2,000 samples. The MPS-minus-ANN annualized return difference
was -0.92 percentage points, with a 95% interval from -4.63 to +2.63 points.
Because the interval includes zero, this run does not show a reliable
difference between the two signal agents.

Results also changed by year. MPS beat ANN in 2020 and 2023, while ANN was
better in 2019, 2021, and 2022. Neither PPO signal agent beat the equal-weight
benchmark over the full test period.

## What I learned

The result supports a limited conclusion: in this setup, the MPS signal did
not improve trading performance over the equally sized ANN signal. It did
slightly better on two prediction metrics, but the ANN produced a higher
Sharpe in all three trading runs. This is the prediction-decision gap that
motivated the experiment.

I do not think this result rules out tensor-network methods. The study uses
only one time split, three PPO seeds, one MPS bond dimension, and a relatively
short 5,000-step training budget. Trading activity also became very low later
in the test period, which suggests that some policies settled into nearly
static positions.

Before testing a tensor-network policy architecture, I would first repeat this
study with more seeds, longer PPO training, and rolling retraining periods. A
later policy-level comparison would then have a stronger baseline.

## Reproduction

The repository includes the full pipeline, tests, daily equity curves,
per-seed and yearly metrics, bootstrap output, and a run manifest containing
the pinned data checksum and FinRL commit.

## References

1. Biamonte, J. et al. “Quantum Machine Learning.” *Nature* 549, 195-202
   (2017). https://www.nature.com/articles/nature23474
2. Liu, X.-Y. and Fang, Y. “Quantum Tensor Networks for Variational
   Reinforcement Learning.” NeurIPS 2020 Quantum Tensor Networks in Machine
   Learning Workshop.
3. FinRL Contest 2025, Task 1: FinRL-DeepSeek for Stock Trading.
   https://finrl-contest.readthedocs.io/en/latest/finrl2025/task1.html
4. FinRL repository, pinned in this experiment to commit
   `2334a5fe6d30629157f13c3b0319e1637e15e123`.
