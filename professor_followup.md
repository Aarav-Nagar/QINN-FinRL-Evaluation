**Subject:** Reproducible ANN vs. MPS FinRL evaluation case

Dear Professor Liu,

Thank you for your guidance. I strengthened the current ANN-versus-MPS FinRL
study before moving toward a tensor-network policy, as you recommended.

The completed evaluation now includes:

- Three matched PPO random seeds
- A paired 20-trading-day block-bootstrap confidence interval
- Gross turnover calculated from executed trades
- Transaction costs and cost-to-notional reconciliation
- Maximum drawdown and other risk-adjusted metrics
- Calendar-year results for 2019 through 2023
- The exact construction of the Base, ANN-signal, and MPS-signal PPO states
- A pinned FinRL commit, checksum-verified data, tests, and a run manifest

The main result remained negative but informative. The MPS encoder achieved
slightly lower prediction error and higher directional accuracy, but the MPS
PPO agent did not outperform the ANN agent. Mean out-of-sample Sharpe was 0.694
for MPS versus 0.735 for ANN, and MPS trailed ANN in all three paired seeds.
The annualized MPS-minus-ANN mean-return difference was -0.92 percentage
points, with a 95% block-bootstrap interval of [-4.63, +2.63] points.

Mean annualized turnover was 24.14% for MPS and 27.32% for ANN. With a 0.10%
cost on each buy and sell, mean five-year transaction costs were $1,253.61 for
MPS and $1,429.65 for ANN from a $1,000,000 starting portfolio. Neither agent
beat the equal-weight buy-and-hold benchmark.

Repository:
https://github.com/Aarav-Nagar/QINN-FinRL-Evaluation

The repository includes the reproducible pipeline, verified result files, and
the attached short technical report describing the protocol, results, and
limitations.

I would be very grateful for your feedback on whether this could serve as a
useful reproducible ATL evaluation case. I would also appreciate the
opportunity to help with any focused part of ATL or FinRL, even through a small
supporting contribution such as reproducing an experiment, preparing data,
running benchmarks, or documenting results. My main goal is to learn how
rigorous research is conducted while contributing something genuinely useful.

Thank you again for your time and guidance.

Best,
Aarav Nagar
