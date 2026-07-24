**Subject:** Follow-up: ANN and MPS signal evaluation in FinRL

Dear Professor Liu,

Thank you for your feedback on the initial experiment. I have now completed
the additional analysis you recommended.

The revised study includes:

- three matched PPO random seeds and a paired block-bootstrap confidence
  interval;
- turnover, transaction costs, and maximum drawdown;
- results across each calendar year from 2019 through 2023; and
- the exact construction of the Base, ANN-signal, and MPS-signal PPO states.

The main finding remained the same. Although the MPS model achieved slightly
lower prediction error and higher directional accuracy, the ANN-based trading
agent performed better on the primary trading measures. Mean out-of-sample
Sharpe was 0.735 for ANN and 0.694 for MPS, with MPS trailing ANN in all three
paired seeds. The 95% block-bootstrap interval for the MPS-minus-ANN
annualized return difference was -4.63 to +2.63 percentage points, so the
experiment does not establish a reliable MPS advantage.

I have included the complete code, result files, and technical report in the
repository below:

https://github.com/Aarav-Nagar/QINN-FinRL-Evaluation

I would be grateful for your feedback on the experimental protocol and on
whether this could serve as an ATL evaluation case. I would also welcome the
opportunity to assist with a focused part of ATL or FinRL, including a small
supporting task involving reproducibility, data preparation, benchmark
testing, or documentation.

Thank you again for your time and guidance.

Best,
Aarav Nagar
