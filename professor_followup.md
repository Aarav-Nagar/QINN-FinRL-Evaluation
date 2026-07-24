**Subject:** Follow-up: ANN and MPS signals in FinRL

Dear Professor Liu,

Thank you for the specific feedback. I went back to the experiment and added
the checks you recommended:

- three matched PPO seeds and a paired confidence interval;
- turnover, transaction costs, and maximum drawdown;
- calendar-year results for 2019-2023; and
- the exact way each prediction signal is added to the PPO state.

The main result did not change. The MPS model had slightly better prediction
MSE and directional accuracy, but that did not lead to better trading results.
Its mean out-of-sample Sharpe was 0.694, compared with 0.735 for the ANN, and
it trailed the ANN in all three seeds. The 95% block-bootstrap interval for the
MPS-minus-ANN annualized return difference was -4.63 to +2.63 percentage
points, so I cannot claim a reliable advantage for either signal from this
run.

I put the code, result files, and a short technical note here:
https://github.com/Aarav-Nagar/QINN-FinRL-Evaluation

I would appreciate your feedback on whether the protocol is reasonable and
whether this could be useful as a small ATL evaluation case. If there is a
focused task I could help with next, even a small one involving
reproducibility, data preparation, benchmark testing, or documentation, I
would be glad to contribute and learn from your guidance.

Thank you again for your time.

Best,
Aarav Nagar
