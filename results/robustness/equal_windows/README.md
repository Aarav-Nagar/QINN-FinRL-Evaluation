# Equal-Length Temporal Robustness Evidence

This directory contains the exploratory three-window panel frozen in
`docs/EQUAL_WINDOW_PROTOCOL.md` before the two new outcomes were viewed. It
does not replace the corrected 2019--2023 primary evaluation or the original
2017--2018 shifted evaluation.

## Design

Each cell evaluates a separate two-calendar-year portfolio period after the
same-length four-calendar-year PPO training period. Encoder fitting uses the
first three training years; the fourth is reserved for validation and early
stopping. Every cell uses Base FinRL, ANN signal, and dimension-2 classical MPS
signal conditions; 20,000 PPO steps; matched seeds 0--9; 0.10% executed
transaction costs; and CPU execution.

The 2017--2018 cell is reused from `../shifted/` because it already matches
the equal-window protocol exactly. The `2019-2020/` and `2021-2022/`
directories contain the two newly completed 30-endpoint run bundles.

## Main exploratory result

| Evaluation | ANN Sharpe | MPS Sharpe | Paired MPS-ANN | Seed wins | Seed-bootstrap 95% interval |
|---|---:|---:|---:|---:|---:|
| 2017--2018 | 0.627 | 0.714 | +0.086 | 9/10 | [0.001, 0.195] |
| 2019--2020 | 0.870 | 0.963 | +0.092 | 7/10 | [-0.008, 0.197] |
| 2021--2022 | 0.038 | 0.177 | +0.140 | 6/10 | [-0.068, 0.338] |

Mean MPS-minus-ANN Sharpe is positive in all three equal-length windows.
However, the two newer intervals include zero, seed variation is largest in
2021--2022, and the frozen five-year 2019--2023 primary evaluation has the
opposite mean sign. The honest conclusion is sensitivity to evaluation
horizon and training-window design, not stable MPS superiority.

## Requested secondary differences

All entries are paired MPS minus ANN means across the ten seeds.

| Evaluation | Annual return | Max drawdown | Ann. turnover | Modeled cost |
|---|---:|---:|---:|---:|
| 2017--2018 | +0.0225 | -0.0068 | +0.0227 | +$41.23 |
| 2019--2020 | +0.0307 | +0.0052 | +0.0036 | +$1.98 |
| 2021--2022 | +0.0405 | +0.0676 | -0.0260 | -$42.20 |

For maximum drawdown, a positive difference means the MPS drawdown was less
severe. For turnover and cost, a negative difference means less trading or
lower modeled cost. Descriptive seed-bootstrap intervals for every metric are
in `window_paired_metric_summary.csv`; most secondary intervals include zero.

## Prediction quality

MPS has slightly lower test MSE and MAE in 2017--2018 and 2019--2020, but
higher MSE and MAE in 2021--2022. Its information coefficient is higher in the
first two windows and lower in the last. Despite that reversal in prediction
quality, mean MPS trading Sharpe remains higher in all three cells. This
supports the paper's central failure-analysis point: prediction metrics do not
map reliably or monotonically to downstream PPO outcomes.

## Artifact map

- `window_condition_summary.csv`: Base, ANN, and MPS means and standard
  deviations for all requested portfolio metrics.
- `window_paired_seed_effects.csv`: every MPS-minus-ANN seed difference.
- `window_paired_metric_summary.csv`: paired means, dispersion, win counts,
  and deterministic descriptive intervals.
- `window_signal_quality.csv`: ANN/MPS prediction metrics and differences.
- `equal_window_inference.json`: bounded cross-window sign description.
- `equal_window_manifest.json`: protocol, configuration, and SHA-256
  provenance.
- `../../figures/equal_window_paired_effect.{png,pdf}`: seed-level visual.

## Reproduction

The full matrix can be resumed without overwriting matching completed runs:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_equal_window_robustness.ps1
```

Generate the cross-window tables with
`scripts/summarize_equal_windows.py`, then generate the figure with
`scripts/plot_equal_windows.py`. The focused safeguards are in
`test_equal_window_summary.py`, `test_equal_window_figure.py`, and
`test_equal_window_evidence.py`.

These are classical tensor-network experiments on one 15-stock universe. The
three historical cells are not independent replications, do not identify a
market-regime cause, and do not establish deployable profitability.
