# Post-hoc Market-State Trend Audit

## Bottom line

The most useful new trend is an upside/downside asymmetry. Across all three
non-overlapping two-year windows, the paired MPS-minus-ANN annualized
mean-return difference was negative on benchmark-down days and positive on
nonnegative benchmark days:

| Window | Benchmark-down days | Nonnegative days | Down-day offset of positive contribution |
|---|---:|---:|---:|
| 2017--2018 | -6.63 pp | +8.56 pp | 53.7% |
| 2019--2020 | -0.56 pp | +4.99 pp | 8.0% |
| 2021--2022 | -2.15 pp | +9.94 pp | 20.5% |

These figures annualize each state's mean daily MPS-minus-ANN return by
multiplying by 252; they are not compounded state returns. The contribution
column additionally weights each state effect by its observed day share and
shows how much the negative-day contribution offsets the nonnegative-day
contribution. The exact reconciled full-window differences are +2.34, +2.68,
and +4.05 percentage points.

Every direction-state seed-bootstrap interval includes zero. The finding is a
cross-window descriptive pattern, not a population-level significance claim.
It suggests that the positive equal-window aggregates came from stronger
relative upside capture rather than protection on falling-market days.

## Other complete trends

- **Return tails:** the top benchmark-return decile favored MPS in all three
  windows (+28.77, +11.20, and +10.00 pp), but intervals were wide. The bottom
  decile was unfavorable in 2017--2018 and 2019--2020, then favorable in
  2021--2022 (-19.47, -12.85, and +16.95 pp). Tail behavior is not stable
  enough for a stronger claim.
- **Volatility:** both high-volatility and low-volatility states had positive
  mean differences in every window. High-volatility effects were +2.33,
  +3.61, and +0.93 pp; low-volatility effects were +2.54, +1.93, and +7.59
  pp. The pattern is therefore not specifically a high-volatility effect.
- **Benchmark drawdown:** both at/new-peak and below-peak states had positive
  mean differences in every window, but the magnitude varied sharply. This
  does not overturn the negative benchmark-down-day pattern because a
  drawdown state can contain both rising and falling days.
- **Calendar years:** mean MPS-minus-ANN return and Sharpe were positive in
  five of six years. Only 2018 had a positive return and Sharpe interval that
  excluded zero; 2017 Sharpe was negative with an interval excluding zero.
  The other yearly intervals included zero, reinforcing year sensitivity.
- **Downside deviation:** MPS had higher downside deviation in 2017--2018,
  roughly similar downside deviation in 2019--2020, and lower downside
  deviation in 2021--2022. There is no stable downside-risk advantage.
- **Trading activity:** annual turnover and modeled-cost differences changed
  sign across years. Only 2017 showed a clearly positive interval for both,
  so a consistent activity/cost mechanism is not supported.

## Evidence inventory

| Artifact | Purpose |
|---|---|
| `market_state_seed_effects.csv` | 270 window/state/seed conditional effects |
| `market_state_summary.csv` | 27 state cells with paired-seed intervals |
| `market_state_cross_window.csv` | equal-weight cross-window sign audit |
| `direction_contribution.csv` | exact direction-state contribution reconciliation |
| `calendar_year_paired_metrics.csv` | 300 year/seed/metric ANN--MPS comparisons |
| `calendar_year_summary.csv` | 30 year/metric summaries and intervals |
| `market_state_inference.json` | bounded machine-readable interpretation |
| `market_state_manifest.json` | source/output hashes and row counts |

The controlling design is
[`docs/MARKET_STATE_TREND_PROTOCOL.md`](../../../docs/MARKET_STATE_TREND_PROTOCOL.md).
It was committed before the outcomes were calculated.

## Reproduce

From the repository root:

```powershell
python scripts\summarize_market_states.py `
  --window-2017-2018 results\robustness\shifted `
  --window-2019-2020 results\robustness\equal_windows\2019-2020 `
  --window-2021-2022 results\robustness\equal_windows\2021-2022 `
  --protocol docs\MARKET_STATE_TREND_PROTOCOL.md `
  --output-dir .cache\reproduced-market-states
python -m pytest -q test_market_state_trends.py
```

## Interpretation boundary

This is a post-hoc analysis of previously frozen curves. Market states overlap,
the three windows are not a large sample of regimes, and daily observations
are serially dependent. The intervals describe PPO-seed dispersion
conditional on each historical window; they do not quantify historical-sample
uncertainty. No causal regime claim, multiple-testing claim, quantum advantage,
or model-selection decision is supported.
