# Post-hoc Market-State Trend Protocol

Frozen before outcome calculation: 2026-08-02

## Purpose and status

This audit asks whether the paired MPS-minus-ANN trading difference is
concentrated in observable market states even though the aggregate results do
not establish a stable winner. It is an exploratory failure-analysis extension,
not a new primary experiment, a model-selection exercise, or a causal regime
test.

The corrected 2019--2023 primary result, the three equal-length window means,
and the nested-horizon results were known before this protocol. No market-state
result had been calculated when the definitions below were frozen.

## Frozen evidence

The audit uses only the saved daily equity curves from the three complete,
non-overlapping two-year evaluation windows:

1. `results/robustness/shifted/equity_curves.csv` (2017--2018);
2. `results/robustness/equal_windows/2019-2020/equity_curves.csv`; and
3. `results/robustness/equal_windows/2021-2022/equity_curves.csv`.

Every window must contain ten matched PPO seeds for `ANN signal` and
`QINN-MPS signal` and one saved `Equal-weight buy-and-hold` benchmark on an
identical daily grid. The benchmark defines market states independently of
either learned representation. No encoder or PPO policy is retrained.

## Prespecified questions

The analysis reports all of the following views:

1. **Calendar year:** each of 2017 through 2022 separately.
2. **Benchmark direction:** negative benchmark-return days versus nonnegative
   benchmark-return days.
3. **Benchmark volatility:** low versus high trailing 20-session annualized
   benchmark volatility, split at the within-window median after excluding the
   first 19 observations.
4. **Benchmark drawdown:** benchmark account value below its prior running
   maximum versus at or above that maximum.
5. **Benchmark return tails:** bottom decile, middle 80%, and top decile of
   benchmark daily returns, using within-window empirical quantiles.

Thresholds are fixed from the benchmark only. No alternative threshold,
subperiod, state intersection, asset subset, or outcome-driven regrouping will
be introduced after results are seen.

## Metrics

For every window, state, and PPO seed, calculate:

- number of included observations;
- ANN mean daily return;
- MPS mean daily return;
- paired MPS-minus-ANN mean daily return;
- the same paired mean multiplied by 252, labeled an annualized mean-return
  difference rather than a compounded return;
- ANN downside deviation and MPS downside deviation on the included days; and
- the paired MPS-minus-ANN downside-deviation difference.

For each window-state cell, report:

- the mean paired effect across the ten seeds;
- the number of seeds with a strictly positive paired effect;
- a deterministic 95% percentile bootstrap interval over the ten paired seed
  effects using seed `20260802` and 10,000 resamples; and
- whether the sign of the mean effect is positive, negative, or exactly zero.

Calendar-year output additionally reports paired annual return, Sharpe,
maximum drawdown, annualized turnover, and modeled-cost differences using the
already saved `annual_period_metrics.csv` tables. Those values must reproduce
a direct rescore from the daily curves within numerical tolerance.

## Cross-window summaries

For each non-calendar state label, the audit reports:

- the unweighted mean of the three window-level paired means;
- the range across the three window-level paired means; and
- the number of windows with positive, negative, and zero mean effects.

The cross-window mean gives each two-year window equal weight. It is
descriptive and receives no population-level confidence interval because there
are only three windows.

## Validation gates

The implementation must reject:

- missing, duplicate, or unexpected conditions;
- any seed set other than integers 0 through 9;
- different date grids between ANN, MPS, and the benchmark;
- evaluation dates outside the three frozen two-year windows;
- missing or non-finite daily returns, account values, turnover, or cost;
- a benchmark with more than one row per date;
- a mismatch between the five requested annual metrics and the saved annual
  period table; and
- output that omits any prespecified year, state family, or state label.

The manifest must hash this protocol, every source curve and annual table, the
analysis script, and every generated table. Results are written under
`results/robustness/market_states/` and remain separate from the scientific
freeze and primary estimand.

## Interpretation rules

All states and all windows are reported, including unfavorable and null cells.
The paper may describe a pattern as exploratory only when its sign is
consistent across all three windows or when a clearly identified individual
window is being described. A seed-bootstrap interval quantifies PPO-seed
dispersion conditional on one historical window; it does not represent
calendar-sampling uncertainty.

The analysis cannot establish that a market state causes MPS or ANN to perform
better. States overlap, daily observations are serially dependent, and the
models were not optimized for these categories. No multiple-comparison
significance claim, quantum advantage, live-trading claim, or general
tensor-network conclusion is permitted.
