# Nested Evaluation-Horizon Protocol

Version 1.0, locked on 2026-07-31 before computing the new prefix outcomes.

## Status and purpose

This is a separately versioned, post-hoc exploratory re-analysis requested
after the primary and equal-window results were known. It does not unfreeze,
replace, or reclassify either completed experiment. In particular, the
five-year 2019--2023 primary MPS-minus-ANN Sharpe estimate was already known
to be negative when this protocol was written.

The purpose is narrower than the equal-window study: determine how the
descriptive ANN-versus-MPS comparison changes as the evaluation horizon grows
while holding the trained policies, training data, PPO seeds, representation
models, assets, costs, and evaluation start date fixed.

## Fixed source and no-retraining rule

The only input is `results/final/equity_curves.csv` from the frozen primary
ten-seed evaluation. No encoder or PPO policy will be refit. Every horizon
therefore uses the same realized daily paths from:

- Base FinRL, ANN signal, and QINN-MPS signal;
- matched PPO seeds 0 through 9;
- 20,000 PPO steps;
- MPS bond dimension 2;
- the 15-stock universe and frozen feature/state construction;
- a 0.10% transaction cost and USD 1,000,000 initial portfolio; and
- the 2019-01-02 evaluation start.

The source file and frozen primary manifest must pass hash and configuration
validation before any prefix is summarized.

## Locked cumulative horizons

The analysis will report every calendar-year endpoint available in the
source, rather than stop after a favorable or unfavorable prefix:

| Horizon | Inclusive evaluation period |
|---|---|
| 1 year | 2019-01-02 through the last saved trading date of 2019 |
| 2 years | 2019-01-02 through the last saved trading date of 2020 |
| 3 years | 2019-01-02 through the last saved trading date of 2021 |
| 4 years | 2019-01-02 through the last saved trading date of 2022 |
| 5 years | 2019-01-02 through 2023-12-28 |

The one-, two-, and three-year prefixes directly answer the requested
question. The four- and five-year prefixes are mandatory safeguards against
selectively stopping the reported path. The five-year endpoint must reproduce
the frozen primary metrics within numerical tolerance.

## Metrics and calculations

For every condition, seed, and horizon, recompute from the prefix:

- total and annualized return;
- annualized volatility and Sharpe ratio, with zero risk-free rate;
- maximum drawdown;
- annualized gross one-way turnover; and
- total modeled transaction cost.

Calculations must match `run_experiment.py`: 252 trading days per year,
geometric annualization using the number of saved trading transitions,
sample-standard-deviation volatility, and daily-return Sharpe. Transaction
cost and turnover are summed only through the applicable cutoff.

For each horizon and metric, report condition means and all ten paired
MPS-minus-ANN seed differences. Annualized Sharpe is the organizing outcome.
Report the mean Sharpe difference, its deterministic paired-seed bootstrap
95% interval using 10,000 samples and seed 20260728, and the number of seeds
in which MPS exceeds ANN.

The primary artifact contains only full-period aggregate encoder-prediction
metrics, not date-level predictions. Prediction quality will therefore not be
invented or retroactively approximated for these prefixes. The existing
full-period signal-quality comparison remains available separately.

## Interpretation boundary

The horizons are nested and strongly dependent: a two-year result contains
the one-year result, and so on. They are not independent replications and
their intervals must not be combined into a population-level trend test.
Changes may reflect accumulated market observations, compounding, policy
path dependence, or year-specific returns; this design cannot identify which
mechanism caused them.

Allowed language is limited to descriptive statements such as:

- which representation has the higher mean metric at each cumulative cutoff;
- whether the paired mean difference changes sign as the horizon expands;
- seed-level consistency and uncertainty; and
- whether turnover, cost, or drawdown differences accompany the Sharpe path.

The analysis may not claim an optimal horizon, a stable MPS advantage,
quantum advantage, or a causal horizon effect. Every locked prefix and every
unfavorable result must be reported. Because the design follows earlier
outcomes, it will be labeled exploratory and post hoc wherever cited.

## Evidence gate

The results may enter the paper only after:

1. the source hash and primary configuration validate;
2. each condition has exactly seeds 0--9 on one common date grid;
3. every locked year-end cutoff is present;
4. the five-year recomputation matches the frozen primary metrics;
5. generated tables and figures are hash-bound to a manifest;
6. focused and full repository tests pass; and
7. paper wording retains the post-hoc, nested-dependence limitations.
