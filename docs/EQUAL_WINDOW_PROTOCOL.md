# Equal-Length Temporal Robustness Protocol

Version 1.0, frozen on 2026-07-31 before viewing outcomes for the two new
windows.

## Purpose

The completed primary and shifted evaluations remain frozen and retain their
existing roles in the paper. The primary evaluation covers 2019--2023, while
the shifted evaluation covers 2017--2018. Because those periods differ in both
calendar regime and evaluation length, their sign reversal cannot distinguish
time sensitivity from horizon sensitivity.

This additional secondary analysis compares three non-overlapping two-calendar-
year evaluations. Each evaluation is preceded by the same four-calendar-year
PPO training span. Within that span, the encoder uses the first three years for
fitting and the fourth year for validation and early stopping. This controls
evaluation and training-span length while allowing the available information
set to advance chronologically.

The 2017--2018 cell reuses the completed shifted-window run because its dates,
seeds, and scientific settings exactly match this protocol. It will not be
retrained merely to create a new artifact.

## Frozen windows

| Window | Encoder fitting | Encoder validation | PPO training | Portfolio evaluation |
|---|---|---|---|---|
| 2017--2018 | 2013-01-02 through 2015-12-31 | 2016-01-01 through 2016-12-30 | 2013-01-02 through 2016-12-30 | 2017-01-03 through 2018-12-28 |
| 2019--2020 | 2015-01-02 through 2017-12-29 | 2018-01-01 through 2018-12-28 | 2015-01-02 through 2018-12-28 | 2019-01-02 through 2020-12-31 |
| 2021--2022 | 2017-01-03 through 2019-12-31 | 2020-01-02 through 2020-12-31 | 2017-01-03 through 2020-12-31 | 2021-01-04 through 2022-12-30 |

The boundary dates are the first and last available market dates in the
checksum-verified repository dataset for each stage. Features may use prior
lookback observations. Encoder fitting, early stopping, normalization, PPO
training, and reward calculation may not observe an evaluation-period target
or reward. The existing next-day-target boundary guard remains mandatory.

## Frozen scientific settings

- Conditions: Base FinRL, ANN signal, and QINN-MPS signal.
- PPO steps: 20,000 per condition and seed.
- PPO seeds: 0 through 9, matched across conditions and windows.
- MPS bond dimension: 2.
- Representation seed: 2026.
- Encoder maximum epochs: 60; patience: 10; batch size: 512.
- Encoder device and PPO device: CPU.
- Initial portfolio: USD 1,000,000.
- Transaction cost: 0.10% per executed buy or sell.
- Assets, features, state construction, PPO architecture, dataset checksums,
  and FinRL commit: unchanged from the frozen evaluations.

No setting may be changed after a new window begins because its result is
unfavorable. Interrupted endpoints may resume only from a configuration-
matching checkpoint.

## Prespecified outcomes

For each window, report all ten matched seed differences for MPS minus ANN and
condition-level means and standard deviations for:

- annualized Sharpe ratio;
- annualized return;
- maximum drawdown;
- annualized turnover;
- total transaction cost.

Also report ANN and MPS evaluation-period signal MSE, MAE, directional
accuracy, and information coefficient, plus their MPS-minus-ANN differences.
The Sharpe difference remains the organizing trading outcome. Other metrics
are descriptive secondary outcomes, not alternative result-selection rules.

## Interpretation and multiplicity

Every completed window will be reported, including unfavorable or
inconclusive windows. The panel is exploratory temporal-robustness evidence,
not a new primary hypothesis test. A pattern will be described only as:

- sign consistency or heterogeneity across these three fixed windows;
- seed-level uncertainty within each window;
- descriptive alignment, or lack of alignment, between signal quality and
  downstream portfolio outcomes.

The analysis will not select a preferred window, pool the three correlated
historical estimates into a universal effect, or claim a market-regime cause.
No additional date windows will be added in response to observed results.

## Evidence gate

The paper may cite this panel only after:

1. both new 30-endpoint matrices complete;
2. manifests confirm the exact configurations above;
3. every condition/seed key is present exactly once;
4. source and output hashes validate;
5. the cross-window tables are generated from all three run directories; and
6. the exploratory wording is checked against this protocol.
