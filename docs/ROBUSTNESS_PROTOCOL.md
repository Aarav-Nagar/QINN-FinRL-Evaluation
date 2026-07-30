# Temporal Robustness Protocol

Version 0.1, frozen before viewing shifted-window outcomes on 2026-07-29.

## Purpose

The primary study fits PPO through 2018 and evaluates 2019-2023. This secondary
analysis asks whether the sign and broad magnitude of the paired
MPS-minus-ANN result persist in an earlier, non-overlapping evaluation period.
It is a robustness check, not a replacement primary estimand.

## Frozen shifted window

| Stage | Dates |
|---|---|
| Encoder fitting | 2013-01-02 through 2015-12-31 |
| Encoder validation and early stopping | 2016-01-01 through 2016-12-30 |
| PPO training | 2013-01-02 through 2016-12-30 |
| Portfolio evaluation | 2017-01-03 through 2018-12-28 |

The evaluation period does not overlap the primary 2019-2023 evaluation.
Features may use historical lookback observations, but neither encoder fitting,
early stopping, normalization, nor PPO training may observe a target or reward
from 2017-2018. Boundary rows whose next-day target enters the evaluation
period must be excluded from encoder fitting.

## Fixed settings

- Conditions: Base FinRL, ANN signal, and QINN-MPS signal.
- PPO budget: 20,000 environment steps.
- PPO seeds: 0 through 9, matched across conditions.
- MPS bond dimension: 2, selected before final and robustness outcomes.
- Representation seed: 2026.
- Encoder maximum epochs: 60.
- Encoder patience: 10.
- Encoder batch size: 512.
- Encoder device: CPU.
- Initial portfolio: USD 1,000,000.
- Transaction cost: 0.10% per executed buy or sell.
- Asset universe, state construction, feature definitions, PPO architecture,
  and FinRL commit: unchanged from the primary study.

No robustness setting may be changed because an intermediate result is
unfavorable. Failed endpoints may be resumed only with the same configuration.

## Outcomes and interpretation

The robustness estimand is the mean paired MPS-minus-ANN annualized Sharpe
difference across seeds 0-9. Report:

- every seed-level difference;
- condition means and standard deviations;
- paired-seed bootstrap interval using the same deterministic procedure;
- the count of positive, negative, and zero paired differences;
- annual return, maximum drawdown, turnover, and transaction costs;
- validation/test signal metrics and runtime metadata.

Directional stability means only that the paired mean has the same sign in the
primary and shifted windows. It does not imply statistical replication,
stationarity, causal validity, or deployable profitability. A sign change is
reported directly and is not used to alter either analysis.

## Known design limitations

- The shifted evaluation covers two years rather than the primary five.
- Its PPO training history is shorter.
- Market regime and available training sample change together, so the analysis
  cannot isolate calendar regime from sample-size effects.
- The two evaluations use the same asset universe and data-generation process.
- Ten seeds provide limited resolution for training-seed uncertainty.

## Evidence gate

Paper claim C08 remains pending until the complete configured endpoint set,
run manifest, paired summary, source hashes, and figure or compact table are
generated and validated. Annual slices of the primary trajectory do not satisfy
this protocol.
