# ATL MPS v2 evaluation result

Verified August 20, 2026 using the frozen protocol in
`ATL_MPS_AGENT_V2_PROTOCOL.md`.

## Data and integrity

- Completed ATL snapshots: 469 fit/validation and 161 untouched evaluation.
- Supervised rows after removing overnight/weekend targets: 2,883 fit, 350
  validation, and 1,110 test.
- Test decisions: 138 hourly targets across 23 trading days.
- Symbols observed: 30 DJIA constituents.
- Feature audit: zero non-finite values and zero degenerate features.
- Combined benchmark data SHA-256:
  `c20d222fdc650500540e54277442b345bb461abe650839f5cbf1928cfbb0667a`.

The deployable seed-2026 MPS completed 34 epochs with validation MSE 0.365605
in squared percentage-point units. Its best validation threshold still had
modeled net edge of -0.078268 percentage points after the 10-basis-point cost,
so the frozen policy disabled new entries.

## Ten-seed matched test comparison

| Metric, mean across paired seeds 0-9 | MPS, 369 parameters | ANN, 369 parameters |
|---|---:|---:|
| Prediction MSE, squared pp | 0.311415 | 0.310693 |
| Directional accuracy | 54.61% | 54.89% |
| Rank correlation | 0.1105 | 0.1559 |
| Cost-adjusted total return | -2.1039% | -2.0996% |
| Maximum drawdown | -3.4521% | -3.9723% |
| Cumulative turnover | 56.60 | 70.60 |
| Modeled cost, percent of initial value | 5.66% | 7.06% |

The paired mean MPS-minus-ANN total-return difference was -0.0043 percentage
points. Its paired-seed bootstrap 95% interval was [-3.7329, 2.7257] percentage
points. The interval includes zero, so this experiment does not establish a
return advantage for either learned representation. MPS used less turnover in
this implementation, but that descriptive difference is not evidence of alpha.

## Simpler controls

| Control | Cost-adjusted return | Maximum drawdown | Cumulative turnover |
|---|---:|---:|---:|
| Ridge regression | -3.7873% | -4.3122% | 29.00 |
| Three-hour momentum | -3.9261% | -7.8836% | 131.67 |
| One-hour mean reversion | -22.0498% | -22.0498% | 194.33 |
| Dynamic equal weight | -6.4722% | -7.6745% | 73.28 |

These controls use the changing ATL `top_signals` candidate set and are not a
claim about conventional DJIA index strategies. Cash (zero trades and zero
return) is the relevant outcome for the deployed model because its validation
gate did not clear modeled costs.

## Main finding

V2 turns an overtrading proof of concept into a leakage-audited decision system
that can decline to trade. The empirical finding is bounded and negative: the
MPS did not beat the matched ANN, and neither learned signal validated a
positive cost-adjusted edge. The contribution is the reproducible ATL-native
agent, matched evaluation, and enforced abstention behavior—not a fabricated
performance win.

The hosted multi-week integration result and immutable ATL version identifiers
are recorded separately in `ATL_MPS_AGENT_RESULT.md`.
