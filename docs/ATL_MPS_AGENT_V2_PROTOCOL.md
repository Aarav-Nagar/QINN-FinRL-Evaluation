# ATL MPS v2 frozen evaluation protocol

## Research question

Does a 369-parameter classical matrix-product-state return signal provide a
cost-adjusted advantage over a 369-parameter neural signal and simpler controls
when every method receives the same ATL-native hourly information?

The MPS is a classical tensor-network simulation. No quantum hardware is used.

## Frozen data boundaries

- Fit: January 5 through March 31, 2026.
- Validation and threshold calibration: April 1 through April 10, 2026.
- Untouched evaluation: April 15 through May 15, 2026.
- A row is excluded if its next-hour target crosses a partition boundary.
- A row is excluded if the next observed snapshot is more than two hours away,
  preventing overnight and weekend returns from being labeled as hourly.
- Feature means and scales are estimated from fit rows only.

ATL collection uses exclusive end dates. The raw API files stay out of Git
because they contain run/session metadata; the evidence package records a
canonical SHA-256 digest of the combined snapshots.

## Feature repair

`cash_fraction` and `position_fraction` are removed from prediction. The 13
inputs are market-only RSI, MACD, moving-average, Bollinger, lagged-return,
cross-sectional rank, cross-sectional volatility, and breadth features.
Unavailable zero-valued price indicators fall back to a neutral value. Training
fails if any feature remains non-finite or constant in the fit partition.

## Matched comparison

- MPS: bond dimension 4, 369 trainable parameters.
- ANN: 13-20-4-1 multilayer perceptron, 369 trainable parameters.
- Seeds: 0 through 9, paired across MPS and ANN.
- Shared target: next-hour return in percentage points.
- Shared optimizer, epoch ceiling, early stopping, standardization, dates, and
  candidate rows.
- Controls: ridge regression, three-hour momentum, one-hour mean reversion, and
  dynamic equal weight.
- Modeled transaction cost: 10 basis points per unit of portfolio turnover.

Prediction metrics are MSE, MAE, directional accuracy, and rank correlation.
Portfolio metrics are total return, maximum drawdown, turnover, modeled cost,
activity, and mean net hourly return. Sharpe is not a primary endpoint because
the evaluation is short and hourly observations are dependent.

## Execution controls

The hosted policy uses a validation-selected entry threshold, residual-based
confidence, exponential score smoothing, a three-decision minimum hold, a
two-decision cooldown, at most three positions, a 25% position cap, and a 25%
per-decision turnover cap. If no validation threshold has positive modeled net
edge, calibration disables new entries. It only calls ATL's `safe_trading` historical
backtest route and explicitly disables live trading on the registered agent.

## Interpretation constraints

ATL exposes only `top_signals` rather than a stable full-DJIA panel. Candidate
membership can therefore change across hours, and the dynamic-equal-weight
control is not a conventional investable DJIA index. The offline benchmark uses
fractional weights for a clean model comparison; the hosted verification uses
ATL's integer orders. The benchmark is self-reported until independently
reproduced or promoted by ATL maintainers. A confidence interval including zero
must be reported as inconclusive, regardless of the point estimate.
