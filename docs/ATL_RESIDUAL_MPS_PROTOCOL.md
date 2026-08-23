# ATL residual-MPS ensemble frozen protocol

## Objective

Test whether aligning the prediction target with execution and maintaining
risk-gated market exposure can produce a positive after-cost result without
tuning on the final evaluation window. Positive return is a goal, not a
guaranteed or selectively reported outcome.

## Architecture

Each of five ensemble members contains:

- a 13-site classical MPS with bond dimension 5;
- a linear residual path over the same standardized inputs; and
- learned MPS and residual mixing coefficients.

Each member has 586 trainable parameters. Training uses seeds 0-4, smooth-L1
loss, a 0.15-weighted within-timestamp ranking loss, training-only
normalization, early stopping, and no quantum hardware.

The target is the return seven ATL observations ahead, approximately the next
trading session, rather than the old one-hour target. Price history is retained
across sessions because overnight and weekend changes are valid information for
this next-session horizon.

## Portfolio calibration

Seven profiles were declared before the final test. Every eligible deployment
profile gives the MPS at least 50% of the combined cross-sectional rank. The
profiles vary only daily versus five-session rebalancing, 50-100% MPS rank
weight, and whether a positive market-trend gate is required.

Selection uses June validation return after 10 basis points per traded notional,
drawdown, turnover, and whole-share feasibility with $1,000. A profile must
have positive validation return and at least three trades to be eligible.

The selected policy is frozen as:

- 75% uncertainty-adjusted MPS rank and 25% observable trend rank;
- `ensemble mean - 0.5 * ensemble standard deviation`;
- positive median market-trend gate;
- maximum three whole-share positions; and
- one rebalance every five trading sessions.

Its approximate stateful validation simulation returned +3.5011% after modeled
cost, with -1.5668% maximum drawdown, 18 trades, and 0.4845% cost relative to
initial capital. This was selection evidence, not a final result.

## Frozen periods

- Training cutoff: May 31, 2026, using observations beginning January 5.
- Portfolio validation: June 1-30, 2026.
- Untouched one-time evaluation: July 1-August 15, 2026.

The architecture, artifact, tests, and this protocol are committed before the
July-August snapshots are collected. Rows whose forward target crosses a split
are embargoed. The evaluation result will be reported even if it is negative.

## Comparisons and endpoints

- Matched 586-parameter ANN ensemble under the same target and dates.
- MPS-only, trend-only, and cash system controls.
- ATL-native DJIA and buy-and-hold references from the hosted run.
- Prediction error, directional accuracy, and within-timestamp rank correlation.
- Return, drawdown, turnover, modeled cost, trades, and invested-day fraction.

The offline simulator approximates whole-share ATL execution; the hosted ATL
run is authoritative for actual platform orders and portfolio accounting.
