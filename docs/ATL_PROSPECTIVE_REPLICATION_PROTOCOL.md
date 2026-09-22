# ATL prospective multi-week replication protocol

Registered on August 21, 2026, before the evaluation window begins.

## Objective

Test whether the frozen `Aarav Residual MPS Ensemble` produces a positive,
flat, or negative result over a new non-overlapping multi-week ATL window, and
whether a second identical hosted execution reproduces the order ledger and
metrics.

## Frozen window

- Start: August 24, 2026.
- Exclusive end: September 19, 2026.
- Intended market coverage: August 24 through September 18.
- Earliest execution time: September 21, 2026 at 6:00 PM America/New_York.
- Initial simulated capital: $1,000.

The window is the next complete four-week block after the August 17-20
micro-extension. It is selected by chronology, not by observed performance.
Market holidays and missing ATL partitions will be retained and reported rather
than replaced with favorable dates.

## Frozen policy

- Artifact SHA-256:
  `a2be3a5286cf7bb72116dac8a2f23b82173ddecdb44c7c81153edbe123d999b6`.
- Original policy freeze commit:
  `7e2a926663d1a94572084cc9ee77609d39aade9f`.
- Immutable ATL release: `agv_c23fdf3230ad`.
- Five bond-dimension-5 residual-MPS members, seeds 0-4, 586 parameters each.
- Seven-observation next-session prediction target.
- 75% uncertainty-adjusted MPS rank plus 25% observable trend rank.
- Positive median-trend gate, three whole-share slots, and one rebalance every
  five trading sessions.
- Fresh policy state at the start of each hosted run.
- Live trading disabled and no quantum hardware.

No feature, model weight, seed, threshold, gate, position limit, rebalance
frequency, artifact, or date may change after this protocol is committed.

## Execution sequence

1. Confirm that the current date is at least September 21, 2026 and the protocol
   commit predates all evaluation observations.
2. Run ATL from `2026-08-24` to exclusive end `2026-09-19`, capturing snapshots
   and the complete hosted result.
3. Repeat the identical run from a fresh policy object and compare metrics,
   equity curve, decisions, and trades exactly.
4. Replay the policy locally over hosted-captured snapshots and require every
   submitted action batch to match ATL.
5. Rerun the frozen matched ANN, MPS-only, trend-only, fixed-basket, and cash
   controls on the captured hosted context without selecting a replacement.
6. Apply 0, 10, 25, 50, and 100 basis points per recorded traded notional and
   report the break-even cost.
7. Report weekly equity contribution and whether one week accounts for more
   than the full-window gain or loss.
8. Verify timestamp uniqueness, seven-observation daily coverage when markets
   are open, finite prices, symbol coverage, and raw-input hashes.
9. Publish the result even when negative, flat, inactive, incomplete, or
   inconsistent.

## Endpoints

The primary endpoint is estimated return after 10 basis points per recorded buy
or sell notional. Secondary endpoints are ATL gross return, final equity,
maximum drawdown, trades, turnover, timeout holds, invested fraction, weekly
concentration, and ATL-native DJIA and buy-and-hold references.

Prediction endpoints are next-session MSE, MAE, directional accuracy, and
within-timestamp rank correlation for the residual-MPS and exactly matched ANN
ensembles.

## Interpretation

The temporal result is classified from the estimated after-cost return as
`positive`, `flat`, or `negative`. An exact second run establishes deterministic
reproducibility for the fixed window, not an independent sample. The result will
not support MPS-specific value unless the MPS comparison differs from the
matched ANN under the same inputs, policy mechanics, and costs.

A single additional four-week window remains insufficient for a general alpha,
significance, or future-profitability claim. It is a stronger prospective test
than repeating already observed dates.
