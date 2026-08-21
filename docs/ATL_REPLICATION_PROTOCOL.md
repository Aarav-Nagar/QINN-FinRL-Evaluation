# ATL residual-MPS replication protocol

Frozen before replication runs on August 21, 2026.

## Question

Does the suffix-free `Aarav Residual MPS Ensemble` reproduce its published
hosted result when ATL reruns the identical dates, and what happens in the
immediately following complete historical window without retraining or policy
selection?

## Fixed implementation

- Artifact SHA-256:
  `a2be3a5286cf7bb72116dac8a2f23b82173ddecdb44c7c81153edbe123d999b6`.
- Original freeze commit: `7e2a926663d1a94572084cc9ee77609d39aade9f`.
- Five seeds: 0-4; 586 parameters per residual-MPS member.
- Seven-observation next-session target.
- 75% uncertainty-adjusted MPS rank and 25% observable trend rank.
- Positive median-trend exposure gate, three whole-share slots, and one
  rebalance every five trading sessions.
- $1,000 initial capital and live trading disabled.

No model, feature, threshold, rank weight, gate, position count, or rebalance
frequency may change after the runs begin.

## Test A: exact hosted rerun

Rerun July 1 through exclusive end August 16, 2026 from a fresh policy object.
Capture every ATL snapshot. Compare the new run with
`ext_20260821_152912_49e74732` on timestamps, orders, traded notional, final
equity, return, drawdown, trade count, and timeout holds.

Exact equality is expected if ATL's historical inputs and execution engine are
deterministic. Any mismatch must be retained and attributed, not averaged away.

## Test B: immediate temporal extension

Run August 17 through exclusive end August 21, 2026. This is the immediately
following set of complete trading days available when the protocol is frozen.
It is selected by chronology, not performance. Report return, drawdown, trades,
timeouts, ATL-native DJIA and buy-and-hold references, and the same 10-basis-
point estimate on every recorded buy and sell notional.

This four-day extension is an out-of-time micro-replication, not enough evidence
for statistical significance or a general profitability claim.

## Test C: data and robustness audit

- Verify snapshot timestamp uniqueness, chronological order, expected seven
  hourly decisions per trading day, finite prices, and symbol coverage.
- Compare repeated market snapshots at matching timestamps after removing
  portfolio state.
- Recompute gross return from the final equity independently.
- Apply 0, 10, 25, 50, and 100 basis points per traded notional to the recorded
  hosted orders.
- Decompose the original run into calendar weeks and report the concentration of
  gains and losses. These dependent subperiods are diagnostics, not independent
  replications.
- Preserve the matched ANN, MPS-only, trend-only, fixed-basket, and cash controls
  from the frozen benchmark; do not select a replacement system from them.

## Decision rule

Replication is `exact` only if Test A reproduces the order ledger and headline
metrics. Temporal evidence is `positive`, `flat`, or `negative` from Test B's
estimated after-cost return. Overall evidence may still be only `share with
caveats` because one exact rerun and a four-day extension do not establish
independent statistical replication.
