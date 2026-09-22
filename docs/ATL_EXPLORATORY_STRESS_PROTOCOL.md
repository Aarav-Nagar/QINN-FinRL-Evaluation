# ATL exploratory date and market stress protocol

## Status and purpose

This plan is recorded before launching the subwindow and China-market runs.
It is exploratory, not a second untouched evaluation: the aggregate
July-August U.S. outcome is already known. Every result will be retained,
including failures, flat windows, and negative windows.

The frozen model artifact, feature contract, ensemble weights, trend gate,
five-session rebalance, and $1,000 whole-share allocation remain unchanged.
No result may be described as groundbreaking, alpha, or model superiority
without a comparable baseline and evidence that survives the full matrix.

## U.S. external-agent date-boundary stress

Run the account-owned external agent with fresh policy state on these
non-overlapping partitions of the already observed evaluation path:

| Label | Start | Exclusive end |
|---|---|---|
| US-A | 2026-07-01 | 2026-07-16 |
| US-B | 2026-07-16 | 2026-08-01 |
| US-C | 2026-08-01 | 2026-08-16 |
| US-D | 2026-08-17 | 2026-08-21 |

The full-window account run remains the reference. Compare the sum of
continuous full-run subperiod contributions with fresh-state partition runs to
measure start-date and state-initialization sensitivity. Capture hosted inputs
to a non-OneDrive temporary directory first, then copy complete evidence into
the repository.

## China A-share platform and transfer stress

ATL exposes iFinD 60-minute data for April 1 to May 1, 2026. Run its declared
rule-based decision path on both visible universes:

- `a_share_demo_6`: six named A-shares.
- `csi300_sample_20_2026h2`: twenty named A-shares.

These are ATL platform controls, not MPS results. Inspect their returned run,
decision, trade, and chart data for whether the frozen 13-feature MPS contract
can be reconstructed at identical timestamp and symbol grain. Run a local MPS
transfer stress only if every required market-derived input is available or
can be deterministically computed from pre-decision OHLCV without leakage. Do
not synthesize missing indicators or silently map U.S. symbols/features to
China.

## Required checks

- complete every declared run or record the platform error;
- verify timestamp uniqueness, ordering, window bounds, symbol coverage,
  finite prices/features, decision count, timeout count, and trade ledger;
- report gross return, estimated return after 10 bps per traded notional,
  drawdown, trades, turnover or traded notional, and invested exposure;
- compare U.S. windows with cash, ATL references when present, and corrected
  hosted-context controls where reconstructable;
- label China currency, FX conversion, lot-size, and benchmark behavior;
- show both per-window results and aggregate distributions;
- treat overlapping dates, repeated historical paths, and multiple testing as
  limitations rather than independent evidence.

## Decision rule

A finding is notable only if it is mechanically reproducible, not driven by a
single date boundary, remains directionally useful after modeled costs, and is
supported by a matched comparison. Otherwise report it as a diagnostic or
hypothesis for the preregistered future window.
