# ATL residual-MPS replication audit

## Overall assessment: share with caveats

The published July 1-August 16 ATL result is exactly reproducible in ATL, but
the immediately following four-day window is flat rather than positive. The
same-window tests establish deterministic reproducibility, not independent
statistical replication or future profitability.

## Exact same-window replication

Three complete hosted executions used the same frozen artifact and policy:

| ATL run | Return | Final equity | Drawdown | Trades | Timeouts |
|---|---:|---:|---:|---:|---:|
| `ext_20260821_152912_49e74732` | +1.29364% | $1,012.9364 | -2.6894% | 11 | 0 |
| `ext_20260821_154414_5f333614` | +1.29364% | $1,012.9364 | -2.6894% | 11 | 0 |
| `ext_20260821_154958_09af96ad` | +1.29364% | $1,012.9364 | -2.6894% | 11 | 0 |

The metrics, 224-point equity curve, 11-trade ledger, and 224 submitted decision
batches are byte-for-byte equivalent after excluding run metadata. Replaying the
frozen local policy over captured hosted snapshots also reproduced all 224 ATL
action batches with zero mismatches.

This is strong execution reproducibility for fixed historical inputs. Because
all three runs use the same dates and historical market path, they are not
independent samples of performance.

## Immediate temporal extension

The chronology-selected extension covers August 17-20, 2026, the four complete
trading days immediately after the original evaluation.

| Result | Agent | ATL DJIA | ATL buy-and-hold |
|---|---:|---:|---:|
| Return | 0.0000% | -0.4370% | 0.0000% |
| Trades | 0 | 0 | 1 |
| Maximum drawdown | 0.0000% | -1.8651% | 0.0000% |

Both hosted extension runs were exactly identical: 28 decisions, zero trades,
zero timeouts, and $1,000 final equity. The positive-trend gate rejected exposure
on the first rebalance, and four days ended before the next five-session
rebalance. The agent protected capital relative to the declining DJIA, but this
does not replicate positive returns.

## Data-lineage finding

The initial offline benchmark used seven separately collected weekly snapshot
files. Comparing them with snapshots captured inside the hosted execution found:

- 224/224 timestamps in common and zero duplicate timestamps;
- prices matched exactly in 1,641/1,641 overlapping symbol rows;
- candidate symbol sets matched on only 51/224 timestamps (22.77%);
- no complete `top_signals` payload matched exactly;
- RSI, MACD, SMA, Bollinger, relative-rank, breadth, and volatility-derived
  features changed with collection context.

This is a high-impact data-lineage issue. ATL's prices are stable, but its
candidate/indicator state is run-context dependent. Therefore the earlier
-3.5493% replay was not based on the same model inputs as the hosted agent and
must not be treated as an execution-faithful contradiction.

## Corrected hosted-context controls

The frozen benchmark was rerun without policy selection on the 224 snapshots
captured inside the hosted execution.

| System | Return after 10 bps modeled cost | Drawdown | Trades |
|---|---:|---:|---:|
| Selected 75% MPS + 25% trend | +1.2834% | -1.8664% | 13 |
| Matched ANN + trend | +1.2834% | -1.8664% | 13 |
| MPS-only rank | +0.2272% | -2.6930% | 35 |
| Trend-only | +0.2878% | -2.6615% | 13 |
| Fixed initial trend basket | +6.0083% | -1.6309% | 3 |
| Cash | 0.0000% | 0.0000% | 0 |

The corrected selected-system replay is positive and close to the +1.2936%
hosted gross result, although it still approximates execution and produces 13
rather than 11 trades. The combined system exceeds either component in this
approximation, but the matched ANN combination produces exactly the same
portfolio result. The evidence therefore supports the combined deployment
architecture, not unique MPS value. The fixed passive basket remains materially
stronger on this one window.

On the corrected input context, residual MPS versus matched ANN prediction MSE
is 6.20303 versus 6.20510 squared percentage points, directional accuracy is
81.67% versus 81.43%, and rank correlation is 0.31872 versus 0.31713. MPS MAE is
slightly worse: 1.63782 versus 1.63757 percentage points. These are small,
mixed differences.

## Cost and concentration stress tests

The hosted run traded $3,086.6736 of cumulative notional. Estimated return is:

| Cost per traded notional | Estimated return |
|---:|---:|
| 0 bps | +1.2936% |
| 10 bps | +0.9850% |
| 25 bps | +0.5220% |
| 50 bps | -0.2497% |
| 100 bps | -1.7930% |

The estimated break-even cost is 41.91 basis points per traded notional. The
result is positive under moderate costs but not high frictions.

Weekly equity contributions show concentration: week 29 contributed -0.8805
percentage points, week 30 +0.7955, week 32 +1.3101, and week 33 +0.0685; the
other three weeks were flat. More than the full net gain came from one week,
offset by an earlier loss.

## Data-quality checks

The captured primary data has 224 unique, sorted timestamps, 32 trading days,
exactly seven hourly observations per day, 10 visible signals per snapshot, 30
distinct symbols, no nonpositive prices, and no non-finite signal values. The
extension has the same properties across 28 observations and four trading days.

The first snapshot-capture implementation accidentally wired the output flag to
the historical runner. The hosted results were unaffected; the wiring was fixed
at commit `97d9a33`, and the capture runs were repeated. One additional capture
attempt was excluded after a local OneDrive atomic-file lock stopped it before
the first submitted decision.

## Conclusion

The honest finding is stronger and narrower than “the agent made money”:

1. The hosted execution is exactly reproducible on the same historical window.
2. Corrected hosted-context offline controls also produce a positive combined
   system, resolving the earlier negative replay discrepancy.
3. The immediate out-of-time extension is flat, not positive.
4. The matched ANN produces the same combined portfolio outcome, so MPS-specific
   trading value is unproven.
5. Performance is cost-sensitive and concentrated in one profitable week.

This package is ready to share as a reproducibility and failure-analysis
contribution, with these caveats kept visible.
