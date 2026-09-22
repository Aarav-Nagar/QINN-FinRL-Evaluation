# ATL exploratory date and China-market stress results

## Bottom line

The frozen account-owned agent works reproducibly, but this test did **not**
find groundbreaking trading performance or evidence that the MPS is superior.
It found something more useful for improving the agent: its decisions depend
strongly on how much earlier market history and rebalance state it carries into
a date window.

The four U.S. runs used the frozen artifact and fresh state. Three stayed in
cash. Only the August 1-16 run traded, returning +0.4166% gross and an estimated
+0.3539% after 10 basis points per traded notional. That positive run still
trailed both ATL references for the same dates.

| Fresh-start window | Agent gross | After cost | ATL DJIA | ATL buy/hold | Trades |
|---|---:|---:|---:|---:|---:|
| Jul 1-Jul 15 | 0.0000% | 0.0000% | +0.3050% | -0.0600% | 0 |
| Jul 16-Jul 31 | 0.0000% | 0.0000% | +1.1000% | -0.2100% | 0 |
| Aug 1-Aug 15 | +0.4166% | +0.3539% | +1.3570% | +0.9110% | 3 |
| Aug 17-Aug 20 | 0.0000% | 0.0000% | -0.4370% | 0.0000% | 0 |

Across the four windows, the mean gross return was +0.1041%, the median was
0.0000%, and only one window was positive. The agent beat the ATL DJIA index
in one of four windows. It beat ATL's buy/hold reference in two, tied once,
and lost once. These are short, overlapping historical diagnostics, not four
independent trials.

## The reproducible finding: start-state sensitivity

The original continuous July 1-August 15 run returned +1.2936% gross. Restarting
the same frozen policy at each subwindow produced only +0.4166% gross in total,
a difference of -0.8771 percentage points. On the August window alone, the
fresh run returned +0.4166%, while the same dates contributed about +1.3786
percentage points inside the continuous run.

This is not random execution noise:

- every one of the 252 fresh-window action batches replayed exactly from the
  saved ATL inputs and frozen artifact;
- a second hosted August run reproduced the first exactly: metrics, equity
  curve, trades, decisions, and all 70 input snapshots;
- only 96 of 224 action batches matched between the fresh partitions and the
  continuous run on common timestamps;
- the match rate fell from 100% in the first partition to 25% in the second
  and 7.1% in the August partition.

The mechanism is expected from the current design: price history, the
positive-trend gate, and the five-session rebalance clock build state across
time. A fresh backtest starts those components over. The result is therefore a
real deployment-design issue: the agent needs an explicit warm-up/history
contract if results should be comparable across arbitrary ATL start dates.

## Cost and exposure stress

The only trading window bought three whole-share positions with $626.51 of
recorded traded notional. Average invested exposure across its 70 hourly
snapshots was 30.48%, peak exposure was 63.02%, and the policy held positions
in 34 snapshots. Its return remained positive at 10, 25, and 50 basis points
per traded notional, but turned negative at 100 basis points. The estimated
break-even cost was 66.49 basis points. This is useful robustness information,
but one short winning window is not a performance claim.

## China A-share tests

ATL displayed iFinD 60-minute data and accepted the declared April 1-May 1
configuration. Both tests used ATL's **rule-based** control, not the external
MPS:

- A-Share Demo 6 reached a result page marked `Failed` and `Backtest did not
  start`, with ATL's message that the iFinD backtest failed.
- CSI 300 Sample 20 failed during launch and did not create a completed run or
  result record.

Neither failure is a 0% return. ATL returned no China decision ledger, trades,
chart data, OHLCV, currency conversion, lot-size behavior, or feature payload.
Therefore the frozen 13-feature MPS contract could not be reconstructed
without inventing inputs, so no China-MPS result is claimed.

This is a concrete ATL contribution opportunity: make the iFinD failure
diagnostic available to users, restore the registered-universe path, and add an
external-agent market/universe contract so the same uploaded agent can be
tested on A-shares with auditable pre-decision inputs.

## Verification

The compact result manifest is
[`exploratory_results.json`](../atl_mps_agent/evidence/exploratory/exploratory_results.json).
The eight raw U.S. result/snapshot files are retained under
[`evidence/exploratory/us`](../atl_mps_agent/evidence/exploratory/us). The test
`atl_mps_agent/test_exploratory_results.py` recomputes file hashes, timestamp
coverage, signal validity, modeled costs, aggregate returns, and the 96/224
continuous-versus-fresh action comparison. The second hosted August run IDs
and canonical hashes are preserved in the manifest.

## Claim boundary

This matrix was frozen before the subwindow outcomes, but the aggregate U.S.
path had already been observed. The dates overlap prior evaluation data, the
windows are short, and the China provider failed. The supported conclusion is
limited to deterministic execution, strong start-state sensitivity, and a
reproducible ATL China integration blocker. It is not alpha, independent
replication, expected future profit, or proof of unique MPS value.
