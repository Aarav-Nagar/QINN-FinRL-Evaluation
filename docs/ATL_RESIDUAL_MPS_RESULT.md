# ATL residual-MPS ensemble result

## Current frozen evaluation

The next-session, whole-share policy was frozen at commit `7e2a926` before its
July 1-August 15 snapshots were collected. The first and only hosted evaluation
of that frozen policy was positive.

| Hosted ATL endpoint | Result |
|---|---:|
| Gross return | +1.2936% |
| Estimated return after 10 bps per traded notional | +0.9850% |
| Maximum drawdown | -2.6894% |
| Recorded trades | 11 |
| Decisions / timeout holds | 224 / 0 |
| ATL DJIA reference | +4.5710% |
| ATL buy-and-hold reference | +0.9160% |

The run ID is `ext_20260821_152912_49e74732`. ATL's hosted engine reports gross
portfolio performance. The after-cost estimate subtracts $3.0867 from the
$3,086.6736 total recorded buy/sell notional, using the 10-basis-point cost
declared before evaluation. It is therefore an explicit estimate, not an ATL
native net-return field.

The architecture remained the validation-selected 75% uncertainty-adjusted MPS
rank, 25% observable trend rank, positive-trend gate, three whole-share slots,
and five-session rebalance. June validation was +3.5011% after modeled cost;
the hosted result above is the untouched evaluation.

## Matched prediction comparison

| Untouched next-session metric | Residual MPS | Matched ANN |
|---|---:|---:|
| Parameters per member | 586 | 586 |
| MSE, squared percentage points | 5.5131 | 5.6828 |
| MAE, percentage points | 1.6206 | 1.6583 |
| Directional accuracy | 73.30% | 72.51% |
| Within-timestamp rank correlation | 0.2028 | 0.2005 |

The MPS was modestly better than the matched ANN on these four predictive
metrics. However, the approximate offline portfolios selected the same holdings
for the combined MPS/ANN systems, and the MPS-only control lost 7.0961% after
modeled cost. The evidence therefore does not support a claim that the MPS
component created the hosted profit.

## Replay discrepancy and control results

The offline stateful replay returned -3.5493% for the selected system, while the
hosted ATL engine returned +1.2936% gross. The replay cannot update a held
symbol's price when it leaves ATL's changing `top_signals` subset; the hosted
engine retains that information through `current_holdings`. The replay also
generated 21 trades versus ATL's 11. It is retained as a diagnostic and is not
silently substituted for the authoritative platform result.

| Approximate offline system | Return after modeled cost |
|---|---:|
| Selected 75% MPS + 25% trend | -3.5493% |
| Matched ANN + trend | -3.5493% |
| MPS-only rank | -7.0961% |
| Trend-only | +3.0857% |
| Fixed initial trend basket | +5.1581% |
| Cash | 0.0000% |

This is a useful weakness finding: predictive improvements did not reliably
translate into portfolio ranking value, and the offline evaluator needs a full
held-symbol price feed before it can serve as an execution-faithful oracle.

## Supported conclusion

The current agent achieved the requested positive historical ATL result and
remained positive under the prespecified transaction-cost adjustment. It also
reduced the prototype's 13 trades in one day to 11 trades across 32 trading
days, with no timeouts. It did not beat ATL's DJIA reference, and this single
window does not establish alpha, statistical significance, future profitability,
or MPS superiority.

The sections below preserve the earlier one-hour forecasting experiment and its
zero-exposure hosted result as historical evidence.

## Published ATL version

- Agent: `Aarav Residual MPS Ensemble` (`agent_63a8501e0235`).
- Model label: `residual-mps-ensemble`.
- Immutable ATL version: `3.0.0` (`agv_63e931b72e43`).
- Evidence commit: `8af8e49b535e3609ccc46fd8a6c461fddcc317f4`.
- ATL configuration hash: `b587aeba5195dc64`.
- Verification level: `self_reported`.
- Live trading: disabled.

ATL's public agent endpoint returned the upgraded name, model label, current
runtime metadata, and the 210-decision hosted run after registration. The immutable
version binds the exact architecture and evidence commit; verification remains
self-reported rather than maintainer-reviewed.

## Fresh data

- Completed fresh ATL snapshots: 210.
- Training rows: 3,233.
- Validation rows: 1,110.
- Fresh test rows: 1,509 across 180 hourly targets.
- Fresh dates: May 18-June 30, 2026.
- Non-finite features: zero.
- Degenerate features: zero.

## Predictive comparison

| Fresh-test metric | Residual-MPS ensemble | Matched ANN ensemble |
|---|---:|---:|
| Parameters per member | 586 | 586 |
| Members | 5 | 5 |
| MSE, squared percentage points | 0.160147 | 0.157780 |
| MAE, percentage points | 0.281636 | 0.279682 |
| Directional accuracy | 57.39% | 58.58% |
| Rank correlation | 0.1772 | 0.2233 |
| Mean member uncertainty | 0.01341 pp | 0.02265 pp |

The ANN remained modestly better on every primary prediction metric. The residual-MPS ensemble is a
stronger tensor-network architecture than v2, but this fresh test does not show
that it is a better forecaster than the matched ANN.

Across ten paired individual members, mean cost-adjusted return was -1.0988%
for residual MPS and -1.4058% for ANN. The mean MPS-minus-ANN difference was
+0.3070 percentage points with bootstrap 95% interval [-0.9607, 1.5258]. The
interval includes zero.

## Complete-system decision

| Validation-frozen ensemble system | Fresh return | Drawdown | Turnover |
|---|---:|---:|---:|
| Residual MPS | 0.0000% | 0.0000% | 0.00 |
| Matched ANN | -0.6985% | -1.2409% | 39.00 |

The residual-MPS validation lower bound had negative modeled net edge, so its
policy abstained. The ANN validation edge was positive, so its policy traded
and subsequently lost after costs. Thus the MPS system made the better
deployment decision on this fresh window, while its underlying predictions
were still weaker. This is evidence for uncertainty-aware deployment gating,
not MPS alpha.

## Hosted ATL verification

- Run: `ext_20260821_000552_16ce0d59`.
- Decisions: 210.
- Trades and decision timeouts: zero.
- Initial/final equity: $1,000 / $1,000.
- ATL DJIA reference: +2.1130% with -2.3746% maximum drawdown.

The hosted agent reproduced the validation-frozen abstention behavior. It did
not beat the passive DJIA reference. ATL's separate whole-share buy-and-hold
reference again bought no symbols at the $1,000 allocation and is not an
invested comparison.

## Conclusion

The current agent provides a materially better technical architecture: residual tensor and
linear paths, exact parameter matching, a cross-sectional ranking objective,
deep-ensemble uncertainty, temporal-gap correction, and frozen validation
gating. The honest scientific result remains bounded: architecture quality and
deployment safety improved, but predictive superiority did not.
