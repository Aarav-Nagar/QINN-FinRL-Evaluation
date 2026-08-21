# ATL residual-MPS ensemble result

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
