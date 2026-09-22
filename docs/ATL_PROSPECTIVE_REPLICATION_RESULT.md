# ATL prospective four-week replication result

## Outcome

The preregistered August 24-September 18, 2026 ATL test is complete and the
prespecified result is **negative**. The frozen agent finished at $975.25 from
$1,000, a gross return of -2.4750%. Subtracting 10 basis points from each
recorded buy or sell notional gives an estimated final value of $973.58 and a
return of **-2.6424%**.

This result was published as observed. It does not support a claim of alpha,
statistical significance, future profitability, or unique MPS value.

## Hosted result

| Measure | Result |
|---|---:|
| Window | Aug. 24-Sep. 18, 2026 |
| Starting value | $1,000.00 |
| Gross final value | $975.25 |
| Gross return | -2.4750% |
| Estimated return after 10 bps | -2.6424% |
| Maximum drawdown | -3.3655% |
| Trades | 6 |
| Hourly decisions | 133 |
| Timeout holds | 0 |
| Recorded traded notional | $1,674.32 |

The arithmetic break-even cost is -147.82 basis points, meaning no
nonnegative transaction-cost assumption makes this window profitable.

## Exact replication and data checks

The primary run (`ext_20260922_001740_4f36e194`) and fresh exact rerun
(`ext_20260922_002036_d6ad5c88`) matched exactly on metrics, the equity curve,
all trades, all decisions, and the complete snapshot payload. Local replay of
the unchanged artifact reproduced all 133 submitted action batches in both
runs with zero mismatches. This establishes deterministic execution for this
fixed window, not an independent statistical replication.

ATL supplied 133 unique hourly observations across 19 market days, with seven
observations on every included day. The platform reported 3,990/3,990 usable
symbol-bars across all 30 DJIA names and zero dropped, missing, duplicate,
off-grid, or invalid bars. The captured candidate rows contained no invalid
prices or nonfinite model inputs. There were no timeouts.

## Matched controls

The following controls used the hosted-captured context, the frozen policy
configuration, and a uniform 10-basis-point cost estimate. These locally
simulated values are directly comparable with each other; the hosted ledger
above remains the primary endpoint.

| System | Return after modeled cost | Max drawdown | Trades |
|---|---:|---:|---:|
| Frozen MPS + trend | -2.6795% | -2.8766% | 6 |
| Matched ANN + trend | -2.6795% | -2.8766% | 6 |
| MPS only | -5.0478% | -5.0478% | 17 |
| Trend only | -2.4344% | -2.4344% | 6 |
| Fixed initial trend basket | -0.9077% | -1.6203% | 3 |
| Cash | 0.0000% | 0.0000% | 0 |

The MPS and equally sized ANN systems produced identical portfolio results.
The MPS-only control was worst, while the trend-only and fixed-basket controls
lost less. This window therefore provides no evidence that the MPS component
added trading value.

For prediction only, MPS error was slightly lower than ANN (MSE 6.6606 versus
6.6936 percentage-points squared) and rank correlation was slightly higher
(0.3516 versus 0.3428), but those small prediction differences did not change
the selected portfolio's orders or return.

## ATL references

ATL's DJIA reference returned -2.8540%, so the agent's gross return was 0.3790
percentage points less negative. ATL's nominal buy-and-hold reference returned
0%, but its own metadata says it bought zero of ten requested symbols and kept
100% cash; it is therefore not an invested buy-and-hold comparison.

## Costs and weekly concentration

| Cost per traded notional | Estimated return |
|---:|---:|
| 0 bps | -2.4750% |
| 10 bps | -2.6424% |
| 25 bps | -2.8936% |
| 50 bps | -3.3122% |
| 100 bps | -4.1493% |

Weekly dollar changes were $0.00, -$10.31, -$14.45, and $0.00. The largest
week contributed 58.36% of the full-window loss, so no single week exceeded
the total loss.

## Registration and deployment identity

The protocol was publicly committed before the window at
`4ea061a55fb45c61c0adeb8cacd3ef0f0b122e76`, then bound at
`3f759c2df9c4d32c9a5c5d1580ca7761930c6e79`. The account migration commit is
`c8c28f7c7fc2b0d6c9bf02518a52cd0e5c015b7f`. The test used account-owned ATL
agent `agent_3cc6d5aac07b`, immutable version `agv_1c48b644b71e`, and artifact
SHA-256 `a2be3a5286cf7bb72116dac8a2f23b82173ddecdb44c7c81153edbe123d999b6`.

The protocol names the earlier anonymous release `agv_c23fdf3230ad`; the later
account migration changed only ownership and registration identity, not the
artifact, features, policy, seeds, thresholds, or dates. Legacy agent
`agent_63a8501e0235` is retained only as provenance. Live trading remained off.

## Evidence

The hash-bound result is in
`atl_mps_agent/evidence/prospective/result_manifest.json`. Raw hosted results,
captured snapshots, cost and weekly tables, and the matched-control files are
stored beside it. `atl_mps_agent/test_prospective_result.py` rebuilds the
manifest from those sources and checks the registered integrity conditions.

The ATL package passed 38 tests in the fresh locked validation environment.
The machine's older default Python environment remains a documented legacy
blocker: its existing PyTorch installation fails to import because
`torch._vendor.packaging.version` is missing. It was not used for these runs or
verification.
