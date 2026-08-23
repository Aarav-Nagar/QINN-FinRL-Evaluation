# Account-owned ATL deployment result

## Identity

- Agent: `Aarav Residual MPS Ensemble`
- Account-owned ATL ID: `agent_3cc6d5aac07b`
- Model label: `residual-mps-ensemble`
- Immutable version: `agv_1c48b644b71e` (`1.0.0`)
- Configuration hash: `56941595c89341bf`
- Code commit at registration: `3f759c2df9c4d32c9a5c5d1580ca7761930c6e79`
- Live trading: disabled
- Verification level: self-reported

ATL displayed this agent under Aarav's authenticated account in **My Agents**.
The access key is stored only in the ignored local `.atl` directory and is not
included in this repository.

## Account-linked hosted run

| Metric | Result |
|---|---:|
| Run ID | `ext_20260822_193243_03d90180` |
| Window | July 1 to exclusive August 16, 2026 |
| Initial equity | $1,000.00 |
| Final equity | $1,012.9364 |
| Gross return | +1.2936% |
| Estimated return after 10 bps per traded notional | +0.9850% |
| Maximum drawdown | -2.6894% |
| Trades | 11 |
| Decisions / timeout holds | 224 / 0 |

The full hosted response is stored at
`atl_mps_agent/evidence/account/account_hosted_run.json` with SHA-256
`7c535f733a76a59b46fbb2c2ee36608335ed10f86871364baaa06fff1aa5aa77`.
The frozen model artifact remains
`a2be3a5286cf7bb72116dac8a2f23b82173ddecdb44c7c81153edbe123d999b6`.

## Interpretation

The account-owned run exactly reproduces the original anonymous-session run's
headline metrics. This verifies that moving ownership did not change the frozen
policy or ATL execution result. It is another deterministic run over the same
historical market path, not an independent performance sample and not evidence
of alpha, MPS superiority, or future profitability.

The original anonymous agent remains in the replication audit so its
preregistered run history is not rewritten. New prospective tests should use
the account-owned agent.
