# Live ATL integration result

Verified on August 20, 2026 against the hosted Agentic Trading Lab API.

## Current positive hosted result

The same single registered agent now runs the frozen next-session,
trend-gated residual-MPS policy. Its July 1-August 16, 2026 ATL run
`ext_20260821_152912_49e74732` returned +1.2936% gross with 11 trades,
-2.6894% maximum drawdown, 224 decisions, and zero timeouts. Applying the
prespecified 10-basis-point estimate to every recorded buy and sell gives
+0.9850% after estimated costs. ATL's DJIA reference returned +4.5710%.

The complete comparison, offline discrepancy, and claim boundaries are in
`docs/ATL_RESIDUAL_MPS_RESULT.md`.

## Current registered architecture

- Agent/model label: `Aarav Residual MPS Ensemble` /
  `residual-mps-ensemble`
- Registered agent ID: `agent_63a8501e0235`
- Immutable version ID: `agv_c23fdf3230ad`
- Semantic version: `4.0.0`
- Evidence code commit: `5b92dcb76f75c38a86ed00923da924b565d73771`
- ATL configuration hash: `e8a8c1f2c51de43f`
- Architecture: five-member bond-dimension-5 residual-MPS ensemble
- Parameters: 586 per member, 2,930 total
- Verification level: `self_reported`
- Live trading enabled: `false`

The existing agent was upgraded in place so its earlier v1/v2 runs remain
traceable. ATL's public agent endpoint confirmed the new name, model label,
runtime metadata, latest hosted run, and three-version run history after the
update. Full residual-MPS results and claim boundaries are in
`docs/ATL_RESIDUAL_MPS_RESULT.md`.

## Version 2 hosted verification

- Registered agent ID: `agent_63a8501e0235`
- Agent/model label: `Aarav MPS Signal Agent v2` /
  `classical-mps-bond-4-v2`
- Evaluation: April 15 through May 15, 2026
- ATL run: `ext_20260820_234121_d91eafa4`
- Decisions: 161
- Trades: 0
- Decision timeouts: 0
- Initial/final equity: $1,000.00 / $1,000.00
- Total return / maximum drawdown: 0.0000% / 0.0000%

The zero-trade outcome is the frozen validation behavior, not a failed runner:
no validation threshold showed positive modeled edge after transaction cost, so
v2 disabled entries and submitted explicit hold decisions throughout the hosted
run. ATL's native DJIA reference returned +2.6990% with -1.6561% maximum
drawdown, so v2 did not beat the passive index. ATL's whole-share buy-and-hold
reference returned 0%, but it bought zero of ten requested symbols with the
$1,000 allocation and is therefore not an invested comparison.

ATL records LLM-call and token estimates for external decision submissions.
This implementation calls no LLM API, so those platform fields are not treated
as actual model usage.

## Immutable ATL version

- Agent version ID: `agv_28edcd572581`
- Semantic version: `2.0.0`
- Evidence code commit: `fcae56fb5855f57bd852473f02d41ab91a0f22a3`
- ATL configuration hash: `7f84796f4046f0cf`
- Verification level: `self_reported`
- Live trading enabled: `false`

The ATL version points to the published commit containing the model artifact,
hashed evidence manifest, benchmark outputs, tests, protocol, and Agent Card.

## Version 1 prototype

## Registered agent

- Name: `Aarav MPS Signal Agent`
- ATL agent ID: `agent_63a8501e0235`
- Model label: `classical-mps-bond-4`
- Category: U.S. stocks
- Runtime: connected external agent
- Allocation: $1,000 simulated capital

## Training boundary

- ATL snapshot window: April 1-10, 2026 (`2026-04-01` to exclusive end
  `2026-04-11`)
- Collector run: `ext_20260820_225907_6d0ec652`
- Hourly snapshots: 49
- Chronological training rows: 308
- Chronological validation rows: 76
- MPS bond dimension: 4
- Trainable parameters: 369
- Best validation MSE: 0.385506 in squared percentage-point return units

The feature standardizer is fit on the training partition only. The evaluation
date below is later than the entire collection and training window.

## Held-out ATL run

- Evaluation window: April 15, 2026 (`2026-04-15` to exclusive end
  `2026-04-16`)
- ATL run ID: `ext_20260820_230246_378572c6`
- Initial equity: $1,000.00
- Final equity: $1,001.8249
- Total return: 0.18249%
- Maximum drawdown: -0.28869%
- Recorded trades: 13
- Hourly decisions: 7
- Decision timeouts: 0

This one-day result verifies that the MPS model can consume ATL snapshots,
produce valid decisions, execute simulated orders, and leave a traceable result.
It is not evidence of persistent alpha or superiority over another strategy.
ATL also reports estimated LLM calls, tokens, and cost for external decisions;
this agent made no LLM API calls, so those fields are not treated as actual
model-usage measurements.

## Scope boundary

The live agent is an ATL-specific classical MPS prototype. It is not the paper's
daily FinRL PPO policy and uses no quantum hardware. Its purpose is to make the
tensor-network research runnable inside ATL's current hourly external-agent
contract without mislabeling a prompt template as an MPS implementation.
