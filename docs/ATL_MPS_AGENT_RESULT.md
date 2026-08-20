# Live ATL integration result

Verified on August 20, 2026 against the hosted Agentic Trading Lab API.

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
