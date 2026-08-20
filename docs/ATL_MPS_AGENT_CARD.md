# Agent Card: Aarav MPS Signal Agent v2

## Identity

- Version: 2.0.0
- Type: connected external ATL research agent
- Architecture: classical 13-site matrix product state, bond dimension 4
- Trainable parameters: 369
- Decision frequency: hourly
- Quantum hardware: none

## Intended use

Reproducible historical evaluation of a tensor-network return signal inside
Agentic Trading Lab. The supported execution surface is ATL `safe_trading`
backtesting with simulated capital. Live trading is disabled.

## Inputs and output

The predictor consumes 13 market-only features derived from the current and
earlier ATL snapshots. It predicts the next-hour percentage-point return for
currently exposed candidates. Portfolio state is used only by risk controls.
The policy may buy, sell, hold, or abstain; the validated v2 artifact abstains
because no validation threshold had positive modeled net edge.

## Evaluation

Fit, validation, and test dates are frozen in
`ATL_MPS_AGENT_V2_PROTOCOL.md`. The primary comparison pairs ten MPS and ANN
seeds under identical 369-parameter capacity, rows, normalization, optimizer,
dates, and 10-basis-point turnover cost. Full machine-readable results are in
`atl_mps_agent/evidence/v2`.

## Risk controls

- Validation-gated entries and residual-calibrated confidence
- Signal smoothing, minimum hold, and cooldown
- Three-position maximum and 25% position cap
- 25% per-decision turnover cap
- Long-only simulated orders
- No live-trading route

## Limitations

ATL exposes a changing `top_signals` set rather than a stable full-universe
panel. Raw indicators sometimes use zero as an unavailable-data placeholder.
The evaluation covers one recent historical regime, and hourly observations
are dependent. The offline model comparison uses fractional weights, while the
hosted ATL verification uses integer shares. Results are self-reported until
independently reproduced. This agent is not investment advice and is not
validated for real capital.
