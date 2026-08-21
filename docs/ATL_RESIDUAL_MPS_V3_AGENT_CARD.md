# Agent Card: Aarav Residual MPS Ensemble v3

## Model

- Classical residual matrix-product-state deep ensemble
- Five bond-dimension-5 members
- 586 parameters per member; 2,930 total
- Linear residual path and rank-aware training
- Ensemble lower-confidence-bound decisions
- No quantum hardware

## Intended use

Historical Agentic Trading Lab research and reproducible comparison with a
matched ANN ensemble. The agent uses ATL safe-trading backtests only. Live
trading is disabled.

## Risk behavior

New entries require positive validation net edge after modeled cost. The score
must also clear an uncertainty-adjusted lower bound. Execution adds smoothing,
a four-hour minimum hold, three-hour cooldown, three-position maximum, 25%
position cap, and 20% per-decision turnover cap.

The published artifact abstains because its validation lower bound did not
clear modeled transaction cost.

## Limitations

The fresh test covers one recent six-week regime and ATL exposes a changing
`top_signals` candidate set. The MPS did not outperform its matched ANN on
fresh prediction metrics. The zero-return deployment result reflects deliberate
cash abstention and should not be interpreted as alpha. This is a self-reported
research system, not investment advice or validation for real capital.
