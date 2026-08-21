# Agent Card: Aarav Residual MPS Ensemble

## Model

- Classical residual matrix-product-state deep ensemble
- Five bond-dimension-5 members
- 586 parameters per member; 2,930 total
- Linear residual path and next-session rank-aware training
- 75% uncertainty-adjusted MPS rank plus 25% observable trend rank
- Positive-trend exposure gate and five-session rebalancing
- No quantum hardware

## Intended use

Historical Agentic Trading Lab research and reproducible comparison with a
matched ANN ensemble. The agent uses ATL safe-trading backtests only. Live
trading is disabled.

## Risk behavior

The policy was selected using a stateful, whole-share validation simulation
with 10 basis points of modeled cost per traded notional. It holds at most three
positions, trades only every fifth session, and moves to cash when the median
visible market trend is not positive. Position sizing uses actual share prices
and the $1,000 ATL allocation, so an artificial 25% cap cannot block every buy.

## Limitations

ATL exposes a changing `top_signals` candidate set, and the offline whole-share
simulator uses the most recently observed price when a held symbol temporarily
leaves that subset. A positive validation result may not persist in the frozen
July-August evaluation. The trend overlay means performance cannot be attributed
to the MPS alone; MPS-only and trend-only controls are therefore required. This
is a self-reported research system, not investment advice or validation for real
capital.
