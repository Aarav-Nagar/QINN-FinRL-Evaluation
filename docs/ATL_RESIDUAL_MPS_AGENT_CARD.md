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
leaves that subset. That approximation did not reproduce the hosted ATL orders
or portfolio accounting in the frozen July-August evaluation, so hosted results
and offline diagnostics are reported separately. The trend overlay means
performance cannot be attributed to the MPS alone; MPS-only and trend-only
controls are therefore required.

The replication audit found that separately collected ATL snapshots shared all
224 timestamps and identical overlapping prices with hosted-captured snapshots,
but only 22.77% of candidate symbol sets and none of the full indicator payloads
matched. Offline comparisons must therefore use snapshots captured inside the
same hosted run context. Under that correction the selected system was positive,
but the matched ANN combination produced the same portfolio result.

## Frozen evaluation

The hosted July 1-August 16, 2026 run returned +1.2936% gross with 11 trades,
-2.6894% maximum drawdown, and zero timeouts. Subtracting the declared 10 basis
points from every recorded buy and sell notional gives an estimated +0.9850%
after-cost return. ATL's DJIA reference returned +4.5710%, so the agent did not
beat the broad platform reference. A positive historical run does not establish
alpha, statistical significance, or future profitability. This remains a
self-reported research system, not investment advice or validation for real
capital.
