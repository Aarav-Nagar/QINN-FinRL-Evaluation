"""Decision policy that turns MPS signals into ATL action payloads."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .features import snapshot_features
from .training import load_artifact


class MPSPolicy:
    """Risk-bounded long-only policy driven by MPS next-hour return scores."""

    def __init__(self, artifact_path: Path, *, max_positions: int = 3):
        self.model, self.means, self.scales, self.metadata = load_artifact(artifact_path)
        self.max_positions = max_positions
        self.price_history: dict[str, list[float]] = defaultdict(list)

    def scores(self, snapshot: dict[str, Any], valid_symbols: list[str]) -> dict[str, float]:
        symbols = [
            symbol
            for symbol in valid_symbols
            if symbol in (snapshot.get("top_signals") or {})
        ]
        if not symbols:
            return {}
        raw = np.vstack(
            [snapshot_features(snapshot, symbol, self.price_history) for symbol in symbols]
        )
        standardized = np.clip((raw - self.means) / self.scales, -8.0, 8.0)
        with torch.no_grad():
            predictions = self.model(torch.as_tensor(standardized, dtype=torch.float32))
        return dict(zip(symbols, predictions.cpu().numpy().astype(float)))

    def decide(self, snapshot: dict[str, Any], valid_symbols: list[str]) -> list[dict[str, Any]]:
        signals = snapshot.get("top_signals") or {}
        holdings = snapshot.get("current_holdings") or {}
        portfolio = snapshot.get("portfolio") or {}
        cash = float(portfolio.get("cash") or 0.0)
        equity = max(float(portfolio.get("total_equity") or cash or 1.0), 1.0)
        scores = self.scores(snapshot, valid_symbols)
        ranked = sorted(scores, key=scores.get, reverse=True)
        held_symbols = [
            symbol
            for symbol, held in holdings.items()
            if float((held or {}).get("shares", (held or {}).get("quantity", 0.0)) or 0.0) > 0
        ]
        actions: list[dict[str, Any]] = []

        keep = set(ranked[: self.max_positions])
        for symbol in held_symbols:
            shares = int(float((holdings[symbol] or {}).get("shares", 0) or 0))
            if symbol not in keep and shares > 0:
                actions.append(
                    self._action(
                        "sell",
                        symbol,
                        shares,
                        scores.get(symbol, 0.0),
                        "MPS score left the top-ranked risk-limited set",
                    )
                )

        open_slots = max(0, self.max_positions - len(held_symbols))
        per_position_budget = min(0.25 * equity, cash / max(1, open_slots))
        for symbol in ranked:
            if open_slots <= 0 or symbol in held_symbols:
                continue
            row = signals.get(symbol) or {}
            price = float(row.get("price") or 0.0)
            if price <= 0:
                continue
            quantity = int(per_position_budget // price)
            if quantity <= 0:
                continue
            actions.append(
                self._action(
                    "buy",
                    symbol,
                    quantity,
                    scores[symbol],
                    "Highest MPS next-hour score within 25% position cap",
                )
            )
            cash -= quantity * price
            open_slots -= 1

        if not actions:
            fallback = ranked[0] if ranked else (valid_symbols[0] if valid_symbols else "AAPL")
            actions.append(
                self._action(
                    "hold",
                    fallback,
                    0,
                    scores.get(fallback, 0.0),
                    "No risk-bounded rebalance required from current MPS ranking",
                )
            )

        for symbol, row in signals.items():
            price = float((row or {}).get("price") or 0.0)
            if price > 0:
                self.price_history[symbol].append(price)
        return actions

    @staticmethod
    def _action(
        action: str, symbol: str, quantity: int, score: float, reason: str
    ) -> dict[str, Any]:
        confidence = min(0.95, max(0.05, 0.5 + 0.1 * abs(float(score))))
        return {
            "action": action,
            "symbol": symbol,
            "confidence": round(confidence, 4),
            "reasoning": f"{reason}; classical MPS score={score:.5f}",
            "position_size": int(quantity),
            "stop_loss_price": None,
            "take_profit_price": None,
        }
