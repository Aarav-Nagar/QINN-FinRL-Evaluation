"""Whole-share, low-turnover execution for the current residual-MPS agent."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .deployment_training import combined_scores, load_deployment_ensemble, trend_signal
from .features import snapshot_features


class DeploymentMPSPolicy:
    """Use next-session MPS ranks inside a validation-selected exposure policy."""

    def __init__(self, artifact_path: Path):
        self.models, self.means, self.scales, self.metadata = load_deployment_ensemble(artifact_path)
        self.profile = dict(self.metadata["selected_policy"])
        self.price_history: dict[str, list[float]] = defaultdict(list)
        self.last_date: str | None = None
        self.trading_day_index = -1
        self.latest_means: dict[str, float] = {}
        self.latest_uncertainties: dict[str, float] = {}
        self.latest_scores: dict[str, float] = {}

    def scores(self, snapshot: dict[str, Any], valid_symbols: list[str]) -> dict[str, float]:
        rows = snapshot.get("top_signals") or {}
        symbols = [symbol for symbol in valid_symbols if symbol in rows and float((rows[symbol] or {}).get("price") or 0.0) > 0.0]
        if not symbols:
            return {}
        raw = np.vstack([snapshot_features(snapshot, symbol, self.price_history) for symbol in symbols])
        standardized = np.clip((raw - self.means) / self.scales, -8.0, 8.0)
        tensor = torch.as_tensor(standardized, dtype=torch.float32)
        with torch.no_grad():
            members = np.vstack([model(tensor).cpu().numpy() for model in self.models])
        means = members.mean(axis=0)
        uncertainties = members.std(axis=0, ddof=0)
        self.latest_means = dict(zip(symbols, means.astype(float)))
        self.latest_uncertainties = dict(zip(symbols, uncertainties.astype(float)))
        self.latest_scores = combined_scores(self.latest_means, self.latest_uncertainties, rows, self.profile)
        return dict(self.latest_scores)

    @staticmethod
    def _holding_quantity(holding: dict[str, Any]) -> int:
        return int(float(holding.get("shares", holding.get("quantity", 0.0)) or 0.0))

    @staticmethod
    def _holding_price(holding: dict[str, Any]) -> float:
        return float(holding.get("current_price") or holding.get("average_price") or 0.0)

    def decide(self, snapshot: dict[str, Any], valid_symbols: list[str]) -> list[dict[str, Any]]:
        timestamp = str(snapshot.get("timestamp", ""))
        try:
            current_date = datetime.fromisoformat(timestamp).date().isoformat()
        except ValueError:
            current_date = timestamp[:10]
        is_new_date = current_date != self.last_date
        if is_new_date:
            self.trading_day_index += 1
            self.last_date = current_date
        scores = self.scores(snapshot, valid_symbols)
        rows = snapshot.get("top_signals") or {}
        holdings = snapshot.get("current_holdings") or {}
        portfolio = snapshot.get("portfolio") or {}
        cash = float(portfolio.get("cash") or 0.0)
        equity = max(float(portfolio.get("total_equity") or cash or 1.0), 1.0)
        rebalance = is_new_date and self.trading_day_index % int(self.profile["rebalance_days"]) == 0

        if not rebalance or not scores:
            self._update_history(rows)
            symbol = max(scores, key=scores.get) if scores else (valid_symbols[0] if valid_symbols else "AAPL")
            return [self._action("hold", symbol, 0, "Between scheduled low-turnover rebalances")]

        market_trend = float(np.median([trend_signal(row or {}) for row in rows.values()])) if rows else 0.0
        ranked = sorted(scores, key=lambda symbol: (scores[symbol], symbol), reverse=True)
        maximum = int(self.profile["max_positions"])
        allocation = equity / max(maximum, 1)
        targets: list[str] = []
        if not bool(self.profile["positive_trend_gate"]) or market_trend > 0.0:
            targets = [symbol for symbol in ranked if float((rows[symbol] or {}).get("price") or 0.0) <= allocation][:maximum]

        held_quantities = {symbol: self._holding_quantity(holding or {}) for symbol, holding in holdings.items()}
        desired: dict[str, int] = {}
        for symbol in targets:
            price = float((rows[symbol] or {}).get("price") or 0.0)
            desired[symbol] = int(allocation // price) if price > 0.0 else 0

        actions: list[dict[str, Any]] = []
        projected_cash = cash
        # Sell removals and excess shares first so their proceeds can fund the
        # subsequent buys in the same ordered ATL decision batch.
        for symbol, current in held_quantities.items():
            target_quantity = desired.get(symbol, 0)
            quantity = max(0, current - target_quantity)
            if quantity <= 0:
                continue
            price = float((rows.get(symbol) or {}).get("price") or self._holding_price(holdings[symbol] or {}))
            if price <= 0.0:
                continue
            projected_cash += quantity * price * 0.999
            actions.append(self._action("sell", symbol, quantity, "Scheduled whole-share rebalance reduced exposure"))

        for symbol in targets:
            price = float((rows[symbol] or {}).get("price") or 0.0)
            requested = max(0, desired[symbol] - held_quantities.get(symbol, 0))
            affordable = int(projected_cash // (price * 1.001)) if price > 0.0 else 0
            quantity = min(requested, affordable)
            if quantity <= 0:
                continue
            projected_cash -= quantity * price * 1.001
            actions.append(self._action("buy", symbol, quantity, "Validation-selected MPS and trend rank established exposure"))

        self._update_history(rows)
        if actions:
            return actions
        symbol = ranked[0] if ranked else (valid_symbols[0] if valid_symbols else "AAPL")
        reason = "Positive-trend gate moved the portfolio to cash" if not targets else "Whole-share targets already satisfied"
        return [self._action("hold", symbol, 0, reason)]

    def _update_history(self, rows: dict[str, Any]) -> None:
        for symbol, row in rows.items():
            price = float((row or {}).get("price") or 0.0)
            if price > 0.0:
                self.price_history[symbol].append(price)

    def _action(self, action: str, symbol: str, quantity: int, reason: str) -> dict[str, Any]:
        score = float(self.latest_scores.get(symbol, 0.5))
        mean = float(self.latest_means.get(symbol, 0.0))
        uncertainty = float(self.latest_uncertainties.get(symbol, 0.0))
        confidence = min(0.95, max(0.50, 0.50 + 0.45 * abs(score - 0.5) * 2.0))
        return {
            "action": action,
            "symbol": symbol,
            "confidence": round(confidence, 4),
            "reasoning": (
                f"{reason}; combined rank={score:.4f}, next-session MPS mean={mean:.5f}pp, "
                f"ensemble uncertainty={uncertainty:.5f}pp, policy={self.profile['name']}"
            ),
            "position_size": int(quantity),
            "stop_loss_price": None,
            "take_profit_price": None,
        }
