"""Cost-aware policy turning calibrated MPS signals into ATL actions."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .features import snapshot_features
from .training import load_artifact


class MPSPolicy:
    """Long-only MPS policy with abstention, persistence, and turnover limits."""

    def __init__(
        self,
        artifact_path: Path,
        *,
        max_positions: int = 3,
        min_hold_steps: int = 3,
        cooldown_steps: int = 2,
        max_step_turnover: float = 0.25,
        smoothing_alpha: float = 0.35,
    ):
        self.model, self.means, self.scales, self.metadata = load_artifact(artifact_path)
        self.max_positions = max_positions
        self.min_hold_steps = min_hold_steps
        self.cooldown_steps = cooldown_steps
        self.max_step_turnover = max_step_turnover
        self.smoothing_alpha = smoothing_alpha
        calibration = self.metadata.get("calibration") or {}
        self.entry_threshold = float(calibration.get("trade_threshold_pp", 0.10))
        self.entries_enabled = not bool(
            calibration.get("abstain_without_positive_validation_edge", False)
        )
        self.exit_threshold = -0.25 * self.entry_threshold
        self.residual_scale = max(float(calibration.get("residual_rmse_pp", 1.0)), 1e-6)
        self.price_history: dict[str, list[float]] = defaultdict(list)
        self.smoothed_scores: dict[str, float] = {}
        self.holding_steps: dict[str, int] = defaultdict(int)
        self.last_trade_step: dict[str, int] = defaultdict(lambda: -10_000)
        self.decision_step = 0

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
        current = dict(zip(symbols, predictions.cpu().numpy().astype(float)))
        for symbol, score in current.items():
            previous = self.smoothed_scores.get(symbol, score)
            self.smoothed_scores[symbol] = (
                self.smoothing_alpha * score + (1.0 - self.smoothing_alpha) * previous
            )
        return {symbol: self.smoothed_scores[symbol] for symbol in symbols}

    def decide(self, snapshot: dict[str, Any], valid_symbols: list[str]) -> list[dict[str, Any]]:
        self.decision_step += 1
        signals = snapshot.get("top_signals") or {}
        holdings = snapshot.get("current_holdings") or {}
        portfolio = snapshot.get("portfolio") or {}
        cash = float(portfolio.get("cash") or 0.0)
        equity = max(float(portfolio.get("total_equity") or cash or 1.0), 1.0)
        scores = self.scores(snapshot, valid_symbols)
        ranked = sorted(scores, key=scores.get, reverse=True)
        held_symbols = []
        for symbol, held in holdings.items():
            shares = float((held or {}).get("shares", (held or {}).get("quantity", 0.0)) or 0.0)
            if shares > 0:
                held_symbols.append(symbol)
                self.holding_steps[symbol] += 1
        for symbol in list(self.holding_steps):
            if symbol not in held_symbols:
                self.holding_steps.pop(symbol, None)

        eligible = (
            [symbol for symbol in ranked if scores[symbol] >= self.entry_threshold]
            if self.entries_enabled
            else []
        )
        target = set(eligible[: self.max_positions])
        actions: list[dict[str, Any]] = []
        turnover_budget = self.max_step_turnover * equity
        planned_sells = 0

        for symbol in held_symbols:
            score = scores.get(symbol, self.smoothed_scores.get(symbol, 0.0))
            minimum_held = self.holding_steps[symbol] >= self.min_hold_steps
            should_exit = score < self.exit_threshold or symbol not in target
            if not (minimum_held and should_exit and turnover_budget > 0.0):
                continue
            held = holdings[symbol] or {}
            shares = int(float(held.get("shares", held.get("quantity", 0.0)) or 0.0))
            row = signals.get(symbol) or {}
            price = float(
                row.get("price")
                or held.get("current_price")
                or held.get("average_price")
                or 0.0
            )
            if shares <= 0:
                continue
            if price > 0.0:
                shares = min(shares, max(1, int(turnover_budget // price)))
                turnover_budget -= shares * price
            actions.append(
                self._action(
                    "sell",
                    symbol,
                    shares,
                    score,
                    "Calibrated MPS edge exited after minimum hold",
                )
            )
            self.last_trade_step[symbol] = self.decision_step
            planned_sells += 1

        open_slots = max(0, self.max_positions - len(held_symbols) + planned_sells)
        per_position_cap = 0.25 * equity
        for symbol in eligible:
            if open_slots <= 0 or symbol in held_symbols or turnover_budget <= 0.0:
                continue
            if self.decision_step - self.last_trade_step[symbol] <= self.cooldown_steps:
                continue
            row = signals.get(symbol) or {}
            price = float(row.get("price") or 0.0)
            if price <= 0:
                continue
            budget = min(per_position_cap, cash, turnover_budget)
            quantity = int(budget // price)
            if quantity <= 0:
                continue
            actions.append(
                self._action(
                    "buy",
                    symbol,
                    quantity,
                    scores[symbol],
                    "MPS edge clears validation-calibrated cost threshold",
                )
            )
            notional = quantity * price
            cash -= notional
            turnover_budget -= notional
            self.last_trade_step[symbol] = self.decision_step
            open_slots -= 1

        if not actions:
            fallback = ranked[0] if ranked else (valid_symbols[0] if valid_symbols else "AAPL")
            score = scores.get(fallback, 0.0)
            if not self.entries_enabled:
                reason = "Abstained: validation found no positive modeled net edge"
            elif score < self.entry_threshold:
                reason = "Abstained: no signal cleared cost and persistence controls"
            else:
                reason = "No risk-bounded rebalance required"
            actions.append(self._action("hold", fallback, 0, score, reason))

        for symbol, row in signals.items():
            price = float((row or {}).get("price") or 0.0)
            if price > 0:
                self.price_history[symbol].append(price)
        return actions

    def _confidence(self, score: float) -> float:
        probability = 0.5 + 0.5 * math.erf(abs(float(score)) / (self.residual_scale * math.sqrt(2.0)))
        return min(0.95, max(0.50, probability))

    def _action(
        self, action: str, symbol: str, quantity: int, score: float, reason: str
    ) -> dict[str, Any]:
        return {
            "action": action,
            "symbol": symbol,
            "confidence": round(self._confidence(score), 4),
            "reasoning": (
                f"{reason}; smoothed classical MPS score={score:.5f}pp, "
                f"entry threshold={self.entry_threshold:.5f}pp"
            ),
            "position_size": int(quantity),
            "stop_loss_price": None,
            "take_profit_price": None,
        }
