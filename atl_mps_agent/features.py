"""Feature engineering for ATL hourly market snapshots."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np


FEATURE_NAMES = (
    "rsi_centered",
    "macd_to_price",
    "macd_gap_to_price",
    "price_to_sma20",
    "price_to_sma50",
    "sma20_to_sma50",
    "bollinger_position",
    "bollinger_width",
    "return_1h",
    "return_3h",
    "cash_fraction",
    "position_fraction",
    "market_breadth",
)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if abs(denominator) > 1e-12 else default


def snapshot_features(
    snapshot: dict[str, Any],
    symbol: str,
    price_history: dict[str, list[float]] | None = None,
) -> np.ndarray:
    """Return 13 finite, dimensionless features for one symbol.

    Only information available in the current or earlier snapshots is used.
    """

    history = price_history or {}
    signals = snapshot.get("top_signals") or {}
    row = signals.get(symbol) or {}
    price = _finite(row.get("price"))
    rsi = _finite(row.get("rsi"), 50.0)
    macd = _finite(row.get("macd"))
    macd_signal = _finite(row.get("macd_signal"))
    sma20 = _finite(row.get("sma20"), price)
    sma50 = _finite(row.get("sma50"), price)
    upper = _finite(row.get("bb_upper"), price)
    lower = _finite(row.get("bb_lower"), price)

    prior = history.get(symbol, [])
    return_1h = _ratio(price, prior[-1], 1.0) - 1.0 if prior else 0.0
    return_3h = _ratio(price, prior[-3], 1.0) - 1.0 if len(prior) >= 3 else 0.0

    portfolio = snapshot.get("portfolio") or {}
    equity = _finite(portfolio.get("total_equity"), 1.0)
    cash = _finite(portfolio.get("cash"), equity)
    holdings = snapshot.get("current_holdings") or {}
    held = holdings.get(symbol) or {}
    shares = _finite(held.get("shares", held.get("quantity", 0.0)))

    valid_rows = [value for value in signals.values() if isinstance(value, dict)]
    breadth_values = []
    for value in valid_rows:
        value_macd = _finite(value.get("macd"))
        value_signal = _finite(value.get("macd_signal"))
        breadth_values.append(1.0 if value_macd > value_signal else 0.0)
    breadth = float(np.mean(breadth_values)) if breadth_values else 0.5

    features = np.asarray(
        [
            (rsi - 50.0) / 50.0,
            _ratio(macd, price),
            _ratio(macd - macd_signal, price),
            _ratio(price, sma20, 1.0) - 1.0,
            _ratio(price, sma50, 1.0) - 1.0,
            _ratio(sma20, sma50, 1.0) - 1.0,
            2.0 * _ratio(price - lower, upper - lower, 0.5) - 1.0,
            _ratio(upper - lower, price),
            return_1h,
            return_3h,
            _ratio(cash, equity),
            _ratio(shares * price, equity),
            2.0 * breadth - 1.0,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def build_supervised_rows(
    snapshots: Iterable[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Build current-feature/next-hour-return pairs in chronological order."""

    ordered = list(snapshots)
    history: dict[str, list[float]] = defaultdict(list)
    features: list[np.ndarray] = []
    targets: list[float] = []
    metadata: list[dict[str, Any]] = []
    for index, current in enumerate(ordered[:-1]):
        following = ordered[index + 1]
        current_signals = current.get("top_signals") or {}
        next_signals = following.get("top_signals") or {}
        for symbol in sorted(set(current_signals) & set(next_signals)):
            current_price = _finite((current_signals.get(symbol) or {}).get("price"))
            next_price = _finite((next_signals.get(symbol) or {}).get("price"))
            if current_price <= 0 or next_price <= 0:
                continue
            features.append(snapshot_features(current, symbol, history))
            targets.append(100.0 * (next_price / current_price - 1.0))
            metadata.append(
                {
                    "timestamp": current.get("timestamp"),
                    "next_timestamp": following.get("timestamp"),
                    "symbol": symbol,
                }
            )
        for symbol, row in current_signals.items():
            price = _finite((row or {}).get("price"))
            if price > 0:
                history[symbol].append(price)
    if not features:
        raise ValueError("No supervised rows could be built from the snapshots")
    return np.vstack(features), np.asarray(targets, dtype=np.float32), metadata
