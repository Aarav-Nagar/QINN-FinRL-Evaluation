"""Leakage-safe market features for ATL hourly snapshots."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Iterable

import numpy as np


# Keep 13 inputs so the bond-dimension-4 MPS remains the prespecified
# 369-parameter model. Portfolio state belongs in policy, not the predictor.
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
    "relative_return_rank_1h",
    "cross_sectional_volatility_1h",
    "market_breadth_1h",
)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _positive(value: Any, default: float) -> float:
    number = _finite(value, default)
    return number if number > 0.0 else default


def _ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if abs(denominator) > 1e-12 else default


def _current_returns(
    signals: dict[str, Any], history: dict[str, list[float]]
) -> dict[str, float]:
    returns: dict[str, float] = {}
    for candidate, value in signals.items():
        if not isinstance(value, dict):
            continue
        price = _finite(value.get("price"))
        prior = history.get(candidate, [])
        if price > 0.0 and prior and prior[-1] > 0.0:
            returns[candidate] = price / prior[-1] - 1.0
    return returns


def snapshot_features(
    snapshot: dict[str, Any],
    symbol: str,
    price_history: dict[str, list[float]] | None = None,
) -> np.ndarray:
    """Return 13 finite market-only features available at decision time.

    ATL sometimes encodes an unavailable indicator as zero. Price-derived
    indicators therefore fall back to the current price rather than producing
    an artificial extreme ratio.
    """

    history = price_history or {}
    signals = snapshot.get("top_signals") or {}
    row = signals.get(symbol) or {}
    price = _positive(row.get("price"), 1.0)
    rsi = _finite(row.get("rsi"), 50.0)
    macd = _finite(row.get("macd"))
    macd_signal = _finite(row.get("macd_signal"))
    sma20 = _positive(row.get("sma20"), price)
    sma50 = _positive(row.get("sma50"), price)
    upper = _positive(row.get("bb_upper"), price)
    lower = _positive(row.get("bb_lower"), price)
    if upper <= lower:
        upper = lower = price

    prior = history.get(symbol, [])
    return_1h = price / prior[-1] - 1.0 if prior and prior[-1] > 0.0 else 0.0
    return_3h = price / prior[-3] - 1.0 if len(prior) >= 3 and prior[-3] > 0.0 else 0.0

    cross_returns = _current_returns(signals, history)
    observed = np.asarray(list(cross_returns.values()), dtype=np.float64)
    if symbol in cross_returns and len(observed) > 1:
        ordered = sorted(cross_returns.values())
        rank = sum(value <= cross_returns[symbol] for value in ordered) - 1
        relative_rank = 2.0 * rank / (len(ordered) - 1) - 1.0
    else:
        relative_rank = 0.0
    cross_volatility = float(observed.std(ddof=0)) if len(observed) > 1 else 0.0
    breadth = float(np.mean(observed > 0.0)) if len(observed) else 0.5

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
            relative_rank,
            cross_volatility,
            2.0 * breadth - 1.0,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def build_supervised_rows(
    snapshots: Iterable[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Build current-feature/next-hour-return pairs in chronological order."""

    ordered = sorted(list(snapshots), key=lambda item: str(item.get("timestamp", "")))
    history: dict[str, list[float]] = defaultdict(list)
    features: list[np.ndarray] = []
    targets: list[float] = []
    metadata: list[dict[str, Any]] = []
    for index, current in enumerate(ordered[:-1]):
        following = ordered[index + 1]
        current_signals = current.get("top_signals") or {}
        next_signals = following.get("top_signals") or {}
        try:
            current_time = datetime.fromisoformat(str(current.get("timestamp")))
            following_time = datetime.fromisoformat(str(following.get("timestamp")))
            horizon_hours = (following_time - current_time).total_seconds() / 3600.0
        except (TypeError, ValueError):
            horizon_hours = float("inf")
        if 0.0 < horizon_hours <= 2.0:
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
                        "horizon_hours": horizon_hours,
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


def audit_features(inputs: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a serializable feature/data-quality audit."""

    if inputs.ndim != 2 or inputs.shape[1] != len(FEATURE_NAMES):
        raise ValueError("Feature matrix does not match the declared contract")
    feature_rows = []
    for index, name in enumerate(FEATURE_NAMES):
        values = inputs[:, index]
        feature_rows.append(
            {
                "name": name,
                "mean": float(np.mean(values)),
                "standard_deviation": float(np.std(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "zero_fraction": float(np.mean(values == 0.0)),
                "nonfinite_count": int(np.sum(~np.isfinite(values))),
                "degenerate": bool(np.std(values) < 1e-8),
            }
        )
    timestamps = sorted({str(row["timestamp"]) for row in rows})
    symbols = sorted({str(row["symbol"]) for row in rows})
    return {
        "row_count": int(len(rows)),
        "timestamp_count": len(timestamps),
        "first_timestamp": timestamps[0] if timestamps else None,
        "last_timestamp": timestamps[-1] if timestamps else None,
        "symbol_count": len(symbols),
        "symbols": symbols,
        "nonfinite_total": int(np.sum(~np.isfinite(inputs))),
        "degenerate_features": [row["name"] for row in feature_rows if row["degenerate"]],
        "features": feature_rows,
    }


def audit_raw_inputs(snapshots: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Measure availability of ATL's raw indicator fields."""

    fields = ("price", "rsi", "macd", "macd_signal", "sma20", "sma50", "bb_upper", "bb_lower")
    observed = {field: 0 for field in fields}
    usable = {field: 0 for field in fields}
    row_count = 0
    for snapshot in snapshots:
        for row in (snapshot.get("top_signals") or {}).values():
            if not isinstance(row, dict):
                continue
            row_count += 1
            for field in fields:
                if field not in row:
                    continue
                observed[field] += 1
                value = _finite(row.get(field), float("nan"))
                if np.isfinite(value) and (field == "rsi" or value != 0.0) and (
                    field not in {"price", "sma20", "sma50", "bb_upper", "bb_lower"}
                    or value > 0.0
                ):
                    usable[field] += 1
    return {
        "signal_row_count": row_count,
        "fields": {
            field: {
                "observed_count": observed[field],
                "usable_count": usable[field],
                "usable_fraction": usable[field] / row_count if row_count else 0.0,
            }
            for field in fields
        },
        "note": "Nonzero is an availability proxy for MACD fields because ATL uses zero placeholders.",
    }
