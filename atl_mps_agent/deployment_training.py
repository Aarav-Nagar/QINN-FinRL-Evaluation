"""Training and portfolio-level calibration for the current ATL agent."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .features import FEATURE_NAMES, snapshot_features
from .model import ResidualMPSRegressor, parameter_count
from .training import prepare_dataset
from .v3_training import ensemble_predictions, fit_rank_aware_regressor


POLICY_PROFILES: tuple[dict[str, Any], ...] = (
    {"name": "mps_rank_daily", "model_weight": 1.0, "uncertainty_z": 0.5, "max_positions": 3, "rebalance_days": 1, "positive_trend_gate": False},
    {"name": "mps_rank_weekly", "model_weight": 1.0, "uncertainty_z": 0.5, "max_positions": 3, "rebalance_days": 5, "positive_trend_gate": False},
    {"name": "mps_trend_daily", "model_weight": 0.75, "uncertainty_z": 0.5, "max_positions": 3, "rebalance_days": 1, "positive_trend_gate": False},
    {"name": "mps_trend_weekly", "model_weight": 0.75, "uncertainty_z": 0.5, "max_positions": 3, "rebalance_days": 5, "positive_trend_gate": False},
    {"name": "balanced_weekly", "model_weight": 0.50, "uncertainty_z": 0.5, "max_positions": 3, "rebalance_days": 5, "positive_trend_gate": False},
    {"name": "gated_mps_trend_daily", "model_weight": 0.75, "uncertainty_z": 0.5, "max_positions": 3, "rebalance_days": 1, "positive_trend_gate": True},
    {"name": "gated_mps_trend_weekly", "model_weight": 0.75, "uncertainty_z": 0.5, "max_positions": 3, "rebalance_days": 5, "positive_trend_gate": True},
)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def trend_signal(row: dict[str, Any]) -> float:
    """A scale-free medium-term trend score available at decision time."""

    price = _finite(row.get("price"))
    if price <= 0.0:
        return 0.0
    sma20 = _finite(row.get("sma20"), price) or price
    sma50 = _finite(row.get("sma50"), price) or price
    macd = _finite(row.get("macd"))
    return (price / sma50 - 1.0) + (sma20 / sma50 - 1.0) + 0.25 * macd / price


def _unit_ranks(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values, key=lambda symbol: (values[symbol], symbol))
    if len(ordered) == 1:
        return {ordered[0]: 0.5}
    return {symbol: index / (len(ordered) - 1) for index, symbol in enumerate(ordered)}


def combined_scores(
    means: dict[str, float],
    uncertainties: dict[str, float],
    rows: dict[str, dict[str, Any]],
    profile: dict[str, Any],
) -> dict[str, float]:
    """Blend cross-sectional MPS confidence ranks with observable trend ranks."""

    z = float(profile["uncertainty_z"])
    lower = {symbol: means[symbol] - z * uncertainties.get(symbol, 0.0) for symbol in means}
    trends = {symbol: trend_signal(rows.get(symbol) or {}) for symbol in means}
    model_ranks = _unit_ranks(lower)
    trend_ranks = _unit_ranks(trends)
    weight = float(profile["model_weight"])
    return {
        symbol: weight * model_ranks[symbol] + (1.0 - weight) * trend_ranks[symbol]
        for symbol in means
    }


def predict_snapshots(
    models: list[torch.nn.Module],
    means: np.ndarray,
    scales: np.ndarray,
    snapshots: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Generate runtime-equivalent predictions for every visible symbol."""

    history: dict[str, list[float]] = defaultdict(list)
    output: dict[str, dict[str, dict[str, float]]] = {}
    for snapshot in sorted(snapshots, key=lambda item: str(item.get("timestamp", ""))):
        timestamp = str(snapshot.get("timestamp"))
        rows = snapshot.get("top_signals") or {}
        symbols = [symbol for symbol, row in rows.items() if _finite((row or {}).get("price")) > 0.0]
        if symbols:
            raw = np.vstack([snapshot_features(snapshot, symbol, history) for symbol in symbols])
            standardized = np.clip((raw - means) / scales, -8.0, 8.0)
            prediction_mean, uncertainty = ensemble_predictions(models, standardized)
            output[timestamp] = {
                symbol: {"mean_pp": float(mean), "uncertainty_pp": float(std)}
                for symbol, mean, std in zip(symbols, prediction_mean, uncertainty)
            }
        for symbol, row in rows.items():
            price = _finite((row or {}).get("price"))
            if price > 0.0:
                history[symbol].append(price)
    return output


def simulate_whole_share_policy(
    snapshots: list[dict[str, Any]],
    prediction_map: dict[str, dict[str, dict[str, float]]],
    profile: dict[str, Any],
    *,
    start: str,
    end: str,
    initial_cash: float = 1000.0,
    transaction_cost_bps: float = 10.0,
) -> dict[str, Any]:
    """Approximate ATL execution with whole shares and stateful holdings."""

    first_by_day: dict[str, dict[str, Any]] = {}
    for snapshot in sorted(snapshots, key=lambda item: str(item.get("timestamp", ""))):
        timestamp = str(snapshot.get("timestamp", ""))
        if start <= timestamp <= end:
            first_by_day.setdefault(timestamp[:10], snapshot)
    cash = float(initial_cash)
    holdings: dict[str, int] = {}
    last_prices: dict[str, float] = {}
    peak = float(initial_cash)
    max_drawdown = 0.0
    cumulative_turnover = 0.0
    cumulative_cost = 0.0
    trade_count = 0
    invested_days = 0
    equity_curve: list[dict[str, Any]] = []
    cost_rate = transaction_cost_bps / 10_000.0

    for day_index, (day, snapshot) in enumerate(sorted(first_by_day.items())):
        rows = snapshot.get("top_signals") or {}
        timestamp = str(snapshot.get("timestamp"))
        for symbol, row in rows.items():
            price = _finite((row or {}).get("price"))
            if price > 0.0:
                last_prices[symbol] = price
        equity = cash + sum(quantity * last_prices.get(symbol, 0.0) for symbol, quantity in holdings.items())
        if holdings:
            invested_days += 1
        if day_index % int(profile["rebalance_days"]) == 0:
            visible_predictions = prediction_map.get(timestamp) or {}
            model_means = {symbol: values["mean_pp"] for symbol, values in visible_predictions.items() if symbol in rows}
            uncertainties = {symbol: values["uncertainty_pp"] for symbol, values in visible_predictions.items() if symbol in rows}
            scores = combined_scores(model_means, uncertainties, rows, profile)
            market_trend = float(np.median([trend_signal(row or {}) for row in rows.values()])) if rows else 0.0
            ranked = sorted(scores, key=lambda symbol: (scores[symbol], symbol), reverse=True)
            maximum = int(profile["max_positions"])
            target_symbols: list[str] = []
            if not bool(profile["positive_trend_gate"]) or market_trend > 0.0:
                allocation = equity / max(maximum, 1)
                target_symbols = [symbol for symbol in ranked if last_prices.get(symbol, float("inf")) <= allocation][:maximum]

            # Sell removals first, using the most recently observed price when a
            # held symbol is temporarily absent from ATL's top-signal subset.
            for symbol in list(holdings):
                if symbol in target_symbols:
                    continue
                price = last_prices.get(symbol, 0.0)
                quantity = holdings.pop(symbol)
                if price <= 0.0 or quantity <= 0:
                    continue
                gross = quantity * price
                cost = gross * cost_rate
                cash += gross - cost
                cumulative_turnover += gross
                cumulative_cost += cost
                trade_count += 1

            equity = cash + sum(quantity * last_prices.get(symbol, 0.0) for symbol, quantity in holdings.items())
            allocation = equity / max(len(target_symbols), 1)
            desired = {symbol: int(allocation // last_prices[symbol]) for symbol in target_symbols}
            # Reduce oversized retained positions before funding new entries.
            for symbol in target_symbols:
                current = holdings.get(symbol, 0)
                if current <= desired[symbol]:
                    continue
                quantity = current - desired[symbol]
                gross = quantity * last_prices[symbol]
                cost = gross * cost_rate
                holdings[symbol] = desired[symbol]
                cash += gross - cost
                cumulative_turnover += gross
                cumulative_cost += cost
                trade_count += 1
            for symbol in target_symbols:
                price = last_prices[symbol]
                requested = max(0, desired[symbol] - holdings.get(symbol, 0))
                affordable = int(cash // (price * (1.0 + cost_rate)))
                quantity = min(requested, affordable)
                if quantity <= 0:
                    continue
                gross = quantity * price
                cost = gross * cost_rate
                cash -= gross + cost
                holdings[symbol] = holdings.get(symbol, 0) + quantity
                cumulative_turnover += gross
                cumulative_cost += cost
                trade_count += 1

        equity = cash + sum(quantity * last_prices.get(symbol, 0.0) for symbol, quantity in holdings.items())
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity / peak - 1.0)
        equity_curve.append({"date": day, "equity": equity})

    final_equity = equity_curve[-1]["equity"] if equity_curve else initial_cash
    day_count = max(len(first_by_day), 1)
    return {
        "initial_equity": initial_cash,
        "final_equity": final_equity,
        "total_return_pct": 100.0 * (final_equity / initial_cash - 1.0),
        "maximum_drawdown_pct": 100.0 * max_drawdown,
        "trade_count": trade_count,
        "cumulative_turnover_multiple": cumulative_turnover / initial_cash,
        "modeled_cost_pct_initial": 100.0 * cumulative_cost / initial_cash,
        "invested_day_fraction": invested_days / day_count,
        "decision_days": len(first_by_day),
        "ending_holdings": holdings,
        "equity_curve": equity_curve,
    }


def train_deployment_ensemble(
    snapshots: list[dict[str, Any]],
    artifact_path: Path,
    *,
    train_end: str,
    validation_start: str,
    validation_end: str,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    epochs: int = 250,
    patience: int = 30,
    transaction_cost_bps: float = 10.0,
) -> dict[str, Any]:
    """Fit a next-session MPS ensemble and select policy only on validation."""

    data = prepare_dataset(
        snapshots,
        train_end=train_end,
        validation_end=validation_end,
        reset_history_on_gap=False,
        horizon_steps=7,
        max_horizon_hours=96.0,
    )
    models: list[ResidualMPSRegressor] = []
    histories: dict[str, list[dict[str, float]]] = {}
    for seed in seeds:
        model, history = fit_rank_aware_regressor(
            lambda seed=seed: ResidualMPSRegressor(13, seed=seed),
            data,
            seed=seed,
            epochs=epochs,
            patience=patience,
            rank_weight=0.15,
        )
        models.append(model)
        histories[str(seed)] = history
    predictions = predict_snapshots(models, data["means"], data["scales"], snapshots)
    profile_results = []
    for profile in POLICY_PROFILES:
        metrics = simulate_whole_share_policy(
            snapshots,
            predictions,
            profile,
            start=validation_start,
            end=validation_end,
            transaction_cost_bps=transaction_cost_bps,
        )
        utility = (
            metrics["total_return_pct"]
            - 0.20 * abs(metrics["maximum_drawdown_pct"])
            - 0.02 * metrics["cumulative_turnover_multiple"]
        )
        profile_results.append({"profile": dict(profile), "metrics": {key: value for key, value in metrics.items() if key != "equity_curve"}, "selection_utility": utility})
    eligible = [row for row in profile_results if row["metrics"]["total_return_pct"] > 0.0 and row["metrics"]["trade_count"] >= 3]
    pool = eligible or profile_results
    selected = max(pool, key=lambda row: (row["selection_utility"], -row["metrics"]["cumulative_turnover_multiple"], row["profile"]["name"]))
    canonical = json.dumps(snapshots, sort_keys=True, separators=(",", ":")).encode()
    artifact = {
        "schema_version": 4,
        "model_type": "next_session_residual_mps_ensemble",
        "feature_names": list(FEATURE_NAMES),
        "bond_dimension": 5,
        "parameter_count_per_member": parameter_count(models[0]),
        "member_count": len(models),
        "ensemble_parameter_count": sum(parameter_count(model) for model in models),
        "seeds": list(seeds),
        "feature_means": data["means"].tolist(),
        "feature_scales": data["scales"].tolist(),
        "state_dicts": [model.state_dict() for model in models],
        "training_rows": int(data["train_mask"].sum()),
        "validation_rows": int(data["validation_mask"].sum()),
        "train_end": train_end,
        "validation_start": validation_start,
        "validation_end": validation_end,
        "target_horizon_steps": 7,
        "target_max_horizon_hours": 96.0,
        "selected_policy": selected["profile"],
        "selected_validation_metrics": selected["metrics"],
        "validation_profile_results": profile_results,
        "transaction_cost_bps": transaction_cost_bps,
        "feature_audit": data["audit"],
        "dataset_sha256": hashlib.sha256(canonical).hexdigest(),
        "training_histories": histories,
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, artifact_path)
    summary = {key: value for key, value in artifact.items() if key not in {"state_dicts", "training_histories"}}
    summary["epochs_completed_by_seed"] = {seed: len(history) for seed, history in histories.items()}
    summary["artifact_sha256"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    artifact_path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8", newline="\n")
    return summary


def load_deployment_ensemble(
    path: Path,
) -> tuple[list[ResidualMPSRegressor], np.ndarray, np.ndarray, dict[str, Any]]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("schema_version") != 4:
        raise ValueError("Expected the current ATL deployment artifact")
    if artifact.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Artifact feature contract does not match the current agent")
    seeds = [int(seed) for seed in artifact["seeds"]]
    models = [ResidualMPSRegressor(13, seed=seed) for seed in seeds]
    for model, state in zip(models, artifact["state_dicts"]):
        model.load_state_dict(state)
        model.eval()
    metadata = {key: value for key, value in artifact.items() if key != "state_dicts"}
    return models, np.asarray(artifact["feature_means"], dtype=np.float32), np.asarray(artifact["feature_scales"], dtype=np.float32), metadata
