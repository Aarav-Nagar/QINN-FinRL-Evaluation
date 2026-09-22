"""Matched, cost-aware ATL benchmark for MPS v2 and classical controls."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .features import FEATURE_NAMES
from .model import MPSRegressor, MatchedANNRegressor, parameter_count
from .training import calibrate_predictions, fit_torch_regressor, prepare_dataset


def _predict(model: torch.nn.Module, values: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.as_tensor(values, dtype=torch.float32)).cpu().numpy()


def _signal_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    rank_ic = 0.0
    if np.std(predictions) > 0.0 and np.std(targets) > 0.0:
        rank_ic = float(np.corrcoef(np.argsort(np.argsort(predictions)), np.argsort(np.argsort(targets)))[0, 1])
    return {
        "mse_pp2": float(np.mean((predictions - targets) ** 2)),
        "mae_pp": float(np.mean(np.abs(predictions - targets))),
        "directional_accuracy": float(np.mean(np.sign(predictions) == np.sign(targets))),
        "rank_ic": rank_ic,
    }


def _portfolio_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    rows: list[dict[str, Any]],
    mask: np.ndarray,
    *,
    threshold_pp: float,
    transaction_cost_bps: float,
    max_positions: int = 3,
    equal_weight_all: bool = False,
) -> tuple[dict[str, float], list[float]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index in np.flatnonzero(mask):
        grouped[str(rows[index]["timestamp"])].append(int(index))
    previous: dict[str, float] = {}
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    cumulative_turnover = 0.0
    cumulative_cost = 0.0
    returns: list[float] = []
    active = 0
    for timestamp in sorted(grouped):
        indices = grouped[timestamp]
        if equal_weight_all:
            selected = indices
        else:
            selected = [index for index in indices if predictions[index] >= threshold_pp]
            selected = sorted(selected, key=lambda index: predictions[index], reverse=True)[:max_positions]
        weights = (
            {str(rows[index]["symbol"]): 1.0 / len(selected) for index in selected}
            if selected
            else {}
        )
        if weights:
            active += 1
        symbols = set(previous) | set(weights)
        turnover = sum(abs(weights.get(symbol, 0.0) - previous.get(symbol, 0.0)) for symbol in symbols)
        gross = float(
            sum(weights[str(rows[index]["symbol"])] * float(targets[index]) / 100.0 for index in selected)
        )
        cost = turnover * transaction_cost_bps / 10_000.0
        net = gross - cost
        equity *= 1.0 + net
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity / peak - 1.0)
        cumulative_turnover += turnover
        cumulative_cost += cost
        returns.append(net)
        previous = weights
    count = max(len(grouped), 1)
    return {
        "total_return_pct": 100.0 * (equity - 1.0),
        "maximum_drawdown_pct": 100.0 * max_drawdown,
        "cumulative_turnover": cumulative_turnover,
        "modeled_cost_pct_initial": 100.0 * cumulative_cost,
        "decision_count": len(grouped),
        "active_decision_fraction": active / count,
        "mean_net_return_bps": 10_000.0 * float(np.mean(returns)) if returns else 0.0,
    }, returns


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = sorted(
        key for key, value in records[0].items() if key != "seed" and isinstance(value, (int, float))
    )
    return {
        "seed_count": len(records),
        "mean": {key: float(np.mean([record[key] for record in records])) for key in numeric},
        "standard_deviation": {
            key: (
                float(np.std([record[key] for record in records], ddof=1))
                if len(records) > 1
                else 0.0
            )
            for key in numeric
        },
        "per_seed": records,
    }


def _paired_bootstrap(mps: list[dict[str, Any]], ann: list[dict[str, Any]]) -> dict[str, float]:
    differences = np.asarray(
        [left["total_return_pct"] - right["total_return_pct"] for left, right in zip(mps, ann)],
        dtype=float,
    )
    generator = np.random.default_rng(2026)
    draws = generator.choice(differences, size=(10_000, len(differences)), replace=True).mean(axis=1)
    return {
        "mean_mps_minus_ann_total_return_pp": float(np.mean(differences)),
        "bootstrap_95pct_low_pp": float(np.quantile(draws, 0.025)),
        "bootstrap_95pct_high_pp": float(np.quantile(draws, 0.975)),
        "paired_seed_count": len(differences),
    }


def run_benchmark(
    snapshots: list[dict[str, Any]],
    output_dir: Path,
    *,
    train_end: str,
    validation_end: str,
    test_start: str,
    test_end: str,
    seeds: tuple[int, ...] = tuple(range(10)),
    epochs: int = 200,
    patience: int = 25,
    transaction_cost_bps: float = 10.0,
) -> dict[str, Any]:
    data = prepare_dataset(snapshots, train_end=train_end, validation_end=validation_end)
    timestamps = np.asarray([str(row["timestamp"]) for row in data["rows"]])
    next_timestamps = np.asarray([str(row["next_timestamp"]) for row in data["rows"]])
    test_mask = (
        (timestamps >= test_start)
        & (timestamps <= test_end)
        & (next_timestamps <= test_end)
    )
    if not test_mask.any():
        raise ValueError("The untouched test partition is empty")
    targets = data["targets"]
    standardized = data["standardized"]
    rows = data["rows"]
    mps_records: list[dict[str, Any]] = []
    ann_records: list[dict[str, Any]] = []

    for seed in seeds:
        mps, _, mps_calibration = fit_torch_regressor(
            lambda seed=seed: MPSRegressor(13, 4, seed=seed),
            data,
            seed=seed,
            epochs=epochs,
            patience=patience,
        )
        ann, _, ann_calibration = fit_torch_regressor(
            MatchedANNRegressor,
            data,
            seed=seed,
            epochs=epochs,
            patience=patience,
        )
        for model, calibration, destination in (
            (mps, mps_calibration, mps_records),
            (ann, ann_calibration, ann_records),
        ):
            predictions = _predict(model, standardized)
            signal = _signal_metrics(predictions[test_mask], targets[test_mask])
            portfolio, _ = _portfolio_metrics(
                predictions,
                targets,
                rows,
                test_mask,
                threshold_pp=float(calibration["trade_threshold_pp"]),
                transaction_cost_bps=transaction_cost_bps,
            )
            destination.append({"seed": seed, **signal, **portfolio})

    train_x = standardized[data["train_mask"]]
    train_y = targets[data["train_mask"]]
    design = np.column_stack([train_x, np.ones(len(train_x))])
    penalty = np.eye(design.shape[1]) * 1e-3
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ train_y)
    all_design = np.column_stack([standardized, np.ones(len(standardized))])
    linear_predictions = all_design @ coefficients
    linear_calibration = calibrate_predictions(
        linear_predictions[data["validation_mask"]],
        targets[data["validation_mask"]],
        transaction_cost_bps=transaction_cost_bps,
    )

    controls: dict[str, dict[str, float]] = {}
    feature_index = {name: index for index, name in enumerate(FEATURE_NAMES)}
    raw = data["inputs"]
    candidates = {
        "linear_ridge": (linear_predictions, float(linear_calibration["trade_threshold_pp"]), False),
        "momentum_3h": (100.0 * raw[:, feature_index["return_3h"]], transaction_cost_bps / 100.0, False),
        "mean_reversion_1h": (-100.0 * raw[:, feature_index["return_1h"]], transaction_cost_bps / 100.0, False),
        "dynamic_equal_weight": (np.ones(len(targets)), -float("inf"), True),
    }
    for name, (predictions, threshold, equal_all) in candidates.items():
        signal = _signal_metrics(predictions[test_mask], targets[test_mask])
        portfolio, _ = _portfolio_metrics(
            predictions,
            targets,
            rows,
            test_mask,
            threshold_pp=threshold,
            transaction_cost_bps=transaction_cost_bps,
            equal_weight_all=equal_all,
        )
        controls[name] = {**signal, **portfolio}

    canonical = json.dumps(snapshots, sort_keys=True, separators=(",", ":")).encode()
    result = {
        "schema_version": 1,
        "method": "ATL-native hourly next-return benchmark",
        "quantum_hardware": False,
        "dataset_sha256": hashlib.sha256(canonical).hexdigest(),
        "feature_names": list(FEATURE_NAMES),
        "feature_audit": data["audit"],
        "split": {
            "train_end": train_end,
            "validation_end": validation_end,
            "test_start": test_start,
            "test_end": test_end,
            "training_rows": int(data["train_mask"].sum()),
            "validation_rows": int(data["validation_mask"].sum()),
            "test_rows": int(test_mask.sum()),
            "test_timestamps": int(len(set(timestamps[test_mask].tolist()))),
        },
        "configuration": {
            "seeds": list(seeds),
            "epochs": epochs,
            "patience": patience,
            "transaction_cost_bps_per_unit_turnover": transaction_cost_bps,
            "mps_parameters": parameter_count(MPSRegressor(13, 4)),
            "ann_parameters": parameter_count(MatchedANNRegressor()),
            "max_positions": 3,
        },
        "mps": _summary(mps_records),
        "matched_ann": _summary(ann_records),
        "paired_inference": _paired_bootstrap(mps_records, ann_records),
        "controls": controls,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output_dir / "feature_audit.json").write_text(json.dumps(data["audit"], indent=2), encoding="utf-8")
    with (output_dir / "benchmark_seed_results.csv").open("w", newline="", encoding="utf-8") as handle:
        rows_out = [{"model": "mps", **row} for row in mps_records] + [
            {"model": "matched_ann", **row} for row in ann_records
        ]
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    return result
