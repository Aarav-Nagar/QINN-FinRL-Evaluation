"""Frozen-window evaluation for the current ATL deployment policy."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .deployment_training import (
    load_deployment_ensemble,
    predict_snapshots,
    simulate_whole_share_policy,
)
from .features import build_supervised_rows
from .model import MatchedANNV3Regressor, parameter_count
from .training import prepare_dataset
from .v3_training import ensemble_predictions, fit_rank_aware_regressor


def _signal_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["timestamp"])].append(index)
    correlations = []
    for indices in grouped.values():
        left = predictions[indices]
        right = targets[indices]
        if len(indices) > 1 and np.std(left) > 0.0 and np.std(right) > 0.0:
            correlations.append(float(np.corrcoef(np.argsort(np.argsort(left)), np.argsort(np.argsort(right)))[0, 1]))
    return {
        "mse_pp2": float(np.mean((predictions - targets) ** 2)),
        "mae_pp": float(np.mean(np.abs(predictions - targets))),
        "directional_accuracy": float(np.mean(np.sign(predictions) == np.sign(targets))),
        "mean_within_timestamp_rank_ic": float(np.mean(correlations)) if correlations else 0.0,
    }


def _compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "equity_curve"}


def run_deployment_benchmark(
    development_snapshots: list[dict[str, Any]],
    fresh_snapshots: list[dict[str, Any]],
    artifact_path: Path,
    output_dir: Path,
    *,
    test_start: str,
    test_end: str,
    train_end: str,
    validation_end: str,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    transaction_cost_bps: float = 10.0,
    epochs: int = 250,
    patience: int = 30,
) -> dict[str, Any]:
    """Evaluate the frozen system and matched controls without reselection."""

    models, means, scales, metadata = load_deployment_ensemble(artifact_path)
    selected_profile = dict(metadata["selected_policy"])
    fresh_predictions = predict_snapshots(models, means, scales, fresh_snapshots)
    selected = simulate_whole_share_policy(
        fresh_snapshots,
        fresh_predictions,
        selected_profile,
        start=test_start,
        end=test_end,
        transaction_cost_bps=transaction_cost_bps,
    )
    mps_only_profile = {
        **selected_profile,
        "name": "mps_only_rank",
        "model_weight": 1.0,
        "positive_trend_gate": False,
    }
    trend_only_profile = {
        **selected_profile,
        "name": "trend_only",
        "model_weight": 0.0,
    }
    initial_basket_profile = {
        **trend_only_profile,
        "name": "fixed_initial_trend_basket",
        "rebalance_days": 10_000,
        "positive_trend_gate": False,
    }
    controls = {
        "mps_only_rank": simulate_whole_share_policy(
            fresh_snapshots,
            fresh_predictions,
            mps_only_profile,
            start=test_start,
            end=test_end,
            transaction_cost_bps=transaction_cost_bps,
        ),
        "trend_only": simulate_whole_share_policy(
            fresh_snapshots,
            fresh_predictions,
            trend_only_profile,
            start=test_start,
            end=test_end,
            transaction_cost_bps=transaction_cost_bps,
        ),
        "fixed_initial_trend_basket": simulate_whole_share_policy(
            fresh_snapshots,
            fresh_predictions,
            initial_basket_profile,
            start=test_start,
            end=test_end,
            transaction_cost_bps=transaction_cost_bps,
        ),
    }

    # Fit the matched ANN only on the same development partitions. Its fresh
    # predictions are generated after training with no policy reselection.
    development_data = prepare_dataset(
        development_snapshots,
        train_end=train_end,
        validation_end=validation_end,
        reset_history_on_gap=False,
        horizon_steps=7,
        max_horizon_hours=96.0,
    )
    ann_models = []
    for seed in seeds:
        ann, _ = fit_rank_aware_regressor(
            MatchedANNV3Regressor,
            development_data,
            seed=seed,
            epochs=epochs,
            patience=patience,
            rank_weight=0.15,
        )
        ann_models.append(ann)
    ann_fresh_predictions = predict_snapshots(ann_models, means, scales, fresh_snapshots)
    matched_ann_system = simulate_whole_share_policy(
        fresh_snapshots,
        ann_fresh_predictions,
        selected_profile,
        start=test_start,
        end=test_end,
        transaction_cost_bps=transaction_cost_bps,
    )

    test_inputs, test_targets, test_rows = build_supervised_rows(
        fresh_snapshots,
        reset_history_on_gap=False,
        horizon_steps=7,
        max_horizon_hours=96.0,
    )
    standardized = np.clip((test_inputs - means) / scales, -8.0, 8.0)
    mps_mean, mps_uncertainty = ensemble_predictions(models, standardized)
    ann_mean, ann_uncertainty = ensemble_predictions(ann_models, standardized)
    canonical = json.dumps(fresh_snapshots, sort_keys=True, separators=(",", ":")).encode()
    result = {
        "schema_version": 1,
        "artifact": "current_next_session_deployment_evaluation",
        "freeze_commit": "7e2a926663d1a94572084cc9ee77609d39aade9f",
        "architecture_selected_without_fresh_test": True,
        "fresh_test_consumed_once": True,
        "quantum_hardware": False,
        "fresh_dataset_sha256": hashlib.sha256(canonical).hexdigest(),
        "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "split": {
            "test_start": test_start,
            "test_end": test_end,
            "snapshot_count": len(fresh_snapshots),
            "decision_days": selected["decision_days"],
            "prediction_rows": len(test_rows),
        },
        "configuration": {
            "selected_policy": selected_profile,
            "transaction_cost_bps_per_traded_notional": transaction_cost_bps,
            "target_horizon_steps": 7,
            "mps_parameters_per_member": parameter_count(models[0]),
            "ann_parameters_per_member": parameter_count(ann_models[0]),
            "ensemble_members": len(models),
            "seeds": list(seeds),
        },
        "prediction_metrics": {
            "residual_mps": _signal_metrics(mps_mean, test_targets, test_rows),
            "matched_ann": _signal_metrics(ann_mean, test_targets, test_rows),
            "mps_mean_uncertainty_pp": float(np.mean(mps_uncertainty)),
            "ann_mean_uncertainty_pp": float(np.mean(ann_uncertainty)),
        },
        "systems": {
            "selected_mps_trend": _compact(selected),
            "matched_ann_trend": _compact(matched_ann_system),
            **{name: _compact(metrics) for name, metrics in controls.items()},
            "cash": {
                "initial_equity": 1000.0,
                "final_equity": 1000.0,
                "total_return_pct": 0.0,
                "maximum_drawdown_pct": 0.0,
                "trade_count": 0,
                "cumulative_turnover_multiple": 0.0,
                "modeled_cost_pct_initial": 0.0,
                "invested_day_fraction": 0.0,
                "decision_days": selected["decision_days"],
                "ending_holdings": {},
            },
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "current_benchmark_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8", newline="\n"
    )
    with (output_dir / "current_system_results.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = [{"system": name, **metrics} for name, metrics in result["systems"].items()]
        fields = [key for key in rows[0] if key != "ending_holdings"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return result
