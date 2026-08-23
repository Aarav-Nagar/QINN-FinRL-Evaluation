"""Fresh-window benchmark for the residual-MPS v3 architecture."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .benchmark import (
    _paired_bootstrap,
    _portfolio_metrics,
    _signal_metrics,
    _summary,
)
from .features import FEATURE_NAMES
from .model import MatchedANNV3Regressor, ResidualMPSRegressor, parameter_count
from .training import calibrate_predictions, prepare_dataset
from .v3_training import ensemble_predictions, fit_rank_aware_regressor


def run_v3_benchmark(
    snapshots: list[dict[str, Any]],
    output_dir: Path,
    *,
    train_end: str,
    validation_end: str,
    test_start: str,
    test_end: str,
    seeds: tuple[int, ...] = tuple(range(10)),
    ensemble_seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    uncertainty_penalty_z: float = 1.0,
    transaction_cost_bps: float = 10.0,
    epochs: int = 250,
    patience: int = 30,
) -> dict[str, Any]:
    data = prepare_dataset(
        snapshots,
        train_end=train_end,
        validation_end=validation_end,
        reset_history_on_gap=True,
    )
    timestamps = np.asarray([str(row["timestamp"]) for row in data["rows"]])
    next_timestamps = np.asarray(
        [str(row["next_timestamp"]) for row in data["rows"]]
    )
    test_mask = (
        (timestamps >= test_start)
        & (timestamps <= test_end)
        & (next_timestamps <= test_end)
    )
    if not test_mask.any():
        raise ValueError("The fresh v3 test partition is empty")
    targets = data["targets"]
    values = data["standardized"]
    rows = data["rows"]
    mps_models = []
    ann_models = []
    mps_records: list[dict[str, Any]] = []
    ann_records: list[dict[str, Any]] = []
    mps_predictions: dict[int, np.ndarray] = {}
    ann_predictions: dict[int, np.ndarray] = {}

    for seed in seeds:
        mps, _ = fit_rank_aware_regressor(
            lambda seed=seed: ResidualMPSRegressor(13, seed=seed),
            data,
            seed=seed,
            epochs=epochs,
            patience=patience,
        )
        ann, _ = fit_rank_aware_regressor(
            MatchedANNV3Regressor,
            data,
            seed=seed,
            epochs=epochs,
            patience=patience,
        )
        mps_models.append(mps)
        ann_models.append(ann)
        for model, store, destination in (
            (mps, mps_predictions, mps_records),
            (ann, ann_predictions, ann_records),
        ):
            predictions, _ = ensemble_predictions([model], values)
            store[seed] = predictions
            calibration = calibrate_predictions(
                predictions[data["validation_mask"]],
                targets[data["validation_mask"]],
                transaction_cost_bps=transaction_cost_bps,
            )
            signal = _signal_metrics(predictions[test_mask], targets[test_mask])
            threshold = (
                float(np.max(predictions[test_mask]) + 1.0)
                if calibration["abstain_without_positive_validation_edge"]
                else float(calibration["trade_threshold_pp"])
            )
            portfolio, _ = _portfolio_metrics(
                predictions,
                targets,
                rows,
                test_mask,
                threshold_pp=threshold,
                transaction_cost_bps=transaction_cost_bps,
            )
            destination.append(
                {
                    "seed": seed,
                    "validation_abstained": bool(
                        calibration["abstain_without_positive_validation_edge"]
                    ),
                    **signal,
                    **portfolio,
                }
            )

    ensemble_indices = [seeds.index(seed) for seed in ensemble_seeds]

    def ensemble_result(models: list[Any]) -> dict[str, Any]:
        selected = [models[index] for index in ensemble_indices]
        mean, uncertainty = ensemble_predictions(selected, values)
        lower_bound = mean - uncertainty_penalty_z * uncertainty
        calibration = calibrate_predictions(
            lower_bound[data["validation_mask"]],
            targets[data["validation_mask"]],
            transaction_cost_bps=transaction_cost_bps,
        )
        signal = _signal_metrics(mean[test_mask], targets[test_mask])
        if calibration["abstain_without_positive_validation_edge"]:
            # Use an unreachable threshold so the benchmark mirrors deployment.
            threshold = float(np.max(lower_bound[test_mask]) + 1.0)
        else:
            threshold = float(calibration["trade_threshold_pp"])
        portfolio, returns = _portfolio_metrics(
            lower_bound,
            targets,
            rows,
            test_mask,
            threshold_pp=threshold,
            transaction_cost_bps=transaction_cost_bps,
        )
        return {
            "member_seeds": list(ensemble_seeds),
            "calibration": calibration,
            "mean_test_uncertainty_pp": float(np.mean(uncertainty[test_mask])),
            "signal_metrics": signal,
            "portfolio_metrics": portfolio,
            "net_returns": returns,
        }

    mps_ensemble = ensemble_result(mps_models)
    ann_ensemble = ensemble_result(ann_models)
    canonical = json.dumps(snapshots, sort_keys=True, separators=(",", ":")).encode()
    result = {
        "schema_version": 1,
        "architecture": "rank-aware residual-MPS deep ensemble",
        "architecture_selected_without_fresh_test": True,
        "quantum_hardware": False,
        "dataset_sha256": hashlib.sha256(canonical).hexdigest(),
        "feature_names": list(FEATURE_NAMES),
        "feature_audit": data["audit"],
        "split": {
            "train_end": train_end,
            "validation_end": validation_end,
            "fresh_test_start": test_start,
            "fresh_test_end": test_end,
            "training_rows": int(data["train_mask"].sum()),
            "validation_rows": int(data["validation_mask"].sum()),
            "test_rows": int(test_mask.sum()),
            "test_timestamps": int(len(set(timestamps[test_mask].tolist()))),
        },
        "configuration": {
            "paired_member_seeds": list(seeds),
            "deployment_ensemble_seeds": list(ensemble_seeds),
            "rank_loss_weight": 0.10,
            "uncertainty_penalty_z": uncertainty_penalty_z,
            "transaction_cost_bps_per_unit_turnover": transaction_cost_bps,
            "mps_parameters_per_member": parameter_count(ResidualMPSRegressor()),
            "ann_parameters_per_member": parameter_count(MatchedANNV3Regressor()),
        },
        "mps_members": _summary(mps_records),
        "matched_ann_members": _summary(ann_records),
        "paired_member_inference": _paired_bootstrap(mps_records, ann_records),
        "mps_ensemble": {key: value for key, value in mps_ensemble.items() if key != "net_returns"},
        "matched_ann_ensemble": {key: value for key, value in ann_ensemble.items() if key != "net_returns"},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "v3_benchmark_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8", newline="\n"
    )
    with (output_dir / "v3_member_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        output_rows = [{"model": "residual_mps", **row} for row in mps_records] + [
            {"model": "matched_ann", **row} for row in ann_records
        ]
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    return result
