"""Leakage-safe training, calibration, and artifact provenance."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .features import FEATURE_NAMES, audit_features, audit_raw_inputs, build_supervised_rows
from .model import MPSRegressor, parameter_count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_dataset(
    snapshots: list[dict[str, Any]],
    *,
    train_end: str | None = None,
    validation_end: str | None = None,
    reset_history_on_gap: bool = False,
    horizon_steps: int = 1,
    max_horizon_hours: float = 2.0,
) -> dict[str, Any]:
    inputs, targets, rows = build_supervised_rows(
        snapshots,
        reset_history_on_gap=reset_history_on_gap,
        horizon_steps=horizon_steps,
        max_horizon_hours=max_horizon_hours,
    )
    audit = audit_features(inputs, rows)
    audit["raw_input_coverage"] = audit_raw_inputs(snapshots)
    timestamps = np.asarray([str(item["timestamp"]) for item in rows])
    next_timestamps = np.asarray([str(item["next_timestamp"]) for item in rows])
    unique_times = sorted(set(timestamps.tolist()))
    if len(unique_times) < 5:
        raise ValueError("At least five hourly timestamps are required")
    if train_end is None:
        train_end = unique_times[max(1, int(0.8 * len(unique_times))) - 1]
    if validation_end is None:
        validation_end = unique_times[-1]
    if train_end >= validation_end:
        raise ValueError("train_end must precede validation_end")

    # next_timestamp constraints remove rows whose target crosses a boundary.
    train_mask = (timestamps <= train_end) & (next_timestamps <= train_end)
    validation_mask = (
        (timestamps > train_end)
        & (timestamps <= validation_end)
        & (next_timestamps <= validation_end)
    )
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("The requested chronological split has an empty partition")

    means = inputs[train_mask].mean(axis=0)
    raw_scales = inputs[train_mask].std(axis=0)
    degenerate = [
        FEATURE_NAMES[index] for index, value in enumerate(raw_scales) if value < 1e-8
    ]
    if degenerate:
        raise ValueError(f"Training features are degenerate: {', '.join(degenerate)}")
    standardized = np.clip((inputs - means) / raw_scales, -8.0, 8.0)
    return {
        "inputs": inputs,
        "targets": targets,
        "rows": rows,
        "audit": audit,
        "train_mask": train_mask,
        "validation_mask": validation_mask,
        "means": means,
        "scales": raw_scales,
        "standardized": standardized,
        "train_end": train_end,
        "validation_end": validation_end,
        "horizon_steps": horizon_steps,
        "max_horizon_hours": max_horizon_hours,
    }


def fit_torch_regressor(
    model_factory: Callable[[], torch.nn.Module],
    data: dict[str, Any],
    *,
    seed: int,
    epochs: int,
    patience: int,
    learning_rate: float = 0.01,
) -> tuple[torch.nn.Module, list[dict[str, float]], dict[str, float]]:
    _seed_everything(seed)
    model = model_factory()
    train_mask = data["train_mask"]
    validation_mask = data["validation_mask"]
    standardized = data["standardized"]
    targets = data["targets"]
    x_train = torch.as_tensor(standardized[train_mask], dtype=torch.float32)
    y_train = torch.as_tensor(targets[train_mask], dtype=torch.float32)
    x_val = torch.as_tensor(standardized[validation_mask], dtype=torch.float32)
    y_val = torch.as_tensor(targets[validation_mask], dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    wait = 0
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_loss = torch.mean((model(x_train) - y_train) ** 2)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = torch.mean((model(x_val) - y_val) ** 2)
        record = {
            "epoch": float(epoch + 1),
            "train_mse": float(train_loss.detach()),
            "validation_mse": float(validation_loss.detach()),
        }
        history.append(record)
        if record["validation_mse"] < best_loss - 1e-8:
            best_loss = record["validation_mse"]
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = model(x_val).cpu().numpy()
    calibration = calibrate_predictions(
        predictions, targets[validation_mask], transaction_cost_bps=10.0
    )
    calibration["best_validation_mse"] = best_loss
    return model, history, calibration


def calibrate_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    transaction_cost_bps: float,
) -> dict[str, Any]:
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    residual_scale = float(np.sqrt(np.mean((predictions - targets) ** 2)))
    positive = predictions[predictions > 0.0]
    candidates = [transaction_cost_bps / 100.0]
    if len(positive):
        candidates.extend(float(np.quantile(positive, q)) for q in (0.5, 0.6, 0.7, 0.8, 0.9))
    best_threshold = candidates[0]
    best_net_edge = float("-inf")
    best_coverage = 0.0
    for threshold in sorted(set(candidates)):
        selected = predictions >= threshold
        if int(selected.sum()) < max(5, int(0.05 * len(predictions))):
            continue
        net_edge = float(np.mean(targets[selected] - transaction_cost_bps / 100.0))
        if net_edge > best_net_edge:
            best_threshold = threshold
            best_net_edge = net_edge
            best_coverage = float(np.mean(selected))
    if not np.isfinite(best_net_edge):
        best_net_edge = 0.0
    abstain_all = best_net_edge <= 0.0
    if abstain_all:
        best_threshold = max(
            transaction_cost_bps / 100.0,
            float(np.max(predictions)) + 1e-6 if len(predictions) else 0.0,
        )
        best_coverage = 0.0
    direction = float(np.mean(np.sign(predictions) == np.sign(targets)))
    rank_ic = 0.0
    if np.std(predictions) > 0.0 and np.std(targets) > 0.0:
        rank_ic = float(np.corrcoef(np.argsort(np.argsort(predictions)), np.argsort(np.argsort(targets)))[0, 1])
    return {
        "residual_rmse_pp": residual_scale,
        "trade_threshold_pp": float(max(transaction_cost_bps / 100.0, best_threshold)),
        "validation_net_edge_pp": best_net_edge,
        "validation_coverage": best_coverage,
        "validation_directional_accuracy": direction,
        "validation_rank_ic": rank_ic,
        "transaction_cost_bps": transaction_cost_bps,
        "abstain_without_positive_validation_edge": abstain_all,
    }


def train_from_snapshots(
    snapshots: list[dict[str, Any]],
    artifact_path: Path,
    *,
    bond_dimension: int = 4,
    epochs: int = 250,
    patience: int = 30,
    seed: int = 2026,
    train_end: str | None = None,
    validation_end: str | None = None,
) -> dict[str, Any]:
    data = prepare_dataset(
        snapshots, train_end=train_end, validation_end=validation_end
    )
    model, history, calibration = fit_torch_regressor(
        lambda: MPSRegressor(len(FEATURE_NAMES), bond_dimension, seed=seed),
        data,
        seed=seed,
        epochs=epochs,
        patience=patience,
    )
    canonical_data = json.dumps(snapshots, sort_keys=True, separators=(",", ":")).encode()
    dataset_sha256 = hashlib.sha256(canonical_data).hexdigest()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": 2,
        "model_type": "classical_mps_regressor",
        "feature_names": list(FEATURE_NAMES),
        "bond_dimension": bond_dimension,
        "parameter_count": parameter_count(model),
        "seed": seed,
        "feature_means": data["means"].tolist(),
        "feature_scales": data["scales"].tolist(),
        "state_dict": model.state_dict(),
        "training_rows": int(data["train_mask"].sum()),
        "validation_rows": int(data["validation_mask"].sum()),
        "train_end": data["train_end"],
        "validation_end": data["validation_end"],
        "first_timestamp": data["audit"]["first_timestamp"],
        "last_timestamp": data["audit"]["last_timestamp"],
        "best_validation_mse": calibration["best_validation_mse"],
        "epochs_completed": len(history),
        "calibration": calibration,
        "feature_audit": data["audit"],
        "dataset_sha256": dataset_sha256,
    }
    torch.save(artifact, artifact_path)
    summary = {key: value for key, value in artifact.items() if key != "state_dict"}
    summary["artifact_sha256"] = _sha256(artifact_path)
    artifact_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def load_artifact(path: Path) -> tuple[MPSRegressor, np.ndarray, np.ndarray, dict[str, Any]]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("schema_version") != 2:
        raise ValueError("This policy requires an ATL MPS v2 artifact")
    if artifact.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Artifact feature contract does not match this agent")
    model = MPSRegressor(len(FEATURE_NAMES), int(artifact["bond_dimension"]))
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    means = np.asarray(artifact["feature_means"], dtype=np.float32)
    scales = np.asarray(artifact["feature_scales"], dtype=np.float32)
    metadata = {key: value for key, value in artifact.items() if key != "state_dict"}
    return model, means, scales, metadata
