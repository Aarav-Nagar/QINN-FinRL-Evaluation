"""Leakage-aware training and artifact persistence for the ATL MPS model."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .features import FEATURE_NAMES, build_supervised_rows
from .model import MPSRegressor, parameter_count


def train_from_snapshots(
    snapshots: list[dict[str, Any]],
    artifact_path: Path,
    *,
    bond_dimension: int = 4,
    epochs: int = 250,
    patience: int = 30,
) -> dict[str, Any]:
    inputs, targets, rows = build_supervised_rows(snapshots)
    timestamps = [str(item["timestamp"]) for item in rows]
    unique_times = sorted(set(timestamps))
    if len(unique_times) < 5:
        raise ValueError("At least five hourly timestamps are required")
    split_time = unique_times[max(1, int(0.8 * len(unique_times))) - 1]
    train_mask = np.asarray([stamp <= split_time for stamp in timestamps])
    validation_mask = ~train_mask
    if not validation_mask.any():
        validation_mask[-max(1, len(validation_mask) // 5) :] = True
        train_mask = ~validation_mask

    means = inputs[train_mask].mean(axis=0)
    scales = inputs[train_mask].std(axis=0)
    scales = np.where(scales < 1e-8, 1.0, scales)
    standardized = np.clip((inputs - means) / scales, -8.0, 8.0)

    x_train = torch.as_tensor(standardized[train_mask], dtype=torch.float32)
    y_train = torch.as_tensor(targets[train_mask], dtype=torch.float32)
    x_val = torch.as_tensor(standardized[validation_mask], dtype=torch.float32)
    y_val = torch.as_tensor(targets[validation_mask], dtype=torch.float32)

    torch.manual_seed(2026)
    model = MPSRegressor(len(FEATURE_NAMES), bond_dimension)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    wait = 0
    history = []
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
            "epoch": epoch + 1,
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

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": 1,
        "model_type": "classical_mps_regressor",
        "feature_names": list(FEATURE_NAMES),
        "bond_dimension": bond_dimension,
        "parameter_count": parameter_count(model),
        "feature_means": means.tolist(),
        "feature_scales": scales.tolist(),
        "state_dict": model.state_dict(),
        "training_rows": int(train_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "split_timestamp": split_time,
        "first_timestamp": unique_times[0],
        "last_timestamp": unique_times[-1],
        "best_validation_mse": best_loss,
        "epochs_completed": len(history),
    }
    torch.save(artifact, artifact_path)
    summary = {key: value for key, value in artifact.items() if key != "state_dict"}
    artifact_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def load_artifact(path: Path) -> tuple[MPSRegressor, np.ndarray, np.ndarray, dict[str, Any]]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Artifact feature contract does not match this agent")
    model = MPSRegressor(len(FEATURE_NAMES), int(artifact["bond_dimension"]))
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    means = np.asarray(artifact["feature_means"], dtype=np.float32)
    scales = np.asarray(artifact["feature_scales"], dtype=np.float32)
    metadata = {key: value for key, value in artifact.items() if key != "state_dict"}
    return model, means, scales, metadata
