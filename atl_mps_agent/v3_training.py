"""Rank-aware residual-MPS ensemble training for ATL v3."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .features import FEATURE_NAMES
from .model import ResidualMPSRegressor, parameter_count
from .training import calibrate_predictions, prepare_dataset


def _pair_indices(
    rows: list[dict[str, Any]], mask: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index in np.flatnonzero(mask):
        grouped[str(rows[int(index)]["timestamp"])].append(int(index))
    left: list[int] = []
    right: list[int] = []
    direction: list[float] = []
    for indices in grouped.values():
        for position, first in enumerate(indices):
            for second in indices[position + 1 :]:
                difference = float(targets[first] - targets[second])
                if abs(difference) <= 1e-8:
                    continue
                left.append(first)
                right.append(second)
                direction.append(1.0 if difference > 0.0 else -1.0)
    return (
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        np.asarray(direction, dtype=np.float32),
    )


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit_rank_aware_regressor(
    model_factory: Callable[[], nn.Module],
    data: dict[str, Any],
    *,
    seed: int,
    epochs: int = 250,
    patience: int = 30,
    rank_weight: float = 0.10,
    learning_rate: float = 0.005,
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Fit one model with robust regression and timestamp-local ranking loss."""

    _seed(seed)
    model = model_factory()
    values = torch.as_tensor(data["standardized"], dtype=torch.float32)
    targets = torch.as_tensor(data["targets"], dtype=torch.float32)
    train_indices = np.flatnonzero(data["train_mask"])
    validation_indices = np.flatnonzero(data["validation_mask"])
    train_idx = torch.as_tensor(train_indices, dtype=torch.long)
    val_idx = torch.as_tensor(validation_indices, dtype=torch.long)
    train_pairs = _pair_indices(data["rows"], data["train_mask"], data["targets"])
    val_pairs = _pair_indices(data["rows"], data["validation_mask"], data["targets"])
    pair_tensors = [
        tuple(torch.as_tensor(part) for part in pair_set)
        for pair_set in (train_pairs, val_pairs)
    ]

    def objective(indices: torch.Tensor, pairs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        predictions = model(values)
        regression = F.smooth_l1_loss(
            predictions[indices], targets[indices], beta=0.25
        )
        left, right, direction = pairs
        ranking = F.softplus(-direction * (predictions[left] - predictions[right])).mean()
        return regression + rank_weight * ranking, regression, ranking

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    wait = 0
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_loss, train_regression, train_ranking = objective(train_idx, pair_tensors[0])
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss, val_regression, val_ranking = objective(val_idx, pair_tensors[1])
        record = {
            "epoch": float(epoch + 1),
            "train_loss": float(train_loss.detach()),
            "train_huber": float(train_regression.detach()),
            "train_rank_loss": float(train_ranking.detach()),
            "validation_loss": float(val_loss.detach()),
            "validation_huber": float(val_regression.detach()),
            "validation_rank_loss": float(val_ranking.detach()),
        }
        history.append(record)
        if record["validation_loss"] < best_loss - 1e-7:
            best_loss = record["validation_loss"]
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def ensemble_predictions(models: list[nn.Module], values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    with torch.no_grad():
        members = np.vstack([model(tensor).cpu().numpy() for model in models])
    return members.mean(axis=0), members.std(axis=0, ddof=0)


def train_v3_ensemble(
    snapshots: list[dict[str, Any]],
    artifact_path: Path,
    *,
    train_end: str,
    validation_end: str,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
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
    models: list[nn.Module] = []
    histories: dict[str, list[dict[str, float]]] = {}
    for seed in seeds:
        model, history = fit_rank_aware_regressor(
            lambda seed=seed: ResidualMPSRegressor(13, seed=seed),
            data,
            seed=seed,
            epochs=epochs,
            patience=patience,
        )
        models.append(model)
        histories[str(seed)] = history
    mean, uncertainty = ensemble_predictions(models, data["standardized"])
    lower_bound = mean - uncertainty_penalty_z * uncertainty
    calibration = calibrate_predictions(
        lower_bound[data["validation_mask"]],
        data["targets"][data["validation_mask"]],
        transaction_cost_bps=transaction_cost_bps,
    )
    calibration.update(
        {
            "uncertainty_penalty_z": uncertainty_penalty_z,
            "validation_mean_uncertainty_pp": float(
                np.mean(uncertainty[data["validation_mask"]])
            ),
        }
    )
    canonical = json.dumps(snapshots, sort_keys=True, separators=(",", ":")).encode()
    artifact = {
        "schema_version": 3,
        "model_type": "residual_mps_deep_ensemble",
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
        "train_end": data["train_end"],
        "validation_end": data["validation_end"],
        "calibration": calibration,
        "feature_audit": data["audit"],
        "dataset_sha256": hashlib.sha256(canonical).hexdigest(),
        "training_histories": histories,
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, artifact_path)
    summary = {key: value for key, value in artifact.items() if key not in {"state_dicts", "training_histories"}}
    summary["epochs_completed_by_seed"] = {
        seed: len(history) for seed, history in histories.items()
    }
    summary["artifact_sha256"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    artifact_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8", newline="\n"
    )
    return summary


def load_v3_ensemble(
    path: Path,
) -> tuple[list[ResidualMPSRegressor], np.ndarray, np.ndarray, dict[str, Any]]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("schema_version") != 3:
        raise ValueError("Expected an ATL residual-MPS v3 artifact")
    if artifact.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Artifact feature contract does not match v3")
    seeds = [int(seed) for seed in artifact["seeds"]]
    models = [ResidualMPSRegressor(13, seed=seed) for seed in seeds]
    for model, state in zip(models, artifact["state_dicts"]):
        model.load_state_dict(state)
        model.eval()
    metadata = {key: value for key, value in artifact.items() if key != "state_dicts"}
    return (
        models,
        np.asarray(artifact["feature_means"], dtype=np.float32),
        np.asarray(artifact["feature_scales"], dtype=np.float32),
        metadata,
    )
