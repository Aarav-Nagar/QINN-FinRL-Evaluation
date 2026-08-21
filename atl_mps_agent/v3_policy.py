"""Uncertainty-aware execution policy for the ATL residual-MPS ensemble."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .features import snapshot_features
from .policy import MPSPolicy
from .v3_training import load_v3_ensemble


class ResidualMPSEnsemblePolicy(MPSPolicy):
    """Use ensemble lower confidence bounds with v2's execution safeguards."""

    def __init__(self, artifact_path: Path):
        self.models, self.means, self.scales, self.metadata = load_v3_ensemble(
            artifact_path
        )
        calibration = self.metadata.get("calibration") or {}
        self.max_positions = 3
        self.min_hold_steps = 4
        self.cooldown_steps = 3
        self.max_step_turnover = 0.20
        self.smoothing_alpha = 0.30
        self.entry_threshold = float(calibration.get("trade_threshold_pp", 0.10))
        self.entries_enabled = not bool(
            calibration.get("abstain_without_positive_validation_edge", False)
        )
        self.exit_threshold = -0.25 * self.entry_threshold
        self.residual_scale = max(
            float(calibration.get("residual_rmse_pp", 1.0)), 1e-6
        )
        self.uncertainty_penalty_z = float(
            calibration.get("uncertainty_penalty_z", 1.0)
        )
        self.price_history: dict[str, list[float]] = defaultdict(list)
        self.smoothed_scores: dict[str, float] = {}
        self.holding_steps: dict[str, int] = defaultdict(int)
        self.last_trade_step: dict[str, int] = defaultdict(lambda: -10_000)
        self.decision_step = 0
        self.latest_uncertainty: dict[str, float] = {}
        self.latest_mean: dict[str, float] = {}
        self.last_snapshot_time: datetime | None = None

    def scores(
        self, snapshot: dict[str, Any], valid_symbols: list[str]
    ) -> dict[str, float]:
        symbols = [
            symbol
            for symbol in valid_symbols
            if symbol in (snapshot.get("top_signals") or {})
        ]
        if not symbols:
            return {}
        try:
            current_time = datetime.fromisoformat(str(snapshot.get("timestamp")))
        except (TypeError, ValueError):
            current_time = None
        if (
            current_time is None
            or (
                self.last_snapshot_time is not None
                and (current_time - self.last_snapshot_time).total_seconds() > 7200
            )
        ):
            self.price_history.clear()
            self.smoothed_scores.clear()
        self.last_snapshot_time = current_time
        raw = np.vstack(
            [snapshot_features(snapshot, symbol, self.price_history) for symbol in symbols]
        )
        standardized = np.clip((raw - self.means) / self.scales, -8.0, 8.0)
        values = torch.as_tensor(standardized, dtype=torch.float32)
        with torch.no_grad():
            member_predictions = np.vstack(
                [model(values).cpu().numpy() for model in self.models]
            )
        means = member_predictions.mean(axis=0)
        uncertainties = member_predictions.std(axis=0, ddof=0)
        lower_bounds = means - self.uncertainty_penalty_z * uncertainties
        for symbol, mean, uncertainty, score in zip(
            symbols, means, uncertainties, lower_bounds
        ):
            self.latest_mean[symbol] = float(mean)
            self.latest_uncertainty[symbol] = float(uncertainty)
            previous = self.smoothed_scores.get(symbol, float(score))
            self.smoothed_scores[symbol] = (
                self.smoothing_alpha * float(score)
                + (1.0 - self.smoothing_alpha) * previous
            )
        return {symbol: self.smoothed_scores[symbol] for symbol in symbols}

    def _action(
        self, action: str, symbol: str, quantity: int, score: float, reason: str
    ) -> dict[str, Any]:
        uncertainty = self.latest_uncertainty.get(symbol, 0.0)
        total_scale = math.sqrt(self.residual_scale**2 + uncertainty**2)
        confidence = 0.5 + 0.5 * math.erf(
            abs(float(score)) / (max(total_scale, 1e-6) * math.sqrt(2.0))
        )
        confidence = min(0.95, max(0.50, confidence))
        return {
            "action": action,
            "symbol": symbol,
            "confidence": round(confidence, 4),
            "reasoning": (
                f"{reason}; residual-MPS ensemble mean="
                f"{self.latest_mean.get(symbol, score):.5f}pp, "
                f"uncertainty={uncertainty:.5f}pp, lower bound={score:.5f}pp, "
                f"entry threshold={self.entry_threshold:.5f}pp"
            ),
            "position_size": int(quantity),
            "stop_loss_price": None,
            "take_profit_price": None,
        }
