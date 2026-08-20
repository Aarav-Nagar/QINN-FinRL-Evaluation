from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from atl_mps_agent.features import FEATURE_NAMES, build_supervised_rows, snapshot_features
from atl_mps_agent.model import MPSRegressor, parameter_count
from atl_mps_agent.policy import MPSPolicy


def sample_snapshot(price: float, cash: float = 1000.0) -> dict:
    return {
        "timestamp": "2026-04-01T10:00:00-04:00",
        "portfolio": {"cash": cash, "total_equity": 1000.0},
        "current_holdings": {},
        "top_signals": {
            "AAPL": {
                "price": price,
                "rsi": 52.0,
                "macd": 1.2,
                "macd_signal": 0.9,
                "sma20": price * 0.99,
                "sma50": price * 0.98,
                "bb_upper": price * 1.04,
                "bb_lower": price * 0.96,
            },
            "MSFT": {
                "price": 400.0,
                "rsi": 48.0,
                "macd": -0.2,
                "macd_signal": -0.1,
                "sma20": 402.0,
                "sma50": 405.0,
                "bb_upper": 420.0,
                "bb_lower": 380.0,
            },
        },
    }


def test_mps_bond_four_has_recorded_369_parameters():
    assert parameter_count(MPSRegressor(13, 4)) == 369


def test_snapshot_feature_contract_is_finite_and_thirteen_dimensional():
    values = snapshot_features(sample_snapshot(200.0), "AAPL", {"AAPL": [195.0]})
    assert values.shape == (len(FEATURE_NAMES),) == (13,)
    assert np.isfinite(values).all()


def test_supervised_target_uses_the_following_snapshot():
    current = sample_snapshot(200.0)
    following = sample_snapshot(202.0)
    following["timestamp"] = "2026-04-01T11:00:00-04:00"
    inputs, targets, rows = build_supervised_rows([current, following])
    aapl = next(index for index, row in enumerate(rows) if row["symbol"] == "AAPL")
    assert inputs.shape[1] == 13
    assert targets[aapl] == np.float32(1.0)


def test_policy_emits_atl_action_contract(tmp_path: Path):
    model = MPSRegressor(13, 4)
    artifact = {
        "schema_version": 1,
        "model_type": "classical_mps_regressor",
        "feature_names": list(FEATURE_NAMES),
        "bond_dimension": 4,
        "parameter_count": 369,
        "feature_means": [0.0] * 13,
        "feature_scales": [1.0] * 13,
        "state_dict": model.state_dict(),
    }
    path = tmp_path / "model.pt"
    torch.save(artifact, path)
    actions = MPSPolicy(path).decide(sample_snapshot(200.0), ["AAPL", "MSFT"])
    assert actions
    required = {"action", "symbol", "confidence", "reasoning", "position_size"}
    assert required <= set(actions[0])
    assert actions[0]["action"] in {"buy", "sell", "hold"}
    assert 0.0 <= actions[0]["confidence"] <= 1.0
