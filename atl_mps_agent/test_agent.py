from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import torch

from atl_mps_agent.features import FEATURE_NAMES, audit_features, build_supervised_rows, snapshot_features
from atl_mps_agent.model import (
    MPSRegressor,
    MatchedANNRegressor,
    MatchedANNV3Regressor,
    ResidualMPSRegressor,
    parameter_count,
)
from atl_mps_agent.policy import MPSPolicy
from atl_mps_agent.benchmark import run_benchmark
from atl_mps_agent.v3_benchmark import run_v3_benchmark
from atl_mps_agent.v3_policy import ResidualMPSEnsemblePolicy


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
    assert parameter_count(MatchedANNRegressor()) == 369


def test_v3_residual_mps_and_ann_are_exactly_parameter_matched():
    assert parameter_count(ResidualMPSRegressor()) == 586
    assert parameter_count(MatchedANNV3Regressor()) == 586


def test_snapshot_feature_contract_is_finite_and_thirteen_dimensional():
    values = snapshot_features(sample_snapshot(200.0), "AAPL", {"AAPL": [195.0]})
    assert values.shape == (len(FEATURE_NAMES),) == (13,)
    assert np.isfinite(values).all()


def test_predictor_features_do_not_depend_on_portfolio_state():
    history = {"AAPL": [195.0], "MSFT": [405.0]}
    cash_only = sample_snapshot(200.0, cash=1000.0)
    invested = sample_snapshot(200.0, cash=0.0)
    invested["current_holdings"] = {"AAPL": {"shares": 5}}
    assert np.array_equal(
        snapshot_features(cash_only, "AAPL", history),
        snapshot_features(invested, "AAPL", history),
    )


def test_zero_placeholder_indicators_do_not_create_extreme_ratios():
    snapshot = sample_snapshot(200.0)
    for key in ("sma20", "sma50", "bb_upper", "bb_lower"):
        snapshot["top_signals"]["AAPL"][key] = 0.0
    values = snapshot_features(snapshot, "AAPL", {"AAPL": [199.0]})
    assert np.max(np.abs(values)) <= 1.0


def test_supervised_target_uses_the_following_snapshot():
    current = sample_snapshot(200.0)
    following = sample_snapshot(202.0)
    following["timestamp"] = "2026-04-01T11:00:00-04:00"
    inputs, targets, rows = build_supervised_rows([current, following])
    aapl = next(index for index, row in enumerate(rows) if row["symbol"] == "AAPL")
    assert inputs.shape[1] == 13
    assert targets[aapl] == np.float32(1.0)
    audit = audit_features(inputs, rows)
    assert audit["row_count"] == len(rows)
    assert audit["nonfinite_total"] == 0


def test_v3_temporal_features_reset_after_overnight_gap():
    friday = sample_snapshot(200.0)
    friday["timestamp"] = "2026-04-10T16:00:00-04:00"
    monday = sample_snapshot(210.0)
    monday["timestamp"] = "2026-04-13T10:00:00-04:00"
    monday_next = sample_snapshot(211.0)
    monday_next["timestamp"] = "2026-04-13T11:00:00-04:00"
    inputs, _, rows = build_supervised_rows(
        [friday, monday, monday_next], reset_history_on_gap=True
    )
    aapl = next(index for index, row in enumerate(rows) if row["symbol"] == "AAPL")
    return_1h_index = FEATURE_NAMES.index("return_1h")
    assert inputs[aapl, return_1h_index] == 0.0


def test_policy_emits_atl_action_contract(tmp_path: Path):
    model = MPSRegressor(13, 4)
    artifact = {
        "schema_version": 2,
        "model_type": "classical_mps_regressor",
        "feature_names": list(FEATURE_NAMES),
        "bond_dimension": 4,
        "parameter_count": 369,
        "feature_means": [0.0] * 13,
        "feature_scales": [1.0] * 13,
        "calibration": {
            "trade_threshold_pp": -999.0,
            "residual_rmse_pp": 0.5,
            "abstain_without_positive_validation_edge": True,
        },
        "state_dict": model.state_dict(),
    }
    path = tmp_path / "model.pt"
    torch.save(artifact, path)
    actions = MPSPolicy(path).decide(sample_snapshot(200.0), ["AAPL", "MSFT"])
    assert actions
    required = {"action", "symbol", "confidence", "reasoning", "position_size"}
    assert required <= set(actions[0])
    assert actions[0]["action"] in {"buy", "sell", "hold"}
    assert actions[0]["action"] == "hold"
    assert 0.0 <= actions[0]["confidence"] <= 1.0


def test_benchmark_writes_matched_cost_aware_evidence(tmp_path: Path):
    snapshots = []
    start = datetime(2026, 1, 5, 10, tzinfo=timezone.utc)
    symbols = ("AAPL", "MSFT", "JPM", "NKE")
    for step in range(24):
        signals = {}
        for offset, symbol in enumerate(symbols):
            wave = np.sin((step + 1) * (offset + 1) * 0.7)
            price = 100.0 + 5.0 * offset + 0.3 * step + wave
            signals[symbol] = {
                "price": price,
                "rsi": 50.0 + 30.0 * wave,
                "macd": 0.3 * wave,
                "macd_signal": 0.2 * np.cos(step + offset),
                "sma20": price * (1.0 - 0.01 * wave),
                "sma50": price * (1.0 - 0.015 * np.cos(step * 0.5 + offset)),
                "bb_upper": price * (1.03 + 0.002 * offset + 0.002 * wave),
                "bb_lower": price * (0.97 - 0.002 * offset),
            }
        snapshots.append(
            {
                "timestamp": (start + timedelta(hours=step)).isoformat(),
                "top_signals": signals,
                "portfolio": {"cash": 1000.0, "total_equity": 1000.0},
                "current_holdings": {},
            }
        )
    result = run_benchmark(
        snapshots,
        tmp_path,
        train_end=snapshots[12]["timestamp"],
        validation_end=snapshots[17]["timestamp"],
        test_start=snapshots[18]["timestamp"],
        test_end=snapshots[-1]["timestamp"],
        seeds=(0,),
        epochs=3,
        patience=2,
    )
    assert result["configuration"]["mps_parameters"] == 369
    assert result["configuration"]["ann_parameters"] == 369
    assert result["split"]["test_rows"] > 0
    assert (tmp_path / "benchmark_results.json").is_file()
    assert (tmp_path / "benchmark_seed_results.csv").is_file()


def test_v3_policy_enforces_ensemble_validation_abstention(tmp_path: Path):
    models = [ResidualMPSRegressor(seed=seed) for seed in (0, 1)]
    artifact = {
        "schema_version": 3,
        "model_type": "residual_mps_deep_ensemble",
        "feature_names": list(FEATURE_NAMES),
        "seeds": [0, 1],
        "feature_means": [0.0] * 13,
        "feature_scales": [1.0] * 13,
        "state_dicts": [model.state_dict() for model in models],
        "calibration": {
            "trade_threshold_pp": -999.0,
            "residual_rmse_pp": 0.5,
            "uncertainty_penalty_z": 1.0,
            "abstain_without_positive_validation_edge": True,
        },
    }
    path = tmp_path / "v3.pt"
    torch.save(artifact, path)
    action = ResidualMPSEnsemblePolicy(path).decide(
        sample_snapshot(200.0), ["AAPL", "MSFT"]
    )[0]
    assert action["action"] == "hold"
    assert "uncertainty=" in action["reasoning"]


def test_v3_benchmark_uses_a_fresh_split_and_matched_ensembles(tmp_path: Path):
    snapshots = []
    start = datetime(2026, 1, 5, 10, tzinfo=timezone.utc)
    symbols = ("AAPL", "MSFT", "JPM", "NKE")
    for step in range(30):
        signals = {}
        for offset, symbol in enumerate(symbols):
            wave = np.sin((step + 1) * (offset + 1) * 0.7)
            price = 100.0 + 5.0 * offset + 0.3 * step + wave
            signals[symbol] = {
                "price": price,
                "rsi": 50.0 + 30.0 * wave,
                "macd": 0.3 * wave,
                "macd_signal": 0.2 * np.cos(step + offset),
                "sma20": price * (1.0 - 0.01 * wave),
                "sma50": price * (1.0 - 0.015 * np.cos(step * 0.5 + offset)),
                "bb_upper": price * (1.03 + 0.002 * offset + 0.002 * wave),
                "bb_lower": price * (0.97 - 0.002 * offset),
            }
        snapshots.append(
            {
                "timestamp": (start + timedelta(hours=step)).isoformat(),
                "top_signals": signals,
                "portfolio": {"cash": 1000.0, "total_equity": 1000.0},
                "current_holdings": {},
            }
        )
    result = run_v3_benchmark(
        snapshots,
        tmp_path,
        train_end=snapshots[14]["timestamp"],
        validation_end=snapshots[21]["timestamp"],
        test_start=snapshots[22]["timestamp"],
        test_end=snapshots[-1]["timestamp"],
        seeds=(0,),
        ensemble_seeds=(0,),
        epochs=2,
        patience=1,
    )
    assert result["architecture_selected_without_fresh_test"] is True
    assert result["configuration"]["mps_parameters_per_member"] == 586
    assert result["configuration"]["ann_parameters_per_member"] == 586
    assert result["split"]["test_rows"] > 0
