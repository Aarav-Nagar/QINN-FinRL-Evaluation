from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


SCRIPT = Path(__file__).with_name("run_experiment.py")
SPEC = importlib.util.spec_from_file_location("experiment", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
experiment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = experiment
SPEC.loader.exec_module(experiment)


def test_parameter_matched_encoders() -> None:
    ann = experiment.ANNRegressor(len(experiment.REPRESENTATION_FEATURES))
    mps = experiment.MPSRegressor(len(experiment.REPRESENTATION_FEATURES), 4)
    assert experiment.parameter_count(ann) == 369
    assert experiment.parameter_count(mps) == 369


def test_bond_dimension_changes_mps_capacity() -> None:
    counts = [
        experiment.parameter_count(
            experiment.MPSRegressor(len(experiment.REPRESENTATION_FEATURES), dimension)
        )
        for dimension in (2, 4, 8)
    ]
    assert counts == [97, 369, 1441]


def test_valid_sensitivity_configs_are_accepted() -> None:
    for dimension in (2, 4, 8):
        experiment.validate_config(
            experiment.ExperimentConfig(
                ppo_seeds=(0,),
                ppo_timesteps=512,
                mps_bond_dimension=dimension,
                encoder_epochs=1,
                encoder_patience=1,
            )
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ppo_seeds", (), "At least one PPO seed"),
        ("ppo_seeds", (0, 0), "unique"),
        ("ppo_seeds", (-1,), "non-negative"),
        ("ppo_timesteps", 0, "positive"),
        ("mps_bond_dimension", 0, "positive"),
        ("encoder_epochs", 0, "positive"),
        ("encoder_patience", 0, "positive"),
        ("encoder_batch_size", 0, "positive"),
        ("encoder_learning_rate", 0.0, "positive"),
        ("transaction_cost", 1.0, r"\[0, 1\)"),
        ("encoder_device", "tpu", "auto, cpu, cuda"),
    ],
)
def test_invalid_configs_are_rejected(field: str, value, message: str) -> None:
    config = replace(experiment.ExperimentConfig(), **{field: value})
    with pytest.raises(ValueError, match=message):
        experiment.validate_config(config)


def test_mps_feature_map_is_unit_normalized() -> None:
    values = torch.linspace(-5, 5, 100).reshape(10, 10)
    mapped = experiment.MPSRegressor.local_feature_map(values)
    norms = torch.linalg.vector_norm(mapped, dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms))


def test_encoder_device_resolution_is_explicit(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert experiment.resolve_encoder_device("auto") == torch.device("cpu")
    assert experiment.resolve_encoder_device("cpu") == torch.device("cpu")
    with pytest.raises(RuntimeError, match="cannot access CUDA"):
        experiment.resolve_encoder_device("cuda")


def test_runtime_metadata_records_environment(monkeypatch) -> None:
    monkeypatch.setattr(experiment, "current_git_commit", lambda: "abc123")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    started = experiment.datetime(2026, 7, 27, tzinfo=experiment.UTC)
    metadata = experiment.runtime_metadata(
        experiment.ExperimentConfig(encoder_device="auto"),
        started,
        elapsed_seconds=12.5,
    )
    assert metadata["status"] == "completed"
    assert metadata["git_commit"] == "abc123"
    assert metadata["encoder_device_resolved"] == "cpu"
    assert metadata["ppo_device"] == "cpu"
    assert metadata["elapsed_seconds"] == 12.5


def test_partial_results_resume_complete_condition_seed_pairs(
    tmp_path: Path,
) -> None:
    config = experiment.ExperimentConfig()
    (tmp_path / "partial_config.json").write_text(
        experiment.json.dumps(experiment.asdict(config)), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "condition": ["Base FinRL", "ANN signal"],
            "seed": [0, 0],
            "sharpe": [0.5, 0.6],
        }
    ).to_csv(tmp_path / "ppo_backtest_metrics.partial.csv", index=False)
    pd.DataFrame(
        {
            "condition": ["Base FinRL", "Base FinRL", "ANN signal"],
            "seed": [0, 0, 0],
            "date": ["2020-01-01", "2020-01-02", "2020-01-01"],
            "account_value": [100.0, 101.0, 100.0],
        }
    ).to_csv(tmp_path / "equity_curves.partial.csv", index=False)
    metrics, curves = experiment.load_partial_results(tmp_path, config)
    assert [(row["condition"], row["seed"]) for row in metrics] == [
        ("Base FinRL", 0),
        ("ANN signal", 0),
    ]
    assert len(curves) == 2


def test_partial_results_reject_mismatched_run_sets(tmp_path: Path) -> None:
    config = experiment.ExperimentConfig()
    (tmp_path / "partial_config.json").write_text(
        experiment.json.dumps(experiment.asdict(config)), encoding="utf-8"
    )
    pd.DataFrame(
        {"condition": ["Base FinRL"], "seed": [0]}
    ).to_csv(tmp_path / "ppo_backtest_metrics.partial.csv", index=False)
    pd.DataFrame(
        {
            "condition": ["ANN signal"],
            "seed": [0],
            "date": ["2020-01-01"],
            "account_value": [100.0],
        }
    ).to_csv(tmp_path / "equity_curves.partial.csv", index=False)
    with pytest.raises(RuntimeError, match="different runs"):
        experiment.load_partial_results(tmp_path, config)


def test_partial_results_reject_different_configuration(tmp_path: Path) -> None:
    saved = experiment.ExperimentConfig(ppo_timesteps=512)
    requested = experiment.ExperimentConfig(ppo_timesteps=1024)
    (tmp_path / "partial_config.json").write_text(
        experiment.json.dumps(experiment.asdict(saved)), encoding="utf-8"
    )
    pd.DataFrame(
        {"condition": ["Base FinRL"], "seed": [0]}
    ).to_csv(tmp_path / "ppo_backtest_metrics.partial.csv", index=False)
    pd.DataFrame(
        {
            "condition": ["Base FinRL"],
            "seed": [0],
            "date": ["2020-01-01"],
            "account_value": [100.0],
        }
    ).to_csv(tmp_path / "equity_curves.partial.csv", index=False)
    with pytest.raises(RuntimeError, match="different configuration"):
        experiment.load_partial_results(tmp_path, requested)


def test_metrics_identify_drawdown_and_return() -> None:
    dates = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
    values = [100.0, 110.0, 88.0, 121.0]
    metrics, curve = experiment.calculate_metrics(
        dates,
        values,
        1.0,
        2,
        gross_traded_notionals=[10.0, 20.0, 30.0],
        daily_turnovers=[0.10, 0.20, 0.30],
        transaction_cost_rate=0.01,
    )
    assert np.isclose(metrics["total_return"], 0.21)
    assert np.isclose(metrics["max_drawdown"], -0.20)
    assert metrics["total_cost"] == 1.0
    assert metrics["trade_count"] == 2
    assert metrics["gross_traded_notional"] == 60.0
    assert np.isclose(metrics["cumulative_turnover"], 0.60)
    assert np.isclose(metrics["average_daily_turnover"], 0.20)
    assert np.isclose(metrics["annualized_turnover"], 50.4)
    assert list(curve.columns) == [
        "date",
        "account_value",
        "gross_traded_notional",
        "daily_turnover",
        "transaction_cost",
        "daily_return",
    ]
    assert np.isclose(curve["transaction_cost"].sum(), 0.60)


def test_finrl_state_dimension_formula() -> None:
    stock_dim = len(experiment.TICKERS)
    base = 1 + 2 * stock_dim + len(experiment.FINRL_FEATURES) * stock_dim
    signal = (
        1 + 2 * stock_dim + (len(experiment.FINRL_FEATURES) + 1) * stock_dim
    )
    assert base == 181
    assert signal == 196


def test_calendar_period_metrics_are_separated_by_year() -> None:
    curves = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2019-12-31", "2020-01-02", "2020-01-03", "2021-01-04"]
            ),
            "account_value": [100.0, 110.0, 99.0, 118.8],
            "daily_return": [0.0, 0.10, -0.10, 0.20],
            "gross_traded_notional": [0.0, 10.0, 20.0, 30.0],
            "daily_turnover": [0.0, 0.10, 0.20, 0.30],
            "transaction_cost": [0.0, 0.01, 0.02, 0.03],
            "condition": ["ANN signal"] * 4,
            "seed": [0] * 4,
        }
    )
    annual = experiment.period_metrics(curves)
    assert annual["year"].tolist() == [2019, 2020, 2021]
    year_2020 = annual.loc[annual["year"] == 2020].iloc[0]
    assert np.isclose(year_2020["period_return"], -0.01)
    assert np.isclose(year_2020["cumulative_turnover"], 0.30)
    assert np.isclose(year_2020["total_cost"], 0.03)


def test_seed_summary_uses_student_t_interval() -> None:
    frame = pd.DataFrame(
        {
            "condition": ["ANN signal"] * 3,
            "seed": [0, 1, 2],
            "sharpe": [0.5, 0.7, 0.9],
        }
    )
    summary = experiment.summarize_across_seeds(
        frame, ["condition"], ["sharpe"]
    ).iloc[0]
    expected_margin = 4.303 * 0.2 / np.sqrt(3)
    assert summary["n_seeds"] == 3
    assert np.isclose(summary["mean"], 0.7)
    assert np.isclose(summary["ci95_lower"], 0.7 - expected_margin)
    assert np.isclose(summary["ci95_upper"], 0.7 + expected_margin)
