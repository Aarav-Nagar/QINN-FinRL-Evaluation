from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).parent / "scripts" / "summarize_dimension_pilot.py"
SPEC = importlib.util.spec_from_file_location("dimension_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
dimension_summary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dimension_summary
SPEC.loader.exec_module(dimension_summary)


def make_run(
    path: Path,
    dimension: int,
    validation_mse: float,
    *,
    missing_seed: bool = False,
    timesteps: int = 20_000,
    control_offset: float = 0.0,
) -> Path:
    path.mkdir()
    experiment = {
        "ppo_timesteps": timesteps,
        "ppo_seeds": [0, 1],
        "mps_bond_dimension": dimension,
        "encoder_epochs": 60,
        "encoder_device": "cpu",
    }
    manifest = {
        "experiment": experiment,
        "mps_parameter_count": {2: 97, 4: 369, 8: 1441}[dimension],
        "encoder_runtime": {
            "mps_fit_seconds": float(dimension),
            "signal_inference_seconds": 0.1,
        },
        "runtime": {
            "status": "completed",
            "git_commit": "abc123",
            "completed_at_utc": "2026-07-28T12:00:00+00:00",
        },
    }
    (path / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    portfolio_rows = []
    for condition, offset in zip(
        dimension_summary.PPO_CONDITIONS, (0.0, 0.1, 0.2), strict=True
    ):
        for seed in (0, 1):
            if missing_seed and condition == "ANN signal" and seed == 1:
                continue
            control_drift = (
                control_offset if condition in ("Base FinRL", "ANN signal") else 0
            )
            portfolio_rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "annual_return": offset + seed * 0.01 + control_drift,
                    "sharpe": offset + seed * 0.02 + control_drift,
                    "max_drawdown": -0.2,
                    "annualized_turnover": 0.3,
                    "total_cost": 100.0,
                    "condition_elapsed_seconds": 10.0,
                }
            )
    pd.DataFrame(portfolio_rows).to_csv(
        path / "ppo_backtest_metrics.csv", index=False
    )

    signal_rows = []
    for split in ("validation_2018", "test_2019_2023"):
        for model in ("ANN", "QINN-MPS"):
            signal_rows.append(
                {
                    "split": split,
                    "model": model,
                    "parameter_count": (
                        manifest["mps_parameter_count"]
                        if model == "QINN-MPS"
                        else 369
                    ),
                    "mse": (
                        validation_mse
                        if (split, model) == ("validation_2018", "QINN-MPS")
                        else 1.2
                    ),
                    "mae": 0.8,
                    "directional_accuracy": 0.51,
                    "information_coefficient": 0.01,
                }
            )
    pd.DataFrame(signal_rows).to_csv(path / "signal_metrics.csv", index=False)
    return path


def test_summary_selects_smallest_model_within_one_percent_mse(
    tmp_path: Path,
) -> None:
    runs = [
        make_run(tmp_path / "bd2", 2, 1.005),
        make_run(tmp_path / "bd4", 4, 1.000),
        make_run(tmp_path / "bd8", 8, 1.020),
    ]
    summary, paired = dimension_summary.summarize(runs)
    selected = summary.loc[summary["selected_primary"], "bond_dimension"].item()
    assert selected == 2
    assert summary["within_validation_mse_1pct"].tolist() == [True, True, False]
    assert len(paired) == 6
    assert paired["mps_minus_ann_sharpe"].unique() == pytest.approx([0.1])


def test_summary_selects_clear_validation_winner(tmp_path: Path) -> None:
    runs = [
        make_run(tmp_path / "bd2", 2, 1.02),
        make_run(tmp_path / "bd4", 4, 1.00),
    ]
    summary, _ = dimension_summary.summarize(runs)
    selected = summary.loc[summary["selected_primary"], "bond_dimension"].item()
    assert selected == 4


def test_summary_rejects_unmatched_seeds(tmp_path: Path) -> None:
    run = make_run(tmp_path / "bd2", 2, 1.0, missing_seed=True)
    with pytest.raises(ValueError, match="Unmatched seeds"):
        dimension_summary.summarize([run])


def test_summary_rejects_changed_fixed_settings(tmp_path: Path) -> None:
    runs = [
        make_run(tmp_path / "bd2", 2, 1.0),
        make_run(tmp_path / "bd4", 4, 1.0, timesteps=10_000),
    ]
    with pytest.raises(ValueError, match="fixed experiment settings"):
        dimension_summary.summarize(runs)


def test_summary_rejects_control_drift(tmp_path: Path) -> None:
    runs = [
        make_run(tmp_path / "bd2", 2, 1.0),
        make_run(tmp_path / "bd4", 4, 1.0, control_offset=0.01),
    ]
    with pytest.raises(ValueError, match="control drifted"):
        dimension_summary.summarize(runs)


def test_summary_reports_missing_portfolio_schema(tmp_path: Path) -> None:
    run = make_run(tmp_path / "bd2", 2, 1.0)
    metrics_path = run / "ppo_backtest_metrics.csv"
    metrics = pd.read_csv(metrics_path).drop(columns="condition_elapsed_seconds")
    metrics.to_csv(metrics_path, index=False)
    with pytest.raises(
        ValueError, match="Missing portfolio columns.*condition_elapsed_seconds"
    ):
        dimension_summary.summarize([run])


def test_summary_requires_expected_dimension_set(tmp_path: Path) -> None:
    runs = [
        make_run(tmp_path / "bd2", 2, 1.0),
        make_run(tmp_path / "bd4", 4, 1.0),
    ]
    with pytest.raises(ValueError, match="Dimension set mismatch"):
        dimension_summary.summarize(
            runs, expected_dimensions={2, 4, 8}
        )


def test_artifact_manifest_hashes_inputs_and_outputs(tmp_path: Path) -> None:
    runs = [
        make_run(tmp_path / "bd2", 2, 1.005),
        make_run(tmp_path / "bd4", 4, 1.000),
        make_run(tmp_path / "bd8", 8, 1.020),
    ]
    summary, paired = dimension_summary.summarize(
        runs, expected_dimensions={2, 4, 8}
    )
    summary_path = tmp_path / "summary.csv"
    paired_path = tmp_path / "paired.csv"
    summary.to_csv(summary_path, index=False)
    paired.to_csv(paired_path, index=False)
    manifest = dimension_summary.build_artifact_manifest(
        runs, summary, summary_path, paired_path
    )
    assert manifest["selection"]["selected_bond_dimension"] == 2
    assert manifest["selection"]["test_or_trading_metrics_used_for_selection"] is False
    assert len(manifest["inputs"]) == 3
    expected_hash = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    assert manifest["outputs"]["summary.csv"] == expected_hash
