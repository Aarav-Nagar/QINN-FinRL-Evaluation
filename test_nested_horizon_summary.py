import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parent
SCRIPT_PATH = ROOT / "scripts" / "summarize_nested_horizons.py"
SPEC = importlib.util.spec_from_file_location("summarize_nested_horizons", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def synthetic_curves() -> pd.DataFrame:
    dates = pd.to_datetime(
        [
            "2019-01-02",
            "2019-12-31",
            "2020-12-31",
            "2021-12-31",
            "2022-12-30",
            "2023-12-28",
        ]
    )
    rows = []
    condition_growth = {
        "Base FinRL": 0.010,
        "ANN signal": 0.012,
        "QINN-MPS signal": 0.014,
    }
    for condition in MODULE.CONDITIONS:
        for seed in MODULE.SEEDS:
            values = [1_000_000.0]
            for step in range(1, len(dates)):
                growth = condition_growth[condition] + seed * 0.0001
                values.append(values[-1] * (1.0 + growth + step * 0.0002))
            for index, (date, value) in enumerate(zip(dates, values, strict=True)):
                rows.append(
                    {
                        "date": date,
                        "account_value": value,
                        "gross_traded_notional": 0.0 if index == 0 else 1_000.0,
                        "daily_turnover": 0.0 if index == 0 else 0.001,
                        "transaction_cost": 0.0 if index == 0 else 1.0,
                        "daily_return": 0.0,
                        "condition": condition,
                        "seed": seed,
                    }
                )
    return pd.DataFrame(rows)


def test_compute_prefix_metrics_uses_saved_steps_and_costs():
    prefix = pd.DataFrame(
        {
            "date": pd.to_datetime(["2019-01-02", "2019-01-03", "2019-01-04"]),
            "account_value": [100.0, 110.0, 121.0],
            "gross_traded_notional": [0.0, 100.0, 200.0],
            "daily_turnover": [0.0, 0.1, 0.2],
            "transaction_cost": [0.0, 1.0, 2.0],
        }
    )
    metrics = MODULE.compute_prefix_metrics(prefix)
    assert metrics["observations"] == 3
    assert metrics["trading_steps"] == 2
    assert metrics["total_return"] == pytest.approx(0.21)
    assert metrics["annualized_turnover"] == pytest.approx(37.8)
    assert metrics["total_cost"] == pytest.approx(3.0)


def test_validate_curves_rejects_one_drifted_date_grid():
    curves = synthetic_curves()
    mask = (
        (curves["condition"] == "ANN signal")
        & (curves["seed"] == 3)
        & (curves["date"] == pd.Timestamp("2020-12-31"))
    )
    curves.loc[mask, "date"] = pd.Timestamp("2020-12-30")
    with pytest.raises(ValueError, match="Date grid drift"):
        MODULE.validate_curves(curves)


def test_validate_curves_explicitly_excludes_saved_benchmark():
    curves = synthetic_curves()
    reference = curves[
        (curves["condition"] == "Base FinRL") & (curves["seed"] == 0)
    ].copy()
    reference["condition"] = MODULE.BENCHMARK_CONDITION
    reference["seed"] = -1
    with_benchmark = pd.concat([curves, reference], ignore_index=True)
    validated = MODULE.validate_curves(with_benchmark)
    assert MODULE.BENCHMARK_CONDITION not in validated["condition"].unique()
    assert len(validated) == len(curves)


def test_validate_curves_rejects_unknown_condition():
    curves = synthetic_curves()
    extra = curves[
        (curves["condition"] == "Base FinRL") & (curves["seed"] == 0)
    ].copy()
    extra["condition"] = "Unregistered strategy"
    with pytest.raises(ValueError, match="unexpected conditions"):
        MODULE.validate_curves(pd.concat([curves, extra], ignore_index=True))


def test_score_and_summarize_report_every_locked_prefix():
    scored = MODULE.score_horizons(synthetic_curves())
    assert len(scored) == 150
    assert sorted(scored["horizon_years"].unique().tolist()) == [1, 2, 3, 4, 5]
    assert scored.groupby("horizon_years").size().tolist() == [30] * 5
    condition, paired, paired_summary, inference = MODULE.summarize_scores(scored)
    assert len(condition) == 15
    assert len(paired) == 50
    assert len(paired_summary) == 25
    assert inference["horizons_years"] == [1, 2, 3, 4, 5]
    sharpe = paired_summary[paired_summary["metric"] == "sharpe"]
    assert len(sharpe) == 5
    assert sharpe["n_matched_seeds"].tolist() == [10] * 5


def test_full_horizon_validator_detects_metric_drift():
    scored = MODULE.score_horizons(synthetic_curves())
    frozen = scored[scored["horizon_years"] == 5][
        ["condition", "seed", *MODULE.METRICS]
    ].copy()
    MODULE.validate_full_horizon(scored, frozen)
    frozen.loc[frozen.index[0], "sharpe"] += 0.01
    with pytest.raises(ValueError, match="does not reproduce"):
        MODULE.validate_full_horizon(scored, frozen)


def test_paired_bootstrap_is_deterministic():
    values = pd.Series(np.linspace(-0.2, 0.3, 10))
    first = MODULE.paired_bootstrap(values)
    second = MODULE.paired_bootstrap(values)
    assert first == second
