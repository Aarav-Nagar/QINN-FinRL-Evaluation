from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).parent / "scripts" / "summarize_budget_pilot.py"
SPEC = importlib.util.spec_from_file_location("budget_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
budget_summary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = budget_summary
SPEC.loader.exec_module(budget_summary)


def make_run(path: Path, budget: int, missing_seed: bool = False) -> Path:
    path.mkdir()
    experiment = {
        "ppo_timesteps": budget,
        "ppo_seeds": [0, 1],
        "mps_bond_dimension": 4,
        "encoder_epochs": 60,
        "encoder_device": "cpu",
    }
    (path / "run_manifest.json").write_text(
        json.dumps({"experiment": experiment}), encoding="utf-8"
    )
    rows = []
    for condition, offset in zip(
        budget_summary.PPO_CONDITIONS, (0.0, 0.1, 0.2), strict=True
    ):
        for seed in (0, 1):
            if missing_seed and condition == "ANN signal" and seed == 1:
                continue
            rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "annual_return": offset + seed * 0.01,
                    "sharpe": offset + seed * 0.02,
                    "max_drawdown": -0.2,
                    "annualized_turnover": 0.3,
                    "total_cost": 100.0,
                    "condition_elapsed_seconds": 10.0,
                    "train_explained_variance": 0.8,
                }
            )
    pd.DataFrame(rows).to_csv(path / "ppo_backtest_metrics.csv", index=False)
    return path


def test_budget_summary_preserves_paired_seed_differences(tmp_path: Path) -> None:
    run = make_run(tmp_path / "run", 5_000)
    summary, paired = budget_summary.summarize([run])
    assert len(summary) == 3
    assert paired.iloc[0]["n_seeds"] == 2
    assert paired.iloc[0]["mps_minus_ann_sharpe_mean"] == pytest.approx(0.1)
    assert paired.iloc[0]["mps_minus_ann_sharpe_positive_seeds"] == 2


def test_budget_summary_rejects_unmatched_seeds(tmp_path: Path) -> None:
    run = make_run(tmp_path / "run", 5_000, missing_seed=True)
    with pytest.raises(ValueError, match="Unmatched seeds"):
        budget_summary.summarize([run])


def test_budget_summary_rejects_duplicate_budgets(tmp_path: Path) -> None:
    first = make_run(tmp_path / "first", 5_000)
    second = make_run(tmp_path / "second", 5_000)
    with pytest.raises(ValueError, match="Duplicate PPO budget"):
        budget_summary.summarize([first, second])
