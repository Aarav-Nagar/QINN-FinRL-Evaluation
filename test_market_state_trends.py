from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).parent / "scripts" / "summarize_market_states.py"
SPEC = importlib.util.spec_from_file_location("market_state_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
summary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = summary
SPEC.loader.exec_module(summary)


def synthetic_curves(periods: int = 40) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=periods)
    benchmark_returns = np.linspace(-0.02, 0.02, periods)
    benchmark_values = 1_000_000 * np.cumprod(1 + benchmark_returns)
    rows = []
    for date, daily_return, value in zip(
        dates,
        benchmark_returns,
        benchmark_values,
        strict=True,
    ):
        rows.append(
            {
                "date": date,
                "account_value": value,
                "daily_return": daily_return,
                "condition": summary.BENCHMARK,
                "seed": -1,
            }
        )
    for condition, offset in ((summary.ANN, 0.0), (summary.MPS, 0.0001)):
        for seed in summary.SEEDS:
            for date, benchmark_return in zip(
                dates,
                benchmark_returns,
                strict=True,
            ):
                rows.append(
                    {
                        "date": date,
                        "account_value": 1_000_000,
                        "daily_return": benchmark_return / 2 + offset,
                        "condition": condition,
                        "seed": seed,
                    }
                )
    return pd.DataFrame(rows)


def test_state_masks_partition_every_eligible_date() -> None:
    curves = synthetic_curves()
    masks = summary.benchmark_state_masks(curves)

    for family, labels in summary.STATE_LABELS.items():
        coverage = sum(masks[(family, label)].astype(int) for label in labels)
        if family == "benchmark_volatility":
            assert (coverage.iloc[:19] == 0).all()
            assert (coverage.iloc[19:] == 1).all()
        else:
            assert (coverage == 1).all()


def test_market_state_scoring_reports_all_cells_and_matched_seeds() -> None:
    scored = summary.score_market_states("2019-2020", synthetic_curves())
    expected_cells = sum(len(labels) for labels in summary.STATE_LABELS.values())

    assert len(scored) == expected_cells * 10
    assert scored["mps_minus_ann_annualized_mean_return"].to_numpy() == (
        pytest.approx([0.0252] * len(scored))
    )
    summarized = summary.summarize_market_states(scored)
    assert len(summarized) == expected_cells
    assert (summarized["positive_seeds"] == 10).all()


def test_cross_window_summary_requires_and_reports_all_windows() -> None:
    base = summary.summarize_market_states(
        summary.score_market_states("2019-2020", synthetic_curves())
    )
    frames = []
    for label, multiplier in zip(summary.WINDOWS, (1.0, -1.0, 2.0), strict=True):
        frame = base.copy()
        frame["window"] = label
        frame["mean_mps_minus_ann_annualized_return"] *= multiplier
        frames.append(frame)
    result = summary.cross_window_summary(pd.concat(frames, ignore_index=True))

    assert len(result) == sum(
        len(labels) for labels in summary.STATE_LABELS.values()
    )
    assert not result["sign_consistent_across_windows"].any()
    assert (result["positive_windows"] == 2).all()
    assert (result["negative_windows"] == 1).all()


def test_direction_contribution_exactly_reconciles_each_window() -> None:
    base = summary.summarize_market_states(
        summary.score_market_states("2019-2020", synthetic_curves())
    )
    frames = []
    for label in summary.WINDOWS:
        frame = base.copy()
        frame["window"] = label
        frames.append(frame)
    result = summary.direction_contribution(
        pd.concat(frames, ignore_index=True)
    )

    assert len(result) == 3
    assert (
        result["negative_day_fraction"]
        + result["nonnegative_day_fraction"]
    ).to_numpy() == pytest.approx([1.0] * 3)
    assert result[
        "reconciled_full_window_annualized_mean_difference"
    ].to_numpy() == pytest.approx([0.0252] * 3)


def test_paired_bootstrap_is_deterministic() -> None:
    values = np.linspace(-0.1, 0.2, 10)
    assert summary.paired_bootstrap(values) == summary.paired_bootstrap(values)


def test_published_market_state_bundle_is_complete_and_hash_bound() -> None:
    root = Path(__file__).parent
    output = root / "results" / "robustness" / "market_states"
    manifest_path = output / "market_state_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["row_counts"] == {
        "calendar_year_paired_metrics": 300,
        "calendar_year_summary": 30,
        "direction_contribution": 3,
        "market_state_cross_window": 9,
        "market_state_seed_effects": 270,
        "market_state_summary": 27,
    }
    for relative, expected in (
        manifest["source_sha256"] | manifest["output_sha256"]
    ).items():
        assert summary.sha256(root / relative) == expected

    inference = json.loads(
        (output / "market_state_inference.json").read_text(encoding="utf-8")
    )
    assert inference["analysis_role"].startswith("post-hoc exploratory")
    assert inference["market_state_cells"] == 27
