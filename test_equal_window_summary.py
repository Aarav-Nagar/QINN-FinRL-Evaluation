from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).parent / "scripts" / "summarize_equal_windows.py"
SPEC = importlib.util.spec_from_file_location("equal_window_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
summary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = summary
SPEC.loader.exec_module(summary)


def make_run(path: Path, label: str, *, window_name: str | None = None) -> Path:
    path.mkdir()
    dates = summary.WINDOWS[label]
    experiment = (
        summary.COMMON_CONFIGURATION
        | dates
        | {"window_name": window_name or dates["window_name"]}
    )
    manifest = {
        "experiment": experiment,
        "runtime": {"status": "completed", "git_commit": "abc123"},
    }
    (path / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    rows = []
    offsets = {"Base FinRL": 0.0, "ANN signal": 0.1, "QINN-MPS signal": 0.2}
    for condition, offset in offsets.items():
        for seed in summary.SEEDS:
            rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "sharpe": offset + seed / 100,
                    "annual_return": offset + seed / 200,
                    "max_drawdown": -0.3 + offset,
                    "annualized_turnover": 0.4 - offset,
                    "total_cost": 1000 - 100 * offset,
                }
            )
    pd.DataFrame(rows).to_csv(path / "ppo_backtest_metrics.csv", index=False)
    signal_rows = []
    for model, offset in (("ANN", 0.0), ("QINN-MPS", 0.1)):
        signal_rows.append(
            {
                "split": summary.expected_test_split(label),
                "model": model,
                "mse": 1.0 - offset,
                "mae": 0.8 - offset,
                "directional_accuracy": 0.5 + offset,
                "information_coefficient": 0.01 + offset,
            }
        )
    pd.DataFrame(signal_rows).to_csv(path / "signal_metrics.csv", index=False)
    return path


def make_panel(tmp_path: Path) -> dict[str, Path]:
    return {
        label: make_run(tmp_path / label, label) for label in summary.WINDOWS
    }


def test_load_window_rejects_configuration_drift(tmp_path: Path) -> None:
    run = make_run(
        tmp_path / "run",
        "2019-2020",
        window_name="equal_2021_2022",
    )
    with pytest.raises(ValueError, match="2019-2020 window_name"):
        summary.load_window("2019-2020", run)


def test_load_window_requires_every_matched_seed(tmp_path: Path) -> None:
    run = make_run(tmp_path / "run", "2021-2022")
    metrics = pd.read_csv(run / "ppo_backtest_metrics.csv")
    metrics = metrics[
        ~((metrics["condition"] == "ANN signal") & (metrics["seed"] == 9))
    ]
    metrics.to_csv(run / "ppo_backtest_metrics.csv", index=False)
    with pytest.raises(ValueError, match="seeds for ANN signal"):
        summary.load_window("2021-2022", run)


def test_summary_reports_all_metrics_and_windows(tmp_path: Path) -> None:
    loaded = {
        label: summary.load_window(label, path)
        for label, path in make_panel(tmp_path).items()
    }
    condition, paired, paired_summary, signal, inference = summary.summarize(
        loaded
    )
    assert len(condition) == 9
    assert len(paired) == 30
    assert len(paired_summary) == 15
    assert len(signal) == 12
    assert set(paired_summary["metric"]) == set(summary.PORTFOLIO_METRICS)
    assert paired["mps_minus_ann_sharpe"].tolist() == pytest.approx([0.1] * 30)
    assert inference["mean_sharpe_sign_pattern"] == "positive_in_all_windows"


def test_write_outputs_hashes_all_inputs_and_is_deterministic(
    tmp_path: Path,
) -> None:
    runs = make_panel(tmp_path)
    protocol = tmp_path / "protocol.md"
    protocol.write_text("frozen before outcomes\n", encoding="utf-8")
    output = tmp_path / "output"
    summary.write_outputs(runs, output, protocol)
    first = {
        path.name: path.read_bytes()
        for path in output.iterdir()
        if path.is_file()
    }
    summary.write_outputs(runs, output, protocol)
    second = {
        path.name: path.read_bytes()
        for path in output.iterdir()
        if path.is_file()
    }
    assert first == second
    assert all(b"\r\n" not in content for content in second.values())
    manifest = json.loads(second["equal_window_manifest.json"])
    assert manifest["artifact"] == "equal_length_temporal_robustness"
    assert len(manifest["inputs"]) == 10
    assert set(manifest["outputs"]) == {
        "window_condition_summary.csv",
        "window_paired_seed_effects.csv",
        "window_paired_metric_summary.csv",
        "window_signal_quality.csv",
        "equal_window_inference.json",
    }
