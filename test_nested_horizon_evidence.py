from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "results" / "robustness" / "nested_horizons"
RUN_DIR = ROOT / "results" / "final"
FIGURES = ROOT / "results" / "figures"
EXPECTED = {
    1: (1.5485939838361393, 1.5534201108895043, 0.0048261270533650254, 4),
    2: (1.086568877824651, 1.0239858819245524, -0.06258299590009872, 2),
    3: (1.1683385100128163, 1.0862272857451745, -0.08211122426764186, 3),
    4: (0.5968129123929505, 0.5683638787261989, -0.02844903366675155, 3),
    5: (0.8210048315496895, 0.78439743209746, -0.03660739945222944, 3),
}

SCRIPT_PATH = ROOT / "scripts" / "summarize_nested_horizons.py"
SPEC = importlib.util.spec_from_file_location(
    "summarize_nested_horizons_evidence", SCRIPT_PATH
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def digest(path: Path) -> str:
    payload = path.read_bytes()
    if path.suffix.lower() not in {".png", ".pdf"}:
        payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def test_nested_horizon_manifest_binds_every_input_output_and_figure() -> None:
    manifest = json.loads(
        (EVIDENCE / "nested_horizon_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["artifact"] == "nested_evaluation_horizon_analysis"
    assert manifest["analysis_role"] == "post-hoc exploratory"
    assert manifest["retraining"] is False
    assert manifest["horizons_years"] == [1, 2, 3, 4, 5]

    input_paths = {
        "protocol": ROOT / "docs" / "NESTED_HORIZON_PROTOCOL.md",
        "scientific_freeze": ROOT / "results" / "SCIENTIFIC_RESULTS_FREEZE.json",
        "run_manifest.json": RUN_DIR / "run_manifest.json",
        "ppo_backtest_metrics.csv": RUN_DIR / "ppo_backtest_metrics.csv",
        "equity_curves.csv": RUN_DIR / "equity_curves.csv",
    }
    for key, path in input_paths.items():
        assert digest(path) == manifest["inputs"][key]
    for filename, expected_hash in manifest["outputs"].items():
        assert digest(EVIDENCE / filename) == expected_hash

    plot_script = ROOT / manifest["figure_generator"]["script"]
    assert digest(plot_script) == manifest["figure_generator"]["script_sha256"]
    for filename, expected_hash in manifest["figures"].items():
        assert digest(FIGURES / filename) == expected_hash


def test_nested_horizon_saved_rows_reproduce_frozen_curves_and_claims() -> None:
    curves = pd.read_csv(RUN_DIR / "equity_curves.csv")
    recomputed = MODULE.score_horizons(curves)
    saved = pd.read_csv(EVIDENCE / "horizon_seed_metrics.csv")
    pd.testing.assert_frame_equal(
        saved,
        recomputed,
        check_exact=False,
        rtol=1e-13,
        atol=1e-13,
    )
    assert len(saved) == 150
    assert not saved.duplicated(
        ["horizon_years", "condition", "seed"]
    ).any()

    condition = pd.read_csv(
        EVIDENCE / "horizon_condition_summary.csv"
    ).set_index(["horizon_years", "condition"])
    summary = pd.read_csv(
        EVIDENCE / "horizon_paired_metric_summary.csv"
    ).set_index(["horizon_years", "metric"])
    paired = pd.read_csv(EVIDENCE / "horizon_paired_seed_effects.csv")
    assert len(condition) == 15
    assert len(summary) == 25
    assert len(paired) == 50
    for horizon, (ann, mps, difference, wins) in EXPECTED.items():
        assert condition.loc[(horizon, "ANN signal"), "sharpe_mean"] == pytest.approx(
            ann
        )
        assert condition.loc[
            (horizon, "QINN-MPS signal"), "sharpe_mean"
        ] == pytest.approx(mps)
        row = summary.loc[(horizon, "sharpe")]
        assert row["mean_difference"] == pytest.approx(difference)
        assert int(row["positive_seeds"]) == wins
        assert row["paired_seed_bootstrap_95pct_lower"] <= 0
        assert row["paired_seed_bootstrap_95pct_upper"] >= 0

    final_metrics = pd.read_csv(RUN_DIR / "ppo_backtest_metrics.csv")
    MODULE.validate_full_horizon(saved, final_metrics)


def test_nested_horizon_paper_and_visuals_retain_reporting_boundaries() -> None:
    inference = json.loads(
        (EVIDENCE / "nested_horizon_inference.json").read_text(encoding="utf-8")
    )
    assert inference["mps_higher_mean_sharpe_horizons"] == [1]
    assert inference["ann_higher_mean_sharpe_horizons"] == [2, 3, 4, 5]
    assert inference["sharpe_intervals_including_zero"] == [1, 2, 3, 4, 5]

    protocol = (ROOT / "docs" / "NESTED_HORIZON_PROTOCOL.md").read_text(
        encoding="utf-8"
    )
    for boundary in (
        "post-hoc exploratory",
        "No encoder or PPO policy will be refit",
        "four- and five-year prefixes are mandatory",
        "nested and strongly dependent",
    ):
        assert boundary in protocol

    paper = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    for token in (
        "+0.005",
        "-0.063",
        "-0.082",
        "-0.028",
        "-0.037",
        "4/10",
        "2/10",
        "post-hoc",
        "All five paired-seed intervals include zero",
    ):
        assert token in paper

    png = FIGURES / "nested_horizon_paired_effect.png"
    pdf = FIGURES / "nested_horizon_paired_effect.pdf"
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf.read_bytes().startswith(b"%PDF")
    assert png.stat().st_size > 100_000
    assert pdf.stat().st_size > 10_000
