from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).parent
EVIDENCE = ROOT / "results" / "robustness" / "equal_windows"
RUNS = {
    "2017-2018": ROOT / "results" / "robustness" / "shifted",
    "2019-2020": EVIDENCE / "2019-2020",
    "2021-2022": EVIDENCE / "2021-2022",
}
EXPECTED = {
    "2017-2018": (0.6274637144159894, 0.7137091951839826, 0.08624548076799318, 9),
    "2019-2020": (0.8703265765716106, 0.9627771811936399, 0.09245060462202927, 7),
    "2021-2022": (0.037626712688394966, 0.1772887936045965, 0.13966208091620153, 6),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_equal_window_manifest_hashes_every_source_and_output() -> None:
    manifest = json.loads(
        (EVIDENCE / "equal_window_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["artifact"] == "equal_length_temporal_robustness"
    assert manifest["windows"].keys() == EXPECTED.keys()
    protocol = ROOT / manifest["protocol"]
    assert digest(protocol) == manifest["inputs"]["protocol"]
    for key, expected_hash in manifest["inputs"].items():
        if key == "protocol":
            continue
        label, filename = key.split("/", 1)
        assert digest(RUNS[label] / filename) == expected_hash
    for filename, expected_hash in manifest["outputs"].items():
        assert digest(EVIDENCE / filename) == expected_hash


def test_equal_window_raw_rows_reproduce_published_sharpe_claims() -> None:
    paired_summary = pd.read_csv(
        EVIDENCE / "window_paired_metric_summary.csv"
    ).set_index(["window", "metric"])
    for label, (ann_expected, mps_expected, difference_expected, wins_expected) in (
        EXPECTED.items()
    ):
        manifest = json.loads(
            (RUNS[label] / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["runtime"]["status"] == "completed"
        assert manifest["experiment"]["ppo_seeds"] == list(range(10))
        assert manifest["experiment"]["ppo_timesteps"] == 20_000
        assert manifest["experiment"]["mps_bond_dimension"] == 2
        raw = pd.read_csv(RUNS[label] / "ppo_backtest_metrics.csv")
        raw = raw[raw["seed"] >= 0]
        assert len(raw) == 30
        assert not raw.duplicated(["condition", "seed"]).any()
        pivot = raw.pivot(index="seed", columns="condition")
        ann = float(pivot["sharpe"]["ANN signal"].mean())
        mps = float(pivot["sharpe"]["QINN-MPS signal"].mean())
        difference = pivot["sharpe"]["QINN-MPS signal"] - pivot["sharpe"][
            "ANN signal"
        ]
        assert ann == pytest.approx(ann_expected)
        assert mps == pytest.approx(mps_expected)
        assert float(difference.mean()) == pytest.approx(difference_expected)
        assert int((difference > 0).sum()) == wins_expected
        saved = paired_summary.loc[(label, "sharpe")]
        assert saved["mean_difference"] == pytest.approx(difference_expected)
        assert int(saved["positive_seeds"]) == wins_expected


def test_equal_window_claims_and_figure_are_bound_to_evidence() -> None:
    inference = json.loads(
        (EVIDENCE / "equal_window_inference.json").read_text(encoding="utf-8")
    )
    assert inference["mean_sharpe_sign_pattern"] == "positive_in_all_windows"
    assert inference["windows_with_positive_mean_sharpe_difference"] == 3
    summary = pd.read_csv(EVIDENCE / "window_paired_metric_summary.csv")
    sharpe = summary[summary["metric"] == "sharpe"].set_index("window")
    assert sharpe.loc[
        "2017-2018", "paired_seed_bootstrap_95pct_lower"
    ] == pytest.approx(0.0011190756593376)
    assert sharpe.loc[
        "2019-2020", "paired_seed_bootstrap_95pct_lower"
    ] == pytest.approx(-0.0079794396860069)
    assert sharpe.loc[
        "2021-2022", "paired_seed_bootstrap_95pct_lower"
    ] == pytest.approx(-0.0679706446440746)
    paper = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    for token in ("+0.086", "+0.092", "+0.140", "9/10", "7/10", "6/10"):
        assert token in paper
    png = ROOT / "results" / "figures" / "equal_window_paired_effect.png"
    pdf = ROOT / "results" / "figures" / "equal_window_paired_effect.pdf"
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf.read_bytes().startswith(b"%PDF")
    assert png.stat().st_size > 100_000
    assert pdf.stat().st_size > 10_000
