#!/usr/bin/env python
"""Verify the frozen scientific evidence and all published result claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


PPO_CONDITIONS = ("Base FinRL", "ANN signal", "QINN-MPS signal")
SUMMARY_METRICS = (
    "annual_return",
    "sharpe",
    "max_drawdown",
    "annualized_turnover",
    "total_cost",
)

TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".txt"}

def _sha256(path: Path) -> str:
    payload = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _assert_close(actual: float, expected: float) -> None:
    if not math.isclose(
        float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12
    ):
        raise AssertionError(f"{actual!r} does not match {expected!r}")


def _verify_hashes(root: Path, freeze: dict[str, object]) -> int:
    hashes = freeze["sha256"]
    assert isinstance(hashes, dict)
    for relative, expected in hashes.items():
        path = root / relative
        if not path.is_file():
            raise AssertionError(f"Frozen artifact is missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise AssertionError(
                f"Frozen artifact changed: {relative}: {actual} != {expected}"
            )
    return len(hashes)


def _paired_seed_bootstrap(differences: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(20_260_728)
    indices = rng.integers(
        0, len(differences), size=(10_000, len(differences))
    )
    return tuple(
        float(value)
        for value in np.quantile(differences[indices].mean(axis=1), [0.025, 0.975])
    )


def _sign_test(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    positive = int((nonzero > 0).sum())
    tail = min(positive, len(nonzero) - positive)
    probability = sum(
        math.comb(len(nonzero), k) for k in range(tail + 1)
    ) / (2 ** len(nonzero))
    return min(1.0, 2 * probability)


def _moving_block_bootstrap(curves: pd.DataFrame) -> dict[str, float]:
    ann = curves[curves["condition"] == "ANN signal"].pivot(
        index="date", columns="seed", values="daily_return"
    )
    mps = curves[curves["condition"] == "QINN-MPS signal"].pivot(
        index="date", columns="seed", values="daily_return"
    )
    aligned = pd.concat(
        [ann.mean(axis=1).rename("ann"), mps.mean(axis=1).rename("mps")],
        axis=1,
    ).dropna()
    values = aligned.to_numpy()
    rng = np.random.default_rng(2026)
    differences = np.empty(2_000, dtype=float)
    for sample in range(2_000):
        indices: list[int] = []
        while len(indices) < len(values):
            start = int(rng.integers(0, len(values)))
            indices.extend((start + offset) % len(values) for offset in range(20))
        draw = values[np.asarray(indices[: len(values)])]
        differences[sample] = (
            draw[:, 1].mean() - draw[:, 0].mean()
        ) * 252
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {
        "point": float((values[:, 1].mean() - values[:, 0].mean()) * 252),
        "lower": float(lower),
        "upper": float(upper),
        "probability_positive": float(np.mean(differences > 0)),
    }


def _verify_run(
    directory: Path,
    inference_name: str,
    expected_window: str,
    frozen_claims: dict[str, object],
) -> tuple[dict[str, object], int]:
    raw_all = pd.read_csv(directory / "ppo_backtest_metrics.csv")
    raw = raw_all[raw_all["seed"] >= 0].copy()
    expected_keys = {
        (condition, seed)
        for condition in PPO_CONDITIONS
        for seed in range(10)
    }
    actual_keys = set(
        zip(raw["condition"], raw["seed"].astype(int), strict=True)
    )
    if actual_keys != expected_keys or raw.duplicated(["condition", "seed"]).any():
        raise AssertionError(f"{expected_window} PPO endpoints are incomplete")

    manifest = json.loads(
        (directory / "run_manifest.json").read_text(encoding="utf-8")
    )
    if manifest["runtime"]["status"] != "completed":
        raise AssertionError(f"{expected_window} run is not completed")
    if manifest["experiment"]["window_name"] != expected_window:
        raise AssertionError(f"Unexpected run window: {expected_window}")

    summary = pd.read_csv(directory / "condition_summary.csv").set_index(
        "condition"
    )
    for condition, rows in raw.groupby("condition"):
        if int(summary.loc[condition, "n_seeds"]) != 10:
            raise AssertionError(f"Unexpected seed count for {condition}")
        for metric in SUMMARY_METRICS:
            _assert_close(
                rows[metric].mean(), summary.loc[condition, f"{metric}_mean"]
            )
            _assert_close(
                rows[metric].std(ddof=1),
                summary.loc[condition, f"{metric}_std"],
            )
            _assert_close(
                rows[metric].median(),
                summary.loc[condition, f"{metric}_median"],
            )
        for metric, expected in frozen_claims["condition_means"][condition].items():
            _assert_close(summary.loc[condition, f"{metric}_mean"], expected)

    pivot = raw.pivot(index="seed", columns="condition")
    paired_saved = pd.read_csv(
        directory / "paired_seed_effects.csv"
    ).set_index("seed")
    for metric in SUMMARY_METRICS:
        calculated = (
            pivot[metric]["QINN-MPS signal"] - pivot[metric]["ANN signal"]
        )
        for seed, value in calculated.items():
            _assert_close(
                value, paired_saved.loc[seed, f"mps_minus_ann_{metric}"]
            )

    differences = (
        pivot["sharpe"]["QINN-MPS signal"]
        - pivot["sharpe"]["ANN signal"]
    ).to_numpy()
    inference = json.loads(
        (directory / inference_name).read_text(encoding="utf-8")
    )
    paired_claim = frozen_claims["paired_sharpe"]
    _assert_close(differences.mean(), inference["mean_difference"])
    _assert_close(differences.mean(), paired_claim["mean"])
    _assert_close(differences.std(ddof=1), paired_claim["standard_deviation"])
    _assert_close(np.median(differences), paired_claim["median"])
    if int((differences > 0).sum()) != paired_claim["positive_seeds"]:
        raise AssertionError("Positive paired-seed count changed")
    if int((differences < 0).sum()) != paired_claim["negative_seeds"]:
        raise AssertionError("Negative paired-seed count changed")
    lower, upper = _paired_seed_bootstrap(differences)
    _assert_close(lower, paired_claim["bootstrap_95pct"][0])
    _assert_close(upper, paired_claim["bootstrap_95pct"][1])
    _assert_close(
        _sign_test(differences), paired_claim["exact_two_sided_sign_test_p"]
    )

    block = _moving_block_bootstrap(
        pd.read_csv(directory / "equity_curves.csv")
    )
    block_claim = frozen_claims["moving_block_annualized_return_difference"]
    _assert_close(block["point"], block_claim["point"])
    _assert_close(block["lower"], block_claim["bootstrap_95pct"][0])
    _assert_close(block["upper"], block_claim["bootstrap_95pct"][1])

    signals = pd.read_csv(directory / "signal_metrics.csv").set_index(
        ["split", "model"]
    )
    test_split = next(
        split for split in signals.index.get_level_values("split").unique()
        if split.startswith("test_")
    )
    for model, expected in frozen_claims["test_prediction_mse"].items():
        _assert_close(signals.loc[(test_split, model), "mse"], expected)
    return manifest, len(raw)


def _verify_configuration(
    primary: dict[str, object], shifted: dict[str, object]
) -> None:
    primary_experiment = primary["experiment"]
    shifted_experiment = shifted["experiment"]
    expected_dates = {
        "primary": {
            "representation_train_end": "2017-12-29",
            "representation_validation_start": "2018-01-01",
            "test_start": "2019-01-02",
            "test_end": "2023-12-28",
        },
        "shifted": {
            "representation_train_end": "2015-12-31",
            "representation_validation_start": "2016-01-01",
            "test_start": "2017-01-03",
            "test_end": "2018-12-28",
        },
    }
    for name, manifest, experiment in (
        ("primary", primary, primary_experiment),
        ("shifted", shifted, shifted_experiment),
    ):
        for field, expected in expected_dates[name].items():
            if experiment[field] != expected:
                raise AssertionError(f"{name} {field} changed")
        expected_controls = {
            "ppo_seeds": list(range(10)),
            "ppo_timesteps": 20_000,
            "mps_bond_dimension": 2,
            "transaction_cost": 0.001,
            "initial_amount": 1_000_000,
            "hmax": 100,
            "reward_scaling": 0.0001,
        }
        for field, expected in expected_controls.items():
            if experiment[field] != expected:
                raise AssertionError(f"{name} {field} changed")
        if len(manifest["tickers"]) != 15:
            raise AssertionError(f"{name} stock universe changed")
        if len(manifest["representation_features"]) != 13:
            raise AssertionError(f"{name} representation inputs changed")
        if manifest["state_construction"]["base_dimension"] != 181:
            raise AssertionError(f"{name} base state dimension changed")
        if manifest["state_construction"]["signal_dimension"] != 196:
            raise AssertionError(f"{name} signal state dimension changed")
        runtime = manifest["runtime"]
        if (
            runtime["python"] != "3.12.10"
            or runtime["torch"] != "2.10.0+cpu"
            or runtime["cuda_available"] is not False
        ):
            raise AssertionError(f"{name} recorded runtime changed")


def _verify_capacity(root: Path, claims: dict[str, object]) -> int:
    directory = root / "results" / "pilots" / "2026-07-28"
    summary = pd.read_csv(directory / "dimension_summary.csv").set_index(
        "bond_dimension"
    )
    if list(summary.index) != claims["tested_bond_dimensions"]:
        raise AssertionError("Capacity dimensions changed")
    minimum = float(summary["validation_mse"].min())
    expected_tied = summary["validation_mse"] <= minimum * 1.01
    if not expected_tied.equals(summary["within_validation_mse_1pct"]):
        raise AssertionError("Capacity tolerance flags changed")
    selected = summary.index[summary["selected_primary"]].tolist()
    if selected != [claims["selected_bond_dimension"]]:
        raise AssertionError("Selected bond dimension changed")
    selected_count = int(
        summary.loc[claims["selected_bond_dimension"], "mps_parameter_count"]
    )
    if selected_count != claims["selected_parameter_count"]:
        raise AssertionError("Selected MPS parameter count changed")
    for dimension in claims["tested_bond_dimensions"]:
        key = str(dimension)
        _assert_close(
            summary.loc[dimension, "validation_mse"],
            claims["validation_mse"][key],
        )
        _assert_close(
            summary.loc[dimension, "mps_fit_seconds"],
            claims["fit_seconds"][key],
        )
        _assert_close(
            summary.loc[dimension, "mps_minus_ann_sharpe_mean"],
            claims["mps_minus_ann_sharpe_mean"][key],
        )
    paired = pd.read_csv(directory / "dimension_paired.csv")
    for dimension, rows in paired.groupby("bond_dimension"):
        _assert_close(
            rows["mps_minus_ann_sharpe"].mean(),
            summary.loc[dimension, "mps_minus_ann_sharpe_mean"],
        )
        if int((rows["mps_minus_ann_sharpe"] > 0).sum()) != 1:
            raise AssertionError("Capacity pilot positive-seed count changed")
    return len(summary)


def _verify_paper(root: Path) -> int:
    paper = (root / "paper" / "main.tex").read_text(encoding="utf-8")
    claim_groups = {
        "abstract": (
            "mean Sharpe was 0.821 for ANN and 0.784 for MPS",
            "[-0.113, 0.045]",
            "MPS was higher in three of ten seeds",
            "MPS mean Sharpe was 0.714 versus 0.627 for ANN",
        ),
        "capacity": (
            "bond dimension 2",
            "97 parameters",
            "1.280290",
            "1.281044",
            "1.275307",
            "\\(-0.045\\), \\(-0.027\\), and",
            "\\(-0.046\\)",
        ),
        "primary": (
            "standard deviation 0.134; median",
            "-0.087",
            "0.344",
            "1.504 versus 1.515",
        ),
        "shifted": (
            "paired MPS-minus-ANN mean was +0.086",
            "[0.001, 0.195]",
            "+2.34 percentage points",
            "[-0.83, 5.98] percentage points",
        ),
        "conclusion": (
            "Sharpe difference of -0.037",
            "evaluation-window sensitivity",
            "not a stable",
        ),
    }
    for group, fragments in claim_groups.items():
        for fragment in fragments:
            if fragment not in paper:
                raise AssertionError(
                    f"Frozen paper claim changed in {group}: {fragment}"
                )
    for stale in (
        "mean Sharpe was 0.800 for ANN",
        "Shifted-period robustness remains pending",
    ):
        if stale in paper:
            raise AssertionError(f"Stale paper claim returned: {stale}")
    return len(claim_groups)


def _verify_figure_structure(root: Path) -> None:
    for relative in (
        "results/figures/dimension_sensitivity.png",
        "results/figures/final_paired_effect.png",
        "results/figures/shifted_paired_effect.png",
    ):
        if not (root / relative).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise AssertionError(f"Invalid PNG structure: {relative}")
    for relative in (
        "results/figures/dimension_sensitivity.pdf",
        "results/figures/final_paired_effect.pdf",
        "results/figures/shifted_paired_effect.pdf",
    ):
        payload = (root / relative).read_bytes()
        if not payload.startswith(b"%PDF-") or b"%%EOF" not in payload[-1024:]:
            raise AssertionError(f"Invalid PDF structure: {relative}")


def verify_freeze(root: Path) -> dict[str, int]:
    freeze = json.loads(
        (root / "results" / "SCIENTIFIC_RESULTS_FREEZE.json").read_text(
            encoding="utf-8"
        )
    )
    frozen_hashes = _verify_hashes(root, freeze)
    claims = freeze["claims"]
    primary_manifest, primary_endpoints = _verify_run(
        root / "results" / "final",
        "primary_inference.json",
        "primary",
        claims["primary"],
    )
    shifted_manifest, shifted_endpoints = _verify_run(
        root / "results" / "robustness" / "shifted",
        "robustness_inference.json",
        "shifted",
        claims["shifted"],
    )
    _verify_configuration(primary_manifest, shifted_manifest)
    capacity_dimensions = _verify_capacity(root, claims["capacity"])
    paper_claim_groups = _verify_paper(root)
    _verify_figure_structure(root)
    return {
        "frozen_hashes": frozen_hashes,
        "primary_ppo_endpoints": primary_endpoints,
        "shifted_ppo_endpoints": shifted_endpoints,
        "capacity_dimensions": capacity_dimensions,
        "paper_claim_groups": paper_claim_groups,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (defaults to the script's parent repository).",
    )
    return parser.parse_args()


def main() -> None:
    report = verify_freeze(parse_args().root.resolve())
    print("Scientific results freeze verified:")
    for field, value in report.items():
        print(f"- {field}: {value}")


if __name__ == "__main__":
    main()
