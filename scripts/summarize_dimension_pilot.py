#!/usr/bin/env python
"""Summarize matched MPS bond-dimension sensitivity runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


PPO_CONDITIONS = ["Base FinRL", "ANN signal", "QINN-MPS signal"]
PORTFOLIO_METRICS = [
    "annual_return",
    "sharpe",
    "max_drawdown",
    "annualized_turnover",
    "total_cost",
    "condition_elapsed_seconds",
]
SIGNAL_METRICS = [
    "mse",
    "mae",
    "directional_accuracy",
    "information_coefficient",
]
CONTROL_METRICS = [
    "seed",
    *(metric for metric in PORTFOLIO_METRICS if metric != "condition_elapsed_seconds"),
]
SELECTION_RELATIVE_TOLERANCE = 0.01
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".txt"}


def _load_run(
    run_dir: Path,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("runtime", {}).get("status") != "completed":
        raise ValueError(f"Run is not completed: {run_dir}")

    portfolio = pd.read_csv(run_dir / "ppo_backtest_metrics.csv")
    required_portfolio_columns = {"condition", "seed", *PORTFOLIO_METRICS}
    missing_portfolio_columns = required_portfolio_columns - set(portfolio.columns)
    if missing_portfolio_columns:
        raise ValueError(
            f"Missing portfolio columns in {run_dir}: "
            f"{sorted(missing_portfolio_columns)}"
        )
    portfolio = portfolio[portfolio["seed"] >= 0].copy()
    observed_conditions = sorted(portfolio["condition"].unique())
    if observed_conditions != sorted(PPO_CONDITIONS):
        raise ValueError(
            f"Unexpected PPO conditions in {run_dir}: {observed_conditions}"
        )

    expected_seeds = sorted(manifest["experiment"]["ppo_seeds"])
    for condition in PPO_CONDITIONS:
        observed_seeds = sorted(
            portfolio.loc[portfolio["condition"] == condition, "seed"].astype(int)
        )
        if observed_seeds != expected_seeds:
            raise ValueError(f"Unmatched seeds for {condition} in {run_dir}")

    signal = pd.read_csv(run_dir / "signal_metrics.csv")
    required_signal_columns = {
        "split",
        "model",
        "parameter_count",
        *SIGNAL_METRICS,
    }
    missing_signal_columns = required_signal_columns - set(signal.columns)
    if missing_signal_columns:
        raise ValueError(
            f"Missing signal columns in {run_dir}: "
            f"{sorted(missing_signal_columns)}"
        )
    expected_signal_rows = {
        ("validation_2018", "ANN"),
        ("validation_2018", "QINN-MPS"),
        ("test_2019_2023", "ANN"),
        ("test_2019_2023", "QINN-MPS"),
    }
    observed_signal_rows = set(zip(signal["split"], signal["model"], strict=True))
    if observed_signal_rows != expected_signal_rows or len(signal) != 4:
        raise ValueError(f"Unexpected signal metric rows in {run_dir}")
    return manifest, portfolio, signal


def _fixed_settings(experiment: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in experiment.items()
        if key != "mps_bond_dimension"
    }


def _validate_control_stability(
    controls: dict[str, pd.DataFrame],
    dimension: int,
    portfolio: pd.DataFrame,
) -> None:
    for condition in ("Base FinRL", "ANN signal"):
        current = (
            portfolio.loc[portfolio["condition"] == condition, CONTROL_METRICS]
            .sort_values("seed")
            .reset_index(drop=True)
        )
        if condition not in controls:
            controls[condition] = current
            continue
        try:
            pd.testing.assert_frame_equal(
                controls[condition],
                current,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )
        except AssertionError as error:
            raise ValueError(
                f"{condition} control drifted at bond dimension {dimension}"
            ) from error


def _select_dimension(summary: pd.DataFrame) -> pd.DataFrame:
    minimum_mse = float(summary["validation_mse"].min())
    threshold = minimum_mse * (1 + SELECTION_RELATIVE_TOLERANCE)
    summary = summary.copy()
    summary["within_validation_mse_1pct"] = (
        summary["validation_mse"] <= threshold
    )
    eligible = summary[summary["within_validation_mse_1pct"]].sort_values(
        ["mps_parameter_count", "mps_fit_seconds", "bond_dimension"]
    )
    selected_dimension = int(eligible.iloc[0]["bond_dimension"])
    summary["selected_primary"] = (
        summary["bond_dimension"] == selected_dimension
    )
    return summary.sort_values("bond_dimension").reset_index(drop=True)


def summarize(
    run_dirs: list[Path],
    expected_dimensions: set[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    seen_dimensions: set[int] = set()
    reference_settings: dict[str, object] | None = None
    controls: dict[str, pd.DataFrame] = {}

    for run_dir in run_dirs:
        manifest, portfolio, signal = _load_run(run_dir)
        experiment = manifest["experiment"]
        dimension = int(experiment["mps_bond_dimension"])
        if dimension in seen_dimensions:
            raise ValueError(f"Duplicate MPS bond dimension: {dimension}")
        seen_dimensions.add(dimension)

        settings = _fixed_settings(experiment)
        if reference_settings is None:
            reference_settings = settings
        elif settings != reference_settings:
            raise ValueError("Dimension runs differ on fixed experiment settings")
        _validate_control_stability(controls, dimension, portfolio)

        mps_signal = signal[signal["model"] == "QINN-MPS"].set_index("split")
        mps_portfolio = portfolio[
            portfolio["condition"] == "QINN-MPS signal"
        ]
        row: dict[str, object] = {
            "bond_dimension": dimension,
            "n_seeds": len(experiment["ppo_seeds"]),
            "ppo_timesteps": int(experiment["ppo_timesteps"]),
            "mps_parameter_count": int(manifest["mps_parameter_count"]),
            "mps_fit_seconds": float(manifest["encoder_runtime"]["mps_fit_seconds"]),
            "signal_inference_seconds": float(
                manifest["encoder_runtime"]["signal_inference_seconds"]
            ),
        }
        for split, prefix in (
            ("validation_2018", "validation"),
            ("test_2019_2023", "test"),
        ):
            for metric in SIGNAL_METRICS:
                row[f"{prefix}_{metric}"] = float(
                    mps_signal.loc[split, metric]
                )
        for metric in PORTFOLIO_METRICS:
            row[f"mps_{metric}_mean"] = float(mps_portfolio[metric].mean())
            row[f"mps_{metric}_std"] = float(
                mps_portfolio[metric].std(ddof=1)
            )

        pivot = portfolio.pivot(index="seed", columns="condition")
        for seed in experiment["ppo_seeds"]:
            paired_rows.append(
                {
                    "bond_dimension": dimension,
                    "seed": int(seed),
                    "mps_minus_ann_sharpe": float(
                        pivot.loc[seed, ("sharpe", "QINN-MPS signal")]
                        - pivot.loc[seed, ("sharpe", "ANN signal")]
                    ),
                    "mps_minus_ann_annual_return": float(
                        pivot.loc[seed, ("annual_return", "QINN-MPS signal")]
                        - pivot.loc[seed, ("annual_return", "ANN signal")]
                    ),
                }
            )
        paired = pd.DataFrame(
            row for row in paired_rows if row["bond_dimension"] == dimension
        )
        row["mps_minus_ann_sharpe_mean"] = float(
            paired["mps_minus_ann_sharpe"].mean()
        )
        row["mps_minus_ann_sharpe_positive_seeds"] = int(
            (paired["mps_minus_ann_sharpe"] > 0).sum()
        )
        row["mps_minus_ann_annual_return_mean"] = float(
            paired["mps_minus_ann_annual_return"].mean()
        )
        summary_rows.append(row)

    if not summary_rows:
        raise ValueError("At least one dimension run is required")
    if expected_dimensions is not None and seen_dimensions != expected_dimensions:
        raise ValueError(
            "Dimension set mismatch: "
            f"expected {sorted(expected_dimensions)}, "
            f"observed {sorted(seen_dimensions)}"
        )
    summary = _select_dimension(pd.DataFrame(summary_rows))
    paired = pd.DataFrame(paired_rows).sort_values(
        ["bond_dimension", "seed"]
    )
    return summary, paired.reset_index(drop=True)


def _sha256(path: Path) -> str:
    payload = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def build_artifact_manifest(
    run_dirs: list[Path],
    summary: pd.DataFrame,
    summary_output: Path,
    paired_output: Path,
) -> dict[str, object]:
    inputs = []
    for run_dir in sorted(run_dirs, key=lambda path: path.name):
        source_manifest = json.loads(
            (run_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        inputs.append(
            {
                "job_id": run_dir.name,
                "bond_dimension": int(
                    source_manifest["experiment"]["mps_bond_dimension"]
                ),
                "source_commit": source_manifest["runtime"]["git_commit"],
                "completed_at_utc": source_manifest["runtime"][
                    "completed_at_utc"
                ],
                "sha256": {
                    filename: _sha256(run_dir / filename)
                    for filename in (
                        "run_manifest.json",
                        "ppo_backtest_metrics.csv",
                        "signal_metrics.csv",
                    )
                },
            }
        )
    selected_dimension = int(
        summary.loc[summary["selected_primary"], "bond_dimension"].item()
    )
    return {
        "artifact": "mps_bond_dimension_pilot",
        "hash_policy": (
            "Text inputs and outputs are normalized to LF before SHA-256; "
            "binary files are hashed as stored."
        ),
        "expected_dimensions": [2, 4, 8],
        "selection": {
            "validation_mse_relative_tolerance": SELECTION_RELATIVE_TOLERANCE,
            "tie_breakers": [
                "mps_parameter_count",
                "mps_fit_seconds",
                "bond_dimension",
            ],
            "selected_bond_dimension": selected_dimension,
            "test_or_trading_metrics_used_for_selection": False,
        },
        "inputs": inputs,
        "outputs": {
            summary_output.name: _sha256(summary_output),
            paired_output.name: _sha256(paired_output),
        },
    }


def write_artifacts(
    run_dirs: list[Path],
    summary_output: Path,
    paired_output: Path,
    manifest_output: Path,
    *,
    expected_dimensions: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary, paired = summarize(
        run_dirs, expected_dimensions=expected_dimensions
    )
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    paired_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_output, index=False, lineterminator="\n")
    paired.to_csv(paired_output, index=False, lineterminator="\n")
    artifact_manifest = build_artifact_manifest(
        run_dirs, summary, summary_output, paired_output
    )
    manifest_output.write_bytes(
        (json.dumps(artifact_manifest, indent=2) + "\n").encode("utf-8")
    )
    return summary, paired


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--paired-output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    parser.add_argument(
        "--expected-dimensions",
        type=int,
        nargs="+",
        default=[2, 4, 8],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, paired = write_artifacts(
        args.run_dirs,
        args.summary_output,
        args.paired_output,
        args.manifest_output,
        expected_dimensions=set(args.expected_dimensions),
    )
    print(summary.to_string(index=False))
    print()
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
