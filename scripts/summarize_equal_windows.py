#!/usr/bin/env python
"""Validate and summarize the prespecified equal-length temporal windows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CONDITIONS = ["Base FinRL", "ANN signal", "QINN-MPS signal"]
SEEDS = list(range(10))
PORTFOLIO_METRICS = [
    "sharpe",
    "annual_return",
    "max_drawdown",
    "annualized_turnover",
    "total_cost",
]
SIGNAL_METRICS = [
    "mse",
    "mae",
    "directional_accuracy",
    "information_coefficient",
]
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_731
COMMON_CONFIGURATION: dict[str, object] = {
    "ppo_seeds": SEEDS,
    "ppo_timesteps": 20_000,
    "mps_bond_dimension": 2,
    "representation_seed": 2026,
    "encoder_epochs": 60,
    "encoder_patience": 10,
    "encoder_batch_size": 512,
    "encoder_device": "cpu",
    "transaction_cost": 0.001,
    "initial_amount": 1_000_000,
}
WINDOWS: dict[str, dict[str, str]] = {
    "2017-2018": {
        "window_name": "shifted",
        "representation_train_end": "2015-12-31",
        "representation_validation_start": "2016-01-01",
        "representation_validation_end": "2016-12-30",
        "train_start": "2013-01-02",
        "train_end": "2016-12-30",
        "test_start": "2017-01-03",
        "test_end": "2018-12-28",
    },
    "2019-2020": {
        "window_name": "equal_2019_2020",
        "representation_train_end": "2017-12-29",
        "representation_validation_start": "2018-01-01",
        "representation_validation_end": "2018-12-28",
        "train_start": "2015-01-02",
        "train_end": "2018-12-28",
        "test_start": "2019-01-02",
        "test_end": "2020-12-31",
    },
    "2021-2022": {
        "window_name": "equal_2021_2022",
        "representation_train_end": "2019-12-31",
        "representation_validation_start": "2020-01-02",
        "representation_validation_end": "2020-12-31",
        "train_start": "2017-01-03",
        "train_end": "2020-12-31",
        "test_start": "2021-01-04",
        "test_end": "2022-12-30",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_test_split(label: str) -> str:
    return f"test_{label.replace('-', '_')}"


def _require_equal(actual: object, expected: object, context: str) -> None:
    if actual != expected:
        raise ValueError(f"{context}: expected {expected!r}, found {actual!r}")


def load_window(
    label: str, run_dir: Path
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    if label not in WINDOWS:
        raise ValueError(f"Unknown equal-window label: {label}")
    manifest_path = run_dir / "run_manifest.json"
    metrics_path = run_dir / "ppo_backtest_metrics.csv"
    signal_path = run_dir / "signal_metrics.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require_equal(
        manifest.get("runtime", {}).get("status"),
        "completed",
        f"{label} runtime status",
    )
    experiment = manifest["experiment"]
    for field, expected in (COMMON_CONFIGURATION | WINDOWS[label]).items():
        _require_equal(experiment.get(field), expected, f"{label} {field}")

    metrics = pd.read_csv(metrics_path)
    required = {"condition", "seed", *PORTFOLIO_METRICS}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"{label} missing portfolio columns: {sorted(missing)}")
    metrics = metrics[metrics["seed"] >= 0].copy()
    if metrics.duplicated(["condition", "seed"]).any():
        raise ValueError(f"{label} has duplicate condition/seed rows")
    _require_equal(
        sorted(metrics["condition"].unique()),
        sorted(CONDITIONS),
        f"{label} conditions",
    )
    for condition in CONDITIONS:
        observed = sorted(
            metrics.loc[metrics["condition"] == condition, "seed"]
            .astype(int)
            .tolist()
        )
        _require_equal(observed, SEEDS, f"{label} seeds for {condition}")

    signal = pd.read_csv(signal_path)
    required_signal = {"split", "model", *SIGNAL_METRICS}
    missing_signal = required_signal - set(signal.columns)
    if missing_signal:
        raise ValueError(f"{label} missing signal columns: {sorted(missing_signal)}")
    signal = signal[signal["split"] == expected_test_split(label)].copy()
    _require_equal(
        sorted(signal["model"].tolist()),
        ["ANN", "QINN-MPS"],
        f"{label} evaluation signal models",
    )
    return manifest, metrics, signal


def paired_bootstrap(
    values: pd.Series, *, seed_offset: int
) -> tuple[float, float]:
    array = values.to_numpy(dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_SAMPLES, len(array)))
    means = array[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def summarize(
    loaded: dict[str, tuple[dict[str, object], pd.DataFrame, pd.DataFrame]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    condition_rows: list[dict[str, object]] = []
    paired_frames: list[pd.DataFrame] = []
    paired_summary_rows: list[dict[str, object]] = []
    signal_rows: list[dict[str, object]] = []

    for window_index, label in enumerate(WINDOWS):
        _, metrics, signal = loaded[label]
        for condition in CONDITIONS:
            selected = metrics[metrics["condition"] == condition]
            row: dict[str, object] = {
                "window": label,
                "condition": condition,
                "n_seeds": len(selected),
            }
            for metric in PORTFOLIO_METRICS:
                row[f"{metric}_mean"] = float(selected[metric].mean())
                row[f"{metric}_std"] = float(selected[metric].std(ddof=1))
            condition_rows.append(row)

        pivot = metrics.pivot(index="seed", columns="condition")
        paired = pd.DataFrame({"window": label, "seed": SEEDS})
        for metric_index, metric in enumerate(PORTFOLIO_METRICS):
            column = f"mps_minus_ann_{metric}"
            paired[column] = (
                pivot[metric]["QINN-MPS signal"] - pivot[metric]["ANN signal"]
            ).to_numpy()
            differences = paired[column]
            lower, upper = paired_bootstrap(
                differences,
                seed_offset=window_index * len(PORTFOLIO_METRICS) + metric_index,
            )
            paired_summary_rows.append(
                {
                    "window": label,
                    "metric": metric,
                    "n_matched_seeds": len(differences),
                    "mean_difference": float(differences.mean()),
                    "standard_deviation": float(differences.std(ddof=1)),
                    "median_difference": float(differences.median()),
                    "positive_seeds": int((differences > 0).sum()),
                    "negative_seeds": int((differences < 0).sum()),
                    "zero_seeds": int((differences == 0).sum()),
                    "paired_seed_bootstrap_95pct_lower": lower,
                    "paired_seed_bootstrap_95pct_upper": upper,
                }
            )
        paired_frames.append(paired)

        signal_by_model = signal.set_index("model")
        for metric in SIGNAL_METRICS:
            ann = float(signal_by_model.loc["ANN", metric])
            mps = float(signal_by_model.loc["QINN-MPS", metric])
            signal_rows.append(
                {
                    "window": label,
                    "metric": metric,
                    "ann": ann,
                    "mps": mps,
                    "mps_minus_ann": mps - ann,
                }
            )

    paired_summary = pd.DataFrame(paired_summary_rows)
    sharpe = paired_summary[paired_summary["metric"] == "sharpe"].copy()
    signs = np.sign(sharpe["mean_difference"].to_numpy(dtype=float))
    nonzero_signs = set(signs[signs != 0].tolist())
    if nonzero_signs == {1.0}:
        sign_pattern = "positive_in_all_windows"
    elif nonzero_signs == {-1.0}:
        sign_pattern = "negative_in_all_windows"
    elif len(nonzero_signs) > 1:
        sign_pattern = "sign_heterogeneity"
    else:
        sign_pattern = "all_zero_or_partly_zero"
    inference = {
        "analysis_role": "exploratory equal-length temporal robustness",
        "windows": list(WINDOWS),
        "primary_panel_metric": "paired MPS-minus-ANN annualized Sharpe",
        "mean_sharpe_sign_pattern": sign_pattern,
        "windows_with_positive_mean_sharpe_difference": int(
            (sharpe["mean_difference"] > 0).sum()
        ),
        "windows_with_negative_mean_sharpe_difference": int(
            (sharpe["mean_difference"] < 0).sum()
        ),
        "bootstrap_samples_per_window_metric": BOOTSTRAP_SAMPLES,
        "bootstrap_base_seed": BOOTSTRAP_SEED,
        "metric_orientation": {
            "sharpe": "higher is better",
            "annual_return": "higher is better",
            "max_drawdown": "higher means a less severe drawdown",
            "annualized_turnover": "lower means less trading",
            "total_cost": "lower means lower modeled transaction cost",
            "mse": "lower is better",
            "mae": "lower is better",
            "directional_accuracy": "higher is better",
            "information_coefficient": "higher is better",
        },
        "interpretation_boundary": (
            "All windows are reported. The panel is exploratory, is not pooled "
            "into a universal effect, and cannot identify a market-regime cause."
        ),
    }
    return (
        pd.DataFrame(condition_rows),
        pd.concat(paired_frames, ignore_index=True),
        paired_summary,
        pd.DataFrame(signal_rows),
        inference,
    )


def write_outputs(
    run_dirs: dict[str, Path], output_dir: Path, protocol: Path
) -> None:
    loaded = {label: load_window(label, run_dirs[label]) for label in WINDOWS}
    condition, paired, paired_summary, signal, inference = summarize(loaded)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "window_condition_summary.csv": condition,
        "window_paired_seed_effects.csv": paired,
        "window_paired_metric_summary.csv": paired_summary,
        "window_signal_quality.csv": signal,
    }
    for filename, frame in output_paths.items():
        frame.to_csv(output_dir / filename, index=False, lineterminator="\n")
    inference_path = output_dir / "equal_window_inference.json"
    inference_path.write_bytes((json.dumps(inference, indent=2) + "\n").encode())

    inputs: dict[str, str] = {"protocol": sha256(protocol)}
    for label, run_dir in run_dirs.items():
        for filename in (
            "run_manifest.json",
            "ppo_backtest_metrics.csv",
            "signal_metrics.csv",
        ):
            inputs[f"{label}/{filename}"] = sha256(run_dir / filename)
    manifest = {
        "artifact": "equal_length_temporal_robustness",
        "protocol": str(protocol.as_posix()),
        "windows": WINDOWS,
        "common_configuration": COMMON_CONFIGURATION,
        "inputs": inputs,
        "outputs": {
            filename: sha256(output_dir / filename) for filename in output_paths
        }
        | {inference_path.name: sha256(inference_path)},
        "reporting_rule": (
            "Report every fixed window, including unfavorable results; "
            "interpret patterns as exploratory."
        ),
    }
    (output_dir / "equal_window_manifest.json").write_bytes(
        (json.dumps(manifest, indent=2) + "\n").encode()
    )
    print(paired_summary.to_string(index=False))
    print()
    print(signal.to_string(index=False))
    print()
    print(json.dumps(inference, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-2017-2018", type=Path, required=True)
    parser.add_argument("--window-2019-2020", type=Path, required=True)
    parser.add_argument("--window-2021-2022", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("docs/EQUAL_WINDOW_PROTOCOL.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_outputs(
        {
            "2017-2018": args.window_2017_2018,
            "2019-2020": args.window_2019_2020,
            "2021-2022": args.window_2021_2022,
        },
        args.output_dir,
        args.protocol,
    )


if __name__ == "__main__":
    main()
