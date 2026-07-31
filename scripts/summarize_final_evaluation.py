#!/usr/bin/env python
"""Validate and summarize the prespecified ten-seed final evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


PPO_CONDITIONS = ["Base FinRL", "ANN signal", "QINN-MPS signal"]
EXPECTED_SEEDS = list(range(10))
EXPECTED_TIMESTEPS = 20_000
EXPECTED_BOND_DIMENSION = 2
SUMMARY_METRICS = [
    "annual_return",
    "sharpe",
    "max_drawdown",
    "annualized_turnover",
    "total_cost",
]
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_728


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_final_run(run_dir: Path) -> tuple[dict[str, object], pd.DataFrame]:
    manifest_path = run_dir / "run_manifest.json"
    metrics_path = run_dir / "ppo_backtest_metrics.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("runtime", {}).get("status") != "completed":
        raise ValueError("Final evaluation is not completed")

    experiment = manifest["experiment"]
    expected = {
        "ppo_timesteps": EXPECTED_TIMESTEPS,
        "mps_bond_dimension": EXPECTED_BOND_DIMENSION,
        "ppo_seeds": EXPECTED_SEEDS,
    }
    for field, value in expected.items():
        if experiment.get(field) != value:
            raise ValueError(
                f"Unexpected final {field}: {experiment.get(field)!r}"
            )

    metrics = pd.read_csv(metrics_path)
    required = {"condition", "seed", *SUMMARY_METRICS}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Missing final metric columns: {sorted(missing)}")
    metrics = metrics[metrics["seed"] >= 0].copy()
    if sorted(metrics["condition"].unique()) != sorted(PPO_CONDITIONS):
        raise ValueError("Unexpected final PPO conditions")
    for condition in PPO_CONDITIONS:
        seeds = sorted(
            metrics.loc[metrics["condition"] == condition, "seed"]
            .astype(int)
            .tolist()
        )
        if seeds != EXPECTED_SEEDS:
            raise ValueError(f"Unmatched final seeds for {condition}")
    if metrics.duplicated(["condition", "seed"]).any():
        raise ValueError("Duplicate final condition/seed rows")
    return manifest, metrics


def paired_seed_bootstrap(
    differences: pd.Series,
) -> tuple[float, float]:
    values = differences.to_numpy(dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def exact_two_sided_sign_p(differences: pd.Series) -> float:
    nonzero = differences[differences != 0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    positive = int((nonzero > 0).sum())
    tail = min(positive, n - positive)
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return min(1.0, 2 * probability)


def summarize(
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    condition_rows: list[dict[str, object]] = []
    for condition in PPO_CONDITIONS:
        selected = metrics[metrics["condition"] == condition]
        row: dict[str, object] = {
            "condition": condition,
            "n_seeds": len(selected),
        }
        for metric in SUMMARY_METRICS:
            row[f"{metric}_mean"] = float(selected[metric].mean())
            row[f"{metric}_std"] = float(selected[metric].std(ddof=1))
            row[f"{metric}_median"] = float(selected[metric].median())
        condition_rows.append(row)

    pivot = metrics.pivot(index="seed", columns="condition")
    paired = pd.DataFrame({"seed": EXPECTED_SEEDS})
    for metric in SUMMARY_METRICS:
        paired[f"mps_minus_ann_{metric}"] = (
            pivot[metric]["QINN-MPS signal"] - pivot[metric]["ANN signal"]
        ).to_numpy()

    primary = paired["mps_minus_ann_sharpe"]
    lower, upper = paired_seed_bootstrap(primary)
    inference = {
        "primary_estimand": "mean paired MPS-minus-ANN Sharpe difference",
        "n_matched_seeds": len(primary),
        "mean_difference": float(primary.mean()),
        "standard_deviation": float(primary.std(ddof=1)),
        "median_difference": float(primary.median()),
        "positive_seeds": int((primary > 0).sum()),
        "negative_seeds": int((primary < 0).sum()),
        "zero_seeds": int((primary == 0).sum()),
        "paired_seed_bootstrap_95pct_lower": lower,
        "paired_seed_bootstrap_95pct_upper": upper,
        "paired_seed_bootstrap_samples": BOOTSTRAP_SAMPLES,
        "paired_seed_bootstrap_seed": BOOTSTRAP_SEED,
        "exact_two_sided_sign_test_p": exact_two_sided_sign_p(primary),
        "interpretation_boundary": (
            "Descriptive uncertainty across ten matched training seeds on one "
            "fixed historical split; not a population-level causal interval."
        ),
    }
    return pd.DataFrame(condition_rows), paired, inference


def write_artifacts(
    run_dir: Path,
    condition_output: Path,
    paired_output: Path,
    inference_output: Path,
    manifest_output: Path,
    artifact_name: str = "final_ten_seed_evaluation",
) -> None:
    manifest, metrics = load_final_run(run_dir)
    condition, paired, inference = summarize(metrics)
    for path in (condition_output, paired_output, inference_output, manifest_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    condition.to_csv(condition_output, index=False, lineterminator="\n")
    paired.to_csv(paired_output, index=False, lineterminator="\n")
    inference_output.write_bytes(
        (json.dumps(inference, indent=2) + "\n").encode("utf-8")
    )
    provenance = {
        "artifact": artifact_name,
        "source_commit": manifest["runtime"].get("git_commit"),
        "completed_at_utc": manifest["runtime"].get("completed_at_utc"),
        "prespecified_configuration": {
            "ppo_timesteps": EXPECTED_TIMESTEPS,
            "mps_bond_dimension": EXPECTED_BOND_DIMENSION,
            "ppo_seeds": EXPECTED_SEEDS,
        },
        "inputs": {
            "run_manifest.json": sha256(run_dir / "run_manifest.json"),
            "ppo_backtest_metrics.csv": sha256(
                run_dir / "ppo_backtest_metrics.csv"
            ),
            "equity_curves.csv": sha256(run_dir / "equity_curves.csv"),
        },
        "outputs": {
            condition_output.name: sha256(condition_output),
            paired_output.name: sha256(paired_output),
            inference_output.name: sha256(inference_output),
        },
    }
    manifest_output.write_bytes(
        (json.dumps(provenance, indent=2) + "\n").encode("utf-8")
    )
    print(condition.to_string(index=False))
    print()
    print(paired.to_string(index=False))
    print()
    print(json.dumps(inference, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--condition-output", required=True, type=Path)
    parser.add_argument("--paired-output", required=True, type=Path)
    parser.add_argument("--inference-output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    parser.add_argument(
        "--artifact-name",
        default="final_ten_seed_evaluation",
        help="Stable provenance label for the summarized evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_artifacts(
        args.run_dir,
        args.condition_output,
        args.paired_output,
        args.inference_output,
        args.manifest_output,
        args.artifact_name,
    )


if __name__ == "__main__":
    main()
