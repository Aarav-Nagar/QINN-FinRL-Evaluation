#!/usr/bin/env python
"""Summarize matched PPO budget endpoints and paired ANN/MPS differences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PPO_CONDITIONS = ["Base FinRL", "ANN signal", "QINN-MPS signal"]
SUMMARY_METRICS = [
    "annual_return",
    "sharpe",
    "max_drawdown",
    "annualized_turnover",
    "total_cost",
    "condition_elapsed_seconds",
    "train_explained_variance",
]


def load_budget_run(run_dir: Path) -> tuple[dict[str, object], pd.DataFrame]:
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    metrics = pd.read_csv(run_dir / "ppo_backtest_metrics.csv")
    metrics = metrics[metrics["seed"] >= 0].copy()
    observed = sorted(metrics["condition"].unique())
    if observed != sorted(PPO_CONDITIONS):
        raise ValueError(f"Unexpected PPO conditions in {run_dir}: {observed}")
    expected_seeds = sorted(manifest["experiment"]["ppo_seeds"])
    for condition in PPO_CONDITIONS:
        condition_seeds = sorted(
            metrics.loc[metrics["condition"] == condition, "seed"].astype(int)
        )
        if condition_seeds != expected_seeds:
            raise ValueError(f"Unmatched seeds for {condition} in {run_dir}")
    return manifest, metrics


def summarize(
    run_dirs: list[Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    seen_budgets: set[int] = set()
    reference_settings: dict[str, object] | None = None
    for run_dir in run_dirs:
        manifest, metrics = load_budget_run(run_dir)
        experiment = manifest["experiment"]
        budget = int(experiment["ppo_timesteps"])
        if budget in seen_budgets:
            raise ValueError(f"Duplicate PPO budget: {budget}")
        seen_budgets.add(budget)
        settings = {
            "ppo_seeds": experiment["ppo_seeds"],
            "mps_bond_dimension": experiment["mps_bond_dimension"],
            "encoder_epochs": experiment["encoder_epochs"],
            "encoder_device": experiment["encoder_device"],
        }
        if reference_settings is None:
            reference_settings = settings
        elif settings != reference_settings:
            raise ValueError("Budget runs differ on non-budget settings")

        for condition in PPO_CONDITIONS:
            selected = metrics[metrics["condition"] == condition]
            row: dict[str, object] = {
                "ppo_timesteps": budget,
                "condition": condition,
                "n_seeds": len(selected),
            }
            for metric in SUMMARY_METRICS:
                row[f"{metric}_mean"] = float(selected[metric].mean())
                row[f"{metric}_std"] = float(selected[metric].std(ddof=1))
            summary_rows.append(row)

        pivot = metrics.pivot(index="seed", columns="condition")
        paired: dict[str, object] = {
            "ppo_timesteps": budget,
            "n_seeds": len(experiment["ppo_seeds"]),
        }
        for metric in ("annual_return", "sharpe"):
            difference = (
                pivot[metric]["QINN-MPS signal"] - pivot[metric]["ANN signal"]
            )
            paired[f"mps_minus_ann_{metric}_mean"] = float(difference.mean())
            paired[f"mps_minus_ann_{metric}_std"] = float(difference.std(ddof=1))
            paired[f"mps_minus_ann_{metric}_positive_seeds"] = int(
                (difference > 0).sum()
            )
        paired_rows.append(paired)

    return (
        pd.DataFrame(summary_rows).sort_values(["ppo_timesteps", "condition"]),
        pd.DataFrame(paired_rows).sort_values("ppo_timesteps"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--paired-output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, paired = summarize(args.run_dirs)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.paired_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    paired.to_csv(args.paired_output, index=False)
    print(summary.to_string(index=False))
    print()
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
