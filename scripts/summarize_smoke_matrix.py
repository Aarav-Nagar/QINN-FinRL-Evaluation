#!/usr/bin/env python
"""Summarize configuration smoke runs without treating them as paper evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_CONDITIONS = {"Base FinRL", "ANN signal", "QINN-MPS signal"}


def load_smoke_run(run_dir: Path) -> dict[str, object]:
    manifest_path = run_dir / "run_manifest.json"
    metrics_path = run_dir / "ppo_backtest_metrics.csv"
    if not manifest_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(f"Incomplete smoke run: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = pd.read_csv(metrics_path)
    observed = set(metrics.loc[metrics["seed"] >= 0, "condition"])
    if observed != REQUIRED_CONDITIONS:
        raise ValueError(
            f"{run_dir} conditions {sorted(observed)} do not match "
            f"{sorted(REQUIRED_CONDITIONS)}"
        )
    experiment = manifest["experiment"]
    runtime = manifest["runtime"]
    return {
        "run_dir": str(run_dir),
        "bond_dimension": int(experiment["mps_bond_dimension"]),
        "mps_parameter_count": int(manifest["mps_parameter_count"]),
        "ann_parameter_count": int(manifest["ann_parameter_count"]),
        "ppo_timesteps": int(experiment["ppo_timesteps"]),
        "seeds": ",".join(str(seed) for seed in experiment["ppo_seeds"]),
        "encoder_epochs": int(experiment["encoder_epochs"]),
        "encoder_device": runtime["encoder_device_resolved"],
        "elapsed_seconds": float(runtime["elapsed_seconds"]),
        "base_sharpe": float(
            metrics.loc[metrics["condition"] == "Base FinRL", "sharpe"].iloc[0]
        ),
        "ann_sharpe": float(
            metrics.loc[metrics["condition"] == "ANN signal", "sharpe"].iloc[0]
        ),
        "mps_sharpe": float(
            metrics.loc[metrics["condition"] == "QINN-MPS signal", "sharpe"].iloc[0]
        ),
        "paper_evidence": False,
    }


def summarize(run_dirs: list[Path]) -> pd.DataFrame:
    rows = [load_smoke_run(path) for path in run_dirs]
    frame = pd.DataFrame(rows).sort_values("bond_dimension").reset_index(drop=True)
    if frame["bond_dimension"].duplicated().any():
        raise ValueError("Each smoke run must use a distinct bond dimension")
    comparable = ["ppo_timesteps", "seeds", "encoder_epochs", "encoder_device"]
    for column in comparable:
        if frame[column].nunique(dropna=False) != 1:
            raise ValueError(f"Smoke runs differ on {column}")
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = summarize(args.run_dirs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
