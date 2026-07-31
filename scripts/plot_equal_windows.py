#!/usr/bin/env python
"""Plot seed-level Sharpe effects across the fixed equal-length windows."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WINDOWS = ["2017-2018", "2019-2020", "2021-2022"]
PAIRED_COLUMN = "mps_minus_ann_sharpe"
COLORS = ["#0F766E", "#1D4ED8", "#C2410C"]


def validate_inputs(
    paired: pd.DataFrame, summary: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired_required = {"window", "seed", PAIRED_COLUMN}
    summary_required = {
        "window",
        "metric",
        "n_matched_seeds",
        "mean_difference",
        "paired_seed_bootstrap_95pct_lower",
        "paired_seed_bootstrap_95pct_upper",
    }
    if missing := paired_required - set(paired.columns):
        raise ValueError(f"Missing paired columns: {sorted(missing)}")
    if missing := summary_required - set(summary.columns):
        raise ValueError(f"Missing summary columns: {sorted(missing)}")
    paired = paired[paired["window"].isin(WINDOWS)].copy()
    summary = summary[
        (summary["window"].isin(WINDOWS)) & (summary["metric"] == "sharpe")
    ].copy()
    if paired.duplicated(["window", "seed"]).any():
        raise ValueError("Duplicate paired window/seed rows")
    for window in WINDOWS:
        seeds = sorted(
            paired.loc[paired["window"] == window, "seed"].astype(int).tolist()
        )
        if seeds != list(range(10)):
            raise ValueError(f"Expected seeds 0-9 for {window}")
    if summary["window"].tolist() != WINDOWS:
        summary = summary.set_index("window").loc[WINDOWS].reset_index()
    if summary["window"].tolist() != WINDOWS or len(summary) != len(WINDOWS):
        raise ValueError("Expected exactly one Sharpe summary per fixed window")
    if summary["n_matched_seeds"].astype(int).tolist() != [10, 10, 10]:
        raise ValueError("Sharpe summaries must each contain ten matched seeds")
    return paired, summary


def plot_equal_windows(
    paired: pd.DataFrame,
    summary: pd.DataFrame,
    png_output: Path,
    pdf_output: Path,
) -> None:
    paired, summary = validate_inputs(paired, summary)
    figure, axis = plt.subplots(figsize=(7.2, 3.9))
    positions = np.arange(len(WINDOWS), dtype=float)
    offsets = np.linspace(-0.14, 0.14, 10)
    for index, (window, color) in enumerate(zip(WINDOWS, COLORS, strict=True)):
        values = (
            paired[paired["window"] == window]
            .sort_values("seed")[PAIRED_COLUMN]
            .to_numpy(dtype=float)
        )
        axis.scatter(
            positions[index] + offsets,
            values,
            color=color,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.45,
            s=34,
            zorder=3,
        )
        row = summary.iloc[index]
        mean = float(row["mean_difference"])
        lower = float(row["paired_seed_bootstrap_95pct_lower"])
        upper = float(row["paired_seed_bootstrap_95pct_upper"])
        axis.errorbar(
            positions[index],
            mean,
            yerr=np.array([[mean - lower], [upper - mean]]),
            fmt="D",
            color="#111827",
            markerfacecolor="white",
            markersize=6.5,
            capsize=5,
            linewidth=1.6,
            zorder=4,
        )
    axis.axhline(0, color="#4B5563", linewidth=1.0, linestyle="--", zorder=1)
    axis.set_xticks(positions, WINDOWS)
    axis.set_ylabel("MPS minus ANN annualized Sharpe")
    axis.set_xlabel("Two-year portfolio evaluation window")
    axis.set_title("Equal-length temporal robustness (exploratory)", loc="left")
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.65, alpha=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    figure.text(
        0.5,
        0.015,
        "Points: matched PPO seeds 0-9. Diamonds and bars: mean and descriptive "
        "seed-bootstrap 95% interval.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#374151",
    )
    figure.tight_layout(rect=(0, 0.075, 1, 1))
    for output in (png_output, pdf_output):
        output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_output, dpi=300, bbox_inches="tight")
    figure.savefig(
        pdf_output,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--png-output", type=Path, required=True)
    parser.add_argument("--pdf-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_equal_windows(
        pd.read_csv(args.paired),
        pd.read_csv(args.summary),
        args.png_output,
        args.pdf_output,
    )


if __name__ == "__main__":
    main()
