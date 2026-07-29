#!/usr/bin/env python
"""Generate the paper-ready MPS bond-dimension sensitivity figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {
    "bond_dimension",
    "mps_parameter_count",
    "mps_fit_seconds",
    "validation_mse",
    "within_validation_mse_1pct",
    "selected_primary",
}


def load_summary(path: Path) -> pd.DataFrame:
    summary = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(summary.columns)
    if missing:
        raise ValueError(f"Missing dimension figure columns: {sorted(missing)}")
    if len(summary) < 2:
        raise ValueError("Dimension figure requires at least two configurations")
    if summary["bond_dimension"].duplicated().any():
        raise ValueError("Bond dimensions must be unique")
    if int(summary["selected_primary"].sum()) != 1:
        raise ValueError("Exactly one primary dimension must be selected")
    return summary.sort_values("bond_dimension").reset_index(drop=True)


def plot_dimension_pilot(
    summary: pd.DataFrame,
    png_output: Path,
    pdf_output: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.85))
    dimensions = summary["bond_dimension"]
    selected = summary["selected_primary"].astype(bool)
    colors = ["#176B87" if value else "#9FB8C5" for value in selected]

    axes[0].plot(
        dimensions,
        summary["validation_mse"],
        color="#34495E",
        linewidth=1.4,
        zorder=1,
    )
    axes[0].scatter(
        dimensions,
        summary["validation_mse"],
        c=colors,
        edgecolor="#243746",
        linewidth=0.6,
        s=58,
        zorder=2,
    )
    axes[0].set_xlabel("MPS bond dimension")
    axes[0].set_ylabel("Validation MSE")
    axes[0].set_xticks(dimensions)
    axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)

    axes[1].bar(
        dimensions,
        summary["mps_parameter_count"],
        color=colors,
        edgecolor="#243746",
        linewidth=0.6,
        width=1.2,
    )
    axes[1].set_xlabel("MPS bond dimension")
    axes[1].set_ylabel("Trainable parameters")
    axes[1].set_xticks(dimensions)
    axes[1].set_yscale("log")
    axes[1].set_yticks([100, 1_000])
    axes[1].set_yticklabels(["100", "1,000"])

    selected_row = summary[selected].iloc[0]
    axes[0].annotate(
        "selected by frozen rule",
        (
            selected_row["bond_dimension"],
            selected_row["validation_mse"],
        ),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=7,
        color="#176B87",
    )
    figure.suptitle(
        "MPS capacity sensitivity and prespecified selection",
        fontsize=10,
        y=1.01,
    )
    figure.tight_layout()
    for output in (png_output, pdf_output):
        output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_output, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_output, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--png-output", required=True, type=Path)
    parser.add_argument("--pdf-output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_dimension_pilot(
        load_summary(args.summary),
        args.png_output,
        args.pdf_output,
    )


if __name__ == "__main__":
    main()
