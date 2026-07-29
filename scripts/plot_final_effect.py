#!/usr/bin/env python
"""Plot the final paired seed-level MPS-minus-ANN Sharpe effect."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PAIRED_COLUMNS = {"seed", "mps_minus_ann_sharpe"}
INFERENCE_FIELDS = {
    "mean_difference",
    "paired_seed_bootstrap_95pct_lower",
    "paired_seed_bootstrap_95pct_upper",
    "n_matched_seeds",
}


def load_inputs(
    paired_path: Path,
    inference_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    paired = pd.read_csv(paired_path)
    missing_columns = PAIRED_COLUMNS - set(paired.columns)
    if missing_columns:
        raise ValueError(f"Missing paired effect columns: {sorted(missing_columns)}")
    if paired["seed"].duplicated().any():
        raise ValueError("Paired effect seeds must be unique")
    paired = paired.sort_values("seed").reset_index(drop=True)

    inference = json.loads(inference_path.read_text(encoding="utf-8"))
    missing_fields = INFERENCE_FIELDS - set(inference)
    if missing_fields:
        raise ValueError(
            f"Missing final inference fields: {sorted(missing_fields)}"
        )
    if int(inference["n_matched_seeds"]) != len(paired):
        raise ValueError("Inference seed count does not match paired rows")
    return paired, inference


def plot_final_effect(
    paired: pd.DataFrame,
    inference: dict[str, object],
    png_output: Path,
    pdf_output: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(3.45, 2.75))
    differences = paired["mps_minus_ann_sharpe"]
    colors = ["#176B87" if value > 0 else "#B14A4A" for value in differences]
    axis.axhline(0, color="#4A4A4A", linewidth=0.8, linestyle="--")
    axis.scatter(
        paired["seed"],
        differences,
        c=colors,
        edgecolor="#243746",
        linewidth=0.5,
        s=36,
        zorder=2,
    )
    lower = float(inference["paired_seed_bootstrap_95pct_lower"])
    upper = float(inference["paired_seed_bootstrap_95pct_upper"])
    mean = float(inference["mean_difference"])
    axis.axhspan(lower, upper, color="#9FB8C5", alpha=0.28, linewidth=0)
    axis.axhline(mean, color="#176B87", linewidth=1.5, label="paired mean")
    axis.set_xlabel("Matched PPO seed")
    axis.set_ylabel("Sharpe difference (MPS - ANN)")
    axis.set_xticks(paired["seed"])
    axis.legend(loc="best", frameon=False, fontsize=7)
    figure.tight_layout()
    for output in (png_output, pdf_output):
        output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_output, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_output, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paired", type=Path)
    parser.add_argument("inference", type=Path)
    parser.add_argument("--png-output", required=True, type=Path)
    parser.add_argument("--pdf-output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paired, inference = load_inputs(args.paired, args.inference)
    plot_final_effect(paired, inference, args.png_output, args.pdf_output)


if __name__ == "__main__":
    main()
