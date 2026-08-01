#!/usr/bin/env python
"""Plot paired Sharpe effects over nested cumulative evaluation horizons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HORIZONS = [1, 2, 3, 4, 5]
PAIRED_COLUMN = "mps_minus_ann_sharpe"


def canonical_sha256(path: Path) -> str:
    payload = path.read_bytes()
    if path.suffix.lower() not in {".png", ".pdf"}:
        payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def validate_inputs(
    paired: pd.DataFrame, summary: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired_required = {"horizon_years", "seed", PAIRED_COLUMN}
    summary_required = {
        "horizon_years",
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
    selected_paired = paired[paired["horizon_years"].isin(HORIZONS)].copy()
    selected_summary = summary[
        (summary["horizon_years"].isin(HORIZONS))
        & (summary["metric"] == "sharpe")
    ].copy()
    if selected_paired.duplicated(["horizon_years", "seed"]).any():
        raise ValueError("Duplicate paired horizon/seed rows")
    for horizon in HORIZONS:
        seeds = sorted(
            selected_paired.loc[
                selected_paired["horizon_years"] == horizon, "seed"
            ].astype(int).tolist()
        )
        if seeds != list(range(10)):
            raise ValueError(f"Expected seeds 0-9 for horizon {horizon}")
    selected_summary = (
        selected_summary.set_index("horizon_years").loc[HORIZONS].reset_index()
    )
    if len(selected_summary) != len(HORIZONS):
        raise ValueError("Expected exactly one Sharpe summary per horizon")
    if selected_summary["n_matched_seeds"].astype(int).tolist() != [10] * 5:
        raise ValueError("Each horizon must contain ten matched seeds")
    return selected_paired, selected_summary


def plot_nested_horizons(
    paired: pd.DataFrame,
    summary: pd.DataFrame,
    png_output: Path,
    pdf_output: Path,
) -> None:
    paired, summary = validate_inputs(paired, summary)
    figure, axis = plt.subplots(figsize=(7.2, 3.9))
    colors = ["#0F766E", "#0D9488", "#2563EB", "#7C3AED", "#C2410C"]
    offsets = np.linspace(-0.12, 0.12, 10)
    for index, (horizon, color) in enumerate(
        zip(HORIZONS, colors, strict=True)
    ):
        values = (
            paired[paired["horizon_years"] == horizon]
            .sort_values("seed")[PAIRED_COLUMN]
            .to_numpy(dtype=float)
        )
        axis.scatter(
            horizon + offsets,
            values,
            color=color,
            alpha=0.7,
            edgecolor="white",
            linewidth=0.45,
            s=32,
            zorder=3,
        )
        row = summary.iloc[index]
        mean = float(row["mean_difference"])
        lower = float(row["paired_seed_bootstrap_95pct_lower"])
        upper = float(row["paired_seed_bootstrap_95pct_upper"])
        axis.errorbar(
            horizon,
            mean,
            yerr=np.array([[mean - lower], [upper - mean]]),
            fmt="D",
            color="#111827",
            markerfacecolor="white",
            markersize=6.5,
            capsize=4.5,
            linewidth=1.5,
            zorder=4,
        )
    axis.axhline(0, color="#4B5563", linewidth=1.0, linestyle="--", zorder=1)
    axis.plot(
        HORIZONS,
        summary["mean_difference"].to_numpy(dtype=float),
        color="#6B7280",
        linewidth=1.0,
        alpha=0.75,
        zorder=2,
    )
    axis.set_xticks(HORIZONS)
    axis.set_xlabel("Cumulative evaluation horizon from 2019 start (years)")
    axis.set_ylabel("MPS minus ANN annualized Sharpe")
    axis.set_title("Nested evaluation horizon (post-hoc exploratory)", loc="left")
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.65, alpha=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    figure.text(
        0.5,
        0.012,
        "Points: matched PPO seeds 0-9. Diamonds/bars: mean and paired-seed "
        "bootstrap interval. Prefixes are nested, not independent.",
        ha="center",
        va="bottom",
        fontsize=8.3,
        color="#374151",
    )
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    for output in (png_output, pdf_output):
        output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_output, dpi=300, bbox_inches="tight")
    figure.savefig(
        pdf_output,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def bind_figure_artifacts(
    manifest_path: Path,
    paired_path: Path,
    summary_path: Path,
    png_output: Path,
    pdf_output: Path,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for source in (paired_path, summary_path):
        expected = manifest.get("outputs", {}).get(source.name)
        if expected is None:
            raise ValueError(f"Evidence manifest does not register {source.name}")
        observed = canonical_sha256(source)
        if observed != expected:
            raise ValueError(
                f"Figure source hash mismatch for {source.name}: "
                f"expected {expected}, found {observed}"
            )
    manifest["figure_generator"] = {
        "script": "scripts/plot_nested_horizons.py",
        "script_sha256": canonical_sha256(Path(__file__)),
        "sources": {
            paired_path.name: canonical_sha256(paired_path),
            summary_path.name: canonical_sha256(summary_path),
        },
    }
    manifest["figures"] = {
        png_output.name: canonical_sha256(png_output),
        pdf_output.name: canonical_sha256(pdf_output),
    }
    manifest_path.write_bytes(
        (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--png-output", type=Path, required=True)
    parser.add_argument("--pdf-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_nested_horizons(
        pd.read_csv(args.paired),
        pd.read_csv(args.summary),
        args.png_output,
        args.pdf_output,
    )
    bind_figure_artifacts(
        args.manifest,
        args.paired,
        args.summary,
        args.png_output,
        args.pdf_output,
    )


if __name__ == "__main__":
    main()
