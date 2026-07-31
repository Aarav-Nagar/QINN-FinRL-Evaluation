from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).parent / "scripts" / "plot_equal_windows.py"
SPEC = importlib.util.spec_from_file_location("equal_window_figure", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
figure = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = figure
SPEC.loader.exec_module(figure)


def frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    paired_rows = []
    summary_rows = []
    for window_index, window in enumerate(figure.WINDOWS):
        values = [-0.1 + window_index * 0.1 + seed / 100 for seed in range(10)]
        for seed, value in enumerate(values):
            paired_rows.append(
                {
                    "window": window,
                    "seed": seed,
                    "mps_minus_ann_sharpe": value,
                }
            )
        summary_rows.append(
            {
                "window": window,
                "metric": "sharpe",
                "n_matched_seeds": 10,
                "mean_difference": sum(values) / len(values),
                "paired_seed_bootstrap_95pct_lower": min(values),
                "paired_seed_bootstrap_95pct_upper": max(values),
            }
        )
    return pd.DataFrame(paired_rows), pd.DataFrame(summary_rows)


def test_plot_writes_png_and_vector_pdf(tmp_path: Path) -> None:
    paired, summary = frames()
    png = tmp_path / "figure.png"
    pdf = tmp_path / "figure.pdf"
    figure.plot_equal_windows(paired, summary, png, pdf)
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf.read_bytes().startswith(b"%PDF")
    assert png.stat().st_size > 10_000
    assert pdf.stat().st_size > 1_000


def test_plot_rejects_missing_seed(tmp_path: Path) -> None:
    paired, summary = frames()
    paired = paired[
        ~((paired["window"] == "2019-2020") & (paired["seed"] == 9))
    ]
    with pytest.raises(ValueError, match="Expected seeds 0-9"):
        figure.plot_equal_windows(
            paired,
            summary,
            tmp_path / "figure.png",
            tmp_path / "figure.pdf",
        )


def test_plot_rejects_duplicate_seed(tmp_path: Path) -> None:
    paired, summary = frames()
    paired = pd.concat([paired, paired.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        figure.plot_equal_windows(
            paired,
            summary,
            tmp_path / "figure.png",
            tmp_path / "figure.pdf",
        )
