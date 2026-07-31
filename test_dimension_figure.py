from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).parent / "scripts" / "plot_dimension_pilot.py"
SPEC = importlib.util.spec_from_file_location("dimension_figure", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
dimension_figure = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dimension_figure
SPEC.loader.exec_module(dimension_figure)


def write_summary(path: Path) -> Path:
    pd.DataFrame(
        {
            "bond_dimension": [2, 4, 8],
            "mps_parameter_count": [97, 369, 1441],
            "mps_fit_seconds": [48.7, 4.7, 9.3],
            "validation_mse": [1.280, 1.281, 1.275],
            "within_validation_mse_1pct": [True, True, True],
            "selected_primary": [True, False, False],
        }
    ).to_csv(path, index=False)
    return path


def test_load_summary_requires_figure_schema(tmp_path: Path) -> None:
    path = write_summary(tmp_path / "summary.csv")
    pd.read_csv(path).drop(columns="validation_mse").to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing dimension figure columns"):
        dimension_figure.load_summary(path)


def test_load_summary_requires_one_selection(tmp_path: Path) -> None:
    path = write_summary(tmp_path / "summary.csv")
    summary = pd.read_csv(path)
    summary["selected_primary"] = False
    summary.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Exactly one"):
        dimension_figure.load_summary(path)


def test_plot_writes_png_and_pdf(tmp_path: Path) -> None:
    summary = dimension_figure.load_summary(
        write_summary(tmp_path / "summary.csv")
    )
    png = tmp_path / "figure.png"
    pdf = tmp_path / "figure.pdf"
    dimension_figure.plot_dimension_pilot(summary, png, pdf)
    assert png.read_bytes().startswith(b"\x89PNG")
    assert pdf.read_bytes().startswith(b"%PDF")
    assert png.stat().st_size > 10_000
    assert pdf.stat().st_size > 1_000
    assert b"/CreationDate" not in pdf.read_bytes()
