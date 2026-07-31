from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).parent / "scripts" / "plot_final_effect.py"
SPEC = importlib.util.spec_from_file_location("final_figure", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
final_figure = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = final_figure
SPEC.loader.exec_module(final_figure)


def write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    paired_path = tmp_path / "paired.csv"
    inference_path = tmp_path / "inference.json"
    pd.DataFrame(
        {
            "seed": list(range(10)),
            "mps_minus_ann_sharpe": [
                -0.10,
                0.03,
                -0.02,
                0.06,
                -0.04,
                0.01,
                -0.08,
                0.04,
                -0.01,
                0.02,
            ],
        }
    ).to_csv(paired_path, index=False)
    inference_path.write_text(
        json.dumps(
            {
                "mean_difference": -0.009,
                "paired_seed_bootstrap_95pct_lower": -0.04,
                "paired_seed_bootstrap_95pct_upper": 0.02,
                "n_matched_seeds": 10,
            }
        ),
        encoding="utf-8",
    )
    return paired_path, inference_path


def test_load_inputs_rejects_seed_count_mismatch(tmp_path: Path) -> None:
    paired, inference = write_inputs(tmp_path)
    payload = json.loads(inference.read_text(encoding="utf-8"))
    payload["n_matched_seeds"] = 9
    inference.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="seed count"):
        final_figure.load_inputs(paired, inference)


def test_load_inputs_requires_paired_schema(tmp_path: Path) -> None:
    paired, inference = write_inputs(tmp_path)
    pd.read_csv(paired).drop(columns="mps_minus_ann_sharpe").to_csv(
        paired, index=False
    )
    with pytest.raises(ValueError, match="Missing paired effect columns"):
        final_figure.load_inputs(paired, inference)


def test_plot_writes_png_and_pdf(tmp_path: Path) -> None:
    paired_path, inference_path = write_inputs(tmp_path)
    paired, inference = final_figure.load_inputs(paired_path, inference_path)
    png = tmp_path / "effect.png"
    pdf = tmp_path / "effect.pdf"
    final_figure.plot_final_effect(paired, inference, png, pdf)
    assert png.read_bytes().startswith(b"\x89PNG")
    assert pdf.read_bytes().startswith(b"%PDF")
    assert b"/CreationDate" not in pdf.read_bytes()
    assert png.stat().st_size > 10_000
    assert pdf.stat().st_size > 1_000
