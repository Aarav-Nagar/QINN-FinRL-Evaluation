import importlib.util
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parent
SCRIPT_PATH = ROOT / "scripts" / "plot_nested_horizons.py"
SPEC = importlib.util.spec_from_file_location("plot_nested_horizons", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    paired_rows = []
    summary_rows = []
    for horizon in range(1, 6):
        values = []
        for seed in range(10):
            value = (seed - 4.5) / 100 + horizon / 100
            values.append(value)
            paired_rows.append(
                {
                    "horizon_years": horizon,
                    "seed": seed,
                    "mps_minus_ann_sharpe": value,
                }
            )
        mean = sum(values) / len(values)
        summary_rows.append(
            {
                "horizon_years": horizon,
                "metric": "sharpe",
                "n_matched_seeds": 10,
                "mean_difference": mean,
                "paired_seed_bootstrap_95pct_lower": mean - 0.02,
                "paired_seed_bootstrap_95pct_upper": mean + 0.02,
            }
        )
    return pd.DataFrame(paired_rows), pd.DataFrame(summary_rows)


def test_plot_writes_png_and_pdf(tmp_path):
    paired, summary = inputs()
    png = tmp_path / "nested.png"
    pdf = tmp_path / "nested.pdf"
    MODULE.plot_nested_horizons(paired, summary, png, pdf)
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf.read_bytes().startswith(b"%PDF")


def test_figure_artifacts_are_bound_to_registered_sources(tmp_path):
    paired, summary = inputs()
    paired_path = tmp_path / "paired.csv"
    summary_path = tmp_path / "summary.csv"
    paired.to_csv(paired_path, index=False, lineterminator="\n")
    summary.to_csv(summary_path, index=False, lineterminator="\n")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        __import__("json").dumps(
            {
                "outputs": {
                    paired_path.name: MODULE.canonical_sha256(paired_path),
                    summary_path.name: MODULE.canonical_sha256(summary_path),
                }
            }
        ),
        encoding="utf-8",
    )
    png = tmp_path / "nested.png"
    pdf = tmp_path / "nested.pdf"
    MODULE.plot_nested_horizons(paired, summary, png, pdf)
    MODULE.bind_figure_artifacts(
        manifest_path, paired_path, summary_path, png, pdf
    )
    bound = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    assert bound["figures"][png.name] == MODULE.canonical_sha256(png)
    assert bound["figures"][pdf.name] == MODULE.canonical_sha256(pdf)
    assert bound["figure_generator"]["script"] == "scripts/plot_nested_horizons.py"


def test_plot_rejects_missing_seed(tmp_path):
    paired, summary = inputs()
    paired = paired[
        ~((paired["horizon_years"] == 2) & (paired["seed"] == 9))
    ]
    with pytest.raises(ValueError, match="Expected seeds 0-9"):
        MODULE.plot_nested_horizons(
            paired,
            summary,
            tmp_path / "bad.png",
            tmp_path / "bad.pdf",
        )


def test_plot_rejects_duplicate_seed(tmp_path):
    paired, summary = inputs()
    paired = pd.concat([paired, paired.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        MODULE.plot_nested_horizons(
            paired,
            summary,
            tmp_path / "bad.png",
            tmp_path / "bad.pdf",
        )
