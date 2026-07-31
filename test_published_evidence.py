from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parent
SCRIPT = ROOT / "scripts" / "summarize_final_evaluation.py"
SPEC = importlib.util.spec_from_file_location("published_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
published_summary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = published_summary
SPEC.loader.exec_module(published_summary)
PRIMARY = ROOT / "results" / "final"
SHIFTED = ROOT / "results" / "robustness" / "shifted"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_published_primary_and_shifted_runs_are_complete() -> None:
    primary_manifest, primary = published_summary.load_final_run(PRIMARY)
    shifted_manifest, shifted = published_summary.load_final_run(SHIFTED)
    assert primary_manifest["experiment"]["window_name"] == "primary"
    assert shifted_manifest["experiment"]["window_name"] == "shifted"
    assert len(primary) == 30
    assert len(shifted) == 30
    assert primary.groupby("condition").size().to_dict() == {
        "ANN signal": 10,
        "Base FinRL": 10,
        "QINN-MPS signal": 10,
    }
    assert shifted.groupby("condition").size().to_dict() == {
        "ANN signal": 10,
        "Base FinRL": 10,
        "QINN-MPS signal": 10,
    }


@pytest.mark.parametrize(
    ("directory", "manifest_name", "artifact_name"),
    [
        (PRIMARY, "final_manifest.json", "final_ten_seed_evaluation"),
        (
            SHIFTED,
            "robustness_manifest.json",
            "shifted_window_ten_seed_evaluation",
        ),
    ],
)
def test_published_provenance_hashes_match(
    directory: Path,
    manifest_name: str,
    artifact_name: str,
) -> None:
    manifest = json.loads((directory / manifest_name).read_text(encoding="utf-8"))
    assert manifest["artifact"] == artifact_name
    for name, digest in manifest["inputs"].items():
        assert sha256(directory / name) == digest
    for name, digest in manifest["outputs"].items():
        assert sha256(directory / name) == digest


def test_manuscript_uses_corrected_published_results() -> None:
    manuscript = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    primary = json.loads((PRIMARY / "primary_inference.json").read_text())
    shifted = json.loads((SHIFTED / "robustness_inference.json").read_text())
    assert float(primary["mean_difference"]) == pytest.approx(-0.03660739945)
    assert float(shifted["mean_difference"]) == pytest.approx(0.08624548077)
    assert "mean Sharpe was 0.821 for ANN and 0.784 for MPS" in manuscript
    assert "Mean Sharpe was 0.714 for MPS, 0.627 for ANN" in manuscript
    assert "Shifted-period robustness remains pending" not in manuscript
    assert "mean Sharpe was 0.800 for ANN and 0.762 for MPS" not in manuscript