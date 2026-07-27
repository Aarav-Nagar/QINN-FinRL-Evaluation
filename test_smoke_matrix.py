from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).parent / "scripts" / "summarize_smoke_matrix.py"
SPEC = importlib.util.spec_from_file_location("smoke_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
smoke_summary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = smoke_summary
SPEC.loader.exec_module(smoke_summary)


def make_run(path: Path, dimension: int, timesteps: int = 512) -> Path:
    path.mkdir()
    manifest = {
        "experiment": {
            "mps_bond_dimension": dimension,
            "ppo_timesteps": timesteps,
            "ppo_seeds": [0],
            "encoder_epochs": 1,
        },
        "runtime": {
            "encoder_device_resolved": "cpu",
            "elapsed_seconds": 12.0,
        },
        "ann_parameter_count": 369,
        "mps_parameter_count": dimension * 10,
    }
    (path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pd.DataFrame(
        {
            "condition": ["Base FinRL", "ANN signal", "QINN-MPS signal"],
            "seed": [0, 0, 0],
            "sharpe": [0.5, 0.6, 0.7],
        }
    ).to_csv(path / "ppo_backtest_metrics.csv", index=False)
    return path


def test_smoke_summary_orders_dimensions_and_marks_non_evidence(tmp_path: Path) -> None:
    run8 = make_run(tmp_path / "bd8", 8)
    run2 = make_run(tmp_path / "bd2", 2)
    frame = smoke_summary.summarize([run8, run2])
    assert frame["bond_dimension"].tolist() == [2, 8]
    assert frame["paper_evidence"].tolist() == [False, False]
    assert frame["mps_sharpe"].tolist() == [0.7, 0.7]


def test_smoke_summary_rejects_mismatched_budgets(tmp_path: Path) -> None:
    run2 = make_run(tmp_path / "bd2", 2, timesteps=512)
    run4 = make_run(tmp_path / "bd4", 4, timesteps=1024)
    with pytest.raises(ValueError, match="ppo_timesteps"):
        smoke_summary.summarize([run2, run4])
