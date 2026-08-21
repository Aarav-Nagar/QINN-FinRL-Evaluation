from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_current_freeze_precedes_fresh_data_and_binds_inputs() -> None:
    evidence = Path(__file__).parent / "evidence" / "current"
    manifest = json.loads((evidence / "freeze_manifest.json").read_text(encoding="utf-8"))
    assert manifest["architecture_selected_without_fresh_test"] is True
    assert manifest["fresh_data_collected"] is False
    assert manifest["quantum_hardware"] is False
    assert manifest["selected_mps_rank_weight"] >= manifest["minimum_mps_rank_weight"] >= 0.5
    for relative, expected in manifest["files_sha256"].items():
        assert _digest(evidence / relative) == expected


def test_current_artifact_has_positive_validation_system_result() -> None:
    summary_path = Path(__file__).parent / "artifacts" / "atl_residual_mps_current.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["target_horizon_steps"] == 7
    assert summary["selected_policy"]["model_weight"] >= 0.5
    assert summary["selected_policy"]["rebalance_days"] == 5
    assert summary["selected_validation_metrics"]["total_return_pct"] > 0.0
    assert summary["selected_validation_metrics"]["trade_count"] >= 3
