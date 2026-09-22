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


def test_current_evaluation_is_frozen_positive_and_hash_bound() -> None:
    evidence = Path(__file__).parent / "evidence" / "current"
    manifest = json.loads(
        (evidence / "evaluation_manifest.json").read_text(encoding="utf-8")
    )
    hosted = json.loads(
        (evidence / "current_hosted_run_summary.json").read_text(encoding="utf-8")
    )
    benchmark = json.loads(
        (evidence / "current_benchmark_results.json").read_text(encoding="utf-8")
    )
    assert manifest["freeze_commit"] == benchmark["freeze_commit"]
    assert manifest["architecture_selected_without_fresh_test"] is True
    assert manifest["fresh_test_consumed_once"] is True
    assert hosted["run"]["timeout_holds"] == 0
    assert hosted["estimated_cost_adjustment"]["estimated_after_cost_return_pct"] > 0.0
    assert hosted["run"]["gross_total_return_pct"] > hosted[
        "estimated_cost_adjustment"
    ]["estimated_after_cost_return_pct"]
    assert hosted["atl_native_references"]["djia"]["total_return_pct"] > hosted[
        "run"
    ]["gross_total_return_pct"]
    assert (
        benchmark["configuration"]["mps_parameters_per_member"]
        == benchmark["configuration"]["ann_parameters_per_member"]
        == 586
    )
    assert benchmark["prediction_metrics"]["residual_mps"]["mse_pp2"] < benchmark[
        "prediction_metrics"
    ]["matched_ann"]["mse_pp2"]
    for relative, expected in manifest["files_sha256"].items():
        assert _digest(evidence / relative) == expected
