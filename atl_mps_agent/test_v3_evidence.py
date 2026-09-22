from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v3_manifest_binds_model_benchmark_and_hosted_run() -> None:
    evidence = Path(__file__).parent / "evidence" / "v3"
    manifest = json.loads(
        (evidence / "evidence_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["architecture_selected_without_fresh_test"] is True
    assert manifest["quantum_hardware"] is False
    for relative, expected in manifest["files_sha256"].items():
        assert _digest(evidence / relative) == expected


def test_v3_claims_distinguish_prediction_from_system_behavior() -> None:
    evidence = Path(__file__).parent / "evidence" / "v3"
    result = json.loads(
        (evidence / "v3_benchmark_results.json").read_text(encoding="utf-8")
    )
    assert result["configuration"]["mps_parameters_per_member"] == 586
    assert result["configuration"]["ann_parameters_per_member"] == 586
    mps_signal = result["mps_ensemble"]["signal_metrics"]
    ann_signal = result["matched_ann_ensemble"]["signal_metrics"]
    assert mps_signal["mse_pp2"] > ann_signal["mse_pp2"]
    assert mps_signal["rank_ic"] < ann_signal["rank_ic"]
    mps_return = result["mps_ensemble"]["portfolio_metrics"]["total_return_pct"]
    ann_return = result["matched_ann_ensemble"]["portfolio_metrics"]["total_return_pct"]
    assert mps_return > ann_return
    interval = result["paired_member_inference"]
    assert interval["bootstrap_95pct_low_pp"] <= 0.0 <= interval["bootstrap_95pct_high_pp"]
