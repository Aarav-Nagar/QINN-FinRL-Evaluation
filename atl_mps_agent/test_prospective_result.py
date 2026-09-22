import json
from pathlib import Path

from atl_mps_agent.prospective_evaluation import build_prospective_evaluation


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "atl_mps_agent" / "evidence" / "prospective"
ARTIFACT = ROOT / "atl_mps_agent" / "artifacts" / "atl_residual_mps_current.pt"


def _rebuild() -> tuple[dict, dict]:
    published = json.loads((EVIDENCE / "result_manifest.json").read_text())
    rebuilt = build_prospective_evaluation(
        primary_result_path=EVIDENCE / "primary_result.json",
        primary_snapshots_path=EVIDENCE / "primary_snapshots.json",
        rerun_result_path=EVIDENCE / "rerun_result.json",
        rerun_snapshots_path=EVIDENCE / "rerun_snapshots.json",
        benchmark_path=EVIDENCE / "controls" / "benchmark_results.json",
        artifact_path=ARTIFACT,
        baselines=published["atl_native_references"],
    )
    return published, rebuilt


def test_published_prospective_result_rebuilds_exactly():
    published, rebuilt = _rebuild()
    assert rebuilt == published


def test_prospective_result_satisfies_registered_integrity_checks():
    result, _ = _rebuild()
    assert result["status"] == "complete"
    assert result["classification_after_10_bps"] == "negative"
    assert result["deployment"]["artifact_sha256"] == (
        "a2be3a5286cf7bb72116dac8a2f23b82173ddecdb44c7c81153edbe123d999b6"
    )
    assert result["exact_rerun"]["exact_replication"] is True
    assert result["exact_rerun"]["snapshot_payload_exact"] is True
    assert result["local_action_replay"]["primary"]["mismatch_count"] == 0
    assert result["local_action_replay"]["rerun"]["mismatch_count"] == 0
    assert result["data_quality"]["all_market_quality_error_counts_zero"] is True
    assert result["matched_model_comparison"]["return_difference_percentage_points"] == 0.0
    assert result["matched_model_comparison"]["supports_unique_mps_value"] is False
    assert result["protocol"]["policy_changed"] is False
    assert result["protocol"]["retrained_or_tuned"] is False


def test_primary_endpoint_and_baseline_caveat_are_preserved():
    result, _ = _rebuild()
    primary = result["hosted_primary"]
    assert primary["gross_return_pct"] == -2.475
    assert primary["estimated_return_pct_after_10_bps"] < -2.64
    assert primary["trade_count"] == 6
    assert primary["timeout_holds"] == 0
    buyhold = result["atl_native_references"]["buy_and_hold"]
    assert buyhold["invested_ratio"] == 0.0
    assert buyhold["symbols_bought"] == 0
