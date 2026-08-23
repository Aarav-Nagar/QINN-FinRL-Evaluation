from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v2_evidence_manifest_binds_public_artifacts() -> None:
    evidence = Path(__file__).parent / "evidence" / "v2"
    manifest = json.loads((evidence / "evidence_manifest.json").read_text(encoding="utf-8"))
    assert manifest["quantum_hardware"] is False
    assert manifest["verification_level"] == "self_reported"
    for relative, expected in manifest["files_sha256"].items():
        assert _digest(evidence / relative) == expected


def test_v2_benchmark_is_parameter_matched_and_inconclusive() -> None:
    evidence = Path(__file__).parent / "evidence" / "v2"
    result = json.loads((evidence / "benchmark_results.json").read_text(encoding="utf-8"))
    assert result["configuration"]["mps_parameters"] == 369
    assert result["configuration"]["ann_parameters"] == 369
    interval = result["paired_inference"]
    assert interval["bootstrap_95pct_low_pp"] <= 0.0 <= interval["bootstrap_95pct_high_pp"]
    assert result["feature_audit"]["nonfinite_total"] == 0
    assert result["feature_audit"]["degenerate_features"] == []
