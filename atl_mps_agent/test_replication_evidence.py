from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_replication_evidence_is_exact_bounded_and_hash_bound() -> None:
    evidence = Path(__file__).parent / "evidence" / "replication"
    manifest = json.loads(
        (evidence / "replication_manifest.json").read_text(encoding="utf-8")
    )
    audit = json.loads(
        (evidence / "replication_audit.json").read_text(encoding="utf-8")
    )
    assert manifest["policy_changed_after_original_freeze"] is False
    assert audit["same_window_replication"]["exact_replication"] is True
    assert audit["same_window_runtime_replay"]["all_action_batches_exact"] is True
    assert audit["temporal_extension"]["classification"] == "flat"
    assert audit["temporal_extension"]["runtime_replay"][
        "all_action_batches_exact"
    ] is True
    comparison = audit["collection_context_comparison"]
    assert comparison["raw_field_consistency"]["price"]["exact_match_fraction"] == 1.0
    assert comparison["symbol_set_exact_match_fraction"] < 0.25
    systems = audit["hosted_context_benchmark"]["systems"]
    assert systems["selected_mps_trend"]["total_return_pct"] > 0.0
    assert systems["selected_mps_trend"]["total_return_pct"] == systems[
        "matched_ann_trend"
    ]["total_return_pct"]
    costs = {
        str(row["basis_points_per_traded_notional"]): row
        for row in audit["cost_sensitivity"]
    }
    assert costs["10"]["estimated_return_pct"] > 0.0
    assert costs["50"]["estimated_return_pct"] < 0.0
    contribution = sum(
        row["contribution_pct_initial"] for row in audit["weekly_decomposition"]
    )
    assert abs(contribution - 1.29364) < 1e-10
    for relative, expected in manifest["files_sha256"].items():
        assert _digest(evidence / relative) == expected
