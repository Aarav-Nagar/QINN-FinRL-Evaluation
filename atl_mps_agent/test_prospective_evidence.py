from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prospective_window_is_registered_before_outcome() -> None:
    evidence = Path(__file__).parent / "evidence" / "prospective"
    manifest = json.loads(
        (evidence / "prospective_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["registered_at_date"] < manifest["window_start"]
    assert manifest["outcome_observed"] is False
    assert manifest["policy_changed_for_prospective_test"] is False
    assert manifest["window_start"] == "2026-08-24"
    assert manifest["window_end_exclusive"] == "2026-09-19"
    assert "10 basis points" in manifest["primary_endpoint"]
    for relative, expected in manifest["files_sha256"].items():
        assert _digest(evidence / relative) == expected
