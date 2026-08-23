from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_account_owned_agent_evidence_is_bound_and_safe() -> None:
    evidence = Path(__file__).parent / "evidence" / "account"
    manifest = json.loads((evidence / "account_manifest.json").read_text(encoding="utf-8"))
    hosted = json.loads((evidence / "account_hosted_run.json").read_text(encoding="utf-8"))

    assert manifest["agent_id"] == "agent_3cc6d5aac07b"
    assert manifest["agent_version_id"] == "agv_1c48b644b71e"
    assert manifest["live_trading_enabled"] is False
    assert manifest["policy_changed_for_account_migration"] is False
    assert manifest["account_linked_run"]["run_id"] == hosted["run"]["run_id"]
    assert hosted["run"]["final_equity"] == 1012.9364
    assert hosted["run"]["num_trades"] == 11
    assert hosted["run"]["metadata"]["timeout_holds"] == 0
    assert "api_key" not in json.dumps(manifest).lower()

    for relative, expected in manifest["files_sha256"].items():
        assert _digest(evidence / relative) == expected
