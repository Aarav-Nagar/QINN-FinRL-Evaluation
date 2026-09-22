from __future__ import annotations

import json
from pathlib import Path


def test_exploratory_matrix_is_fixed_and_bounded() -> None:
    path = Path(__file__).parent / "evidence" / "exploratory" / "plan_manifest.json"
    plan = json.loads(path.read_text(encoding="utf-8"))

    assert plan["outcomes_observed_for_declared_subwindow_runs"] is False
    assert plan["aggregate_us_evaluation_outcome_previously_observed"] is True
    assert plan["policy_changed"] is False
    assert len(plan["us_windows"]) == 4
    assert len(plan["china_runs"]) == 2
    assert {item["decision_source"] for item in plan["china_runs"]} == {"rule_based"}
    assert "without synthetic imputation" in plan["china_mps_transfer_gate"]
