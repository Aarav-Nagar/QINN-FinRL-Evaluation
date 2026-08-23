import hashlib
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence" / "exploratory"
RESULTS = json.loads((EVIDENCE / "exploratory_results.json").read_text(encoding="utf-8"))
PLAN = json.loads((EVIDENCE / "plan_manifest.json").read_text(encoding="utf-8"))
ACCOUNT = json.loads(
    (ROOT / "evidence" / "account" / "account_hosted_run.json").read_text(
        encoding="utf-8"
    )
)
SIGNAL_FIELDS = (
    "price",
    "rsi",
    "macd",
    "macd_signal",
    "sma20",
    "sma50",
    "bb_upper",
    "bb_lower",
)


def _canonical_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _window_files(label):
    slug = label.lower().replace("-", "_")
    result = json.loads(
        (EVIDENCE / "us" / f"{slug}_result.json").read_text(encoding="utf-8")
    )
    snapshots = json.loads(
        (EVIDENCE / "us" / f"{slug}_snapshots.json").read_text(encoding="utf-8")
    )
    return result, snapshots


def test_exploratory_plan_was_frozen_before_subwindow_outcomes():
    assert PLAN["outcomes_observed_for_declared_subwindow_runs"] is False
    assert PLAN["policy_changed"] is False
    assert RESULTS["plan_commit"] == "c8c4c87ea61ee2332ed22fd9a6d26fb5bda02228"
    assert RESULTS["policy_changed"] is False
    assert RESULTS["classification"] == "exploratory historical stress test"


def test_us_primary_files_match_manifest_and_pass_quality_contract():
    expected_signals = set(SIGNAL_FIELDS)
    for window in RESULTS["us_windows"]:
        result, snapshots = _window_files(window["label"])
        timestamps = [snapshot["timestamp"] for snapshot in snapshots]
        day_counts = Counter(timestamp[:10] for timestamp in timestamps)

        assert result["run"]["run_id"] == window["run_id"]
        assert math.isclose(
            100.0 * result["metrics"]["total_return"],
            window["gross_return_pct"],
            abs_tol=1e-12,
        )
        assert math.isclose(
            100.0 * result["metrics"]["max_drawdown"],
            window["max_drawdown_pct"],
            abs_tol=1e-12,
        )
        assert result["metrics"]["num_trades"] == window["trades"]
        assert len(result["decisions"]) == window["decisions"]
        assert result["metrics"]["timeout_holds"] == window["timeouts"]
        assert len(snapshots) == window["snapshot_count"]
        assert len(set(timestamps)) == len(timestamps)
        assert timestamps == sorted(timestamps)
        assert len(day_counts) == window["trading_days"]
        assert set(day_counts.values()) == {7}
        assert _canonical_sha256(result) == window["canonical_result_sha256"]
        assert _canonical_sha256(snapshots) == window["canonical_snapshots_sha256"]

        symbols = set()
        for snapshot in snapshots:
            signals = snapshot["top_signals"]
            assert len(signals) == 10
            symbols.update(signals)
            for row in signals.values():
                assert expected_signals <= set(row)
                for field in SIGNAL_FIELDS:
                    value = float(row[field])
                    assert math.isfinite(value)
                assert float(row["price"]) > 0.0
        assert len(symbols) == window["distinct_symbols"]
        assert window["snapshot_quality_passed"] is True


def test_cost_calculation_and_window_aggregate_are_recomputed():
    gross = []
    after_cost = []
    for window in RESULTS["us_windows"]:
        result, _ = _window_files(window["label"])
        initial = float(result["run"]["initial_equity"])
        final = float(result["run"]["final_equity"])
        notional = sum(float(trade["value"]) for trade in result["trades"])
        net_return = 100.0 * ((final - notional * 0.001) / initial - 1.0)
        assert notional == window["traded_notional"]
        assert math.isclose(net_return, window["after_cost_return_pct"], abs_tol=1e-12)
        gross.append(window["gross_return_pct"])
        after_cost.append(window["after_cost_return_pct"])

    aggregate = RESULTS["aggregate_us"]
    assert sum(gross) == aggregate["fresh_state_sum_gross_return_pct"]
    assert sum(after_cost) == aggregate["fresh_state_sum_after_cost_return_pct"]
    assert sum(value > 0.0 for value in gross) == aggregate["positive_window_count"]
    assert sum(value == 0.0 for value in gross) == aggregate["flat_window_count"]


def test_continuous_path_action_divergence_is_recomputed():
    continuous_actions = {
        decision["timestamp"]: decision.get("actions_submitted") or []
        for decision in ACCOUNT["decisions"]
    }
    total_common = 0
    total_matches = 0
    for window in RESULTS["us_windows"]:
        result, _ = _window_files(window["label"])
        common = [
            decision
            for decision in result["decisions"]
            if decision["timestamp"] in continuous_actions
        ]
        matches = sum(
            (decision.get("actions_submitted") or [])
            == continuous_actions[decision["timestamp"]]
            for decision in common
        )
        assert len(common) == window["continuous_run_common_decisions"]
        assert matches == window["continuous_run_action_matches"]
        total_common += len(common)
        total_matches += matches

    aggregate = RESULTS["aggregate_us"]
    assert total_common == aggregate["continuous_overlap_decisions"] == 224
    assert total_matches == aggregate["continuous_overlap_action_matches"] == 96
    assert math.isclose(100.0 * total_matches / total_common, 42.857142857142854)
    expected_gap = (
        aggregate["fresh_state_sum_gross_return_pct"]
        - aggregate["continuous_reference_return_pct_through_us_c"]
    )
    assert math.isclose(
        expected_gap, aggregate["fresh_state_minus_continuous_pct_points"]
    )


def test_runtime_and_repeat_evidence_are_exact_not_performance_claims():
    for window in RESULTS["us_windows"]:
        assert window["runtime_replay_exact_batches"] == window["decisions"]
        assert window["runtime_replay_mismatches"] == 0
    repeat = RESULTS["us_c_exact_repeat"]
    assert len(set(repeat["run_ids"])) == 2
    assert repeat["metrics_exact"] is True
    assert repeat["equity_curve_exact"] is True
    assert repeat["trade_ledger_exact"] is True
    assert repeat["decision_ledger_exact"] is True
    assert repeat["snapshots_exact"] is True
    assert RESULTS["finding"]["performance_claim"].startswith("No groundbreaking")


def test_china_failures_are_not_encoded_as_zero_returns_or_mps_results():
    for control in RESULTS["china_controls"]:
        assert control["status"].startswith("failed_before")
        assert control["return_is_observed"] is False
        assert "return_pct" not in control
        assert control["decision_source"] == "rule_based"
    transfer = RESULTS["china_mps_transfer"]
    assert transfer["status"] == "not_run"
    assert transfer["platform_controls_are_mps_results"] is False
