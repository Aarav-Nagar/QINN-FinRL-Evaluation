"""Build the hash-bound result for the preregistered ATL replication."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .replication_audit import (
    _cost_sensitivity,
    _replication_comparison,
    _runtime_replay,
    _snapshot_profile,
    _weekly_decomposition,
)


def _file_sha256(path: Path) -> str:
    if path.suffix.lower() == ".json":
        payload = json.dumps(
            _read_json(path), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_prospective_evaluation(
    *,
    primary_result_path: Path,
    primary_snapshots_path: Path,
    rerun_result_path: Path,
    rerun_snapshots_path: Path,
    benchmark_path: Path,
    artifact_path: Path,
    baselines: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Reconcile hosted evidence without changing or refitting the frozen policy."""

    primary = _read_json(primary_result_path)
    rerun = _read_json(rerun_result_path)
    primary_snapshots = _read_json(primary_snapshots_path)
    rerun_snapshots = _read_json(rerun_snapshots_path)
    benchmark = _read_json(benchmark_path)

    replication = _replication_comparison([primary, rerun])
    primary_profile = _snapshot_profile(primary_snapshots)
    rerun_profile = _snapshot_profile(rerun_snapshots)
    primary_replay = _runtime_replay(primary, primary_snapshots, artifact_path)
    rerun_replay = _runtime_replay(rerun, rerun_snapshots, artifact_path)
    costs = _cost_sensitivity(primary)
    weeks = _weekly_decomposition(primary)
    after_10_bps = next(
        row for row in costs if row["basis_points_per_traded_notional"] == 10
    )
    after_cost_return = float(after_10_bps["estimated_return_pct"])
    classification = "flat"
    if after_cost_return > 0.0:
        classification = "positive"
    elif after_cost_return < 0.0:
        classification = "negative"

    full_window_change = float(primary["run"]["final_equity"]) - float(
        primary["run"]["initial_equity"]
    )
    largest_week = max(weeks, key=lambda row: abs(float(row["dollar_change"])))
    largest_week_fraction = (
        abs(float(largest_week["dollar_change"])) / abs(full_window_change)
        if full_window_change
        else None
    )
    market_quality = dict(
        (primary.get("run", {}).get("metadata") or {}).get("market_data_quality")
        or {}
    )
    systems = benchmark["systems"]
    mps_system = systems["selected_mps_trend"]
    ann_system = systems["matched_ann_trend"]

    source_paths = {
        "primary_result": primary_result_path,
        "primary_snapshots": primary_snapshots_path,
        "rerun_result": rerun_result_path,
        "rerun_snapshots": rerun_snapshots_path,
        "benchmark": benchmark_path,
        "frozen_artifact": artifact_path,
    }
    return {
        "schema_version": 1,
        "artifact": "atl_prospective_four_week_replication_result",
        "generated_at_utc": "2026-09-22T00:30:00Z",
        "status": "complete",
        "classification_after_10_bps": classification,
        "protocol": {
            "registered_commit": "4ea061a55fb45c61c0adeb8cacd3ef0f0b122e76",
            "registration_binding_commit": "3f759c2df9c4d32c9a5c5d1580ca7761930c6e79",
            "account_binding_commit": "c8c28f7c7fc2b0d6c9bf02518a52cd0e5c015b7f",
            "start_date": "2026-08-24",
            "exclusive_end_date": "2026-09-19",
            "executed_after_earliest_time": True,
            "policy_changed": False,
            "retrained_or_tuned": False,
        },
        "deployment": {
            "account_agent_id": "agent_3cc6d5aac07b",
            "account_immutable_version_id": "agv_1c48b644b71e",
            "legacy_anonymous_agent_id": "agent_63a8501e0235",
            "legacy_anonymous_version_id": "agv_c23fdf3230ad",
            "account_migration_only": True,
            "live_trading_enabled": False,
            "artifact_sha256": _file_sha256(artifact_path),
        },
        "hosted_primary": {
            "run_id": primary["run"]["run_id"],
            "initial_equity": primary["run"]["initial_equity"],
            "gross_final_equity": primary["run"]["final_equity"],
            "gross_return_pct": 100.0 * float(primary["metrics"]["total_return"]),
            "maximum_drawdown_pct": 100.0
            * float(primary["metrics"]["max_drawdown"]),
            "trade_count": primary["metrics"]["num_trades"],
            "decision_count": len(primary.get("decisions") or []),
            "timeout_holds": primary["metrics"]["timeout_holds"],
            "traded_notional": after_10_bps["traded_notional"],
            "estimated_10_bps_cost": after_10_bps["estimated_cost"],
            "estimated_final_equity_after_10_bps": after_10_bps[
                "estimated_final_equity"
            ],
            "estimated_return_pct_after_10_bps": after_cost_return,
        },
        "exact_rerun": {
            **replication,
            "snapshot_payload_exact": primary_snapshots == rerun_snapshots,
            "snapshot_hash_exact": primary_profile["canonical_sha256"]
            == rerun_profile["canonical_sha256"],
        },
        "local_action_replay": {
            "primary": primary_replay,
            "rerun": rerun_replay,
        },
        "data_quality": {
            "primary_snapshot_profile": primary_profile,
            "rerun_snapshot_profile": rerun_profile,
            "atl_market_data_quality": market_quality,
            "all_market_quality_error_counts_zero": all(
                int(market_quality.get(field, 0)) == 0
                for field in (
                    "dropped_decision_bars",
                    "missing_source_bars",
                    "duplicate_source_bars",
                    "off_grid_source_bars",
                    "invalid_source_bars",
                )
            ),
        },
        "cost_sensitivity": costs,
        "weekly_decomposition": weeks,
        "weekly_concentration": {
            "full_window_dollar_change": full_window_change,
            "largest_absolute_week": largest_week["week"],
            "largest_absolute_week_dollar_change": largest_week["dollar_change"],
            "largest_absolute_week_fraction_of_full_window_change": largest_week_fraction,
            "one_week_exceeds_full_window_change": bool(
                largest_week_fraction is not None and largest_week_fraction > 1.0
            ),
        },
        "matched_controls_after_10_bps": systems,
        "matched_model_comparison": {
            "selected_mps_trend_return_pct": mps_system["total_return_pct"],
            "matched_ann_trend_return_pct": ann_system["total_return_pct"],
            "return_difference_percentage_points": float(
                mps_system["total_return_pct"]
            )
            - float(ann_system["total_return_pct"]),
            "same_trade_count": mps_system["trade_count"] == ann_system["trade_count"],
            "supports_unique_mps_value": False,
            "note": "The frozen MPS and matched ANN produced identical portfolio results in this window.",
        },
        "atl_native_references": baselines,
        "limitations": [
            "One four-week window is not evidence of alpha, significance, or future profitability.",
            "The exact rerun tests determinism on one fixed sample, not independent replication.",
            "The ATL buy-and-hold reference remained entirely in cash because all ten requested whole-share purchases were skipped.",
            "The local matched-control simulator is comparable within its table but is not the hosted execution ledger used for the primary endpoint.",
        ],
        "source_file_sha256": {
            name: _file_sha256(path) for name, path in source_paths.items()
        },
    }


def write_prospective_evaluation(
    result: dict[str, Any], output_dir: Path
) -> None:
    """Write the canonical result and its two human-auditable tables."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result_manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8", newline="\n"
    )
    for name, rows in (
        ("cost_sensitivity.csv", result["cost_sensitivity"]),
        ("weekly_decomposition.csv", result["weekly_decomposition"]),
    ):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            fieldnames = list(
                dict.fromkeys(key for row in rows for key in row)
            )
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
