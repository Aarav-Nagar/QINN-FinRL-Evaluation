"""Reconcile hosted ATL replications, snapshots, and robustness diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .features import FEATURE_NAMES, snapshot_features
from .deployment_policy import DeploymentMPSPolicy


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
METRIC_FIELDS = (
    "total_return",
    "sharpe_ratio",
    "max_drawdown",
    "num_trades",
    "final_equity",
    "timeout_holds",
)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _deduplicate_snapshots(
    groups: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for group in groups:
        for snapshot in group:
            unique[str(snapshot.get("timestamp"))] = snapshot
    return [unique[key] for key in sorted(unique)]


def _snapshot_profile(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    timestamps = [str(snapshot.get("timestamp")) for snapshot in snapshots]
    day_counts = Counter(timestamp[:10] for timestamp in timestamps)
    offsets = []
    price_rows = 0
    invalid_prices = 0
    nonfinite_fields = 0
    symbol_counts = []
    symbols: set[str] = set()
    for snapshot in snapshots:
        try:
            offset = datetime.fromisoformat(str(snapshot.get("timestamp"))).utcoffset()
            offsets.append(str(offset))
        except (TypeError, ValueError):
            offsets.append("invalid")
        signals = snapshot.get("top_signals") or {}
        symbol_counts.append(len(signals))
        symbols.update(signals)
        for row in signals.values():
            price_rows += 1
            try:
                price = float((row or {}).get("price"))
            except (TypeError, ValueError):
                price = float("nan")
            if not math.isfinite(price) or price <= 0.0:
                invalid_prices += 1
            for field in SIGNAL_FIELDS:
                try:
                    value = float((row or {}).get(field))
                except (TypeError, ValueError):
                    value = float("nan")
                if not math.isfinite(value):
                    nonfinite_fields += 1
    duplicates = len(timestamps) - len(set(timestamps))
    return {
        "snapshot_count": len(snapshots),
        "unique_timestamp_count": len(set(timestamps)),
        "duplicate_timestamp_count": duplicates,
        "chronologically_sorted": timestamps == sorted(timestamps),
        "first_timestamp": min(timestamps) if timestamps else None,
        "last_timestamp": max(timestamps) if timestamps else None,
        "trading_day_count": len(day_counts),
        "observations_per_day": dict(sorted(day_counts.items())),
        "all_days_have_seven_observations": bool(day_counts)
        and all(count == 7 for count in day_counts.values()),
        "timezone_offsets": sorted(set(offsets)),
        "distinct_symbol_count": len(symbols),
        "signals_per_snapshot_min": min(symbol_counts) if symbol_counts else 0,
        "signals_per_snapshot_max": max(symbol_counts) if symbol_counts else 0,
        "signal_price_row_count": price_rows,
        "invalid_or_nonpositive_price_count": invalid_prices,
        "nonfinite_signal_field_count": nonfinite_fields,
        "portfolio_state_present_count": sum(
            bool(snapshot.get("portfolio")) for snapshot in snapshots
        ),
        "nonempty_holdings_snapshot_count": sum(
            bool(snapshot.get("current_holdings")) for snapshot in snapshots
        ),
        "canonical_sha256": _canonical_sha256(snapshots),
    }


def _feature_map(
    snapshots: list[dict[str, Any]],
) -> dict[tuple[str, str], np.ndarray]:
    history: dict[str, list[float]] = defaultdict(list)
    output: dict[tuple[str, str], np.ndarray] = {}
    for snapshot in snapshots:
        timestamp = str(snapshot.get("timestamp"))
        signals = snapshot.get("top_signals") or {}
        for symbol in sorted(signals):
            output[(timestamp, symbol)] = snapshot_features(snapshot, symbol, history)
        for symbol, row in signals.items():
            try:
                price = float((row or {}).get("price"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(price) and price > 0.0:
                history[symbol].append(price)
    return output


def _compare_snapshot_sources(
    collected: list[dict[str, Any]], hosted: list[dict[str, Any]]
) -> dict[str, Any]:
    left = {str(snapshot.get("timestamp")): snapshot for snapshot in collected}
    right = {str(snapshot.get("timestamp")): snapshot for snapshot in hosted}
    common = sorted(set(left) & set(right))
    field_stats = {
        field: {"comparisons": 0, "exact_matches": 0, "absolute_differences": []}
        for field in SIGNAL_FIELDS
    }
    symbol_set_matches = 0
    exact_payload_matches = 0
    for timestamp in common:
        left_signals = left[timestamp].get("top_signals") or {}
        right_signals = right[timestamp].get("top_signals") or {}
        if set(left_signals) == set(right_signals):
            symbol_set_matches += 1
        if left_signals == right_signals:
            exact_payload_matches += 1
        for symbol in sorted(set(left_signals) & set(right_signals)):
            for field in SIGNAL_FIELDS:
                try:
                    left_value = float((left_signals[symbol] or {}).get(field))
                    right_value = float((right_signals[symbol] or {}).get(field))
                except (TypeError, ValueError):
                    continue
                if not (math.isfinite(left_value) and math.isfinite(right_value)):
                    continue
                stats = field_stats[field]
                difference = abs(left_value - right_value)
                stats["comparisons"] += 1
                stats["exact_matches"] += int(difference == 0.0)
                stats["absolute_differences"].append(difference)

    field_results = {}
    for field, stats in field_stats.items():
        differences = stats.pop("absolute_differences")
        comparisons = int(stats["comparisons"])
        field_results[field] = {
            **stats,
            "exact_match_fraction": stats["exact_matches"] / comparisons
            if comparisons
            else 0.0,
            "mean_absolute_difference": float(np.mean(differences))
            if differences
            else 0.0,
            "maximum_absolute_difference": max(differences) if differences else 0.0,
        }

    collected_features = _feature_map(collected)
    hosted_features = _feature_map(hosted)
    common_features = sorted(set(collected_features) & set(hosted_features))
    feature_results = []
    for index, name in enumerate(FEATURE_NAMES):
        differences = np.asarray(
            [
                abs(
                    float(collected_features[key][index])
                    - float(hosted_features[key][index])
                )
                for key in common_features
            ],
            dtype=float,
        )
        feature_results.append(
            {
                "feature": name,
                "comparisons": len(differences),
                "exact_match_fraction": float(np.mean(differences == 0.0))
                if len(differences)
                else 0.0,
                "mean_absolute_difference": float(np.mean(differences))
                if len(differences)
                else 0.0,
                "maximum_absolute_difference": float(np.max(differences))
                if len(differences)
                else 0.0,
            }
        )
    return {
        "collected_snapshot_count": len(collected),
        "hosted_snapshot_count": len(hosted),
        "common_timestamp_count": len(common),
        "symbol_set_exact_match_count": symbol_set_matches,
        "symbol_set_exact_match_fraction": symbol_set_matches / len(common)
        if common
        else 0.0,
        "top_signals_payload_exact_match_count": exact_payload_matches,
        "top_signals_payload_exact_match_fraction": exact_payload_matches
        / len(common)
        if common
        else 0.0,
        "raw_field_consistency": field_results,
        "model_feature_consistency": feature_results,
    }


def _result_signature(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics") or {}
    return {
        "metrics": {field: metrics.get(field) for field in METRIC_FIELDS},
        "equity_curve": result.get("equity_curve") or [],
        "trades": result.get("trades") or [],
        "decisions": result.get("decisions") or [],
    }


def _runtime_replay(
    result: dict[str, Any],
    snapshots: list[dict[str, Any]],
    artifact_path: Path | None,
) -> dict[str, Any]:
    if artifact_path is None:
        return {"status": "not_run", "reason": "artifact path not supplied"}
    policy = DeploymentMPSPolicy(artifact_path)
    valid_symbols = sorted(
        {
            symbol
            for snapshot in snapshots
            for symbol in (snapshot.get("top_signals") or {})
        }
    )
    ordered = sorted(snapshots, key=lambda snapshot: str(snapshot.get("timestamp")))
    replayed = [policy.decide(snapshot, valid_symbols) for snapshot in ordered]
    recorded = [
        decision.get("actions_submitted") or []
        for decision in (result.get("decisions") or [])
    ]
    mismatch_indices = [
        index
        for index, (left, right) in enumerate(zip(replayed, recorded))
        if left != right
    ]
    if len(replayed) != len(recorded):
        mismatch_indices.extend(
            range(min(len(replayed), len(recorded)), max(len(replayed), len(recorded)))
        )
    return {
        "status": "complete",
        "snapshot_count": len(ordered),
        "recorded_decision_count": len(recorded),
        "replayed_decision_count": len(replayed),
        "valid_symbol_count": len(valid_symbols),
        "exact_action_batch_count": len(replayed) - len(set(mismatch_indices)),
        "mismatch_count": len(set(mismatch_indices)),
        "first_mismatch_indices": sorted(set(mismatch_indices))[:10],
        "all_action_batches_exact": not mismatch_indices
        and len(replayed) == len(recorded),
    }


def _replication_comparison(results: list[dict[str, Any]]) -> dict[str, Any]:
    signatures = [_result_signature(result) for result in results]
    first = signatures[0]
    return {
        "run_ids": [str(result.get("run", {}).get("run_id")) for result in results],
        "run_count": len(results),
        "metrics_exact": all(signature["metrics"] == first["metrics"] for signature in signatures),
        "equity_curve_exact": all(
            signature["equity_curve"] == first["equity_curve"]
            for signature in signatures
        ),
        "trade_ledger_exact": all(
            signature["trades"] == first["trades"] for signature in signatures
        ),
        "decision_ledger_exact": all(
            signature["decisions"] == first["decisions"]
            for signature in signatures
        ),
        "reference_metrics": first["metrics"],
        "exact_replication": all(signature == first for signature in signatures),
    }


def _cost_sensitivity(result: dict[str, Any]) -> list[dict[str, Any]]:
    initial = float(result["run"]["initial_equity"])
    gross_final = float(result["run"]["final_equity"])
    gross_return_pct = 100.0 * (gross_final / initial - 1.0)
    notional = sum(float(trade.get("value") or 0.0) for trade in result.get("trades") or [])
    rows = []
    for basis_points in (0, 10, 25, 50, 100):
        cost = notional * basis_points / 10_000.0
        final_equity = gross_final - cost
        rows.append(
            {
                "basis_points_per_traded_notional": basis_points,
                "traded_notional": notional,
                "estimated_cost": cost,
                "estimated_final_equity": final_equity,
                "estimated_return_pct": 100.0 * (final_equity / initial - 1.0),
            }
        )
    return rows + [
        {
            "basis_points_per_traded_notional": "break_even",
            "traded_notional": notional,
            "estimated_cost": gross_final - initial,
            "estimated_final_equity": initial,
            "estimated_return_pct": 0.0,
            "break_even_basis_points": (gross_final - initial) / notional * 10_000.0
            if notional
            else None,
            "gross_return_pct": gross_return_pct,
        }
    ]


def _weekly_decomposition(result: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for point in result.get("equity_curve") or []:
        timestamp = datetime.fromisoformat(str(point["timestamp"]))
        iso_year, iso_week, _ = timestamp.isocalendar()
        grouped[f"{iso_year}-W{iso_week:02d}"].append(point)
    trade_counts = Counter()
    for trade in result.get("trades") or []:
        timestamp = datetime.fromisoformat(str(trade["timestamp"]))
        iso_year, iso_week, _ = timestamp.isocalendar()
        trade_counts[f"{iso_year}-W{iso_week:02d}"] += 1
    prior_equity = float(result["run"]["initial_equity"])
    initial_equity = prior_equity
    rows = []
    for week in sorted(grouped):
        points = sorted(grouped[week], key=lambda point: str(point["timestamp"]))
        ending_equity = float(points[-1]["equity"])
        rows.append(
            {
                "week": week,
                "start_reference_equity": prior_equity,
                "ending_equity": ending_equity,
                "dollar_change": ending_equity - prior_equity,
                "return_pct_from_prior_week_end": 100.0
                * (ending_equity / prior_equity - 1.0),
                "contribution_pct_initial": 100.0
                * (ending_equity - prior_equity)
                / initial_equity,
                "trade_count": trade_counts[week],
            }
        )
        prior_equity = ending_equity
    return rows


def run_replication_audit(
    same_window_results: list[dict[str, Any]],
    original_snapshots: list[list[dict[str, Any]]],
    hosted_snapshots: list[dict[str, Any]],
    extension_results: list[dict[str, Any]],
    extension_snapshots: list[dict[str, Any]],
    output_dir: Path,
    *,
    artifact_path: Path | None = None,
    hosted_context_benchmark: dict[str, Any] | None = None,
    extension_djia_return_pct: float,
    extension_buyhold_return_pct: float,
) -> dict[str, Any]:
    collected = _deduplicate_snapshots(original_snapshots)
    hosted = _deduplicate_snapshots([hosted_snapshots])
    extension = _deduplicate_snapshots([extension_snapshots])
    exact = _replication_comparison(same_window_results)
    extension_comparison = _replication_comparison(extension_results)
    costs = _cost_sensitivity(same_window_results[0])
    weeks = _weekly_decomposition(same_window_results[0])
    source_comparison = _compare_snapshot_sources(collected, hosted)
    extension_metrics = extension_comparison["reference_metrics"]
    result = {
        "schema_version": 1,
        "artifact": "atl_replication_audit",
        "protocol_commit": "673b4c7064c1725cbdfb79848c7eaef5fa209273",
        "snapshot_capture_fix_commit": "97d9a33ea3d1b1b4c1e35d662f085daf7028d9f9",
        "policy_changed_after_original_freeze": False,
        "same_window_replication": exact,
        "same_window_snapshot_profile": _snapshot_profile(hosted),
        "same_window_runtime_replay": _runtime_replay(
            same_window_results[0], hosted, artifact_path
        ),
        "collection_context_comparison": source_comparison,
        "temporal_extension": {
            **extension_comparison,
            "classification": "flat"
            if float(extension_metrics["total_return"] or 0.0) == 0.0
            else (
                "positive"
                if float(extension_metrics["total_return"] or 0.0) > 0.0
                else "negative"
            ),
            "snapshot_profile": _snapshot_profile(extension),
            "runtime_replay": _runtime_replay(
                extension_results[0], extension, artifact_path
            ),
            "atl_djia_return_pct": extension_djia_return_pct,
            "atl_buyhold_return_pct": extension_buyhold_return_pct,
        },
        "cost_sensitivity": costs,
        "weekly_decomposition": weeks,
        "hosted_context_benchmark": {
            "status": "complete" if hosted_context_benchmark else "not_run",
            "split": (hosted_context_benchmark or {}).get("split"),
            "prediction_metrics": (hosted_context_benchmark or {}).get(
                "prediction_metrics"
            ),
            "systems": (hosted_context_benchmark or {}).get("systems"),
        },
        "overall_assessment": "share_with_caveats",
        "supported_conclusion": (
            "The hosted same-window result is exactly deterministic across complete reruns. "
            "The immediately following four-day window is flat because the frozen trend gate "
            "keeps the fresh policy in cash. Separately collected ATL indicator payloads differ "
            "from hosted-run payloads at the same timestamps, so the earlier offline replay is "
            "not execution-faithful. The corrected-context controls reproduce a positive "
            "combined-system result, but the matched ANN system produces the same portfolio "
            "outcome, so incremental MPS value is not established."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "replication_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8", newline="\n"
    )
    for filename, rows in (
        ("cost_sensitivity.csv", costs),
        ("weekly_decomposition.csv", weeks),
        ("feature_consistency.csv", source_comparison["model_feature_consistency"]),
    ):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            fields = sorted({key for row in rows for key in row})
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return result
