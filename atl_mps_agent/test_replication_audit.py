from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from atl_mps_agent.replication_audit import run_replication_audit


def _result(run_id: str, final_equity: float, trades: list[dict]) -> dict:
    total_return = final_equity / 1000.0 - 1.0
    return {
        "run": {
            "run_id": run_id,
            "initial_equity": 1000.0,
            "final_equity": final_equity,
        },
        "metrics": {
            "total_return": total_return,
            "sharpe_ratio": 1.0 if total_return else 0.0,
            "max_drawdown": -0.01 if total_return else 0.0,
            "num_trades": len(trades),
            "final_equity": final_equity,
            "timeout_holds": 0,
        },
        "equity_curve": [
            {
                "timestamp": "2026-07-01T14:00:00+00:00",
                "equity": final_equity,
            }
        ],
        "trades": trades,
        "decisions": [{"timestamp": "2026-07-01T14:00:00+00:00", "action": "hold"}],
    }


def _snapshot(sma50: float) -> dict:
    return {
        "timestamp": "2026-07-01T14:00:00+00:00",
        "portfolio": {"cash": 1000.0, "total_equity": 1000.0},
        "current_holdings": {},
        "top_signals": {
            "AAA": {
                "price": 100.0,
                "rsi": 50.0,
                "macd": 0.1,
                "macd_signal": 0.05,
                "sma20": 99.0,
                "sma50": sma50,
                "bb_upper": 105.0,
                "bb_lower": 95.0,
            }
        },
    }


def test_replication_audit_detects_exact_runs_and_context_drift(tmp_path: Path):
    trade = {
        "timestamp": "2026-07-01T14:00:00+00:00",
        "symbol": "AAA",
        "quantity": 1,
        "side": "BUY",
        "price": 100.0,
        "value": 100.0,
    }
    primary = _result("one", 1010.0, [trade])
    replica = deepcopy(primary)
    replica["run"]["run_id"] = "two"
    flat = _result("extension-one", 1000.0, [])
    flat_replica = deepcopy(flat)
    flat_replica["run"]["run_id"] = "extension-two"
    result = run_replication_audit(
        [primary, replica],
        [[_snapshot(98.0)]],
        [_snapshot(97.0)],
        [flat, flat_replica],
        [_snapshot(97.0)],
        tmp_path,
        extension_djia_return_pct=-0.5,
        extension_buyhold_return_pct=0.0,
    )
    assert result["same_window_replication"]["exact_replication"] is True
    assert result["temporal_extension"]["classification"] == "flat"
    assert result["collection_context_comparison"]["raw_field_consistency"]["price"]["exact_match_fraction"] == 1.0
    assert result["collection_context_comparison"]["raw_field_consistency"]["sma50"]["exact_match_fraction"] == 0.0
    assert result["cost_sensitivity"][-1]["break_even_basis_points"] == 1000.0
    assert (tmp_path / "replication_audit.json").is_file()
    assert (tmp_path / "cost_sensitivity.csv").is_file()
