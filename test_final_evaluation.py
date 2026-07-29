from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).parent / "scripts" / "summarize_final_evaluation.py"
SPEC = importlib.util.spec_from_file_location("final_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
final_summary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = final_summary
SPEC.loader.exec_module(final_summary)


def make_run(
    path: Path,
    *,
    completed: bool = True,
    dimension: int = 2,
    seeds: list[int] | None = None,
) -> Path:
    path.mkdir()
    configured_seeds = list(range(10)) if seeds is None else seeds
    manifest = {
        "experiment": {
            "ppo_timesteps": 20_000,
            "mps_bond_dimension": dimension,
            "ppo_seeds": configured_seeds,
        },
        "runtime": {
            "status": "completed" if completed else "running",
            "git_commit": "abc123",
            "completed_at_utc": "2026-07-28T20:00:00+00:00",
        },
    }
    (path / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    rows = []
    for condition, offset in zip(
        final_summary.PPO_CONDITIONS, (0.0, 0.1, 0.2), strict=True
    ):
        for seed in configured_seeds:
            rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "annual_return": offset + seed / 100,
                    "sharpe": offset + seed / 50,
                    "max_drawdown": -0.2,
                    "annualized_turnover": 0.3,
                    "total_cost": 100.0,
                }
            )
    pd.DataFrame(rows).to_csv(path / "ppo_backtest_metrics.csv", index=False)
    (path / "equity_curves.csv").write_text("date,value\n", encoding="utf-8")
    return path


def test_load_requires_completed_prespecified_run(tmp_path: Path) -> None:
    run = make_run(tmp_path / "run", completed=False)
    with pytest.raises(ValueError, match="not completed"):
        final_summary.load_final_run(run)


def test_load_requires_selected_dimension(tmp_path: Path) -> None:
    run = make_run(tmp_path / "run", dimension=4)
    with pytest.raises(ValueError, match="mps_bond_dimension"):
        final_summary.load_final_run(run)


def test_load_requires_all_ten_seeds(tmp_path: Path) -> None:
    run = make_run(tmp_path / "run", seeds=list(range(9)))
    with pytest.raises(ValueError, match="ppo_seeds"):
        final_summary.load_final_run(run)


def test_summary_preserves_paired_seed_differences(tmp_path: Path) -> None:
    run = make_run(tmp_path / "run")
    _, metrics = final_summary.load_final_run(run)
    condition, paired, inference = final_summary.summarize(metrics)
    assert len(condition) == 3
    assert len(paired) == 10
    assert paired["mps_minus_ann_sharpe"].tolist() == pytest.approx([0.1] * 10)
    assert inference["mean_difference"] == pytest.approx(0.1)
    assert inference["positive_seeds"] == 10


def test_paired_bootstrap_is_deterministic() -> None:
    differences = pd.Series([-0.2, -0.1, 0.0, 0.1, 0.2])
    first = final_summary.paired_seed_bootstrap(differences)
    second = final_summary.paired_seed_bootstrap(differences)
    assert first == second
    assert first[0] < 0 < first[1]


def test_exact_sign_test_ignores_ties() -> None:
    assert final_summary.exact_two_sided_sign_p(
        pd.Series([1.0, 2.0, 0.0, -1.0])
    ) == pytest.approx(1.0)
    assert final_summary.exact_two_sided_sign_p(
        pd.Series([1.0] * 10)
    ) == pytest.approx(2 / 1024)


def test_writer_hashes_source_and_outputs(tmp_path: Path) -> None:
    run = make_run(tmp_path / "run")
    condition = tmp_path / "out" / "condition.csv"
    paired = tmp_path / "out" / "paired.csv"
    inference = tmp_path / "out" / "inference.json"
    manifest = tmp_path / "out" / "manifest.json"
    final_summary.write_artifacts(
        run, condition, paired, inference, manifest
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["artifact"] == "final_ten_seed_evaluation"
    assert payload["prespecified_configuration"]["ppo_seeds"] == list(range(10))
    assert set(payload["outputs"]) == {
        condition.name,
        paired.name,
        inference.name,
    }
