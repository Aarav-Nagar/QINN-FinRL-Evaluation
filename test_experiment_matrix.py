from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parent / "scripts" / "run_experiment_matrix.py"
SPEC = importlib.util.spec_from_file_location("experiment_matrix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
experiment_matrix = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = experiment_matrix
SPEC.loader.exec_module(experiment_matrix)


def test_build_jobs_is_deterministic() -> None:
    jobs = experiment_matrix.build_jobs(
        "budget-pilot", [20_000, 5_000], [4], (0, 1, 2), 60, 10, 512, "cpu"
    )
    assert [job.timesteps for job in jobs] == [5_000, 20_000]
    assert jobs[0].job_id == "budget-pilot_steps5000_bd4_seeds0-1-2_epochs60"


def test_build_jobs_rejects_duplicate_axes() -> None:
    with pytest.raises(ValueError, match="Timesteps must be unique"):
        experiment_matrix.build_jobs(
            "pilot", [5_000, 5_000], [4], (0,), 1, 1, 512, "cpu"
        )
    with pytest.raises(ValueError, match="Bond dimensions must be unique"):
        experiment_matrix.build_jobs(
            "pilot", [5_000], [4, 4], (0,), 1, 1, 512, "cpu"
        )


def test_command_contains_full_job_configuration(tmp_path: Path) -> None:
    job = experiment_matrix.MatrixJob(
        "pilot", 5_000, 8, (0, 2), 3, 2, 256, "auto"
    )
    command = experiment_matrix.command_for_job(
        job,
        Path("run_experiment.py"),
        Path("data"),
        Path("FinRL"),
        tmp_path,
    )
    assert command[0] == sys.executable
    assert command[command.index("--seeds") + 1 : command.index("--bond-dimension")] == [
        "0",
        "2",
    ]
    assert command[command.index("--bond-dimension") + 1] == "8"
    assert command[command.index("--encoder-batch-size") + 1] == "256"


def test_write_plan_is_machine_readable(tmp_path: Path) -> None:
    jobs = experiment_matrix.build_jobs(
        "pilot", [512], [2, 4], (0,), 1, 1, 512, "cpu"
    )
    destination = tmp_path / "matrix_plan.json"
    experiment_matrix.write_plan(destination, jobs)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert [row["bond_dimension"] for row in payload] == [2, 4]
    assert all(row["job_id"].startswith("pilot_") for row in payload)
