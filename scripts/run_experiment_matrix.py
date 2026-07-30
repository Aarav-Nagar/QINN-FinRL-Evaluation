#!/usr/bin/env python
"""Plan and execute reproducible QINN-FinRL experiment matrices."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class MatrixJob:
    phase: str
    window: str
    timesteps: int
    bond_dimension: int
    seeds: tuple[int, ...]
    encoder_epochs: int
    encoder_patience: int
    encoder_batch_size: int
    encoder_device: str

    @property
    def job_id(self) -> str:
        seed_token = "-".join(str(seed) for seed in self.seeds)
        window_token = "" if self.window == "primary" else f"_{self.window}"
        return (
            f"{self.phase}{window_token}_steps{self.timesteps}_bd{self.bond_dimension}"
            f"_seeds{seed_token}_epochs{self.encoder_epochs}"
            f"_batch{self.encoder_batch_size}_{self.encoder_device}"
        )


def build_jobs(
    phase: str,
    window: str,
    timesteps: list[int],
    bond_dimensions: list[int],
    seeds: tuple[int, ...],
    encoder_epochs: int,
    encoder_patience: int,
    encoder_batch_size: int,
    encoder_device: str,
) -> list[MatrixJob]:
    if not phase or any(character.isspace() for character in phase):
        raise ValueError("Phase must be a non-empty token without whitespace")
    if not timesteps or not bond_dimensions:
        raise ValueError("At least one timestep and bond dimension are required")
    if len(set(timesteps)) != len(timesteps):
        raise ValueError("Timesteps must be unique")
    if len(set(bond_dimensions)) != len(bond_dimensions):
        raise ValueError("Bond dimensions must be unique")
    jobs = [
        MatrixJob(
            phase=phase,
            window=window,
            timesteps=steps,
            bond_dimension=dimension,
            seeds=seeds,
            encoder_epochs=encoder_epochs,
            encoder_patience=encoder_patience,
            encoder_batch_size=encoder_batch_size,
            encoder_device=encoder_device,
        )
        for steps, dimension in itertools.product(timesteps, bond_dimensions)
    ]
    return sorted(jobs, key=lambda job: (job.timesteps, job.bond_dimension))


def command_for_job(
    job: MatrixJob,
    runner: Path,
    data_dir: Path,
    finrl_dir: Path,
    output_root: Path,
) -> list[str]:
    return [
        sys.executable,
        str(runner),
        "--data-dir",
        str(data_dir),
        "--finrl-dir",
        str(finrl_dir),
        "--output-dir",
        str(output_root / job.job_id),
        "--timesteps",
        str(job.timesteps),
        "--window",
        job.window,
        "--seeds",
        *(str(seed) for seed in job.seeds),
        "--bond-dimension",
        str(job.bond_dimension),
        "--encoder-epochs",
        str(job.encoder_epochs),
        "--encoder-patience",
        str(job.encoder_patience),
        "--encoder-batch-size",
        str(job.encoder_batch_size),
        "--encoder-device",
        job.encoder_device,
    ]


def write_plan(path: Path, jobs: list[MatrixJob]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(job) | {"job_id": job.job_id} for job in jobs], indent=2),
        encoding="utf-8",
    )


def job_state(job: MatrixJob, output_root: Path) -> str:
    """Classify an output directory without silently accepting stale results."""
    output_dir = output_root / job.job_id
    status_path = output_dir / "run_status.json"
    manifest_path = output_dir / "run_manifest.json"
    if not output_dir.exists():
        return "missing"
    if not status_path.exists():
        return "incomplete"
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    experiment = payload.get("experiment", {})
    expected: dict[str, object] = {
        "ppo_seeds": list(job.seeds),
        "ppo_timesteps": job.timesteps,
        "mps_bond_dimension": job.bond_dimension,
        "encoder_epochs": job.encoder_epochs,
        "encoder_patience": job.encoder_patience,
        "encoder_batch_size": job.encoder_batch_size,
        "encoder_device": job.encoder_device,
    }
    if job.window != "primary":
        expected["window_name"] = job.window
    if any(experiment.get(key) != value for key, value in expected.items()):
        return "stale"
    if payload.get("runtime", {}).get("status") != "completed":
        return "incomplete"
    if not manifest_path.exists():
        return "incomplete"
    return "completed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--window", choices=("primary", "shifted"), default="primary")
    parser.add_argument("--runner", type=Path, default=Path("run_experiment.py"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--finrl-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, nargs="+", required=True)
    parser.add_argument("--bond-dimensions", type=int, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--encoder-epochs", type=int, default=60)
    parser.add_argument("--encoder-patience", type=int, default=10)
    parser.add_argument("--encoder-batch-size", type=int, default=512)
    parser.add_argument(
        "--encoder-device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Run jobs even when matching completed artifacts already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = build_jobs(
        args.phase,
        args.window,
        args.timesteps,
        args.bond_dimensions,
        tuple(args.seeds),
        args.encoder_epochs,
        args.encoder_patience,
        args.encoder_batch_size,
        args.encoder_device,
    )
    write_plan(args.output_root / "matrix_plan.json", jobs)
    for job in jobs:
        state = job_state(job, args.output_root)
        if state == "stale":
            raise RuntimeError(
                f"Refusing to overwrite stale configuration: {job.job_id}"
            )
        if state == "completed" and not args.rerun_completed:
            print(f"[skipped: completed] {job.job_id}", flush=True)
            continue
        command = command_for_job(
            job, args.runner, args.data_dir, args.finrl_dir, args.output_root
        )
        print(f"[{state}] {job.job_id}", flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
