#!/usr/bin/env python
"""Record whether the local experiment environment can use NVIDIA acceleration."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch


def query_nvidia_smi() -> dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        return {"available": False, "error": type(error).__name__}
    first_line = result.stdout.strip().splitlines()[0]
    fields = [field.strip() for field in first_line.split(",")]
    if len(fields) != 3:
        return {"available": False, "error": "unexpected_output"}
    return {
        "available": True,
        "name": fields[0],
        "memory_mib": int(fields[1]),
        "driver_version": fields[2],
    }


def acceleration_report() -> dict[str, object]:
    gpu = query_nvidia_smi()
    cuda_available = torch.cuda.is_available()
    report: dict[str, object] = {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "nvidia_smi": gpu,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torch_cuda_available": cuda_available,
        "torch_device_count": torch.cuda.device_count(),
        "usable_encoder_device": "cuda" if cuda_available else "cpu",
    }
    if cuda_available:
        report["torch_device_name"] = torch.cuda.get_device_name(0)
        report["status"] = "cuda_ready"
    elif gpu.get("available"):
        report["status"] = "gpu_visible_but_torch_cpu_only"
    else:
        report["status"] = "no_nvidia_gpu_visible"
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = acceleration_report()
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
