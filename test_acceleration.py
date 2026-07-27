from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import torch


SCRIPT = Path(__file__).parent / "scripts" / "check_acceleration.py"
SPEC = importlib.util.spec_from_file_location("check_acceleration", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
check_acceleration = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = check_acceleration
SPEC.loader.exec_module(check_acceleration)


def test_nvidia_smi_parser(monkeypatch) -> None:
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="NVIDIA RTX Test, 8192, 600.00\n"
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: completed)
    result = check_acceleration.query_nvidia_smi()
    assert result == {
        "available": True,
        "name": "NVIDIA RTX Test",
        "memory_mib": 8192,
        "driver_version": "600.00",
    }


def test_report_distinguishes_visible_gpu_from_cpu_torch(monkeypatch) -> None:
    monkeypatch.setattr(
        check_acceleration,
        "query_nvidia_smi",
        lambda: {"available": True, "name": "GPU"},
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    report = check_acceleration.acceleration_report()
    assert report["status"] == "gpu_visible_but_torch_cpu_only"
    assert report["usable_encoder_device"] == "cpu"
