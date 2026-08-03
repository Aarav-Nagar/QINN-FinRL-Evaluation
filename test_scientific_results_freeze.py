from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).parent
SCRIPT = ROOT / "scripts" / "verify_scientific_results_freeze.py"
SPEC = importlib.util.spec_from_file_location("results_freeze", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
results_freeze = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = results_freeze
SPEC.loader.exec_module(results_freeze)


def test_scientific_results_freeze_matches_published_evidence() -> None:
    report = results_freeze.verify_freeze(ROOT)
    assert report == {
        "frozen_hashes": 38,
        "primary_ppo_endpoints": 30,
        "shifted_ppo_endpoints": 30,
        "capacity_dimensions": 3,
        "paper_claim_groups": 6,
    }
