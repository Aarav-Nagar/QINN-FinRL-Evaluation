from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).parent
SCRIPT = ROOT / "run_experiment.py"
SPEC = importlib.util.spec_from_file_location("experiment_contract", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
experiment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = experiment
SPEC.loader.exec_module(experiment)

PRIMARY = ROOT / "results" / "final" / "run_manifest.json"
SHIFTED = (
    ROOT / "results" / "robustness" / "shifted" / "run_manifest.json"
)
CAPACITY = (
    ROOT / "results" / "pilots" / "2026-07-28" / "dimension_manifest.json"
)
CAPACITY_RAW = CAPACITY.parent / "raw"


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha256(path: Path) -> str:
    payload = path.read_bytes()
    if path.suffix.lower() in {".csv", ".json", ".md", ".py", ".txt"}:
        payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def test_frozen_manifests_match_implementation_contract() -> None:
    primary = load_json(PRIMARY)
    shifted = load_json(SHIFTED)
    expected_data_hashes = {
        name: record["sha256"]
        for name, record in experiment.DATA_FILES.items()
    }

    for manifest in (primary, shifted):
        assert manifest["tickers"] == experiment.TICKERS
        assert manifest["finrl_features"] == experiment.FINRL_FEATURES
        assert (
            manifest["representation_features"]
            == experiment.REPRESENTATION_FEATURES
        )
        assert manifest["dataset_sha256"] == expected_data_hashes
        assert manifest["finrl_commit"] == experiment.FINRL_COMMIT
        assert manifest["state_construction"]["base_dimension"] == 181
        assert manifest["state_construction"]["signal_dimension"] == 196
        assert manifest["ann_parameter_count"] == 369
        assert manifest["mps_parameter_count"] == 97

        config = manifest["experiment"]
        assert config["ppo_seeds"] == list(range(10))
        assert config["ppo_timesteps"] == 20_000
        assert config["transaction_cost"] == 0.001
        assert config["initial_amount"] == 1_000_000
        assert config["hmax"] == 100
        assert config["reward_scaling"] == 1e-4
        assert config["mps_bond_dimension"] == 2

    assert primary["experiment"] | {
        "representation_train_end": "2017-12-29",
        "representation_validation_start": "2018-01-01",
        "representation_validation_end": "2018-12-28",
        "train_start": "2013-01-02",
        "train_end": "2018-12-28",
        "test_start": "2019-01-02",
        "test_end": "2023-12-28",
    } == primary["experiment"]
    assert shifted["experiment"] | {
        "representation_train_end": "2015-12-31",
        "representation_validation_start": "2016-01-01",
        "representation_validation_end": "2016-12-30",
        "train_start": "2013-01-02",
        "train_end": "2016-12-30",
        "test_start": "2017-01-03",
        "test_end": "2018-12-28",
    } == shifted["experiment"]


def test_capacity_manifest_sources_are_packaged_and_hash_valid() -> None:
    manifest = load_json(CAPACITY)
    assert manifest["expected_dimensions"] == [2, 4, 8]
    assert manifest["selection"]["selected_bond_dimension"] == 2
    assert (
        manifest["selection"]["test_or_trading_metrics_used_for_selection"]
        is False
    )
    assert len(manifest["inputs"]) == 3

    for source in manifest["inputs"]:
        source_dir = CAPACITY_RAW / source["job_id"]
        for filename, expected in source["sha256"].items():
            path = source_dir / filename
            assert path.is_file()
            assert canonical_sha256(path) == expected


def test_public_docs_distinguish_current_and_historical_protocols() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    reproduction = (ROOT / "REPRODUCIBILITY.md").read_text(encoding="utf-8")
    docstring = experiment.__doc__ or ""

    assert "## Current SecureFinAI protocol" in readme
    assert "## Historical reference protocol" in readme
    assert "selected final ANN/MPS pair\nis not parameter matched" in readme
    assert "parameter count, PPO configuration" not in readme
    assert "Fit parameter-matched ANN" not in docstring
    assert "untouched 2019-2023 trading period" not in docstring
    assert "--output-root work\\" not in reproduction
    assert "--output-root local_runs\\" in reproduction
    assert "## Clean-checkout verification" in reproduction
