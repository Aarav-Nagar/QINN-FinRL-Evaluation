"""Command line workflow for collecting, training, registering, and running."""

from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

from .client import ATLClient
from .benchmark import run_benchmark
from .deployment_policy import DeploymentMPSPolicy
from .deployment_training import train_deployment_ensemble
from .policy import MPSPolicy
from .training import train_from_snapshots
from .v3_benchmark import run_v3_benchmark
from .v3_policy import ResidualMPSEnsemblePolicy
from .v3_training import train_v3_ensemble


def hold_strategy(snapshot: dict, valid_symbols: list[str]) -> list[dict]:
    symbol = "AAPL" if "AAPL" in valid_symbols else (valid_symbols[0] if valid_symbols else "AAPL")
    return [
        {
            "action": "hold",
            "symbol": symbol,
            "confidence": 0.5,
            "reasoning": "Historical feature collection only; no simulated trade requested",
            "position_size": 0,
            "stop_loss_price": None,
            "take_profit_price": None,
        }
    ]


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("--start", required=True)
    collect.add_argument("--end", required=True)
    collect.add_argument("--output", type=Path, required=True)
    train = sub.add_parser("train")
    train.add_argument("--snapshots", type=Path, nargs="+", required=True)
    train.add_argument("--artifact", type=Path, required=True)
    train.add_argument("--train-end")
    train.add_argument("--validation-end")
    train.add_argument("--seed", type=int, default=2026)
    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--snapshots", type=Path, nargs="+", required=True)
    benchmark.add_argument("--output-dir", type=Path, required=True)
    benchmark.add_argument("--train-end", required=True)
    benchmark.add_argument("--validation-end", required=True)
    benchmark.add_argument("--test-start", required=True)
    benchmark.add_argument("--test-end", required=True)
    train_v3 = sub.add_parser("train-v3")
    train_v3.add_argument("--snapshots", type=Path, nargs="+", required=True)
    train_v3.add_argument("--artifact", type=Path, required=True)
    train_v3.add_argument("--train-end", required=True)
    train_v3.add_argument("--validation-end", required=True)
    benchmark_v3 = sub.add_parser("benchmark-v3")
    benchmark_v3.add_argument("--snapshots", type=Path, nargs="+", required=True)
    benchmark_v3.add_argument("--output-dir", type=Path, required=True)
    benchmark_v3.add_argument("--train-end", required=True)
    benchmark_v3.add_argument("--validation-end", required=True)
    benchmark_v3.add_argument("--test-start", required=True)
    benchmark_v3.add_argument("--test-end", required=True)
    register = sub.add_parser("register")
    register.add_argument("--credentials", type=Path, required=True)
    register.add_argument("--source-repo", required=True)
    run = sub.add_parser("run")
    run.add_argument("--start", required=True)
    run.add_argument("--end", required=True)
    run.add_argument("--artifact", type=Path, required=True)
    run.add_argument("--credentials", type=Path, required=True)
    run.add_argument("--result", type=Path, required=True)
    run_v3 = sub.add_parser("run-v3")
    run_v3.add_argument("--start", required=True)
    run_v3.add_argument("--end", required=True)
    run_v3.add_argument("--artifact", type=Path, required=True)
    run_v3.add_argument("--credentials", type=Path, required=True)
    run_v3.add_argument("--result", type=Path, required=True)
    train_deployment = sub.add_parser("train-deployment")
    train_deployment.add_argument("--snapshots", type=Path, nargs="+", required=True)
    train_deployment.add_argument("--artifact", type=Path, required=True)
    train_deployment.add_argument("--train-end", required=True)
    train_deployment.add_argument("--validation-start", required=True)
    train_deployment.add_argument("--validation-end", required=True)
    run_deployment = sub.add_parser("run-deployment")
    run_deployment.add_argument("--start", required=True)
    run_deployment.add_argument("--end", required=True)
    run_deployment.add_argument("--artifact", type=Path, required=True)
    run_deployment.add_argument("--credentials", type=Path, required=True)
    run_deployment.add_argument("--result", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "collect":
        session = "mps-training-" + secrets.token_hex(12)
        client = ATLClient(session_id=session)
        result = client.run_loop(
            args.start,
            args.end,
            agent_name="Aarav MPS Training Data Collector",
            model_name="deterministic-hold",
            strategy=hold_strategy,
            snapshots_path=args.output,
        )
        print(json.dumps({"run": result["run"], "snapshots": str(args.output)}, indent=2))
    elif args.command == "train":
        snapshots = []
        for path in args.snapshots:
            snapshots.extend(json.loads(path.read_text(encoding="utf-8")))
        snapshots = list({str(item.get("timestamp")): item for item in snapshots}.values())
        summary = train_from_snapshots(
            snapshots,
            args.artifact,
            seed=args.seed,
            train_end=args.train_end,
            validation_end=args.validation_end,
        )
        print(json.dumps(summary, indent=2))
    elif args.command == "benchmark":
        snapshots = []
        for path in args.snapshots:
            snapshots.extend(json.loads(path.read_text(encoding="utf-8")))
        unique = {str(item.get("timestamp")): item for item in snapshots}
        summary = run_benchmark(
            list(unique.values()),
            args.output_dir,
            train_end=args.train_end,
            validation_end=args.validation_end,
            test_start=args.test_start,
            test_end=args.test_end,
        )
        print(json.dumps({"split": summary["split"], "paired_inference": summary["paired_inference"]}, indent=2))
    elif args.command in {"train-v3", "benchmark-v3"}:
        snapshots = []
        for path in args.snapshots:
            snapshots.extend(json.loads(path.read_text(encoding="utf-8")))
        snapshots = list({str(item.get("timestamp")): item for item in snapshots}.values())
        if args.command == "train-v3":
            summary = train_v3_ensemble(
                snapshots,
                args.artifact,
                train_end=args.train_end,
                validation_end=args.validation_end,
            )
            print(json.dumps(summary, indent=2))
        else:
            summary = run_v3_benchmark(
                snapshots,
                args.output_dir,
                train_end=args.train_end,
                validation_end=args.validation_end,
                test_start=args.test_start,
                test_end=args.test_end,
            )
            print(
                json.dumps(
                    {
                        "split": summary["split"],
                        "paired_member_inference": summary[
                            "paired_member_inference"
                        ],
                        "mps_ensemble": summary["mps_ensemble"],
                        "matched_ann_ensemble": summary[
                            "matched_ann_ensemble"
                        ],
                    },
                    indent=2,
                )
            )
    elif args.command == "train-deployment":
        snapshots = []
        for path in args.snapshots:
            snapshots.extend(json.loads(path.read_text(encoding="utf-8")))
        snapshots = list({str(item.get("timestamp")): item for item in snapshots}.values())
        summary = train_deployment_ensemble(
            snapshots,
            args.artifact,
            train_end=args.train_end,
            validation_start=args.validation_start,
            validation_end=args.validation_end,
        )
        print(json.dumps(summary, indent=2))
    elif args.command == "register":
        owner_session = "aarav-mps-agent-" + secrets.token_hex(12)
        client = ATLClient(session_id=owner_session)
        response = client.register_agent(
            name="Aarav MPS Signal Agent v2",
            description=(
                "Cost-aware classical bond-dimension-4 MPS with market-only features, "
                "validation-calibrated abstention, and leakage-safe ATL evaluation."
            ),
            source_repo=args.source_repo,
        )
        args.credentials.parent.mkdir(parents=True, exist_ok=True)
        args.credentials.write_text(
            json.dumps(
                {
                    "agent_id": response["agent"]["agent_id"],
                    "owner_session_id": owner_session,
                    "session_id": response["session_id"],
                    "api_key": response["api_key"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "agent_id": response["agent"]["agent_id"],
                    "name": response["agent"]["name"],
                    "credentials_saved": str(args.credentials),
                },
                indent=2,
            )
        )
    elif args.command == "run":
        credentials = json.loads(args.credentials.read_text(encoding="utf-8"))
        client = ATLClient(api_key=credentials["api_key"])
        policy = MPSPolicy(args.artifact)
        result = client.run_loop(
            args.start,
            args.end,
            agent_name="Aarav MPS Signal Agent v2",
            model_name="classical-mps-bond-4-v2",
            strategy=policy.decide,
        )
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"run": result["run"], "metrics": result["metrics"]}, indent=2))
    elif args.command == "run-v3":
        credentials = json.loads(args.credentials.read_text(encoding="utf-8"))
        client = ATLClient(api_key=credentials["api_key"])
        policy = ResidualMPSEnsemblePolicy(args.artifact)
        result = client.run_loop(
            args.start,
            args.end,
            agent_name="Aarav Residual MPS Ensemble",
            model_name="residual-mps-ensemble",
            strategy=policy.decide,
        )
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"run": result["run"], "metrics": result["metrics"]}, indent=2))
    elif args.command == "run-deployment":
        credentials = json.loads(args.credentials.read_text(encoding="utf-8"))
        client = ATLClient(api_key=credentials["api_key"])
        policy = DeploymentMPSPolicy(args.artifact)
        result = client.run_loop(
            args.start,
            args.end,
            agent_name="Aarav Residual MPS Ensemble",
            model_name="residual-mps-ensemble",
            strategy=policy.decide,
        )
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"run": result["run"], "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
