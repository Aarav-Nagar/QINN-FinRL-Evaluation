"""Command line workflow for collecting, training, registering, and running."""

from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

from .client import ATLClient
from .policy import MPSPolicy
from .training import train_from_snapshots


def hold_strategy(snapshot: dict, valid_symbols: list[str]) -> list[dict]:
    symbol = "AAPL" if "AAPL" in valid_symbols else valid_symbols[0]
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
    train.add_argument("--snapshots", type=Path, required=True)
    train.add_argument("--artifact", type=Path, required=True)
    register = sub.add_parser("register")
    register.add_argument("--credentials", type=Path, required=True)
    register.add_argument("--source-repo", required=True)
    run = sub.add_parser("run")
    run.add_argument("--start", required=True)
    run.add_argument("--end", required=True)
    run.add_argument("--artifact", type=Path, required=True)
    run.add_argument("--credentials", type=Path, required=True)
    run.add_argument("--result", type=Path, required=True)
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
        snapshots = json.loads(args.snapshots.read_text(encoding="utf-8"))
        summary = train_from_snapshots(snapshots, args.artifact)
        print(json.dumps(summary, indent=2))
    elif args.command == "register":
        owner_session = "aarav-mps-agent-" + secrets.token_hex(12)
        client = ATLClient(session_id=owner_session)
        response = client.register_agent(
            name="Aarav MPS Signal Agent",
            description=(
                "Research prototype using a classical bond-dimension-4 MPS trained on "
                "earlier ATL hourly snapshots; risk-bounded historical simulation only."
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
            agent_name="Aarav MPS Signal Agent",
            model_name="classical-mps-bond-4",
            strategy=policy.decide,
        )
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"run": result["run"], "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
