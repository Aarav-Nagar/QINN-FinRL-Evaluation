"""Small dependency-free client for Agentic Trading Lab's external-agent API."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class ATLClient:
    def __init__(
        self,
        *,
        base_url: str = "https://agentictrading.onrender.com",
        session_id: str | None = None,
        api_key: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id
        self.api_key = api_key
        if api_key and not session_id:
            resolved = self._request("GET", "/api/v1/agents/resolve", api_key=api_key)
            self.session_id = resolved["session_id"]
        if not self.session_id:
            raise ValueError("session_id or api_key is required")

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        api_key: str | None = None,
        timeout: int = 60,
    ) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        if self.session_id:
            headers["X-Session-Id"] = self.session_id
        if api_key:
            headers["X-API-Key"] = api_key
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode("utf-8")
        request = Request(self.base_url + path, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ATL HTTP {exc.code} for {path}: {detail}") from exc

    def register_agent(
        self,
        *,
        name: str,
        description: str,
        source_repo: str,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/agents",
            {
                "name": name,
                "model_name": "classical-mps-bond-4",
                "agent_type": "external",
                "description": description,
                "category": "us_stocks",
                "runtime_type": "pipeline",
                "runtime_config": {
                    "source_repo": source_repo,
                    "method": "ATL-native 13-feature classical MPS signal",
                    "quantum_hardware": False,
                },
                "cash_allocation": 1000.0,
                "backtest_allocation": 1000.0,
            },
        )

    def start_backtest(
        self, start_date: str, end_date: str, *, agent_name: str, model_name: str
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/backtest/start",
            {
                "start_date": start_date,
                "end_date": end_date,
                "agent_name": agent_name,
                "model_name": model_name,
                "mode": "safe_trading",
            },
        )

    def run_loop(
        self,
        start_date: str,
        end_date: str,
        *,
        agent_name: str,
        model_name: str,
        strategy: Callable[[dict[str, Any], list[str]], list[dict[str, Any]]],
        snapshots_path: Path | None = None,
    ) -> dict[str, Any]:
        started = self.start_backtest(
            start_date, end_date, agent_name=agent_name, model_name=model_name
        )
        backtest_id = started["backtest_id"]
        snapshots: list[dict[str, Any]] = []
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            step = self._request(
                "GET", f"/api/v1/backtest/{backtest_id}/steps/current"
            )
            status = step.get("status")
            if status == "waiting_decision":
                snapshot = dict(step.get("market_snapshot") or {})
                snapshot["timestamp"] = step.get("timestamp")
                snapshots.append(snapshot)
                actions = strategy(snapshot, list(step.get("valid_symbols") or []))
                self._request(
                    "POST",
                    f"/api/v1/backtest/{backtest_id}/steps/current/decisions",
                    {"actions": actions},
                )
            elif status == "completed":
                if snapshots_path:
                    snapshots_path.parent.mkdir(parents=True, exist_ok=True)
                    snapshots_path.write_text(
                        json.dumps(snapshots, indent=2), encoding="utf-8"
                    )
                run_id = step["run_id"]
                return self._request(
                    "GET", f"/api/v1/backtest/runs/{run_id}/result"
                )
            elif status == "failed":
                raise RuntimeError(step.get("error") or "ATL backtest failed")
            time.sleep(0.5)
        raise TimeoutError(f"ATL backtest {backtest_id} did not finish in 10 minutes")
