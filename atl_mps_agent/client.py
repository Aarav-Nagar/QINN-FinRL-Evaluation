"""Small dependency-free client for Agentic Trading Lab's external-agent API."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
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
        attempts = 3 if method == "GET" else 1
        for attempt in range(attempts):
            try:
                with urlopen(request, timeout=timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if method == "GET" and exc.code in {429, 500, 502, 503, 504} and attempt + 1 < attempts:
                    time.sleep(0.5 * 2**attempt)
                    continue
                raise RuntimeError(f"ATL HTTP {exc.code} for {path}: {detail}") from exc
            except URLError as exc:
                if method == "GET" and attempt + 1 < attempts:
                    time.sleep(0.5 * 2**attempt)
                    continue
                raise RuntimeError(f"ATL network error for {path}: {exc}") from exc
        raise RuntimeError(f"ATL request attempts exhausted for {path}")

    @staticmethod
    def _write_json_atomic(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)

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
                "model_name": "classical-mps-bond-4-v2",
                "agent_type": "external",
                "description": description,
                "category": "us_stocks",
                "runtime_type": "pipeline",
                "runtime_config": {
                    "source_repo": source_repo,
                    "method": "ATL-native market-only 13-feature classical MPS v2 signal",
                    "quantum_hardware": False,
                    "cost_aware": True,
                    "feature_contract_version": 2,
                },
                "cash_allocation": 1000.0,
                "backtest_allocation": 1000.0,
            },
        )

    def update_agent(
        self,
        agent_id: str,
        *,
        name: str,
        model_name: str,
        description: str,
        runtime_config: dict[str, Any],
    ) -> dict[str, Any]:
        return self._request(
            "PATCH",
            f"/api/v1/agents/{agent_id}",
            {
                "name": name,
                "model_name": model_name,
                "description": description,
                "runtime_type": "pipeline",
                "runtime_config": runtime_config,
                "live_trading_enabled": False,
                "category": "us_stocks",
            },
            api_key=self.api_key,
        )

    def create_version(
        self,
        agent_id: str,
        *,
        version: str,
        code_commit: str,
        config: dict[str, Any],
        verification_level: str = "self_reported",
        architecture: str = "classical matrix product state",
        model_backbones: list[str] | None = None,
    ) -> dict[str, Any]:
        if model_backbones is None:
            model_backbones = ["bond-dimension-4 MPS", "13 market-only features"]
        return self._request(
            "POST",
            f"/api/v1/agents/{agent_id}/versions",
            {
                "version": version,
                "execution_mode": "external",
                "architecture": architecture,
                "model_backbones": model_backbones,
                "decision_frequency": "1h",
                "code_commit": code_commit,
                "config": config,
                "verification_level": verification_level,
            },
            api_key=self.api_key,
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
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            step = self._request(
                "GET", f"/api/v1/backtest/{backtest_id}/steps/current"
            )
            status = step.get("status")
            if status == "waiting_decision":
                snapshot = dict(step.get("market_snapshot") or {})
                snapshot["timestamp"] = step.get("timestamp")
                snapshots.append(snapshot)
                if snapshots_path:
                    self._write_json_atomic(snapshots_path, snapshots)
                actions = strategy(snapshot, list(step.get("valid_symbols") or []))
                self._request(
                    "POST",
                    f"/api/v1/backtest/{backtest_id}/steps/current/decisions",
                    {"actions": actions},
                )
            elif status == "completed":
                if snapshots_path:
                    self._write_json_atomic(snapshots_path, snapshots)
                run_id = step["run_id"]
                return self._request(
                    "GET", f"/api/v1/backtest/runs/{run_id}/result"
                )
            elif status == "failed":
                raise RuntimeError(step.get("error") or "ATL backtest failed")
            time.sleep(0.5)
        raise TimeoutError(f"ATL backtest {backtest_id} did not finish in 30 minutes")
