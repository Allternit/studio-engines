# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Allternit LLC
"""Shared fixtures: an Engine API v1 server for any adapter, in dry-run mode."""

from __future__ import annotations

import copy
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytest

from studio_engine_adapters import ADAPTERS, load
from studio_engine_kit.jobs import JobManager
from studio_engine_kit.server import serve

FIXTURES = Path(__file__).parent / "fixtures"
TOKEN = "test-token-123"
TERMINAL = {"succeeded", "failed", "cancelled"}
HW_OK = "cuda-linux"          # every adapter here runs on it


def no_device_problem(hardware):
    return None


def fixture(name: str) -> Dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def spec_for(engine: str, **over: Any) -> Dict[str, Any]:
    """The lora-valid fixture aimed at `engine`. Dotted keys set nested fields."""
    spec = fixture("lora-valid")
    spec["engine"] = engine
    for key, value in over.items():
        node = spec
        parts = key.split("__")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return spec


class Client:
    def __init__(self, base: str, token: Optional[str] = TOKEN) -> None:
        self.base, self.token = base, token

    def call(self, method: str, path: str, body: Any = None, token: Any = "default") -> Tuple[int, Any]:
        data = body if isinstance(body, (bytes, type(None))) else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method)
        tok = self.token if token == "default" else token
        if tok is not None:
            req.add_header("Authorization", f"Bearer {tok}")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return resp.status, json.loads(resp.read() or b"null")
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"null")

    def get(self, path: str, **kw: Any) -> Tuple[int, Any]:
        return self.call("GET", path, **kw)

    def post(self, path: str, body: Any = None, **kw: Any) -> Tuple[int, Any]:
        return self.call("POST", path, body, **kw)

    def wait(self, job_id: str, states=TERMINAL, timeout: float = 30) -> Dict[str, Any]:
        end = time.time() + timeout
        while time.time() < end:
            _, job = self.get(f"/v1/jobs/{job_id}")
            if job["state"] in states:
                return job
            time.sleep(0.05)
        raise AssertionError(f"job {job_id} did not reach {states}: {job['state']}")


def local_base(spec: Dict[str, Any], folder: Any) -> Dict[str, Any]:
    """Point the spec at a local model folder with base.path (what the host sends)."""
    spec["base"] = {"path": str(folder)}
    return spec


def start(engine: str, tmp_path: Path, dry_run: bool = True, hardware: Any = HW_OK,
          device_check=no_device_problem) -> Tuple[Client, JobManager, Any]:
    manager = JobManager(load(engine), tmp_path / "data", dry_run=dry_run,
                         hardware=hardware, kill_grace=2.0, device_check=device_check)
    server = serve(manager, TOKEN, 0)
    threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.05}, daemon=True).start()
    return Client(f"http://127.0.0.1:{server.server_address[1]}"), manager, server


@pytest.fixture(params=sorted(ADAPTERS))
def engine(request) -> str:
    return request.param


@pytest.fixture
def api(engine, tmp_path):
    client, manager, server = start(engine, tmp_path)
    yield client
    for job in list(manager.jobs.values()):
        try:
            manager.cancel(job.id)
        except Exception:  # noqa: BLE001
            pass
    server.shutdown()
    server.server_close()
