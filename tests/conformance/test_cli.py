# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Allternit LLC
"""The launch path the host uses: a process, a ready line, a token."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from studio_engine_adapters import ADAPTERS

from .conftest import Client, spec_for

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("engine", sorted(ADAPTERS))
def test_process_prints_ready_line_and_serves_a_job(engine: str, tmp_path):
    out_dir = tmp_path / "out"
    env = dict(os.environ, STUDIO_ENGINE_TOKEN="cli-token", PYTHONPATH=os.pathsep.join([str(ROOT / "kit"), str(ROOT / "adapters")]),
               ALLTERNIT_STUDIO_OUTPUT_DIR=str(out_dir), ALLTERNIT_STUDIO_HARDWARE="cuda-linux")
    proc = subprocess.Popen([sys.executable, "-m", "studio_engine_kit", "--engine", engine, "--dry-run"],
                            stdout=subprocess.PIPE, text=True, env=env)
    try:
        ready = json.loads(proc.stdout.readline())
        assert ready["ready"] is True and ready["engine"] == engine and ready["dry_run"] is True
        assert ready["hardware"] == "cuda-linux" and ready["output_dir"] == str(out_dir.resolve())
        client = Client(f"http://127.0.0.1:{ready['port']}", token="cli-token")
        assert client.get("/v1/engine")[1]["id"] == engine
        status, job = client.post("/v1/jobs", spec_for(engine))
        done = client.wait(job["id"])
        assert status == 201 and done["state"] == "succeeded" and done["outputs"]
        for out in done["outputs"]:      # absolute, inside the host's output dir
            path = Path(out["path"])
            assert path.is_absolute() and path.exists()
            assert path.is_relative_to(out_dir.resolve()) if hasattr(path, "is_relative_to") else \
                str(path).startswith(str(out_dir.resolve()) + os.sep)
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
