# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Allternit LLC
"""Engine API v1 over loopback HTTP (services/studio-spec/openapi/engine-api.yaml).

Binds 127.0.0.1 only; every request needs "Authorization: Bearer <token>".
Standard library only.
"""

from __future__ import annotations

import hmac
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from . import __version__
from .adapter import Adapter
from .fit import fit
from .jobs import AlreadyFinished, HardwareUnavailable, JobManager, JobRunning
from .spec import MAX_SEQ_LEN, SpecError, load_catalog

MAX_BODY = 1 << 20
_JOB_PATH = re.compile(r"^/v1/jobs/([A-Za-z0-9_-]+)(/events|/cancel)?$")


def engine_card(adapter: Adapter, manager: JobManager) -> Dict[str, Any]:
    """EngineInfo with `fit` for ALLTERNIT_STUDIO_HARDWARE (omitted when the host
    did not pass a hardware id; limits.unavailable_here then says so)."""
    entry = load_catalog()[adapter.id]
    card: Dict[str, Any] = {
        "id": adapter.id, "name": entry["name"], "license": entry["license"], "version": adapter.pin,
        "methods": adapter.methods(), "hardware": list(entry["hardware"]),
        "install_state": "installed",
        "limits": {
            "max_seq_len": MAX_SEQ_LEN, "dry_run": manager.dry_run, "kit_version": __version__,
            "unavailable_here": None if manager.dry_run and manager.hardware else manager.device_check(manager.hardware),
            "base_model": "base.path, a local folder; the host fetches and verifies weights",
        },
        "published": entry["published"], "process": entry["process"], "default": False,
        "backends": [dict(b) for b in entry.get("backends", [])],
    }
    if entry.get("notes"):
        card["notes"] = entry["notes"]
    if manager.hardware:
        card["fit"] = fit(entry, manager.hardware)
    return card


def make_handler(manager: JobManager, token: str):
    expected = f"Bearer {token}".encode()

    class Handler(BaseHTTPRequestHandler):
        server_version = f"studio-engine-kit/{__version__}"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            if os.environ.get("STUDIO_ENGINE_ACCESS_LOG") == "1":
                super().log_message(fmt, *args)

        def _send(self, status: int, body: Dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, status: int, code: str, message: str) -> None:
            self._send(status, {"error": code, "message": message})

        def _authorized(self) -> bool:
            if hmac.compare_digest((self.headers.get("Authorization") or "").encode(), expected):
                return True
            try:
                self._drain()
            except ValueError:
                self.close_connection = True
            self._error(401, "unauthorized", "missing or wrong bearer token")
            return False

        def _drain(self) -> bytes:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise ValueError("body too large")
            return self.rfile.read(length) if length > 0 else b""

        def _route(self) -> Tuple[str, Optional[str], Optional[str]]:
            path = urlparse(self.path).path.rstrip("/") or "/"
            m = _JOB_PATH.match(path)
            if m:
                return "job", m.group(1), (m.group(2) or "").lstrip("/") or None
            return path, None, None

        def do_GET(self) -> None:  # noqa: N802
            if not self._authorized():
                return
            route, job_id, sub = self._route()
            if route == "/v1/engine":
                return self._send(200, engine_card(manager.adapter, manager))
            if route == "job" and sub in (None, "events"):
                job = manager.get(job_id or "")
                if job is None:
                    return self._error(404, "not_found", f"no job {job_id}")
                if sub is None:
                    return self._send(200, job.to_json())
                query = parse_qs(urlparse(self.path).query)
                try:
                    after = int(query.get("after", ["0"])[0])
                except ValueError:
                    return self._error(400, "bad_request", "after must be an integer")
                return self._send(200, job.events_after(max(after, 0)))
            self._error(404, "not_found", f"no route GET {route}")

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorized():
                return
            route, job_id, sub = self._route()
            if route == "/v1/jobs":
                return self._start()
            if route == "job" and sub == "cancel":
                self._drain()
                if manager.get(job_id or "") is None:
                    return self._error(404, "not_found", f"no job {job_id}")
                try:
                    return self._send(200, manager.cancel(job_id or "").to_json())
                except AlreadyFinished as exc:
                    return self._error(409, "already_finished", str(exc))
            self._error(404, "not_found", f"no route POST {route}")

        def _start(self) -> None:
            try:
                raw = self._drain()
            except ValueError:
                self.close_connection = True
                return self._error(413, "too_large", f"job spec is over {MAX_BODY} bytes")
            try:
                spec = json.loads(raw.decode("utf-8") or "null")
            except (ValueError, UnicodeDecodeError) as exc:
                return self._error(422, "invalid_spec", f"job spec is not valid JSON: {exc}")
            try:
                job = manager.submit(spec)
            except SpecError as exc:
                return self._error(422, exc.code, exc.message)
            except HardwareUnavailable as exc:
                return self._error(503, "hardware_unavailable", str(exc))
            except JobRunning as exc:
                return self._error(409, "job_running", str(exc))
            self._send(201, job.snapshot)

    return Handler


def serve(manager: JobManager, token: str, port: int = 0) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(manager, token))
    server.daemon_threads = True
    return server
