# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Allternit LLC
"""Engine API v1 conformance: every adapter must pass this whole file."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from studio_engine_adapters import load
from studio_engine_kit.fit import fit
from studio_engine_kit.hardware import HARDWARE_ENV, device_unavailable
from studio_engine_kit.spec import load_catalog

from .conftest import HW_OK, Client, local_base, spec_for, start

LLAMA = "meta-llama/Llama-3.2-1B-Instruct"   # every adapter maps this base, for every method
ORDER = ["queued", "preparing", "running", "exporting", "succeeded"]


def test_every_route_needs_the_token(api: Client):
    for method, path in [("GET", "/v1/engine"), ("POST", "/v1/jobs"), ("GET", "/v1/jobs/job_0001"),
                         ("GET", "/v1/jobs/job_0001/events"), ("POST", "/v1/jobs/job_0001/cancel")]:
        for token in (None, "wrong"):
            status, body = api.call(method, path, {} if method == "POST" else None, token=token)
            assert status == 401 and body["error"] == "unauthorized", (method, path, token)


def test_engine_card_matches_the_registry(api: Client, engine: str):
    status, card = api.get("/v1/engine")
    entry = load_catalog()[engine]
    assert status == 200
    assert card["id"] == engine and card["install_state"] == "installed"
    assert card["process"] == "separate_process" and card["published"] is entry["published"] is True
    assert card["license"] == entry["license"] and card["hardware"] == entry["hardware"]
    assert card["version"] == load(engine).pin == entry["version"]
    assert card["methods"] == entry["methods"] and card.get("notes") == entry.get("notes")
    assert card["fit"] == fit(entry, HW_OK) and card["fit"]["ok"] is True and card["fit"]["hardware"] == HW_OK


def test_fixture_job_runs_to_succeeded_with_ordered_events(api: Client, engine: str):
    status, job = api.post("/v1/jobs", spec_for(engine))
    assert status == 201 and job["state"] == "queued" and job["last_seq"] == 1
    done = api.wait(job["id"])
    assert done["state"] == "succeeded", done["error"]
    _, page = api.get(f"/v1/jobs/{job['id']}/events")
    events = page["events"]
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    states = [e["message"] for e in events if e["kind"] == "state"]
    assert states == ORDER
    assert events[0]["message"] == "queued" and events[-1]["progress"] == 1.0
    assert any(e["kind"] == "progress" and 0 < e["progress"] <= 1 for e in events)
    assert page["state"] == "succeeded" and done["last_seq"] == events[-1]["seq"]


def test_events_after_returns_only_newer_events(api: Client, engine: str):
    _, job = api.post("/v1/jobs", spec_for(engine))
    api.wait(job["id"])
    _, page = api.get(f"/v1/jobs/{job['id']}/events")
    mid = page["events"][2]["seq"]
    _, tail = api.get(f"/v1/jobs/{job['id']}/events?after={mid}")
    assert [e["seq"] for e in tail["events"]] == [e["seq"] for e in page["events"] if e["seq"] > mid]
    _, none = api.get(f"/v1/jobs/{job['id']}/events?after={page['events'][-1]['seq']}")
    assert none["events"] == []
    assert api.get(f"/v1/jobs/{job['id']}/events?after=x")[0] == 400


def test_outputs_are_registered_with_real_paths(api: Client, engine: str):
    _, job = api.post("/v1/jobs", spec_for(engine))
    done = api.wait(job["id"])
    kinds = [o["kind"] for o in done["outputs"]]
    assert "adapter" in kinds and "gguf" in kinds        # the fixture asks for adapter + q4_k_m
    for out in done["outputs"]:
        assert Path(out["path"]).exists() and out["bytes"] > 0
    assert any(e["kind"] == "output" for e in api.get(f"/v1/jobs/{job['id']}/events")[1]["events"])


def test_config_and_command_are_written_for_the_job(api: Client, engine: str, tmp_path):
    _, job = api.post("/v1/jobs", spec_for(engine))
    done = api.wait(job["id"])
    folder = next(p for p in Path(done["outputs"][0]["path"]).parents if p.name == job["id"])
    command = json.loads((folder / "command.json").read_text())
    assert command[0]["name"] == "train" and command[0]["argv"]
    assert (folder / "spec.json").exists() and any((folder / "config").iterdir())


@pytest.mark.parametrize("method", ["lora", "qlora", "full", "dpo"])
def test_each_registry_method_is_translated_or_refused_plainly(api: Client, engine: str, method: str):
    over = {"method": method, "base__repo": LLAMA}
    if method == "dpo":
        over["dataset__format"] = "dpo"
    status, body = api.post("/v1/jobs", spec_for(engine, **over))
    if method not in load_catalog()[engine]["methods"]:
        assert status == 422 and body["error"] == "unknown_method" and body["message"]
        return
    assert status == 201, body
    done = api.wait(body["id"])
    assert done["state"] == "succeeded", done["error"]
    kinds = [o["kind"] for o in done["outputs"]]
    assert ("merged" in kinds) if method == "full" else ("adapter" in kinds)


def test_a_second_job_while_one_runs_is_409_then_cancel_works(engine: str, tmp_path, monkeypatch):
    monkeypatch.setenv("STUB_DELAY", "0.5")
    monkeypatch.setenv("STUB_STEPS", "40")
    client, manager, server = start(engine, tmp_path)
    try:
        _, job = client.post("/v1/jobs", spec_for(engine))
        client.wait(job["id"], states={"running"})
        status, body = client.post("/v1/jobs", spec_for(engine))
        assert status == 409 and body["error"] == "job_running"
        status, cancelled = client.post(f"/v1/jobs/{job['id']}/cancel")
        assert status == 200 and cancelled["state"] == "cancelled"
        status, body = client.post(f"/v1/jobs/{job['id']}/cancel")
        assert status == 409 and body["error"] == "already_finished"
        assert manager.get(job["id"]).state == "cancelled"
        deadline = __import__("time").time() + 5          # the trainer process is gone, a new job may start
        while manager.get(job["id"]).proc is not None and __import__("time").time() < deadline:
            __import__("time").sleep(0.05)
        assert manager.get(job["id"]).proc is None
        assert client.post("/v1/jobs", spec_for(engine))[0] == 201
    finally:
        server.shutdown()
        server.server_close()
        for j in list(manager.jobs.values()):
            try:
                manager.cancel(j.id)
            except Exception:  # noqa: BLE001
                pass


def test_a_failing_trainer_ends_failed_with_the_reason(engine: str, tmp_path, monkeypatch):
    monkeypatch.setenv("STUB_FAIL", "train")
    client, _, server = start(engine, tmp_path)
    try:
        _, job = client.post("/v1/jobs", spec_for(engine))
        done = client.wait(job["id"])
        assert done["state"] == "failed" and "code 3" in done["error"] and "stub failure" in done["error"]
        assert client.get(f"/v1/jobs/{job['id']}/events")[1]["events"][-1]["message"] == "failed"
        assert client.post(f"/v1/jobs/{job['id']}/cancel")[0] == 409
    finally:
        server.shutdown()
        server.server_close()


def test_unknown_job_is_404(api: Client):
    assert api.get("/v1/jobs/job_9999")[0] == 404
    assert api.get("/v1/jobs/job_9999/events")[0] == 404
    assert api.post("/v1/jobs/job_9999/cancel")[0] == 404


# ---- real mode (no dry-run): device check, base.path, a missing engine -------------

def test_an_unusable_device_is_503_hardware_unavailable(engine: str, tmp_path):
    why = "the host reported an NVIDIA GPU, but nvidia-smi lists none (driver not loaded or device lost)."
    client, _, server = start(engine, tmp_path, dry_run=False, device_check=lambda hw: why)
    try:
        status, body = client.post("/v1/jobs", spec_for(engine))
        assert status == 503 and body == {"error": "hardware_unavailable", "message": why}
        assert client.get("/v1/engine")[1]["limits"]["unavailable_here"] == why
    finally:
        server.shutdown()
        server.server_close()


def test_without_a_hardware_id_from_the_host_jobs_are_refused(engine: str, tmp_path):
    client, _, server = start(engine, tmp_path, hardware=None, device_check=device_unavailable)
    try:
        status, body = client.post("/v1/jobs", spec_for(engine))
        assert status == 503 and body["error"] == "hardware_unavailable" and HARDWARE_ENV in body["message"]
        assert "fit" not in client.get("/v1/engine")[1]
    finally:
        server.shutdown()
        server.server_close()


def test_device_check(monkeypatch):
    assert HARDWARE_ENV in device_unavailable(None)
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr(os.path, "exists", lambda p: False)
    assert "NVIDIA driver is not loaded" in device_unavailable("cuda-linux")
    assert "NVIDIA" in device_unavailable("cuda-windows")
    for hw in ("apple-silicon", "rocm-linux", "xpu-linux", "xpu-windows", "cpu"):
        assert device_unavailable(hw) is None


def test_real_mode_wants_base_path_and_reports_a_missing_engine(engine: str, tmp_path):
    base = tmp_path / "Llama-3.2-1B-Instruct"
    base.mkdir()
    (base / "config.json").write_text("{}")
    data = tmp_path / "train.jsonl"
    data.write_text('{"instruction":"a","input":"","output":"b"}\n')
    client, _, server = start(engine, tmp_path, dry_run=False)
    try:
        status, body = client.post("/v1/jobs", spec_for(engine, base__repo=LLAMA, dataset__path=str(data),
                                                        outputs__gguf=[]))
        assert status == 422 and body["error"] == "invalid_base" and "base.path" in body["message"]
        missing = local_base(spec_for(engine, dataset__path=str(data), outputs__gguf=[]), tmp_path / "nope")
        status, body = client.post("/v1/jobs", missing)
        assert status == 422 and body["error"] == "invalid_base" and "not found" in body["message"]
        relative = local_base(spec_for(engine, dataset__path=str(data), outputs__gguf=[]), "models/x")
        assert client.post("/v1/jobs", relative)[1]["error"] == "invalid_base"
        _, job = client.post("/v1/jobs", local_base(spec_for(engine, dataset__path=str(data), outputs__gguf=[]),
                                                    base))
        done = client.wait(job["id"])
        assert done["state"] == "failed" and "cannot start" in done["error"]
    finally:
        server.shutdown()
        server.server_close()
