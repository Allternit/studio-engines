# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Allternit LLC
"""Conformance against the platform's fixtures/fit-table.json.

1. The kit's port of fit.rs gives the table's answer for every engine
   (allternit-tune included) on every hardware id, reasons word for word.
2. Each adapter's engine process refuses or accepts every (hardware, method)
   pair exactly as the table says, with the contract codes, over HTTP.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from studio_engine_adapters import ADAPTERS
from studio_engine_kit.fit import check_job, fit
from studio_engine_kit.hardware import HARDWARE_IDS
from studio_engine_kit.spec import METHODS, load_catalog

from .conftest import spec_for, start

TABLE = json.loads((Path(__file__).parent / "fixtures" / "fit-table.json").read_text())
CATALOG = load_catalog()
LLAMA = "meta-llama/Llama-3.2-1B-Instruct"
CELLS = [(e, hw) for e in sorted(TABLE["engines"]) for hw in TABLE["hardware"]]


def test_the_table_covers_every_catalog_engine_and_hardware_id():
    assert tuple(TABLE["hardware"]) == HARDWARE_IDS
    assert set(TABLE["engines"]) == set(CATALOG)
    for engine, by_hw in TABLE["engines"].items():
        assert set(by_hw) == set(HARDWARE_IDS), engine


@pytest.mark.parametrize("engine,hw", CELLS)
def test_fit_matches_the_table(engine, hw):
    want = TABLE["engines"][engine][hw]
    card = fit(CATALOG[engine], hw)
    if isinstance(want, str):
        assert card["ok"] is False and card["reason"] == want
        return
    assert card["ok"] is True
    assert set(want["methods"]) == set(CATALOG[engine]["methods"])
    for method, verdict in want["methods"].items():
        assert card["methods_unavailable"].get(method, "ok") == verdict, method
        spec = spec_for(engine, method=method)
        backend, errs = check_job(spec, CATALOG[engine], hw)
        if verdict == "ok":
            assert errs == [] and backend == want["backend"][method], method
        else:
            assert [(e.code, e.message) for e in errs] == [("method_not_supported", verdict)]
    assert card["outputs_unavailable"].get("mlx", "ok") == want["outputs"]["mlx"]
    assert card.get("warning") == want["warning"]


def _job(engine, method, **over):
    over.setdefault("base__repo", LLAMA)      # every adapter maps it, for every method
    if method == "dpo":
        over.setdefault("dataset__format", "dpo")
    return spec_for(engine, method=method, **over)


@pytest.mark.parametrize("engine,hw", [(e, hw) for e in sorted(ADAPTERS) for hw in HARDWARE_IDS])
def test_engine_process_follows_the_table(engine, hw, tmp_path):
    want = TABLE["engines"][engine][hw]
    client, manager, server = start(engine, tmp_path, hardware=hw)
    try:
        card = client.get("/v1/engine")[1]
        assert card["fit"] == fit(CATALOG[engine], hw)
        for method in METHODS:
            status, body = client.post("/v1/jobs", _job(engine, method))
            if method not in CATALOG[engine]["methods"]:
                assert status == 422 and body["error"] == "unknown_method", (method, body)
            elif isinstance(want, str):
                assert (status, body) == (422, {"error": "hardware_not_supported", "message": want}), method
            elif want["methods"][method] != "ok":
                assert (status, body) == (422, {"error": "method_not_supported", "message": want["methods"][method]})
            else:
                assert status == 201, (method, body)
                done = client.wait(body["id"])
                assert done["state"] == "succeeded", (method, done["error"])
        if not isinstance(want, str):
            status, body = client.post("/v1/jobs", _job(engine, "lora", outputs__mlx=True))
            assert (status, body) == (422, {"error": "output_not_supported", "message": want["outputs"]["mlx"]})
            status, body = client.post("/v1/jobs", _job(engine, "lora", resources__backend="mlx"))
            assert status == 422 and body["error"] == "backend_not_supported"
    finally:
        server.shutdown()
        server.server_close()


def test_torchtune_config_names_the_device_for_the_hardware(tmp_path):
    for hw, device in (("apple-silicon", "mps"), ("cuda-linux", "cuda"), ("rocm-linux", "cuda"), ("xpu-linux", "xpu")):
        client, manager, server = start("torchtune", tmp_path / hw, hardware=hw)
        try:
            _, job = client.post("/v1/jobs", _job("torchtune", "lora"))
            client.wait(job["id"])
            config = (manager.data_dir / "jobs" / job["id"] / "config" / "torchtune.yaml").read_text()
            assert f"device: \"{device}\"" in config, hw
        finally:
            server.shutdown()
            server.server_close()
