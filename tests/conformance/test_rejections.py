# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Allternit LLC
"""Specs the Rust contract refuses (studio-spec tests/contract.rs) are refused the
same way here: HTTP 422 and the same stable error code."""

from __future__ import annotations

import pytest

from .conftest import Client, fixture, spec_for

EXPECTED = {
    "invalid-rented": "rented_not_available",
    "invalid-publish": "publish_not_available",
    "invalid-engine": "unknown_engine",
    "invalid-method": "unknown_method",
    "invalid-dataset": "invalid_dataset",
}


def test_base_needs_exactly_one_of_repo_or_path(api: Client, engine: str):
    both = spec_for(engine)
    both["base"]["path"] = "/models/x"
    neither = spec_for(engine)
    neither["base"] = {}
    for spec in (both, neither):
        status, body = api.post("/v1/jobs", spec)
        assert status == 422 and body["error"] == "invalid_base"
        assert "base needs exactly one of repo or path" in body["message"]


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_rejection_fixture_gets_the_contract_code(api: Client, name: str):
    status, body = api.post("/v1/jobs", fixture(name))
    assert status == 422
    assert body["error"] == EXPECTED[name] and body["message"]


def test_a_spec_for_another_engine_is_refused(api: Client, engine: str):
    # The lora-valid fixture names allternit-tune; an engine process must not run it.
    status, body = api.post("/v1/jobs", fixture("lora-valid"))
    assert status == 422 and body["error"] == "unknown_engine" and engine in body["message"]


def test_wrong_spec_version_unknown_field_and_bad_json(api: Client, engine: str):
    assert api.post("/v1/jobs", spec_for(engine, spec_version=2))[1]["error"] == "unsupported_spec_version"
    spec = spec_for(engine)
    spec["surprise"] = 1
    assert api.post("/v1/jobs", spec)[1]["error"] == "invalid_spec"
    status, body = api.post("/v1/jobs", b"{not json")
    assert status == 422 and body["error"] == "invalid_spec"
    spec = spec_for(engine)
    del spec["method"]
    assert api.post("/v1/jobs", spec)[1]["error"] == "invalid_spec"


def test_param_and_output_refusals(api: Client, engine: str):
    assert api.post("/v1/jobs", spec_for(engine, params={"nope": 1}))[1]["error"] == "invalid_spec"
    assert api.post("/v1/jobs", spec_for(engine, params={"rank": 0}))[1]["error"] == "invalid_spec"
    assert api.post("/v1/jobs", spec_for(engine, outputs={"mlx": True}))[1]["error"] == "output_not_supported"
    assert api.post("/v1/jobs", spec_for(engine, resources__backend="mlx"))[1]["error"] == "backend_not_supported"
    assert api.post("/v1/jobs", spec_for(engine, resources__backend="cuda"))[1]["error"] == "invalid_spec"
    assert api.post("/v1/jobs", spec_for(engine, outputs={"gguf": ["q9"]}))[1]["error"] == "invalid_spec"
    assert api.post("/v1/jobs", spec_for(engine, dataset__format="dpo"))[1]["error"] == "invalid_dataset"


def test_a_refused_spec_does_not_block_the_next_job(api: Client, engine: str):
    assert api.post("/v1/jobs", fixture("invalid-rented"))[0] == 422
    status, job = api.post("/v1/jobs", spec_for(engine))
    assert status == 201
    api.wait(job["id"])
