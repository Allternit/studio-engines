# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Allternit LLC
"""Golden tests: the config and command each adapter builds for fixtures/lora-valid.json.

Regenerate after a deliberate change with:  UPDATE_GOLDEN=1 pytest tests/conformance/test_golden.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from studio_engine_adapters import ADAPTERS, load
from studio_engine_kit.spec import build_settings, contract_errors

from .conftest import spec_for

GOLDEN = Path(__file__).parent / "golden"


@pytest.fixture(autouse=True)
def fixed_home(monkeypatch):
    monkeypatch.setenv("HOME", "/home/studio")      # "~/data/train.jsonl" expands the same everywhere


def built(engine: str, **over):
    spec = spec_for(engine, **over)
    assert contract_errors(spec) == []
    settings = build_settings(spec, dry_run=True)
    adapter = load(engine)
    adapter.check_spec(settings)
    build = adapter.build(settings, Path("/unused"))
    files = {}
    for rel, text in build.files.items():
        files[rel] = json.loads(text) if rel.endswith(".json") else yaml.safe_load(text)
    return {"files": files, "steps": [{"name": s.name, "argv": s.argv} for s in build.steps],
            "outputs": [{"kind": o.kind, "path": o.path} for o in build.outputs], "notes": build.notes}


@pytest.mark.parametrize("engine", sorted(ADAPTERS))
def test_lora_valid_fixture_golden(engine: str):
    got = built(engine)
    path = GOLDEN / f"{engine}.lora-valid.json"
    if os.environ.get("UPDATE_GOLDEN") == "1":
        path.write_text(json.dumps(got, indent=2, sort_keys=True) + "\n")
    assert got == json.loads(path.read_text())


@pytest.mark.parametrize("engine", sorted(ADAPTERS))
def test_generated_yaml_is_valid_and_mentions_the_spec_values(engine: str):
    got = built(engine)
    text = json.dumps(got["files"])
    for value in ("freeweights/qwen2.5-0.5b-instruct", "~/data/train.jsonl", "0.0002"):
        assert value in text or "/home/studio/data/train.jsonl" in text, value


def test_unsloth_adapter_uses_upstreams_own_commands_only():
    got = built("unsloth-upstream")
    assert got["steps"][0]["argv"] == ["unsloth", "train", "--config", "config/unsloth.yaml"]
    cfg = got["files"]["config/unsloth.yaml"]
    assert cfg["training"]["load_in_4bit"] is False and cfg["lora"]["lora_r"] == 16


def test_qlora_turns_on_four_bit_everywhere():
    assert built("unsloth-upstream", method="qlora")["files"]["config/unsloth.yaml"]["training"]["load_in_4bit"]
    assert built("axolotl", method="qlora")["files"]["config/axolotl.yaml"]["adapter"] == "qlora"
    assert built("llama-factory", method="qlora")["files"]["config/train.yaml"]["quantization_bit"] == 4
    assert "qlora_" in built("torchtune", method="qlora", base__repo="meta-llama/Llama-3.2-1B")["files"][
        "config/torchtune.yaml"]["model"]["_component_"]


def test_dpo_translation():
    spec = dict(method="dpo", dataset__format="dpo")
    assert built("axolotl", **spec)["files"]["config/axolotl.yaml"]["rl"] == "dpo"
    lf = built("llama-factory", **spec)
    assert lf["files"]["config/train.yaml"]["stage"] == "dpo"
    assert lf["files"]["config/dataset_info.json"]["studio_train"]["ranking"] is True
    tt = built("torchtune", **spec)
    assert tt["steps"][0]["argv"][2] == "lora_dpo_single_device"


def test_torchtune_refuses_what_it_cannot_map():
    from studio_engine_kit.spec import SpecError
    adapter = load("torchtune")
    for over in ({"base__repo": "some/unknown-model"}, {"method": "qlora"}):   # qlora on Qwen has no builder
        settings = build_settings(spec_for("torchtune", **over), dry_run=True)
        with pytest.raises(SpecError):
            adapter.check_spec(settings)


def test_adapter_pins_match_pins_json():
    pins = json.loads((Path(__file__).resolve().parents[2] / "pins.json").read_text())["engines"]
    assert sorted(pins) == sorted(ADAPTERS)
    for engine in ADAPTERS:
        assert load(engine).pin == pins[engine]["version"]
