# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Allternit LLC
"""The vendored contract files are the platform's, byte for byte.

kit/studio_engine_kit/spec-sync.json records the sha256 of every file copied
from the platform's services/studio-spec (tools/sync_spec.py writes it).

- Always: each copy here still has the recorded sha256 (nobody edited a copy).
- With ALLTERNIT_STUDIO_SPEC_DIR=<platform checkout>/services/studio-spec: the
  platform file still has it too, so a contract change on the platform fails
  here until tools/sync_spec.py is run against it and the copies are committed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RECORD = json.loads((ROOT / "kit" / "studio_engine_kit" / "spec-sync.json").read_text())
FILES = sorted(RECORD["files"].items())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_record_covers_the_catalog_and_the_fit_table():
    assert {"engines.json", "fixtures/fit-table.json", "fixtures/lora-valid.json"} <= set(RECORD["files"])
    assert len(RECORD["platform_commit"]) == 40


@pytest.mark.parametrize("name,entry", FILES, ids=[n for n, _ in FILES])
def test_vendored_copy_matches_the_record(name, entry):
    assert _sha(ROOT / entry["copy"]) == entry["sha256"], f"{entry['copy']} was edited; re-run tools/sync_spec.py"


@pytest.mark.parametrize("name,entry", FILES, ids=[n for n, _ in FILES])
def test_platform_file_has_not_drifted(name, entry):
    spec_dir = os.environ.get("ALLTERNIT_STUDIO_SPEC_DIR")
    if not spec_dir:
        pytest.skip("set ALLTERNIT_STUDIO_SPEC_DIR to a platform services/studio-spec folder")
    platform_file = Path(spec_dir) / name
    assert platform_file.is_file(), f"{name} is gone from the platform contract"
    assert _sha(platform_file) == entry["sha256"], (
        f"{name} changed on the platform: run tools/sync_spec.py {spec_dir} --commit <sha> and commit the copies")


def test_no_new_platform_fixture_is_missing_here():
    spec_dir = os.environ.get("ALLTERNIT_STUDIO_SPEC_DIR")
    if not spec_dir:
        pytest.skip("set ALLTERNIT_STUDIO_SPEC_DIR to a platform services/studio-spec folder")
    platform = {f"fixtures/{p.name}" for p in (Path(spec_dir) / "fixtures").glob("*.json")}
    assert platform <= set(RECORD["files"]), "new platform fixtures: run tools/sync_spec.py"
