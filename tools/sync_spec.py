# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Allternit LLC
"""Copy the contract files from the platform repo and record their sha256.

    python3 tools/sync_spec.py <platform checkout>/services/studio-spec [--commit SHA]

Copies engines.json into the kit and fixtures/*.json into the conformance
tests, then rewrites kit/studio_engine_kit/spec-sync.json with the sha256 of
each copied file. tests/conformance/test_spec_sync.py fails when a vendored copy
no longer matches that record, and (with ALLTERNIT_STUDIO_SPEC_DIR set to a
platform services/studio-spec folder) when the platform file has moved on.

Refresh after the platform changes the contract: run this script against the
new checkout, run the tests, commit the copies and spec-sync.json together.
Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "kit" / "studio_engine_kit" / "spec-sync.json"
FIXTURES = ROOT / "tests" / "conformance" / "fixtures"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan(spec_dir: Path) -> dict:
    """platform path (relative to services/studio-spec) -> repo path (relative to ROOT)."""
    files = {"engines.json": "kit/studio_engine_kit/engines.json"}
    for fx in sorted((spec_dir / "fixtures").glob("*.json")):
        files[f"fixtures/{fx.name}"] = f"tests/conformance/fixtures/{fx.name}"
    return files


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("spec_dir", type=Path, help="services/studio-spec in a platform checkout")
    p.add_argument("--commit", default="", help="platform commit the files come from")
    args = p.parse_args(argv)
    spec_dir = args.spec_dir.resolve()
    if not (spec_dir / "engines.json").is_file():
        p.error(f"{spec_dir} has no engines.json")
    record = {
        "_about": "sha256 of each contract file copied from the platform's services/studio-spec. "
                  "Refresh with tools/sync_spec.py; checked by tests/conformance/test_spec_sync.py.",
        "source": "services/studio-spec",
        "platform_commit": args.commit,
        "files": {},
    }
    for src_rel, dst_rel in plan(spec_dir).items():
        dst = ROOT / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(spec_dir / src_rel, dst)
        record["files"][src_rel] = {"copy": dst_rel, "sha256": sha256(dst)}
    RECORD.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"synced {len(record['files'])} files; record in {RECORD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
