# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Allternit LLC
"""python -m studio_engine_kit --engine axolotl --port 0 --token T [--dry-run]

Prints one JSON line {"ready": true, "port": N, ...} to stdout once listening.
The token can come from STUDIO_ENGINE_TOKEN so it stays out of `ps`.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any, Optional

from . import __version__
from .jobs import AlreadyFinished, JobManager
from .server import serve


def main(argv: Optional[list] = None) -> int:
    from studio_engine_adapters import ADAPTERS, load

    p = argparse.ArgumentParser(prog="python -m studio_engine_kit")
    p.add_argument("--engine", required=True, choices=sorted(ADAPTERS))
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--token", default=os.environ.get("STUDIO_ENGINE_TOKEN"))
    p.add_argument("--data-dir", default=os.environ.get("ALLTERNIT_STUDIO_OUTPUT_DIR"),
                   help="where job folders and outputs go (default ALLTERNIT_STUDIO_OUTPUT_DIR, else "
                        "STUDIO_ENGINE_DATA_DIR/<engine>, else ~/.allternit/studio/engines/<engine>)")
    p.add_argument("--dry-run", action="store_true",
                   default=os.environ.get("STUDIO_ENGINE_DRY_RUN") == "1",
                   help="build config and command, run a stub instead of the trainer")
    args = p.parse_args(argv)
    if not args.token:
        p.error("--token (or STUDIO_ENGINE_TOKEN) is required")
    adapter = load(args.engine)
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        legacy = os.environ.get("STUDIO_ENGINE_DATA_DIR") or str(Path.home() / ".allternit" / "studio" / "engines")
        data_dir = Path(legacy) / args.engine
    manager = JobManager(adapter, data_dir, dry_run=args.dry_run)
    server = serve(manager, args.token, args.port)
    print(json.dumps({"ready": True, "port": server.server_address[1], "engine": adapter.id,
                      "version": adapter.pin, "kit": __version__, "dry_run": args.dry_run,
                      "hardware": manager.hardware, "output_dir": str(manager.data_dir)}), flush=True)

    def _term(signum: int, frame: Any) -> None:  # noqa: ARG001
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _term)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for job in list(manager.jobs.values()):
            try:
                manager.cancel(job.id)
            except AlreadyFinished:
                pass
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
