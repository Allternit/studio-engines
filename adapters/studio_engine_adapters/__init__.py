# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Allternit LLC
"""Engine adapters. Each maps a job spec v1 to one upstream engine's own config
and command line. None imports its engine: the boundary is a separate process."""

from __future__ import annotations

import importlib
from typing import Dict

ADAPTERS: Dict[str, str] = {
    "unsloth-upstream": "studio_engine_adapters.unsloth_upstream",
    "axolotl": "studio_engine_adapters.axolotl",
    "llama-factory": "studio_engine_adapters.llama_factory",
    "torchtune": "studio_engine_adapters.torchtune",
}


def load(engine_id: str):
    """Return an instance of the adapter for `engine_id`."""
    module = importlib.import_module(ADAPTERS[engine_id])
    return module.ADAPTER
