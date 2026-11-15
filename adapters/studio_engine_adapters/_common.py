# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Allternit LLC
"""Helpers shared by adapters."""

from __future__ import annotations

import math
from typing import List

from studio_engine_kit.spec import Settings, SpecError

DEFAULT_LORA_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def whole_epochs(s: Settings, engine: str) -> int:
    if s.epochs != math.floor(s.epochs):
        raise SpecError("invalid_spec", f"params.epochs must be a whole number for {engine}")
    return int(s.epochs)


def linear_modules(s: Settings) -> List[str]:
    return list(DEFAULT_LORA_LINEAR) if s.target_modules == "all-linear" else list(s.target_modules)


def refuse_revision(s: Settings, engine: str) -> None:
    if s.dataset_revision:
        raise SpecError("invalid_dataset", f"dataset.revision is not supported by {engine}; pin the data by file path")
