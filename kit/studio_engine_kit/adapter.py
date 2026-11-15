# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Allternit LLC
"""The adapter contract: a job spec v1 in, the engine's own config and command out.

An adapter never imports the engine. It writes the engine's config file(s) and
returns the command line that starts the engine as a SEPARATE PROCESS inside the
engine's own virtualenv. The kit runs that command and speaks Engine API v1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .spec import GGUF_DIRECT, Settings, SpecError


@dataclass
class Step:
    """One process to run. The first step trains; later steps export."""

    name: str                      # "train", "merge", "gguf"
    argv: List[str]
    env: Dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None      # relative to the job folder; None = job folder


@dataclass
class OutputSpec:
    """A result to register. `path` is relative to the job folder and may hold
    one `*`; with several matches the last in sorted order wins."""

    kind: str                      # adapter | merged | gguf
    path: str


@dataclass
class Build:
    """What an adapter produces for one job."""

    files: Dict[str, str]          # relative path -> text (configs, dataset_info.json, ...)
    steps: List[Step]
    outputs: List[OutputSpec]
    notes: List[str] = field(default_factory=list)

    def command(self) -> List[List[str]]:
        return [s.argv for s in self.steps]


class Adapter:
    """Subclass per engine. Class attributes come from engines.json plus the pin."""

    id: str = ""
    pin: str = ""                  # upstream version, mirrors pins.json
    executable: str = ""           # how the engine's CLI is called, e.g. "axolotl"
    progress_patterns = (
        re.compile(r"(\d+)\s*/\s*(\d+)\s*\["),          # tqdm: 12/100 [
        re.compile(r"[Ss]tep\s+(\d+)\s*/\s*(\d+)"),     # "step 3/10"
    )

    def methods(self) -> List[str]:
        """The methods engines.json lists for this engine (the contract's list)."""
        from .spec import load_catalog
        return list(load_catalog()[self.id]["methods"])

    def check_spec(self, s: Settings) -> None:
        """Refuse what this engine cannot do. Raise SpecError (422)."""

    def build(self, s: Settings, job_dir: Path) -> Build:
        raise NotImplementedError

    def parse_progress(self, line: str) -> Optional[float]:
        for pattern in self.progress_patterns:
            m = pattern.search(line)
            if m and int(m.group(2)) > 0:
                return min(1.0, int(m.group(1)) / int(m.group(2)))
        return None

    def engine_command(self) -> List[str]:
        """Prefix that launches the engine CLI. ENGINE_<ID>_BIN overrides it so a
        host can point at the venv it created (`/path/venv/bin/axolotl`)."""
        import os
        override = os.environ.get("STUDIO_ENGINE_BIN_" + self.id.upper().replace("-", "_"))
        return [override] if override else [self.executable]


def gguf_steps(merged_rel: str, quants: List[str], out_rel: str, llama_cpp_dir: str) -> List[Step]:
    """Steps that turn a merged Hugging Face folder into GGUF files with llama.cpp.

    `llama_cpp_dir` holds convert_hf_to_gguf.py and (for quantised outputs)
    a llama-quantize binary. The converter is llama.cpp's own tool run as a
    process, nothing here links against it.
    """
    steps: List[Step] = []
    f16 = f"{out_rel}/model-f16.gguf"
    need_f16 = any(q not in GGUF_DIRECT for q in quants) or "f16" in quants
    direct = [q for q in quants if q in GGUF_DIRECT]
    if need_f16 and "f16" not in direct:
        direct.append("f16")
    for q in direct:
        steps.append(Step(f"gguf-{q}", [
            "python3", f"{llama_cpp_dir}/convert_hf_to_gguf.py", merged_rel,
            "--outfile", f"{out_rel}/model-{q}.gguf", "--outtype", q]))
    for q in quants:
        if q in GGUF_DIRECT:
            continue
        steps.append(Step(f"gguf-{q}", [
            f"{llama_cpp_dir}/llama-quantize", f16, f"{out_rel}/model-{q}.gguf", q.upper()]))
    return steps


def gguf_outputs(quants: List[str], out_rel: str) -> List[OutputSpec]:
    return [OutputSpec("gguf", f"{out_rel}/model-{q}.gguf") for q in quants]


def require_llama_cpp(s: Settings) -> str:
    """Folder with llama.cpp's converter. Refuse when GGUF is wanted and it is missing."""
    import os
    folder = os.environ.get("LLAMA_CPP_DIR", "")
    if s.dry_run:
        return folder or "$LLAMA_CPP_DIR"
    if not folder or not (Path(folder) / "convert_hf_to_gguf.py").is_file():
        raise SpecError("invalid_spec",
                        "outputs.gguf needs llama.cpp's convert_hf_to_gguf.py: set LLAMA_CPP_DIR to a llama.cpp checkout")
    return folder
