# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Allternit LLC
"""Job spec v1 checks, shared by every adapter.

Mirrors services/studio-spec/src/spec.rs of the platform repo: the same
refusal codes in the same order, collected the same way (the first code is the
one the HTTP error carries). On top of that the kit normalises `params` into a
`Settings` object so each adapter only maps values to its engine's own config.

Pure standard library. Python 3.9+.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

SPEC_VERSION = 1
METHODS = ("lora", "qlora", "full", "dpo")
PLACES = {"local", "paired", "compute", "rented"}
BACKENDS = {"auto", "torch", "mlx"}
DATASET_FORMATS = {"jsonl", "chat", "alpaca", "text", "completion", "dpo"}
DATASET_SUFFIXES = {".jsonl", ".json", ".parquet"}
MAX_SEQ_LEN = 8192
TOP_LEVEL_KEYS = {
    "spec_version", "engine", "engine_version", "base", "dataset", "method",
    "params", "resources", "outputs", "eval", "publish",
}

# Quants convert_hf_to_gguf.py writes directly, and the ones llama-quantize makes.
GGUF_DIRECT = {"f32", "f16", "bf16", "q8_0"}
GGUF_QUANTIZE = {
    "q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "q4_0", "q4_1", "q4_k_s", "q4_k_m",
    "q5_0", "q5_1", "q5_k_s", "q5_k_m", "q6_k", "iq4_xs", "iq4_nl",
}

_CATALOG_PATH = Path(__file__).with_name("engines.json")


class SpecError(Exception):
    """A refused spec. Becomes HTTP 422 {error: code, message}."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def load_catalog() -> Dict[str, Dict[str, Any]]:
    """engines.json copied from services/studio-spec (see NOTICE for the commit)."""
    entries = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    return {e["id"]: e for e in entries}


# ---- the Rust contract, ported -------------------------------------------------

def contract_errors(spec: Any, catalog: Optional[Dict[str, Dict[str, Any]]] = None) -> List[SpecError]:
    """Every problem the platform contract (spec.rs `JobSpec::validate`) finds.

    Order matches the Rust code: version, engine/method, base, dataset,
    rented, publish. A shape error comes back as one `invalid_spec`.
    """
    catalog = catalog or load_catalog()
    if not isinstance(spec, dict):
        return [SpecError("invalid_spec", "job spec must be a JSON object")]
    unknown = sorted(set(spec) - TOP_LEVEL_KEYS)
    if unknown:
        return [SpecError("invalid_spec", f"unknown field(s): {', '.join(unknown)}")]
    for key in ("spec_version", "engine", "base", "dataset", "method", "resources"):
        if key not in spec:
            return [SpecError("invalid_spec", f"missing field: {key}")]
    if not isinstance(spec["engine"], str) or not isinstance(spec["method"], str):
        return [SpecError("invalid_spec", "engine and method must be strings")]
    for key in ("base", "dataset", "resources"):
        if not isinstance(spec[key], dict):
            return [SpecError("invalid_spec", f"{key} must be an object")]
    resources = spec["resources"]
    if resources.get("place") not in PLACES:
        return [SpecError("invalid_spec", "resources.place must be local, paired, compute or rented")]
    if resources.get("backend") is not None and resources["backend"] not in BACKENDS:
        return [SpecError("invalid_spec", "resources.backend must be auto, torch or mlx")]
    base = spec["base"]
    for key in ("repo", "path"):
        if base.get(key) is not None and not isinstance(base[key], str):
            return [SpecError("invalid_spec", f"base.{key} must be a string")]

    errs: List[SpecError] = []
    if spec["spec_version"] != SPEC_VERSION:
        errs.append(SpecError(
            "unsupported_spec_version",
            f"spec_version {spec['spec_version']} is not supported (expected {SPEC_VERSION})",
        ))
    entry = catalog.get(spec["engine"])
    if entry is None:
        errs.append(SpecError("unknown_engine", f"engine '{spec['engine']}' is not in the catalog"))
    elif spec["method"] not in entry["methods"]:
        errs.append(SpecError(
            "unknown_method", f"engine '{spec['engine']}' does not support method '{spec['method']}'"))
    if bool(base.get("repo")) == (base.get("path") is not None):
        errs.append(SpecError("invalid_base", "base needs exactly one of repo or path"))
    ds = spec["dataset"]
    if bool(ds.get("repo")) == bool(ds.get("path")):
        errs.append(SpecError("invalid_dataset", "dataset needs exactly one of repo or path"))
    if resources["place"] == "rented":
        errs.append(SpecError("rented_not_available", "rented GPUs are not available yet"))
    if spec.get("publish") is not None:
        errs.append(SpecError("publish_not_available", "publishing from a job is not available yet"))
    return errs


# ---- normalised settings -------------------------------------------------------

_PARAM_TYPES: Dict[str, type] = {
    "rank": int, "alpha": int, "dropout": float, "target_modules": list, "lr": float,
    "epochs": float, "max_steps": int, "batch_size": int, "grad_accum": int,
    "warmup_steps": int, "weight_decay": float, "max_seq_len": int, "seed": int,
    "beta": float, "lr_scheduler": str, "gradient_checkpointing": bool,
}
KNOWN_PARAMS = tuple(sorted(_PARAM_TYPES))
_DEFAULT_LR = {"lora": 2e-4, "qlora": 2e-4, "full": 2e-5, "dpo": 5e-5}


@dataclass
class Settings:
    """Everything an adapter needs, after the spec passed every check."""

    engine: str
    method: str
    base: str                      # base.path, absolute (or a base.repo hub id in dry-run only)
    dataset_path: Optional[str]
    dataset_repo: Optional[str]
    dataset_revision: Optional[str]
    dataset_format: str
    rank: Optional[int]
    alpha: Optional[int]
    dropout: float
    target_modules: Any            # "all-linear" or a list of module names
    lr: float
    epochs: float
    max_steps: int                 # -1 means "use epochs"
    batch_size: int
    grad_accum: int
    warmup_steps: int
    weight_decay: float
    max_seq_len: int
    seed: int
    beta: Optional[float]
    lr_scheduler: str
    gradient_checkpointing: bool
    want_adapter: bool
    want_merged: bool
    gguf: List[str]
    max_hours: Optional[float]
    dry_run: bool = False
    notes: List[str] = field(default_factory=list)
    hardware: Optional[str] = None     # ALLTERNIT_STUDIO_HARDWARE id the job runs on
    backend: Optional[str] = None      # torch-cuda, torch-mps, ... (fit.check_job)

    @property
    def is_lora(self) -> bool:
        return self.method in ("lora", "qlora", "dpo")


def _expand(path: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(path)))


def _looks_like_path(value: str) -> bool:
    return value.startswith(("/", "~", "./", "../", "$")) or value in (".", "..")


def _params(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SpecError("invalid_spec", "params must be an object")
    unknown = sorted(set(raw) - set(_PARAM_TYPES))
    if unknown:
        raise SpecError("invalid_spec", f"unknown params: {', '.join(unknown)} (known: {', '.join(KNOWN_PARAMS)})")
    out: Dict[str, Any] = {}
    for key, value in raw.items():
        want = _PARAM_TYPES[key]
        if want is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if want is list and value == "all-linear":
            pass
        elif want in (int, float) and isinstance(value, bool):
            raise SpecError("invalid_spec", f"params.{key} must be {want.__name__}")
        elif not isinstance(value, want):
            raise SpecError("invalid_spec", f"params.{key} must be {want.__name__}")
        elif want is list and not all(isinstance(m, str) for m in value):
            raise SpecError("invalid_spec", f"params.{key} must be a list of module names")
        if want in (int, float) and value < 0:
            raise SpecError("invalid_spec", f"params.{key} must not be negative")
        out[key] = value
    for key in ("rank", "alpha", "lr", "batch_size", "grad_accum", "max_seq_len"):
        if key in out and out[key] <= 0:
            raise SpecError("invalid_spec", f"params.{key} must be greater than 0")
    if out.get("max_seq_len", 0) > MAX_SEQ_LEN:
        raise SpecError("invalid_spec", f"params.max_seq_len is above the limit {MAX_SEQ_LEN}")
    if out.get("dropout", 0) >= 1:
        raise SpecError("invalid_spec", "params.dropout must be below 1")
    return out


def _dataset(ds: Dict[str, Any], method: str, dry_run: bool) -> Dict[str, Optional[str]]:
    fmt = ds.get("format")
    if fmt not in DATASET_FORMATS:
        raise SpecError("invalid_dataset", f"dataset.format '{fmt}' is not one of {', '.join(sorted(DATASET_FORMATS))}")
    if method == "dpo" and fmt not in ("dpo", "jsonl"):
        raise SpecError("invalid_dataset", "method dpo needs dataset.format dpo (prompt, chosen, rejected)")
    if method != "dpo" and fmt == "dpo":
        raise SpecError("invalid_dataset", f"dataset.format dpo is only for method dpo, not {method}")
    out: Dict[str, Optional[str]] = {"path": None, "repo": None, "revision": ds.get("revision"), "format": fmt}
    if ds.get("path"):
        if not isinstance(ds["path"], str):
            raise SpecError("invalid_dataset", "dataset.path must be a string")
        file = _expand(ds["path"])
        if file.suffix.lower() not in DATASET_SUFFIXES:
            raise SpecError("invalid_dataset", f"dataset file must be .jsonl, .json or .parquet: {file}")
        if not dry_run and not file.is_file():
            raise SpecError("invalid_dataset", f"dataset file not found: {file}")
        out["path"] = str(file)
    else:
        repo = ds.get("repo")
        if not isinstance(repo, str) or _looks_like_path(repo):
            raise SpecError("invalid_dataset", f"dataset.repo '{repo}' is not a Hugging Face dataset id")
        out["repo"] = repo
    return out


def _base(base: Dict[str, Any], dry_run: bool) -> str:
    """The model folder the trainer loads: base.path, absolute.

    Engines never download weights. The host fetches and verifies the model and
    sends its folder in base.path. A hub id in base.repo is accepted in dry-run
    only (config and command are built, nothing loads it).
    """
    if base.get("path") is None:
        value = base["repo"].strip()
        if dry_run:
            return value
        raise SpecError(
            "invalid_base",
            f"base.repo '{value}' is a repo id. Engines never download weights: the host "
            "fetches and verifies the model, then sends its local folder in base.path.",
        )
    raw = base["path"].strip()
    folder = _expand(raw)
    if not raw or not folder.is_absolute():
        raise SpecError("invalid_base", f"base.path must be an absolute path (or ~/...), not '{raw}'")
    if dry_run:
        return str(folder)
    if not folder.is_dir():
        raise SpecError("invalid_base", f"base model folder not found: {folder}")
    if not (folder / "config.json").is_file():
        raise SpecError("invalid_base", f"base model folder has no config.json: {folder}")
    return str(folder.resolve())


def build_settings(spec: Dict[str, Any], dry_run: bool = False,
                   hardware: Optional[str] = None, backend: Optional[str] = None) -> Settings:
    """Normalise a spec that already passed `contract_errors`. Raises SpecError."""
    method = spec["method"]
    params = _params(spec.get("params"))
    dataset = _dataset(spec["dataset"], method, dry_run)
    base = _base(spec["base"], dry_run)

    raw_out = spec.get("outputs") or {}
    if not isinstance(raw_out, dict):
        raise SpecError("invalid_spec", "outputs must be an object")
    if raw_out.get("mlx"):
        # check_job refuses this first with the fit reason; kept for direct callers.
        raise SpecError("output_not_supported", "MLX files are made only on a Mac with Apple Silicon.")
    gguf = raw_out.get("gguf") or []
    if not isinstance(gguf, list) or not all(isinstance(q, str) for q in gguf):
        raise SpecError("invalid_spec", "outputs.gguf must be a list of quant names")
    gguf = [q.lower() for q in gguf]
    bad = [q for q in gguf if q not in GGUF_DIRECT | GGUF_QUANTIZE]
    if bad:
        raise SpecError("invalid_spec", f"unknown GGUF quant: {', '.join(bad)}")
    if method == "full":
        want_adapter, want_merged = False, True      # a full fine-tune is its own merged model
    else:
        want_adapter, want_merged = True, bool(raw_out.get("merged"))

    max_hours = spec["resources"].get("max_hours")
    if max_hours is not None and (isinstance(max_hours, bool) or not isinstance(max_hours, (int, float)) or max_hours <= 0):
        raise SpecError("invalid_spec", "resources.max_hours must be a positive number")

    if method != "dpo" and "beta" in params:
        raise SpecError("invalid_spec", "params.beta is only for method dpo")
    lora = method != "full"
    rank = params.get("rank", 16) if lora else None
    notes: List[str] = []
    ev = spec.get("eval") or {}
    if isinstance(ev, dict) and ev.get("sets"):
        notes.append("eval sets are not run by this engine; the job trains and exports only")

    return Settings(
        engine=spec["engine"], method=method, base=base,
        dataset_path=dataset["path"], dataset_repo=dataset["repo"],
        dataset_revision=dataset["revision"], dataset_format=dataset["format"] or "jsonl",
        rank=rank, alpha=params.get("alpha", (rank or 16) * 2) if lora else None,
        dropout=params.get("dropout", 0.0),
        target_modules=params.get("target_modules", "all-linear"),
        lr=params.get("lr", _DEFAULT_LR[method]), epochs=params.get("epochs", 1.0),
        max_steps=params.get("max_steps", -1), batch_size=params.get("batch_size", 2),
        grad_accum=params.get("grad_accum", 4), warmup_steps=params.get("warmup_steps", 0),
        weight_decay=params.get("weight_decay", 0.0), max_seq_len=params.get("max_seq_len", 2048),
        seed=params.get("seed", 42), beta=params.get("beta", 0.1) if method == "dpo" else None,
        lr_scheduler=params.get("lr_scheduler", "linear"),
        gradient_checkpointing=params.get("gradient_checkpointing", False),
        want_adapter=want_adapter, want_merged=want_merged, gguf=gguf,
        max_hours=float(max_hours) if max_hours is not None else None,
        dry_run=dry_run, notes=notes, hardware=hardware, backend=backend,
    )
