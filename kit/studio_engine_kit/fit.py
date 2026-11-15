# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Allternit LLC
"""Which methods and outputs an engine runs on a hardware id.

A port of studio-spec src/fit.rs: the same reasons, word for word, and the same
refusal codes (`hardware_not_supported`, `method_not_supported`,
`backend_not_supported`, `output_not_supported`). Everything is derived from
engines.json, so tests/conformance/test_fit_table.py can assert it against
the platform's fixtures/fit-table.json. Standard library only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .hardware import HARDWARE_IDS, LABELS, TORCH_BACKEND
from .spec import SpecError

CPU_WARNING = "Training on the CPU is slow. Use a small base model; even a small LoRA run can take hours."
BACKEND_CHOICES = ("auto", "torch", "mlx")
_METHOD_NAMES = {"lora": "LoRA", "qlora": "QLoRA", "full": "full fine-tuning", "dpo": "DPO"}


def _method_name(m: str) -> str:
    return _METHOD_NAMES.get(m, m)


def _capitalize(s: str) -> str:
    return s[:1].upper() + s[1:]


def _join(items: List[str], last: str) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} {last} {items[-1]}"


def _has_mlx(engine: Dict[str, Any]) -> bool:
    return any(b.get("id") == "mlx" and b.get("status") == "available" for b in engine.get("backends", []))


def _mlx_unsupported(engine: Dict[str, Any]) -> List[str]:
    return list((engine.get("limits") or {}).get("mlx_unsupported_methods") or [])


def _runs_on(engine: Dict[str, Any], hw: str) -> bool:
    return hw in engine["hardware"]


def engine_refusal(engine: Dict[str, Any], hw: str) -> Optional[str]:
    if _runs_on(engine, hw):
        return None
    labels = [LABELS[h] for h in HARDWARE_IDS if _runs_on(engine, h)]
    return f"{engine['name']} runs only on {_join(labels, 'or')}, not on {LABELS[hw]}."


def method_refusal(engine: Dict[str, Any], hw: str, method: str) -> Optional[str]:  # noqa: ARG001
    if hw == "cpu" and method != "lora":
        return f"{_capitalize(_method_name(method))} needs a GPU. Only LoRA runs on the CPU."
    return None


def backend_for(engine: Dict[str, Any], hw: str, method: str, choice: str = "auto") -> Tuple[Optional[str], Optional[str]]:
    """(backend, None) or (None, reason for backend_not_supported)."""
    mlx_here = hw == "apple-silicon" and _has_mlx(engine)
    mlx_method = method not in _mlx_unsupported(engine)
    if choice == "mlx":
        if hw != "apple-silicon":
            return None, "MLX runs only on a Mac with Apple Silicon."
        if not mlx_here:
            return None, f"{engine['name']} has no MLX backend."
        if not mlx_method:
            return None, (f"MLX does not run {_method_name(method)} here. "
                          "Pick the automatic or PyTorch backend.")
        return "mlx", None
    if choice == "auto" and mlx_here and mlx_method:
        return "mlx", None
    return TORCH_BACKEND[hw], None


def output_refusal(engine: Dict[str, Any], hw: str, output: str) -> Optional[str]:
    if output != "mlx":
        return None
    if hw != "apple-silicon":
        return "MLX files are made only on a Mac with Apple Silicon."
    if not _has_mlx(engine):
        return f"{engine['name']} cannot make MLX files. Allternit Tune can."
    return None


def _ok_reason(engine: Dict[str, Any], hw: str) -> str:
    if hw == "apple-silicon" and _has_mlx(engine):
        off = _mlx_unsupported(engine)
        mlx = [_method_name(m) for m in engine["methods"] if m not in off]
        torch = [_method_name(m) for m in engine["methods"] if m in off]
        if not torch:
            return f"Runs on this Mac's GPU with MLX ({_join(mlx, 'and')})."
        return f"Runs on this Mac's GPU: {_join(mlx, 'and')} with MLX, {_join(torch, 'and')} with PyTorch (MPS)."
    return {
        "apple-silicon": "Runs on this Mac's GPU with PyTorch (MPS).",
        "cuda-linux": "Runs on this computer's NVIDIA GPU.",
        "cuda-windows": "Runs on this computer's NVIDIA GPU.",
        "rocm-linux": "Runs on this computer's AMD GPU (ROCm).",
        "xpu-linux": "Runs on this computer's Intel GPU (XPU).",
        "xpu-windows": "Runs on this computer's Intel GPU (XPU).",
        "cpu": "Runs on this computer's CPU.",
    }[hw]


def fit(engine: Dict[str, Any], hw: str) -> Dict[str, Any]:
    """The `fit` object (studio-api.yaml Fit) for one engine on `hw`."""
    refused = engine_refusal(engine, hw)
    if refused:
        return {"ok": False, "hardware": hw, "reason": refused,
                "methods_unavailable": {}, "outputs_unavailable": {}}
    methods_unavailable: Dict[str, str] = {}
    backend: Optional[str] = None
    for m in engine["methods"]:
        why = method_refusal(engine, hw, m)
        if why:
            methods_unavailable[m] = why
        elif backend is None:
            backend = backend_for(engine, hw, m)[0]
    outputs_unavailable = {o: r for o in ("adapter", "merged", "gguf", "mlx")
                           for r in [output_refusal(engine, hw, o)] if r}
    card: Dict[str, Any] = {"ok": True, "hardware": hw, "reason": _ok_reason(engine, hw),
                            "methods_unavailable": methods_unavailable, "outputs_unavailable": outputs_unavailable}
    if backend:
        card["backend"] = backend
    if hw == "cpu":
        card["warning"] = CPU_WARNING
    return card


def check_job(spec: Dict[str, Any], engine: Dict[str, Any], hw: str) -> Tuple[Optional[str], List[SpecError]]:
    """(backend, []) when the job runs here, else (None, refusals).

    Same order as fit.rs `check_job`: a refused engine stops at
    hardware_not_supported; otherwise method, backend and outputs are collected.
    The spec already passed `contract_errors`.
    """
    card = fit(engine, hw)
    if not card["ok"]:
        return None, [SpecError("hardware_not_supported", card["reason"])]
    errs: List[SpecError] = []
    method = spec["method"]
    if method in card["methods_unavailable"]:
        errs.append(SpecError("method_not_supported", card["methods_unavailable"][method]))
    choice = (spec.get("resources") or {}).get("backend") or "auto"
    backend, why = backend_for(engine, hw, method, choice)
    if why:
        errs.append(SpecError("backend_not_supported", why))
    out = spec.get("outputs") or {}
    asked = (("adapter", bool(out.get("adapter"))), ("merged", bool(out.get("merged"))),
             ("gguf", bool(out.get("gguf"))), ("mlx", bool(out.get("mlx"))))
    for name, on in asked:
        if on and name in card["outputs_unavailable"]:
            errs.append(SpecError("output_not_supported", card["outputs_unavailable"][name]))
    return (backend if not errs else None), errs
