# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Allternit LLC
"""A tiny YAML writer so adapters need no YAML library at run time.

Strings are written as JSON strings, which is valid YAML, so quoting is always
safe. Mappings keep insertion order so generated configs are stable.
"""

from __future__ import annotations

import json
from typing import Any, List


def _scalar(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    return json.dumps(str(v), ensure_ascii=False)


def _lines(v: Any, indent: int) -> List[str]:
    pad = "  " * indent
    out: List[str] = []
    if isinstance(v, dict):
        for k, val in v.items():
            if isinstance(val, dict) and val:
                out.append(f"{pad}{k}:")
                out += _lines(val, indent + 1)
            elif isinstance(val, list) and val:
                out.append(f"{pad}{k}:")
                out += _lines(val, indent + 1)
            elif isinstance(val, (dict, list)):
                out.append(f"{pad}{k}: {'{}' if isinstance(val, dict) else '[]'}")
            else:
                out.append(f"{pad}{k}: {_scalar(val)}")
    else:
        for item in v:
            if isinstance(item, dict) and item:
                sub = _lines(item, indent + 1)
                out.append(f"{pad}- {sub[0].lstrip()}")
                out += sub[1:]
            elif isinstance(item, list) and item:
                raise ValueError("nested lists are not supported")
            else:
                out.append(f"{pad}- {_scalar(item)}")
    return out


def dump(data: Any, header: str = "") -> str:
    text = "\n".join(_lines(data, 0)) + "\n"
    return (header + "\n" if header else "") + text
