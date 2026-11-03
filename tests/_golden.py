"""Tiny golden-file helper: ``check(name, value)`` compares JSON-normalised ``value`` with the stored golden.

Run with ``UPDATE_GOLDEN=1`` to (re)capture; the pass-2 goldens were captured from the pre-split code.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

PATH = Path(__file__).parent / "golden" / "pass2_golden.json"


def _norm(value):
    return json.loads(json.dumps(value, sort_keys=False, default=str))


def check(name: str, value) -> None:
    value = _norm(value)
    data = json.loads(PATH.read_text()) if PATH.exists() else {}
    if os.environ.get("UPDATE_GOLDEN"):
        data[name] = value
        PATH.write_text("{\n" + ",\n".join(f"{json.dumps(k)}: {json.dumps(data[k], sort_keys=True)}"
                                           for k in sorted(data)) + "\n}\n")
        return
    assert name in data, f"no golden for {name}; run with UPDATE_GOLDEN=1 on the pre-refactor code"
    if value != data[name]:
        import difflib
        a = json.dumps(data[name], indent=1, sort_keys=True).splitlines()
        b = json.dumps(value, indent=1, sort_keys=True).splitlines()
        raise AssertionError("golden mismatch:\n" + "\n".join(list(difflib.unified_diff(a, b, "golden", "now", lineterm=""))[:40]))
