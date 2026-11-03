"""24 h symbol-list caches of the five scanners: round trip, expiry, and tolerance of a corrupt file."""
from __future__ import annotations

import json
import time

import pytest
from _scanner_import import import_scanner

try:                       # pass-2 modules; absent on the pre-split commit
    import scan_common
    import us_symbols
except ImportError:
    scan_common = us_symbols = None

# (module getter, cache-path attribute, load, save, payload, expected loaded)
CASES = {
    "indian": ("_SYMBOL_CACHE", "_load_symbol_cache", "_save_symbol_cache", (["A", "B"], ["C"]), (["A", "B"], ["C"])),
    "us": ("_SYMBOL_CACHE", "_load_symbol_cache", "_save_symbol_cache", (["A"], ["B"]), (["A"], ["B"])),
    "japan": ("_CACHE_FILE", "_load_cache", "_save_cache", (["7203.T"],), ["7203.T"]),
    "korea": ("_CACHE_FILE", "_load_cache", "_save_cache", ([{"code": "005930"}],), [{"code": "005930"}]),
}
EMPTY = {"indian": (None, None), "us": (None, None), "japan": None, "korea": None}


def _mod(name):
    return us_symbols if (name == "us" and us_symbols is not None) else import_scanner(name)


@pytest.mark.parametrize("name", sorted(CASES))
def test_cache_round_trip_expiry_and_corruption(name, tmp_path, monkeypatch):
    attr, load, save, payload, expected = CASES[name]
    mod = _mod(name)
    path = tmp_path / "cache.json"
    monkeypatch.setattr(mod, attr, path)
    assert getattr(mod, load)() == EMPTY[name]            # missing file
    getattr(mod, save)(*payload)
    assert getattr(mod, load)() == expected               # fresh
    data = json.loads(path.read_text())
    data["ts"] = time.time() - 10 * 86400
    path.write_text(json.dumps(data))
    assert getattr(mod, load)() == EMPTY[name]            # expired
    path.write_text("{not json")
    assert getattr(mod, load)() == EMPTY[name]            # corrupt
    path.write_text("[1, 2]")
    assert getattr(mod, load)() == EMPTY[name]            # valid JSON, wrong shape
    monkeypatch.setattr(mod, attr, tmp_path / "missing_dir" / "c.json")
    getattr(mod, save)(*payload)                           # unwritable location is tolerated


def test_shared_helpers_directly(tmp_path):
    if scan_common is None:
        pytest.skip("scan_common not present on this commit")
    p = tmp_path / "c.json"
    assert scan_common.load_json_cache(p, 60) is None
    scan_common.save_json_cache(p, {"x": 1})
    assert scan_common.load_json_cache(p, 60)["x"] == 1
    assert scan_common.load_json_cache(p, -1) is None
