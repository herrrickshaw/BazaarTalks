"""Import the full_*_market_scan modules without network or optional market libraries.

The scanners ``sys.exit`` at import time when bseindia / pykrx are absent; tests only need their
pure functions, so a missing third-party package is replaced by an inert stub (never when installed).
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import types


def _installed(name: str) -> bool:
    return name in sys.modules or importlib.util.find_spec(name) is not None


def _install_stubs() -> None:
    if not _installed("bseindia"):
        bse = types.ModuleType("bseindia")
        lib = types.ModuleType("bseindia.libutil")
        lib.SECURITY_MASTER_URL = ""
        bse.libutil = lib
        sys.modules["bseindia"], sys.modules["bseindia.libutil"] = bse, lib
    if not _installed("pykrx"):
        krx = types.ModuleType("pykrx")
        krx.stock = types.ModuleType("pykrx.stock")
        sys.modules["pykrx"] = krx


def import_scanner(market: str):
    """Return ``full_<market>_market_scan`` (market in us/indian/japan/korea/european)."""
    _install_stubs()
    return importlib.import_module(f"full_{market}_market_scan")
