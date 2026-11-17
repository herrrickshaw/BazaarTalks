"""
Regression tests for findings of the 2026-10-25 static-analysis sweep (ruff/mypy).
Offline, deterministic; nothing is written outside tmp_path.
"""
import importlib
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _Sentinel(Exception):
    """Raised by the patched marketdata.market_list so main() stops right there."""


# ── F821: five CLI entry points call `marketdata.market_list()` without binding
#    the module name (they only `from marketdata import ...`), so `--all` (and the
#    default "no --market") path dies with NameError before doing any work.
@pytest.mark.parametrize("modname", [
    "accumulation_screener", "corporate_actions", "earnings_liquidity",
    "liquidity_factor", "pead_factor",
])
def test_cli_all_markets_resolves_marketdata(monkeypatch, modname):
    import marketdata
    mod = importlib.import_module(modname)

    def _boom():
        raise _Sentinel

    monkeypatch.setattr(marketdata, "market_list", _boom)
    monkeypatch.setattr(sys, "argv", [modname, "--all"])
    # Before the fix this raises NameError (not _Sentinel): `marketdata` is undefined.
    with pytest.raises(_Sentinel):
        mod.main()


# ── RET503: MarketCache.get_ohlc returned None (not a DataFrame) when the parquet
#    file exists but the meta index has no "to" date for it (meta lost / rebuilt).
def test_get_ohlc_with_parquet_but_no_meta_date_returns_dataframe(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))          # module creates dirs under ~ on import
    sys.modules.pop("market_data_cache", None)
    mdc = importlib.import_module("market_data_cache")
    monkeypatch.setattr(mdc, "OHLC_DIR", tmp_path / "ohlc")
    monkeypatch.setattr(mdc, "META_FILE", tmp_path / "meta.json")
    mdc.OHLC_DIR.mkdir()

    idx = pd.to_datetime(["2026-01-05", "2026-01-06"])
    existing = pd.DataFrame({"Open": [1.0, 2.0], "High": [1.0, 2.0], "Low": [1.0, 2.0],
                             "Close": [1.0, 2.0], "Volume": [10, 20]}, index=idx)
    existing.to_parquet(mdc.OHLC_DIR / "ABC.parquet")

    cache = mdc.MarketCache(verbose=False)
    cache._meta = {"ohlc:ABC": {"updated": "2000-01-01T00:00:00"}}   # stale, no "to"
    monkeypatch.setattr(cache, "_yf_download_single", lambda ticker, start: existing.copy())

    out = cache.get_ohlc("ABC")
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 2
