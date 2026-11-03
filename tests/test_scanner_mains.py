"""Characterisation of the five full_*_market_scan modules' main() and fundamental_scan().

Everything external is faked (universe lists, bulk OHLC, yfinance Ticker, symbol caches); the temp dir
receives the Excel output, which is read back and compared. Goldens were captured before the pass-2 split.
"""
from __future__ import annotations

import sys
from concurrent.futures import Future

import pandas as pd
import pytest
from _golden import check
from _scan_fakes import FakeTicker, KINDS, ohlc_panel, read_workbooks, scrub  # noqa: F401
from _scanner_import import import_scanner

import yfinance


class _SyncPool:
    """ThreadPoolExecutor stand-in: runs jobs inline, in submission order."""

    def __init__(self, max_workers=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def submit(self, fn, *a, **k):
        f = Future()
        try:
            f.set_result(fn(*a, **k))
        except BaseException as e:  # noqa: BLE001 - mirrors a worker exception
            f.set_exception(e)
        return f


def _ordered(fs):
    return iter(list(fs))


def _patch_pool(monkeypatch, mod):
    """Make the stage-4 pool deterministic in the module itself or in scan_common (after the split)."""
    holders = [mod]
    try:
        import scan_common
        holders.append(scan_common)
    except ImportError:
        pass
    for h in holders:
        if hasattr(h, "ThreadPoolExecutor"):
            monkeypatch.setattr(h, "ThreadPoolExecutor", _SyncPool)
        if hasattr(h, "as_completed"):
            monkeypatch.setattr(h, "as_completed", _ordered)


@pytest.fixture(autouse=True)
def _fake_yf(monkeypatch):
    monkeypatch.setattr(yfinance, "Ticker", FakeTicker)


SYMS = [f"S{i}" for i in range(10)]


def _run(mod, fn, tmp_path, capsys, *a, **k):
    ret = fn(*a, **k)
    out = scrub(capsys.readouterr().out, tmp_path)
    if isinstance(ret, dict) and "excel_path" in ret:
        ret = {**ret, "excel_path": scrub(ret["excel_path"], tmp_path)}
    return {"ret": ret, "out": out, "books": read_workbooks(tmp_path)}


# ── fundamental_scan ───────────────────────────────────────────────────────────

FUND_SYMS = [f"F{i}" for i in range(12)]


@pytest.mark.parametrize("market,args", [("us", ()), ("indian", (".NS",)), ("japan", ()), ("korea", (".KS",)),
                                         ("european", ())])
def test_fundamental_scan_matches_golden(market, args):
    mod = import_scanner(market)
    out = {s: mod.fundamental_scan(s, *args) for s in FUND_SYMS}
    check(f"fundamental_scan/{market}", out)


# ── mains ──────────────────────────────────────────────────────────────────────

def _setup_us(mod, mp, tmp_path, cached):
    mp.setattr(mod, "DOWNLOAD_DIR", tmp_path)
    syms = SYMS
    if cached:
        mp.setattr(mod, "_load_symbol_cache", lambda: (syms[:7], syms[5:]))
    else:
        mp.setattr(mod, "_load_symbol_cache", lambda: (None, None))
        mp.setattr(mod, "fetch_nasdaq_symbols", lambda: syms[:7])
        mp.setattr(mod, "fetch_nyse_symbols", lambda: syms[5:])
    mp.setattr(mod, "_save_symbol_cache", lambda a, b: None)
    mp.setattr(mod, "bulk_download_ohlc", lambda tickers, period="1y": {t: d for t, d in ohlc_panel(tickers).items()})
    frame = pd.DataFrame({"Close": [float(i) for i in range(1, 260)]})
    mp.setattr(yfinance, "download", lambda *a, **k: frame)
    mp.setattr(mod, "_NSE_OK", True)


class _FakeML:
    def predict_batch(self, ohlc, max_workers=1):
        rows = [{"symbol": s, "direction": ["BULLISH", "BEARISH", "NEUTRAL"][i % 3],
                 "predicted_ret%": i * 0.5, "confidence": 0.1 * i, "train_rmse": 1.0 + i}
                for i, s in enumerate(ohlc)]
        return pd.DataFrame(rows)


@pytest.mark.parametrize("scn", ["cached_ml", "fetched_noml_noscans", "nasdaq_only_top"])
def test_us_main(monkeypatch, tmp_path, capsys, scn):
    mod = import_scanner("us")
    _patch_pool(monkeypatch, mod)
    _setup_us(mod, monkeypatch, tmp_path, cached=(scn != "fetched_noml_noscans"))
    monkeypatch.setattr(mod, "_ML_OK", scn == "cached_ml")
    monkeypatch.setattr(mod, "_ML_ENGINE", _FakeML(), raising=False)
    kw = {"cached_ml": dict(workers=1), "fetched_noml_noscans": dict(run_scans=False, workers=1),
          "nasdaq_only_top": dict(nasdaq_only=True, top=4, workers=1, min_price=0.001)}[scn]
    check(f"main/us/{scn}", _run(mod, mod.main, tmp_path, capsys, **kw))


def _setup_india(mod, mp, tmp_path, cached):
    mp.setattr(mod, "DOWNLOAD_DIR", tmp_path)
    nse, bse = SYMS[:7], ["B1", "B2", "B3"]
    if cached:
        mp.setattr(mod, "_load_symbol_cache", lambda: (nse, bse))
    else:
        mp.setattr(mod, "_load_symbol_cache", lambda: (None, None))
        mp.setattr(mod, "fetch_nse_symbols", lambda: nse)
        mp.setattr(mod, "fetch_bse_only_symbols", lambda s: bse)
    mp.setattr(mod, "_save_symbol_cache", lambda a, b: None)
    mp.setattr(mod, "bulk_download_ohlc", lambda tickers, period="1y": ohlc_panel(tickers))

    class Fetcher:
        def print_live_context(self):
            print("LIVE CONTEXT")

    mp.setattr(mod, "_NSE_FETCHER", Fetcher())
    mp.setattr(mod, "_USE_NSE_FETCHER", True)


@pytest.mark.parametrize("scn", ["cached", "fetched_noscans", "nse_only_top"])
def test_indian_main(monkeypatch, tmp_path, capsys, scn):
    mod = import_scanner("indian")
    _patch_pool(monkeypatch, mod)
    _setup_india(mod, monkeypatch, tmp_path, cached=(scn != "fetched_noscans"))
    kw = {"cached": dict(workers=1), "fetched_noscans": dict(run_scans=False),
          "nse_only_top": dict(nse_only=True, top=5, workers=1)}[scn]
    check(f"main/indian/{scn}", _run(mod, mod.main, tmp_path, capsys, **kw))


def _universe(n, suffix=".T"):
    return [{"yf_ticker": f"{1000 + i}{suffix}", "code": str(1000 + i), "name": f"Name {i}", "sector": "Tech"}
            for i in range(n)]


@pytest.mark.parametrize("scn", ["default", "cap2", "noscans_top"])
def test_japan_main(monkeypatch, tmp_path, capsys, scn):
    mod = import_scanner("japan")
    _patch_pool(monkeypatch, mod)
    monkeypatch.setattr(mod, "DOWNLOAD_DIR", tmp_path)
    monkeypatch.setattr(mod, "build_ticker_universe", lambda: _universe(10))
    monkeypatch.setattr(mod, "bulk_download_ohlc", lambda tickers, period="3mo": ohlc_panel(tickers))
    argv = {"default": [], "cap2": [], "noscans_top": ["--no-scans", "--top", "6"]}[scn]
    if scn == "cap2":
        monkeypatch.setattr(mod, "MAX_FUND_CANDIDATES", 2)
    monkeypatch.setattr(sys, "argv", ["full_japan_market_scan.py", "--workers", "1", *argv])
    check(f"main/japan/{scn}", _run(mod, mod.main, tmp_path, capsys))


def _krx(n):
    return [{"code": f"{5000 + i:06d}", "name": f"K{i}", "market": "KOSPI" if i % 3 else "KOSDAQ",
             "yf_suffix": ".KS" if i % 3 else ".KQ"} for i in range(n)]


@pytest.mark.parametrize("scn", ["default", "cap2", "kospi_noscans_top"])
def test_korea_main(monkeypatch, tmp_path, capsys, scn):
    mod = import_scanner("korea")
    _patch_pool(monkeypatch, mod)
    monkeypatch.setattr(mod, "DOWNLOAD_DIR", tmp_path)
    monkeypatch.setattr(mod, "build_krx_universe", lambda kospi_only=False: _krx(10))

    def fake_ohlc(universe, months=3):
        panel = ohlc_panel([s["code"] for s in universe])
        panel.pop(universe[-1]["code"], None)        # one ticker without data
        return panel

    monkeypatch.setattr(mod, "fetch_krx_ohlc", fake_ohlc)
    argv = {"default": [], "cap2": [], "kospi_noscans_top": ["--kospi-only", "--no-scans", "--top", "6"]}[scn]
    if scn == "cap2":
        monkeypatch.setattr(mod, "MAX_FUND_CANDIDATES", 2)
    monkeypatch.setattr(sys, "argv", ["full_korea_market_scan.py", "--workers", "1", *argv])
    check(f"main/korea/{scn}", _run(mod, mod.main, tmp_path, capsys))


@pytest.mark.parametrize("scn", ["default", "noscans_top", "download_fails"])
def test_european_main(monkeypatch, tmp_path, capsys, scn):
    mod = import_scanner("european")
    _patch_pool(monkeypatch, mod)
    monkeypatch.setattr(mod, "DOWNLOAD_DIR", tmp_path)
    meta = {f"E{i}.DE": (f"Euro {i}", "Industrials") for i in range(10)}
    monkeypatch.setattr(mod, "EURO_STOXX_50_META", meta)

    def fake_download(symbols, **kw):
        if scn == "download_fails":
            raise RuntimeError("no network")
        return pd.concat(ohlc_panel(list(symbols)), axis=1)

    monkeypatch.setattr(yfinance, "download", fake_download)
    argv = {"default": [], "noscans_top": ["--no-scans", "--top", "6"], "download_fails": []}[scn]
    monkeypatch.setattr(sys, "argv", ["full_european_market_scan.py", *argv])
    if scn == "download_fails":
        with pytest.raises(SystemExit) as ei:
            mod.main()
        check("main/european/download_fails", {"exit": str(ei.value), "out": scrub(capsys.readouterr().out, tmp_path)})
        return
    check(f"main/european/{scn}", _run(mod, mod.main, tmp_path, capsys))
