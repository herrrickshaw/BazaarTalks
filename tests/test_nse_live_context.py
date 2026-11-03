"""Characterisation of NSEDataFetcher's institutional score, strategy advice, live context and its printout."""
from __future__ import annotations

import pandas as pd
import pytest
from _golden import check

import nse_data_fetcher as ndf

REGIMES = ["BULL", "BULL_VOLATILE", "BEAR", "BEAR_EXTREME", "SIDEWAYS", "UNKNOWN"]


def test_strategy_recommendation_grid():
    out = {f"{r}/{v}/{s}": ndf._get_strategy_recommendation(r, v, s)
           for r in REGIMES for v in (None, 10.0, 15.0, 20.0, 30.0) for s in (-3, 0, 2)}
    check("nse/strategy_grid", out)


FII = {
    "buy": pd.DataFrame({"FII Net": [3e7, 2e7, 1e6], "DII Net": [-1e7, -2e7, 5.0]}),
    "sell": pd.DataFrame({"fii net": [-3e7], "dii net": [4.0]}),
    "no_cols": pd.DataFrame({"x": [1]}),
    "empty": pd.DataFrame(),
}
BULK = {
    "buys": pd.DataFrame({"Buy/Sell": ["BUY", "buy", "SELL", "BUY"]}),
    "sells": pd.DataFrame({"Buy/Sell": ["SELL", "sell", "BUY"]}),
    "even": pd.DataFrame({"Buy/Sell": ["BUY", "SELL"]}),
    "none": pd.DataFrame({"Other": [1]}),
    "empty": pd.DataFrame(),
}


@pytest.mark.parametrize("fii", sorted(FII))
@pytest.mark.parametrize("bulk", sorted(BULK))
def test_institutional_score(monkeypatch, fii, bulk):
    f = ndf.NSEDataFetcher()
    monkeypatch.setattr(f, "get_fii_dii", lambda: FII[fii])
    monkeypatch.setattr(f, "get_bulk_deals", lambda: BULK[bulk])
    check(f"nse/inst/{fii}/{bulk}", f.get_institutional_activity_score())


def test_institutional_score_survives_exceptions(monkeypatch):
    f = ndf.NSEDataFetcher()
    monkeypatch.setattr(f, "get_fii_dii", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(f, "get_bulk_deals", lambda: (_ for _ in ()).throw(RuntimeError("y")))
    check("nse/inst/errors", f.get_institutional_activity_score())


def test_get_live_context_and_print(monkeypatch, capsys):
    f = ndf.NSEDataFetcher()
    monkeypatch.setattr(f, "get_regime", lambda: {"regime": "BULL", "vix": 14.0, "nifty_last": 25000.0, "dma200": 23000.0,
                                                  "pct_from_dma200": 8.7, "pct_from_52wh": -1.2, "fii_sentiment": "BUYING",
                                                  "fii_net_cr": 12.5})
    monkeypatch.setattr(f, "get_market_breadth", lambda: {"vix": 99, "vix_regime": "NORMAL", "nifty_52w_low": 19000.0,
                                                          "nifty_52w_high": 26000.0})
    monkeypatch.setattr(f, "get_institutional_activity_score",
                        lambda: {"score": 2, "label": "BUY", "bulk_buy_ratio": 66.7})
    monkeypatch.setattr(f, "get_results_calendar",
                        lambda days_ahead=7: pd.DataFrame({"symbol": [f"S{i}" for i in range(10)]}))
    ctx = f.get_live_context()
    out = {"ctx": ctx}
    capsys.readouterr()
    f.print_live_context()
    out["printed"] = capsys.readouterr().out.replace(ndf.datetime.now().strftime("%d %b %Y"), "<DATE>")
    import re
    out["printed"] = re.sub(r"\d{2}:\d{2} IST", "<TIME> IST", out["printed"])
    check("nse/live_context", out)


def test_print_live_context_sparse(monkeypatch, capsys):
    f = ndf.NSEDataFetcher()
    monkeypatch.setattr(f, "get_live_context", lambda: {"regime": "UNKNOWN", "vix": None, "institutional_score": 0})
    f.print_live_context()
    import re
    check("nse/live_context_sparse", re.sub(r"\d{2} \w{3} \d{4} \d{2}:\d{2} IST", "<TS>", capsys.readouterr().out))


@pytest.mark.parametrize("vix", [None, 11.0, 15.0, 20.0, 25.0, 40.0])
def test_market_breadth(monkeypatch, vix):
    monkeypatch.setattr(ndf, "_NSEPY_OK", True)
    monkeypatch.setattr(ndf, "indiavix", lambda: vix, raising=False)
    monkeypatch.setattr(ndf, "nse_get_index_quote",
                        lambda n: {"last": "24000", "yearHigh": "26000", "yearLow": "20000"}, raising=False)
    check(f"nse/breadth/{vix}", ndf.NSEDataFetcher().get_market_breadth())


def _pe_df():
    return pd.DataFrame({"PE": [20.0, 21.5], "PB": [3.1, 3.2], "DivYield": [1.2, 1.3]},
                        index=pd.to_datetime(["2026-10-30", "2026-10-31"]))


def test_nifty_pe_pb_paths(monkeypatch, tmp_path):
    import sys
    import types
    monkeypatch.setattr(ndf, "CACHE_DIR", tmp_path)
    fake = types.ModuleType("nsepython")
    calls = []
    fake.index_pe_pb_div = lambda name, a, b: calls.append((name, len(a), len(b))) or _pe_df()
    monkeypatch.setitem(sys.modules, "nsepython", fake)
    f = ndf.NSEDataFetcher()
    out = {}
    monkeypatch.setattr(ndf, "_NSEPY_OK", False)
    out["no_lib"] = f.get_nifty_pe_pb()
    monkeypatch.setattr(ndf, "_NSEPY_OK", True)
    out["fetched"] = f.get_nifty_pe_pb()
    out["file_written"] = (tmp_path / "nifty_pe_pb.json").exists()
    fake.index_pe_pb_div = lambda *a: (_ for _ in ()).throw(RuntimeError("down"))
    out["from_cache"] = f.get_nifty_pe_pb()                    # fresh cache file wins over a broken source
    (tmp_path / "nifty_pe_pb.json").unlink()
    out["source_error"] = f.get_nifty_pe_pb()
    fake.index_pe_pb_div = lambda *a: pd.DataFrame()
    out["empty_df"] = f.get_nifty_pe_pb()
    out["calls"] = calls
    check("nse/pe_pb", out)
