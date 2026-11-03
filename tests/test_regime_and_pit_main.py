"""Characterisation of NSEDataFetcher.get_regime and pit_backtest.main with every external call faked."""
from __future__ import annotations

import sqlite3
import sys
import types

import numpy as np
import pandas as pd
import pytest
from _golden import check

import nse_data_fetcher as ndf
import pit_backtest


def _index_frame(n, drift, multi=False):
    idx = pd.bdate_range("2025-01-01", periods=n)
    close = 100 + np.cumsum(np.full(n, drift)) + np.sin(np.arange(n)) * 0.5
    df = pd.DataFrame({"Close": close}, index=idx)
    if multi:
        df.columns = pd.MultiIndex.from_tuples([("Close", "^NSEI")])
    return df


class _Yf:
    def __init__(self, frame):
        self.frame = frame

    def download(self, *a, **k):
        if isinstance(self.frame, Exception):
            raise self.frame
        return self.frame


def _fii(cols, vals):
    return pd.DataFrame({c: v for c, v in zip(cols, vals)})


SCENARIOS = {
    "bull_calm": dict(quote={"last": "25000", "yearHigh": "26000", "yearLow": "20000"}, frame=_index_frame(250, 0.4),
                      vix=13.2, fii=_fii(["FII Net", "x"], [[3e7, 1e7, 2e7], [1, 2, 3]])),
    "bull_volatile": dict(quote={"last": 1, "yearHigh": 0, "yearLow": 0}, frame=_index_frame(250, 0.4, multi=True),
                          vix=21.0, fii=_fii(["net buy"], [[-5.0, -1.0]])),
    "bear_extreme": dict(quote=None, frame=_index_frame(250, -0.4), vix=30.0, fii=pd.DataFrame()),
    "bear": dict(quote=RuntimeError("x"), frame=_index_frame(30, -0.5), vix=None, fii=RuntimeError("y")),
    "sideways": dict(quote={"last": 10, "yearHigh": 10, "yearLow": 5}, frame=_index_frame(250, 0.0),
                     vix=RuntimeError("v"), fii=_fii(["a"], [[1]])),
    "yf_empty_and_error": dict(quote={}, frame=pd.DataFrame(), vix=0, fii=None),
    "yf_raises": dict(quote={}, frame=RuntimeError("net"), vix=18.0, fii=None),
}


def _maybe(v):
    def f(*a, **k):
        if isinstance(v, Exception):
            raise v
        return v
    return f


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_get_regime(monkeypatch, name):
    sc = SCENARIOS[name]
    monkeypatch.setattr(ndf, "_NSEPY_OK", True)
    monkeypatch.setattr(ndf, "_YF_OK", True)
    monkeypatch.setattr(ndf, "yf", _Yf(sc["frame"]), raising=False)
    monkeypatch.setattr(ndf, "nse_get_index_quote", _maybe(sc["quote"]), raising=False)
    monkeypatch.setattr(ndf, "indiavix", _maybe(sc["vix"]), raising=False)
    monkeypatch.setattr(ndf, "nse_fiidii", _maybe(sc["fii"]), raising=False)
    res = ndf.NSEDataFetcher().get_regime()
    res["timestamp"] = "T"
    check(f"get_regime/{name}", res)


def test_get_regime_without_libs(monkeypatch):
    monkeypatch.setattr(ndf, "_NSEPY_OK", False)
    monkeypatch.setattr(ndf, "_YF_OK", False)
    res = ndf.NSEDataFetcher().get_regime()
    res["timestamp"] = "T"
    check("get_regime/no_libs", res)


def _ohlc(seed, n=330, base=50.0, vol=4e5):
    rng = np.random.RandomState(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    close = base * np.exp(np.cumsum(rng.normal(0.0008, 0.015, n)))
    return pd.DataFrame({"Close": close, "Volume": rng.uniform(0.5, 1.5, n) * vol}, index=idx)


def test_pit_backtest_main(monkeypatch, tmp_path, capsys):
    data = {f"T{i}": _ohlc(i, vol=1e5 * (i + 1)) for i in range(7)}
    data["BAD"] = pd.DataFrame({"x": [1, 2]})
    fake = types.ModuleType("market_store")
    fake.cached_download = lambda tickers, years=5: {t: data[t] for t in tickers if t in data}
    monkeypatch.setitem(sys.modules, "market_store", fake)
    monkeypatch.setattr(pit_backtest, "UNIVERSE", [*data])
    monkeypatch.setattr(pit_backtest, "piotroski_asof", lambda t, d: (int(t[1:]) % 4 + 5 if t != "BAD" else None, {}))
    monkeypatch.setattr(pit_backtest, "as_of", lambda t, d: {"shares": 1e6} if int(t[1:] or 0) % 2 else None)
    monkeypatch.setattr(pit_backtest, "coffeecan_asof", lambda t, d, mktcap=None: (int(t[1:]) % 3 == 0, {}))
    db = tmp_path / "pit.db"
    monkeypatch.setattr(sys, "argv", ["pit_backtest.py", "--years", "2", "--db", str(db), "--min-dollar-vol", "1.5e7"])
    pit_backtest.main()
    cap = capsys.readouterr()
    con = sqlite3.connect(db)
    rows = con.execute("select * from arm_summary").fetchall()
    con.close()
    check("pit_main", {"out": cap.out, "err": cap.err, "rows": rows})
