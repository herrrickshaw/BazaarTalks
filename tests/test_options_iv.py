"""Offline tests for options_iv.py (chain aggregators + faked yfinance fetch)."""
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import options_iv as oi  # noqa: E402


def _chain(strikes, ivs, oi_=None, vol=None):
    d = {"strike": strikes, "impliedVolatility": ivs}
    if oi_ is not None:
        d["openInterest"] = oi_
    if vol is not None:
        d["volume"] = vol
    return pd.DataFrame(d)


CALLS = _chain([90, 100, 110, 120], [0.30, 0.25, 0.22, 0.20], [10, 20, 30, 40], [1, 2, 3, 4])
PUTS = _chain([80, 90, 100, 110], [0.45, 0.35, 0.27, 0.24], [50, 60, 70, 20], [5, 5, 5, 5])


def test_atm_iv_averages_nearest_strikes():
    assert oi.atm_iv(CALLS, PUTS, 100.0) == pytest.approx((0.25 + 0.27) / 2)
    # spot 104 -> calls nearest 100 (0.25); puts nearest 100 (|100-104|=4 vs 110:6) (0.27)
    assert oi.atm_iv(CALLS, PUTS, 104.0) == pytest.approx(0.26)
    # spot 106 -> calls nearest 110 (0.22), puts nearest 110 (0.24)
    assert oi.atm_iv(CALLS, PUTS, 106.0) == pytest.approx(0.23)


def test_atm_iv_tie_picks_first_strike():
    calls = _chain([95, 105], [0.10, 0.50])
    assert oi.atm_iv(calls, None, 100.0) == pytest.approx(0.10)    # idxmin keeps first


def test_atm_iv_skips_bad_sides():
    assert oi.atm_iv(CALLS, None, 100.0) == pytest.approx(0.25)
    assert oi.atm_iv(None, PUTS, 100.0) == pytest.approx(0.27)
    assert oi.atm_iv(CALLS, pd.DataFrame(), 100.0) == pytest.approx(0.25)
    assert oi.atm_iv(CALLS, pd.DataFrame({"strike": [100]}), 100.0) == pytest.approx(0.25)  # no IV col
    zero = _chain([100], [0.0])
    nan = _chain([100], [np.nan])
    assert oi.atm_iv(zero, nan, 100.0) != oi.atm_iv(zero, nan, 100.0)   # both rejected -> NaN
    assert oi.atm_iv(CALLS, zero, 100.0) == pytest.approx(0.25)         # zero IV excluded, not averaged
    assert np.isnan(oi.atm_iv(None, None, 100.0))


def test_put_call_ratio_by_open_interest_and_volume():
    assert oi.put_call_ratio(CALLS, PUTS) == pytest.approx(200 / 100)
    assert oi.put_call_ratio(CALLS, PUTS, field="volume") == pytest.approx(20 / 10)


def test_put_call_ratio_degenerate_denominators():
    zero_calls = _chain([100], [0.2], [0])
    assert np.isnan(oi.put_call_ratio(zero_calls, PUTS))
    assert np.isnan(oi.put_call_ratio(CALLS, _chain([100], [0.2])))      # puts lack the field
    assert np.isnan(oi.put_call_ratio(_chain([100], [0.2]), PUTS))       # calls lack the field
    assert oi.put_call_ratio(CALLS, _chain([100], [0.2], [0])) == 0.0    # no puts -> 0 ratio
    nan_calls = _chain([100], [0.2], [np.nan])
    assert np.isnan(oi.put_call_ratio(nan_calls, PUTS))


def test_iv_skew_positive_when_puts_richer():
    # spot 100, moneyness 10%: put target 90 -> 0.35 ; call target 110 -> 0.22
    assert oi.iv_skew(CALLS, PUTS, 100.0) == pytest.approx(0.35 - 0.22)


def test_iv_skew_custom_moneyness_and_nearest_snap():
    # 20%: put target 80 -> 0.45 ; call target 120 -> 0.20
    assert oi.iv_skew(CALLS, PUTS, 100.0, moneyness=0.20) == pytest.approx(0.25)
    # 5%: put target 95 (tie 90/100 -> first, 90 -> 0.35); call target 105 (tie 100/110 -> first, 100 -> 0.25)
    assert oi.iv_skew(CALLS, PUTS, 100.0, moneyness=0.05) == pytest.approx(0.10)


def test_iv_skew_nan_when_a_side_missing():
    assert np.isnan(oi.iv_skew(None, PUTS, 100.0))
    assert np.isnan(oi.iv_skew(CALLS, pd.DataFrame(), 100.0))
    assert np.isnan(oi.iv_skew(CALLS, _chain([90], [np.nan]), 100.0))


def test_iv_skew_negative_when_calls_richer():
    calls = _chain([110], [0.40])
    puts = _chain([90], [0.30])
    assert oi.iv_skew(calls, puts, 100.0) == pytest.approx(-0.10)


def test_chain_metrics_rounding_and_keys():
    m = oi.chain_metrics(CALLS, PUTS, 100.0)
    assert m == {"atm_iv%": 26.0, "put_call": 2.0, "iv_skew%": 13.0}


def test_chain_metrics_nan_branches():
    empty = pd.DataFrame(columns=["strike", "impliedVolatility", "openInterest"])
    m = oi.chain_metrics(empty, empty, 100.0)
    assert np.isnan(m["atm_iv%"]) and np.isnan(m["iv_skew%"]) and np.isnan(m["put_call"])


# ── fetch (faked yfinance + apiclient) ───────────────────────────────────────
def _install(monkeypatch, options=("2026-12-18",), spot=100.0, chain=(CALLS, PUTS), raises=None):
    api = types.ModuleType("apiclient")
    api.calls = []

    def robust(src, fn, retries=5):
        api.calls.append((src, retries))
        return fn()
    api.robust = robust
    yf = types.ModuleType("yfinance")
    seen = {}

    class Ticker:
        def __init__(self, t):
            seen["ticker"] = t
            self.fast_info = {"lastPrice": spot}

        @property
        def options(self):
            if raises:
                raise raises
            return options

        def option_chain(self, exp):
            seen["exp"] = exp
            return types.SimpleNamespace(calls=chain[0], puts=chain[1])
    yf.Ticker = Ticker
    monkeypatch.setitem(sys.modules, "apiclient", api)
    monkeypatch.setitem(sys.modules, "yfinance", yf)
    return api, seen


def test_fetch_chain_uses_nearest_expiry(monkeypatch):
    api, seen = _install(monkeypatch, options=("2026-12-18", "2027-01-15"))
    calls, puts, spot = oi.fetch_chain_yf("AAPL")
    assert calls is CALLS and puts is PUTS and spot == 100.0
    assert seen == {"ticker": "AAPL", "exp": "2026-12-18"}
    assert [c[0] for c in api.calls] == ["yfinance"] * 3 and all(c[1] == 2 for c in api.calls)


@pytest.mark.parametrize("opts", [(), None])
def test_fetch_chain_no_expiries(monkeypatch, opts):
    _install(monkeypatch, options=opts)
    assert oi.fetch_chain_yf("X") == (None, None, None)


def test_fetch_chain_missing_spot_is_nan(monkeypatch):
    _install(monkeypatch, spot=None)
    calls, puts, spot = oi.fetch_chain_yf("X")
    assert calls is CALLS and np.isnan(spot)


def test_fetch_chain_error_degrades(monkeypatch):
    _install(monkeypatch, raises=RuntimeError("offline"))
    assert oi.fetch_chain_yf("X") == (None, None, None)


# ── main ──────────────────────────────────────────────────────────────────────
def test_main_sorts_by_skew_and_skips_failed_fetches(monkeypatch, capsys):
    flat_puts = _chain([90, 100], [0.20, 0.25], [10, 10])
    flat_calls = _chain([100, 110], [0.25, 0.20], [10, 10])
    fetched = {"AAA": (CALLS, PUTS, 100.0),                 # skew +13.0
               "BBB": (flat_calls, flat_puts, 100.0),       # skew 0.0
               "CCC": (None, None, None),
               "DDD": (CALLS, PUTS, np.nan)}
    monkeypatch.setattr(oi, "fetch_chain_yf", lambda t: fetched[t])
    monkeypatch.setattr(sys, "argv", ["options_iv.py", "--tickers", "aaa, bbb,ccc,ddd,"])
    oi.main()
    out = capsys.readouterr().out
    assert "— 2 names" in out
    rows = [l.split() for l in out.splitlines() if l.strip().startswith(("AAA", "BBB"))]
    assert [r[0] for r in rows] == ["AAA", "BBB"]
    assert rows[0] == ["AAA", "100.00", "26.0", "2.0", "13.0"]
    assert rows[1][-1] == "0.0"


def test_main_offline_message(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(oi, "fetch_chain_yf", lambda t: seen.append(t) or (None, None, None))
    monkeypatch.setattr(sys, "argv", ["options_iv.py"])
    oi.main()
    assert seen == ["AAPL", "NVDA", "SPY", "MSFT", "TSLA"]
    assert "no option chains fetched" in capsys.readouterr().out
