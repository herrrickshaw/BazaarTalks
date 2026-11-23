"""Offline tests for liquidity_multiyear.py (Fama-MacBeth liquidity-premium test)."""
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import liquidity_multiyear as lm  # noqa: E402
import marketdata  # noqa: E402


def _panel(n_dates=8, n_per=15, slope=1.0, noise=0.0, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for d in range(n_dates):
        for i in range(n_per):
            il = float(i + 1)
            rows.append({"date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=90 * d),
                         "ticker": f"T{i}", "illiq": il,
                         "fwd_ret": slope * il / 1000.0 + noise * rng.randn()})
    return pd.DataFrame(rows)


# ── fama_macbeth_quintiles ────────────────────────────────────────────────────
def test_fm_clear_premium():
    res = lm.fama_macbeth_quintiles(_panel(noise=0.002))
    assert res["n_dates"] == 8
    q = res["quintile_fwd%"]
    assert q == sorted(q) and len(q) == 5
    assert res["Q5_minus_Q1%"] == pytest.approx(q[-1] - q[0], abs=0.011)
    assert res["Q5_minus_Q1%"] > 0
    assert res["t_stat"] > 2
    assert res["monotonicity"] == 1.0
    assert res["avg_IC"] > 0.9


def test_fm_inverted_premium_has_negative_stats():
    res = lm.fama_macbeth_quintiles(_panel(slope=-1.0, noise=0.002))
    assert res["Q5_minus_Q1%"] < 0 and res["t_stat"] < -2
    assert res["monotonicity"] == -1.0 and res["avg_IC"] < -0.9


def test_fm_exact_quintile_means_without_noise():
    res = lm.fama_macbeth_quintiles(_panel())
    # 15 names / 5 quintiles = 3 each: mean illiq 2,5,8,11,14 -> /1000 -> %
    assert res["quintile_fwd%"] == [0.2, 0.5, 0.8, 1.1, 1.4]
    assert res["Q5_minus_Q1%"] == 1.2
    assert np.isnan(res["t_stat"])                 # zero-variance spread -> undefined t


def test_fm_too_few_dates_returns_only_count():
    assert lm.fama_macbeth_quintiles(_panel(n_dates=4)) == {"n_dates": 4}
    assert lm.fama_macbeth_quintiles(_panel(n_dates=5, noise=0.001))["n_dates"] == 5   # boundary


def test_fm_skips_small_cross_sections_and_nans():
    p = _panel(n_dates=6, noise=0.001)
    small = p[p["date"] == p["date"].iloc[0]].head(14)          # one date reduced to 14 < q*3
    p2 = pd.concat([p[p["date"] != p["date"].iloc[0]], small])
    assert lm.fama_macbeth_quintiles(p2)["n_dates"] == 5
    p3 = p.copy()
    p3.loc[p3["date"] == p3["date"].iloc[0], "illiq"] = np.nan  # whole date NaN -> skipped
    assert lm.fama_macbeth_quintiles(p3)["n_dates"] == 5


def test_fm_ties_that_collapse_buckets_are_skipped():
    p = _panel(n_dates=6, noise=0.001)
    first = p["date"] == p["date"].iloc[0]
    p.loc[first, "illiq"] = 1.0                    # all tied -> qcut cannot form 5 bins
    out = lm.fama_macbeth_quintiles(p)
    assert out["n_dates"] == 5


def test_fm_ic_nan_when_fwd_constant():
    p = _panel(n_dates=6)
    p["fwd_ret"] = 0.01
    out = lm.fama_macbeth_quintiles(p)
    assert np.isnan(out["avg_IC"])
    assert out["quintile_fwd%"] == [1.0] * 5


def test_fm_custom_q():
    res = lm.fama_macbeth_quintiles(_panel(n_per=12, noise=0.001), q=3)
    assert len(res["quintile_fwd%"]) == 3 and res["monotonicity"] == 1.0


# ── universe / build_panel ────────────────────────────────────────────────────
def test_universe_missing_cache(monkeypatch):
    monkeypatch.setattr(marketdata, "close_volume", lambda m: (None, None))
    assert lm.universe(5) == []


def test_universe_truncates_to_n(monkeypatch):
    close = pd.DataFrame(np.ones((10, 3)), columns=list("ABC"))
    monkeypatch.setattr(marketdata, "close_volume", lambda m: (close, close))
    monkeypatch.setattr(marketdata, "liquid_symbols", lambda c, v: ["C", "B", "A"])
    assert lm.universe(2) == ["C", "B"]
    assert lm.universe(10) == ["C", "B", "A"]


def _px(n=450, seed=0, vol=True, start=100.0):
    rng = np.random.RandomState(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = start * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    df = pd.DataFrame({"Close": close}, index=idx)
    if vol:
        df["Volume"] = rng.uniform(1e5, 2e5, n)
    return df


def test_build_panel_rows_and_forward_return():
    df = _px()
    out = lm.build_panel({"AAA": df}, years=2)
    Ts = list(range(252, 450 - 63, 63))             # 252, 315, 378
    assert len(out) == len(Ts) == 3
    assert out["date"].tolist() == [df.index[t] for t in Ts]
    c = df["Close"].to_numpy()
    assert out["fwd_ret"].tolist() == pytest.approx([c[t + 63] / c[t] - 1 for t in Ts])
    assert (out["illiq"] > 0).all() and set(out["ticker"]) == {"AAA"}


def test_build_panel_skips_short_none_and_missing_cols():
    short = _px(n=252 + 63 + 4)                       # one bar short of lookback+horizon+5
    ok = _px(n=252 + 63 + 5)
    out = lm.build_panel({"S": short, "N": None, "NC": pd.DataFrame({"Open": [1.0] * 400}),
                          "OK": ok}, years=1)
    assert set(out["ticker"]) == {"OK"} and len(out) == 1


def test_build_panel_without_volume_yields_nothing():
    out = lm.build_panel({"X": _px(vol=False)}, years=2)
    assert out.empty


def test_build_panel_skips_bad_prices_and_honours_params():
    df = _px()
    df.iloc[252, 0] = np.nan                           # NaN start price at first T
    out = lm.build_panel({"A": df}, years=2)
    assert df.index[252] not in set(out["date"]) and len(out) == 2
    out2 = lm.build_panel({"A": _px()}, years=2, lookback=100, horizon=20, step=100)
    Ts = list(range(100, 450 - 20, 100))
    assert len(out2) == len(Ts)


# ── CLI ───────────────────────────────────────────────────────────────────────
def _run_main(monkeypatch, fake_prices, *argv, universe=("AAA",)):
    calls = {}

    def yf_download(tk, period):
        calls["tk"], calls["period"] = tk, period
        return fake_prices

    monkeypatch.setitem(sys.modules, "apiclient", types.SimpleNamespace(yf_download=yf_download))
    monkeypatch.setattr(lm, "universe", lambda n: list(universe)[:n])
    monkeypatch.setattr(sys, "argv", ["liquidity_multiyear.py", *argv])
    lm.main()
    return calls


def test_main_insufficient_cross_sections(monkeypatch, capsys):
    calls = _run_main(monkeypatch, {"AAA": _px()}, "--n", "5", "--years", "3", "--horizon", "63")
    cap = capsys.readouterr()
    assert calls == {"tk": ["AAA"], "period": "3y"}
    assert "fetching 1 US tickers, 3y daily" in cap.err and "got 1 price series" in cap.err
    assert "MULTI-YEAR (3y)" in cap.out
    assert "insufficient cross-sections" in cap.out


def test_main_full_report_significant(monkeypatch, capsys):
    rng = np.random.RandomState(3)
    prices = {}
    for i in range(20):
        n = 252 + 63 * 8 + 70
        idx = pd.bdate_range("2010-01-01", periods=n)
        # illiquid names (small volume) get a higher drift -> positive premium
        drift = 0.0002 + 0.0015 * (19 - i) / 19.0
        close = 50 * np.cumprod(1 + drift + rng.normal(0, 0.004, n))
        prices[f"T{i}"] = pd.DataFrame({"Close": close,
                                        "Volume": np.full(n, 1e4 * (i + 1) ** 3)}, index=idx)
    _run_main(monkeypatch, prices, "--years", "10", universe=tuple(prices))
    out = capsys.readouterr().out
    assert "20 names · 20 in panel" in out
    assert "quarterly cross-sections" in out
    assert "mean 63d fwd return by ILLIQ quintile" in out
    assert "Q1=" in out and "Q5=" in out
    assert "Q5−Q1 =" in out and "avg IC =" in out
    assert "-> SIGNIFICANT liquidity premium (illiquid earn more, |t|>2)" in out


def test_main_not_significant_branch(monkeypatch, capsys):
    rng = np.random.RandomState(5)
    prices = {}
    for i in range(20):
        n = 252 + 63 * 8 + 70
        idx = pd.bdate_range("2010-01-01", periods=n)
        close = 50 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
        prices[f"T{i}"] = pd.DataFrame({"Close": close, "Volume": rng.uniform(1e5, 2e5, n)}, index=idx)
    _run_main(monkeypatch, prices, universe=tuple(prices))
    out = capsys.readouterr().out
    assert "-> not significant (illiquid earn more, |t|>2)" in out


def test_main_filters_extreme_forward_returns(monkeypatch, capsys):
    df = _px()
    df.iloc[252 + 63:, 0] = df["Close"].iloc[252 + 63:] * 10          # +900% jump inside window
    _run_main(monkeypatch, {"AAA": df}, "--years", "2")
    out = capsys.readouterr().out
    assert "1 names" in out and "stock-obs" in out
