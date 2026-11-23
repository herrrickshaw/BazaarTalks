"""Offline tests for peer_network.py (peer lead-lag signal, panel build, CLI)."""
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import peer_network as pn  # noqa: E402


# ── basket_return ─────────────────────────────────────────────────────────────
def _mp():
    return pd.DataFrame({"A": [100.0, 110.0, 121.0], "B": [50.0, 50.0, 100.0]})


def test_basket_return_equal_weight_mean_of_member_returns():
    assert pn.basket_return(_mp(), 0, 2) == pytest.approx((0.21 + 1.0) / 2)
    assert pn.basket_return(_mp(), 1, 2) == pytest.approx((0.10 + 1.0) / 2)


def test_basket_return_bounds():
    mp = _mp()
    assert np.isnan(pn.basket_return(mp, 0, 3))              # end == len -> out of range
    assert not np.isnan(pn.basket_return(mp, 0, 2))          # end == len-1 ok
    assert np.isnan(pn.basket_return(mp, -1, 2))
    assert pn.basket_return(mp, 0, 0) == 0.0                 # start == end allowed
    assert np.isnan(pn.basket_return(mp.iloc[:, :0], 0, 1))  # no members


def test_basket_return_skips_bad_members():
    mp = pd.DataFrame({"A": [100.0, 110.0], "NANSTART": [np.nan, 5.0], "ZERO": [0.0, 5.0],
                       "NEG": [-1.0, 5.0], "NANEND": [10.0, np.nan], "INF": [10.0, np.inf]})
    assert pn.basket_return(mp, 0, 1) == pytest.approx(0.10)


def test_basket_return_all_members_bad_is_nan():
    mp = pd.DataFrame({"A": [0.0, 1.0], "B": [np.nan, 1.0]})
    assert np.isnan(pn.basket_return(mp, 0, 1))


# ── peer_signal ───────────────────────────────────────────────────────────────
def test_peer_signal_passthrough_and_nonfinite():
    assert pn.peer_signal(0.25) == 0.25
    assert pn.peer_signal(-0.5) == -0.5
    assert np.isnan(pn.peer_signal(np.nan))
    assert np.isnan(pn.peer_signal(np.inf))
    assert np.isnan(pn.peer_signal(-np.inf))
    assert isinstance(pn.peer_signal(np.float32(0.5)), float)


# ── build_panel ───────────────────────────────────────────────────────────────
N = 60


def _px(growth, base=100.0):
    return base * (1.0 + growth) ** np.arange(N)


def _wide():
    idx = pd.bdate_range("2024-01-02", periods=N)
    cols = {"A1.NS": _px(0.010), "A2.NS": _px(0.020), "A3.NS": _px(0.000), "A4.NS": _px(-0.010),
            "B1": _px(0.01), "B2": _px(0.02), "B3": _px(0.03),            # 3 members -> skipped
            "NOIND": _px(0.01), "NANIND": _px(0.01), "ZZ": _px(0.01)}
    return pd.DataFrame(cols, index=idx)


@pytest.fixture
def env(tmp_path, monkeypatch):
    comp_path = tmp_path / "companies_industry.parquet"
    comp_path.write_bytes(b"x")
    monkeypatch.setattr(pn, "COMPANIES", str(comp_path))
    comp = pd.DataFrame({
        "ticker": ["A1", "A2", "A3.BO", "A4", "B1", "B2", "B3", "NANIND", "ZZ"],
        "industry": ["Auto"] * 4 + ["Bio"] * 3 + [np.nan, ""],
        "extra": range(9)})
    monkeypatch.setattr(pn.pd, "read_parquet", lambda p, *a, **k: comp.copy())
    wide = {"v": (_wide(), None)}
    lf = types.ModuleType("liquidity_factor")
    lf._market_wide = lambda m: (wide["v"][0], wide["v"][0] * 0 + 1) if wide["v"] is not None else None
    pfm = types.ModuleType("pead_factor")
    pfm._liquid_symbols = lambda close, vol: list(close.columns)
    monkeypatch.setitem(sys.modules, "liquidity_factor", lf)
    monkeypatch.setitem(sys.modules, "pead_factor", pfm)
    return wide


def test_build_panel_missing_companies_returns_empty(env, monkeypatch, tmp_path):
    monkeypatch.setattr(pn, "COMPANIES", str(tmp_path / "gone.parquet"))
    assert pn.build_panel("US").empty


def test_build_panel_no_price_data_returns_empty(env, monkeypatch):
    lf = sys.modules["liquidity_factor"]
    monkeypatch.setattr(lf, "_market_wide", lambda m: None)
    assert pn.build_panel("US").empty


def test_build_panel_only_industries_with_four_members(env):
    df = pn.build_panel("US", lag=5, horizon=5)
    assert set(df["industry"]) == {"Auto"}
    assert sorted(df["ticker"]) == ["A1.NS", "A2.NS", "A3.NS", "A4.NS"]
    assert (df["market"] == "US").all()
    assert list(df.columns) == ["market", "ticker", "industry", "peer_signal", "own_lag", "fwd_ret"]


def test_build_panel_values_use_point_in_time_window(env):
    lag, hor = 5, 5
    T = N - hor - 1                                      # 54
    df = pn.build_panel("US", lag=lag, horizon=hor).set_index("ticker")
    g = {"A1.NS": 0.010, "A2.NS": 0.020, "A3.NS": 0.0, "A4.NS": -0.010}
    for s in g:
        fwd = (1 + g[s]) ** hor - 1
        own = (1 + g[s]) ** lag - 1
        peers = [(1 + g[o]) ** lag - 1 for o in g if o != s]
        assert df.loc[s, "fwd_ret"] == pytest.approx(fwd)
        assert df.loc[s, "own_lag"] == pytest.approx(own)
        assert df.loc[s, "peer_signal"] == pytest.approx(np.mean(peers))
    assert T == 54


def test_build_panel_excludes_self_from_peer_basket(env):
    df = pn.build_panel("US", lag=5, horizon=5).set_index("ticker")
    # if self were included, every peer_signal would be identical (the basket mean)
    assert df["peer_signal"].nunique() == 4
    assert df.loc["A2.NS", "peer_signal"] < df.loc["A4.NS", "peer_signal"]


def test_build_panel_history_too_short(env):
    # T = N - horizon - 1 = 59-... choose lag so that T - lag < 0
    assert pn.build_panel("US", lag=60, horizon=5).empty
    assert not pn.build_panel("US", lag=54, horizon=5).empty      # T - lag == 0 is allowed


def test_build_panel_skips_stock_with_bad_prices(env):
    w = _wide()
    w.loc[w.index[54], "A1.NS"] = np.nan                           # p0 not finite
    env["v"] = (w, None)
    df = pn.build_panel("US", lag=5, horizon=5)
    assert "A1.NS" not in set(df["ticker"])
    assert len(df) == 3
    w2 = _wide()
    w2.loc[w2.index[54], "A2.NS"] = 0.0                            # p0 not > 0
    env["v"] = (w2, None)
    assert "A2.NS" not in set(pn.build_panel("US", lag=5, horizon=5)["ticker"])


def test_build_panel_nan_own_lag_but_valid_row_kept(env):
    w = _wide()
    w.loc[w.index[49], "A1.NS"] = np.nan                           # c[T-lag] NaN -> own_lag NaN
    env["v"] = (w, None)
    df = pn.build_panel("US", lag=5, horizon=5).set_index("ticker")
    assert np.isnan(df.loc["A1.NS", "own_lag"])
    assert np.isfinite(df.loc["A1.NS", "peer_signal"])
    # the other stocks' peer baskets skip A1 (NaN start price)
    assert df.loc["A2.NS", "peer_signal"] == pytest.approx(np.mean([0.0, 0.99 ** 5 - 1]))


def test_build_panel_defaults_are_module_constants(env):
    a = pn.build_panel("US")
    b = pn.build_panel("US", pn.LAG, pn.HORIZON)
    assert a.equals(b) and not a.empty
    assert (pn.LAG, pn.HORIZON) == (21, 21)


# ── CLI ───────────────────────────────────────────────────────────────────────
def _fake_accum(monkeypatch, curve=None):
    m = types.ModuleType("accumulation_screener")
    calls = {}

    def quantile_returns(panel, col):
        calls["qr"] = (len(panel), col)
        return curve if curve is not None else pd.DataFrame(
            {"bucket": ["Q1", "Q2", "Q3"], "median_fwd%": [1.0, 2.0, 4.5]})

    m.quantile_returns = quantile_returns
    m.information_coefficient = lambda s, f: 0.0421
    m.monotonicity = lambda c: 1.0
    monkeypatch.setitem(sys.modules, "accumulation_screener", m)
    return calls


def _panel(n=6, fwd=None):
    return pd.DataFrame({
        "market": ["US"] * n, "ticker": [f"T{i}" for i in range(n)],
        "industry": ["Industry-" + "x" * 40] * n,
        "peer_signal": np.linspace(-0.1, 0.4, n), "own_lag": 0.0,
        "fwd_ret": fwd if fwd is not None else np.linspace(-0.05, 0.05, n)})


def test_main_prints_curve_ic_and_top_list(monkeypatch, capsys):
    calls = _fake_accum(monkeypatch)
    seen = []
    monkeypatch.setattr(pn, "build_panel", lambda m, lag, hor: (seen.append((m, lag, hor)), _panel())[1])
    monkeypatch.setattr(sys, "argv", ["peer_network.py", "--market", "US", "--lag", "10",
                                      "--horizon", "7", "--top", "2"])
    pn.main()
    out = capsys.readouterr().out
    assert seen == [("US", 10, 7)]
    assert calls["qr"] == (6, "peer_signal")
    assert "PEER-NETWORK LEAD-LAG — 6 stock-obs, peer 10d lag -> 7d forward" in out
    assert "fwd return by peer-signal quantile: Q1=+1.0 Q2=+2.0 Q3=+4.5" in out
    assert "Q5-Q1 median +3.50%  IC=+0.042  mono=+1.00" in out
    assert "current top 2 by peer signal" in out
    data_rows = [ln for ln in out.splitlines() if ln.lstrip().startswith("US") and "T" in ln]
    assert [r.split()[1] for r in data_rows] == ["T5", "T4"]       # highest peer signal first
    assert data_rows[0].rstrip().endswith("40.0")
    assert "Industry-" + "x" * 17 in data_rows[0]                  # industry truncated to 26 chars
    assert "x" * 18 not in data_rows[0]


def test_main_filters_extreme_forward_returns(monkeypatch, capsys):
    _fake_accum(monkeypatch)
    fwd = [0.01, 2.99, 3.0, -3.5, 0.02, -2.9]
    monkeypatch.setattr(pn, "build_panel", lambda m, lag, hor: _panel(6, fwd))
    monkeypatch.setattr(sys, "argv", ["peer_network.py", "--market", "US"])
    pn.main()
    assert "4 stock-obs" in capsys.readouterr().out                # |3.0| and |-3.5| removed (strict <3)


def test_main_empty_curve_skips_quantile_lines(monkeypatch, capsys):
    _fake_accum(monkeypatch, curve=pd.DataFrame({"bucket": [], "median_fwd%": []}))
    monkeypatch.setattr(pn, "build_panel", lambda m, lag, hor: _panel())
    monkeypatch.setattr(sys, "argv", ["peer_network.py", "--market", "US"])
    pn.main()
    out = capsys.readouterr().out
    assert "fwd return by peer-signal quantile" not in out
    assert "current top 15 by peer signal" in out


def test_main_no_args_uses_all_markets_and_concats(monkeypatch, capsys):
    _fake_accum(monkeypatch)
    monkeypatch.setattr(pn.marketdata, "market_list", lambda: ["US", "JP"])
    monkeypatch.setattr(pn, "build_panel", lambda m, lag, hor: _panel(3).assign(market=m))
    monkeypatch.setattr(sys, "argv", ["peer_network.py"])
    pn.main()
    assert "6 stock-obs, peer 21d lag -> 21d forward" in capsys.readouterr().out


def test_main_empty_panel_exits(monkeypatch):
    _fake_accum(monkeypatch)
    monkeypatch.setattr(pn, "build_panel", lambda m, lag, hor: pd.DataFrame())
    monkeypatch.setattr(sys, "argv", ["peer_network.py", "--market", "US"])
    with pytest.raises(SystemExit) as ei:
        pn.main()
    assert "no peer-network panel" in str(ei.value.code)
