"""Offline tests for factor_research.py (Markowitz / CAPM / Fama-French proposals)."""
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import factor_research as fr  # noqa: E402


# ── load_universe ─────────────────────────────────────────────────────────────
class _FakeExcel:
    def __init__(self, path):
        self.path = path

    def parse(self, sheet):
        assert sheet == "All_Stocks"
        return pd.DataFrame({"Symbol": ["AAA", "", np.nan, "BBB", "CCC"]})


def test_load_universe_uses_latest_file_and_cleans(monkeypatch):
    seen = {}
    monkeypatch.setattr(fr.glob, "glob", lambda pat, recursive: seen.update(pat=pat, rec=recursive) or
                        ["/x/us_full_scan_2024.xlsx", "/x/us_full_scan_2026.xlsx", "/x/us_full_scan_2025.xlsx"])
    opened = []
    monkeypatch.setattr(pd, "ExcelFile", lambda p: opened.append(p) or _FakeExcel(p))
    out = fr.load_universe(None)
    assert opened == ["/x/us_full_scan_2026.xlsx"]                    # lexicographically last = latest
    assert out == ["AAA", "BBB", "CCC"]                                # '' and NaN removed
    assert seen["rec"] is True and seen["pat"].endswith("us_full_scan_*.xlsx")
    assert os.path.isabs(seen["pat"]) and "~" not in seen["pat"]


@pytest.mark.parametrize("limit,expected", [(2, ["AAA", "BBB"]), (3, ["AAA", "BBB", "CCC"]),
                                             (10, ["AAA", "BBB", "CCC"]), (0, ["AAA", "BBB", "CCC"])])
def test_load_universe_limit(monkeypatch, limit, expected):
    monkeypatch.setattr(fr.glob, "glob", lambda pat, recursive: ["/x/a.xlsx"])
    monkeypatch.setattr(pd, "ExcelFile", _FakeExcel)
    assert fr.load_universe(limit) == expected


def test_load_universe_no_files_raises(monkeypatch):
    monkeypatch.setattr(fr.glob, "glob", lambda pat, recursive: [])
    with pytest.raises(IndexError):
        fr.load_universe(5)


# ── ols / quintile_table ──────────────────────────────────────────────────────
def test_ols_recovers_known_coefficients():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(500, 2))
    y = 1.0 + 2.0 * X[:, 0] - 0.5 * X[:, 1] + rng.normal(0, 0.1, 500)
    r = fr.ols(y, X, ["a", "b"])
    assert list(r) == ["intercept", "a", "b", "_R2"]
    assert r["intercept"][0] == pytest.approx(1.0, abs=0.02)
    assert r["a"][0] == pytest.approx(2.0, abs=0.02) and r["a"][1] > 50
    assert r["b"][0] == pytest.approx(-0.5, abs=0.02) and r["b"][1] < -10
    assert r["_R2"] > 0.98


def test_ols_matches_numpy_and_scipy_pvalue_consistency():
    from scipy import stats
    rng = np.random.RandomState(1)
    x = rng.normal(size=60)
    y = 0.3 * x + rng.normal(size=60)
    r = fr.ols(y, x.reshape(-1, 1), ["x"])
    lr = stats.linregress(x, y)
    assert r["x"][0] == pytest.approx(lr.slope, abs=1e-4)
    assert r["intercept"][0] == pytest.approx(lr.intercept, abs=1e-4)
    assert r["_R2"] == pytest.approx(lr.rvalue ** 2, abs=1e-3)
    assert r["x"][1] == pytest.approx(lr.slope / lr.stderr, abs=1e-2)


def test_ols_noise_has_insignificant_slope_and_low_r2():
    rng = np.random.RandomState(5)
    r = fr.ols(rng.normal(size=300), rng.normal(size=(300, 1)), ["z"])
    assert abs(r["z"][1]) < 3 and r["_R2"] < 0.05


def test_quintile_table_means_counts_and_labels():
    df = pd.DataFrame({"x": np.arange(50.0), "fwd_ret": np.arange(50.0) * 2})
    t = fr.quintile_table(df, "x")
    assert t.index.tolist() == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert t["count"].tolist() == [10] * 5
    assert t["mean"].tolist() == [9.0, 29.0, 49.0, 69.0, 89.0]
    t2 = fr.quintile_table(df.assign(other=df["fwd_ret"] + 1), "x", ret="other")
    assert t2["mean"].iloc[0] == 10.0


@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason="factor_research.py:83 pd.qcut(..., labels=[Q1..Q5], duplicates='drop') raises "
                          "'Bin labels must be one fewer than the number of bin edges' when ties collapse "
                          "bin edges, so duplicates='drop' never actually works")
def test_quintile_table_duplicate_edges_drop_buckets():
    df = pd.DataFrame({"x": [1.0] * 30 + list(range(2, 12)), "fwd_ret": np.arange(40.0)})
    t = fr.quintile_table(df, "x")
    assert len(t) < 5 and t["count"].sum() == 40


# ── cross_section (with a fake PIT-fundamentals module) ───────────────────────
def _ohlc(n=600, tickers=("A", "B", "C", "D"), seed=0, vol=1e6, price0=50.0):
    rng = np.random.RandomState(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    out = {}
    for k, t in enumerate(tickers):
        close = price0 * (k + 1) * np.cumprod(1 + rng.normal(0.0004, 0.01, n))
        out[t] = pd.DataFrame({"Close": close, "Volume": np.full(n, vol)}, index=idx)
    return out


class _PIT:
    """Fake pit_fundamentals: as_of(t, date) -> {'shares':..., 'ni': ...}; _g unwraps."""
    calls = []

    @staticmethod
    def as_of(t, date):
        _PIT.calls.append((t, date))
        return {"shares": 1e6 * (1 + ord(t[0]) % 5), "ni": {"v": 2e6}} if t != "NOFUND" else None

    @staticmethod
    def _g(x):
        return x["v"] if isinstance(x, dict) else x


@pytest.fixture
def pit(monkeypatch):
    _PIT.calls = []
    monkeypatch.setitem(sys.modules, "pit_fundamentals",
                        types.SimpleNamespace(as_of=_PIT.as_of, _g=_PIT._g))
    return _PIT


def test_cross_section_point_in_time_and_forward_return(pit):
    ohlc = _ohlc()
    df, mkt = fr.cross_section(ohlc)
    assert df["ticker"].tolist() == ["A", "B", "C", "D"]
    assert set(df.columns) == {"ticker", "beta", "size", "log_size", "value_ey", "fwd_ret"}
    T = ohlc["A"].index[-253]                                          # fwd_days=252 -> index[-253]
    assert {c[1] for c in pit.calls} == {str(T.date())}               # fundamentals as of T, not today
    a = ohlc["A"]["Close"]
    assert df.set_index("ticker").loc["A", "fwd_ret"] == pytest.approx((a.iloc[-1] / a.loc[T] - 1) * 100)
    shares = 1e6 * (1 + ord("A") % 5)
    row = df.set_index("ticker").loc["A"]
    assert row["size"] == pytest.approx(a.loc[T] * shares)
    assert row["log_size"] == pytest.approx(np.log10(a.loc[T] * shares))
    assert row["value_ey"] == pytest.approx(2e6 / (a.loc[T] * shares))
    # market proxy: equal-weight mean of daily returns of all eligible names
    expect = pd.DataFrame({t: d["Close"].pct_change() for t, d in ohlc.items()}).mean(axis=1)
    pd.testing.assert_series_equal(mkt, expect, check_names=False)
    # beta is cov/var against that proxy over the 252 days up to T
    r = a.pct_change().loc[:T].tail(252)
    m = mkt.reindex(r.index)
    assert row["beta"] == pytest.approx(r.cov(m) / m.var())


def test_cross_section_filters(pit):
    ohlc = _ohlc(tickers=("A", "B", "C", "NOFUND"))
    ohlc["SHORT"] = ohlc["A"].iloc[-400:]                              # < lookback+fwd+5 -> skipped
    ohlc["ILLIQ"] = ohlc["B"].assign(Volume=1.0)                      # median $vol below threshold
    df, _ = fr.cross_section(ohlc)
    assert set(df["ticker"]) == {"A", "B", "C"}                       # NOFUND has no fundamentals
    thin = _ohlc(tickers=("A",), price0=1.0, vol=100.0)               # $vol ~ 100 < 2e6
    assert fr.cross_section(thin)[0].empty
    assert len(fr.cross_section(thin, min_dollar_vol=0)[0]) == 1


def test_cross_section_dollar_volume_threshold_is_inclusive(pit):
    ohlc = _ohlc(tickers=("A",), vol=1000.0)
    med = (ohlc["A"]["Close"] * ohlc["A"]["Volume"]).tail(252).median()
    assert len(fr.cross_section(ohlc, min_dollar_vol=med)[0]) == 1    # median == threshold -> kept
    assert fr.cross_section(ohlc, min_dollar_vol=med * 1.0001)[0].empty


def test_cross_section_missing_fundamental_fields(pit, monkeypatch):
    monkeypatch.setitem(sys.modules, "pit_fundamentals", types.SimpleNamespace(
        as_of=lambda t, d: {"shares": None, "ni": {"v": 1.0}}, _g=_PIT._g))
    assert fr.cross_section(_ohlc())[0].empty                          # no shares -> no market cap
    monkeypatch.setitem(sys.modules, "pit_fundamentals", types.SimpleNamespace(
        as_of=lambda t, d: {"shares": 10.0, "ni": None}, _g=_PIT._g))
    assert fr.cross_section(_ohlc())[0].empty                          # no earnings -> no value proxy
    monkeypatch.setitem(sys.modules, "pit_fundamentals", types.SimpleNamespace(
        as_of=lambda t, d: {"shares": 10.0, "ni": {"v": 0}}, _g=_PIT._g))
    assert fr.cross_section(_ohlc())[0].empty                          # ni == 0 is falsy -> skipped


def test_cross_section_zero_market_variance_drops_stock(pit):
    flat = {t: pd.DataFrame({"Close": np.full(600, 100.0), "Volume": np.full(600, 1e6)},
                            index=pd.bdate_range("2020-01-01", periods=600)) for t in "AB"}
    assert fr.cross_section(flat)[0].empty                             # beta NaN -> row skipped


# ── markowitz ─────────────────────────────────────────────────────────────────
def _mk_inputs(n_tickers=8, seed=0):
    tickers = tuple(f"T{i}" for i in range(n_tickers))
    ohlc = _ohlc(tickers=tickers, seed=seed, n=600)
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({"ticker": list(tickers), "fwd_ret": rng.normal(10, 5, n_tickers)})
    return ohlc, df


def test_markowitz_portfolios_structure_and_math():
    ohlc, df = _mk_inputs()
    res = fr.markowitz(ohlc, df)
    assert list(res) == ["Concentrated(1)", "EqualWeight", "MinVariance", "MaxSharpe"]
    for v in res.values():
        assert set(v) == {"fwd_ret%", "ex_ante_vol%", "ret/vol"}
    fwd = np.array([ohlc[t]["Close"].iloc[-1] / ohlc[t]["Close"].iloc[-253] - 1 for t in df["ticker"]]) * 100
    order = df.sort_values("fwd_ret", ascending=False)["ticker"].tolist()
    assert res["EqualWeight"]["fwd_ret%"] == pytest.approx(round(float(np.mean([
        ohlc[t]["Close"].iloc[-1] / ohlc[t]["Close"].iloc[-253] - 1 for t in order])) * 100, 2), abs=0.01)
    # diversification: min-variance vol cannot exceed equal-weight or single-stock vol
    assert res["MinVariance"]["ex_ante_vol%"] <= res["EqualWeight"]["ex_ante_vol%"] + 1e-9
    assert res["MinVariance"]["ex_ante_vol%"] <= res["Concentrated(1)"]["ex_ante_vol%"] + 1e-9
    assert res["EqualWeight"]["ex_ante_vol%"] <= res["Concentrated(1)"]["ex_ante_vol%"] + 1e-9
    assert fwd.shape == (8,)
    for v in res.values():
        assert v["ret/vol"] == round(v["fwd_ret%"] / v["ex_ante_vol%"], 2)


def test_markowitz_candidate_set_is_top30_by_fwd_ret():
    ohlc, df = _mk_inputs(n_tickers=40)
    res = fr.markowitz(ohlc, df)
    assert res is not None
    # the 10 lowest fwd_ret names must not be in the portfolio -> equal-weight over exactly 30
    top = df.sort_values("fwd_ret", ascending=False)["ticker"].head(30)
    exp = np.mean([ohlc[t]["Close"].iloc[-1] / ohlc[t]["Close"].iloc[-253] - 1 for t in top]) * 100
    assert res["EqualWeight"]["fwd_ret%"] == pytest.approx(exp, abs=0.01)


def test_markowitz_too_few_assets_returns_none():
    ohlc, df = _mk_inputs(n_tickers=4)
    assert fr.markowitz(ohlc, df) is None
    ohlc5, df5 = _mk_inputs(n_tickers=5)
    assert fr.markowitz(ohlc5, df5) is not None                        # boundary: 5 assets is enough


def test_markowitz_drops_assets_with_missing_history():
    ohlc, df = _mk_inputs(n_tickers=6)
    c = ohlc["T0"]["Close"].copy()
    c.iloc[-300:-260] = np.nan                                         # NaNs inside the lookback window
    ohlc["T0"] = ohlc["T0"].assign(Close=c)
    # T0 has gaps in returns before T -> dropped by dropna(axis=1); 5 left still runs
    res = fr.markowitz(ohlc, df)
    assert res is not None
    ohlc2, df2 = _mk_inputs(n_tickers=5)
    ohlc2["T0"] = ohlc2["T0"].assign(Close=c.iloc[:0].reindex(ohlc2["T0"].index))   # all-NaN column
    assert fr.markowitz(ohlc2, df2) is None                            # 4 left < 5


# ── CLI ───────────────────────────────────────────────────────────────────────
def _fake_store(monkeypatch, ohlc, tickers):
    calls = {}

    def cached_download(t, years):
        calls["args"] = (list(t), years)
        return ohlc

    monkeypatch.setitem(sys.modules, "market_store", types.SimpleNamespace(cached_download=cached_download))
    monkeypatch.setattr(fr, "load_universe", lambda limit: calls.update(limit=limit) or list(tickers))
    return calls


def _run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["factor_research.py", *argv])
    fr.main()


def _fake_cross_section(n=120, beta_effect=0.0, size_effect=0.0, seed=0):
    rng = np.random.RandomState(seed)
    beta = rng.normal(1, 0.3, n)
    logsz = rng.normal(9, 1, n)
    ey = rng.normal(0.05, 0.02, n)
    ret = beta_effect * beta + size_effect * logsz + rng.normal(0, 1, n)
    return pd.DataFrame({"ticker": [f"T{i}" for i in range(n)], "beta": beta, "size": 10 ** logsz,
                         "log_size": logsz, "value_ey": ey, "fwd_ret": ret}), None


def test_main_too_few_stocks(monkeypatch, capsys):
    _fake_store(monkeypatch, {}, ["A"])
    monkeypatch.setattr(fr, "cross_section", lambda ohlc, min_dollar_vol: _fake_cross_section(n=49))
    _run_main(monkeypatch)
    cap = capsys.readouterr()
    assert "too few stocks" in cap.err and "factor panel: 49 stocks" in cap.err
    assert cap.out == ""


def test_main_full_report_capm_not_supported(monkeypatch, capsys):
    calls = _fake_store(monkeypatch, {}, ["A", "B"])
    got = {}
    monkeypatch.setattr(fr, "cross_section", lambda ohlc, min_dollar_vol:
                        got.update(mdv=min_dollar_vol) or _fake_cross_section())
    monkeypatch.setattr(fr, "markowitz", lambda ohlc, df: {
        "Concentrated(1)": {"fwd_ret%": 5.0, "ex_ante_vol%": 40.0, "ret/vol": 0.12},
        "MinVariance": {"fwd_ret%": 4.0, "ex_ante_vol%": 12.0, "ret/vol": 0.33}})
    _run_main(monkeypatch, "--limit", "77", "--min-dollar-vol", "5e6")
    cap = capsys.readouterr()
    assert calls["limit"] == 77 and calls["args"] == (["A", "B"], 5) and got["mdv"] == 5e6
    assert "loading 2 US tickers" in cap.err and "factor panel: 120 stocks" in cap.err
    out = cap.out
    assert "P1 — MARKOWITZ (1952)" in out and "Concentrated(1)" in out and "MinVariance" in out
    assert "P2 — SHARPE (1964) CAPM: is beta priced?" in out
    assert "-> CAPM NOT supported" in out
    assert "P3 — FAMA-FRENCH (1992)" in out and "size quintiles (Q1=small):" in out
    assert "multivariate OLS fwd_ret ~ beta + log_size + value_ey" in out
    assert "P4 — FAMA (1991) EMH" in out
    assert "statistically significant factors: NONE (efficient — no exploitable premium)" in out
    assert "beta significant? NO" in out


def test_main_detects_priced_beta_and_size(monkeypatch, capsys):
    _fake_store(monkeypatch, {}, ["A"])
    monkeypatch.setattr(fr, "cross_section", lambda ohlc, min_dollar_vol:
                        _fake_cross_section(n=400, beta_effect=3.0, size_effect=-1.5))
    monkeypatch.setattr(fr, "markowitz", lambda ohlc, df: None)       # P1 section skipped quietly
    _run_main(monkeypatch)
    out = capsys.readouterr().out
    assert "-> CAPM SUPPORTED" in out
    assert "statistically significant factors: ['beta', 'log_size']" in out
    assert "beta significant? yes" in out
    assert "beta       coef=" in out and "***" in out
    assert "Concentrated" not in out


def test_main_beta_insignificant_message(monkeypatch, capsys):
    _fake_store(monkeypatch, {}, ["A"])
    monkeypatch.setattr(fr, "cross_section", lambda ohlc, min_dollar_vol:
                        _fake_cross_section(n=200, size_effect=2.0, seed=3))
    monkeypatch.setattr(fr, "markowitz", lambda ohlc, df: None)
    _run_main(monkeypatch)
    out = capsys.readouterr().out
    assert "beta significant? NO — matches FF: beta explains ~nothing" in out
    assert "['log_size']" in out
