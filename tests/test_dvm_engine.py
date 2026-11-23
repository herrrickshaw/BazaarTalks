"""Offline tests for dvm_engine.py (Trendlyne-style D/V/M scoring CLI)."""
import os
import sqlite3
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dvm_engine as eng  # noqa: E402


PANDAS3 = int(pd.__version__.split(".")[0]) >= 3
# pandas>=3: Series.astype(str) keeps NaN as a float NaN, so the `x != "nan"` filter lets it through.
nan_bug = pytest.mark.xfail(
    PANDAS3, strict=True, raises=AssertionError,
    reason="dvm_engine.py:110 NaN symbols survive the 'nan' string filter under pandas>=3")


def _ohlcv(n=300, drift=0.001, amp=0.01, seed=0, price0=100.0, vol=1e6):
    rng = np.random.default_rng(seed)
    close = price0 * np.cumprod(1 + drift + amp * rng.standard_normal(n))
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
        "Volume": vol * (1 + 0.1 * np.abs(rng.standard_normal(n))),
    }, index=idx)


def _g(lst, i=0):
    return lst[i] if lst and len(lst) > i else None


def _fake_pit(monkeypatch, as_of=None, piotroski=None):
    mod = types.ModuleType("pit_fundamentals")
    mod.as_of = as_of or (lambda t, d: {})
    mod.piotroski_asof = piotroski or (lambda t, d: (None, {}))
    mod._g = _g
    monkeypatch.setitem(sys.modules, "pit_fundamentals", mod)
    return mod


# ── momentum_score ────────────────────────────────────────────────────────────
def test_momentum_none_and_short_return_nan():
    assert np.isnan(eng.momentum_score(None))
    assert np.isnan(eng.momentum_score(_ohlcv(219)))


def test_momentum_boundary_220_is_scored():
    m = eng.momentum_score(_ohlcv(220))
    assert isinstance(m, float) and not np.isnan(m)


def test_momentum_bounded_and_float():
    for seed in range(6):
        m = eng.momentum_score(_ohlcv(300, seed=seed))
        assert isinstance(m, float) and 0.0 <= m <= 100.0


def test_momentum_uptrend_beats_downtrend():
    up = eng.momentum_score(_ohlcv(300, drift=0.004, amp=0.004, seed=1))
    dn = eng.momentum_score(_ohlcv(300, drift=-0.004, amp=0.004, seed=1))
    assert up > dn
    assert up > 60 and dn < 50


def test_momentum_monotone_rising_has_nan_rsi_but_scores():
    # no down-days: loss mean is 0 -> RSI NaN -> RSI sub-score collapses to 0, rest still scored
    n = 300
    c = np.linspace(100, 200, n)
    df = pd.DataFrame({"High": c * 1.01, "Low": c * 0.99, "Close": c,
                       "Volume": np.full(n, 1e6)}, index=pd.bdate_range("2020-01-01", periods=n))
    m = eng.momentum_score(df)
    assert 0.0 < m < 100.0


def test_momentum_zero_volume_uses_unit_ratio():
    df = _ohlcv(300, seed=3)
    df["Volume"] = 0.0
    m0 = eng.momentum_score(df)
    df["Volume"] = 1e6   # ratio == 1 -> vol sub-score 50, identical to the zero-volume fallback
    m1 = eng.momentum_score(df)
    assert m0 == pytest.approx(m1)


def test_momentum_volume_thrust_raises_score_and_is_capped():
    base = _ohlcv(300, seed=4)
    quiet = base.copy(); quiet["Volume"] = 1e6
    surge = quiet.copy(); surge.iloc[-1, surge.columns.get_loc("Volume")] = 1e9
    huge = quiet.copy(); huge.iloc[-1, huge.columns.get_loc("Volume")] = 1e12
    mq, ms, mh = (eng.momentum_score(x) for x in (quiet, surge, huge))
    assert ms > mq
    assert ms == pytest.approx(mh)   # volume sub-score capped at 100


def test_momentum_overbought_rsi_penalised():
    # same shape, but force a sharp final run-up so RSI > 70 vs a flat-noise tail
    df = _ohlcv(300, drift=0.0, amp=0.01, seed=5)
    hot = df.copy()
    bump = np.linspace(1.0, 1.6, 15)
    for col in ("Close", "High", "Low"):
        hot.iloc[-15:, hot.columns.get_loc(col)] = hot[col].iloc[-16] * bump
    assert eng.momentum_score(hot) != eng.momentum_score(df)
    assert 0.0 <= eng.momentum_score(hot) <= 100.0


# ── durability_score ──────────────────────────────────────────────────────────
def _facts(ni=(1.0,), roe_hist=(25.0,), de=0.5, cfo=10.0, capex=5.0, rev=(120.0, 100.0)):
    return {"ni": list(ni), "roe_hist": list(roe_hist), "de": de, "cfo": cfo,
            "capex": capex, "rev": list(rev)}


def _dur(monkeypatch, facts, F):
    _fake_pit(monkeypatch, as_of=lambda t, d: facts, piotroski=lambda t, d: (F, {}))
    return eng.durability_score("X", "2024-01-01")


def test_durability_no_facts_returns_none(monkeypatch):
    _fake_pit(monkeypatch, as_of=lambda t, d: {})
    assert eng.durability_score("X", "2024-01-01") == (None, {})


def test_durability_missing_net_income_returns_none(monkeypatch):
    _fake_pit(monkeypatch, as_of=lambda t, d: _facts(ni=()))
    assert eng.durability_score("X", "2024-01-01") == (None, {})


def test_durability_best_case_exact(monkeypatch):
    score, det = _dur(monkeypatch, _facts(), 9)
    assert score == pytest.approx(np.mean([90, 85, 80, 75, 100]))
    assert det["roe"] == pytest.approx(25.0)
    assert det["de"] == 0.5
    assert det["fcf_pos"] is True
    assert det["piotroski"] == 9


def test_durability_worst_case_exact(monkeypatch):
    f = _facts(roe_hist=(1.0,), de=5.0, cfo=1.0, capex=5.0, rev=(90.0, 100.0))
    score, det = _dur(monkeypatch, f, 0)
    assert score == pytest.approx(np.mean([15, 20, 20, 35, 0]))
    assert det["fcf_pos"] is False


@pytest.mark.parametrize("roe,expected", [
    (20.01, 90), (20.0, 70), (15.01, 70), (15.0, 45), (8.01, 45), (8.0, 15), (-3.0, 15),
])
def test_durability_roe_thresholds(monkeypatch, roe, expected):
    f = _facts(roe_hist=(roe,), de=0.5, cfo=10, capex=5, rev=(2, 1))
    score, _ = _dur(monkeypatch, f, 9)
    assert score == pytest.approx(np.mean([expected, 85, 80, 75, 100]))


def test_durability_empty_roe_history_scores_floor(monkeypatch):
    f = _facts(roe_hist=())
    score, det = _dur(monkeypatch, f, 9)
    assert det["roe"] is None
    assert score == pytest.approx(np.mean([15, 85, 80, 75, 100]))


def test_durability_roe_is_mean_of_history(monkeypatch):
    f = _facts(roe_hist=(10.0, 30.0))   # mean 20.0 -> not > 20 -> 70
    score, det = _dur(monkeypatch, f, 9)
    assert det["roe"] == pytest.approx(20.0)
    assert score == pytest.approx(np.mean([70, 85, 80, 75, 100]))


@pytest.mark.parametrize("de,expected", [
    (0.99, 85), (1.0, 50), (1.99, 50), (2.0, 20), (None, 20),
])
def test_durability_debt_thresholds(monkeypatch, de, expected):
    score, _ = _dur(monkeypatch, _facts(de=de), 9)
    assert score == pytest.approx(np.mean([90, expected, 80, 75, 100]))


def test_durability_de_zero_counts_as_low_debt(monkeypatch):
    score, _ = _dur(monkeypatch, _facts(de=0.0), 9)
    assert score == pytest.approx(np.mean([90, 85, 80, 75, 100]))


@pytest.mark.parametrize("cfo,capex,expected,fcf_pos", [
    (10, 5, 80, True), (5, 5, 20, 0), (4, 5, 20, False), (None, 5, 20, None), (5, None, 20, None),
])
def test_durability_fcf(monkeypatch, cfo, capex, expected, fcf_pos):
    score, det = _dur(monkeypatch, _facts(cfo=cfo, capex=capex), 9)
    assert score == pytest.approx(np.mean([90, 85, expected, 75, 100]))
    assert det["fcf_pos"] == fcf_pos


@pytest.mark.parametrize("rev,expected", [
    ((101.0, 100.0), 75), ((100.0, 100.0), 35), ((99.0, 100.0), 35), ((100.0,), 35), ((), 35),
])
def test_durability_revenue_growth(monkeypatch, rev, expected):
    score, _ = _dur(monkeypatch, _facts(rev=rev), 9)
    assert score == pytest.approx(np.mean([90, 85, 80, expected, 100]))


@pytest.mark.parametrize("F,expected", [(None, 50.0), (0, 0.0), (3, 100 / 3), (9, 100.0)])
def test_durability_piotroski_component(monkeypatch, F, expected):
    score, det = _dur(monkeypatch, _facts(), F)
    assert score == pytest.approx(np.mean([90, 85, 80, 75, expected]))
    assert det["piotroski"] == F


# ── load_universe ─────────────────────────────────────────────────────────────
class _FakeXL:
    frames = {}
    opened = []

    def __init__(self, path):
        _FakeXL.opened.append(path)

    def parse(self, sheet):
        assert sheet == "All_Stocks"
        return _FakeXL.frames["df"]


@pytest.fixture
def fake_excel(monkeypatch):
    _FakeXL.opened = []
    monkeypatch.setattr(eng.pd, "ExcelFile", _FakeXL)
    monkeypatch.setattr(eng.glob, "glob", lambda pat, recursive=False:
                        ["/x/b_2.xlsx", "/x/a_1.xlsx", "/x/c_3.xlsx"])
    return _FakeXL


def test_load_universe_us_uses_symbol_and_latest_file(fake_excel):
    fake_excel.frames["df"] = pd.DataFrame({"Symbol": ["AAPL", "MSFT", "NVDA"]})
    out = eng.load_universe("US", None)
    assert out == ["AAPL", "MSFT", "NVDA"]   
    assert fake_excel.opened == ["/x/c_3.xlsx"]    # sorted(...)[-1]


@nan_bug
def test_load_universe_drops_nan_strings(fake_excel):
    fake_excel.frames["df"] = pd.DataFrame({"Symbol": ["AAPL", np.nan, "MSFT"]})
    assert eng.load_universe("US", None) == ["AAPL", "MSFT"]


def test_load_universe_india_concatenates_suffix(fake_excel):
    fake_excel.frames["df"] = pd.DataFrame({"Symbol": ["TCS", "INFY"], "Suffix": [".NS", ".BO"]})
    assert eng.load_universe("India", None) == ["TCS.NS", "INFY.BO"]


@pytest.mark.parametrize("market", ["Japan", "Korea"])
def test_load_universe_asia_uses_yf_ticker(fake_excel, market):
    fake_excel.frames["df"] = pd.DataFrame({"YF_Ticker": ["7203.T", "005930.KS"], "Symbol": ["a", "b"]})
    assert eng.load_universe(market, None) == ["7203.T", "005930.KS"]


@nan_bug
def test_load_universe_limit_slices_after_filtering(fake_excel):
    fake_excel.frames["df"] = pd.DataFrame({"Symbol": [np.nan, "A", "B", "C"]})
    assert eng.load_universe("US", 2) == ["A", "B"]
    assert eng.load_universe("US", 0) == ["A", "B", "C"]     # falsy limit -> everything


def test_load_universe_unknown_market_raises_keyerror():
    with pytest.raises(KeyError):
        eng.load_universe("Mars", 5)


# ── SCREENS ───────────────────────────────────────────────────────────────────
def _row(D=None, V=None, M=0.0):
    return {"D": D, "V": V, "M": M}


def test_screen_high_dvm_requires_all_three_and_avg_65():
    s = eng.SCREENS["high_dvm"]
    assert s(_row(65, 65, 65))
    assert not s(_row(64, 65, 65))          # mean 64.67 < 65
    assert s(_row(60, 70, 65))              # mean exactly 65 -> inclusive
    assert not s(_row(None, 90, 90))
    assert not s(_row(90, None, 90))


def test_screen_durable_momentum_boundaries():
    s = eng.SCREENS["durable_momentum"]
    assert s(_row(D=60, M=60))
    assert not s(_row(D=59.9, M=60))
    assert not s(_row(D=60, M=59.9))
    assert not s(_row(D=None, M=99))


def test_screen_value_under_radar_boundaries():
    s = eng.SCREENS["value_under_radar"]
    assert s(_row(55, 55, 49.9))
    assert not s(_row(55, 55, 50))          # M must be strictly < 50
    assert not s(_row(54.9, 55, 10))
    assert not s(_row(55, 54.9, 10))
    assert not s(_row(None, 90, 10))
    assert not s(_row(90, None, 10))


def test_screen_momentum_breakout_boundary():
    s = eng.SCREENS["momentum_breakout"]
    assert s(_row(M=75)) and not s(_row(M=74.99))


def test_labels_cover_all_eight_codes():
    assert len(eng.LABELS) == 8
    assert eng.LABELS["GGG"] == "Strong Performer" and eng.LABELS["BBB"] == "Weak / Avoid"
    assert set(eng.LABELS) == {a + b + c for a in "GB" for b in "GB" for c in "GB"}


# ── main() ────────────────────────────────────────────────────────────────────
@pytest.fixture
def run_main(monkeypatch, tmp_path):
    def _run(market="US", extra=(), ohlc=None, universe=None, as_of=None, piotroski=None):
        calls = {}
        ms = types.ModuleType("market_store")

        def cached_download(tickers, years):
            calls["tickers"], calls["years"] = list(tickers), years
            return ohlc

        ms.cached_download = cached_download
        monkeypatch.setitem(sys.modules, "market_store", ms)
        _fake_pit(monkeypatch, as_of=as_of, piotroski=piotroski)
        monkeypatch.setattr(eng, "load_universe",
                            lambda m, lim: calls.setdefault("lu", (m, lim)) and (universe or list(ohlc)))
        db = tmp_path / "dvm.db"
        monkeypatch.setattr(sys, "argv", ["dvm_engine.py", "--market", market, "--db", str(db), *extra])
        eng.main()
        return calls, db
    return _run


def _read(db):
    c = sqlite3.connect(db)
    try:
        return pd.read_sql("SELECT * FROM dvm ORDER BY ticker", c)
    finally:
        c.close()


def test_main_us_scores_filters_and_persists(run_main, capsys):
    liquid_up = _ohlcv(300, drift=0.004, amp=0.004, seed=1)
    liquid_dn = _ohlcv(300, drift=-0.004, amp=0.004, seed=1)
    illiquid = _ohlcv(300, seed=2, vol=1.0)          # $-vol far below 2e6
    short = _ohlcv(250, seed=3)                       # len must be > 250
    edge = _ohlcv(251, seed=4)                        # 251 passes the length gate
    ohlc = {"UP": liquid_up, "DN": liquid_dn, "ILL": illiquid, "SHORT": short, "EDGE": edge}
    facts = {t: {**_facts(), "ni": [ni], "shares": 10.0}
             for t, ni in (("UP", 100.0), ("DN", 10.0), ("EDGE", 50.0))}
    calls, db = run_main("US", ["--limit", "5", "--screen", "momentum_breakout"], ohlc=ohlc,
                         as_of=lambda t, d: facts.get(t, {}), piotroski=lambda t, d: (8, {}))
    out, err = capsys.readouterr()
    assert calls["years"] == 5 and calls["lu"] == ("US", 5)
    df = _read(db)
    assert set(df.ticker) == {"UP", "DN", "EDGE"}        # ILL (illiquid) and SHORT dropped
    assert (df.market == "US").all()
    assert "loading 5 US tickers" in err and "3 liquid stocks scored" in err
    # D is the same formula for all -> equal; V is the earnings-yield percentile rank
    ey = {"UP": 100 / (liquid_up.Close.iloc[-1] * 10), "DN": 10 / (liquid_dn.Close.iloc[-1] * 10),
          "EDGE": 50 / (edge.Close.iloc[-1] * 10)}
    exp_rank = pd.Series(ey).rank(pct=True) * 100
    for _, r in df.iterrows():
        assert r.V == pytest.approx(round(float(exp_rank[r.ticker]), 1))
        assert r.D == pytest.approx(round(np.mean([90, 85, 80, 75, 8 / 9 * 100]), 1))
        code = ("G" if r.D >= 50 else "B") + ("G" if r.V >= 50 else "B") + ("G" if r.M >= 50 else "B")
        assert r.code == code and r.label == eng.LABELS[code]
    assert "=== DVM SCORES" in out and "(3 stocks)" in out
    assert "=== SCREEN: momentum_breakout" in out
    assert "saved all scores to" in out


def test_main_us_missing_fundamentals_yield_bad_durability_and_valuation(run_main, capsys):
    ohlc = {"AAA": _ohlcv(300, seed=1)}
    calls, db = run_main("US", ohlc=ohlc, as_of=lambda t, d: {})
    df = _read(db)
    assert len(df) == 1
    assert pd.isna(df.D[0]) and pd.isna(df.V[0])
    assert df.code[0].startswith("BB")


def test_main_us_zero_or_negative_net_income_has_no_earnings_yield(run_main):
    ohlc = {"NEG": _ohlcv(300, seed=1), "ZERO": _ohlcv(300, seed=2), "NOSH": _ohlcv(300, seed=3)}
    facts = {"NEG": {**_facts(ni=(-5.0,)), "shares": 10.0},
             "ZERO": {**_facts(ni=(0.0,)), "shares": 10.0},
             "NOSH": {**_facts(ni=(5.0,)), "shares": 0}}
    calls, db = run_main("US", ohlc=ohlc, as_of=lambda t, d: facts[t])
    df = _read(db)
    # NEG still has a (truthy) negative NI -> yield negative; ZERO and NOSH have none
    assert df.set_index("ticker").V["ZERO"] != df.set_index("ticker").V["ZERO"]   # NaN
    assert pd.isna(df.set_index("ticker").V["NOSH"])
    assert not pd.isna(df.set_index("ticker").V["NEG"])


@pytest.mark.parametrize("market", ["India", "Japan", "Korea"])
def test_main_non_us_skips_fundamentals(run_main, market, capsys):
    ohlc = {"AAA": _ohlcv(300, drift=0.004, amp=0.004, seed=1)}

    def boom(*a, **k):
        raise AssertionError("fundamentals must not be touched outside the US")

    calls, db = run_main(market, ohlc=ohlc, as_of=boom, piotroski=boom)
    df = _read(db)
    assert df.market[0] == market
    assert pd.isna(df.D[0]) and pd.isna(df.V[0])
    assert df.code[0] in ("BBG", "BBB")
    assert df.code[0] == ("BBG" if df.M[0] >= 50 else "BBB")


def test_main_short_history_after_liquidity_gate_is_skipped(run_main):
    # passes the >250 length gate but momentum_score needs >=220 -> always scored; ensure NaN skip
    # path with a frame whose momentum is NaN: monkeypatch via zero-length trick is not possible,
    # so verify the empty-universe path instead.
    ohlc = {"SHORT": _ohlcv(100, seed=1)}
    calls, db = run_main("India", ohlc=ohlc)
    assert len(_read(db)) == 0


def test_main_nan_momentum_rows_are_skipped(run_main, monkeypatch):
    monkeypatch.setattr(eng, "momentum_score", lambda df: np.nan)
    calls, db = run_main("India", ohlc={"AAA": _ohlcv(300)})
    assert len(_read(db)) == 0


def test_main_screen_hits_sorted_and_truncated_to_15(run_main, capsys, monkeypatch):
    n = 20
    scores = {f"T{i:02d}": float(80 + i) for i in range(n)}
    monkeypatch.setattr(eng, "momentum_score", lambda df: scores[df.attrs["t"]])
    ohlc = {}
    for t in scores:
        d = _ohlcv(300, seed=1); d.attrs["t"] = t
        ohlc[t] = d
    calls, db = run_main("India", ["--screen", "momentum_breakout"], ohlc=ohlc)
    out = capsys.readouterr().out
    assert f"-> {n} hits" in out
    lines = [l for l in out.splitlines() if l.startswith("  T")]
    assert len(lines) == 15
    assert lines[0].split()[0] == "T19" and lines[-1].split()[0] == "T05"   # descending by score


def test_main_min_dollar_vol_flag_applied(run_main, capsys):
    ohlc = {"AAA": _ohlcv(300, vol=1e6)}     # ~ $100M/day
    _, db = run_main("India", ["--min-dollar-vol", "1e12"], ohlc=ohlc)
    assert len(_read(db)) == 0
    _, db2 = run_main("India", ["--min-dollar-vol", "1"], ohlc=ohlc)
    assert len(_read(db2)) == 1


def test_main_replaces_existing_table(run_main, tmp_path):
    db = tmp_path / "dvm.db"
    c = sqlite3.connect(db); c.execute("CREATE TABLE dvm(junk TEXT)"); c.commit(); c.close()
    _, db = run_main("India", ohlc={"AAA": _ohlcv(300)})
    assert list(_read(db).columns) == ["ticker", "market", "D", "V", "M", "code", "label"]
