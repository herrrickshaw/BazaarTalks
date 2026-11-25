"""Offline tests for pit_backtest.py (point-in-time Darvas / Piotroski / Coffee-Can backtest)."""
import importlib
import os
import sqlite3
import sys
import types

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PANDAS3 = int(pd.__version__.split(".")[0]) >= 3


@pytest.fixture
def pb():
    """pit_backtest imported against a fake pit_fundamentals (the real one opens a SQLite file
    next to the source at import time and needs the network); sys.modules is restored afterwards."""
    names = ("pit_fundamentals", "pit_backtest")
    saved = {n: sys.modules.get(n) for n in names}
    fake = types.ModuleType("pit_fundamentals")
    fake.piotroski_asof = lambda t, d: (None, {})
    fake.coffeecan_asof = lambda t, d, mktcap=None: (False, {})
    fake.as_of = lambda t, d: {}
    fake._g = lambda lst, i=0: lst[i] if lst and len(lst) > i else None
    sys.modules["pit_fundamentals"] = fake
    sys.modules.pop("pit_backtest", None)
    mod = importlib.import_module("pit_backtest")
    mod._fake_fund = fake
    try:
        yield mod
    finally:
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m


# ── constants ─────────────────────────────────────────────────────────────────
def test_constants_and_builtin_universe(pb):
    assert pb.US_COST == 0.10 and pb.BREAKOUT_LB == 60
    assert len(pb.UNIVERSE) == 100 and len(set(pb.UNIVERSE)) == 100
    assert pb.UNIVERSE[:3] == ["AAPL", "MSFT", "NVDA"] and pb.UNIVERSE[-1] == "PH"


# ── darvas_breakout ───────────────────────────────────────────────────────────
def _series(vals, start="2022-01-03"):
    return pd.Series(vals, index=pd.bdate_range(start, periods=len(vals)), dtype=float)


def test_darvas_needs_sixty_prior_bars(pb):
    s = _series(np.arange(1, 101))
    assert pb.darvas_breakout(s, s.index[59]) is False     # loc 59 < 60 even though it is a new high
    assert pb.darvas_breakout(s, s.index[60]) is True


def test_darvas_tie_with_prior_high_counts(pb):
    v = [10.0] * 60 + [10.0]
    s = _series(v)
    assert pb.darvas_breakout(s, s.index[60]) is True


def test_darvas_below_prior_high_is_not_breakout(pb):
    v = [10.0] * 59 + [11.0] + [10.99]
    s = _series(v)
    assert pb.darvas_breakout(s, s.index[60]) is False


def test_darvas_window_excludes_current_bar_and_older_than_60(pb):
    # a spike 61 bars before t falls out of the window: price below it is still a breakout
    v = [100.0] + [10.0] * 60 + [11.0]
    s = _series(v)
    assert pb.darvas_breakout(s, s.index[61]) is True
    # the same spike inside the 60-bar window blocks the breakout
    v2 = [10.0] + [100.0] + [10.0] * 59 + [11.0]
    s2 = _series(v2)
    assert pb.darvas_breakout(s2, s2.index[61]) is False


def test_darvas_returns_python_bool(pb):
    s = _series(np.arange(1, 100))
    assert type(pb.darvas_breakout(s, s.index[80])) is bool


# ── metrics ───────────────────────────────────────────────────────────────────
def test_metrics_empty(pb):
    assert pb.metrics([]) == {"n_months": 0}


def test_metrics_known_values(pb):
    m = pb.metrics([10.0, -10.0, 20.0, 0.0])
    r = np.array([0.1, -0.1, 0.2, 0.0])
    eq = np.cumprod(1 + r)
    assert m["n_months"] == 4
    assert m["avg_mth%"] == round(r.mean() * 100, 3) == 5.0
    assert m["ann_return%"] == round((eq[-1] ** 3 - 1) * 100, 2)
    assert m["hit_rate%"] == 50.0
    assert m["sharpe"] == round(r.mean() / r.std() * np.sqrt(12), 2)
    assert m["max_dd%"] == -10.0


def test_metrics_constant_returns_have_zero_sharpe_and_no_drawdown(pb):
    m = pb.metrics([2.0, 2.0, 2.0])
    assert m["sharpe"] == 0 and m["max_dd%"] == 0.0
    assert m["hit_rate%"] == 100.0
    assert m["ann_return%"] == round((1.02 ** 12 - 1) * 100, 2)


def test_metrics_single_month_annualises_twelvefold(pb):
    m = pb.metrics([1.0])
    assert m["n_months"] == 1 and m["sharpe"] == 0
    assert m["ann_return%"] == round((1.01 ** 12 - 1) * 100, 2)


def test_metrics_drawdown_is_peak_to_trough(pb):
    m = pb.metrics([50.0, -50.0, -50.0])      # 1.5 -> 0.75 -> 0.375: dd = (0.375-1.5)/1.5
    assert m["max_dd%"] == -75.0


def test_metrics_zero_return_is_not_a_hit(pb):
    assert pb.metrics([0.0, 0.0, 1.0])["hit_rate%"] == pytest.approx(33.3)


# ── load_scan_universe ────────────────────────────────────────────────────────
class _XL:
    df = None
    opened = []

    def __init__(self, path):
        _XL.opened.append(path)

    def parse(self, sheet):
        assert sheet == "All_Stocks"
        return _XL.df


@pytest.fixture
def scan(pb, monkeypatch):
    _XL.opened = []
    monkeypatch.setattr(pb.pd, "ExcelFile", _XL)
    monkeypatch.setattr(pb.glob, "glob", lambda p, recursive=False: ["/d/z2.xlsx", "/d/a1.xlsx"])
    return _XL


def test_load_scan_universe_no_files(pb, monkeypatch):
    monkeypatch.setattr(pb.glob, "glob", lambda p, recursive=False: [])
    assert pb.load_scan_universe() == []


def test_load_scan_universe_strips_and_uses_latest_file(pb, scan):
    scan.df = pd.DataFrame({"Symbol": [" AAPL ", "MSFT\t", "NVDA"]})
    assert pb.load_scan_universe() == ["AAPL", "MSFT", "NVDA"]
    assert scan.opened == ["/d/z2.xlsx"]


def test_load_scan_universe_limit(pb, scan):
    scan.df = pd.DataFrame({"Symbol": list("ABCDE")})
    assert pb.load_scan_universe(2) == ["A", "B"]
    assert pb.load_scan_universe(0) == list("ABCDE")


def test_load_scan_universe_drops_blank_strings(pb, scan):
    scan.df = pd.DataFrame({"Symbol": ["A", "   ", "", "B"]})
    assert pb.load_scan_universe() == ["A", "B"]


@pytest.mark.xfail(PANDAS3, strict=True, raises=AssertionError,
                   reason="pit_backtest.py:58 NaN symbols survive the 'nan' string filter under pandas>=3")
def test_load_scan_universe_drops_nan(pb, scan):
    scan.df = pd.DataFrame({"Symbol": ["A", np.nan, "B"]})
    assert pb.load_scan_universe() == ["A", "B"]


# ── main ──────────────────────────────────────────────────────────────────────
def _prices(n, start="2021-01-04", daily=0.001, base=100.0, vol=1e6):
    idx = pd.bdate_range(start, periods=n)
    close = base * (1 + daily) ** np.arange(n)
    return pd.DataFrame({"Close": close, "Volume": np.full(n, vol)}, index=idx)


def _snap(idx, ts):
    return idx[idx <= ts][-1]


@pytest.fixture
def run(pb, monkeypatch, tmp_path):
    def _run(ohlc, *argv, piotroski=None, coffee=None, as_of=None):
        calls = {"pio": [], "cc": [], "dl": None}
        ms = types.ModuleType("market_store")

        def cached_download(tickers, years):
            calls["dl"] = (list(tickers), years)
            return ohlc

        ms.cached_download = cached_download
        monkeypatch.setitem(sys.modules, "market_store", ms)

        def pio(t, d):
            calls["pio"].append((t, d))
            return (piotroski or (lambda t, d: (None, {})))(t, d)

        def cc(t, d, mktcap=None):
            calls["cc"].append((t, d, mktcap))
            return (coffee or (lambda t, d, m: False))(t, d, mktcap), {}

        monkeypatch.setattr(pb, "piotroski_asof", pio)
        monkeypatch.setattr(pb, "coffeecan_asof", cc)
        monkeypatch.setattr(pb, "as_of", as_of or (lambda t, d: {}))
        db = tmp_path / "pit.db"
        monkeypatch.setattr(sys, "argv", ["pit_backtest.py", "--db", str(db), *argv])
        pb.main()
        c = sqlite3.connect(db)
        try:
            res = pd.read_sql("SELECT * FROM arm_summary", c)
        finally:
            c.close()
        return res.set_index("arm"), calls
    return _run


def _expected_arm(closes, picks_by_month, t_pairs, cost=0.10):
    rets = []
    for (t0, t1), picks in zip(t_pairs, picks_by_month):
        vals = []
        for t in picks:
            s = closes[t]
            e0, e1 = _snap(s.index, t0), _snap(s.index, t1)
            vals.append((s.loc[e1] / s.loc[e0] - 1) * 100)
        if vals:
            rets.append(float(np.mean(vals)) - cost)
    return rets


def _rebal_pairs(closes):
    all_idx = sorted(set().union(*[s.index for s in closes.values()]))
    rebal = pd.Series(1, index=pd.DatetimeIndex(all_idx)).resample("MS").first().index
    rebal = [d for d in rebal if d >= pd.Timestamp(all_idx[61])]
    return list(zip(rebal[:-1], rebal[1:]))


def test_main_arms_against_independent_reference(run, pb, capsys):
    ohlc = {"UP": _prices(330, daily=0.002), "DN": _prices(330, daily=-0.001, base=80.0),
            "FLAT": _prices(330, daily=0.0, base=50.0)}
    closes = {t: d["Close"] for t, d in ohlc.items()}
    pairs = _rebal_pairs(closes)
    res, calls = run(ohlc, "--limit", "3",
                     piotroski=lambda t, d: (7, {}) if t == "UP" else (6, {}),
                     coffee=lambda t, d, m: True)
    # UP rises every day -> breakout every month; FLAT ties its own prior max -> breakout too; DN never
    breakout = ["UP", "FLAT"]
    exp_a = _expected_arm(closes, [breakout] * len(pairs), pairs)
    exp_b = _expected_arm(closes, [["UP"]] * len(pairs), pairs)
    exp_bench = [float(np.mean([(s.loc[_snap(s.index, t1)] / s.loc[_snap(s.index, t0)] - 1) * 100
                                for s in closes.values()])) - 0.10 for t0, t1 in pairs]
    assert list(res.index) == ["A_darvas", "B_darvas_F7", "C_triple_hit", "BENCH_all"]
    for arm, exp in (("A_darvas", exp_a), ("B_darvas_F7", exp_b), ("C_triple_hit", exp_b),
                     ("BENCH_all", exp_bench)):
        ref = pb.metrics(exp)
        assert res.loc[arm, "n_months"] == ref["n_months"] == len(pairs)
        assert res.loc[arm, "avg_mth%"] == pytest.approx(ref["avg_mth%"])
        assert res.loc[arm, "ann_return%"] == pytest.approx(ref["ann_return%"])
        assert res.loc[arm, "hit_rate%"] == pytest.approx(ref["hit_rate%"])
        assert res.loc[arm, "max_dd%"] == pytest.approx(ref["max_dd%"])
    assert res.loc["A_darvas", "avg_picks"] == "2.0" or float(res.loc["A_darvas", "avg_picks"]) == 2.0
    assert float(res.loc["B_darvas_F7", "avg_picks"]) == 1.0
    assert res.loc["BENCH_all", "avg_picks"] == "all"
    cap = capsys.readouterr()
    assert "TRIPLE-HIT vs GATE COMPONENTS" in cap.out and "A=Darvas" in cap.out
    assert "3 usable series" in cap.err and "breakouts=2 F7=1 triple=1" in cap.err


def test_main_piotroski_threshold_is_inclusive_at_seven(run):
    ohlc = {"A": _prices(330, daily=0.002), "B": _prices(330, daily=0.002, base=90.0),
            "C": _prices(330, daily=0.002, base=70.0), "D": _prices(330, daily=0.002, base=60.0)}
    F = {"A": 7, "B": 6, "C": None, "D": 9}
    res, _ = run(ohlc, piotroski=lambda t, d: (F[t], {}))
    assert float(res.loc["A_darvas", "avg_picks"]) == 4.0
    assert float(res.loc["B_darvas_F7", "avg_picks"]) == 2.0       # A (7) and D (9) only
    assert res.loc["C_triple_hit", "n_months"] == 0                # coffee-can fake says False


def test_main_empty_arm_reports_zero_months_and_zero_picks(run):
    ohlc = {"DN": _prices(330, daily=-0.002), "DN2": _prices(330, daily=-0.001, base=70.0)}
    res, _ = run(ohlc)
    assert res.loc["A_darvas", "n_months"] == 0
    assert res.loc["A_darvas", "avg_picks"] == 0.0 or float(res.loc["A_darvas", "avg_picks"]) == 0.0
    assert res.loc["BENCH_all", "n_months"] > 0


def test_main_coffee_can_gets_pit_date_and_market_cap(run):
    ohlc = {"UP": _prices(330, daily=0.002)}
    s = ohlc["UP"]["Close"]
    res, calls = run(ohlc, piotroski=lambda t, d: (8, {}), coffee=lambda t, d, m: True,
                     as_of=lambda t, d: {"shares": 1000.0})
    assert calls["cc"], "Coffee-Can must be consulted for F>=7 breakouts"
    for t, d, mcap in calls["cc"]:
        t0 = pd.Timestamp(d)
        assert mcap == pytest.approx(s.loc[_snap(s.index, t0)] * 1000.0)
        assert t0.day == 1                         # rebalance dates are month starts
    assert [d for _, d in calls["pio"]] == [d for _, d, _ in calls["cc"]]


def test_main_missing_shares_gives_none_mktcap(run):
    ohlc = {"UP": _prices(330, daily=0.002)}
    _, calls = run(ohlc, piotroski=lambda t, d: (8, {}), coffee=lambda t, d, m: True,
                   as_of=lambda t, d: {"shares": 0})
    assert calls["cc"] and all(m is None for _, _, m in calls["cc"])


def test_main_liquidity_filter_and_short_series(run, capsys):
    ohlc = {"LIQ": _prices(330, daily=0.002),
            "ILLIQ": _prices(330, daily=0.002, vol=1.0),
            "SHORT": _prices(80, daily=0.002),            # 80 <= BREAKOUT_LB + 20 -> unusable
            "NOCLOSE": pd.DataFrame({"Volume": [1.0] * 300}, index=pd.bdate_range("2021-01-04", periods=300))}
    res, calls = run(ohlc, "--min-dollar-vol", "2e6", piotroski=lambda t, d: (8, {}))
    err = capsys.readouterr().err
    assert "2 usable series" in err        # LIQ and ILLIQ pass the length check
    assert "2 -> 1 tradeable names" in err
    assert "$2,000,000/day" in err
    assert float(res.loc["A_darvas", "avg_picks"]) == 1.0


def test_main_min_dollar_vol_boundary_is_inclusive(run, capsys):
    hi = _prices(330, daily=0.002, base=100.0, vol=1e4)
    lo = _prices(330, daily=0.002, base=100.0, vol=1e4 * 0.999)
    thr = float((hi["Close"] * hi["Volume"]).tail(252).median())     # exactly hi's median $-volume
    run({"HI": hi, "LO": lo}, "--min-dollar-vol", repr(thr))
    assert "2 -> 1 tradeable names" in capsys.readouterr().err


@pytest.mark.xfail(strict=True, raises=KeyError,
                   reason="pit_backtest.py:181 every arm empty (e.g. liquidity filter removes all names) -> "
                          "pd.DataFrame(rows)[[...]] KeyError instead of an empty report")
def test_main_nothing_tradeable_still_writes_report(run):
    res, _ = run({"X": _prices(330, daily=0.002, vol=1.0)})
    assert (res["n_months"] == 0).all()


def test_main_zero_price_does_not_poison_benchmark(run):
    z = _prices(330, daily=0.0, base=100.0)
    # a month where the close at the rebalance date is exactly 0 -> forward return is inf
    z.loc["2021-12-01":"2021-12-31", "Close"] = 0.0
    ohlc = {"UP": _prices(330, daily=0.002), "ZERO": z}
    res, _ = run(ohlc)
    row = res.loc["BENCH_all"]
    assert np.isfinite(row["avg_mth%"]) and np.isfinite(row["ann_return%"])


def test_main_limit_and_universe_selection(run, pb, monkeypatch):
    ohlc = {"UP": _prices(330, daily=0.002)}
    _, calls = run(ohlc, "--limit", "7", "--years", "3")
    assert calls["dl"] == (pb.UNIVERSE[:7], 3)
    _, calls = run(ohlc)
    assert calls["dl"] == (pb.UNIVERSE, 5)
    monkeypatch.setattr(pb, "load_scan_universe", lambda limit=None: ["Q1", "Q2", "Q3"][:limit])
    _, calls = run(ohlc, "--universe", "scan", "--limit", "2")
    assert calls["dl"] == (["Q1", "Q2"], 5)


def test_main_scan_universe_without_limit(run, pb, monkeypatch):
    monkeypatch.setattr(pb, "load_scan_universe", lambda limit=None: ["Q1", "Q2", "Q3"])
    _, calls = run({"UP": _prices(330, daily=0.002)}, "--universe", "scan")
    assert calls["dl"][0] == ["Q1", "Q2", "Q3"]


def test_main_db_replaced_on_rerun(run, tmp_path):
    ohlc = {"UP": _prices(330, daily=0.002)}
    run(ohlc)
    res, _ = run(ohlc)
    assert len(res) == 4


def test_main_ticker_listed_midway_is_skipped_before_it_exists(run, capsys):
    ohlc = {"UP": _prices(330, daily=0.002), "LATE": _prices(150, start="2021-09-01", daily=0.002, base=60.0)}
    res, _ = run(ohlc)
    lines = [l for l in capsys.readouterr().err.splitlines() if "breakouts=" in l]
    assert lines[0].strip().endswith("breakouts=1 F7=0 triple=0")           # LATE not yet trading
    assert lines[-1].strip().endswith("breakouts=2 F7=0 triple=0")          # LATE trading and rising
    assert res.loc["BENCH_all", "n_months"] == len(lines)
