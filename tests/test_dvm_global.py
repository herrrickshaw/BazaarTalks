"""Offline tests for dvm_global.py (global Trendlyne-style technical screens)."""
import concurrent.futures as cf
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dvm_global as dg  # noqa: E402


def _ohlcv(n=320, drift=0.001, amp=0.01, seed=0, price0=100.0, vol=1e6, end=None):
    rng = np.random.default_rng(seed)
    close = price0 * np.cumprod(1 + drift + amp * rng.standard_normal(n))
    idx = pd.bdate_range(end="2024-12-31", periods=n) if end is None else pd.bdate_range(end=end, periods=n)
    return pd.DataFrame({
        "High": close * 1.01, "Low": close * 0.99, "Close": close,
        "Volume": vol * (1 + 0.1 * np.abs(rng.standard_normal(n))),
    }, index=idx)


def _tech(df, mkt=None):
    mkt = mkt if mkt is not None else df["Close"].pct_change()
    return dg._tech(df["Close"], df["High"], df["Low"], df["Volume"], mkt)


def _long(frames: dict) -> pd.DataFrame:
    parts = []
    for sym, df in frames.items():
        d = df.reset_index().rename(columns={"index": "Date"})
        d["Symbol"] = sym
        parts.append(d[["Date", "Symbol", "Close", "High", "Low", "Volume"]])
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=0)   # unsorted on purpose


# ── _tech ─────────────────────────────────────────────────────────────────────
def test_tech_min_bars_boundary():
    assert _tech(_ohlcv(199)) is None
    assert _tech(_ohlcv(200)) is not None


def test_tech_keys_types_and_ranges():
    r = _tech(_ohlcv(320, seed=1))
    assert set(r) == {"M", "rsi", "mfi", "adx", "dist_52w", "above_200dma", "golden_cross",
                      "sma50_above_200", "macd_bull", "vol_ratio", "beta"}
    assert 0 <= r["M"] <= 100 and 0 <= r["rsi"] <= 100 and 0 <= r["mfi"] <= 100
    assert r["dist_52w"] <= 0.0 + 1e-9            # price never above its own 252d max
    for k in ("above_200dma", "golden_cross", "sma50_above_200", "macd_bull"):
        assert isinstance(r[k], bool)


def test_tech_uptrend_vs_downtrend():
    up = _tech(_ohlcv(320, drift=0.004, amp=0.004, seed=1))
    dn = _tech(_ohlcv(320, drift=-0.004, amp=0.004, seed=1))
    assert up["M"] > dn["M"]
    assert up["above_200dma"] and up["sma50_above_200"]
    assert not dn["above_200dma"] and not dn["sma50_above_200"]
    assert up["dist_52w"] > dn["dist_52w"]


def test_tech_new_high_has_zero_distance():
    n = 320
    c = np.linspace(50, 150, n)
    df = pd.DataFrame({"High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": np.full(n, 1e6)},
                      index=pd.bdate_range("2023-01-02", periods=n))
    r = _tech(df)
    assert r["dist_52w"] == 0.0
    assert r["vol_ratio"] == 1.0
    assert r["rsi"] is None            # no down-days -> RSI undefined
    assert r["above_200dma"] is True


def test_tech_zero_volume_ratio_defaults_to_one():
    df = _ohlcv(320, seed=2)
    df["Volume"] = 0.0
    assert _tech(df)["vol_ratio"] == 1.0


def _cross_frame(end="2024-12-31"):
    n = 400
    c = np.concatenate([np.linspace(200, 100, 300), np.linspace(100, 400, 100)])
    return pd.DataFrame({"High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": np.full(n, 1e6)},
                        index=pd.bdate_range(end=end, periods=n))


def _first_cross_index(df):
    d50 = df.Close.rolling(50).mean(); d200 = df.Close.rolling(200).mean()
    return [i for i in range(201, len(df)) if d50.iloc[i] > d200.iloc[i] and d50.iloc[i - 1] <= d200.iloc[i - 1]][0]


@pytest.mark.parametrize("sessions_after,expected", [(-1, False), (0, True), (4, True), (5, False)])
def test_tech_golden_cross_window_is_last_five_sessions(sessions_after, expected):
    df = _cross_frame()
    i = _first_cross_index(df)
    assert _tech(df.iloc[:i + sessions_after + 1])["golden_cross"] is expected


def test_tech_beta_against_market_series():
    df = _ohlcv(320, seed=3)
    mkt = df["Close"].pct_change()
    assert _tech(df, mkt)["beta"] == pytest.approx(1.0, abs=0.01)
    assert _tech(df, mkt * 2)["beta"] == pytest.approx(0.5, abs=0.01)
    assert _tech(df, -mkt)["beta"] == pytest.approx(-1.0, abs=0.01)


def test_tech_beta_none_for_zero_variance_market():
    df = _ohlcv(320, seed=3)
    mkt = pd.Series(0.0, index=df.index)
    assert _tech(df, mkt)["beta"] is None


# ── SCREENS ───────────────────────────────────────────────────────────────────
def _r(**kw):
    base = {"M": 0.0, "rsi": 60.0, "mfi": 60.0, "adx": 30.0, "dist_52w": -5.0, "above_200dma": True,
            "golden_cross": False, "sma50_above_200": True, "macd_bull": True, "vol_ratio": 1.5, "beta": 1.0}
    base.update(kw)
    return base


def test_screen_momentum_breakout_boundaries():
    s = dg.SCREENS["momentum_breakout"]
    assert s(_r(M=70, dist_52w=-10, adx=25, vol_ratio=1.2))
    assert not s(_r(M=69.9, dist_52w=-10, adx=25, vol_ratio=1.2))
    assert not s(_r(M=70, dist_52w=-10.1, adx=25, vol_ratio=1.2))
    assert not s(_r(M=70, dist_52w=-10, adx=24.9, vol_ratio=1.2))
    assert not s(_r(M=70, dist_52w=-10, adx=25, vol_ratio=1.19))
    assert not s(_r(M=70, dist_52w=-10, adx=None, vol_ratio=1.2))


def test_screen_high_momentum_boundary():
    s = dg.SCREENS["high_momentum"]
    assert s(_r(M=75)) and not s(_r(M=74.9))


def test_screen_golden_crossover_follows_flag():
    s = dg.SCREENS["golden_crossover"]
    assert s(_r(golden_cross=True)) and not s(_r(golden_cross=False))


def test_screen_uptrend_quality_boundaries():
    s = dg.SCREENS["uptrend_quality"]
    assert s(_r(rsi=50, adx=20)) and s(_r(rsi=70, adx=20))
    assert not s(_r(rsi=49.9, adx=20)) and not s(_r(rsi=70.1, adx=20))
    assert not s(_r(rsi=60, adx=19.9))
    assert not s(_r(above_200dma=False))
    assert not s(_r(rsi=None))
    assert not s(_r(adx=None))


def test_screen_trendlyne_technical_boundaries():
    s = dg.SCREENS["trendlyne_technical"]
    assert s(_r(rsi=50, mfi=50, macd_bull=True))
    assert not s(_r(rsi=49.9)) and not s(_r(rsi=70.1))
    assert not s(_r(mfi=49.9)) and not s(_r(mfi=None))
    assert not s(_r(macd_bull=False))


def test_screen_sma_golden_needs_both_flags():
    s = dg.SCREENS["sma_golden"]
    assert s(_r(sma50_above_200=True, above_200dma=True))
    assert not s(_r(sma50_above_200=False, above_200dma=True))
    assert not s(_r(sma50_above_200=True, above_200dma=False))


# ── process_market ────────────────────────────────────────────────────────────
@pytest.fixture
def seed(monkeypatch, tmp_path):
    d = tmp_path / "seed"
    d.mkdir()
    monkeypatch.setattr(dg, "SEED", str(d))
    store = {}

    def fake_read_parquet(path, *a, **k):
        return store[os.path.basename(path)].copy()

    monkeypatch.setattr(dg.pd, "read_parquet", fake_read_parquet)

    def put(market, frames):
        name = f"cleaned_long_{market}.parquet"
        (d / name).write_bytes(b"")
        store[name] = _long(frames)

    return put


def test_process_market_scores_every_long_enough_ticker(seed):
    seed("XX", {"UP": _ohlcv(320, drift=0.004, amp=0.004, seed=1),
                "DN": _ohlcv(320, drift=-0.004, amp=0.004, seed=1),
                "TINY": _ohlcv(100, seed=2)})
    rows, hits = dg.process_market("XX", "high_momentum")
    by = {r["ticker"]: r for r in rows}
    assert set(by) == {"UP", "DN"}          # TINY has < MIN_BARS observations
    assert all(r["market"] == "XX" for r in rows)
    assert by["UP"]["M"] > by["DN"]["M"]
    assert by["UP"]["above_200dma"] and not by["DN"]["above_200dma"]
    assert [h["ticker"] for h in hits] == [r["ticker"] for r in rows if r["M"] >= 75]


def test_process_market_min_bars_boundary(seed):
    seed("XX", {"A199": _ohlcv(199, seed=1, end="2024-12-31"),
                "A200": _ohlcv(200, seed=2, end="2024-12-31"),
                "LONG": _ohlcv(320, seed=3)})
    rows, _ = dg.process_market("XX", "high_momentum")
    got = {r["ticker"] for r in rows}
    assert "A200" in got and "A199" not in got and "LONG" in got


def test_process_market_matches_per_ticker_reference(seed):
    frames = {"A": _ohlcv(320, drift=0.003, amp=0.006, seed=5),
              "B": _ohlcv(320, drift=-0.001, amp=0.012, seed=6),
              "C": _ohlcv(320, drift=0.0, amp=0.02, seed=7)}
    seed("XX", frames)
    rows, _ = dg.process_market("XX", "high_momentum")
    by = {r["ticker"]: r for r in rows}
    for t, df in frames.items():
        ref = _tech(df, pd.DataFrame({k: f["Close"] for k, f in frames.items()}).pct_change().mean(axis=1))
        got = by[t]
        assert got["M"] == pytest.approx(ref["M"], abs=0.11)
        assert got["rsi"] == pytest.approx(ref["rsi"], abs=0.11)
        assert got["mfi"] == pytest.approx(ref["mfi"], abs=0.11)
        assert got["adx"] == pytest.approx(ref["adx"], abs=0.11)
        assert got["dist_52w"] == pytest.approx(ref["dist_52w"], abs=0.11)
        assert got["vol_ratio"] == pytest.approx(ref["vol_ratio"], abs=0.011)
        assert got["above_200dma"] == ref["above_200dma"]
        assert got["macd_bull"] == ref["macd_bull"]
        assert got["sma50_above_200"] == ref["sma50_above_200"]
        assert got["beta"] == pytest.approx(ref["beta"], abs=0.05)


def test_process_market_equal_weight_beta_of_identical_stocks_is_one(seed):
    base = _ohlcv(320, seed=9)
    seed("XX", {"A": base, "B": base.copy(), "C": base.copy()})
    rows, _ = dg.process_market("XX", "high_momentum")
    assert all(r["beta"] == pytest.approx(1.0, abs=0.02) for r in rows)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="dvm_global.py:139 beta = population cov / sample var -> 0.995 not 1.0 (_tech uses sample/sample)")
def test_process_market_beta_of_market_clone_is_exactly_one(seed):
    base = _ohlcv(320, seed=9)
    seed("XX", {"A": base, "B": base.copy(), "C": base.copy()})
    rows, _ = dg.process_market("XX", "high_momentum")
    assert all(r["beta"] == pytest.approx(1.0, abs=0.005) for r in rows)


def test_process_market_golden_cross_on_last_bar(seed):
    df = _cross_frame()
    i = _first_cross_index(df)
    seed("XX", {"S": df.iloc[:i + 1]})
    rows, hits = dg.process_market("XX", "golden_crossover")
    assert rows[0]["golden_cross"] is True and len(hits) == 1
    seed("XX", {"S": df.iloc[:i]})
    rows, hits = dg.process_market("XX", "golden_crossover")
    assert rows[0]["golden_cross"] is False and hits == []


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="dvm_global.py:147 gc uses shift(1) (cross on last bar only) but the module docstring and _tech use 'last 5 sessions'")
def test_process_market_golden_cross_within_last_five_sessions(seed):
    df = _cross_frame()
    i = _first_cross_index(df)
    seed("XX", {"S": df.iloc[:i + 3]})       # crossed 2 sessions before the last bar
    rows, _ = dg.process_market("XX", "golden_crossover")
    assert rows[0]["golden_cross"] is True


def test_process_market_all_tickers_too_short_gives_empty(seed):
    seed("XX", {"A": _ohlcv(50, seed=1)})
    rows, hits = dg.process_market("XX", "high_momentum")
    assert rows == [] and hits == []


def test_process_market_unknown_screen_raises_keyerror_only_with_rows(seed):
    seed("XX", {"A": _ohlcv(320, seed=1)})
    with pytest.raises(KeyError):
        dg.process_market("XX", "nope")


# ── main ──────────────────────────────────────────────────────────────────────
class _SyncExecutor:
    """In-process stand-in for ProcessPoolExecutor (no pickling / spawn)."""
    def __init__(self, max_workers=None):
        _SyncExecutor.workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def submit(self, fn, *args):
        f = cf.Future()
        try:
            f.set_result(fn(*args))
        except Exception as e:      # pragma: no cover
            f.set_exception(e)
        return f


@pytest.fixture
def run_main(monkeypatch, tmp_path, seed, capsys):
    monkeypatch.setattr(dg, "ProcessPoolExecutor", _SyncExecutor)

    def _run(*argv):
        db = tmp_path / "g.db"
        monkeypatch.setattr(sys, "argv", ["dvm_global.py", "--db", str(db), *argv])
        dg.main()
        return db, capsys.readouterr()
    return _run


def _setup_markets(seed):
    seed("AA", {"U1": _ohlcv(320, drift=0.004, amp=0.004, seed=1),
                "U2": _ohlcv(320, drift=0.004, amp=0.004, seed=2),
                "D1": _ohlcv(320, drift=-0.004, amp=0.004, seed=3)})
    seed("BB", {"X": _ohlcv(320, drift=-0.004, amp=0.004, seed=4)})
    seed("CC", {"S": _ohlcv(40, seed=5)})


def test_main_all_markets_by_default_and_db_schema(run_main, seed):
    _setup_markets(seed)
    db, cap = run_main("--screen", "high_momentum", "--workers", "3")
    assert "markets with data: ['AA', 'BB', 'CC']" in cap.err
    assert _SyncExecutor.workers == 3
    c = sqlite3.connect(db)
    try:
        df = pd.read_sql("SELECT * FROM dvm_global", c)
    finally:
        c.close()
    assert set(df.market) == {"AA", "BB"}           # CC has no tickers with enough history
    assert len(df) == 4
    assert list(df.columns) == ["market", "ticker", "M", "rsi", "mfi", "adx", "dist_52w",
                                "above_200dma", "golden_cross", "sma50_above_200", "macd_bull",
                                "vol_ratio", "beta"]
    assert set(df.above_200dma) <= {0, 1} and set(df.macd_bull) <= {0, 1}


def test_main_summary_sorted_by_hits_and_totals(run_main, seed):
    _setup_markets(seed)
    db, cap = run_main("--screen", "high_momentum")
    out = cap.out
    assert "GLOBAL DVM / TRENDLYNE SCREEN: high_momentum (3 markets)" in out
    lines = out.splitlines()
    mk = [l.split() for l in lines if l.startswith("  ") and l.split() and l.split()[0] in ("AA", "BB", "CC")]
    hits = [int(m[2]) for m in mk]
    assert hits == sorted(hits, reverse=True)
    total = [l.split() for l in lines if l.strip().startswith("TOTAL")][0]
    assert int(total[1]) == 4
    assert int(total[2]) == sum(hits)
    cc = [m for m in mk if m[0] == "CC"][0]
    assert cc[1:] == ["0", "0", "0"]                  # empty market: avg momentum reported as 0
    assert "all metrics saved to" in out
    assert "AA: scored     3" in cap.err


def test_main_markets_subset(run_main, seed):
    _setup_markets(seed)
    db, cap = run_main("--markets", "BB", "--screen", "golden_crossover")
    c = sqlite3.connect(db)
    try:
        df = pd.read_sql("SELECT market FROM dvm_global", c)
    finally:
        c.close()
    assert set(df.market) == {"BB"}
    assert "(1 markets)" in cap.out


def test_main_rerun_replaces_table(run_main, seed):
    _setup_markets(seed)
    db, _ = run_main("--markets", "AA")
    db, _ = run_main("--markets", "AA")
    c = sqlite3.connect(db)
    try:
        n = c.execute("SELECT COUNT(*) FROM dvm_global").fetchone()[0]
    finally:
        c.close()
    assert n == 3


def test_main_rejects_unknown_screen(run_main, seed):
    with pytest.raises(SystemExit):
        run_main("--screen", "nope")
