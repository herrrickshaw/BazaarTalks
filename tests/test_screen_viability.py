"""Offline tests for screen_viability.py (full-universe price/technical screen viability backtest)."""
import os
import sqlite3
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import screen_viability as sv  # noqa: E402


def _ohlcv(n=400, drift=0.0005, amp=0.01, seed=0, price0=100.0, vol=1e6, start="2020-01-01"):
    rng = np.random.default_rng(seed)
    close = price0 * np.cumprod(1 + drift + amp * rng.standard_normal(n))
    return pd.DataFrame({
        "High": close * 1.01, "Low": close * 0.99, "Close": close,
        "Volume": vol * (1 + 0.1 * np.abs(rng.standard_normal(n))),
    }, index=pd.bdate_range(start, periods=n))


def _flat(closes, vol=1e6, start="2020-01-01"):
    c = np.asarray(closes, dtype=float)
    v = np.full(len(c), vol) if np.isscalar(vol) else np.asarray(vol, dtype=float)
    return pd.DataFrame({"High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": v},
                        index=pd.bdate_range(start, periods=len(c)))


# ── load_universe ─────────────────────────────────────────────────────────────
class _XL:
    df = None
    opened = []

    def __init__(self, p):
        _XL.opened.append(p)

    def parse(self, sheet):
        assert sheet == "All_Stocks"
        return _XL.df


@pytest.fixture
def excel(monkeypatch):
    _XL.opened = []
    seen = []
    monkeypatch.setattr(sv.pd, "ExcelFile", _XL)
    monkeypatch.setattr(sv.glob, "glob", lambda p, recursive=False: (seen.append(p), ["/d/b.xlsx", "/d/a.xlsx"])[1])
    _XL.globs = seen
    return _XL


def test_scan_globs_cover_five_markets():
    assert set(sv.SCAN_GLOBS) == {"US", "India", "Japan", "Korea", "Europe"}


def test_load_universe_no_files_returns_empty(monkeypatch):
    monkeypatch.setattr(sv.glob, "glob", lambda p, recursive=False: [])
    assert sv.load_universe("US") == []


def test_load_universe_uses_market_glob_and_latest_file(excel):
    excel.df = pd.DataFrame({"Symbol": ["A", "B"]})
    sv.load_universe("Europe")
    assert excel.opened == ["/d/b.xlsx"]
    assert excel.globs[0].endswith(sv.SCAN_GLOBS["Europe"])
    assert excel.globs[0].startswith(sv.DL)


def test_load_universe_us_europe_use_symbol_and_strip(excel):
    excel.df = pd.DataFrame({"Symbol": [" AAPL ", "MSFT", "AAPL", np.nan, "NVDA"]})
    assert sv.load_universe("US") == ["AAPL", "MSFT", "NVDA"]        # deduped, order kept, NaN dropped
    assert sv.load_universe("Europe") == ["AAPL", "MSFT", "NVDA"]


def test_load_universe_india_concatenates_suffix(excel):
    excel.df = pd.DataFrame({"Symbol": ["TCS ", "INFY"], "Suffix": [".NS", " .BO"]})
    assert sv.load_universe("India") == ["TCS.NS", "INFY.BO"]


@pytest.mark.parametrize("market", ["Japan", "Korea"])
def test_load_universe_asia_uses_yf_ticker_and_skips_blank(excel, market):
    excel.df = pd.DataFrame({"YF_Ticker": ["7203.T", np.nan, "", "7203.T", "6758.T"], "Symbol": list("abcde")})
    assert sv.load_universe(market) == ["7203.T", "6758.T"]


def test_load_universe_limit_applies_after_dedupe(excel):
    excel.df = pd.DataFrame({"Symbol": ["A", "A", "B", "C"]})
    assert sv.load_universe("US", 2) == ["A", "B"]
    assert sv.load_universe("US", None) == ["A", "B", "C"]


# ── screens_for ───────────────────────────────────────────────────────────────
def test_screens_for_returns_five_named_boolean_series():
    sig = sv.screens_for(_ohlcv(400))
    assert set(sig) == {"rsi_oversold", "near_52w_high", "price_vol_breakout",
                        "darvas_proximity", "golden_crossover"}
    for s in sig.values():
        assert isinstance(s, pd.Series) and s.dtype == bool and len(s) == 400


def test_rsi_oversold_on_persistent_decline_not_on_rise():
    dn = sv.screens_for(_flat(np.linspace(200, 50, 300)))["rsi_oversold"]
    assert dn.iloc[-1] and not dn.iloc[:13].any()          # RSI undefined before 14 bars
    up = sv.screens_for(_flat(np.linspace(50, 200, 300)))["rsi_oversold"]
    assert not up.any()                                    # no losses -> RSI NaN -> never oversold


def test_near_52w_high_boundary():
    base = [100.0] * 300
    assert sv.screens_for(_flat(base + [90.0]))["near_52w_high"].iloc[-1]          # exactly 90% counts
    assert not sv.screens_for(_flat(base + [89.99]))["near_52w_high"].iloc[-1]
    assert sv.screens_for(_flat(base))["near_52w_high"].iloc[-1]


def test_near_52w_high_needs_120_bars():
    s = sv.screens_for(_flat([100.0] * 130))["near_52w_high"]
    assert not s.iloc[:119].any() and s.iloc[119:].all()


def test_price_vol_breakout_needs_volume_spike_and_up_day():
    c = [100.0] * 60 + [101.0]
    spike = sv.screens_for(_flat(c, vol=[1e6] * 60 + [1e7]))["price_vol_breakout"]
    assert spike.iloc[-1] and not spike.iloc[:-1].any()
    mild = sv.screens_for(_flat(c, vol=[1e6] * 60 + [3e6]))["price_vol_breakout"]
    assert not mild.iloc[-1]
    down = sv.screens_for(_flat([100.0] * 60 + [99.0], vol=[1e6] * 60 + [1e7]))["price_vol_breakout"]
    assert not down.iloc[-1]
    flat = sv.screens_for(_flat([100.0] * 61, vol=[1e6] * 60 + [1e7]))["price_vol_breakout"]
    assert not flat.iloc[-1]                               # unchanged close is not an up day


def test_darvas_proximity_all_conditions():
    # 300 bars rising 20 -> 100 then within 10% of the high, price>10, volume>1e5
    rise = list(np.linspace(20, 100, 300))
    ok = sv.screens_for(_flat(rise, vol=2e5))["darvas_proximity"]
    assert ok.iloc[-1]
    assert not sv.screens_for(_flat(rise, vol=1e5))["darvas_proximity"].iloc[-1]      # vol must be > 1e5
    low_px = sv.screens_for(_flat(list(np.linspace(2, 10, 300)), vol=2e5))["darvas_proximity"]
    assert not low_px.iloc[-1]                              # price must be > 10 (10.0 is not)
    off_high = sv.screens_for(_flat(rise + [85.0], vol=2e5))["darvas_proximity"]
    assert not off_high.iloc[-1]                            # >10% below the 52w high
    flat = sv.screens_for(_flat([50.0] * 300, vol=2e5))["darvas_proximity"]
    assert not flat.iloc[-1]                                # needs >=10% above the 52w low


def test_golden_crossover_fires_only_on_the_cross_bar():
    c = np.concatenate([np.linspace(200, 100, 300), np.linspace(100, 400, 100)])
    gc = sv.screens_for(_flat(c))["golden_crossover"]
    d = pd.Series(c)
    cross = ((d.rolling(50).mean() > d.rolling(200).mean()) &
             (d.rolling(50).mean().shift(1) <= d.rolling(200).mean().shift(1))).values
    assert gc.sum() == cross.sum() == 1
    assert np.array_equal(gc.values, cross)


# ── eval_ticker ───────────────────────────────────────────────────────────────
def test_eval_ticker_short_or_none_returns_empty():
    assert sv.eval_ticker(None) == []
    assert sv.eval_ticker(_ohlcv(sv.MIN_BARS + 5 - 1)) == []
    assert sv.eval_ticker(_ohlcv(sv.MIN_BARS + 5)) != []


def test_eval_ticker_horizon_changes_min_length():
    assert sv.eval_ticker(_ohlcv(sv.MIN_BARS + 20 - 1), horizon=20) == []
    assert sv.eval_ticker(_ohlcv(sv.MIN_BARS + 20), horizon=20) != []


def test_eval_ticker_row_schema_and_edge_identity():
    rows = sv.eval_ticker(_ohlcv(600, seed=3))
    assert [r["screen"] for r in rows] == ["rsi_oversold", "near_52w_high", "price_vol_breakout",
                                           "darvas_proximity", "golden_crossover"]
    for r in rows:
        assert set(r) == {"screen", "n_signals", "avg_fwd5d", "hit_pct", "baseline", "edge"}
        if r["n_signals"]:
            assert r["edge"] == pytest.approx(round(r["avg_fwd5d"] - r["baseline"], 4), abs=2e-4)
            assert 0 <= r["hit_pct"] <= 100
        else:
            assert r["avg_fwd5d"] is None and r["hit_pct"] is None and r["edge"] is None


def test_eval_ticker_values_against_manual_reference():
    df = _ohlcv(600, seed=4)
    rows = {r["screen"]: r for r in sv.eval_ticker(df)}
    close = df["Close"]
    fwd = (close.pct_change(5).shift(-5) * 100).clip(-30, 30)
    mask = sv.screens_for(df)["near_52w_high"]
    r = fwd[mask].dropna()
    assert rows["near_52w_high"]["n_signals"] == len(r)
    assert rows["near_52w_high"]["avg_fwd5d"] == round(float(r.mean()), 4)
    assert rows["near_52w_high"]["hit_pct"] == round(float((r > 0).mean() * 100), 1)
    assert rows["near_52w_high"]["baseline"] == round(float(fwd.mean()), 4)


def test_eval_ticker_clip_limits_outlier_influence():
    c = np.concatenate([np.full(400, 100.0), [100.0] * 5])
    # one +1000% spike 5 bars ahead of the (always true) near-high signal bars
    c[300 + 5] = 1100.0
    df = _flat(c)
    wide = {r["screen"]: r for r in sv.eval_ticker(df, clip=10000)}["near_52w_high"]
    tight = {r["screen"]: r for r in sv.eval_ticker(df, clip=30)}["near_52w_high"]
    assert wide["avg_fwd5d"] > tight["avg_fwd5d"]
    assert tight["avg_fwd5d"] <= 30


def test_eval_ticker_signal_with_no_forward_return_counts_zero():
    # a golden cross on the very last bar has no 5d forward return -> n_signals 0, None metrics
    c = np.concatenate([np.linspace(200, 100, 300), np.linspace(100, 400, 100)])
    d = pd.Series(c)
    cross = int(np.where((d.rolling(50).mean() > d.rolling(200).mean()) &
                         (d.rolling(50).mean().shift(1) <= d.rolling(200).mean().shift(1)))[0][0])
    assert cross + 1 >= sv.MIN_BARS + 5
    rows = {r["screen"]: r for r in sv.eval_ticker(_flat(c[:cross + 1]))}
    assert rows["golden_crossover"]["n_signals"] == 0 and rows["golden_crossover"]["edge"] is None


@pytest.fixture
def fake_ml(monkeypatch):
    """Real ml_signal_engine functions are exercised; see test below for the failure path."""
    import ml_signal_engine as mse
    return mse


def test_eval_ticker_ml_bullish_row_present_with_real_engine(fake_ml):
    df = _ohlcv(700, drift=0.002, amp=0.01, seed=5)
    eng = fake_ml.MLSignalEngine(model_type="ridge")
    rows = {r["screen"]: r for r in sv.eval_ticker(df, include_ml=True, ml_engine=eng)}
    assert "ml_bullish" in rows
    r = rows["ml_bullish"]
    assert r["n_signals"] >= 0 and (r["n_signals"] == 0) == (r["edge"] is None)


def test_eval_ticker_ml_skipped_without_engine_or_flag(fake_ml):
    df = _ohlcv(700, seed=5)
    assert "ml_bullish" not in {r["screen"] for r in sv.eval_ticker(df, include_ml=True, ml_engine=None)}
    eng = fake_ml.MLSignalEngine(model_type="ridge")
    assert "ml_bullish" not in {r["screen"] for r in sv.eval_ticker(df, include_ml=False, ml_engine=eng)}


def test_eval_ticker_ml_failure_is_swallowed(fake_ml):
    class Boom:
        def _make_sequences(self, *a):
            raise RuntimeError("x")
    df = _ohlcv(700, seed=5)
    rows = sv.eval_ticker(df, include_ml=True, ml_engine=Boom())
    assert "ml_bullish" not in {r["screen"] for r in rows}
    assert len(rows) == 5


def test_eval_ticker_ml_short_history_gives_empty_signal_row(fake_ml):
    df = _ohlcv(sv.MIN_BARS + 10, seed=5)           # too short for the walk-forward window
    eng = fake_ml.MLSignalEngine(model_type="ridge")
    rows = {r["screen"]: r for r in sv.eval_ticker(df, include_ml=True, ml_engine=eng)}
    assert rows["ml_bullish"]["n_signals"] == 0 and rows["ml_bullish"]["avg_fwd5d"] is None


# ── sqlite helpers ────────────────────────────────────────────────────────────
def test_init_db_creates_tables_and_is_idempotent(tmp_path):
    p = tmp_path / "v.db"
    sv.init_db(str(p)).close()
    conn = sv.init_db(str(p))
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"ticker_screen", "done"} <= names
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
    conn.close()


def test_already_done_filters_by_market(tmp_path):
    conn = sv.init_db(str(tmp_path / "v.db"))
    conn.executemany("INSERT INTO done VALUES (?,?)", [("US", "A"), ("US", "B"), ("India", "C.NS")])
    assert sv.already_done(conn, "US") == {"A", "B"}
    assert sv.already_done(conn, "Korea") == set()
    conn.close()


def _insert(conn, rows):
    conn.executemany("INSERT INTO ticker_screen VALUES (?,?,?,?,?,?,?,?)", rows)


def test_write_summary_aggregates_and_viability_rule(tmp_path):
    conn = sv.init_db(str(tmp_path / "v.db"))
    _insert(conn, [
        # market, ticker, screen, n, avg, hit, base, edge
        ("US", "A", "s1", 10, 1.0, 60.0, 0.5, 0.5),
        ("US", "B", "s1", 20, 2.0, 70.0, 0.5, 1.5),
        ("US", "C", "s1", 0, None, None, 0.5, None),        # no signals -> excluded
        ("US", "A", "s2", 5, 0.2, 50.0, 0.5, 0.1),          # hit exactly 50 -> not viable
        ("US", "B", "s2", 5, 0.2, 50.0, 0.5, 0.1),
        ("US", "A", "s3", 5, -1.0, 70.0, 0.5, -1.5),        # negative edge -> not viable
        ("India", "X", "s1", 7, 1.0, 55.0, 0.0, 1.0),
        ("India", "Y", "s1", 3, -0.5, 40.0, 0.0, -0.5),
    ])
    conn.commit()
    sv.write_summary(conn)
    df = pd.read_sql("SELECT * FROM market_screen_summary", conn).set_index(["market", "screen"])
    r = df.loc[("US", "s1")]
    assert r.n_tickers == 2 and r.total_signals == 30
    assert r.avg_fwd5d == 1.5 and r.avg_hit_pct == 65.0 and r.avg_edge == 1.0
    assert r.pct_tickers_pos_edge == 100.0 and r.viable == "YES"
    assert df.loc[("US", "s2"), "viable"] == "no"          # AVG(hit) == 50 is not > 50
    assert df.loc[("US", "s3"), "viable"] == "no"
    assert df.loc[("India", "s1"), "pct_tickers_pos_edge"] == 50.0
    assert df.loc[("India", "s1"), "avg_edge"] == 0.25 and df.loc[("India", "s1"), "viable"] == "no"  # hit avg 47.5
    # ordered by market then avg_edge desc
    order = list(pd.read_sql("SELECT market, screen FROM market_screen_summary", conn).itertuples(index=False))
    assert order == [("India", "s1"), ("US", "s1"), ("US", "s2"), ("US", "s3")]
    conn.close()


def test_write_summary_replaces_previous_table(tmp_path):
    conn = sv.init_db(str(tmp_path / "v.db"))
    _insert(conn, [("US", "A", "s1", 1, 1.0, 60.0, 0.0, 1.0)])
    sv.write_summary(conn)
    conn.execute("DELETE FROM ticker_screen")
    sv.write_summary(conn)
    assert conn.execute("SELECT COUNT(*) FROM market_screen_summary").fetchone()[0] == 0
    conn.close()


# ── main ──────────────────────────────────────────────────────────────────────
@pytest.fixture
def run_main(monkeypatch, tmp_path):
    def _run(*argv, data=None, universes=None):
        calls = {"dl": [], "eval": [], "lu": []}
        ms = types.ModuleType("market_store")

        def cached_download(batch, years):
            calls["dl"].append((list(batch), years))
            return {t: data[t] for t in batch if data and t in data}

        ms.cached_download = cached_download
        monkeypatch.setitem(sys.modules, "market_store", ms)

        def lu(market, limit=None):
            calls["lu"].append((market, limit))
            return list((universes or {}).get(market, []))

        monkeypatch.setattr(sv, "load_universe", lu)
        real_eval = sv.eval_ticker

        def spy(df, horizon=sv.FWD, clip=30, include_ml=False, ml_engine=None):
            calls["eval"].append((horizon, clip, include_ml, ml_engine))
            return real_eval(df, horizon, clip, False, None)
        monkeypatch.setattr(sv, "eval_ticker", spy)
        db = tmp_path / "v.db"
        monkeypatch.setattr(sys, "argv", ["screen_viability.py", "--db", str(db), *argv])
        sv.main()
        return db, calls
    return _run


def _q(db, sql):
    c = sqlite3.connect(db)
    try:
        return pd.read_sql(sql, c)
    finally:
        c.close()


def test_main_processes_batches_and_marks_done(run_main, capsys):
    data = {t: _ohlcv(500, seed=i) for i, t in enumerate("ABCDE")}
    db, calls = run_main("--markets", "US", "--batch", "2", "--limit", "5",
                         data=data, universes={"US": list("ABCDE")})
    assert [b for b, _ in calls["dl"]] == [["A", "B"], ["C", "D"], ["E"]]
    assert all(y == 5 for _, y in calls["dl"])
    assert calls["lu"] == [("US", 5)]
    assert set(_q(db, "SELECT ticker FROM done").ticker) == set("ABCDE")
    ts = _q(db, "SELECT * FROM ticker_screen")
    assert set(ts.ticker) == set("ABCDE") and len(ts) == 25 and set(ts.market) == {"US"}
    cap = capsys.readouterr()
    assert "[US] universe=5 done=0 todo=5" in cap.err
    assert "[US] 2/5 processed" in cap.err and "[US] 5/5 processed" in cap.err
    assert "SCREEN VIABILITY BY MARKET" in cap.out


def test_main_is_resumable_skips_done_tickers(run_main, tmp_path, capsys):
    data = {t: _ohlcv(500, seed=i) for i, t in enumerate("ABC")}
    db0 = tmp_path / "v.db"
    conn = sv.init_db(str(db0))
    conn.execute("INSERT INTO done VALUES ('US','A')"); conn.commit(); conn.close()
    db, calls = run_main("--markets", "US", data=data, universes={"US": list("ABC")})
    assert [b for b, _ in calls["dl"]] == [["B", "C"]]
    assert "universe=3 done=1 todo=2" in capsys.readouterr().err


def test_main_missing_download_marks_ticker_done_without_rows(run_main):
    data = {"A": _ohlcv(500, seed=1)}
    db, _ = run_main("--markets", "US", data=data, universes={"US": ["A", "GONE"]})
    assert set(_q(db, "SELECT ticker FROM done").ticker) == {"A", "GONE"}
    assert set(_q(db, "SELECT ticker FROM ticker_screen").ticker) == {"A"}


@pytest.mark.parametrize("horizon,clip", [(5, 30), (20, 60), (21, 61), (1, 13)])
def test_main_clip_scales_with_sqrt_horizon(run_main, horizon, clip):
    data = {"A": _ohlcv(500, seed=1)}
    _, calls = run_main("--markets", "US", "--horizon", str(horizon),
                        data=data, universes={"US": ["A"]})
    assert calls["eval"][0][:2] == (horizon, clip)


def test_main_include_ml_builds_engine_once(run_main, monkeypatch):
    built = []

    class FakeEngine:
        def __init__(self, model_type):
            built.append(model_type)
    fake = types.ModuleType("ml_signal_engine")
    fake.MLSignalEngine = FakeEngine
    monkeypatch.setitem(sys.modules, "ml_signal_engine", fake)
    data = {"A": _ohlcv(500, seed=1), "B": _ohlcv(500, seed=2)}
    _, calls = run_main("--markets", "US", "--include-ml", data=data, universes={"US": ["A", "B"]})
    assert built == ["ridge"]
    assert all(c[2] is True and isinstance(c[3], FakeEngine) for c in calls["eval"])


def test_main_without_ml_flag_passes_no_engine(run_main):
    _, calls = run_main("--markets", "US", data={"A": _ohlcv(500)}, universes={"US": ["A"]})
    assert calls["eval"][0][2:] == (False, None)


def test_main_multiple_markets_default_all_five(run_main):
    _, calls = run_main(data={}, universes={})
    assert [m for m, _ in calls["lu"]] == ["US", "India", "Japan", "Korea", "Europe"]


def test_main_summary_table_written(run_main):
    data = {"A": _ohlcv(500, seed=1), "B": _ohlcv(500, seed=2)}
    db, _ = run_main("--markets", "US", data=data, universes={"US": ["A", "B"]})
    s = _q(db, "SELECT * FROM market_screen_summary")
    assert set(s.market) == {"US"} and s.viable.isin(["YES", "no"]).all()


def test_main_export_summary_writes_small_db_and_exits(run_main, tmp_path, capsys, monkeypatch):
    db = tmp_path / "v.db"
    conn = sv.init_db(str(db))
    _insert(conn, [("US", "A", "s1", 10, 1.0, 60.0, 0.5, 0.5)])
    conn.commit(); conn.close()
    out = tmp_path / "summary.db"
    monkeypatch.setattr(sys, "argv", ["screen_viability.py", "--db", str(db), "--export-summary", str(out)])
    monkeypatch.setitem(sys.modules, "market_store", None)       # importing it would raise: must not be reached
    sv.main()
    c = sqlite3.connect(out)
    try:
        names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        rows = c.execute("SELECT market, screen, viable FROM market_screen_summary").fetchall()
    finally:
        c.close()
    assert names == {"market_screen_summary"}
    assert rows == [("US", "s1", "YES")]
    assert "exported summary (1 rows)" in capsys.readouterr().err
