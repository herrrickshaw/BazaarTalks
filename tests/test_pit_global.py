"""Offline tests for pit_global.py (price-only point-in-time Darvas backtest for every market)."""
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


@pytest.fixture
def pg():
    """pit_global imported against a fake pit_fundamentals (the real one opens a SQLite file at import);
    pit_backtest / pit_global are re-imported fresh and sys.modules is restored afterwards."""
    names = ("pit_fundamentals", "pit_backtest", "pit_global")
    saved = {n: sys.modules.get(n) for n in names}
    fake = types.ModuleType("pit_fundamentals")
    fake.piotroski_asof = lambda t, d: (None, {})
    fake.coffeecan_asof = lambda t, d, mktcap=None: (False, {})
    fake.as_of = lambda t, d: {}
    fake._g = lambda lst, i=0: lst[i] if lst and len(lst) > i else None
    sys.modules["pit_fundamentals"] = fake
    sys.modules.pop("pit_backtest", None)
    sys.modules.pop("pit_global", None)
    mod = importlib.import_module("pit_global")
    try:
        yield mod
    finally:
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m


# ── market_cost ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("market,cost", [
    ("US", 0.10), ("JP", 0.20), ("KR", 0.25),
    ("DE", 0.40), ("EU", 0.40), ("FI", 0.40), ("DK", 0.40), ("CH", 0.40), ("SE", 0.40), ("UK", 0.40),
    ("IN", 0.30), ("ZZ", 0.30), ("", 0.30),
])
def test_market_cost_buckets_and_fallback(pg, market, cost):
    assert pg.market_cost(market) == cost


def test_market_cost_is_case_sensitive(pg):
    assert pg.market_cost("us") == 0.30


# ── load_closes ───────────────────────────────────────────────────────────────
def _long(frames):
    parts = []
    for sym, s in frames.items():
        parts.append(pd.DataFrame({"Date": s.index, "Symbol": sym, "Close": s.values}))
    return pd.concat(parts, ignore_index=True)


@pytest.fixture
def seed(pg, monkeypatch, tmp_path):
    d = tmp_path / "seed"
    d.mkdir()
    monkeypatch.setattr(pg, "SEED", str(d))
    store = {}
    monkeypatch.setattr(pg.pd, "read_parquet", lambda path, *a, **k: store[os.path.basename(path)].copy())

    def put(market, frames):
        name = f"cleaned_long_{market}.parquet"
        (d / name).write_bytes(b"")
        store[name] = _long(frames) if isinstance(frames, dict) else frames
    return put


def _px(n, start="2021-01-04", daily=0.002, base=100.0):
    return pd.Series(base * (1 + daily) ** np.arange(n), index=pd.bdate_range(start, periods=n))


def test_load_closes_missing_file_returns_empty(pg, seed):
    assert pg.load_closes("NOPE", 5) == {}


def test_load_closes_length_threshold_is_strictly_above_80(pg, seed):
    seed("XX", {"A80": _px(80), "A81": _px(81), "A200": _px(200)})
    out = pg.load_closes("XX", 50)
    assert set(out) == {"A81", "A200"}
    assert all(isinstance(s, pd.Series) and s.dtype == float for s in out.values())


def test_load_closes_sorted_and_drops_nan(pg, seed):
    s = _px(120)
    s.iloc[10] = np.nan
    df = _long({"A": s}).sample(frac=1.0, random_state=1)
    seed("XX", df)
    out = pg.load_closes("XX", 50)["A"]
    assert out.index.is_monotonic_increasing and len(out) == 119 and out.notna().all()


def test_load_closes_cutoff_window(pg, seed):
    # 3 years of daily data; years=1 keeps dates >= max - (365 + 30) days
    s = _px(800, start="2020-01-01")
    seed("XX", {"A": s})
    out = pg.load_closes("XX", 1)["A"]
    cutoff = s.index.max() - pd.Timedelta(days=395)
    assert out.index.min() >= cutoff
    assert out.index.min() == s.index[s.index >= cutoff][0]
    assert out.index.max() == s.index.max()


def test_load_closes_cutoff_is_inclusive(pg, seed):
    s = pd.Series(np.arange(1, 1001, dtype=float),
                  index=pd.date_range("2019-01-01", periods=1000, freq="D"))
    seed("XX", {"A": s})
    out = pg.load_closes("XX", 1)["A"]          # cutoff = max - 395 days, row on the cutoff is kept
    assert out.index.min() == s.index.max() - pd.Timedelta(days=395)
    assert len(out) == 396


def test_load_closes_parses_string_dates(pg, seed):
    s = _px(120)
    df = _long({"A": s})
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    seed("XX", df)
    out = pg.load_closes("XX", 5)["A"]
    assert isinstance(out.index, pd.DatetimeIndex)


# ── durable_names ─────────────────────────────────────────────────────────────
@pytest.fixture
def fund_db(pg, monkeypatch, tmp_path):
    monkeypatch.setattr(pg, "HERE", str(tmp_path))

    def make(rows):
        con = sqlite3.connect(tmp_path / "fundamentals_cache.db")
        con.execute("CREATE TABLE fund(market TEXT, ticker TEXT, roe, de)")
        con.executemany("INSERT INTO fund VALUES (?,?,?,?)", rows)
        con.commit(); con.close()
    return make


def test_durable_names_no_cache_file(pg, fund_db):
    assert pg.durable_names("JP") == set()


def test_durable_names_thresholds_are_strict(pg, fund_db):
    fund_db([
        ("JP", "OK.T", 12.01, 1.49),
        ("JP", "ROE_EQ.T", 12.0, 0.5),      # roe must be > 12
        ("JP", "DE_EQ.T", 20.0, 1.5),       # de must be < 1.5
        ("JP", "LOW.T", 5.0, 0.2),
        ("KR", "OTHER.KS", 30.0, 0.1),      # other market
    ])
    assert pg.durable_names("JP") == {"OK"}


def test_durable_names_normalises_ticker_and_coerces_text(pg, fund_db):
    fund_db([
        ("JP", "7203.t", "15.5", "0.4"),
        ("JP", "abc", 20, 1.0),
        ("JP", "BAD.T", "n/a", 0.5),       # non-numeric roe -> NaN -> excluded
        ("JP", "NODE.T", 20.0, None),      # missing de -> NaN -> excluded
    ])
    assert pg.durable_names("JP") == {"7203", "ABC"}


def test_durable_names_unknown_market_empty(pg, fund_db):
    fund_db([("JP", "A.T", 20.0, 0.5)])
    assert pg.durable_names("ZZ") == set()


# ── run_market ────────────────────────────────────────────────────────────────
def _universe(n_up=12, n_dn=12, n=330):
    out = {}
    for i in range(n_up):
        out[f"U{i:02d}.T"] = _px(n, daily=0.001 + 0.0002 * i, base=50 + i)
    for i in range(n_dn):
        out[f"D{i:02d}.T"] = _px(n, daily=-0.0005 - 0.0001 * i, base=80 + i)
    return out


def _snap(idx, ts):
    return idx[idx <= ts][-1]


def _pairs(closes):
    all_idx = sorted(set().union(*[s.index for s in closes.values()]))
    rebal = pd.Series(1, index=pd.DatetimeIndex(all_idx)).resample("MS").first().index
    rebal = [d for d in rebal if d >= pd.Timestamp(all_idx[61])]
    return list(zip(rebal[:-1], rebal[1:]))


def _fwd(s, t0, t1):
    return (s.loc[_snap(s.index, t1)] / s.loc[_snap(s.index, t0)] - 1) * 100


def test_run_market_skips_small_universe(pg, monkeypatch, capsys):
    monkeypatch.setattr(pg, "load_closes", lambda m, y: {f"S{i}": _px(120) for i in range(19)})
    out = pg.run_market("JP", 5, False)
    assert out.empty
    assert "[skip] JP: only 19 usable series" in capsys.readouterr().err


def test_run_market_twenty_series_is_enough(pg, monkeypatch):
    monkeypatch.setattr(pg, "load_closes", lambda m, y: {f"S{i}": _px(330, daily=0.001) for i in range(20)})
    assert not pg.run_market("JP", 5, False).empty


def test_run_market_matches_independent_reference(pg, monkeypatch):
    closes = _universe()
    monkeypatch.setattr(pg, "load_closes", lambda m, y: closes)
    res = pg.run_market("JP", 5, False)
    cost = 0.20
    pairs = _pairs(closes)
    bo = pg.darvas_breakout
    a, b = [], []
    for t0, t1 in pairs:
        held = [_fwd(s, t0, t1) for s in closes.values()]
        b.append(float(np.mean(held)) - cost)
        sel = [t for t, s in closes.items() if bo(s, _snap(s.index, t0))]
        if sel:
            a.append(float(np.mean([_fwd(closes[t], t0, t1) for t in sel])) - cost)
    assert list(res["arm"]) == ["A_darvas", "BENCH_all"]
    assert (res["market"] == "JP").all()
    assert list(res.columns) == ["market", "arm", "n_months", "avg_mth%", "ann_return%",
                                 "hit_rate%", "sharpe", "max_dd%"]
    for arm, rets in (("A_darvas", a), ("BENCH_all", b)):
        ref = pg.metrics(rets)
        row = res[res.arm == arm].iloc[0]
        assert row["n_months"] == ref["n_months"]
        for k in ("avg_mth%", "ann_return%", "hit_rate%", "sharpe", "max_dd%"):
            assert row[k] == pytest.approx(ref[k])
    assert res[res.arm == "BENCH_all"].iloc[0]["n_months"] == len(pairs)


def test_run_market_cost_follows_market_bucket(pg, monkeypatch):
    closes = _universe()
    monkeypatch.setattr(pg, "load_closes", lambda m, y: closes)
    us = pg.run_market("US", 5, False).set_index("arm")
    de = pg.run_market("DE", 5, False).set_index("arm")
    assert us.loc["BENCH_all", "avg_mth%"] - de.loc["BENCH_all", "avg_mth%"] == pytest.approx(0.30, abs=2e-3)
    assert us.loc["A_darvas", "avg_mth%"] - de.loc["A_darvas", "avg_mth%"] == pytest.approx(0.30, abs=2e-3)


def test_run_market_overlay_adds_durable_arm(pg, monkeypatch):
    closes = _universe()
    monkeypatch.setattr(pg, "load_closes", lambda m, y: closes)
    monkeypatch.setattr(pg, "durable_names", lambda m: {"U03", "U07", "D00"})
    res = pg.run_market("JP", 5, True)
    assert list(res["arm"]) == ["A_darvas", "BENCH_all", "D_darvas_durable"]
    assert res.set_index("arm").loc["D_darvas_durable", "n_months"] > 0
    cost = 0.20
    pairs = _pairs(closes)
    bo = pg.darvas_breakout
    d = []
    for t0, t1 in pairs:
        sel = [t for t, s in closes.items()
               if bo(s, _snap(s.index, t0)) and t.split(".")[0].upper() in {"U03", "U07", "D00"}]
        if sel:
            d.append(float(np.mean([_fwd(closes[t], t0, t1) for t in sel])) - cost)
    ref = pg.metrics(d)
    row = res.set_index("arm").loc["D_darvas_durable"]
    assert row["n_months"] == ref["n_months"] and row["ann_return%"] == pytest.approx(ref["ann_return%"])


def test_run_market_overlay_not_called_without_flag(pg, monkeypatch):
    monkeypatch.setattr(pg, "load_closes", lambda m, y: _universe())

    def boom(m):
        raise AssertionError("durable_names must only be consulted with --overlay")
    monkeypatch.setattr(pg, "durable_names", boom)
    assert "D_darvas_durable" not in set(pg.run_market("JP", 5, False)["arm"])


def test_run_market_overlay_without_cache_has_empty_durable_arm(pg, monkeypatch):
    monkeypatch.setattr(pg, "load_closes", lambda m, y: _universe())
    monkeypatch.setattr(pg, "durable_names", lambda m: set())
    res = pg.run_market("JP", 5, True).set_index("arm")
    assert res.loc["D_darvas_durable", "n_months"] == 0


def test_run_market_ignores_tickers_not_trading_at_the_window(pg, monkeypatch):
    closes = _universe()
    closes["LATE.T"] = _px(120, start="2021-09-01", daily=0.01, base=10.0)      # lists mid-sample
    closes["GONE.T"] = _px(150, start="2021-01-04", daily=0.01, base=10.0)      # delisted mid-sample
    monkeypatch.setattr(pg, "load_closes", lambda m, y: closes)
    res = pg.run_market("JP", 5, False).set_index("arm")
    exp = []
    for t0, t1 in _pairs(closes):
        held = []
        for s in closes.values():
            if not (s.index <= t0).any() or not (s.index <= t1).any():
                continue
            if _snap(s.index, t0) >= _snap(s.index, t1):
                continue
            held.append(_fwd(s, t0, t1))
        exp.append(float(np.mean(held)) - 0.20)
    ref = pg.metrics(exp)
    assert res.loc["BENCH_all", "n_months"] == ref["n_months"]
    assert res.loc["BENCH_all", "avg_mth%"] == pytest.approx(ref["avg_mth%"])
    assert res.loc["BENCH_all", "ann_return%"] == pytest.approx(ref["ann_return%"])


def test_run_market_zero_price_forward_return_is_skipped(pg, monkeypatch):
    closes = _universe()
    z = _px(330, daily=0.0, base=100.0)
    z.loc["2021-12-01":"2021-12-31"] = 0.0
    closes["Z00.T"] = z
    monkeypatch.setattr(pg, "load_closes", lambda m, y: closes)
    row = pg.run_market("JP", 5, False).set_index("arm").loc["BENCH_all"]
    assert np.isfinite(row["avg_mth%"]) and np.isfinite(row["ann_return%"])


@pytest.mark.xfail(strict=True, raises=KeyError,
                   reason="pit_global.py:133 column list is taken from rows[0]; when A_darvas has no months "
                          "(no breakouts) the BENCH metrics columns are silently dropped")
def test_run_market_no_breakouts_still_reports_benchmark_metrics(pg, monkeypatch):
    closes = {f"D{i:02d}.T": _px(330, daily=-0.001 - 0.0001 * i, base=80 + i) for i in range(22)}
    monkeypatch.setattr(pg, "load_closes", lambda m, y: closes)
    res = pg.run_market("JP", 5, False).set_index("arm")
    assert res.loc["A_darvas", "n_months"] == 0
    assert res.loc["BENCH_all", "ann_return%"] == res.loc["BENCH_all", "ann_return%"]   # column must exist


# ── main ──────────────────────────────────────────────────────────────────────
@pytest.fixture
def run_main(pg, monkeypatch, tmp_path, capsys):
    def _run(*argv, closes=None, markets=None):
        monkeypatch.setattr(pg, "load_closes", lambda m, y: (closes or {}).get(m, {}))
        if markets is not None:
            monkeypatch.setattr(pg.marketdata, "market_list", lambda: markets)
        monkeypatch.setattr(pg, "durable_names", lambda m: {"U03", "U05"})
        db = tmp_path / "g.db"
        monkeypatch.setattr(sys, "argv", ["pit_global.py", "--db", str(db), *argv])
        pg.main()
        cap = capsys.readouterr()
        return db, cap
    return _run


def _read(db):
    c = sqlite3.connect(db)
    try:
        return pd.read_sql("SELECT * FROM global_arm_summary", c)
    finally:
        c.close()


def test_main_default_market_is_jp_and_saves(run_main):
    db, cap = run_main(closes={"JP": _universe()})
    df = _read(db)
    assert set(df.market) == {"JP"} and list(df.arm) == ["A_darvas", "BENCH_all"]
    assert "=== JP — Darvas breakout, monthly, net of 0.20%" in cap.out
    assert "saved 2 rows" in cap.err
    assert "NOTE:" not in cap.out


def test_main_market_flag_and_overlay_note(run_main):
    db, cap = run_main("--market", "US", "--overlay", closes={"US": _universe()})
    df = _read(db)
    assert set(df.market) == {"US"} and "D_darvas_durable" in set(df.arm)
    assert "net of 0.10%" in cap.out
    assert "NOTE: D_darvas_durable uses CURRENT fundamentals" in cap.out


def test_main_all_iterates_market_list_and_skips_empty(run_main):
    db, cap = run_main("--all", closes={"JP": _universe(), "KR": _universe(), "XX": {}},
                       markets=["JP", "KR", "XX"])
    df = _read(db)
    assert set(df.market) == {"JP", "KR"} and len(df) == 4
    assert "[skip] XX" in cap.err
    assert cap.out.count("=== ") == 2
    assert "net of 0.25%" in cap.out


def test_main_nothing_to_save_creates_no_db(run_main, tmp_path):
    db, cap = run_main("--market", "XX", closes={})
    assert not db.exists()
    assert "saved" not in cap.err


def test_main_years_argument_forwarded(pg, monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(pg, "load_closes", lambda m, y: seen.append((m, y)) or {})
    monkeypatch.setattr(sys, "argv", ["pit_global.py", "--market", "KR", "--years", "3",
                                      "--db", str(tmp_path / "x.db")])
    pg.main()
    assert seen == [("KR", 3)]


def test_main_replaces_previous_table(run_main):
    run_main(closes={"JP": _universe()})
    db, _ = run_main(closes={"JP": _universe()})
    assert len(_read(db)) == 2
