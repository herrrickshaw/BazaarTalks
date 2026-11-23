"""Offline tests for market_data_cache.MarketCache.

Everything external is faked: HOME is redirected to tmp_path before a *fresh* import (the
module creates ~/Downloads/market_cache at import time), parquet I/O is replaced with
pickle (no pyarrow), yfinance is a scripted fake, and the module's own ``date``/``datetime``
/``time`` names are frozen so no test depends on the wall clock.
"""
import importlib
import json
import os
import sys
import types
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_NOW = {"v": datetime(2026, 6, 15, 12, 0, 0)}


class _FD(date):
    @classmethod
    def today(cls):
        return _NOW["v"].date()


class _FDT(datetime):
    @classmethod
    def now(cls, tz=None):
        return _NOW["v"]


def at(*args):
    _NOW["v"] = datetime(*args)


def make_ohlc(n, end="2026-06-12", price=100.0, seed=0):
    idx = pd.bdate_range(end=end, periods=n)
    rng = np.random.default_rng(seed)
    close = price * np.cumprod(1 + rng.normal(0, 0.01, n))
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99,
                         "Close": close, "Volume": 1000.0 + np.arange(n)}, index=idx)


def yf_frame(frames):
    """Mimic yfinance's (field, ticker) MultiIndex columns."""
    return pd.concat(frames, axis=1).swaplevel(axis=1).sort_index(axis=1)


class FakeYF:
    def __init__(self):
        self.store = {}
        self.calls = []
        self.handler = self._serve
        self.tickers = {}

    def _serve(self, tickers, start):
        names = [tickers] if isinstance(tickers, str) else list(tickers)
        frames = {}
        for t in names:
            if t in self.store:
                df = self.store[t]
                frames[t] = df[df.index >= pd.Timestamp(start)]
        if not frames:
            return pd.DataFrame()
        return yf_frame(frames)

    def download(self, tickers, start=None, **kw):
        self.calls.append((tickers, start, kw))
        return self.handler(tickers, start)

    def Ticker(self, sym):
        return self.tickers[sym]() if callable(self.tickers.get(sym)) else self.tickers[sym]


@pytest.fixture
def env(tmp_path, monkeypatch):
    at(2026, 6, 15, 12, 0, 0)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delitem(sys.modules, "market_data_cache", raising=False)
    mod = importlib.import_module("market_data_cache")
    # parquet -> pickle
    monkeypatch.setattr(pd.DataFrame, "to_parquet",
                        lambda self, path, *a, **k: self.to_pickle(path), raising=False)
    monkeypatch.setattr(pd, "read_parquet", lambda path, *a, **k: pd.read_pickle(path))
    sleeps = []
    monkeypatch.setattr(mod, "time", types.SimpleNamespace(sleep=sleeps.append))
    monkeypatch.setattr(mod, "date", _FD)
    monkeypatch.setattr(mod, "datetime", _FDT)
    fake = FakeYF()
    monkeypatch.setattr(mod, "yf", fake, raising=False)
    monkeypatch.setattr(mod, "_YF_OK", True)
    monkeypatch.setattr(mod, "_CACHE_INSTANCE", None)
    ns = types.SimpleNamespace(mod=mod, yf=fake, sleeps=sleeps, home=tmp_path)
    yield ns
    sys.modules.pop("market_data_cache", None)


@pytest.fixture
def cache(env):
    return env.mod.MarketCache(verbose=True)


# ── import side effects / config ─────────────────────────────────────────────
def test_import_creates_cache_dirs_under_home(env):
    root = env.home / "Downloads" / "market_cache"
    assert env.mod.CACHE_ROOT == root
    for sub in ("ohlc", "fundamentals", "index", "meta"):
        assert (root / sub).is_dir()
    assert env.mod.META_FILE == root / "meta" / "cache_index.json"


# ── meta helpers ─────────────────────────────────────────────────────────────
def test_load_meta_missing_valid_and_corrupt(env):
    assert env.mod.MarketCache()._meta == {}
    env.mod.META_FILE.write_text(json.dumps({"ohlc:A": {"rows": 3}}))
    assert env.mod.MarketCache()._meta == {"ohlc:A": {"rows": 3}}
    env.mod.META_FILE.write_text("{not json")
    assert env.mod.MarketCache()._meta == {}


def test_update_meta_records_and_persists(cache, env):
    df = make_ohlc(5, end="2026-06-12")
    p = env.mod.OHLC_DIR / "A.parquet"
    cache._update_meta("ohlc:A", p, df)
    m = cache._meta["ohlc:A"]
    assert m == {"rows": 5, "from": "2026-06-08", "to": "2026-06-12",
                 "updated": "2026-06-15T12:00:00", "file": str(p)}
    assert json.loads(env.mod.META_FILE.read_text())["ohlc:A"]["rows"] == 5


def test_update_meta_empty_frame_has_null_dates(cache, env):
    cache._update_meta("ohlc:E", env.mod.OHLC_DIR / "E.parquet",
                       pd.DataFrame(columns=["Close"], index=pd.DatetimeIndex([])))
    assert cache._meta["ohlc:E"]["from"] is None and cache._meta["ohlc:E"]["to"] is None
    assert cache._meta["ohlc:E"]["rows"] == 0


def test_is_stale_boundaries(cache):
    assert cache._is_stale("missing", 20) is True
    cache._meta["k"] = {"updated": "2026-06-14T16:00:00"}       # exactly 20h old
    assert cache._is_stale("k", 20) is False                    # strictly greater only
    cache._meta["k"] = {"updated": "2026-06-14T15:59:59"}       # 20h + 1s
    assert cache._is_stale("k", 20) is True
    cache._meta["k"] = {"updated": "garbage"}
    assert cache._is_stale("k", 20) is True
    cache._meta["k"] = {}
    assert cache._is_stale("k", 20) is True
    cache._meta["k"] = None
    assert cache._is_stale("k", 20) is True


def test_last_cached_date(cache):
    assert cache._last_cached_date("none") is None
    cache._meta["a"] = {"to": "2026-06-12"}
    assert cache._last_cached_date("a") == date(2026, 6, 12)
    cache._meta["a"] = {"to": None}
    assert cache._last_cached_date("a") is None
    cache._meta["a"] = {"to": "not-a-date"}
    assert cache._last_cached_date("a") is None
    cache._meta["a"] = {"rows": 1}
    assert cache._last_cached_date("a") is None


def test_ohlc_path_sanitises_slash_and_colon(cache, env):
    assert cache._ohlc_path("BRK/B:X") == env.mod.OHLC_DIR / "BRK_B_X.parquet"
    assert cache._ohlc_path("RELIANCE.NS").name == "RELIANCE.NS.parquet"


# ── get_ohlc ─────────────────────────────────────────────────────────────────
def test_get_ohlc_cold_start_downloads_five_years_and_writes_cache(cache, env, capsys):
    env.yf.store["AAA.NS"] = make_ohlc(300)
    df = cache.get_ohlc("AAA.NS")
    assert len(df) == 300
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    ticker, start, kw = env.yf.calls[0]
    assert ticker == "AAA.NS"
    assert start == date(2026, 6, 15) - timedelta(days=5 * 365 + 30)
    assert kw["auto_adjust"] is True and kw["progress"] is False
    assert (env.mod.OHLC_DIR / "AAA.NS.parquet").exists()
    m = cache._meta["ohlc:AAA.NS"]
    assert m["rows"] == 300 and m["to"] == "2026-06-12"
    out = capsys.readouterr().out
    assert f"Cache created: AAA.NS 300 rows ({df.index.min().date()} – {df.index.max().date()})" in out


def test_get_ohlc_cold_start_empty_download_writes_nothing(cache, env):
    df = cache.get_ohlc("NOPE.NS")
    assert df.empty
    assert not (env.mod.OHLC_DIR / "NOPE.NS.parquet").exists()
    assert "ohlc:NOPE.NS" not in cache._meta


def test_get_ohlc_quiet_mode_prints_nothing(env, capsys):
    env.yf.store["Q"] = make_ohlc(10)
    env.mod.MarketCache(verbose=False).get_ohlc("Q")
    assert capsys.readouterr().out == ""


def test_get_ohlc_fresh_reads_disk_once_then_memory(cache, env, monkeypatch):
    env.yf.store["A"] = make_ohlc(50)
    cache.get_ohlc("A")                          # cold start
    reads = []
    real = pd.read_parquet
    monkeypatch.setattr(pd, "read_parquet", lambda p, *a, **k: (reads.append(p), real(p))[1])
    first = cache.get_ohlc("A")
    second = cache.get_ohlc("A")
    assert len(reads) == 1                        # second call served from L1
    assert second is first
    assert len(env.yf.calls) == 1                 # never hit the network again


def test_get_ohlc_force_redownloads_full_history(cache, env):
    env.yf.store["A"] = make_ohlc(50)
    cache.get_ohlc("A")
    cache.get_ohlc("A", force=True)
    assert len(env.yf.calls) == 2
    assert env.yf.calls[1][1] == date(2026, 6, 15) - timedelta(days=5 * 365 + 30)


def test_get_ohlc_stale_fetches_only_after_last_cached_date(cache, env, capsys):
    full = make_ohlc(40, end="2026-06-16")
    env.yf.store["A"] = full[full.index <= "2026-06-12"]
    first = cache.get_ohlc("A")
    n0 = len(first)
    env.yf.store["A"] = full
    capsys.readouterr()
    at(2026, 6, 17, 12, 0, 0)                      # 48h later -> stale
    df = cache.get_ohlc("A")
    assert env.yf.calls[-1][1] == date(2026, 6, 13)            # last_date + 1 day
    assert len(df) == n0 + 2                                    # 06-15 and 06-16
    assert df.index.is_monotonic_increasing and not df.index.duplicated().any()
    assert cache._meta["ohlc:A"]["to"] == "2026-06-16"
    assert f"Cache updated: A +2 rows (total {n0 + 2})" in capsys.readouterr().out


def test_get_ohlc_stale_overlap_keeps_latest_values(cache, env):
    base = make_ohlc(20, end="2026-06-12")
    env.yf.store["A"] = base
    cache.get_ohlc("A")
    revised = base.iloc[-1:] * 2                    # same date, new numbers
    new_day = make_ohlc(1, end="2026-06-16", price=50.0)
    env.yf.store["A"] = pd.concat([revised, new_day])
    cache._last_cached_date = lambda key: date(2026, 6, 11)   # force overlap with 06-12
    at(2026, 6, 17, 12, 0, 0)
    df = cache.get_ohlc("A")
    assert df.loc["2026-06-12", "Close"] == pytest.approx(base["Close"].iloc[-1] * 2)
    assert len(df) == 21


def test_get_ohlc_stale_but_already_up_to_date_makes_no_request(cache, env):
    env.yf.store["A"] = make_ohlc(20, end="2026-06-12")
    at(2026, 6, 12, 20, 0, 0)
    cache.get_ohlc("A")
    n_calls = len(env.yf.calls)
    at(2026, 6, 13, 20, 0, 0)                      # 24h later: stale, today == last+1
    df = cache.get_ohlc("A")
    assert len(env.yf.calls) == n_calls            # start >= today -> skip fetch
    assert len(df) == 20


def test_get_ohlc_stale_day_before_boundary_does_fetch(cache, env):
    env.yf.store["A"] = make_ohlc(20, end="2026-06-12")
    at(2026, 6, 12, 20, 0, 0)
    cache.get_ohlc("A")
    n_calls = len(env.yf.calls)
    at(2026, 6, 14, 20, 0, 0)                      # start (06-13) < today (06-14)
    before = dict(cache._meta["ohlc:A"])
    df = cache.get_ohlc("A")
    assert len(env.yf.calls) > n_calls
    assert len(df) == 20                           # nothing new -> existing returned
    assert cache._meta["ohlc:A"] == before         # and cache not rewritten


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="market_data_cache.py:get_ohlc existing parquet + stale/missing meta "
                          "(no 'to' date) falls off the end and returns None instead of a frame")
def test_get_ohlc_existing_file_with_lost_meta_returns_frame(cache, env):
    env.yf.store["A"] = make_ohlc(20)
    cache.get_ohlc("A")
    cache._meta = {}                                # cache_index.json lost / corrupt
    out = cache.get_ohlc("A")
    assert isinstance(out, pd.DataFrame)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="market_data_cache.py:get_ohlc the L1 _mem copy is never invalidated "
                          "when force-refresh/incremental rewrites the parquet, so later calls "
                          "return the superseded frame")
def test_get_ohlc_memory_copy_refreshed_after_force(cache, env):
    env.yf.store["A"] = make_ohlc(30)
    cache.get_ohlc("A")
    cache.get_ohlc("A")                             # populates L1
    env.yf.store["A"] = make_ohlc(40)
    refreshed = cache.get_ohlc("A", force=True)
    assert len(refreshed) == 40
    assert len(cache.get_ohlc("A")) == 40


# ── _yf_download_single ──────────────────────────────────────────────────────
def test_yf_single_returns_only_ohlcv_and_drops_all_nan_rows(cache, env):
    df = make_ohlc(5)
    df["Adj"] = 1.0
    df.iloc[2, :5] = np.nan
    env.yf.handler = lambda t, s: df.copy()
    out = cache._yf_download_single("A", date(2026, 1, 1))
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(out) == 4


def test_yf_single_flattens_multiindex(cache, env):
    env.yf.store["A"] = make_ohlc(5)
    out = cache._yf_download_single("A", date(2020, 1, 1))
    assert not isinstance(out.columns, pd.MultiIndex) and len(out) == 5


def test_yf_single_without_yfinance_returns_empty(cache, env, monkeypatch):
    monkeypatch.setattr(env.mod, "_YF_OK", False)
    assert cache._yf_download_single("A", date(2020, 1, 1)).empty
    assert env.yf.calls == []


def test_yf_single_empty_result_retried_three_times(cache, env):
    env.yf.handler = lambda t, s: pd.DataFrame()
    assert cache._yf_download_single("A", date(2020, 1, 1)).empty
    assert len(env.yf.calls) == 3
    assert env.sleeps == []


def test_yf_single_rate_limit_backs_off_then_succeeds(cache, env):
    good = make_ohlc(3)
    state = {"n": 0}

    def handler(t, s):
        state["n"] += 1
        if state["n"] <= 2:
            raise RuntimeError("HTTP 429 slow down" if state["n"] == 1 else "Rate limited")
        return good

    env.yf.handler = handler
    out = cache._yf_download_single("A", date(2020, 1, 1))
    assert len(out) == 3
    assert env.sleeps == [15, 30]


def test_yf_single_rate_limit_exhausted_returns_empty(cache, env):
    def handler(t, s):
        raise RuntimeError("429")
    env.yf.handler = handler
    assert cache._yf_download_single("A", date(2020, 1, 1)).empty
    assert env.sleeps == [15, 30, 45]


def test_yf_single_other_error_breaks_immediately(cache, env):
    def handler(t, s):
        raise ValueError("boom")
    env.yf.handler = handler
    assert cache._yf_download_single("A", date(2020, 1, 1)).empty
    assert len(env.yf.calls) == 1 and env.sleeps == []


# ── _bulk_yf_download ────────────────────────────────────────────────────────
def test_bulk_download_batches_and_sleeps_between_batches_only(cache, env, monkeypatch, capsys):
    monkeypatch.setattr(env.mod, "BATCH_SIZE", 2)
    for t in "ABC":
        env.yf.store[t] = make_ohlc(5)
    out = cache._bulk_yf_download(["A", "B", "C"], date(2020, 1, 1))
    assert sorted(out) == ["A", "B", "C"]
    assert [c[0] for c in env.yf.calls] == [["A", "B"], ["C"]]
    assert env.yf.calls[0][2]["threads"] is True
    assert env.sleeps == [env.mod.BATCH_SLEEP]
    assert list(out["A"].columns) == ["Open", "High", "Low", "Close", "Volume"]
    txt = capsys.readouterr().out
    assert "Downloading batch 1/2 (2 tickers)" in txt and "OK (2/2)" in txt
    assert "Bulk download complete: 3/3 tickers" in txt


def test_bulk_download_skips_missing_and_all_nan_tickers(cache, env, capsys):
    nan_df = make_ohlc(5) * np.nan
    env.yf.store["A"] = make_ohlc(5)
    env.yf.store["N"] = nan_df
    out = cache._bulk_yf_download(["A", "N", "GONE"], date(2020, 1, 1))
    assert list(out) == ["A"]
    assert "OK (1/3)" in capsys.readouterr().out


def test_bulk_download_flat_frame_goes_to_first_ticker(cache, env):
    env.yf.handler = lambda t, s: make_ohlc(4)
    out = cache._bulk_yf_download(["X", "Y"], date(2020, 1, 1))
    assert list(out) == ["X"] and len(out["X"]) == 4


def test_bulk_download_empty_response_not_retried(cache, env, capsys):
    env.yf.handler = lambda t, s: pd.DataFrame()
    assert cache._bulk_yf_download(["A"], date(2020, 1, 1)) == {}
    assert len(env.yf.calls) == 1
    assert "empty" in capsys.readouterr().out


def test_bulk_download_rate_limit_backoff_schedule(cache, env):
    state = {"n": 0}
    env.yf.store["A"] = make_ohlc(3)

    def handler(t, s):
        state["n"] += 1
        if state["n"] < 3:
            raise RuntimeError("Too Many Requests")
        return env.yf._serve(t, s)

    env.yf.handler = handler
    out = cache._bulk_yf_download(["A"], date(2020, 1, 1))
    assert list(out) == ["A"]
    assert env.sleeps == [30, 60]


def test_bulk_download_rate_limit_gives_up_after_three(cache, env):
    def handler(t, s):
        raise RuntimeError("429")
    env.yf.handler = handler
    assert cache._bulk_yf_download(["A"], date(2020, 1, 1)) == {}
    assert env.sleeps == [30, 60, 90]


def test_bulk_download_other_error_reported_once(cache, env, capsys):
    def handler(t, s):
        raise ValueError("kaput")
    env.yf.handler = handler
    assert cache._bulk_yf_download(["A"], date(2020, 1, 1)) == {}
    assert len(env.yf.calls) == 1
    assert "ERROR: kaput" in capsys.readouterr().out


def test_bulk_download_without_yfinance(cache, env, monkeypatch):
    monkeypatch.setattr(env.mod, "_YF_OK", False)
    assert cache._bulk_yf_download(["A"], date(2020, 1, 1)) == {}


# ── get_ohlc_bulk ────────────────────────────────────────────────────────────
def test_bulk_classifies_fresh_stale_and_missing(cache, env, capsys):
    for t in ("FRESH", "STALE", "NEW"):
        env.yf.store[t] = make_ohlc(30)
    cache.get_ohlc_bulk(["FRESH", "STALE"])
    cache._meta["ohlc:STALE"]["updated"] = "2026-06-10T00:00:00"
    env.yf.calls.clear()
    capsys.readouterr()
    res = cache.get_ohlc_bulk(["FRESH", "STALE", "NEW"])
    out = capsys.readouterr().out
    assert "Cache: 1 fresh | 1 incremental | 1 cold-start" in out
    assert "Cache result: 3/3 tickers loaded" in out
    assert sorted(res) == ["FRESH", "NEW", "STALE"]
    starts = {tuple(c[0]): c[1] for c in env.yf.calls}
    assert starts[("NEW",)] == date(2026, 6, 15) - timedelta(days=5 * 365 + 30)
    assert starts[("STALE",)] == date(2026, 6, 10)             # today - 5 days
    assert len(env.yf.calls) == 2


def test_bulk_all_fresh_makes_no_requests_and_quiet_mode_is_silent(env, capsys):
    env.yf.store["A"] = make_ohlc(20)
    c = env.mod.MarketCache(verbose=False)
    c.get_ohlc_bulk(["A"])
    capsys.readouterr()
    env.yf.calls.clear()
    res = c.get_ohlc_bulk(["A"])
    assert len(res["A"]) == 20
    assert env.yf.calls == []
    assert capsys.readouterr().out == ""


def test_bulk_force_refreshes_even_fresh_tickers(cache, env):
    env.yf.store["A"] = make_ohlc(20)
    cache.get_ohlc_bulk(["A"])
    env.yf.calls.clear()
    cache.get_ohlc_bulk(["A"], force=True)
    assert len(env.yf.calls) == 1
    assert env.yf.calls[0][1] == date(2026, 6, 10)


def test_bulk_fresh_reads_promote_to_memory(cache, env, monkeypatch):
    env.yf.store["A"] = make_ohlc(20)
    cache.get_ohlc_bulk(["A"])
    assert "A" not in cache._mem                    # cold-start path does not fill L1
    cache.get_ohlc_bulk(["A"])
    assert "A" in cache._mem
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("disk hit"))
    assert len(cache.get_ohlc_bulk(["A"])["A"]) == 20


def test_bulk_unreadable_fresh_file_falls_back_to_download(cache, env):
    env.yf.store["A"] = make_ohlc(20)
    cache.get_ohlc_bulk(["A"])
    cache._ohlc_path("A").write_bytes(b"corrupt")
    env.yf.calls.clear()
    res = cache.get_ohlc_bulk(["A"])
    assert len(res["A"]) == 20
    assert len(env.yf.calls) == 1


def test_bulk_incremental_appends_new_rows_and_new_values_win(cache, env):
    base = make_ohlc(30, end="2026-06-12")
    env.yf.store["A"] = base
    cache.get_ohlc_bulk(["A"])
    cache._meta["ohlc:A"]["updated"] = "2026-06-01T00:00:00"
    overlap = base[base.index >= "2026-06-11"] * 2
    fresh_day = make_ohlc(1, end="2026-06-15", price=77.0)
    env.yf.store["A"] = pd.concat([overlap, fresh_day])
    res = cache.get_ohlc_bulk(["A"])["A"]
    assert len(res) == 31
    assert res.loc["2026-06-12", "Close"] == pytest.approx(base["Close"].iloc[-1] * 2)
    assert res.index[-1] == pd.Timestamp("2026-06-15")
    assert cache._meta["ohlc:A"]["rows"] == 31 and cache._meta["ohlc:A"]["to"] == "2026-06-15"


def test_bulk_incremental_without_new_rows_returns_existing(cache, env):
    env.yf.store["A"] = make_ohlc(20)
    cache.get_ohlc_bulk(["A"])
    cache._meta["ohlc:A"]["updated"] = "2026-06-01T00:00:00"
    env.yf.store.clear()                            # download returns nothing
    res = cache.get_ohlc_bulk(["A"])
    assert len(res["A"]) == 20


def test_bulk_incremental_with_corrupt_file_uses_downloaded_frame(cache, env):
    env.yf.store["A"] = make_ohlc(20)
    cache.get_ohlc_bulk(["A"])
    cache._meta["ohlc:A"]["updated"] = "2026-06-01T00:00:00"
    cache._ohlc_path("A").write_bytes(b"corrupt")
    fresh = make_ohlc(3, end="2026-06-12")
    env.yf.store["A"] = fresh
    res = cache.get_ohlc_bulk(["A"])
    assert len(res["A"]) == 3
    # corrupt + nothing downloaded -> ticker simply absent
    env.yf.store.clear()
    assert "A" not in cache.get_ohlc_bulk(["A"], force=True)


def test_bulk_empty_ticker_list(cache, env, capsys):
    assert cache.get_ohlc_bulk([]) == {}
    assert "Cache result: 0/0 tickers loaded" in capsys.readouterr().out
    assert env.yf.calls == []


# ── get_financials ───────────────────────────────────────────────────────────
def _stmt(vals, dates=("2024-03-31", "2023-03-31")):
    return pd.DataFrame({pd.Timestamp(d): v for d, v in zip(dates, vals)},
                        index=["Revenue", "NetIncome"])


class FakeTicker:
    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


def test_first_df_picks_first_nonempty_dataframe(env):
    f = env.mod._first_df
    good = pd.DataFrame({"a": [1]})
    t = FakeTicker(a=None, b=pd.DataFrame(), c="notdf", d=good)
    assert f(t, "a", "b", "c", "d") is good
    assert f(t, "a", "b", "c", "zzz") is None
    assert f(t) is None


def test_financials_annual_fetch_stack_and_return(cache, env):
    inc, bal, cf = _stmt([[10.0, 8.0], [2.0, 1.0]]), _stmt([[5.0, 4.0], [1.0, 1.0]]), _stmt([[3.0, 2.0], [1.0, 0.5]])
    env.yf.tickers["T.NS"] = FakeTicker(income_stmt=inc, balance_sheet=bal, cash_flow=cf)
    out = cache.get_financials("T.NS")
    assert set(out) == {"income_stmt", "balance_sheet", "cash_flow"}
    assert out["income_stmt"] is inc
    path = env.mod.FUND_DIR / "T.NS_annual.parquet"
    assert path.exists()
    stacked = pd.read_parquet(path)
    assert stacked["_sheet"].tolist() == ["income_stmt"] * 2 + ["balance_sheet"] * 2 + ["cash_flow"] * 2
    assert all(isinstance(c, str) for c in stacked.columns)
    meta = cache._meta["fund:T.NS:annual"]
    assert meta["sheets"] == ["income_stmt", "balance_sheet", "cash_flow"]


def test_financials_fallback_attribute_names(cache, env):
    inc, cf = _stmt([[1.0, 1.0], [1.0, 1.0]]), _stmt([[2.0, 2.0], [2.0, 2.0]])
    env.yf.tickers["T"] = FakeTicker(income_stmt=pd.DataFrame(), financials=inc,
                                     balance_sheet=None, cashflow=cf)
    out = cache.get_financials("T")
    assert set(out) == {"income_stmt", "cash_flow"}          # empty balance sheet omitted
    assert out["income_stmt"] is inc and out["cash_flow"] is cf


def test_financials_second_call_served_from_cache(cache, env):
    inc = _stmt([[10.0, 8.0], [2.0, 1.0]])
    bal = _stmt([[5.0, 4.0], [1.0, 1.0]])
    env.yf.tickers["T"] = FakeTicker(income_stmt=inc, balance_sheet=bal, cash_flow=None)
    cache.get_financials("T")
    env.yf.tickers["T"] = None                               # any new fetch would crash
    again = cache.get_financials("T")
    assert set(again) == {"income_stmt", "balance_sheet"}
    assert again["income_stmt"].columns.tolist() == [pd.Timestamp("2024-03-31"), pd.Timestamp("2023-03-31")]
    assert again["income_stmt"].loc["Revenue"].tolist() == [10.0, 2.0]
    assert again["balance_sheet"].loc["NetIncome"].tolist() == [4.0, 1.0]


def test_financials_cache_expires_after_seven_days(cache, env):
    env.yf.tickers["T"] = FakeTicker(income_stmt=_stmt([[1.0, 1.0], [1.0, 1.0]]))
    cache.get_financials("T")
    env.yf.tickers["T"] = FakeTicker(income_stmt=_stmt([[99.0, 98.0], [1.0, 1.0]]))
    at(2026, 6, 22, 11, 59, 59)                              # 6d23h59m59s later
    assert cache.get_financials("T")["income_stmt"].loc["Revenue"].iloc[0] == 1.0
    at(2026, 6, 22, 12, 0, 1)                                # 7d + 1s
    assert cache.get_financials("T")["income_stmt"].loc["Revenue"].iloc[0] == 99.0


def test_financials_force_bypasses_cache(cache, env):
    env.yf.tickers["T"] = FakeTicker(income_stmt=_stmt([[1.0, 1.0], [1.0, 1.0]]))
    cache.get_financials("T")
    env.yf.tickers["T"] = FakeTicker(income_stmt=_stmt([[7.0, 7.0], [1.0, 1.0]]))
    assert cache.get_financials("T", force=True)["income_stmt"].loc["Revenue"].iloc[0] == 7.0


def test_financials_quarterly_uses_quarterly_attributes(cache, env):
    q = _stmt([[1.0, 2.0], [3.0, 4.0]], dates=("2024-03-31", "2023-12-31"))
    env.yf.tickers["T"] = FakeTicker(quarterly_income_stmt=None, quarterly_financials=q,
                                     income_stmt=_stmt([[9.0, 9.0], [9.0, 9.0]]))
    out = cache.get_financials("T", freq="quarterly")
    assert list(out) == ["quarterly_income"]
    assert out["quarterly_income"] is q
    assert (env.mod.FUND_DIR / "T_quarterly.parquet").exists()
    assert "fund:T:quarterly" in cache._meta


def test_financials_nothing_available_writes_nothing(cache, env):
    env.yf.tickers["T"] = FakeTicker()
    assert cache.get_financials("T") == {}
    assert not (env.mod.FUND_DIR / "T_annual.parquet").exists()
    assert "fund:T:annual" not in cache._meta


def test_financials_slash_in_ticker_sanitised(cache, env):
    env.yf.tickers["A/B"] = FakeTicker(income_stmt=_stmt([[1.0, 1.0], [1.0, 1.0]]))
    cache.get_financials("A/B")
    assert (env.mod.FUND_DIR / "A_B_annual.parquet").exists()


def test_financials_without_yfinance_returns_empty(cache, env, monkeypatch):
    monkeypatch.setattr(env.mod, "_YF_OK", False)
    assert cache.get_financials("T") == {}


def test_financials_fetch_error_is_swallowed_and_reported(cache, env, capsys):
    env.yf.tickers["T"] = None            # attribute access on None -> AttributeError path
    class Boom:
        def __getattr__(self, name):
            raise RuntimeError("no net")
    env.yf.tickers["T"] = Boom()
    assert cache.get_financials("T") == {}
    assert "Financial fetch failed for T: no net" in capsys.readouterr().out


def test_financials_fetch_error_quiet_mode(env, capsys):
    class Boom:
        def __getattr__(self, name):
            raise RuntimeError("no net")
    env.yf.tickers["T"] = Boom()
    assert env.mod.MarketCache(verbose=False).get_financials("T") == {}
    assert capsys.readouterr().out == ""


def test_financials_corrupt_cache_file_refetches(cache, env):
    env.yf.tickers["T"] = FakeTicker(income_stmt=_stmt([[1.0, 1.0], [1.0, 1.0]]))
    cache.get_financials("T")
    (env.mod.FUND_DIR / "T_annual.parquet").write_bytes(b"junk")
    env.yf.tickers["T"] = FakeTicker(income_stmt=_stmt([[5.0, 5.0], [1.0, 1.0]]))
    assert cache.get_financials("T")["income_stmt"].loc["Revenue"].iloc[0] == 5.0


# ── get_index ────────────────────────────────────────────────────────────────
def test_get_index_computes_regime_indicators_and_caches(cache, env, capsys):
    raw = make_ohlc(260)
    env.yf.store["^NSEI"] = raw
    df = cache.get_index("^NSEI")
    close = raw["Close"]
    assert env.yf.calls[0][1] == date(2026, 6, 15) - timedelta(days=10 * 365 + 30)
    assert df["dma50"].iloc[-1] == pytest.approx(close.iloc[-50:].mean())
    assert df["dma200"].iloc[-1] == pytest.approx(close.iloc[-200:].mean())
    assert df["dma50"].iloc[:49].isna().all() and pd.notna(df["dma50"].iloc[49])
    assert df["dma200"].iloc[:199].isna().all() and pd.notna(df["dma200"].iloc[199])
    assert df["dma200_sl"].iloc[-1] == pytest.approx(
        close.iloc[-200:].mean() - close.iloc[-205:-5].mean())
    exp_vol = close.pct_change().iloc[-20:].std() * np.sqrt(252) * 100
    assert df["vol20"].iloc[-1] == pytest.approx(exp_vol)
    assert (env.mod.INDEX_DIR / "NSEI.parquet").exists()
    m = cache._meta["index:^NSEI"]
    assert m["rows"] == 260 and m["to"] == "2026-06-12"
    assert "Index cached: ^NSEI 260 bars" in capsys.readouterr().out


def test_get_index_fresh_cache_hit_and_force(cache, env):
    env.yf.store["^GSPC"] = make_ohlc(30)
    first = cache.get_index("^GSPC")
    assert (env.mod.INDEX_DIR / "GSPC.parquet").exists()
    again = cache.get_index("^GSPC")
    assert len(env.yf.calls) == 1
    pd.testing.assert_frame_equal(first, again)
    cache.get_index("^GSPC", force=True)
    assert len(env.yf.calls) == 2


def test_get_index_stale_after_twenty_hours(cache, env):
    env.yf.store["^GSPC"] = make_ohlc(30)
    cache.get_index("^GSPC")
    at(2026, 6, 16, 7, 59, 0)                       # 19h59m
    cache.get_index("^GSPC")
    assert len(env.yf.calls) == 1
    at(2026, 6, 16, 8, 1, 0)                        # 20h01m
    cache.get_index("^GSPC")
    assert len(env.yf.calls) == 2


def test_get_index_empty_download_returns_empty_and_writes_nothing(cache, env):
    out = cache.get_index("^NOPE")
    assert out.empty
    assert not (env.mod.INDEX_DIR / "NOPE.parquet").exists()
    assert "index:^NOPE" not in cache._meta


def test_get_index_name_sanitises_caret_and_slash(cache, env):
    env.yf.store["^A/B"] = make_ohlc(10)
    cache.get_index("^A/B")
    assert (env.mod.INDEX_DIR / "A_B.parquet").exists()


# ── report ───────────────────────────────────────────────────────────────────
def test_report_empty_cache(cache, capsys):
    out = cache.report()
    assert out.empty
    assert "Cache is empty" in capsys.readouterr().out


def test_report_counts_size_and_age_buckets(cache, env, capsys):
    (env.mod.OHLC_DIR / "x.parquet").write_bytes(b"0" * 2_000_000)
    now = _NOW["v"]

    def ago(**kw):
        return (now - timedelta(**kw)).isoformat()

    cache._meta = {
        "ohlc:A": {"rows": 10, "from": "2020-01-01", "to": "2020-02-01", "updated": ago(minutes=30)},
        "ohlc:B": {"rows": 1, "updated": ago(minutes=59, seconds=59)},
        "ohlc:C": {"updated": ago(hours=1)},
        "ohlc:D": {"updated": ago(hours=23, minutes=59)},
        "fund:E:annual": {"updated": ago(hours=24)},
        "index:^F": {"updated": ago(hours=49)},
        "ohlc:G": {"updated": "bad"},
        "ohlc:H": {},
        "ohlc:" + "Z" * 60: {"updated": ago(minutes=5)},
    }
    df = cache.report()
    assert list(df.columns) == ["Key", "Rows", "From", "To", "Age"]
    age = dict(zip(df["Key"], df["Age"]))
    assert age["ohlc:A"] == "30m"
    assert age["ohlc:B"] == "59m"
    assert age["ohlc:C"] == "1h"
    assert age["ohlc:D"] == "23h"
    assert age["fund:E:annual"] == "1d"
    assert age["index:^F"] == "2d"
    assert age["ohlc:G"] == "—" and age["ohlc:H"] == "—"
    row_a = df[df["Key"] == "ohlc:A"].iloc[0]
    assert (row_a["Rows"], row_a["From"], row_a["To"]) == (10, "2020-01-01", "2020-02-01")
    row_h = df[df["Key"] == "ohlc:H"].iloc[0]
    assert (row_h["Rows"], row_h["From"], row_h["To"]) == ("—", "—", "—")
    assert df["Key"].str.len().max() == 50                     # truncated
    assert "Cache report: 7 OHLC | 1 fundamental | 1 index | 2.0 MB on disk" in capsys.readouterr().out


# ── purge ────────────────────────────────────────────────────────────────────
def test_purge_ticker_removes_its_files_and_meta_only(cache, env):
    env.yf.store["A.NS"] = make_ohlc(10)
    env.yf.store["B.NS"] = make_ohlc(10)
    cache.get_ohlc("A.NS")
    cache.get_ohlc("B.NS")
    for fq in ("annual", "quarterly"):
        (env.mod.FUND_DIR / f"A.NS_{fq}.parquet").write_bytes(b"x")
    cache._meta["fund:A.NS:annual"] = {"updated": "2026-06-15T00:00:00"}
    cache.purge("A.NS")
    assert not cache._ohlc_path("A.NS").exists()
    assert not list(env.mod.FUND_DIR.glob("A.NS_*"))
    assert cache._ohlc_path("B.NS").exists()
    assert "ohlc:B.NS" in cache._meta
    assert not [k for k in cache._meta if "A.NS" in k]
    assert "ohlc:A.NS" not in json.loads(env.mod.META_FILE.read_text())


def test_purge_unknown_ticker_is_a_noop(cache, env):
    env.yf.store["A"] = make_ohlc(10)
    cache.get_ohlc("A")
    cache.purge("ZZZ")
    assert cache._ohlc_path("A").exists() and "ohlc:A" in cache._meta


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="market_data_cache.py:purge matches meta keys by substring "
                          "(`ticker in k`), so purging TCS.NS also deletes the meta of NTCS.NS "
                          "while leaving its parquet file behind")
def test_purge_does_not_touch_other_tickers_with_similar_names(cache, env):
    cache._meta = {"ohlc:TCS.NS": {"updated": "x"}, "ohlc:NTCS.NS": {"updated": "x"}}
    cache.purge("TCS.NS")
    assert "ohlc:NTCS.NS" in cache._meta


def test_purge_all_wipes_everything(cache, env, capsys):
    env.yf.store["A"] = make_ohlc(10)
    cache.get_ohlc("A")
    (env.mod.FUND_DIR / "f.parquet").write_bytes(b"x")
    (env.mod.INDEX_DIR / "i.parquet").write_bytes(b"x")
    (env.mod.OHLC_DIR / "keep.txt").write_text("not parquet")
    cache.purge()
    assert not list(env.mod.OHLC_DIR.glob("*.parquet"))
    assert not list(env.mod.FUND_DIR.glob("*.parquet"))
    assert not list(env.mod.INDEX_DIR.glob("*.parquet"))
    assert (env.mod.OHLC_DIR / "keep.txt").exists()
    assert cache._meta == {} and json.loads(env.mod.META_FILE.read_text()) == {}
    assert "Cache wiped" in capsys.readouterr().out


# ── _clean_ohlc / _write_parquet ─────────────────────────────────────────────
def test_clean_ohlc_empty_passthrough(env):
    e = pd.DataFrame()
    assert env.mod.MarketCache._clean_ohlc(e) is e


def test_clean_ohlc_sorts_dedups_and_drops_bad_dates(env):
    df = make_ohlc(4)
    df.index = df.index.astype(str)
    df = pd.concat([df, df.iloc[[0]]])                       # duplicate first date
    df.loc["not-a-date"] = df.iloc[0]
    df = df.iloc[::-1]                                       # unsorted
    out = env.mod.MarketCache._clean_ohlc(df)
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.is_monotonic_increasing and not out.index.duplicated().any()
    assert len(out) == 4


def test_clean_ohlc_applies_central_gate_nonpositive_close_dropped(env):
    df = make_ohlc(5)
    df.iloc[2, df.columns.get_loc("Close")] = 0.0
    out = env.mod.MarketCache._clean_ohlc(df)
    assert len(out) == 4 and (out["Close"] > 0).all()


def test_clean_ohlc_all_rows_unusable_returns_empty_frame(env):
    df = make_ohlc(3)
    df["Close"] = -1.0
    out = env.mod.MarketCache._clean_ohlc(df)
    assert out.empty and list(out.columns) == list(df.columns)


def test_clean_ohlc_fallback_without_stock_utils(env, monkeypatch):
    monkeypatch.setattr(env.mod, "_clean_ohlcv_central", None)
    df = make_ohlc(4)
    df.iloc[1, df.columns.get_loc("Close")] = np.nan
    df.iloc[0, df.columns.get_loc("Open")] = np.nan
    dup = df.iloc[[3]].copy()
    dup["Close"] = 555.0
    out = env.mod.MarketCache._clean_ohlc(pd.concat([df, dup]))
    assert len(out) == 4
    assert out["Close"].iloc[3] == 555.0                    # keep="last" on duplicate
    assert out["Close"].iloc[1] == df["Close"].iloc[0]      # ffilled
    assert out["Open"].iloc[0] == df["Open"].iloc[1]        # bfilled
    assert not out.isna().any().any()


def test_write_parquet_atomic_success_leaves_no_tmp(env, tmp_path):
    p = tmp_path / "w.parquet"
    env.mod.MarketCache._write_parquet(p, make_ohlc(3))
    assert p.exists() and not p.with_suffix(".tmp").exists()
    assert len(pd.read_parquet(p)) == 3


def test_write_parquet_failure_cleans_tmp_and_reraises(env, tmp_path, monkeypatch):
    p = tmp_path / "w.parquet"

    def boom(self, path, *a, **k):
        open(path, "wb").write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    with pytest.raises(OSError, match="disk full"):
        env.mod.MarketCache._write_parquet(p, make_ohlc(3))
    assert not p.exists() and not p.with_suffix(".tmp").exists()


def test_write_parquet_failure_before_tmp_created_still_reraises(env, tmp_path, monkeypatch):
    def boom(self, path, *a, **k):
        raise ValueError("bad frame")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    with pytest.raises(ValueError, match="bad frame"):
        env.mod.MarketCache._write_parquet(tmp_path / "x.parquet", make_ohlc(2))


# ── get_cache / warm_cache ───────────────────────────────────────────────────
def test_get_cache_is_singleton_and_verbose(env):
    a = env.mod.get_cache()
    assert a is env.mod.get_cache()
    assert isinstance(a, env.mod.MarketCache) and a.verbose is True


class _StubCache:
    def __init__(self):
        self.events = []

    def get_index(self, sym):
        self.events.append(("index", sym))

    def get_ohlc_bulk(self, symbols):
        self.events.append(("bulk", list(symbols)))

    def report(self):
        self.events.append(("report",))


def test_warm_cache_default_indices_then_bulk_then_report(env, monkeypatch, capsys):
    stub = _StubCache()
    monkeypatch.setattr(env.mod, "get_cache", lambda: stub)
    env.mod.warm_cache(["A.NS"] * 250)
    assert stub.events == [("index", "^NSEI"), ("index", "^GSPC"),
                           ("bulk", ["A.NS"] * 250), ("report",)]
    out = capsys.readouterr().out
    assert "Warming cache: 250 stocks, 5-year window" in out
    assert "First run: ~6 minutes" in out                     # 250//100 * 3
    assert "Cache warm" in out


def test_warm_cache_custom_indices_and_small_universe(env, monkeypatch, capsys):
    stub = _StubCache()
    monkeypatch.setattr(env.mod, "get_cache", lambda: stub)
    env.mod.warm_cache(["X"], period_years=7, index_tickers=["^FTSE"])
    assert stub.events == [("index", "^FTSE"), ("bulk", ["X"]), ("report",)]
    out = capsys.readouterr().out
    assert "7-year window" in out and "First run: ~0 minutes" in out
