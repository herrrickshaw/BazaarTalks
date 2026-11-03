"""Characterisation of MarketCache download helpers and get_ohlc_bulk with a fake yfinance (HOME is a temp dir)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from _golden import check

import market_data_cache as mdc

COLS = ["Open", "High", "Low", "Close", "Volume"]


def _frame(seed, n=12, start="2026-01-01"):
    rng = np.random.RandomState(seed)
    idx = pd.bdate_range(start, periods=n)
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"Open": c, "High": c + 1, "Low": c - 1, "Close": c, "Volume": rng.randint(1e5, 1e6, n)},
                        index=idx)


class FakeYf:
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.calls = []

    def download(self, tickers, **kw):
        self.calls.append((tickers if isinstance(tickers, str) else list(tickers), sorted(kw)))
        return self.behaviour(tickers, kw, len(self.calls))


def _multi(tickers, kw, n):
    tick = [t for t in tickers if t != "MISSING"]
    return pd.concat({t: _frame(sum(map(ord, t))) for t in tick}, axis=1).swaplevel(axis=1).sort_index(axis=1)


def _to_dict(df):
    return {c: [round(float(x), 6) for x in df[c]] for c in df.columns}, [str(i)[:10] for i in df.index]


@pytest.fixture()
def cache(monkeypatch, tmp_path):
    for name in ("OHLC_DIR", "META_DIR"):
        monkeypatch.setattr(mdc, name, tmp_path / name.lower())
        (tmp_path / name.lower()).mkdir()
    monkeypatch.setattr(mdc, "META_FILE", tmp_path / "meta_dir" / "cache_index.json")
    monkeypatch.setattr(mdc.time, "sleep", lambda s: None)
    monkeypatch.setattr(mdc, "_YF_OK", True)
    return mdc.MarketCache(verbose=True)


def test_bulk_download_multiindex_and_missing(cache, monkeypatch, capsys):
    fake = FakeYf(_multi)
    monkeypatch.setattr(mdc, "yf", fake, raising=False)
    monkeypatch.setattr(mdc, "BATCH_SIZE", 2)
    out = cache._bulk_yf_download(["AAA", "MISSING", "BBB", "CCC"], start=pd.Timestamp("2026-01-01").date())
    check("mdc/bulk_multi", {"out": {k: _to_dict(v) for k, v in out.items()}, "stdout": capsys.readouterr().out,
                            "calls": fake.calls})


def test_bulk_download_single_frame_empty_and_errors(cache, monkeypatch, capsys):
    seq = [pd.DataFrame(), _frame(1), RuntimeError("boom")]
    fake = FakeYf(lambda t, kw, n: (_ for _ in ()).throw(seq[n - 1]) if isinstance(seq[n - 1], Exception) else seq[n - 1])
    monkeypatch.setattr(mdc, "yf", fake, raising=False)
    monkeypatch.setattr(mdc, "BATCH_SIZE", 1)
    out = cache._bulk_yf_download(["A", "B", "C"], start=pd.Timestamp("2026-01-01").date())
    check("mdc/bulk_single", {"keys": sorted(out), "stdout": capsys.readouterr().out})


def test_bulk_download_rate_limit_retry(cache, monkeypatch, capsys):
    def beh(t, kw, n):
        if n < 3:
            raise RuntimeError("429 Too Many Requests")
        return _multi(t, kw, n)
    fake = FakeYf(beh)
    monkeypatch.setattr(mdc, "yf", fake, raising=False)
    out = cache._bulk_yf_download(["AAA", "BBB"], start=pd.Timestamp("2026-01-01").date())
    check("mdc/bulk_ratelimit", {"keys": sorted(out), "stdout": capsys.readouterr().out, "ncalls": len(fake.calls)})


def test_download_single(cache, monkeypatch):
    seq = {"OK": lambda: _frame(3), "EMPTY": lambda: pd.DataFrame(), "RATE": None}

    def beh(t, kw, n):
        if t == "RATE":
            raise RuntimeError("429")
        if t == "BAD":
            raise ValueError("other")
        return seq[t]() if t in seq else pd.DataFrame()

    monkeypatch.setattr(mdc, "yf", FakeYf(beh), raising=False)
    s = pd.Timestamp("2026-01-01").date()
    res = {t: cache._yf_download_single(t, s) for t in ("OK", "EMPTY", "RATE", "BAD")}
    check("mdc/single", {k: (_to_dict(v) if len(v) else None) for k, v in res.items()})
    monkeypatch.setattr(mdc, "_YF_OK", False)
    assert cache._yf_download_single("OK", s).empty and cache._bulk_yf_download(["OK"], s) == {}


def test_get_ohlc_bulk_cold_incremental_fresh(cache, monkeypatch, capsys):
    pytest.importorskip("pyarrow")
    monkeypatch.setattr(mdc, "yf", FakeYf(_multi), raising=False)
    first = cache.get_ohlc_bulk(["AAA", "BBB"])            # cold start
    cache._mem.clear()
    second = cache.get_ohlc_bulk(["AAA", "BBB", "CCC"])    # fresh x2 + cold
    cache._meta = {k: {**v, "updated": "2000-01-01T00:00:00"} if isinstance(v, dict) else v
                   for k, v in cache._meta.items()}
    third = cache.get_ohlc_bulk(["AAA"], force=True)       # incremental
    check("mdc/get_ohlc_bulk", {"first": {k: _to_dict(v) for k, v in sorted(first.items())},
                               "second": sorted(second), "third": {k: _to_dict(v) for k, v in third.items()},
                               "stdout": capsys.readouterr().out})


def test_write_parquet_is_atomic_and_cleans_tmp(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    path = tmp_path / "x.parquet"
    mdc.MarketCache._write_parquet(path, _frame(1))
    assert path.exists() and not path.with_suffix(".tmp").exists()
    assert len(pd.read_parquet(path)) == 12

    def boom(self, tmp, **kw):
        tmp.write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    with pytest.raises(OSError):
        mdc.MarketCache._write_parquet(tmp_path / "y.parquet", _frame(2))
    assert not (tmp_path / "y.tmp").exists() and not (tmp_path / "y.parquet").exists()


def test_clean_ohlc_empty_unparseable_index_and_fallback(monkeypatch):
    assert mdc.MarketCache._clean_ohlc(pd.DataFrame()).empty
    df = _frame(4)
    df.index = list(df.index[:-1].astype(str)) + ["not-a-date"]
    out = mdc.MarketCache._clean_ohlc(df)
    assert isinstance(out.index, pd.DatetimeIndex) and out.index.is_monotonic_increasing
    monkeypatch.setattr(mdc, "_clean_ohlcv_central", None)
    dup = pd.concat([_frame(5), _frame(5).iloc[-2:]])
    dup.iloc[3, 3] = np.nan
    fb = mdc.MarketCache._clean_ohlc(dup)
    check("mdc/clean_fallback", {"n": len(fb), "nan": int(fb.isna().sum().sum()), "close": _to_dict(fb)[0]["Close"]})
