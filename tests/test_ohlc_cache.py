"""ohlc_cache: cache policy tested against an in-memory fake store (no Cassandra, Kafka or network)."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

import ohlc_cache as oc

TODAY = date(2026, 6, 15)


def _cov(first_ago, last_ago, n=1000):
    return (TODAY - timedelta(days=first_ago), TODAY - timedelta(days=last_ago), n)


def test_plan_fetch_buckets():
    cov = [("FRESH", _cov(1900, 1)), ("STALE", _cov(1900, 30)), ("SHORT", _cov(100, 1)), ("NONE", None)]
    fresh, stale, misses = oc.plan_fetch(cov, TODAY, years=5, refresh_days=3)
    assert fresh == ["FRESH"]
    assert stale == [("STALE", TODAY - timedelta(days=30))]
    assert misses == ["SHORT", "NONE"]


def test_plan_fetch_freshness_boundary_is_inclusive():
    edge = _cov(1900, 3 + oc.HOLIDAY_GAP_DAYS)
    assert oc.plan_fetch([("E", edge)], TODAY, 5, 3)[0] == ["E"]
    assert oc.plan_fetch([("E", _cov(1900, 3 + oc.HOLIDAY_GAP_DAYS + 1))], TODAY, 5, 3)[1]


class FakeStore:
    def __init__(self, cov=None, unreachable=False):
        self.cov, self.unreachable, self.calls, self.bars = cov or {}, unreachable, [], {}

    def connect(self):
        self.calls.append("connect")
        if self.unreachable:
            raise RuntimeError("store down")

    def coverage(self, t):
        return self.cov.get(t)

    def get_ohlc(self, t):
        return self.bars.get(t, pd.DataFrame({"Close": [1.0]}))

    def put_ohlc(self, t, df):
        self.calls.append(("put", t, len(df)))
        self.bars[t] = df


def _downloader(log):
    def dl(tickers, **kw):
        log.append((tuple(tickers), kw))
        return {t: pd.DataFrame({"Close": [2.0, 3.0]}) for t in tickers}
    return dl


def test_cached_download_fresh_never_touches_network(capsys):
    store, log = FakeStore({"A": _cov(1900, 1)}), []
    out = oc.cached_download(["A"], store, _downloader(log), today=TODAY)
    assert list(out) == ["A"] and log == []
    assert "no network" in capsys.readouterr().err


def test_cached_download_stale_fetches_only_new_bars_then_misses_full_period():
    last = TODAY - timedelta(days=30)
    store = FakeStore({"S": _cov(1900, 30)})
    log = []
    out = oc.cached_download(["S", "M"], store, _downloader(log), years=5, today=TODAY)
    assert log[0] == (("S",), {"start": str(last + timedelta(days=1))})
    assert log[1] == (("M",), {"period": "5y"})
    assert list(out) == ["S", "M"]
    assert ("put", "S", 2) in store.calls and ("put", "M", 2) in store.calls


def test_cached_download_unreachable_store_raises_before_any_download():
    log = []
    with pytest.raises(RuntimeError):
        oc.cached_download(["A"], FakeStore(unreachable=True), _downloader(log), today=TODAY)
    assert log == []
