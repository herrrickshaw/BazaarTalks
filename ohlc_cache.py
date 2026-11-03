"""
ohlc_cache.py
-------------
The cache *policy* behind ``market_store.cached_download``, separated from the Cassandra / Kafka
plumbing so it can be reasoned about and tested without either:

* ``plan_fetch``       pure: coverage per ticker + today  ->  fresh / stale / miss buckets
* ``cached_download``  orchestration over an ``OhlcStore`` and a ``downloader`` (both injected)

``market_store`` supplies the real Cassandra-backed store and the governed yfinance downloader;
tests supply in-memory fakes. Behaviour (bucket rules, call order, stderr messages) is exactly
what ``market_store.cached_download`` did before the split.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Tuple

import pandas as pd

# (min_date, max_date, n_bars) held for a ticker, or None when nothing is cached.
Coverage = Optional[Tuple[date, date, int]]

HOLIDAY_GAP_DAYS = 4        # tolerate a long weekend / holiday between last bar and "today"
HISTORY_BUFFER_DAYS = 30    # a "5y" pull starts ~exactly 5y ago; don't demand more than it returns


class OhlcStore(Protocol):
    """What the cache policy needs from a store (Cassandra in production, a dict in tests)."""

    def connect(self) -> None: ...
    def coverage(self, ticker: str) -> Coverage: ...
    def get_ohlc(self, ticker: str) -> pd.DataFrame: ...
    def put_ohlc(self, ticker: str, df: pd.DataFrame) -> None: ...


Downloader = Callable[..., Dict[str, pd.DataFrame]]   # yf_download(tickers, start=… | period=…)


def plan_fetch(coverage_by_ticker: Iterable[Tuple[str, Coverage]], today: date, years: int,
               refresh_days: int) -> Tuple[List[str], List[Tuple[str, date]], List[str]]:
    """Split tickers into ``(fresh, stale, misses)``.

    fresh  — cached with enough history and a bar within ``refresh_days`` (+ holiday gap): serve as is.
    stale  — enough history but newer bars missing: fetch only the days after the last cached bar.
    misses — absent or too little history: full pull.
    """
    fresh_cut = today - timedelta(days=refresh_days + HOLIDAY_GAP_DAYS)
    need_start = today - timedelta(days=int(years * 365) - HISTORY_BUFFER_DAYS)
    fresh: List[str] = []
    stale: List[Tuple[str, date]] = []
    misses: List[str] = []
    for ticker, cov in coverage_by_ticker:
        if cov and cov[0] <= need_start and cov[1] >= fresh_cut:
            fresh.append(ticker)
        elif cov and cov[0] <= need_start:
            stale.append((ticker, cov[1]))
        else:
            misses.append(ticker)
    return fresh, stale, misses


def cached_download(tickers: Iterable[str], store: OhlcStore, downloader: Downloader,
                    years: int = 5, refresh_days: int = 3,
                    today: Optional[date] = None) -> Dict[str, pd.DataFrame]:
    """OHLC per ticker: served from ``store`` when fresh, else downloaded and written back.

    ``store.connect()`` runs first so an unreachable store raises before any download happens.
    """
    today = today or date.today()
    store.connect()
    coverage = [(t, store.coverage(t)) for t in tickers]
    fresh, stale, misses = plan_fetch(coverage, today, years, refresh_days)

    out: Dict[str, pd.DataFrame] = {t: store.get_ohlc(t) for t in fresh}

    if stale:
        print(f"  [market_store] {len(stale)} stale -> fetching only new bars since last cache…",
              file=sys.stderr, flush=True)
        for t, last in stale:
            new = downloader([t], start=str(last + timedelta(days=1))).get(t)
            if new is not None and not new.empty:
                store.put_ohlc(t, new)           # upsert (PK = ticker,d) — idempotent
            out[t] = store.get_ohlc(t)

    if misses:
        print(f"  [market_store] {len(out)} from Cassandra, downloading {len(misses)} misses…",
              file=sys.stderr, flush=True)
        for t, df in downloader(misses, period=f"{years}y").items():
            out[t] = df
            store.put_ohlc(t, df)
    elif not stale:
        print(f"  [market_store] all {len(out)} tickers served from Cassandra (no network)",
              file=sys.stderr, flush=True)
    return out
