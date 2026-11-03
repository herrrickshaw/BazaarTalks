"""yfinance download helpers for the Parquet market cache (retry on rate limits, batching).

The yfinance module, batch size and sleep function are parameters so the cache can be exercised with a fake.
"""
from __future__ import annotations

import time
from datetime import date

import pandas as pd

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _is_rate_limited(exc: Exception, extra: tuple = ()) -> bool:
    msg = str(exc)
    return "Rate" in msg or "429" in msg or any(x in msg for x in extra)


def download_single(yf_mod, ticker: str, start: date, sleep=time.sleep) -> pd.DataFrame:
    """Download OHLCV for a single ticker with up to 3 attempts (backs off 15 s * n on rate limits)."""
    for attempt in range(3):
        try:
            df = yf_mod.download(ticker, start=start, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df = df.xs(ticker, axis=1, level=1)
            if not df.empty:
                return df[OHLCV].dropna(how="all")
        except Exception as e:
            if _is_rate_limited(e):
                sleep(15 * (attempt + 1))
            else:
                break
    return pd.DataFrame()


def _split_batch(raw: pd.DataFrame, batch: list) -> dict:
    """{ticker: OHLCV frame} out of one yf.download result (MultiIndex or single-ticker layout)."""
    result = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for t in batch:
            try:
                df = raw.xs(t, axis=1, level=1).dropna(how="all")
                if not df.empty and "Close" in df.columns:
                    result[t] = df[OHLCV]
            except KeyError:
                pass
    elif not raw.empty:
        result[batch[0]] = raw[OHLCV]
    return result


def download_bulk(yf_mod, tickers: list, start: date, batch_size: int, batch_sleep: float,
                  sleep=time.sleep) -> dict:
    """Batch-download OHLCV for many tickers, retrying rate-limited batches up to 3 times (30 s * n)."""
    result = {}
    batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
    n_total = len(tickers)
    n_done = 0

    for idx, batch in enumerate(batches, 1):
        print(f"    Downloading batch {idx}/{len(batches)} "
              f"({len(batch)} tickers) …", end=" ", flush=True)
        for attempt in range(3):
            try:
                raw = yf_mod.download(batch, start=start, auto_adjust=True, threads=True, progress=False)
                if raw.empty:
                    print("empty"); break
                result.update(_split_batch(raw, batch))
                ok = sum(1 for t in batch if t in result)
                print(f"OK ({ok}/{len(batch)})")
                n_done += ok
                break
            except Exception as e:
                if _is_rate_limited(e, ("Too Many",)):
                    wait = 30 * (attempt + 1)
                    print(f"\n    Rate limited — waiting {wait}s …", end="", flush=True)
                    sleep(wait)
                else:
                    print(f"ERROR: {e}"); break
        if idx < len(batches):
            sleep(batch_sleep)

    print(f"  Bulk download complete: {n_done}/{n_total} tickers")
    return result
