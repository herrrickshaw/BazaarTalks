"""OHLC persistence helpers for the Parquet market cache: pre-cache cleaning and atomic Parquet writes."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pandas as pd


def clean_ohlc(df: pd.DataFrame, central_clean: Optional[Callable] = None) -> pd.DataFrame:
    """Clean OHLC before caching — delegates to the central clean_ohlcv gate.

    Cache-specific prep: coerce to a sorted DatetimeIndex and drop unparseable
    dates first, then hand off to ``central_clean`` (stock_utils.clean_ohlcv) so the cache stores
    exactly the same hygiene (numeric coercion, dedup, non-positive-price drop,
    OHLC-integrity repair, bad-print/split neutralisation) every consumer sees.
    Falls back to the prior ffill/bfill behaviour if ``central_clean`` is None.
    """
    if df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    if central_clean is not None:
        cleaned = central_clean(df, min_bars=1)
        return cleaned if cleaned is not None else df.iloc[0:0]
    # fallback (stock_utils not importable)
    df = df[~df.index.duplicated(keep="last")]
    return df.ffill().bfill()


def write_parquet_atomic(path: Path, df: pd.DataFrame) -> None:
    """Atomic write: temp file → rename (prevents corruption on crash)."""
    tmp = path.with_suffix(".tmp")
    try:
        df.to_parquet(tmp, compression="snappy", index=True)
        tmp.rename(path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
