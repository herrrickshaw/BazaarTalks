#!/usr/bin/env python3
"""
build_india_seed.py
--------------------
Converts the existing NSE Bhavcopy cache (market_data_consolidated/india/
nse_bhav_cache.db — 2,740 tickers, ~1y daily OHLCV, already collected, just
never wired into this platform's warehouse) into cleaned_long_IN.parquet,
matching the exact schema of the other 18 markets' seed files (Date, Open,
High, Low, Close as float32, Volume as int64, Symbol).

Written to cache_seed_local/ *inside this repo*, not the external
~/Downloads/code/python_files/cache_seed/ tree the other 18 markets use —
that directory is shared with a parallel session that has previously wiped
untracked/gitignored files (documented loss of prior India collections), so
India's seed lives here instead, where it's under this repo's own git
history. warehouse.py's ohlc view unions both locations.

Usage:
  python build_india_seed.py
"""
from __future__ import annotations

import os
import sqlite3

import pandas as pd

SOURCE_DB = "/Users/umashankar/market_data_consolidated/india/nse_bhav_cache.db"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "cache_seed_local")
OUT_FILE = os.path.join(OUT_DIR, "cleaned_long_IN.parquet")


def main():
    if not os.path.exists(SOURCE_DB):
        print(f"source not found: {SOURCE_DB}")
        return

    conn = sqlite3.connect(SOURCE_DB)
    df = pd.read_sql_query("SELECT symbol, d, open, high, low, close, volume FROM prices", conn)
    conn.close()

    df = df.rename(columns={
        "symbol": "Symbol", "d": "Date", "open": "Open",
        "high": "High", "low": "Low", "close": "Close", "volume": "Volume",
    })
    df["Date"] = pd.to_datetime(df["Date"])
    for col in ("Open", "High", "Low", "Close"):
        df[col] = df[col].astype("float32")
    df["Volume"] = df["Volume"].fillna(0).astype("int64")
    df = df[["Date", "Open", "High", "Low", "Close", "Volume", "Symbol"]]
    df = df.dropna(subset=["Date", "Symbol"]).sort_values(["Symbol", "Date"])

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_parquet(OUT_FILE, index=False, compression="snappy")

    print(f"wrote {OUT_FILE}")
    print(f"  {len(df):,} rows, {df['Symbol'].nunique():,} tickers, "
          f"{df['Date'].min().date()} - {df['Date'].max().date()}")
    print(f"  {os.path.getsize(OUT_FILE) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
