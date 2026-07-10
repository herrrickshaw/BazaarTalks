#!/usr/bin/env python3
"""
live_fundamentals.py
---------------------
Priority-fallback + batch collector over the two live, cookie-authenticated
sources (trendlyne_session.py, screener_session.py) — modeled on the sibling
working-files-repo project's fundamental_metrics.py priority order (Trendlyne
first for throughput, Screener.in fills gaps its endpoint doesn't cover),
except here the two are *merged* rather than strict fallback, since their
field coverage is complementary: Trendlyne gives pe/pb/mktcap/instihold,
Screener.in gives roe/roce/de/growth/div_yield/book_value.

Writes into fundamentals_cache.db's existing `fund` table (same schema
fundamentals_global.py already creates: ticker, market, pe, pb, roe, roa, de,
rev_growth, earn_growth, op_margin, div_yield, mktcap, sector), plus one
additive `source` column — so results are immediately visible through the
existing DuckDB warehouse.py `fundamentals` view, ticker_detail(), and
ticker_view.py with no other wiring.

Usage:
  python live_fundamentals.py RELIANCE                    # single ticker, print only
  python live_fundamentals.py --batch RELIANCE,TCS,INFY    # batch, writes to cache
  python live_fundamentals.py --batch-file tickers.txt --pause 1.5
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import time

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fundamentals_cache.db")

_SCHEMA_COLS = [
    "ticker", "market", "pe", "pb", "roe", "roa", "de", "rev_growth",
    "earn_growth", "op_margin", "div_yield", "mktcap", "sector", "source",
]


def fetch_one(symbol: str, market: str = "IN") -> dict:
    """Trendlyne + Screener.in merged (Trendlyne's fields win on overlap, since
    it's the faster/higher-throughput source per the existing platform convention)."""
    result = {"ticker": symbol, "market": market}
    sources = []

    try:
        import trendlyne_session
        tl = trendlyne_session.fundamentals(symbol)
        if tl:
            result.update({k: v for k, v in tl.items() if k not in ("source", "ticker")})
            sources.append("trendlyne")
    except Exception as e:
        print(f"  [live] trendlyne failed for {symbol}: {e}")

    try:
        import screener_session
        sc = screener_session.company_financials(symbol)
        if sc:
            for k, v in sc.items():
                if k in ("source", "ticker"):
                    continue
                result.setdefault(k, v)  # fill gaps only, don't overwrite trendlyne
            sources.append("screener.in")
    except Exception as e:
        print(f"  [live] screener.in failed for {symbol}: {e}")

    result["source"] = "+".join(sources) if sources else None
    return result


def _cache_conn() -> sqlite3.Connection:
    c = sqlite3.connect(CACHE)
    c.execute("PRAGMA journal_mode=DELETE;")
    c.execute("""CREATE TABLE IF NOT EXISTS fund(ticker TEXT PRIMARY KEY, market TEXT,
        pe REAL, pb REAL, roe REAL, roa REAL, de REAL, rev_growth REAL,
        earn_growth REAL, op_margin REAL, div_yield REAL, mktcap REAL, sector TEXT)""")
    # additive: only add `source` if this db predates it (fundamentals_global.py's
    # schema doesn't have it)
    cols = {r[1] for r in c.execute("PRAGMA table_info(fund)").fetchall()}
    if "source" not in cols:
        c.execute("ALTER TABLE fund ADD COLUMN source TEXT")
    c.commit()
    return c


def _write_row(conn: sqlite3.Connection, row: dict) -> None:
    vals = [row.get(col) for col in _SCHEMA_COLS]
    placeholders = ",".join("?" * len(_SCHEMA_COLS))
    conn.execute(f"INSERT OR REPLACE INTO fund({','.join(_SCHEMA_COLS)}) VALUES ({placeholders})", vals)


def fetch_batch(symbols: list[str], market: str = "IN", pause: float = 1.5, verbose: bool = True) -> dict:
    """Rate-limited batch fetch (mirrors screener_session.fundamentals_batch's
    politeness pause), writes each result into fundamentals_cache.db as it goes
    so a long run can be interrupted without losing progress."""
    conn = _cache_conn()
    results = {}
    try:
        for i, sym in enumerate(symbols):
            row = fetch_one(sym, market)
            results[sym] = row
            _write_row(conn, row)
            conn.commit()
            if verbose:
                print(f"  [{i+1}/{len(symbols)}] {sym}: source={row.get('source')}")
            if i < len(symbols) - 1:
                time.sleep(pause)
    finally:
        conn.close()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol", nargs="?")
    ap.add_argument("--batch", help="comma-separated ticker list")
    ap.add_argument("--batch-file", help="path to a file with one ticker per line")
    ap.add_argument("--market", default="IN")
    ap.add_argument("--pause", type=float, default=1.5)
    args = ap.parse_args()

    if args.batch or args.batch_file:
        symbols = (args.batch.split(",") if args.batch else
                   [l.strip() for l in open(args.batch_file) if l.strip()])
        results = fetch_batch([s.strip() for s in symbols if s.strip()], args.market, args.pause)
        n_ok = sum(1 for r in results.values() if r.get("source"))
        print(f"\n  {n_ok}/{len(results)} tickers got at least one live source")
    elif args.symbol:
        print(fetch_one(args.symbol, args.market))
    else:
        print("usage: live_fundamentals.py SYMBOL | --batch A,B,C | --batch-file tickers.txt")


if __name__ == "__main__":
    main()
