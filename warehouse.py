#!/usr/bin/env python3
"""
warehouse.py
------------
A DuckDB analytical layer over all the platform's data — one SQL surface to
update, process, filter, and depict results across the whole system, without
copying data. DuckDB reads the parquets directly and attaches the SQLite result
DBs, so views always reflect the latest files (easy "update" = just re-run the
producers; the warehouse sees the new data).

Two-tier design (Modern Data Architecture Blueprint):
  Cassandra (market_store)  = operational store: OHLC cache, CDC source, streaming.
  DuckDB    (this)          = analytical / serving layer: fast filtering & rollups.

Unified views:
  ohlc          all 19 markets' daily OHLC (from cleaned_long_*.parquet, market column)
  companies     industry/peer dataset (companies_industry.parquet)
  fundamentals  yfinance fundamentals (fundamentals_cache.db)
  dvm_global    technical DVM/Trendlyne metrics per stock (dvm_global.db)
  dvm_composite global GGG/GGB/BBG classification (dvm_composite.db)
  viability     screen-viability summary (viability_summary.db)

Usage:
  python warehouse.py --build                       # (re)create the warehouse views
  python warehouse.py --show ggg_global             # pre-built result views
  python warehouse.py --show markets
  python warehouse.py --filter "roe>15 and de<1 and M>=70"   # ad-hoc DVM filter
  python warehouse.py --sql "SELECT market, count(*) FROM ohlc GROUP BY 1"
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import duckdb

import incremental   # F9.1 partition-incremental refresh (pure pandas helpers)

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = os.path.expanduser("~/Downloads/code/python_files/cache_seed")
# India's seed lives inside this repo (cache_seed_local/, via build_india_seed.py),
# not the external SEED tree above — that directory is shared with a parallel
# session that has previously wiped untracked/gitignored files there (a
# documented prior loss of India collections specifically), so India's data
# survives under this repo's own git history instead.
SEED_LOCAL = os.path.join(HERE, "cache_seed_local")
DB = os.path.join(HERE, "market.duckdb")

# SQLite result DBs to attach (alias -> (file, {view: table}))
SQLITE_SOURCES = {
    "fund_db":   ("fundamentals_cache.db", {"fundamentals": "fund"}),
    "dvmg_db":   ("dvm_global.db", {"dvm_global": "dvm_global"}),
    "dvmc_db":   ("dvm_composite.db", {"dvm_composite": "dvm_composite"}),
    "viab_db":   ("viability_summary.db", {"viability": "market_screen_summary"}),
}


def build(con):
    con.execute("INSTALL sqlite; LOAD sqlite;")
    # OHLC across all markets, market derived from the parquet filename.
    # Two glob roots unioned: the external 18-market seed, plus this repo's
    # own cache_seed_local/ (India — see SEED_LOCAL above for why it's separate).
    globs = [os.path.join(SEED, "cleaned_long_*.parquet")]
    if os.path.isdir(SEED_LOCAL) and os.listdir(SEED_LOCAL):
        globs.append(os.path.join(SEED_LOCAL, "cleaned_long_*.parquet"))
    glob_list = "[" + ", ".join(f"'{g}'" for g in globs) + "]"
    con.execute(f"""
        CREATE OR REPLACE VIEW ohlc AS
        SELECT Symbol AS ticker, Date, Open, High, Low, Close, Volume,
               regexp_extract(filename, 'cleaned_long_([A-Za-z]+)\\.parquet', 1) AS market
        FROM read_parquet({glob_list}, filename=true)
    """)
    ci = os.path.join(HERE, "companies_industry.parquet")
    if os.path.exists(ci):
        con.execute(f"CREATE OR REPLACE VIEW companies AS SELECT * FROM read_parquet('{ci}')")
    for alias, (fn, views) in SQLITE_SOURCES.items():
        path = os.path.join(HERE, fn)
        if not os.path.exists(path):
            print(f"  [skip] {fn} not present", file=sys.stderr); continue
        con.execute(f"ATTACH IF NOT EXISTS '{path}' AS {alias} (TYPE sqlite)")
        for view, table in views.items():
            try:
                if view == "fundamentals":   # yfinance can store 'Infinity' as text -> coerce
                    base_cols = """ticker, market,
                        TRY_CAST(pe AS DOUBLE) pe, TRY_CAST(pb AS DOUBLE) pb,
                        TRY_CAST(roe AS DOUBLE) roe, TRY_CAST(roa AS DOUBLE) roa,
                        TRY_CAST(de AS DOUBLE) de, TRY_CAST(rev_growth AS DOUBLE) rev_growth,
                        TRY_CAST(earn_growth AS DOUBLE) earn_growth, TRY_CAST(op_margin AS DOUBLE) op_margin,
                        TRY_CAST(div_yield AS DOUBLE) div_yield, TRY_CAST(mktcap AS DOUBLE) mktcap,
                        sector"""
                    try:
                        # `source` (which live_fundamentals.py stamps: trendlyne/screener.in/
                        # both) is additive — older fund tables built by fundamentals_global.py
                        # alone won't have it yet
                        con.execute(f"""CREATE OR REPLACE VIEW fundamentals AS
                            SELECT {base_cols}, source FROM {alias}.{table}""")
                    except Exception:
                        con.execute(f"""CREATE OR REPLACE VIEW fundamentals AS
                            SELECT {base_cols}, CAST(NULL AS VARCHAR) AS source FROM {alias}.{table}""")
                else:
                    con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM {alias}.{table}")
            except Exception as e:
                print(f"  [skip] {view}: {e}", file=sys.stderr)
    print("  warehouse built:", [r[0] for r in con.execute("SHOW TABLES").fetchall()],
          file=sys.stderr)


# Pre-built "depict results" queries
SHOWS = {
    "markets": "SELECT market, count(DISTINCT ticker) AS tickers, count(*) AS bars, "
               "min(Date) AS from_, max(Date) AS to_ FROM ohlc GROUP BY 1 ORDER BY tickers DESC",
    "ggg_global": "SELECT market, ticker, D, V, M, composite, label FROM dvm_composite "
                  "WHERE code='GGG' ORDER BY composite DESC LIMIT 25",
    "dvm_dist": "SELECT code, label, count(*) n FROM dvm_composite GROUP BY 1,2 ORDER BY n DESC",
    "momentum_by_market": "SELECT market, count(*) scored, "
                          "round(avg(M),1) avg_M, sum(CASE WHEN M>=70 THEN 1 ELSE 0 END) hi_mom "
                          "FROM dvm_global GROUP BY 1 ORDER BY hi_mom DESC",
    "high_roe_low_de": "SELECT market, ticker, round(roe,1) roe, round(de,2) de, round(pe,1) pe, sector "
                       "FROM fundamentals WHERE roe>15 AND de<1 AND de IS NOT NULL "
                       "ORDER BY roe DESC LIMIT 25",
    "industry_segments": "SELECT segment, n_companies FROM (SELECT * FROM companies) "
                         "USING SAMPLE 0 ROWS",  # placeholder; companies has list cols
}

# Tables joined per ticker for ticker_detail(); source view -> result key.
# ohlc is handled separately (bar history, not a single-row join).
_TICKER_JOIN_VIEWS = {
    "fundamentals": "fundamentals",
    "dvm_global": "dvm_technical",
    "dvm_composite": "dvm_composite",
}


def json_safe(value):
    """Recursively replace NaN/+-Infinity with None.

    Real fundamentals data legitimately has undefined ratios (a non-dividend
    stock's div_yield, a divide-by-zero P/E, ...), which pandas represents as
    float('nan'). Two independent consumers need this normalized to None
    rather than a bare NaN:
      - JSON encoders that reject non-finite floats outright (Starlette's
        default JSONResponse passes `allow_nan=False`, so any route
        returning a DataFrame with a NaN anywhere crashes the whole request
        with "Out of range float values are not JSON compliant" -- found by
        running the real /filter endpoint against real production data,
        where BRK-B's div_yield is NaN because it pays no dividend).
      - ticker_view.py's `_card()`, which checks `if value is None` to
        decide whether to show "n/a" -- a bare NaN fails that check (NaN is
        not None) and would display the literal text "nan" instead.
    Operates on whatever to_dict()/to_dict("records") already produced (a
    dict, or a list of dicts), not on the DataFrame itself, to sidestep
    pandas' object-dtype casting quirks when mixing NaN and None in the same
    column.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def ticker_detail(con, ticker: str, market: str, bars: int = 60) -> dict:
    """
    Single-ticker lookup across every view the warehouse currently has built —
    OHLC history plus whichever of fundamentals/dvm_global/dvm_composite are
    present. Missing views (not every environment has run every producer) are
    silently omitted rather than erroring, matching the rest of the platform's
    "skip what's absent" convention. Uses bound parameters (?, not string
    interpolation) since ticker/market come from a request path/query param.
    """
    available = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    result: dict = {"ticker": ticker, "market": market}

    if "ohlc" in available:
        df = con.execute(
            "SELECT Date, Open, High, Low, Close, Volume FROM ohlc "
            "WHERE ticker = ? AND market = ? ORDER BY Date DESC LIMIT ?",
            [ticker, market, bars],
        ).df()
        result["ohlc"] = json_safe(df.to_dict("records"))
    else:
        result["ohlc"] = []

    for view, key in _TICKER_JOIN_VIEWS.items():
        if view not in available:
            result[key] = None
            continue
        row = con.execute(
            f"SELECT * FROM {view} WHERE ticker = ? AND market = ? LIMIT 1",
            [ticker, market],
        ).df()
        result[key] = json_safe(row.to_dict("records")[0]) if not row.empty else None

    return result


def refresh_ohlc_partition(market: str, new_parquet: str) -> dict:
    """F9.1: append only genuinely new dates from `new_parquet` into a market's
    cleaned_long parquet (per-symbol high-water mark), instead of rebuilding it.
    Returns a small summary of what was appended."""
    import pandas as pd
    base_path = os.path.join(SEED, f"cleaned_long_{market}.parquet")
    base = pd.read_parquet(base_path) if os.path.exists(base_path) else pd.DataFrame()
    new = pd.read_parquet(new_parquet)
    merged = incremental.append_new_dates(base, new, date_col="Date", key_col="Symbol")
    added = len(merged) - len(base)
    merged.to_parquet(base_path, index=False, compression="snappy")
    return {"market": market, "rows_before": len(base), "rows_added": added,
            "rows_after": len(merged)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--show", choices=[k for k in SHOWS if k != "industry_segments"])
    ap.add_argument("--filter", help="ad-hoc predicate over the dvm_composite⋈fundamentals join")
    ap.add_argument("--sql", help="run arbitrary SQL")
    ap.add_argument("--refresh-ohlc", nargs=2, metavar=("MARKET", "NEW_PARQUET"),
                    help="F9.1: incrementally append new dates into a market's parquet")
    args = ap.parse_args()

    if args.refresh_ohlc:
        print(refresh_ohlc_partition(*args.refresh_ohlc)); return

    con = duckdb.connect(DB)
    # DuckDB sqlite ATTACHments are per-session, so (re)build views every run — it's
    # just cheap view definitions over the live parquets/SQLite (that's the "update").
    build(con)

    con.execute("SET max_expression_depth=10000")
    if args.sql:
        print(con.execute(args.sql).df().to_string(index=False))
    elif args.show:
        print(con.execute(SHOWS[args.show]).df().to_string(index=False))
    elif args.filter:
        q = (f"SELECT c.market, c.ticker, c.D, c.V, c.M, c.composite, c.code, "
             f"f.roe, f.de, f.pe, f.sector FROM dvm_composite c "
             f"LEFT JOIN fundamentals f ON c.ticker=f.ticker "
             f"WHERE {args.filter} ORDER BY c.composite DESC LIMIT 30")
        print(con.execute(q).df().to_string(index=False))
    else:
        # default: quick health summary
        print("=== warehouse tables ===")
        for r in con.execute("SHOW TABLES").fetchall():
            try:
                n = con.execute(f"SELECT count(*) FROM {r[0]}").fetchone()[0]
                print(f"  {r[0]:16} {n:>10,} rows")
            except Exception:
                print(f"  {r[0]:16} (view)")
    con.close()


if __name__ == "__main__":
    main()
