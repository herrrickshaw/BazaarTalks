#!/usr/bin/env python3
"""
pipeline.py
-----------
End-to-end orchestrator: source -> validate -> process -> analyze -> graphics,
over the platform's existing scripts (nothing here reimplements what already
exists — see README.md's architecture table for what each stage wraps).

Note on "source": full_{market}_market_scan.py (Excel-output scanners) is a
separate, older sourcing path from the 19-market cleaned_long_*.parquet seed
that warehouse.py reads (that seed is refreshed by a process outside this
repo, in the shared ~/Downloads/code/python_files/cache_seed dir — see
warehouse.py's SEED constant). --source runs the scanner scripts; it does not
refresh the DuckDB seed data itself.

Stages:
  --source MARKET      full_{market}_market_scan.py  (us|indian|japan|korea|european)
  --validate           data_quality.py                (subprocess; owns its own exit code)
  --process             warehouse.build()               (rebuild DuckDB views)
  --analyze MARKET      DVM views + accumulation_screener, for one warehouse market code
  --graphics MARKET     dashboard.py's render_html (chart-augmented) -> HTML file
  --all MARKET          process -> validate -> analyze -> graphics, in order
                        (skips --source: it's slow/network-heavy, opt in explicitly)

Usage:
  python pipeline.py --process
  python pipeline.py --validate
  python pipeline.py --analyze US
  python pipeline.py --graphics US
  python pipeline.py --all US
  python pipeline.py --source indian
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

MARKET_SCRIPTS = {
    "us": "full_us_market_scan.py",
    "indian": "full_indian_market_scan.py",
    "japan": "full_japan_market_scan.py",
    "korea": "full_korea_market_scan.py",
    "european": "full_european_market_scan.py",
}


def stage_source(market: str, extra_args: list[str] | None = None) -> bool:
    script = MARKET_SCRIPTS.get(market)
    if not script:
        print(f"  [source] unknown market '{market}'; choices: {sorted(MARKET_SCRIPTS)}", file=sys.stderr)
        return False
    print(f"  [source] running {script} ...")
    r = subprocess.run([sys.executable, os.path.join(HERE, script), *(extra_args or [])])
    return r.returncode == 0


def stage_validate(extra_args: list[str] | None = None) -> bool:
    print("  [validate] running data_quality.py ...")
    r = subprocess.run([sys.executable, os.path.join(HERE, "data_quality.py"), *(extra_args or [])])
    return r.returncode == 0


def stage_process() -> bool:
    import duckdb
    import warehouse
    print("  [process] rebuilding warehouse views ...")
    con = duckdb.connect(warehouse.DB)
    try:
        warehouse.build(con)
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        print(f"  [process] views ready: {tables}")
        return True
    finally:
        con.close()


def stage_analyze(market: str) -> bool:
    import duckdb
    import warehouse
    print(f"  [analyze] {market}: DVM + accumulation screen ...")
    con = duckdb.connect(warehouse.DB)
    try:
        warehouse.build(con)
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if "dvm_global" in tables:
            df = con.execute(
                "SELECT count(*) scored, round(avg(M),1) avg_M FROM dvm_global WHERE market = ?",
                [market],
            ).df()
            print(f"  [analyze] DVM momentum ({market}):\n{df.to_string(index=False)}")
        else:
            print("  [analyze] dvm_global not built locally — skipping momentum summary")
        if "dvm_composite" in tables:
            df = con.execute(
                "SELECT code, label, count(*) n FROM dvm_composite WHERE market = ? GROUP BY 1,2 ORDER BY n DESC",
                [market],
            ).df()
            print(f"  [analyze] screener.in-style classification ({market}):\n{df.to_string(index=False)}")
        else:
            print("  [analyze] dvm_composite not built locally — skipping classification summary")
    finally:
        con.close()

    try:
        import accumulation_screener as acc
        panel = acc.current_screen(market)
        print(f"  [analyze] accumulation screen ({market}): {len(panel)} tickers scored")
    except Exception as e:                          # noqa: BLE001
        print(f"  [analyze] accumulation screen skipped: {e}")

    return True


def stage_live(market: str, limit: int | None, pause: float) -> bool:
    """Live fundamentals backfill/refresh via trendlyne_session/screener_session
    (see live_fundamentals.py) — currently only meaningful for IN (India), the
    one market both sources cover. India isn't in the ohlc warehouse view at all
    (the 19-market cleaned_long_*.parquet seed doesn't include it), so the
    ticker list instead comes from the `companies` industry/peer view, stripping
    the yfinance .NS/.BO suffix Trendlyne/Screener.in don't use."""
    import duckdb
    import warehouse
    import live_fundamentals

    con = duckdb.connect(warehouse.DB)
    try:
        warehouse.build(con)
        if market == "IN":
            q = "SELECT DISTINCT ticker FROM companies WHERE country = 'India' ORDER BY ticker"
            if limit:
                q += f" LIMIT {int(limit)}"
            raw = [r[0] for r in con.execute(q).fetchall()]
            tickers = sorted({t.split(".")[0] for t in raw if t})
            if limit:
                tickers = tickers[:limit]
        else:
            q = "SELECT DISTINCT ticker FROM ohlc WHERE market = ? ORDER BY ticker"
            if limit:
                q += f" LIMIT {int(limit)}"
            tickers = [r[0] for r in con.execute(q, [market]).fetchall()]
    finally:
        con.close()

    if not tickers:
        print(f"  [live] no tickers found for market={market}")
        return False

    print(f"  [live] fetching {len(tickers)} tickers for {market} (pause={pause}s)...")
    results = live_fundamentals.fetch_batch(tickers, market=market, pause=pause)
    n_ok = sum(1 for r in results.values() if r.get("source"))
    print(f"  [live] {n_ok}/{len(results)} tickers got at least one live source")
    return n_ok > 0


def stage_graphics(market: str, out: str | None = None) -> bool:
    import dashboard
    print(f"  [graphics] rendering dashboard for {market} ...")
    sections = dashboard._query_all(include_accum=True)
    html = dashboard.render_html(sections)
    out = out or os.path.join(HERE, f"pipeline_dashboard_{market}.html")
    open(out, "w").write(html)
    print(f"  [graphics] wrote {out} ({len(html)//1024} KB)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", metavar="MARKET", help="one of: " + ", ".join(sorted(MARKET_SCRIPTS)))
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--process", action="store_true")
    ap.add_argument("--analyze", metavar="MARKET", help="warehouse market code, e.g. US, JP, IN")
    ap.add_argument("--graphics", metavar="MARKET")
    ap.add_argument("--live", metavar="MARKET", help="live Trendlyne/Screener.in fundamentals backfill (IN only, currently)")
    ap.add_argument("--limit", type=int, help="cap tickers for --live (e.g. for a quick test run)")
    ap.add_argument("--pause", type=float, default=1.5, help="seconds between --live requests (politeness)")
    ap.add_argument("--all", metavar="MARKET", help="process -> validate -> analyze -> graphics")
    args = ap.parse_args()

    if not any([args.source, args.validate, args.process, args.analyze, args.graphics, args.live, args.all]):
        ap.print_help()
        return

    results: dict[str, bool] = {}

    if args.source:
        results["source"] = stage_source(args.source)
    if args.process:
        results["process"] = stage_process()
    if args.validate:
        results["validate"] = stage_validate()
    if args.analyze:
        results["analyze"] = stage_analyze(args.analyze.upper())
    if args.live:
        results["live"] = stage_live(args.live.upper(), args.limit, args.pause)
    if args.graphics:
        results["graphics"] = stage_graphics(args.graphics.upper())

    if args.all:
        market = args.all.upper()
        results["process"] = stage_process()
        results["validate"] = stage_validate()
        results["analyze"] = stage_analyze(market)
        results["graphics"] = stage_graphics(market)

    print("\n=== PIPELINE SUMMARY ===")
    for stage, ok in results.items():
        print(f"  {stage:10} {'PASS' if ok else 'FAIL'}")
    sys.exit(1 if not all(results.values()) else 0)


if __name__ == "__main__":
    main()
