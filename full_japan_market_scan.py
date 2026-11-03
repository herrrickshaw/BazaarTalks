# full_japan_market_scan.py
# =========================
# Tokyo Stock Exchange (TSE) universe scanner using kabupy + yfinance.
#
# Pipeline:
#   Stage 1 — Fetch TSE Prime / Standard equity universe via kabupy (+ JPX CSV fallback)
#   Stage 2 — Bulk OHLC download via yfinance (tickers suffixed with .T)
#   Stage 3 — Darvas Box screen + 200-day MA trend on every stock
#   Stage 4 — Piotroski F-Score + Coffee Can on Darvas BREAKOUT candidates only
#   Stage 5 — Save styled Excel workbook with ranked results
#
# Output sheets:
#   All_Stocks      — price summary for every stock scanned
#   Darvas_Signals  — breakout / breakdown alerts ranked by upside
#   Fundamentals    — Piotroski + Coffee Can results for breakout candidates
#   Triple_Hits     — BREAKOUT_BUY + Piotroski ≥ 7 + Coffee Can PASS
#
# Usage:
#   python full_japan_market_scan.py
#   python full_japan_market_scan.py --top 200     # limit to first 200 tickers
#   python full_japan_market_scan.py --no-scans    # Darvas only, skip fundamentals
#   python full_japan_market_scan.py --workers 8
#
# Install:
#   pip install kabupy yfinance pandas openpyxl requests

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path

import pandas as pd

import darvas_core
import requests
import yfinance as yf

try:
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    OPENPYXL_OK = True
except ImportError:
    OPENPYXL_OK = False

# ── Configuration ─────────────────────────────────────────────────────────────
DOWNLOAD_DIR     = Path("./japan_scan")
DOWNLOAD_DIR.mkdir(exist_ok=True)

DARVAS_CONFIRM      = 3
BATCH_SIZE          = 200
SLEEP_BETWEEN       = 1.5
MAX_WORKERS         = 10
PIOTROSKI_STRONG    = 7
MAX_FUND_CANDIDATES = 200
SYMBOL_CACHE_TTL    = 86400   # 24 h

_CACHE_FILE = DOWNLOAD_DIR / ".symbols_cache.json"

# JPX publishes a daily stock master CSV (TSE Prime + Standard + Growth).
# This is the authoritative free source for the full TSE universe.
JPX_MASTER_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/"
    "tvdivq0000001vg2-att/data_j.xls"
)

# ── Symbol universe ───────────────────────────────────────────────────────────

def _load_cache():
    try:
        if _CACHE_FILE.exists():
            data = json.loads(_CACHE_FILE.read_text())
            if time.time() - data.get("ts", 0) < SYMBOL_CACHE_TTL:
                return data.get("tickers", [])
    except Exception:
        pass
    return None


def _save_cache(tickers):
    try:
        _CACHE_FILE.write_text(json.dumps({"ts": time.time(), "tickers": tickers}))
    except Exception:
        pass


def fetch_tse_universe_kabupy() -> list[dict]:
    """Fetch TSE equity universe via kabupy.Jpx().issues (official JPX data)."""
    try:
        import kabupy
        issues = kabupy.Jpx().issues   # list of dicts, ~4000+ records
        result = []
        for s in issues:
            cat = str(s.get("category", ""))
            # Keep Prime and Standard domestic equities; skip ETF/REIT/foreign
            if "内国株式" not in cat:
                continue
            code = str(s.get("security_code", "")).split(".")[0].zfill(4)
            if not code.isdigit() or len(code) != 4:
                continue
            result.append({
                "code":   code,
                "name":   s.get("name", ""),
                "sector": s.get("33_industry_category", s.get("17_industry_category", "—")),
                "market": cat.replace("（内国株式）", ""),
            })
        if result:
            print(f"  kabupy → {len(result)} TSE domestic equities")
        return result
    except ImportError:
        print("  kabupy not installed — falling back to JPX master CSV")
    except Exception as e:
        print(f"  kabupy error: {e} — falling back to JPX master CSV")
    return []


def fetch_tse_universe_jpx() -> list[dict]:
    """Download TSE stock master Excel from JPX (no auth required, free)."""
    print("  Fetching TSE universe from JPX stock master …")
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
        r = requests.get(JPX_MASTER_URL, headers=headers, timeout=30)
        r.raise_for_status()
        df = pd.read_excel(BytesIO(r.content), dtype=str)
        # Expected columns vary by JPX file version; normalise
        df.columns = [c.strip() for c in df.columns]
        code_col   = next((c for c in df.columns if "コード" in c or "Code" in c), None)
        name_col   = next((c for c in df.columns if "銘柄名" in c or "銘柄" in c or "Name" in c.capitalize()), None)
        sector_col = next((c for c in df.columns if "業種" in c or "Sector" in c.capitalize()), None)
        market_col = next((c for c in df.columns if "市場" in c or "Market" in c.capitalize()), None)

        if code_col is None:
            print("  ⚠️  Cannot identify code column in JPX file")
            return []

        # Filter to Prime + Standard markets (equities only)
        if market_col:
            df = df[df[market_col].str.contains("Prime|Standard|プライム|スタンダード", na=False)]

        result = []
        for _, row in df.iterrows():
            code = str(row[code_col]).strip().split(".")[0].zfill(4)
            if not code.isdigit() or len(code) != 4:
                continue
            result.append({
                "code":   code,
                "name":   str(row[name_col]).strip() if name_col else "",
                "sector": str(row[sector_col]).strip() if sector_col else "—",
            })
        return result
    except Exception as e:
        print(f"  ⚠️  JPX fetch error: {e}")
        return []


def build_ticker_universe() -> list[dict]:
    """Return deduplicated TSE ticker list with name/sector, cached for 24 h."""
    cached = _load_cache()
    if cached:
        print(f"  Symbol cache hit: {len(cached)} TSE tickers")
        return cached

    stocks = fetch_tse_universe_kabupy()
    if not stocks:
        stocks = fetch_tse_universe_jpx()

    if not stocks:
        sys.exit("❌  Could not fetch TSE universe from kabupy or JPX. Check network/install.")

    # Deduplicate on code, build yfinance ticker (.T suffix)
    seen = set()
    result = []
    for s in stocks:
        code = s["code"]
        if code in seen:
            continue
        seen.add(code)
        result.append({
            "yf_ticker": f"{code}.T",
            "code":      code,
            "name":      s["name"],
            "sector":    s["sector"],
        })

    _save_cache(result)
    print(f"  → {len(result)} unique TSE equity tickers")
    return result


# ── Bulk OHLC ─────────────────────────────────────────────────────────────────

def bulk_download_ohlc(tickers: list[str], period: str = "3mo") -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    print(f"  Downloading OHLC for {len(tickers)} tickers in {len(batches)} batches …")

    for idx, batch in enumerate(batches, 1):
        print(f"    Batch {idx}/{len(batches)} ({len(batch)} tickers) …", end=" ", flush=True)
        try:
            raw = yf.download(batch, period=period, auto_adjust=True,
                              threads=True, progress=False)
            if raw.empty:
                print("empty")
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                for tkr in batch:
                    try:
                        df = raw.xs(tkr, axis=1, level=1).dropna(how="all")
                        if not df.empty and len(df) >= DARVAS_CONFIRM + 5:
                            result[tkr] = df
                    except KeyError:
                        pass
            else:
                tkr = batch[0]
                if not raw.empty:
                    result[tkr] = raw
            print(f"OK ({sum(1 for t in batch if t in result)} usable)")
        except Exception as e:
            print(f"ERROR — {e}")
        if idx < len(batches):
            time.sleep(SLEEP_BETWEEN)

    return result


# ── Darvas Box ────────────────────────────────────────────────────────────────

def compute_darvas_box(df: pd.DataFrame, confirm: int = DARVAS_CONFIRM) -> dict:
    """Detect Darvas Box from High/Low/Close; prices rounded to whole currency units."""
    if df is None or df.empty or len(df) < confirm + 5:
        return {"signal": "INSUFFICIENT_DATA", "box_top": None, "box_bottom": None}

    highs  = darvas_core.numeric_list(df, "High")
    lows   = darvas_core.numeric_list(df, "Low")
    closes = darvas_core.numeric_list(df, "Close")

    current = closes[-1]
    box_top, box_bottom = darvas_core.find_box(highs[:-1], lows[:-1], confirm)

    if box_top is None:
        return {"signal": "NO_BOX", "box_top": None, "box_bottom": None, "current_price": current}
    if box_bottom is None:
        return {"signal": "NO_BOX", "box_top": round(box_top, 0), "box_bottom": None,
                "current_price": round(current, 0)}

    signal = darvas_core.classify(current, box_top, box_bottom)
    _, upside, pos = darvas_core.box_geometry(current, box_top, box_bottom)

    return {
        "signal":       signal,
        "box_top":      round(box_top,    0),
        "box_bottom":   round(box_bottom, 0),
        "current_price":round(current,    0),
        "upside_pct":   round(upside, 2),
        "pos_in_box":   round(pos, 1),
        "data_points":  len(closes),
    }


# ── Fundamental helpers (shared; see stock_utils.py) ──────────────────────────
from scan_common import (OHLC_SIGNALS, asia_fundamental_scan, last_two_closes, ma200_and_distance,
                         make_write_sheet, run_fundamental_pool, trend_signal)


# ── Fundamental scan ─────────────────────────────────────────────────────────

def fundamental_scan(yf_ticker: str) -> dict:
    """Piotroski F-Score + Japan-adapted Coffee Can in a single yfinance call."""
    result = {"yf_ticker": yf_ticker, "f_score": None, "cc_qualifies": "FAIL", "error": None}
    return asia_fundamental_scan(result, yf, yf_ticker, PIOTROSKI_STRONG,
                                 mcap_key="market_cap_b_jpy", mcap_div=1e9)


# ── Excel styling ─────────────────────────────────────────────────────────────

def style_sheet(ws):
    if not OPENPYXL_OK:
        return
    fill_hdr  = PatternFill(start_color="1A3A5C", end_color="1A3A5C", fill_type="solid")
    fill_alt  = PatternFill(start_color="F0F4F8", end_color="F0F4F8", fill_type="solid")
    font_hdr  = Font(name="Calibri", size=11, bold=True,  color="FFFFFF")
    font_body = Font(name="Calibri", size=11, bold=False, color="000000")
    thin = Border(
        left=Side(style="thin", color="CBD5E0"), right=Side(style="thin", color="CBD5E0"),
        top=Side(style="thin", color="CBD5E0"),  bottom=Side(style="thin", color="CBD5E0"),
    )
    for col_idx in range(1, ws.max_column + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = font_hdr; cell.fill = fill_hdr; cell.border = thin
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for row_idx in range(2, ws.max_row + 1):
        for col_idx in range(1, ws.max_column + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.font = font_body; cell.border = thin
            if row_idx % 2 == 1:
                cell.fill = fill_alt
            hdr = str(ws.cell(row=1, column=col_idx).value or "").upper()
            val = cell.value
            if isinstance(val, (int, float)):
                cell.alignment = Alignment(horizontal="right", vertical="center")
                if any(k in hdr for k in ["%", "CAGR", "ROCE", "YIELD"]):
                    cell.number_format = '0.00"%"'
                elif any(k in hdr for k in ["CAP", "LTP", "BOX", "PRICE", "200"]):
                    cell.number_format = '#,##0'
                else:
                    cell.number_format = '0.00'
            else:
                cell.alignment = Alignment(horizontal="left", vertical="center")

    ws.row_dimensions[1].height = 28
    for r in range(2, ws.max_row + 1):
        ws.row_dimensions[r].height = 20
    for col in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        ws.column_dimensions[get_column_letter(col[0].column)].width = max(max_len + 4, 12)


# ── Main ─────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Full TSE scanner — Darvas + Piotroski + Coffee Can")
    parser.add_argument("--top",      type=int, default=0,     help="Limit to first N tickers")
    parser.add_argument("--no-scans", action="store_true",     help="Skip fundamental scans")
    parser.add_argument("--workers",  type=int, default=MAX_WORKERS, help="Parallel threads")
    return parser.parse_args(argv)


def _technical_row(yf_tkr: str, df: pd.DataFrame, info: dict) -> dict:
    """All_Stocks row for one ticker: price (JPY), 200-MA trend and Darvas box."""
    darv    = compute_darvas_box(df)
    closes, ltp, _prev, chg_pct = last_two_closes(df, 0)
    ma200, dist_ma = ma200_and_distance(closes, ltp, 0)
    trend   = trend_signal(dist_ma, "Insufficient History")
    return {
        "YF_Ticker":          yf_tkr,
        "Code":               info.get("code", yf_tkr.replace(".T", "")),
        "Name":               info.get("name", ""),
        "Sector":             info.get("sector", "—"),
        "LTP_JPY":            ltp,
        "Change%":            chg_pct,
        "200_Day_MA":         ma200,
        "Distance_to_200MA%": dist_ma,
        "Trend_Signal":       trend,
        "Darvas_Signal":      darv.get("signal"),
        "Box_Top":            darv.get("box_top"),
        "Box_Bottom":         darv.get("box_bottom"),
        "Upside_to_Top%":     darv.get("upside_pct"),
        "Position_in_Box%":   darv.get("pos_in_box"),
        "Data_Points":        darv.get("data_points"),
    }


def _technical_stage(ohlc: dict, meta: dict) -> tuple:
    """(all_rows, darvas_rows, breakout_tickers) over every ticker with OHLC data."""
    all_rows, darvas_rows, breakout_tickers = [], [], []
    for yf_tkr, df in ohlc.items():
        row = _technical_row(yf_tkr, df, meta.get(yf_tkr, {}))
        all_rows.append(row)
        if row["Darvas_Signal"] in OHLC_SIGNALS:
            darvas_rows.append(row.copy())
        if row["Darvas_Signal"] == "BREAKOUT_BUY":
            breakout_tickers.append(yf_tkr)
    return all_rows, darvas_rows, breakout_tickers


def _cap_candidates(breakout_tickers: list, darvas_rows: list) -> list:
    """Keep only the MAX_FUND_CANDIDATES freshest breakouts (smallest distance to the box top)."""
    def _upside(tkr):
        row = next((r for r in darvas_rows if r["YF_Ticker"] == tkr), None)
        return abs(row.get("Upside_to_Top%") or 999)
    if len(breakout_tickers) > MAX_FUND_CANDIDATES:
        breakout_tickers = sorted(breakout_tickers, key=_upside)[:MAX_FUND_CANDIDATES]
        print(f"  (capped to {MAX_FUND_CANDIDATES} freshest breakouts)")
    return breakout_tickers


def _fund_row(tkr: str, tech: dict, res: dict) -> dict:
    """Fundamentals row: technical columns plus the scan result."""
    return {
        "YF_Ticker":       tkr,
        "Code":            tech.get("Code"),
        "Name":            tech.get("Name"),
        "Sector":          tech.get("Sector"),
        "LTP_JPY":         tech.get("LTP_JPY"),
        "Change%":         tech.get("Change%"),
        "Darvas_Signal":   tech.get("Darvas_Signal"),
        "Upside_to_Top%":  tech.get("Upside_to_Top%"),
        "200_Day_MA":      tech.get("200_Day_MA"),
        "Piotroski_Score": res.get("f_score"),
        "Piotroski_Strong":("YES" if res.get("f_strong") else "NO"),
        "CoffeeCan":       res.get("cc_qualifies", "FAIL"),
        "CC_Score":        res.get("cc_score", "—"),
        "Rev_CAGR%":       res.get("cc_rev_cagr"),
        "ROCE_Avg%":       res.get("cc_roce_avg"),
        "Debt_Equity":     res.get("cc_debt_equity"),
        "MCap_B_JPY":      res.get("market_cap_b_jpy"),
        "Error":           res.get("error", ""),
    }


def _fundamental_stage(breakout_tickers: list, darvas_rows: list, workers: int) -> tuple:
    """(fund_rows, triple_rows) for the capped breakout candidates."""
    fund_rows, triple_rows = [], []
    print(f"\nStage 4 — Fundamental scans ({len(breakout_tickers)} candidates, "
          f"{workers} workers) …")

    def handle(tkr, res, done):
        tech = next((r for r in darvas_rows if r["YF_Ticker"] == tkr), {})
        fund_row = _fund_row(tkr, tech, res)
        fund_rows.append(fund_row)
        if res.get("f_strong") and res.get("cc_qualifies") == "PASS":
            triple_rows.append(fund_row.copy())
        if done % 20 == 0 or done == len(breakout_tickers):
            print(f"    {done}/{len(breakout_tickers)} done  "
                  f"(triple hits: {len(triple_rows)})")

    run_fundamental_pool([(tkr, (tkr,)) for tkr in breakout_tickers], fundamental_scan, workers, handle)
    return fund_rows, triple_rows


def _save_workbook(all_rows, darvas_rows, fund_rows, triple_rows):
    """Write the four-sheet Excel report and return its path."""
    date_str = datetime.today().strftime("%Y%m%d_%H%M")
    path = DOWNLOAD_DIR / f"japan_market_scan_{date_str}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        write_sheet = make_write_sheet(writer, style=style_sheet if OPENPYXL_OK else None)
        write_sheet(all_rows,    "All_Stocks",    sort_col="Change%")
        write_sheet(darvas_rows, "Darvas_Signals", sort_col="Upside_to_Top%")
        write_sheet(fund_rows,   "Fundamentals",  sort_col="Piotroski_Score")
        write_sheet(triple_rows, "Triple_Hits",   sort_col="Piotroski_Score")
    return path


def _print_summary(all_rows, breakout_tickers, fund_rows, triple_rows, path) -> None:
    print(f"\n{'='*60}")
    print(f"  SCAN COMPLETE — {datetime.now().strftime('%d %b %Y  %H:%M:%S')}")
    print(f"  TSE stocks scanned:    {len(all_rows)}")
    print(f"  Darvas Breakouts:      {len(breakout_tickers)}")
    print(f"  Fundamentals scanned:  {len(fund_rows)}")
    print(f"  ★ TRIPLE HITS:         {len(triple_rows)}")
    if triple_rows:
        print("\n  Triple-hit stocks:")
        for r in sorted(triple_rows, key=lambda x: x.get("Piotroski_Score") or 0, reverse=True):
            print(f"    [{r['Code']}] {r['Name']:<30}  F={r['Piotroski_Score']}/9  "
                  f"CC={r['CC_Score']}  ¥{r['LTP_JPY']:,.0f}")
    else:
        print("\n  No Triple Hit stocks found today.")
    print(f"\n  📊 Excel saved → {path}\n")
    print(f"{'='*60}\n")


def main():
    args = _parse_args()

    print(f"\n{'#'*60}")
    print("  FULL JAPAN (TSE) MARKET SCAN")
    print(f"  Started: {datetime.now().strftime('%d %b %Y  %H:%M:%S')}")
    print(f"{'#'*60}\n")

    # Stage 1 — Universe
    print("Stage 1 — Building TSE equity universe …")
    universe = build_ticker_universe()
    meta = {s["yf_ticker"]: s for s in universe}
    all_yf_tickers = [s["yf_ticker"] for s in universe]
    if args.top:
        all_yf_tickers = all_yf_tickers[:args.top]
        print(f"  (limited to first {args.top} tickers)")

    # Stage 2 — Bulk OHLC
    print(f"\nStage 2 — Bulk OHLC download ({len(all_yf_tickers)} tickers) …")
    ohlc = bulk_download_ohlc(all_yf_tickers, period="3mo")
    print(f"  → {len(ohlc)} tickers with usable OHLC data")

    # Stage 3 — Darvas Box + 200-day MA
    print("\nStage 3 — Darvas Box & 200-day MA screen …")
    all_rows, darvas_rows, breakout_tickers = _technical_stage(ohlc, meta)
    print(f"  Breakout BUY:   {len(breakout_tickers)}")
    print(f"  Breakdown SELL: {sum(1 for r in darvas_rows if r['Darvas_Signal'] == 'BREAKDOWN_SELL')}")
    print(f"  In Box:         {len(all_rows) - len(darvas_rows)}")

    # Stage 4 — Fundamentals on breakout candidates
    fund_rows, triple_rows = [], []
    if not args.no_scans and breakout_tickers:
        breakout_tickers = _cap_candidates(breakout_tickers, darvas_rows)
        fund_rows, triple_rows = _fundamental_stage(breakout_tickers, darvas_rows, args.workers)
    else:
        print("\nStage 4 — Skipped")

    # Stage 5 — Excel
    print("\nStage 5 — Saving Excel workbook …")
    path = _save_workbook(all_rows, darvas_rows, fund_rows, triple_rows)
    _print_summary(all_rows, breakout_tickers, fund_rows, triple_rows, path)


if __name__ == "__main__":
    main()
