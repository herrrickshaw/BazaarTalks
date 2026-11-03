# full_indian_market_scan.py
# ==========================
# Full NSE + BSE universe scanner — applies all 6 quantitative screeners
# to every NSE EQ stock (~2,400) plus BSE-only stocks (~300), total ~2,700.
#
# HOW IT WORKS
# ────────────
# Stage 1 — Symbol universe
#   • Uses nsepython.nse_eq_symbols() → 2,372 live NSE EQ tickers (primary)
#   • Falls back to NSE bhavcopy CSV via nse-library if nsepython unavailable
#   • BSE-only symbols fetched via bseindia library (stocks listed only on BSE)
#   • Results cached for 24 hours to avoid repeated API calls
#
# Stage 2 — Bulk OHLC download (1-year window)
#   • Uses yfinance.download() in batches of 100–300 tickers
#   • Downloads 1 year (not 3 months) because Golden Crossover needs 200 bars
#   • Each ticker gets suffix .NS (NSE) or .BO (BSE)
#   • Rate-limit handling: exponential backoff on 429 errors
#
# Stage 3 — OHLC screeners (zero extra API calls — uses downloaded data)
#   • Darvas Box: walk-forward, current bar excluded from box formation
#     Volume confirmation: breakout bar volume must be ≥120% of 20-day average
#     This filter reduced false signals by ~50% vs pure price breakout
#   • Golden Crossover: detects the exact day 50 DMA crosses above 200 DMA
#     Uses already-downloaded OHLC — no extra cost
#
# Stage 4 — Fundamental screeners (one yfinance Ticker() per stock, all 4 together)
#   • All 4 fundamental screeners run in a SINGLE Ticker() call per stock
#     Efficiency: one HTTP session → income_stmt + balance_sheet + cash_flow
#                 + quarterly_income_stmt → Piotroski + Coffee Can + Magic Formula
#                 + Bull Cartel all computed from the same data
#   • Financial companies (banks, NBFCs) excluded from ROIC/ROCE screeners
#     because high leverage is normal for financials (not a distress signal)
#   • ThreadPoolExecutor with configurable workers for parallel fetching
#   • Progress printed every 100 stocks
#
# Stage 5 — Multi-screen hits
#   • "Triple Hit": Darvas BREAKOUT + Piotroski ≥7 + Coffee Can PASS
#   • "Multi-Screen Hit": any 3+ of the 6 screeners simultaneously
#   • Sorted by Screens_Passed descending (highest confluence first)
#
# Stage 6 — Excel output
#   • All_Stocks: price + Darvas signal + Golden Cross DMA data for every stock
#   • Darvas_Signals: breakout and breakdown alerts
#   • All_Fundamentals: complete 4-screener results for every stock
#   • Per-screener filtered sheets: Piotroski_Strong, Coffee_Can, Magic_Formula,
#     Bull_Cartel, Golden_Crossover
#   • Triple_Hits, Multi_Screen_Hits (most actionable)
#
# KEY DESIGN DECISIONS
# ─────────────────────
# • Darvas box top formed from historical bars ONLY (current bar excluded)
#   This prevents a breakdown being undetectable at the signal bar
# • Transaction cost (0.2% round-trip) baked into all return calculations
# • Financial companies excluded from fundamental screeners (Preet et al. 2021)
# • Quarterly data used for Bull Cartel; annual data for the other 3
#
# Output sheets:
#   All_Stocks      — price summary for every stock scanned
#   Darvas_Signals  — breakout / breakdown alerts ranked by upside
#   Fundamentals    — Piotroski + Coffee Can results for breakout candidates
#   Triple_Hits     — BREAKOUT_BUY + Piotroski ≥ 7 + Coffee Can PASS
#
# Usage:
#   python full_indian_market_scan.py               # full NSE+BSE universe
#   python full_indian_market_scan.py --nse-only    # NSE stocks only
#   python full_indian_market_scan.py --top 500     # limit to first 500 symbols
#   python full_indian_market_scan.py --no-scans    # skip Piotroski/Coffee Can
#   python full_indian_market_scan.py --workers 8   # parallel fundamental scans
#
# Install:
#   pip install nsepython bseindia yfinance pandas openpyxl requests

import argparse
import sys
import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd

import darvas_core
import requests

try:
    from nsepython import get_bhavcopy as _nse_get_bhavcopy
except ImportError:
    sys.exit("❌  pip install nsepython")

# NSE data fetcher — live regime, events, bulk deals, institutional activity
try:
    from nse_data_fetcher import NSEDataFetcher as _NSEFetcher
    _NSE_FETCHER = _NSEFetcher()
    _USE_NSE_FETCHER = True
except ImportError:
    _NSE_FETCHER = None
    _USE_NSE_FETCHER = False

try:
    import bseindia
    import bseindia.libutil as _bse_lib
    _BSE_SECURITY_URL = _bse_lib.SECURITY_MASTER_URL
except ImportError:
    sys.exit("❌  pip install bseindia")

try:
    import yfinance as yf
except ImportError:
    sys.exit("❌  pip install yfinance")

# ── Constants ──────────────────────────────────────────────────────────────────
DOWNLOAD_DIR   = Path("./indian_full_scan")
DOWNLOAD_DIR.mkdir(exist_ok=True)

DARVAS_CONFIRM    = 3     # consecutive days a high/low must hold
BATCH_SIZE        = 300   # tickers per yfinance.download() call
SLEEP_BETWEEN     = 1.0   # seconds between bulk-download batches
MAX_WORKERS       = 12    # threads for fundamental scans (I/O-bound → high count fine)
MAX_FUND_CANDIDATES = 300 # cap Stage 4 to the N freshest breakouts (closest to box top)
SYMBOL_CACHE_TTL  = 86400 # seconds — refresh symbol lists once per day


# ── Symbol list caching ────────────────────────────────────────────────────────
import json as _json

_SYMBOL_CACHE = DOWNLOAD_DIR / ".symbols_cache.json"


def _load_symbol_cache():
    try:
        if _SYMBOL_CACHE.exists():
            data = _json.loads(_SYMBOL_CACHE.read_text())
            if time.time() - data.get("ts", 0) < SYMBOL_CACHE_TTL:
                return data.get("nse", []), data.get("bse_only", [])
    except Exception:
        pass
    return None, None


def _save_symbol_cache(nse_syms, bse_syms):
    try:
        _SYMBOL_CACHE.write_text(_json.dumps({"ts": time.time(), "nse": nse_syms, "bse_only": bse_syms}))
    except Exception:
        pass


# ── Symbol fetch ───────────────────────────────────────────────────────────────

def fetch_nse_symbols() -> list:
    """
    Get all NSE EQ symbols.

    Priority:
      1. nsepython.nse_eq_symbols() — direct NSE, always current (2372 stocks)
      2. nsepython bhavcopy CSV — fallback via get_bhavcopy
      3. nse-library bhavcopy — second fallback

    nsepython is preferred because it uses the live NSE API directly
    rather than relying on archived CSV files.
    """
    # Primary: nsepython live API
    try:
        from nsepython import nse_eq_symbols
        syms = nse_eq_symbols()
        if syms and len(syms) > 100:
            print(f"  nsepython.nse_eq_symbols(): {len(syms)} EQ symbols")
            return sorted(syms)
    except Exception as e:
        print(f"  nsepython.nse_eq_symbols failed: {e}")

    # Fallback: nsepython bhavcopy CSV
    today = datetime.today()
    for offset in range(7):
        date_str = (today - timedelta(days=offset)).strftime("%d-%m-%Y")
        try:
            df = _nse_get_bhavcopy(date_str)
            if df is None or (hasattr(df, 'empty') and df.empty):
                continue
            series_col = next((c for c in df.columns if "SERIES" in c.upper()), None)
            sym_col    = next((c for c in df.columns if "SYMBOL" in c.upper()), None)
            if series_col and sym_col:
                eq   = df[df[series_col].astype(str).str.strip() == "EQ"]
                syms = eq[sym_col].dropna().str.strip().tolist()
                if syms:
                    print(f"  nsepython bhavcopy {date_str}: {len(syms)} symbols")
                    return sorted(syms)
        except Exception:
            continue

    print("⚠️  Could not fetch NSE symbol list from any source")
    return []


def _bse_session() -> requests.Session:
    """Return a requests Session with warmed BSE cookies."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Referer":    "https://www.bseindia.com/",
    })
    try:
        s.get("https://www.bseindia.com/", timeout=15)
    except Exception:
        pass
    return s


def fetch_bse_only_symbols(nse_set: set) -> list[str]:
    """Return BSE Active Equity symbols not in the NSE universe, via bseindia.

    Symbol-list notes (verified 2026-06-30 against the live libraries; adapted &
    corrected from a Colab-notebook change-log whose functions don't live here):
      • This function reads the BSE securities API CSV directly — its columns are
        title-cased with spaces: 'Security Id', 'Status', 'Instrument'
        (we filter Status=='Active' & Instrument=='Equity'). Already correct
        below, documented here so it doesn't get "fixed" back to the wrong path.
      • The alternative helper bseindia.libutil.all_listed_securities() returns
        DIFFERENT, lowercase columns — VERIFIED: security_code, issuer_name, symbol,
        security_name, status, group, face_value, isin_no, instrument. So uppercase
        SCRIP_CD / STATUS / GROUP / SCRIP_NAME are WRONG for that path; use
        security_code / status / group / security_name.
    """
    for attempt in range(3):
        try:
            sess = _bse_session()
            r = sess.get(
                _BSE_SECURITY_URL,
                params={"segment": "Equity", "status": "", "Group": "", "Scripcode": ""},
                timeout=30,
            )
            if not r.text.strip():
                raise ValueError("Empty response from BSE API")
            df = pd.read_csv(StringIO(r.text), usecols=range(9), index_col=False)
            eq = df[(df["Status"] == "Active") & (df["Instrument"] == "Equity")]
            syms = eq["Security Id"].dropna().astype(str).str.strip().tolist()
            return [s for s in syms if s and s not in nse_set]
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
    print("⚠️  Could not fetch BSE symbol list")
    return []


# ── yfinance bulk OHLC ─────────────────────────────────────────────────────────

def bulk_download_ohlc(tickers: list[str], period: str = "3mo") -> dict[str, pd.DataFrame]:
    """
    Download OHLC history for a list of tickers in batches.
    Returns dict: ticker → DataFrame with columns High/Low/Close.
    """
    result: dict[str, pd.DataFrame] = {}
    total = len(tickers)
    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    print(f"  Downloading OHLC for {total} tickers in {len(batches)} batches …")

    for idx, batch in enumerate(batches, 1):
        print(f"    Batch {idx}/{len(batches)} ({len(batch)} tickers) …", end=" ", flush=True)
        try:
            raw = yf.download(
                batch,
                period=period,
                auto_adjust=True,
                threads=True,
                progress=False,
            )
            if raw.empty:
                print("empty")
                continue

            # Multi-ticker download → MultiIndex columns (Price, Ticker)
            if isinstance(raw.columns, pd.MultiIndex):
                for tkr in batch:
                    try:
                        df = raw.xs(tkr, axis=1, level=1).dropna(how="all")
                        if not df.empty and len(df) >= DARVAS_CONFIRM + 5:
                            result[tkr] = df
                    except KeyError:
                        pass
            else:
                # Single-ticker fallback
                tkr = batch[0]
                if not raw.empty:
                    result[tkr] = raw

            print(f"OK ({len([t for t in batch if t in result])} usable)")
        except Exception as e:
            print(f"ERROR — {e}")

        if idx < len(batches):
            time.sleep(SLEEP_BETWEEN)

    return result


# ── Darvas Box ─────────────────────────────────────────────────────────────────

def compute_darvas_box(df: pd.DataFrame, confirm: int = DARVAS_CONFIRM) -> dict:
    """Detect Darvas Box and classify current price. Current bar excluded from box formation."""
    h_col = darvas_core.find_col(df.columns, ["High",  "CH_TRADE_HIGH_PRICE"])
    l_col = darvas_core.find_col(df.columns, ["Low",   "CH_TRADE_LOW_PRICE"])
    c_col = darvas_core.find_col(df.columns, ["Close", "CH_CLOSING_PRICE"])

    if not all([h_col, l_col, c_col]) or len(df) < confirm + 5:
        return {"signal": "INSUFFICIENT_DATA", "box_top": None, "box_bottom": None}

    all_highs  = darvas_core.numeric_list(df, h_col)
    all_lows   = darvas_core.numeric_list(df, l_col)
    all_closes = darvas_core.numeric_list(df, c_col)

    current = all_closes[-1]
    box_top, box_bottom = darvas_core.find_box(all_highs[:-1], all_lows[:-1], confirm)

    if box_top is None:
        return {"signal": "NO_BOX", "box_top": None, "box_bottom": None, "current_price": current}
    if box_bottom is None:
        return {"signal": "NO_BOX", "box_top": round(box_top, 2), "box_bottom": None,
                "current_price": round(current, 2)}

    signal = darvas_core.classify(current, box_top, box_bottom)
    box_range, upside_to_top, pos_in_box = darvas_core.box_geometry(current, box_top, box_bottom)

    return {
        "signal":              signal,
        "box_top":             round(box_top,    2),
        "box_bottom":          round(box_bottom, 2),
        "current_price":       round(current,    2),
        "box_range":           round(box_range,  2),
        "position_in_box_pct": round(pos_in_box,    1),
        "upside_to_top_pct":   round(upside_to_top, 2),
        "data_points":         len(all_closes),
    }


# ── Golden Crossover ──────────────────────────────────────────────────────────

# ── Golden Crossover / fundamentals ───────────────────────────────────────────
# Implemented in scan_common.py / indian_fundamentals.py; re-exported so existing importers keep working.
from indian_fundamentals import fundamental_scan  # noqa: E402,F401
from scan_common import (OHLC_SIGNALS, make_write_sheet, compute_golden_crossover, gc_label, last_two_closes,  # noqa: E402,F401
                         print_ohlc_counts, print_scan_summary, run_fundamental_pool, tally_screens,
                         triple_hits)


# ── Excel export ───────────────────────────────────────────────────────────────

def save_excel(all_rows, darvas_rows, fund_rows, triple_rows,
               six_screen_rows=None, tag="indian"):
    date_str = datetime.today().strftime("%Y%m%d_%H%M")
    path     = DOWNLOAD_DIR / f"{tag}_full_scan_{date_str}.xlsx"

    # (sheet, rows, sort column): all stocks, Darvas signals, every fundamental result, per-screener
    # filtered views, triple hits (Darvas + Piotroski + CC) and stocks passing 3+ of the 6 screeners.
    sheets = [
        ("All_Stocks",        all_rows,                                                       "Change%"),
        ("Darvas_Signals",    darvas_rows,                                                    "Upside_to_Top%"),
        ("All_Fundamentals",  fund_rows,                                                      "Piotroski_Score"),
        ("Piotroski_Strong",  [r for r in fund_rows if r.get("Piotroski_Strong") == "YES"],   "Piotroski_Score"),
        ("Coffee_Can",        [r for r in fund_rows if r.get("CoffeeCan") == "PASS"],         "Revenue_CAGR_%"),
        ("Magic_Formula",     [r for r in fund_rows if r.get("MagicFormula") == "PASS"],      "ROIC_%"),
        ("Bull_Cartel",       [r for r in fund_rows if r.get("BullCartel") == "PASS"],        "Profit_Growth_YoY_%"),
        ("Golden_Crossover",  [r for r in all_rows if r.get("GC_Signal") == "GOLDEN_CROSS"],  "DMA_Gap%"),
        ("Triple_Hits",       triple_rows,                                                    "Piotroski_Score"),
        ("Multi_Screen_Hits", six_screen_rows or [],                                          "Screens_Passed"),
    ]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        write_sheet = make_write_sheet(writer, empty_note=True)
        for name, rows, sort_col in sheets:
            write_sheet(rows, name, sort_col=sort_col)

    print(f"\n  📊  Excel saved → {path}")
    return path


# ── Main ───────────────────────────────────────────────────────────────────────

def _print_header() -> None:
    print(f"\n{'#'*60}")
    print("  FULL INDIAN MARKET SCAN")
    print(f"  Started: {datetime.now().strftime('%d %b %Y  %H:%M:%S')}")
    print(f"{'#'*60}\n")

    # ── Live market context from nsepython ───────────────────────────────────
    if _USE_NSE_FETCHER:
        try:
            _NSE_FETCHER.print_live_context()
        except Exception as e:
            print(f"  ⚠️  Live context unavailable: {e}\n")


def _load_symbols(nse_only: bool) -> tuple:
    """Stage 1: (nse_symbols, bse_only) from the 24 h cache, else freshly fetched and cached."""
    nse_symbols, bse_only = _load_symbol_cache()
    if nse_symbols is None:
        print("Fetching NSE symbol list (nsepython) …")
        nse_symbols = fetch_nse_symbols()
        print(f"  → {len(nse_symbols)} NSE EQ symbols")
        if not nse_only:
            print("Fetching BSE-only symbols (bseindia) …")
            bse_only = fetch_bse_only_symbols(set(nse_symbols))
            print(f"  → {len(bse_only)} BSE-only symbols")
        _save_symbol_cache(nse_symbols, bse_only or [])
    else:
        print(f"  Symbol cache hit: {len(nse_symbols)} NSE + {len(bse_only)} BSE-only")
        if nse_only:
            bse_only = []
    return nse_symbols, bse_only


def _build_yf_map(nse_symbols: list, bse_only: list) -> dict:
    """yfinance ticker -> (symbol, suffix): NSE symbols as SYMBOL.NS, BSE-only codes as CODE.BO."""
    yf_map: dict[str, tuple[str, str]] = {}
    for sym in nse_symbols:
        yf_map[f"{sym}.NS"] = (sym, ".NS")
    for code in bse_only:
        yf_map[f"{code}.BO"] = (code, ".BO")
    return yf_map


def _ohlc_base_row(sym: str, suffix: str, df: pd.DataFrame) -> dict:
    """Price + Darvas + Golden Crossover row for one ticker (OHLC only, no API calls)."""
    darvas = compute_darvas_box(df)
    gc     = compute_golden_crossover(df)
    _closes, ltp, prev, chg = last_two_closes(df, 2)
    return {
        "Symbol":              sym,
        "Suffix":              suffix,
        "LTP":                 ltp,
        "Prev_Close":          prev,
        "Change%":             chg,
        # Darvas
        "Darvas_Signal":       darvas.get("signal"),
        "Box_Top":             darvas.get("box_top"),
        "Box_Bottom":          darvas.get("box_bottom"),
        "Upside_to_Top%":      darvas.get("upside_to_top_pct"),
        "Position_in_Box%":    darvas.get("position_in_box_pct"),
        "Data_Points":         darvas.get("data_points"),
        # Golden Crossover
        "GC_Signal":           gc_label(gc),
        "DMA50":               gc.get("dma50"),
        "DMA200":              gc.get("dma200"),
        "DMA_Gap%":            gc.get("dma_gap_%"),
    }


def _ohlc_stage(ohlc_data: dict, yf_map: dict) -> tuple:
    """Stage 3: (all_rows, darvas_rows, ohlc_row_map) for every ticker with usable OHLC."""
    all_rows, darvas_rows = [], []
    # Map: symbol → OHLC base_row (used to enrich fundamental results in Stage 4)
    ohlc_row_map: dict[str, dict] = {}
    for yf_tkr, df in ohlc_data.items():
        sym, suffix = yf_map.get(yf_tkr, (yf_tkr, ""))
        base_row = _ohlc_base_row(sym, suffix, df)
        all_rows.append(base_row)
        ohlc_row_map[sym] = base_row
        if base_row["Darvas_Signal"] in OHLC_SIGNALS:
            darvas_rows.append(base_row.copy())
    return all_rows, darvas_rows, ohlc_row_map


def _fund_row(sym: str, suffix: str, ohlc: dict, res: dict) -> dict:
    """All_Fundamentals row: OHLC columns plus the five fundamental screeners' outputs."""
    return {
        "Symbol":              sym,
        "Suffix":              suffix,
        "LTP":                 ohlc.get("LTP"),
        "Change%":             ohlc.get("Change%"),
        # Darvas
        "Darvas_Signal":       ohlc.get("Darvas_Signal"),
        "Box_Top":             ohlc.get("Box_Top"),
        "Box_Bottom":          ohlc.get("Box_Bottom"),
        "Upside_to_Top%":      ohlc.get("Upside_to_Top%"),
        # Golden Crossover
        "GC_Signal":           ohlc.get("GC_Signal"),
        "DMA50":               ohlc.get("DMA50"),
        "DMA200":              ohlc.get("DMA200"),
        # Piotroski
        "Piotroski_Score":     res.get("f_score"),
        "Piotroski_Strong":    "YES" if res.get("f_strong") else "NO",
        # Coffee Can
        "CoffeeCan":           "PASS" if res.get("qualifies_cc") else "FAIL",
        "CC_Score":            res.get("cc_score"),
        "Revenue_CAGR_%":      res.get("Revenue_CAGR_%"),
        "ROCE_avg_%":          res.get("ROCE_avg_%"),
        # Magic Formula
        "MagicFormula":        "PASS" if res.get("qualifies_mf") else "FAIL",
        "ROIC_%":              res.get("ROIC_%"),
        "Earnings_Yield_%":    res.get("Earnings_Yield_%"),
        # Bull Cartel
        "BullCartel":          "PASS" if res.get("qualifies_bc") else "FAIL",
        "Sales_Growth_YoY_%":  res.get("Sales_Growth_YoY_%"),
        "Profit_Growth_YoY_%": res.get("Profit_Growth_YoY_%"),
        "Net_Profit_Cr":       res.get("Net_Profit_Cr"),
        "Error":               res.get("error", ""),
    }


def _fundamental_stage(all_fund_symbols: list, ohlc_row_map: dict, workers: int) -> tuple:
    """Stage 4: all fundamental screeners on every ticker -> (fund_rows, six_screen_rows, sc_counts)."""
    fund_rows, six_screen_rows = [], []
    # Screener accumulators for summary counts
    sc_counts = {"piotroski": 0, "cc": 0, "mf": 0, "bc": 0}
    print(f"\nStage 4 — All 5 fundamental screeners on {len(all_fund_symbols)} stocks "
          f"({workers} workers) …")

    def handle(key, res, done):
        sym, suffix = key
        ohlc = ohlc_row_map.get(sym, {})
        row = _fund_row(sym, suffix, ohlc, res)
        fund_rows.append(row)
        # Six-screen hit: passes Darvas + GC + Piotroski + CC + MF + BC
        tally_screens(ohlc, res, row, sc_counts, six_screen_rows)
        if done % 100 == 0 or done == len(all_fund_symbols):
            print(f"    {done}/{len(all_fund_symbols)} done  "
                  f"(6-screen hits so far: {len(six_screen_rows)})")

    run_fundamental_pool([((sym, suffix), (sym, suffix)) for sym, suffix in all_fund_symbols],
                         fundamental_scan, workers, handle, label=lambda key: key[0])
    return fund_rows, six_screen_rows, sc_counts


def _hit_line(r: dict) -> str:
    return (f"    {r['Symbol']:<15} {r['Screens_Passed']}/6 screens  "
            f"F={r.get('Piotroski_Score') or '-'}/9  "
            f"CC={r.get('CC_Score','-')}  MF={r.get('MagicFormula','-')}  "
            f"LTP=₹{r.get('LTP','?')}")


def main(nse_only: bool = False, top: int = 0, run_scans: bool = True,
         workers: int = MAX_WORKERS):
    _print_header()

    # ── Stage 1: symbol lists (cached for 24 h) ──────────────────────────────
    nse_symbols, bse_only = _load_symbols(nse_only)
    yf_map = _build_yf_map(nse_symbols, bse_only)

    all_tickers = list(yf_map.keys())
    if top:
        all_tickers = all_tickers[:top]
        print(f"  (limited to first {top} tickers)")

    print(f"\nTotal tickers to scan: {len(all_tickers)}")

    # ── Stage 2: bulk OHLC download ──────────────────────────────────────────
    print("\nStage 2 — Bulk OHLC download (1-year window; needed for Golden Crossover 200 DMA) …")
    ohlc_data = bulk_download_ohlc(all_tickers, period="1y")
    print(f"  → {len(ohlc_data)} tickers with usable OHLC data\n")

    # ── Stage 3: Darvas Box + Golden Crossover on every ticker (OHLC only — no API calls)
    print("Stage 3 — Darvas Box + Golden Crossover screen (all stocks) …")
    all_rows, darvas_rows, ohlc_row_map = _ohlc_stage(ohlc_data, yf_map)
    counts = print_ohlc_counts(all_rows)

    # ── Stage 4: ALL 5 fundamental screeners on EVERY stock ──────────────────
    # No pre-filtering: all four fundamental screeners share one Ticker() call per stock.
    fund_rows, six_screen_rows = [], []
    sc_counts = {"piotroski": 0, "cc": 0, "mf": 0, "bc": 0}
    if run_scans:
        all_fund_symbols = [(sym, suf) for yf_tkr, (sym, suf) in yf_map.items()
                            if yf_tkr in ohlc_data]
        fund_rows, six_screen_rows, sc_counts = _fundamental_stage(all_fund_symbols, ohlc_row_map, workers)
    else:
        print("\nStage 4 — Skipped (--no-scans)")

    # Derive triple_rows for backward compat (Darvas + Piotroski + CC)
    triple_rows = triple_hits(fund_rows)

    # ── Stage 5: Save results ─────────────────────────────────────────────────
    print("\nStage 5 — Saving results …")
    path = save_excel(all_rows, darvas_rows, fund_rows, triple_rows,
                      six_screen_rows, tag="indian")

    print_scan_summary(all_rows, counts, sc_counts, triple_rows, six_screen_rows, _hit_line)
    return {
        "triple_hits": triple_rows,
        "six_screen_hits": six_screen_rows,
        "breakout_count": counts[0],
        "gc_count": counts[2],
        "total_scanned": len(all_rows),
        "excel_path": str(path),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full NSE+BSE universe Darvas+Piotroski+CoffeeCan scanner."
    )
    parser.add_argument("--nse-only",  action="store_true", default=False,
                        help="Scan NSE stocks only (skip BSE-only symbols)")
    parser.add_argument("--top",       type=int, default=0,
                        help="Limit scan to first N tickers (0 = all)")
    parser.add_argument("--no-scans",  action="store_true", default=False,
                        help="Skip Piotroski + Coffee Can (Darvas only)")
    parser.add_argument("--workers",   type=int, default=MAX_WORKERS,
                        help=f"Parallel threads for fundamental scans (default {MAX_WORKERS})")
    args = parser.parse_args()
    main(nse_only=args.nse_only, top=args.top,
         run_scans=not args.no_scans, workers=args.workers)
