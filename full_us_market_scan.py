# full_us_market_scan.py
# =======================
# Full NASDAQ + NYSE universe scanner (~8,000+ US equity listings).
#
# Pipeline:
#   Stage 1 — Fetch full NASDAQ + NYSE symbol lists (NASDAQ FTP / fallback)
#   Stage 2 — Bulk OHLC download via yfinance (batched, 3-month window)
#   Stage 3 — Darvas Box screen on every stock
#   Stage 4 — Piotroski F-Score + Coffee Can on Darvas BREAKOUT candidates only
#   Stage 5 — Save Excel workbook with ranked results
#
# Output sheets:
#   All_Stocks      — price summary for every stock scanned
#   Darvas_Signals  — breakout / breakdown alerts ranked by upside
#   Fundamentals    — Piotroski + Coffee Can for breakout candidates
#   Triple_Hits     — BREAKOUT_BUY + Piotroski ≥ 7 + Coffee Can PASS
#
# Usage:
#   python full_us_market_scan.py                   # full NASDAQ + NYSE
#   python full_us_market_scan.py --nasdaq-only     # NASDAQ listings only
#   python full_us_market_scan.py --top 500         # limit to first 500 symbols
#   python full_us_market_scan.py --no-scans        # Darvas only (fast mode)
#   python full_us_market_scan.py --workers 8       # parallel fundamental scans
#   python full_us_market_scan.py --min-price 5     # exclude penny stocks (< $5)
#
# Install:
#   pip install yfinance pandas openpyxl requests
#   pip install "git+https://github.com/Nasdaq/NasdaqCloudDataService-SDK-Python.git"
#
# Nasdaq Cloud Data Service (NCDS) SDK — optional, requires paid Nasdaq credentials.
# Place credentials at ~/.nasdaq/ncds_auth.json:
#   {
#     "oauth": {
#       "endpoint": "https://<auth_host>/auth/realms/pro-realm/protocol/openid-connect/token",
#       "client_id": "<client_id>",
#       "client_secret": "<client_secret>"
#     },
#     "kafka": { "bootstrap_servers": "<streams_host>:9094" },
#     "topic": "NLSPLUS-CTA-V4"
#   }
# Without credentials, symbols fall back to NASDAQ FTP → SEC EDGAR.

import argparse
import sys
import time
from datetime import datetime
from typing import Optional

import pandas as pd

import darvas_core

# Persistent cache (avoids re-downloading on subsequent runs)
try:
    from market_data_cache import MarketCache as _MarketCache
    _US_CACHE = _MarketCache(verbose=True)
    _CACHE_OK  = True
except ImportError:
    _US_CACHE = None
    _CACHE_OK  = False

# ML signal engine (AlQahtani et al. 2025 — Ridge regression)
try:
    from ml_signal_engine import MLSignalEngine as _MLEngine
    _ML_ENGINE = _MLEngine(model_type="ridge")
    _ML_OK     = True
except ImportError:
    _ML_ENGINE = None
    _ML_OK     = False

# NSE data fetcher for live context
try:
    from nse_data_fetcher import NSEDataFetcher as _NSEFetcher
    _US_FETCHER = _NSEFetcher()
    _NSE_OK     = True
except ImportError:
    _US_FETCHER = None
    _NSE_OK     = False

try:
    import yfinance as yf
except ImportError:
    sys.exit("❌  pip install yfinance")

# ── Constants ──────────────────────────────────────────────────────────────────
DARVAS_CONFIRM      = 3
BATCH_SIZE          = 300
SLEEP_BETWEEN       = 1.0
MAX_WORKERS         = 12
MAX_FUND_CANDIDATES = 300  # cap Stage 4 to N freshest breakouts

# Symbol universe (NCDS / NASDAQ FTP / SEC) lives in us_symbols.py; re-exported for existing importers.
from us_symbols import (DOWNLOAD_DIR, NASDAQ_FTP_BASE, NASDAQ_FTP_BASE_HTTP, SYMBOL_CACHE_TTL,  # noqa: F401,E402
                        _load_symbol_cache, _save_symbol_cache, fetch_all_us_symbols_from_sec,
                        fetch_nasdaq_symbols, fetch_nyse_symbols, fetch_symbols_from_ncds)

# ── Bulk OHLC download (cache-aware) ──────────────────────────────────────────

def bulk_download_ohlc(tickers: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
    """
    Download OHLC for US stocks with 3-tier caching (memory → disk → network).

    US tickers have no suffix (AAPL, MSFT etc.) — stored as bare ticker in cache.
    Cache stores 5 years as recommended by AlQahtani et al. (2025).

    Tier 1 — Memory cache: 0.3 ms/stock (same session, already fetched)
    Tier 2 — Parquet cache: ~94 ms/stock (previously downloaded, on disk)
    Tier 3 — yfinance network: ~3s per batch of 100 (cold download)

    For 7,000+ US stocks:
      First run:       ~15-20 min (network)
      Subsequent runs: ~4 min (disk Parquet) | <1s (memory within session)
    """
    if _CACHE_OK:
        return _US_CACHE.get_ohlc_bulk(tickers, force=False)

    # Fallback: direct yfinance (no cache)
    result: dict[str, pd.DataFrame] = {}
    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    print(f"  Downloading OHLC for {len(tickers)} tickers in {len(batches)} batches …")
    for idx, batch in enumerate(batches, 1):
        print(f"    Batch {idx}/{len(batches)} ({len(batch)}) …", end=" ", flush=True)
        for attempt in range(3):
            try:
                raw = yf.download(batch, period=period, auto_adjust=True,
                                  threads=True, progress=False)
                if raw.empty:
                    print("empty"); break
                if isinstance(raw.columns, pd.MultiIndex):
                    for tkr in batch:
                        try:
                            df = raw.xs(tkr, axis=1, level=1).dropna(how="all")
                            if not df.empty and len(df) >= DARVAS_CONFIRM + 5:
                                result[tkr] = df
                        except KeyError:
                            pass
                else:
                    if not raw.empty:
                        result[batch[0]] = raw
                print(f"OK ({sum(1 for t in batch if t in result)} usable)")
                break
            except Exception as e:
                if "Rate" in str(e) or "429" in str(e):
                    time.sleep(30 * (attempt + 1))
                else:
                    print(f"ERROR — {e}"); break
        if idx < len(batches):
            time.sleep(SLEEP_BETWEEN)
    return result


# ── Darvas Box ─────────────────────────────────────────────────────────────────

def compute_darvas_box(df: pd.DataFrame, confirm: int = DARVAS_CONFIRM) -> dict:
    """Darvas Box detection (fuzzy column match). Current bar excluded from box formation."""
    h_col = darvas_core.find_col(df.columns, ["High"])
    l_col = darvas_core.find_col(df.columns, ["Low"])
    c_col = darvas_core.find_col(df.columns, ["Close"])

    if not all([h_col, l_col, c_col]) or len(df) < confirm + 5:
        return {"signal": "INSUFFICIENT_DATA", "box_top": None, "box_bottom": None}

    all_highs  = darvas_core.numeric_list(df, h_col)
    all_lows   = darvas_core.numeric_list(df, l_col)
    all_closes = darvas_core.numeric_list(df, c_col)

    current = all_closes[-1]
    box_top, box_bottom = darvas_core.find_box(all_highs[:-1], all_lows[:-1], confirm)

    if box_top is None:
        return {"signal": "NO_BOX", "box_top": None, "box_bottom": None,
                "current_price": round(current, 2)}
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
        "upside_to_top_pct":   round(upside_to_top, 2),
        "position_in_box_pct": round(pos_in_box,    1),
        "data_points":         len(all_closes),
    }


# ── Golden Crossover ──────────────────────────────────────────────────────────

# ── Fundamental scan ───────────────────────────────────────────────────────────
# Implemented in us_fundamentals.py; re-exported so existing importers keep working.
from scan_common import (OHLC_SIGNALS, compute_golden_crossover, make_write_sheet, gc_label, print_ohlc_counts, print_scan_summary,  # noqa: E402
                         run_fundamental_pool, tally_screens, triple_hits)
from us_fundamentals import fundamental_scan  # noqa: E402,F401


# ── Excel export ───────────────────────────────────────────────────────────────

def save_excel(all_rows, darvas_rows, fund_rows, triple_rows,
               six_screen_rows=None, tag="us", ml_signal_map=None):
    date_str = datetime.today().strftime("%Y%m%d_%H%M")
    path     = DOWNLOAD_DIR / f"{tag}_full_scan_{date_str}.xlsx"

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        write_sheet = make_write_sheet(writer)

        write_sheet(all_rows,    "All_Stocks",       sort_col="Change%")
        write_sheet(darvas_rows, "Darvas_Signals",   sort_col="Upside_to_Top%")
        write_sheet(fund_rows,   "All_Fundamentals", sort_col="Piotroski_Score")
        write_sheet([r for r in fund_rows if r.get("Piotroski_Strong") == "YES"],
                    "Piotroski_Strong", sort_col="Piotroski_Score")
        write_sheet([r for r in fund_rows if r.get("CoffeeCan") == "PASS"],
                    "Coffee_Can",       sort_col="Revenue_CAGR_%")
        write_sheet([r for r in fund_rows if r.get("MagicFormula") == "PASS"],
                    "Magic_Formula",    sort_col="ROIC_%")
        write_sheet([r for r in fund_rows if r.get("BullCartel") == "PASS"],
                    "Bull_Cartel",      sort_col="Profit_Growth_YoY_%")
        write_sheet([r for r in all_rows if r.get("GC_Signal") == "GOLDEN_CROSS"],
                    "Golden_Crossover", sort_col="DMA_Gap%")
        write_sheet(triple_rows, "Triple_Hits",      sort_col="Piotroski_Score")
        write_sheet(six_screen_rows or [], "Multi_Screen_Hits", sort_col="Screens_Passed")
        # ML signal sheet — all stocks with ML_Direction, sorted by predicted return
        if ml_signal_map:
            ml_rows = [{"Symbol": s, **v} for s, v in ml_signal_map.items()]
            write_sheet(
                [r for r in ml_rows if r.get("ML_Direction")=="BULLISH"],
                "ML_Bullish",  sort_col="ML_Pred_Ret%"
            )
            write_sheet(
                [r for r in ml_rows if r.get("ML_Direction")=="BEARISH"],
                "ML_Bearish",  sort_col="ML_Pred_Ret%"
            )

    print(f"\n  📊  Excel saved → {path}")
    return path


# ── Main ───────────────────────────────────────────────────────────────────────

def _print_header() -> None:
    print(f"\n{'#'*60}")
    print("  FULL US MARKET SCAN — NASDAQ + NYSE")
    print(f"  Started: {datetime.now().strftime('%d %b %Y  %H:%M:%S')}")
    print(f"  Cache: {'✅ active (Parquet)' if _CACHE_OK else '⚠️  disabled'}")
    print(f"  ML signal: {'✅ Ridge regression (AlQahtani et al. 2025)' if _ML_OK else '⚠️  disabled'}")
    print(f"{'#'*60}\n")


def _print_market_context() -> None:
    """Live S&P 500 regime and VIX line (best effort; silent when yfinance fails)."""
    if not _NSE_OK:
        return
    try:
        import yfinance as yf
        sp500 = yf.download("^GSPC", period="1y", auto_adjust=True, progress=False)
        if isinstance(sp500.columns, pd.MultiIndex):
            sp500 = sp500.xs("^GSPC", axis=1, level=1)
        if not sp500.empty:
            sp_last  = float(sp500["Close"].iloc[-1])
            sp_dma200= float(sp500["Close"].rolling(200).mean().iloc[-1])
            sp_regime= "BULL" if sp_last > sp_dma200 else "BEAR"
            sp_pct   = (sp_last - sp_dma200) / sp_dma200 * 100
            print(f"  S&P 500: {sp_last:,.0f}  |  200 DMA: {sp_dma200:,.0f}  "
                  f"({sp_pct:+.2f}%)  |  Regime: {sp_regime}")
            # VIX from nsepython not applicable for US — use yfinance VIX
            try:
                vix = yf.download("^VIX", period="5d", progress=False)
                if not vix.empty:
                    vix_val = float(vix["Close"].iloc[-1])
                    vix_lvl = ("NORMAL" if vix_val<18 else "ELEVATED" if vix_val<25 else "PANIC")
                    print(f"  CBOE VIX: {vix_val:.1f}  [{vix_lvl}]")
            except Exception:
                pass
    except Exception:
        pass


def _collect_symbols(nasdaq_only: bool, top: int) -> list:
    """Stage 1: unique NASDAQ + NYSE symbols (24 h cache, else fetched), optionally limited to ``top``."""
    print("Stage 1 — Fetching symbol lists …")
    nasdaq_syms, nyse_syms = _load_symbol_cache()
    if nasdaq_syms is None:
        nasdaq_syms = fetch_nasdaq_symbols()
        print(f"  NASDAQ:       {len(nasdaq_syms)} symbols")
        nyse_syms = []
        if not nasdaq_only:
            nyse_syms = fetch_nyse_symbols()
            print(f"  NYSE/Other:   {len(nyse_syms)} symbols")
        _save_symbol_cache(nasdaq_syms, nyse_syms)
    else:
        print(f"  Symbol cache hit: {len(nasdaq_syms)} NASDAQ + {len(nyse_syms)} NYSE/other")
        if nasdaq_only:
            nyse_syms = []

    seen = set()
    all_symbols = []
    for s in nasdaq_syms + nyse_syms:
        if s not in seen:
            seen.add(s)
            all_symbols.append(s)

    if top:
        all_symbols = all_symbols[:top]
        print(f"  (limited to first {top} symbols)")

    print(f"  Total unique symbols: {len(all_symbols)}\n")
    return all_symbols


def _ohlc_base_row(ticker: str, df: pd.DataFrame, min_price: float) -> Optional[dict]:
    """Price + Darvas + Golden Crossover row, or None when there is no price or it is below ``min_price``."""
    closes = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if closes.empty:
        return None
    ltp = round(float(closes.iloc[-1]), 2)
    if ltp < min_price:
        return None
    prev    = round(float(closes.iloc[-2]), 2) if len(closes) >= 2 else None
    chg_pct = round((ltp - prev) / prev * 100, 2) if (prev and prev) else None

    darvas = compute_darvas_box(df)
    gc     = compute_golden_crossover(df)
    return {
        "Symbol":           ticker,
        "LTP":              ltp,
        "Prev_Close":       prev,
        "Change%":          chg_pct,
        "Darvas_Signal":    darvas.get("signal"),
        "Box_Top":          darvas.get("box_top"),
        "Box_Bottom":       darvas.get("box_bottom"),
        "Upside_to_Top%":   darvas.get("upside_to_top_pct"),
        "Position_in_Box%": darvas.get("position_in_box_pct"),
        "Data_Points":      darvas.get("data_points"),
        "GC_Signal":        gc_label(gc),
        "DMA50":            gc.get("dma50"),
        "DMA200":           gc.get("dma200"),
        "DMA_Gap%":         gc.get("dma_gap_%"),
    }


def _ohlc_stage(ohlc_data: dict, min_price: float) -> tuple:
    """Stage 3: (all_rows, darvas_rows, ohlc_row_map) for every priced ticker at or above ``min_price``."""
    all_rows, darvas_rows = [], []
    ohlc_row_map: dict[str, dict] = {}
    for ticker, df in ohlc_data.items():
        base_row = _ohlc_base_row(ticker, df, min_price)
        if base_row is None:
            continue
        all_rows.append(base_row)
        ohlc_row_map[ticker] = base_row
        if base_row["Darvas_Signal"] in OHLC_SIGNALS:
            darvas_rows.append(base_row.copy())
    return all_rows, darvas_rows, ohlc_row_map


def _fund_row(sym: str, ohlc: dict, res: dict) -> dict:
    """All_Fundamentals row: OHLC columns plus the five fundamental screeners' outputs."""
    return {
        "Symbol":              sym,
        "Name":                res.get("name", ""),
        "Sector":              res.get("sector", ""),
        "LTP":                 ohlc.get("LTP"),
        "Change%":             ohlc.get("Change%"),
        "Darvas_Signal":       ohlc.get("Darvas_Signal"),
        "Box_Top":             ohlc.get("Box_Top"),
        "Box_Bottom":          ohlc.get("Box_Bottom"),
        "Upside_to_Top%":      ohlc.get("Upside_to_Top%"),
        "GC_Signal":           ohlc.get("GC_Signal"),
        "DMA50":               ohlc.get("DMA50"),
        "DMA200":              ohlc.get("DMA200"),
        "Piotroski_Score":     res.get("f_score"),
        "Piotroski_Strong":    "YES" if res.get("f_strong") else "NO",
        "CoffeeCan":           "PASS" if res.get("qualifies_cc") else "FAIL",
        "CC_Score":            res.get("cc_score"),
        "Revenue_CAGR_%":      res.get("Revenue_CAGR_%"),
        "ROE_avg_%":           res.get("ROE_avg_%"),
        "MagicFormula":        "PASS" if res.get("qualifies_mf") else "FAIL",
        "ROIC_%":              res.get("ROIC_%"),
        "Earnings_Yield_%":    res.get("Earnings_Yield_%"),
        "BullCartel":          "PASS" if res.get("qualifies_bc") else "FAIL",
        "Sales_Growth_YoY_%":  res.get("Sales_Growth_YoY_%"),
        "Profit_Growth_YoY_%": res.get("Profit_Growth_YoY_%"),
        "Net_Profit_$M":       res.get("Net_Profit_$M"),
        "Error":               res.get("error", ""),
    }


def _fundamental_stage(symbols: list, ohlc_row_map: dict, workers: int) -> tuple:
    """Stage 4: all fundamental screeners on every ticker -> (fund_rows, six_screen_rows, sc_counts)."""
    fund_rows, six_screen_rows = [], []
    sc_counts = {"piotroski": 0, "cc": 0, "mf": 0, "bc": 0}
    print(f"\nStage 4 — All 5 fundamental screeners on {len(symbols)} stocks "
          f"({workers} workers) …")

    def handle(sym, res, done):
        ohlc = ohlc_row_map.get(sym, {})
        row = _fund_row(sym, ohlc, res)
        fund_rows.append(row)
        tally_screens(ohlc, res, row, sc_counts, six_screen_rows)
        if done % 200 == 0 or done == len(symbols):
            print(f"    {done}/{len(symbols)} done  "
                  f"(multi-screen hits: {len(six_screen_rows)})")

    run_fundamental_pool([(sym, (sym,)) for sym in symbols], fundamental_scan, workers, handle)
    return fund_rows, six_screen_rows, sc_counts


def _ml_stage(ohlc_data: dict, workers: int, all_rows: list, fund_rows: list, six_screen_rows: list) -> dict:
    """Stage 4b: Ridge-regression ML signal per ticker, attached in place to the row lists; returns the map."""
    ml_signal_map: dict = {}
    if not (_ML_OK and ohlc_data):
        return ml_signal_map
    print(f"\nStage 4b — ML signal (Ridge regression) on {len(ohlc_data)} stocks …")
    try:
        ml_df = _ML_ENGINE.predict_batch(ohlc_data, max_workers=workers)
        for _, row in ml_df.iterrows():
            ml_signal_map[row["symbol"]] = {
                "ML_Direction":    row.get("direction", "NEUTRAL"),
                "ML_Pred_Ret%":    row.get("predicted_ret%", 0),
                "ML_Confidence":   row.get("confidence", 0),
                "ML_TrainRMSE":    row.get("train_rmse"),
            }
    except Exception as e:
        print(f"  ML signal error: {e}")

    # Attach ML signals to all_rows and fund_rows
    for r in all_rows:
        r.update(ml_signal_map.get(r.get("Symbol",""), {}))
    for r in fund_rows:
        r.update(ml_signal_map.get(r.get("Symbol",""), {}))

    # Count signals
    n_bull = sum(1 for v in ml_signal_map.values() if v.get("ML_Direction")=="BULLISH")
    n_bear = sum(1 for v in ml_signal_map.values() if v.get("ML_Direction")=="BEARISH")
    print(f"  ML: BULLISH={n_bull} | BEARISH={n_bear} | "
          f"NEUTRAL={len(ml_signal_map)-n_bull-n_bear}")

    # High-conviction: fundamental screener signal + ML BULLISH
    for r in six_screen_rows:
        ml = ml_signal_map.get(r.get("Symbol",""), {})
        r.update(ml)
        r["ML_Confirmed"] = "YES" if ml.get("ML_Direction")=="BULLISH" else "NO"
    return ml_signal_map


def _hit_line(r: dict) -> str:
    return (f"    {r['Symbol']:<10} {r.get('Name',''):<30} {r['Screens_Passed']}/6 "
            f"F={r.get('Piotroski_Score') or '-'}/9  CC={r.get('CC_Score','-')}  "
            f"MF={r.get('MagicFormula','-')}  LTP=${r.get('LTP','?')}")


def main(nasdaq_only: bool = False, top: int = 0, run_scans: bool = True,
         workers: int = MAX_WORKERS, min_price: float = 1.0):
    _print_header()
    _print_market_context()
    print()

    all_symbols = _collect_symbols(nasdaq_only, top)

    # ── Stage 2: bulk OHLC download ──────────────────────────────────────────
    print("Stage 2 — Bulk OHLC download (1-year window; needed for Golden Crossover 200 DMA) …")
    ohlc_data = bulk_download_ohlc(all_symbols, period="1y")
    print(f"  → {len(ohlc_data)} tickers with usable data\n")

    # Stage 3: Darvas Box + Golden Crossover on every ticker (no extra API calls)
    print("Stage 3 — Darvas Box + Golden Crossover screen (all stocks) …")
    all_rows, darvas_rows, ohlc_row_map = _ohlc_stage(ohlc_data, min_price)
    counts = print_ohlc_counts(all_rows)

    # ── Stage 4: ALL 5 fundamental screeners on EVERY stock ──────────────────
    fund_rows, six_screen_rows = [], []
    sc_counts = {"piotroski": 0, "cc": 0, "mf": 0, "bc": 0}
    if run_scans:
        fund_rows, six_screen_rows, sc_counts = _fundamental_stage(list(ohlc_row_map.keys()), ohlc_row_map, workers)
    else:
        print("\nStage 4 — Skipped (--no-scans)")

    # Backward-compat triple_rows
    triple_rows = triple_hits(fund_rows)

    # ── Stage 4b: ML signal on every stock with OHLC (AlQahtani et al. 2025) ─
    ml_signal_map = _ml_stage(ohlc_data, workers, all_rows, fund_rows, six_screen_rows)

    # ── Stage 5: Save results ─────────────────────────────────────────────────
    print("\nStage 5 — Saving results …")
    path = save_excel(all_rows, darvas_rows, fund_rows, triple_rows,
                      six_screen_rows, tag="us", ml_signal_map=ml_signal_map)

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
        description="Full NASDAQ+NYSE universe Darvas+Piotroski+CoffeeCan scanner."
    )
    parser.add_argument("--nasdaq-only", action="store_true", default=False,
                        help="Scan NASDAQ-listed stocks only (skip NYSE/AMEX)")
    parser.add_argument("--top",         type=int,   default=0,
                        help="Limit scan to first N symbols (0 = all)")
    parser.add_argument("--no-scans",    action="store_true", default=False,
                        help="Skip Piotroski + Coffee Can (Darvas only, much faster)")
    parser.add_argument("--workers",     type=int,   default=MAX_WORKERS,
                        help=f"Parallel threads for fundamental scans (default {MAX_WORKERS})")
    parser.add_argument("--min-price",   type=float, default=1.0,
                        help="Exclude stocks priced below this (default $1 — filters penny stocks)")
    args = parser.parse_args()
    main(nasdaq_only=args.nasdaq_only, top=args.top,
         run_scans=not args.no_scans, workers=args.workers,
         min_price=args.min_price)
