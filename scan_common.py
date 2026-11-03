"""Pieces shared by the five full_<market>_market_scan modules (pure helpers + the stage-4 worker pool).

Each scanner keeps its own universe, row layout and Excel sheets; only code that was copy-pasted
identically across them lives here.
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Optional

import pandas as pd

from stock_utils import first_df as _first_df, row as _row, series as _series

logger = logging.getLogger(__name__)

OHLC_SIGNALS = ("BREAKOUT_BUY", "BREAKDOWN_SELL")
GC_PASSING = ("GOLDEN_CROSS", "ABOVE_200DMA")


# ── Piotroski ──────────────────────────────────────────────────────────────────

def piotroski_f_score(inc, bal, cf) -> int:
    """9-point Piotroski F-score from annual income / balance-sheet / cash-flow frames (0-9)."""
    ni0 = _row(inc, "Net Income", col=0);  a0 = _row(bal, "Total Assets", col=0)
    ni1 = _row(inc, "Net Income", col=1);  a1 = _row(bal, "Total Assets", col=1)
    roa0 = (ni0 / a0) if (ni0 and a0) else None
    roa1 = (ni1 / a1) if (ni1 and a1) else None
    ocf0 = _row(cf, "Operating Cash Flow", "Total Cash From Operating Activities")
    ltd0 = _row(bal, "Long Term Debt", col=0) or 0
    ltd1 = _row(bal, "Long Term Debt", col=1) or 0
    ca0 = _row(bal, "Current Assets", "Total Current Assets", col=0)
    cl0 = _row(bal, "Current Liabilities", "Total Current Liabilities", col=0)
    ca1 = _row(bal, "Current Assets", "Total Current Assets", col=1)
    cl1 = _row(bal, "Current Liabilities", "Total Current Liabilities", col=1)
    sh0 = _row(bal, "Share Issued", col=0);  sh1 = _row(bal, "Share Issued", col=1)
    rev0 = _row(inc, "Total Revenue", col=0);  gp0 = _row(inc, "Gross Profit", col=0)
    rev1 = _row(inc, "Total Revenue", col=1);  gp1 = _row(inc, "Gross Profit", col=1)
    return (
        (1 if (roa0 and roa0 > 0) else 0) +
        (1 if (ocf0 and ocf0 > 0) else 0) +
        (1 if (roa0 and roa1 and roa0 > roa1) else 0) +
        (1 if (ocf0 and a0 and roa0 and (ocf0 / a0) > roa0) else 0) +
        (1 if (a0 and a1 and (ltd0 / a0) < (ltd1 / a1)) else 0) +
        (1 if (ca0 and cl0 and ca1 and cl1 and (ca0 / cl0) > (ca1 / cl1)) else 0) +
        ((1 if sh0 <= sh1 else 0) if (sh0 and sh1) else 1) +
        (1 if (gp0 and rev0 and gp1 and rev1 and (gp0 / rev0) > (gp1 / rev1)) else 0) +
        (1 if (rev0 and a0 and rev1 and a1 and (rev0 / a0) > (rev1 / a1)) else 0)
    )


# ── Golden Crossover ───────────────────────────────────────────────────────────

def compute_golden_crossover(df: pd.DataFrame) -> dict:
    """
    Detect if the 50 DMA just crossed above the 200 DMA (today's bar).
    Uses only the already-downloaded bulk OHLC — zero extra API calls.
    Requires 201+ bars (≈ 1 year of trading days).
    """
    if df is None or df.empty:
        return {"gc_signal": False, "dma50_above_200": False, "dma50": None, "dma200": None}
    closes = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(closes) < 201:
        return {"gc_signal": False, "dma50_above_200": False, "dma50": None, "dma200": None}
    dma50  = closes.rolling(50).mean()
    dma200 = closes.rolling(200).mean()
    d50_t, d200_t = float(dma50.iloc[-1]), float(dma200.iloc[-1])
    d50_p, d200_p = float(dma50.iloc[-2]), float(dma200.iloc[-2])
    gc_today = (d50_p < d200_p) and (d50_t > d200_t)
    gap_pct  = round((d50_t - d200_t) / d200_t * 100, 2) if d200_t else 0
    return {
        "gc_signal":      gc_today,              # strict: crossed exactly today
        "dma50_above_200": d50_t > d200_t,       # broader: currently above
        "dma50":          round(d50_t, 2),
        "dma200":         round(d200_t, 2),
        "dma_gap_%":      gap_pct,
    }


# ── Price-row helpers ──────────────────────────────────────────────────────────

def gc_label(gc: dict) -> str:
    """Golden-crossover dict -> GOLDEN_CROSS / ABOVE_200DMA / BELOW_200DMA."""
    return "GOLDEN_CROSS" if gc.get("gc_signal") else (
        "ABOVE_200DMA" if gc.get("dma50_above_200") else "BELOW_200DMA")


def trend_signal(dist_ma: Optional[float], fallback: str) -> str:
    """Trend label from % distance to the 200-DMA (+-5% band); ``fallback`` when the MA is unavailable."""
    return ("Above 200MA (Uptrend)" if dist_ma and dist_ma > 5 else
            "Below 200MA (Downtrend)" if dist_ma and dist_ma < -5 else
            "Near 200MA (Consolidation)" if dist_ma else fallback)


def last_two_closes(df: pd.DataFrame, ndigits: int) -> tuple:
    """(closes, ltp, prev, change%) from a frame's Close column, rounded to ``ndigits``; None when missing."""
    closes = pd.to_numeric(df["Close"], errors="coerce").dropna()
    ltp = round(float(closes.iloc[-1]), ndigits) if not closes.empty else None
    prev = round(float(closes.iloc[-2]), ndigits) if len(closes) >= 2 else None
    chg = round((ltp - prev) / prev * 100, 2) if (ltp and prev) else None
    return closes, ltp, prev, chg


def ma200_and_distance(closes: pd.Series, ltp, ndigits: int) -> tuple:
    """(200-DMA, % distance of ltp from it); both None below 200 bars."""
    ma200 = round(closes.rolling(200).mean().iloc[-1], ndigits) if len(closes) >= 200 else None
    dist = round((ltp - ma200) / ma200 * 100, 2) if (ma200 and ltp) else None
    return ma200, dist


# ── Stage 3 / 4 / summary pieces shared by the US and India scanners ───────────

def print_ohlc_counts(all_rows: list) -> tuple:
    """Print and return (breakouts, breakdowns, golden crosses, dma50>dma200) over the OHLC rows."""
    breakout = sum(1 for r in all_rows if r["Darvas_Signal"] == "BREAKOUT_BUY")
    breakdown = sum(1 for r in all_rows if r["Darvas_Signal"] == "BREAKDOWN_SELL")
    gc = sum(1 for r in all_rows if r["GC_Signal"] == "GOLDEN_CROSS")
    above = sum(1 for r in all_rows if r["GC_Signal"] in GC_PASSING)
    print(f"  Darvas Breakout BUY:    {breakout}")
    print(f"  Darvas Breakdown SELL:  {breakdown}")
    print(f"  Golden Cross (today):   {gc}")
    print(f"  DMA50 above DMA200:     {above}")
    return breakout, breakdown, gc, above


def tally_screens(ohlc: dict, res: dict, row: dict, sc_counts: dict, six_screen_rows: list) -> None:
    """Update per-screener counts and append ``row`` to the multi-screen list when 3+ of 6 screens pass."""
    if res.get("f_strong"):
        sc_counts["piotroski"] += 1
    if res.get("qualifies_cc"):
        sc_counts["cc"] += 1
    if res.get("qualifies_mf"):
        sc_counts["mf"] += 1
    if res.get("qualifies_bc"):
        sc_counts["bc"] += 1
    screens_passed = sum([
        ohlc.get("Darvas_Signal") == "BREAKOUT_BUY",
        ohlc.get("GC_Signal") in GC_PASSING,
        bool(res.get("f_strong")),
        bool(res.get("qualifies_cc")),
        bool(res.get("qualifies_mf")),
        bool(res.get("qualifies_bc")),
    ])
    if screens_passed >= 3:
        six_screen_rows.append({**row, "Screens_Passed": screens_passed})


def triple_hits(fund_rows: list) -> list:
    """Rows with a Darvas breakout + strong Piotroski + Coffee Can pass."""
    return [r for r in fund_rows
            if r.get("Darvas_Signal") == "BREAKOUT_BUY"
            and r.get("Piotroski_Strong") == "YES"
            and r.get("CoffeeCan") == "PASS"]


def print_scan_summary(all_rows, counts, sc_counts, triple_rows, six_screen_rows, hit_line: Callable) -> None:
    """The closing SCAN COMPLETE block; ``hit_line(row)`` formats one top multi-screen hit."""
    from datetime import datetime
    breakout_count, breakdown_count, gc_count, above_200_count = counts
    print(f"\n{'='*70}")
    print(f"  SCAN COMPLETE — {datetime.now().strftime('%d %b %Y  %H:%M:%S')}")
    print(f"  Tickers scanned:          {len(all_rows)}")
    print("  ── OHLC Screeners ──────────────────────────────────────")
    print(f"  Darvas Breakouts:         {breakout_count}")
    print(f"  Darvas Breakdowns:        {breakdown_count}")
    print(f"  Golden Cross (today):     {gc_count}")
    print(f"  DMA50 above DMA200:       {above_200_count}")
    print("  ── Fundamental Screeners ────────────────────────────────")
    print(f"  Piotroski ≥7 (Strong):    {sc_counts.get('piotroski', 0)}")
    print(f"  Coffee Can PASS:          {sc_counts.get('cc', 0)}")
    print(f"  Magic Formula PASS:       {sc_counts.get('mf', 0)}")
    print(f"  Bull Cartel PASS:         {sc_counts.get('bc', 0)}")
    print("  ── Combined ─────────────────────────────────────────────")
    print(f"  Triple Hits (D+P+CC):     {len(triple_rows)}")
    print(f"  ★ Multi-Screen Hits (3+): {len(six_screen_rows)}")
    if six_screen_rows:
        top = sorted(six_screen_rows, key=lambda x: x.get("Screens_Passed", 0), reverse=True)[:10]
        print("\n  Top Multi-Screen Hits:")
        for r in top:
            print(hit_line(r))
    print(f"{'='*70}\n")


def run_fundamental_pool(jobs: Iterable, scan_fn: Callable, workers: int, handle: Callable,
                         error_fmt: str = "    {label}: error — {exc}",
                         label: Callable = lambda key: key) -> None:
    """Run ``scan_fn(*args)`` for every ``(key, args)`` job on a thread pool.

    ``handle(key, result, done)`` runs as each future completes (``done`` = 1-based completion count);
    an exception in the scan or in ``handle`` is printed with ``error_fmt`` and the loop continues.
    """
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(scan_fn, *args): key for key, args in jobs}
        for future in as_completed(futures):
            key = futures[future]
            done += 1
            try:
                handle(key, future.result(), done)
            except Exception as exc:  # per-ticker failure must not abort the scan
                print(error_fmt.format(label=label(key), exc=exc))


# ── Excel sheet writer ─────────────────────────────────────────────────────────

def make_write_sheet(writer, style: Optional[Callable] = None, empty_note: bool = False) -> Callable:
    """Return ``write_sheet(rows, name, sort_col=None, sort_asc=False)`` bound to an open ExcelWriter.

    ``style(ws)`` is applied to non-empty sheets when given; ``empty_note`` writes a one-line Note sheet
    instead of an empty one when there are no rows.
    """
    def write_sheet(rows, name, sort_col=None, sort_asc=False):
        if not rows:
            empty = pd.DataFrame({"Note": ["No qualifying stocks today"]}) if empty_note else pd.DataFrame()
            empty.to_excel(writer, sheet_name=name, index=False)
            return
        df = pd.DataFrame(rows)
        if sort_col and sort_col in df.columns:
            df = df.sort_values(sort_col, ascending=sort_asc)
        df.to_excel(writer, sheet_name=name, index=False)
        if style is not None:
            style(writer.sheets[name])
    return write_sheet


# ── Japan / Korea fundamental scan (identical apart from labels and market-cap unit) ──

def _asia_coffee_can(inc, bal, cf, ticker, mcap_key: str, mcap_div: float) -> dict:
    """Coffee Can fields: C1 rev CAGR>10, C2 ROCE>15, C3 D/E<1, C4 mcap>=1e11, C5 profitable, C6 FCF>0."""
    c = {}
    revs = _series(inc, "Total Revenue")
    if len(revs) >= 2 and revs[-1] > 0:
        cagr = ((revs[0] / revs[-1]) ** (1 / (len(revs) - 1)) - 1) * 100
        c["C1"] = 1 if cagr > 10 else 0
    else:
        cagr = None; c["C1"] = 0

    ebit_s = _series(inc, "EBIT", "Operating Income", "Ebit")
    ta_s   = _series(bal, "Total Assets")
    cl_s   = _series(bal, "Current Liabilities", "Total Current Liabilities")
    roce_l = [ebit_s[i] / (ta_s[i] - cl_s[i]) * 100
              for i in range(min(len(ebit_s), len(ta_s), len(cl_s)))
              if (ta_s[i] - cl_s[i]) > 0]
    avg_roce = sum(roce_l) / len(roce_l) if roce_l else None
    c["C2"] = 1 if (avg_roce and avg_roce > 15) else 0

    ltd_s = _series(bal, "Long Term Debt")
    eq_s  = _series(bal, "Stockholders Equity", "Total Stockholder Equity",
                    "Total Equity Gross Minority Interest")
    de = (ltd_s[0] / abs(eq_s[0])) if (ltd_s and eq_s and eq_s[0] != 0) else None
    c["C3"] = 1 if (de is not None and de < 1) else 0

    try:
        mcap = ticker.fast_info.market_cap or 0
    except Exception:
        mcap = 0
    c["C4"] = 1 if mcap >= 1e11 else 0

    ni_s = _series(inc, "Net Income")
    c["C5"] = 1 if (ni_s and all(n > 0 for n in ni_s)) else 0

    fcf_s = _series(cf, "Free Cash Flow")
    if fcf_s:
        c["C6"] = 1 if fcf_s[0] > 0 else 0
    else:
        ocf_s   = _series(cf, "Operating Cash Flow")
        capex_s = _series(cf, "Capital Expenditure")
        c["C6"] = 1 if (ocf_s and capex_s and (ocf_s[0] - abs(capex_s[0])) > 0) else 0

    qualifies = sum(c.values()) == len(c)
    return {
        "cc_qualifies":   "PASS" if qualifies else "FAIL",
        "cc_score":       f"{sum(c.values())}/{len(c)}",
        "cc_rev_cagr":    round(cagr, 2) if cagr else None,
        "cc_roce_avg":    round(avg_roce, 2) if avg_roce else None,
        "cc_debt_equity": round(de, 2) if de is not None else None,
        mcap_key:         round(mcap / mcap_div, 1) if mcap else None,
    }


def asia_fundamental_scan(result: dict, yf_mod, yf_ticker: str, strong_threshold: int,
                          mcap_key: str, mcap_div: float) -> dict:
    """Piotroski + market-adapted Coffee Can for one TSE/KRX ticker; fills and returns ``result``."""
    try:
        ticker = yf_mod.Ticker(yf_ticker)
        inc = _first_df(ticker, "income_stmt", "financials")
        bal = _first_df(ticker, "balance_sheet")
        cf  = _first_df(ticker, "cash_flow", "cashflow")
        if inc is None:
            result["error"] = "no_financial_data"
            return result
    except Exception as e:
        result["error"] = str(e)[:80]
        return result

    try:
        f_score = piotroski_f_score(inc, bal, cf)
        result["f_score"]  = f_score
        result["f_strong"] = f_score >= strong_threshold
    except Exception:
        return result

    if not result.get("f_strong"):
        result["cc_score"] = "—"
        return result

    try:
        result.update(_asia_coffee_can(inc, bal, cf, ticker, mcap_key, mcap_div))
    except Exception as exc:  # a malformed statement leaves the Coffee Can fields at their FAIL defaults
        logger.debug("coffee-can failed for %s: %s", yf_ticker, exc)
    return result


# ── 24 h JSON symbol-list cache ────────────────────────────────────────────────

def load_json_cache(path, ttl_sec: float) -> Optional[dict]:
    """The cached JSON object if ``path`` holds one younger than ``ttl_sec`` (its ``ts`` field), else None."""
    try:
        if path.exists():
            data = json.loads(path.read_text())
            if time.time() - data.get("ts", 0) < ttl_sec:
                return data
    except (OSError, ValueError, AttributeError, TypeError) as exc:
        logger.debug("symbol cache %s unreadable: %s", path, exc)
    return None


def save_json_cache(path, payload: dict) -> None:
    """Write ``payload`` plus a ``ts`` timestamp to ``path``; an unwritable location is logged, not raised."""
    try:
        path.write_text(json.dumps({"ts": time.time(), **payload}))
    except (OSError, TypeError, ValueError) as exc:
        logger.debug("symbol cache %s not written: %s", path, exc)
