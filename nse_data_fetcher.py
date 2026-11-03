# nse_data_fetcher.py
# ====================
# Unified NSE + BSE data layer using NSEpy + nsepython + yfinance.
#
# Library capabilities (tested on macOS):
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ Library      │ What it provides                  │ Status on Mac            │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ nsepython    │ Live quotes, VIX, FII/DII, bulk    │ ✅ Works reliably        │
# │              │ deals, block deals, events calendar│                          │
# │              │ symbol list (2372 EQ stocks)       │                          │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ NSEpy        │ Historical OHLC (NSE direct)       │ ⚠️  SSL error on new API │
# │              │ Index history, derivatives history │   Use yfinance fallback  │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ yfinance     │ Historical OHLC (via Yahoo Finance)│ ✅ Works (rate limits)   │
# │              │ Annual + quarterly financial stmts │ ✅ Best for financials   │
# │              │ Insider trades, institutional data │                          │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# Architecture:
#   Historical OHLC    → yfinance primary (with NSEpy fallback when fixed)
#   Symbol universe    → nsepython.nse_eq_symbols() — direct NSE, 2372 stocks
#   Market regime      → nsepython: VIX + Nifty live + FII/DII trend
#   Filing events      → nsepython.nse_events() — results/board meeting calendar
#   Bulk/block deals   → nsepython: institutional activity confirmation
#   Fundamental data   → yfinance (Piotroski, Coffee Can, Magic Formula)
#
# Usage:
#   from nse_data_fetcher import NSEDataFetcher
#   nse = NSEDataFetcher()
#   regime   = nse.get_regime()            # {'regime':'BULL','vix':13.05,...}
#   symbols  = nse.get_all_symbols()       # ['RELIANCE','TCS',...]
#   events   = nse.get_upcoming_events()   # DataFrame of results/board meetings
#   bulk     = nse.get_bulk_deals()        # today's bulk deals
#   fiidii   = nse.get_fii_dii()          # FII/DII net activity
#   ohlc     = nse.get_ohlc('RELIANCE',period='5y')  # historical OHLC
#
# Install:
#   pip install nsepython nsepy yfinance pandas "nse[local]"

import logging
import time
import warnings
from datetime import datetime, timedelta, date
from pathlib import Path
from functools import lru_cache

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

from nse_regime import (RegimeSources, build_live_context, build_regime, market_breadth, nifty_pe_pb, classify_regime, institutional_score,  # noqa: E402,F401
                        print_live_context_report, strategy_recommendation)

# ── Imports with graceful fallback ─────────────────────────────────────────────
try:
    import yfinance as yf
    _YF_OK = True
except ImportError:
    _YF_OK = False
    print("⚠️  yfinance not installed: pip install yfinance")

try:
    from nsepython import (
        nse_eq_symbols, indiavix, get_bulkdeals, get_blockdeals,
        nse_events, nse_get_index_quote, nse_fiidii,
        nse_get_top_gainers, nse_get_top_losers, nse_circular,
        holiday_master, fnolist, expiry_list,
    )
    _NSEPY_OK = True
except ImportError:
    _NSEPY_OK = False
    print("⚠️  nsepython not installed: pip install nsepython")

try:
    from nsepy import get_history as _nsepy_get_history
    _NSEPY_LIB_OK = True
except ImportError:
    _NSEPY_LIB_OK = False

CACHE_DIR = Path("./nse_cache")
CACHE_DIR.mkdir(exist_ok=True)

def _load_index_pe_pb_div():
    """Lazy nsepython lookup (the function is optional in older nsepython releases)."""
    from nsepython import index_pe_pb_div
    return index_pe_pb_div


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DATA FETCHER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class NSEDataFetcher:
    """
    Unified NSE data access layer.

    Combines the strengths of nsepython (live market data, events, sentiment)
    with yfinance (historical OHLC, financial statements) to give a complete
    picture for both live screening and backtesting.
    """

    def __init__(self, cache_ttl_sec: int = 3600):
        self._cache_ttl = cache_ttl_sec
        self._cache: dict = {}
        self._index_cache: dict = {}

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1: SYMBOL UNIVERSE
    # ─────────────────────────────────────────────────────────────────────────

    def get_all_symbols(self, series: str = "EQ") -> list:
        """
        Get all NSE-listed equity symbols directly from NSE via nsepython.
        nsepython.nse_eq_symbols() returns the live NSE EQ universe (~2372 stocks).
        Falls back to nse-library bhavcopy if nsepython is unavailable.

        Returns: list of NSE ticker symbols (e.g. ['RELIANCE', 'TCS', ...])
        """
        cache_key = f"symbols_{series}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        # Primary: nsepython (direct NSE, always current)
        if _NSEPY_OK:
            try:
                syms = nse_eq_symbols()
                if syms and len(syms) > 100:
                    print(f"  nsepython: {len(syms)} NSE EQ symbols")
                    self._set_cache(cache_key, syms)
                    return syms
            except Exception as e:
                print(f"  nsepython.nse_eq_symbols failed: {e}")

        # Fallback: nse library bhavcopy
        try:
            from nse import NSE
            today = datetime.today()
            with NSE(download_folder=str(CACHE_DIR), server=False) as nse:
                for offset in range(7):
                    d = today - timedelta(days=offset)
                    try:
                        result = nse.equityBhavcopy(d)
                        if hasattr(result, "exists") and result.exists():
                            df = pd.read_csv(result)
                            if "SctySrs" in df.columns:
                                syms = sorted(df[df["SctySrs"]==series]["TckrSymb"]
                                              .dropna().str.strip().tolist())
                                if syms:
                                    print(f"  nse bhavcopy {d.date()}: {len(syms)} EQ symbols")
                                    self._set_cache(cache_key, syms)
                                    return syms
                    except Exception:
                        continue
        except ImportError:
            pass

        print("  ⚠️  Could not fetch NSE symbol list — using Nifty 50 fallback")
        return [
            "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK",
            "BAJAJ-AUTO","BAJFINANCE","BAJAJFINSV","BPCL","BHARTIARTL",
            "BRITANNIA","CIPLA","COALINDIA","DIVISLAB","DRREDDY",
            "EICHERMOT","GRASIM","HCLTECH","HDFCBANK","HDFCLIFE",
            "HEROMOTOCO","HINDALCO","HINDUNILVR","ICICIBANK","ITC",
            "INDUSINDBK","INFY","JSWSTEEL","KOTAKBANK","LT",
            "M&M","MARUTI","NTPC","NESTLEIND","ONGC",
            "POWERGRID","RELIANCE","SBILIFE","SHRIRAMFIN","SBIN",
            "SUNPHARMA","TCS","TATACONSUM","TATAMOTORS","TATASTEEL",
            "TECHM","TITAN","TRENT","ULTRACEMCO","WIPRO",
        ]

    def get_fno_symbols(self) -> list:
        """Get the list of F&O eligible stocks from NSE."""
        if _NSEPY_OK:
            try:
                fo = fnolist()
                if isinstance(fo, (list, pd.Series)):
                    return list(fo)
            except Exception:
                pass
        return []

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2: MARKET REGIME (live)
    # ─────────────────────────────────────────────────────────────────────────

    def get_regime(self) -> dict:
        """
        Determine the current market regime using multiple inputs:

        Primary signals:
          - Nifty 50 vs 200 DMA      (trend direction)
          - India VIX level           (fear / volatility)
          - FII net activity          (foreign institutional sentiment)

        Regime classification:
          BULL     : Nifty > 200 DMA, DMA upsloping, VIX < 18
          BULL_VOLATILE: Nifty > 200 DMA but VIX ≥ 18 (caution zone)
          BEAR     : Nifty < 200 DMA, DMA downsloping, VIX > 18
          BEAR_EXTREME: Nifty < 200 DMA, VIX > 25 (panic/crash zone)
          SIDEWAYS : Nifty within 1.5% of 200 DMA

        Returns dict with all inputs + final regime classification.
        """
        sources = RegimeSources(
            index_quote=nse_get_index_quote if _NSEPY_OK else None,
            download=yf.download if _YF_OK else None,
            india_vix=indiavix if _NSEPY_OK else None,
            fii_dii=nse_fiidii if _NSEPY_OK else None,
        )
        return build_regime(sources)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3: HISTORICAL OHLC
    # ─────────────────────────────────────────────────────────────────────────

    def get_ohlc(self, symbol: str, period: str = "1y",
                 start: date = None, end: date = None) -> pd.DataFrame:
        """
        Fetch OHLC history for one NSE stock.

        Priority:
          1. NSEpy get_history() — direct NSE, no rate limits
             (currently failing on Mac due to NSE SSL change; use when fixed)
          2. yfinance with .NS suffix — reliable, slight rate-limit risk
             Use smaller batches and sleep between calls.

        Returns DataFrame with columns: Open, High, Low, Close, Volume
        Index: DatetimeIndex (sorted ascending)
        """
        # Try NSEpy first (best quality, direct NSE data)
        if _NSEPY_LIB_OK and start and end:
            try:
                df = _nsepy_get_history(symbol=symbol, start=start, end=end)
                if df is not None and not df.empty:
                    return df[["Open","High","Low","Close","Volume"]].sort_index()
            except Exception:
                pass  # Fall through to yfinance

        # yfinance fallback
        if _YF_OK:
            try:
                kwargs = {"auto_adjust": True, "progress": False}
                if start and end:
                    kwargs["start"] = start
                    kwargs["end"]   = end
                else:
                    kwargs["period"] = period

                df = yf.download(f"{symbol}.NS", **kwargs)
                if isinstance(df.columns, pd.MultiIndex):
                    df = df.xs(f"{symbol}.NS", axis=1, level=1)
                if not df.empty:
                    return df[["Open","High","Low","Close","Volume"]].dropna().sort_index()
            except Exception as e:
                print(f"  yfinance failed for {symbol}: {e}")

        return pd.DataFrame()

    def get_ohlc_bulk(self, symbols: list, period: str = "1y",
                      batch_size: int = 100, sleep_sec: float = 2.0) -> dict:
        """
        Bulk OHLC download for multiple symbols via yfinance.
        Returns {symbol: DataFrame}.

        batch_size: reduce to 50-100 for 5y/10y to avoid rate limits.
        sleep_sec: pause between batches (increase to 3-5 for large universes).
        """
        result = {}
        tickers = [f"{s}.NS" for s in symbols]
        batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
        print(f"  Bulk OHLC: {len(symbols)} stocks | {len(batches)} batches "
              f"| period={period} | batch_size={batch_size}")

        for idx, batch in enumerate(batches, 1):
            print(f"    Batch {idx}/{len(batches)} ({len(batch)}) …", end=" ", flush=True)
            for attempt in range(3):
                try:
                    raw = yf.download(batch, period=period, auto_adjust=True,
                                      threads=True, progress=False)
                    if raw.empty:
                        print("empty"); break

                    if isinstance(raw.columns, pd.MultiIndex):
                        for t in batch:
                            sym = t.replace(".NS", "")
                            try:
                                df = raw.xs(t, axis=1, level=1).dropna(how="all")
                                if not df.empty and len(df) >= 30:
                                    result[sym] = df[["Open","High","Low","Close","Volume"]]
                            except KeyError:
                                pass
                    else:
                        sym = batch[0].replace(".NS", "")
                        if not raw.empty:
                            result[sym] = raw[["Open","High","Low","Close","Volume"]]

                    ok = sum(1 for t in batch if t.replace(".NS","") in result)
                    print(f"OK ({ok} usable)")
                    break

                except Exception as e:
                    if "Rate" in str(e) or "429" in str(e) or "Too Many" in str(e):
                        wait = 30 * (attempt + 1)
                        print(f"\n      Rate limited — waiting {wait}s …", end="", flush=True)
                        time.sleep(wait)
                    else:
                        print(f"ERROR: {e}"); break

            if idx < len(batches):
                time.sleep(sleep_sec)

        print(f"  Total: {len(result)}/{len(symbols)} tickers downloaded")
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4: FILING EVENTS CALENDAR
    # ─────────────────────────────────────────────────────────────────────────

    def get_upcoming_events(self, days_ahead: int = 30) -> pd.DataFrame:
        """
        Fetch upcoming corporate events from NSE via nsepython.nse_events().

        Events include:
          - Board meetings (results, dividends, fund raising)
          - AGM / EGM announcements
          - Corporate actions (ex-date, record date)

        This is the "trends in company filings to regulator" data source.
        Use to:
          1. Identify stocks with upcoming results (potential catalyst)
          2. Pre-filter for high-quality earnings dates for event-study backtest
          3. Confirm Darvas/Piotroski signals with imminent fundamental trigger

        Returns DataFrame: symbol | company | purpose | date
        """
        if not _NSEPY_OK:
            return pd.DataFrame()
        try:
            df = nse_events()
            if isinstance(df, pd.DataFrame) and not df.empty:
                # Parse dates and filter to window
                df["date_parsed"] = pd.to_datetime(df["date"], format="%d-%b-%Y",
                                                    errors="coerce")
                cutoff = datetime.today() + timedelta(days=days_ahead)
                df = df[df["date_parsed"] <= cutoff].copy()
                df["days_away"] = (df["date_parsed"] - datetime.today()).dt.days
                df = df.sort_values("date_parsed")
                return df[["symbol","company","purpose","date","days_away"]].reset_index(drop=True)
        except Exception as e:
            print(f"  nse_events error: {e}")
        return pd.DataFrame()

    def get_results_calendar(self, days_ahead: int = 30) -> pd.DataFrame:
        """
        Filter upcoming events to ONLY quarterly/annual results announcements.
        Use as the 'filing event dates' for the event-study backtest.
        """
        ev = self.get_upcoming_events(days_ahead)
        if ev.empty:
            return ev
        results_keywords = ["result", "financial", "quarterly", "annual", "q1","q2","q3","q4"]
        mask = ev["purpose"].str.lower().str.contains("|".join(results_keywords), na=False)
        return ev[mask].reset_index(drop=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5: INSTITUTIONAL ACTIVITY
    # ─────────────────────────────────────────────────────────────────────────

    def get_bulk_deals(self) -> pd.DataFrame:
        """
        Today's NSE bulk deals — institutional-size trades that crossed
        0.5% of outstanding shares. Strong signal of conviction buying/selling.

        Usage: confirm Darvas breakout signals — a breakout with a
        simultaneous bulk buy from a known institution is higher conviction.

        Returns DataFrame: Date | Symbol | Security Name | Client Name |
                           Buy/Sell | Quantity Traded | Trade Price
        """
        if not _NSEPY_OK:
            return pd.DataFrame()
        try:
            df = get_bulkdeals()
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception as e:
            print(f"  get_bulkdeals error: {e}")
        return pd.DataFrame()

    def get_block_deals(self) -> pd.DataFrame:
        """
        Today's block deals (≥ ₹10 Cr single trade, disclosed on block window).
        Higher threshold than bulk deals — strong institutional signals.
        """
        if not _NSEPY_OK:
            return pd.DataFrame()
        try:
            df = get_blockdeals()
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception as e:
            print(f"  get_blockdeals error: {e}")
        return pd.DataFrame()

    def get_fii_dii(self) -> pd.DataFrame:
        """
        FII and DII net buy/sell activity from NSE.
        Use as a macro-level sentiment filter:
          FII net buying > 0 → institutional tailwind → higher screener conviction
          FII net selling → headwind → tighten stop-losses

        Returns DataFrame with net FII/DII flows (₹ Crore).
        """
        if not _NSEPY_OK:
            return pd.DataFrame()
        try:
            df = nse_fiidii()
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        except Exception as e:
            print(f"  nse_fiidii error: {e}")
        return pd.DataFrame()

    def get_institutional_activity_score(self) -> dict:
        """
        Composite institutional activity score combining FII/DII + bulk deals.
        Score ranges from -3 (strong selling) to +3 (strong buying).

        Components:
          +1/-1 : FII net activity (5-day sum positive/negative)
          +1/-1 : DII net activity (supporting/opposing FII)
          +1/-1 : Bulk deal buy/sell ratio (today)

        Use as a multiplier on screener expected value:
          score = +3 → increase position size
          score = -3 → reduce or avoid position
        """
        fii_df = bulk_df = None
        try:
            fii_df = self.get_fii_dii()
        except Exception as exc:
            logger.debug("fii/dii unavailable: %s", exc)
        try:
            bulk_df = self.get_bulk_deals()
        except Exception as exc:
            logger.debug("bulk deals unavailable: %s", exc)
        return institutional_score(fii_df, bulk_df)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 6: MARKET BREADTH + SENTIMENT
    # ─────────────────────────────────────────────────────────────────────────

    def get_market_breadth(self) -> dict:
        """
        Market breadth using India VIX and Nifty 50 live data.

        Breadth metrics:
          - VIX level: fear gauge (>25 = high fear, <15 = complacency)
          - Nifty % from 52-week high: market strength indicator
          - % from 52-week low: recovery / oversold signal

        These complement the 200 DMA regime with short-term sentiment context.
        """
        return market_breadth(indiavix if _NSEPY_OK else None, nse_get_index_quote if _NSEPY_OK else None)

    def get_nifty_pe_pb(self) -> dict:
        """
        Nifty 50 P/E and P/B ratios — macro valuation context.
        Nifty P/E > 25 = expensive; < 18 = cheap; 18-25 = fair value.
        Use to calibrate screener thresholds for the current market cycle.
        """
        return nifty_pe_pb(CACHE_DIR / "nifty_pe_pb.json", self._cache_ttl,
                           _load_index_pe_pb_div if _NSEPY_OK else None)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 7: LIVE SCREENING INTEGRATION
    # ─────────────────────────────────────────────────────────────────────────

    def get_live_context(self) -> dict:
        """
        Comprehensive live market context combining all data sources.
        Run this once at the start of each screening session to calibrate
        which strategies and parameters to use today.

        Returns a single dict with all live context:
          - regime (BULL/BEAR/SIDEWAYS)
          - vix + vix_regime
          - fii/dii sentiment score
          - upcoming results calendar (next 7 days)
          - institutional activity score
          - strategy recommendations based on current conditions
        """
        return build_live_context(self)

    def print_live_context(self):
        """Pretty-print the live market context for the daily dashboard."""
        ctx = self.get_live_context()
        print_live_context_report(ctx)
        return ctx

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 8: CACHE HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _get_cache(self, key: str):
        entry = self._cache.get(key)
        if entry and (time.time() - entry["ts"] < self._cache_ttl):
            return entry["data"]
        return None

    def _set_cache(self, key: str, data):
        self._cache[key] = {"data": data, "ts": time.time()}


_get_strategy_recommendation = strategy_recommendation   # backward-compatible alias


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE FUNCTIONS (for backward compatibility with existing scripts)
# ═══════════════════════════════════════════════════════════════════════════════

_fetcher = None

def get_fetcher() -> NSEDataFetcher:
    """Get or create the singleton NSEDataFetcher."""
    global _fetcher
    if _fetcher is None:
        _fetcher = NSEDataFetcher()
    return _fetcher


def get_nse_symbols() -> list:
    """Get all NSE EQ symbols. Drop-in replacement for yfinance-based fetching."""
    return get_fetcher().get_all_symbols()


def get_live_regime() -> dict:
    """Get current market regime with VIX, FII/DII, and recommendations."""
    return get_fetcher().get_regime()


def get_upcoming_results(days: int = 7) -> pd.DataFrame:
    """Get stocks announcing results in the next N days."""
    return get_fetcher().get_results_calendar(days_ahead=days)


def get_institutional_confirmation(symbol: str) -> bool:
    """Check if a symbol had a bulk/block buy deal today."""
    bd = get_fetcher().get_bulk_deals()
    if bd.empty or "Symbol" not in bd.columns:
        return False
    sym_deals = bd[bd["Symbol"].str.upper() == symbol.upper()]
    if sym_deals.empty:
        return False
    buy_col = "Buy/Sell" if "Buy/Sell" in sym_deals.columns else None
    if buy_col:
        return (sym_deals[buy_col].str.upper() == "BUY").any()
    return not sym_deals.empty


# ═══════════════════════════════════════════════════════════════════════════════
# CLI — run as script for daily dashboard
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NSE live market context dashboard")
    parser.add_argument("--regime",  action="store_true", help="Show market regime only")
    parser.add_argument("--symbols", action="store_true", help="Show NSE symbol count")
    parser.add_argument("--events",  action="store_true", help="Show upcoming events (30d)")
    parser.add_argument("--bulk",    action="store_true", help="Show today's bulk deals")
    parser.add_argument("--full",    action="store_true", default=True, help="Full dashboard")
    args = parser.parse_args()

    fetcher = NSEDataFetcher()

    if args.symbols:
        syms = fetcher.get_all_symbols()
        print(f"NSE EQ Universe: {len(syms)} symbols")
        print(f"Sample: {syms[:10]}")

    if args.events:
        ev = fetcher.get_upcoming_events(30)
        print(f"\nUpcoming Events (30 days):")
        print(ev.to_string(index=False) if not ev.empty else "No events")

    if args.bulk:
        bd = fetcher.get_bulk_deals()
        print(f"\nToday's Bulk Deals ({len(bd)}):")
        print(bd.to_string(index=False) if not bd.empty else "No bulk deals today")

    if args.full or args.regime:
        fetcher.print_live_context()
