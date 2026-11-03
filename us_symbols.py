"""US symbol universe: NASDAQ Data Link (NCDS) -> NASDAQ Trader FTP -> SEC EDGAR fallbacks, with a 24 h cache.

Split out of full_us_market_scan.py (which re-exports every public name here).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

try:
    from ncdssdk import NCDSClient as _NCDSClient
    _NCDS_AVAILABLE = True
except ImportError:
    _NCDS_AVAILABLE = False

_NCDS_CREDS_PATH = Path.home() / ".nasdaq" / "ncds_auth.json"

DOWNLOAD_DIR = Path("./us_full_scan")
DOWNLOAD_DIR.mkdir(exist_ok=True)

SYMBOL_CACHE_TTL     = 86400
NASDAQ_FTP_BASE      = "https://ftp.nasdaqtrader.com/dynamic/SymDir"
NASDAQ_FTP_BASE_HTTP = "http://ftp.nasdaqtrader.com/dynamic/SymDir"

_SYMBOL_CACHE = DOWNLOAD_DIR / ".symbols_cache.json"


def _load_symbol_cache():
    try:
        if _SYMBOL_CACHE.exists():
            data = json.loads(_SYMBOL_CACHE.read_text())
            if time.time() - data.get("ts", 0) < SYMBOL_CACHE_TTL:
                return data.get("nasdaq", []), data.get("nyse", [])
    except Exception:
        pass
    return None, None


def _save_symbol_cache(nasdaq_syms, nyse_syms):
    try:
        _SYMBOL_CACHE.write_text(json.dumps({"ts": time.time(), "nasdaq": nasdaq_syms, "nyse": nyse_syms}))
    except Exception:
        pass


# ── Symbol fetch ───────────────────────────────────────────────────────────────

def _load_ncds_creds():
    """Load Nasdaq NCDS credentials from ~/.nasdaq/ncds_auth.json, or None."""
    if not _NCDS_AVAILABLE or not _NCDS_CREDS_PATH.exists():
        return None
    try:
        return json.loads(_NCDS_CREDS_PATH.read_text())
    except Exception:
        return None


def fetch_symbols_from_ncds() -> tuple:
    """
    Fetch NASDAQ + NYSE symbol lists via Nasdaq Cloud Data Service SDK.
    Consumes SeqDirectoryMessage events from the configured Kafka topic.
    Returns (nasdaq_list, nyse_list) or ([], []) if unavailable.
    """
    creds = _load_ncds_creds()
    if not creds:
        return [], []

    oauth = creds.get("oauth", {})
    kafka = creds.get("kafka", {})
    topic = creds.get("topic", "NLSPLUS-CTA-V4")

    security_cfg = {
        "oauth.token.endpoint.uri": oauth.get("endpoint", ""),
        "oauth.client.id":          oauth.get("client_id", ""),
        "oauth.client.secret":      oauth.get("client_secret", ""),
    }
    kafka_cfg = {
        "bootstrap.servers": kafka.get("bootstrap_servers", ""),
        "auto.offset.reset": "earliest",
    }

    try:
        client   = _NCDSClient(security_cfg, kafka_cfg)
        consumer = client.ncds_kafka_consumer(topic)
        nasdaq, nyse = [], []
        empty_polls  = 0
        while empty_polls < 5:
            msgs = consumer.consume(num_messages=500, timeout=3)
            if not msgs:
                empty_polls += 1
                continue
            empty_polls = 0
            for msg in msgs:
                val = msg.value() if callable(msg.value) else msg.value
                if not isinstance(val, dict):
                    continue
                if val.get("schema_name") != "SeqDirectoryMessage":
                    continue
                sym   = str(val.get("symbol", "")).strip().upper()
                mkt   = str(val.get("marketClass", "")).strip().upper()
                if not sym or any(c in sym for c in ("^", "/", "$", ".")):
                    continue
                if mkt in ("Q", "G", "S", ""):    # NASDAQ market tiers
                    nasdaq.append(sym)
                else:
                    nyse.append(sym)
        consumer.close()
        return list(dict.fromkeys(nasdaq)), list(dict.fromkeys(nyse))
    except Exception as e:
        print(f"  ⚠️  NCDS SDK error: {e}")
        return [], []


def _parse_nasdaq_file(text: str, exchange_filter=None) -> list[str]:
    """
    Parse a NASDAQ FTP symbol file (pipe-delimited).
    nasdaqlisted.txt columns: Symbol|Security Name|Market Category|Test Issue|...
    otherlisted.txt columns:  ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|...
    """
    symbols = []
    lines   = text.strip().splitlines()
    if not lines:
        return symbols

    header = [h.strip().lower() for h in lines[0].split("|")]
    # Find relevant column indices
    sym_idx  = next((i for i, h in enumerate(header) if h in ("symbol", "act symbol")), 0)
    test_idx = next((i for i, h in enumerate(header) if "test" in h), None)
    etf_idx  = next((i for i, h in enumerate(header) if h == "etf"), None)
    exch_idx = next((i for i, h in enumerate(header) if h == "exchange"), None)

    for line in lines[1:]:
        if line.startswith("File Creation Time"):
            continue
        parts = line.split("|")
        if len(parts) <= sym_idx:
            continue
        sym = parts[sym_idx].strip()
        if not sym or len(sym) > 5:          # skip blanks and long symbols (warrants etc.)
            continue
        if any(c in sym for c in ("^", "/", "$", ".")):  # skip special securities
            continue
        if test_idx and test_idx < len(parts) and parts[test_idx].strip().upper() == "Y":
            continue
        if etf_idx and etf_idx < len(parts) and parts[etf_idx].strip().upper() == "Y":
            continue
        if exchange_filter and exch_idx and exch_idx < len(parts):
            if parts[exch_idx].strip().upper() != exchange_filter.upper():
                continue
        symbols.append(sym)
    return symbols


def _fetch_nasdaq_file(filename: str) -> str:
    """Try HTTPS first, fall back to HTTP for NASDAQ FTP files."""
    headers = {"User-Agent": "Mozilla/5.0 StockScanner"}
    for base in (NASDAQ_FTP_BASE, NASDAQ_FTP_BASE_HTTP):
        try:
            resp = requests.get(f"{base}/{filename}", timeout=20, headers=headers)
            resp.raise_for_status()
            if resp.text:
                return resp.text
        except Exception:
            continue
    return ""


def fetch_all_us_symbols_from_sec() -> tuple:
    """Fetch NASDAQ + NYSE symbols from SEC EDGAR company_tickers_exchange.json."""
    url = "https://www.sec.gov/files/company_tickers_exchange.json"
    headers = {"User-Agent": "StockScanner umashankartd1991@gmail.com"}
    try:
        resp = requests.get(url, timeout=30, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("data", [])
        # fields: [cik, name, ticker, exchange]
        nasdaq, nyse = [], []
        for row in rows:
            ticker   = str(row[2]).strip().upper() if len(row) > 2 else ""
            exchange = str(row[3]).strip().lower() if len(row) > 3 else ""
            if not ticker or any(c in ticker for c in [" ", ".", "^"]):
                continue
            if "nasdaq" in exchange:
                nasdaq.append(ticker)
            elif "nyse" in exchange or "amex" in exchange or "arca" in exchange:
                nyse.append(ticker)
        return nasdaq, nyse
    except Exception as e:
        print(f"  ⚠️  SEC EDGAR fetch failed: {e}")
        return [], []


_ncds_symbol_cache: dict = {}   # populated once; shared by both fetch functions


def fetch_nasdaq_symbols() -> list[str]:
    """
    Fetch NASDAQ-listed equity symbols.
    Priority: Nasdaq NCDS SDK → NASDAQ FTP → SEC EDGAR.
    """
    if _NCDS_AVAILABLE and _load_ncds_creds() and not _ncds_symbol_cache:
        print("  Trying Nasdaq Cloud Data Service SDK …")
        nasdaq, nyse = fetch_symbols_from_ncds()
        _ncds_symbol_cache["nasdaq"] = nasdaq
        _ncds_symbol_cache["nyse"]   = nyse
        if nasdaq:
            print(f"  NCDS: {len(nasdaq)} NASDAQ symbols")
            return nasdaq
        print("  ⚠️  NCDS returned no NASDAQ symbols — trying FTP")

    text = _fetch_nasdaq_file("nasdaqlisted.txt")
    if text:
        syms = _parse_nasdaq_file(text)
        if syms:
            return syms
    print("  ⚠️  NASDAQ FTP unavailable — using SEC EDGAR fallback")
    nasdaq, _ = fetch_all_us_symbols_from_sec()
    return nasdaq


def fetch_nyse_symbols() -> list[str]:
    """
    Fetch NYSE/AMEX symbols.
    Priority: Nasdaq NCDS SDK (cached) → NASDAQ FTP → SEC EDGAR.
    """
    if _ncds_symbol_cache.get("nyse"):
        return _ncds_symbol_cache["nyse"]

    text = _fetch_nasdaq_file("otherlisted.txt")
    if text:
        syms = _parse_nasdaq_file(text)
        if syms:
            return syms
    print("  ⚠️  NYSE FTP unavailable — using SEC EDGAR fallback")
    _, nyse = fetch_all_us_symbols_from_sec()
    return nyse
