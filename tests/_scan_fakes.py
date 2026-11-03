"""Deterministic fakes for the full_*_market_scan characterisation tests (no network, no Cassandra)."""
from __future__ import annotations

import re
import zlib

import numpy as np
import pandas as pd


def make_ohlc(kind: str, seed: int, n: int = 260, base: float = 100.0) -> pd.DataFrame:
    """Synthetic High/Low/Close/Volume frame; ``kind`` shapes the final bars (breakout/breakdown/range/walk)."""
    rng = np.random.RandomState(seed)
    idx = pd.bdate_range("2025-01-01", periods=n)
    if kind == "short":
        n = 40
        idx = idx[:n]
    t = np.arange(n)
    if kind in ("breakout", "breakdown", "range", "short", "penny"):
        close = base + 5 * np.sin(t / 3.0) + rng.normal(0, 0.3, n)
        if kind == "breakout":
            close[-1] = base + 25
        elif kind == "breakdown":
            close[-1] = base - 25
    elif kind == "flat":
        close = np.full(n, base)
    else:
        close = base * np.exp(np.cumsum(rng.normal(0.0006, 0.015, n)))
    if kind == "penny":
        close = close / 200.0
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    vol = rng.uniform(0.8, 1.2, n) * 1e6
    if kind == "breakout":
        vol[-1] *= 3
    return pd.DataFrame({"High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


KINDS = ["breakout", "breakout", "breakdown", "range", "walk", "walk", "short", "penny", "breakout", "flat"]


def ohlc_panel(tickers: list) -> dict:
    return {t: make_ohlc(KINDS[i % len(KINDS)], i) for i, t in enumerate(tickers)}


def _stmt(rows: dict, ncols: int, rng, scale: float = 1.0) -> pd.DataFrame:
    cols = pd.to_datetime([f"{2025 - i}-03-31" for i in range(ncols)])
    data = {}
    for name, level in rows.items():
        growth = 1 + rng.uniform(-0.08, 0.2)
        data[name] = [level * scale * growth ** (-i) * rng.uniform(0.95, 1.05) for i in range(ncols)]
    return pd.DataFrame(data, index=cols).T


class FakeTicker:
    """yfinance.Ticker stand-in; the symbol's crc32 picks one of four statement shapes."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        seed = zlib.crc32(symbol.encode())
        self.variant = seed % 5
        rng = np.random.RandomState(seed % (2**31))
        if self.variant == 3:
            raise RuntimeError("ticker boom")
        if self.variant == 4:
            self.income_stmt = self.balance_sheet = self.cash_flow = self.quarterly_income_stmt = None
            self.financials = self.cashflow = self.quarterly_financials = pd.DataFrame()
            self.info = {}
            self.fast_info = type("F", (), {"market_cap": 0})()
            return
        bad = self.variant == 1
        self.income_stmt = _stmt({"Net Income": 8e9, "Total Revenue": 9e10, "Gross Profit": 3e10,
                                  "EBIT": 1.2e10, "Operating Income": 1.2e10}, 4, rng, -1 if bad else 1)
        self.balance_sheet = _stmt({"Total Assets": 1e11, "Long Term Debt": 2e10, "Current Assets": 4e10,
                                    "Current Liabilities": 2e10, "Share Issued": 1e9, "Stockholders Equity": 6e10,
                                    "Total Current Assets": 4e10, "Total Current Liabilities": 2e10}, 4, rng)
        self.cash_flow = _stmt({"Operating Cash Flow": 1.1e10, "Free Cash Flow": 6e9, "Capital Expenditure": -4e9},
                               4, rng, -1 if bad else 1)
        self.quarterly_income_stmt = _stmt({"Total Revenue": 3e10, "Net Income": 3e9}, 6, rng)
        self.info = {"debtToEquity": 35.0 + 20 * (seed % 3), "totalDebt": 2e10, "totalCash": 1e10, "bookValue": 40.0,
                     "trailingPE": 18.5, "forwardPE": 16.0, "priceToBook": 3.1, "pegRatio": 1.4,
                     "dividendYield": 0.021, "marketCap": 4e12, "trailingEps": 5.5, "regularMarketPrice": 100.0,
                     "longName": f"{symbol} Corp", "sector": "Technology"}
        self.fast_info = type("F", (), {"market_cap": 4e12 + 1e11 * (seed % 4)})()


_FILE_STAMP = re.compile(r"_\d{8}_\d{4}")
_STAMP = re.compile(r"\d{2} [A-Za-z]{3} \d{4}  \d{2}:\d{2}:\d{2}")


def scrub(text: str, tmp) -> str:
    """Remove wall-clock stamps and the temp dir from captured stdout."""
    text = text.replace(str(tmp), "<TMP>").replace(str(tmp.resolve()), "<TMP>")
    return _FILE_STAMP.sub("_<DATE>", _STAMP.sub("<TS>", text))


def read_workbooks(directory) -> dict:
    """{file-stem-without-timestamp: {sheet: list-of-row-dicts}} for every xlsx written under ``directory``."""
    from pathlib import Path
    out = {}
    for p in sorted(Path(directory).rglob("*.xlsx")):
        stem = re.sub(r"_\d{8}_\d{4}$", "", p.stem)
        sheets = pd.read_excel(p, sheet_name=None)
        out[stem] = {k: df.astype(object).where(df.notna(), None).to_dict("records") for k, df in sheets.items()}
    return out
