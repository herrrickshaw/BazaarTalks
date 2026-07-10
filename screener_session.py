#!/usr/bin/env python3
# screener_session.py
# ====================
# Read-only screener.in access via a session cookie you obtain yourself by
# logging in through your own browser — this module never submits a password
# anywhere. Unlocks per-company financial exports (full P&L / balance sheet /
# cash flow / ratios) for the India fundamental screeners.
#
# SECURITY: log into screener.in yourself in a normal browser, then copy the
# `sessionid` and `csrftoken` cookie values (DevTools > Application > Cookies)
# into a local, gitignored .env file — never paste them into chat, never
# hard-code them:
#     SCREENER_SESSIONID=...
#     SCREENER_CSRFTOKEN=...
# Sessions expire; when this starts failing, just re-log-in and refresh the
# two values. Respect screener.in's Terms of Service and rate limits (this is
# for your own account's data, fetched politely).
#
#   from screener_session import session, company_financials
#   s = session()                       # authenticated requests.Session (cached)
#   f = company_financials("RELIANCE")  # -> fundamentals dict for strategies
#
# Parsing logic (page_ratios, company_financials) ported unchanged from the
# working-files-repo screener_in_auth.py — only session() changed, from a
# password POST to cookie reuse.

from __future__ import annotations

import io
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

_UA = {"User-Agent": "Mozilla/5.0 (research)"}
BASE = "https://www.screener.in"


class AuthError(RuntimeError):
    pass


def _load_dotenv() -> None:
    """Load SCREENER_* from a local, gitignored .env (KEY=VALUE lines) into the
    environment if not already set. Also checks the sibling working-files-repo
    location so both projects can share one .env."""
    candidates = [
        Path(__file__).parent / ".env",
        Path("/Users/umashankar/working-files-repo/python_files/.env"),
    ]
    for envf in candidates:
        if not envf.exists():
            continue
        for line in envf.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


@lru_cache(maxsize=1)
def session() -> requests.Session:
    """Return an authenticated session built from a cookie you obtained by
    logging in yourself — no password is ever sent by this module."""
    _load_dotenv()
    sid = os.environ.get("SCREENER_SESSIONID")
    csrf = os.environ.get("SCREENER_CSRFTOKEN")
    if not sid:
        raise AuthError(
            "set SCREENER_SESSIONID (and ideally SCREENER_CSRFTOKEN) in .env — "
            "log into screener.in yourself and copy the cookie value, never a password"
        )
    s = requests.Session()
    s.headers.update(_UA)
    s.cookies.set("sessionid", sid, domain="www.screener.in")
    if csrf:
        s.cookies.set("csrftoken", csrf, domain="www.screener.in")
    who = s.get(f"{BASE}/", timeout=25)
    if "logout" not in who.text.lower():
        raise AuthError("session cookie rejected or expired — re-log-in and refresh SCREENER_SESSIONID")
    return s


def _num(x):
    try:
        return float(str(x).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def company_export_bytes(symbol: str) -> Optional[bytes]:
    """Download the authenticated 'Export to Excel' workbook for a company.

    The button is a POST form: <button formaction="/user/company/export/<id>/">.
    We read that action + the CSRF token and POST to it."""
    s = session()
    page = s.get(f"{BASE}/company/{symbol}/", timeout=25)
    m = re.search(
        r'(?:formaction|action|href)=["\']([^"\']*company/export/\d+/?[^"\']*)["\']', page.text
    )
    if not m:
        return None
    url = m.group(1)
    if url.startswith("/"):
        url = BASE + url
    mt = re.search(r'name=["\']csrfmiddlewaretoken["\'] value=["\']([^"\']+)', page.text)
    tok = mt.group(1) if mt else s.cookies.get("csrftoken")
    headers = {"Referer": f"{BASE}/company/{symbol}/"}
    for how in ("post", "get"):
        try:
            if how == "post":
                r = s.post(url, data={"csrfmiddlewaretoken": tok}, headers=headers, timeout=40)
            else:
                r = s.get(url, headers=headers, timeout=40)
            if r.status_code == 200 and r.content[:2] == b"PK":  # xlsx = zip (PK)
                return r.content
        except Exception:
            continue
    return None


_RATIO_KEYS = {
    "market cap": "market_cap_cr",
    "current price": "current_price",
    "stock p/e": "pe",
    "p/e": "pe",
    "dividend yield": "dividend_yield",
    "roce": "roce_page",
    "roe": "roe_page",
    "book value": "book_value",
    "face value": "face_value",
}


def page_ratios(symbol: str) -> dict:
    """Top-of-page ratios from the company page (market cap, P/E, ROE, ROCE,
    dividend yield, current price) — the market/valuation inputs the export lacks."""
    try:
        html = session().get(f"{BASE}/company/{symbol}/", timeout=25).text
    except Exception:
        return {}
    out = {}
    for name, num in re.findall(
        r'<span class="name">\s*([^<]+?)\s*</span>.*?<span class="(?:nowrap )?value">(.*?)</span>',
        html,
        re.S,
    ):
        key = _RATIO_KEYS.get(name.strip().lower())
        if not key:
            continue
        v = _num(re.sub(r"<[^>]+>", "", num))
        if v is not None:
            out[key] = v
    return out


def company_financials(symbol: str) -> dict:
    """Parse the screener.in export + company-page ratios into a strategy-ready
    fundamentals dict."""
    raw = company_export_bytes(symbol)
    if not raw:
        return {}
    try:
        xl = pd.ExcelFile(io.BytesIO(raw))
        sheet = next((s for s in xl.sheet_names if "data" in s.lower()), xl.sheet_names[0])
        df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet, header=None)
    except Exception:
        return {}

    def row(*labels):
        for lab in labels:
            m = df[
                df.apply(
                    lambda r: r.astype(str).str.strip().str.lower().eq(lab.lower()).any(), axis=1
                )
            ]
            if not m.empty:
                vals = [_num(v) for v in m.iloc[0].tolist()[1:] if _num(v) is not None]
                if vals:
                    return vals
        return []

    def g(lst, i=0):
        return lst[i] if len(lst) > i else None

    ni = row("Net profit", "Net Profit")
    debt = row("Borrowings", "Borrowings+", "Total Debt")
    reserves = row("Reserves")
    capital = row("Equity Capital", "Equity Share Capital", "Share Capital")
    rev = row("Sales", "Sales+", "Revenue", "Total Revenue")

    eq = (
        [
            (capital[i] if i < len(capital) else 0) + (reserves[i] if i < len(reserves) else 0)
            for i in range(max(len(capital), len(reserves)))
        ]
        if (capital or reserves)
        else []
    )

    def pct(a, b):
        return (a - b) / abs(b) * 100 if (a is not None and b not in (None, 0)) else None

    r = page_ratios(symbol)
    out = {
        "source": "screener.in",
        "ticker": symbol,
        "pe": r.get("pe"),
        "roe": r.get("roe_page") or (round(g(ni) / g(eq) * 100, 1) if g(eq) else None),
        "roce": r.get("roce_page"),
        "de": (round(g(debt) / g(eq), 2) if g(eq) else None),
        "mktcap": r.get("market_cap_cr"),
        "div_yield": r.get("dividend_yield"),
        "rev_growth": pct(g(rev), g(rev, 1)),
        "earn_growth": pct(g(ni), g(ni, 1)),
        "book_value": r.get("book_value"),
    }
    return {k: v for k, v in out.items() if v is not None}


def fundamentals_batch(symbols, pause: float = 1.0, verbose: bool = True) -> dict:
    """Rate-limited batch fetch — politeness pause between requests."""
    import time

    out = {}
    for i, sym in enumerate(symbols):
        try:
            out[sym] = company_financials(sym)
        except Exception as e:
            out[sym] = {"error": str(e)}
        if verbose:
            print(f"  [{i+1}/{len(symbols)}] {sym}: {'ok' if out[sym] else 'empty'}")
        if i < len(symbols) - 1:
            time.sleep(pause)
    return out


if __name__ == "__main__":
    import sys

    sym = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE"
    try:
        session()
        print("logged in ✓")
    except AuthError as e:
        print(f"AUTH: {e}")
        sys.exit(1)
    print(company_financials(sym))
