#!/usr/bin/env python3
# trendlyne_session.py
# =====================
# Read-only Trendlyne access via a session cookie you obtain yourself by
# logging in through your own browser — this module never submits a password
# anywhere, and never automates the login form (Trendlyne's login is behind
# reCAPTCHA v3, which is specifically designed to detect/block headless
# automation — respecting that boundary rather than trying to defeat it).
#
# Trendlyne is a JS-rendered SPA (unlike screener.in's server-rendered Django
# pages), so this uses Playwright (a real browser context) rather than plain
# `requests` — the site's own API endpoints (getStockMetricParameterList,
# member/api/ac_snames) work fine once the browser has the right cookies and
# has navigated once to establish a Referer.
#
# SECURITY: log into trendlyne.com yourself in a normal browser, then copy the
# session cookie (DevTools > Application > Cookies > trendlyne.com — look for
# an HttpOnly+Secure cookie, commonly named `.trendlyne` or `sessionid`) and
# `csrftoken` into a local, gitignored .env file — never paste them into chat,
# never hard-code them:
#     TRENDLYNE_SESSIONID=...
#     TRENDLYNE_CSRFTOKEN=...
# Sessions expire; when this starts failing, re-log-in and refresh the value.
#
#   from trendlyne_session import session, fundamentals
#   ctx = session()                      # authenticated Playwright context (cached)
#   f = fundamentals("RELIANCE", ctx)    # -> fundamentals dict (pe, pb, mktcap, sector)

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

BASE = "https://trendlyne.com"
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# module-level Playwright handles — kept alive for the process lifetime,
# closed explicitly via close() if needed
_pw = None
_browser = None
_context = None
_home_page = None

# ticker -> numeric Trendlyne stock id (populated by _resolve_id, avoids a
# repeat autocomplete call for the same symbol within one run)
_ID_CACHE: dict = {}


class AuthError(RuntimeError):
    pass


def _load_dotenv() -> None:
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


def session():
    """Return an authenticated Playwright BrowserContext built from a cookie you
    obtained by logging in yourself — no password/login form is ever touched
    by this module. Cached for the process lifetime."""
    global _pw, _browser, _context, _home_page
    if _context is not None:
        return _context

    _load_dotenv()
    sid = os.environ.get("TRENDLYNE_SESSIONID")
    csrf = os.environ.get("TRENDLYNE_CSRFTOKEN")
    if not sid:
        raise AuthError(
            "set TRENDLYNE_SESSIONID (and ideally TRENDLYNE_CSRFTOKEN) in .env — "
            "log into trendlyne.com yourself and copy the session cookie, never a password"
        )

    from playwright.sync_api import sync_playwright

    _pw = sync_playwright().start()
    _browser = _pw.chromium.launch(headless=True)
    _context = _browser.new_context(user_agent=_UA)
    cookies = [{"name": ".trendlyne", "value": sid, "domain": "trendlyne.com", "path": "/"}]
    if csrf:
        cookies.append({"name": "csrftoken", "value": csrf, "domain": "trendlyne.com", "path": "/"})
    _context.add_cookies(cookies)

    _home_page = _context.new_page()
    _home_page.goto(BASE + "/", timeout=30000, wait_until="domcontentloaded")
    _home_page.wait_for_timeout(2500)

    has_login_btn = any(
        "login / sign up" in (t or "").strip().lower()
        for t in _home_page.eval_on_selector_all(
            "a, button", "els => els.map(e => e.textContent)"
        )
    )
    if has_login_btn:
        close()
        raise AuthError(
            "session cookie rejected or expired — re-log-in on trendlyne.com "
            "and refresh TRENDLYNE_SESSIONID"
        )
    return _context


def close() -> None:
    """Explicitly tear down the cached browser (optional; also fine to just let the process exit)."""
    global _pw, _browser, _context, _home_page
    try:
        if _browser:
            _browser.close()
        if _pw:
            _pw.stop()
    except Exception:
        pass
    _pw = _browser = _context = _home_page = None
    _ID_CACHE.clear()


def _resolve_id(symbol: str) -> Optional[int]:
    """NSE ticker -> numeric Trendlyne stock id, via the site's own autocomplete API."""
    if symbol in _ID_CACHE:
        return _ID_CACHE[symbol]
    session()  # ensure logged in / _home_page exists
    resp = _home_page.request.get(
        f"{BASE}/member/api/ac_snames/all/?all-results=true&term={symbol}",
        headers={"Referer": BASE + "/", "X-Requested-With": "XMLHttpRequest"},
        timeout=20000,
    )
    if resp.status != 200:
        return None
    try:
        results = resp.json()
    except Exception:
        return None
    for r in results:
        if r.get("NSEcode") == symbol or r.get("stock_code") == symbol:
            _ID_CACHE[symbol] = r["k"]
            return r["k"]
    if results:  # fall back to the top hit
        _ID_CACHE[symbol] = results[0]["k"]
        return results[0]["k"]
    return None


# getStockMetricParameterList keys -> our target fundamentals schema
_METRIC_MAP = {
    "PE_TTM": "pe",
    "PBV_A": "pb",
    "MCAP_Q": "mktcap",
    "INSTIHOLD": "instihold",
}


def fundamentals(symbol: str, ctx=None) -> dict:
    """Fundamentals for one ticker via Trendlyne's stock-metric API.

    Covers PE/PB/market-cap/institutional-holding directly; ROE/D-E/growth are
    not exposed by this particular endpoint — screener_session.company_financials
    is the richer fallback for those (see live_fundamentals.py's priority order).
    """
    ctx = ctx or session()
    sid = _resolve_id(symbol)
    if sid is None:
        return {}

    resp = _home_page.request.get(f"{BASE}/equity/getStockMetricParameterList/{sid}/", timeout=20000)
    if resp.status != 200:
        return {}
    try:
        body = resp.json().get("body", {})
    except Exception:
        return {}

    out = {"source": "trendlyne", "ticker": symbol}
    for tl_key, our_key in _METRIC_MAP.items():
        entry = body.get(tl_key)
        if entry and entry.get("value") is not None:
            out[our_key] = entry["value"]
    return {k: v for k, v in out.items() if v is not None} if len(out) > 2 else {}


if __name__ == "__main__":
    import sys

    sym = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE"
    try:
        session()
        print("logged in ✓")
    except AuthError as e:
        print(f"AUTH: {e}")
        sys.exit(1)
    print(fundamentals(sym))
    close()
