#!/usr/bin/env python3
"""
ticker_view.py
---------------
Ticker-wise data extraction + display dashboard (first-attempt BazaarTalks
feature): given one ticker, pulls its recent OHLC history plus every screener
signal the warehouse currently has for it — fundamentals, DVM technical/
composite classification, and the accumulation/CMF signal — and renders a
single-page "screener scorecard" so the different screens can be compared
side by side for that ticker, instead of running each script separately.

Same pure-function pattern as dashboard.py: `render_html` takes plain data,
returns an HTML string, and is unit-testable without a live warehouse.

Usage:
  python ticker_view.py AAPL --market US
  python ticker_view.py RELIANCE --market IN --open
"""
from __future__ import annotations

import argparse
import datetime as dt
import os

HERE = os.path.dirname(os.path.abspath(__file__))

_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2rem;color:#1a1a1a;background:#fafafa}
h1{font-size:1.5rem}h2{font-size:1.1rem;margin-top:2rem;border-bottom:2px solid #e0e0e0;padding-bottom:.3rem}
table{border-collapse:collapse;width:100%;margin:.5rem 0;background:#fff;font-size:.85rem}
th,td{border:1px solid #e5e5e5;padding:.35rem .6rem;text-align:right}
th{background:#f0f3f7;text-align:left}td:first-child,th:first-child{text-align:left}
tr:nth-child(even){background:#fbfcfd}.meta{color:#888;font-size:.8rem}
.badge{display:inline-block;background:#2b6cb0;color:#fff;border-radius:3px;padding:.1rem .4rem;font-size:.75rem}
.scorecard{display:flex;flex-wrap:wrap;gap:.75rem;margin:.5rem 0}
.card{background:#fff;border:1px solid #e5e5e5;border-radius:6px;padding:.75rem 1rem;min-width:150px}
.card .label{font-size:.75rem;color:#888;text-transform:uppercase}
.card .value{font-size:1.3rem;font-weight:600;margin-top:.15rem}
.card .value.na{color:#bbb;font-weight:400;font-size:1rem}
"""


def _card(label: str, value) -> str:
    if value is None:
        return f'<div class="card"><div class="label">{label}</div><div class="value na">n/a</div></div>'
    return f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div></div>'


def render_html(ticker: str, market: str, detail: dict, accum_row: dict | None,
                 generated: str | None = None) -> str:
    """
    detail: the dict returned by warehouse.ticker_detail() —
            {ticker, market, ohlc: [...], fundamentals: {...}|None,
             dvm_technical: {...}|None, dvm_composite: {...}|None}
    accum_row: one row from accumulation_screener.current_screen(), or None
    """
    generated = generated or dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    fund = detail.get("fundamentals") or {}
    dvm = detail.get("dvm_technical") or {}
    comp = detail.get("dvm_composite") or {}
    ohlc = detail.get("ohlc") or []

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{ticker} · {market}</title><style>{_CSS}</style></head><body>",
        f"<h1>{ticker} <span class='meta'>{market}</span> <span class='badge'>ticker view</span></h1>",
        f"<p class='meta'>generated {generated}</p>",
    ]

    # ── Screener scorecard: one card per signal, side by side ──────────────
    parts.append("<h2>Screener scorecard</h2><div class='scorecard'>")
    parts.append(_card("DVM composite", comp.get("label") or comp.get("code")))
    parts.append(_card("Momentum (M)", dvm.get("M")))
    parts.append(_card("RSI", dvm.get("rsi")))
    parts.append(_card("Accumulation (CMF)", None if accum_row is None else round(accum_row.get("cmf", 0), 2)))
    parts.append(_card("ROE %", fund.get("roe")))
    parts.append(_card("P/E", fund.get("pe")))
    parts.append(_card("D/E", fund.get("de")))
    parts.append("</div>")

    # ── Fundamentals ─────────────────────────────────────────────────────
    parts.append("<h2>Fundamentals</h2>")
    if fund:
        rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in fund.items() if k not in ("ticker", "market"))
        parts.append(f"<table>{rows}</table>")
    else:
        parts.append("<p class='meta'>no fundamentals cached for this ticker</p>")

    # ── DVM technical / composite ────────────────────────────────────────
    parts.append("<h2>DVM technical &amp; composite classification</h2>")
    merged = {**{f"tech_{k}": v for k, v in dvm.items() if k not in ("ticker", "market")},
              **{f"composite_{k}": v for k, v in comp.items() if k not in ("ticker", "market")}}
    if merged:
        rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in merged.items())
        parts.append(f"<table>{rows}</table>")
    else:
        parts.append("<p class='meta'>no DVM data cached for this ticker</p>")

    # ── Recent OHLC ───────────────────────────────────────────────────────
    parts.append(f"<h2>Recent OHLC ({len(ohlc)} bars)</h2>")
    if ohlc:
        head = "".join(f"<th>{k}</th>" for k in ohlc[0].keys())
        body = "".join(
            "<tr>" + "".join(f"<td>{v}</td>" for v in row.values()) + "</tr>" for row in ohlc
        )
        parts.append(f"<table><tr>{head}</tr>{body}</table>")
    else:
        parts.append("<p class='meta'>no OHLC history for this ticker/market</p>")

    parts.append("</body></html>")
    return "".join(parts)


def _fetch(ticker: str, market: str, bars: int) -> dict:
    import duckdb
    import warehouse
    con = duckdb.connect(os.path.join(HERE, "market.duckdb"))
    warehouse.build(con)
    con.execute("SET max_expression_depth=10000")
    try:
        return warehouse.ticker_detail(con, ticker, market, bars=bars)
    finally:
        con.close()


def _fetch_accum_row(ticker: str, market: str) -> dict | None:
    """Best-effort: the accumulation screen runs over a whole market's OHLC
    panel (no per-ticker entry point exists), so isolate failures the same
    way dashboard.py does — a screener hiccup never breaks the rest of the page."""
    try:
        import accumulation_screener as acc
        panel = acc.current_screen(market)
        if panel.empty:
            return None
        row = panel[panel["ticker"] == ticker]
        return row.iloc[0].to_dict() if not row.empty else None
    except Exception as e:                          # noqa: BLE001
        print(f"  [ticker_view] accumulation lookup skipped: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("--market", required=True)
    ap.add_argument("--bars", type=int, default=60, help="OHLC bars to include (default 60)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--no-accum", action="store_true", help="skip the accumulation/CMF lookup")
    args = ap.parse_args()

    ticker = args.ticker.upper()
    market = args.market.upper()
    out = args.out or f"ticker_{ticker}_{market}.html"

    detail = _fetch(ticker, market, args.bars)
    accum_row = None if args.no_accum else _fetch_accum_row(ticker, market)

    html = render_html(ticker, market, detail, accum_row)
    open(out, "w").write(html)
    print(f"wrote {out} ({len(html)//1024} KB)")
    if args.open:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(out)}")


if __name__ == "__main__":
    main()
