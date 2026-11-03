"""US fundamental screeners (Piotroski, US Coffee Can, Magic Formula, Bull Cartel) from one yfinance Ticker.

Split out of full_us_market_scan.py (which re-exports ``fundamental_scan``).
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf

from scan_common import piotroski_f_score
from stock_utils import first_df as _first_df, row as _row


def _fetch_statements(symbol: str) -> tuple:
    """(inc, bal, cf, inc_q, mcap, info) for one symbol via a single yfinance Ticker."""
    ticker = yf.Ticker(symbol)
    inc    = _first_df(ticker, "income_stmt", "financials")
    bal    = _first_df(ticker, "balance_sheet")
    cf     = _first_df(ticker, "cash_flow", "cashflow")
    inc_q  = _first_df(ticker, "quarterly_income_stmt", "quarterly_financials")
    try:
        mcap = ticker.fast_info.market_cap or 0
    except Exception:
        mcap = 0
    try:
        info = ticker.info or {}
    except Exception:
        info = {}
    return inc, bal, cf, inc_q, mcap, info


def _annual_series(df, *rows) -> list:
    """All non-NaN values of the first matching row across periods (newest first)."""
    for name_ in rows:
        if df is not None and name_ in df.index:
            return [float(v) for v in df.loc[name_].dropna() if pd.notna(v)]
    return []


def _us_coffee_can(inc, bal, cf, info: dict, mcap: float) -> tuple:
    """US Coffee Can (6 criteria): (qualifies, score text, rev CAGR %, avg ROE %)."""
    revs = _annual_series(inc, "Total Revenue")
    cagr = ((revs[0]/revs[-1])**(1/(len(revs)-1))-1)*100 if len(revs)>=2 and revs[-1]>0 else None
    ni_s = _annual_series(inc, "Net Income")
    eq_s = _annual_series(bal, "Stockholders Equity", "Total Stockholder Equity",
                          "Total Equity Gross Minority Interest")
    roe_l = [ni_s[i]/eq_s[i]*100 for i in range(min(len(ni_s),len(eq_s))) if eq_s[i]>0]
    avg_roe = sum(roe_l)/len(roe_l) if roe_l else None
    de_raw = info.get("debtToEquity")
    if de_raw is not None:
        de = de_raw/100 if de_raw > 10 else de_raw
    else:
        ltd_s = _annual_series(bal, "Long Term Debt")
        de = (ltd_s[0]/abs(eq_s[0])) if (ltd_s and eq_s and eq_s[0]!=0) else None
    fcf_s   = _annual_series(cf, "Free Cash Flow")
    ocf_s   = _annual_series(cf, "Operating Cash Flow", "Total Cash From Operating Activities")
    capex_s = _annual_series(cf, "Capital Expenditure", "Capital Expenditures")
    fcf_pos = (fcf_s[0]>0) if fcf_s else ((ocf_s[0]-abs(capex_s[0]))>0 if (ocf_s and capex_s) else False)
    cc_bits = [
        1 if (cagr    and cagr    > 10)  else 0,
        1 if (avg_roe and avg_roe > 15)  else 0,
        1 if (de is not None and de < 1) else 0,
        1 if mcap >= 1e9                 else 0,   # ≥ $1B
        1 if (ni_s and all(n>0 for n in ni_s)) else 0,
        1 if fcf_pos                     else 0,
    ]
    return sum(cc_bits) == 6, f"{sum(cc_bits)}/6", cagr, avg_roe


def _us_magic_formula(inc, bal, info: dict, mcap: float) -> tuple:
    """Magic Formula, US adapted (MCap > $50M): (qualifies, ROIC %, earnings yield %)."""
    a0  = _row(bal, "Total Assets", col=0)
    cl0 = _row(bal, "Current Liabilities", "Total Current Liabilities", col=0)
    ebit     = _row(inc, "EBIT", "Operating Income", "Ebit")
    cap_emp  = (a0 - (cl0 or 0)) if a0 else None
    tot_dbt  = info.get("totalDebt", 0) or 0
    cash_val = info.get("totalCash", 0) or 0
    ev       = (mcap + tot_dbt - cash_val) if mcap else None
    roic     = (ebit/cap_emp*100) if (ebit and cap_emp and cap_emp>0) else None
    ey       = (ebit/ev*100)       if (ebit and ev      and ev>0)     else None
    bv       = info.get("bookValue")
    return bool(roic and roic>25 and ey and ey>15 and bv and bv>0 and mcap>50e6), roic, ey


def _annual_screens(out: dict, inc, bal, cf, info: dict, mcap: float) -> None:
    """Fill Piotroski, US Coffee Can and Magic Formula fields (all from the annual statements)."""
    if inc is None:
        out.update({"f_score": None, "f_strong": False,
                    "qualifies_cc": False, "cc_score": "N/A",
                    "qualifies_mf": False, "ROIC_%": None, "Earnings_Yield_%": None,
                    "Revenue_CAGR_%": None, "ROE_avg_%": None})
        return
    f_score = piotroski_f_score(inc, bal, cf)
    out["f_score"]  = f_score
    out["f_strong"] = f_score >= 7

    qualifies_cc, cc_score, cagr, avg_roe = _us_coffee_can(inc, bal, cf, info, mcap)
    out["qualifies_cc"] = qualifies_cc
    out["cc_score"]     = cc_score
    out["Revenue_CAGR_%"] = round(cagr,    2) if cagr    is not None else None
    out["ROE_avg_%"]      = round(avg_roe,  2) if avg_roe is not None else None

    qualifies_mf, roic, ey = _us_magic_formula(inc, bal, info, mcap)
    out["qualifies_mf"]     = qualifies_mf
    out["ROIC_%"]           = round(roic, 2) if roic is not None else None
    out["Earnings_Yield_%"] = round(ey,   2) if ey   is not None else None


def _bull_cartel(out: dict, inc_q) -> None:
    """Bull Cartel: YoY quarterly sales >15% and profit >20% with net profit > $1M."""
    if inc_q is not None and len(inc_q.columns) >= 5:
        rev_q0 = _row(inc_q, "Total Revenue", col=0)
        rev_q4 = _row(inc_q, "Total Revenue", col=4)
        ni_q0  = _row(inc_q, "Net Income",    col=0)
        ni_q4  = _row(inc_q, "Net Income",    col=4)
        sales_g  = ((rev_q0-rev_q4)/abs(rev_q4)*100) if (rev_q0 and rev_q4 and rev_q4!=0) else None
        profit_g = ((ni_q0-ni_q4)/abs(ni_q4)*100)    if (ni_q0  and ni_q4  and ni_q4!=0)  else None
        ni_usd_m = ni_q0/1e6 if ni_q0 else None
        out["qualifies_bc"]        = bool(sales_g and sales_g>15
                                          and profit_g and profit_g>20
                                          and ni_usd_m and ni_usd_m>1)   # >$1M net profit
        out["Sales_Growth_YoY_%"]  = round(sales_g,  2) if sales_g  is not None else None
        out["Profit_Growth_YoY_%"] = round(profit_g, 2) if profit_g is not None else None
        out["Net_Profit_$M"]       = round(ni_usd_m, 2) if ni_usd_m is not None else None
    else:
        out.update({"qualifies_bc": False, "Sales_Growth_YoY_%": None,
                    "Profit_Growth_YoY_%": None, "Net_Profit_$M": None})


def fundamental_scan(symbol: str) -> dict:
    """
    Run ALL 5 fundamental screeners for one US symbol in a single Ticker() call:
      1. Piotroski F-Score  — 9-point accounting quality (≥7 = strong)
      2. US Coffee Can      — Rev CAGR>10%, ROE>15%, D/E<1, MCap≥$1B, no loss, FCF>0
      3. Magic Formula      — ROIC>25%, Earnings Yield>15%, MCap>$50M  (US-adapted)
      4. Bull Cartel        — YoY quarterly sales growth>15%, profit growth>20%
    """
    try:
        inc, bal, cf, inc_q, mcap, info = _fetch_statements(symbol)
        name   = info.get("shortName", "") or info.get("longName", "")
        sector = info.get("sector", "")
    except Exception as e:
        return {"symbol": symbol, "name": "", "sector": "", "error": str(e),
                "f_score": None, "f_strong": False,
                "qualifies_cc": False, "cc_score": "N/A",
                "qualifies_mf": False, "qualifies_bc": False}

    out = {"symbol": symbol, "name": name, "sector": sector, "error": ""}
    _annual_screens(out, inc, bal, cf, info, mcap)
    _bull_cartel(out, inc_q)
    return out
