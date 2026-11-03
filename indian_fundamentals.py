"""India (NSE/BSE) fundamental screeners: Piotroski, Coffee Can, Magic Formula, Bull Cartel from one yfinance Ticker.

Split out of full_indian_market_scan.py (which re-exports ``fundamental_scan``).
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf

from scan_common import piotroski_f_score
from stock_utils import first_df as _first_df, row as _row


def _fetch_statements(symbol: str, yf_suffix: str) -> tuple:
    """(inc, bal, cf, inc_q, mcap, info) for one symbol via a single yfinance Ticker."""
    ticker = yf.Ticker(f"{symbol}{yf_suffix}")
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


def _failed_scan(symbol: str, error: str) -> dict:
    """Result row for a ticker whose statements could not be fetched."""
    return {"symbol": symbol, "error": error,
            "f_score": None, "f_strong": False,
            "qualifies_cc": False, "cc_score": "N/A",
            "qualifies_mf": False, "qualifies_bc": False,
            "ROIC_%": None, "Earnings_Yield_%": None,
            "Sales_Growth_YoY_%": None, "Profit_Growth_YoY_%": None}


def _annual_series(df, *rows) -> list:
    """All non-NaN values of the first matching row across periods (newest first)."""
    for name in rows:
        if df is not None and name in df.index:
            return [float(v) for v in df.loc[name].dropna() if pd.notna(v)]
    return []


def _coffee_can(inc, bal, info: dict, mcap: float) -> tuple:
    """Coffee Can bits: (qualifies, score text, rev CAGR %, avg ROCE %)."""
    revs = _annual_series(inc, "Total Revenue")
    cagr = ((revs[0]/revs[-1])**(1/(len(revs)-1))-1)*100 if len(revs)>=2 and revs[-1]>0 else None

    ebit_s = _annual_series(inc, "EBIT", "Operating Income", "Ebit")
    ta_s   = _annual_series(bal, "Total Assets")
    cl_s   = _annual_series(bal, "Current Liabilities", "Total Current Liabilities")
    roce_l = [ebit_s[i]/(ta_s[i]-cl_s[i])*100
              for i in range(min(len(ebit_s),len(ta_s),len(cl_s))) if (ta_s[i]-cl_s[i])>0]
    avg_roce = sum(roce_l)/len(roce_l) if roce_l else None

    de_raw = info.get("debtToEquity")
    if de_raw is not None:
        de = de_raw/100 if de_raw > 10 else de_raw
    else:
        ltd_s = _annual_series(bal, "Long Term Debt")
        eq_s  = _annual_series(bal, "Stockholders Equity", "Total Stockholder Equity",
                               "Total Equity Gross Minority Interest")
        de = (ltd_s[0]/eq_s[0]) if (ltd_s and eq_s and eq_s[0]!=0) else None

    mcap_cr = mcap / 1e7
    ni_s    = _annual_series(inc, "Net Income")
    cc_bits = [
        1 if (cagr    and cagr    > 10)  else 0,
        1 if (avg_roce and avg_roce > 15) else 0,
        1 if (de is not None and de < 1)  else 0,
        1 if mcap_cr >= 500               else 0,
        1 if (ni_s and all(n > 0 for n in ni_s)) else 0,
    ]
    return sum(cc_bits) == 5, f"{sum(cc_bits)}/5", cagr, avg_roce


def _magic_formula(inc, bal, info: dict, mcap: float) -> tuple:
    """Magic Formula: (qualifies, ROIC %, earnings yield %) reusing the annual statements."""
    a0  = _row(bal, "Total Assets", col=0)
    cl0 = _row(bal, "Current Liabilities", "Total Current Liabilities", col=0)
    ebit      = _row(inc, "EBIT", "Operating Income", "Ebit")
    cap_emp   = (a0 - (cl0 or 0)) if a0 else None
    total_dbt = info.get("totalDebt", 0) or 0
    cash_val  = info.get("totalCash", 0) or 0
    ev        = (mcap + total_dbt - cash_val) if mcap else None
    roic      = (ebit/cap_emp*100) if (ebit and cap_emp and cap_emp>0) else None
    ey        = (ebit/ev*100)       if (ebit and ev      and ev>0)     else None
    bv        = info.get("bookValue")
    mcap_cr   = mcap / 1e7
    qualifies = bool(roic and roic>25 and ey and ey>15 and bv and bv>0 and mcap_cr>15)
    return qualifies, roic, ey


def _annual_screens(out: dict, inc, bal, cf, info: dict, mcap: float) -> None:
    """Fill Piotroski, Coffee Can and Magic Formula fields (all from the annual statements)."""
    if inc is None:
        out.update({"f_score": None, "f_strong": False,
                    "qualifies_cc": False, "cc_score": "N/A",
                    "qualifies_mf": False, "ROIC_%": None, "Earnings_Yield_%": None,
                    "Revenue_CAGR_%": None, "ROCE_avg_%": None})
        return
    f_score = piotroski_f_score(inc, bal, cf)
    out["f_score"]  = f_score
    out["f_strong"] = f_score >= 7

    qualifies_cc, cc_score, cagr, avg_roce = _coffee_can(inc, bal, info, mcap)
    out["qualifies_cc"] = qualifies_cc
    out["cc_score"]     = cc_score
    out["Revenue_CAGR_%"] = round(cagr,     2) if cagr     is not None else None
    out["ROCE_avg_%"]     = round(avg_roce,  2) if avg_roce is not None else None

    qualifies_mf, roic, ey = _magic_formula(inc, bal, info, mcap)
    out["qualifies_mf"]      = qualifies_mf
    out["ROIC_%"]            = round(roic, 2) if roic is not None else None
    out["Earnings_Yield_%"]  = round(ey,   2) if ey   is not None else None


def _bull_cartel(out: dict, inc_q) -> None:
    """Bull Cartel: YoY quarterly sales >15% and profit >20% with net profit > Rs 1 Cr."""
    if inc_q is not None and len(inc_q.columns) >= 5:
        rev_q0 = _row(inc_q, "Total Revenue", col=0)
        rev_q4 = _row(inc_q, "Total Revenue", col=4)
        ni_q0  = _row(inc_q, "Net Income",    col=0)
        ni_q4  = _row(inc_q, "Net Income",    col=4)
        sales_g  = ((rev_q0-rev_q4)/abs(rev_q4)*100) if (rev_q0 and rev_q4 and rev_q4!=0) else None
        profit_g = ((ni_q0-ni_q4)/abs(ni_q4)*100)   if (ni_q0  and ni_q4  and ni_q4!=0)  else None
        ni_cr    = ni_q0 / 1e7 if ni_q0 else None
        out["qualifies_bc"]        = bool(sales_g and sales_g>15
                                          and profit_g and profit_g>20
                                          and ni_cr and ni_cr>1)
        out["Sales_Growth_YoY_%"]  = round(sales_g,  2) if sales_g  is not None else None
        out["Profit_Growth_YoY_%"] = round(profit_g, 2) if profit_g is not None else None
        out["Net_Profit_Cr"]       = round(ni_cr,    2) if ni_cr    is not None else None
    else:
        out.update({"qualifies_bc": False, "Sales_Growth_YoY_%": None,
                    "Profit_Growth_YoY_%": None, "Net_Profit_Cr": None})


def fundamental_scan(symbol: str, yf_suffix: str = ".NS") -> dict:
    """
    Run ALL 5 fundamental screeners for one symbol in a single Ticker() call.

    Screeners computed here (Golden Crossover uses OHLC, computed separately):
      1. Piotroski F-Score  — 9-point accounting quality score
      2. Coffee Can         — CAGR, ROCE, D/E, MCap, no loss-year
      3. Magic Formula      — ROIC > 25%, Earnings Yield > 15%  (reuses same annual data)
      4. Bull Cartel        — YoY quarterly sales growth > 15%, profit > 20%

    One yfinance Ticker() → one set of HTTP calls → four screener results.
    This is the core efficiency mechanism for running all screeners on the full universe.
    """
    try:
        inc, bal, cf, inc_q, mcap, info = _fetch_statements(symbol, yf_suffix)
    except Exception as e:
        return _failed_scan(symbol, str(e))

    out = {"symbol": symbol, "error": ""}
    _annual_screens(out, inc, bal, cf, info, mcap)
    _bull_cartel(out, inc_q)
    return out
