"""Live NSE market-regime logic with injected data sources, plus the scoring / advice / report built on it.

``NSEDataFetcher`` (nse_data_fetcher.py) wires the real nsepython / yfinance callables in; tests pass fakes.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegimeSources:
    """Data sources for the regime; ``None`` means the source is unavailable and its step is skipped."""
    index_quote: Optional[Callable] = None    # nse_get_index_quote("NIFTY 50") -> dict
    download: Optional[Callable] = None       # yfinance.download
    india_vix: Optional[Callable] = None      # indiavix() -> float
    fii_dii: Optional[Callable] = None        # nse_fiidii() -> DataFrame


def _regime_nifty_quote(result: dict, get_index_quote: Optional[Callable]) -> None:
    """Fill Nifty 50 last / 52-week range from the live NSE quote (best effort)."""
    if get_index_quote is None:
        return
    try:
        q = get_index_quote("NIFTY 50")
        if q:
            last   = float(q.get("last", 0))
            yh     = float(q.get("yearHigh", 0))
            yl     = float(q.get("yearLow", 0))
            result["nifty_last"]  = last
            result["nifty_52w_h"] = yh
            result["nifty_52w_l"] = yl
            result["pct_from_52wh"] = round((last - yh) / yh * 100, 2) if yh else None
    except Exception as exc:
        logger.debug("nifty quote unavailable: %s", exc)


def _regime_nifty_dma200(result: dict, download: Optional[Callable]) -> None:
    """Fill the Nifty 200-DMA, its 5-bar slope and % distance from yfinance closes (best effort)."""
    if download is None:
        return
    try:
        idx = download("^NSEI", period="1y", auto_adjust=True, progress=False)
        if isinstance(idx.columns, pd.MultiIndex):
            idx = idx.xs("^NSEI", axis=1, level=1)
        if not idx.empty:
            dma200 = float(idx["Close"].rolling(min(200, len(idx))).mean().iloc[-1])
            dma200_sl = float(idx["Close"].rolling(min(200, len(idx))).mean().diff(5).iloc[-1])
            last = float(idx["Close"].iloc[-1])
            result["dma200"]    = round(dma200, 2)
            result["dma200_sl"] = round(dma200_sl, 2)
            if not result["nifty_last"]:
                result["nifty_last"] = round(last, 2)
            pct_from_dma = (last - dma200) / dma200 * 100
            result["pct_from_dma200"] = round(pct_from_dma, 2)
    except Exception as exc:
        logger.debug("nifty 200-DMA unavailable: %s", exc)


def _regime_india_vix(result: dict, india_vix: Optional[Callable]) -> None:
    """Fill the India VIX level (best effort)."""
    if india_vix is None:
        return
    try:
        vix = india_vix()
        result["vix"] = round(float(vix), 2) if vix else None
    except Exception as exc:
        logger.debug("india vix unavailable: %s", exc)


def _regime_fii_flow(result: dict, fii_dii: Optional[Callable]) -> None:
    """Fill the 5-session FII net flow and BUYING/SELLING sentiment (best effort)."""
    if fii_dii is None:
        return
    try:
        fii_df = fii_dii()
        if isinstance(fii_df, pd.DataFrame) and not fii_df.empty:
            # Sum recent FII net activity (last 5 sessions)
            net_col = next((c for c in fii_df.columns
                            if "net" in c.lower() and "fii" in c.lower()), None)
            if not net_col:
                net_col = next((c for c in fii_df.columns if "net" in c.lower()), None)
            if net_col:
                recent = fii_df[net_col].dropna().head(5)
                net_5d = float(recent.sum())
                result["fii_net_cr"]    = round(net_5d / 1e7, 2) \
                                           if abs(net_5d) > 1e6 else round(net_5d, 2)
                result["fii_sentiment"] = "BUYING" if net_5d > 0 else "SELLING"
    except Exception as exc:
        logger.debug("fii/dii unavailable: %s", exc)


def classify_regime(result: dict) -> str:
    """Pure: SIDEWAYS within 1.5% of the 200-DMA, else BULL/BULL_VOLATILE (VIX>=18) or BEAR/BEAR_EXTREME (VIX>25)."""
    vix  = result.get("vix") or 15
    pct  = result.get("pct_from_dma200", 0) or 0
    if abs(pct) <= 1.5:
        return "SIDEWAYS"
    if pct > 1.5:
        return "BULL_VOLATILE" if vix >= 18 else "BULL"
    return "BEAR_EXTREME" if vix > 25 else "BEAR"


def build_regime(sources: RegimeSources) -> dict:
    """Run every regime step against ``sources`` and classify; returns the full regime dict."""
    result = {
        "regime":        "UNKNOWN",
        "nifty_last":    None,
        "nifty_52w_h":   None,
        "nifty_52w_l":   None,
        "vix":           None,
        "pct_from_52wh": None,
        "fii_net_cr":    None,
        "fii_sentiment": "UNKNOWN",
        "timestamp":     datetime.now().isoformat(),
    }
    _regime_nifty_quote(result, sources.index_quote)
    _regime_nifty_dma200(result, sources.download)
    _regime_india_vix(result, sources.india_vix)
    _regime_fii_flow(result, sources.fii_dii)
    result["regime"] = classify_regime(result)
    return result


def institutional_score(fii_df: Optional[pd.DataFrame], bulk_df: Optional[pd.DataFrame]) -> dict:
    """Composite score in [-3, 3] (FII 5-day net, DII 5-day net, bulk-deal buy ratio) with a label.

    ``None`` for either input means that source failed and contributes nothing.
    """
    score = 0
    details = {}

    # FII/DII
    try:
        if fii_df is not None and not fii_df.empty:
            net_cols = [c for c in fii_df.columns if "net" in c.lower()]
            fii_col  = next((c for c in net_cols if "fii" in c.lower()), None)
            dii_col  = next((c for c in net_cols if "dii" in c.lower()), None)
            if fii_col:
                fii_net = float(fii_df[fii_col].dropna().head(5).sum())
                score  += 1 if fii_net > 0 else -1
                details["fii_5d_net_cr"] = round(fii_net / 1e7, 2) if abs(fii_net) > 1e6 else fii_net
            if dii_col:
                dii_net = float(fii_df[dii_col].dropna().head(5).sum())
                score  += 1 if dii_net > 0 else -1
                details["dii_5d_net_cr"] = round(dii_net / 1e7, 2) if abs(dii_net) > 1e6 else dii_net
    except Exception as exc:
        logger.debug("fii/dii score skipped: %s", exc)

    # Bulk deals
    try:
        if bulk_df is not None and not bulk_df.empty and "Buy/Sell" in bulk_df.columns:
            buys  = (bulk_df["Buy/Sell"].str.upper() == "BUY").sum()
            sells = (bulk_df["Buy/Sell"].str.upper() == "SELL").sum()
            total = buys + sells
            if total > 0:
                buy_ratio = buys / total
                score += 1 if buy_ratio > 0.6 else (-1 if buy_ratio < 0.4 else 0)
                details["bulk_buys"]  = int(buys)
                details["bulk_sells"] = int(sells)
                details["bulk_buy_ratio"] = round(buy_ratio * 100, 1)
    except Exception as exc:
        logger.debug("bulk-deal score skipped: %s", exc)

    label = "STRONG_BUY" if score >= 3 else \
            "BUY"        if score >= 1 else \
            "NEUTRAL"    if score == 0 else \
            "SELL"       if score >= -2 else "STRONG_SELL"

    return {"score": score, "label": label, **details}


def strategy_recommendation(regime: str, vix: float, inst_score: int) -> list:
    """
    Map live market conditions to actionable screener recommendations.
    Based on walk-forward backtest results from the research framework.

    Key findings from backtesting:
      BULL + low VIX:     Darvas (1-3mo), Golden Cross (1yr)
      BEAR + any VIX:     Darvas BEAR breakout (1-3mo), wait for reversal
      SIDEWAYS:           Golden Cross setup, Coffee Can accumulation
      HIGH VIX (>22):     Reduce size, focus on quality (Piotroski ≥7)
      FII buying:         Higher conviction on all signals
    """
    recs = []

    # Base recommendations by regime
    regime_map = {
        "BULL": [
            "📈 BULL MARKET — Use Darvas breakouts (T+63d to T+126d horizon)",
            "   Best EV: Darvas +6-13% at 1-3 months | Alpha vs Nifty: +3-8%",
            "   For 1-year horizon: Golden Cross signals give EV +30-46%",
            "   Confirmation: prefer volume >120% of 20-day average on breakout",
        ],
        "BULL_VOLATILE": [
            "⚡ BULL VOLATILE — Nifty above 200 DMA but VIX elevated",
            "   Reduce position size by 30% | Tighten stop-loss to box bottom",
            "   Use Coffee Can quality filter before acting on Darvas signals",
            "   Avoid leveraged positions until VIX drops below 18",
        ],
        "BEAR": [
            "📉 BEAR MARKET — Conservative mode activated",
            "   Short-term (T+1d to T+5d): expected value ≈ 0%, AVOID",
            "   Medium-term (T+63d): Darvas BEAR breakouts work — EV +13%",
            "   Only act on stocks with: Piotroski ≥7 AND Bull Cartel criteria",
            "   Cash allocation target: increase to 40-60% of portfolio",
        ],
        "BEAR_EXTREME": [
            "🚨 BEAR EXTREME — VIX >25 (panic zone)",
            "   DO NOT initiate new long positions",
            "   Consider systematic accumulation of Coffee Can stocks (long-term)",
            "   Watch for VIX reversal below 22 as signal to re-enter",
            "   These conditions historically precede the best Darvas breakouts",
        ],
        "SIDEWAYS": [
            "⬛ SIDEWAYS MARKET — Range-bound conditions",
            "   Best screener: Golden Cross for medium-term (EV +2-7% at T+63d)",
            "   For 1-year horizon: Golden Cross EV +40-68% (strongest signal)",
            "   Accumulate Coffee Can stocks on dips — structural quality holds",
            "   Darvas expected value ≈ 0% in sideways — use only with fundamental filter",
        ],
    }

    recs.extend(regime_map.get(regime, ["⚠️  Unknown regime — wait for clarity"]))

    # VIX overlay
    if vix and vix > 25:
        recs.append(f"   🔴 VIX={vix:.1f} (PANIC) — halve standard position sizes")
    elif vix and vix > 18:
        recs.append(f"   🟡 VIX={vix:.1f} (ELEVATED) — reduce size by 25%")
    elif vix and vix < 12:
        recs.append(f"   ⚠️  VIX={vix:.1f} (COMPLACENCY) — consider taking partial profits")

    # Institutional score overlay
    if inst_score >= 2:
        recs.append(f"   ✅ Institutional buying ({inst_score:+d}) — increase conviction on signals")
    elif inst_score <= -2:
        recs.append(f"   ❌ Institutional selling ({inst_score:+d}) — avoid new entries, protect gains")

    return recs


def print_live_context_report(ctx: dict) -> None:
    """Pretty-print the live market context for the daily dashboard."""
    print(f"\n{'═'*65}")
    print(f"  📊 LIVE MARKET CONTEXT — {datetime.now().strftime('%d %b %Y %H:%M IST')}")
    print(f"{'═'*65}")

    # Regime
    regime = ctx.get("regime", "UNKNOWN")
    regime_emoji = {"BULL":"🟢","BEAR":"🔴","BEAR_EXTREME":"🔴🔴",
                    "BULL_VOLATILE":"🟡","SIDEWAYS":"🟡"}.get(regime, "⚪")
    print(f"  Market Regime:  {regime_emoji} {regime}")
    print(f"  Nifty 50:       {ctx.get('nifty_last','—')}")
    print(f"  200 DMA:        {ctx.get('dma200','—')}  "
          f"({ctx.get('pct_from_dma200','—'):+.2f}% from DMA)" if ctx.get('pct_from_dma200') else "  200 DMA:        —")
    print(f"  52-wk range:    {ctx.get('nifty_52w_low','—')} – {ctx.get('nifty_52w_high','—')}")
    print(f"  % from 52wH:    {ctx.get('pct_from_52wh','—'):+.2f}%" if ctx.get('pct_from_52wh') else "  % from 52wH:    —")

    # VIX
    vix = ctx.get("vix")
    vix_regime = ctx.get("vix_regime", "")
    vix_color = "🟢" if (vix and vix < 15) else "🟡" if (vix and vix < 22) else "🔴"
    print(f"\n  India VIX:      {vix_color} {vix}  [{vix_regime}]")
    print(f"  Interpretation: {'Low volatility — trend-following works well' if vix and vix<15 else 'Elevated — widen stop-losses' if vix and vix<22 else 'HIGH FEAR — reduce position size'}")

    # FII/DII
    print(f"\n  FII Sentiment:  {ctx.get('fii_sentiment','—')}"
          f"  (5d net: ₹{ctx.get('fii_net_cr','—')} Cr)")
    print(f"  Inst. Activity: {ctx.get('institutional_label','—')} "
          f"(score={ctx.get('institutional_score',0):+d})")
    if ctx.get("bulk_buy_ratio"):
        print(f"  Bulk deal buy%: {ctx['bulk_buy_ratio']:.1f}%")

    # Upcoming results
    r7 = ctx.get("results_next7d", [])
    print(f"\n  Results next 7d: {len(r7)} stocks — {', '.join(r7[:8])}{'...' if len(r7)>8 else ''}")

    # Recommendations
    recs = ctx.get("recommendations", [])
    print("\n  ── TODAY'S STRATEGY RECOMMENDATIONS ──────────────────────")
    for r in recs:
        print(f"  {r}")

    print(f"{'═'*65}\n")


def market_breadth(india_vix: Optional[Callable], index_quote: Optional[Callable]) -> dict:
    """VIX level / regime and Nifty position vs its 52-week range (empty values when a source is None)."""
    result = {"vix": None, "vix_regime": "NORMAL", "nifty_52w_pct": None}

    if india_vix is not None:
        try:
            vix = india_vix()
            result["vix"] = round(float(vix), 2) if vix else None
            if vix:
                result["vix_regime"] = (
                    "PANIC"        if vix > 30 else
                    "HIGH_FEAR"    if vix > 22 else
                    "ELEVATED"     if vix > 17 else
                    "NORMAL"       if vix > 12 else
                    "COMPLACENCY"
                )

            q = index_quote("NIFTY 50")
            if q:
                last, yh, yl = (float(q.get(k, 0) or 0)
                                for k in ("last","yearHigh","yearLow"))
                if yh and yl:
                    result["nifty_last"]     = last
                    result["nifty_52w_high"] = yh
                    result["nifty_52w_low"]  = yl
                    result["pct_from_52wh"]  = round((last - yh)/yh*100, 2)
                    result["pct_from_52wl"]  = round((last - yl)/yl*100, 2)
        except Exception as exc:
            logger.debug("market breadth unavailable: %s", exc)

    return result


def build_live_context(fetcher) -> dict:
    """Regime + breadth + institutional score + 7-day results calendar + strategy advice in one dict.

    ``fetcher`` provides get_regime / get_market_breadth / get_institutional_activity_score /
    get_results_calendar (NSEDataFetcher, or a fake).
    """
    print("  Fetching live market context …")
    context = {}

    # Regime
    regime_data = fetcher.get_regime()
    context.update(regime_data)

    # Market breadth
    breadth = fetcher.get_market_breadth()
    context.update({k: v for k, v in breadth.items() if k not in context})

    # Institutional score
    inst = fetcher.get_institutional_activity_score()
    context["institutional_score"]  = inst.get("score", 0)
    context["institutional_label"]  = inst.get("label", "NEUTRAL")
    context["bulk_buy_ratio"]        = inst.get("bulk_buy_ratio")

    # Upcoming results (next 7 days)
    results_cal = fetcher.get_results_calendar(days_ahead=7)
    context["results_next7d"]       = results_cal["symbol"].tolist() if not results_cal.empty else []
    context["results_count_7d"]     = len(context["results_next7d"])

    # Strategy recommendation matrix based on live context
    regime = context.get("regime", "UNKNOWN")
    vix    = context.get("vix", 15) or 15
    inst_s = context.get("institutional_score", 0)

    # Recommend best screener + horizon for today
    context["recommendations"] = strategy_recommendation(regime, vix, inst_s)

    return context


def nifty_pe_pb(cache_file: Path, ttl_sec: float, get_source: Optional[Callable[[], Callable]]) -> dict:
    """Nifty 50 P/E, P/B and dividend yield: a fresh JSON cache first, else the nsepython index table.

    ``get_source()`` returns ``index_pe_pb_div`` (None when nsepython is unavailable); every failure
    yields the all-None result.
    """
    # yfinance doesn't provide this; use cached NSE data if available
    if cache_file.exists():
        age = datetime.now().timestamp() - cache_file.stat().st_mtime
        if age < ttl_sec:
            with open(cache_file) as f:
                return json.load(f)

    if get_source is not None:
        try:
            index_pe_pb_div = get_source()
            today_str = datetime.today().strftime("%d-%m-%Y")
            week_ago  = (datetime.today() - timedelta(days=7)).strftime("%d-%m-%Y")
            df = index_pe_pb_div("NIFTY 50", week_ago, today_str)
            if isinstance(df, pd.DataFrame) and not df.empty:
                latest = df.iloc[-1]
                result = {
                    "pe":   float(latest.get("PE", 0)),
                    "pb":   float(latest.get("PB", 0)),
                    "div_yield": float(latest.get("DivYield", 0)),
                    "date": str(df.index[-1]),
                }
                with open(cache_file, "w") as f:
                    json.dump(result, f)
                return result
        except Exception as exc:
            logger.debug("nifty pe/pb unavailable: %s", exc)

    return {"pe": None, "pb": None, "div_yield": None, "date": None}
