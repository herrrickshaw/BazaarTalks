"""Offline tests for news_sentiment.py (lexicon scorer + faked yfinance news)."""
import os
import sys
import types

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import news_sentiment as ns  # noqa: E402


def test_tokenize_lowercases_and_splits_on_non_letters():
    assert ns.tokenize("Q3 Profit-Beat: EPS $1.20!") == ["q", "profit", "beat", "eps"]
    assert ns.tokenize("") == [] and ns.tokenize(None) == []


def test_score_text_counts_and_sign():
    r = ns.score_text("Record profit beats estimates")
    assert r == {"pos": 3, "neg": 0, "sentiment": 1.0}              # record, profit, beats
    r = ns.score_text("Fraud probe triggers plunge")
    assert r == {"pos": 0, "neg": 3, "sentiment": -1.0}
    assert ns.score_text("Profit warning after loss") == {"pos": 1, "neg": 2, "sentiment": -0.333}


def test_score_text_no_hits_is_zero_not_nan():
    assert ns.score_text("Company holds annual meeting") == {"pos": 0, "neg": 0, "sentiment": 0.0}
    assert ns.score_text("") == {"pos": 0, "neg": 0, "sentiment": 0.0}


def test_score_text_balanced_is_zero():
    assert ns.score_text("growth but decline")["sentiment"] == 0.0


def test_score_text_repeated_words_count_each_time_and_case_insensitive():
    r = ns.score_text("GAIN gain Gain loss")
    assert (r["pos"], r["neg"], r["sentiment"]) == (3, 1, 0.5)


def test_score_text_whole_word_matching_only():
    # 'cutting' / 'profits' / 'dropping' are not lexicon entries (no stemming)
    assert ns.score_text("cutting profits dropping")["pos"] + ns.score_text("cutting profits dropping")["neg"] == 0


def test_score_text_custom_lexicon():
    r = ns.score_text("moon rekt moon", positive={"moon"}, negative={"rekt"})
    assert r == {"pos": 2, "neg": 1, "sentiment": 0.333}


def test_lexicons_are_disjoint():
    assert not (ns.POSITIVE & ns.NEGATIVE)


def test_score_headlines_exact_values():
    r = ns.score_headlines(["Record profit", "Lawsuit and fraud", "Meeting held"])
    assert r == {"n_headlines": 3, "sentiment": round((1.0 + -1.0 + 0.0) / 3, 3), "lexicon_hits": 4}
    r2 = ns.score_headlines(["Strong growth", "Weak sales"])
    assert r2 == {"n_headlines": 2, "sentiment": 0.0, "lexicon_hits": 3}


def test_score_headlines_skips_falsy_and_handles_empty():
    assert ns.score_headlines([]) == {"n_headlines": 0, "sentiment": 0.0, "lexicon_hits": 0}
    r = ns.score_headlines(["", None, "Record profit"])
    assert r["n_headlines"] == 1 and r["sentiment"] == 1.0


def test_score_headlines_rounds_to_three_places():
    r = ns.score_headlines(["gain", "gain", "loss"])                 # (1+1-1)/3
    assert r["sentiment"] == 0.333


# ── fetch ─────────────────────────────────────────────────────────────────────
def _install(monkeypatch, news=None, raises=None):
    api = types.ModuleType("apiclient")
    api.calls = []

    def robust(src, fn, retries=5):
        api.calls.append((src, retries))
        return fn()
    api.robust = robust
    yf = types.ModuleType("yfinance")

    class Ticker:
        def __init__(self, t):
            self.t = t

        @property
        def news(self):
            if raises:
                raise raises
            return news
    yf.Ticker = Ticker
    monkeypatch.setitem(sys.modules, "apiclient", api)
    monkeypatch.setitem(sys.modules, "yfinance", yf)
    return api


def test_fetch_news_handles_both_schemas_and_skips_untitled(monkeypatch):
    api = _install(monkeypatch, [
        {"title": "old schema"},
        {"content": {"title": "new schema"}},
        {"content": None},
        {"title": ""},
        {"title": "", "content": {"title": "fallback"}},
        {},
    ])
    assert ns.fetch_news_yf("AAPL") == ["old schema", "new schema", "fallback"]
    assert api.calls == [("yfinance", 2)]


def test_fetch_news_limit_applies_before_filtering(monkeypatch):
    items = [{"title": f"h{i}"} for i in range(30)]
    _install(monkeypatch, items)
    assert ns.fetch_news_yf("X") == [f"h{i}" for i in range(15)]
    assert ns.fetch_news_yf("X", limit=3) == ["h0", "h1", "h2"]
    _install(monkeypatch, [{}, {}, {"title": "late"}])
    assert ns.fetch_news_yf("X", limit=2) == []                      # 'late' is past the limit


@pytest.mark.parametrize("news", [None, []])
def test_fetch_news_empty(monkeypatch, news):
    _install(monkeypatch, news)
    assert ns.fetch_news_yf("X") == []


def test_fetch_news_error_degrades(monkeypatch):
    _install(monkeypatch, raises=RuntimeError("offline"))
    assert ns.fetch_news_yf("X") == []


# ── universe / main ───────────────────────────────────────────────────────────
def test_universe_from_tickers_arg():
    args = types.SimpleNamespace(tickers=" aapl,msft ,, nvda", market="US", top=2)
    assert ns._universe(args) == ["AAPL", "MSFT", "NVDA"]


def test_universe_from_liquid_symbols(monkeypatch):
    lf = types.ModuleType("liquidity_factor")
    pf = types.ModuleType("pead_factor")
    seen = {}
    lf._market_wide = lambda m: seen.setdefault("m", m) and ("CLOSE", "VOL")
    pf._liquid_symbols = lambda c, v: ["S1", "S2", "S3", "S4"] if (c, v) == ("CLOSE", "VOL") else []
    monkeypatch.setitem(sys.modules, "liquidity_factor", lf)
    monkeypatch.setitem(sys.modules, "pead_factor", pf)
    args = types.SimpleNamespace(tickers=None, market=None, top=3)
    assert ns._universe(args) == ["S1", "S2", "S3"]
    assert seen["m"] == "US"                                         # market defaults to US


def test_main_sorts_by_sentiment_and_drops_headlineless(monkeypatch, capsys):
    heads = {"AAA": ["Record profit"], "BBB": ["Fraud probe", "Fraud"], "CCC": [], "DDD": ["Mixed gain loss"]}
    monkeypatch.setattr(ns, "fetch_news_yf", lambda t: heads[t])
    monkeypatch.setattr(sys, "argv", ["news_sentiment.py", "--tickers", "bbb,aaa,ccc,ddd"])
    ns.main()
    out = capsys.readouterr().out
    assert "— 3 names" in out
    rows = [l.split() for l in out.splitlines() if l.strip().startswith(("AAA", "BBB", "DDD"))]
    assert [r[0] for r in rows] == ["AAA", "DDD", "BBB"]
    assert rows[0] == ["AAA", "1.00", "1", "2"]
    assert rows[2] == ["BBB", "-1.00", "2", "3"]


def test_main_offline_message(monkeypatch, capsys):
    monkeypatch.setattr(ns, "fetch_news_yf", lambda t: [])
    monkeypatch.setattr(sys, "argv", ["news_sentiment.py", "--tickers", "AAPL"])
    ns.main()
    assert "no news fetched" in capsys.readouterr().out
