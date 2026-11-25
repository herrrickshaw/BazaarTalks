"""Offline tests for esg_screen.py (grading, scoring, ranking, faked yfinance fetch)."""
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import esg_screen as es  # noqa: E402


@pytest.mark.parametrize("risk,grade", [
    (0, "negligible"), (9.99, "negligible"), (10, "low"), (19.99, "low"),
    (20, "medium"), (29.99, "medium"), (30, "high"), (39.99, "high"),
    (40, "severe"), (55, "severe"), (-3, "negligible"),
])
def test_esg_grade_band_edges(risk, grade):
    assert es.esg_grade(risk) == grade


@pytest.mark.parametrize("bad", [None, np.nan, np.inf, -np.inf])
def test_esg_grade_na_for_missing_or_nonfinite(bad):
    assert es.esg_grade(bad) == "n/a"


def test_esg_score_linear_and_clamped():
    assert es.esg_score_0_100(0) == 100.0
    assert es.esg_score_0_100(25) == 50.0
    assert es.esg_score_0_100(50) == 0.0
    assert es.esg_score_0_100(80) == 0.0            # clamped, never negative
    assert es.esg_score_0_100(12.34) == 75.3        # 100 - 24.68 rounded to 1dp
    assert es.esg_score_0_100(-5) == 110.0          # negative risk is NOT clamped on the top side


@pytest.mark.parametrize("bad", [None, np.nan, np.inf])
def test_esg_score_nan_for_missing(bad):
    assert np.isnan(es.esg_score_0_100(bad))


def test_rank_esg_orders_lowest_risk_first_and_drops_nan():
    df = pd.DataFrame({"ticker": ["A", "B", "C", "D"],
                       "total_esg": [35.0, 8.0, np.nan, 22.0]})
    out = es.rank_esg(df)
    assert out["ticker"].tolist() == ["B", "D", "A"]
    assert out["grade"].tolist() == ["negligible", "medium", "high"]
    assert out["esg_score"].tolist() == [84.0, 56.0, 30.0]
    assert "esg_score" not in df.columns             # input not mutated


def test_rank_esg_empty_after_dropna():
    out = es.rank_esg(pd.DataFrame({"ticker": ["A"], "total_esg": [np.nan]}))
    assert out.empty


def _install_fakes(monkeypatch, sustainability=None, raises=None):
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
        def sustainability(self):
            if raises:
                raise raises
            return sustainability
    yf.Ticker = Ticker
    monkeypatch.setitem(sys.modules, "apiclient", api)
    monkeypatch.setitem(sys.modules, "yfinance", yf)
    return api


def _sus_frame(**vals):
    return pd.DataFrame({"Value": pd.Series(vals)})


def test_fetch_esg_yf_parses_all_fields(monkeypatch):
    api = _install_fakes(monkeypatch, _sus_frame(
        totalEsg=21.5, environmentScore=3.0, socialScore=9.5, governanceScore=9.0,
        highestControversy=2.0))
    out = es.fetch_esg_yf("AAPL")
    assert out == {"ticker": "AAPL", "total_esg": 21.5, "env": 3.0, "social": 9.5,
                   "gov": 9.0, "controversy": 2.0}
    assert api.calls == [("yfinance", 2)]


def test_fetch_esg_yf_missing_and_nan_keys_become_nan(monkeypatch):
    _install_fakes(monkeypatch, _sus_frame(totalEsg=30.0, environmentScore=np.nan))
    out = es.fetch_esg_yf("X")
    assert out["total_esg"] == 30.0
    assert np.isnan(out["env"]) and np.isnan(out["social"])
    assert np.isnan(out["gov"]) and np.isnan(out["controversy"])


@pytest.mark.parametrize("value", [None, pd.DataFrame()])
def test_fetch_esg_yf_uncovered_returns_empty(monkeypatch, value):
    _install_fakes(monkeypatch, value)
    assert es.fetch_esg_yf("X") == {}


def test_fetch_esg_yf_swallows_errors(monkeypatch):
    _install_fakes(monkeypatch, raises=RuntimeError("boom"))
    assert es.fetch_esg_yf("X") == {}


def test_main_prints_ranked_table(monkeypatch, capsys):
    data = {"AAA": {"ticker": "AAA", "total_esg": 28.0, "env": 5.0, "social": 10.0, "gov": 13.0,
                    "controversy": 3.0},
            "BBB": {"ticker": "BBB", "total_esg": 9.0, "env": 1.0, "social": 3.0, "gov": 5.0,
                    "controversy": 1.0},
            "CCC": {}}
    monkeypatch.setattr(es, "fetch_esg_yf", lambda t: data[t])
    monkeypatch.setattr(sys, "argv", ["esg_screen.py", "--tickers", "aaa, bbb,ccc,"])
    es.main()
    out = capsys.readouterr().out
    assert "— 2 names" in out
    body = [l for l in out.splitlines() if l.strip().startswith(("AAA", "BBB"))]
    assert body[0].split()[:4] == ["BBB", "9.0", "82.0", "negligible"]
    assert body[1].split()[:4] == ["AAA", "28.0", "44.0", "medium"]


def test_main_default_ticker_list_and_nothing_fetched(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(es, "fetch_esg_yf", lambda t: seen.append(t) or {})
    monkeypatch.setattr(sys, "argv", ["esg_screen.py"])
    es.main()
    assert seen == ["AAPL", "MSFT", "XOM", "NVDA", "JNJ"]
    assert "no ESG data fetched" in capsys.readouterr().out
