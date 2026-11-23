"""Offline tests for unlisted_valuation.py (comps valuation; parquet/sqlite inputs faked)."""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unlisted_valuation as uv  # noqa: E402


def test_implied_value_multiplies_and_coerces():
    assert uv.implied_value(5e7, 20) == 1e9
    assert uv.implied_value("100", "2.5") == 250.0
    assert uv.implied_value(0, 99) == 0.0
    assert uv.implied_value(-10, 3) == -30.0


def test_peer_multiple_band_quartiles():
    b = uv.peer_multiple_band([10, 20, 30, 40, 50])
    assert b == {"n": 5, "median": 30.0, "q1": 20.0, "q3": 40.0}


def test_peer_multiple_band_drops_nonpositive_nonfinite_and_none():
    b = uv.peer_multiple_band([10, 20, 0, -5, np.nan, np.inf, -np.inf, None, 30])
    assert b["n"] == 3 and b["median"] == 20.0
    # strict > 0: a zero multiple is meaningless
    assert uv.peer_multiple_band([0, 0.0])["n"] == 0


@pytest.mark.parametrize("vals", [[], [None], [-1, -2], [np.nan], (x for x in [])])
def test_peer_multiple_band_empty_cases(vals):
    assert uv.peer_multiple_band(vals) == {"n": 0, "median": None, "q1": None, "q3": None}


def test_peer_multiple_band_single_value_collapses_band():
    b = uv.peer_multiple_band([12.5])
    assert b == {"n": 1, "median": 12.5, "q1": 12.5, "q3": 12.5}


def test_peer_multiple_band_accepts_series_and_even_count():
    b = uv.peer_multiple_band(pd.Series([10.0, 20.0, 30.0, 40.0]))
    assert b["median"] == 25.0 and b["q1"] == 17.5 and b["q3"] == 32.5


def test_value_range_applies_band():
    band = {"n": 4, "median": 20.0, "q1": 15.0, "q3": 30.0}
    assert uv.value_range(2e6, band) == {"low": 3e7, "mid": 4e7, "high": 6e7, "n_peers": 4}


@pytest.mark.parametrize("band", [None, {}, {"median": None, "n": 0}])
def test_value_range_without_band(band):
    assert uv.value_range(1e6, band) == {"low": None, "mid": None, "high": None, "n_peers": 0}


# ── data assembly (fake parquet + real sqlite in tmp_path) ───────────────────
def _fund_db(path, rows):
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE fund (ticker TEXT, pe REAL, pb REAL, roe REAL, extra TEXT)")
    con.executemany("INSERT INTO fund VALUES (?,?,?,?,?)", [(*r, "x") for r in rows])
    con.commit()
    con.close()


COMP = pd.DataFrame({"ticker": ["A", "B", "C", "D", "Z"],
                     "industry": ["Ads", "Ads", "Ads", "Steel", "Ads"],
                     "sector": ["Comm"] * 3 + ["Mat", "Comm"], "name": ["n"] * 5})


@pytest.fixture
def env(tmp_path, monkeypatch):
    db = tmp_path / "fund.db"
    _fund_db(db, [("A", 10.0, 1.0, 5.0), ("B", 20.0, 2.0, 6.0), ("C", 30.0, 3.0, 7.0),
                  ("D", 8.0, 0.8, 9.0), ("Q", 99.0, 9.0, 1.0)])
    comp, unl = tmp_path / "comp.parquet", tmp_path / "unl.parquet"
    comp.write_text("")
    monkeypatch.setattr(uv, "FUND", str(db))
    monkeypatch.setattr(uv, "COMPANIES", str(comp))
    monkeypatch.setattr(uv, "UNLISTED", str(unl))
    store = {"unl": None}

    def fake_read_parquet(path, *a, **k):
        if str(path) == str(comp):
            return COMP.copy()
        if str(path) == str(unl):
            return store["unl"].copy()
        raise AssertionError(path)
    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    return types_ns(unl=unl, store=store)


def types_ns(**kw):
    import types
    return types.SimpleNamespace(**kw)


def test_load_peers_inner_joins_and_coerces_numeric(env, tmp_path, monkeypatch):
    peers = uv.load_peers()
    assert sorted(peers["ticker"]) == ["A", "B", "C", "D"]            # Z has no fundamentals, Q no industry
    assert list(peers.columns) == ["ticker", "industry", "sector", "pe", "pb", "roe"]
    assert peers["pe"].dtype.kind == "f"


def test_load_peers_nonnumeric_becomes_nan(env, tmp_path, monkeypatch):
    con = sqlite3.connect(uv.FUND)
    con.execute("UPDATE fund SET pe='n/a' WHERE ticker='A'")
    con.commit()
    con.close()
    peers = uv.load_peers().set_index("ticker")
    assert np.isnan(peers.loc["A", "pe"]) and peers.loc["B", "pe"] == 20.0


def test_industry_band_filters_by_exact_industry_and_kind(env):
    peers = uv.load_peers()
    pe = uv.industry_band(peers, "Ads")
    assert pe == {"n": 3, "median": 20.0, "q1": 15.0, "q3": 25.0}
    pb = uv.industry_band(peers, "Ads", "pb")
    assert pb["median"] == 2.0
    assert uv.industry_band(peers, "ads")["n"] == 0                    # case-sensitive exact match
    assert uv.industry_band(peers, "Nonexistent")["median"] is None


def _run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["unlisted_valuation.py", *argv])
    uv.main()


def test_main_requires_inputs(monkeypatch, tmp_path):
    monkeypatch.setattr(uv, "COMPANIES", str(tmp_path / "missing.parquet"))
    monkeypatch.setattr(uv, "FUND", str(tmp_path / "missing.db"))
    with pytest.raises(SystemExit, match="need companies_industry.parquet"):
        _run_main(monkeypatch, "--list")


def test_main_list_without_industry_filters_min_three_peers(env, monkeypatch, capsys):
    _run_main(monkeypatch)
    out = capsys.readouterr().out
    assert "industries with >=3 listed peers (1)" in out
    assert "Ads" in out and "Steel" not in out                         # Steel has only 1 peer
    assert "peers=   3" in out and "median P/E=20.0" in out
    assert "comps for industry" not in out


def test_main_industry_prints_band_and_implied_values(env, monkeypatch, capsys):
    _run_main(monkeypatch, "--industry", "Ads", "--earnings", "1000000", "--book", "2000000")
    out = capsys.readouterr().out
    assert "=== comps for industry: Ads ===" in out
    assert "P/E band (n=3): q1=15.0 med=20.0 q3=25.0" in out
    assert "implied value @ earnings 1,000,000: 15,000,000 — 20,000,000 — 25,000,000 (low—mid—high, 3 peers)" in out
    assert "implied value @ book 2,000,000: 3,000,000 — 4,000,000 — 5,000,000" in out
    assert "industries with" not in out                                # --list not requested


def test_main_industry_with_list_flag_prints_both(env, monkeypatch, capsys):
    _run_main(monkeypatch, "--industry", "Ads", "--list")
    out = capsys.readouterr().out
    assert "industries with >=3 listed peers" in out and "comps for industry: Ads" in out


def test_main_unknown_industry_reports_no_coverage(env, monkeypatch, capsys):
    _run_main(monkeypatch, "--industry", "Nope", "--earnings", "5", "--book", "5")
    out = capsys.readouterr().out
    assert "P/E band: no peer coverage" in out
    assert "implied value" not in out


def test_main_lists_unlisted_candidates(env, monkeypatch, capsys):
    env.unl.write_text("")
    env.store["unl"] = pd.DataFrame({"company_name": [f"Firm{i}" for i in range(12)] + ["Other"],
                                     "segment": ["Digital ADS agency"] * 12 + ["Steel"]})
    _run_main(monkeypatch, "--industry", "Ads")
    cap = capsys.readouterr()
    assert "unlisted firms tagged to this space (12):" in cap.err     # case-insensitive match
    names = [l.strip() for l in cap.out.splitlines() if l.strip().startswith("- ")]
    assert names == [f"- Firm{i}" for i in range(10)]                  # capped at 10
    assert "Other" not in cap.out


def test_main_unlisted_without_segment_column_or_hits(env, monkeypatch, capsys):
    env.unl.write_text("")
    env.store["unl"] = pd.DataFrame({"company_name": ["X"], "industry": ["Ads"]})
    _run_main(monkeypatch, "--industry", "Ads")
    assert "unlisted firms" not in capsys.readouterr().err
    env.store["unl"] = pd.DataFrame({"company_name": ["X"], "segment": ["Steel"]})
    _run_main(monkeypatch, "--industry", "Ads")
    assert "unlisted firms" not in capsys.readouterr().err
