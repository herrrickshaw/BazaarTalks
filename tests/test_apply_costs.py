"""Offline tests for apply_costs.py (net-of-cost viability tables)."""
import os
import sqlite3
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import apply_costs as ac  # noqa: E402


def _make_db(path, rows):
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE market_screen_summary (market TEXT, screen TEXT, "
                "avg_hit_pct REAL, avg_edge REAL)")
    con.executemany("INSERT INTO market_screen_summary VALUES (?,?,?,?)", rows)
    con.commit()
    con.close()


def test_costs_table_matches_documented_values():
    assert ac.COSTS_PCT == {"India": 0.30, "US": 0.10, "Japan": 0.20, "Korea": 0.25, "Europe": 0.40}


def test_load_missing_db_returns_none(tmp_path):
    assert ac.load(str(tmp_path / "nope.db")) is None


def test_load_reads_summary_columns(tmp_path):
    db = tmp_path / "s.db"
    _make_db(db, [("US", "RSI", 55.0, 0.5)])
    df = ac.load(str(db))
    assert list(df.columns) == ["market", "screen", "avg_hit_pct", "avg_edge"]
    assert df.iloc[0].tolist() == ["US", "RSI", 55.0, 0.5]


def test_net_table_boundaries_and_columns():
    df = pd.DataFrame({
        "market": ["US", "US", "US", "US", "India"],
        "screen": ["edge_eq_cost", "just_above", "low_hit", "hit_eq_50", "ind"],
        "avg_hit_pct": [60.0, 60.0, 50.0, 50.0001, 70.0],
        "avg_edge": [0.10, 0.11, 5.0, 5.0, 0.2],
    })
    out = ac.net_table(df)
    assert list(out.columns) == ["market", "screen", "gross_edge", "cost%", "net_edge",
                                 "hit%", "net_viable"]
    by = out.set_index("screen")
    assert by.loc["edge_eq_cost", "net_edge"] == 0.0
    assert by.loc["edge_eq_cost", "net_viable"] == "no"        # net must be strictly > 0
    assert by.loc["just_above", "net_edge"] == pytest.approx(0.01)
    assert by.loc["just_above", "net_viable"] == "YES"
    assert by.loc["low_hit", "net_viable"] == "no"             # hit must be strictly > 50
    assert by.loc["hit_eq_50", "net_viable"] == "YES"
    assert by.loc["ind", "net_edge"] == pytest.approx(-0.1)
    assert by.loc["ind", "net_viable"] == "no"
    assert by.loc["ind", "cost%"] == 0.30


def test_net_table_sort_and_no_mutation():
    df = pd.DataFrame({"market": ["US", "India", "US", "India"], "screen": list("abcd"),
                       "avg_hit_pct": [60.0] * 4, "avg_edge": [1.0, 2.0, 3.0, 0.5]})
    before = df.copy()
    out = ac.net_table(df)
    pd.testing.assert_frame_equal(df, before)                  # input untouched
    assert out["market"].tolist() == ["India", "India", "US", "US"]   # market asc
    assert out["screen"].tolist() == ["b", "d", "c", "a"]      # net_edge desc inside market


def test_net_table_rounds_to_three_places():
    df = pd.DataFrame({"market": ["US"], "screen": ["x"], "avg_hit_pct": [60.0],
                       "avg_edge": [0.123456]})
    assert ac.net_table(df)["net_edge"].iloc[0] == 0.023


def test_net_table_unknown_market_is_nan_and_not_viable():
    df = pd.DataFrame({"market": ["Mars"], "screen": ["x"], "avg_hit_pct": [90.0],
                       "avg_edge": [9.0]})
    out = ac.net_table(df)
    assert pd.isna(out["cost%"].iloc[0]) and pd.isna(out["net_edge"].iloc[0])
    assert out["net_viable"].iloc[0] == "no"


def test_main_writes_csv_and_markdown(tmp_path, monkeypatch, capsys):
    d5, d21 = tmp_path / "v5.db", tmp_path / "v21.db"
    _make_db(d5, [("US", "RSI", 60.0, 0.5), ("India", "Golden", 55.0, 0.1)])
    _make_db(d21, [("Europe", "Vol", 52.0, 1.0)])
    monkeypatch.setattr(ac, "HERE", str(tmp_path))
    monkeypatch.setattr(ac, "SUMMARIES", {"5d": str(d5), "21d": str(d21)})
    # DataFrame.to_markdown needs `tabulate`, which CI does not install
    monkeypatch.setattr(pd.DataFrame, "to_markdown", lambda self, index=False: "TABLE:%d" % len(self),
                        raising=False)
    ac.main()
    csv5 = pd.read_csv(tmp_path / "net_viability_5d.csv")
    assert csv5["screen"].tolist() == ["Golden", "RSI"]
    assert csv5["market"].tolist() == ["India", "US"]
    assert csv5.set_index("screen").loc["RSI", "net_viable"] == "YES"
    assert csv5.set_index("screen").loc["Golden", "net_viable"] == "no"
    csv21 = pd.read_csv(tmp_path / "net_viability_21d.csv")
    assert csv21["net_edge"].tolist() == [0.6]
    md = (tmp_path / "NET_OF_COST.md").read_text()
    assert "## 5d horizon  (1/2 screens keep positive edge net of cost; 2 were positive pre-cost)" in md
    assert "## 21d horizon  (1/1 screens keep positive edge net of cost; 1 were positive pre-cost)" in md
    assert "TABLE:2" in md and "## Bottom line" in md
    err = capsys.readouterr().err
    assert "[5d] wrote" in err and "wrote NET_OF_COST.md" in err


def test_main_skips_missing_db(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ac, "HERE", str(tmp_path))
    monkeypatch.setattr(ac, "SUMMARIES", {"5d": str(tmp_path / "gone.db")})
    ac.main()
    assert "[skip]" in capsys.readouterr().err
    assert not (tmp_path / "net_viability_5d.csv").exists()
    md = (tmp_path / "NET_OF_COST.md").read_text()
    assert "horizon  (" not in md and "## Bottom line" in md
