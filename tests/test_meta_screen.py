"""Offline tests for meta_screen.py (screen fusion / conviction ranking)."""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import meta_screen as ms  # noqa: E402


# ── fuse ──────────────────────────────────────────────────────────────────────
def test_fuse_weighted_mean_with_default_weights():
    comps = {"durability": 80.0, "valuation": 40.0, "momentum": 60.0}
    expected = (0.20 * 80 + 0.12 * 40 + 0.16 * 60) / (0.20 + 0.12 + 0.16)
    assert ms.fuse(comps) == pytest.approx(expected)


def test_fuse_missing_components_renormalise():
    # only durability present -> its score, regardless of weight
    assert ms.fuse({"durability": 70.0, "valuation": None, "momentum": float("nan")}) == pytest.approx(70.0)


def test_fuse_nothing_present_is_zero():
    assert ms.fuse({}) == 0.0
    assert ms.fuse({"durability": None, "valuation": np.nan}) == 0.0


def test_fuse_unknown_and_nonpositive_weights_ignored():
    w = {"a": 1.0, "b": 0.0, "c": -2.0}
    assert ms.fuse({"a": 50.0, "b": 100.0, "c": 100.0, "zzz": 100.0}, weights=w) == pytest.approx(50.0)
    assert ms.fuse({"b": 100.0}, weights=w) == 0.0                # den == 0


def test_fuse_custom_weights_replace_defaults():
    assert ms.fuse({"x": 10.0, "y": 30.0}, weights={"x": 3.0, "y": 1.0}) == pytest.approx(15.0)
    # empty dict is falsy -> falls back to defaults
    assert ms.fuse({"durability": 10.0}, weights={}) == pytest.approx(10.0)


def test_fuse_gate_bonus_per_true_gate_and_cap():
    base = ms.fuse({"durability": 50.0})
    assert ms.fuse({"durability": 50.0}, gates={"triple_hit": True}) == pytest.approx(base + 10)
    assert ms.fuse({"durability": 50.0}, gates={"a": True, "b": True, "c": False}) == pytest.approx(70.0)
    assert ms.fuse({"durability": 50.0}, gates={"a": False}) == pytest.approx(50.0)
    assert ms.fuse({"durability": 50.0}, gates={"a": True}, gate_bonus=2.5) == pytest.approx(52.5)


def test_fuse_clamped_to_0_100():
    assert ms.fuse({"durability": 95.0}, gates={"a": True}) == 100.0
    assert ms.fuse({"durability": -20.0}) == 0.0
    assert ms.fuse({"durability": 10.0}, gates={"a": True}, gate_bonus=-50.0) == 0.0


def test_fuse_gate_only_still_awards_bonus():
    assert ms.fuse({}, gates={"triple_hit": True}) == 10.0


def test_fuse_accepts_ints_and_numpy_scalars():
    assert ms.fuse({"durability": 60}) == 60.0
    assert ms.fuse({"durability": np.float64(60.0), "valuation": np.float64("nan")}) == 60.0


# ── rank ──────────────────────────────────────────────────────────────────────
def _df():
    return pd.DataFrame({"ticker": ["LOW", "HIGH", "MID"], "D": [20.0, 90.0, 55.0],
                         "V": [20.0, 80.0, 50.0], "M": [20.0, 70.0, 60.0]})


def test_rank_sorts_descending_and_adds_columns():
    out = ms.rank(_df())
    assert out["ticker"].tolist() == ["HIGH", "MID", "LOW"]
    assert out["conviction"].tolist() == [
        round((0.20 * 90 + 0.12 * 80 + 0.16 * 70) / 0.48, 1),
        round((0.20 * 55 + 0.12 * 50 + 0.16 * 60) / 0.48, 1),
        20.0]
    assert out["n_confirms"].tolist() == [1, 1, 1]            # D/V/M count as one method


def test_rank_does_not_mutate_input():
    df = _df()
    ms.rank(df)
    assert "conviction" not in df.columns
    assert df["ticker"].tolist() == ["LOW", "HIGH", "MID"]


def test_rank_optional_components_and_confirmation_count():
    df = _df()
    df["ml_signal"] = [np.nan, 100.0, np.nan]
    df["quality_score"] = [np.nan, 100.0, 40.0]
    df["pead_score"] = [np.nan, np.nan, 40.0]
    df["triple_hit"] = [False, True, False]
    out = ms.rank(df).set_index("ticker")
    assert out.loc["HIGH", "n_confirms"] == 4          # D/V/M + ml + quality + triple_hit (pead missing)
    assert out.loc["MID", "n_confirms"] == 3           # D/V/M + quality + pead
    assert out.loc["LOW", "n_confirms"] == 1
    comps = {"durability": 90.0, "valuation": 80.0, "momentum": 70.0, "ml_signal": 100.0, "quality": 100.0}
    assert out.loc["HIGH", "conviction"] == round(min(100.0, ms.fuse(comps) + 10.0), 1)


def test_rank_triple_hit_gate_adds_ten_points():
    df = _df()
    base = ms.rank(df).set_index("ticker")["conviction"]
    df["triple_hit"] = [True, False, False]
    bumped = ms.rank(df).set_index("ticker")["conviction"]
    assert bumped["LOW"] == pytest.approx(base["LOW"] + 10)
    assert bumped["HIGH"] == base["HIGH"] and bumped["MID"] == base["MID"]


def test_rank_missing_dvm_row_still_scored_by_others():
    df = pd.DataFrame({"ticker": ["X"], "D": [np.nan], "V": [np.nan], "M": [np.nan],
                       "quality_score": [64.0]})
    out = ms.rank(df)
    assert out["conviction"].iloc[0] == 64.0
    assert out["n_confirms"].iloc[0] == 1              # quality only; D/V/M all NaN -> not counted


def test_rank_custom_weights_used():
    df = pd.DataFrame({"ticker": ["X"], "D": [100.0], "V": [0.0], "M": [0.0]})
    out = ms.rank(df, weights={"durability": 1.0, "valuation": 1.0, "momentum": 2.0})
    assert out["conviction"].iloc[0] == 25.0


def test_rank_ties_keep_scores_equal():
    df = pd.DataFrame({"ticker": ["A", "B"], "D": [50.0, 50.0], "V": [50.0, 50.0], "M": [50.0, 50.0]})
    out = ms.rank(df)
    assert out["conviction"].tolist() == [50.0, 50.0]


# ── data assembly ─────────────────────────────────────────────────────────────
def _make_db(path):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE dvm_composite (market TEXT, ticker TEXT, D REAL, V REAL, M REAL, "
                "composite REAL, code TEXT, extra TEXT)")
    con.executemany("INSERT INTO dvm_composite VALUES (?,?,?,?,?,?,?,?)", [
        ("US", "AAA", 90, 80, 70, 80, "GGG", "x"), ("US", "BBB", 40, 30, 20, 30, "BBB", "x"),
        ("JP", "7203.T", 60, 55, 70, 61.7, "GGG", "x")])
    con.commit(); con.close()


def test_load_composite_all_and_filtered(tmp_path):
    db = str(tmp_path / "d.db"); _make_db(db)
    allr = ms.load_composite(None, db)
    assert list(allr.columns) == ["market", "ticker", "D", "V", "M", "composite", "code"]
    assert len(allr) == 3
    us = ms.load_composite("US", db)
    assert us["ticker"].tolist() == ["AAA", "BBB"]
    assert ms.load_composite("KR", db).empty


def test_merge_optional_left_join_semantics(tmp_path):
    df = pd.DataFrame({"ticker": ["A", "B"], "D": [1.0, 2.0]})
    csv = tmp_path / "ml.csv"
    csv.write_text("ticker,ml_signal\nA,77\nZ,5\n")
    out = ms._merge_optional(df, str(csv), "ml_signal")
    assert out["ml_signal"].iloc[0] == 77 and np.isnan(out["ml_signal"].iloc[1])
    assert len(out) == 2                                       # extra csv row 'Z' not added


def test_merge_optional_skips_none_and_missing_path(tmp_path):
    df = pd.DataFrame({"ticker": ["A"]})
    assert ms._merge_optional(df, None, "x") is df
    assert ms._merge_optional(df, "", "x") is df
    assert ms._merge_optional(df, str(tmp_path / "missing.csv"), "x") is df


def test_merge_optional_custom_keys(tmp_path):
    df = pd.DataFrame({"market": ["US", "JP"], "ticker": ["A", "A"]})
    csv = tmp_path / "x.csv"
    csv.write_text("market,ticker,quality_score\nUS,A,90\nJP,A,10\n")
    out = ms._merge_optional(df, str(csv), "quality_score", keys=("market", "ticker"))
    assert out["quality_score"].tolist() == [90, 10]


# ── CLI ───────────────────────────────────────────────────────────────────────
def test_main_missing_db_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["meta_screen.py", "--db", str(tmp_path / "no.db")])
    with pytest.raises(SystemExit) as ei:
        ms.main()
    assert "dvm_composite.py first" in str(ei.value.code)


def test_main_basic_output_sorted_and_top(tmp_path, monkeypatch, capsys):
    db = str(tmp_path / "d.db"); _make_db(db)
    monkeypatch.setattr(sys, "argv", ["meta_screen.py", "--db", db, "--top", "2"])
    ms.main()
    cap = capsys.readouterr()
    assert "META-SCREEN conviction — all markets (3 names)" in cap.err
    lines = cap.out.splitlines()
    assert lines[0].split() == ["mkt", "ticker", "D", "V", "M", "conv", "#conf", "code"]
    rows = [ln.split() for ln in lines[1:]]
    assert [r[1] for r in rows] == ["AAA", "7203.T"]           # top 2 by conviction
    assert rows[0] == ["US", "AAA", "90", "80", "70", "80.8", "1", "GGG"]


def test_main_market_filter_and_optional_csvs(tmp_path, monkeypatch, capsys):
    db = str(tmp_path / "d.db"); _make_db(db)
    ml = tmp_path / "ml.csv"; ml.write_text("ticker,ml_signal\nBBB,100\n")
    q = tmp_path / "q.csv"; q.write_text("ticker,quality_score\nBBB,100\n")
    pe = tmp_path / "pe.csv"; pe.write_text("ticker,pead_score\nBBB,100\n")
    lq = tmp_path / "lq.csv"; lq.write_text("ticker,liquidity_score\nBBB,100\n")
    th = tmp_path / "th.csv"; th.write_text("ticker\nBBB\nNOPE\n")
    monkeypatch.setattr(sys, "argv", ["meta_screen.py", "--db", db, "--market", "US",
                                      "--ml", str(ml), "--quality", str(q), "--pead", str(pe),
                                      "--liquidity", str(lq), "--triple-hits", str(th)])
    ms.main()
    cap = capsys.readouterr()
    assert "— US (2 names)" in cap.err
    rows = [ln.split() for ln in cap.out.splitlines()[1:]]
    # BBB: fuse(D40,V30,M20,ml100,q100,pead100,liq100) + 10 gate
    comps = {"durability": 40, "valuation": 30, "momentum": 20, "ml_signal": 100, "quality": 100,
             "pead": 100, "liquidity": 100}
    exp = round(min(100.0, ms.fuse(comps) + 10), 1)
    assert exp == 76.8
    assert [r[1] for r in rows] == ["AAA", "BBB"]              # AAA (80.8) still ahead
    assert float(rows[1][5]) == exp
    assert rows[1][6] == "5"                                       # D/V/M + ml + quality + pead + triple_hit


def test_main_triple_hits_missing_file_ignored(tmp_path, monkeypatch, capsys):
    db = str(tmp_path / "d.db"); _make_db(db)
    monkeypatch.setattr(sys, "argv", ["meta_screen.py", "--db", db,
                                      "--triple-hits", str(tmp_path / "gone.csv")])
    ms.main()
    assert "AAA" in capsys.readouterr().out


def test_main_default_db_is_next_to_module(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "HERE", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["meta_screen.py"])
    with pytest.raises(SystemExit):
        ms.main()                                              # no dvm_composite.db in tmp_path
