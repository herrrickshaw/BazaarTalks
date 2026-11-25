"""Offline tests for sector_rotation.py (12-1 momentum, group ranking, CLI)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sector_rotation as sr  # noqa: E402


# ── member_momentum ───────────────────────────────────────────────────────────
def test_member_momentum_skips_last_month():
    c = pd.Series(np.arange(1.0, 301.0))              # 1..300
    # start = c[-252] = 49 ; end = c[-22] = 279
    assert sr.member_momentum(c) == pytest.approx(279 / 49 - 1)


def test_member_momentum_minimum_length_boundary():
    ok = pd.Series(np.arange(1.0, 254.0))             # len 253 = lookback+1 -> computed
    assert not np.isnan(sr.member_momentum(ok))
    assert sr.member_momentum(ok) == pytest.approx(ok.iloc[-22] / ok.iloc[-252] - 1)
    short = pd.Series(np.arange(1.0, 253.0))          # len 252 -> NaN
    assert np.isnan(sr.member_momentum(short))


def test_member_momentum_drops_nan_before_counting():
    c = pd.Series(np.arange(1.0, 254.0))
    c.iloc[5] = np.nan
    c.iloc[7] = np.nan                                 # 251 valid points -> too short
    assert np.isnan(sr.member_momentum(c))


def test_member_momentum_nonpositive_start_is_nan():
    c = pd.Series(np.arange(1.0, 301.0))
    c.iloc[-252] = 0.0
    assert np.isnan(sr.member_momentum(c))
    c.iloc[-252] = -3.0
    assert np.isnan(sr.member_momentum(c))


def test_member_momentum_custom_params_and_small_series_fallback():
    c = pd.Series([10.0, 20.0, 40.0, 80.0])
    assert sr.member_momentum(c, lookback=3, skip=1) == pytest.approx(40 / 20 - 1)
    # skip >= len(c): falls back to the last price
    assert sr.member_momentum(c, lookback=3, skip=10) == pytest.approx(80 / 20 - 1)
    assert np.isnan(sr.member_momentum(pd.Series([], dtype=float)))


def test_member_momentum_casts_object_series():
    c = pd.Series([str(x) for x in range(1, 301)])
    assert sr.member_momentum(c) == pytest.approx(279 / 49 - 1)


# ── rank_groups ───────────────────────────────────────────────────────────────
def _members():
    return pd.DataFrame({
        "industry": ["A"] * 4 + ["B"] * 3 + ["C"] * 2 + ["D"] * 3,
        "ticker": list("abcdefghijkl"),
        "momentum": [0.30, 0.20, 0.10, -0.10,           # A median 0.15 breadth 75%
                     0.50, 0.40, 0.60,                  # B median 0.50 breadth 100%
                     0.90, 0.80,                        # C only 2 members -> filtered
                     -0.2, -0.1, np.nan],               # D 2 valid -> filtered at min 3
    })


def test_rank_groups_filters_small_groups_and_ranks_descending():
    out = sr.rank_groups(_members(), "industry")
    assert out.columns.tolist() == ["industry", "rank", "mom_median%", "mom_mean%", "breadth%", "n"]
    assert out["industry"].tolist() == ["B", "A"]
    assert out["rank"].tolist() == [1, 2]
    assert out["mom_median%"].tolist() == [50.0, 15.0]
    assert out["mom_mean%"].tolist() == [50.0, 12.5]
    assert out["breadth%"].tolist() == [100.0, 75.0]
    assert out["n"].tolist() == [3, 4]


def test_rank_groups_min_members_threshold_is_inclusive():
    out = sr.rank_groups(_members(), "industry", min_members=2)
    assert set(out["industry"]) == {"A", "B", "C", "D"}
    assert out["industry"].iloc[0] == "C"                      # median 85%
    assert sr.rank_groups(_members(), "industry", min_members=5).empty


def test_rank_groups_nan_momentum_excluded_from_count():
    out = sr.rank_groups(_members(), "industry", min_members=2).set_index("industry")
    assert out.loc["D", "n"] == 2
    assert out.loc["D", "breadth%"] == 0.0


def test_rank_groups_breadth_excludes_zero_returns():
    m = pd.DataFrame({"sector": ["S"] * 4, "ticker": list("abcd"),
                      "momentum": [0.0, 0.0, 0.1, -0.1]})
    out = sr.rank_groups(m, "sector")
    assert out["breadth%"].iloc[0] == 25.0                     # strictly > 0


def test_rank_groups_ties_share_rank_method_average_floor():
    m = pd.DataFrame({"industry": ["X"] * 3 + ["Y"] * 3 + ["Z"] * 3,
                      "ticker": list("abcdefghi"),
                      "momentum": [0.1] * 3 + [0.1] * 3 + [0.0] * 3})
    out = sr.rank_groups(m, "industry").set_index("industry")
    assert out.loc["X", "rank"] == out.loc["Y", "rank"] == 1       # 1.5 average -> int truncation
    assert out.loc["Z", "rank"] == 3


# ── data assembly ─────────────────────────────────────────────────────────────
def _prices(symbols, drift):
    dates = pd.bdate_range("2023-01-02", periods=300)
    rows = []
    for s in symbols:
        p = 100 * (1 + drift[s]) ** np.arange(300)
        rows += [{"Date": d, "Symbol": s, "Close": float(x)} for d, x in zip(dates, p)]
    return pd.DataFrame(rows)


@pytest.fixture
def seed(tmp_path, monkeypatch):
    """Two market parquets (US full, JP short), a companies table, patched paths."""
    sd = tmp_path / "seed"
    sd.mkdir()
    (sd / "cleaned_long_US.parquet").write_bytes(b"x")
    (sd / "cleaned_long_JP.parquet").write_bytes(b"x")
    (sd / "other.txt").write_bytes(b"x")
    comp_path = tmp_path / "companies_industry.parquet"
    comp_path.write_bytes(b"x")
    monkeypatch.setattr(sr, "SEED", str(sd))
    monkeypatch.setattr(sr, "COMPANIES", str(comp_path))
    comp = pd.DataFrame({
        "ticker": ["AAA.NS", "bbb", "CCC", "AAA.BO"], "country": ["IN", "US", "US", "IN"],
        "sector": ["Tech", "Tech", "Fin", "Dup"], "industry": ["Soft", "Soft", "Bank", "Dup"],
        "segment": ["s1", "s1", "s2", "s3"]})
    us = _prices(["AAA", "BBB", "CCC", "SHORT"], {"AAA": 0.002, "BBB": 0.001, "CCC": -0.001, "SHORT": 0.0})
    us = us[~((us["Symbol"] == "SHORT") & (us["Date"] > us["Date"].iloc[10]))]   # too short -> NaN momentum
    jp = _prices(["7203.T"], {"7203.T": 0.003})
    tables = {str(comp_path): comp, str(sd / "cleaned_long_US.parquet"): us,
              str(sd / "cleaned_long_JP.parquet"): jp}
    monkeypatch.setattr(sr.pd, "read_parquet", lambda p, *a, **k: tables[str(p)].copy())
    return tables


def test_build_member_table_single_market_joins_on_clean_key(seed):
    mem = sr.build_member_table("industry", "US")
    assert set(mem["ticker"]) == {"AAA", "BBB", "CCC"}          # SHORT dropped (NaN momentum)
    assert set(mem["market"]) == {"US"}
    by = dict(zip(mem["ticker"], mem["industry"]))
    # AAA joins to the first of the duplicated company keys (AAA.NS) -> drop_duplicates("key")
    assert by == {"AAA": "Soft", "BBB": "Soft", "CCC": "Bank"}
    mom = dict(zip(mem["ticker"], mem["momentum"]))
    assert mom["AAA"] == pytest.approx(1.002 ** 278 / 1.002 ** 48 - 1)


def test_build_member_table_unknown_group_filled(seed):
    mem = sr.build_member_table("sector", "JP")
    assert mem["sector"].tolist() == ["Unknown"]               # 7203 not in companies
    assert mem["ticker"].tolist() == ["7203.T"]


def test_build_member_table_all_markets_scans_seed_dir(seed):
    mem = sr.build_member_table("segment", None)
    assert set(mem["market"]) == {"US", "JP"}
    assert len(mem) == 4


def test_build_member_table_skips_missing_parquet_and_empty(seed, tmp_path, monkeypatch):
    mem = sr.build_member_table("industry", "ZZ")              # no file -> no rows
    assert mem.empty


# ── CLI ───────────────────────────────────────────────────────────────────────
def test_main_missing_companies_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "COMPANIES", str(tmp_path / "nope.parquet"))
    monkeypatch.setattr(sys, "argv", ["sector_rotation.py"])
    with pytest.raises(SystemExit) as ei:
        sr.main()
    assert "build_industry_parquet.py" in str(ei.value.code)


def test_main_no_members_exits(seed, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["sector_rotation.py", "--market", "ZZ"])
    with pytest.raises(SystemExit) as ei:
        sr.main()
    assert "no member momentum" in str(ei.value.code)


def test_main_prints_ranked_groups_and_leaders(seed, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["sector_rotation.py", "--by", "sector", "--market", "US",
                                      "--min-members", "1", "--top", "1"])
    sr.main()
    cap = capsys.readouterr()
    assert "SECTOR ROTATION by sector — US (2 groups)" in cap.err
    assert "Leaders (top group's names):" in cap.err
    lines = cap.out.splitlines()
    assert lines[0].split() == ["#", "sector", "12-1", "mom%", "breadth%", "n"]
    row = lines[1].split()
    assert row[:2] == ["1", "Tech"]                            # AAA+BBB beat CCC
    assert row[-1] == "2"
    leaders = [ln.split() for ln in lines[2:]]
    assert [l[0] for l in leaders] == ["AAA", "BBB"]           # sorted by momentum desc
    assert leaders[0][1].endswith("%")


def test_main_default_label_all_markets(seed, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["sector_rotation.py", "--min-members", "1"])
    sr.main()
    assert "all markets" in capsys.readouterr().err
