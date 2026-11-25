"""Offline tests for pit_panel.py (filed-date point-in-time panel + history gate)."""
import datetime as _dt
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pit_panel as pit  # noqa: E402

PANEL = [
    {"symbol": "AAA", "filed_date": "2023-02-10", "roe": 0.10},
    {"symbol": "AAA", "filed_date": "2024-02-12", "roe": 0.14},
    {"symbol": "AAA", "filed_date": "2025-02-11", "roe": 0.18},
    {"symbol": "BBB", "filed_date": "2024-05-01", "roe": 0.22},
]


# ── _d ────────────────────────────────────────────────────────────────────────
def test_d_accepts_str_date_datetime_timestamp_and_long_strings():
    d = _dt.date(2024, 3, 5)
    assert pit._d("2024-03-05") == d
    assert pit._d("2024-03-05T10:30:00") == d             # only first 10 chars
    assert pit._d(d) == d
    assert pit._d(_dt.datetime(2024, 3, 5, 23, 59)) == d
    assert pit._d(pd.Timestamp("2024-03-05 12:00")) == d


@pytest.mark.parametrize("bad", ["", "not-a-date", "2024/03/05", None])
def test_d_rejects_malformed(bad):
    with pytest.raises(ValueError):
        pit._d(bad)


# ── as_of ─────────────────────────────────────────────────────────────────────
def test_as_of_picks_latest_filing_on_or_before_and_drops_future():
    got = pit.as_of(PANEL, "2024-06-30")
    assert set(got) == {"AAA", "BBB"}
    assert got["AAA"]["roe"] == 0.14
    assert got["BBB"]["roe"] == 0.22


def test_as_of_boundary_filed_equals_asof_is_included():
    assert pit.as_of(PANEL, "2024-02-12")["AAA"]["roe"] == 0.14
    assert pit.as_of(PANEL, "2024-02-11")["AAA"]["roe"] == 0.10     # one day earlier


def test_as_of_symbol_not_yet_filed_is_absent():
    assert "BBB" not in pit.as_of(PANEL, "2024-04-30")
    assert pit.as_of(PANEL, "2020-01-01") == {}


def test_as_of_order_independent_and_first_wins_on_tie():
    shuffled = list(reversed(PANEL))
    assert pit.as_of(shuffled, "2024-06-30")["AAA"]["roe"] == 0.14
    tie = [{"symbol": "X", "filed_date": "2024-01-01", "v": 1},
           {"symbol": "X", "filed_date": "2024-01-01", "v": 2}]
    assert pit.as_of(tie, "2024-12-31")["X"]["v"] == 1       # strict > keeps the first


def test_as_of_dataframe_and_custom_columns_and_date_asof():
    df = pd.DataFrame([{"cd": "1", "filed": pd.Timestamp("2024-01-05"), "v": 1},
                       {"cd": "1", "filed": pd.Timestamp("2024-03-05"), "v": 2}])
    got = pit.as_of(df, _dt.date(2024, 2, 1), symbol_col="cd", filed_col="filed")
    assert got["1"]["v"] == 1
    assert pit.as_of(df, pd.Timestamp("2024-03-05"), symbol_col="cd", filed_col="filed")["1"]["v"] == 2


def test_as_of_empty_inputs():
    assert pit.as_of([], "2024-01-01") == {}
    assert pit.as_of(pd.DataFrame(columns=["symbol", "filed_date"]), "2024-01-01") == {}


def test_as_of_accepts_generator_and_does_not_mutate():
    rows = [dict(r) for r in PANEL]
    pit.as_of((r for r in rows), "2024-06-30")
    assert rows == PANEL


# ── is_point_in_time ──────────────────────────────────────────────────────────
def test_is_point_in_time_true_and_false_cases():
    assert pit.is_point_in_time(PANEL) is True
    assert pit.is_point_in_time([]) is False
    for bad in (None, "", "NaT"):
        assert pit.is_point_in_time(PANEL + [{"symbol": "Z", "filed_date": bad}]) is False
    assert pit.is_point_in_time(PANEL + [{"symbol": "Z"}]) is False          # key missing
    assert pit.is_point_in_time(pd.DataFrame(PANEL)) is True
    rows = [{"symbol": "A", "d": "2024-01-01"}]
    assert pit.is_point_in_time(rows, filed_col="d") is True
    assert pit.is_point_in_time(rows) is False                              # default col absent


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="pit_panel.py:~71 only treats None/''/'NaT' as missing, so pandas NaT "
                          "(and NaN) filed dates in a DataFrame still count as 'real filing dates'")
def test_is_point_in_time_rejects_pandas_missing_dates():
    df = pd.DataFrame({"symbol": ["A", "B"],
                       "filed_date": [pd.Timestamp("2024-01-01"), pd.NaT]})
    assert pit.is_point_in_time(df) is False


# ── leaks_lookahead ───────────────────────────────────────────────────────────
def test_leaks_lookahead_boundaries():
    assert pit.leaks_lookahead(PANEL, "2024-06-30") is True          # 2025 filing is future
    assert pit.leaks_lookahead(PANEL, "2025-02-11") is False         # == asof is known
    assert pit.leaks_lookahead(PANEL, "2025-02-10") is True
    assert pit.leaks_lookahead([], "2024-01-01") is False
    assert pit.leaks_lookahead(pd.DataFrame(PANEL), "2030-01-01") is False


def test_leaks_lookahead_custom_filed_col():
    rows = [{"symbol": "A", "f": "2024-01-02"}]
    assert pit.leaks_lookahead(rows, "2024-01-01", filed_col="f") is True


# ── min_history_ok / gate_markets ─────────────────────────────────────────────
def test_min_history_ok_boundaries():
    assert pit.min_history_ok(756) is True           # 3y * 252 exactly
    assert pit.min_history_ok(755) is False
    assert pit.min_history_ok(0) is False
    assert pit.min_history_ok(126, min_years=0.5) is True
    assert pit.min_history_ok(125, min_years=0.5) is False
    assert pit.min_history_ok(365, min_years=1, trading_days_year=365) is True
    assert pit.min_history_ok(364, min_years=1, trading_days_year=365) is False


def test_gate_markets_splits_sorted_and_reports_thresholds():
    g = pit.gate_markets({"US": 2520, "BR": 900, "JP": 240, "CN": 180, "IN": 756, "KR": 755})
    assert g == {"report": ["BR", "IN", "US"], "withhold": ["CN", "JP", "KR"],
                 "min_years": 3.0, "min_days": 756}


def test_gate_markets_custom_years_and_empty():
    g = pit.gate_markets({"A": 252, "B": 251}, min_years=1)
    assert g["report"] == ["A"] and g["withhold"] == ["B"] and g["min_days"] == 252
    assert pit.gate_markets({}) == {"report": [], "withhold": [], "min_years": 3.0, "min_days": 756}
    assert pit.gate_markets({}, min_years=1.5)["min_days"] == 378


# ── main ──────────────────────────────────────────────────────────────────────
def test_main_demo_output(capsys):
    pit.main()
    out = capsys.readouterr().out
    assert "    AAA: roe=0.14 (filed 2024-02-12)" in out
    assert "    BBB: roe=0.22 (filed 2024-05-01)" in out
    assert "leaks_lookahead(full panel @2024-06-30) = True" in out
    assert "history gate (>= 3.0y = 756d):" in out
    assert "    report:   ['BR', 'US']" in out
    assert "    withhold: ['CN', 'JP']" in out
