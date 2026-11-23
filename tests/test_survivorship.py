"""Offline tests for survivorship.py (PIT universe, delisting returns, survivorship gap)."""
import datetime as _dt
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import survivorship as sv  # noqa: E402

MEMBERSHIP = [
    {"symbol": "ALIVE1", "add_date": "2015-01-01", "remove_date": None},
    {"symbol": "ALIVE2", "add_date": "2018-06-01", "remove_date": ""},
    {"symbol": "DIED", "add_date": "2015-01-01", "remove_date": "2020-03-15"},
    {"symbol": "FUTURE", "add_date": "2024-01-01", "remove_date": None},
]


def test_d_conversions():
    d = _dt.date(2020, 1, 2)
    assert sv._d(None) is None
    assert sv._d(d) == d
    assert sv._d(_dt.datetime(2020, 1, 2, 9, 30)) == d
    assert sv._d("2020-01-02") == d
    assert sv._d("2020-01-02 10:00:00") == d
    assert sv._d(pd.Timestamp("2020-01-02")) == d
    with pytest.raises(ValueError):
        sv._d("02/01/2020")


def test_universe_includes_dead_names_before_death_and_excludes_future_adds():
    assert sv.point_in_time_universe(MEMBERSHIP, "2016-01-01") == ["ALIVE1", "DIED"]
    assert sv.point_in_time_universe(MEMBERSHIP, "2019-01-01") == ["ALIVE1", "ALIVE2", "DIED"]
    assert sv.point_in_time_universe(MEMBERSHIP, "2021-01-01") == ["ALIVE1", "ALIVE2"]
    assert sv.point_in_time_universe(MEMBERSHIP, "2025-01-01") == ["ALIVE1", "ALIVE2", "FUTURE"]


def test_universe_boundaries_add_inclusive_remove_exclusive():
    assert "FUTURE" in sv.point_in_time_universe(MEMBERSHIP, "2024-01-01")       # add == asof
    assert "FUTURE" not in sv.point_in_time_universe(MEMBERSHIP, "2023-12-31")
    assert "DIED" in sv.point_in_time_universe(MEMBERSHIP, "2020-03-14")
    assert "DIED" not in sv.point_in_time_universe(MEMBERSHIP, "2020-03-15")     # remove == asof


def test_universe_ignores_missing_add_date_and_dedupes_sorted():
    m = MEMBERSHIP + [{"symbol": "NOADD", "add_date": None, "remove_date": None},
                      {"symbol": "ALIVE1", "add_date": "2016-01-01", "remove_date": None}]
    out = sv.point_in_time_universe(m, "2021-01-01")
    assert out == ["ALIVE1", "ALIVE2"] and "NOADD" not in out
    assert sv.point_in_time_universe([], "2021-01-01") == []
    assert sv.point_in_time_universe(iter(MEMBERSHIP), _dt.date(2016, 1, 1)) == ["ALIVE1", "DIED"]


def test_coverage_counts():
    c = sv.coverage(MEMBERSHIP, "2021-01-01")
    assert c == {"asof": "2021-01-01", "in_universe": 2, "ever_listed": 4}
    assert sv.coverage([], _dt.date(2021, 1, 1)) == {"asof": "2021-01-01", "in_universe": 0,
                                                     "ever_listed": 0}


def test_survivorship_gap_sign_and_value():
    assert sv.survivorship_gap(-0.29, 0.02) == pytest.approx(0.31)
    assert sv.survivorship_gap(0.05, 0.05) == 0.0
    assert sv.survivorship_gap(0.10, 0.04) == pytest.approx(-0.06)


def _returns():
    idx = pd.date_range("2020-03-12", periods=6, freq="D")
    return pd.DataFrame({"DIED": [0.01] * 6, "OK": [0.02] * 6, "BUYOUT": [0.0] * 6}, index=idx)


def test_apply_delisting_returns_splices_final_and_blanks_after():
    r = _returns()
    out = sv.apply_delisting_returns(r, {"DIED": {"date": "2020-03-15", "final_return": -0.9}})
    died = out["DIED"]
    assert list(died.iloc[:3]) == [0.01, 0.01, 0.01]
    assert died.iloc[3] == -0.9                                   # delisting day
    assert died.iloc[4:].isna().all()                             # gone afterwards
    assert (out["OK"] == 0.02).all()


def test_apply_delisting_returns_does_not_mutate_input():
    r = _returns()
    before = r.copy()
    sv.apply_delisting_returns(r, {"DIED": {"date": "2020-03-13", "final_return": -1.0}})
    pd.testing.assert_frame_equal(r, before)


def test_apply_delisting_returns_multiple_unknown_symbol_and_date_types():
    r = _returns()
    out = sv.apply_delisting_returns(r, {
        "DIED": {"date": _dt.date(2020, 3, 12), "final_return": -0.5},     # first row
        "BUYOUT": {"date": pd.Timestamp("2020-03-17"), "final_return": 0.3},  # last row
        "GHOST": {"date": "2020-03-14", "final_return": -1.0},             # not a column
    })
    assert out["DIED"].iloc[0] == -0.5 and out["DIED"].iloc[1:].isna().all()
    assert out["BUYOUT"].iloc[-1] == 0.3 and (out["BUYOUT"].iloc[:-1] == 0.0).all()
    assert "GHOST" not in out.columns


def test_apply_delisting_returns_final_return_coerced_and_date_after_window():
    r = _returns()
    out = sv.apply_delisting_returns(r, {"DIED": {"date": "2020-03-14", "final_return": "-0.25"}})
    assert out["DIED"].iloc[2] == -0.25
    out2 = sv.apply_delisting_returns(r, {"DIED": {"date": "2030-01-01", "final_return": -1.0}})
    pd.testing.assert_frame_equal(out2, r)                        # never reached
    assert sv.apply_delisting_returns(r, {}).equals(r)


def test_main_prints_each_snapshot(capsys):
    sv.main()
    out = capsys.readouterr().out
    assert "as of 2016-01-01: ['ALIVE1', 'DIED']" in out
    assert "as of 2019-01-01: ['ALIVE1', 'ALIVE2', 'DIED']" in out
    assert "as of 2021-01-01: ['ALIVE1', 'ALIVE2']" in out
    assert "as of 2025-01-01: ['ALIVE1', 'ALIVE2', 'FUTURE']" in out
