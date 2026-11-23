"""Offline tests for market_holidays.py (calendar logic, aliases, frozen 'today')."""
import os
import sys
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import market_holidays as mh  # noqa: E402


def _frozen_date(monkeypatch, y, m, d):
    class FakeDate(date):
        @classmethod
        def today(cls):
            return date(y, m, d)

    monkeypatch.setattr(mh, "date", FakeDate)


@pytest.mark.parametrize("year,expected", [
    (2000, date(2000, 4, 23)), (2024, date(2024, 3, 31)), (2025, date(2025, 4, 20)),
    (2026, date(2026, 4, 5)), (2027, date(2027, 3, 28)),
])
def test_easter_known_dates(year, expected):
    assert mh._easter(year) == expected


def test_europe_good_friday_and_easter_monday_2026():
    assert mh.is_trading_day("Europe", "2026-04-03") is False   # Good Friday
    assert mh.is_trading_day("Europe", "2026-04-06") is False   # Easter Monday
    assert mh.is_trading_day("Europe", "2026-04-02") is True    # Thursday before
    assert mh.is_trading_day("Europe", "2026-04-07") is True    # Tuesday after


def test_aliases_resolve_to_canonical_market():
    # 2026-01-26 is a Monday: Republic Day for every India alias
    for alias in ("NSE", "BSE", "India"):
        assert mh.is_trading_day(alias, "2026-01-26") is False
    for alias in ("NYSE", "NASDAQ", "US"):
        assert mh.is_trading_day(alias, "2026-07-03") is True   # Jul-4 is a Saturday in 2026
    assert mh._canon("XETRA") == "Europe"
    assert mh._canon("KOSPI") == "Korea"
    assert mh._canon("JPX") == "Japan"
    assert mh._canon("Atlantis") == "Atlantis"                  # unknown passes through


def test_unknown_market_only_weekends_are_closed():
    assert mh.is_trading_day("Atlantis", "2026-12-25") is True  # no fixed set -> open
    assert mh.is_trading_day("Atlantis", "2026-12-26") is False  # Saturday


def test_weekend_boundaries():
    assert mh.is_trading_day("US", "2026-06-12") is True   # Fri
    assert mh.is_trading_day("US", "2026-06-13") is False  # Sat
    assert mh.is_trading_day("US", "2026-06-14") is False  # Sun
    assert mh.is_trading_day("US", "2026-06-15") is True   # Mon


def test_input_types_are_coerced():
    assert mh.is_trading_day("US", pd.Timestamp("2026-12-25")) is False
    assert mh.is_trading_day("US", date(2026, 12, 25)) is False
    assert mh.is_trading_day("US", datetime(2026, 12, 24, 15, 30)) is True


def test_curated_holidays_per_market():
    assert mh.is_trading_day("US", "2026-11-26") is False      # Thanksgiving
    assert mh.is_trading_day("US", "2026-06-19") is False      # Juneteenth
    assert mh.is_trading_day("India", "2026-03-04") is False   # Holi (curated)
    assert mh.is_trading_day("Japan", "2026-09-22") is False   # Silver Week
    assert mh.is_trading_day("Korea", "2026-02-17") is False   # Seollal
    assert mh.is_trading_day("Korea", "2026-02-19") is True    # day after the cluster
    assert mh.is_trading_day("Japan", "2026-12-31") is False   # fixed JP year-end


def test_year_without_curated_list_uses_fixed_only():
    # 2028 has no curated entries: Thanksgiving (US floating) is NOT known -> open
    assert mh.is_trading_day("US", "2028-11-23") is True
    assert mh.is_trading_day("US", "2028-12-25") is False       # fixed


def test_trading_days_inclusive_bounds_and_count():
    td = mh.trading_days("US", "2026-12-21", "2026-12-28")
    assert [d.strftime("%Y-%m-%d") for d in td] == [
        "2026-12-21", "2026-12-22", "2026-12-23", "2026-12-24", "2026-12-28"]
    assert isinstance(td, pd.DatetimeIndex)
    assert len(mh.trading_days("US", "2026-01-01", "2026-12-31")) == 253


def test_trading_days_empty_when_range_all_closed():
    td = mh.trading_days("US", "2026-12-25", "2026-12-27")      # Fri hol + weekend
    assert len(td) == 0


def test_next_trading_day_skips_weekend_and_holiday():
    assert mh.next_trading_day("US", "2026-06-12") == date(2026, 6, 15)
    assert mh.next_trading_day("US", "2026-06-18") == date(2026, 6, 22)   # Fri 19th = Juneteenth
    assert mh.next_trading_day("US", date(2026, 12, 24)) == date(2026, 12, 28)


def test_next_trading_day_is_strictly_after_input():
    # input itself is a trading day -> still returns the NEXT session
    assert mh.next_trading_day("US", "2026-06-15") == date(2026, 6, 16)


def test_next_trading_day_defaults_to_today(monkeypatch):
    _frozen_date(monkeypatch, 2026, 6, 12)
    assert mh.next_trading_day("US") == date(2026, 6, 15)


def test_should_run_today_gate(monkeypatch):
    _frozen_date(monkeypatch, 2026, 6, 15)
    assert mh.should_run_today("US") is True
    _frozen_date(monkeypatch, 2026, 6, 13)
    assert mh.should_run_today("US") is False
    _frozen_date(monkeypatch, 2026, 1, 26)
    assert mh.should_run_today("NSE") is False


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="market_holidays.py:119-120 next_trading_day returns a datetime/Timestamp "
                          "(not a date) when given one, so == date(...) is False")
def test_next_trading_day_returns_plain_date_for_datetime_input():
    res = mh.next_trading_day("US", datetime(2026, 1, 2))
    assert res == date(2026, 1, 5)


def test_main_prints_summary(monkeypatch, capsys):
    _frozen_date(monkeypatch, 2026, 6, 12)
    monkeypatch.setattr(sys, "argv", ["market_holidays.py", "US"])
    import runpy
    # run the __main__ block against a fresh copy of the module (script path)
    runpy.run_path(mh.__file__, run_name="__main__")
    out = capsys.readouterr().out
    assert "US: 253 trading days in 2026" in out
    assert "8 weekday holidays skipped" in out
    assert "next session:" in out
