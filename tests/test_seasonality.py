"""Offline tests for seasonality.py (calendar-effect core + CLI)."""
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import seasonality as se  # noqa: E402


def _series(dates, values):
    return pd.Series(values, index=pd.DatetimeIndex(dates), dtype=float)


# ── day_of_week ───────────────────────────────────────────────────────────────
def test_day_of_week_means_counts_and_weekend_dropped():
    # 2024-01-01 is a Monday; include Sat/Sun rows which must be excluded
    idx = pd.date_range("2024-01-01", periods=14)           # 2 full weeks Mon..Sun
    vals = [0.01, 0.02, 0.03, 0.04, 0.05, 0.99, 0.99,
            0.03, 0.02, 0.01, 0.00, -0.01, 0.99, 0.99]
    out = se.day_of_week(_series(idx, vals))
    assert out.columns.tolist() == ["weekday", "mean%", "n"]
    assert out["weekday"].tolist() == ["Mon", "Tue", "Wed", "Thu", "Fri"]
    assert out["mean%"].tolist() == pytest.approx([2.0, 2.0, 2.0, 2.0, 2.0])
    assert out["n"].tolist() == [2] * 5


def test_day_of_week_drops_nan_and_rounds_to_3dp():
    idx = pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-09"])   # Mon, Mon, Tue
    out = se.day_of_week(_series(idx, [0.01, np.nan, 0.0123456]))
    assert out["weekday"].tolist() == ["Mon", "Tue"]
    assert out["n"].tolist() == [1, 1]
    assert out["mean%"].tolist() == [1.0, 1.235]


def test_day_of_week_missing_weekday_uses_real_label():
    idx = pd.to_datetime(["2024-01-03", "2024-01-05"])      # Wed, Fri only
    out = se.day_of_week(_series(idx, [0.01, 0.02]))
    assert out["weekday"].tolist() == ["Wed", "Fri"]


# ── turn_of_month ─────────────────────────────────────────────────────────────
def test_turn_of_month_empty_returns_empty_dict():
    assert se.turn_of_month(pd.Series([], dtype=float, index=pd.DatetimeIndex([]))) == {}
    assert se.turn_of_month(_series(["2024-01-02"], [np.nan])) == {}


def test_turn_of_month_flags_first_window_and_last_day():
    # Jan: 2,3,4,5,8 ; Feb: 1,2,5,6,7  (business days only)
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08",
                          "2024-02-01", "2024-02-02", "2024-02-05", "2024-02-06", "2024-02-07"])
    vals = [0.10, 0.10, 0.10, 0.0, 0.10,      # Jan: first 3 + last (8th) are ToM; 5th is rest
            0.10, 0.10, 0.10, 0.0, 0.10]      # Feb: first 3 + last (7th) are ToM; 6th is rest
    out = se.turn_of_month(_series(idx, vals))
    assert out["tom_n"] == 8 and out["rest_n"] == 2
    assert out["tom_mean%"] == 10.0
    assert out["rest_mean%"] == 0.0
    assert out["edge%"] == 10.0


def test_turn_of_month_window_parameter_shifts_boundary():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])
    r = _series(idx, [0.01, 0.02, 0.03, 0.04, 0.05])
    w1 = se.turn_of_month(r, window=1)       # first day + last day
    assert w1["tom_n"] == 2 and w1["rest_n"] == 3
    assert w1["tom_mean%"] == pytest.approx(3.0)             # (0.01 + 0.05)/2
    assert w1["rest_mean%"] == pytest.approx(3.0)
    w0 = se.turn_of_month(r, window=0)       # last day only
    assert w0["tom_n"] == 1 and w0["tom_mean%"] == 5.0
    w9 = se.turn_of_month(r, window=9)       # everything is ToM
    assert w9["tom_n"] == 5 and w9["rest_n"] == 0
    assert np.isnan(w9["rest_mean%"])


def test_turn_of_month_single_day_month_counted_once():
    idx = pd.to_datetime(["2024-01-31", "2024-02-01", "2024-02-02", "2024-02-05"])
    out = se.turn_of_month(_series(idx, [0.01, 0.02, 0.03, 0.04]), window=1)
    assert out["tom_n"] == 3                  # Jan31 (first==last), Feb1 (first), Feb5 (last)
    assert out["rest_n"] == 1
    assert out["rest_mean%"] == 3.0


# ── monthly seasonality / sell in may ────────────────────────────────────────
def _two_years():
    # one return per month-end for 2 years -> monthly compounded == that return
    idx = pd.date_range("2022-01-31", periods=24, freq=pd.offsets.MonthEnd())
    vals = [0.01 * ((i % 12) + 1) for i in range(24)]      # month m -> m% each year
    return _series(idx, vals)


def test_monthly_seasonality_mean_by_month_and_count():
    out = se.monthly_seasonality(_two_years())
    assert out.columns.tolist() == ["month", "mean%", "n"]
    assert out["month"].tolist() == list(range(1, 13))
    assert out["mean%"].tolist() == pytest.approx([float(m) for m in range(1, 13)])
    assert out["n"].tolist() == [2] * 12


def test_monthly_seasonality_compounds_within_month():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    out = se.monthly_seasonality(_series(idx, [0.10, 0.10]))
    assert out["mean%"].iloc[0] == 21.0 and out["n"].iloc[0] == 1


def test_sell_in_may_split_and_years():
    out = se.sell_in_may(_two_years())
    # Nov-Apr months: 11,12,1,2,3,4 -> mean(11,12,1,2,3,4)=5.5 ; May-Oct: 5..10 -> 7.5
    assert out["nov_apr%"] == pytest.approx(5.5)
    assert out["may_oct%"] == pytest.approx(7.5)
    assert out["edge%"] == pytest.approx(-2.0)
    assert out["years"] == 2


def test_sell_in_may_boundaries_april_may_oct_nov():
    idx = pd.to_datetime(["2023-04-28", "2023-05-31", "2023-10-31", "2023-11-30"])
    out = se.sell_in_may(_series(idx, [0.01, 0.02, 0.04, 0.08]))
    assert out["nov_apr%"] == pytest.approx((1 + 8) / 2)      # Apr, Nov
    assert out["may_oct%"] == pytest.approx((2 + 4) / 2)      # May, Oct
    assert out["years"] == 1


def test_sell_in_may_nan_dropped():
    idx = pd.to_datetime(["2023-01-31", "2023-06-30", "2023-07-31"])
    out = se.sell_in_may(_series(idx, [np.nan, 0.02, 0.04]))
    assert np.isnan(out["nov_apr%"])
    assert out["may_oct%"] == pytest.approx(3.0)


# ── data layer ────────────────────────────────────────────────────────────────
def _install_fakes(monkeypatch, wide, liquid=None):
    lf = types.ModuleType("liquidity_factor")
    lf._market_wide = lambda m: wide
    pfm = types.ModuleType("pead_factor")
    pfm._liquid_symbols = lambda close, vol: list(close.columns) if liquid is None else liquid
    monkeypatch.setitem(sys.modules, "liquidity_factor", lf)
    monkeypatch.setitem(sys.modules, "pead_factor", pfm)


def test_market_returns_missing_parquet_exits(monkeypatch):
    _install_fakes(monkeypatch, None)
    with pytest.raises(SystemExit) as ei:
        se.market_returns("ZZ")
    assert str(ei.value.code) == "no parquet for ZZ"


def test_market_returns_clips_and_equal_weights_liquid_only(monkeypatch):
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    close = pd.DataFrame({"A": [100.0, 110.0, 99.0],
                          "B": [10.0, 30.0, 30.0],          # +200% glitch -> clipped to +50%
                          "JUNK": [1.0, 1000.0, 1.0]}, index=idx)
    vol = close * 0 + 1
    _install_fakes(monkeypatch, (close, vol), liquid=["A", "B"])
    r = se.market_returns("US")
    assert isinstance(r.index, pd.DatetimeIndex)
    assert len(r) == 2
    assert r.iloc[0] == pytest.approx((0.10 + 0.50) / 2)
    assert r.iloc[1] == pytest.approx((-0.10 + 0.0) / 2)


def test_market_returns_string_index_converted_to_datetime(monkeypatch):
    close = pd.DataFrame({"A": [1.0, 2.0]}, index=["2024-01-02", "2024-01-03"])
    _install_fakes(monkeypatch, (close, close))
    r = se.market_returns("US")
    assert r.index[0] == pd.Timestamp("2024-01-03")


# ── CLI ───────────────────────────────────────────────────────────────────────
def _long_series(years):
    idx = pd.bdate_range("2020-01-01", periods=int(260 * years))
    rng = np.random.default_rng(5)
    return pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)


def test_main_short_history_skips_sell_in_may(monkeypatch, capsys):
    monkeypatch.setattr(se, "market_returns", lambda m: _long_series(1.0))
    monkeypatch.setattr(sys, "argv", ["seasonality.py", "--market", "JP"])
    se.main()
    out = capsys.readouterr().out
    assert "SEASONALITY — JP equal-weight proxy (260 days, 1 yr)" in out
    assert "  day-of-week mean return:" in out
    assert "turn-of-month: ToM " in out
    assert "monthly/Sell-in-May: skipped (needs >=3 yr; have 1)" in out
    assert "Sell-in-May: Nov-Apr" not in out


def test_main_long_history_prints_sell_in_may(monkeypatch, capsys):
    monkeypatch.setattr(se, "market_returns", lambda m: _long_series(3.2))
    monkeypatch.setattr(sys, "argv", ["seasonality.py"])
    se.main()
    out = capsys.readouterr().out
    assert "SEASONALITY — US" in out                          # default market
    assert "Sell-in-May: Nov-Apr" in out
    assert "skipped" not in out


def test_main_all_iterates_market_list(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(se.marketdata, "market_list", lambda: ["US", "JP"])
    monkeypatch.setattr(se, "market_returns", lambda m: (seen.append(m), _long_series(1.0))[1])
    monkeypatch.setattr(sys, "argv", ["seasonality.py", "--all", "--market", "KR"])
    se.main()
    assert seen == ["US", "JP"]
    out = capsys.readouterr().out
    assert out.index("SEASONALITY — US") < out.index("SEASONALITY — JP")
