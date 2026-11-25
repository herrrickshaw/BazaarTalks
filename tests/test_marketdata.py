"""Offline tests for marketdata.py (shared loaders, universe filter, cross-sectional stats)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import marketdata as md  # noqa: E402


# ── market_list / _parquet ────────────────────────────────────────────────────
def _touch(d, *names):
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_bytes(b"")


def test_market_list_merges_both_seed_dirs_sorted_and_deduped(tmp_path, monkeypatch):
    ext, loc = tmp_path / "ext", tmp_path / "loc"
    _touch(ext, "cleaned_long_US.parquet", "cleaned_long_JP.parquet", "other.txt",
           "xcleaned_long_ZZ.parquet")
    _touch(loc, "cleaned_long_IN.parquet", "cleaned_long_US.parquet")
    monkeypatch.setattr(md, "SEED", str(ext))
    monkeypatch.setattr(md, "SEED_LOCAL", str(loc))
    assert md.market_list() == ["IN", "JP", "US"]


def test_market_list_missing_dirs_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(md, "SEED", str(tmp_path / "nope"))
    monkeypatch.setattr(md, "SEED_LOCAL", str(tmp_path / "nope2"))
    assert md.market_list() == []


def test_market_list_one_dir_missing(tmp_path, monkeypatch):
    loc = tmp_path / "loc"
    _touch(loc, "cleaned_long_IN.parquet")
    monkeypatch.setattr(md, "SEED", str(tmp_path / "nope"))
    monkeypatch.setattr(md, "SEED_LOCAL", str(loc))
    assert md.market_list() == ["IN"]


def test_parquet_prefers_external_seed_then_local(tmp_path, monkeypatch):
    ext, loc = tmp_path / "ext", tmp_path / "loc"
    _touch(ext, "cleaned_long_US.parquet")
    _touch(loc, "cleaned_long_US.parquet", "cleaned_long_IN.parquet")
    monkeypatch.setattr(md, "SEED", str(ext))
    monkeypatch.setattr(md, "SEED_LOCAL", str(loc))
    assert md._parquet("US") == str(ext / "cleaned_long_US.parquet")
    assert md._parquet("IN") == str(loc / "cleaned_long_IN.parquet")
    # absent everywhere -> the local path (which does not exist)
    assert md._parquet("XX") == str(loc / "cleaned_long_XX.parquet")


# ── wide / close_volume ───────────────────────────────────────────────────────
def _long_frame():
    d = pd.to_datetime(["2024-01-02", "2024-01-03"])
    rows = []
    for sym, base in (("AAA", 10.0), ("BBB", 20.0)):
        for i, day in enumerate(d):
            rows.append({"Date": day, "Symbol": sym, "Close": base + i,
                         "High": base + i + 1, "Low": base + i - 1,
                         "Volume": 100 * (i + 1)})
    # duplicate (Date, Symbol): aggfunc="last" must keep the last one
    rows.append({"Date": d[1], "Symbol": "AAA", "Close": 99.0, "High": 100.0,
                 "Low": 98.0, "Volume": 7})
    return pd.DataFrame(rows)


@pytest.fixture
def seeded(tmp_path, monkeypatch):
    ext = tmp_path / "ext"
    _touch(ext, "cleaned_long_US.parquet")
    monkeypatch.setattr(md, "SEED", str(ext))
    monkeypatch.setattr(md, "SEED_LOCAL", str(tmp_path / "loc"))
    monkeypatch.setattr(pd, "read_parquet", lambda p, *a, **k: _long_frame())
    return ext


def test_wide_returns_all_fields_pivoted_with_last_aggregation(seeded):
    w = md.wide("US")
    assert set(w) == {"Close", "High", "Low", "Volume"}
    assert list(w["Close"].columns) == ["AAA", "BBB"]
    assert w["Close"].loc["2024-01-03", "AAA"] == 99.0         # last duplicate wins
    assert w["Close"].loc["2024-01-02", "BBB"] == 20.0
    assert all(f.dtypes.eq(float).all() for f in w.values())


def test_wide_field_subset_and_missing_market(seeded):
    w = md.wide("US", fields=("Close",))
    assert list(w) == ["Close"]
    assert md.wide("NOPE") is None


def test_close_volume_pair_and_missing(seeded):
    close, vol = md.close_volume("US")
    assert close.loc["2024-01-03", "AAA"] == 99.0
    assert vol.loc["2024-01-03", "AAA"] == 7.0
    assert md.close_volume("NOPE") == (None, None)


# ── liquid_symbols ────────────────────────────────────────────────────────────
def _universe(n_days=300):
    idx = pd.date_range("2022-01-03", periods=n_days, freq="B")
    cols = [f"S{i}" for i in range(1, 11)]
    close = pd.DataFrame(1.0, index=idx, columns=cols)
    vol = pd.DataFrame({c: float(i) for i, c in enumerate(cols, start=1)}, index=idx)
    return close, vol


def test_liquid_symbols_keeps_top_40_percent():
    close, vol = _universe()
    # dollar volumes 1..10, 0.6 quantile = 6.4 -> S7..S10
    assert md.liquid_symbols(close, vol) == ["S7", "S8", "S9", "S10"]


def test_liquid_symbols_threshold_is_inclusive():
    close, vol = _universe()
    # quantile 0.5 over 1..10 -> 5.5; quantile 4/9 -> exactly 5.0 => S5 included (>=)
    assert md.liquid_symbols(close, vol, quantile=4 / 9)[0] == "S5"
    assert "S4" not in md.liquid_symbols(close, vol, quantile=4 / 9)


def test_liquid_symbols_min_history_boundary():
    close, vol = _universe(300)
    close.iloc[:50, close.columns.get_loc("S10")] = np.nan      # 250 valid rows left
    assert "S10" in md.liquid_symbols(close, vol, min_history=250)    # exactly enough
    assert "S10" not in md.liquid_symbols(close, vol, min_history=251)  # one short


def test_liquid_symbols_nan_dollar_volume_excluded():
    close, vol = _universe()
    vol["S9"] = np.nan
    out = md.liquid_symbols(close, vol)
    assert "S9" not in out and "S10" in out


# ── clean_key ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("raw,expected", [
    ("reliance.ns", "RELIANCE"), ("7203.T", "7203"), ("AAPL", "AAPL"),
    ("brk.b", "BRK"), (123, "123"), ("", ""),
])
def test_clean_key(raw, expected):
    assert md.clean_key(raw) == expected


# ── market_proxy ──────────────────────────────────────────────────────────────
def test_market_proxy_equal_weight_and_clip():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    close = pd.DataFrame({"A": [100.0, 110.0, 110.0], "B": [10.0, 40.0, 40.0]}, index=idx)
    p = md.market_proxy(close)
    assert np.isnan(p.iloc[0])
    # day 2: A +10%, B +300% clipped to +50% -> mean 30%
    assert p.iloc[1] == pytest.approx(0.30)
    assert p.iloc[2] == pytest.approx(0.0)


def test_market_proxy_symbol_subset_custom_clip_and_nan_not_filled():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    close = pd.DataFrame({"A": [100.0, 120.0, 120.0, 120.0],
                          "B": [10.0, np.nan, 11.0, 11.0],
                          "C": [5.0, 5.0, 5.0, 5.0]}, index=idx)
    p = md.market_proxy(close, symbols=["A", "B"], clip=0.1)
    assert p.iloc[1] == pytest.approx(0.10)       # A +20% clipped to 10%; B NaN skipped
    # B's gap is not forward-filled: day 3 has no valid B return (NaN prev), only A
    assert p.iloc[2] == pytest.approx(0.0)


# ── zscore ────────────────────────────────────────────────────────────────────
def test_zscore_standardises_population_sd():
    z = md.zscore(pd.Series([1.0, 2.0, 3.0]))
    assert z.mean() == pytest.approx(0.0)
    assert z.std(ddof=0) == pytest.approx(1.0)
    assert z.iloc[2] == pytest.approx(np.sqrt(1.5))


def test_zscore_degenerate_and_inf_handling():
    flat = md.zscore(pd.Series([5.0, 5.0, 5.0], index=list("abc")))
    assert list(flat.index) == list("abc") and (flat == 0.0).all()
    z = md.zscore(pd.Series([1.0, np.inf, 3.0, -np.inf]))
    assert np.isnan(z.iloc[1]) and np.isnan(z.iloc[3])
    assert z.iloc[0] == pytest.approx(-1.0) and z.iloc[2] == pytest.approx(1.0)


def test_zscore_all_nan_and_strings():
    z = md.zscore(pd.Series([np.nan, np.nan], index=[7, 8]))
    assert list(z) == [0.0, 0.0] and list(z.index) == [7, 8]
    z2 = md.zscore(pd.Series(["1", "x", "3"]))
    assert np.isnan(z2.iloc[1]) and z2.iloc[0] == pytest.approx(-1.0)


# ── information_coefficient ───────────────────────────────────────────────────
def test_ic_perfect_positive_and_negative():
    x = list(range(10))
    assert md.information_coefficient(x, [2 * v + 1 for v in x]) == pytest.approx(1.0)
    assert md.information_coefficient(x, [-v for v in x]) == pytest.approx(-1.0)


def test_ic_minimum_sample_boundary():
    assert np.isnan(md.information_coefficient(range(9), range(9)))
    assert md.information_coefficient(range(10), range(10)) == pytest.approx(1.0)


def test_ic_drops_nans_pairwise_and_constant_series():
    s = list(range(12)) + [np.nan]
    r = list(range(12)) + [5.0]
    assert md.information_coefficient(s, r) == pytest.approx(1.0)   # 12 clean pairs
    assert np.isnan(md.information_coefficient([1.0] * 12, list(range(12))))
    assert np.isnan(md.information_coefficient(list(range(12)), [3.0] * 12))
    # 9 clean pairs after NaN removal -> too few
    assert np.isnan(md.information_coefficient(list(range(9)) + [np.nan] * 3, list(range(12))))


def test_ic_ignores_index_alignment():
    s = pd.Series(range(10), index=list("abcdefghij"))
    r = pd.Series(range(10), index=list("jihgfedcba"))     # different labels: positional join
    assert md.information_coefficient(s, r) == pytest.approx(1.0)


# ── monotonicity ──────────────────────────────────────────────────────────────
def test_monotonicity_rising_falling_and_ties():
    up = pd.DataFrame({"r": [1.0, 2.0, 3.0, 4.0]})
    assert md.monotonicity(up, "r") == pytest.approx(1.0)
    assert md.monotonicity(up.iloc[::-1], "r") == pytest.approx(-1.0)
    mixed = pd.DataFrame({"r": [1.0, 3.0, 2.0]})
    assert md.monotonicity(mixed, "r") == pytest.approx(0.5)


def test_monotonicity_guards():
    assert np.isnan(md.monotonicity(None, "r"))
    assert np.isnan(md.monotonicity(pd.DataFrame({"r": [1.0, 2.0]}), "r"))      # < 3 rows
    assert np.isnan(md.monotonicity(pd.DataFrame({"r": [1.0, 2.0, 3.0]}), "x"))  # no col
    assert not np.isnan(md.monotonicity(pd.DataFrame({"r": [1.0, 2.0, 3.0]}), "r"))  # exactly 3


# ── trend_corr ────────────────────────────────────────────────────────────────
def test_trend_corr_directions_and_scale_free():
    assert md.trend_corr([1, 2, 3, 4]) == pytest.approx(1.0)
    assert md.trend_corr([4, 3, 2, 1]) == pytest.approx(-1.0)
    assert md.trend_corr([10, 20, 30]) == pytest.approx(md.trend_corr([1000, 2000, 3000]))


def test_trend_corr_guards_and_nonfinite_dropped():
    assert np.isnan(md.trend_corr([1, 2]))                  # < 3
    assert np.isnan(md.trend_corr([5, 5, 5, 5]))            # flat
    assert np.isnan(md.trend_corr([1, np.nan, np.inf]))     # only one finite value
    assert md.trend_corr([1, np.nan, 2, np.inf, 3]) == pytest.approx(1.0)
