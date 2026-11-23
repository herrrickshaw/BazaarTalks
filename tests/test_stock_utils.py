"""Offline tests for stock_utils (statement helpers, OHLCV cleaning, bulk download, numerics)."""
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import stock_utils as su  # noqa: E402

NAN = float("nan")


# --------------------------------------------------------------- statements
def stmt():
    return pd.DataFrame({"2025": [10.0, NAN], "2024": [20.0, 5.0], "2023": [30.0, 6.0]},
                        index=["Net Income", "Gross Profit"])


def test_first_df_returns_first_non_empty_dataframe_attribute():
    t = types.SimpleNamespace(a=pd.DataFrame(), b=None, c="notadf", d=stmt(), e=stmt() * 2)
    assert su.first_df(t, "a", "b", "c", "missing", "d", "e") is t.d
    assert su.first_df(t, "a", "b", "c", "missing") is None
    assert su.first_df(t) is None


def test_row_extracts_floats_with_alias_and_column_selection():
    df = stmt()
    assert su.row(df, "Net Income") == 10.0
    assert su.row(df, "Net Income", col=2) == 30.0
    assert su.row(df, "Nope", "Net Income") == 10.0
    assert su.row(df, "Gross Profit") is None            # NaN -> None
    assert su.row(df, "Net Income", col=9) is None        # out of range
    assert su.row(df, "Missing") is None
    assert su.row(None, "x") is None and su.row(df.iloc[0:0], "x") is None


def test_series_skips_nans_and_uses_first_known_alias():
    df = stmt()
    assert su.series(df, "Gross Profit") == [5.0, 6.0]
    assert su.series(df, "Nope", "Net Income") == [10.0, 20.0, 30.0]
    assert su.series(df, "Nope") == [] and su.series(None, "x") == [] and su.series(df.iloc[0:0], "x") == []


def test_fin_metric_default_only_when_missing():
    df = stmt()
    assert su.fin_metric(df, ("Net Income",)) == 10.0
    assert su.fin_metric(df, ("Gross Profit",), default=-1.0) == -1.0
    assert su.fin_metric(df, ("Zzz",)) is None
    df2 = pd.DataFrame({"x": [0.0]}, index=["Zero"])
    assert su.fin_metric(df2, ("Zero",), default=9.0) == 0.0     # zero is a value, not missing


# ------------------------------------------------------------- clean_ohlcv
def ohlcv(n=5, **over):
    idx = pd.date_range("2026-01-01", periods=n)
    d = {"Open": [10.0 + i for i in range(n)], "High": [11.0 + i for i in range(n)],
         "Low": [9.0 + i for i in range(n)], "Close": [10.5 + i for i in range(n)],
         "Volume": [1000.0] * n}
    d.update(over)
    return pd.DataFrame(d, index=idx)


def test_clean_ohlcv_passthrough_is_idempotent():
    df = ohlcv()
    out = su.clean_ohlcv(df)
    pd.testing.assert_frame_equal(out, df)
    pd.testing.assert_frame_equal(su.clean_ohlcv(out), out)


def test_clean_ohlcv_rejects_empty_none_and_missing_close():
    assert su.clean_ohlcv(None) is None
    assert su.clean_ohlcv(pd.DataFrame()) is None
    assert su.clean_ohlcv(ohlcv().drop(columns="Close")) is None


def test_clean_ohlcv_does_not_mutate_input():
    df = ohlcv()
    df.iloc[1, df.columns.get_loc("Close")] = NAN
    snapshot = df.copy()
    su.clean_ohlcv(df)
    pd.testing.assert_frame_equal(df, snapshot)


def test_clean_ohlcv_sorts_dedups_keeping_last_and_drops_extra_columns():
    df = ohlcv(3)
    df["Adj Close"] = 1.0
    dup = df.iloc[[0]].copy()
    dup["Close"] = 10.9
    dup["High"] = 12.0
    messy = pd.concat([df.iloc[[2, 1]], df.iloc[[0]], dup])      # unsorted, 2026-01-01 twice
    out = su.clean_ohlcv(messy)
    assert list(out.columns) == su.OHLCV_COLS
    assert out.index.is_monotonic_increasing and len(out) == 3
    assert out.iloc[0]["Close"] == 10.9                           # last duplicate wins


def test_clean_ohlcv_coerces_strings_and_drops_rows_without_close():
    df = ohlcv(4).astype(object)
    df.iloc[1, df.columns.get_loc("Close")] = "garbage"
    out = su.clean_ohlcv(df)
    assert len(out) == 3 and out["Close"].dtype.kind == "f"


def test_clean_ohlcv_fills_ohlc_from_close_and_volume_with_zero_clipped():
    df = ohlcv(3)
    df.iloc[1, df.columns.get_loc("Open")] = NAN
    df.iloc[1, df.columns.get_loc("Volume")] = NAN
    df.iloc[2, df.columns.get_loc("Volume")] = -5.0
    out = su.clean_ohlcv(df)
    assert out.iloc[1]["Open"] == out.iloc[1]["Close"]                # missing Open filled from Close
    assert out["Volume"].tolist() == [1000.0, 0.0, 0.0]


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_clean_ohlcv_drops_non_positive_prices(bad):
    df = ohlcv(4)
    df.iloc[2, df.columns.get_loc("Low")] = bad
    assert len(su.clean_ohlcv(df)) == 3
    all_bad = ohlcv(2, Close=[bad, bad])
    assert su.clean_ohlcv(all_bad) is None


def test_clean_ohlcv_repairs_high_low_envelope():
    df = ohlcv(2)
    df.iloc[0, df.columns.get_loc("High")] = 1.0                  # High below Open/Close: invalid
    df.iloc[1, df.columns.get_loc("Low")] = 99.0                  # Low above Open/Close: invalid
    out = su.clean_ohlcv(df)
    first, second = out.iloc[0], out.iloc[1]
    assert first["High"] == max(first["Open"], first["Close"], 9.0)     # raised to the true max
    assert second["Low"] == min(second["Open"], second["Close"])        # lowered to the true min
    for _, r in out.iterrows():
        assert r["High"] >= max(r["Open"], r["Close"], r["Low"]) and r["Low"] <= min(r["Open"], r["Close"], r["High"])


def test_clean_ohlcv_neutralises_huge_jump_only_on_zero_volume():
    closes = [10.0, 10.0, 100.0, 10.0]
    df = ohlcv(4, Open=closes, High=closes, Low=closes, Close=closes, Volume=[100.0, 100.0, 0.0, 100.0])
    out = su.clean_ohlcv(df)
    assert out["Close"].tolist() == [10.0, 10.0, 10.0, 10.0]      # bad print replaced by previous close
    df2 = ohlcv(4, Open=closes, High=closes, Low=closes, Close=closes, Volume=[100.0] * 4)
    assert su.clean_ohlcv(df2)["Close"].tolist() == closes          # real move with volume is kept


def test_clean_ohlcv_jump_threshold_is_strict_and_configurable():
    closes = [10.0, 15.0]                                          # +50 % exactly
    df = ohlcv(2, Open=closes, High=closes, Low=closes, Close=closes, Volume=[0.0, 0.0])
    assert su.clean_ohlcv(df, max_daily_move=0.5)["Close"].tolist() == closes     # not strictly greater
    assert su.clean_ohlcv(df, max_daily_move=0.49)["Close"].tolist() == [10.0, 10.0]


def test_clean_ohlcv_min_bars_and_verbose(capsys):
    assert su.clean_ohlcv(ohlcv(3), min_bars=4) is None
    assert su.clean_ohlcv(ohlcv(3), min_bars=3) is not None
    df = ohlcv(3)
    df.iloc[1, df.columns.get_loc("Close")] = NAN
    su.clean_ohlcv(df, ticker="ABC", verbose=True)
    assert "clean_ohlcv[ABC]: dropped 1 rows, neutralised 0 bad-print bars -> 2 clean bars" in capsys.readouterr().out
    su.clean_ohlcv(ohlcv(3), ticker="ABC", verbose=True)
    assert capsys.readouterr().out == ""                           # nothing dropped -> quiet


def test_clean_ohlcv_close_only_frame_and_unsortable_index():
    out = su.clean_ohlcv(pd.DataFrame({"Close": [1.0, 2.0]}, index=pd.date_range("2026-01-01", periods=2)))
    assert out["Close"].tolist() == [1.0, 2.0]
    mixed = pd.DataFrame({"Close": [1.0, 2.0]}, index=pd.Index([1, "a"], dtype=object))
    assert len(su.clean_ohlcv(mixed)) == 2                          # sort failure is tolerated


# ---------------------------------------------------------- clean_financials
def test_clean_financials_sorts_newest_first_and_drops_empty_columns():
    df = pd.DataFrame({"2023": [1, 2], "2025": [3, 4], "2024": [NAN, NAN]}, index=["a", "b"]).astype(object)
    df.loc["a", "2023"] = "7"
    out = su.clean_financials(df)
    assert list(out.columns) == ["2025", "2023"]
    assert out.loc["a", "2023"] == 7.0


def test_clean_financials_none_for_empty_or_all_nan():
    assert su.clean_financials(None) is None and su.clean_financials(pd.DataFrame()) is None
    assert su.clean_financials(pd.DataFrame({"x": [NAN]})) is None


def test_clean_financials_keeps_already_sorted_and_tolerates_mixed_column_types():
    df = pd.DataFrame({"2025": [1.0], "2024": [2.0]})
    assert list(su.clean_financials(df).columns) == ["2025", "2024"]
    mixed = pd.DataFrame([[1.0, 2.0]], columns=pd.Index(["a", 5], dtype=object))
    assert list(su.clean_financials(mixed).columns) == ["a", 5]       # unsortable: left as is


# --------------------------------------------------------- extract_ticker_df
def bulk(tickers=("AAA", "BBB")):
    idx = pd.date_range("2026-01-01", periods=4)
    cols = pd.MultiIndex.from_product([su.OHLCV_COLS, list(tickers)])
    data = np.tile(np.arange(1, 5, dtype=float).reshape(4, 1), (1, len(cols)))
    return pd.DataFrame(data, index=idx, columns=cols)


def test_extract_ticker_df_multiindex_and_flat():
    raw = bulk()
    out = su.extract_ticker_df(raw, "AAA")
    assert list(out.columns) == su.OHLCV_COLS and len(out) == 4
    flat = ohlcv(3)
    assert len(su.extract_ticker_df(flat, "ANY")) == 3


def test_extract_ticker_df_missing_ticker_or_empty_is_none():
    assert su.extract_ticker_df(bulk(), "ZZZ") is None
    raw = bulk()
    raw[("Close", "AAA")] = NAN
    for c in su.OHLCV_COLS:
        raw[(c, "AAA")] = NAN
    assert su.extract_ticker_df(raw, "AAA") is None
    assert su.extract_ticker_df(pd.DataFrame(), "AAA") is None


def test_extract_ticker_df_min_bars_and_clean_flag():
    assert su.extract_ticker_df(bulk(), "AAA", min_bars=5) is None
    assert su.extract_ticker_df(bulk(), "AAA", min_bars=4) is not None
    dirty = ohlcv(3)
    dirty.iloc[1, dirty.columns.get_loc("Close")] = -1.0
    assert len(su.extract_ticker_df(dirty, "X", clean=False)) == 3
    assert len(su.extract_ticker_df(dirty, "X", clean=True)) == 2
    assert su.extract_ticker_df(dirty, "X", min_bars=4, clean=False) is None
    extra = ohlcv(3)
    extra["Adj Close"] = 1.0
    assert list(su.extract_ticker_df(extra, "X", clean=False).columns) == su.OHLCV_COLS
    nocols = pd.DataFrame({"Z": [1, 2]})
    assert list(su.extract_ticker_df(nocols, "X", clean=False).columns) == ["Z"]


# ------------------------------------------------------------- rate limiting
@pytest.mark.parametrize("msg, expected", [("HTTP 429", True), ("Too Many Requests", True),
                                           ("YFRateLimitError: Rate limited", True), ("Invalid Crumb", True),
                                           ("connection reset", False), ("", False)])
def test_is_rate_limited(msg, expected):
    assert su.is_rate_limited(Exception(msg)) is expected


def test_strip_suffix():
    assert su._strip_suffix("RELIANCE.NS") == "RELIANCE" and su._strip_suffix("X.BO") == "X"
    assert su._strip_suffix("AAPL") == "AAPL"


# -------------------------------------------------------------- bulk_download
class FakeYF:
    def __init__(self, script):
        self.script = list(script)          # list of results or exceptions, consumed per call
        self.calls = []

    def download(self, batch, **kw):
        self.calls.append((list(batch), kw))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def nosleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(su.time, "sleep", lambda s: sleeps.append(s))
    return sleeps


def install_yf(monkeypatch, script):
    fake = FakeYF(script)
    monkeypatch.setattr(su, "yf", fake, raising=False)
    monkeypatch.setattr(su, "_YF_OK", True)
    return fake


def big_bulk(tickers, n=40):
    idx = pd.date_range("2026-01-01", periods=n)
    cols = pd.MultiIndex.from_product([su.OHLCV_COLS, list(tickers)])
    return pd.DataFrame(np.full((n, len(cols)), 10.0), index=idx, columns=cols)


def test_bulk_download_without_yfinance_returns_empty(monkeypatch):
    monkeypatch.setattr(su, "_YF_OK", False)
    assert su.bulk_download(["A"]) == {}


def test_bulk_download_batches_strips_suffix_and_sleeps_between_batches(monkeypatch, nosleep, capsys):
    fake = install_yf(monkeypatch, [big_bulk(["A.NS", "B.NS"]), big_bulk(["C"])])
    out = su.bulk_download(["A.NS", "B.NS", "C"], batch_size=2, sleep_between=0.25, period="6mo")
    assert sorted(out) == ["A", "B", "C"]
    assert [c[0] for c in fake.calls] == [["A.NS", "B.NS"], ["C"]]
    assert fake.calls[0][1] == {"auto_adjust": True, "threads": True, "progress": False, "period": "6mo"}
    assert nosleep == [0.25]                                         # only between batches
    text = capsys.readouterr().out
    assert "bulk_download: 3 tickers, 2 batches (size=2, period=6mo)" in text
    assert "Batch 1/2 (2)" in text and "OK (2/2)" in text and "bulk_download complete: 3/3 loaded" in text


def test_bulk_download_start_overrides_period_and_min_bars_filters(monkeypatch, nosleep):
    fake = install_yf(monkeypatch, [big_bulk(["A"], n=10)])
    out = su.bulk_download(["A"], start="2026-01-01", min_bars=30, verbose=False)
    assert out == {} and "start" in fake.calls[0][1] and "period" not in fake.calls[0][1]


def test_bulk_download_empty_response_skips_batch(monkeypatch, nosleep, capsys):
    install_yf(monkeypatch, [pd.DataFrame(), None])
    assert su.bulk_download(["A", "B"], batch_size=1) == {}
    assert capsys.readouterr().out.count("empty") == 2


def test_bulk_download_retries_rate_limit_with_growing_backoff(monkeypatch, nosleep):
    fake = install_yf(monkeypatch, [Exception("HTTP 429"), Exception("Too Many Requests"), big_bulk(["A"])])
    out = su.bulk_download(["A"], max_retries=3, verbose=False)
    assert list(out) == ["A"] and len(fake.calls) == 3
    assert nosleep == [20, 40]


def test_bulk_download_gives_up_after_max_retries_and_on_other_errors(monkeypatch, nosleep, capsys):
    fake = install_yf(monkeypatch, [Exception("429"), Exception("429")])
    assert su.bulk_download(["A"], max_retries=2) == {}
    assert len(fake.calls) == 2 and nosleep == [20]                  # last attempt does not sleep first
    fake2 = install_yf(monkeypatch, [ValueError("boom")])
    assert su.bulk_download(["A"], max_retries=3) == {}
    assert len(fake2.calls) == 1 and "ERROR: boom" in capsys.readouterr().out


# ----------------------------------------------------------------- parallel
def test_parallel_map_collects_non_none_results_and_swallows_errors(capsys):
    def fn(x):
        if x == 3:
            raise RuntimeError("bad")
        return None if x % 2 == 0 else x * 10
    out = su.parallel_map(fn, [1, 2, 3, 4, 5], workers=2, progress_every=2, label="things")
    assert sorted(out) == [10, 50]
    text = capsys.readouterr().out
    assert "5/5 things processed, 2 results" in text and "2/5 things" in text and "4/5 things" in text
    assert "1/5" not in text and "3/5" not in text


def test_parallel_map_quiet_and_empty(capsys):
    assert su.parallel_map(lambda x: x, [], verbose=False) == []
    assert sorted(su.parallel_map(lambda x: x, [0, 1], verbose=False)) == [0, 1]
    assert capsys.readouterr().out == ""


def test_parallel_map_zero_result_is_kept():
    assert su.parallel_map(lambda x: 0, [1], verbose=False) == [0]


# ------------------------------------------------------------------ numerics
@pytest.mark.parametrize("new, old, expected", [(110, 100, 10.0), (90, 100, -10.0), (5, -10, 150.0),
                                                (1, 0, None), (None, 1, None), (1, None, None), (0, 100, -100.0)])
def test_pct_change(new, old, expected):
    got = su.pct_change(new, old)
    assert got == expected if expected is None else got == pytest.approx(expected)


@pytest.mark.parametrize("latest, earliest, years, expected", [
    (121, 100, 2, 10.0), (200, 100, 1, 100.0), (50, 100, 1, -50.0),
    (0, 100, 2, None), (100, 0, 2, None), (100, -5, 2, None), (100, 50, 0, None), (None, 5, 2, None),
])
def test_cagr(latest, earliest, years, expected):
    got = su.cagr(latest, earliest, years)
    assert got == expected if expected is None else got == pytest.approx(expected)


@pytest.mark.parametrize("raw, expected", [(None, None), (0.45, 0.45), (10, 10), (10.01, 0.1001), (45.2, 0.452)])
def test_normalise_debt_to_equity(raw, expected):
    got = su.normalise_debt_to_equity(raw)
    assert got == expected if expected is None else got == pytest.approx(expected)


def test_market_cap_crores():
    assert su.market_cap_crores(5e9) == 500.0 and su.market_cap_crores(None) == 0.0 and su.market_cap_crores(0) == 0.0


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="stock_utils.py:cagr() guards `not latest` and `earliest <= 0` but not latest < 0; "
                          "a negative latest value raises a fractional power of a negative float and returns "
                          "a COMPLEX number instead of None.")
def test_cagr_negative_latest_is_none():
    assert su.cagr(-5.0, 10.0, 2) is None
