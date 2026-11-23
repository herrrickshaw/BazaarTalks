"""Offline tests for brazil_quality.py: PIT quality, CVM->ticker mapping, forward returns, quintiles, CLI."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import brazil_cvm  # noqa: E402
import brazil_quality as bq  # noqa: E402
import marketdata  # noqa: E402


def rec(cd, filed, roe, **kw):
    r = {"cd_cvm": cd, "filed_date": filed, "roe": roe, "net_income": 1.0}
    r.update(kw)
    return r


# ── quality_asof ──────────────────────────────────────────────────────────────
def test_quality_asof_uses_latest_known_filing_and_filters_implausible():
    panel = [
        rec("1", "2023-03-01", 0.10),
        rec("1", "2024-03-01", 0.20),
        rec("1", "2025-03-01", 0.90),                 # future vs asof
        rec("2", "2024-01-01", 5.0),                  # bank-taxonomy outlier
        rec("3", "2024-01-01", None),
        rec("4", "2024-01-01", -1.0),                 # band edge is plausible
        rec("5", "2024-06-01", 0.3),                  # filed after asof
    ]
    assert bq.quality_asof(panel, "2024-05-01") == {"1": 0.20, "4": -1.0}


def test_quality_asof_non_roe_key_skips_plausibility_but_not_none():
    panel = [rec("1", "2024-01-01", 0.1, net_income=500.0),
             rec("2", "2024-01-01", 0.1, net_income=None),
             rec("3", "2024-01-01", 0.1, net_income=-9e9)]
    out = bq.quality_asof(panel, "2024-02-01", quality_key="net_income")
    assert out == {"1": 500.0, "3": -9e9}


def test_quality_asof_accepts_dataframe_and_empty():
    df = pd.DataFrame([rec("1", "2024-01-01", 0.15)])
    assert bq.quality_asof(df, "2024-12-31") == {"1": 0.15}
    assert bq.quality_asof([], "2024-12-31") == {}


# ── to_tickers ────────────────────────────────────────────────────────────────
def test_norm_cd():
    assert bq._norm_cd("009512") == "9512"
    assert bq._norm_cd(" 4170 ") == "4170"
    assert bq._norm_cd(4170) == "4170"
    assert bq._norm_cd("000") == "0" and bq._norm_cd("") == "0"


def test_to_tickers_maps_zero_padded_codes_and_drops_unknown():
    q = {"004170": 0.2, "1023": 0.1, "99999": 0.5}
    assert bq.to_tickers(q) == {"VALE3.SA": 0.2, "BBAS3.SA": 0.1}


def test_to_tickers_valid_ticker_filter_and_custom_map():
    q = {"4170": 0.2, "1023": 0.1}
    assert bq.to_tickers(q, valid_tickers={"VALE3.SA"}) == {"VALE3.SA": 0.2}
    assert bq.to_tickers(q, valid_tickers=set()) == {}
    assert bq.to_tickers({"0007": 1.0}, ticker_map={"7": "X.SA"}) == {"X.SA": 1.0}
    assert bq.to_tickers({}, ) == {}


def test_ticker_map_is_wellformed():
    assert all(t.endswith(".SA") for t in bq.TICKER_MAP.values())
    assert len(set(bq.TICKER_MAP.values())) == len(bq.TICKER_MAP)
    assert bq.TICKER_MAP["4170"] == "VALE3.SA"


# ── forward_returns ───────────────────────────────────────────────────────────
@pytest.fixture
def close():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])
    return pd.DataFrame({
        "UP": [100.0, 101.0, 102.0, 105.0, 110.0],
        "DOWN": [50.0, 49.0, 48.0, 45.0, 40.0],
        "NAN0": [np.nan, 10.0, 11.0, 12.0, 13.0],
        "NANEND": [10.0, 11.0, 12.0, 13.0, np.nan],
        "NEG": [-5.0, 1.0, 2.0, 3.0, 4.0],
    }, index=idx)


def test_forward_returns_to_last_day_by_default(close):
    r = bq.forward_returns(close, "2024-01-02")
    assert r == {"UP": pytest.approx(0.10), "DOWN": pytest.approx(-0.20)}   # NaN / non-positive dropped


def test_forward_returns_horizon_in_rows_and_start_rounding_up(close):
    r = bq.forward_returns(close, "2024-01-03", horizon_days=2)           # rows 1 -> 3
    assert r["UP"] == pytest.approx(105 / 101 - 1)
    assert r["NAN0"] == pytest.approx(12 / 10 - 1)
    # start on a non-trading day moves to the next trading day (Saturday -> Monday)
    r2 = bq.forward_returns(close, "2024-01-06")
    assert r2["UP"] == 0.0 and r2["DOWN"] == 0.0                           # start == last row


def test_forward_returns_horizon_is_clamped_to_last_row(close):
    r = bq.forward_returns(close, "2024-01-04", horizon_days=100)
    assert r["UP"] == pytest.approx(110 / 102 - 1)


def test_forward_returns_start_after_data_is_empty(close):
    assert bq.forward_returns(close, "2024-02-01") == {}
    assert bq.forward_returns(close, "2024-02-01", horizon_days=3) == {}


def test_forward_returns_excludes_nan_at_either_end(close):
    r = bq.forward_returns(close, "2024-01-02", horizon_days=4)
    assert "NAN0" not in r and "NANEND" not in r and "NEG" not in r


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="brazil_quality.py:~90 `if horizon_days` treats horizon_days=0 like None, "
                          "returning the return to the LAST day instead of 0 days ahead")
def test_forward_returns_zero_horizon_is_zero_return(close):
    r = bq.forward_returns(close, "2024-01-02", horizon_days=0)
    assert r == {"UP": 0.0, "DOWN": 0.0}


# ── quality_quintiles ─────────────────────────────────────────────────────────
def test_quintiles_perfect_monotone_signal():
    keys = [f"T{i}" for i in range(10)]
    quality = {k: float(i) for i, k in enumerate(keys)}
    fwd = {k: 0.01 * i for i, k in enumerate(keys)}
    res = bq.quality_quintiles(quality, fwd, q=5)
    assert res == {"n": 10, "IC": 1.0, "Q1_ret%": 0.5, "Q5_ret%": 8.5,
                   "Q5_minus_Q1%": 8.0, "monotonicity": 1.0}


def test_quintiles_inverted_signal_is_negative():
    keys = [f"T{i}" for i in range(10)]
    quality = {k: float(i) for i, k in enumerate(keys)}
    fwd = {k: -0.01 * i for i, k in enumerate(keys)}
    res = bq.quality_quintiles(quality, fwd)
    assert res["IC"] == -1.0 and res["Q5_minus_Q1%"] == -8.0 and res["monotonicity"] == -1.0


def test_quintiles_only_joined_names_count_and_nan_dropped():
    quality = {f"T{i}": float(i) for i in range(12)}
    fwd = {f"T{i}": 0.01 * i for i in range(10)}
    fwd["T3"] = float("nan")
    quality["ONLY_Q"] = 5.0
    res = bq.quality_quintiles(quality, fwd, q=3)
    assert res["n"] == 9


def test_quintiles_too_few_names_note_and_boundary():
    quality = {f"T{i}": float(i) for i in range(4)}
    assert bq.quality_quintiles(quality, dict(quality), q=5) == {"n": 4, "note": "too few names to bucket"}
    assert bq.quality_quintiles({}, {}) == {"n": 0, "note": "too few names to bucket"}
    five = {f"T{i}": float(i) for i in range(5)}
    res = bq.quality_quintiles(five, dict(five), q=5)                # exactly q names
    assert res["n"] == 5 and "note" not in res
    assert np.isnan(res["IC"])                                       # < 10 pairs -> IC undefined


def test_quintiles_ties_in_quality_are_ranked_first_come():
    quality = {f"T{i}": 1.0 for i in range(10)}
    fwd = {f"T{i}": 0.01 * i for i in range(10)}
    res = bq.quality_quintiles(quality, fwd, q=5)
    assert res["n"] == 10 and np.isnan(res["IC"])                    # constant signal -> no IC
    assert res["Q1_ret%"] == 0.5 and res["Q5_ret%"] == 8.5


# ── main ──────────────────────────────────────────────────────────────────────
def run_main(monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, "argv", ["brazil_quality.py", *argv])
    bq.main()
    return capsys.readouterr().out


def live_inputs(monkeypatch):
    tickers = list(bq.TICKER_MAP.items())[:10]
    idx = pd.bdate_range("2024-01-02", periods=150)
    close = pd.DataFrame({t: np.linspace(10, 10 * (1 + 0.01 * (i + 1)), len(idx))
                          for i, (_, t) in enumerate(tickers)}, index=idx)
    vol = close * 0 + 1000
    panel = [rec(cd, "2023-12-15", 0.05 * (i + 1)) for i, (cd, _) in enumerate(tickers)]
    calls = []

    def filed_panel(y, doc):
        calls.append((y, doc))
        if y == 2022:
            raise RuntimeError("zip missing")
        return list(panel) if y == 2023 else []
    monkeypatch.setattr(brazil_cvm, "filed_panel", filed_panel)
    monkeypatch.setattr(marketdata, "close_volume", lambda m: (close, vol))
    return calls


def test_main_full_pipeline_prints_quintile_result(monkeypatch, capsys):
    calls = live_inputs(monkeypatch)
    out = run_main(monkeypatch, capsys, "2022", "2023", "2024")
    assert calls == [(2022, "DFP"), (2023, "DFP"), (2024, "DFP")]
    assert "  (skip 2022: zip missing)" in out
    assert "fundamentals: 10 filings over [2022, 2023, 2024]" in out
    assert "quality known as of 2024-01-02: 10 companies (10 mapped to liquid B3 tickers)" in out
    assert "'n': 10" in out and "'IC': 1.0" in out
    assert "short 1-year price cache" in out


def test_main_default_years(monkeypatch, capsys):
    calls = live_inputs(monkeypatch)
    run_main(monkeypatch, capsys)
    assert [c[0] for c in calls] == [2022, 2023, 2024]


def test_main_reports_live_run_unavailable(monkeypatch, capsys):
    monkeypatch.setattr(brazil_cvm, "filed_panel", lambda y, d: [])

    def boom(m):
        raise OSError("no parquet")
    monkeypatch.setattr(marketdata, "close_volume", boom)
    out = run_main(monkeypatch, capsys, "2023")
    assert "(live run unavailable: no parquet)" in out
