"""Offline tests for crowding.py (co-movement crowding proxy)."""
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import crowding as cw  # noqa: E402


# ── corr_to_market ────────────────────────────────────────────────────────────
def _rets(n, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 0.01, n), index=pd.RangeIndex(n))


def test_corr_to_market_perfect_and_inverse():
    m = _rets(30)
    assert cw.corr_to_market(m * 2, m) == pytest.approx(1.0)
    assert cw.corr_to_market(-m, m) == pytest.approx(-1.0)


def test_corr_to_market_min_observations_boundary():
    m = _rets(25)
    s = m * 1.5
    assert not np.isnan(cw.corr_to_market(s.iloc[:20], m.iloc[:20]))       # exactly 20 ok
    assert np.isnan(cw.corr_to_market(s.iloc[:19], m.iloc[:19]))           # 19 -> NaN


def test_corr_to_market_counts_only_overlapping_non_nan():
    m = _rets(30)
    s = (m * 2).copy()
    s.iloc[:11] = np.nan                       # 19 overlapping left
    assert np.isnan(cw.corr_to_market(s, m))
    s2 = (m * 2).copy()
    s2.iloc[:10] = np.nan                      # 20 left
    assert cw.corr_to_market(s2, m) == pytest.approx(1.0)


def test_corr_to_market_zero_variance_is_nan():
    m = _rets(30)
    flat = pd.Series(2.0 ** -6, index=m.index)
    assert np.isnan(cw.corr_to_market(flat, m))
    assert np.isnan(cw.corr_to_market(m, flat))


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="crowding.py:45 `std() == 0` guard misses float noise: a constant 0.01 "
                          "series has std~1e-18 so corr_to_market returns ~1e-17 instead of NaN")
def test_corr_to_market_constant_series_float_noise():
    m = _rets(30)
    assert np.isnan(cw.corr_to_market(pd.Series(0.01, index=m.index), m))


def test_corr_to_market_aligns_on_index():
    m = _rets(40)
    s = (m * 3).iloc[10:]                      # misaligned start, 30 overlapping
    assert cw.corr_to_market(s, m) == pytest.approx(1.0)


# ── rel_strength ──────────────────────────────────────────────────────────────
def test_rel_strength_uses_window_start_price():
    c = pd.Series(np.arange(1.0, 101.0))
    # c[-63] = 38, c[-1] = 100
    assert cw.rel_strength(c, 63) == pytest.approx(100 / 38 - 1)
    assert cw.rel_strength(c, 10) == pytest.approx(100 / 91 - 1)


def test_rel_strength_length_boundary():
    c = pd.Series(np.arange(1.0, 65.0))        # 64 points = lookback+1
    assert not np.isnan(cw.rel_strength(c, 63))
    assert np.isnan(cw.rel_strength(c.iloc[1:], 63))    # 63 points -> NaN


def test_rel_strength_nonpositive_start_and_nan_handling():
    c = pd.Series(np.arange(1.0, 101.0))
    c.iloc[-63] = 0.0
    assert np.isnan(cw.rel_strength(c, 63))
    c2 = pd.Series([np.nan] + list(np.arange(1.0, 65.0)))
    assert cw.rel_strength(c2, 63) == pytest.approx(64 / 2 - 1)    # NaN dropped first


# ── crowding_score ────────────────────────────────────────────────────────────
def test_crowding_score_weights_and_range():
    feat = pd.DataFrame({"corr_mkt": [0.1, 0.5, 0.9], "rel_strength": [0.9, 0.5, 0.1]})
    s = cw.crowding_score(feat)
    # pct ranks: corr [1/3,2/3,1], rs [1,2/3,1/3]
    exp = 100 * (0.65 * np.array([1 / 3, 2 / 3, 1]) + 0.35 * np.array([1, 2 / 3, 1 / 3]))
    assert s.tolist() == pytest.approx(np.round(exp, 1).tolist())
    assert s.iloc[2] > s.iloc[0]               # co-movement outweighs run-up
    assert ((s >= 0) & (s <= 100)).all()


def test_crowding_score_is_robust_to_outliers():
    a = pd.DataFrame({"corr_mkt": [0.1, 0.5, 0.9], "rel_strength": [0.1, 0.2, 0.3]})
    b = a.copy(); b.loc[2, "rel_strength"] = 1e9
    assert cw.crowding_score(a).tolist() == cw.crowding_score(b).tolist()


def test_crowding_score_handles_strings_and_nan():
    feat = pd.DataFrame({"corr_mkt": ["0.2", "bad", "0.8"], "rel_strength": [0.1, 0.2, np.nan]})
    s = cw.crowding_score(feat)
    assert np.isnan(s.iloc[1]) and np.isnan(s.iloc[2])
    assert s.iloc[0] == 50.0                    # ranks among non-NaN only: 0.5 and 0.5


# ── scan_market ───────────────────────────────────────────────────────────────
def _install_fakes(monkeypatch, wide, liquid=None):
    lf = types.ModuleType("liquidity_factor")
    lf._market_wide = lambda m: wide
    pfm = types.ModuleType("pead_factor")
    pfm._liquid_symbols = lambda close, vol: list(close.columns) if liquid is None else liquid
    monkeypatch.setitem(sys.modules, "liquidity_factor", lf)
    monkeypatch.setitem(sys.modules, "pead_factor", pfm)


def _close(n=120, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n)
    common = rng.normal(0.0005, 0.01, n)
    cols = {
        "HERD": 100 * np.cumprod(1 + common + rng.normal(0, 0.001, n)),
        "LONER": 100 * np.cumprod(1 + rng.normal(0, 0.01, n)),
        "FLAT": np.full(n, 50.0),                           # zero variance -> NaN corr -> skipped
        "SHORT": np.concatenate([np.full(n - 30, np.nan), 100 * np.cumprod(1 + rng.normal(0, 0.01, 30))]),
    }
    return pd.DataFrame(cols, index=idx)


def test_scan_market_no_data_returns_empty(monkeypatch):
    _install_fakes(monkeypatch, None)
    out = cw.scan_market("US")
    assert isinstance(out, pd.DataFrame) and out.empty


def test_scan_market_rows_skip_nonfinite_and_score(monkeypatch):
    close = _close()
    _install_fakes(monkeypatch, (close, close * 0 + 1))
    df = cw.scan_market("US")
    assert set(df["ticker"]) == {"HERD", "LONER"}           # FLAT: nan corr; SHORT: <20 obs / <64 px
    assert (df["market"] == "US").all()
    assert {"corr_mkt", "rel_strength", "crowding"} <= set(df.columns)
    herd = df.set_index("ticker").loc["HERD"]
    assert df["corr_mkt"].between(-1, 1).all()
    assert herd["corr_mkt"] == round(herd["corr_mkt"], 3)    # rounded to 3dp
    assert df["crowding"].between(0, 100).all()


def test_scan_market_respects_liquid_filter_and_window(monkeypatch):
    close = _close()
    _install_fakes(monkeypatch, (close, close * 0 + 1), liquid=["HERD"])
    df = cw.scan_market("US", window=30)
    assert df["ticker"].tolist() == ["HERD"]
    # window controls rel_strength lookback: c[-30] -> c[-1]
    exp = close["HERD"].iloc[-1] / close["HERD"].iloc[-30] - 1
    assert df["rel_strength"].iloc[0] == pytest.approx(exp, abs=5e-4)


def test_scan_market_all_skipped_gives_empty_without_crowding_column(monkeypatch):
    close = _close()[["FLAT"]]
    _install_fakes(monkeypatch, (close, close))
    df = cw.scan_market("US")
    assert df.empty and "crowding" not in df.columns


def test_scan_market_clips_extreme_returns(monkeypatch):
    close = _close()
    close.loc[close.index[-5], "LONER"] = close["LONER"].iloc[-6] * 20
    _install_fakes(monkeypatch, (close, close * 0 + 1))
    df = cw.scan_market("US").set_index("ticker")
    assert -1.0 <= df.loc["LONER", "corr_mkt"] <= 1.0


# ── CLI ───────────────────────────────────────────────────────────────────────
def _fake_scan(m, window=cw.WINDOW):
    return pd.DataFrame({"market": [m, m], "ticker": [f"{m}1", f"{m}2"],
                         "corr_mkt": [0.5, 0.9], "rel_strength": [0.1, 0.3],
                         "crowding": [40.0, 90.0]})


def test_main_no_args_scans_all_markets_sorted(monkeypatch, capsys):
    monkeypatch.setattr(cw.marketdata, "market_list", lambda: ["US", "JP"])
    monkeypatch.setattr(cw, "scan_market", _fake_scan)
    monkeypatch.setattr(sys, "argv", ["crowding.py", "--top", "3"])
    cw.main()
    out = capsys.readouterr().out.splitlines()
    assert "CO-MOVEMENT CROWDING — 4 liquid names" in out[1]
    assert "NOT short-interest crowding" in out[3]
    rows = [ln.split() for ln in out[5:]]
    assert len(rows) == 3                                   # --top honoured
    assert [r[-1] for r in rows] == ["90.00", "90.00", "40.00"]
    assert rows[0][2] == "0.90" and rows[0][3] == "30.0"


def test_main_single_market(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(cw, "scan_market", lambda m: (seen.append(m), _fake_scan(m))[1])
    monkeypatch.setattr(sys, "argv", ["crowding.py", "--market", "KR"])
    cw.main()
    assert seen == ["KR"]


def test_main_all_flag_overrides_market(monkeypatch):
    seen = []
    monkeypatch.setattr(cw.marketdata, "market_list", lambda: ["US", "JP"])
    monkeypatch.setattr(cw, "scan_market", lambda m: (seen.append(m), _fake_scan(m))[1])
    monkeypatch.setattr(sys, "argv", ["crowding.py", "--all", "--market", "KR"])
    cw.main()
    assert seen == ["US", "JP"]


def test_main_empty_exits(monkeypatch):
    monkeypatch.setattr(cw, "scan_market", lambda m: pd.DataFrame())
    monkeypatch.setattr(cw.marketdata, "market_list", lambda: ["US"])
    monkeypatch.setattr(sys, "argv", ["crowding.py"])
    with pytest.raises(SystemExit) as ei:
        cw.main()
    assert str(ei.value.code) == "no data"
