"""Offline tests for risk.py (pure return-series metrics + CLI)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import risk  # noqa: E402


def test_equity_curve_compounds():
    eq = risk.equity_curve([0.1, -0.5, 0.2])
    assert eq.tolist() == pytest.approx([1.1, 0.55, 0.66])
    assert risk.equity_curve([]).size == 0


def test_max_drawdown_known_and_empty():
    assert risk.max_drawdown([]) == 0.0
    assert risk.max_drawdown([0.1, -0.5, 0.2]) == pytest.approx(-0.5)
    # monotonically rising -> no drawdown (exactly 0, not negative)
    assert risk.max_drawdown([0.01, 0.02, 0.03]) == 0.0
    # worst trough is later than the first dip
    assert risk.max_drawdown([-0.1, 0.5, -0.2, -0.2]) == pytest.approx(0.64 - 1)


def test_hist_var_positive_loss_fraction():
    r = np.linspace(-0.10, 0.10, 101)            # 5% quantile = -0.09
    assert risk.hist_var(r, 0.05) == pytest.approx(0.09)
    assert risk.hist_var(r, 0.5) == pytest.approx(0.0, abs=1e-12)
    assert risk.hist_var([], 0.05) == 0.0


def test_cvar_is_mean_of_tail_and_ge_var():
    r = np.linspace(-0.10, 0.10, 101)
    # values <= -0.09 : -0.10, -0.098, ... -0.09 (6 pts) -> mean -0.095
    assert risk.cvar(r, 0.05) == pytest.approx(0.095)
    assert risk.cvar(r, 0.05) >= risk.hist_var(r, 0.05)
    assert risk.cvar([], 0.05) == 0.0


def test_cvar_all_nan_falls_back_to_threshold():
    out = risk.cvar([np.nan, np.nan, np.nan], 0.05)
    assert np.isnan(out)


def test_ann_vol_scaling_and_degenerate():
    r = np.array([0.01, -0.01, 0.01, -0.01])
    expected = r.std(ddof=1) * np.sqrt(252)
    assert risk.ann_vol(r) == pytest.approx(expected)
    assert risk.ann_vol(r, periods=12) == pytest.approx(r.std(ddof=1) * np.sqrt(12))
    assert risk.ann_vol([0.05]) == 0.0
    assert risk.ann_vol([]) == 0.0


def test_ann_return():
    assert risk.ann_return([]) == 0.0
    # 252 obs of +0.1% -> exactly the full compounded return
    r = np.full(252, 0.001)
    assert risk.ann_return(r) == pytest.approx(1.001 ** 252 - 1)
    # half a year of data is annualised up
    r2 = np.full(126, 0.001)
    assert risk.ann_return(r2) == pytest.approx(1.001 ** 252 - 1)
    assert risk.ann_return([0.1, 0.1], periods=2) == pytest.approx(0.21)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="risk.py:84 `sd > 0` guard trips on float noise: constant 0.01 series "
                          "has sd~1e-18, so sharpe returns ~8.7e16 instead of 0.0")
def test_sharpe_constant_series_float_noise():
    assert risk.sharpe(np.full(10, 0.01)) == 0.0


def test_sharpe_zero_vol_and_rf():
    assert risk.sharpe(np.full(10, 2.0 ** -6)) == 0.0     # sd exactly 0 -> 0, not inf
    r = np.array([0.02, 0.00, 0.02, 0.00])
    base = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert risk.sharpe(r) == pytest.approx(base)
    ex = r - 0.05 / 252
    assert risk.sharpe(r, rf=0.05) == pytest.approx(ex.mean() / ex.std(ddof=1) * np.sqrt(252))
    assert risk.sharpe(r, rf=0.05) < risk.sharpe(r)


def test_sortino_downside_only():
    r = np.array([0.03, -0.01, 0.02, -0.03, 0.01])
    dd = np.array([-0.01, -0.03]).std(ddof=1)
    assert risk.sortino(r) == pytest.approx(r.mean() / dd * np.sqrt(252))
    # fewer than 2 losing periods -> downside dev undefined -> 0.0
    assert risk.sortino([0.01, 0.02, -0.01]) == 0.0
    assert risk.sortino([0.01, 0.02, 0.03]) == 0.0
    # two identical losses -> dd == 0 -> 0.0
    assert risk.sortino([0.05, -0.01, -0.01]) == 0.0


def test_regime_flag_unknown_when_short():
    out = risk.regime_flag(np.zeros(63), window=63)       # needs window+1
    assert out == {"regime": "unknown", "vol_ratio": None, "in_drawdown": None}
    assert risk.regime_flag(np.zeros(64), window=63)["regime"] != "unknown"


def test_regime_flag_constant_returns_is_risk_on():
    out = risk.regime_flag(np.full(20, 2.0 ** -9), window=5)
    assert out == {"regime": "risk_on", "vol_ratio": 1.0, "in_drawdown": False}


def _calm_then(tail, n_calm=40):
    calm = [0.002, -0.002] * (n_calm // 2)
    return np.array(calm + list(tail))


def test_regime_flag_risk_off_needs_vol_spike_and_drawdown():
    r = _calm_then([0.05, -0.12, 0.04, -0.10])
    out = risk.regime_flag(r, window=4)
    assert out["regime"] == "risk_off"
    assert out["in_drawdown"] is True
    assert out["vol_ratio"] > 1.3


def test_regime_flag_caution_vol_only():
    # vol spike but ending at a fresh equity peak -> not in drawdown
    r = _calm_then([-0.10, 0.12, -0.05, 0.20])
    out = risk.regime_flag(r, window=4)
    assert out["vol_ratio"] > 1.15
    assert out["in_drawdown"] is False
    assert out["regime"] == "caution"


def test_regime_flag_caution_drawdown_only():
    # steady (low, constant vol) decline of >2% -> drawdown but ratio ~1
    r = np.concatenate([np.full(20, 0.01), np.full(10, -0.005)])
    out = risk.regime_flag(r, window=5)
    assert out["in_drawdown"] is True
    assert out["vol_ratio"] <= 1.15
    assert out["regime"] == "caution"


def test_regime_flag_drawdown_threshold_is_two_percent():
    # drop of exactly ~1% from peak is not a drawdown
    r = np.array([0.01] * 10 + [-0.01] + [0.0] * 5)
    assert risk.regime_flag(r, window=5)["in_drawdown"] is False


def test_risk_report_fields_and_rounding():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, 300)
    rep = risk.risk_report(r)
    assert rep["n"] == 300
    assert rep["ann_return%"] == round(risk.ann_return(r) * 100, 2)
    assert rep["ann_vol%"] == round(risk.ann_vol(r) * 100, 2)
    assert rep["sharpe"] == round(risk.sharpe(r), 2)
    assert rep["sortino"] == round(risk.sortino(r), 2)
    assert rep["max_dd%"] == round(risk.max_drawdown(r) * 100, 1)
    assert rep["VaR95%"] == round(risk.hist_var(r) * 100, 2)
    assert rep["CVaR95%"] == round(risk.cvar(r) * 100, 2)
    assert rep["CVaR95%"] >= rep["VaR95%"]
    assert rep["regime"] in {"risk_on", "caution", "risk_off"}


def test_risk_report_short_series_has_unknown_regime():
    rep = risk.risk_report([0.01, -0.02, 0.01])
    assert rep["n"] == 3
    assert rep["regime"] == "unknown" and rep["vol_ratio"] is None


# ── data helper + CLI ─────────────────────────────────────────────────────────
def _fake_long():
    dates = pd.date_range("2024-01-01", periods=4)
    rows = []
    for sym, px in {"A": [100, 110, 121, 121], "B": [50, 50, 55, 66]}.items():
        for d, p in zip(dates, px):
            rows.append({"Date": d, "Symbol": sym, "Close": p})
    # duplicate row: aggfunc="last" must keep the later value
    rows.append({"Date": dates[0], "Symbol": "A", "Close": 100})
    return pd.DataFrame(rows)


def test_market_proxy_returns_missing_parquet_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(risk, "SEED", str(tmp_path))
    with pytest.raises(SystemExit) as ei:
        risk.market_proxy_returns("ZZ")
    assert "ZZ" in str(ei.value.code)


def test_market_proxy_returns_equal_weight(tmp_path, monkeypatch):
    monkeypatch.setattr(risk, "SEED", str(tmp_path))
    (tmp_path / "cleaned_long_US.parquet").write_bytes(b"x")
    monkeypatch.setattr(risk.pd, "read_parquet", lambda p: _fake_long())
    r = risk.market_proxy_returns("US")
    assert len(r) == 3                                    # first NaN row dropped
    assert r.iloc[0] == pytest.approx((0.10 + 0.0) / 2)
    assert r.iloc[1] == pytest.approx((0.10 + 0.10) / 2)
    assert r.iloc[2] == pytest.approx((0.0 + 0.20) / 2)


def test_main_prints_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(risk, "SEED", str(tmp_path))
    (tmp_path / "cleaned_long_US.parquet").write_bytes(b"x")
    monkeypatch.setattr(risk.pd, "read_parquet", lambda p: _fake_long())
    monkeypatch.setattr(sys, "argv", ["risk.py", "--market", "US", "--periods", "12"])
    risk.main()
    out = capsys.readouterr()
    assert "risk report — US equal-weight proxy (3 days)" in out.err
    lines = out.out.splitlines()
    assert lines[0].split()[0] == "n" and lines[0].split()[1] == "3"
    keys = [ln.split()[0] for ln in lines]
    assert keys == ["n", "ann_return%", "ann_vol%", "sharpe", "sortino", "max_dd%",
                    "VaR95%", "CVaR95%", "regime", "vol_ratio", "in_drawdown"]
    # keys are left-padded to the widest key
    assert lines[0].startswith("  n" + " " * (len("ann_return%") - 1))


def test_main_default_market_is_us(tmp_path, monkeypatch, capsys):
    seen = {}

    def fake(m):
        seen["m"] = m
        return pd.Series([0.01, -0.01, 0.02])

    monkeypatch.setattr(risk, "market_proxy_returns", fake)
    monkeypatch.setattr(sys, "argv", ["risk.py"])
    risk.main()
    assert seen["m"] == "US"
    assert "(3 days)" in capsys.readouterr().err


def test_sortino_ignores_exactly_zero_excess_returns_as_downside():
    import risk
    # only one strictly negative excess return -> downside deviation is undefined (0) -> 0.0
    assert risk.sortino([0.0, 0.0, -0.01, 0.02]) == 0.0
    # with a second strictly negative return it is defined and positive-mean => positive ratio
    assert risk.sortino([0.0, -0.01, -0.02, 0.05]) > 0
