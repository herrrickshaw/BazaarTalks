"""Offline tests for accumulation_screener.py (CMF / accumulation signal + validation)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import accumulation_screener as acc  # noqa: E402
import darvas_volume as dv  # noqa: E402
import marketdata  # noqa: E402


def make_long(n_syms=20, n_days=300, seed=0):
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    frames = []
    for k in range(n_syms):
        ret = rng.normal(0.0005, 0.01, n_days)
        vol = rng.uniform(0.8, 1.2, n_days) * 1e5 * (k + 1)
        close = 50.0 * (k + 1) * np.cumprod(1 + ret)
        frames.append(pd.DataFrame({"Date": dates, "Symbol": f"S{k}", "Close": close,
                                    "High": close * 1.01, "Low": close * 0.99, "Volume": vol}))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def seed(tmp_path, monkeypatch):
    store = {}

    def put(market, df):
        (tmp_path / f"cleaned_long_{market}.parquet").write_bytes(b"x")
        store[f"cleaned_long_{market}.parquet"] = df

    monkeypatch.setattr(pd, "read_parquet", lambda p, *a, **k: store[os.path.basename(p)].copy())
    monkeypatch.setattr(marketdata, "SEED", str(tmp_path))
    monkeypatch.setattr(marketdata, "SEED_LOCAL", str(tmp_path / "nolocal"))
    return put


# ── accumulation_signal ───────────────────────────────────────────────────────
def _accumulating(n=30):
    close = np.linspace(10, 20, n)
    return close + 0.2, close - 1.0, close, np.linspace(100, 300, n)


def test_accumulation_signal_is_sum_of_components():
    h, l, c, v = _accumulating()
    sig = acc.accumulation_signal(h, l, c, v)
    assert set(sig) == {"cmf", "accum"}
    expected = (dv.chaikin_money_flow(h, l, c, v) + dv.trend_corr(dv.obv(c, v))
                + dv.trend_corr(dv.chaikin_ad(h, l, c, v)) + dv.trend_corr(v) + 0.0)  # inf U/D -> 0
    assert sig["cmf"] == pytest.approx(dv.chaikin_money_flow(h, l, c, v))
    assert sig["accum"] == pytest.approx(expected)
    assert sig["accum"] > 3


def test_accumulation_signal_ud_term_bounded_and_finite_handling():
    # alternating closes: up/down volume ratio is 2 -> tanh(0.5*ln 2) added
    c = np.array([10, 11, 10, 11, 10, 11, 10, 11, 10, 11.0])
    v = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2.0])
    h, l = c + 0.5, c - 0.5
    ud = dv.up_down_volume_ratio(c, v)
    assert ud == pytest.approx(2.0 * 5 / 4 * 1.0) or ud > 1
    parts = [dv.chaikin_money_flow(h, l, c, v), dv.trend_corr(dv.obv(c, v)),
             dv.trend_corr(dv.chaikin_ad(h, l, c, v)), dv.trend_corr(v), np.tanh(0.5 * np.log(ud))]
    expected = float(np.sum([p for p in parts if np.isfinite(p)]))
    assert acc.accumulation_signal(h, l, c, v)["accum"] == pytest.approx(expected)


def test_accumulation_signal_constant_volume_skips_nan_trend():
    h, l, c, _ = _accumulating()
    v = np.full(len(c), 100.0)                           # trend_corr(volume) is NaN -> excluded, not NaN-poisoned
    sig = acc.accumulation_signal(h, l, c, v)
    assert np.isfinite(sig["accum"])


# ── information_coefficient / quantile_returns / monotonicity ─────────────────
def test_information_coefficient_reexport_and_thresholds():
    assert acc.information_coefficient is marketdata.information_coefficient
    x = np.arange(10.0)
    assert acc.information_coefficient(x, x * 2) == pytest.approx(1.0)
    assert acc.information_coefficient(x, -x) == pytest.approx(-1.0)
    assert np.isnan(acc.information_coefficient(x[:9], x[:9]))          # < 10 pairs
    assert np.isnan(acc.information_coefficient(x, np.ones(10)))         # constant return
    assert np.isnan(acc.information_coefficient(np.ones(12), np.arange(12.0)))


def _qpanel(n=50, sign=1.0):
    s = np.arange(1, n + 1, dtype=float)
    return pd.DataFrame({"sig": s, "fwd_ret": sign * s / 1000.0})


def test_quantile_returns_values_and_columns():
    c = acc.quantile_returns(_qpanel(), "sig")
    assert list(c.columns) == ["bucket", "median_fwd%", "mean_fwd%", "n"]
    assert list(c["bucket"]) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert c["n"].tolist() == [10] * 5
    assert c["median_fwd%"].tolist() == [0.55, 1.55, 2.55, 3.55, 4.55]
    assert c["mean_fwd%"].tolist() == c["median_fwd%"].tolist()


def test_quantile_returns_median_resists_outliers():
    p = _qpanel()
    p.loc[49, "fwd_ret"] = 50.0                                # one 5000% penny outlier in Q5
    c = acc.quantile_returns(p, "sig")
    assert c["median_fwd%"].iloc[-1] == 4.45 or c["median_fwd%"].iloc[-1] < 5
    assert c["mean_fwd%"].iloc[-1] > 100                       # mean is blown up, median is not


def test_quantile_returns_boundaries_and_custom_cols():
    assert acc.quantile_returns(_qpanel(n=14), "sig").empty
    assert len(acc.quantile_returns(_qpanel(n=15), "sig")) == 5
    p = _qpanel(n=15)
    p.loc[0, "sig"] = np.nan
    assert acc.quantile_returns(p, "sig").empty                 # NaN dropped -> 14 rows
    p2 = _qpanel().rename(columns={"fwd_ret": "r"})
    assert len(acc.quantile_returns(p2, "sig", q=2, ret_col="r")) == 2


def test_monotonicity_wrapper():
    inc = acc.quantile_returns(_qpanel(), "sig")
    dec = acc.quantile_returns(_qpanel(sign=-1.0), "sig")
    assert acc.monotonicity(inc) == pytest.approx(1.0)
    assert acc.monotonicity(dec) == pytest.approx(-1.0)
    assert acc.monotonicity(inc, col="mean_fwd%") == pytest.approx(1.0)
    assert np.isnan(acc.monotonicity(inc.head(2)))


@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason="accumulation_screener.py:73 pd.qcut(..., labels=[Q1..Q5], duplicates='drop') raises "
                          "'Bin labels must be one fewer than the number of bin edges' when ties collapse bin "
                          "edges (e.g. a signal that is 0 for most names)")
def test_quantile_returns_tied_signal_degrades_instead_of_raising():
    p = pd.DataFrame({"sig": [0.0] * 30 + list(np.arange(1.0, 11.0)), "fwd_ret": np.linspace(0, 0.1, 40)})
    out = acc.quantile_returns(p, "sig")
    assert len(out) < 5 and out["n"].sum() == 40


# ── build_panel / current_screen ──────────────────────────────────────────────
def test_build_panel_missing_and_too_short(seed):
    assert acc.build_panel("XX", 21).empty
    seed("US", make_long())
    assert acc.build_panel("US", 300).empty                      # T negative
    assert acc.build_panel("US", 21, sig_window=300).empty       # T - sig_window < 0


def test_build_panel_point_in_time_alignment(seed):
    df = make_long()
    seed("US", df)
    panel = acc.build_panel("US", 21)
    assert set(panel["ticker"]) == {f"S{k}" for k in range(12, 20)}
    assert set(panel.columns) == {"market", "ticker", "fwd_ret", "cmf", "accum"}
    T = 300 - 21 - 1
    s = df[df.Symbol == "S15"].sort_values("Date").reset_index(drop=True)
    row = panel.set_index("ticker").loc["S15"]
    assert row["fwd_ret"] == pytest.approx(s["Close"][T + 21] / s["Close"][T] - 1)
    w = s.iloc[T - 42:T]                                         # signal window ends BEFORE T
    expect = acc.accumulation_signal(w.High.values, w.Low.values, w.Close.values, w.Volume.values)
    assert row["cmf"] == pytest.approx(expect["cmf"]) and row["accum"] == pytest.approx(expect["accum"])


def test_build_panel_skips_bad_prices_and_sparse_windows(seed):
    df = make_long()
    dates = pd.bdate_range("2024-01-01", periods=300)
    T = 300 - 21 - 1
    df.loc[(df.Symbol == "S19") & (df.Date == dates[T]), "Close"] = np.nan          # no start price
    df.loc[(df.Symbol == "S18") & (df.Date == dates[T + 21]), "Close"] = np.nan     # no end price
    win = (df.Symbol == "S17") & (df.Date >= dates[T - 42]) & (df.Date < dates[T - 20])
    df.loc[win, "Close"] = np.nan                                                    # 22 of 42 missing -> 20 < 21
    win2 = (df.Symbol == "S16") & (df.Date >= dates[T - 42]) & (df.Date < dates[T - 21])
    df.loc[win2, "Close"] = np.nan                                                   # 21 missing -> 21 finite == 42//2
    seed("US", df)
    got = set(acc.build_panel("US", 21)["ticker"])
    assert "S19" not in got and "S18" not in got and "S17" not in got
    assert "S16" in got


def test_current_screen(seed):
    df = make_long()
    seed("US", df)
    assert acc.current_screen("XX").empty
    scr = acc.current_screen("US")
    assert set(scr["ticker"]) == {f"S{k}" for k in range(12, 20)}
    s = df[df.Symbol == "S14"].sort_values("Date").iloc[-42:]
    row = scr.set_index("ticker").loc["S14"]
    assert row["close"] == round(float(s["Close"].iloc[-1]), 2)
    assert row["cmf"] == pytest.approx(dv.chaikin_money_flow(s.High, s.Low, s.Close, s.Volume))
    assert scr["market"].eq("US").all()
    assert acc.current_screen("US", sig_window=300).empty         # len(c) < sig_window + 2


# ── _report ───────────────────────────────────────────────────────────────────
def test_report_lines(capsys):
    rng = np.random.RandomState(0)
    p = pd.DataFrame({"cmf": np.arange(1, 51.0), "accum": rng.normal(size=50)})
    p["fwd_ret"] = p["cmf"] / 1000.0
    acc._report(p, "1-MONTH (21d)")
    out = capsys.readouterr().out
    assert "--- 1-MONTH (21d) forward (50 stock-obs) ---" in out
    cmf_line = [ln for ln in out.splitlines() if "by cmf" in ln][0]
    assert "Q1=+0.6 Q2=+1.6 Q3=+2.5 Q4=+3.5 Q5=+4.5" in cmf_line
    assert "Q5−Q1 median +4.00%" in cmf_line and "IC=+1.000" in cmf_line and "mono=+1.00" in cmf_line
    assert "by accum" in out


def test_report_too_few_obs(capsys):
    p = pd.DataFrame({"cmf": [1.0, 2.0], "accum": [1.0, 2.0], "fwd_ret": [0.1, 0.2]})
    acc._report(p, "X")
    out = capsys.readouterr().out
    assert "cmf: too few obs" in out and "accum: too few obs" in out


# ── CLI ───────────────────────────────────────────────────────────────────────
def _run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["accumulation_screener.py", *argv])
    acc.main()


def test_main_end_to_end(seed, monkeypatch, capsys, tmp_path):
    seed("US", make_long())
    out_csv = tmp_path / "scr.csv"
    _run_main(monkeypatch, "--market", "US", "--top", "3", "--out", str(out_csv))
    cap = capsys.readouterr()
    assert "VALIDATION (point-in-time, look-ahead-free)" in cap.out
    assert "--- 1-MONTH (21d) forward (8 stock-obs) ---" in cap.out
    assert "--- 6-MONTH (126d) forward (8 stock-obs) ---" in cap.out
    assert "CURRENT SCREEN — top 3 by accum — US (8 liquid names)" in cap.out
    shown = [ln.split()[1] for ln in cap.out.splitlines() if ln.startswith("  US")]
    assert len(shown) == 3
    written = pd.read_csv(out_csv)
    assert len(written) == 8 and {"ticker", "cmf", "accum", "close"} <= set(written.columns)
    assert written["accum"].is_monotonic_decreasing               # sorted by the chosen signal
    assert shown == written["ticker"].head(3).tolist()
    assert f"wrote {out_csv}" in cap.err


def test_main_signal_cmf_sort_and_outlier_filter(monkeypatch, capsys):
    monkeypatch.setattr(acc, "build_panel", lambda m, h, w: pd.DataFrame({
        "cmf": np.arange(1, 21.0), "accum": np.arange(1, 21.0), "fwd_ret": [5.0] * 5 + [0.01] * 15}))
    monkeypatch.setattr(acc, "current_screen", lambda m, w: pd.DataFrame({
        "market": "US", "ticker": ["A", "B", "C"], "close": [1.0, 2.0, 3.0],
        "cmf": [0.1, 0.9, 0.5], "accum": [3.0, 1.0, 2.0]}))
    _run_main(monkeypatch, "--market", "US", "--signal", "cmf", "--sig-window", "30")
    out = capsys.readouterr().out
    assert "forward (15 stock-obs)" in out                        # fwd_ret >= 3.0 rows (5) removed
    assert "top 15 by cmf" in out
    assert [ln.split()[1] for ln in out.splitlines() if ln.startswith("  US")] == ["B", "C", "A"]


def test_main_insufficient_history_message(monkeypatch, capsys):
    monkeypatch.setattr(acc, "build_panel", lambda m, h, w: pd.DataFrame())
    monkeypatch.setattr(acc, "current_screen", lambda m, w: pd.DataFrame({
        "market": ["US"], "ticker": ["A"], "close": [1.0], "cmf": [0.1], "accum": [1.0]}))
    _run_main(monkeypatch, "--market", "US")
    out = capsys.readouterr().out
    assert "1-MONTH (21d): insufficient history for this horizon" in out
    assert "6-MONTH (126d): insufficient history for this horizon" in out


def test_main_passes_horizons_and_window(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(acc, "build_panel", lambda m, h, w: calls.append((m, h, w)) or pd.DataFrame())
    monkeypatch.setattr(acc, "current_screen", lambda m, w: pd.DataFrame({
        "market": ["US"], "ticker": ["A"], "close": [1.0], "cmf": [0.1], "accum": [1.0]}))
    _run_main(monkeypatch, "--market", "JP", "--sig-window", "33")
    assert calls == [("JP", 21, 33), ("JP", 126, 33)]


@pytest.mark.xfail(strict=True, raises=NameError,
                   reason="accumulation_screener.py:main `marketdata.market_list()` - module `marketdata` is "
                          "never imported (only `from marketdata import information_coefficient`), so --all / "
                          "no --market raises NameError")
def test_main_all_markets(monkeypatch, capsys):
    monkeypatch.setattr(acc, "build_panel", lambda m, h, w: pd.DataFrame())
    monkeypatch.setattr(acc, "current_screen", lambda m, w: pd.DataFrame({
        "market": ["US"], "ticker": ["A"], "close": [1.0], "cmf": [0.1], "accum": [1.0]}))
    _run_main(monkeypatch, "--all")
    assert "CURRENT SCREEN" in capsys.readouterr().out


@pytest.mark.xfail(strict=True, raises=KeyError,
                   reason="accumulation_screener.py:main when every market is missing, current_screen returns "
                          "an empty frame with no columns and scr.dropna(subset=[signal]) raises KeyError "
                          "instead of printing an empty screen")
def test_main_missing_market_prints_empty_screen(seed, monkeypatch, capsys):
    _run_main(monkeypatch, "--market", "ZZ")
    assert "CURRENT SCREEN" in capsys.readouterr().out
