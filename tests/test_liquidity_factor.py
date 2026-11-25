"""Offline tests for liquidity_factor.py (Amihud ILLIQ, capacity score, premium study)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import liquidity_factor as lf  # noqa: E402
import marketdata  # noqa: E402


def make_long(n_syms=10, n_days=300, seed=0):
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


# ── amihud / proxies ──────────────────────────────────────────────────────────
def test_amihud_value_and_scale():
    assert lf.amihud_illiq([0.01] * 5, [1e6] * 5) == pytest.approx(0.01)
    assert lf.amihud_illiq([-0.02] * 6, [2e6] * 6) == pytest.approx(0.01)       # abs return
    assert lf.amihud_illiq([0.01] * 5, [1e6] * 5, scale=1.0) == pytest.approx(1e-8)
    # average of ratios, not ratio of averages
    assert lf.amihud_illiq([0.01, 0.02, 0.01, 0.02, 0.01], [1e6, 1e6, 1e6, 1e6, 1e6]) \
        == pytest.approx(0.014)


def test_amihud_minimum_five_valid_days_boundary():
    assert np.isnan(lf.amihud_illiq([0.01] * 4, [1e6] * 4))
    assert not np.isnan(lf.amihud_illiq([0.01] * 5, [1e6] * 5))
    # invalid days (zero / negative / NaN / inf dollar volume) are dropped before counting
    r = [0.01] * 8
    dv = [1e6, 1e6, 1e6, 1e6, 0.0, -5.0, np.nan, np.inf]
    assert np.isnan(lf.amihud_illiq(r, dv))
    dv5 = [1e6] * 5 + [0.0, np.nan, np.inf]
    assert lf.amihud_illiq(r, dv5) == pytest.approx(0.01)
    r_nan = [np.nan] + [0.01] * 5
    assert lf.amihud_illiq(r_nan, [1e6] * 6) == pytest.approx(0.01)


def test_amihud_accepts_series_and_empty():
    s = pd.Series([0.01] * 6)
    assert lf.amihud_illiq(s, s * 0 + 1e6) == pytest.approx(0.01)
    assert np.isnan(lf.amihud_illiq([], []))


def test_zero_return_frac_threshold_and_nan_handling():
    assert lf.zero_return_frac([0.0, 0.0, 0.01, 0.02]) == 0.5
    assert lf.zero_return_frac([1e-4, -1e-4]) == 0.0            # strictly < 1e-4
    assert lf.zero_return_frac([0.99e-4, -0.99e-4]) == 1.0
    assert lf.zero_return_frac([np.nan, 0.0, 0.5]) == 0.5       # NaN dropped from the base
    assert np.isnan(lf.zero_return_frac([]))
    assert np.isnan(lf.zero_return_frac([np.nan, np.nan]))


def test_turnover():
    assert lf.turnover([100, 200, np.nan], 1000) == pytest.approx(0.3)
    assert np.isnan(lf.turnover([100], 0))
    assert np.isnan(lf.turnover([100], None))
    assert lf.turnover([], 10) == 0.0


def test_capacity_score_inverts_illiq_rank():
    s = pd.Series([1.0, 3.0, 2.0], index=list("abc"))
    out = lf.capacity_score(s)
    assert out.to_dict() == {"a": 66.7, "b": 0.0, "c": 33.3}
    assert out.index.tolist() == ["a", "b", "c"]


def test_capacity_score_nan_and_strings():
    s = pd.Series([1.0, np.nan, "2.0", "bad"])
    out = lf.capacity_score(s)
    assert out.iloc[0] == 50.0 and out.iloc[2] == 0.0
    assert out.iloc[1:2].isna().all() and np.isnan(out.iloc[3])


def test_illiq_pctile_is_percent_rank():
    s = pd.Series([10.0, 30.0, 20.0, 40.0])
    assert lf.illiq_pctile(s).tolist() == [25.0, 75.0, 50.0, 100.0]
    assert lf.illiq_pctile(pd.Series([5.0, 5.0])).tolist() == [75.0, 75.0]    # ties average


# ── premium study ─────────────────────────────────────────────────────────────
def _panel(n=50, sign=1.0):
    il = np.arange(1, n + 1, dtype=float)
    return pd.DataFrame({"illiq": il, "fwd_ret": sign * il / 1000.0})


def test_premium_by_illiq_monotone():
    c = lf.premium_by_illiq(_panel())
    assert list(c.columns) == ["bucket", "mean_fwd%", "median_fwd%", "n"]
    assert list(c["bucket"]) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert c["n"].tolist() == [10] * 5
    assert c["mean_fwd%"].tolist() == [0.55, 1.55, 2.55, 3.55, 4.55]
    assert lf.monotonicity(c) == pytest.approx(1.0)
    assert lf.monotonicity(lf.premium_by_illiq(_panel(sign=-1.0))) == pytest.approx(-1.0)


def test_premium_by_illiq_boundaries():
    assert lf.premium_by_illiq(_panel(n=14)).empty              # < q*3
    assert len(lf.premium_by_illiq(_panel(n=15))) == 5
    p = _panel(n=15)
    p.loc[0, "fwd_ret"] = np.nan
    assert lf.premium_by_illiq(p).empty                         # NaN dropped -> 14
    assert len(lf.premium_by_illiq(_panel(), q=2)) == 2


def test_monotonicity_wrapper_edge_cases():
    c = lf.premium_by_illiq(_panel())
    assert np.isnan(lf.monotonicity(c.head(2)))
    assert np.isnan(lf.monotonicity(c, col="nope"))


# ── scan_market ───────────────────────────────────────────────────────────────
def test_scan_market_missing_parquet(seed):
    pan, sc = lf.scan_market("XX")
    assert pan.empty and sc.empty


def test_scan_market_panel_and_scores(seed):
    df = make_long()
    seed("US", df)
    pan, sc = lf.scan_market("US")
    assert len(pan) == 10 and len(sc) == 10
    assert set(pan.columns) == {"market", "ticker", "illiq", "fwd_ret"}
    assert set(sc.columns) == {"market", "ticker", "illiq", "capacity_score", "illiq_pctile"}
    # volume rises with k -> S0 the most illiquid, S9 the most liquid
    sc = sc.set_index("ticker")
    assert sc.loc["S0", "capacity_score"] == 0.0
    assert sc.loc["S9", "capacity_score"] == 90.0
    assert sc.loc["S0", "illiq_pctile"] == 100.0
    assert sc.sort_index()["illiq"].is_monotonic_decreasing
    # fwd return is exactly close[T+h]/close[T]-1 with T = n-h-1
    close = df[df.Symbol == "S3"].sort_values("Date")["Close"].to_numpy()
    T = 300 - 21 - 1
    got = pan.set_index("ticker").loc["S3", "fwd_ret"]
    assert got == pytest.approx(close[T + 21] / close[T] - 1)


def test_scan_market_honours_lookback_and_horizon(seed):
    seed("US", make_long(n_days=300))
    pan, sc = lf.scan_market("US", lookback=60, horizon=10)
    c = make_long(n_days=300)
    assert len(pan) == 10
    s0 = c[c.Symbol == "S0"].sort_values("Date")["Close"].to_numpy()
    assert pan.set_index("ticker").loc["S0", "fwd_ret"] == pytest.approx(s0[299] / s0[289] - 1)
    # ILLIQ differs when the lookback window changes
    sc2 = lf.scan_market("US", lookback=126, horizon=10)[1].set_index("ticker")
    assert sc.set_index("ticker").loc["S0", "illiq"] != sc2.loc["S0", "illiq"]


def test_scan_market_short_history_skipped_at_boundary(seed):
    seed("US", make_long(n_days=151))       # lookback+horizon+5 = 152 -> too short
    pan, sc = lf.scan_market("US")
    assert pan.empty and sc.empty
    seed("US", make_long(n_days=152))
    pan, sc = lf.scan_market("US")
    assert len(sc) == 10 and len(pan) == 10     # T=130 > lookback=126


def test_scan_market_nonpositive_start_price_dropped_from_panel(seed):
    df = make_long()
    T = 300 - 21 - 1
    bad = (df.Symbol == "S2") & (df.Date == pd.bdate_range("2024-01-01", periods=300)[T])
    df.loc[bad, "Close"] = np.nan
    seed("US", df)
    pan, sc = lf.scan_market("US")
    assert "S2" not in set(pan["ticker"])
    assert "S2" in set(sc["ticker"])


# ── CLI ───────────────────────────────────────────────────────────────────────
def _run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["liquidity_factor.py", *argv])
    lf.main()


def _fake_scan(sign):
    def fake(m, lookback, horizon):
        n = 30
        il = np.arange(1, n + 1, dtype=float)
        pan = pd.DataFrame({"market": m, "ticker": [f"T{i}" for i in range(n)], "illiq": il,
                            "fwd_ret": sign * il / 1000.0})
        sc = pan[["market", "ticker", "illiq"]].copy()
        sc["capacity_score"] = lf.capacity_score(sc["illiq"])
        sc["illiq_pctile"] = lf.illiq_pctile(sc["illiq"])
        return pan, sc
    return fake


def test_main_premium_present(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(lf, "scan_market", _fake_scan(1.0))
    out = tmp_path / "liq.csv"
    _run_main(monkeypatch, "--market", "US", "--top", "2", "--out", str(out))
    text = capsys.readouterr().out
    assert "LIQUIDITY PREMIUM — US (30 stock-observations, 21d forward)" in text
    assert "monotonicity = 1.00  ->  liquidity premium PRESENT (illiquid earn more, monotone)" in text
    assert "premium (Q5−Q1 forward) = 2.40%" in text           # mean(25..30)/10 - mean(1..6)/10
    assert "MOST liquid: T0, T1" in text
    assert "LEAST liquid: T28, T29" in text
    w = pd.read_csv(out)
    assert list(w.columns) == ["ticker", "liquidity_score"]
    assert len(w) == 30 and w.set_index("ticker").loc["T0", "liquidity_score"] == 96.7


def test_main_premium_absent_when_inverted(monkeypatch, capsys):
    monkeypatch.setattr(lf, "scan_market", _fake_scan(-1.0))
    _run_main(monkeypatch, "--market", "JP", "--lookback", "50", "--horizon", "7")
    text = capsys.readouterr().out
    assert "7d forward" in text and "JP" in text
    assert "weak/absent" in text and "PRESENT" not in text


def test_main_no_panel_exits(monkeypatch):
    monkeypatch.setattr(lf, "scan_market", lambda *a, **k: (pd.DataFrame(), pd.DataFrame()))
    with pytest.raises(SystemExit) as e:
        _run_main(monkeypatch, "--market", "US")
    assert "no liquidity panel" in str(e.value)


def test_main_end_to_end_on_synthetic_seed(seed, monkeypatch, capsys):
    seed("US", make_long())
    _run_main(monkeypatch, "--market", "US", "--top", "3")
    text = capsys.readouterr().out
    assert "LIQUIDITY PREMIUM — US (10 stock-observations" in text
    assert "forward return by Amihud ILLIQ quantile" not in text    # 10 obs < 15 -> no curve
    assert "MOST liquid: S9, S8, S7" in text


@pytest.mark.xfail(strict=True, raises=NameError,
                   reason="liquidity_factor.py:main `marketdata.market_list()` - `marketdata` is never imported "
                          "at module level, so --all / no --market raises NameError")
def test_main_all_markets(seed, monkeypatch, capsys):
    seed("US", make_long())
    _run_main(monkeypatch, "--all")
    assert "LIQUIDITY PREMIUM" in capsys.readouterr().out
