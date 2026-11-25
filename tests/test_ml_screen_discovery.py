"""Offline tests for ml_screen_discovery.py (supervised + unsupervised + CEM-RL screen discovery)."""
import json
import os
import sqlite3
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ml_screen_discovery as msd  # noqa: E402


def _ohlcv(n=300, drift=0.0005, amp=0.01, seed=0, price0=100.0, vol=1e6):
    rng = np.random.default_rng(seed)
    close = price0 * np.cumprod(1 + drift + amp * rng.standard_normal(n))
    return pd.DataFrame({
        "Close": close, "Volume": vol * (1 + 0.1 * np.abs(rng.standard_normal(n))),
    }, index=pd.bdate_range("2021-01-04", periods=n))


# ── _features_asof ────────────────────────────────────────────────────────────
def test_features_none_and_short_history():
    assert msd._features_asof(None) is None
    assert msd._features_asof(_ohlcv(259)) is None
    assert msd._features_asof(_ohlcv(260)) is not None


def test_features_keys_and_values_match_manual_computation():
    df = _ohlcv(300, seed=1)
    f = msd._features_asof(df)
    assert set(f) == set(msd.FEATURES) | {"_fwd"}
    c, v = df["Close"], df["Volume"]
    assert f["ret_20d"] == pytest.approx((c.iloc[-1] / c.iloc[-21] - 1) * 100)
    assert f["ret_60d"] == pytest.approx((c.iloc[-1] / c.iloc[-61] - 1) * 100)
    assert f["vol_20d"] == pytest.approx(c.pct_change().tail(20).std() * np.sqrt(252) * 100)
    assert f["dist_52w_high"] == pytest.approx((c.iloc[-1] / c.tail(252).max() - 1) * 100)
    assert f["dist_200dma"] == pytest.approx((c.iloc[-1] / c.tail(200).mean() - 1) * 100)
    assert f["vol_ratio"] == pytest.approx(v.iloc[-1] / v.tail(20).mean())
    assert f["dollar_vol_log"] == pytest.approx(np.log10(c.iloc[-1] * v.iloc[-1]))
    assert f["_fwd"] == pytest.approx((c.iloc[-1] / c.iloc[-1 - msd.FWD] - 1) * 100)
    d = c.diff()
    rsi = 100 - 100 / (1 + d.tail(14).clip(lower=0).mean() / (-d.tail(14).clip(upper=0)).mean())
    assert f["rsi_14"] == pytest.approx(rsi)
    assert 0 <= f["rsi_14"] <= 100


def test_features_zero_volume_uses_unit_ratio_and_dollar_vol_floor():
    df = _ohlcv(300, seed=2)
    df["Volume"] = 0.0
    f = msd._features_asof(df)
    assert f["vol_ratio"] == 1.0
    assert f["dollar_vol_log"] == 0.0            # log10(max(0, 1))


def test_features_nan_rsi_returns_none():
    # strictly rising series: no losses -> RSI undefined -> the whole stock is dropped
    n = 300
    df = pd.DataFrame({"Close": np.linspace(50, 150, n), "Volume": np.full(n, 1e6)},
                      index=pd.bdate_range("2021-01-04", periods=n))
    assert msd._features_asof(df) is None


def test_features_nan_close_returns_none():
    df = _ohlcv(300, seed=3)
    df.iloc[-1, df.columns.get_loc("Close")] = np.nan
    assert msd._features_asof(df) is None


def test_build_matrix_drops_unscorable_and_indexes_by_ticker():
    ohlc = {"A": _ohlcv(300, seed=1), "SHORT": _ohlcv(100, seed=2), "NONE": None, "B": _ohlcv(300, seed=3)}
    X = msd.build_matrix(ohlc)
    assert list(X.index) == ["A", "B"]
    assert list(X.columns) == msd.FEATURES + ["_fwd"]


# ── rules ─────────────────────────────────────────────────────────────────────
def _X(n=100, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, len(msd.FEATURES))), columns=msd.FEATURES)
    X["_fwd"] = rng.standard_normal(n) * 5
    return X


def test_rule_from_cluster_uses_interquantile_box():
    X = _X(50)
    mask = np.zeros(50, dtype=bool); mask[:20] = True
    rule = msd.rule_from_cluster(X, mask)
    assert set(rule) == set(msd.FEATURES)
    for f, (lo, hi) in rule.items():
        sub = X.loc[mask, f]
        assert lo == pytest.approx(np.percentile(sub, 20)) and hi == pytest.approx(np.percentile(sub, 80))
        assert lo <= hi
    wide = msd.rule_from_cluster(X, mask, q=(0, 100))
    assert wide["ret_20d"] == (X.loc[mask, "ret_20d"].min(), X.loc[mask, "ret_20d"].max())


def test_apply_rule_bounds_are_inclusive_and_anded():
    X = pd.DataFrame({f: [0.0, 1.0, 2.0, 3.0] for f in msd.FEATURES})
    rule = {f: (1.0, 2.0) for f in msd.FEATURES}
    assert msd.apply_rule(X, rule).tolist() == [False, True, True, False]
    rule["rsi_14"] = (2.0, 2.0)
    assert msd.apply_rule(X, rule).tolist() == [False, False, True, False]
    rule["ret_20d"] = (5.0, 6.0)
    assert not msd.apply_rule(X, rule).any()


def test_apply_rule_empty_rule_selects_everything():
    X = _X(10)
    assert msd.apply_rule(X, {}).all() and len(msd.apply_rule(X, {})) == 10


def test_deviation_is_one_minus_jaccard():
    a = np.array([1, 1, 0, 0], dtype=bool)
    assert msd.deviation(a, a) == 0.0
    assert msd.deviation(a, np.array([0, 0, 1, 1], dtype=bool)) == 1.0
    assert msd.deviation(a, np.array([1, 0, 1, 0], dtype=bool)) == pytest.approx(1 - 1 / 3)
    assert msd.deviation(a, np.array([1, 1, 1, 1], dtype=bool)) == pytest.approx(0.5)


def test_deviation_empty_sets():
    z = np.zeros(4, dtype=bool)
    assert msd.deviation(z, z) == 0.0
    assert msd.deviation(z, np.array([1, 0, 0, 0], dtype=bool)) == 1.0


def test_reward_penalty_for_tiny_screens():
    X = _X(20)
    rule = {f: (-100.0, 100.0) for f in msd.FEATURES}
    none_rule = {f: (100.0, 101.0) for f in msd.FEATURES}
    r, m = msd.reward(X, none_rule, np.ones(20, dtype=bool))
    assert r == -9.9 and not m.any()
    two = {f: (-100.0, 100.0) for f in msd.FEATURES}
    two["ret_20d"] = tuple(sorted(X["ret_20d"].nsmallest(2).values[[0, 1]]))
    r2, m2 = msd.reward(X, two, np.ones(20, dtype=bool))
    assert m2.sum() == 2 and r2 == -9.9           # needs >= 3 picks
    assert msd.reward(X, rule, np.ones(20, dtype=bool))[0] != -9.9


def test_reward_edge_minus_lambda_times_deviation():
    X = _X(30, seed=4)
    rule = {f: (-100.0, 100.0) for f in msd.FEATURES}
    rule["ret_20d"] = (0.0, 100.0)
    known = (X["rsi_14"] > 0).values
    r, m = msd.reward(X, rule, known, lam=2.0)
    edge = X.loc[m, "_fwd"].mean() - X["_fwd"].mean()
    assert r == pytest.approx(edge - 2.0 * msd.deviation(m, known))
    assert msd.reward(X, rule, known)[0] == pytest.approx(edge - 1.5 * msd.deviation(m, known))


def test_reward_selecting_everything_has_zero_edge():
    X = _X(30, seed=5)
    rule = {f: (-100.0, 100.0) for f in msd.FEATURES}
    r, m = msd.reward(X, rule, np.ones(30, dtype=bool))
    assert m.all() and r == pytest.approx(0.0)


# ── rl_refine ─────────────────────────────────────────────────────────────────
def test_rl_refine_never_worse_and_returns_valid_box():
    np.random.seed(0)
    X = _X(80, seed=6)
    known = (X["ret_20d"] > 0).values
    base = {f: (float(X[f].quantile(0.2)), float(X[f].quantile(0.8))) for f in msd.FEATURES}
    r0, _ = msd.reward(X, base, known)
    rule, best = msd.rl_refine(X, base, known, iters=10, pop=30)
    assert best >= r0
    assert set(rule) == set(msd.FEATURES)
    assert all(lo <= hi for lo, hi in rule.values())
    r1, _ = msd.reward(X, rule, known)
    assert r1 == pytest.approx(best)


def test_rl_refine_zero_iterations_returns_input():
    np.random.seed(0)
    X = _X(40)
    base = {f: (-1.0, 1.0) for f in msd.FEATURES}
    rule, best = msd.rl_refine(X, base, np.ones(40, dtype=bool), iters=0)
    assert rule is base
    assert best == msd.reward(X, base, np.ones(40, dtype=bool))[0]


def test_rl_refine_is_reproducible_under_seed():
    X = _X(60, seed=7)
    known = (X["vol_20d"] > 0).values
    base = {f: (-1.0, 1.0) for f in msd.FEATURES}
    np.random.seed(11)
    a = msd.rl_refine(X, base, known, iters=5, pop=20)
    np.random.seed(11)
    b = msd.rl_refine(X, base, known, iters=5, pop=20)
    assert a[1] == b[1] and a[0] == b[0]


# ── load_universe ─────────────────────────────────────────────────────────────
class _XL:
    df = None
    opened = []

    def __init__(self, p):
        _XL.opened.append(p)

    def parse(self, sheet):
        assert sheet == "All_Stocks"
        return _XL.df


@pytest.fixture
def excel(monkeypatch):
    _XL.opened = []
    monkeypatch.setattr(msd.pd, "ExcelFile", _XL)
    monkeypatch.setattr(msd.glob, "glob", lambda p, recursive=False: ["/d/z.xlsx", "/d/a.xlsx"])
    return _XL


PANDAS3 = int(pd.__version__.split(".")[0]) >= 3
nan_bug = pytest.mark.xfail(
    PANDAS3, strict=True, raises=AssertionError,
    reason="ml_screen_discovery.py:158 NaN symbols survive the 'nan' string filter under pandas>=3")


def test_universe_globs_cover_four_markets():
    assert set(msd.UNI) == {"US", "India", "Japan", "Korea"}


def test_load_universe_formats_per_market_and_latest_file(excel):
    excel.df = pd.DataFrame({"Symbol": ["TCS", "INFY"], "Suffix": [".NS", ".BO"],
                             "YF_Ticker": ["7203.T", "005930.KS"]})
    assert msd.load_universe("US", None) == ["TCS", "INFY"]
    assert excel.opened[-1] == "/d/z.xlsx"
    assert msd.load_universe("India", None) == ["TCS.NS", "INFY.BO"]
    assert msd.load_universe("Japan", None) == ["7203.T", "005930.KS"]
    assert msd.load_universe("Korea", None) == ["7203.T", "005930.KS"]


@nan_bug
def test_load_universe_drops_nan_and_applies_limit(excel):
    excel.df = pd.DataFrame({"Symbol": [np.nan, "A", "B", "C"]})
    assert msd.load_universe("US", 2) == ["A", "B"]
    assert msd.load_universe("US", None) == ["A", "B", "C"]


def test_load_universe_limit_zero_means_all(excel):
    excel.df = pd.DataFrame({"Symbol": ["A", "B", "C"]})
    assert msd.load_universe("US", 0) == ["A", "B", "C"]


# ── main ──────────────────────────────────────────────────────────────────────
def _universe(n=60, bars=300):
    out = {}
    for i in range(n):
        out[f"T{i:03d}"] = _ohlcv(bars, drift=-0.001 + 0.00005 * i, amp=0.008 + 0.0002 * (i % 7),
                                  seed=i, price0=20 + i, vol=5e5 * (1 + i % 5))
    return out


@pytest.fixture
def run_main(monkeypatch, tmp_path):
    def _run(*argv, ohlc=None, market_open=True):
        calls = {}
        ms = types.ModuleType("market_store")

        def cached_download(tickers, years):
            calls["dl"] = (list(tickers), years)
            return ohlc

        ms.cached_download = cached_download
        mh = types.ModuleType("market_holidays")
        mh.should_run_today = lambda m: (calls.setdefault("hol", m), market_open)[1]
        monkeypatch.setitem(sys.modules, "market_store", ms)
        monkeypatch.setitem(sys.modules, "market_holidays", mh)
        monkeypatch.setattr(msd, "load_universe", lambda m, lim: calls.setdefault("lu", (m, lim)) and list(ohlc))
        db = tmp_path / "reco.db"
        monkeypatch.setattr(sys, "argv", ["ml_screen_discovery.py", "--db", str(db), *argv])
        np.random.seed(0)
        msd.main()
        return db, calls
    return _run


def _reco(db):
    c = sqlite3.connect(db)
    try:
        return pd.read_sql("SELECT * FROM screen_reco", c)
    finally:
        c.close()


def test_main_too_few_stocks_exits_early_without_db(run_main, capsys):
    ohlc = _universe(n=30)
    db, calls = run_main("--limit", "30", ohlc=ohlc)
    assert not db.exists()
    assert "too few stocks after filtering" in capsys.readouterr().err


def test_main_forty_stocks_is_enough(run_main, capsys):
    db, calls = run_main("--limit", "40", ohlc=_universe(n=40))
    assert db.exists()
    assert "40 liquid stocks with features" in capsys.readouterr().err


def test_main_filters_short_and_illiquid_before_features(run_main, capsys):
    ohlc = _universe(n=45)
    ohlc["ILL"] = _ohlcv(300, seed=99, vol=1.0)
    ohlc["SHORT"] = _ohlcv(260, seed=98)       # len must be > 260
    db, calls = run_main("--dev-threshold", "2", ohlc=ohlc)
    err = capsys.readouterr().err
    assert "45 liquid stocks with features" in err
    assert "market open today? True" in err and calls["hol"] == "US"
    assert calls["dl"][1] == 5


def test_main_writes_ranked_recommendations(run_main, capsys):
    db, calls = run_main("--dev-threshold", "2", ohlc=_universe(60))
    df = _reco(db)
    assert len(df) >= 1
    assert list(df.columns) == ["market", "cluster", "n_picks", "cluster_fwd", "screen_edge",
                                "deviation", "rl_refined", "rule"]
    assert set(df.market) == {"US"}
    assert (df.rl_refined == 0).all()                         # deviation can never exceed 1 <= 2
    assert list(df.screen_edge) == sorted(df.screen_edge, reverse=True)
    assert df.cluster.is_unique and (df.cluster >= 0).all()
    assert df.deviation.between(0, 1).all() and (df.n_picks >= 0).all()
    for rule in df.rule:
        r = json.loads(rule)
        assert set(r) == set(msd.FEATURES) and all(lo <= hi for lo, hi in r.values())
    out = capsys.readouterr().out
    assert "=== NEW SCREEN RECOMMENDATIONS — US (market fwd" in out
    assert "[RL-refined]" not in out
    assert f"{len(df)} outperforming clusters -> new screens (0 RL-refined)" in out


def test_main_cluster_screens_only_for_outperformers(run_main, capsys):
    ohlc = _universe(60)
    db, _ = run_main("--dev-threshold", "2", ohlc=ohlc)
    df = _reco(db)
    X = msd.build_matrix({t: d for t, d in ohlc.items()})
    mkt = X["_fwd"].mean()
    assert (df.cluster_fwd > round(mkt, 2) - 0.01).all()      # every kept cluster beat the market mean


def test_main_forces_rl_when_threshold_negative(run_main, capsys):
    db, _ = run_main("--dev-threshold", "-1", ohlc=_universe(60))
    df = _reco(db)
    assert len(df) >= 1 and (df.rl_refined == 1).all()
    out = capsys.readouterr().out
    assert "[RL-refined]" in out
    assert f"({len(df)} RL-refined)" in out


def test_main_prints_at_most_six_screens(run_main, capsys):
    db, _ = run_main("--dev-threshold", "2", ohlc=_universe(80))
    out = capsys.readouterr().out
    assert out.count("screen#") <= 6


def test_main_market_choice_and_replace_table(run_main, tmp_path):
    ohlc = _universe(60)
    c = sqlite3.connect(tmp_path / "reco.db")
    c.execute("CREATE TABLE screen_reco(junk TEXT)"); c.execute("INSERT INTO screen_reco VALUES ('x')")
    c.commit(); c.close()
    db, calls = run_main("--market", "Japan", "--dev-threshold", "2", ohlc=ohlc)
    df = _reco(db)
    assert set(df.market) == {"Japan"} and "junk" not in df.columns
    assert calls["lu"] == ("Japan", 400) and calls["hol"] == "Japan"


def test_main_rejects_unknown_market(run_main):
    with pytest.raises(SystemExit):
        run_main("--market", "Mars", ohlc=_universe(10))
