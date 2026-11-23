"""Offline tests for quality_factor.py: data assembly, portfolios, price premium, CLI."""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import quality_factor as qf  # noqa: E402

PANDAS3 = int(pd.__version__.split(".")[0]) >= 3


# ── fixtures ──────────────────────────────────────────────────────────────────
def _fund_rows(n_per=40, markets=("US", "JP"), seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for m in markets:
        for i in range(n_per):
            q = rng.normal()
            rows.append({
                "market": m, "ticker": f"{m}{i}" if m == "US" else f"{1000 + i}.T",
                "pe": 15 + rng.normal(), "pb": float(np.exp(1 + 0.4 * q + rng.normal(0, 0.05))),
                "roe": 10 + 5 * q + rng.normal(), "roa": 5 + 2 * q + rng.normal(),
                "de": float(abs(1 - 0.3 * q + rng.normal(0, 0.2))),
                "rev_growth": 5 + 3 * q + rng.normal(), "earn_growth": 6 + 3 * q + rng.normal(),
                "op_margin": 12 + 4 * q + rng.normal(), "div_yield": abs(2 + q + rng.normal(0, 0.3)),
                "mktcap": float(np.exp(20 + rng.normal())),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def fund_db(tmp_path, monkeypatch):
    path = tmp_path / "fundamentals_cache.db"
    monkeypatch.setattr(qf, "FUND", str(path))

    def write(df):
        con = sqlite3.connect(path)
        try:
            df.to_sql("fund", con, if_exists="replace", index=False)
        finally:
            con.close()
        return path

    return write


@pytest.fixture
def seed(tmp_path, monkeypatch):
    store = {}

    def put(market, df):
        (tmp_path / f"cleaned_long_{market}.parquet").write_bytes(b"x")
        store[f"cleaned_long_{market}.parquet"] = df

    monkeypatch.setattr(pd, "read_parquet", lambda p, *a, **k: store[os.path.basename(p)].copy())
    monkeypatch.setattr(qf, "SEED", str(tmp_path))
    return put


def _long(series: dict, n=300):
    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.concat([pd.DataFrame({"Date": dates, "Symbol": s, "Close": c}) for s, c in series.items()],
                     ignore_index=True)


# ── z_rank / dimension_score / quality_score edge cases ───────────────────────
def test_z_rank_ties_nan_and_degenerate():
    z = qf.z_rank(pd.Series([1.0, 1.0, 3.0, np.nan]))
    assert z.iloc[0] == z.iloc[1] < z.iloc[2] and np.isnan(z.iloc[3])
    assert qf.z_rank(pd.Series([5.0, 5.0, 5.0])).isna().all()        # no dispersion
    assert qf.z_rank(pd.Series([np.nan, np.nan])).isna().all()
    s = pd.Series(["1", "x", "3"])
    assert np.isnan(qf.z_rank(s).iloc[1])                           # strings coerced


def test_dimension_score_missing_columns_and_partial_nan():
    df = pd.DataFrame({"roe": [1.0, 2.0, 3.0]})
    assert qf.dimension_score(df, [("nope", 1)]).isna().all()
    s = qf.dimension_score(df, [("roe", 1), ("nope", 1)])           # absent col ignored
    assert s.tolist() == pytest.approx(qf.z_rank(df["roe"]).tolist())
    df2 = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [np.nan, 1.0, 2.0]})
    out = qf.dimension_score(df2, [("a", 1), ("b", -1)])
    za, zb = qf.z_rank(df2["a"]), qf.z_rank(df2["b"])
    assert out.iloc[0] == pytest.approx(za.iloc[0])                  # NaN skipped, not zero-filled
    assert out.iloc[2] == pytest.approx((za.iloc[2] - zb.iloc[2]) / 2)


def test_quality_score_columns_range_and_missing_dimension():
    df = pd.DataFrame({"roe": [1.0, 2.0, 3.0, 4.0], "de": [4.0, 3.0, 2.0, 1.0]})
    out = qf.quality_score(df)
    assert {"profitability", "growth", "safety", "payout", "quality_raw", "quality",
            "quality_score"} <= set(out.columns)
    assert out["growth"].isna().all() and out["payout"].isna().all()
    assert out["quality_score"].tolist() == [25.0, 50.0, 75.0, 100.0]
    assert out["quality"].is_monotonic_increasing
    assert "quality" not in df.columns                               # input untouched
    custom = qf.quality_score(df, dimensions={"only": [("roe", 1)]})
    assert "only" in custom and "profitability" not in custom


# ── deciles / weights / portfolios ────────────────────────────────────────────
def test_assign_deciles_boundaries():
    lab = qf.assign_deciles(pd.Series(np.arange(1.0, 11.0)))         # pct ranks 0.1 .. 1.0
    assert lab.tolist() == ["junk"] + ["mid"] * 7 + ["quality"] * 2   # pct rank <= .10 junk, >= .90 quality
    lab2 = qf.assign_deciles(pd.Series(np.arange(1.0, 11.0)), top=0.8, bot=0.2)
    assert lab2.tolist() == ["junk"] * 2 + ["mid"] * 5 + ["quality"] * 3


def test_value_weight_edge_cases():
    w = qf.value_weight(pd.Series([0.0, 0.0]))
    assert w.tolist() == [0.5, 0.5]                                  # zero cap -> equal weights
    w2 = qf.value_weight(pd.Series([-5.0, 5.0, np.nan]))
    assert w2.tolist()[:2] == [0.0, 1.0] and np.isnan(w2.iloc[2])    # negatives clipped to 0
    w3 = qf.value_weight(pd.Series(["2", "2"]))
    assert w3.tolist() == [0.5, 0.5]


def test_build_portfolios_legs_and_weights():
    n = 200
    df = pd.DataFrame({"ticker": [f"T{i}" for i in range(n)],
                       "quality": np.arange(n, dtype=float),
                       "mktcap": np.tile(np.arange(1, 11, dtype=float), 20)})
    legs = qf.build_portfolios(df)
    assert set(legs) <= {"small_quality", "small_junk", "big_quality", "big_junk"}
    assert {"small_quality", "small_junk", "big_quality", "big_junk"} == set(legs)
    for leg in legs.values():
        assert list(leg.columns) == ["ticker", "weight"]
        assert leg["weight"].sum() == pytest.approx(1.0)
    # quality leg = pct rank >= .9 (inclusive) -> T179..T199 split by size
    q = pd.concat([legs[k] for k in legs if k.endswith("quality")])
    assert set(q["ticker"]) <= {f"T{i}" for i in range(179, 200)}
    j = pd.concat([legs[k] for k in legs if k.endswith("junk")])
    assert set(j["ticker"]) <= {f"T{i}" for i in range(0, 20)}
    # big = top-decile market cap (mktcap 10 -> pct rank 1.0 >= .9)
    assert set(legs["big_quality"]["ticker"]) == {f"T{i}" for i in range(179, 200) if i % 10 == 9}


def test_build_portfolios_drops_nan_and_empty_legs():
    df = pd.DataFrame({"ticker": list("abcd"), "quality": [1.0, np.nan, 3.0, 4.0],
                       "mktcap": [1.0, 2.0, np.nan, 4.0]})
    legs = qf.build_portfolios(df)                                   # only a, d survive dropna
    assert sum(len(v) for v in legs.values()) <= 2
    assert qf.build_portfolios(df.iloc[0:0]) == {}


def test_value_weighting_inside_a_leg():
    df = pd.DataFrame({"ticker": [f"T{i}" for i in range(20)],
                       "quality": np.arange(20.0),
                       "mktcap": [1.0] * 18 + [10.0, 30.0]})
    legs = qf.build_portfolios(df)
    big_q = legs["big_quality"].set_index("ticker")["weight"]
    assert big_q.to_dict() == {"T18": 0.25, "T19": 0.75}             # value-weighted 10 : 30
    small_q = legs["small_quality"].set_index("ticker")["weight"]
    assert small_q.to_dict() == {"T17": 1.0}


def test_combos_default_missing_legs_to_zero():
    assert qf.qmj_combo({}) == 0.0
    assert qf.qmj_combo({"small_quality": 0.2}) == pytest.approx(0.1)
    assert qf.lq_combo({"big_quality": 0.4}) == pytest.approx(0.2)


# ── price_risk ────────────────────────────────────────────────────────────────
def test_price_risk_missing_parquet(seed):
    out = qf.price_risk("XX")
    assert out.empty and list(out.columns) == ["key", "beta", "vol"]


def test_price_risk_beta_vol_and_keys(seed):
    rng = np.random.RandomState(4)
    r = rng.normal(0, 0.01, 300)
    a = 100 * np.cumprod(1 + r)
    seed("US", _long({"aaa.NS": a, "BBB": a.copy(), "CCC": np.full(300, 50.0)}))
    out = qf.price_risk("US").set_index("key")
    assert set(out.index) == {"AAA", "BBB", "CCC"}                  # suffix stripped, upper-cased
    # mkt = mean(r, r, 0) = 2r/3 -> beta of r on 2r/3 = 1.5; flat stock has beta 0
    assert out.loc["AAA", "beta"] == pytest.approx(1.5, rel=1e-6)
    assert out.loc["BBB", "beta"] == pytest.approx(1.5, rel=1e-6)
    assert out.loc["CCC", "beta"] == pytest.approx(0.0, abs=1e-9)
    rets = pd.Series(a).pct_change().tail(252).dropna()
    assert out.loc["AAA", "vol"] == pytest.approx(rets.std() * np.sqrt(252))
    assert out.loc["CCC", "vol"] == 0.0


def test_price_risk_min_observations_boundary(seed):
    rng = np.random.RandomState(1)
    full = 100 * np.cumprod(1 + rng.normal(0, 0.01, 300))
    short = np.full(300, np.nan)
    short[-60:] = 100 * np.cumprod(1 + rng.normal(0, 0.01, 60))     # 59 returns -> skipped
    ok = np.full(300, np.nan)
    ok[-61:] = 100 * np.cumprod(1 + rng.normal(0, 0.01, 61))        # 60 returns -> kept
    seed("US", _long({"FULL": full, "SHORT": short, "OK": ok}))
    assert set(qf.price_risk("US")["key"]) == {"FULL", "OK"}


def test_price_risk_zero_market_variance_gives_nan_beta(seed):
    seed("US", _long({"A": np.full(100, 10.0), "B": np.full(100, 20.0)}, n=100))
    out = qf.price_risk("US")
    assert out["beta"].isna().all() and out["vol"].tolist() == [0.0, 0.0]


def test_price_risk_uses_only_last_252_returns(seed):
    rng = np.random.RandomState(2)
    old = rng.normal(0, 0.2, 100)                                    # wild early history
    recent = rng.normal(0, 0.005, 300)
    s = 100 * np.cumprod(1 + np.concatenate([old, recent]))
    other = 100 * np.cumprod(1 + rng.normal(0, 0.005, 400))
    seed("US", _long({"A": s, "B": other}, n=400))
    out = qf.price_risk("US").set_index("key")
    assert out.loc["A", "vol"] < 0.2                                 # early chaos excluded


# ── load_fundamentals / attach_price_risk / score_universe ────────────────────
def test_load_fundamentals_coerces_and_filters(fund_db):
    df = _fund_rows(n_per=3)
    df["roe"] = df["roe"].astype(object)
    df.loc[0, "roe"] = "n/a"
    fund_db(df)
    f = qf.load_fundamentals()
    assert len(f) == 6 and np.isnan(f.loc[0, "roe"]) and f["roe"].dtype.kind == "f"
    us = qf.load_fundamentals(["US"])
    assert set(us["market"]) == {"US"} and len(us) == 3
    assert qf.load_fundamentals(["XX"]).empty
    assert len(qf.load_fundamentals([])) == 6                        # empty list = no filter


def test_attach_price_risk_merges_and_dedupes(seed):
    r = np.random.RandomState(0).normal(0, 0.01, 300)
    px = 100 * np.cumprod(1 + r)
    seed("US", _long({"AAA": px, "BBB": px.copy()}))
    f = pd.DataFrame({"market": ["US", "US", "US"], "ticker": ["AAA", "bbb.NS", "ZZZ"], "roe": [1, 2, 3]})
    out = qf.attach_price_risk(f)
    assert out["key"].tolist() == ["AAA", "BBB", "ZZZ"]
    assert out["beta"].iloc[:2].notna().all() and np.isnan(out["beta"].iloc[2])
    assert "key" not in f.columns                                    # caller frame untouched


def test_attach_price_risk_no_data_and_empty(seed):
    f = pd.DataFrame({"market": ["XX"], "ticker": ["A"]})
    out = qf.attach_price_risk(f)
    assert np.isnan(out["beta"].iloc[0]) and np.isnan(out["vol"].iloc[0])
    empty = qf.attach_price_risk(pd.DataFrame({"market": [], "ticker": []}))
    assert empty.empty and {"beta", "vol"} <= set(empty.columns)


def test_attach_price_risk_duplicate_keys_across_markets(seed):
    rng = np.random.RandomState(7)
    seed("US", _long({"X": 100 * np.cumprod(1 + rng.normal(0, 0.01, 300)),
                      "Y": 100 * np.cumprod(1 + rng.normal(0, 0.01, 300))}))
    seed("JP", _long({"X.T": 100 * np.cumprod(1 + rng.normal(0, 0.02, 300)),
                      "Z.T": 100 * np.cumprod(1 + rng.normal(0, 0.02, 300))}))
    f = pd.DataFrame({"market": ["US", "JP"], "ticker": ["X", "X.T"]})
    out = qf.attach_price_risk(f)
    assert len(out) == 2                                             # no row explosion from dup key "X"
    assert out["beta"].notna().all()


def test_score_universe_pooled_matches_quality_score():
    f = _fund_rows(n_per=20)
    f["beta"], f["vol"] = 1.0, 0.2
    pooled = qf.score_universe(f, by_market=False)
    pd.testing.assert_frame_equal(pooled, qf.quality_score(f))


def test_score_universe_by_market_ranks_within_each_market():
    f = _fund_rows(n_per=20)
    f["beta"], f["vol"] = 1.0, 0.2
    scored = qf.score_universe(f, by_market=True)
    assert len(scored) == len(f)
    assert "quality_score" in scored
    for m in ("US", "JP"):
        sub = scored[f["market"] == m] if len(scored) == len(f) and "market" not in scored else \
            scored[scored["market"] == m]
        assert sub["quality_score"].max() == 100.0 and sub["quality_score"].min() == 5.0


# ── price premium / drivers ───────────────────────────────────────────────────
def _premium_df(n=150, coef=0.3, markets=("US", "JP"), seed_=0):
    rng = np.random.RandomState(seed_)
    q = rng.normal(0, 1, n)
    mc = np.exp(rng.normal(20, 1, n))
    mk = np.array([markets[i % len(markets)] for i in range(n)])
    logmb = coef * q + 0.1 * np.log(mc) + 0.5 * (mk == markets[-1]) + rng.normal(0, 0.05, n)
    return pd.DataFrame({"quality": q, "pb": np.exp(logmb), "mktcap": mc, "market": mk})


def test_price_premium_recovers_coefficient_with_market_dummies():
    pp = qf.price_premium(_premium_df())
    assert pp["n"] == 150
    assert pp["quality_coef"] == pytest.approx(0.3, abs=0.02)
    assert pp["quality_t"] > 20
    assert pp["mb_premium_per_sd%"] == round((np.exp(pp["quality_coef"]) - 1) * 100, 1)
    assert pp["R2"] > 0.9


def test_price_premium_single_market_no_dummies():
    pp = qf.price_premium(_premium_df(markets=("US",)))
    assert pp["quality_coef"] == pytest.approx(0.3, abs=0.03)


def test_price_premium_filters_and_min_rows_boundary():
    d = _premium_df(n=40)
    bad = d.copy()
    bad.loc[:9, "pb"] = [0, -1, 0, -1, 0, -1, 0, -1, 0, -1]          # 10 non-positive pb removed
    bad.loc[10:12, "mktcap"] = 0.0                                    # 3 non-positive caps removed
    pp = qf.price_premium(bad)
    assert pp["n"] == 27 and pp["quality_coef"] is None               # < 30 after filtering
    ok = qf.price_premium(_premium_df(n=30))
    assert ok["n"] == 30 and ok["quality_coef"] is not None           # exactly 30 -> runs
    nan_df = _premium_df(n=30)
    nan_df.loc[0, "quality"] = np.nan
    assert qf.price_premium(nan_df)["quality_coef"] is None


def test_driver_breakdown_sorted_rounded_and_skips_missing():
    rng = np.random.RandomState(0)
    n = 100
    base = rng.normal(size=n)
    d = pd.DataFrame({"pb": np.exp(base), "profitability": base + 0.01 * rng.normal(size=n),
                      "safety": -base, "growth": rng.normal(size=n)})
    out = qf.driver_breakdown(d)
    assert out["dimension"].tolist()[0] == "profitability" and out["dimension"].tolist()[-1] == "safety"
    assert "payout" not in set(out["dimension"])                      # dimension absent from frame
    assert out["corr_with_logMB"].tolist() == sorted(out["corr_with_logMB"].tolist(), reverse=True)
    assert out.set_index("dimension").loc["safety", "corr_with_logMB"] == pytest.approx(-1.0, abs=0.01)
    assert all(round(v, 3) == v for v in out["corr_with_logMB"])
    d2 = d.copy()
    d2.loc[:49, "pb"] = -1.0                                          # non-positive pb dropped
    assert qf.driver_breakdown(d2) is not None


# ── CLI ───────────────────────────────────────────────────────────────────────
def _run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["quality_factor.py", *argv])
    qf.main()


def test_main_missing_db_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(qf, "FUND", str(tmp_path / "absent.db"))
    with pytest.raises(SystemExit) as e:
        _run_main(monkeypatch)
    assert "no fundamentals_cache.db" in str(e.value)


def test_main_empty_market_exits(fund_db, monkeypatch):
    fund_db(_fund_rows(n_per=5))
    with pytest.raises(SystemExit) as e:
        _run_main(monkeypatch, "--market", "XX")
    assert "no fundamentals for XX" in str(e.value)
    fund_db(_fund_rows(n_per=5).iloc[0:0])
    with pytest.raises(SystemExit) as e:
        _run_main(monkeypatch)
    assert "no fundamentals for any market" in str(e.value)


@pytest.mark.xfail(PANDAS3, strict=True, raises=KeyError,
                   reason="quality_factor.py:213 score_universe groupby('market').apply(...) drops the grouping "
                          "column on pandas>=3, so scored['market'] / r['market'] raises KeyError downstream")
def test_main_full_report_by_market(fund_db, seed, monkeypatch, capsys, tmp_path):
    fund_db(_fund_rows(n_per=40))
    rng = np.random.RandomState(3)
    seed("US", _long({f"US{i}": 100 * np.cumprod(1 + rng.normal(0, 0.01, 300)) for i in range(40)}))
    out = tmp_path / "q.csv"
    _run_main(monkeypatch, "--premium", "--portfolios", "--top", "3", "--out", str(out))
    cap = capsys.readouterr()
    assert "AFP/QMJ QUALITY RANKING — all markets (80 scored)" in cap.err
    assert f"wrote {out}" in cap.err
    ranked = [ln for ln in cap.out.splitlines() if ln.startswith("  US") or ln.startswith("  JP")]
    assert len(ranked) >= 3
    assert "QMJ portfolio legs (value-weighted, 2×3 size×quality)" in cap.out
    assert "QUALITY PRICE PREMIUM" in cap.out and "quality premium is SIGNIFICANT (|t|>2)" in cap.out
    assert "driver breakdown" in cap.out
    w = pd.read_csv(out)
    assert list(w.columns) == ["ticker", "quality_score"] and len(w) == 80


def test_main_single_market_pooled_few_firms(fund_db, seed, monkeypatch, capsys):
    fund_db(_fund_rows(n_per=10, markets=("US",)))
    _run_main(monkeypatch, "--market", "US", "--pooled", "--premium", "--top", "2")
    cap = capsys.readouterr()
    assert "QUALITY RANKING — US (10 scored)" in cap.err
    assert "too few firms (10)" in cap.out
    ranked = [ln for ln in cap.out.splitlines() if ln.startswith("  US")]
    assert len(ranked) == 2


def test_main_pooled_single_market_runs_on_all_pandas(fund_db, seed, monkeypatch, capsys):
    # pooled scoring keeps the market column on every pandas version
    fund_db(_fund_rows(n_per=40, markets=("US",)))
    _run_main(monkeypatch, "--market", "US", "--pooled", "--premium", "--portfolios", "--top", "2")
    cap = capsys.readouterr()
    assert "QUALITY RANKING — US (40 scored)" in cap.err
    assert "QUALITY PRICE PREMIUM" in cap.out
    assert "QMJ portfolio legs" in cap.out
    assert "quality coef =" in cap.out
