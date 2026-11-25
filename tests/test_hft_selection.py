"""Offline tests for hft_selection.py (daily-OHLC microstructure proxies + archetype picks)."""
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hft_selection as hs  # noqa: E402


# ── daily_range_pct / avg_range / range_stability ────────────────────────────
def test_daily_range_pct_values_and_nonpositive_close():
    out = hs.daily_range_pct([11, 12, 5], [9, 10, 4], [10, 0, -1])
    assert out[0] == pytest.approx(0.2)
    assert np.isnan(out[1]) and np.isnan(out[2])           # close <= 0 -> NaN, no div-by-zero


def test_daily_range_pct_accepts_series_and_lists():
    out = hs.daily_range_pct(pd.Series([102.0]), [98.0], np.array([100.0]))
    assert out.tolist() == pytest.approx([0.04])


def test_avg_range_ignores_nan():
    assert hs.avg_range([11, 12], [9, 10], [10, 0]) == pytest.approx(0.2)


def test_range_stability_zero_for_constant_range_and_positive_otherwise():
    assert hs.range_stability([102, 204], [98, 196], [100, 200]) == pytest.approx(0.0)
    assert hs.range_stability([101, 110], [99, 90], [100, 100]) == pytest.approx(0.09)  # std of [0.02,0.20]


# ── corwin_schultz_spread ─────────────────────────────────────────────────────
def _cs_pair(h0, l0, h1, l1):
    k = 3 - 2 * np.sqrt(2)
    beta = np.log(h0 / l0) ** 2 + np.log(h1 / l1) ** 2
    gamma = np.log(max(h0, h1) / min(l0, l1)) ** 2
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    return max(2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha)), 0.0)


def test_corwin_schultz_matches_formula_for_one_pair():
    got = hs.corwin_schultz_spread([101.0, 102.0], [99.0, 100.0])
    assert got == pytest.approx(_cs_pair(101, 99, 102, 100))


def test_corwin_schultz_averages_consecutive_pairs():
    h = [101.0, 102.0, 103.5]; l = [99.0, 100.0, 100.5]
    exp = np.mean([_cs_pair(101, 99, 102, 100), _cs_pair(102, 100, 103.5, 100.5)])
    assert hs.corwin_schultz_spread(h, l) == pytest.approx(exp)


def test_corwin_schultz_negative_estimates_floored_at_zero():
    # disjoint, widely separated 2-day ranges -> gamma huge -> alpha<0 -> floored to 0.0
    got = hs.corwin_schultz_spread([101.0, 201.0], [99.0, 199.0])
    assert got == 0.0


def test_corwin_schultz_too_short_and_bad_lows():
    assert np.isnan(hs.corwin_schultz_spread([100.0], [99.0]))
    assert np.isnan(hs.corwin_schultz_spread([], []))
    # every pair has a non-positive low -> no spreads -> NaN
    assert np.isnan(hs.corwin_schultz_spread([10.0, 10.0], [0.0, 9.0]))
    # one bad pair is skipped, the good one still used
    h = [101.0, 102.0, 103.0]; l = [0.0, 100.0, 101.0]
    assert hs.corwin_schultz_spread(h, l) == pytest.approx(_cs_pair(102, 100, 103, 101))


# ── efficiency_ratio ──────────────────────────────────────────────────────────
def test_efficiency_ratio_trending_choppy_and_flat():
    assert hs.efficiency_ratio([1, 2, 3, 4]) == pytest.approx(1.0)
    assert hs.efficiency_ratio([4, 3, 2, 1]) == pytest.approx(1.0)          # abs net move
    assert hs.efficiency_ratio([1, 2, 1, 2, 1]) == pytest.approx(0.0)
    assert hs.efficiency_ratio([10, 12, 11, 13]) == pytest.approx(3 / 5)
    assert np.isnan(hs.efficiency_ratio([5, 5, 5]))                         # zero travel
    assert np.isnan(hs.efficiency_ratio([5]))


# ── lag1_autocorr ─────────────────────────────────────────────────────────────
def test_lag1_autocorr_sign_and_degenerate():
    assert hs.lag1_autocorr([1, -1, 1, -1, 1, -1]) == pytest.approx(-1.0)
    assert hs.lag1_autocorr([1, 2, 3, 4, 5, 6]) == pytest.approx(1.0)
    assert np.isnan(hs.lag1_autocorr([1, 2]))                               # < 3 points
    assert np.isnan(hs.lag1_autocorr([3, 3, 3, 3]))                         # zero variance


def test_lag1_autocorr_drops_nonfinite_before_pairing():
    assert hs.lag1_autocorr([1, np.nan, 2, np.inf, 3, 4, 5]) == pytest.approx(1.0)
    assert np.isnan(hs.lag1_autocorr([1, np.nan, 2]))                       # only 2 finite left


# ── ou_half_life ──────────────────────────────────────────────────────────────
def test_ou_half_life_exact_ar1():
    # p_{t+1} - 100 = 0.5 (p_t - 100)  => b = -0.5 => half-life = -ln2/ln(0.5) = 1 day
    p = [100 + 16 * 0.5 ** t for t in range(8)]
    assert hs.ou_half_life(p) == pytest.approx(1.0)
    # b = -0.1 -> -ln2/ln(0.9)
    q = [100 + 20 * 0.9 ** t for t in range(10)]
    assert hs.ou_half_life(q) == pytest.approx(-np.log(2) / np.log(0.9))


def test_ou_half_life_not_reverting_is_inf():
    assert hs.ou_half_life([1, 2, 3, 4, 5, 6]) == np.inf                     # trending: b >= 0
    assert hs.ou_half_life([100 * 1.02 ** t for t in range(10)]) == np.inf   # explosive


def test_ou_half_life_overshooting_b_le_minus_one_is_inf():
    p = [110.0, 90.0, 110.0, 90.0, 110.0]       # dP = -2*(P-100): b = -2 -> (1+b) <= 0
    assert hs.ou_half_life(p) == np.inf


def test_ou_half_life_short_or_nan_input():
    assert np.isnan(hs.ou_half_life([1, 2, 3]))
    assert np.isnan(hs.ou_half_life([1, np.nan, np.nan, 2]))                # 2 finite points left
    hl = hs.ou_half_life([100 + 16 * 0.5 ** t if t != 3 else np.nan for t in range(9)])
    assert np.isfinite(hl)                                                   # NaNs dropped first


# ── archetype_scores ──────────────────────────────────────────────────────────
def _feat(with_peer=True):
    f = pd.DataFrame({
        "ticker": list("ABCD"),
        "avg_range%": [1.0, 2.0, 3.0, 4.0], "range_stability": [0.1, 0.2, 0.3, 0.4],
        "eff_ratio": [0.1, 0.4, 0.2, 0.9], "ret_autocorr": [-0.5, 0.0, 0.3, 0.6],
        "vol_autocorr": [0.1, 0.3, 0.2, 0.9], "half_life": [2.0, np.inf, 10.0, 500.0]})
    if with_peer:
        f["peer_corr"] = [0.9, 0.5, np.nan, 0.1]
        f["peer_dev"] = [-0.05, 0.01, 0.0, 0.2]
    return f


def _z(s):
    s = pd.Series(s, dtype=float)
    return (s - s.mean()) / s.std(ddof=0)


def test_archetype_scores_formulas():
    f = _feat()
    out = hs.archetype_scores(f)
    hl = pd.Series([2.0, np.nan, 10.0, 60.0])                                # inf->NaN, clip to 60
    exp_mm = (-_z(f["avg_range%"]) - _z(f["range_stability"]) - _z(f["eff_ratio"])).round(3)
    exp_sa = (-_z(f["ret_autocorr"]) - _z(f["eff_ratio"]) - _z(hl)).round(3)
    exp_lat = (_z(f["eff_ratio"]) + _z(f["vol_autocorr"]) + _z(f["ret_autocorr"])).round(3)
    assert out["market_making"].tolist() == pytest.approx(exp_mm.tolist())
    assert out["stat_arb"].tolist() == pytest.approx(exp_sa.tolist(), nan_ok=True)
    assert out["latency"].tolist() == pytest.approx(exp_lat.tolist())
    assert np.isnan(out.loc[1, "stat_arb"])                                  # inf half-life -> NaN score
    assert not np.isnan(out.loc[3, "stat_arb"])                              # clipped (500 -> 60), not NaN


def test_archetype_scores_market_making_prefers_tight_stable_low_toxicity():
    out = hs.archetype_scores(_feat())
    assert out["market_making"].idxmax() == 0                                # A: lowest range/instability/toxicity
    assert out["latency"].idxmax() == 3                                      # D: most persistent


def test_archetype_scores_etf_arb_with_peer_columns():
    f = _feat()
    out = hs.archetype_scores(f)
    exp = (_z(f["peer_corr"]) + _z(f["peer_dev"].abs())).round(3)
    assert out["etf_arb"].tolist() == pytest.approx(exp.tolist(), nan_ok=True)
    assert np.isnan(out.loc[2, "etf_arb"])


def test_archetype_scores_etf_arb_nan_without_or_all_nan_peer():
    assert hs.archetype_scores(_feat(with_peer=False))["etf_arb"].isna().all()
    f = _feat(); f["peer_corr"] = np.nan
    assert hs.archetype_scores(f)["etf_arb"].isna().all()


def test_archetype_scores_does_not_mutate_input():
    f = _feat()
    cols = list(f.columns)
    hs.archetype_scores(f)
    assert list(f.columns) == cols
    assert f["half_life"].iloc[1] == np.inf


def test_archetype_scores_degenerate_cross_section_scores_zero():
    f = _feat(with_peer=False)
    for c in ["avg_range%", "range_stability", "eff_ratio", "ret_autocorr", "vol_autocorr"]:
        f[c] = 1.0
    out = hs.archetype_scores(f)
    assert (out["market_making"] == 0).all() and (out["latency"] == 0).all()


def test_archetypes_constant():
    assert hs.ARCHETYPES == ["market_making", "stat_arb", "latency", "etf_arb"]
    assert (hs.WINDOW, hs.STAT_WINDOW, hs.TRADEABLE_MAX_RANGE) == (5, 20, 0.12)


# ── _sector_map ───────────────────────────────────────────────────────────────
def test_sector_map_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(hs, "COMPANIES", str(tmp_path / "none.parquet"))
    assert hs._sector_map() == {}


def test_sector_map_normalises_keys(tmp_path, monkeypatch):
    p = tmp_path / "c.parquet"; p.write_bytes(b"x")
    monkeypatch.setattr(hs, "COMPANIES", str(p))
    comp = pd.DataFrame({"ticker": ["abc.NS", "7203.T", "XYZ"], "sector": ["Tech", "Auto", "Fin"],
                         "other": [1, 2, 3]})
    monkeypatch.setattr(hs.pd, "read_parquet", lambda path, *a, **k: comp.copy())
    assert hs._sector_map() == {"ABC": "Tech", "7203": "Auto", "XYZ": "Fin"}


# ── scan_market ───────────────────────────────────────────────────────────────
N = 60


def _wide_data():
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2024-01-02", periods=N)
    common = rng.normal(0, 0.01, N)
    close = {}
    for s in ["T1", "T2", "T3", "U1", "U2"]:
        close[s] = 100 * np.cumprod(1 + common * (0.5 + rng.random()) + rng.normal(0, 0.005, N))
    close["WILD"] = 100 * np.cumprod(1 + rng.normal(0, 0.01, N))
    short = np.full(N, np.nan); short[-10:] = 50.0 + np.arange(10)
    close["SHORT"] = short
    close = pd.DataFrame(close, index=idx)
    vol = pd.DataFrame(rng.integers(1000, 5000, close.shape).astype(float), index=idx, columns=close.columns)
    high = close * 1.01
    low = close * 0.99
    high["WILD"] = close["WILD"] * 1.3
    low["WILD"] = close["WILD"] * 0.7
    return close, vol, high, low


def _long(close, high, low):
    rows = []
    for s in close.columns:
        for d in close.index:
            rows.append({"Date": d, "Symbol": s, "Close": close.at[d, s],
                         "High": high.at[d, s], "Low": low.at[d, s]})
    return pd.DataFrame(rows)


@pytest.fixture
def scan_env(tmp_path, monkeypatch):
    close, vol, high, low = _wide_data()
    state = {"wide": (close, vol)}
    lf = types.ModuleType("liquidity_factor")
    lf._market_wide = lambda m: state["wide"]
    pfm = types.ModuleType("pead_factor")
    pfm._liquid_symbols = lambda c, v: list(c.columns)
    monkeypatch.setitem(sys.modules, "liquidity_factor", lf)
    monkeypatch.setitem(sys.modules, "pead_factor", pfm)
    comp_path = tmp_path / "companies_industry.parquet"; comp_path.write_bytes(b"x")
    monkeypatch.setattr(hs, "COMPANIES", str(comp_path))
    monkeypatch.setattr(hs, "SEED", str(tmp_path))
    comp = pd.DataFrame({"ticker": ["T1.NS", "T2", "T3"], "sector": ["Tech"] * 3})
    long_px = _long(close, high, low)

    def fake_read(path, *a, **k):
        return comp.copy() if str(path) == str(comp_path) else long_px.copy()

    monkeypatch.setattr(hs.pd, "read_parquet", fake_read)
    return types.SimpleNamespace(close=close, vol=vol, high=high, low=low, state=state)


def test_scan_market_no_data_returns_empty(scan_env):
    scan_env.state["wide"] = (None, None)
    assert hs.scan_market("US").empty


def test_scan_market_filters_wild_and_short_histories(scan_env):
    df = hs.scan_market("US")
    assert set(df["ticker"]) == {"T1", "T2", "T3", "U1", "U2"}.intersection(set(df["ticker"]))
    assert "WILD" not in set(df["ticker"])         # 60% avg range > TRADEABLE_MAX_RANGE
    assert "SHORT" not in set(df["ticker"])        # fewer than STAT_WINDOW+2 prices
    assert set(df["ticker"]) == {"T1", "T2", "T3", "U1", "U2"}
    assert (df["market"] == "US").all()


def test_scan_market_feature_values(scan_env):
    df = hs.scan_market("US").set_index("ticker")
    assert df["avg_range%"].tolist() == pytest.approx([2.0] * 5, abs=0.01)   # (1.01-0.99)/1 = 2% of close
    assert df["range_stability"].abs().max() < 1e-9
    c = scan_env.close["T1"].tail(5)
    assert df.loc["T1", "eff_ratio"] == pytest.approx(hs.efficiency_ratio(c.values))
    rets = scan_env.close.pct_change(fill_method=None)
    assert df.loc["T1", "ret_autocorr"] == pytest.approx(hs.lag1_autocorr(rets["T1"].tail(20).values))
    assert df.loc["T1", "vol_autocorr"] == pytest.approx(hs.lag1_autocorr(scan_env.vol["T1"].tail(20).values))
    assert df.loc["T1", "half_life"] == pytest.approx(hs.ou_half_life(scan_env.close["T1"].tail(20).values)) \
        or (np.isinf(df.loc["T1", "half_life"]))
    assert df.loc["T1", "cs_spread"] == pytest.approx(
        hs.corwin_schultz_spread(scan_env.high["T1"].tail(5), scan_env.low["T1"].tail(5)))


def test_scan_market_peer_features_use_sector_basket(scan_env):
    df = hs.scan_market("US").set_index("ticker")
    assert {"peer_corr", "peer_dev"} <= set(df.columns)
    rets = scan_env.close.pct_change(fill_method=None)
    basket = rets[["T1", "T2", "T3"]].mean(axis=1)               # Tech sector (3 members) basket
    rr = rets["T1"].tail(20)
    assert df.loc["T1", "peer_corr"] == pytest.approx(rr.corr(basket.reindex(rr.index)))
    cw = scan_env.close["T1"].tail(5)
    exp_dev = (cw.iloc[-1] / cw.iloc[0] - 1) - (1 + basket.reindex(rr.index).tail(5)).prod() + 1
    assert df.loc["T1", "peer_dev"] == pytest.approx(exp_dev)
    # U1/U2 are "Unknown" sector: 4 members in Unknown (U1,U2,WILD,SHORT) -> also have a basket
    assert df.loc[["U1", "U2"], "peer_corr"].notna().all()


def test_scan_market_small_sector_has_no_peer_features(scan_env, monkeypatch):
    comp = pd.DataFrame({"ticker": ["T1", "T2"], "sector": ["Tech", "Tech"]})
    real = hs.pd.read_parquet
    monkeypatch.setattr(hs.pd, "read_parquet",
                        lambda p, *a, **k: comp.copy() if str(p) == hs.COMPANIES else real(p))
    df = hs.scan_market("US").set_index("ticker")
    assert df.loc[["T1", "T2"], "peer_corr"].isna().all()         # only 2 Tech members (<3)
    assert df.loc["T3", "peer_corr"] == df.loc["T3", "peer_corr"]  # T3 now 'Unknown' with U1,U2,T3,...


def test_scan_market_window_param_changes_features(scan_env):
    a = hs.scan_market("US", window=5).set_index("ticker")
    b = hs.scan_market("US", window=10).set_index("ticker")
    assert a.loc["T1", "eff_ratio"] != b.loc["T1", "eff_ratio"]
    c = hs.scan_market("US", stat_window=70)                      # needs 72 prices: nobody has them
    assert c.empty


# ── pick ──────────────────────────────────────────────────────────────────────
def test_pick_sorts_drops_nan_and_limits():
    scan = pd.DataFrame({"market": ["US"] * 4, "ticker": list("ABCD"), "avg_range%": 1.0,
                         "cs_spread": 0.01, "range_stability": 0.0, "eff_ratio": 0.5,
                         "ret_autocorr": 0.0, "vol_autocorr": 0.0, "half_life": 3.0,
                         "market_making": [1.0, np.nan, 3.0, 2.0], "junk": 1})
    out = hs.pick(scan, "market_making", 2)
    assert out["ticker"].tolist() == ["C", "D"]
    assert "junk" not in out.columns and out.columns[-1] == "market_making"
    assert len(hs.pick(scan, "market_making", 10)) == 3
    assert hs.pick(scan, "market_making", 0).empty


def test_pick_missing_optional_columns_tolerated():
    scan = pd.DataFrame({"market": ["US"], "ticker": ["A"], "stat_arb": [1.0]})
    assert hs.pick(scan, "stat_arb", 5).columns.tolist() == ["market", "ticker", "stat_arb"]


# ── CLI ───────────────────────────────────────────────────────────────────────
def _fake_scan(market, window=hs.WINDOW):
    f = _feat()
    f["market"] = market
    f["ticker"] = [f"{market}{t}" for t in f["ticker"]]
    f["cs_spread"] = [0.01, 0.0, np.nan, 0.02]
    return f


def test_main_prints_all_archetypes(monkeypatch, capsys):
    monkeypatch.setattr(hs, "scan_market", _fake_scan)
    monkeypatch.setattr(sys, "argv", ["hft_selection.py", "--market", "US", "--top", "2"])
    hs.main()
    out = capsys.readouterr().out
    assert "HFT-ARCHETYPE STOCK PICKS — US, 5-day (1-week) window, 4 liquid names" in out
    for lab in ("MARKET MAKING (stable earnable spread, low toxicity)",
                "STATISTICAL ARB (fast mean reversion)",
                "LATENCY / ORDER-ANTICIPATION (predictable, persistent flow)",
                "INDEX/ETF ARB (mispriced leg of a peer relationship)"):
        assert f"-- {lab} --" in out
    assert "Tier-1 (universe/tradeability) proxies" in out
    sec = out.split("-- MARKET MAKING")[1].split("-- STATISTICAL")[0]
    rows = [ln for ln in sec.splitlines() if ln.startswith("  US")]
    assert len(rows) == 2 and rows[0].split()[1] == "USA"       # best market-maker first, --top 2


def test_main_no_eligible_names_message(monkeypatch, capsys):
    monkeypatch.setattr(hs, "scan_market", lambda m, w=5: _fake_scan(m).drop(columns=["peer_corr", "peer_dev"]))
    monkeypatch.setattr(sys, "argv", ["hft_selection.py", "--market", "US"])
    hs.main()
    out = capsys.readouterr().out
    assert "-- INDEX/ETF ARB (mispriced leg of a peer relationship): no eligible names --" in out


def test_main_all_markets_tag_and_concat(monkeypatch, capsys):
    monkeypatch.setattr(hs.marketdata, "market_list", lambda: ["US", "JP"])
    monkeypatch.setattr(hs, "scan_market", _fake_scan)
    monkeypatch.setattr(sys, "argv", ["hft_selection.py", "--all", "--archetype", "stat_arb"])
    hs.main()
    out = capsys.readouterr().out
    assert "PICKS — US, JP, 5-day" in out and "8 liquid names" in out
    monkeypatch.setattr(hs.marketdata, "market_list", lambda: ["A", "B", "C", "D"])
    hs.main()
    assert "PICKS — 4 markets," in capsys.readouterr().out


def test_main_no_market_defaults_to_all(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(hs.marketdata, "market_list", lambda: ["US", "JP"])
    monkeypatch.setattr(hs, "scan_market", lambda m, w=5: (seen.append(m), _fake_scan(m))[1])
    monkeypatch.setattr(sys, "argv", ["hft_selection.py"])
    hs.main()
    assert seen == ["US", "JP"]


def test_main_all_empty_scans_exits(monkeypatch):
    monkeypatch.setattr(hs.marketdata, "market_list", lambda: ["US"])
    monkeypatch.setattr(hs, "scan_market", lambda m, w=5: pd.DataFrame())
    monkeypatch.setattr(sys, "argv", ["hft_selection.py"])
    with pytest.raises(SystemExit) as ei:
        hs.main()
    assert "no data" in str(ei.value.code)
