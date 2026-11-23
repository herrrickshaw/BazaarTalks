"""Offline tests for watchlists.py (filter/merge core, builders with faked factor modules, CLI)."""
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import watchlists as wl  # noqa: E402


def _scored(**cols):
    base = {"ticker": ["A", "B", "C", "D", "E"], "market": ["US"] * 5,
            "quality_score": [90.0, 60.0, 59.99, np.nan, 75.0],
            "roe": [20.0, 15.0, 30.0, 30.0, 16.0],
            "de": [0.5, 0.5, 0.5, 0.5, 1.0],
            "sector": list("vwxyz")}
    base.update(cols)
    return pd.DataFrame(base)


def test_strong_from_scores_floor_is_inclusive_and_nan_dropped():
    out = wl.strong_from_scores(_scored())
    assert out["ticker"].tolist() == ["A", "E", "B"]                 # 60.0 kept, 59.99 and NaN dropped
    assert out["quality_score"].tolist() == [90.0, 75.0, 60.0]


def test_strong_profile_uses_strict_thresholds():
    out = wl.strong_from_scores(_scored()).set_index("ticker")
    assert bool(out.loc["A", "strong_profile"]) is True              # roe 20 > 15, de 0.5 < 1
    assert bool(out.loc["B", "strong_profile"]) is False             # roe == 15 is NOT > 15
    assert bool(out.loc["E", "strong_profile"]) is False             # de == 1 is NOT < 1


def test_strong_profile_false_when_roe_or_de_nan_or_text():
    d = _scored(roe=[np.nan, "n/a", 20.0, 20.0, 20.0], de=[0.5, 0.5, np.nan, 0.5, "x"])
    out = wl.strong_from_scores(d, min_quality=0).set_index("ticker")
    assert bool(out.loc["A", "strong_profile"]) is False
    assert bool(out.loc["B", "strong_profile"]) is False
    assert bool(out.loc["C", "strong_profile"]) is False
    assert bool(out.loc["E", "strong_profile"]) is False


def test_strong_from_scores_adds_clean_key_and_does_not_mutate():
    d = _scored(ticker=["a.ns", "7203.T", "B", "C", "E"])
    before = d.copy()
    out = wl.strong_from_scores(d)
    assert out.set_index("ticker")["key"].to_dict() == {"a.ns": "A", "E": "E", "7203.T": "7203"}
    pd.testing.assert_frame_equal(d, before)


def test_strong_from_scores_custom_floor_and_empty():
    assert wl.strong_from_scores(_scored(), min_quality=80)["ticker"].tolist() == ["A"]
    assert wl.strong_from_scores(_scored(), min_quality=99).empty


@pytest.mark.xfail(strict=True, raises=AttributeError,
                   reason="watchlists.py:60-62 strong_from_scores crashes when the scored frame has "
                          "no roe/de column (d.get(...) is None -> comparison yields a scalar, .fillna fails)")
def test_strong_from_scores_tolerates_missing_roe_de_columns():
    d = pd.DataFrame({"ticker": ["A"], "market": ["US"], "quality_score": [70.0]})
    out = wl.strong_from_scores(d)
    assert out["strong_profile"].tolist() == [False]


def _scan():
    return pd.DataFrame({
        "ticker": ["P", "Q", "R", "S", "T"],
        "state": ["in_box", "breakout", "in_box", "in_box", "in_box"],
        "accumulation": [3.0, 9.0, 0.49, 0.5, 1.5],
        "position": [0.1, 0.2, 0.3, 0.4, 0.5], "cmf": [0.1] * 5})


def test_accumulated_from_scan_filters_and_ranks():
    out = wl.accumulated_from_scan(_scan())
    assert out["ticker"].tolist() == ["P", "T", "S"]                 # 0.5 inclusive, 0.49 out, breakout out
    assert out["key"].tolist() == ["P", "T", "S"]


def test_accumulated_from_scan_custom_floor_and_empty():
    assert wl.accumulated_from_scan(_scan(), min_accum=2.0)["ticker"].tolist() == ["P"]
    assert wl.accumulated_from_scan(_scan(), min_accum=10).empty


def test_default_thresholds():
    assert wl.MIN_QUALITY == 60.0 and wl.MIN_ACCUM == 0.5


def _strong_frame():
    return wl.strong_from_scores(pd.DataFrame({
        "ticker": ["AAA.NS", "BBB", "CCC"], "market": ["IN", "US", "US"],
        "quality_score": [80.0, 95.0, 70.0], "roe": [20] * 3, "de": [0.5] * 3, "sector": ["s"] * 3}))


def _accum_frame():
    return wl.accumulated_from_scan(pd.DataFrame({
        "ticker": ["AAA", "BBB", "ZZZ"], "state": ["in_box"] * 3, "accumulation": [1.0, 2.0, 3.0],
        "position": [0.5, 0.6, 0.7], "cmf": [0.1, 0.2, 0.3]}))


def test_intersect_joins_on_clean_key_and_orders_by_quality_then_accumulation():
    both = wl.intersect(_strong_frame(), _accum_frame())
    assert both["key"].tolist() == ["BBB", "AAA"]                    # quality 95 then 80; CCC/ZZZ unmatched
    assert both["ticker_f"].tolist() == ["BBB", "AAA.NS"]            # original strong tickers kept
    assert both["ticker_a"].tolist() == ["BBB", "AAA"]
    assert list(both.columns) == ["key", "ticker_f", "market", "quality_score",
                                  "ticker_a", "accumulation", "position", "cmf"]


def test_intersect_empty_inputs_and_no_overlap():
    assert wl.intersect(pd.DataFrame(), _accum_frame()).empty
    assert wl.intersect(_strong_frame(), pd.DataFrame()).empty
    only_z = wl.accumulated_from_scan(pd.DataFrame({
        "ticker": ["ZZZ"], "state": ["in_box"], "accumulation": [1.0], "position": [0.1], "cmf": [0.1]}))
    res = wl.intersect(_strong_frame(), only_z)
    assert res.empty and "key" in res.columns


# ── builders (faked quality_factor / darvas_volume) ──────────────────────────
def test_build_strong_pipeline(monkeypatch):
    qf = types.ModuleType("quality_factor")
    log = {}
    raw = pd.DataFrame({"ticker": ["X"], "market": ["US"]})
    qf.load_fundamentals = lambda m: (log.setdefault("mk", m), raw)[1]
    qf.attach_price_risk = lambda f: (log.setdefault("attach", f), f)[1]

    def score(f, by_market):
        log["by_market"] = by_market
        return pd.DataFrame({"ticker": ["X", "Y"], "market": ["US", "US"],
                             "quality_score": [65.0, 10.0], "roe": [20, 20], "de": [0.1, 0.1]})
    qf.score_universe = score
    monkeypatch.setitem(sys.modules, "quality_factor", qf)
    out = wl.build_strong(["US", "IN"])
    assert log["mk"] == ["US", "IN"] and log["by_market"] is True and log["attach"] is raw
    assert out["ticker"].tolist() == ["X"]


def test_build_strong_empty_fundamentals(monkeypatch):
    qf = types.ModuleType("quality_factor")
    qf.load_fundamentals = lambda m: pd.DataFrame()
    qf.attach_price_risk = lambda f: pytest.fail("must not run on empty")
    monkeypatch.setitem(sys.modules, "quality_factor", qf)
    assert wl.build_strong(["US"]).empty


def test_build_accumulated_concats_nonempty_scans(monkeypatch):
    dv = types.ModuleType("darvas_volume")
    scans = {"US": pd.DataFrame({"ticker": ["U1"], "state": ["in_box"], "accumulation": [1.0]}),
             "IN": pd.DataFrame(),
             "JP": pd.DataFrame({"ticker": ["J1"], "state": ["in_box"], "accumulation": [2.0]})}
    dv.scan_market = lambda m: scans[m]
    monkeypatch.setitem(sys.modules, "darvas_volume", dv)
    out = wl.build_accumulated(["US", "IN", "JP"])
    assert out["ticker"].tolist() == ["J1", "U1"]                    # ranked by accumulation
    assert out.index.tolist() in ([0, 1], [1, 0])


def test_build_accumulated_all_empty(monkeypatch):
    dv = types.ModuleType("darvas_volume")
    dv.scan_market = lambda m: pd.DataFrame()
    monkeypatch.setitem(sys.modules, "darvas_volume", dv)
    assert wl.build_accumulated(["US"]).empty


# ── _markets / main ──────────────────────────────────────────────────────────
def _args(**kw):
    d = dict(all=False, market=None)
    d.update(kw)
    return types.SimpleNamespace(**d)


def test_markets_selection(monkeypatch):
    monkeypatch.setattr(wl.marketdata, "market_list", lambda: ["IN", "US"])
    assert wl._markets(_args(market="US")) == ["US"]
    assert wl._markets(_args(market="US", all=True)) == ["IN", "US"]
    assert wl._markets(_args()) == ["IN", "US"]


def _patch_builders(monkeypatch, strong, accum):
    seen = {}
    monkeypatch.setattr(wl, "build_strong", lambda m: seen.setdefault("s", m) and strong)
    monkeypatch.setattr(wl, "build_accumulated", lambda m: seen.setdefault("a", m) and accum)
    return seen


def _full_accum():
    d = _accum_frame().copy()
    d["market"] = "US"
    d["close"] = d["ticker"].map({"AAA": 10.0, "BBB": 20.0, "ZZZ": 30.0})
    d["obv_trend"] = d["ticker"].map({"AAA": 0.1, "BBB": 0.2, "ZZZ": 0.3})
    d["ud_vol_ratio"] = d["ticker"].map({"AAA": 1.5, "BBB": np.inf, "ZZZ": 2.0})
    return d


def test_main_prints_all_three_lists_and_writes_csv(tmp_path, monkeypatch, capsys):
    seen = _patch_builders(monkeypatch, _strong_frame(), _full_accum())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["watchlists.py", "--market", "US", "--top", "2", "--both",
                                      "--csv", "wl"])
    wl.main()
    cap = capsys.readouterr()
    out = cap.out
    assert seen["s"] == ["US"]
    assert "FUNDAMENTALLY STRONG  (3 names)" in out
    assert "BEING ACCUMULATED  (3 names)" in out
    assert "BOTH — strong AND accumulated  (2 names)" in out
    rows = [l.split() for l in out.splitlines()]
    bbb = next(r for r in rows if r[:2] == ["US", "BBB"])
    assert bbb[2] == "95.0" and "★" in bbb                            # strong profile star
    zzz = next(r for r in rows if r[:2] == ["US", "ZZZ"])
    assert zzz[2:] == ["30.00", "0.70", "0.30", "0.30", "2.00", "3.00"]
    assert any(r[:2] == ["US", "BBB"] and "inf" in r for r in rows)   # infinite up/down ratio printed as inf
    assert not any(r[:2] == ["US", "CCC"] for r in rows)               # --top 2 hides the 3rd strong row
    assert "wrote wl_strong.csv / wl_accum.csv" in cap.err
    assert len(pd.read_csv(tmp_path / "wl_strong.csv")) == 2
    assert len(pd.read_csv(tmp_path / "wl_accum.csv")) == 2


def test_main_empty_lists_and_no_overlap(tmp_path, monkeypatch, capsys):
    _patch_builders(monkeypatch, pd.DataFrame(), pd.DataFrame())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["watchlists.py", "--market", "US", "--both", "--csv", "e"])
    wl.main()
    cap = capsys.readouterr()
    assert "(no fundamentals for the selected market)" in cap.out
    assert "(no accumulation coils found)" in cap.out
    assert "(no overlap" in cap.out
    assert not (tmp_path / "e_strong.csv").exists() and not (tmp_path / "e_accum.csv").exists()


def test_main_default_all_markets_without_both(monkeypatch, capsys):
    monkeypatch.setattr(wl.marketdata, "market_list", lambda: ["IN", "US"])
    seen = _patch_builders(monkeypatch, pd.DataFrame(), pd.DataFrame())
    monkeypatch.setattr(sys, "argv", ["watchlists.py"])
    wl.main()
    assert seen["s"] == ["IN", "US"]
    assert "BOTH" not in capsys.readouterr().out
