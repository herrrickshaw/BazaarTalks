"""Offline tests for darvas_volume.py (Darvas box + volume-acquisition monitor)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import darvas_volume as dv  # noqa: E402
import marketdata  # noqa: E402


# ── primitives ────────────────────────────────────────────────────────────────
def test_obv_signs_and_flat_days():
    close = [10, 11, 11, 10, 12]
    vol = [100, 200, 300, 400, 500]
    assert dv.obv(close, vol).tolist() == [0, 200, 200, -200, 300]    # first bar and flat bar add 0
    assert dv.obv([5], [10]).tolist() == [0]


def test_chaikin_ad_multiplier_and_zero_range():
    h = [12, 12, 10]
    l = [10, 10, 10]
    c = [12, 10, 10]                                  # mfm: +1, -1, zero-range -> 0
    v = [100, 50, 999]
    assert dv.chaikin_ad(h, l, c, v).tolist() == [100, 50, 50]
    c2 = [11, 11, 11]                                 # mid-range close -> multiplier 0
    assert dv.chaikin_ad([12] * 3, [10] * 3, c2, [1, 2, 3]).tolist() == [0, 0, 0]


def test_cmf_value_period_and_zero_volume():
    h = np.array([12.0, 12.0, 12.0, 12.0])
    l = np.array([10.0, 10.0, 10.0, 10.0])
    c = np.array([12.0, 12.0, 10.0, 10.0])           # +1,+1,-1,-1
    v = np.array([1.0, 1.0, 3.0, 3.0])
    assert dv.chaikin_money_flow(h, l, c, v) == pytest.approx((2 - 6) / 8)
    assert dv.chaikin_money_flow(h, l, c, v, period=2) == pytest.approx(-1.0)   # last two bars only
    assert dv.chaikin_money_flow(h, l, c, v, period=0) == pytest.approx(-0.5)   # 0 = full window
    assert np.isnan(dv.chaikin_money_flow(h, l, c, v * 0))
    assert dv.chaikin_money_flow([5, 5], [5, 5], [5, 5], [1, 1]) == 0.0         # flat bars -> 0


def test_up_down_volume_ratio():
    assert dv.up_down_volume_ratio([1, 2, 1, 2], [10, 30, 10, 10]) == pytest.approx(40 / 10)
    assert dv.up_down_volume_ratio([1, 2, 3], [1, 1, 1]) == np.inf              # no down volume
    assert np.isnan(dv.up_down_volume_ratio([1, 1, 1], [5, 5, 5]))              # no moves at all
    assert dv.up_down_volume_ratio([3, 2, 1], [1, 4, 6]) == 0.0                 # only down volume


def test_trend_corr_is_marketdata_reexport():
    assert dv.trend_corr is marketdata.trend_corr
    assert dv.trend_corr([1, 2, 3, 4]) == pytest.approx(1.0)
    assert np.isnan(dv.trend_corr([1, 1, 1]))


# ── darvas_box ────────────────────────────────────────────────────────────────
def test_darvas_box_top_held_and_bottom():
    h = [10, 11, 15, 14, 14, 13, 14, 99]              # last bar (99) is the current one -> excluded
    l = [9, 10, 12, 11, 12, 11.5, 12, 98]
    b = dv.darvas_box(h, l, lookback=40, confirm=3)
    assert b == {"top": 15.0, "bottom": 11.0, "len": 5, "held": True}


def test_darvas_box_current_bar_excluded_by_default():
    h = [10, 12, 11, 11, 11, 20]
    l = [9, 10, 9.5, 9.5, 9.5, 19]
    assert dv.darvas_box(h, l)["top"] == 12.0
    assert dv.darvas_box(h, l, exclude_current=False)["top"] == 20.0


def test_darvas_box_confirm_boundary():
    # top at index 2 of 7 usable bars; held bars after it = 4
    h = [1, 2, 9, 3, 3, 3, 3, 0]
    l = [0, 0, 5, 1, 1, 1, 1, 0]
    assert dv.darvas_box(h, l, confirm=4)["held"] is True
    assert dv.darvas_box(h, l, confirm=5)["held"] is False
    # a later bar that exceeds the top breaks the hold even if time passed
    h2 = [1, 2, 9, 3, 9.5, 3, 3, 0]
    assert dv.darvas_box(h2, l, confirm=3)["top"] == 9.5


def test_darvas_box_too_short_returns_nan_box():
    b = dv.darvas_box([1, 2, 3, 4, 5], [0, 1, 2, 3, 4], confirm=3)   # 4 usable < confirm+2
    assert np.isnan(b["top"]) and np.isnan(b["bottom"]) and b["len"] == 0
    assert "held" not in b
    ok = dv.darvas_box([1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5], confirm=3)     # 5 usable == confirm+2
    assert np.isfinite(ok["top"])


def test_darvas_box_lookback_window_and_nan_highs():
    h = [50] + [10] * 10 + [0]                        # old high of 50 is outside lookback=5
    l = [1] * 12
    b = dv.darvas_box(h, l, lookback=5, confirm=2)
    assert b["top"] == 10.0 and b["len"] == 5
    hn = [np.nan, 10, 12, 11, 11, 11, 11, 0]
    assert dv.darvas_box(hn, [1] * 8, confirm=3)["top"] == 12.0


# ── box_state ─────────────────────────────────────────────────────────────────
BOX = {"top": 20.0, "bottom": 10.0}


@pytest.mark.parametrize("close,state", [(20.01, "breakout"), (20.0, "in_box"), (10.0, "in_box"),
                                         (9.99, "breakdown"), (15.0, "in_box")])
def test_box_state_classification_edges(close, state):
    assert dv.box_state(close, close, close, BOX, 1.0, 1.0)["state"] == state


def test_box_state_position_and_clipping():
    assert dv.box_state(15, 15, 15, BOX, 1, 1)["position"] == 0.5
    assert dv.box_state(25, 25, 25, BOX, 1, 1)["position"] == 1.2          # clipped at 1.2
    assert dv.box_state(5, 5, 5, BOX, 1, 1)["position"] == 0.0             # clipped at 0
    assert dv.box_state(13.3333, 0, 0, BOX, 1, 1)["position"] == 0.333     # rounded to 3dp


def test_box_state_volume_confirmation_boundary():
    assert dv.box_state(25, 25, 25, BOX, 150, 100)["vol_confirmed"] is True     # == 1.5x
    assert dv.box_state(25, 25, 25, BOX, 149.99, 100)["vol_confirmed"] is False
    assert dv.box_state(25, 25, 25, BOX, 10, 0)["vol_confirmed"] is False       # no baseline volume


@pytest.mark.parametrize("box", [{"top": np.nan, "bottom": 1.0}, {"top": 5.0, "bottom": np.nan},
                                 {"top": 5.0, "bottom": 5.0}, {"top": 4.0, "bottom": 5.0}])
def test_box_state_no_box(box):
    r = dv.box_state(5, 5, 5, box, 1, 1)
    assert r["state"] == "no_box" and np.isnan(r["position"]) and r["vol_confirmed"] is False


# ── accumulation features / score ─────────────────────────────────────────────
def test_accumulation_features_keys_and_signs():
    n = 30
    close = np.linspace(10, 20, n)
    high, low = close + 0.1, close - 1.0               # closes near the top of the range
    vol = np.linspace(100, 300, n)
    f = dv.accumulation_features(high, low, close, vol)
    assert set(f) == {"obv_trend", "ad_trend", "cmf", "ud_vol_ratio", "vol_trend"}
    assert f["obv_trend"] > 0.9 and f["ad_trend"] > 0.9 and f["cmf"] > 0.5
    assert f["vol_trend"] == pytest.approx(1.0)
    assert f["ud_vol_ratio"] == np.inf                  # monotone rise -> no down volume


def _feat(rows):
    return pd.DataFrame(rows, columns=["obv_trend", "ad_trend", "cmf", "ud_vol_ratio", "vol_trend"])


def test_accumulation_score_values_and_eff_ratio_penalty():
    feat = _feat([[-1, -1, -1, 0.5, -1], [1, 1, 1, 2.0, 1]])
    assert dv.accumulation_score(feat).tolist() == [-5.0, 5.0]                  # five +/-1 z-scores
    feat["eff_ratio"] = [0.1, 0.9]                                              # pinned price is rewarded
    assert dv.accumulation_score(feat).tolist() == [-4.0, 4.0]
    feat["eff_ratio"] = [0.9, 0.1]
    assert dv.accumulation_score(feat).tolist() == [-6.0, 6.0]


def test_accumulation_score_rounds_to_3dp_and_orders():
    rng = np.random.RandomState(0)
    feat = _feat(rng.normal(size=(20, 5)))
    feat["ud_vol_ratio"] = np.exp(feat["ud_vol_ratio"])
    s = dv.accumulation_score(feat)
    assert (s.round(3) == s).all() and s.index.equals(feat.index)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="darvas_volume.py:152 accumulation_score replaces inf up/down-volume ratio by NaN "
                          "so a name with ONLY up-volume (the strongest accumulation) gets a NaN score and "
                          "sorts last")
def test_accumulation_score_all_upvolume_name_still_scored():
    feat = _feat([[0.5, 0.5, 0.5, np.inf, 0.5], [-0.5, -0.5, -0.5, 1.0, -0.5], [0.0, 0.0, 0.0, 2.0, 0.0]])
    assert np.isfinite(dv.accumulation_score(feat).iloc[0])


# ── scan_market ───────────────────────────────────────────────────────────────
def make_long(n_syms=20, n_days=300, seed=0, last_moves=None):
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    last_moves = last_moves or {}
    frames = []
    for k in range(n_syms):
        ret = rng.normal(0.0005, 0.01, n_days)
        vol = rng.uniform(0.8, 1.2, n_days) * 1e5 * (k + 1)
        if k in last_moves:
            ret[-1], mult = last_moves[k]
            vol[-1] *= mult
        close = 50.0 * (k + 1) * np.cumprod(1 + ret)
        frames.append(pd.DataFrame({"Date": dates, "Symbol": f"S{k}", "Close": close,
                                    "High": close * 1.005, "Low": close * 0.995, "Volume": vol}))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def seed(tmp_path, monkeypatch):
    store = {}

    def put(market, df):
        (tmp_path / f"cleaned_long_{market}.parquet").write_bytes(b"x")
        store[f"cleaned_long_{market}.parquet"] = df

    monkeypatch.setattr(pd, "read_parquet", lambda p, *a, **k: store[os.path.basename(p)].copy())
    for mod in (marketdata, dv):
        monkeypatch.setattr(mod, "SEED", str(tmp_path))
    monkeypatch.setattr(marketdata, "SEED_LOCAL", str(tmp_path / "nolocal"))
    return put


def test_scan_market_missing_and_degenerate(seed):
    assert dv.scan_market("XX").empty
    seed("US", make_long())
    assert dv.scan_market("US", lookback=300).empty            # len(c)=300 < lookback+5 -> skipped
    assert dv.scan_market("US", lookback=3).empty              # box needs confirm+2 usable bars


def test_scan_market_states_and_columns(seed):
    df = make_long(last_moves={19: (0.12, 6.0), 18: (-0.12, 3.0), 17: (0.12, 1.0)})
    seed("US", df)
    out = dv.scan_market("US")
    assert set(out["ticker"]) == {f"S{k}" for k in range(12, 20)}          # top-40% liquid only
    assert {"market", "ticker", "close", "box_top", "box_bottom", "state", "position", "vol_confirmed",
            "obv_trend", "ad_trend", "cmf", "ud_vol_ratio", "vol_trend", "eff_ratio", "vol_autocorr",
            "accumulation"} <= set(out.columns)
    r = out.set_index("ticker")
    assert r.loc["S19", "state"] == "breakout" and bool(r.loc["S19", "vol_confirmed"]) is True
    assert r.loc["S18", "state"] == "breakdown"
    assert r.loc["S17", "state"] == "breakout" and bool(r.loc["S17", "vol_confirmed"]) is False
    assert (r["box_top"] > r["box_bottom"]).all()
    c19 = df[df.Symbol == "S19"].sort_values("Date")["Close"].iloc[-1]
    assert r.loc["S19", "close"] == round(float(c19), 2)
    assert r["accumulation"].notna().all() and out["market"].eq("US").all()


def test_scan_market_box_excludes_current_bar(seed):
    df = make_long(last_moves={19: (0.2, 5.0)})
    seed("US", df)
    s19 = df[df.Symbol == "S19"].sort_values("Date")
    row = dv.scan_market("US").set_index("ticker").loc["S19"]
    prior_high = s19["High"].iloc[-41:-1].max()
    assert row["box_top"] == round(float(prior_high), 2)                  # last bar's high not in the box
    assert s19["High"].iloc[-1] > prior_high and row["state"] == "breakout"


def test_scan_market_accumulation_features_exclude_current_bar(seed):
    df = make_long()
    seed("US", df)
    s = df[df.Symbol == "S15"].sort_values("Date").iloc[-41:-1]
    row = dv.scan_market("US").set_index("ticker").loc["S15"]
    assert row["cmf"] == pytest.approx(dv.chaikin_money_flow(s.High, s.Low, s.Close, s.Volume))
    assert row["obv_trend"] == pytest.approx(dv.trend_corr(dv.obv(s.Close, s.Volume)))


# ── CLI ───────────────────────────────────────────────────────────────────────
def _scan_frame():
    rows = []
    for i, (st, conf, acc, ud) in enumerate([
            ("in_box", False, 3.0, 2.5), ("in_box", False, 9.0, np.inf), ("in_box", False, 1.0, 0.8),
            ("breakout", True, 2.0, 1.5), ("breakout", False, 8.0, 1.2), ("breakdown", False, 4.0, 0.5)]):
        rows.append({"market": "US", "ticker": f"T{i}", "close": 10.0 + i, "box_top": 12.0 + i,
                     "box_bottom": 8.0 + i, "state": st, "position": 0.5, "vol_confirmed": conf,
                     "obv_trend": 0.3, "cmf": 0.1, "ud_vol_ratio": ud, "eff_ratio": 0.2,
                     "vol_autocorr": 0.05, "accumulation": acc})
    return pd.DataFrame(rows)


def _run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["darvas_volume.py", *argv])
    dv.main()


def test_main_in_box_default_sorted_by_accumulation(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(dv, "scan_market", lambda m, lb: seen.append((m, lb)) or _scan_frame())
    _run_main(monkeypatch, "--market", "US", "--lookback", "25")
    out = capsys.readouterr().out
    assert seen == [("US", 25)]
    assert "DARVAS × VOLUME-ACQUISITION MONITOR — US" in out
    assert "COILING IN BOX — volume being acquired (stealth accumulation)  ·  3 names" in out
    rows = [ln.split()[1] for ln in out.splitlines() if ln.startswith("  US")]
    assert rows == ["T1", "T0", "T2"]                                   # 9.0, 3.0, 1.0
    inf_line = [ln for ln in out.splitlines() if ln.startswith("  US") and "T1" in ln][0]
    assert " inf " in inf_line                                          # infinite U/D shown as 'inf'


def test_main_breakout_keeps_only_volume_confirmed(monkeypatch, capsys):
    monkeypatch.setattr(dv, "scan_market", lambda m, lb: _scan_frame())
    _run_main(monkeypatch, "--market", "US", "--state", "breakout")
    out = capsys.readouterr().out
    assert "VOLUME-CONFIRMED BREAKOUTS" in out and "·  1 names" in out
    rows = [ln.split()[1] for ln in out.splitlines() if ln.startswith("  US")]
    assert rows == ["T3"]                                               # T4 (unconfirmed, acc 8) dropped


def test_main_breakdown_and_top_limit(monkeypatch, capsys):
    monkeypatch.setattr(dv, "scan_market", lambda m, lb: _scan_frame())
    _run_main(monkeypatch, "--market", "US", "--state", "breakdown")
    out = capsys.readouterr().out
    assert "BREAKDOWNS (box bottom lost)  ·  1 names" in out
    _run_main(monkeypatch, "--market", "US", "--top", "2")
    rows = [ln.split()[1] for ln in capsys.readouterr().out.splitlines() if ln.startswith("  US")]
    assert rows == ["T1", "T0"]


def test_main_all_markets_uses_market_list_and_tags(monkeypatch, capsys):
    monkeypatch.setattr(marketdata, "market_list", lambda: ["US", "JP", "KR", "DE"])
    seen = []
    monkeypatch.setattr(dv, "scan_market", lambda m, lb: seen.append(m) or (_scan_frame() if m != "JP"
                                                                           else pd.DataFrame()))
    _run_main(monkeypatch, "--all")
    out = capsys.readouterr().out
    assert seen == ["US", "JP", "KR", "DE"]
    assert "MONITOR — 4 markets" in out
    assert out.count("\n  US ") == 9                                    # 3 in-box rows x 3 non-empty markets
    seen.clear()
    monkeypatch.setattr(marketdata, "market_list", lambda: ["US", "JP"])
    _run_main(monkeypatch)                                              # no --market => all markets
    assert seen == ["US", "JP"] and "MONITOR — US, JP" in capsys.readouterr().out


def test_main_no_data_exits(monkeypatch):
    monkeypatch.setattr(dv, "scan_market", lambda m, lb: pd.DataFrame())
    with pytest.raises(SystemExit) as e:
        _run_main(monkeypatch, "--market", "US")
    assert "no data" in str(e.value)


def test_main_end_to_end_on_synthetic_seed(seed, monkeypatch, capsys):
    seed("US", make_long(last_moves={19: (0.12, 6.0)}))
    _run_main(monkeypatch, "--market", "US", "--state", "breakout")
    out = capsys.readouterr().out
    assert "1 names" in out and "S19" in out
