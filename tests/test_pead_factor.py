"""Offline tests for pead_factor.py (PEAD event study on synthetic prices)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import marketdata  # noqa: E402
import pead_factor as pf  # noqa: E402


# ── synthetic seed helpers ────────────────────────────────────────────────────
def make_long(n_syms=10, n_days=300, seed=0, events=None):
    """Long OHLCV frame. events: {(sym_idx, day): (ret, vol_mult)}."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    events = events or {}
    frames = []
    for k in range(n_syms):
        ret = rng.normal(0.0005, 0.01, n_days)
        vol = rng.uniform(0.8, 1.2, n_days) * 1e5 * (k + 1)
        for (kk, d), (r, m) in events.items():
            if kk == k:
                ret[d] = r
                vol[d] *= m
        close = 50.0 * (k + 1) * np.cumprod(1 + ret)
        frames.append(pd.DataFrame({"Date": dates, "Symbol": f"S{k}", "Close": close,
                                    "High": close * 1.01, "Low": close * 0.99,
                                    "Volume": vol}))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def seed(tmp_path, monkeypatch):
    store = {}

    def put(market, df):
        (tmp_path / f"cleaned_long_{market}.parquet").write_bytes(b"x")
        store[f"cleaned_long_{market}.parquet"] = df

    monkeypatch.setattr(pd, "read_parquet", lambda p, *a, **k: store[os.path.basename(p)].copy())
    for mod in (marketdata, pf):
        monkeypatch.setattr(mod, "SEED", str(tmp_path))
    monkeypatch.setattr(marketdata, "SEED_LOCAL", str(tmp_path / "nolocal"))
    return put


EVENTS = {(0, 100): (0.08, 6), (0, 200): (-0.07, 6), (1, 120): (0.09, 7)}


# ── pure core ─────────────────────────────────────────────────────────────────
def test_sue_basic_and_degenerate():
    assert pf.sue(1.5, 1.0, 0.25) == pytest.approx(2.0)
    assert pf.sue(0.5, 1.0, 0.25) == pytest.approx(-2.0)
    assert np.isnan(pf.sue(1, 0, 0))
    assert np.isnan(pf.sue(1, 0, float("nan")))
    assert np.isnan(pf.sue(1, 0, None))


def test_market_adjust_beta_and_alignment():
    idx = pd.date_range("2024-01-01", periods=3)
    s = pd.Series([0.02, 0.01, -0.01], index=idx)
    m = pd.Series([0.01, 0.01, 0.01], index=idx)
    assert pf.market_adjust(s, m).tolist() == pytest.approx([0.01, 0.0, -0.02])
    assert pf.market_adjust(s, m, beta=2.0).tolist() == pytest.approx([0.0, -0.01, -0.03])
    short = m.iloc[:2]                     # market missing last day -> NaN abnormal
    out = pf.market_adjust(s, short)
    assert np.isnan(out.iloc[2]) and out.iloc[0] == pytest.approx(0.01)


def test_car_window_is_inclusive_and_tolerant():
    a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert pf.car(a, 1, 3) == 9.0               # positions 1..3 inclusive
    assert pf.car(a, 2, 2) == 3.0               # single bar
    assert pf.car(a, -3, 1) == 3.0              # negative start clipped to 0
    assert pf.car(a, 3, 99) == 9.0              # end past range
    assert pf.car(a, 10, 12) == 0.0             # fully out of range
    assert pf.car(a, 3, 1) == 0.0               # empty window


def test_event_surprise_uses_tight_window():
    a = pd.Series([0.0, 0.01, 0.05, -0.02, 0.5])
    assert pf.event_surprise(a, 2) == pytest.approx(0.04)           # 0.01+0.05-0.02
    assert pf.event_surprise(a, 2, half=0) == pytest.approx(0.05)
    assert pf.event_surprise(a, 0) == pytest.approx(0.01)           # start clipped at 0


def _spike_series(n=200, spikes=()):
    rng = np.random.RandomState(1)
    ret = rng.normal(0, 0.01, n)
    vol = np.full(n, 1000.0) * rng.uniform(0.9, 1.1, n)
    for i in spikes:
        ret[i] = 0.2
        vol[i] = 20000.0
    close = pd.Series(100 * np.cumprod(1 + ret))
    return close, pd.Series(vol)


def test_detect_events_finds_spike_and_requires_both_conditions():
    close, vol = _spike_series(spikes=[80])
    assert pf.detect_events(close, vol) == [80]
    # volume spike alone (no price jump) is not an event
    close2, vol2 = _spike_series(spikes=[])
    vol2.iloc[80] = 50000.0
    assert pf.detect_events(close2, vol2) == []
    # price jump alone is not an event either
    close3, vol3 = _spike_series(spikes=[])
    close3.iloc[80:] *= 1.3
    assert pf.detect_events(close3, vol3) == []


def test_detect_events_declusters_on_min_gap_boundary():
    close, vol = _spike_series(spikes=[60, 99])           # gap 39 < 40 -> merged
    assert pf.detect_events(close, vol) == [60]
    close, vol = _spike_series(spikes=[60, 100])          # gap 40 >= 40 -> both kept
    assert pf.detect_events(close, vol) == [60, 100]
    close, vol = _spike_series(spikes=[60, 99])
    assert pf.detect_events(close, vol, min_gap=39) == [60, 99]


def test_detect_events_ignores_warmup_bars():
    close, vol = _spike_series(spikes=[10])               # inside the 20-bar warm-up
    assert pf.detect_events(close, vol) == []


@pytest.mark.parametrize("surprise,days,expected", [
    (0.1, -1, 50.0),            # before the event
    (0.1, 61, 50.0),            # past the horizon
    (float("nan"), 5, 50.0),    # undefined surprise
    (0.0, 5, 50.0),             # no surprise -> neutral
])
def test_pead_score_neutral_cases(surprise, days, expected):
    assert pf.pead_score(surprise, days) == expected


def test_pead_score_decay_and_sign_and_boundaries():
    s = 0.05
    now = pf.pead_score(s, 0)
    assert now == pytest.approx(50 + 50 * np.tanh(0.4))
    assert pf.pead_score(s, 20) == pytest.approx(50 + 50 * np.tanh(0.4) * 0.5)   # one half-life
    assert pf.pead_score(-s, 0) == pytest.approx(100 - now)                      # symmetric
    assert pf.pead_score(s, 60) != 50.0                                          # horizon inclusive
    assert pf.pead_score(s, 61) == 50.0
    assert pf.pead_score(s, 10, horizon=10) != 50.0
    assert pf.pead_score(s, 11, horizon=10) == 50.0
    assert pf.pead_score(s, 5, halflife=5) == pytest.approx(50 + 50 * np.tanh(0.4) * 0.5)
    assert 0.0 <= pf.pead_score(10.0, 0) <= 100.0


def _events(n=40, slope=1.0):
    s = np.linspace(-0.1, 0.1, n)
    return pd.DataFrame({"surprise": s, "fwd_car": s * slope})


def test_drift_by_surprise_monotone_curve():
    c = pf.drift_by_surprise(_events())
    assert list(c.columns) == ["bucket", "mean%", "median%", "n"]
    assert list(c["bucket"]) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert c["n"].tolist() == [8] * 5
    assert c["mean%"].is_monotonic_increasing
    assert c["mean%"].iloc[0] < 0 < c["mean%"].iloc[-1]
    assert pf.monotonicity(c) == pytest.approx(1.0)


def test_drift_by_surprise_min_rows_boundary_and_nan_drop():
    assert pf.drift_by_surprise(_events(n=14)).empty            # < q*3
    assert not pf.drift_by_surprise(_events(n=15)).empty        # == q*3
    ev = _events(n=15)
    ev.loc[0, "fwd_car"] = np.nan                               # drops to 14 -> empty
    assert pf.drift_by_surprise(ev).empty
    assert len(pf.drift_by_surprise(_events(), q=2)) == 2


def test_monotonicity_inverse_and_short_curve():
    c = pf.drift_by_surprise(_events(slope=-1.0))
    assert pf.monotonicity(c) == pytest.approx(-1.0)
    assert np.isnan(pf.monotonicity(c.head(2)))


# ── data assembly ─────────────────────────────────────────────────────────────
def test_market_wide_missing_and_present(seed):
    assert pf._market_wide("XX") == (None, None)
    seed("US", make_long(n_syms=3, n_days=20))
    close, vol = pf._market_wide("US")
    assert close.shape == (20, 3) and vol.shape == (20, 3)
    assert list(close.columns) == ["S0", "S1", "S2"]
    assert close.dtypes.eq(float).all()


def test_liquid_symbols_delegates_with_module_constants(seed):
    seed("US", make_long(n_syms=10, n_days=300))
    close, vol = pf._market_wide("US")
    liq = pf._liquid_symbols(close, vol)
    # S0..S9 have increasing dollar volume (k+1)^2-ish; top 40% only
    assert liq == ["S6", "S7", "S8", "S9"]
    # short history -> nobody qualifies (MIN_HISTORY=250)
    assert pf._liquid_symbols(close.head(200), vol.head(200)) == []


def test_scan_market_missing_returns_empty_frames(seed):
    ev, sig = pf.scan_market("NOPE")
    assert ev.empty and sig.empty


def test_scan_market_events_and_signals(seed):
    ev_cfg = {(9, 100): (0.08, 8), (9, 200): (-0.07, 8), (8, 120): (0.09, 8),
              (7, 295): (0.08, 8)}
    seed("US", make_long(events=ev_cfg))
    events, sig = pf.scan_market("US", horizon=60)
    assert set(events.columns) == {"market", "ticker", "surprise", "fwd_car"}
    assert set(events["market"]) == {"US"}
    # S9 has two events with a full 60d window, S8 one; S7's event @295 has no complete window
    assert (events["ticker"] == "S9").sum() == 2
    assert (events["ticker"] == "S8").sum() == 1
    assert "S7" not in set(events["ticker"])
    s9 = events[events["ticker"] == "S9"].sort_values("surprise")
    assert s9["surprise"].iloc[0] < -0.05 and s9["surprise"].iloc[1] > 0.05
    # signals: one row per stock with a (non-glitch) event, from the most recent one
    assert set(sig["ticker"]) == {"S9", "S8", "S7"}
    row = sig.set_index("ticker")
    assert row.loc["S9", "days_since"] == 299 - 200
    assert row.loc["S7", "days_since"] == 299 - 295
    assert row.loc["S9", "pead_score"] == 50.0                  # past the 60d horizon
    assert row.loc["S7", "pead_score"] > 50.0                   # fresh positive surprise
    assert row.loc["S7", "pead_score"] == round(pf.pead_score(row.loc["S7", "surprise"], 4, 60), 1)


def test_scan_market_drops_glitch_events_above_surprise_cap(seed):
    # three +30% days (clipped to 25%/day) -> 3-day surprise 0.75 > SURPRISE_CAP
    seed("US", make_long(events={(9, 99): (0.3, 1), (9, 100): (0.3, 8), (9, 101): (0.3, 1),
                                 (8, 120): (0.08, 8)}))
    events, sig = pf.scan_market("US")
    assert set(events["ticker"]) == {"S8"}
    assert set(sig["ticker"]) == {"S8"}


def test_scan_market_horizon_boundary(seed):
    # event at day 100, horizon 199 -> ev+h == len-1 (299) is kept; 200 is not
    seed("US", make_long(events={(9, 100): (0.08, 8)}))
    kept, _ = pf.scan_market("US", horizon=199)
    dropped, _ = pf.scan_market("US", horizon=200)
    assert (kept["ticker"] == "S9").sum() == 1
    assert dropped.empty or "S9" not in set(dropped["ticker"])


# ── CLI ───────────────────────────────────────────────────────────────────────
def _run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["pead_factor.py", *argv])
    pf.main()


def test_main_market_prints_study_and_writes_out(seed, monkeypatch, capsys, tmp_path):
    cfg = {}
    for k in range(12, 20):
        for j, d in enumerate((60, 130, 200)):
            sign = 1 if (k + j) % 2 else -1
            cfg[(k, d)] = (sign * (0.05 + 0.005 * ((k * 7 + j * 3) % 8)), 8)
    seed("US", make_long(n_syms=20, events=cfg))
    out = tmp_path / "pead.csv"
    _run_main(monkeypatch, "--market", "US", "--horizon", "40", "--top", "3", "--out", str(out))
    text = capsys.readouterr().out
    assert "PEAD EVENT STUDY — US" in text
    assert "horizon 40d" in text
    assert "forward CAR by surprise quantile" in text
    assert "monotonicity =" in text
    assert "long-short (Q5−Q1) forward CAR" in text
    assert "current PEAD signals — top 3" in text
    written = pd.read_csv(out)
    assert list(written.columns) == ["ticker", "pead_score"]
    assert len(written) > 0


def test_main_no_events_exits(seed, monkeypatch):
    seed("US", make_long(n_syms=10))                            # no injected events
    # random data may still trigger none -> SystemExit; assert the message
    with pytest.raises(SystemExit) as e:
        _run_main(monkeypatch, "--market", "US")
    assert "no events detected" in str(e.value)


def test_main_missing_market_exits(seed, monkeypatch):
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, "--market", "ZZ")


def test_main_few_events_skips_curve(seed, monkeypatch, capsys):
    seed("US", make_long(events={(9, 100): (0.08, 8)}))
    _run_main(monkeypatch, "--market", "US", "--top", "2")
    text = capsys.readouterr().out
    assert "1 earnings-proxy events" in text
    assert "forward CAR by surprise quantile" not in text       # < 15 events -> no curve


@pytest.mark.xfail(strict=True, raises=NameError,
                   reason="pead_factor.py:main `marketdata.market_list()` - module `marketdata` is never "
                          "imported at module level (only inside helper functions), so --all / no --market "
                          "raises NameError")
def test_main_all_markets_uses_market_list(seed, monkeypatch, capsys):
    seed("US", make_long(events={(9, 100): (0.08, 8)}))
    _run_main(monkeypatch, "--all")
    assert "PEAD EVENT STUDY" in capsys.readouterr().out
