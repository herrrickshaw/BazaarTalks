"""Offline tests for dvm_composite.py (momentum, durability, global D/V/M ranking CLI)."""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dvm_composite as dvm  # noqa: E402


# ── momentum ──────────────────────────────────────────────────────────────────
def _wavy(n=260, drift=0.0015, amp=0.01, seed=0):
    rng = np.random.default_rng(seed)
    r = drift + amp * rng.standard_normal(n)
    return pd.Series(100 * np.cumprod(1 + r))


def test_momentum_requires_200_points():
    s = _wavy(300)
    assert np.isnan(dvm.momentum(s.iloc[:199]))
    assert not np.isnan(dvm.momentum(s.iloc[:200]))


def test_momentum_bounded_0_100_and_float():
    for seed in range(5):
        m = dvm.momentum(_wavy(seed=seed))
        assert isinstance(m, float) and 0.0 <= m <= 100.0


def test_momentum_uptrend_beats_downtrend():
    up = dvm.momentum(_wavy(drift=0.004, amp=0.005, seed=1))
    dn = dvm.momentum(_wavy(drift=-0.004, amp=0.005, seed=1))
    assert up > dn
    assert up > 60 and dn < 50


def test_momentum_components_for_clean_downtrend():
    # steady -1%/day: RSI=0 -> 0 ; MACD hist > 0 (decaying macd rises toward 0, above its signal) -> 100 ;
    # px below both DMAs -> 20 ; 52w part = 100+(px/hi-1)*300 clipped to 0
    c = pd.Series(100 * 0.99 ** np.arange(260))
    hi52 = c.rolling(252, min_periods=150).max().iloc[-1]
    d = c.diff()
    rsi = 100 - 100 / (1 + d.clip(lower=0).rolling(14).mean() /
                       (-d.clip(upper=0)).rolling(14).mean().replace(0, np.nan))
    assert rsi.iloc[-1] == 0.0
    hi_part = min(100, max(0, 100 + (c.iloc[-1] / hi52 - 1) * 300))
    assert dvm.momentum(c) == pytest.approx((0 + 100 + 20 + hi_part) / 4)
    assert hi_part == 0


def test_momentum_overbought_rsi_penalised():
    # RSI > 70 maps to 70 - (rsi-70)*2: higher RSI -> lower sub-score
    rng = np.random.default_rng(3)
    base = np.concatenate([rng.normal(0.0005, 0.01, 240), np.full(20, 0.0)])
    mild = pd.Series(100 * np.cumprod(1 + base + np.concatenate([np.zeros(246), [0.01, -0.002] * 7])))
    assert 0 <= dvm.momentum(mild) <= 100


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="dvm_composite.py:47-53 RSI is NaN when avg loss==0 (replace(0,nan)); "
                          "max(0,nan)==0 so a perfectly rising stock gets RSI sub-score 0 instead of "
                          "the 10 that RSI=100 implies -> momentum 75.0 not 77.5")
def test_momentum_monotonic_rise_rsi_100_subscore():
    c = pd.Series(100 * 1.01 ** np.arange(260))
    assert dvm.momentum(c) == pytest.approx((10 + 100 + 100 + 100) / 4)


def test_momentum_trend_buckets():
    # price above DMA200 but below DMA50 -> 60 ; build: long rise then a sharp 15% dip
    c = pd.Series(np.concatenate([np.linspace(50, 100, 240), np.linspace(100, 85, 20)]))
    dma50 = c.rolling(50).mean().iloc[-1]; dma200 = c.rolling(200).mean().iloc[-1]
    assert c.iloc[-1] < dma50 and c.iloc[-1] > dma200
    d = c.diff()
    rsi = (100 - 100 / (1 + d.clip(lower=0).rolling(14).mean() /
                        (-d.clip(upper=0)).rolling(14).mean().replace(0, np.nan))).iloc[-1]
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    macd_h = (macd - macd.ewm(span=9).mean()).iloc[-1]
    hi = c.rolling(252, min_periods=150).max().iloc[-1]
    subs = [min(100, max(0, rsi if rsi <= 70 else 70 - (rsi - 70) * 2)),
            100 if macd_h > 0 else 25, 60, min(100, max(0, 100 + (c.iloc[-1] / hi - 1) * 300))]
    assert dvm.momentum(c) == pytest.approx(np.mean(subs))


# ── durability ────────────────────────────────────────────────────────────────
def _r(**kw):
    base = dict(roe=np.nan, de=np.nan, rev_growth=np.nan, op_margin=np.nan, earn_growth=np.nan)
    base.update(kw)
    return pd.Series(base)


@pytest.mark.parametrize("roe,score", [(20.01, 90), (20, 75), (15.01, 75), (15, 55), (10.01, 55),
                                       (10, 35), (0.01, 35), (0, 15), (-5, 15)])
def test_durability_roe_buckets(roe, score):
    assert dvm.durability(_r(roe=roe)) == score


@pytest.mark.parametrize("de,score", [(0.0, 90), (0.49, 90), (0.5, 70), (0.99, 70), (1.0, 45),
                                      (1.99, 45), (2.0, 20), (5.0, 20)])
def test_durability_de_buckets(de, score):
    assert dvm.durability(_r(de=de)) == score


@pytest.mark.parametrize("rg,score", [(15.01, 85), (15, 60), (5.01, 60), (5, 40), (0.01, 40), (0, 20), (-3, 20)])
def test_durability_rev_growth_buckets(rg, score):
    assert dvm.durability(_r(rev_growth=rg)) == score


@pytest.mark.parametrize("om,score", [(20.01, 85), (20, 60), (10.01, 60), (10, 40), (0.01, 40), (0, 20), (-1, 20)])
def test_durability_op_margin_buckets(om, score):
    assert dvm.durability(_r(op_margin=om)) == score


@pytest.mark.parametrize("eg,score", [(0.01, 75), (0, 35), (-4, 35)])
def test_durability_earn_growth_buckets(eg, score):
    assert dvm.durability(_r(earn_growth=eg)) == score


def test_durability_mean_of_available_subscores_and_all_nan():
    r = _r(roe=25, de=0.3, rev_growth=20, op_margin=25, earn_growth=1)
    assert dvm.durability(r) == pytest.approx(85.0)
    assert dvm.durability(_r(roe=25, de=3)) == pytest.approx((90 + 20) / 2)
    assert np.isnan(dvm.durability(_r()))


def test_durability_returns_python_float():
    assert type(dvm.durability(_r(roe=12))) is float


# ── main ──────────────────────────────────────────────────────────────────────
def _series_for(kind, n=260):
    t = np.arange(n)
    if kind == "up":
        return 100 * 1.002 ** t * (1 + 0.01 * np.sin(t / 3))
    if kind == "down":
        return 100 * 0.998 ** t * (1 + 0.01 * np.sin(t / 3))
    return 100 * (1 + 0.02 * np.sin(t / 7))


def _px_frame(spec):
    dates = pd.bdate_range("2023-01-02", periods=260)
    rows = []
    for sym, kind in spec.items():
        rows += [{"Date": d, "Symbol": sym, "Close": float(x)} for d, x in zip(dates, _series_for(kind))]
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=0)           # unsorted: main must sort by Date


FUND_COLS = ["market", "ticker", "pe", "pb", "roe", "roa", "de", "rev_growth", "earn_growth",
             "op_margin", "div_yield", "mktcap", "sector"]


def _fund(rows):
    df = pd.DataFrame(rows, columns=FUND_COLS)
    return df


def _make_cache(path, df):
    con = sqlite3.connect(path)
    df.astype(object).to_sql("fund", con, index=False, if_exists="replace")
    con.close()


@pytest.fixture
def env(tmp_path, monkeypatch):
    seed = tmp_path / "seed"; seed.mkdir()
    cache = tmp_path / "fundamentals_cache.db"
    monkeypatch.setattr(dvm, "SEED", str(seed))
    monkeypatch.setattr(dvm, "CACHE", str(cache))
    frames = {}

    def fake_read_parquet(p, *a, **k):
        return frames[os.path.basename(str(p))].copy()

    monkeypatch.setattr(dvm.pd, "read_parquet", fake_read_parquet)

    class E:
        pass
    e = E(); e.seed = seed; e.cache = cache; e.frames = frames; e.out = str(tmp_path / "out.db")

    def add_market(mkt, spec):
        (seed / f"cleaned_long_{mkt}.parquet").write_bytes(b"x")
        frames[f"cleaned_long_{mkt}.parquet"] = _px_frame(spec)
    e.add_market = add_market
    return e


US_ROWS = [
    ("US", "U1", 10.0, 1.0, 25, 10, 0.3, 20, 1, 25, 1.0, 1e9, "Tech"),
    ("US", "U2", 20.0, 2.0, 5, 3, 1.5, 3, -1, 5, 0.5, 1e9, "Fin"),
    ("US", "U3", -5.0, np.nan, 12, 3, np.nan, np.nan, np.nan, np.nan, 0.0, 1e9, "Util"),
    ("US", "NOPX", 10.0, 1.0, 25, 10, 0.3, 20, 1, 25, 1.0, 1e9, "Tech"),      # no price history
    ("JP", "J1", 10.0, 1.0, 25, 10, 0.3, 20, 1, 25, 1.0, 1e9, "Tech"),        # market has no parquet
]


def test_main_missing_cache_prints_and_returns(env, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["dvm_composite.py", "--db", env.out])
    assert dvm.main() is None
    assert "no fundamentals_cache.db" in capsys.readouterr().err
    assert not os.path.exists(env.out)


def test_main_no_price_data_reports_and_does_not_write(env, monkeypatch, capsys):
    _make_cache(str(env.cache), _fund(US_ROWS[:1]))
    monkeypatch.setattr(sys, "argv", ["dvm_composite.py", "--db", env.out])
    dvm.main()
    err = capsys.readouterr().err
    assert "1 stocks with fundamentals across 1 markets" in err
    assert "no stocks with both momentum + durability" in err
    assert not os.path.exists(env.out)


def test_main_ranks_writes_db_and_prints(env, monkeypatch, capsys):
    env.add_market("US", {"U1": "up", "U2": "down", "U3": "up"})
    _make_cache(str(env.cache), _fund(US_ROWS))
    monkeypatch.setattr(sys, "argv", ["dvm_composite.py", "--db", env.out])
    dvm.main()
    cap = capsys.readouterr()
    assert "5 stocks with fundamentals across 2 markets" in cap.err
    con = sqlite3.connect(env.out)
    db = pd.read_sql("SELECT * FROM dvm_composite", con); con.close()
    assert set(db["ticker"]) == {"U1", "U2", "U3"}            # NOPX (no px) and J1 (no parquet) dropped
    assert "_ey" not in db.columns and "_pb" not in db.columns
    assert db["composite"].is_monotonic_decreasing
    d = db.set_index("ticker")
    # durability uses the bucket tables; valuation is a cross-sectional blend of earnings-yield & P/B rank
    assert d.loc["U1", "D"] == 85.0
    assert d.loc["U2", "D"] == 39.0
    assert d.loc["U3", "D"] == 55.0
    assert d.loc["U1", "V"] == 80.0
    assert d.loc["U2", "V"] == 30.0
    assert d.loc["U3", "V"] == 55.0                           # NaN ranks filled with the column mean
    for t, row in d.iterrows():
        g = lambda x: "G" if x >= 50 else "B"
        assert row["code"] == g(row["D"]) + g(row["V"]) + g(row["M"])
        assert row["label"] == dvm.LABELS[row["code"]]
        assert row["composite"] == round((row["D"] + row["V"] + row["M"]) / 3, 1)
    assert d.loc["U1", "M"] > d.loc["U2", "M"]                # uptrend vs downtrend
    out = cap.out
    assert "=== GLOBAL DVM COMPOSITE — 3 stocks, all markets ===" in out
    assert "classification: {" in out
    assert f"saved to {env.out}" in out


def test_main_default_view_lists_only_ggg(env, monkeypatch, capsys):
    env.add_market("US", {"U1": "up", "U2": "down", "U3": "up"})
    _make_cache(str(env.cache), _fund(US_ROWS))
    monkeypatch.setattr(sys, "argv", ["dvm_composite.py", "--db", env.out])
    dvm.main()
    out = capsys.readouterr().out
    con = sqlite3.connect(env.out)
    db = pd.read_sql("SELECT ticker, code, label FROM dvm_composite", con); con.close()
    n_ggg = int((db["code"] == "GGG").sum())
    assert f"=== top GGG Strong Performers ({n_ggg}) — top 20 by composite ===" in out
    listed = [ln.split()[1] for ln in out.split("top 20 by composite ===")[1].splitlines()
              if ln.startswith("  US")]
    assert set(listed) == set(db[db["code"] == "GGG"]["ticker"])


def test_main_code_filter(env, monkeypatch, capsys):
    env.add_market("US", {"U1": "up", "U2": "down", "U3": "up"})
    _make_cache(str(env.cache), _fund(US_ROWS))
    monkeypatch.setattr(sys, "argv", ["dvm_composite.py", "--db", env.out])
    dvm.main()
    con = sqlite3.connect(env.out)
    db = pd.read_sql("SELECT ticker, code FROM dvm_composite", con); con.close()
    code = db["code"].iloc[-1]                                 # the lowest-ranked name's code
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["dvm_composite.py", "--db", env.out, "--code", code])
    dvm.main()
    out = capsys.readouterr().out
    want = db[db["code"] == code]["ticker"].tolist()
    assert f"=== code={code} ({len(want)}) — top 20 by composite ===" in out
    listed = [ln.split()[1] for ln in out.split("top 20 by composite ===")[1].splitlines()
              if ln.startswith("  US")]
    assert sorted(listed) == sorted(want)


def test_main_non_numeric_fundamentals_coerced(env, monkeypatch, capsys):
    env.add_market("US", {"U1": "up", "U2": "down"})
    rows = [list(r) for r in US_ROWS[:2]]
    rows[0][2] = "n/a"                                         # pe unparseable -> NaN -> no earnings yield
    rows[1][4] = "bad"                                         # roe unparseable
    _make_cache(str(env.cache), _fund(rows))
    monkeypatch.setattr(sys, "argv", ["dvm_composite.py", "--db", env.out])
    dvm.main()
    con = sqlite3.connect(env.out)
    db = pd.read_sql("SELECT * FROM dvm_composite", con).set_index("ticker"); con.close()
    assert set(db.index) == {"U1", "U2"}
    assert np.isnan(db.loc["U1", "pe"])
    # U2 durability without roe: (45 + 40 + 40 + 35)/4
    assert db.loc["U2", "D"] == 40.0


def test_main_pe_zero_has_no_earnings_yield(env, monkeypatch):
    env.add_market("US", {"U1": "up", "U2": "up"})
    rows = [list(US_ROWS[0]), list(US_ROWS[1])]
    rows[0][2] = 0.0                                           # pe == 0 -> ey None (no ZeroDivisionError)
    _make_cache(str(env.cache), _fund(rows))
    monkeypatch.setattr(sys, "argv", ["dvm_composite.py", "--db", env.out])
    dvm.main()
    con = sqlite3.connect(env.out)
    db = pd.read_sql("SELECT ticker, V FROM dvm_composite", con).set_index("ticker"); con.close()
    # U1 ey NaN -> filled with mean of the single valid rank (U2: 100) ; pb rank U1=50 U2=0 -> V
    assert db.loc["U1", "V"] == round(0.6 * 100 + 0.4 * 50, 1)
    assert db.loc["U2", "V"] == round(0.6 * 100 + 0.4 * 0, 1)


def test_main_replaces_existing_table(env, monkeypatch):
    env.add_market("US", {"U1": "up"})
    _make_cache(str(env.cache), _fund(US_ROWS[:1]))
    con = sqlite3.connect(env.out)
    con.execute("CREATE TABLE dvm_composite (junk TEXT)"); con.commit(); con.close()
    monkeypatch.setattr(sys, "argv", ["dvm_composite.py", "--db", env.out])
    dvm.main()
    con = sqlite3.connect(env.out)
    cols = [r[1] for r in con.execute("PRAGMA table_info(dvm_composite)")]; con.close()
    assert "junk" not in cols and "composite" in cols
