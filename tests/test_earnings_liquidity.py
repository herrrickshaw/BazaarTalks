"""Offline tests for earnings_liquidity.py (PEAD x liquidity conditioning study)."""
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import earnings_liquidity as el  # noqa: E402
import marketdata  # noqa: E402


# ── synthetic seed ────────────────────────────────────────────────────────────
def make_long(n_syms=20, n_days=300, seed=0, events=None):
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


def _panel(n=60, market="US", sign=1.0):
    k = np.arange(1, n + 1, dtype=float)
    return pd.DataFrame({"market": market, "ticker": [f"T{i}" for i in range(n)],
                         "price": k, "illiq": k, "dollar_vol": k[::-1].copy(),
                         "vol_surge": np.full(n, 3.0), "surprise": 0.05, "fwd_car": 0.01,
                         "dir_drift": sign * k / 500.0})


# ── pure helpers ──────────────────────────────────────────────────────────────
def test_directional_drift_sign_and_invalid():
    assert el.directional_drift(0.05, 0.02) == pytest.approx(0.02)
    assert el.directional_drift(-0.05, 0.02) == pytest.approx(-0.02)    # drift against the surprise
    assert el.directional_drift(-0.05, -0.02) == pytest.approx(0.02)    # continues down
    assert np.isnan(el.directional_drift(0.0, 0.02))
    assert np.isnan(el.directional_drift(np.nan, 0.02))
    assert np.isnan(el.directional_drift(0.05, np.nan))
    assert np.isnan(el.directional_drift(np.inf, 0.02))
    assert np.isnan(el.directional_drift(0.05, -np.inf))


def test_bucket_stats_columns_values_and_labels():
    t = el.bucket_stats(_panel(), "illiq", "dir_drift")
    assert list(t.columns) == ["bucket", "dir_drift_mean%", "dir_drift_med%", "n"]
    assert list(t["bucket"]) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert t["n"].tolist() == [12] * 5
    assert t["dir_drift_mean%"].tolist() == [1.3, 3.7, 6.1, 8.5, 10.9]   # mean(1..12)/5 etc.
    assert t["dir_drift_med%"].tolist() == [1.3, 3.7, 6.1, 8.5, 10.9]
    lab = el.bucket_stats(_panel(), "illiq", "dir_drift", q=2, labels=["low", "high"])
    assert list(lab["bucket"]) == ["low", "high"] and lab["n"].tolist() == [30, 30]


def test_bucket_stats_min_rows_boundary_and_nan():
    assert el.bucket_stats(_panel(n=14), "illiq", "dir_drift").empty       # < q*3
    assert not el.bucket_stats(_panel(n=15), "illiq", "dir_drift").empty
    p = _panel(n=15)
    p.loc[0, "illiq"] = np.nan
    assert el.bucket_stats(p, "illiq", "dir_drift").empty                  # NaN dropped -> 14


def test_spread_qhigh_qlow():
    t = el.bucket_stats(_panel(), "illiq", "dir_drift")
    assert el.spread_qhigh_qlow(t, "dir_drift_med%") == 9.6
    assert el.spread_qhigh_qlow(el.bucket_stats(_panel(sign=-1), "illiq", "dir_drift"),
                                "dir_drift_med%") == -9.6
    assert np.isnan(el.spread_qhigh_qlow(pd.DataFrame(), "x"))


def test_per_market_summary_ic_signs_and_ordering():
    up, down = _panel(n=50, market="AA"), _panel(n=50, market="BB", sign=-1.0)
    up["vol_surge"] = 4.26
    out = el.per_market_summary({"BB": down, "AA": up})
    assert out["market"].tolist() == ["AA", "BB"]                  # sorted by illiq_IC desc
    a = out.set_index("market")
    assert a.loc["AA", "illiq_IC"] == 1.0 and a.loc["BB", "illiq_IC"] == -1.0
    assert a.loc["AA", "vol_IC"] == -1.0                           # dollar_vol decreases with k
    assert a.loc["AA", "price_IC"] == 1.0
    assert a.loc["AA", "events"] == 50 and a.loc["AA", "vol_surge×"] == 4.3
    assert a.loc["BB", "vol_surge×"] == 3.0


def test_per_market_summary_min_events_filter_boundary_and_glitch_rows():
    assert not el.per_market_summary({"A": _panel(n=40)}).empty      # == min_events kept
    assert el.per_market_summary({"A": _panel(n=39)}).empty
    p = _panel(n=45)
    p.loc[:5, "dir_drift"] = 2.0                                     # |drift| >= 2 are dropped
    assert el.per_market_summary({"A": p}).empty                     # 45-6 = 39 < 40
    p2 = _panel(n=46)
    p2.loc[:5, "dir_drift"] = -2.5
    assert el.per_market_summary({"A": p2})["events"].tolist() == [40]
    assert el.per_market_summary({"A": p}, min_events=39)["events"].tolist() == [39]
    assert el.per_market_summary({}).empty
    assert el.per_market_summary({"E": pd.DataFrame()}).empty


# ── EDGAR helpers ─────────────────────────────────────────────────────────────
def test_parse_submissions_filters_sorts_dedupes():
    payload = {"filings": {"recent": {
        "form": ["10-K", "8-K", "10-Q", "10-Q", "10-Q/A", "10-K"],
        "filingDate": ["2024-02-20", "2024-03-01", "2023-11-01", "2023-11-01", "2023-12-01", "2022-02-20"]}}}
    assert el.parse_submissions(payload) == ["2022-02-20", "2023-11-01", "2024-02-20"]
    assert el.parse_submissions(payload, forms=("8-K",)) == ["2024-03-01"]
    assert el.parse_submissions(payload, forms=()) == []


def test_parse_submissions_ragged_and_missing():
    assert el.parse_submissions({}) == []
    assert el.parse_submissions({"filings": {}}) == []
    ragged = {"filings": {"recent": {"form": ["10-Q", "10-K", "10-Q"], "filingDate": ["2024-01-02"]}}}
    assert el.parse_submissions(ragged) == ["2024-01-02"]             # truncated to the shorter list


def test_cik_map_loads_once_and_caches(monkeypatch):
    calls = []

    def ticker_cik():
        calls.append(1)
        return {"AAPL": 320193}

    monkeypatch.setitem(sys.modules, "pit_fundamentals", types.SimpleNamespace(_ticker_cik=ticker_cik))
    monkeypatch.setattr(el, "_CIK", None)
    assert el._cik_map() == {"AAPL": 320193}
    assert el._cik_map() is el._cik_map()
    assert len(calls) == 1


def test_cik_map_failure_degrades_to_empty(monkeypatch):
    def boom():
        raise RuntimeError("db missing")

    monkeypatch.setitem(sys.modules, "pit_fundamentals", types.SimpleNamespace(_ticker_cik=boom))
    monkeypatch.setattr(el, "_CIK", None)
    assert el._cik_map() == {}
    assert el._CIK == {}


class _Resp:
    def __init__(self, status, payload):
        self.status_code, self._p = status, payload

    def json(self):
        return self._p


def _fake_api(monkeypatch, resp):
    calls = []

    def http_get(name, url, headers=None, retries=None):
        calls.append({"name": name, "url": url, "headers": headers, "retries": retries})
        if isinstance(resp, Exception):
            raise resp
        return resp

    monkeypatch.setitem(sys.modules, "apiclient", types.SimpleNamespace(http_get=http_get))
    return calls


_SUB = {"filings": {"recent": {"form": ["10-Q", "10-K", "8-K"],
                               "filingDate": ["2023-05-01", "2022-02-01", "2023-06-01"]}}}


def test_fetch_earnings_dates_edgar_url_headers_and_since(monkeypatch):
    monkeypatch.setattr(el, "_CIK", {"AAPL": 320193})
    monkeypatch.delenv("SEC_UA", raising=False)
    calls = _fake_api(monkeypatch, _Resp(200, _SUB))
    assert el.fetch_earnings_dates_edgar("aapl") == ["2022-02-01", "2023-05-01"]   # case-insensitive
    assert calls[0]["url"] == "https://data.sec.gov/submissions/CIK0000320193.json"
    assert calls[0]["name"] == "edgar" and calls[0]["retries"] == 2
    assert "admin@gms.dev" in calls[0]["headers"]["User-Agent"]
    # since is inclusive
    assert el.fetch_earnings_dates_edgar("AAPL", since="2023-05-01") == ["2023-05-01"]
    assert el.fetch_earnings_dates_edgar("AAPL", since="2023-05-02") == []


def test_fetch_earnings_dates_edgar_env_user_agent(monkeypatch):
    monkeypatch.setattr(el, "_CIK", {"X": 1})
    monkeypatch.setenv("SEC_UA", "me test@example.org")
    calls = _fake_api(monkeypatch, _Resp(200, _SUB))
    el.fetch_earnings_dates_edgar("X")
    assert calls[0]["headers"] == {"User-Agent": "me test@example.org"}
    assert calls[0]["url"].endswith("CIK0000000001.json")


def test_fetch_earnings_dates_edgar_failures_return_empty(monkeypatch):
    monkeypatch.setattr(el, "_CIK", {"AAPL": 320193})
    _fake_api(monkeypatch, _Resp(404, _SUB))
    assert el.fetch_earnings_dates_edgar("AAPL") == []
    _fake_api(monkeypatch, RuntimeError("offline"))
    assert el.fetch_earnings_dates_edgar("AAPL") == []
    calls = _fake_api(monkeypatch, _Resp(200, _SUB))
    assert el.fetch_earnings_dates_edgar("UNKNOWN") == []          # no CIK -> no request at all
    assert calls == []


# ── data assembly ─────────────────────────────────────────────────────────────
def test_scan_market_missing_parquet(seed):
    assert el.scan_market("XX").empty


def test_scan_market_rows_match_event_definition(seed):
    ev = {(19, 100): (0.08, 8), (19, 200): (-0.07, 8), (18, 120): (0.09, 6)}
    df = make_long(events=ev)
    seed("US", df)
    out = el.scan_market("US", horizon=40, pre=42)
    assert set(out.columns) == {"market", "ticker", "price", "illiq", "dollar_vol",
                                "vol_surge", "surprise", "fwd_car", "dir_drift"}
    assert (out["ticker"] == "S19").sum() == 2 and (out["ticker"] == "S18").sum() == 1
    r = out[(out["ticker"] == "S19")].sort_values("surprise")
    neg, pos = r.iloc[0], r.iloc[1]
    c = df[df.Symbol == "S19"].sort_values("Date")
    assert pos["price"] == pytest.approx(c["Close"].iloc[100])
    assert neg["price"] == pytest.approx(c["Close"].iloc[200])
    assert pos["vol_surge"] == pytest.approx(c["Volume"].iloc[100] / c["Volume"].iloc[58:100].mean())
    assert pos["dollar_vol"] == pytest.approx((c["Close"] * c["Volume"]).iloc[58:100].mean())
    assert pos["surprise"] > 0 > neg["surprise"]
    assert out["dir_drift"].tolist() == pytest.approx(
        [f * np.sign(s) for f, s in zip(out["fwd_car"], out["surprise"])])
    assert (out["illiq"] > 0).all() and out["market"].eq("US").all()


def test_scan_market_window_boundaries(seed):
    # event at 100: pre=42 fits, horizon 199 fits exactly (100+199 == len-1), 200 does not
    seed("US", make_long(events={(19, 100): (0.08, 8)}))
    assert len(el.scan_market("US", horizon=199)) == 1
    assert el.scan_market("US", horizon=200).empty
    # an event inside the pre-window (ev < pre) is skipped
    seed("US", make_long(events={(19, 30): (0.08, 8)}))
    assert el.scan_market("US", pre=42).empty
    assert len(el.scan_market("US", pre=30)) == 1                     # ev == pre is kept


def test_scan_market_drops_glitch_surprise(seed):
    seed("US", make_long(events={(19, 99): (0.3, 1), (19, 100): (0.3, 8), (19, 101): (0.3, 1)}))
    assert el.scan_market("US").empty


def _wide_frames(n=120, tickers=("A", "B"), seed_=1, start="2024-01-01"):
    rng = np.random.RandomState(seed_)
    idx = pd.bdate_range(start, periods=n)
    close = pd.DataFrame({t: 100 * np.cumprod(1 + rng.normal(0, 0.01, n)) for t in tickers}, index=idx)
    vol = pd.DataFrame({t: rng.uniform(1e5, 2e5, n) for t in tickers}, index=idx)
    return {"Close": close, "High": close, "Low": close, "Volume": vol}


def test_scan_us_edgar_missing_data(monkeypatch):
    monkeypatch.setattr(el, "_wide", lambda m: None)
    assert el.scan_us_edgar(["A"]).empty


def test_scan_us_edgar_event_alignment_and_boundaries(monkeypatch):
    w = _wide_frames(n=120)
    idx = w["Close"].index
    monkeypatch.setattr(el, "_wide", lambda m: w)
    import pead_factor
    monkeypatch.setattr(pead_factor, "_liquid_symbols", lambda c, v: ["A", "B"])
    # make the announcement day a clear move so surprise != 0
    w["Close"].iloc[60:, 0] *= 1.05
    dates = {
        "A": [
            str(idx[59].date() + pd.Timedelta(days=0)),                # day before the jump
            str((idx[59] + pd.Timedelta(days=1)).date()),               # may be a weekend -> next bar
            str(idx[21].date()),                                        # ev == pre (21): kept
            str(idx[20].date()),                                        # ev < pre: skipped
            str(idx[119 - 40].date()),                                  # ev + horizon == len-1: kept
            str(idx[119 - 39].date()),                                  # one past: skipped
        ],
        "B": [],                                                        # no filings -> nothing
    }
    monkeypatch.setattr(el, "fetch_earnings_dates_edgar", lambda s, since: dates.get(s, []))
    out = el.scan_us_edgar(["A", "B", "ZZZ"], horizon=40, pre=21)
    assert set(out["ticker"]) == {"A"}
    got = set(out["date"])
    assert str(idx[21].date()) in got and str(idx[79].date()) in got
    assert str(idx[20].date()) not in got and str(idx[80].date()) not in got
    assert out["market"].eq("US").all()
    assert (out["illiq"] > 0).all() and out["dollar_vol"].gt(0).all()
    assert out["dir_drift"].tolist() == pytest.approx(
        [f * np.sign(s) for f, s in zip(out["fwd_car"], out["surprise"])])


def test_scan_us_edgar_since_is_first_index_date_and_short_history(monkeypatch):
    w = _wide_frames(n=120)
    monkeypatch.setattr(el, "_wide", lambda m: w)
    seen = []
    monkeypatch.setattr(el, "fetch_earnings_dates_edgar", lambda s, since: seen.append((s, since)) or [])
    el.scan_us_edgar(["A"])
    assert seen == [("A", str(w["Close"].index[0].date()))]
    short = _wide_frames(n=21 + 40 + 10 - 1)                         # len(c) < pre+horizon+10 -> skipped
    monkeypatch.setattr(el, "_wide", lambda m: short)
    seen.clear()
    assert el.scan_us_edgar(["A"]).empty and seen == []
    long_enough = _wide_frames(n=21 + 40 + 10)
    monkeypatch.setattr(el, "_wide", lambda m: long_enough)
    el.scan_us_edgar(["A"])
    assert len(seen) == 1


def test_scan_us_edgar_drops_glitch_surprise(monkeypatch):
    w = _wide_frames(n=120)
    w["Close"].iloc[60:, 0] *= 3.0                                  # +200% jump -> |surprise| > cap
    monkeypatch.setattr(el, "_wide", lambda m: w)
    monkeypatch.setattr(el, "fetch_earnings_dates_edgar", lambda s, since: [str(w["Close"].index[60].date())])
    out = el.scan_us_edgar(["A"])
    assert out.empty or 60 not in [w["Close"].index.get_loc(pd.Timestamp(d)) for d in out["date"]]


# ── CLI ───────────────────────────────────────────────────────────────────────
def _run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["earnings_liquidity.py", *argv])
    el.main()


def test_main_default_study_output(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(el, "scan_market", lambda m, h: seen.append((m, h)) or _panel())
    _run_main(monkeypatch, "--market", "US", "--horizon", "25")
    text = capsys.readouterr().out
    assert seen == [("US", 25)]
    assert "US  ·  60 announcement-proxy events  ·  25d drift" in text
    assert "by Amihud ILLIQUIDITY (Q1=liquid … Q5=illiquid):" in text
    assert "median dir-drift: Q1=+1.3 Q2=+3.7 Q3=+6.1 Q4=+8.5 Q5=+10.9" in text
    assert "Q5−Q1 = +9.60%" in text and "IC(signal,dir_drift) = +1.000" in text
    assert "by dollar VOLUME (Q1=low-vol … Q5=high-vol):" in text
    assert "IC(signal,dir_drift) = -1.000" in text                 # dollar_vol falls with drift
    assert "by PRICE level (Q1=low-price … Q5=high-price):" in text
    assert "announcement-day volume surge: median 3.0× pre-event average (p90 3.0×)" in text
    assert "Chordia-Sadka" in text


def test_main_default_filters_extreme_drifts_and_exits_when_empty(monkeypatch, capsys):
    p = _panel()
    p.loc[:9, "dir_drift"] = 2.0                                    # filtered out: 50 events remain
    monkeypatch.setattr(el, "scan_market", lambda m, h: p)
    _run_main(monkeypatch, "--market", "US")
    assert "50 announcement-proxy events" in capsys.readouterr().out
    monkeypatch.setattr(el, "scan_market", lambda m, h: pd.DataFrame())
    with pytest.raises(SystemExit) as e:
        _run_main(monkeypatch, "--market", "US")
    assert "no earnings-event panel" in str(e.value)


def test_main_default_skips_buckets_when_too_few_rows(monkeypatch, capsys):
    monkeypatch.setattr(el, "scan_market", lambda m, h: _panel(n=10))
    _run_main(monkeypatch, "--market", "US")
    text = capsys.readouterr().out
    assert "10 announcement-proxy events" in text and "median dir-drift" not in text
    assert "announcement-day volume surge: median 3.0×" in text


def test_main_by_market_table_and_summary(monkeypatch, capsys):
    panels = {"AA": _panel(n=50, market="AA"), "BB": _panel(n=50, market="BB", sign=-1.0),
              "CC": _panel(n=10, market="CC")}
    monkeypatch.setattr(el, "scan_market", lambda m, h: panels[m])
    monkeypatch.setattr(marketdata, "market_list", lambda: list(panels), raising=True)
    monkeypatch.setattr(el, "marketdata", marketdata, raising=False)
    _run_main(monkeypatch, "--all", "--by-market")
    text = capsys.readouterr().out
    assert "PER-COUNTRY test (2 markets)" in text
    lines = [ln for ln in text.splitlines() if ln.startswith("  AA") or ln.startswith("  BB")]
    assert lines[0].split()[:3] == ["AA", "50", "3.0"] and lines[1].split()[0] == "BB"
    assert "1/2 markets show the Chordia-Sadka sign (illiq_IC>0); strongest: AA, BB" in text


def test_main_by_market_insufficient_events(monkeypatch, capsys):
    monkeypatch.setattr(el, "scan_market", lambda m, h: _panel(n=10))
    _run_main(monkeypatch, "--market", "US", "--by-market")
    assert "insufficient events per market" in capsys.readouterr().out


def test_main_edgar_mode_report(monkeypatch, capsys):
    import pead_factor
    w = _wide_frames(n=60, tickers=("A", "B", "C"))
    monkeypatch.setattr(el, "_wide", lambda m: w)
    monkeypatch.setattr(pead_factor, "_liquid_symbols", lambda c, v: ["A", "B", "C"])
    got = {}

    def fake_scan(tickers, horizon):
        got["args"] = (list(tickers), horizon)
        p = _panel()
        p["date"] = "2024-01-01"
        return p

    monkeypatch.setattr(el, "scan_us_edgar", fake_scan)
    _run_main(monkeypatch, "--edgar", "--limit", "2", "--horizon", "30")
    cap = capsys.readouterr()
    assert got["args"] == (["A", "B"], 30)                           # --limit truncates the universe
    assert "fetching EDGAR 10-Q/10-K filing dates for 2 liquid US names" in cap.err
    assert "(60 10-Q/10-K events)" in cap.out
    assert "Q1=+1.3 Q2=+3.7 Q3=+6.1 Q4=+8.5 Q5=+10.9" in cap.out
    assert "Q5−Q1 = +9.60%" in cap.out
    assert "illiq_IC (real dates) = +1.000" in cap.out
    assert "median 3.0×" in cap.out


def test_main_edgar_mode_no_events(monkeypatch, capsys):
    import pead_factor
    w = _wide_frames(n=60)
    monkeypatch.setattr(el, "_wide", lambda m: w)
    monkeypatch.setattr(pead_factor, "_liquid_symbols", lambda c, v: ["A"])
    monkeypatch.setattr(el, "scan_us_edgar", lambda t, h: pd.DataFrame())
    _run_main(monkeypatch, "--edgar")
    assert "no EDGAR-dated events (offline, or no filings in the price window)" in capsys.readouterr().out


def test_main_edgar_small_panel_skips_quintile_lines(monkeypatch, capsys):
    import pead_factor
    monkeypatch.setattr(el, "_wide", lambda m: _wide_frames(n=60))
    monkeypatch.setattr(pead_factor, "_liquid_symbols", lambda c, v: ["A"])
    monkeypatch.setattr(el, "scan_us_edgar", lambda t, h: _panel(n=8))
    _run_main(monkeypatch, "--edgar")
    out = capsys.readouterr().out
    assert "(8 10-Q/10-K events)" in out and "by illiquidity quintile" not in out


@pytest.mark.xfail(strict=True, raises=NameError,
                   reason="earnings_liquidity.py:main `marketdata.market_list()` - `marketdata` is never "
                          "imported at module level (only `from marketdata import wide`), so --all / no "
                          "--market raises NameError")
def test_main_all_markets_resolves_market_list(monkeypatch, capsys):
    monkeypatch.setattr(el, "scan_market", lambda m, h: _panel())
    _run_main(monkeypatch, "--all")
    assert "EARNINGS × LIQUIDITY" in capsys.readouterr().out
