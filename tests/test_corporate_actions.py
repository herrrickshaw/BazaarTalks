"""Offline tests for corporate_actions.py: split / rights detectors, EDGAR parsing, screeners, CLI."""
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import apiclient  # noqa: E402
import corporate_actions as ca  # noqa: E402


# ── helpers ───────────────────────────────────────────────────────────────────
def flat(n=40, price=100.0, vol=1000.0):
    return np.full(n, price), np.full(n, vol)


def with_step(c, i, factor):
    c = c.copy()
    c[i:] = c[i - 1] * factor
    return c


# ── nearest_split_ratio ───────────────────────────────────────────────────────
@pytest.mark.parametrize("gap,expected", [
    (0.5, ("2:1", 0.5)), (1 / 3, ("3:1", 1 / 3)), (2 / 3, ("3:2", 2 / 3)),
    (0.1, ("10:1", 0.1)), (2.0, ("1:2", 2.0)), (10.0, ("1:10", 10.0)),
    (0.66, ("3:2", 2 / 3)),
])
def test_nearest_split_ratio_hits(gap, expected):
    assert ca.nearest_split_ratio(gap) == expected


@pytest.mark.parametrize("gap", [1.0, 0.8, 0.0, -0.5, float("nan"), float("inf"), 7.0])
def test_nearest_split_ratio_misses(gap):
    assert ca.nearest_split_ratio(gap) is None


def test_nearest_split_ratio_tolerance_boundary():
    assert ca.nearest_split_ratio(0.514) == ("2:1", 0.5)         # 2.8% off
    assert ca.nearest_split_ratio(0.516) is None                 # 3.2% off
    assert ca.nearest_split_ratio(0.516, tol=0.04) == ("2:1", 0.5)


def test_nearest_split_ratio_picks_closest_when_tolerance_is_wide():
    assert ca.nearest_split_ratio(0.45, tol=0.5) == ("2:1", 0.5)
    assert ca.nearest_split_ratio(0.30, tol=0.5)[0] == "3:1"


# ── detect_splits ─────────────────────────────────────────────────────────────
def test_detect_splits_forward_split_with_volume_spike():
    c, v = flat()
    c = with_step(c, 25, 0.5)
    v[25] = 2000.0
    dates = [f"d{i}" for i in range(40)]
    out = ca.detect_splits(c, v, dates)
    assert out == [{"i": 25, "date": "d25", "ratio": "2:1", "gap": 0.5, "vol_x": 2.0,
                    "type": "forward"}]


def test_detect_splits_date_defaults_to_index():
    c, v = flat()
    c = with_step(c, 30, 0.25)
    v[30] = 5000
    ev = ca.detect_splits(c, v)[0]
    assert ev["date"] == 30 and ev["ratio"] == "4:1"


def test_detect_splits_forward_volume_threshold_inclusive():
    c, v = flat()
    c = with_step(c, 25, 0.5)
    v[25] = 1300.0                                  # exactly VOL_MULT * avg
    assert len(ca.detect_splits(c, v)) == 1
    v[25] = 1299.0
    assert ca.detect_splits(c, v) == []


def test_detect_splits_reverse_split_needs_volume_drop():
    c, v = flat()
    c = with_step(c, 25, 2.0)
    v[25] = 700.0                                   # <= avg/1.3 = 769.2
    out = ca.detect_splits(c, v)
    assert [(e["ratio"], e["type"], e["vol_x"]) for e in out] == [("1:2", "reverse", 0.7)]
    v[25] = 800.0
    assert ca.detect_splits(c, v) == []


def test_detect_splits_rejects_immediate_reversal_but_accepts_last_bar():
    c, v = flat()
    c[25:] = 50.0
    c[26:] = 100.0                                  # price snaps back next day
    v[25] = 3000.0
    assert ca.detect_splits(c, v) == []
    c2, v2 = flat(26)
    c2[25] = 50.0
    v2[25] = 3000.0
    assert len(ca.detect_splits(c2, v2)) == 1       # no next bar -> persistence assumed


def test_detect_splits_next_day_nan_counts_as_persisting():
    c, v = flat()
    c = with_step(c, 25, 0.5)
    c[26] = np.nan
    v[25] = 2000.0
    assert [e["i"] for e in ca.detect_splits(c, v)] == [25]


def test_detect_splits_no_event_in_first_21_bars_and_short_series():
    c, v = flat()
    c = with_step(c, 21, 0.5)
    v[21] = 5000
    assert [e["i"] for e in ca.detect_splits(c, v)] == [21]
    c = with_step(flat()[0], 20, 0.5)
    v = flat()[1]
    v[20] = 5000
    assert ca.detect_splits(c, v) == []             # scan starts at index 21
    assert ca.detect_splits([1.0] * 10, [1.0] * 10) == []
    assert ca.detect_splits([], []) == []


def test_detect_splits_glitch_guard_ignores_collapse_to_near_zero():
    c, v = flat()
    c = with_step(c, 25, 0.07)                      # below MIN_GAP
    v[25] = 9000
    assert ca.detect_splits(c, v, tol=0.35) == []   # would match 10:1 at this tolerance
    c = with_step(flat()[0], 25, 0.09)
    out = ca.detect_splits(c, v, tol=0.35)
    assert out and out[0]["ratio"] == "10:1"


def test_detect_splits_skips_nan_and_nonpositive_previous_close():
    c, v = flat()
    c[24] = np.nan
    c[25:] = 50.0
    v[25] = 2000
    assert ca.detect_splits(c, v) == []
    c, v = flat()
    c[24] = 0.0
    c[25:] = 50.0
    assert ca.detect_splits(c, v) == []


def test_detect_splits_zero_average_volume_gives_nan_vol_x():
    c, v = flat(vol=0.0)
    c = with_step(c, 25, 0.5)
    ev = ca.detect_splits(c, v)
    assert len(ev) == 1 and np.isnan(ev[0]["vol_x"])


def test_detect_splits_accepts_lists_and_custom_tolerance():
    c, v = flat()
    c = with_step(c, 25, 0.516)
    v[25] = 2000
    assert ca.detect_splits(list(c), list(v)) == []
    assert len(ca.detect_splits(list(c), list(v), tol=0.04)) == 1


# ── detect_rights ─────────────────────────────────────────────────────────────
def rights_case(drop=-0.15, vol_spike=2000.0, mkt_ret=0.0, after=1.0, n=40):
    c, v = flat(n)
    i = 25
    c = with_step(c, i, 1 + drop)
    c[i + 3:] = c[i] * after
    v[i] = vol_spike
    m = np.zeros(n)
    m[i] = mkt_ret
    return c, v, m


def test_detect_rights_basic_event_and_fields():
    c, v, m = rights_case()
    dates = [f"d{i}" for i in range(40)]
    out = ca.detect_rights(c, v, m, dates)
    assert out == [{"i": 25, "date": "d25", "drop%": -15.0, "vs_mkt%": -15.0, "vol_x": 2.0}]
    assert ca.detect_rights(c, v, m)[0]["date"] == 25


def test_detect_rights_band_edges():
    for drop, expect in [(-0.09, True), (-0.07, False), (-0.30, True), (-0.36, False), (0.10, False)]:
        c, v, m = rights_case(drop=drop)
        assert bool(ca.detect_rights(c, v, m)) is expect, drop


def test_detect_rights_excludes_clean_split_ratio_inside_band():
    c, v, m = rights_case(drop=-0.33)               # 0.67 ~ 3:2 split
    assert ca.detect_rights(c, v, m) == []


def test_detect_rights_requires_idiosyncratic_drop():
    c, v, m = rights_case(drop=-0.15, mkt_ret=-0.10)    # market margin 0.05 < 0.08
    assert ca.detect_rights(c, v, m) == []
    c, v, m = rights_case(drop=-0.15, mkt_ret=-0.06)    # margin 0.09
    assert len(ca.detect_rights(c, v, m)) == 1


def test_detect_rights_market_series_short_or_nan_treated_as_zero():
    c, v, m = rights_case()
    assert len(ca.detect_rights(c, v, m[:10])) == 1
    m2 = m.copy()
    m2[25] = np.nan
    assert len(ca.detect_rights(c, v, m2)) == 1


def test_detect_rights_volume_gate():
    c, v, m = rights_case(vol_spike=1299.0)
    assert ca.detect_rights(c, v, m) == []
    c, v, m = rights_case(vol_spike=1300.0)
    assert len(ca.detect_rights(c, v, m)) == 1
    c, v, m = rights_case()
    v[:25] = 0.0
    assert ca.detect_rights(c, v, m) == []              # zero average volume


def test_detect_rights_requires_stabilisation():
    c, v, m = rights_case(after=1.10)
    assert len(ca.detect_rights(c, v, m)) == 1          # 10% drift: still stable
    c, v, m = rights_case(after=1.20)
    assert ca.detect_rights(c, v, m) == []              # bouncing back
    c, v, m = rights_case(after=0.80)
    assert ca.detect_rights(c, v, m) == []              # still falling
    c, v, m = rights_case()
    c[28] = np.nan
    assert ca.detect_rights(c, v, m) == []


def test_detect_rights_last_three_bars_and_nan_prev_excluded():
    c, v, m = rights_case(n=29)                         # i=25 -> i+3=28 is last index
    assert len(ca.detect_rights(c, v, m)) == 1
    c, v, m = rights_case(n=28)                         # not enough lookahead
    assert ca.detect_rights(c, v, m) == []
    c, v, m = rights_case()
    c[24] = np.nan
    assert ca.detect_rights(c, v, m) == []


# ── label_ratio ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("factor,label", [
    (10.0, "10:1"), (2.0, "2:1"), (1.0, "1:1"), (1.5, "1.50:1"), (3.04, "3:1"),
    (3.06, "3.06:1"), (0.5, "1:2"), (0.1, "1:10"), (0.4, "1:2.50"),
    (1 / 3, "1:3"),
])
def test_label_ratio(factor, label):
    assert ca.label_ratio(factor) == label


# ── confirm_split_yf ──────────────────────────────────────────────────────────
@pytest.fixture
def fake_yf(monkeypatch):
    mod = types.ModuleType("yfinance")
    state = types.SimpleNamespace(splits=pd.Series(dtype=float), tickers=[], raises=None)

    class Ticker:
        def __init__(self, t):
            state.tickers.append(t)

        @property
        def splits(self):
            if state.raises:
                raise state.raises
            return state.splits
    mod.Ticker = Ticker
    monkeypatch.setitem(sys.modules, "yfinance", mod)
    seen = []

    def robust(src, fn, *a, **k):
        seen.append((src, k))
        return fn(*a)
    monkeypatch.setattr(apiclient, "robust", robust)
    state.robust_calls = seen
    return state


def test_confirm_split_yf_returns_last_five_as_date_factor(fake_yf):
    idx = pd.to_datetime([f"2020-0{m}-15" for m in range(1, 8)])
    fake_yf.splits = pd.Series([2, 3, 4, 5, 6, 7, 8], index=idx, dtype=float)
    out = ca.confirm_split_yf("AAPL")
    assert out == [("2020-03-15", 4.0), ("2020-04-15", 5.0), ("2020-05-15", 6.0),
                   ("2020-06-15", 7.0), ("2020-07-15", 8.0)]
    assert fake_yf.tickers == ["AAPL"]
    assert fake_yf.robust_calls == [("yfinance", {"retries": 2})]


def test_confirm_split_yf_empty_and_errors(fake_yf):
    assert ca.confirm_split_yf("X") == []
    fake_yf.raises = RuntimeError("down")
    assert ca.confirm_split_yf("X") == []


# ── parse_edgar_hits ──────────────────────────────────────────────────────────
def hit(names, forms=None, date="2025-03-01", **extra):
    src = {"display_names": names, "file_date": date}
    if forms is not None:
        src["root_forms"] = forms
    src.update(extra)
    return {"_source": src}


def test_parse_edgar_hits_extracts_company_ticker_date_form():
    p = {"hits": {"hits": [
        hit(["ACME CORP  (ACM, ACMX)  (CIK 0000123456)"], ["8-K"]),
        hit(["BERKSHIRE HATHAWAY INC  (BRK.B)  (CIK 0001067983)"], ["8-K/A"], date="2025-04-02"),
    ]}}
    rows = ca.parse_edgar_hits(p, "stock split")
    assert rows == [
        {"company": "ACME CORP", "ticker": "ACM", "date": "2025-03-01", "form": "8-K",
         "event": "stock split"},
        {"company": "BERKSHIRE HATHAWAY INC", "ticker": "BRK.B", "date": "2025-04-02",
         "form": "8-K/A", "event": "stock split"},
    ]


def test_parse_edgar_hits_missing_pieces():
    p = {"hits": {"hits": [
        {"_source": {}},
        {},
        hit([], None, form_type="8-K"),
        hit(["NO TICKER INC  (CIK 0000000001)"], []),
        hit(["lowercase co  (abc)  (CIK 1)"], "10-K", date=None),
    ]}}
    rows = ca.parse_edgar_hits(p)
    assert rows[0] == {"company": "", "ticker": "", "date": None, "form": "", "event": ""}
    assert rows[1] == rows[0]
    assert rows[2]["form"] == "8-K" and rows[2]["company"] == ""        # form_type string
    assert rows[3] == {"company": "NO TICKER INC", "ticker": "", "date": "2025-03-01",
                       "form": "", "event": ""}
    assert rows[4]["ticker"] == "" and rows[4]["form"] == "10-K" and rows[4]["date"] is None
    assert ca.parse_edgar_hits({}) == [] and ca.parse_edgar_hits({"hits": {}}) == []


# ── announcements_edgar ───────────────────────────────────────────────────────
class Resp:
    def __init__(self, code=200, payload=None):
        self.status_code = code
        self._p = payload or {}

    def json(self):
        return self._p


def edgar_payload(n=3):
    return {"hits": {"hits": [
        hit([f"CO{i} INC  (T{i})  (CIK 000000000{i})"], ["8-K"], date=f"2025-01-0{i + 1}")
        for i in range(n)]}}


def test_announcements_edgar_builds_url_headers_and_limits(monkeypatch):
    calls = []

    def http_get(source, url, headers=None, retries=None):
        calls.append((source, url, headers, retries))
        return Resp(200, edgar_payload(3))
    monkeypatch.setattr(apiclient, "http_get", http_get)
    monkeypatch.delenv("SEC_UA", raising=False)
    df = ca.announcements_edgar("forward stock split", forms="8-K", since="2024-05-01", limit=2)
    assert list(df["ticker"]) == ["T0", "T1"]
    assert list(df.columns) == ["company", "ticker", "date", "form", "event"]
    assert set(df["event"]) == {"forward stock split"}
    src, url, headers, retries = calls[0]
    assert src == "edgar" and retries == 2
    assert url == ("https://efts.sec.gov/LATEST/search-index?q=%22forward+stock+split%22"
                   "&forms=8-K&startdt=2024-05-01&enddt=2035-01-01")
    assert headers == {"User-Agent": "global-market-scanners research admin@gms.dev"}


def test_announcements_edgar_user_agent_override(monkeypatch):
    seen = {}
    monkeypatch.setenv("SEC_UA", "me me@example.org")
    monkeypatch.setattr(apiclient, "http_get",
                        lambda s, u, headers=None, retries=None: seen.update(h=headers) or Resp(200, {}))
    assert ca.announcements_edgar("x").empty
    assert seen["h"] == {"User-Agent": "me me@example.org"}


def test_announcements_edgar_non_200_and_exception(monkeypatch, capsys):
    monkeypatch.setattr(apiclient, "http_get", lambda *a, **k: Resp(403))
    assert ca.announcements_edgar("x").empty
    assert capsys.readouterr().err == ""

    def boom(*a, **k):
        raise RuntimeError("offline")
    monkeypatch.setattr(apiclient, "http_get", boom)
    assert ca.announcements_edgar("rights offering").empty
    assert capsys.readouterr().err == "  [edgar] 'rights offering' unavailable: offline\n"


# ── announcements ─────────────────────────────────────────────────────────────
def frame(rows):
    return pd.DataFrame(rows, columns=["company", "ticker", "date", "form", "event"])


def test_announcements_union_dedup_sorted_desc_and_kind_keywords(monkeypatch):
    seen = []
    data = {
        "forward stock split": frame([["A", "A", "2025-01-05", "8-K", "f"],
                                      ["B", "B", "2025-03-01", "8-K", "f"]]),
        "reverse stock split": frame([["A", "A", "2025-01-05", "8-K", "r"]]),   # dup of A
        "stock split": pd.DataFrame(),
    }

    def fake(k, since=None, limit=None):
        seen.append((k, since, limit))
        return data[k]
    monkeypatch.setattr(ca, "announcements_edgar", fake)
    df = ca.announcements("split", since="2024-01-01", limit=7)
    assert [s[0] for s in seen] == ca.SPLIT_KEYWORDS
    assert all(s[1:] == ("2024-01-01", 7) for s in seen)
    assert list(df["company"]) == ["B", "A"]
    assert df.set_index("company").loc["A", "event"] == "f"        # first occurrence kept


def test_announcements_rights_keywords_and_all_empty(monkeypatch):
    seen = []
    monkeypatch.setattr(ca, "announcements_edgar",
                        lambda k, since=None, limit=None: seen.append(k) or pd.DataFrame())
    assert ca.announcements("rights").empty
    assert seen == ca.RIGHTS_KEYWORDS == ["rights offering", "rights issue"]


# ── screeners over wide frames ────────────────────────────────────────────────
@pytest.fixture
def pead(monkeypatch):
    mod = types.ModuleType("pead_factor")
    mod._liquid_symbols = lambda close, vol: list(close.columns)
    monkeypatch.setitem(sys.modules, "pead_factor", mod)
    return mod


def wide_frames(n=40, extra=0):
    idx = pd.bdate_range("2024-01-01", periods=n)
    close = pd.DataFrame(100.0, index=idx, columns=["X", "Y"] + [f"F{i}" for i in range(extra)])
    vol = pd.DataFrame(1000.0, index=idx, columns=close.columns)
    return close, vol


def test_scan_splits_finds_event_per_symbol(monkeypatch, pead):
    close, vol = wide_frames()
    close.loc[close.index[25]:, "X"] = 50.0
    vol.iloc[25, 0] = 2500.0
    monkeypatch.setattr(ca, "_wide", lambda m: (close, vol))
    df = ca.scan_splits("US")
    assert list(df.columns) == ["market", "ticker", "date", "ratio", "gap", "vol_x", "type"]
    assert df.to_dict("records") == [{
        "market": "US", "ticker": "X", "date": str(close.index[25].date()), "ratio": "2:1",
        "gap": 0.5, "vol_x": 2.5, "type": "forward"}]


def test_scan_splits_no_events_is_empty_frame(monkeypatch, pead):
    close, vol = wide_frames()
    monkeypatch.setattr(ca, "_wide", lambda m: (close, vol))
    assert ca.scan_splits("US").empty


def test_scan_rights_uses_equal_weight_market_proxy(monkeypatch, pead):
    close, vol = wide_frames(extra=4)
    close.loc[close.index[25]:, "X"] = 85.0
    vol.iloc[25, 0] = 2000.0
    monkeypatch.setattr(ca, "_wide", lambda m: (close, vol))
    df = ca.scan_rights("US")
    rec = df.to_dict("records")
    assert rec == [{"market": "US", "ticker": "X", "date": str(close.index[25].date()),
                    "drop%": -15.0, "vs_mkt%": -12.5, "vol_x": 2.0}]


def test_scan_rights_two_name_market_is_not_idiosyncratic(monkeypatch, pead):
    close, vol = wide_frames()                      # X drags the 2-name proxy to -7.5%
    close.loc[close.index[25]:, "X"] = 85.0
    vol.iloc[25, 0] = 2000.0
    monkeypatch.setattr(ca, "_wide", lambda m: (close, vol))
    assert ca.scan_rights("US").empty


def test_scan_functions_return_empty_when_wide_is_none(monkeypatch, pead):
    monkeypatch.setattr(ca, "_wide", lambda m: None)
    assert ca.scan_splits("US").empty
    assert ca.scan_rights("US").empty
    assert ca.screen_splits_yf("US").empty


@pytest.mark.xfail(strict=True, raises=AttributeError,
                   reason="corporate_actions.py:~118 _wide is marketdata.close_volume, which returns "
                          "(None, None) not None for a missing market, so the `w is None` guard never fires")
def test_scan_splits_missing_market_returns_empty_frame(monkeypatch, pead):
    import marketdata
    monkeypatch.setattr(marketdata, "SEED", "/nonexistent-seed-dir")
    monkeypatch.setattr(marketdata, "SEED_LOCAL", "/nonexistent-local-dir")
    assert ca.scan_splits("ZZ").empty


@pytest.mark.xfail(strict=True, raises=AttributeError,
                   reason="corporate_actions.py:~136 same (None, None) vs None mismatch in scan_rights")
def test_scan_rights_missing_market_returns_empty_frame(monkeypatch, pead):
    import marketdata
    monkeypatch.setattr(marketdata, "SEED", "/nonexistent-seed-dir")
    monkeypatch.setattr(marketdata, "SEED_LOCAL", "/nonexistent-local-dir")
    assert ca.scan_rights("ZZ").empty


@pytest.mark.xfail(strict=True, raises=AttributeError,
                   reason="corporate_actions.py:~250 same (None, None) vs None mismatch in screen_splits_yf")
def test_screen_splits_yf_missing_market_returns_empty_frame(monkeypatch, pead):
    import marketdata
    monkeypatch.setattr(marketdata, "SEED", "/nonexistent-seed-dir")
    monkeypatch.setattr(marketdata, "SEED_LOCAL", "/nonexistent-local-dir")
    assert ca.screen_splits_yf("ZZ").empty


# ── screen_splits_yf ──────────────────────────────────────────────────────────
@pytest.fixture
def yf_screen(monkeypatch, pead):
    idx = pd.bdate_range("2024-01-01", periods=10)
    close = pd.DataFrame({"BIG": 100.0, "MID": 50.0, "SMALL": 1.0}, index=idx)
    vol = pd.DataFrame({"BIG": 1000.0, "MID": 1000.0, "SMALL": 1000.0}, index=idx)
    monkeypatch.setattr(ca, "_wide", lambda m: (close, vol))
    splits = {
        "BIG": [("2023-12-31", 2.0), ("2024-02-01", 4.0)],
        "MID": [("2024-01-01", 0.5), ("2024-03-01", 1.5)],
        "SMALL": [("2024-02-02", 10.0)],
    }
    asked = []

    def confirm(t):
        asked.append(t)
        return splits[t]
    monkeypatch.setattr(ca, "confirm_split_yf", confirm)
    return types.SimpleNamespace(asked=asked, close=close)


def test_screen_splits_yf_limits_to_most_liquid_and_filters_window(yf_screen):
    df = ca.screen_splits_yf("US", limit=2)
    assert yf_screen.asked == ["BIG", "MID"]                    # ranked by dollar volume
    assert df.to_dict("records") == [
        {"market": "US", "ticker": "BIG", "date": "2024-02-01", "ratio": "4:1",
         "factor": 4.0, "type": "forward"},                     # 2023-12-31 is before window
        {"market": "US", "ticker": "MID", "date": "2024-01-01", "ratio": "1:2",
         "factor": 0.5, "type": "reverse"},                     # == min_date is kept
        {"market": "US", "ticker": "MID", "date": "2024-03-01", "ratio": "1.50:1",
         "factor": 1.5, "type": "forward"},
    ]


def test_screen_splits_yf_explicit_min_date_and_factor_rounding(yf_screen, monkeypatch):
    monkeypatch.setattr(ca, "confirm_split_yf", lambda t: [("2024-03-01", 1.23456)])
    df = ca.screen_splits_yf("US", limit=1, min_date="2024-03-02")
    assert df.empty
    df = ca.screen_splits_yf("US", limit=1, min_date="2024-03-01")
    assert df["factor"].tolist() == [1.235]


# ── CLI ───────────────────────────────────────────────────────────────────────
def run_main(monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, "argv", ["corporate_actions.py", *argv])
    ca.main()
    return capsys.readouterr().out


def test_main_news_mode_prints_both_kinds(monkeypatch, capsys):
    asked = []

    def ann(kind, since=None, limit=None):
        asked.append((kind, since, limit))
        if kind == "rights":
            return pd.DataFrame()
        return pd.DataFrame([{"company": "A VERY LONG COMPANY NAME THAT EXCEEDS THIRTY EIGHT", "ticker": "",
                              "date": "2025-02-01", "form": "8-K", "event": "stock split"},
                             {"company": "Beta", "ticker": "BET", "date": "2025-01-01",
                              "form": "8-K", "event": "stock split"}])
    monkeypatch.setattr(ca, "announcements", ann)
    out = run_main(monkeypatch, capsys, "--news", "--action", "both", "--since", "2024-06-01", "--top", "1")
    assert asked == [("split", "2024-06-01", 1), ("rights", "2024-06-01", 1)]
    assert "=== STOCK SPLIT — SEC EDGAR public announcements since 2024-06-01 (2 filings) ===" in out
    assert "  —         2025-02-01  8-K      A VERY LONG COMPANY NAME THAT EXCEEDS   ·  stock split" in out
    assert "Beta" not in out                                     # --top 1
    assert "=== RIGHTS ISSUE — SEC EDGAR public announcements since 2024-06-01 (0 filings) ===" in out
    assert "  none found (offline, or no matching 8-K filings)" in out


def test_main_news_mode_respects_action(monkeypatch, capsys):
    kinds = []
    monkeypatch.setattr(ca, "announcements",
                        lambda kind, since=None, limit=None: kinds.append(kind) or pd.DataFrame())
    run_main(monkeypatch, capsys, "--news", "--action", "rights")
    run_main(monkeypatch, capsys, "--news")                      # default action split
    assert kinds == ["rights", "split"]


SPLIT_ROWS = pd.DataFrame([
    {"market": "US", "ticker": "AAA", "date": "2024-01-02", "ratio": "2:1", "gap": 0.5,
     "vol_x": 2.5, "type": "forward"},
    {"market": "US", "ticker": "BBB", "date": "2024-02-03", "ratio": "1:2", "gap": 2.0,
     "vol_x": 0.7, "type": "reverse"}])


def test_main_heuristic_split_report(monkeypatch, capsys):
    got = []
    monkeypatch.setattr(ca, "scan_splits", lambda m: got.append(m) or SPLIT_ROWS)
    out = run_main(monkeypatch, capsys, "--market", "US", "--heuristic")
    assert got == ["US"]
    assert "=== STOCK SPLIT SCREENER (offline heuristic) — 2 candidate events ===" in out
    assert "  US  AAA         2024-01-02      2:1   0.500   2.50  forward" in out
    assert "  US  BBB         2024-02-03      1:2   2.000   0.70  reverse" in out


def test_main_heuristic_split_empty_prints_only_header(monkeypatch, capsys):
    monkeypatch.setattr(ca, "scan_splits", lambda m: pd.DataFrame())
    out = run_main(monkeypatch, capsys, "--market", "US", "--heuristic")
    assert "0 candidate events" in out and "ticker" not in out


def test_main_yfinance_split_report_sorted_newest_first_and_limit_passed(monkeypatch, capsys):
    seen = []
    rows = pd.DataFrame([{"market": "US", "ticker": "OLD", "date": "2024-01-02", "ratio": "2:1",
                          "factor": 2.0, "type": "forward"},
                         {"market": "US", "ticker": "NEW", "date": "2024-05-02", "ratio": "1:2",
                          "factor": 0.5, "type": "reverse"}])
    monkeypatch.setattr(ca, "screen_splits_yf", lambda m, limit: seen.append((m, limit)) or rows)
    out = run_main(monkeypatch, capsys, "--market", "US", "--limit", "12", "--top", "1")
    assert seen == [("US", 12)]
    assert "2 splits in window" in out and "top-12 liquid names per market" in out
    assert "NEW" in out and "OLD" not in out                     # newest first, truncated to --top


def test_main_yfinance_split_empty_message(monkeypatch, capsys):
    monkeypatch.setattr(ca, "screen_splits_yf", lambda m, limit: pd.DataFrame())
    out = run_main(monkeypatch, capsys, "--market", "JP")
    assert "0 splits in window" in out
    assert "no splits recorded for the queried names in this window (offline? try --heuristic)" in out


def test_main_rights_report_sorted_by_deepest_drop(monkeypatch, capsys):
    rows = pd.DataFrame([
        {"market": "US", "ticker": "SHALLOW", "date": "2024-01-02", "drop%": -9.0, "vs_mkt%": -8.0, "vol_x": 1.5},
        {"market": "US", "ticker": "DEEP", "date": "2024-02-02", "drop%": -25.5, "vs_mkt%": -20.1, "vol_x": 3.25},
    ])
    monkeypatch.setattr(ca, "scan_rights", lambda m: rows)
    out = run_main(monkeypatch, capsys, "--action", "rights", "--market", "US", "--top", "5")
    assert "2 candidate ex-dates ===" in out
    assert "HONEST CAVEAT" in out
    assert out.index("DEEP") < out.index("SHALLOW")
    assert "  US  DEEP        2024-02-02     -25.5    -20.1   3.25" in out


def test_main_rights_empty_prints_no_table(monkeypatch, capsys):
    monkeypatch.setattr(ca, "scan_rights", lambda m: pd.DataFrame())
    out = run_main(monkeypatch, capsys, "--action", "rights", "--market", "US")
    assert "0 candidate ex-dates" in out and "vs_mkt%" not in out


def test_main_both_runs_split_then_rights(monkeypatch, capsys):
    monkeypatch.setattr(ca, "screen_splits_yf", lambda m, limit: pd.DataFrame())
    monkeypatch.setattr(ca, "scan_rights", lambda m: pd.DataFrame())
    out = run_main(monkeypatch, capsys, "--action", "both", "--market", "US")
    assert out.index("STOCK SPLIT SCREENER") < out.index("RIGHTS-ISSUE")


@pytest.mark.xfail(strict=True, raises=NameError,
                   reason="corporate_actions.py:~323 main() calls marketdata.market_list() but only "
                          "`from marketdata import close_volume` is imported, so --all / no --market "
                          "raises NameError")
def test_main_without_market_enumerates_all_markets(monkeypatch, capsys):
    import marketdata
    monkeypatch.setattr(marketdata, "market_list", lambda: ["US", "JP"])
    got = []
    monkeypatch.setattr(ca, "scan_splits", lambda m: got.append(m) or pd.DataFrame())
    run_main(monkeypatch, capsys, "--heuristic", "--all")
    assert got == ["US", "JP"]
