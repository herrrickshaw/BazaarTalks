"""Offline tests for pit_fundamentals.py (point-in-time EDGAR fundamentals).

The module opens ./edgar_facts.db at import time, so each test imports a private copy with
sqlite3.connect redirected to an in-memory database; nothing under the repo is touched.
"""
import importlib.util
import json
import os
import sqlite3
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture
def pf(tmp_path, monkeypatch):
    real_connect = sqlite3.connect
    with monkeypatch.context() as m:
        m.setattr(sqlite3, "connect", lambda *a, **k: real_connect(":memory:"))
        spec = importlib.util.spec_from_file_location(
            "pit_fundamentals", os.path.join(ROOT, "pit_fundamentals.py"))
        mod = importlib.util.module_from_spec(spec)
        m.setitem(sys.modules, "pit_fundamentals", mod)
        spec.loader.exec_module(mod)
    monkeypatch.setitem(sys.modules, "pit_fundamentals", mod)
    monkeypatch.setattr(mod, "CACHE_DIR", str(tmp_path / "edgar_cache"))
    yield mod
    mod._conn.close()


# ── fact builders ─────────────────────────────────────────────────────────────
def fy(end, filed, val, start="auto", form="10-K", fp="FY"):
    """One companyfacts record. start='auto' -> ~365d before end."""
    from datetime import date, timedelta
    if start == "auto":
        start = str(date.fromisoformat(end) - timedelta(days=364))
    r = {"end": end, "filed": filed, "val": val, "form": form, "fp": fp}
    if start is not None:
        r["start"] = start
    return r


def node(*recs, unit="USD"):
    return {"units": {unit: list(recs)}}


def instant(end, filed, val):
    return fy(end, filed, val, start=None)


# ── _prune / _store / _read ───────────────────────────────────────────────────
def test_prune_keeps_needed_concepts_and_only_annual_10k(pf):
    facts = {"us-gaap": {
        "NetIncomeLoss": {"units": {"USD": [
            fy("2020-12-31", "2021-02-01", 10), fy("2020-09-30", "2020-11-01", 3, form="10-Q", fp="Q3"),
            fy("2020-12-31", "2021-02-01", 11, fp="Q4")]}},
        "Assets": {"units": {"USD": [fy("2020-12-31", "2021-02-01", 99, form="10-Q")]}},   # all dropped
        "SomeOtherConcept": {"units": {"USD": [fy("2020-12-31", "2021-02-01", 1)]}},       # not needed
    }}
    out = pf._prune(facts)
    assert set(out["us-gaap"]) == {"NetIncomeLoss"}
    assert [r["val"] for r in out["us-gaap"]["NetIncomeLoss"]["units"]["USD"]] == [10]
    assert pf._prune({}) == {"us-gaap": {}}
    assert pf._prune({"us-gaap": {"Assets": {}}}) == {"us-gaap": {}}


def test_store_and_read_roundtrip_marks_records_as_10k_fy(pf):
    facts = {"us-gaap": {
        "NetIncomeLoss": node(fy("2020-12-31", "2021-02-01", 10.0), fy("2021-12-31", "2022-02-01", 12.0)),
        "Assets": {"units": {"USD": [instant("2021-12-31", "2022-02-01", 500.0)],
                             "EUR": [instant("2021-12-31", "2022-02-02", 450.0)]}},
    }}
    assert pf._read("0000000001") is None
    pf._store("0000000001", facts)
    back = pf._read("0000000001")
    ni = back["us-gaap"]["NetIncomeLoss"]["units"]["USD"]
    assert sorted(r["val"] for r in ni) == [10.0, 12.0]
    assert all(r["form"] == "10-K" and r["fp"] == "FY" for r in ni)
    assert set(back["us-gaap"]["Assets"]["units"]) == {"USD", "EUR"}
    assert back["us-gaap"]["Assets"]["units"]["USD"][0]["start"] is None
    assert pf._read("0000000002") is None


def test_store_empty_facts_still_marks_fetched(pf):
    pf._store("0000000009", {"us-gaap": {}})
    assert pf._read("0000000009") == {"us-gaap": {}}


# ── _annual_asof ──────────────────────────────────────────────────────────────
def test_annual_asof_newest_first_and_limit(pf):
    f = {"us-gaap": {"NetIncomeLoss": node(
        *[fy(f"{y}-12-31", f"{y + 1}-02-15", float(y)) for y in range(2015, 2022)])}}
    assert pf._annual_asof(f, "NetIncomeLoss", "2030-01-01") == [2021.0, 2020.0, 2019.0, 2018.0, 2017.0]
    assert pf._annual_asof(f, "NetIncomeLoss", "2030-01-01", n=2) == [2021.0, 2020.0]
    assert pf._annual_asof(f, "NetIncomeLoss", "2030-01-01", n=0) == []


def test_annual_asof_filed_date_boundary_is_inclusive(pf):
    f = {"us-gaap": {"NetIncomeLoss": node(fy("2020-12-31", "2021-02-15", 7.0))}}
    assert pf._annual_asof(f, "NetIncomeLoss", "2021-02-15") == [7.0]
    assert pf._annual_asof(f, "NetIncomeLoss", "2021-02-14") == []


def test_annual_asof_uses_latest_restatement_public_by_asof(pf):
    f = {"us-gaap": {"NetIncomeLoss": node(
        fy("2020-12-31", "2021-02-15", 100.0),
        fy("2020-12-31", "2022-02-15", 90.0),               # restated a year later
        fy("2020-12-31", "2021-11-01", 95.0))}}
    assert pf._annual_asof(f, "NetIncomeLoss", "2021-03-01") == [100.0]
    assert pf._annual_asof(f, "NetIncomeLoss", "2021-12-01") == [95.0]
    assert pf._annual_asof(f, "NetIncomeLoss", "2022-02-15") == [90.0]


def test_annual_asof_duration_window_350_to_385_days(pf):
    def one(days):
        from datetime import date, timedelta
        end = date(2020, 12, 31)
        r = fy(str(end), "2021-02-01", 1.0, start=str(end - timedelta(days=days)))
        return pf._annual_asof({"us-gaap": {"X": node(r)}}, "X", "2022-01-01")
    assert one(349) == [] and one(350) == [1.0]
    assert one(385) == [1.0] and one(386) == []
    assert one(90) == []                                    # quarterly frame mis-tagged FY


def test_annual_asof_instant_concepts_have_no_duration_check(pf):
    f = {"us-gaap": {"Assets": node(instant("2020-12-31", "2021-02-01", 5.0))}}
    assert pf._annual_asof(f, "Assets", "2021-06-01") == [5.0]


def test_annual_asof_skips_bad_records(pf):
    f = {"us-gaap": {"X": node(
        fy("2020-12-31", "2021-02-01", None),                      # null value
        fy("2019-12-31", "", 1.0),                                 # no filed date
        fy("", "2021-02-01", 1.0, start=None),                     # no end
        fy("2018-12-31", "2019-02-01", 1.0, form="10-Q"),
        fy("2017-12-31", "2018-02-01", 1.0, fp="Q1"),
        fy("2016-12-31", "2017-02-01", 0.0))}}                     # zero is a real value
    assert pf._annual_asof(f, "X", "2022-01-01") == [0.0]


def test_annual_asof_missing_concept_and_units_merge(pf):
    assert pf._annual_asof({}, "X", "2022-01-01") == []
    assert pf._annual_asof({"us-gaap": {}}, "X", "2022-01-01") == []
    f = {"us-gaap": {"X": {"units": {
        "USD": [fy("2020-12-31", "2021-02-01", 1.0)],
        "shares": [fy("2019-12-31", "2020-02-01", 2.0)]}}}}
    assert pf._annual_asof(f, "X", "2022-01-01") == [1.0, 2.0]


def test_annual_asof_equal_filed_dates_keep_first_seen(pf):
    f = {"us-gaap": {"X": node(fy("2020-12-31", "2021-02-01", 1.0), fy("2020-12-31", "2021-02-01", 2.0))}}
    assert pf._annual_asof(f, "X", "2022-01-01") == [1.0]


# ── tiny helpers ──────────────────────────────────────────────────────────────
def test_first_ratio_and_getter_helpers(pf):
    f = {"us-gaap": {"B": node(fy("2020-12-31", "2021-02-01", 4.0))}}
    assert pf._first(f, "2022-01-01", "A", "B") == [4.0]          # falls through missing concept
    assert pf._first(f, "2022-01-01", "A") == []
    assert pf._first(f, "2020-01-01", "B") == []                  # nothing known yet
    assert pf._r(6, 3) == 2.0 and pf._r(0, 5) == 0.0
    assert pf._r(None, 5) is None and pf._r(5, None) is None and pf._r(5, 0) is None
    assert pf._g([1, 2, 3]) == 1 and pf._g([1, 2, 3], 2) == 3 and pf._g([1], 1) is None
    assert pf._g([], 0) is None and pf._g(None) is None and pf._g([1, 2, 3], -1) == 3
    assert pf._d("2020-02-29").isoformat() == "2020-02-29"


# ── _ticker_cik ───────────────────────────────────────────────────────────────
def test_ticker_cik_maps_upper_and_zero_pads_and_is_cached(pf, monkeypatch):
    calls = []

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"0": {"ticker": "aapl", "cik_str": 320193}, "1": {"ticker": "BRK-B", "cik_str": 1067983}}

    def get(url, headers=None, timeout=None):
        calls.append((url, headers, timeout))
        return R()
    monkeypatch.setattr(pf.requests, "get", get)
    pf._ticker_cik.cache_clear()
    m = pf._ticker_cik()
    assert m == {"AAPL": "0000320193", "BRK-B": "0001067983"}
    assert pf._ticker_cik() is m and len(calls) == 1
    assert calls[0][0] == "https://www.sec.gov/files/company_tickers.json"
    assert "User-Agent" in calls[0][1] and calls[0][2] == 30


def test_ticker_cik_http_error_propagates(pf, monkeypatch):
    class R:
        def raise_for_status(self):
            raise RuntimeError("403")
    monkeypatch.setattr(pf.requests, "get", lambda *a, **k: R())
    pf._ticker_cik.cache_clear()
    with pytest.raises(RuntimeError, match="403"):
        pf._ticker_cik()


# ── _migrate_json ─────────────────────────────────────────────────────────────
def test_migrate_json_imports_pruned_legacy_files_once(pf, tmp_path):
    cache = tmp_path / "edgar_cache"
    cache.mkdir()
    facts = {"us-gaap": {
        "NetIncomeLoss": node(fy("2020-12-31", "2021-02-01", 5.0), fy("2020-09-30", "2020-11-01", 1.0, form="10-Q", fp="Q3")),
        "Junk": node(fy("2020-12-31", "2021-02-01", 1.0))}}
    (cache / "CIK0000000042.json").write_text(json.dumps(facts))
    (cache / "CIK0000000043.json").write_text("{broken")
    (cache / "notes.txt").write_text("ignore")
    pf._migrate_json()
    assert pf._MIGRATED is True
    got = pf._read("0000000042")
    assert list(got["us-gaap"]) == ["NetIncomeLoss"]
    assert [r["val"] for r in got["us-gaap"]["NetIncomeLoss"]["units"]["USD"]] == [5.0]
    assert pf._read("0000000043") is None                       # bad JSON skipped silently
    rows_before = pf._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    pf._migrate_json()                                           # already fetched -> not re-imported
    assert pf._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == rows_before


def test_migrate_json_without_cache_dir_is_noop(pf):
    pf._migrate_json()
    assert pf._MIGRATED is True
    assert pf._conn.execute("SELECT COUNT(*) FROM fetched").fetchone()[0] == 0


# ── _load_facts ───────────────────────────────────────────────────────────────
@pytest.fixture
def net(pf, monkeypatch):
    import apiclient
    pf._MIGRATED = True
    pf._ticker_cik.cache_clear()
    monkeypatch.setattr(pf, "_ticker_cik", lambda: {"AAA": "0000000001"})
    st = types.SimpleNamespace(calls=[], resp=None, exc=None)

    def http_get(source, url, headers=None, **kw):
        st.calls.append((source, url, headers))
        if st.exc:
            raise st.exc
        return st.resp
    monkeypatch.setattr(apiclient, "http_get", http_get)
    return st


class Resp:
    def __init__(self, code, payload=None):
        self.status_code = code
        self._p = payload

    def json(self):
        return self._p


def test_load_facts_unknown_ticker_is_none_and_no_network(pf, net):
    assert pf._load_facts("ZZZ") is None
    assert net.calls == []


def test_load_facts_downloads_prunes_stores_then_serves_from_sqlite(pf, net):
    net.resp = Resp(200, {"facts": {"us-gaap": {
        "NetIncomeLoss": node(fy("2020-12-31", "2021-02-01", 5.0), fy("2020-09-30", "2020-11-01", 1.0, form="10-Q", fp="Q3")),
        "Junk": node(fy("2020-12-31", "2021-02-01", 1.0))}}})
    f = pf._load_facts("aaa")
    assert list(f["us-gaap"]) == ["NetIncomeLoss"]
    assert [r["val"] for r in f["us-gaap"]["NetIncomeLoss"]["units"]["USD"]] == [5.0]
    src, url, headers = net.calls[0]
    assert src == "edgar" and url == "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json"
    assert "User-Agent" in headers
    pf._load_facts.cache_clear()
    net.exc = RuntimeError("must not hit network again")
    again = pf._load_facts("AAA")                                # now from SQLite
    assert [r["val"] for r in again["us-gaap"]["NetIncomeLoss"]["units"]["USD"]] == [5.0]
    assert len(net.calls) == 1


def test_load_facts_non_200_negative_caches(pf, net):
    net.resp = Resp(404)
    assert pf._load_facts("AAA") == {"us-gaap": {}}
    pf._load_facts.cache_clear()
    net.exc = RuntimeError("no more network")
    assert pf._load_facts("AAA") == {"us-gaap": {}}              # remembered as fetched
    assert len(net.calls) == 1


def test_load_facts_network_exception_returns_none_and_does_not_cache(pf, net):
    net.exc = RuntimeError("timeout")
    assert pf._load_facts("AAA") is None
    assert pf._read("0000000001") is None


def test_load_facts_triggers_one_time_legacy_migration(pf, net, tmp_path):
    pf._MIGRATED = False
    cache = tmp_path / "edgar_cache"
    cache.mkdir()
    (cache / "CIK0000000001.json").write_text(json.dumps(
        {"us-gaap": {"Assets": node(fy("2020-12-31", "2021-02-01", 8.0, start=None))}}))
    f = pf._load_facts("AAA")
    assert pf._MIGRATED is True and net.calls == []
    assert f["us-gaap"]["Assets"]["units"]["USD"][0]["val"] == 8.0


# ── as_of ─────────────────────────────────────────────────────────────────────
def rich_facts():
    """Three fiscal years of every concept, newest FY2022 (filed 2023-02)."""
    def series(vals, durations=True):
        recs = []
        for y, v in zip((2022, 2021, 2020), vals):
            recs.append(fy(f"{y}-12-31", f"{y + 1}-02-15", v) if durations
                        else instant(f"{y}-12-31", f"{y + 1}-02-15", v))
        return node(*recs)
    return {"us-gaap": {
        "NetIncomeLoss": series([120.0, 100.0, 60.0]),
        "Revenues": series([1200.0, 1000.0, 800.0]),
        "Assets": series([1000.0, 1000.0, 1000.0], False),
        "Liabilities": series([400.0, 500.0, 600.0], False),
        "StockholdersEquity": series([600.0, 500.0, 400.0], False),
        "AssetsCurrent": series([300.0, 200.0, 200.0], False),
        "LiabilitiesCurrent": series([100.0, 100.0, 100.0], False),
        "GrossProfit": series([480.0, 350.0, 240.0]),
        "NetCashProvidedByUsedInOperatingActivities": series([150.0, 110.0, 70.0]),
        "CommonStockSharesOutstanding": series([100.0, 100.0, 100.0], False),
        "PaymentsToAcquirePropertyPlantAndEquipment": series([30.0, 25.0, 20.0]),
    }}


@pytest.fixture
def rich(pf, monkeypatch):
    facts = rich_facts()
    monkeypatch.setattr(pf, "_load_facts", lambda t: facts)
    return facts


def test_as_of_returns_empty_dict_without_facts(pf, monkeypatch):
    monkeypatch.setattr(pf, "_load_facts", lambda t: None)
    assert pf.as_of("X", "2024-01-01") == {}
    monkeypatch.setattr(pf, "_load_facts", lambda t: {})
    assert pf.as_of("X", "2024-01-01") == {}


def test_as_of_computes_ratios_from_latest_two_years(pf, rich):
    d = pf.as_of("X", "2023-06-01")
    assert d["ni"] == [120.0, 100.0, 60.0] and d["assets"] == [1000.0] * 3
    assert d["cfo"] == 150.0 and d["capex"] == 30.0
    assert d["roa"] == pytest.approx(0.12) and d["roa_prev"] == pytest.approx(0.10)
    assert d["cur"] == pytest.approx(3.0) and d["cur_prev"] == pytest.approx(2.0)
    assert d["lev"] == pytest.approx(0.4) and d["lev_prev"] == pytest.approx(0.5)
    assert d["gm"] == pytest.approx(0.4) and d["gm_prev"] == pytest.approx(0.35)
    assert d["at"] == pytest.approx(1.2) and d["at_prev"] == pytest.approx(1.0)
    assert d["shares"] == 100.0 and d["shares_prev"] == 100.0
    assert d["de"] == pytest.approx(400 / 600)
    assert d["roe_hist"] == pytest.approx([20.0, 20.0, 15.0])


def test_as_of_is_point_in_time_future_filings_are_invisible(pf, rich):
    d = pf.as_of("X", "2022-06-01")                       # FY2022 not filed until 2023-02-15
    assert d["ni"] == [100.0, 60.0] and d["cfo"] == 110.0
    d2 = pf.as_of("X", "2023-02-15")
    assert d2["ni"][0] == 120.0
    assert pf.as_of("X", "2023-02-14")["ni"][0] == 100.0
    early = pf.as_of("X", "2020-01-01")
    assert early["ni"] == [] and early["roa"] is None and early["cfo"] is None
    assert early["roe_hist"] == []


def test_as_of_revenue_and_cfo_fall_back_to_alternate_concepts(pf, monkeypatch):
    f = rich_facts()
    f["us-gaap"]["SalesRevenueNet"] = f["us-gaap"].pop("Revenues")
    f["us-gaap"]["NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"] = \
        f["us-gaap"].pop("NetCashProvidedByUsedInOperatingActivities")
    f["us-gaap"]["WeightedAverageNumberOfSharesOutstandingBasic"] = f["us-gaap"].pop("CommonStockSharesOutstanding")
    monkeypatch.setattr(pf, "_load_facts", lambda t: f)
    d = pf.as_of("X", "2023-06-01")
    assert d["rev"] == [1200.0, 1000.0, 800.0] and d["cfo"] == 150.0 and d["shares"] == 100.0


def test_as_of_zero_equity_skipped_in_roe_history(pf, monkeypatch):
    f = rich_facts()
    f["us-gaap"]["StockholdersEquity"] = node(
        instant("2022-12-31", "2023-02-15", 0.0), instant("2021-12-31", "2022-02-15", 500.0),
        instant("2020-12-31", "2021-02-15", 400.0))
    monkeypatch.setattr(pf, "_load_facts", lambda t: f)
    d = pf.as_of("X", "2023-06-01")
    assert d["roe_hist"] == pytest.approx([20.0, 15.0])
    assert d["de"] is None                                       # division by zero equity guarded


# ── piotroski_asof ────────────────────────────────────────────────────────────
ALL_TESTS = {"roa_pos", "cfo_pos", "d_roa", "accrual", "d_lev", "d_cur", "no_dilute", "d_margin", "d_turn"}


def test_piotroski_perfect_score(pf, rich):
    s, det = pf.piotroski_asof("X", "2023-06-01")
    assert s == 9 and set(det) == ALL_TESTS and all(det.values())


def test_piotroski_none_when_insufficient_data(pf, monkeypatch):
    monkeypatch.setattr(pf, "_load_facts", lambda t: None)
    assert pf.piotroski_asof("X", "2023-01-01") == (None, {})
    f = rich_facts()
    del f["us-gaap"]["Assets"]
    monkeypatch.setattr(pf, "_load_facts", lambda t: f)
    assert pf.piotroski_asof("X", "2023-06-01") == (None, {})
    f = rich_facts()
    del f["us-gaap"]["NetIncomeLoss"]
    monkeypatch.setattr(pf, "_load_facts", lambda t: f)
    assert pf.piotroski_asof("X", "2023-06-01") == (None, {})


def _score_with(pf, monkeypatch, mutate):
    f = rich_facts()
    mutate(f["us-gaap"])
    monkeypatch.setattr(pf, "_load_facts", lambda t: f)
    return pf.piotroski_asof("X", "2023-06-01")


def test_piotroski_each_test_flips_individually(pf, monkeypatch):
    def setc(g, concept, vals, instantaneous=True):
        recs = [(instant if instantaneous else fy)(f"{y}-12-31", f"{y + 1}-02-15", v)
                for y, v in zip((2022, 2021, 2020), vals)]
        g[concept] = node(*recs)

    s, det = _score_with(pf, monkeypatch, lambda g: setc(g, "NetIncomeLoss", [-5.0, 100.0, 60.0], False))
    assert det["roa_pos"] is False and det["d_roa"] is False and det["accrual"] is True and s == 7
    s, det = _score_with(pf, monkeypatch, lambda g: setc(g, "NetCashProvidedByUsedInOperatingActivities", [-1.0, 110.0, 70.0], False))
    assert det["cfo_pos"] is False and det["accrual"] is False
    # accrual is strict: CFO == NI fails
    s, det = _score_with(pf, monkeypatch, lambda g: setc(g, "NetCashProvidedByUsedInOperatingActivities", [120.0, 110.0, 70.0], False))
    assert det["accrual"] is False and det["cfo_pos"] is True
    s, det = _score_with(pf, monkeypatch, lambda g: setc(g, "Liabilities", [500.0, 500.0, 600.0]))
    assert det["d_lev"] is False                                  # equal leverage is not an improvement
    s, det = _score_with(pf, monkeypatch, lambda g: setc(g, "AssetsCurrent", [200.0, 200.0, 200.0]))
    assert det["d_cur"] is False
    s, det = _score_with(pf, monkeypatch, lambda g: setc(g, "GrossProfit", [350.0, 350.0, 240.0], False))
    assert det["d_margin"] is False or det["d_margin"] is True    # 350/1200 < 350/1000
    assert det["d_margin"] is False
    s, det = _score_with(pf, monkeypatch, lambda g: setc(g, "Revenues", [1000.0, 1000.0, 800.0], False))
    assert det["d_turn"] is False


def test_piotroski_dilution_tolerance_is_one_percent(pf, monkeypatch):
    def shares(new):
        def m(g):
            g["CommonStockSharesOutstanding"] = node(
                instant("2022-12-31", "2023-02-15", new), instant("2021-12-31", "2022-02-15", 100.0),
                instant("2020-12-31", "2021-02-15", 100.0))
        return m
    assert _score_with(pf, monkeypatch, shares(101.0))[1]["no_dilute"] is True      # == prev*1.01
    assert _score_with(pf, monkeypatch, shares(101.5))[1]["no_dilute"] is False
    assert _score_with(pf, monkeypatch, shares(90.0))[1]["no_dilute"] is True       # buyback


def test_piotroski_missing_inputs_fail_their_tests_not_crash(pf, monkeypatch):
    def drop(g):
        for c in ("CommonStockSharesOutstanding", "GrossProfit", "LiabilitiesCurrent",
                  "NetCashProvidedByUsedInOperatingActivities"):
            del g[c]
    s, det = _score_with(pf, monkeypatch, drop)
    assert det["no_dilute"] is False and det["d_margin"] is False and det["d_cur"] is False
    assert det["cfo_pos"] is False and det["accrual"] is False
    assert s == 4 and all(isinstance(v, bool) for v in det.values())


# ── coffeecan_asof ────────────────────────────────────────────────────────────
def test_coffeecan_passes_for_quality_compounder(pf, rich):
    ok, det = pf.coffeecan_asof("X", "2023-06-01", mktcap=5e9)
    assert ok is True
    assert det == {"avg_roe>15": True, "earnings_all+": True, "fcf_pos": True,
                   "rev_growth": True, "leverage_ok": True, "cap>1B": True}


def test_coffeecan_market_cap_gate(pf, rich):
    assert pf.coffeecan_asof("X", "2023-06-01")[0] is True                    # None = no gate
    assert pf.coffeecan_asof("X", "2023-06-01", mktcap=1e9)[0] is False       # strict >
    assert pf.coffeecan_asof("X", "2023-06-01", mktcap=1e9 + 1)[0] is True
    ok, det = pf.coffeecan_asof("X", "2023-06-01", mktcap=5e8)
    assert det["cap>1B"] is False and sum(det.values()) == 5


def test_coffeecan_needs_three_years_of_earnings(pf, rich):
    assert pf.coffeecan_asof("X", "2021-06-01") == (False, {})               # only FY2020 known
    assert pf.coffeecan_asof("X", "2022-06-01") == (False, {})               # two years
    assert pf.coffeecan_asof("X", "2023-06-01")[0] is True


def test_coffeecan_individual_criteria_fail(pf, monkeypatch):
    def run(mutate, mktcap=None):
        f = rich_facts()
        mutate(f["us-gaap"])
        monkeypatch.setattr(pf, "_load_facts", lambda t: f)
        return pf.coffeecan_asof("X", "2023-06-01", mktcap)

    def ni(vals):
        return lambda g: g.__setitem__("NetIncomeLoss", node(
            *[fy(f"{y}-12-31", f"{y + 1}-02-15", v) for y, v in zip((2022, 2021, 2020), vals)]))
    ok, det = run(ni([120.0, -1.0, 60.0]))
    assert ok is False and det["earnings_all+"] is False
    ok, det = run(ni([120.0, 0.0, 60.0]))
    assert det["earnings_all+"] is False                                      # zero is not positive
    ok, det = run(lambda g: g.__setitem__("PaymentsToAcquirePropertyPlantAndEquipment", node(
        fy("2022-12-31", "2023-02-15", 150.0))))
    assert det["fcf_pos"] is False                                            # capex == cfo -> fcf 0
    ok, det = run(lambda g: g.pop("PaymentsToAcquirePropertyPlantAndEquipment"))
    assert det["fcf_pos"] is False                                            # unknown capex
    ok, det = run(lambda g: g.__setitem__("Revenues", node(
        *[fy(f"{y}-12-31", f"{y + 1}-02-15", v) for y, v in zip((2022, 2021, 2020), (700.0, 900.0, 800.0))])))
    assert det["rev_growth"] is False                                         # newest vs oldest
    ok, det = run(lambda g: g.__setitem__("Liabilities", node(
        *[instant(f"{y}-12-31", f"{y + 1}-02-15", v) for y, v in zip((2022, 2021, 2020), (600.0, 500.0, 600.0))])))
    assert det["leverage_ok"] is False                                        # 0.6 is not < 0.6
    ok, det = run(lambda g: g.__setitem__("StockholdersEquity", node(
        *[instant(f"{y}-12-31", f"{y + 1}-02-15", v) for y, v in zip((2022, 2021, 2020), (800.0, 900.0, 1000.0))])))
    assert det["avg_roe>15"] is False
