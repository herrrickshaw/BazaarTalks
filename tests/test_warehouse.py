"""Offline tests for warehouse.py.

duckdb is NOT installed in CI, so it is replaced via sys.modules with a recording fake
connection. That means the SQL text itself (view DDL, read_parquet/ATTACH syntax,
regexp_extract) is checked only as strings: its real execution against DuckDB is NOT
covered here. What is covered: view-selection logic (which globs/files/sources are
used, fallbacks, skips), ticker_detail's parameter binding and result shaping,
json_safe, the incremental parquet refresh, and the CLI dispatch.
"""
import importlib
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Result:
    def __init__(self, rows=None, df=None):
        self._rows, self._df = rows or [], df

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0]

    def df(self):
        return self._df if self._df is not None else pd.DataFrame()


class FakeCon:
    """Records executed SQL; behaviour is scripted by attributes."""

    def __init__(self):
        self.log = []                 # (sql, params)
        self.tables = []              # names returned by SHOW TABLES
        self.fail_on = {}             # substring -> Exception (raised once if once=True)
        self.ohlc = pd.DataFrame()
        self.views = {}               # view -> DataFrame (all rows)
        self.counts = {}
        self.closed = False
        self.frames = {}              # exact-sql -> DataFrame for .df() in main

    def execute(self, sql, params=None):
        self.log.append((sql, params))
        for sub, exc in list(self.fail_on.items()):
            if sub in sql:
                raise exc
        s = " ".join(sql.split())
        if s == "SHOW TABLES":
            return Result(rows=[(t,) for t in self.tables])
        if s.startswith("SELECT count(*) FROM"):
            name = s.rsplit(" ", 1)[1]
            return Result(rows=[(self.counts[name],)])
        if s.startswith("SELECT Date, Open, High, Low, Close, Volume FROM ohlc"):
            t, m, n = params
            d = self.ohlc
            d = d[(d["ticker"] == t) & (d["market"] == m)].sort_values("Date", ascending=False).head(n)
            return Result(df=d[["Date", "Open", "High", "Low", "Close", "Volume"]].reset_index(drop=True))
        if s.startswith("SELECT * FROM") and "WHERE ticker = ? AND market = ?" in s:
            view = s.split()[3]
            d = self.views[view]
            t, m = params
            return Result(df=d[(d["ticker"] == t) & (d["market"] == m)].head(1).reset_index(drop=True))
        if sql in self.frames:
            return Result(df=self.frames[sql])
        return Result()

    def sqls(self):
        return [" ".join(s.split()) for s, _ in self.log]

    def close(self):
        self.closed = True


@pytest.fixture
def wh(monkeypatch, tmp_path):
    """Fresh `warehouse` import against a fake duckdb; modules removed afterwards."""
    duck = types.ModuleType("duckdb")
    duck.con = FakeCon()
    duck.paths = []
    duck.connect = lambda p: duck.paths.append(p) or duck.con
    monkeypatch.setitem(sys.modules, "duckdb", duck)
    monkeypatch.delitem(sys.modules, "warehouse", raising=False)
    mod = importlib.import_module("warehouse")
    monkeypatch.setattr(mod, "HERE", str(tmp_path / "here"))
    monkeypatch.setattr(mod, "SEED", str(tmp_path / "seed"))
    monkeypatch.setattr(mod, "SEED_LOCAL", str(tmp_path / "here" / "cache_seed_local"))
    monkeypatch.setattr(mod, "DB", str(tmp_path / "market.duckdb"))
    os.makedirs(tmp_path / "here")
    mod.duck = duck
    yield mod
    sys.modules.pop("warehouse", None)


# ── json_safe ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("v", [float("nan"), float("inf"), float("-inf"), np.float64("nan"), np.float64("inf")])
def test_json_safe_non_finite_floats_become_none(wh, v):
    assert wh.json_safe(v) is None


@pytest.mark.parametrize("v", [0.0, -1.5, 1e308, 3, 0, None, "nan", "", True, (float("nan"),)])
def test_json_safe_leaves_finite_and_non_float_alone(wh, v):
    out = wh.json_safe(v)
    assert out == v or (isinstance(v, tuple) and out is v)


def test_json_safe_recurses_dicts_and_lists_without_mutating(wh):
    src = {"a": float("nan"), "b": [1.0, float("inf"), {"c": float("-inf"), "d": "x"}], "e": 2}
    out = wh.json_safe(src)
    assert out == {"a": None, "b": [1.0, None, {"c": None, "d": "x"}], "e": 2}
    assert np.isnan(src["a"])                                   # input untouched


def test_json_safe_records_from_dataframe(wh):
    df = pd.DataFrame({"x": [1.0, np.nan], "y": ["a", "b"]})
    assert wh.json_safe(df.to_dict("records")) == [{"x": 1.0, "y": "a"}, {"x": None, "y": "b"}]


# ── ticker_detail ───────────────────────────────────────────────────────────
def _con_with_data():
    con = FakeCon()
    con.tables = ["ohlc", "fundamentals", "dvm_global", "dvm_composite"]
    con.ohlc = pd.DataFrame({
        "ticker": ["AAPL"] * 4 + ["MSFT"], "market": ["US"] * 4 + ["US"],
        "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-04"]),
        "Open": [1.0, 2.0, 3.0, 4.0, 9.0], "High": [1.5, 2.5, 3.5, 4.5, 9.5],
        "Low": [0.5, 1.5, 2.5, 3.5, 8.5], "Close": [1.2, 2.2, 3.2, np.nan, 9.2],
        "Volume": [10, 20, 30, 40, 90]})
    row = lambda **kw: pd.DataFrame([{"ticker": "AAPL", "market": "US", **kw}])
    con.views = {"fundamentals": row(pe=np.nan, roe=15.0),
                 "dvm_global": row(M=70, rsi=55.0),
                 "dvm_composite": row(code="GGG", composite=80.0)}
    return con


def test_ticker_detail_full_shape_newest_first_and_json_safe(wh):
    con = _con_with_data()
    d = wh.ticker_detail(con, "AAPL", "US", bars=3)
    assert d["ticker"] == "AAPL" and d["market"] == "US"
    assert [r["Volume"] for r in d["ohlc"]] == [40, 30, 20]                # newest first, limited to bars
    assert d["ohlc"][0]["Close"] is None                                   # NaN -> None
    assert d["fundamentals"] == {"ticker": "AAPL", "market": "US", "pe": None, "roe": 15.0}
    assert d["dvm_technical"]["M"] == 70
    assert d["dvm_composite"]["code"] == "GGG"


def test_ticker_detail_default_bars_is_60(wh):
    con = _con_with_data()
    wh.ticker_detail(con, "AAPL", "US")
    assert con.log[1][1] == ["AAPL", "US", 60]


def test_ticker_detail_uses_bound_parameters_not_interpolation(wh):
    con = _con_with_data()
    evil = "AAPL'; DROP TABLE ohlc;--"
    wh.ticker_detail(con, evil, "US")
    for sql, params in con.log[1:]:
        assert evil not in sql and "?" in sql
        assert params[0] == evil


def test_ticker_detail_filters_by_market(wh):
    con = _con_with_data()
    d = wh.ticker_detail(con, "AAPL", "IN")
    assert d["ohlc"] == [] and d["fundamentals"] is None and d["dvm_composite"] is None


def test_ticker_detail_missing_views_omitted(wh):
    con = _con_with_data()
    con.tables = ["fundamentals"]
    d = wh.ticker_detail(con, "AAPL", "US")
    assert d["ohlc"] == []                                  # no ohlc view -> empty list
    assert d["dvm_technical"] is None and d["dvm_composite"] is None
    assert d["fundamentals"]["roe"] == 15.0
    assert not any("FROM ohlc" in s for s in con.sqls())
    assert not any("dvm_global" in s for s in con.sqls())


def test_ticker_detail_no_tables_at_all(wh):
    con = FakeCon()
    assert wh.ticker_detail(con, "A", "US") == {
        "ticker": "A", "market": "US", "ohlc": [], "fundamentals": None,
        "dvm_technical": None, "dvm_composite": None}


def test_ticker_detail_zero_bars(wh):
    con = _con_with_data()
    assert wh.ticker_detail(con, "AAPL", "US", bars=0)["ohlc"] == []


# ── build ───────────────────────────────────────────────────────────────────
def _touch(path, text="x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(text)


def test_build_ohlc_view_single_glob_when_no_local_seed(wh, tmp_path, capsys):
    con = FakeCon()
    wh.build(con)
    sqls = con.sqls()
    assert sqls[0] == "INSTALL sqlite; LOAD sqlite;"
    ohlc = [s for s in sqls if "VIEW ohlc" in s][0]
    assert f"read_parquet(['{tmp_path}/seed/cleaned_long_*.parquet'], filename=true)" in ohlc
    assert "regexp_extract(filename, 'cleaned_long_([A-Za-z]+)\\.parquet', 1) AS market" in ohlc
    assert "Symbol AS ticker" in ohlc
    err = capsys.readouterr().err
    assert "[skip] fundamentals_cache.db not present" in err
    assert "warehouse built:" in err
    assert not any("VIEW companies" in s for s in sqls)


def test_build_includes_local_seed_only_when_nonempty(wh, tmp_path):
    os.makedirs(wh.SEED_LOCAL)                                # exists but empty -> ignored
    con = FakeCon(); wh.build(con)
    assert [s for s in con.sqls() if "VIEW ohlc" in s][0].count("cleaned_long_*.parquet'") == 1
    _touch(os.path.join(wh.SEED_LOCAL, "cleaned_long_IN.parquet"))
    con = FakeCon(); wh.build(con)
    ohlc = [s for s in con.sqls() if "VIEW ohlc" in s][0]
    assert (f"read_parquet(['{tmp_path}/seed/cleaned_long_*.parquet', "
            f"'{wh.SEED_LOCAL}/cleaned_long_*.parquet'], filename=true)") in ohlc


def test_build_companies_view_only_if_file_exists(wh):
    _touch(os.path.join(wh.HERE, "companies_industry.parquet"))
    con = FakeCon(); wh.build(con)
    assert (f"CREATE OR REPLACE VIEW companies AS SELECT * FROM read_parquet("
            f"'{wh.HERE}/companies_industry.parquet')") in con.sqls()


def test_build_attaches_each_present_sqlite_with_plain_views(wh):
    for fn in ("dvm_global.db", "dvm_composite.db", "viability_summary.db"):
        _touch(os.path.join(wh.HERE, fn))
    con = FakeCon(); wh.build(con)
    sqls = con.sqls()
    assert f"ATTACH IF NOT EXISTS '{wh.HERE}/dvm_global.db' AS dvmg_db (TYPE sqlite)" in sqls
    assert "CREATE OR REPLACE VIEW dvm_global AS SELECT * FROM dvmg_db.dvm_global" in sqls
    assert "CREATE OR REPLACE VIEW dvm_composite AS SELECT * FROM dvmc_db.dvm_composite" in sqls
    assert "CREATE OR REPLACE VIEW viability AS SELECT * FROM viab_db.market_screen_summary" in sqls
    assert not any("ATTACH" in s and "fundamentals_cache" in s for s in sqls)   # absent file skipped


def test_build_fundamentals_view_casts_and_selects_source(wh):
    _touch(os.path.join(wh.HERE, "fundamentals_cache.db"))
    con = FakeCon(); wh.build(con)
    fund = [s for s in con.sqls() if "VIEW fundamentals" in s]
    assert len(fund) == 1
    assert fund[0].endswith("sector, source FROM fund_db.fund")
    for col in ("pe", "pb", "roe", "roa", "de", "rev_growth", "earn_growth", "op_margin", "div_yield", "mktcap"):
        assert f"TRY_CAST({col} AS DOUBLE) {col}" in fund[0]


def test_build_fundamentals_falls_back_to_null_source_column(wh, capsys):
    _touch(os.path.join(wh.HERE, "fundamentals_cache.db"))
    con = FakeCon()
    con.fail_on = {", source FROM": RuntimeError("no such column: source")}
    wh.build(con)
    fund = [s for s in con.sqls() if "VIEW fundamentals" in s]
    assert len(fund) == 2                                         # tried twice
    assert fund[1].endswith("sector, CAST(NULL AS VARCHAR) AS source FROM fund_db.fund")
    assert "[skip] fundamentals" not in capsys.readouterr().err


def test_build_view_failure_is_skipped_not_fatal(wh, capsys):
    _touch(os.path.join(wh.HERE, "dvm_global.db"))
    _touch(os.path.join(wh.HERE, "dvm_composite.db"))
    con = FakeCon()
    con.fail_on = {"VIEW dvm_global": RuntimeError("table missing")}
    wh.build(con)
    err = capsys.readouterr().err
    assert "[skip] dvm_global: table missing" in err
    assert any("VIEW dvm_composite" in s for s in con.sqls())      # later sources still built


def test_build_fundamentals_total_failure_skipped(wh, capsys):
    _touch(os.path.join(wh.HERE, "fundamentals_cache.db"))
    con = FakeCon()
    con.fail_on = {"VIEW fundamentals": RuntimeError("corrupt")}
    wh.build(con)
    assert "[skip] fundamentals: corrupt" in capsys.readouterr().err


def test_build_reports_tables_from_show_tables(wh, capsys):
    con = FakeCon(); con.tables = ["ohlc", "companies"]
    wh.build(con)
    assert "warehouse built: ['ohlc', 'companies']" in capsys.readouterr().err


# ── static tables ───────────────────────────────────────────────────────────
def test_shows_queries_and_ticker_join_views(wh):
    assert set(wh.SHOWS) == {"markets", "ggg_global", "dvm_dist", "momentum_by_market",
                             "high_roe_low_de", "industry_segments"}
    assert wh._TICKER_JOIN_VIEWS == {"fundamentals": "fundamentals", "dvm_global": "dvm_technical",
                                     "dvm_composite": "dvm_composite"}
    assert "code='GGG'" in wh.SHOWS["ggg_global"] and "LIMIT 25" in wh.SHOWS["ggg_global"]
    assert set(wh.SQLITE_SOURCES) == {"fund_db", "dvmg_db", "dvmc_db", "viab_db"}


# ── refresh_ohlc_partition ──────────────────────────────────────────────────
@pytest.fixture
def pq(monkeypatch):
    written = {}
    monkeypatch.setattr(pd, "read_parquet", lambda p, *a, **k: pd.read_pickle(p))

    def to_parquet(self, p, *a, **k):
        written["kwargs"] = k
        self.to_pickle(p)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", to_parquet)
    return written


def _px(rows):
    return pd.DataFrame(rows, columns=["Symbol", "Date", "Close"]).assign(
        Date=lambda d: pd.to_datetime(d["Date"]))


def test_refresh_creates_base_when_missing(wh, pq, tmp_path):
    os.makedirs(wh.SEED)
    new = _px([("A", "2024-01-01", 1.0), ("B", "2024-01-01", 2.0)])
    new.to_pickle(tmp_path / "new.parquet")
    out = wh.refresh_ohlc_partition("US", str(tmp_path / "new.parquet"))
    assert out == {"market": "US", "rows_before": 0, "rows_added": 2, "rows_after": 2}
    assert pq["kwargs"] == {"index": False, "compression": "zstd", "compression_level": 19}
    assert len(pd.read_pickle(os.path.join(wh.SEED, "cleaned_long_US.parquet"))) == 2


def test_refresh_appends_only_newer_dates_per_symbol(wh, pq, tmp_path):
    os.makedirs(wh.SEED)
    base = _px([("A", "2024-01-01", 1.0), ("A", "2024-01-02", 1.1), ("B", "2024-01-01", 2.0)])
    base.to_pickle(os.path.join(wh.SEED, "cleaned_long_US.parquet"))
    new = _px([("A", "2024-01-02", 9.9),      # same as high-water mark: NOT appended (strict >)
               ("A", "2024-01-03", 1.2),      # new for A
               ("B", "2024-01-02", 2.1),      # new for B (B's mark is 01-01)
               ("C", "2023-01-01", 3.0)])     # unseen symbol: any date is newer than Timestamp.min
    new.to_pickle(tmp_path / "new.parquet")
    out = wh.refresh_ohlc_partition("US", str(tmp_path / "new.parquet"))
    assert out == {"market": "US", "rows_before": 3, "rows_added": 3, "rows_after": 6}
    merged = pd.read_pickle(os.path.join(wh.SEED, "cleaned_long_US.parquet"))
    tail = merged.iloc[3:]
    assert sorted(zip(tail["Symbol"], tail["Close"])) == [("A", 1.2), ("B", 2.1), ("C", 3.0)]
    assert 9.9 not in merged["Close"].tolist()


def test_refresh_noop_when_nothing_new(wh, pq, tmp_path):
    os.makedirs(wh.SEED)
    base = _px([("A", "2024-01-05", 1.0)])
    base.to_pickle(os.path.join(wh.SEED, "cleaned_long_JP.parquet"))
    base.to_pickle(tmp_path / "new.parquet")
    out = wh.refresh_ohlc_partition("JP", str(tmp_path / "new.parquet"))
    assert out["rows_added"] == 0 and out["rows_after"] == 1


# ── main ────────────────────────────────────────────────────────────────────
def _main(wh, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["warehouse.py", *argv])
    wh.main()


def test_main_refresh_does_not_open_db(wh, monkeypatch, capsys):
    monkeypatch.setattr(wh, "refresh_ohlc_partition", lambda m, p: {"m": m, "p": p})
    _main(wh, monkeypatch, "--refresh-ohlc", "US", "/x.parquet")
    assert capsys.readouterr().out == "{'m': 'US', 'p': '/x.parquet'}\n"
    assert wh.duck.paths == []


def test_main_sql(wh, monkeypatch, capsys):
    con = wh.duck.con
    con.frames["SELECT 1 AS n"] = pd.DataFrame({"n": [1]})
    _main(wh, monkeypatch, "--sql", "SELECT 1 AS n")
    out = capsys.readouterr().out
    assert out.split() == ["n", "1"]
    assert wh.duck.paths == [wh.DB]
    sqls = con.sqls()
    assert "SET max_expression_depth=10000" in sqls and sqls[-1] == "SELECT 1 AS n"
    assert sqls[0] == "INSTALL sqlite; LOAD sqlite;"              # views rebuilt each run
    assert con.closed


def test_main_show_runs_prebuilt_query(wh, monkeypatch, capsys):
    con = wh.duck.con
    con.frames[wh.SHOWS["markets"]] = pd.DataFrame({"market": ["US"], "tickers": [5]})
    _main(wh, monkeypatch, "--show", "markets")
    assert capsys.readouterr().out.split() == ["market", "tickers", "US", "5"]
    assert con.sqls()[-1] == " ".join(wh.SHOWS["markets"].split())


def test_main_show_rejects_placeholder_and_unknown(wh, monkeypatch):
    with pytest.raises(SystemExit):
        _main(wh, monkeypatch, "--show", "industry_segments")
    with pytest.raises(SystemExit):
        _main(wh, monkeypatch, "--show", "nope")


def test_main_filter_builds_join_query(wh, monkeypatch):
    _main(wh, monkeypatch, "--filter", "roe>15 and M>=70")
    q = wh.duck.con.sqls()[-1]
    assert q == ("SELECT c.market, c.ticker, c.D, c.V, c.M, c.composite, c.code, "
                 "f.roe, f.de, f.pe, f.sector FROM dvm_composite c "
                 "LEFT JOIN fundamentals f ON c.ticker=f.ticker "
                 "WHERE roe>15 and M>=70 ORDER BY c.composite DESC LIMIT 30")


def test_main_sql_takes_precedence_over_show_and_filter(wh, monkeypatch):
    _main(wh, monkeypatch, "--sql", "SELECT 7", "--show", "markets", "--filter", "1=1")
    assert wh.duck.con.sqls()[-1] == "SELECT 7"


def test_main_default_health_summary(wh, monkeypatch, capsys):
    con = wh.duck.con
    con.tables = ["ohlc", "dvm_global", "broken"]
    con.counts = {"ohlc": 1234567, "dvm_global": 42}              # 'broken' -> KeyError -> "(view)"
    _main(wh, monkeypatch)
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "=== warehouse tables ==="
    assert out[1].split() == ["ohlc", "1,234,567", "rows"]
    assert out[2].split() == ["dvm_global", "42", "rows"]
    assert out[3].split() == ["broken", "(view)"]
    assert con.closed
