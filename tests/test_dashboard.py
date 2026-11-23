"""Offline tests for dashboard.py (HTML rendering + faked warehouse/screener)."""
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dashboard as db  # noqa: E402


def test_render_html_structure_and_generated_stamp():
    html = db.render_html({"Market coverage": pd.DataFrame({"market": ["US"], "n": [3]})},
                          generated="2026-01-02 03:04")
    assert html.startswith("<!doctype html><html>")
    assert html.endswith("</body></html>")
    assert "<p class='meta'>generated 2026-01-02 03:04</p>" in html
    assert "<h2>Market coverage</h2>" in html
    assert "<table" in html and "<td>US</td>" in html
    assert "<svg" not in html                           # bar chart only for the DVM section


def test_render_html_default_generated_uses_clock(monkeypatch):
    class _DT:
        class datetime:
            @staticmethod
            def now():
                import datetime as real
                return real.datetime(2030, 5, 6, 7, 8)
    monkeypatch.setattr(db, "dt", _DT)
    assert "generated 2030-05-06 07:08" in db.render_html({})


@pytest.mark.parametrize("empty", [None, pd.DataFrame()])
def test_render_html_empty_section_says_no_data(empty):
    html = db.render_html({"X": empty})
    assert "<h2>X</h2><p class='meta'>no data</p>" in html
    assert "<table" not in html


def test_render_html_dvm_section_gets_bar_chart():
    df = pd.DataFrame({"label": ["GGG", "BBB"], "n": [7, 2]})
    html = db.render_html({"DVM classification": df})
    assert html.count("<svg") == 1
    assert ">GGG<" in html and ">7<" in html
    assert html.index("<svg") < html.index("<table")    # chart precedes the table


def test_render_html_dvm_section_without_label_columns_has_no_chart():
    html = db.render_html({"DVM classification": pd.DataFrame({"x": [1]})})
    assert "<svg" not in html and "<table" in html


def test_render_html_chart_only_for_exact_title():
    df = pd.DataFrame({"label": ["A"], "n": [1]})
    assert "<svg" not in db.render_html({"dvm classification": df})


def test_render_html_preserves_section_order_and_no_index():
    html = db.render_html({"Second": pd.DataFrame({"a": [1]}), "First": pd.DataFrame({"b": [2]})})
    assert html.index("<h2>Second</h2>") < html.index("<h2>First</h2>")
    assert "<th></th>" not in html                       # index=False


# ── _accumulation_section with a fake accumulation_screener ───────────────────
def _fake_acc(monkeypatch, seed_dir, frames):
    mod = types.ModuleType("accumulation_screener")
    mod.SEED = str(seed_dir)
    mod.calls = []

    def current_screen(m):
        mod.calls.append(m)
        return frames.get(m, pd.DataFrame())
    mod.current_screen = current_screen
    monkeypatch.setitem(sys.modules, "accumulation_screener", mod)
    return mod


def _screen(market, rows):
    return pd.DataFrame([{"market": market, "ticker": t, "close": 10.0, "cmf": c, "accum": a,
                          "extra": 1} for t, c, a in rows])


def test_accumulation_section_discovers_markets_sorts_rounds_and_limits(tmp_path, monkeypatch):
    for n in ("cleaned_long_US.parquet", "cleaned_long_IN.parquet", "notes.txt"):
        (tmp_path / n).write_text("")
    mod = _fake_acc(monkeypatch, tmp_path, {
        "US": _screen("US", [("AAA", 0.12345, 1.2349), ("BBB", 0.2, np.nan)]),
        "IN": _screen("IN", [("CCC", 0.3333, 3.3333), ("DDD", 0.1, 0.5)]),
    })
    out = db._accumulation_section(top=2)
    assert mod.calls == ["IN", "US"]                                # sorted, non-parquet skipped
    assert list(out.columns) == ["market", "ticker", "close", "cmf", "accum"]
    assert out["ticker"].tolist() == ["CCC", "AAA"]                 # NaN accum dropped, top-2 by accum
    assert out["accum"].tolist() == [3.33, 1.23]
    assert out["cmf"].tolist() == [0.33, 0.12]


def test_accumulation_section_explicit_markets_and_empty(tmp_path, monkeypatch):
    mod = _fake_acc(monkeypatch, tmp_path, {})
    out = db._accumulation_section(markets=["XX", "YY"])
    assert mod.calls == ["XX", "YY"]
    assert out.empty


def test_accumulation_section_default_top_is_15(tmp_path, monkeypatch):
    rows = [(f"T{i:02d}", 0.1, float(i)) for i in range(20)]
    _fake_acc(monkeypatch, tmp_path, {"US": _screen("US", rows)})
    out = db._accumulation_section(markets=["US"])
    assert len(out) == 15 and out["ticker"].iloc[0] == "T19"


# ── _query_all with fake duckdb + warehouse ───────────────────────────────────
class _Res:
    def __init__(self, df):
        self._df = df

    def df(self):
        return self._df


def _fake_backend(monkeypatch, shows, build_error=None):
    duck = types.ModuleType("duckdb")
    wh = types.ModuleType("warehouse")
    state = {"executed": [], "closed": False, "path": None}

    class Con:
        def execute(self, sql):
            state["executed"].append(sql)
            if sql in shows and isinstance(shows[sql], Exception):
                raise shows[sql]
            return _Res(shows.get(sql, pd.DataFrame()))

        def close(self):
            state["closed"] = True

    def connect(path):
        state["path"] = path
        return Con()
    duck.connect = connect
    wh.SHOWS = {"markets": "Q_MARKETS", "dvm_dist": "Q_DVM", "ggg_global": "Q_GGG",
                "high_roe_low_de": "Q_ROE"}
    wh.build = lambda con: state.update(built=True)
    monkeypatch.setitem(sys.modules, "duckdb", duck)
    monkeypatch.setitem(sys.modules, "warehouse", wh)
    return state


ACC_KEY = "Accumulation / CMF screen (top by accumulation, validated 6m signal)"


def test_query_all_collects_sections_and_isolates_failures(monkeypatch, capsys):
    good = pd.DataFrame({"a": [1]})
    state = _fake_backend(monkeypatch, {"Q_MARKETS": good, "Q_DVM": RuntimeError("no dvm"),
                                         "Q_GGG": good, "Q_ROE": good})
    monkeypatch.setattr(db, "_accumulation_section", lambda top=15: pd.DataFrame({"t": [top]}))
    sec = db._query_all(accum_top=7)
    assert list(sec) == ["Market coverage", "DVM classification", "Top GGG Strong Performers",
                         "High ROE / low D-E", ACC_KEY]
    assert sec["DVM classification"].empty                  # failed section degrades to empty
    assert sec["Market coverage"] is good
    assert sec[ACC_KEY]["t"].tolist() == [7]
    assert state["closed"] and state["built"]
    assert state["path"] == os.path.join(db.HERE, "market.duckdb")
    assert state["executed"][0] == "SET max_expression_depth=10000"
    assert "'DVM classification' section skipped: no dvm" in capsys.readouterr().out


def test_query_all_skips_accumulation_when_disabled(monkeypatch):
    _fake_backend(monkeypatch, {})
    monkeypatch.setattr(db, "_accumulation_section",
                        lambda top=15: pytest.fail("must not be called"))
    sec = db._query_all(include_accum=False)
    assert ACC_KEY not in sec and len(sec) == 4


def test_query_all_accumulation_failure_is_isolated(monkeypatch, capsys):
    _fake_backend(monkeypatch, {})

    def boom(top=15):
        raise ValueError("screener down")
    monkeypatch.setattr(db, "_accumulation_section", boom)
    sec = db._query_all()
    assert sec[ACC_KEY].empty
    assert "accumulation section skipped: screener down" in capsys.readouterr().out


# ── main ──────────────────────────────────────────────────────────────────────
def test_main_writes_file_and_passes_flags(tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_query(accum_top, include_accum):
        seen.update(top=accum_top, inc=include_accum)
        return {"T": pd.DataFrame({"a": [1]})}
    monkeypatch.setattr(db, "_query_all", fake_query)
    out = tmp_path / "dash.html"
    monkeypatch.setattr(sys, "argv", ["dashboard.py", "--out", str(out), "--accum-top", "3",
                                      "--no-accum"])
    db.main()
    assert seen == {"top": 3, "inc": False}
    assert "<h2>T</h2>" in out.read_text()
    assert capsys.readouterr().out.startswith(f"wrote {out} (")


def test_main_open_launches_browser(tmp_path, monkeypatch):
    import webbrowser
    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))
    monkeypatch.setattr(db, "_query_all", lambda accum_top, include_accum: {})
    out = tmp_path / "d.html"
    monkeypatch.setattr(sys, "argv", ["dashboard.py", "--out", str(out), "--open"])
    db.main()
    assert opened == [f"file://{os.path.abspath(out)}"]


def test_main_defaults(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(db, "_query_all", lambda accum_top, include_accum: seen.update(
        top=accum_top, inc=include_accum) or {})
    monkeypatch.setattr(db, "OUT", str(tmp_path / "x.html"))
    monkeypatch.setattr(sys, "argv", ["dashboard.py"])
    # argparse default for --out was bound at import; just make sure defaults for the rest hold
    monkeypatch.setattr(db, "open", lambda *a, **k: __import__("io").StringIO(), raising=False)
    db.main()
    assert seen == {"top": 15, "inc": True}
