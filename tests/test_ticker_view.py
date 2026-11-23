"""Offline tests for ticker_view.py (HTML scorecard renderer + CLI wiring)."""
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ticker_view as tv  # noqa: E402

OHLC = [  # newest first, as warehouse.ticker_detail returns it
    {"Date": "2024-01-03 00:00:00", "Open": 3.0, "High": 4.0, "Low": 2.0, "Close": 3.5, "Volume": 30},
    {"Date": "2024-01-02 00:00:00", "Open": 2.0, "High": 3.0, "Low": 1.0, "Close": 2.5, "Volume": 20},
]


def _detail(**kw):
    d = {"ticker": "AAPL", "market": "US", "ohlc": [], "fundamentals": None,
         "dvm_technical": None, "dvm_composite": None}
    d.update(kw)
    return d


# ── _card ───────────────────────────────────────────────────────────────────
def test_card_value():
    assert tv._card("ROE %", 12.5) == ('<div class="card"><div class="label">ROE %</div>'
                                       '<div class="value">12.5</div></div>')


def test_card_none_is_na_but_zero_is_not():
    assert 'class="value na">n/a<' in tv._card("X", None)
    assert '<div class="value">0</div>' in tv._card("X", 0)
    assert '<div class="value">False</div>' in tv._card("X", False)


def test_card_nan_renders_literal_nan():
    assert '<div class="value">nan</div>' in tv._card("X", float("nan"))   # json_safe must pre-clean


# ── render_html ─────────────────────────────────────────────────────────────
def test_render_empty_detail_placeholders():
    html = tv.render_html("AAPL", "US", _detail(), None, generated="G")
    assert html.startswith("<!doctype html>") and html.endswith("</body></html>")
    assert "<title>AAPL · US</title>" in html
    assert "generated G" in html
    assert html.count('class="value na">n/a<') == 7                # all 7 scorecard cards empty
    assert "no OHLC history to chart" in html
    assert "no fundamentals cached for this ticker" in html
    assert "no DVM data cached for this ticker" in html
    assert "no OHLC history for this ticker/market" in html
    assert "Recent OHLC (0 bars)" in html
    assert "<svg" not in html                                      # no gauges when no M / composite


def test_render_handles_missing_keys_entirely():
    html = tv.render_html("X", "IN", {}, None, generated="G")
    assert "Recent OHLC (0 bars)" in html


def test_render_generated_defaults_to_formatted_now():
    html = tv.render_html("X", "US", _detail(), None)
    import re
    assert re.search(r"generated \d{4}-\d{2}-\d{2} \d{2}:\d{2}<", html)


def test_render_scorecard_values_and_label_precedence():
    d = _detail(fundamentals={"roe": 18.2, "pe": 25.0, "de": 0.5, "source": "trendlyne"},
                dvm_technical={"M": 72, "rsi": 61.5},
                dvm_composite={"label": "Strong", "code": "GGG", "composite": 80})
    html = tv.render_html("AAPL", "US", d, {"cmf": 0.12345})
    assert '<div class="label">DVM composite</div><div class="value">Strong</div>' in html
    assert '<div class="label">Momentum (M)</div><div class="value">72</div>' in html
    assert '<div class="label">RSI</div><div class="value">61.5</div>' in html
    assert '<div class="label">Accumulation (CMF)</div><div class="value">0.12</div>' in html
    assert '<div class="label">ROE %</div><div class="value">18.2</div>' in html
    assert '<div class="label">P/E</div><div class="value">25.0</div>' in html
    assert '<div class="label">D/E</div><div class="value">0.5</div>' in html


def test_render_label_falls_back_to_code_then_na():
    html = tv.render_html("A", "US", _detail(dvm_composite={"code": "GGB"}), None)
    assert '<div class="label">DVM composite</div><div class="value">GGB</div>' in html
    html = tv.render_html("A", "US", _detail(dvm_composite={"label": "", "code": "GGB"}), None)
    assert '<div class="value">GGB</div>' in html                  # empty label is falsy -> code
    html = tv.render_html("A", "US", _detail(dvm_composite={"label": None, "code": None}), None)
    assert html.count('class="value na">n/a<') >= 1


def test_render_accum_cmf_rounding_missing_key_and_zero():
    d = _detail()
    assert '<div class="value">0.13</div>' in tv.render_html("A", "US", d, {"cmf": 0.125 + 1e-9})
    assert '<div class="value">0</div>' in tv.render_html("A", "US", d, {})          # {} -> cmf 0 -> round(0,2)
    assert '<div class="value">-0.46</div>' in tv.render_html("A", "US", d, {"cmf": -0.456})


def test_render_ohlc_table_and_chart_oldest_first(monkeypatch):
    seen = {}

    def fake_line(dates, closes, width, height, title):
        seen.update(dates=dates, closes=closes, width=width, height=height, title=title)
        return "<CHART/>"

    monkeypatch.setattr(tv.charts, "line_chart", fake_line)
    html = tv.render_html("AAPL", "US", _detail(ohlc=OHLC), None)
    assert seen == {"dates": ["2024-01-02", "2024-01-03"], "closes": [2.5, 3.5],
                    "width": 600, "height": 160, "title": "AAPL close, last 2 bars"}
    assert "<CHART/>" in html and "no OHLC history to chart" not in html
    assert "Recent OHLC (2 bars)" in html
    assert "<th>Date</th><th>Open</th><th>High</th><th>Low</th><th>Close</th><th>Volume</th>" in html
    first_row = "<tr><td>2024-01-03 00:00:00</td><td>3.0</td><td>4.0</td><td>2.0</td><td>3.5</td><td>30</td></tr>"
    second_row = first_row.replace("2024-01-03", "2024-01-02").replace("<td>3.0", "<td>2.0") \
        .replace("<td>4.0", "<td>3.0").replace("<td>2.0</td><td>3.5", "<td>1.0</td><td>2.5") \
        .replace("<td>30<", "<td>20<")
    assert html.index(first_row) < html.index("2024-01-02 00:00:00")  # table keeps newest-first order


def test_render_ohlc_chart_real_charts_module():
    html = tv.render_html("AAPL", "US", _detail(ohlc=OHLC), None)
    assert "<svg" in html


def test_render_gauges_conditions(monkeypatch):
    monkeypatch.setattr(tv.charts, "gauge", lambda v, label="": f"<G {label}={v}/>")
    h = tv.render_html("A", "US", _detail(dvm_technical={"M": 55}), None)
    assert "<G Momentum (M)=55/>" in h and "DVM composite score" not in h
    h = tv.render_html("A", "US", _detail(dvm_composite={"composite": 61.5}), None)
    assert "<G DVM composite score=61.5/>" in h and "Momentum (M)=" not in h
    h = tv.render_html("A", "US", _detail(dvm_technical={"M": 0}, dvm_composite={"composite": 0}), None)
    assert "<G Momentum (M)=0/>" in h and "<G DVM composite score=0/>" in h   # zero is not None
    h = tv.render_html("A", "US", _detail(dvm_technical={"M": None}, dvm_composite={"composite": None}), None)
    assert "<G " not in h


def test_render_fundamentals_table_excludes_identity_keys_and_badge():
    f = {"ticker": "AAPL", "market": "US", "source": "screener.in", "pe": 20.0, "roe": 30.0}
    html = tv.render_html("AAPL", "US", _detail(fundamentals=f), None)
    assert "<h2>Fundamentals <span class='badge'>via screener.in</span></h2>" in html
    assert "<table><tr><td>pe</td><td>20.0</td></tr><tr><td>roe</td><td>30.0</td></tr></table>" in html


def test_render_fundamentals_without_source_has_no_badge():
    html = tv.render_html("A", "US", _detail(fundamentals={"pe": 1}), None)
    assert "<h2>Fundamentals</h2>" in html and "via" not in html


def test_render_fundamentals_only_identity_keys_gives_empty_table():
    html = tv.render_html("A", "US", _detail(fundamentals={"ticker": "A", "market": "US"}), None)
    assert "<table></table>" in html


def test_render_dvm_merged_table_prefixes_and_drops_identity():
    d = _detail(dvm_technical={"ticker": "A", "market": "US", "M": 70},
                dvm_composite={"ticker": "A", "market": "US", "code": "GGG"})
    html = tv.render_html("A", "US", d, None)
    assert "<tr><td>tech_M</td><td>70</td></tr><tr><td>composite_code</td><td>GGG</td></tr>" in html
    assert "tech_ticker" not in html and "composite_market" not in html
    assert "no DVM data cached" not in html


def test_render_dvm_only_identity_keys_shows_placeholder():
    html = tv.render_html("A", "US", _detail(dvm_technical={"ticker": "A", "market": "US"}), None)
    assert "no DVM data cached for this ticker" in html


# ── _fetch ──────────────────────────────────────────────────────────────────
class FakeCon:
    def __init__(self):
        self.executed, self.closed = [], False

    def execute(self, sql):
        self.executed.append(sql)

    def close(self):
        self.closed = True


@pytest.fixture
def fake_db(monkeypatch):
    con = FakeCon()
    duck = types.ModuleType("duckdb")
    duck.paths = []
    duck.connect = lambda p: duck.paths.append(p) or con
    wh = types.ModuleType("warehouse")
    wh.log = []
    wh.build = lambda c: wh.log.append(("build", c))

    def ticker_detail(c, t, m, bars=60):
        wh.log.append(("detail", t, m, bars))
        if wh.fail:
            raise RuntimeError("query failed")
        return {"ticker": t, "market": m, "ohlc": []}

    wh.fail = False
    wh.ticker_detail = ticker_detail
    monkeypatch.setitem(sys.modules, "duckdb", duck)
    monkeypatch.setitem(sys.modules, "warehouse", wh)
    return types.SimpleNamespace(con=con, duck=duck, wh=wh)


def test_fetch_builds_then_queries_and_closes(fake_db):
    out = tv._fetch("AAPL", "US", 25)
    assert out == {"ticker": "AAPL", "market": "US", "ohlc": []}
    assert fake_db.duck.paths == [os.path.join(tv.HERE, "market.duckdb")]
    assert fake_db.wh.log == [("build", fake_db.con), ("detail", "AAPL", "US", 25)]
    assert fake_db.con.executed == ["SET max_expression_depth=10000"]
    assert fake_db.con.closed


def test_fetch_closes_connection_on_error(fake_db):
    fake_db.wh.fail = True
    with pytest.raises(RuntimeError, match="query failed"):
        tv._fetch("A", "US", 5)
    assert fake_db.con.closed


# ── _fetch_accum_row ────────────────────────────────────────────────────────
def _install_acc(monkeypatch, fn):
    acc = types.ModuleType("accumulation_screener")
    acc.current_screen = fn
    monkeypatch.setitem(sys.modules, "accumulation_screener", acc)


def test_accum_row_found(monkeypatch):
    _install_acc(monkeypatch, lambda m: pd.DataFrame({"ticker": ["AAA", "BBB"], "cmf": [0.1, 0.2]}))
    assert tv._fetch_accum_row("BBB", "US") == {"ticker": "BBB", "cmf": 0.2}


def test_accum_row_first_match_wins(monkeypatch):
    _install_acc(monkeypatch, lambda m: pd.DataFrame({"ticker": ["A", "A"], "cmf": [0.1, 0.9]}))
    assert tv._fetch_accum_row("A", "US")["cmf"] == 0.1


def test_accum_row_missing_ticker_or_empty_panel(monkeypatch):
    _install_acc(monkeypatch, lambda m: pd.DataFrame({"ticker": ["AAA"], "cmf": [0.1]}))
    assert tv._fetch_accum_row("ZZZ", "US") is None
    _install_acc(monkeypatch, lambda m: pd.DataFrame(columns=["ticker", "cmf"]))
    assert tv._fetch_accum_row("AAA", "US") is None


def test_accum_row_passes_market(monkeypatch):
    seen = []
    _install_acc(monkeypatch, lambda m: seen.append(m) or pd.DataFrame())
    tv._fetch_accum_row("A", "IN")
    assert seen == ["IN"]


def test_accum_row_swallows_errors(monkeypatch, capsys):
    def boom(m):
        raise ValueError("no panel")
    _install_acc(monkeypatch, boom)
    assert tv._fetch_accum_row("A", "US") is None
    assert "accumulation lookup skipped: no panel" in capsys.readouterr().out


def test_accum_row_import_failure_swallowed(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "accumulation_screener", None)   # import raises ImportError
    assert tv._fetch_accum_row("A", "US") is None
    assert "accumulation lookup skipped" in capsys.readouterr().out


# ── main ────────────────────────────────────────────────────────────────────
@pytest.fixture
def wired(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    s = types.SimpleNamespace(fetch=[], accum=[], render=[])
    monkeypatch.setattr(tv, "_fetch", lambda t, m, b: s.fetch.append((t, m, b)) or {"d": 1})
    monkeypatch.setattr(tv, "_fetch_accum_row", lambda t, m: s.accum.append((t, m)) or {"cmf": 1})
    monkeypatch.setattr(tv, "render_html",
                        lambda t, m, d, a: s.render.append((t, m, d, a)) or "<html>x</html>")
    return s


def _main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["ticker_view.py", *argv])
    tv.main()


def test_main_defaults_uppercase_and_output(monkeypatch, wired, tmp_path, capsys):
    _main(monkeypatch, "aapl", "--market", "us")
    assert wired.fetch == [("AAPL", "US", 60)]
    assert wired.accum == [("AAPL", "US")]
    assert wired.render == [("AAPL", "US", {"d": 1}, {"cmf": 1})]
    assert (tmp_path / "ticker_AAPL_US.html").read_text() == "<html>x</html>"
    assert capsys.readouterr().out == "wrote ticker_AAPL_US.html (0 KB)\n"


def test_main_bars_out_and_no_accum(monkeypatch, wired, tmp_path, capsys):
    _main(monkeypatch, "tcs", "--market", "in", "--bars", "5", "--out", "custom.html", "--no-accum")
    assert wired.fetch == [("TCS", "IN", 5)]
    assert wired.accum == []
    assert wired.render[0][3] is None
    assert (tmp_path / "custom.html").exists() and not (tmp_path / "ticker_TCS_IN.html").exists()
    assert "wrote custom.html" in capsys.readouterr().out


def test_main_size_kb_floor(monkeypatch, wired, capsys):
    monkeypatch.setattr(tv, "render_html", lambda *a: "x" * 2048)
    _main(monkeypatch, "A", "--market", "us")
    assert "(2 KB)" in capsys.readouterr().out


def test_main_open_launches_browser(monkeypatch, wired, tmp_path):
    import webbrowser
    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda u: opened.append(u))
    _main(monkeypatch, "a", "--market", "us", "--out", "o.html", "--open")
    assert opened == [f"file://{os.path.abspath('o.html')}"]


def test_main_no_open_by_default(monkeypatch, wired):
    import webbrowser
    monkeypatch.setattr(webbrowser, "open", lambda u: pytest.fail("should not open"))
    _main(monkeypatch, "a", "--market", "us")


def test_main_market_required(monkeypatch, wired):
    with pytest.raises(SystemExit):
        _main(monkeypatch, "a")
