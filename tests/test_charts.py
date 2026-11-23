"""Offline tests for charts.py (pure SVG generators)."""
import os
import re
import sys
import xml.etree.ElementTree as ET

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import charts  # noqa: E402

NS = "{http://www.w3.org/2000/svg}"


def _parse(svg):
    return ET.fromstring(svg)


def test_scale_basic_and_degenerate_span():
    assert charts._scale([0, 5, 10], 0, 10, 0, 100) == [0, 50, 100]
    assert charts._scale([10, 0], 0, 10, 100, 0) == [0, 100]            # inverted range
    assert charts._scale([7, 7], 7, 7, 10, 20) == [10, 10]              # lo == hi -> span 1


def test_line_chart_empty_returns_placeholder():
    assert charts.line_chart([], []) == "<p class='meta'>no data</p>"


def test_line_chart_is_valid_svg_with_geometry():
    svg = charts.line_chart(["a", "b", "c"], [1.0, 3.0, 2.0], width=100, height=60, title="T")
    root = _parse(svg)
    assert root.tag == NS + "svg" and root.get("width") == "100" and root.get("height") == "60"
    pts = root.find(NS + "polyline").get("points").split()
    # pad=24: x from 24 to 76; y: min->height-pad=36, max->pad=24
    assert pts == ["24.0,36.0", "50.0,24.0", "76.0,30.0"]
    area = root.find(NS + "polygon").get("points")
    assert area == "24,36 24.0,36.0 50.0,24.0 76.0,30.0 76,36"
    texts = [t.text for t in root.iter(NS + "text")]
    assert texts == ["T", "a", "c", "3.00", "1.00"]


def test_line_chart_without_title_or_labels():
    root = _parse(charts.line_chart([], [1, 2]))
    texts = [t.text for t in root.iter(NS + "text")]
    assert texts == [None, None, "2.00", "1.00"]                        # empty first/last labels, no title


def test_line_chart_single_point_and_flat_series_do_not_divide_by_zero():
    root = _parse(charts.line_chart(["only"], [5.0], width=100, height=60))
    assert root.find(NS + "polyline").get("points") == "24.0,36.0"
    flat = _parse(charts.line_chart(["a", "b"], [4, 4], width=100, height=60))
    ys = {p.split(",")[1] for p in flat.find(NS + "polyline").get("points").split()}
    assert ys == {"36.0"}


def test_bar_chart_empty_returns_placeholder():
    assert charts.bar_chart([], []) == "<p class='meta'>no data</p>"


def test_bar_chart_bar_heights_proportional():
    svg = charts.bar_chart(["A", "B", "C"], [10, 5, 0], width=332, height=200, title="DVM")
    root = _parse(svg)
    rects = root.findall(NS + "rect")
    assert len(rects) == 3
    # plot_h = 200-40-24 = 136; heights 136, 68, 0
    assert [float(r.get("height")) for r in rects] == [136.0, 68.0, 0.0]
    # bars sit on the baseline y = pad_t + plot_h = 160
    assert [round(float(r.get("y")) + float(r.get("height")), 1) for r in rects] == [160.0] * 3
    texts = [t.text for t in root.iter(NS + "text")]
    assert texts == ["DVM", "A", "10", "B", "5", "C", "0"]


def test_bar_chart_all_zero_values_no_crash():
    root = _parse(charts.bar_chart(["a", "b"], [0, 0]))
    assert [float(r.get("height")) for r in root.findall(NS + "rect")] == [0.0, 0.0]


@pytest.mark.parametrize("value,color", [
    (0, "#c0392b"), (33, "#c0392b"), (34, "#d68910"), (66, "#d68910"),
    (67, "#1e8449"), (100, "#1e8449"),
])
def test_gauge_color_bands(value, color):
    svg = charts.gauge(value)
    assert f"fill='{color}'" in svg


def test_gauge_fill_width_and_clamping():
    root = _parse(charts.gauge(50, width=108))
    fill = root.findall(NS + "rect")[1]
    assert float(fill.get("width")) == 50.0                              # (108-8) * 0.5
    over = _parse(charts.gauge(250)).findall(NS + "rect")
    assert float(over[1].get("width")) == float(over[0].get("width"))   # clamped to full
    assert _parse(charts.gauge(250)).find(NS + "text").text == "100"
    under = _parse(charts.gauge(-9)).findall(NS + "rect")
    assert float(under[1].get("width")) == 0.0


def test_gauge_custom_range_label_and_degenerate_range():
    root = _parse(charts.gauge(5, lo=0, hi=10, label="ROE", width=108))
    assert float(root.findall(NS + "rect")[1].get("width")) == 50.0
    assert [t.text for t in root.iter(NS + "text")] == ["5", "ROE"]
    flat = charts.gauge(3, lo=3, hi=3)                                   # hi == lo -> frac 0
    assert "fill='#c0392b'" in flat


def test_demo_main_writes_html(tmp_path, monkeypatch, capsys):
    out = tmp_path / "d.html"
    monkeypatch.setattr(sys, "argv", ["charts.py", "--demo", "--out", str(out)])
    charts.main()
    html = out.read_text()
    assert html.count("<svg") == 3
    assert "Demo close price" in html and "DVM classification" in html and "Momentum (M)" in html
    assert capsys.readouterr().out.strip() == f"wrote {out}"


def test_main_without_demo_writes_nothing(tmp_path, monkeypatch, capsys):
    out = tmp_path / "d.html"
    monkeypatch.setattr(sys, "argv", ["charts.py", "--out", str(out)])
    charts.main()
    assert not out.exists() and capsys.readouterr().out == ""
