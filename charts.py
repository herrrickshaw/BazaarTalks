#!/usr/bin/env python3
"""
charts.py
---------
Graphics stage of the pipeline (pipeline.py --graphics). Pure, dependency-free
SVG generators — no matplotlib/plotly, matching the platform's stated
"dependency-free... unit-testable and importable anywhere" ethos (see
incremental.py). Each function takes plain data, returns a raw `<svg>...</svg>`
string — same purity contract as dashboard.py's render_html — so it can be
embedded in an HTML page (ticker_view.py, dashboard.py) or served standalone
(serve.py's GET /chart/ticker/{symbol}).

Usage:
  python charts.py --demo    # writes demo_chart.html with one of each chart
"""
from __future__ import annotations

_STROKE = "#2b6cb0"
_FILL = "#bcd7ee"
_AXIS = "#ccc"
_TEXT = "#666"


def _scale(values: list, lo: float, hi: float, out_lo: float, out_hi: float) -> list:
    span = (hi - lo) or 1.0
    return [out_lo + (v - lo) / span * (out_hi - out_lo) for v in values]


def line_chart(labels: list, values: list, width: int = 480, height: int = 120,
                title: str = "") -> str:
    """OHLC-close-style line chart. `labels` (dates/x-axis) only used for the
    first/last tick; `values` is the y series. Returns '<p>no data</p>' if empty."""
    if not values:
        return "<p class='meta'>no data</p>"

    pad = 24
    plot_w, plot_h = width - 2 * pad, height - 2 * pad
    lo, hi = min(values), max(values)
    xs = _scale(list(range(len(values))), 0, max(1, len(values) - 1), pad, width - pad)
    ys = _scale(values, lo, hi, height - pad, pad)  # invert: high value -> small y

    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    area = f"{pad},{height - pad} {points} {width - pad},{height - pad}"

    title_svg = f"<text x='{pad}' y='14' font-size='11' fill='{_TEXT}'>{title}</text>" if title else ""
    first_lbl = str(labels[0]) if labels else ""
    last_lbl = str(labels[-1]) if labels else ""

    return (
        f"<svg viewBox='0 0 {width} {height}' width='{width}' height='{height}' "
        f"xmlns='http://www.w3.org/2000/svg'>"
        f"{title_svg}"
        f"<line x1='{pad}' y1='{height - pad}' x2='{width - pad}' y2='{height - pad}' stroke='{_AXIS}'/>"
        f"<polygon points='{area}' fill='{_FILL}' opacity='0.5'/>"
        f"<polyline points='{points}' fill='none' stroke='{_STROKE}' stroke-width='2'/>"
        f"<text x='{pad}' y='{height - 6}' font-size='9' fill='{_TEXT}'>{first_lbl}</text>"
        f"<text x='{width - pad}' y='{height - 6}' font-size='9' fill='{_TEXT}' text-anchor='end'>{last_lbl}</text>"
        f"<text x='{width - pad}' y='{pad}' font-size='9' fill='{_TEXT}' text-anchor='end'>{hi:.2f}</text>"
        f"<text x='{width - pad}' y='{height - pad - 2}' font-size='9' fill='{_TEXT}' text-anchor='end'>{lo:.2f}</text>"
        f"</svg>"
    )


def bar_chart(labels: list, values: list, width: int = 480, height: int = 220,
              title: str = "") -> str:
    """Simple vertical bar chart — e.g. DVM classification counts, market coverage."""
    if not values:
        return "<p class='meta'>no data</p>"

    pad_l, pad_b, pad_t = 32, 40, 24
    plot_w, plot_h = width - pad_l - 10, height - pad_b - pad_t
    hi = max(values) or 1
    n = len(values)
    bw = plot_w / n * 0.7
    gap = plot_w / n

    title_svg = f"<text x='{pad_l}' y='14' font-size='11' fill='{_TEXT}'>{title}</text>" if title else ""
    bars = []
    for i, (label, v) in enumerate(zip(labels, values)):
        bh = (v / hi) * plot_h
        x = pad_l + i * gap + (gap - bw) / 2
        y = pad_t + (plot_h - bh)
        bars.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bw:.1f}' height='{bh:.1f}' fill='{_STROKE}'/>")
        bars.append(
            f"<text x='{x + bw / 2:.1f}' y='{pad_t + plot_h + 12}' font-size='9' fill='{_TEXT}' "
            f"text-anchor='middle'>{label}</text>"
        )
        bars.append(
            f"<text x='{x + bw / 2:.1f}' y='{y - 3:.1f}' font-size='9' fill='{_TEXT}' "
            f"text-anchor='middle'>{v}</text>"
        )

    return (
        f"<svg viewBox='0 0 {width} {height}' width='{width}' height='{height}' "
        f"xmlns='http://www.w3.org/2000/svg'>"
        f"{title_svg}"
        f"<line x1='{pad_l}' y1='{pad_t + plot_h}' x2='{width - 10}' y2='{pad_t + plot_h}' stroke='{_AXIS}'/>"
        f"{''.join(bars)}"
        f"</svg>"
    )


def gauge(value: float, lo: float = 0, hi: float = 100, label: str = "",
          width: int = 160, height: int = 54) -> str:
    """Horizontal bar gauge for a single bounded metric (momentum M, ROE, DVM score, ...)."""
    value = max(lo, min(hi, value))
    frac = (value - lo) / ((hi - lo) or 1)
    color = "#c0392b" if frac < 0.34 else ("#d68910" if frac < 0.67 else "#1e8449")

    bar_x, bar_y, bar_w, bar_h = 4, 24, width - 8, 14
    fill_w = bar_w * frac

    return (
        f"<svg viewBox='0 0 {width} {height}' width='{width}' height='{height}' "
        f"xmlns='http://www.w3.org/2000/svg'>"
        f"<text x='{bar_x}' y='14' font-size='13' font-weight='600' fill='#1a1a1a'>{value:.0f}</text>"
        f"<rect x='{bar_x}' y='{bar_y}' width='{bar_w}' height='{bar_h}' rx='3' fill='#e5e5e5'/>"
        f"<rect x='{bar_x}' y='{bar_y}' width='{fill_w:.1f}' height='{bar_h}' rx='3' fill='{color}'/>"
        f"<text x='{bar_x}' y='{bar_y + bar_h + 12}' font-size='9' fill='{_TEXT}'>{label}</text>"
        f"</svg>"
    )


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default="demo_chart.html")
    args = ap.parse_args()

    if args.demo:
        dates = [f"d{i}" for i in range(20)]
        values = [100, 102, 101, 105, 108, 107, 110, 112, 109, 111,
                  115, 117, 116, 120, 119, 122, 125, 123, 126, 128]
        html = (
            "<html><body style='font-family:sans-serif'>"
            f"<h3>line_chart</h3>{line_chart(dates, values, title='Demo close price')}"
            f"<h3>bar_chart</h3>{bar_chart(['GGG', 'GGB', 'BBG', 'BBB'], [42, 18, 9, 3], title='DVM classification')}"
            f"<h3>gauge</h3>{gauge(72, label='Momentum (M)')}"
            "</body></html>"
        )
        open(args.out, "w").write(html)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
