"""
darvas_core.py
--------------
The single implementation of the Darvas-box kernel shared by the five
``full_*_market_scan.py`` scanners (they used to carry five copy-pasted variants).

Pure functions only: lists of floats in, numbers out, no I/O and no pandas column
lookup. Each scanner keeps its own ``compute_darvas_box`` wrapper (column lookup,
rounding, result keys), so scanner output is unchanged; only the box-finding logic
and the signal classification live here.

Design rule (see feedback "Darvas Box design rule"): the *current* (last) bar is
always excluded from box formation, so a breakout is never part of its own box.
"""

from __future__ import annotations

BREAKOUT_BUY = "BREAKOUT_BUY"
BREAKDOWN_SELL = "BREAKDOWN_SELL"
IN_BOX = "IN_BOX"


def find_box(highs: list[float], lows: list[float],
             confirm: int) -> tuple[float | None, float | None]:
    """Return ``(box_top, box_bottom)`` from *historical* highs/lows (current bar already removed).

    box_top    = most recent high that the next ``confirm`` highs all stay below.
    box_bottom = first low at/after the top's bar that the next ``confirm`` lows all stay above;
                 falls back to the minimum positive low of that segment.
    Zeros (NaN-filled gaps) never qualify as a candidate. ``(None, None)`` = no top found;
    ``(top, None)`` = a top but no usable bottom.
    """
    n = len(highs)
    box_top_idx = box_top = None
    for i in range(n - confirm - 1, -1, -1):
        candidate = highs[i]
        if candidate == 0:
            continue
        window = highs[i + 1: i + 1 + confirm]
        if len(window) == confirm and all(h < candidate for h in window):
            box_top_idx, box_top = i, candidate
            break
    if box_top is None:
        return None, None

    segment = lows[box_top_idx:]
    box_bottom = None
    for i in range(len(segment) - confirm):
        candidate = segment[i]
        if candidate == 0:
            continue
        window = segment[i + 1: i + 1 + confirm]
        if len(window) == confirm and all(low > candidate for low in window):
            box_bottom = candidate
            break
    if box_bottom is None:
        valid = [low for low in segment if low > 0]
        box_bottom = min(valid) if valid else None
    return box_top, box_bottom


def classify(current: float, box_top: float, box_bottom: float) -> str:
    """BREAKOUT_BUY above the box, BREAKDOWN_SELL below it, else IN_BOX."""
    if current > box_top:
        return BREAKOUT_BUY
    if current < box_bottom:
        return BREAKDOWN_SELL
    return IN_BOX


def box_geometry(current: float, box_top: float,
                 box_bottom: float) -> tuple[float, float, float]:
    """``(box_range, upside_to_top_pct, position_in_box_pct)`` — zero denominators give 0."""
    box_range = box_top - box_bottom
    upside = ((box_top - current) / current * 100) if current else 0
    pos = ((current - box_bottom) / box_range * 100) if box_range else 0
    return box_range, upside, pos


def find_col(columns, candidates) -> str | None:
    """First column whose upper-cased name contains a candidate (candidates tried in order)."""
    for cand in candidates:
        match = next((col for col in columns if cand.upper() in col.upper()), None)
        if match:
            return match
    return None


def numeric_list(df, col) -> list[float]:
    """Column as a list of floats; non-numeric / missing values become 0."""
    import pandas as pd
    return pd.to_numeric(df[col], errors="coerce").fillna(0).tolist()
