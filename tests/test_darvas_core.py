"""Darvas kernel: unit tests + golden characterisation of the five scanners' compute_darvas_box.

The golden file was captured from the pre-refactor scanners on deterministic synthetic panels
(tests/_darvas_panels.py); it pins per-market keys, key order, rounding and error behaviour.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from _darvas_panels import MARKETS, panels
from _scanner_import import import_scanner

import darvas_core as dc

GOLDEN = json.loads((Path(__file__).parent / "golden" / "darvas_golden.json").read_text())


def test_find_box_top_then_bottom():
    highs = [10, 12, 11, 10.5, 10.4, 10.3]
    lows = [9, 9.5, 9.2, 9.4, 9.6, 9.7]
    assert dc.find_box(highs, lows, 3) == (11, 9.2)  # most recent confirmed top


def test_find_box_none_when_no_confirmed_top():
    rising = [1, 2, 3, 4, 5, 6, 7]
    assert dc.find_box(rising, rising, 3) == (None, None)


def test_find_box_ignores_zero_gaps_and_falls_back_to_min_low():
    # a confirmed top exists but every low in its segment is 0 -> no bottom
    assert dc.find_box([5, 4, 3, 2, 1], [0, 0, 0, 0, 0], 3) == (4, None)
    # no confirmed low -> falls back to the smallest positive low
    assert dc.find_box([5, 4, 3, 2, 1], [5, 4, 3, 2, 1], 3) == (4, 1)


@pytest.mark.parametrize("cur,sig", [(13, dc.BREAKOUT_BUY), (8, dc.BREAKDOWN_SELL), (10, dc.IN_BOX),
                                     (12, dc.IN_BOX), (9, dc.IN_BOX)])
def test_classify_boundaries_are_inclusive(cur, sig):
    assert dc.classify(cur, 12, 9) == sig


def test_box_geometry_zero_denominators():
    assert dc.box_geometry(0, 5, 5) == (0, 0, 0)
    assert dc.box_geometry(10, 12, 8) == (4, pytest.approx(20.0), pytest.approx(50.0))


def test_find_col_prefers_earlier_candidate():
    assert dc.find_col(["CH_CLOSING_PRICE", "Close"], ["Close", "CH_CLOSING_PRICE"]) == "Close"
    assert dc.find_col(["x"], ["High"]) is None


@pytest.mark.parametrize("market", MARKETS)
def test_scanner_output_matches_pre_refactor_golden(market):
    mod = import_scanner(market)
    for name, df in panels().items():
        want_kind, want = GOLDEN[f"{market}/{name}"]
        if want_kind == "err":
            with pytest.raises(Exception) as ei:
                mod.compute_darvas_box(df)
            assert type(ei.value).__name__ == want
        else:
            got = json.loads(json.dumps(list(mod.compute_darvas_box(df).items()), default=str))
            assert got == want, f"{market}/{name}"
