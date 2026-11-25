"""Offline tests for net_of_cost.py (pure cost-model arithmetic)."""
import math
import os
import runpy
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import net_of_cost as nc  # noqa: E402


def test_amihud_impact_basic_and_guards():
    assert nc.amihud_impact(1e-9, 1e6) == pytest.approx(1e-3)
    assert nc.amihud_impact(0, 1e6) == 0.0
    assert nc.amihud_impact(-1e-9, 1e6) == 0.0
    assert nc.amihud_impact(1e-9, 0) == 0.0
    assert nc.amihud_impact(1e-9, -5) == 0.0


def test_amihud_impact_is_clipped_at_half():
    assert nc.amihud_impact(1.0, 1.0) == 0.5
    assert nc.amihud_impact(1e-6, 5e5) == pytest.approx(0.5)     # exactly at the cap
    assert nc.amihud_impact(1e-6, 4e5) == pytest.approx(0.4)     # just below


def test_round_trip_cost_defaults_and_formula():
    assert nc.round_trip_cost() == pytest.approx(0.002)          # 2*(5bp+5bp)
    assert nc.round_trip_cost(commission=0.001, half_spread=0.002) == pytest.approx(0.006)
    got = nc.round_trip_cost(0.0, 0.0, amihud_illiq=1e-9, trade_value=1e6)
    assert got == pytest.approx(2 * 1e-3)


def test_round_trip_cost_negative_inputs_floored():
    assert nc.round_trip_cost(commission=-0.01, half_spread=-0.01) == 0.0
    assert nc.round_trip_cost(commission=-0.01, half_spread=0.001) == pytest.approx(0.002)


def test_net_spread_scales_with_turnover():
    assert nc.net_spread(0.05, 0.01) == pytest.approx(0.04)
    assert nc.net_spread(0.05, 0.01, turnover=2.5) == pytest.approx(0.025)
    assert nc.net_spread(0.05, 0.01, turnover=0) == 0.05
    assert nc.net_spread(0.05, 0.01, turnover=-3) == 0.05        # negative turnover floored
    assert nc.net_spread(0.05, -0.01, turnover=1) == 0.05        # negative cost floored
    assert nc.net_spread(-0.01, 0.01) == pytest.approx(-0.02)


def test_break_even_cost():
    assert nc.break_even_cost(0.04, 2.0) == pytest.approx(0.02)
    assert nc.break_even_cost(0.04) == 0.04
    assert nc.break_even_cost(0.04, 0) == math.inf
    assert nc.break_even_cost(0.04, -1) == math.inf


def test_survives_costs_strict_inequality_at_boundary():
    # US cost 0.0010: break-even exactly equal -> NOT surviving (strict >)
    assert nc.survives_costs(0.0010, "US", 1.0) is False
    assert nc.survives_costs(0.0011, "US", 1.0) is True
    assert nc.survives_costs(0.0009, "US", 1.0) is False


def test_survives_costs_market_lookup_default_and_override():
    assert nc.survives_costs(0.0035, "Atlantis") is True         # default 0.0030
    assert nc.survives_costs(0.0025, "Atlantis") is False
    assert nc.survives_costs(0.0035, "BR") is False              # BR cost 0.0035 -> equal -> False
    assert nc.survives_costs(0.0040, "BR") is True
    assert nc.survives_costs(0.01, "US", round_trip=0.02) is False
    assert nc.survives_costs(0.01, "US", round_trip=0.0) is True  # override of 0 is honoured
    assert nc.survives_costs(0.0015, "US", turnover=2.0) is False  # be=0.00075 < 0.001
    assert nc.survives_costs(0.0, "US", turnover=0) is True      # no trading -> inf


def test_net_report_values_and_rounding():
    r = nc.net_report(0.108, "US", 0.5)
    assert r == {"market": "US", "gross_spread": 0.108, "round_trip_cost": 0.001,
                 "turnover": 0.5, "net_spread": 0.1075, "break_even_cost": 0.216,
                 "survives": True}


def test_net_report_zero_turnover_keeps_infinity_unrounded():
    r = nc.net_report(0.02, "IN", 0)
    assert r["break_even_cost"] == math.inf
    assert r["net_spread"] == 0.02
    assert r["survives"] is True


def test_net_report_rounds_to_five_places_and_override():
    r = nc.net_report(0.0123456789, "Atlantis", 1.0, round_trip=0.0012345678)
    assert r["gross_spread"] == 0.01235
    assert r["round_trip_cost"] == 0.00123
    assert r["net_spread"] == pytest.approx(0.01111, abs=1e-9)
    assert r["market"] == "Atlantis"
    assert nc.net_report(0.01, "Atlantis")["round_trip_cost"] == nc.DEFAULT_ROUND_TRIP


def test_net_report_not_surviving():
    r = nc.net_report(0.004, "BR", 2.0)
    assert r["survives"] is False
    assert r["net_spread"] == pytest.approx(0.004 - 2 * 0.0035)


def test_round_trip_table_is_consistent_with_aliases():
    assert nc.ROUND_TRIP["IN"] == nc.ROUND_TRIP["India"]
    assert nc.ROUND_TRIP["EU"] == nc.ROUND_TRIP["Europe"]
    assert all(0 < v < 0.01 for v in nc.ROUND_TRIP.values())


def test_main_prints_worked_examples(capsys):
    runpy.run_path(nc.__file__, run_name="__main__")
    out = capsys.readouterr().out
    assert "remediation L8" in out
    lines = [l for l in out.splitlines() if l.startswith(("accumulation", "liquidity", "marginal"))]
    assert len(lines) == 4
    assert lines[0].rstrip().endswith("YES")
    assert lines[2].rstrip().endswith("YES")        # 4.24% vs 0.35% BR
    assert lines[3].rstrip().endswith("YES")        # 0.40% break-even vs 0.10% US cost
    assert "0.40%" in lines[3]
