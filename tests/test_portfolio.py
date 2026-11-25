"""Offline tests for portfolio.py (constrained weights, turnover, CLI)."""
import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import portfolio as pf  # noqa: E402


# ── optimiser primitives ──────────────────────────────────────────────────────
def test_min_variance_identity_cov_is_equal_weight():
    w = pf.min_variance_weights(np.eye(4))
    assert w == pytest.approx(np.full(4, 0.25))


def test_min_variance_favours_low_vol_asset():
    w = pf.min_variance_weights(np.diag([0.01, 0.04]))
    assert w.sum() == pytest.approx(1.0)
    assert w == pytest.approx([0.8, 0.2])


def test_max_sharpe_proportional_to_inv_cov_mu():
    cov = np.diag([0.04, 0.04])
    w = pf.max_sharpe_weights(np.array([0.10, 0.30]), cov)
    assert w == pytest.approx([0.25, 0.75])


def test_max_sharpe_zero_sum_falls_back_to_equal():
    # Σ⁻¹μ sums to exactly 0 -> equal weights, not a division by zero
    w = pf.max_sharpe_weights(np.array([0.1, -0.1]), np.eye(2))
    assert w == pytest.approx([0.5, 0.5])


def test_long_only_clips_and_renormalises():
    w = pf.long_only(np.array([0.6, -0.2, 0.6]))
    assert w == pytest.approx([0.5, 0.0, 0.5])


def test_long_only_all_nonpositive_is_equal_weight():
    assert pf.long_only(np.array([-1.0, 0.0, -2.0])) == pytest.approx([1 / 3] * 3)


# ── caps ──────────────────────────────────────────────────────────────────────
def test_cap_weights_pushes_excess_to_uncapped():
    w = pf.cap_weights(np.array([0.7, 0.2, 0.1]), 0.5)
    assert w.max() <= 0.5 + 1e-9
    assert w.sum() == pytest.approx(1.0)
    assert w[0] == pytest.approx(0.5)
    # excess 0.2 is split in proportion 2:1 between the free names
    assert w[1:] == pytest.approx([0.2 + 0.2 * 2 / 3, 0.1 + 0.2 / 3])


def test_cap_weights_noop_when_all_under_cap():
    w = np.array([0.4, 0.3, 0.3])
    assert pf.cap_weights(w, 0.4) == pytest.approx(w)      # == cap is not "over"


def test_cap_weights_infeasible_returns_equal():
    assert pf.cap_weights(np.array([0.9, 0.1]), 0.4) == pytest.approx([0.5, 0.5])


def test_cap_weights_exactly_feasible_boundary():
    # n*cap == 1 is feasible: everything ends up at cap
    w = pf.cap_weights(np.array([0.9, 0.05, 0.05]), 1 / 3)
    assert w == pytest.approx([1 / 3] * 3)


def test_cap_weights_no_free_names_clamps_all_to_cap():
    # zero-weight names are not "free" (w>0 required) -> branch sets all to cap
    w = pf.cap_weights(np.array([0.6, 0.4, 0.0, 0.0]), 0.25)
    assert w == pytest.approx([0.25] * 4)


def test_cap_weights_does_not_mutate_input():
    src = np.array([0.7, 0.2, 0.1])
    pf.cap_weights(src, 0.5)
    assert src.tolist() == [0.7, 0.2, 0.1]


def test_sector_cap_redistributes():
    w = np.array([0.4, 0.3, 0.2, 0.1])
    out = pf.apply_sector_cap(w, ["A", "A", "B", "C"], 0.5)
    assert out[:2].sum() == pytest.approx(0.5)
    assert out.sum() == pytest.approx(1.0)
    assert out[0] / out[1] == pytest.approx(4 / 3)           # within-sector proportions kept
    assert out[2] / out[3] == pytest.approx(2.0)             # free names scaled proportionally


def test_sector_cap_infeasible_leaves_weights():
    w = np.array([0.5, 0.5])
    out = pf.apply_sector_cap(w, ["A", "A"], 0.5)            # 0.5 * 1 sector < 1
    assert out == pytest.approx(w)


def test_sector_cap_noop_when_under():
    w = np.array([0.25, 0.25, 0.25, 0.25])
    out = pf.apply_sector_cap(w, ["A", "A", "B", "B"], 0.5)  # exactly at cap, not over
    assert out == pytest.approx(w)


def test_sector_cap_no_free_names_breaks_out():
    # sector B has zero weight -> not free; loop must terminate without error
    w = np.array([0.6, 0.4, 0.0])
    out = pf.apply_sector_cap(w, ["A", "A", "B"], 0.5)
    assert out.sum() == pytest.approx(1.0)
    assert out == pytest.approx([0.6, 0.4, 0.0])             # rescaled to 0.5/0.5 -> renormalised


# ── turnover ──────────────────────────────────────────────────────────────────
def test_turnover_known_values():
    assert pf.turnover([0.5, 0.5], [0.5, 0.5]) == 0.0
    assert pf.turnover([1.0, 0.0], [0.0, 1.0]) == 1.0
    assert pf.turnover([0.6, 0.4], [0.5, 0.5]) == pytest.approx(0.1)


def test_blend_within_budget_returns_target_unchanged():
    new = np.array([0.6, 0.4]); old = np.array([0.5, 0.5])
    assert pf.blend_to_turnover(new, old, 0.1) is new        # t == budget -> not blended
    assert pf.blend_to_turnover(new, new.copy(), 0.0) is new  # t == 0


def test_blend_limits_turnover_to_budget():
    new = np.array([1.0, 0.0]); old = np.array([0.0, 1.0])
    w = pf.blend_to_turnover(new, old, 0.25)
    assert pf.turnover(w, old) == pytest.approx(0.25)
    assert w == pytest.approx([0.25, 0.75])
    assert w.sum() == pytest.approx(1.0)


# ── build_weights ─────────────────────────────────────────────────────────────
def _returns(seed=1, n=300):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 0.01, (n, 1))
    data = {
        "A": base[:, 0] * 1.0 + rng.normal(0, 0.002, n),
        "B": base[:, 0] * 0.5 + rng.normal(0, 0.004, n),
        "C": rng.normal(0.001, 0.02, n),
        "D": rng.normal(0.0005, 0.015, n),
    }
    return pd.DataFrame(data)


def test_build_weights_needs_two_complete_assets():
    df = pd.DataFrame({"A": [0.01, 0.02, 0.03], "B": [np.nan, 0.01, 0.02]})
    with pytest.raises(ValueError, match="need >=2 assets"):
        pf.build_weights(df)


def test_build_weights_min_var_long_only_sorted_and_sums_to_one():
    w = pf.build_weights(_returns())
    assert isinstance(w, pd.Series)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()
    assert list(w.values) == sorted(w.values, reverse=True)
    assert w.index[0] == "A" or w.index[0] == "B"            # low-vol names dominate


def test_build_weights_drops_incomplete_columns():
    df = _returns()
    df.loc[0, "C"] = np.nan
    w = pf.build_weights(df)
    assert "C" not in w.index
    assert set(w.index) == {"A", "B", "D"}


def test_build_weights_max_sharpe_differs_from_min_var():
    df = _returns()
    a = pf.build_weights(df, method="min_var")
    b = pf.build_weights(df, method="max_sharpe")
    assert b.sum() == pytest.approx(1.0)
    assert not np.allclose(a.sort_index().values, b.sort_index().values)


def test_build_weights_position_cap_applied():
    uncapped = pf.build_weights(_returns())
    assert uncapped.max() > 0.5
    w = pf.build_weights(_returns(), cap=0.5)
    assert w.max() <= 0.5 + 1e-9
    assert w.sum() == pytest.approx(1.0)


def test_build_weights_cap_zero_or_none_means_no_cap():
    base = pf.build_weights(_returns())
    assert pf.build_weights(_returns(), cap=0).equals(base)
    assert pf.build_weights(_returns(), cap=None).equals(base)


def test_build_weights_sector_cap_and_retightened_position_cap():
    df = _returns()
    sectors = ["X", "X", "Y", "Z"]
    w = pf.build_weights(df, cap=0.5, sectors=sectors, sector_cap=0.6)
    assert w.sum() == pytest.approx(1.0)
    assert w.max() <= 0.5 + 1e-9
    assert w[["A", "B"]].sum() <= 0.6 + 1e-6


def test_build_weights_sector_cap_ignored_without_sectors_or_cap():
    df = _returns()
    base = pf.build_weights(df)
    assert pf.build_weights(df, sectors=None, sector_cap=0.2).equals(base)
    assert pf.build_weights(df, sectors=["X", "X", "Y", "Z"], sector_cap=None).equals(base)


def test_build_weights_sectors_mapped_by_original_columns_when_column_dropped():
    df = _returns()
    df.loc[3, "A"] = np.nan                                  # A is dropped from R
    w = pf.build_weights(df, sectors=["S1", "S1", "S2", "S3"], sector_cap=0.5)
    assert "A" not in w.index
    assert w.sum() == pytest.approx(1.0)


def test_build_weights_turnover_blend():
    df = _returns()
    target = pf.build_weights(df)
    prev = {"A": 0.0, "B": 0.0, "C": 0.5, "D": 0.5}
    w = pf.build_weights(df, prev=prev, max_turnover=0.1)
    old = np.array([prev[c] for c in w.index])
    assert pf.turnover(w.values, old) == pytest.approx(0.1, abs=1e-9)
    assert pf.turnover(target.reindex(w.index).values, old) > 0.1
    assert w.sum() == pytest.approx(1.0)


def test_build_weights_turnover_skipped_without_both_args_or_empty_prev():
    df = _returns()
    base = pf.build_weights(df)
    assert pf.build_weights(df, prev={"A": 1.0}).equals(base)           # no max_turnover
    assert pf.build_weights(df, max_turnover=0.1).equals(base)          # no prev
    assert pf.build_weights(df, prev={"Z": 1.0}, max_turnover=0.1).equals(base)  # old book sums 0


# ── data helpers ──────────────────────────────────────────────────────────────
def _make_db(path, rows):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE dvm_composite (market TEXT, ticker TEXT, sector TEXT, "
                "composite REAL, code TEXT)")
    con.executemany("INSERT INTO dvm_composite VALUES (?,?,?,?,?)", rows)
    con.commit(); con.close()


ROWS = [("US", "A", "Tech", 90.0, "GGG"), ("US", "B", "Tech", 80.0, "GGG"),
        ("US", "C", "Fin", 85.0, "GGG"), ("US", "D", "Fin", 99.0, "GGB"),
        ("US", "E", "Util", 70.0, "GGG"), ("JP", "J1", "Tech", 95.0, "GGG")]


def test_candidates_from_composite_filters_orders_limits(tmp_path):
    db = str(tmp_path / "d.db"); _make_db(db, ROWS)
    df = pf.candidates_from_composite("US", 3, db)
    assert df["ticker"].tolist() == ["A", "C", "B"]          # GGG only, composite DESC, LIMIT 3
    assert list(df.columns) == ["ticker", "sector", "composite"]
    assert pf.candidates_from_composite("XX", 3, db).empty


def test_candidates_from_composite_n_is_coerced_to_int(tmp_path):
    db = str(tmp_path / "d.db"); _make_db(db, ROWS)
    assert len(pf.candidates_from_composite("US", "2", db)) == 2


def _px(symbols, n=300):
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2023-01-02", periods=n)
    rows = []
    for s in symbols:
        p = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
        rows += [{"Date": d, "Symbol": s, "Close": float(x)} for d, x in zip(dates, p)]
    return pd.DataFrame(rows)


def test_returns_for_filters_symbols_and_tails_252(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "SEED", str(tmp_path))
    monkeypatch.setattr(pf.pd, "read_parquet", lambda p: _px(["A", "B", "OTHER"]))
    R = pf.returns_for("US", ["A", "B"])
    assert list(R.columns) == ["A", "B"]
    assert len(R) == 252
    assert R.notna().all().all()


def test_returns_for_short_history_keeps_nan_first_row(tmp_path, monkeypatch):
    monkeypatch.setattr(pf.pd, "read_parquet", lambda p: _px(["A"], n=10))
    R = pf.returns_for("US", ["A"])
    assert len(R) == 10 and np.isnan(R.iloc[0, 0])


# ── CLI ───────────────────────────────────────────────────────────────────────
def _setup_cli(tmp_path, monkeypatch, rows=ROWS):
    db = str(tmp_path / "d.db"); _make_db(db, rows)
    monkeypatch.setattr(pf, "HERE", str(tmp_path))
    monkeypatch.setattr(pf, "returns_for",
                        lambda m, t: _returns_for_tickers(t))
    return db


def _returns_for_tickers(tickers):
    rng = np.random.default_rng(3)
    return pd.DataFrame({t: rng.normal(0.0005, 0.005 + 0.003 * i, 252)
                         for i, t in enumerate(sorted(tickers))})


def test_main_missing_db_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["portfolio.py", "--db", str(tmp_path / "nope.db")])
    with pytest.raises(SystemExit) as ei:
        pf.main()
    assert "dvm_composite.db" in str(ei.value.code)


def test_main_no_candidates_exits(tmp_path, monkeypatch):
    db = _setup_cli(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["portfolio.py", "--market", "KR", "--db", db])
    with pytest.raises(SystemExit) as ei:
        pf.main()
    assert str(ei.value.code) == "no GGG candidates for KR"


def test_main_end_to_end_prints_and_saves(tmp_path, monkeypatch, capsys):
    db = _setup_cli(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["portfolio.py", "--market", "US", "--n", "4",
                                      "--cap", "0.5", "--db", db])
    pf.main()
    out = capsys.readouterr().out
    assert "=== min_var portfolio — US, 4 names (cap 0.5) ===" in out
    assert "--- portfolio risk (trailing 1y) ---" in out
    assert "turnover vs prev" not in out                       # no --prev -> blank line only
    saved = json.loads((tmp_path / "portfolio_US_min_var.json").read_text())
    assert set(saved) == {"A", "B", "C", "E"}                  # the four US GGG names (D is GGB)
    assert sum(saved.values()) == pytest.approx(1.0, abs=1e-4)
    assert max(saved.values()) <= 0.5 + 1e-6
    # printed sector for each name comes from the candidate table
    for line in out.splitlines():
        if line.startswith("  A "):
            assert line.rstrip().endswith("Tech")
    for k in ("n", "ann_return%", "max_dd%", "CVaR95%", "regime"):
        assert f"    {k:<12} " in out


def test_main_n_limits_candidates(tmp_path, monkeypatch, capsys):
    db = _setup_cli(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["portfolio.py", "--market", "US", "--n", "2",
                                      "--cap", "0.6", "--db", db])
    pf.main()
    saved = json.loads((tmp_path / "portfolio_US_min_var.json").read_text())
    assert set(saved) == {"A", "C"}                            # top-2 composite among GGG


def test_main_prev_prints_turnover_and_max_sharpe(tmp_path, monkeypatch, capsys):
    db = _setup_cli(tmp_path, monkeypatch)
    prev = tmp_path / "prev.json"
    prev.write_text(json.dumps({"A": 0.5, "C": 0.5}))
    monkeypatch.setattr(sys, "argv", ["portfolio.py", "--market", "US", "--n", "4", "--cap", "0.5",
                                      "--method", "max_sharpe", "--sector-cap", "0.7",
                                      "--prev", str(prev), "--max-turnover", "0.2", "--db", db])
    pf.main()
    out = capsys.readouterr().out
    assert "max_sharpe portfolio — US" in out
    line = [ln for ln in out.splitlines() if "turnover vs prev:" in ln][0]
    assert float(line.split(":")[1]) <= 0.2 + 1e-2
    assert (tmp_path / "portfolio_US_max_sharpe.json").exists()


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="portfolio.py:68 feasibility uses n incl. zero-weight names, but spillover "
                          "(line 76) only goes to w>0 names: [0,.5,.3,.2] with cap .3 returns 0.35 > cap")
def test_cap_weights_respects_cap_when_zero_weight_names_present():
    w = pf.cap_weights(np.array([0.0, 0.5, 0.3, 0.2]), 0.3)
    assert w.max() <= 0.3 + 1e-9


def test_cap_weights_leaves_weights_exactly_at_the_cap_tolerance_untouched():
    import portfolio
    w = np.array([0.5 + 1e-12, 0.5 - 1e-12])
    out = portfolio.cap_weights(w, 0.5)
    assert (out == w).all()                      # w == cap + 1e-12 is not "over" (strict >)
