"""Offline tests for data_quality.py (pure rules, scans over fake parquet / temp sqlite, CLI)."""
import datetime as _dt
import os
import sqlite3
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_quality as dq  # noqa: E402

TODAY = _dt.date(2026, 1, 31)


@pytest.fixture
def frozen_today(monkeypatch):
    monkeypatch.setattr(dq, "dt", types.SimpleNamespace(
        date=types.SimpleNamespace(today=lambda: TODAY)))


# ── staleness_days ────────────────────────────────────────────────────────────
def test_staleness_days_uses_newest_date_and_explicit_asof():
    assert dq.staleness_days(["2026-01-01", "2026-01-20", "2026-01-10"], asof="2026-01-30") == 10
    assert dq.staleness_days(["2026-01-30"], asof="2026-01-30") == 0
    assert dq.staleness_days(["2026-02-02"], asof="2026-01-30") == -3     # future-dated data


def test_staleness_days_defaults_to_today(frozen_today):
    assert dq.staleness_days(pd.to_datetime(["2026-01-21"])) == 10
    assert dq.staleness_days(pd.Series(pd.to_datetime(["2026-01-31"]))) == 0


# ── null_rate ─────────────────────────────────────────────────────────────────
def test_null_rate_values():
    assert dq.null_rate([1, None, 3, np.nan]) == 0.5
    assert dq.null_rate([1, 2]) == 0.0
    assert dq.null_rate([np.nan]) == 1.0
    assert dq.null_rate([]) == 1.0                       # empty = fully missing
    assert dq.null_rate(pd.Series([1.0, np.nan, np.nan, np.nan])) == 0.75


# ── outlier_rate ──────────────────────────────────────────────────────────────
def test_outlier_rate_flags_absurd_points():
    v = [10, 11, 9, 10, 10, 11, 9, 10, 1000]
    assert dq.outlier_rate(v) == pytest.approx(1 / 9)
    assert dq.outlier_rate([10, 11, 9, 10, 10, 11, 9, 10, 12]) == 0.0


def test_outlier_rate_k_threshold_is_strict_and_tunable():
    v = [10, 11, 9, 10, 10, 11, 9, 10, 13]               # z of 13 = 3/1.4826 ~= 2.02
    assert dq.outlier_rate(v, k=2.0) == pytest.approx(1 / 9)
    assert dq.outlier_rate(v, k=2.1) == 0.0


def test_outlier_rate_small_samples_and_junk_values():
    assert dq.outlier_rate([1, 1, 1, 1000]) == 0.0       # < 5 finite values
    assert dq.outlier_rate([]) == 0.0
    assert dq.outlier_rate([1, 2, "x", None, np.nan]) == 0.0       # only 2 usable
    assert dq.outlier_rate(["10", "11", "9", "10", "10", "11", "9", "10", "1000"]) == pytest.approx(1 / 9)


def test_outlier_rate_zero_mad_falls_back_to_std():
    flat_plus_spike = [5.0] * 9 + [500.0]                # MAD = 0 -> uses std
    assert dq.outlier_rate(flat_plus_spike) == 0.0       # z = 495/(1.4826*148) ~ 2.3 < 8
    assert dq.outlier_rate([5.0] * 10) == 0.0            # std 0 -> divisor 1, all z = 0


# ── is_monotonic_dates ────────────────────────────────────────────────────────
def test_is_monotonic_dates():
    assert dq.is_monotonic_dates(["2024-01-01", "2024-01-02", "2024-01-03"]) is True
    assert dq.is_monotonic_dates(["2024-01-02", "2024-01-01"]) is False
    assert dq.is_monotonic_dates(["2024-01-01", "2024-01-01"]) is True      # ties allowed
    assert dq.is_monotonic_dates(["2024-01-01"]) is True
    assert dq.is_monotonic_dates([]) is True


# ── evaluate ──────────────────────────────────────────────────────────────────
def test_evaluate_operators_and_boundaries():
    rows = dq.evaluate("src", {"a": 5, "b": 5, "c": True, "d": 6, "e": 4, "f": False},
                       {"a": ("<=", 5), "b": (">=", 5), "c": ("==", True),
                        "d": ("<=", 5), "e": (">=", 5), "f": ("==", True)})
    assert [r["status"] for r in rows] == ["PASS", "PASS", "PASS", "FAIL", "FAIL", "FAIL"]
    assert rows[0] == {"source": "src", "rule": "a", "value": 5, "limit": "<=5", "status": "PASS"}
    assert rows[2]["limit"] == "==True"


def test_evaluate_missing_threshold_raises_and_empty_ok():
    with pytest.raises(KeyError):
        dq.evaluate("s", {"x": 1}, {})
    assert dq.evaluate("s", {}, {}) == []


# ── scan_parquet ──────────────────────────────────────────────────────────────
def _ohlc(last="2026-01-20", closes=None):
    dates = pd.to_datetime(["2026-01-18", "2026-01-19", last])
    return pd.DataFrame({
        "Symbol": ["A"] * 3, "Date": dates,
        "Close": closes if closes is not None else [10.0, 10.5, 11.0]})


def _scan(monkeypatch, df, stale=30, nul=0.2, out=0.01):
    monkeypatch.setattr(pd, "read_parquet", lambda p, *a, **k: df)
    return {r["rule"]: r for r in dq.scan_parquet("x.parquet", "US", stale, nul, out)}


def test_scan_parquet_all_pass(monkeypatch, frozen_today):
    rows = _scan(monkeypatch, _ohlc())
    assert set(rows) == {"stale_days", "close_null_rate", "close_outlier_rate",
                         "dates_sorted_within_symbol"}
    assert all(r["status"] == "PASS" for r in rows.values())
    assert rows["stale_days"]["value"] == 11 and rows["stale_days"]["source"] == "ohlc:US"
    assert rows["stale_days"]["limit"] == "<=30"


def test_scan_parquet_stale_boundary(monkeypatch, frozen_today):
    # newest date 2026-01-20 is exactly 11 days old
    assert _scan(monkeypatch, _ohlc(), stale=11)["stale_days"]["status"] == "PASS"
    assert _scan(monkeypatch, _ohlc(), stale=10)["stale_days"]["status"] == "FAIL"


def test_scan_parquet_null_and_outlier_thresholds(monkeypatch, frozen_today):
    rows = _scan(monkeypatch, _ohlc(closes=[10.0, np.nan, 11.0]), nul=0.2)
    assert rows["close_null_rate"]["status"] == "FAIL"
    assert rows["close_null_rate"]["value"] == pytest.approx(1 / 3)
    assert _scan(monkeypatch, _ohlc(closes=[10.0, np.nan, 11.0]), nul=1 / 3)[
        "close_null_rate"]["status"] == "PASS"
    n = 30
    big = pd.DataFrame({"Symbol": ["A"] * n, "Date": pd.date_range("2026-01-01", periods=n),
                        "Close": [10.0 + (i % 3) for i in range(n - 1)] + [1e6]})
    rows = _scan(monkeypatch, big, out=0.01)
    assert rows["close_outlier_rate"]["value"] == pytest.approx(1 / n)
    assert rows["close_outlier_rate"]["status"] == "FAIL"


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="data_quality.py:~84 sorts by (Symbol, Date) before checking monotonicity, "
                          "so dates_sorted_within_symbol can never FAIL")
def test_scan_parquet_unsorted_within_symbol_is_detected(monkeypatch, frozen_today):
    # Dates are stored out of order for symbol A; the rule is named
    # dates_sorted_within_symbol so it must FAIL.
    df = pd.DataFrame({"Symbol": ["A", "A", "A"],
                       "Date": pd.to_datetime(["2026-01-19", "2026-01-18", "2026-01-20"]),
                       "Close": [1.0, 2.0, 3.0]})
    rows = _scan(monkeypatch, df)
    assert rows["dates_sorted_within_symbol"]["status"] == "FAIL"


# ── scan_fundamentals ─────────────────────────────────────────────────────────
def _fund_db(path, rows, cols="roe, de, pe"):
    con = sqlite3.connect(path)
    con.execute(f"CREATE TABLE fund (symbol TEXT, {cols})")
    ph = ",".join("?" * (1 + len(cols.split(","))))
    con.executemany(f"INSERT INTO fund VALUES ({ph})", rows)
    con.commit()
    con.close()


def test_scan_fundamentals_pass_and_values(tmp_path):
    db = str(tmp_path / "f.db")
    rows = [("S%d" % i, 0.1 + i / 100, 1.0, 15.0 + i % 3) for i in range(10)]
    _fund_db(db, rows)
    res = {r["rule"]: r for r in dq.scan_fundamentals(db, 0.2)}
    assert res["roe_null_rate"]["value"] == 0.0 and res["roe_null_rate"]["status"] == "PASS"
    assert res["pe_outlier_rate"]["limit"] == "<=0.05" and res["pe_outlier_rate"]["status"] == "PASS"
    assert res["roe_null_rate"]["source"] == "fundamentals"


def test_scan_fundamentals_text_numbers_nulls_and_outlier(tmp_path):
    db = str(tmp_path / "f.db")
    rows = [("S%d" % i, "0.2" if i % 2 else None, "1", str(14 + i % 3)) for i in range(9)]
    rows.append(("OUT", "0.2", "1", "100000"))
    _fund_db(db, rows)
    res = {r["rule"]: r for r in dq.scan_fundamentals(db, 0.5)}
    assert res["roe_null_rate"]["value"] == pytest.approx(0.5)        # 5 of 10 null
    assert res["roe_null_rate"]["status"] == "PASS"                   # == limit passes
    assert dq.scan_fundamentals(db, 0.49)[0]["status"] == "FAIL"
    assert res["pe_outlier_rate"]["value"] == pytest.approx(0.1)
    assert res["pe_outlier_rate"]["status"] == "FAIL"


def test_scan_fundamentals_missing_columns(tmp_path):
    db = str(tmp_path / "f.db")
    _fund_db(db, [("A", 1.0)], cols="de")
    res = {r["rule"]: r for r in dq.scan_fundamentals(db, 0.5)}
    assert res["roe_null_rate"]["value"] == 1.0 and res["roe_null_rate"]["status"] == "FAIL"
    assert res["pe_outlier_rate"]["value"] == 0.0 and res["pe_outlier_rate"]["status"] == "PASS"


def test_scan_fundamentals_missing_table_raises_and_closes(tmp_path):
    db = str(tmp_path / "empty.db")
    sqlite3.connect(db).close()
    with pytest.raises(Exception):
        dq.scan_fundamentals(db, 0.2)


# ── CLI ───────────────────────────────────────────────────────────────────────
@pytest.fixture
def env(tmp_path, monkeypatch, frozen_today):
    seed = tmp_path / "seed"
    seed.mkdir()
    monkeypatch.setattr(dq, "SEED", str(seed))
    monkeypatch.setattr(dq, "HERE", str(tmp_path))
    frames = {}

    def read_parquet(p, *a, **k):
        name = os.path.basename(p)
        if name not in frames:
            raise OSError(f"cannot read {name}")
        return frames[name]
    monkeypatch.setattr(pd, "read_parquet", read_parquet)

    def add(market, df=None):
        name = f"cleaned_long_{market}.parquet"
        (seed / name).write_bytes(b"")
        if df is not None:
            frames[name] = df
    return types.SimpleNamespace(seed=seed, root=tmp_path, add=add)


def _run(monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, "argv", ["data_quality.py", *argv])
    code = None
    try:
        dq.main()
    except SystemExit as e:
        code = e.code
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def test_main_all_pass_exits_zero_and_reports_counts(env, monkeypatch, capsys):
    env.add("US", _ohlc())
    code, out, _ = _run(monkeypatch, capsys)
    assert code == 0
    assert "=== DATA QUALITY REPORT ===" in out
    assert "4/4 checks PASS; 0 FAIL" in out
    assert "ohlc:US" in out


def test_main_failure_exits_one_and_flags_unreadable_market(env, monkeypatch, capsys):
    env.add("US", _ohlc())
    env.add("JP")                                         # file exists, read raises
    code, out, _ = _run(monkeypatch, capsys)
    assert code == 1
    assert "ohlc:JP" in out and "readable" in out and "cannot read cleaned_long_JP.parquet" in out
    assert "4/5 checks PASS; 1 FAIL" in out


def test_main_cli_thresholds_are_applied(env, monkeypatch, capsys):
    env.add("US", _ohlc())                                # 11 days stale
    code, out, _ = _run(monkeypatch, capsys, "--max-stale-days", "10")
    assert code == 1 and "3/4 checks PASS; 1 FAIL" in out
    code, out, _ = _run(monkeypatch, capsys, "--max-stale-days", "11")
    assert code == 0
    env.add("US", _ohlc(closes=[1.0, np.nan, 1.0]))
    code, out, _ = _run(monkeypatch, capsys, "--max-null-rate", "0.5")
    assert code == 0
    code, out, _ = _run(monkeypatch, capsys, "--max-null-rate", "0.3")
    assert code == 1


def test_main_includes_fundamentals_db_when_present(env, monkeypatch, capsys):
    env.add("US", _ohlc())
    _fund_db(str(env.root / "fundamentals_cache.db"),
             [("S%d" % i, 0.1, 1.0, 15.0) for i in range(6)])
    code, out, _ = _run(monkeypatch, capsys)
    assert code == 0 and "fundamentals" in out and "6/6 checks PASS" in out


def test_main_unreadable_fundamentals_db_is_a_fail_row(env, monkeypatch, capsys):
    env.add("US", _ohlc())
    sqlite3.connect(str(env.root / "fundamentals_cache.db")).close()   # no `fund` table
    code, out, _ = _run(monkeypatch, capsys)
    assert code == 1
    assert "fundamentals" in out and "readable" in out and "FAIL" in out


def test_main_no_sources_prints_to_stderr_and_returns(env, monkeypatch, capsys):
    code, out, err = _run(monkeypatch, capsys)
    assert code is None and out == ""
    assert err == "no data sources found to check\n"
