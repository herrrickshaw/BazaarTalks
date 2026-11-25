"""Offline tests for ml_viability.py (cross-market walk-forward viability of the ML-bullish screen)."""
import concurrent.futures
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import feature_cache  # noqa: E402
import ml_viability as mv  # noqa: E402


def _ohlcv(n=900, drift=0.001, amp=0.01, seed=0, price0=100.0, vol=1e6):
    rng = np.random.default_rng(seed)
    close = price0 * np.cumprod(1 + drift + amp * rng.standard_normal(n))
    return pd.DataFrame({
        "High": close * 1.01, "Low": close * 0.99, "Close": close,
        "Volume": vol * (1 + 0.1 * np.abs(rng.standard_normal(n))),
    }, index=pd.bdate_range("2018-01-01", periods=n))


@pytest.fixture(autouse=True)
def mem_cache(monkeypatch, tmp_path):
    """In-memory stand-in for the parquet feature cache (pyarrow is not a CI dependency, and the real
    cache would otherwise write next to the source)."""
    store = {}
    monkeypatch.setattr(feature_cache, "CACHE_DIR", str(tmp_path / "fc"))
    monkeypatch.setattr(feature_cache, "load", lambda key: store.get(key))
    monkeypatch.setattr(feature_cache, "save", lambda key, feats: store.__setitem__(key, feats) or key)
    return store


# ── constants / universes ─────────────────────────────────────────────────────
def test_market_universes_shape():
    assert list(mv.MARKET_UNIVERSES) == ["US", "India", "Japan", "Korea", "Europe"]
    assert all(len(set(v)) == len(v) and v for v in mv.MARKET_UNIVERSES.values())
    assert all(t.endswith(".NS") for t in mv.MARKET_UNIVERSES["India"])
    assert all(t.endswith(".T") for t in mv.MARKET_UNIVERSES["Japan"])
    assert all(t.endswith(".KS") for t in mv.MARKET_UNIVERSES["Korea"])


def test_min_needed_matches_engine_constants():
    assert mv.MIN_NEEDED == mv.LOOKBACK + mv.TRAIN_WINDOW + mv.PREDICT_DAYS + 250 == 567


# ── download_ohlc ─────────────────────────────────────────────────────────────
def test_download_ohlc_filters_by_length_and_none(monkeypatch):
    seen = {}
    ms = types.ModuleType("market_store")

    def cached_download(tickers, years):
        seen["args"] = (list(tickers), years)
        return {"OK": _ohlcv(mv.MIN_NEEDED + 1), "EDGE": _ohlcv(mv.MIN_NEEDED),
                "SHORT": _ohlcv(100), "NONE": None}

    ms.cached_download = cached_download
    monkeypatch.setitem(sys.modules, "market_store", ms)
    out = mv.download_ohlc(["OK", "EDGE", "SHORT", "NONE"], 7)
    assert set(out) == {"OK"}                  # len must be strictly greater than MIN_NEEDED
    assert seen["args"] == (["OK", "EDGE", "SHORT", "NONE"], 7)


# ── _eval_ticker_wf ───────────────────────────────────────────────────────────
def test_wf_short_history_returns_zero_tuple():
    assert mv._eval_ticker_wf(("X", _ohlcv(400), 5, "ridge")) == (0, 0, [], [], [], [], 0, 0)


def test_wf_failure_is_swallowed_and_returns_zeros():
    bad = _ohlcv(900).drop(columns=["High"])
    assert mv._eval_ticker_wf(("X", bad, 5, "ridge")) == (0, 0, [], [], [], [], 0, 0)


def test_wf_internal_consistency():
    nc, nt, act, pred, bull, allr, bh, bn = mv._eval_ticker_wf(("X", _ohlcv(900, seed=3), 20, "ridge"))
    assert nt > 5 and len(act) == len(pred) == len(allr) == nt
    assert act == allr
    assert nc == sum((p > 0 and a > 0) or (p < 0 and a < 0) for p, a in zip(pred, act))
    assert bull == [a for p, a in zip(pred, act) if p >= mv.BULLISH_THRESHOLD]
    assert bn == len(bull) and bh == sum(a > 0 for a in bull)
    assert 0 <= nc <= nt and 0 <= bh <= bn <= nt


def test_wf_step_controls_number_of_test_points():
    df = _ohlcv(900, seed=3)
    n20 = mv._eval_ticker_wf(("X", df, 20, "ridge"))[1]
    n40 = mv._eval_ticker_wf(("X", df, 40, "ridge"))[1]
    assert n20 > n40 >= 1
    assert abs(n20 - 2 * n40) <= 2


def test_wf_is_deterministic_and_model_type_is_used():
    df = _ohlcv(900, seed=4)
    a = mv._eval_ticker_wf(("X", df, 30, "ridge"))
    b = mv._eval_ticker_wf(("X", df, 30, "ridge"))
    assert a == b
    c = mv._eval_ticker_wf(("X", df, 30, "lr"))
    assert c[1] == a[1] and c[3] != a[3]        # same test points, different fitted model -> different preds


def test_wf_feature_cache_is_used_on_second_call(monkeypatch, mem_cache):
    calls = []
    real = mv.compute_features
    monkeypatch.setattr(mv, "compute_features", lambda d: (calls.append(1), real(d))[1])
    df = _ohlcv(900, seed=5)
    r1 = mv._eval_ticker_wf(("CACHE_ME", df, 50, "ridge"))
    r2 = mv._eval_ticker_wf(("CACHE_ME", df, 50, "ridge"))
    assert len(calls) == 1 and r1 == r2
    assert len(mem_cache) == 1 and next(iter(mem_cache)).startswith("CACHE_ME__")
    mv._eval_ticker_wf(("CACHE_ME", df.iloc[:-1], 50, "ridge"))        # changed data -> recomputed
    assert len(calls) == 2 and len(mem_cache) == 2


def test_wf_uptrending_stock_has_positive_realised_returns_recorded():
    _, nt, act, *_ = mv._eval_ticker_wf(("UP", _ohlcv(900, drift=0.004, amp=0.004, seed=6), 20, "ridge"))
    assert nt > 0 and np.mean(act) > 0


# ── evaluate_market ───────────────────────────────────────────────────────────
class _SyncPool:
    created = []

    def __init__(self, max_workers=None):
        _SyncPool.created.append(max_workers)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def map(self, fn, tasks):
        return [fn(t) for t in tasks]


@pytest.fixture
def pool(monkeypatch):
    _SyncPool.created = []
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _SyncPool)
    return _SyncPool


def _canned(monkeypatch, per_ticker):
    seen = []

    def fake(args):
        seen.append(args)
        return per_ticker[args[0]]
    monkeypatch.setattr(mv, "_eval_ticker_wf", fake)
    return seen


class _Eng:
    model_type = "lr"


def test_evaluate_market_metrics_exact(monkeypatch, pool):
    # ticker A: preds [1, -1, 2], actuals [2, -3, -1]  -> 2 correct of 3 ; bull = pred>=0.5 -> actuals [2, -1]
    A = (2, 3, [2.0, -3.0, -1.0], [1.0, -1.0, 2.0], [2.0, -1.0], [2.0, -3.0, -1.0], 1, 2)
    B = (1, 1, [4.0], [0.2], [], [4.0], 0, 0)
    seen = _canned(monkeypatch, {"A": A, "B": B})
    df = _ohlcv(10)
    row = mv.evaluate_market("US", {"A": df, "B": df}, _Eng(), step=7, workers=3)
    act = np.array([2, -3, -1, 4.0]); pred = np.array([1, -1, 2, 0.2])
    assert row["Market"] == "US" and row["n_stocks"] == 2 and row["n_preds"] == 4
    assert row["dir_acc%"] == 75.0
    assert row["rmse"] == round(float(np.sqrt(np.mean((act - pred) ** 2))), 3)
    assert row["mae"] == round(float(np.mean(np.abs(act - pred))), 3)
    assert row["base_fwd%"] == round(float(act.mean()), 3) == 0.5
    assert row["bull_fwd%"] == 0.5                      # mean([2, -1])
    assert row["edge%"] == 0.0
    assert row["bull_hit%"] == 50.0 and row["bull_n"] == 2
    assert row["VIABLE"] == "no"                        # edge not > 0
    assert pool.created == [3]
    assert [a[0] for a in seen] == ["A", "B"] and all(a[2:] == (7, "lr") for a in seen)


def test_evaluate_market_viable_requires_all_three_strictly(monkeypatch, pool):
    df = _ohlcv(10)

    def row(nc, nt, bull_ret, base_ret, bh, bn):
        _canned(monkeypatch, {"A": (nc, nt, [1.0] * nt, [1.0] * nt, bull_ret, base_ret, bh, bn)})
        return mv.evaluate_market("M", {"A": df}, _Eng(), 5)

    good = row(6, 10, [3.0, 3.0, 3.0, -1.0], [1.0] * 10, 3, 4)               # acc 60, hit 75, edge 2
    assert good["VIABLE"] == "YES" and good["edge%"] == 1.0 and good["bull_hit%"] == 75.0
    assert row(5, 10, [3.0] * 4, [1.0] * 10, 3, 4)["VIABLE"] == "no"        # dir acc exactly 50
    assert row(6, 10, [3.0] * 4, [1.0] * 10, 2, 4)["VIABLE"] == "no"        # hit exactly 50
    assert row(6, 10, [1.0] * 4, [1.0] * 10, 3, 4)["VIABLE"] == "no"         # edge exactly 0
    assert row(6, 10, [0.5] * 4, [1.0] * 10, 3, 4)["VIABLE"] == "no"         # negative edge


def test_evaluate_market_no_bull_calls(monkeypatch, pool):
    _canned(monkeypatch, {"A": (3, 4, [1.0, 2.0, -1.0, 3.0], [1.0, 1.0, 1.0, 1.0], [], [1.0, 2.0, -1.0, 3.0], 0, 0)})
    r = mv.evaluate_market("M", {"A": _ohlcv(10)}, _Eng(), 5)
    assert r["bull_fwd%"] == 0.0 and r["bull_hit%"] == 0.0 and r["bull_n"] == 0
    assert r["edge%"] == -round(float(np.mean([1.0, 2.0, -1.0, 3.0])), 3)
    assert r["VIABLE"] == "no"


def test_evaluate_market_no_predictions_gives_na_row(monkeypatch, pool):
    _canned(monkeypatch, {"A": (0, 0, [], [], [], [], 0, 0)})
    assert mv.evaluate_market("M", {"A": _ohlcv(10)}, _Eng(), 5) == \
        {"Market": "M", "n_stocks": 1, "n_preds": 0, "VIABLE": "n/a"}


def test_evaluate_market_empty_ohlc(pool):
    assert mv.evaluate_market("M", {}, _Eng(), 5) == {"Market": "M", "n_stocks": 0, "n_preds": 0, "VIABLE": "n/a"}


def test_evaluate_market_end_to_end_with_real_walk_forward(pool):
    ohlc = {"A": _ohlcv(900, seed=1, drift=0.002), "B": _ohlcv(900, seed=2, drift=0.002)}
    r = mv.evaluate_market("T", ohlc, mv.MLSignalEngine(model_type="ridge"), step=40)
    assert r["n_stocks"] == 2 and r["n_preds"] > 0
    assert 0 <= r["dir_acc%"] <= 100 and r["rmse"] >= r["mae"] >= 0
    assert r["VIABLE"] in ("YES", "no")
    assert r["bull_n"] >= 0


# ── main ──────────────────────────────────────────────────────────────────────
@pytest.fixture
def run_main(monkeypatch, tmp_path):
    def _run(*argv, rows=None):
        calls = {"dl": [], "ev": []}

        def dl(tickers, years):
            calls["dl"].append((list(tickers), years))
            return {t: None for t in tickers}

        def ev(name, ohlc, engine, step, workers=None):
            calls["ev"].append((name, sorted(ohlc), engine.model_type, step, workers))
            return (rows or {}).get(name, {"Market": name, "n_stocks": len(ohlc), "n_preds": 10,
                                           "dir_acc%": 55.0, "VIABLE": "YES"})

        monkeypatch.setattr(mv, "download_ohlc", dl)
        monkeypatch.setattr(mv, "evaluate_market", ev)
        out = tmp_path / "out.xlsx"
        monkeypatch.setattr(sys, "argv", ["ml_viability.py", "--out", str(out), *argv])
        mv.main()
        return out, calls
    return _run


@pytest.fixture
def csv_only(monkeypatch):
    def boom(self, *a, **k):
        raise ImportError("no openpyxl")
    monkeypatch.setattr(pd.DataFrame, "to_excel", boom)


def test_main_all_markets_by_default_with_defaults(run_main, csv_only, capsys):
    out, calls = run_main()
    assert [c[0] for c in calls["ev"]] == list(mv.MARKET_UNIVERSES)
    assert [t for t, _ in calls["dl"]] == list(mv.MARKET_UNIVERSES.values())
    assert all(y == 5 for _, y in calls["dl"])
    assert all(c[2:] == ("ridge", 5, os.cpu_count()) for c in calls["ev"])
    cap = capsys.readouterr()
    assert "[US] downloading 12 tickers (5y)" in cap.err
    assert "5-YEAR ML SCREEN VIABILITY BY MARKET" in cap.out and "VIABLE = edge>0" in cap.out


def test_main_options_forwarded(run_main, csv_only):
    out, calls = run_main("--markets", "Japan", "US", "--years", "3", "--step", "10",
                          "--top", "2", "--model", "lr", "--workers", "2")
    assert [c[0] for c in calls["ev"]] == ["Japan", "US"]
    assert calls["dl"] == [(mv.MARKET_UNIVERSES["Japan"][:2], 3), (mv.MARKET_UNIVERSES["US"][:2], 3)]
    assert all(c[2:] == ("lr", 10, 2) for c in calls["ev"])


def test_main_unknown_market_skipped(run_main, csv_only, capsys):
    out, calls = run_main("--markets", "Mars", "US", "--top", "1")
    assert [c[0] for c in calls["ev"]] == ["US"]
    assert "[skip] unknown market Mars" in capsys.readouterr().err


def test_main_falls_back_to_csv_when_excel_unavailable(run_main, csv_only, tmp_path):
    out, _ = run_main("--markets", "US", "India", "--top", "1")
    assert not out.exists()
    df = pd.read_csv(tmp_path / "out.csv")
    assert list(df["Market"]) == ["US", "India"] and list(df["VIABLE"]) == ["YES", "YES"]


def test_main_writes_excel_when_available(run_main, monkeypatch):
    written = {}
    monkeypatch.setattr(pd.DataFrame, "to_excel", lambda self, path, index=True: written.update(path=path, index=index, n=len(self)))
    out, _ = run_main("--markets", "US", "--top", "1")
    assert written == {"path": str(out), "index": False, "n": 1}


def test_main_rejects_unknown_model(run_main):
    with pytest.raises(SystemExit):
        run_main("--model", "xgb")


def test_main_prints_result_rows(run_main, csv_only, capsys):
    run_main("--markets", "US", "--top", "1", rows={"US": {"Market": "US", "n_stocks": 1, "n_preds": 7,
                                                        "dir_acc%": 61.5, "VIABLE": "no"}})
    out = capsys.readouterr().out
    assert "61.5" in out and "n_preds" in out
