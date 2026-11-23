"""Offline tests for ml_signal_engine (features, sequencing, walk-forward signal, batch,
backtest evaluation, model comparison). Synthetic seeded OHLC only; no network."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ml_signal_engine as ml  # noqa: E402
from ml_signal_engine import (FEATURE_NAMES, LOOKBACK, MIN_ROWS, PREDICT_DAYS,  # noqa: E402
                              TRAIN_WINDOW, MLSignalEngine, compare_models,
                              compute_features, z_score_normalise)


def make_df(n, seed=0):
    r = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2025-12-31", periods=n)
    c = 100 * np.exp(np.cumsum(r.normal(0.0003, 0.012, n)))
    return pd.DataFrame({
        "Open": c,
        "High": c * (1 + np.abs(r.normal(0, 0.005, n))),
        "Low": c * (1 - np.abs(r.normal(0, 0.005, n))),
        "Close": c,
        "Volume": 1e6 * (1 + np.abs(r.normal(0, 0.3, n))) + 1,
    }, index=idx)


@pytest.fixture(scope="module")
def big():
    return make_df(600)


class StubModel:
    """Constant-output regressor so thresholds / confidence are testable exactly."""

    def __init__(self, c=0.0, fit_exc=None):
        self.c, self.fit_exc = c, fit_exc

    def fit(self, X, y):
        if self.fit_exc:
            raise self.fit_exc
        return self

    def predict(self, X):
        return np.full(len(X), self.c, dtype=float)


def stub_engine(monkeypatch, c, **kw):
    eng = MLSignalEngine(**kw)
    monkeypatch.setattr(eng, "_make_model", lambda: StubModel(c))
    return eng


# ── constants ────────────────────────────────────────────────────────────────
def test_constants_and_feature_list():
    assert (LOOKBACK, TRAIN_WINDOW, PREDICT_DAYS) == (60, 252, 5)
    assert MIN_ROWS == 60 + 252 + 5 + 10
    assert len(FEATURE_NAMES) == 12 and len(set(FEATURE_NAMES)) == 12


# ── compute_features ─────────────────────────────────────────────────────────
def test_compute_features_shape_columns_and_no_nan(big):
    f = compute_features(big)
    assert list(f.columns) == FEATURE_NAMES
    assert len(f) == 600 - 199                      # 200-DMA is the binding warm-up
    assert f.index[0] == big.index[199] and f.index[-1] == big.index[-1]
    assert not f.isna().any().any()


def test_compute_features_matches_reference_formulas(big):
    f = compute_features(big)
    c, h, lo, v = big["Close"], big["High"], big["Low"], big["Volume"]
    last = f.iloc[-1]
    assert last["ret_1d"] == pytest.approx((c.iloc[-1] / c.iloc[-2] - 1) * 100)
    assert last["ret_5d"] == pytest.approx((c.iloc[-1] / c.iloc[-6] - 1) * 100)
    assert last["vol_20d"] == pytest.approx(c.pct_change().iloc[-20:].std() * np.sqrt(252) * 100)
    assert last["hl_ratio"] == pytest.approx((h.iloc[-1] - lo.iloc[-1]) / c.iloc[-1])
    assert last["vol_ratio"] == pytest.approx(v.iloc[-1] / v.iloc[-20:].mean())
    m20, s20 = c.iloc[-20:].mean(), c.iloc[-20:].std()
    assert last["close_norm"] == pytest.approx((c.iloc[-1] - m20) / s20)
    assert last["bb_pct"] == pytest.approx((c.iloc[-1] - (m20 - 2 * s20)) / (4 * s20))
    assert last["dma50_gap"] == pytest.approx((c.iloc[-1] / c.iloc[-50:].mean() - 1) * 100)
    assert last["dma200_gap"] == pytest.approx((c.iloc[-1] / c.iloc[-200:].mean() - 1) * 100)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    assert last["macd"] == pytest.approx(macd.iloc[-1])
    assert last["macd_signal"] == pytest.approx(macd.ewm(span=9, adjust=False).mean().iloc[-1])
    d = c.diff()
    gain, loss = d.clip(lower=0).iloc[-14:].mean(), (-d.clip(upper=0)).iloc[-14:].mean()
    assert last["rsi_14"] == pytest.approx(100 - 100 / (1 + gain / loss))
    assert 0 <= f["rsi_14"].min() and f["rsi_14"].max() <= 100


def test_compute_features_zero_volume_bar_blanks_the_next_20_rows(big):
    d = big.copy()
    d.iloc[400, d.columns.get_loc("Volume")] = 0.0
    f = compute_features(d)
    assert len(f) == 600 - 199 - 20
    assert big.index[400] not in f.index and big.index[419] not in f.index
    assert big.index[420] in f.index


def test_compute_features_missing_column_raises(big):
    with pytest.raises(KeyError):
        compute_features(big.drop(columns=["Volume"]))


def test_compute_features_integer_input_columns_are_cast(big):
    d = big.copy()
    d["Volume"] = d["Volume"].astype(int)
    d["Close"] = d["Close"].round(0).astype(int).clip(lower=1)
    assert len(compute_features(d)) > 0


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="ml_signal_engine.py:rsi `rs = gain / loss.replace(0, nan)` makes RSI NaN "
                          "(not 100) when a 14-day window has no down days, so a steadily rising "
                          "series loses every feature row")
def test_compute_features_monotonic_uptrend_keeps_rows():
    idx = pd.bdate_range("2023-01-02", periods=300)
    c = 100 + np.arange(300) * 0.5
    d = pd.DataFrame({"Open": c, "High": c + 1, "Low": c - 1, "Close": c, "Volume": 1000.0}, index=idx)
    assert len(compute_features(d)) > 0


# ── z_score_normalise ────────────────────────────────────────────────────────
def test_z_score_known_values_and_zero_variance_column():
    X = np.array([[1.0, 2.0], [3.0, 2.0]])
    out = z_score_normalise(X)
    assert out.tolist() == [[-1.0, 0.0], [1.0, 0.0]]       # constant column -> 0, not NaN


def test_z_score_zero_mean_unit_std_and_input_untouched():
    rng = np.random.default_rng(1)
    X = rng.normal(5, 3, (40, 4))
    X0 = X.copy()
    out = z_score_normalise(X)
    assert np.allclose(out.mean(axis=0), 0) and np.allclose(out.std(axis=0), 1)
    assert np.array_equal(X, X0)


def test_z_score_single_row_is_all_zero():
    assert z_score_normalise(np.array([[4.0, -2.0]])).tolist() == [[0.0, 0.0]]


# ── _make_model ──────────────────────────────────────────────────────────────
def test_make_model_types_and_params():
    from sklearn.linear_model import Ridge
    r = MLSignalEngine("ridge")._make_model()
    assert isinstance(r, Ridge) and r.alpha == 1.0 and r.fit_intercept is True
    lr = MLSignalEngine("lr")._make_model()
    assert type(lr) is LinearRegression and lr.fit_intercept is True
    assert type(MLSignalEngine("anything-else")._make_model()) is LinearRegression
    assert MLSignalEngine._make_model(MLSignalEngine()) is not MLSignalEngine._make_model(MLSignalEngine())


def test_make_model_without_sklearn_returns_none(monkeypatch):
    monkeypatch.setattr(ml, "_SKLEARN_OK", False)
    assert MLSignalEngine()._make_model() is None


def test_engine_defaults():
    e = MLSignalEngine()
    assert (e.model_type, e.workers) == ("ridge", 4)


# ── _make_sequences ──────────────────────────────────────────────────────────
def _seq_inputs(n, nf=3):
    rng = np.random.default_rng(3)
    feats = pd.DataFrame(rng.normal(size=(n, nf)), columns=list("abc")[:nf])
    tgt = pd.Series(np.arange(n, dtype=float))
    return feats, tgt


def test_make_sequences_count_shape_and_target_alignment():
    feats, tgt = _seq_inputs(100)
    X, y = MLSignalEngine()._make_sequences(feats, tgt)
    assert X.shape == (100 - LOOKBACK - PREDICT_DAYS, LOOKBACK * 3)
    assert y[0] == LOOKBACK + PREDICT_DAYS                  # target[i + 5] for i = 60
    assert y[-1] == 99
    np.testing.assert_allclose(X[0].reshape(LOOKBACK, 3), z_score_normalise(feats.values[:LOOKBACK]))
    np.testing.assert_allclose(X[-1].reshape(LOOKBACK, 3),
                               z_score_normalise(feats.values[34:94]))


def test_make_sequences_boundary_minimum_length():
    eng = MLSignalEngine()
    X, y = eng._make_sequences(*_seq_inputs(LOOKBACK + PREDICT_DAYS))      # 65 -> none
    assert X.size == 0 and y.size == 0
    X, y = eng._make_sequences(*_seq_inputs(LOOKBACK + PREDICT_DAYS + 1))  # 66 -> one
    assert X.shape[0] == 1 and y.tolist() == [65.0]


def test_make_sequences_empty_frame():
    X, y = MLSignalEngine()._make_sequences(pd.DataFrame({"a": []}), pd.Series([], dtype=float))
    assert X.size == 0 and y.size == 0


# ── predict ──────────────────────────────────────────────────────────────────
def test_predict_none_and_short_inputs_are_insufficient(big):
    eng = MLSignalEngine("lr")
    expected_base = {"symbol": "S", "direction": "INSUFFICIENT_DATA", "predicted_ret%": 0.0,
                     "confidence": 0.0, "model": "lr", "train_rmse": None, "train_rows": 0}
    assert eng.predict("S", None) == expected_base
    assert eng.predict("S", big.iloc[:MIN_ROWS - 1]) == expected_base
    # enough raw rows, but too few feature rows after the 200-DMA warm-up
    assert eng.predict("S", big.iloc[:MIN_ROWS]) == expected_base


def test_predict_feature_row_boundary_475_vs_476():
    eng = MLSignalEngine()
    assert eng.predict("S", make_df(475))["direction"] == "INSUFFICIENT_DATA"
    out = eng.predict("S", make_df(476))
    assert out["direction"] in ("BULLISH", "NEUTRAL", "BEARISH")
    assert "error" not in out and out["train_rows"] == 187


def test_predict_without_sklearn_is_insufficient(big, monkeypatch):
    monkeypatch.setattr(ml, "_SKLEARN_OK", False)
    assert MLSignalEngine().predict("S", big)["direction"] == "INSUFFICIENT_DATA"


def test_predict_real_model_result_shape_and_determinism(big):
    a = MLSignalEngine("ridge").predict("RELI.NS", big)
    b = MLSignalEngine("ridge").predict("RELI.NS", big.copy())
    assert a == b
    assert set(a) == {"symbol", "direction", "predicted_ret%", "confidence", "model",
                      "train_rmse", "train_rows", "features_used"}
    assert a["symbol"] == "RELI.NS" and a["model"] == "ridge"
    assert a["features_used"] == 12 and a["train_rows"] == TRAIN_WINDOW - LOOKBACK - PREDICT_DAYS
    assert a["train_rmse"] > 0 and 0.0 <= a["confidence"] <= 1.0
    assert a["predicted_ret%"] == round(a["predicted_ret%"], 3)
    lr = MLSignalEngine("lr").predict("RELI.NS", big)
    assert lr["model"] == "lr" and lr["predicted_ret%"] != a["predicted_ret%"]


@pytest.mark.parametrize("c, direction", [
    (0.5, "BULLISH"),          # >= threshold is inclusive
    (0.4999, "NEUTRAL"),
    (0.0, "NEUTRAL"),
    (-0.4999, "NEUTRAL"),
    (-0.5, "BEARISH"),         # <= threshold is inclusive
    (7.0, "BULLISH"),
    (-7.0, "BEARISH"),
])
def test_predict_direction_thresholds(big, monkeypatch, c, direction):
    out = stub_engine(monkeypatch, c).predict("S", big)
    assert out["direction"] == direction
    assert out["predicted_ret%"] == round(c, 3)


def test_predict_confidence_constant_model_uses_unit_std_and_caps_at_one(big, monkeypatch):
    # constant training predictions -> pred_std falls back to 1.0 -> |c| / 2
    assert stub_engine(monkeypatch, 1.0).predict("S", big)["confidence"] == 0.5
    assert stub_engine(monkeypatch, -0.3).predict("S", big)["confidence"] == 0.15
    assert stub_engine(monkeypatch, 2.0).predict("S", big)["confidence"] == 1.0
    assert stub_engine(monkeypatch, 50.0).predict("S", big)["confidence"] == 1.0   # capped
    assert stub_engine(monkeypatch, 0.0).predict("S", big)["confidence"] == 0.0


def test_predict_confidence_uses_training_prediction_std(big, monkeypatch):
    class Ramp:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.array([4.0]) if len(X) == 1 else np.arange(len(X), dtype=float)

    eng = MLSignalEngine()
    monkeypatch.setattr(eng, "_make_model", lambda: Ramp())
    out = eng.predict("S", big)
    std = np.arange(out["train_rows"], dtype=float).std()
    assert out["confidence"] == round(4.0 / (2 * std + 1e-9), 3)
    assert out["predicted_ret%"] == 4.0


def test_predict_train_rmse_and_rows_from_independent_window(big, monkeypatch):
    c = 0.25
    out = stub_engine(monkeypatch, c).predict("S", big)
    feats = compute_features(big)
    close = big["Close"].reindex(feats.index)
    tgt = close.pct_change(PREDICT_DAYS).shift(-PREDICT_DAYS) * 100
    al = feats.join(tgt.rename("target"), how="inner").dropna()
    end = len(al) - PREDICT_DAYS - 1
    X, y = MLSignalEngine()._make_sequences(al[FEATURE_NAMES].iloc[end - TRAIN_WINDOW:end],
                                            al["target"].iloc[end - TRAIN_WINDOW:end])
    assert out["train_rows"] == len(X) == 187
    assert out["train_rmse"] == round(float(np.sqrt(np.mean((y - c) ** 2))), 4)


def test_predict_exception_returns_base_with_error(big, monkeypatch):
    eng = MLSignalEngine()
    monkeypatch.setattr(eng, "_make_model", lambda: StubModel(fit_exc=RuntimeError("fit exploded")))
    out = eng.predict("S", big)
    assert out["error"] == "fit exploded"
    assert out["direction"] == "NEUTRAL" and out["predicted_ret%"] == 0.0 and out["train_rows"] == 0


def test_predict_missing_column_is_reported_not_raised(big):
    out = MLSignalEngine().predict("S", big.drop(columns=["High"]))
    assert "error" in out and out["direction"] == "NEUTRAL"


def test_predict_has_no_lookahead_on_final_bars(big):
    base = MLSignalEngine().predict("S", big)
    tampered = big.copy()
    tampered.iloc[-PREDICT_DAYS:, :] *= 1.5          # last 5 bars are not used by any feature/target
    assert MLSignalEngine().predict("S", tampered) == base
    sensitive = big.copy()
    sensitive.iloc[-PREDICT_DAYS - 1:, :] *= 1.5     # the bar 6 from the end IS the latest window
    assert MLSignalEngine().predict("S", sensitive)["predicted_ret%"] != base["predicted_ret%"]


def test_predict_ignores_regime_older_than_training_window(big):
    base = MLSignalEngine().predict("S", big)
    old = big.copy()
    rng = np.random.default_rng(9)
    noise = np.exp(rng.normal(0, 0.05, 150))
    for col in ("Open", "High", "Low", "Close"):
        old.iloc[:150, old.columns.get_loc(col)] *= noise
    again = MLSignalEngine().predict("S", old)
    assert again["predicted_ret%"] == pytest.approx(base["predicted_ret%"], abs=1e-3)
    assert again["train_rows"] == base["train_rows"]


# ── predict_batch ────────────────────────────────────────────────────────────
class _FixedEngine(MLSignalEngine):
    def __init__(self, table, **kw):
        super().__init__(**kw)
        self.table = table

    def predict(self, symbol, df):
        v = self.table[symbol]
        if isinstance(v, Exception):
            raise v
        return {"symbol": symbol, "direction": v[0], "predicted_ret%": v[1],
                "confidence": 0.1, "model": self.model_type}


def test_predict_batch_sorts_descending_and_summarises(capsys):
    eng = _FixedEngine({"A": ("BULLISH", 1.5), "B": ("BEARISH", -2.0), "C": ("NEUTRAL", 0.1),
                        "D": ("BULLISH", 3.0)})
    out = eng.predict_batch({s: None for s in "ABCD"}, max_workers=1)
    assert out["symbol"].tolist() == ["D", "A", "C", "B"]
    txt = capsys.readouterr().out
    assert "ML signal engine: predicting 4 stocks (ridge, 60d window, 252d train)" in txt
    assert "4/4 done | BULLISH: 2 | BEARISH: 1" in txt
    assert "ML signals computed: 4 stocks" in txt
    assert "BULLISH" in txt and "50.0%" in txt and "25.0%" in txt


def test_predict_batch_worker_exception_becomes_error_row(capsys):
    eng = _FixedEngine({"OK": ("BULLISH", 1.0), "BAD": RuntimeError("thread died")})
    out = eng.predict_batch({"OK": None, "BAD": None})
    bad = out[out["symbol"] == "BAD"].iloc[0]
    assert bad["direction"] == "ERROR" and bad["error"] == "thread died"
    assert bad["predicted_ret%"] == 0.0 and bad["model"] == "ridge"
    assert out["symbol"].tolist() == ["OK", "BAD"]


def test_predict_batch_progress_every_hundred(capsys):
    table = {f"S{i}": ("NEUTRAL", float(i)) for i in range(101)}
    out = _FixedEngine(table).predict_batch({k: None for k in table}, max_workers=1)
    assert len(out) == 101 and out["predicted_ret%"].is_monotonic_decreasing
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "done |" in ln]
    assert [ln.split()[0] for ln in lines] == ["100/101", "101/101"]


def test_predict_batch_empty_input(capsys):
    out = MLSignalEngine().predict_batch({})
    assert out.empty
    txt = capsys.readouterr().out
    assert "predicting 0 stocks" in txt and "ML signals computed: 0 stocks" in txt


def test_predict_batch_real_engine_mixed_inputs(big, capsys):
    out = MLSignalEngine(workers=2).predict_batch({"GOOD": big, "SHORT": big.iloc[:100], "NONE": None})
    assert set(out["symbol"]) == {"GOOD", "SHORT", "NONE"}
    by = out.set_index("symbol")["direction"]
    assert by["SHORT"] == "INSUFFICIENT_DATA" and by["NONE"] == "INSUFFICIENT_DATA"
    assert by["GOOD"] in ("BULLISH", "NEUTRAL", "BEARISH")
    assert out["predicted_ret%"].is_monotonic_decreasing


# ── evaluate_on_backtest ─────────────────────────────────────────────────────
def _aligned(df):
    feats = compute_features(df)
    close = df["Close"].reindex(feats.index)
    tgt = close.pct_change(PREDICT_DAYS).shift(-PREDICT_DAYS) * 100
    return feats.join(tgt.rename("target"), how="inner").dropna()


def test_backtest_matches_independent_walk_forward(big, capsys):
    al = _aligned(big)
    start, end = al.index[-3], al.index[-1]
    res = MLSignalEngine("lr").evaluate_on_backtest({"S": big}, str(start.date()), str(end.date()))
    assert res["n_predictions"] == 3
    assert res["test_period"] == f"{start.date()} – {end.date()}"
    act, pred = [], []
    for ti in range(len(al) - 3, len(al)):
        X, y = MLSignalEngine()._make_sequences(al[FEATURE_NAMES].iloc[ti - TRAIN_WINDOW:ti],
                                                al["target"].iloc[ti - TRAIN_WINDOW:ti])
        m = LinearRegression().fit(X, y)
        z = z_score_normalise(al[FEATURE_NAMES].iloc[ti - LOOKBACK:ti].values).reshape(1, -1)
        pred.append(float(m.predict(z)[0]))
        act.append(float(al["target"].iloc[ti]))
    act, pred = np.array(act), np.array(pred)
    assert res["rmse"] == round(float(np.sqrt(np.mean((act - pred) ** 2))), 4)
    assert res["mae"] == round(float(np.mean(np.abs(act - pred))), 4)
    hits = ((pred > 0) & (act > 0)) | ((pred < 0) & (act < 0))
    assert res["directional_acc"] == round(hits.sum() / 3 * 100, 1)
    assert res["rmse"] >= res["mae"] >= 0
    txt = capsys.readouterr().out
    assert "ML Backtest Evaluation" in txt and "Predictions:      3" in txt


def test_backtest_window_is_inclusive_and_needs_warmup(big, monkeypatch):
    al = _aligned(big)
    eng = stub_engine(monkeypatch, 1.0)
    # position 85 is the first with >= 20 training sequences (85 - 65)
    res = eng.evaluate_on_backtest({"S": big}, str(al.index[0].date()), str(al.index[85].date()))
    assert res["n_predictions"] == 1
    res = eng.evaluate_on_backtest({"S": big}, str(al.index[0].date()), str(al.index[86].date()))
    assert res["n_predictions"] == 2
    one = eng.evaluate_on_backtest({"S": big}, str(al.index[200].date()), str(al.index[200].date()))
    assert one["n_predictions"] == 1


def test_backtest_directional_accuracy_edge_cases(big, monkeypatch):
    al = _aligned(big)
    s, e = str(al.index[300].date()), str(al.index[339].date())
    tgt = al["target"].iloc[300:340]
    up = stub_engine(monkeypatch, 1.0).evaluate_on_backtest({"S": big}, s, e)
    assert up["n_predictions"] == 40
    assert up["directional_acc"] == round((tgt > 0).sum() / 40 * 100, 1)
    down = stub_engine(monkeypatch, -1.0).evaluate_on_backtest({"S": big}, s, e)
    assert down["directional_acc"] == round((tgt < 0).sum() / 40 * 100, 1)
    zero = stub_engine(monkeypatch, 0.0).evaluate_on_backtest({"S": big}, s, e)
    assert zero["directional_acc"] == 0.0            # pred == 0 never counts as a hit
    assert up["directional_acc"] + down["directional_acc"] == pytest.approx(100.0, abs=0.2)


def test_backtest_no_predictions_returns_error(big):
    eng = MLSignalEngine()
    assert eng.evaluate_on_backtest({}, "2024-01-01", "2024-02-01") == {"error": "No test predictions generated"}
    assert eng.evaluate_on_backtest({"S": big}, "2000-01-01", "2000-02-01") == {"error": "No test predictions generated"}
    assert eng.evaluate_on_backtest({"S": big.iloc[:MIN_ROWS - 1], "N": None},
                                    "2025-01-01", "2025-12-31") == {"error": "No test predictions generated"}
    assert eng.evaluate_on_backtest({"S": big}, "2025-12-30", "2025-01-01") == {"error": "No test predictions generated"}


def test_backtest_bad_frame_is_skipped_not_fatal(big, monkeypatch):
    al = _aligned(big)
    bad = big.drop(columns=["Close"]).copy()
    bad = bad.reindex(make_df(600).index)
    eng = stub_engine(monkeypatch, 1.0)
    res = eng.evaluate_on_backtest({"BAD": bad, "S": big}, str(al.index[300].date()), str(al.index[301].date()))
    assert res["n_predictions"] == 2


# ── compare_models ───────────────────────────────────────────────────────────
def test_compare_models_short_or_no_sklearn_returns_empty(big, monkeypatch):
    assert compare_models(big.iloc[:MIN_ROWS - 1]).empty
    monkeypatch.setattr(ml, "_SKLEARN_OK", False)
    assert compare_models(big).empty


def test_compare_models_too_few_sequences_returns_empty():
    d = make_df(330)
    d.iloc[230, d.columns.get_loc("Volume")] = 0.0     # each zero-volume bar blanks 20 feature rows
    d.iloc[270, d.columns.get_loc("Volume")] = 0.0
    assert len(compute_features(d)) == 91
    assert compare_models(d).empty


def test_compare_models_table_matches_independent_split(big, capsys):
    out = compare_models(big, symbol="XYZ")
    assert out["Model"].tolist() == ["LinearRegression", "Ridge(alpha=1)", "Naive(last_ret)"]
    assert list(out.columns) == ["Model", "Train_RMSE", "Train_MAE", "Test_RMSE", "Test_MAE"]
    al = _aligned(big)
    X, y = MLSignalEngine()._make_sequences(al[FEATURE_NAMES], al["target"])
    split = int(len(X) * 0.8)
    ytr, yte = y[:split], y[split:]
    naive = out.set_index("Model").loc["Naive(last_ret)"]
    assert naive["Train_RMSE"] == round(float(np.sqrt(np.mean((ytr - ytr.mean()) ** 2))), 4)
    assert naive["Test_RMSE"] == round(float(np.sqrt(np.mean((yte - ytr.mean()) ** 2))), 4)
    assert naive["Test_MAE"] == round(float(np.mean(np.abs(yte - ytr.mean()))), 4)
    lr = out.set_index("Model").loc["LinearRegression"]
    ref = LinearRegression().fit(X[:split], ytr)
    assert lr["Test_RMSE"] == round(float(np.sqrt(np.mean((yte - ref.predict(X[split:])) ** 2))), 4)
    # 720 features vs ~264 samples: OLS interpolates the training set, ridge nearly does
    assert lr["Train_RMSE"] <= out.set_index("Model").loc["Ridge(alpha=1)"]["Train_RMSE"] + 1e-6
    assert out.set_index("Model").loc["Ridge(alpha=1)"]["Train_RMSE"] < naive["Train_RMSE"]
    txt = capsys.readouterr().out
    assert f"Model comparison for XYZ (train={split} rows, test={len(X) - split} rows)" in txt
    assert "Paper benchmark" in txt


def test_compare_models_default_symbol_label(big, capsys):
    compare_models(big)
    assert "Model comparison for TEST" in capsys.readouterr().out
