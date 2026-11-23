"""Offline tests for feature_cache.py. Parquet is faked with pickle (no pyarrow needed)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import feature_cache as fc  # noqa: E402


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(fc, "CACHE_DIR", str(tmp_path / "fcache"))
    monkeypatch.setattr(pd.DataFrame, "to_parquet",
                        lambda self, path, *a, **k: self.to_pickle(path), raising=False)
    monkeypatch.setattr(pd, "read_parquet", lambda path, *a, **k: pd.read_pickle(path))
    return fc


def _ohlc(n=5, bump=0.0):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    c = np.arange(n, dtype=float) + 100 + bump
    return pd.DataFrame({"Close": c, "High": c + 1, "Low": c - 1,
                         "Volume": np.arange(n) + 1000, "Open": c}, index=idx)


# ── fingerprint ──────────────────────────────────────────────────────────────
def test_fingerprint_is_16_hex_and_deterministic():
    a = fc.frame_fingerprint(_ohlc())
    assert a == fc.frame_fingerprint(_ohlc())
    assert len(a) == 16 and int(a, 16) >= 0


def test_fingerprint_changes_when_a_relevant_value_changes():
    base = _ohlc()
    for col in ("Close", "High", "Low", "Volume"):
        d = base.copy()
        d.iloc[2, d.columns.get_loc(col)] += 1
        assert fc.frame_fingerprint(d) != fc.frame_fingerprint(base), col


def test_fingerprint_ignores_irrelevant_columns():
    base = _ohlc()
    d = base.copy()
    d["Open"] = d["Open"] + 50
    d["Extra"] = 1
    assert fc.frame_fingerprint(d) == fc.frame_fingerprint(base)


def test_fingerprint_changes_with_last_date_or_length():
    base = _ohlc()
    shifted = base.copy()
    shifted.index = shifted.index + pd.Timedelta(days=7)
    assert fc.frame_fingerprint(shifted) != fc.frame_fingerprint(base)
    assert fc.frame_fingerprint(base.iloc[:-1]) != fc.frame_fingerprint(base)


def test_fingerprint_custom_cols_and_missing_cols_subset():
    base = _ohlc()
    d = base.copy()
    d["Open"] = d["Open"] + 1
    assert fc.frame_fingerprint(d, cols=("Open",)) != fc.frame_fingerprint(base, cols=("Open",))
    only_close = base[["Close"]]
    assert fc.frame_fingerprint(only_close) == fc.frame_fingerprint(base, cols=("Close",))


def test_fingerprint_empty_frame_is_stable_and_ignores_columns():
    a = fc.frame_fingerprint(pd.DataFrame({"Close": []}))
    assert a == fc.frame_fingerprint(pd.DataFrame({"Whatever": []}))
    assert a != fc.frame_fingerprint(_ohlc())


@pytest.mark.xfail(int(pd.__version__.split(".")[0]) < 3, strict=True, raises=ValueError,
                   reason="feature_cache.py:frame_fingerprint non-empty frame with none of the "
                          "relevant columns crashes in hash_pandas_object on pandas<3 instead of hashing")
def test_fingerprint_nonempty_frame_without_relevant_columns():
    d = pd.DataFrame({"X": [1, 2]})
    assert len(fc.frame_fingerprint(d)) == 16


# ── cache_key / paths ────────────────────────────────────────────────────────
def test_cache_key_sanitises_ticker():
    df = _ohlc()
    fp = fc.frame_fingerprint(df)
    assert fc.cache_key("RELIANCE.NS", df) == f"RELIANCE_NS__{fp}"
    assert fc.cache_key("^NSEI", df) == f"_NSEI__{fp}"
    assert fc.cache_key("A/B C", df) == f"A_B_C__{fp}"
    assert fc.cache_key(123, df) == f"123__{fp}"


def test_cache_key_differs_per_ticker_and_data():
    a, b = _ohlc(), _ohlc(bump=1)
    assert fc.cache_key("X", a) != fc.cache_key("Y", a)
    assert fc.cache_key("X", a) != fc.cache_key("X", b)


# ── load / save ──────────────────────────────────────────────────────────────
def test_load_missing_returns_none(cache):
    assert cache.load("nope__abc") is None


def test_save_creates_dir_returns_path_and_roundtrips(cache, tmp_path):
    feats = pd.DataFrame({"f": [1.0, 2.0]})
    p = cache.save("k1", feats)
    assert p == os.path.join(str(tmp_path / "fcache"), "k1.parquet")
    assert os.path.isfile(p)
    pd.testing.assert_frame_equal(cache.load("k1"), feats)


# ── cached_features ──────────────────────────────────────────────────────────
def test_cached_features_computes_once_then_hits_cache(cache):
    calls = []

    def compute(df):
        calls.append(len(df))
        return pd.DataFrame({"mean_close": [df["Close"].mean()]})

    df = _ohlc()
    first = cache.cached_features("T.NS", df, compute)
    second = cache.cached_features("T.NS", df.copy(), compute)
    assert calls == [5]
    pd.testing.assert_frame_equal(first, second)
    assert first.loc[0, "mean_close"] == pytest.approx(102.0)


def test_cached_features_recomputes_when_data_changes(cache):
    calls = []

    def compute(df):
        calls.append(1)
        return pd.DataFrame({"v": [df["Close"].iloc[-1]]})

    a = cache.cached_features("T", _ohlc(), compute)
    b = cache.cached_features("T", _ohlc(bump=10), compute)
    assert len(calls) == 2
    assert a.loc[0, "v"] == 104.0 and b.loc[0, "v"] == 114.0


def test_cached_features_separate_entries_per_ticker(cache):
    calls = []

    def compute(df):
        calls.append(1)
        return pd.DataFrame({"v": [1]})

    df = _ohlc()
    cache.cached_features("A", df, compute)
    cache.cached_features("B", df, compute)
    assert len(calls) == 2
    assert len(os.listdir(fc.CACHE_DIR)) == 2
