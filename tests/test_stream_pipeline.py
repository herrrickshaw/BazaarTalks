"""Offline tests for stream_pipeline.py: Kafka and Cassandra are replaced by in-memory fakes."""
import json
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stream_pipeline as sp  # noqa: E402


# ── fakes ───────────────────────────────────────────────────────────────────
class Msg:
    def __init__(self, value=b"", err=None):
        self._v, self._e = value, err

    def value(self):
        return self._v

    def error(self):
        return self._e


class KafkaWorld:
    """Shared state for the fake confluent_kafka module."""
    def __init__(self):
        self.produced = []           # (topic, key, value)
        self.flushes = 0
        self.queue = []              # items returned by Consumer.poll (None = idle)
        self.polls = 0
        self.consumer_conf = None
        self.producer_confs = []
        self.subscribed = None
        self.closed = False


@pytest.fixture
def kafka(monkeypatch):
    w = KafkaWorld()

    class Producer:
        def __init__(self, conf):
            w.producer_confs.append(conf)

        def produce(self, topic, key=None, value=None):
            w.produced.append((topic, key, value))

        def flush(self):
            w.flushes += 1

    class Consumer:
        def __init__(self, conf):
            w.consumer_conf = conf

        def subscribe(self, topics):
            w.subscribed = topics

        def poll(self, timeout):
            w.polls += 1
            return w.queue.pop(0) if w.queue else None

        def close(self):
            w.closed = True

    mod = types.ModuleType("confluent_kafka")
    mod.Producer, mod.Consumer = Producer, Consumer
    monkeypatch.setitem(sys.modules, "confluent_kafka", mod)
    return w


def _install_store(monkeypatch, ohlc=None, cached=None):
    store = types.ModuleType("market_store")
    store.calls = []

    def get_ohlc(t):
        store.calls.append(("get_ohlc", t))
        return ohlc if ohlc is not None else pd.DataFrame()

    def cached_download(tickers):
        store.calls.append(("cached_download", list(tickers)))
        return cached if cached is not None else {}

    store.get_ohlc, store.cached_download = get_ohlc, cached_download
    monkeypatch.setitem(sys.modules, "market_store", store)
    return store


# ── _universe ───────────────────────────────────────────────────────────────
class FakeExcel:
    sheets = {}
    opened = []

    def __init__(self, path):
        FakeExcel.opened.append(path)

    def parse(self, name):
        assert name == "All_Stocks"
        return FakeExcel.sheets[FakeExcel.opened[-1]]


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(pd, "ExcelFile", FakeExcel)
    FakeExcel.opened, FakeExcel.sheets = [], {}
    return tmp_path


def _touch(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").close()
    return str(p)


def test_universe_no_files_empty(home):
    assert sp._universe("US", None) == []
    assert sp._universe("India", 5) == []


def test_universe_us_uses_latest_sorted_file_and_drops_nan(home):
    old = _touch(home / "Downloads/data/us_full_scan/2024/us_full_scan_20240101.xlsx")
    new = _touch(home / "Downloads/data/us_full_scan/2024/us_full_scan_20240301.xlsx")
    FakeExcel.sheets[new] = pd.DataFrame({"Symbol": ["AAPL", "nan", "", "MSFT"]})
    FakeExcel.sheets[old] = pd.DataFrame({"Symbol": ["OLD"]})
    assert sp._universe("US", None) == ["AAPL", "MSFT"]      # newest file; 'nan'/'' removed
    assert FakeExcel.opened == [new]


@pytest.mark.xfail(
    condition=int(pd.__version__.split(".")[0]) >= 3, strict=True, raises=AssertionError,
    reason="stream_pipeline.py:42 astype(str) keeps NaN as a float on pandas>=3 string dtype, "
           "so the `s != 'nan'` filter at :47 lets a NaN ticker through")
def test_universe_drops_blank_excel_cells(home):
    f = _touch(home / "Downloads/data/us_full_scan/2024/us_full_scan_20240301.xlsx")
    FakeExcel.sheets[f] = pd.DataFrame({"Symbol": ["AAPL", np.nan, "MSFT"]})
    assert sp._universe("US", None) == ["AAPL", "MSFT"]


def test_universe_india_concats_suffix_and_limit(home):
    f = _touch(home / "Downloads/data/x/y/indian_full_scan_20240101.xlsx")
    FakeExcel.sheets[f] = pd.DataFrame({"Symbol": ["TCS", "INFY", "WIPRO"], "Suffix": [".NS", ".NS", ".BO"]})
    assert sp._universe("India", None) == ["TCS.NS", "INFY.NS", "WIPRO.BO"]
    assert sp._universe("India", 2) == ["TCS.NS", "INFY.NS"]
    assert sp._universe("India", 0) == ["TCS.NS", "INFY.NS", "WIPRO.BO"]      # 0 = no limit (falsy)


def test_universe_unknown_market_keyerror(home):
    with pytest.raises(KeyError):
        sp._universe("Mars", None)


# ── _signal ─────────────────────────────────────────────────────────────────
def _df(vals):
    return pd.DataFrame({"Close": vals})


def test_signal_no_data_when_store_and_cache_empty(monkeypatch):
    store = _install_store(monkeypatch)
    assert sp._signal("ZZZ") == {"ticker": "ZZZ", "signal": "NO_DATA"}
    assert store.calls == [("get_ohlc", "ZZZ"), ("cached_download", ["ZZZ"])]


def test_signal_falls_back_to_cache_download(monkeypatch):
    store = _install_store(monkeypatch, cached={"AAA": _df(np.linspace(100, 160, 70))})
    out = sp._signal("AAA")
    assert out["signal"] == "BREAKOUT"
    assert [c[0] for c in store.calls] == ["get_ohlc", "cached_download"]


def test_signal_cache_returns_none_is_no_data(monkeypatch):
    _install_store(monkeypatch, cached={"AAA": None})
    assert sp._signal("AAA")["signal"] == "NO_DATA"


def test_signal_min_length_boundary_60(monkeypatch):
    _install_store(monkeypatch, ohlc=_df(np.linspace(100, 150, 59)))
    assert sp._signal("T")["signal"] == "NO_DATA"
    _install_store(monkeypatch, ohlc=_df(np.linspace(100, 150, 60)))
    assert sp._signal("T")["signal"] == "BREAKOUT"


def test_signal_breakout_fields_and_rounding(monkeypatch):
    close = np.linspace(100, 159.123456, 80)
    store = _install_store(monkeypatch, ohlc=_df(close))
    out = sp._signal("UP")
    assert out["ticker"] == "UP" and out["signal"] == "BREAKOUT"
    assert out["close"] == 159.12
    assert np.isnan(out["rsi"])                       # no down days -> 0 loss -> RSI undefined
    assert store.calls == [("get_ohlc", "UP")]        # cache not consulted when OHLC present


def test_signal_flat_series_is_breakout(monkeypatch):
    _install_store(monkeypatch, ohlc=_df([50.0] * 70))
    assert sp._signal("F")["signal"] == "BREAKOUT"    # last >= rolling max (>=)


def test_signal_oversold(monkeypatch):
    _install_store(monkeypatch, ohlc=_df(np.linspace(200, 100, 70)))
    out = sp._signal("DN")
    assert out["signal"] == "OVERSOLD" and out["rsi"] == 0.0 and out["close"] == 100.0


def test_signal_neutral_sawtooth(monkeypatch):
    _install_store(monkeypatch, ohlc=_df([100.0, 102.0] * 30 + [100.0]))   # 61 bars, ends on a dip
    out = sp._signal("N")
    assert out["signal"] == "NEUTRAL"
    assert 30 <= out["rsi"] < 70 and out["close"] == 100.0


def test_signal_close_coerced_from_strings(monkeypatch):
    _install_store(monkeypatch, ohlc=_df([str(v) for v in np.linspace(10, 20, 65)]))
    assert sp._signal("S")["close"] == 20.0


# ── produce ─────────────────────────────────────────────────────────────────
def test_produce_publishes_each_ticker_and_flushes(kafka, monkeypatch, capsys):
    monkeypatch.setattr(sp, "_universe", lambda m, l: ["A", "B", "C"])
    sp.produce("US", 3)
    assert kafka.produced == [(sp.TOPIC_WORK, t, t) for t in "ABC"]
    assert kafka.flushes == 1
    assert kafka.producer_confs == [{"bootstrap.servers": sp.BROKER}]
    assert "produced 3 US tickers -> scan.tickers" in capsys.readouterr().err


def test_produce_empty_universe(kafka, monkeypatch, capsys):
    monkeypatch.setattr(sp, "_universe", lambda m, l: [])
    sp.produce("India", None)
    assert kafka.produced == [] and kafka.flushes == 1
    assert "produced 0 India tickers" in capsys.readouterr().err


# ── consume ─────────────────────────────────────────────────────────────────
def test_consume_processes_messages_then_stops_after_idle(kafka, monkeypatch, capsys):
    monkeypatch.setattr(sp, "_signal", lambda t: {"ticker": t, "signal": "NEUTRAL"})
    kafka.queue = [Msg(b"AAA"), Msg(b"BBB")]
    sp.consume("grp", max_idle=3)
    assert kafka.consumer_conf == {"bootstrap.servers": sp.BROKER, "group.id": "grp",
                                   "auto.offset.reset": "earliest"}
    assert kafka.subscribed == [sp.TOPIC_WORK]
    assert [(t, k, json.loads(v)["signal"]) for t, k, v in kafka.produced] == [
        (sp.TOPIC_RESULT, "AAA", "NEUTRAL"), (sp.TOPIC_RESULT, "BBB", "NEUTRAL")]
    assert kafka.polls == 2 + 3                       # 2 messages then exactly max_idle empty polls
    assert kafka.closed
    assert kafka.flushes == 1                         # final flush only (n < 25)
    err = capsys.readouterr().err
    assert "[worker grp] done, 2 signals -> scan.signals" in err


def test_consume_idle_counter_resets_on_message(kafka, monkeypatch):
    monkeypatch.setattr(sp, "_signal", lambda t: {"ticker": t})
    kafka.queue = [None, None, Msg(b"A"), None, None]
    sp.consume("g", max_idle=3)
    assert kafka.polls == 2 + 1 + 3 and len(kafka.produced) == 1


def test_consume_skips_error_messages(kafka, monkeypatch):
    monkeypatch.setattr(sp, "_signal", lambda t: {"ticker": t})
    kafka.queue = [Msg(b"X", err="boom"), Msg(b"OK")]
    sp.consume("g", max_idle=1)
    assert [k for _, k, _ in kafka.produced] == ["OK"]


def test_consume_flushes_and_reports_every_25(kafka, monkeypatch, capsys):
    monkeypatch.setattr(sp, "_signal", lambda t: {"ticker": t})
    kafka.queue = [Msg(f"T{i}".encode()) for i in range(50)]
    sp.consume("g", max_idle=1)
    assert kafka.flushes == 2 + 1                     # at 25, at 50, and the finally-flush
    err = capsys.readouterr().err
    assert "processed 25" in err and "processed 50" in err and "processed 24" not in err


def test_consume_closes_on_exception(kafka, monkeypatch):
    def boom(t):
        raise RuntimeError("bad ticker")
    monkeypatch.setattr(sp, "_signal", boom)
    kafka.queue = [Msg(b"A")]
    with pytest.raises(RuntimeError):
        sp.consume("g", max_idle=1)
    assert kafka.closed and kafka.flushes == 1


# ── demo ────────────────────────────────────────────────────────────────────
def test_demo_round_trip(kafka, monkeypatch, capsys):
    monkeypatch.setattr(sp, "_signal", lambda t: {"ticker": t, "signal": "X"})
    expected = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
    kafka.queue = [Msg(t.encode()) for t in expected]
    sp.demo()
    assert [k for _, k, _ in kafka.produced] == expected
    assert kafka.consumer_conf["group.id"] == "demo"
    out = capsys.readouterr()
    assert out.out.count("signal:") == 5 and "produced 5 -> scan.tickers" in out.err
    assert kafka.closed and kafka.polls == 5          # stops as soon as 5 received


def test_demo_gives_up_after_10_idle_polls(kafka, monkeypatch, capsys):
    monkeypatch.setattr(sp, "_signal", lambda t: {"ticker": t})
    kafka.queue = [Msg(b"AAPL")]
    sp.demo()
    assert kafka.polls == 1 + 10 and kafka.closed
    assert capsys.readouterr().out.count("signal:") == 1


def test_demo_error_messages_not_counted(kafka, monkeypatch, capsys):
    monkeypatch.setattr(sp, "_signal", lambda t: {"ticker": t})
    kafka.queue = [Msg(b"E", err="x")] + [Msg(t.encode()) for t in ["A", "B", "C", "D", "E"]]
    sp.demo()
    assert capsys.readouterr().out.count("signal:") == 5 and kafka.polls == 6


# ── main ────────────────────────────────────────────────────────────────────
def _main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["stream_pipeline.py", *argv])
    sp.main()


def test_main_produce_defaults_and_args(monkeypatch):
    calls = []
    monkeypatch.setattr(sp, "produce", lambda m, l: calls.append((m, l)))
    _main(monkeypatch, "produce")
    _main(monkeypatch, "produce", "--market", "India", "--limit", "7")
    assert calls == [("US", None), ("India", 7)]


def test_main_consume_group(monkeypatch):
    calls = []
    monkeypatch.setattr(sp, "consume", lambda g: calls.append(g))
    _main(monkeypatch, "consume")
    _main(monkeypatch, "consume", "--group", "g9")
    assert calls == ["scan1", "g9"]


def test_main_demo(monkeypatch):
    calls = []
    monkeypatch.setattr(sp, "demo", lambda: calls.append(1))
    _main(monkeypatch, "demo")
    assert calls == [1]


def test_main_requires_command(monkeypatch):
    with pytest.raises(SystemExit):
        _main(monkeypatch)
