"""Offline tests for apiclient.py: throttling, adaptive penalty, retry/backoff, yfinance + HTTP wrappers."""
import os
import sys
import threading
import types

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import apiclient  # noqa: E402


class Clock:
    """Fake time module: sleep advances a monotonic clock and is recorded."""
    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s

    def time(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(apiclient, "time", c)
    monkeypatch.setattr(apiclient.random, "uniform", lambda a, b: 0.0)
    monkeypatch.setattr(apiclient, "_sources", {})
    monkeypatch.setitem(apiclient.LIMITS, "t", (0.0, 1))        # no throttle wait: backoff only
    return c


# ── rate-error classification ─────────────────────────────────────────────────
@pytest.mark.parametrize("msg", [
    "HTTP 429", "Too Many Requests", "Invalid Crumb", "401 Unauthorized", "rate limit",
    "Service Unavailable", "Read timed out", "Connection reset by peer", "THROTTLED",
])
def test_rate_errors(msg):
    assert apiclient._is_rate_error(Exception(msg))


@pytest.mark.parametrize("msg", ["KeyError: 'x'", "division by zero", "", "404 not found"])
def test_non_rate_errors(msg):
    assert not apiclient._is_rate_error(Exception(msg))


# ── _Source throttle ──────────────────────────────────────────────────────────
def test_source_spaces_calls_by_interval(clock):
    s = apiclient._Source(2.0, 1)
    s.acquire(); s.release()
    assert clock.sleeps == []                       # first call is immediate
    s.acquire(); s.release()
    assert clock.sleeps == [2.0]                    # second waits a full interval
    s.acquire(); s.release()
    assert clock.sleeps == [2.0, 2.0]


def test_source_no_wait_when_interval_already_elapsed(clock):
    s = apiclient._Source(1.0, 1)
    s.acquire(); s.release()
    clock.now += 5.0
    s.acquire(); s.release()
    assert clock.sleeps == []
    assert s.next_ok == clock.now + 1.0             # re-based from now, not stacked


def test_source_penalty_scales_interval_caps_and_relaxes(clock):
    s = apiclient._Source(1.0, 1)
    for _ in range(10):
        s.penalize()
    assert s.penalty == 32.0                        # capped
    s.relax()
    assert s.penalty == pytest.approx(32.0 * 0.92)
    for _ in range(200):
        s.relax()
    assert s.penalty == 1.0                         # floor
    s.penalize()
    assert s.penalty == 2.0
    s.acquire(); s.release(); s.acquire(); s.release()
    assert clock.sleeps == [2.0]


def test_source_concurrency_limit_blocks_extra_holder(clock):
    s = apiclient._Source(0.0, 1)
    s.acquire()
    got = []
    t = threading.Thread(target=lambda: (s.acquire(), got.append(1), s.release()))
    t.start()
    t.join(0.05)
    assert got == [] and t.is_alive()               # blocked on the semaphore
    s.release()
    t.join(2)
    assert got == [1]


def test_src_registry_uses_limits_default_and_caches(clock):
    a = apiclient._src("edgar")
    assert (a.interval, a.sem._initial_value) == (0.12, 5)
    assert apiclient._src("edgar") is a
    d = apiclient._src("unlisted-source")
    assert (d.interval, d.sem._initial_value) == apiclient._DEFAULT


def test_limits_table_documented_values():
    assert apiclient.LIMITS["yfinance"] == (0.40, 3)
    assert apiclient.LIMITS["wikidata"] == (60.0, 1)
    assert apiclient.LIMITS["arxiv"] == (3.00, 1)
    assert apiclient.YF_CHUNK == 50


def test_throttle_context_releases_slot_even_on_exception(clock):
    with pytest.raises(ValueError):
        with apiclient.throttle("t"):
            raise ValueError("boom")
    with apiclient.throttle("t"):                    # would deadlock if not released
        pass


# ── robust ────────────────────────────────────────────────────────────────────
def test_robust_success_passes_args_and_relaxes(clock):
    s = apiclient._src("t")
    s.penalty = 4.0
    out = apiclient.robust("t", lambda a, b=0: a + b, 1, b=2)
    assert out == 3
    assert s.penalty == pytest.approx(4.0 * 0.92)
    assert clock.sleeps == []


def test_robust_rate_error_retries_with_exponential_backoff_then_succeeds(clock):
    n = {"c": 0}

    def f():
        n["c"] += 1
        if n["c"] < 3:
            raise RuntimeError("HTTP 429 rate limited")
        return "ok"
    assert apiclient.robust("t", f) == "ok"
    assert n["c"] == 3
    assert clock.sleeps == [1.5, 3.0]               # 1.5*2**attempt, no jitter
    assert apiclient._src("t").penalty == pytest.approx(4.0 * 0.92)    # 2 penalties then a relax


def test_robust_rate_errors_exhaust_retries_and_raise_last(clock):
    n = {"c": 0}

    def f():
        n["c"] += 1
        raise RuntimeError(f"429 #{n['c']}")
    with pytest.raises(RuntimeError, match="429 #3"):
        apiclient.robust("t", f, retries=3)
    assert n["c"] == 3
    assert clock.sleeps == [1.5, 3.0, 6.0]          # sleeps after every attempt, incl. the last


def test_robust_backoff_is_capped_at_90_seconds(clock):
    def f():
        raise RuntimeError("429")
    with pytest.raises(RuntimeError):
        apiclient.robust("t", f, retries=8)
    assert clock.sleeps[:6] == [1.5, 3.0, 6.0, 12.0, 24.0, 48.0]
    assert clock.sleeps[6:] == [90.0, 90.0]         # 96 -> 90


def test_robust_jitter_added_to_backoff(clock, monkeypatch):
    monkeypatch.setattr(apiclient.random, "uniform", lambda a, b: 0.25)
    calls = {"c": 0}

    def f():
        calls["c"] += 1
        if calls["c"] == 1:
            raise RuntimeError("429")
        return 1
    assert apiclient.robust("t", f) == 1
    assert clock.sleeps == [1.75]


def test_robust_non_rate_error_gives_up_on_third_failure(clock):
    n = {"c": 0}

    def f():
        n["c"] += 1
        raise KeyError("missing")
    with pytest.raises(KeyError):
        apiclient.robust("t", f)
    assert n["c"] == 3                              # attempts 0,1 retried; attempt 2 raises
    assert clock.sleeps == [1.5, 3.0]
    assert apiclient._src("t").penalty == 1.0       # non-rate errors do not penalise


def test_robust_non_rate_error_recovers_before_giving_up(clock):
    n = {"c": 0}

    def f():
        n["c"] += 1
        if n["c"] == 1:
            raise KeyError("flaky")
        return "fine"
    assert apiclient.robust("t", f) == "fine"
    assert clock.sleeps == [1.5]


def test_robust_non_rate_error_with_retries_two_raises_last(clock):
    def f():
        raise KeyError("bad")
    with pytest.raises(KeyError):
        apiclient.robust("t", f, retries=2)
    assert clock.sleeps == [1.5, 3.0]


def test_robust_zero_retries_raises_runtime_error(clock):
    with pytest.raises(RuntimeError, match="t: retries exhausted"):
        apiclient.robust("t", lambda: 1, retries=0)


# ── yf_download ───────────────────────────────────────────────────────────────
def _multi(tickers, n=2):
    idx = pd.date_range("2024-01-01", periods=n)
    cols = pd.MultiIndex.from_product([tickers, ["Open", "Close"]])
    return pd.DataFrame(1.0, index=idx, columns=cols)


@pytest.fixture
def fake_yf(monkeypatch):
    calls = []
    mod = types.ModuleType("yfinance")

    def download(batch, **kw):
        calls.append((list(batch), kw))
        if mod.fail_on and mod.fail_on in batch:
            raise RuntimeError("boom")
        return mod.respond(batch)
    mod.download = download
    mod.fail_on = None
    mod.respond = lambda batch: _multi(batch)
    mod.calls = calls
    monkeypatch.setitem(sys.modules, "yfinance", mod)
    monkeypatch.setattr(apiclient, "robust", lambda src, fn, *a, **k: fn(*a, **k))
    return mod


def test_yf_download_dedupes_chunks_and_forwards_kwargs(fake_yf, monkeypatch):
    monkeypatch.setattr(apiclient, "YF_CHUNK", 2)
    out = apiclient.yf_download(["A", "B", "A", "", None, "C", "D", "E"], period="1y", interval="1d")
    assert [c[0] for c in fake_yf.calls] == [["A", "B"], ["C", "D"], ["E"]]
    kw = fake_yf.calls[0][1]
    assert kw["period"] == "1y" and kw["start"] is None and kw["interval"] == "1d"
    assert kw["auto_adjust"] is True and kw["progress"] is False
    assert kw["group_by"] == "ticker" and kw["threads"] is True
    assert list(out) == ["A", "B", "C", "D", "E"]
    assert list(out["A"].columns) == ["Open", "Close"]


def test_yf_download_accepts_single_string_and_tuple(fake_yf):
    assert list(apiclient.yf_download("AAPL", period="5d")) == ["AAPL"]
    assert list(apiclient.yf_download(("X", "Y"), start="2024-01-01")) == ["X", "Y"]
    assert fake_yf.calls[-1][1]["start"] == "2024-01-01"


def test_yf_download_failed_batch_skipped_others_kept(fake_yf, monkeypatch):
    monkeypatch.setattr(apiclient, "YF_CHUNK", 1)
    fake_yf.fail_on = "B"
    assert list(apiclient.yf_download(["A", "B", "C"])) == ["A", "C"]


def test_yf_download_drops_missing_and_all_nan_tickers(fake_yf):
    def respond(batch):
        df = _multi(["A", "B"])
        df[("B", "Open")] = float("nan")
        df[("B", "Close")] = float("nan")
        return df
    fake_yf.respond = respond
    out = apiclient.yf_download(["A", "B", "ZZ"])
    assert list(out) == ["A"]                       # B all-NaN, ZZ KeyError


def test_yf_download_flat_columns_used_directly_and_nan_rows_dropped(fake_yf):
    flat = pd.DataFrame({"Close": [1.0, float("nan")]}, index=pd.date_range("2024-01-01", periods=2))
    fake_yf.respond = lambda batch: flat
    out = apiclient.yf_download(["ONLY"])
    assert list(out) == ["ONLY"] and len(out["ONLY"]) == 1      # all-NaN row dropped


def test_yf_download_empty_input(fake_yf):
    assert apiclient.yf_download([]) == {}
    assert fake_yf.calls == []


# ── yf_info ───────────────────────────────────────────────────────────────────
def test_yf_info_calls_get_info_through_robust(monkeypatch):
    seen = {}

    class Ticker:
        def __init__(self, t):
            seen["t"] = t

        def get_info(self):
            return {"sector": "Tech"}
    mod = types.ModuleType("yfinance")
    mod.Ticker = Ticker
    monkeypatch.setitem(sys.modules, "yfinance", mod)

    def fake_robust(src, fn, *a, **k):
        seen["src"] = src
        return fn(*a, **k)
    monkeypatch.setattr(apiclient, "robust", fake_robust)
    assert apiclient.yf_info("MSFT") == {"sector": "Tech"}
    assert seen == {"t": "MSFT", "src": "yfinance"}


# ── http_get ──────────────────────────────────────────────────────────────────
class Resp:
    def __init__(self, code):
        self.status_code = code


def _patch_requests(monkeypatch, codes):
    import requests
    seen = []
    it = iter(codes)

    def get(url, headers=None, timeout=None):
        seen.append((url, headers, timeout))
        return Resp(next(it))
    monkeypatch.setattr(requests, "get", get)
    return seen


def test_http_get_passes_through_response_and_args(clock, monkeypatch):
    seen = _patch_requests(monkeypatch, [200])
    r = apiclient.http_get("t", "http://x/y", headers={"A": "b"}, timeout=7)
    assert r.status_code == 200
    assert seen == [("http://x/y", {"A": "b"}, 7)]


def test_http_get_default_timeout_45(clock, monkeypatch):
    seen = _patch_requests(monkeypatch, [200])
    apiclient.http_get("t", "u")
    assert seen[0][2] == 45 and seen[0][1] is None


def test_http_get_retries_429_then_returns(clock, monkeypatch):
    seen = _patch_requests(monkeypatch, [429, 429, 200])
    assert apiclient.http_get("t", "u").status_code == 200
    assert len(seen) == 3 and clock.sleeps == [1.5, 3.0]


def test_http_get_429_exhausts_retries(clock, monkeypatch):
    seen = _patch_requests(monkeypatch, [429, 429])
    with pytest.raises(RuntimeError, match="HTTP 429 rate limited"):
        apiclient.http_get("t", "u", retries=2)
    assert len(seen) == 2


def test_http_get_other_status_codes_are_returned_not_raised(clock, monkeypatch):
    _patch_requests(monkeypatch, [500])
    assert apiclient.http_get("t", "u").status_code == 500
