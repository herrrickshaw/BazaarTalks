"""Offline tests for fx.py (currency mapping, conversion, rate cache, CLI)."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fx  # noqa: E402


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    p = tmp_path / "fx_rates.json"
    monkeypatch.setattr(fx, "CACHE", str(p))
    return p


# ── pure core ─────────────────────────────────────────────────────────────────
def test_market_currency_case_and_default():
    assert fx.market_currency("kr") == "KRW"
    assert fx.market_currency("UK") == "GBP"
    assert fx.market_currency("DE") == "EUR" and fx.market_currency("FI") == "EUR"
    assert fx.market_currency("ZZ") == "USD"          # unknown -> USD


def test_convert_level_and_combine_return():
    assert fx.convert_level("1000", "0.00073") == pytest.approx(0.73)
    assert fx.convert_level(0, 5) == 0.0
    assert fx.combine_return(0.0, 0.0) == 0.0
    assert fx.combine_return(0.10, -0.10) == pytest.approx(-0.01)
    assert fx.combine_return(-1.0, 0.5) == pytest.approx(-1.0)


def test_normalize_cross_market_adds_base_column_without_mutating():
    df = pd.DataFrame({"mktcap": [100, 200, 300, 400], "market": ["KR", "us", "UK", "ZZ"]})
    rates = {"KRW": 0.001, "USD": 1.0, "GBP": 1.5}
    out = fx.normalize_cross_market(df, "mktcap", "market", rates)
    assert "mktcap_usd" not in df.columns
    assert list(out["mktcap_usd"]) == pytest.approx([0.1, 200.0, 450.0, 400.0])
    out2 = fx.normalize_cross_market(df, "mktcap", "market", {"USD": 1.0}, base="eur")
    assert "mktcap_eur" in out2.columns
    assert np.isnan(out2["mktcap_eur"].iloc[0]) and out2["mktcap_eur"].iloc[1] == 200.0


# ── load_rates ────────────────────────────────────────────────────────────────
def test_load_rates_snapshot_when_no_cache_and_is_a_copy():
    r = fx.load_rates()
    assert r == fx.SNAPSHOT_USD_PER
    r["USD"] = 99
    assert fx.SNAPSHOT_USD_PER["USD"] == 1.0


def test_load_rates_reads_matching_cache(_cache):
    _cache.write_text(json.dumps({"base": "USD", "rates": {"USD": 1.0, "EUR": 1.5}}))
    assert fx.load_rates("USD") == {"USD": 1.0, "EUR": 1.5}


@pytest.mark.parametrize("payload", [
    json.dumps({"base": "EUR", "rates": {"USD": 1.0}}),        # base mismatch
    json.dumps({"base": "USD", "rates": {}}),                  # empty rates
    json.dumps({"base": "USD"}),                               # no rates
    "{not json",                                                # corrupt
])
def test_load_rates_bad_cache_falls_back_to_snapshot(_cache, payload):
    _cache.write_text(payload)
    assert fx.load_rates("USD") == fx.SNAPSHOT_USD_PER


def test_load_rates_rebases_snapshot_to_other_base():
    r = fx.load_rates("EUR")
    assert r["EUR"] == 1.0
    assert r["USD"] == round(1 / 1.08, 6)
    assert r["JPY"] == round(0.0064 / 1.08, 6)
    assert set(r) == set(fx.SNAPSHOT_USD_PER)


def test_load_rates_unknown_base_returns_usd_snapshot():
    assert fx.load_rates("XXX") == fx.SNAPSHOT_USD_PER


def test_load_rates_non_usd_cache_hit(_cache):
    _cache.write_text(json.dumps({"base": "EUR", "rates": {"EUR": 1.0, "USD": 0.9}}))
    assert fx.load_rates("EUR") == {"EUR": 1.0, "USD": 0.9}


# ── refresh_rates ─────────────────────────────────────────────────────────────
def _fake_yf(monkeypatch, data, calls=None):
    import apiclient

    def fake(tickers, period=None, **kw):
        if calls is not None:
            calls.append((list(tickers), period))
        return data
    monkeypatch.setattr(apiclient, "yf_download", fake)


def test_refresh_rates_uses_last_close_writes_cache_and_falls_back(monkeypatch, _cache):
    calls = []
    data = {
        "EURUSD=X": pd.DataFrame({"Close": [1.1, 1.2, np.nan]}),   # trailing NaN dropped
        "JPYUSD=X": pd.DataFrame({"Close": [np.nan, np.nan]}),     # all NaN -> snapshot
        # everything else missing -> snapshot
    }
    _fake_yf(monkeypatch, data, calls)
    rates = fx.refresh_rates("USD")
    tickers, period = calls[0]
    assert period == "5d"
    assert "USDUSD=X" not in tickers and tickers == sorted(tickers)
    assert tickers == [f"{c}USD=X" for c in sorted(set(fx.MARKET_CCY.values()) - {"USD"})]
    assert rates["USD"] == 1.0
    assert rates["EUR"] == 1.2
    assert rates["JPY"] == fx.SNAPSHOT_USD_PER["JPY"]
    assert rates["GBP"] == fx.SNAPSHOT_USD_PER["GBP"]
    saved = json.loads(_cache.read_text())
    assert saved == {"base": "USD", "rates": rates}
    assert fx.load_rates("USD") == rates                           # round trip via cache


def test_refresh_rates_rounds_to_six_places(monkeypatch):
    _fake_yf(monkeypatch, {"EURUSD=X": pd.DataFrame({"Close": [1.23456789]})})
    assert fx.refresh_rates("USD")["EUR"] == 1.234568


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="fx.py:~111 failed-fetch fallback uses SNAPSHOT_USD_PER (USD-per-unit) "
                          "even when base != USD, so a EUR-based cache holds USD-based numbers")
def test_refresh_rates_fallback_respects_non_usd_base(monkeypatch):
    _fake_yf(monkeypatch, {})
    rates = fx.refresh_rates("EUR")
    # GBP per-EUR-base should be 1.27 USD / 1.08 USD = ~1.176, not the raw USD 1.27
    assert rates["GBP"] == pytest.approx(1.27 / 1.08, rel=1e-3)


# ── CLI ───────────────────────────────────────────────────────────────────────
def _run(monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, "argv", ["fx.py", *argv])
    fx.main()
    return capsys.readouterr()


def test_main_default_prints_rate_table(monkeypatch, capsys):
    out = _run(monkeypatch, capsys)
    lines = out.out.splitlines()
    assert lines[0] == f"=== USD-per-unit rates ({len(fx.SNAPSHOT_USD_PER)} currencies) ==="
    eur = next(line for line in lines if line.strip().startswith("EUR"))
    assert "1.080000" in eur and "markets: DE,EU,FI" in eur
    cur = [line.split()[0] for line in lines[1:]]
    assert cur == sorted(fx.SNAPSHOT_USD_PER)


def test_main_convert(monkeypatch, capsys):
    out = _run(monkeypatch, capsys, "--convert", "1000000", "--from", "KR")
    assert out.out == "1,000,000.00 KRW  =  730.00 USD\n"


def test_main_convert_requires_from_else_table(monkeypatch, capsys):
    out = _run(monkeypatch, capsys, "--convert", "5")
    assert out.out.startswith("=== USD-per-unit rates")


def test_main_convert_unknown_rate_is_nan(monkeypatch, capsys, _cache):
    _cache.write_text(json.dumps({"base": "USD", "rates": {"USD": 1.0}}))
    out = _run(monkeypatch, capsys, "--convert", "10", "--from", "JP")
    assert out.out == "10.00 JPY  =  nan USD\n"


def test_main_refresh_reports_on_stderr(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(fx, "refresh_rates", lambda base: seen.append(base) or {"EUR": 1.5, "USD": 1.0})
    out = _run(monkeypatch, capsys, "--refresh", "--base", "EUR", "--rates")
    assert seen == ["EUR"]
    assert f"refreshed 2 rates -> {fx.CACHE}" in out.err
    assert out.out.splitlines()[0] == "=== EUR-per-unit rates (2 currencies) ==="
