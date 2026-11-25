"""Offline tests for data_sources.py (static per-market factor-library / filing-system registry)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_sources as ds  # noqa: E402


# ── public_factor_sources ─────────────────────────────────────────────────────
@pytest.mark.parametrize("market,expected", [
    ("IN", ["iffm", "aqr", "ken_french"]),
    ("US", ["aqr", "ken_french"]),
    ("us", ["aqr", "ken_french"]),                  # case-insensitive
    ("ZA", ["aqr", "ken_french"]),                  # not an AQR country, global AQR still listed
    ("XX", ["aqr"]),                                # unknown market: only the global AQR series
])
def test_public_factor_sources(market, expected):
    assert ds.public_factor_sources(market) == expected


def test_public_factor_sources_no_duplicates_for_any_platform_market():
    for m in ds.PLATFORM_MARKETS + ["IN"]:
        out = ds.public_factor_sources(m)
        assert len(out) == len(set(out)) and set(out) <= set(ds.LIBRARIES)
        assert out[-1] == "ken_french"              # every platform market has a KF region


# ── raw_sources ───────────────────────────────────────────────────────────────
def test_raw_sources():
    assert ds.raw_sources("US") == ["yfinance", "edgar"]
    assert ds.raw_sources("us") == ["yfinance", "edgar"]
    assert ds.raw_sources("IN") == ["nsepython", "yfinance"]
    assert ds.raw_sources("JP") == ["yfinance"]
    assert set(ds.raw_sources("IN") + ds.raw_sources("US")) <= set(ds.RAW_SOURCES)


# ── filing_source / filing_coverage ───────────────────────────────────────────
def test_filing_source_lookup():
    br = ds.filing_source("br")
    assert br["system"].startswith("CVM") and br["api"] is True
    assert ds.filing_source("IN")["api"] is False
    assert ds.filing_source("CA") is None           # not registered
    assert ds.filing_source("") is None


def test_filing_coverage_counts_platform_markets_only():
    c = ds.filing_coverage()
    assert c["markets"] == 19
    # IN is registered but is not a platform market; CA/FI/DK/SE/CH/SA are not registered
    assert c["with_filing_dates"] == 13
    assert c["covered"] == ["US", "UK", "DE", "EU", "JP", "AU", "HK", "SG", "BR", "CN",
                            "TW", "KR", "ZA"]
    assert c["with_api"] == 6
    assert c["api_markets"] == ["US", "UK", "JP", "BR", "TW", "KR"]
    assert "IN" not in c["covered"]


def test_filing_coverage_respects_has_filing_dates_flag(monkeypatch):
    patched = dict(ds.REGULATORY_FILINGS)
    patched["US"] = dict(patched["US"], has_filing_dates=False)
    monkeypatch.setattr(ds, "REGULATORY_FILINGS", patched)
    c = ds.filing_coverage()
    assert "US" not in c["covered"] and "US" not in c["api_markets"]
    assert c["with_filing_dates"] == 12 and c["with_api"] == 5


# ── for_market / coverage / paper_sources ─────────────────────────────────────
def test_for_market_entry_shape():
    d = ds.for_market("kr")
    assert d == {"market": "KR", "currency": "KRW", "ken_french_region": "Emerging",
                 "aqr_country_qmj": False, "public_factor_sources": ["aqr", "ken_french"],
                 "raw_sources": ["yfinance"]}
    us = ds.for_market("US")
    assert us["aqr_country_qmj"] is True and us["raw_sources"] == ["yfinance", "edgar"]
    unk = ds.for_market("xx")
    assert unk["currency"] is None and unk["ken_french_region"] is None
    assert unk["aqr_country_qmj"] is False and unk["public_factor_sources"] == ["aqr"]


def test_in_is_aqr_country_via_emerging_set():
    assert ds.for_market("IN")["aqr_country_qmj"] is True
    assert ds.for_market("IN")["currency"] == "INR"


def test_coverage_rows_cover_all_platform_markets_plus_india():
    rows = ds.coverage()
    assert [r["market"] for r in rows] == ds.PLATFORM_MARKETS + ["IN"]
    assert len(rows) == 20


def test_paper_sources_pair():
    p = ds.paper_sources()
    assert p["returns_and_fundamentals"] is ds.RAW_SOURCES["cmie_prowess"]
    assert "COMMERCIAL" in p["returns_and_fundamentals"]["license"]
    assert p["factor_benchmark"] is ds.LIBRARIES["iffm"]


def test_registry_internal_consistency():
    for m in ds.PLATFORM_MARKETS:
        assert m in ds.CURRENCY and m in ds.KEN_FRENCH_REGION
    assert set(ds.REGULATORY_FILINGS) - set(ds.PLATFORM_MARKETS) == {"IN"}
    for m, f in ds.REGULATORY_FILINGS.items():
        assert {"system", "url", "forms", "has_filing_dates"} <= set(f), m


def test_currency_table_agrees_with_fx_module():
    import fx
    for m, ccy in fx.MARKET_CCY.items():
        assert ds.CURRENCY[m] == ccy


def test_fmt_lib():
    assert ds._fmt_lib("iffm") == "iffm (public/free)"
    assert ds._fmt_lib("aqr") == "aqr (public/free (registration))"


# ── CLI ───────────────────────────────────────────────────────────────────────
def _main(monkeypatch, capsys, *argv):
    monkeypatch.setattr("sys.argv", ["data_sources.py", *argv])
    ds.main()
    return capsys.readouterr().out


def test_main_paper(monkeypatch, capsys):
    out = _main(monkeypatch, capsys, "--paper")
    assert "raw returns + fundamentals: CMIE Prowess" in out
    assert "-> COMMERCIAL (subscription)  (https://prowess.cmie.com)" in out
    assert "4-factor benchmark:         IIMA Indian Fama-French-Momentum" in out
    assert out.rstrip().endswith("Only the IFFM factor library is public; CMIE Prowess is commercial.")


def test_main_public_lists_every_library(monkeypatch, capsys):
    out = _main(monkeypatch, capsys, "--public")
    for key in ds.LIBRARIES:
        assert f"  [{key}] {ds.LIBRARIES[key]['name']}" in out
    assert out.count("provides:") == 3


def test_main_market_detail_country_and_global(monkeypatch, capsys):
    out = _main(monkeypatch, capsys, "--market", "US")
    assert out.splitlines()[0] == "=== US (USD) ==="
    assert "  Ken French region : North America" in out
    assert "  AQR country QMJ    : yes" in out
    assert "  raw price/fund     : Yahoo Finance (yfinance), SEC EDGAR XBRL" in out
    assert "aqr (public/free (registration)), ken_french (public/free)" in out
    out2 = _main(monkeypatch, capsys, "--market", "kr")
    assert "AQR country QMJ    : no (use AQR global)" in out2
    assert "       - https://www.aqr.com/Insights/Datasets" in out2


def test_main_default_table(monkeypatch, capsys):
    out = _main(monkeypatch, capsys)
    lines = out.splitlines()
    assert lines[0].startswith("mkt ccy  KenFrench region")
    assert lines[1] == "-" * 78
    us = next(line for line in lines if line.startswith("US "))
    assert "USD" in us and "North America" in us and "country" in us and "aqr, ken_french" in us
    kr = next(line for line in lines if line.startswith("KR "))
    assert "global" in kr
    inn = next(line for line in lines if line.startswith("IN "))
    assert "iffm, aqr, ken_french" in inn
    assert len([line for line in lines[2:] if line[:2].isupper() and line[2] == " "]) == 20
    assert "Raw pipeline sources: yfinance (all)" in out
