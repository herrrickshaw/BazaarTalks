"""Offline tests for benchmark.py (Ken French factors + alpha regression)."""
import io
import os
import sys
import types
import zipfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import benchmark  # noqa: E402


FF_TEXT = ("header junk\n\n"
           ",Mkt-RF,SMB,HML,RMW,CMA,RF\n"
           "20240102, 1.00, 0.50, -0.20, 0.10, 0.05, 0.02\n"
           "20240103, -0.50, -0.25, 0.30, 0.20, 0.00, 0.02\n")


def _zip_bytes(text, name="f.csv"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name, text.encode("latin-1"))
    return buf.getvalue()


class _Resp:
    def __init__(self, status_code=200, content=b""):
        self.status_code = status_code
        self.content = content


@pytest.fixture
def fake_apiclient(monkeypatch):
    mod = types.ModuleType("apiclient")
    mod.calls = []
    mod.responses = {}

    def http_get(source, url, headers=None, timeout=None, retries=None):
        mod.calls.append((source, url, headers, timeout, retries))
        r = mod.responses.get(url.rsplit("/", 1)[-1])
        if isinstance(r, Exception):
            raise r
        return r if r is not None else _Resp(404)

    mod.http_get = http_get
    monkeypatch.setitem(sys.modules, "apiclient", mod)
    return mod


@pytest.fixture
def fake_parquet(monkeypatch, tmp_path):
    """No pyarrow in CI: store 'parquet' as pickle."""
    monkeypatch.setattr(pd, "read_parquet", lambda p, *a, **k: pd.read_pickle(p))
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, p, *a, **k: self.to_pickle(p))
    monkeypatch.setattr(benchmark, "CACHE_DIR", str(tmp_path / "cache"))


# ── parse_ff_csv ────────────────────────────────────────────────────────────
def test_parse_no_header_raises():
    with pytest.raises(ValueError, match="no factor header"):
        benchmark.parse_ff_csv("nothing\n20240102,1,2\n")


def test_parse_mom_header_and_missing_codes():
    text = ("Momentum\n,Mom   \n"
            "20240102, 1.5\n"
            "20240103, -99.99\n"
            "20240104, -999\n"
            "20240105, -98.9\n")
    df = benchmark.parse_ff_csv(text)
    assert list(df.columns) == ["Mom"]
    assert df["Mom"].iloc[0] == pytest.approx(0.015)
    assert np.isnan(df["Mom"].iloc[1]) and np.isnan(df["Mom"].iloc[2])
    assert df["Mom"].iloc[3] == pytest.approx(-0.989)      # just above the -99 cutoff is kept


def test_parse_exactly_minus_99_is_missing():
    df = benchmark.parse_ff_csv(",Mkt-RF,RF\n20240102,-99.0,0.1\n20240103,-98.99,0.1\n")
    assert np.isnan(df["Mkt-RF"].iloc[0])
    assert df["Mkt-RF"].iloc[1] == pytest.approx(-0.9899)


def test_parse_skips_bad_rows_and_stops_at_first_non_date_after_data():
    text = (",Mkt-RF,RF\n"
            "garbage line\n"           # before data: skipped
            "20240102,abc,0.1\n"        # unparsable float: skipped, not a block end
            "2024010,1.0,0.1\n"         # 7 digits: not a date -> before data, skipped
            "20240103,2.0,0.1\n"
            "\n"                        # blank ends the block
            "20240104,3.0,0.1\n")
    df = benchmark.parse_ff_csv(text)
    assert list(df.index.strftime("%Y%m%d")) == ["20240103"]
    assert df["Mkt-RF"].iloc[0] == pytest.approx(0.02)


def test_parse_header_columns_trimmed_and_extra_values_ignored():
    df = benchmark.parse_ff_csv(", Mkt-RF , RF ,\n20240102,1,2,3,4\n")
    assert list(df.columns) == ["Mkt-RF", "RF"]            # trailing blank header dropped
    assert df.iloc[0].tolist() == pytest.approx([0.01, 0.02])   # extra values ignored


def test_parse_header_without_mkt_rf_not_recognised():
    with pytest.raises(ValueError):
        benchmark.parse_ff_csv(",A,B\n20240102,1,2\n")


def test_parse_only_header_gives_empty_frame():
    df = benchmark.parse_ff_csv(",Mkt-RF,RF\n")
    assert df.empty


# ── _download_member ────────────────────────────────────────────────────────
def test_download_ok_uses_kenfrench_url_and_latin1(fake_apiclient):
    fake_apiclient.responses["x.zip"] = _Resp(200, _zip_bytes("caf\xe9"))
    out = benchmark._download_member("x.zip")
    assert out == "caf\xe9"
    src, url, headers, timeout, retries = fake_apiclient.calls[0]
    assert (src, url, timeout, retries) == ("kenfrench", benchmark.KF_BASE + "x.zip", 30, 2)
    assert headers == benchmark._UA


def test_download_non_200_is_none(fake_apiclient):
    fake_apiclient.responses["x.zip"] = _Resp(500, b"")
    assert benchmark._download_member("x.zip") is None


def test_download_exception_is_none_and_logged(fake_apiclient, capsys):
    fake_apiclient.responses["x.zip"] = RuntimeError("boom")
    assert benchmark._download_member("x.zip") is None
    assert "x.zip unavailable: boom" in capsys.readouterr().err


def test_download_bad_zip_is_none(fake_apiclient):
    fake_apiclient.responses["x.zip"] = _Resp(200, b"not a zip")
    assert benchmark._download_member("x.zip") is None


# ── factors ─────────────────────────────────────────────────────────────────
MOM_TEXT = ",Mom\n20240102, 2.0\n20240103, -1.0\n"


def test_factors_unknown_region_none():
    assert benchmark.factors("Atlantis") is None


def test_factors_download_join_and_cache(fake_apiclient, fake_parquet, tmp_path):
    spec = benchmark.REGION_FILES["Europe"]
    fake_apiclient.responses[spec["ff5"]] = _Resp(200, _zip_bytes(FF_TEXT))
    fake_apiclient.responses[spec["mom"]] = _Resp(200, _zip_bytes(MOM_TEXT))
    df = benchmark.factors("Europe")
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF", "Mom"]
    assert df["Mom"].tolist() == pytest.approx([0.02, -0.01])
    assert os.path.exists(tmp_path / "cache" / "kf_Europe.parquet")
    n_calls = len(fake_apiclient.calls)
    again = benchmark.factors("Europe")           # served from cache, no network
    assert len(fake_apiclient.calls) == n_calls
    pd.testing.assert_frame_equal(df, again)


def test_factors_cache_filename_replaces_spaces(fake_apiclient, fake_parquet, tmp_path):
    spec = benchmark.REGION_FILES["Asia Pacific ex Japan"]
    fake_apiclient.responses[spec["ff5"]] = _Resp(200, _zip_bytes(FF_TEXT))
    benchmark.factors("Asia Pacific ex Japan")
    assert os.path.exists(tmp_path / "cache" / "kf_Asia_Pacific_ex_Japan.parquet")


def test_factors_use_cache_false_redownloads(fake_apiclient, fake_parquet, tmp_path):
    spec = benchmark.REGION_FILES["Japan"]
    fake_apiclient.responses[spec["ff5"]] = _Resp(200, _zip_bytes(FF_TEXT))
    stale = pd.DataFrame({"stale": [1]})
    os.makedirs(tmp_path / "cache")
    stale.to_pickle(tmp_path / "cache" / "kf_Japan.parquet")
    assert list(benchmark.factors("Japan")) == ["stale"]
    fresh = benchmark.factors("Japan", use_cache=False)
    assert "Mkt-RF" in fresh.columns and "stale" not in fresh.columns


def test_factors_ff5_unavailable_none(fake_apiclient, fake_parquet):
    assert benchmark.factors("Europe") is None


def test_factors_momentum_missing_still_returns_ff5(fake_apiclient, fake_parquet):
    spec = benchmark.REGION_FILES["Europe"]
    fake_apiclient.responses[spec["ff5"]] = _Resp(200, _zip_bytes(FF_TEXT))
    df = benchmark.factors("Europe")
    assert "Mom" not in df.columns and len(df) == 2


def test_factors_momentum_unparseable_is_ignored(fake_apiclient, fake_parquet):
    spec = benchmark.REGION_FILES["Europe"]
    fake_apiclient.responses[spec["ff5"]] = _Resp(200, _zip_bytes(FF_TEXT))
    fake_apiclient.responses[spec["mom"]] = _Resp(200, _zip_bytes("garbage"))
    df = benchmark.factors("Europe")
    assert "Mom" not in df.columns


def test_factors_mom_left_join_keeps_ff5_dates(fake_apiclient, fake_parquet):
    spec = benchmark.REGION_FILES["Europe"]
    fake_apiclient.responses[spec["ff5"]] = _Resp(200, _zip_bytes(FF_TEXT))
    fake_apiclient.responses[spec["mom"]] = _Resp(200, _zip_bytes(",Mom\n20240102, 2.0\n"))
    df = benchmark.factors("Europe")
    assert len(df) == 2 and np.isnan(df["Mom"].iloc[1])


def test_emerging_uses_developed_proxy_files():
    assert benchmark.REGION_FILES["Emerging"]["ff5"].startswith("Developed_")
    assert benchmark.REGION_FILES["Emerging"]["mom"].startswith("Developed_")


# ── carhart_alpha ───────────────────────────────────────────────────────────
def _fac(n, cols=("Mkt-RF", "SMB", "HML", "Mom"), seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.DataFrame({c: rng.normal(0, 0.01, n) for c in cols}, index=idx)


def test_carhart_too_few_overlap_boundary():
    fac = _fac(30)
    y29 = pd.Series(fac["Mkt-RF"].values, index=fac.index)
    short = benchmark.carhart_alpha(y29.iloc[:29], fac)
    assert short == {"n": 29, "alpha_daily": None}
    ok = benchmark.carhart_alpha(y29, fac)
    assert ok["n"] == 30 and ok["alpha_daily"] is not None


def test_carhart_nan_rows_dropped_before_count():
    fac = _fac(40)
    y = pd.Series(fac["Mkt-RF"].values * 1.1 + np.linspace(0, 1e-4, 40), index=fac.index)
    y.iloc[:11] = np.nan                       # 29 left -> too few
    assert benchmark.carhart_alpha(y, fac)["alpha_daily"] is None
    assert benchmark.carhart_alpha(y, fac)["n"] == 29


def test_carhart_reliable_boundary_399_vs_400():
    for n, expect in [(399, False), (400, True)]:
        fac = _fac(n, seed=3)
        y = 0.9 * fac["Mkt-RF"] + np.random.default_rng(5).normal(0, 1e-3, n)
        assert benchmark.carhart_alpha(y, fac)["alpha_reliable"] is expect


def test_carhart_without_mom_uses_three_factors():
    fac = _fac(60, cols=("Mkt-RF", "SMB", "HML", "RMW"))
    y = 0.5 * fac["Mkt-RF"] + np.random.default_rng(1).normal(0, 1e-3, 60)
    res = benchmark.carhart_alpha(y, fac)
    assert list(res["loadings"]) == ["Mkt-RF", "SMB", "HML"]      # RMW not used, no Mom


def test_carhart_alpha_monthly_and_fields():
    fac = _fac(100, seed=9)
    y = 0.002 + 1.0 * fac["Mkt-RF"]               # exact fit
    res = benchmark.carhart_alpha(y, fac)
    assert res["alpha_daily"] == pytest.approx(0.002, abs=1e-4)
    assert res["alpha_monthly%"] == round(res["alpha_daily"] * 21 * 100, 3)
    assert res["R2"] == pytest.approx(1.0)
    assert res["n"] == 100
    assert set(res["loadings"]) == {"Mkt-RF", "SMB", "HML", "Mom"}


def test_carhart_misaligned_index_inner_joins():
    fac = _fac(50)
    y = pd.Series(np.random.default_rng(2).normal(0, 0.01, 50),
                  index=pd.date_range("2030-01-03", periods=50, freq="B"))
    res = benchmark.carhart_alpha(y, fac)
    assert res == {"n": 0, "alpha_daily": None}


# ── factor_premia ───────────────────────────────────────────────────────────
def test_factor_premia_values_columns_and_dates():
    idx = pd.date_range("2023-01-02", periods=4, freq="B")
    fac = pd.DataFrame({"A": [0.01, 0.03, 0.01, 0.03], "B": [0.0, np.nan, 0.0, 0.0]}, index=idx)
    p = benchmark.factor_premia(fac).set_index("factor")
    assert p.loc["A", "ann_mean%"] == round(0.02 * 252 * 100, 2)
    vol = np.std([0.01, 0.03, 0.01, 0.03], ddof=1) * np.sqrt(252)
    assert p.loc["A", "ann_vol%"] == round(vol * 100, 2)
    assert p.loc["A", "sharpe"] == round(0.02 * 252 / vol, 2)
    assert p.loc["A", "n_days"] == 4 and p.loc["A", "from"] == "2023-01-02"
    assert p.loc["B", "n_days"] == 3                          # NaN dropped
    assert np.isnan(p.loc["B", "sharpe"])                     # zero vol -> NaN sharpe


# ── _clean ─────────────────────────────────────────────────────────────────
def test_clean_strips_suffix_and_uppercases():
    assert benchmark._clean("abc.ns") == "ABC"
    assert benchmark._clean("BRK.B.US") == "BRK"              # only the first segment kept
    assert benchmark._clean(123) == "123"


# ── quality_lq_returns ──────────────────────────────────────────────────────
@pytest.fixture
def fake_qf(monkeypatch):
    qf = types.ModuleType("quality_factor")
    qf.scored = pd.DataFrame({"ticker": ["aaa", "BBB.L", "ccc", "ddd", "eee"],
                              "quality": [1.0, 2.0, 3.0, 4.0, np.nan]})
    qf.fund_empty = False
    qf.load_calls = []

    def load_fundamentals(markets):
        qf.load_calls.append(markets)
        return pd.DataFrame() if qf.fund_empty else pd.DataFrame({"ticker": ["x"]})

    qf.load_fundamentals = load_fundamentals
    qf.attach_price_risk = lambda f: f
    qf.score_universe = lambda f, by_market=True: qf.scored.copy()
    qf.assign_deciles = lambda s: pd.Series(
        ["quality" if v >= 2.0 and v <= 3.0 else "other" for v in s], index=s.index)
    monkeypatch.setitem(sys.modules, "quality_factor", qf)
    return qf


def _prices(symbols, n=4):
    rows = []
    for k, s in enumerate(symbols):
        for i in range(n):
            rows.append({"Symbol": s, "Date": f"2024-01-0{i + 1}", "Close": 100.0 * (1 + 0.01 * (k + 1)) ** i})
    return pd.DataFrame(rows)


def test_lq_empty_fundamentals_none(fake_qf):
    fake_qf.fund_empty = True
    assert benchmark.quality_lq_returns("US") is None
    assert fake_qf.load_calls == [["US"]]


def test_lq_all_quality_nan_none(fake_qf):
    fake_qf.scored = pd.DataFrame({"ticker": ["a"], "quality": [np.nan]})
    assert benchmark.quality_lq_returns("US") is None


def test_lq_missing_price_file_none(fake_qf, monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "SEED", str(tmp_path))
    assert benchmark.quality_lq_returns("US") is None


def test_lq_no_quality_decile_none(fake_qf, monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "SEED", str(tmp_path))
    (tmp_path / "cleaned_long_US.parquet").write_text("x")
    fake_qf.assign_deciles = lambda s: pd.Series(["other"] * len(s), index=s.index)
    assert benchmark.quality_lq_returns("US") is None


def test_lq_equal_weight_return_and_filter(fake_qf, monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "SEED", str(tmp_path))
    (tmp_path / "cleaned_long_US.parquet").write_text("x")
    px = _prices(["AAA", "BBB", "CCC", "DDD"])           # LQ keys: BBB, CCC only (+ none) -> <3
    monkeypatch.setattr(pd, "read_parquet", lambda p, *a, **k: px.copy())
    assert benchmark.quality_lq_returns("US") is None    # only 2 LQ symbols present -> < 3

    fake_qf.scored = pd.DataFrame({"ticker": ["aaa", "BBB.L", "ccc", "ddd"],
                                   "quality": [2.0, 2.0, 3.0, 4.0]})
    ret = benchmark.quality_lq_returns("US")             # LQ = AAA, BBB, CCC
    assert isinstance(ret.index, pd.DatetimeIndex)
    assert len(ret) == 3                                  # first pct_change NaN dropped
    exp = np.mean([1.01 - 1, 1.02 - 1, 1.03 - 1])
    assert ret.iloc[0] == pytest.approx(exp)             # DDD (not LQ) excluded


# ── validate_quality ────────────────────────────────────────────────────────
def test_validate_unknown_market_error():
    assert benchmark.validate_quality("ZZ") == {"error": "no Ken French region for ZZ"}


def test_validate_factors_unavailable(monkeypatch):
    monkeypatch.setattr(benchmark, "factors", lambda r: None)
    assert "unavailable" in benchmark.validate_quality("US")["error"]


def test_validate_portfolio_none(monkeypatch):
    monkeypatch.setattr(benchmark, "factors", lambda r: _fac(5).assign(RF=0.0))
    monkeypatch.setattr(benchmark, "quality_lq_returns", lambda m: None)
    assert benchmark.validate_quality("US")["error"] == "could not form LQ portfolio for US"


def test_validate_end_to_end_lowercase_market(monkeypatch):
    fac = _fac(120, seed=4)
    fac["RF"] = 0.0001
    port = pd.Series(0.0001 + 0.8 * fac["Mkt-RF"], index=fac.index)
    seen = {}
    monkeypatch.setattr(benchmark, "factors", lambda r: seen.setdefault("r", r) and fac)
    monkeypatch.setattr(benchmark, "quality_lq_returns", lambda m: port)
    res = benchmark.validate_quality("us")
    assert res["market"] == "us" and res["region"] == seen["r"]
    assert res["loadings"]["Mkt-RF"][0] == pytest.approx(0.8, abs=0.01)
    assert res["alpha_daily"] == pytest.approx(0.0, abs=1e-4)   # excess = port - RF removes the 1bp


# ── main ────────────────────────────────────────────────────────────────────
def _run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["benchmark.py", *argv])
    benchmark.main()


def test_main_region_prints_premia(monkeypatch, capsys):
    seen = {}
    fac = pd.DataFrame({"Mkt-RF": [0.01, 0.02, 0.03]}, index=pd.date_range("2024-01-01", periods=3))
    monkeypatch.setattr(benchmark, "factors", lambda r, use_cache=True: seen.update(r=r, uc=use_cache) or fac)
    _run_main(monkeypatch, "--region", "Japan", "--no-cache")
    out = capsys.readouterr().out
    assert seen == {"r": "Japan", "uc": False}
    assert "=== real Ken French factor premia — Japan ===" in out
    assert "Mkt-RF" in out and "source: Kenneth French Data Library" in out


def test_main_default_region_from_market(monkeypatch, capsys):
    import data_sources
    seen = {}
    monkeypatch.setattr(benchmark, "factors", lambda r, use_cache=True: seen.update(r=r, uc=use_cache))
    _run_main(monkeypatch, "--market", "jp")
    assert seen == {"r": data_sources.KEN_FRENCH_REGION["JP"], "uc": True}
    assert "unavailable (offline?)" in capsys.readouterr().out


def test_main_unknown_market_falls_back_to_north_america(monkeypatch):
    seen = {}
    monkeypatch.setattr(benchmark, "factors", lambda r, use_cache=True: seen.update(r=r))
    _run_main(monkeypatch, "--market", "ZZ")
    assert seen["r"] == "North America"


def test_main_default_market_is_us(monkeypatch):
    import data_sources
    seen = {}
    monkeypatch.setattr(benchmark, "factors", lambda r, use_cache=True: seen.update(r=r))
    _run_main(monkeypatch)
    assert seen["r"] == data_sources.KEN_FRENCH_REGION["US"]


def test_main_validate_error(monkeypatch, capsys):
    monkeypatch.setattr(benchmark, "validate_quality", lambda m: {"error": "nope"})
    _run_main(monkeypatch, "--validate-quality", "--market", "DE")
    out = capsys.readouterr().out
    assert "QUALITY (LQ) ALPHA vs real Ken French factors — DE" in out and "  nope" in out


def test_main_validate_default_market_us_and_too_few(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(benchmark, "validate_quality",
                        lambda m: seen.append(m) or {"alpha_daily": None, "n": 12})
    _run_main(monkeypatch, "--validate-quality")
    assert seen == ["US"]
    assert "too few overlapping days (12)" in capsys.readouterr().out


def _res(hml, mom, reliable):
    return {"region": "Europe", "n": 500, "R2": 0.5, "alpha_daily": 0.001, "alpha_t": 2.5,
            "alpha_reliable": reliable,
            "loadings": {"Mkt-RF": (1.0, 10.0), "HML": (hml, -3.0), "Mom": (mom, 1.0)}}


def test_main_validate_matches_paper(monkeypatch, capsys):
    monkeypatch.setattr(benchmark, "validate_quality", lambda m: _res(-0.3, 0.2, False))
    _run_main(monkeypatch, "--validate-quality")
    out = capsys.readouterr().out
    assert "region: Europe  ·  500 overlapping days  ·  R² = 0.5" in out
    assert "HML − , Mom +  (matches the paper" in out
    assert "(t=-3.0) *" in out and "(t=1.0) *" not in out          # |t|>2 flagged only
    assert "alpha (diagnostic only): 0.100%/day, t=2.5  [UNRELIABLE" in out


def test_main_validate_differs_and_reliable(monkeypatch, capsys):
    monkeypatch.setattr(benchmark, "validate_quality", lambda m: _res(0.3, -0.2, True))
    _run_main(monkeypatch, "--validate-quality")
    out = capsys.readouterr().out
    assert "HML + , Mom −  (differs from the paper" in out
    assert "UNRELIABLE" not in out


def test_main_validate_zero_loadings_differ(monkeypatch, capsys):
    monkeypatch.setattr(benchmark, "validate_quality", lambda m: _res(0.0, 0.0, True))
    _run_main(monkeypatch, "--validate-quality")
    out = capsys.readouterr().out
    assert "HML + , Mom −  (differs" in out                        # hml<0 and mom>0 both strict


def test_main_validate_missing_loadings_default_zero(monkeypatch, capsys):
    r = _res(0, 0, True)
    r["loadings"] = {"Mkt-RF": (1.0, 10.0)}
    monkeypatch.setattr(benchmark, "validate_quality", lambda m: r)
    _run_main(monkeypatch, "--validate-quality")
    assert "differs from the paper" in capsys.readouterr().out
