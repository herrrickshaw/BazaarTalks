"""Offline tests for brazil_cvm.py: CVM CSV parsers, filed-date panel build, zip cache/fetch, CLI."""
import datetime as _dt
import io
import os
import sys
import zipfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import apiclient  # noqa: E402
import brazil_cvm as cvm  # noqa: E402


def csv_bytes(header, rows, enc="latin-1"):
    lines = [";".join(header)] + [";".join(str(x) for x in r) for r in rows]
    return ("\n".join(lines) + "\n").encode(enc)


HDR = ["CNPJ_CIA", "DT_REFER", "VERSAO", "DENOM_CIA", "CD_CVM", "CATEG_DOC", "DT_RECEB"]
STM = ["CD_CVM", "DT_REFER", "VERSAO", "DT_FIM_EXERC", "ESCALA_MOEDA", "CD_CONTA", "DS_CONTA", "VL_CONTA"]


def header_csv():
    return csv_bytes(HDR, [
        ["11.111.111/0001-11", "2023-12-31", "1", "Companhia São João", "4170", "DFP", "2024-03-01"],
        ["11.111.111/0001-11", "2023-12-31", "2", "Companhia São João", "4170", "DFP", "2024-09-15"],
        ["22.222.222/0001-22", "2023-12-31", "1", "Banco X", "1023", "DFP", "2024-02-20"],
    ])


# ── small helpers ─────────────────────────────────────────────────────────────
def test_norm_strips_accents_case_and_whitespace():
    assert cvm._norm("  Patrimônio Líquido ") == "patrimonio liquido"
    assert cvm._norm(None) == ""
    assert cvm._norm("ÁÉÍÓÚ ç") == "aeiou c"


def test_d_parses_prefix_and_empty():
    assert cvm._d("2024-03-01") == _dt.date(2024, 3, 1)
    assert cvm._d("2024-03-01T00:00:00") == _dt.date(2024, 3, 1)
    assert cvm._d("") is None
    with pytest.raises(ValueError):
        cvm._d("01/03/2024")


@pytest.mark.parametrize("raw,expected", [
    ("12.5", 12.5), ("12,5", 12.5), (" -7 ", -7.0), ("0", 0.0), ("", None),
    (None, None), ("abc", None), ("1.234,5", None),
])
def test_num(raw, expected):
    assert cvm._num(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("MIL", 1000.0), ("mil", 1000.0), (" Mil ", 1000.0), ("UNIDADE", 1.0), ("", 1.0), (None, 1.0),
])
def test_scale(raw, expected):
    assert cvm._scale(raw) == expected


def test_plausible_roe_band_is_inclusive():
    assert cvm.plausible_roe(0.0) and cvm.plausible_roe(1.0) and cvm.plausible_roe(-1.0)
    assert not cvm.plausible_roe(1.0001) and not cvm.plausible_roe(-1.0001)
    assert not cvm.plausible_roe(None)


def test_reader_accepts_bytes_str_and_bytearray():
    raw = csv_bytes(["A", "B"], [["x", "y"]])
    assert list(cvm._reader(raw)) == [{"A": "x", "B": "y"}]
    assert list(cvm._reader(raw.decode("latin-1"))) == [{"A": "x", "B": "y"}]
    assert list(cvm._reader(bytearray(raw))) == [{"A": "x", "B": "y"}]


# ── parse_header ──────────────────────────────────────────────────────────────
def test_parse_header_keys_by_company_period_version_with_latin1_names():
    h = cvm.parse_header(header_csv())
    assert set(h) == {("4170", "2023-12-31", "1"), ("4170", "2023-12-31", "2"),
                      ("1023", "2023-12-31", "1")}
    assert h[("4170", "2023-12-31", "2")] == {
        "filed_date": "2024-09-15", "name": "Companhia São João",
        "cnpj": "11.111.111/0001-11", "categ": "DFP"}


def test_parse_header_strips_whitespace_and_tolerates_missing_optional_cols():
    raw = csv_bytes(["CD_CVM", "DT_REFER", "VERSAO"], [[" 7 ", " 2023-12-31 ", " 1 "]])
    h = cvm.parse_header(raw)
    assert h == {("7", "2023-12-31", "1"): {"filed_date": "", "name": "", "cnpj": "", "categ": ""}}


# ── parse_statement ───────────────────────────────────────────────────────────
WANTED = {"3.01": "revenue", "3.11": "net_income"}


def stmt_rows():
    return [
        ["4170", "2023-12-31", "1", "2023-12-31", "MIL", "3.01", "Receita", "1000"],
        ["4170", "2023-12-31", "1", "2023-12-31", "MIL", "3.11", "Lucro", "150,5"],
        ["4170", "2023-12-31", "1", "2022-12-31", "MIL", "3.11", "Lucro ant", "999"],   # prior period
        ["4170", "2023-12-31", "1", "2023-12-31", "MIL", "3.02", "Custo", "5"],          # unwanted
        ["1023", "2023-12-31", "1", "2023-12-31", "UNIDADE", "3.11", "Lucro", "42"],
        ["1023", "2023-12-31", "1", "2023-12-31", "UNIDADE", "3.01", "Receita", "n/a"],  # unparseable
    ]


def test_parse_statement_scales_filters_period_and_codes():
    out = cvm.parse_statement(csv_bytes(STM, stmt_rows()), WANTED)
    assert out == {
        ("4170", "2023-12-31", "1"): {"revenue": 1_000_000.0, "net_income": 150_500.0},
        ("1023", "2023-12-31", "1"): {"net_income": 42.0},
    }


def test_parse_statement_empty_and_no_matches():
    assert cvm.parse_statement(csv_bytes(STM, []), WANTED) == {}
    assert cvm.parse_statement(csv_bytes(STM, stmt_rows()), {"9.99": "x"}) == {}


# ── parse_equity ──────────────────────────────────────────────────────────────
def test_parse_equity_matches_depth2_patrimonio_liquido_for_industrial_and_bank():
    rows = [
        ["4170", "2023-12-31", "1", "2023-12-31", "MIL", "2.03", "Patrimônio Líquido", "500"],
        ["4170", "2023-12-31", "1", "2023-12-31", "MIL", "2.03.01", "Patrimônio Líquido sub", "1"],  # depth 3
        ["4170", "2023-12-31", "1", "2023-12-31", "MIL", "2.01", "Passivo Circulante", "7"],       # not PL
        ["1023", "2023-12-31", "1", "2023-12-31", "UNIDADE", "2.07", "PATRIMONIO LIQUIDO CONSOLIDADO", "900"],
        ["1023", "2023-12-31", "1", "2022-12-31", "UNIDADE", "2.07", "Patrimônio Líquido", "1"],   # prior period
        ["9999", "2023-12-31", "1", "2023-12-31", "UNIDADE", "2.03", "Patrimônio Líquido", "x"],   # unparseable
    ]
    out = cvm.parse_equity(csv_bytes(STM, rows))
    assert out == {("4170", "2023-12-31", "1"): {"equity": 500_000.0},
                   ("1023", "2023-12-31", "1"): {"equity": 900.0}}


def test_parse_equity_description_must_start_with_patrimonio_liquido():
    rows = [["1", "2023-12-31", "1", "2023-12-31", "UNIDADE", "2.03", "Total do Patrimônio Líquido", "5"]]
    assert cvm.parse_equity(csv_bytes(STM, rows)) == {}


# ── build_panel ───────────────────────────────────────────────────────────────
def test_build_panel_one_record_per_version_with_roe():
    header = cvm.parse_header(header_csv())
    st1 = cvm.parse_statement(csv_bytes(STM, stmt_rows()), WANTED)
    eq = {("4170", "2023-12-31", "1"): {"equity": 1_000_000.0},
          ("4170", "2023-12-31", "2"): {"equity": 0.0, "assets": 10.0},
          ("1023", "2023-12-31", "1"): {"equity": 21.0}}
    panel = cvm.build_panel(header, [st1, eq])
    by = {(r["cd_cvm"], r["version"]): r for r in panel}
    assert set(by) == {("4170", "1"), ("4170", "2"), ("1023", "1")}
    v1 = by[("4170", "1")]
    assert v1 == {"cd_cvm": "4170", "name": "Companhia São João", "filed_date": "2024-03-01",
                  "fiscal_end": "2023-12-31", "version": "1", "revenue": 1_000_000.0,
                  "net_income": 150_500.0, "assets": None, "equity": 1_000_000.0,
                  "roe": 0.1505}
    assert by[("4170", "2")]["roe"] is None                     # zero equity -> no ROE
    assert by[("4170", "2")]["assets"] == 10.0
    assert by[("1023", "1")]["roe"] == 2.0                      # implausible, kept for the caller to gate


def test_build_panel_drops_undated_and_empty_records():
    header = {("1", "2023-12-31", "1"): {"filed_date": "", "name": "A"},
              ("2", "2023-12-31", "1"): {"filed_date": "2024-01-01", "name": "B"},
              ("3", "2023-12-31", "1"): {"filed_date": "2024-01-01", "name": "C"}}
    st = {("1", "2023-12-31", "1"): {"revenue": 5.0}, ("3", "2023-12-31", "1"): {"assets": 0.0}}
    panel = cvm.build_panel(header, [st])
    assert [r["cd_cvm"] for r in panel] == ["3"]                # 1 undated, 2 no metrics; 0.0 counts as data
    assert cvm.build_panel({}, []) == []


def test_build_panel_missing_name_defaults_empty():
    panel = cvm.build_panel({("1", "d", "1"): {"filed_date": "2024-01-01"}},
                            [{("1", "d", "1"): {"net_income": 3.0, "equity": 6.0}}])
    assert panel[0]["name"] == "" and panel[0]["roe"] == 0.5


# ── zip fetch / cache ─────────────────────────────────────────────────────────
def make_zip(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, b in members.items():
            z.writestr(n, b)
    return buf.getvalue()


def full_members(year=2023, pfx="dfp", prefer="con", with_bpp=True, only_ind=False):
    kind = "ind" if only_ind else prefer
    m = {f"{pfx}_cia_aberta_{year}.csv": header_csv(),
         f"{pfx}_cia_aberta_DRE_{kind}_{year}.csv": csv_bytes(STM, stmt_rows() + [
             ["4170", "2023-12-31", "2", "2023-12-31", "MIL", "3.11", "Lucro", "160"]]),
         f"{pfx}_cia_aberta_BPA_{kind}_{year}.csv": csv_bytes(STM, [
             ["4170", "2023-12-31", "1", "2023-12-31", "MIL", "1", "Ativo Total", "9000"]])}
    if with_bpp:
        m[f"{pfx}_cia_aberta_BPP_{kind}_{year}.csv"] = csv_bytes(STM, [
            ["4170", "2023-12-31", "1", "2023-12-31", "MIL", "2.03", "Patrimônio Líquido", "1000"]])
    return m


class Resp:
    def __init__(self, content):
        self.content = content


def test_fetch_zip_reads_existing_cache_without_network(tmp_path, monkeypatch):
    (tmp_path / "dfp_cia_aberta_2023.zip").write_bytes(make_zip({"a.csv": b"x", "b.csv": b"y"}))
    monkeypatch.setattr(apiclient, "http_get",
                        lambda *a, **k: pytest.fail("network must not be used"))
    assert cvm.fetch_zip(2023, "DFP", str(tmp_path)) == {"a.csv": b"x", "b.csv": b"y"}


def test_fetch_zip_downloads_via_governed_client_and_caches(tmp_path, monkeypatch):
    calls = []

    def http_get(src, url, headers=None):
        calls.append((src, url, headers))
        return Resp(make_zip({"m.csv": b"1"}))
    monkeypatch.setattr(apiclient, "http_get", http_get)
    cache = tmp_path / "sub" / "cache"
    out = cvm.fetch_zip(2022, "ITR", str(cache))
    assert out == {"m.csv": b"1"}
    assert calls == [("cvm",
                      "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/ITR/DADOS/itr_cia_aberta_2022.zip",
                      {"User-Agent": "market-research admin@gms.dev"})]
    assert (cache / "itr_cia_aberta_2022.zip").exists()
    cvm.fetch_zip(2022, "ITR", str(cache))
    assert len(calls) == 1                                       # second call is a cache hit


def test_fetch_zip_falls_back_to_requests_when_governed_client_fails(tmp_path, monkeypatch):
    import requests

    def boom(*a, **k):
        raise RuntimeError("down")
    monkeypatch.setattr(apiclient, "http_get", boom)
    seen = {}

    def get(url, headers=None, timeout=None):
        seen.update(url=url, headers=headers, timeout=timeout)
        return Resp(make_zip({"z.csv": b"q"}))
    monkeypatch.setattr(requests, "get", get)
    assert cvm.fetch_zip(2021, "dfp", str(tmp_path)) == {"z.csv": b"q"}
    assert seen["timeout"] == 120 and seen["url"].endswith("/DFP/DADOS/dfp_cia_aberta_2021.zip")


# ── filed_panel ───────────────────────────────────────────────────────────────
def zip_into(tmp_path, members, year=2023, pfx="dfp"):
    (tmp_path / f"{pfx}_cia_aberta_{year}.zip").write_bytes(make_zip(members))


def test_filed_panel_end_to_end_from_cached_zip(tmp_path):
    zip_into(tmp_path, full_members())
    panel = cvm.filed_panel(2023, cache_dir=str(tmp_path))
    by = {(r["cd_cvm"], r["version"]): r for r in panel}
    r = by[("4170", "1")]
    assert (r["revenue"], r["net_income"], r["assets"], r["equity"]) == (
        1_000_000.0, 150_500.0, 9_000_000.0, 1_000_000.0)
    assert r["roe"] == pytest.approx(0.1505)
    assert by[("4170", "2")]["filed_date"] == "2024-09-15"       # restatement is its own observation
    assert by[("1023", "1")]["net_income"] == 42.0


def test_filed_panel_feeds_pit_as_of_without_lookahead(tmp_path):
    import pit_panel as pit
    zip_into(tmp_path, full_members())
    panel = cvm.filed_panel(2023, cache_dir=str(tmp_path))
    early = pit.as_of(panel, "2024-06-01", symbol_col="cd_cvm")
    late = pit.as_of(panel, "2024-12-31", symbol_col="cd_cvm")
    assert early["4170"]["version"] == "1" and late["4170"]["version"] == "2"
    assert pit.leaks_lookahead(panel, "2024-06-01") is True


def test_filed_panel_falls_back_to_individual_statements_and_skips_missing_bpp(tmp_path):
    zip_into(tmp_path, full_members(only_ind=True, with_bpp=False))
    panel = cvm.filed_panel(2023, cache_dir=str(tmp_path))
    r = next(x for x in panel if x["cd_cvm"] == "4170" and x["version"] == "1")
    assert r["net_income"] == 150_500.0 and r["assets"] == 9_000_000.0
    assert r["equity"] is None and r["roe"] is None


def test_filed_panel_prefers_requested_variant_and_supports_itr(tmp_path):
    members = full_members(year=2023, pfx="itr", prefer="ind")
    zip_into(tmp_path, members, pfx="itr")
    panel = cvm.filed_panel(2023, doc="ITR", prefer="ind", cache_dir=str(tmp_path))
    assert any(r["cd_cvm"] == "4170" and r["net_income"] == 150_500.0 for r in panel)


def test_filed_panel_missing_header_member_raises_keyerror(tmp_path):
    members = full_members()
    del members["dfp_cia_aberta_2023.csv"]
    zip_into(tmp_path, members)
    with pytest.raises(KeyError):
        cvm.filed_panel(2023, cache_dir=str(tmp_path))


# ── main ──────────────────────────────────────────────────────────────────────
def sample_panel():
    return [
        {"cd_cvm": "1", "name": "Big Company With A Rather Long Registered Name Ltda",
         "filed_date": "2024-03-01", "fiscal_end": "2023-12-31", "roe": 0.256, "assets": 500.0},
        {"cd_cvm": "2", "name": "Small", "filed_date": "2024-03-02", "fiscal_end": "2023-12-31",
         "roe": None, "assets": None},
        {"cd_cvm": "3", "name": "Mid", "filed_date": "2025-09-01", "fiscal_end": "2024-12-31",
         "roe": -0.1, "assets": 100.0},
    ]


def run_main(monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, "argv", ["brazil_cvm.py", *argv])
    cvm.main()
    return capsys.readouterr().out


def test_main_prints_top_companies_by_assets_and_pit_summary(monkeypatch, capsys):
    got = []
    monkeypatch.setattr(cvm, "filed_panel", lambda y, d: got.append((y, d)) or sample_panel())
    out = run_main(monkeypatch, capsys, "2023", "itr")
    assert got == [(2023, "ITR")]
    assert "brazil_cvm.py — CVM ITR filed-date panel for 2023" in out
    assert "companies with a dated filing: 3" in out
    lines = [line for line in out.splitlines() if line.startswith("  CVM")]
    assert [line.split()[1] for line in lines] == ["1", "3", "2"]       # assets desc, None as 0
    assert "ROE  25.6%  Big Company With A Rather Long Reg" in lines[0]
    assert "ROE   n/a  Small" in lines[2]
    assert "pit_panel.as_of(2024-06-30) -> 2 companies known by then (look-ahead leak: True)" in out


def test_main_reports_fetch_failure_and_returns(monkeypatch, capsys):
    def boom(y, d):
        raise RuntimeError("no network")
    monkeypatch.setattr(cvm, "filed_panel", boom)
    out = run_main(monkeypatch, capsys, "2020")
    assert "(fetch failed: no network)" in out
    assert "companies with a dated filing" not in out


def test_main_default_year_is_previous_calendar_year(monkeypatch, capsys):
    class FakeDT(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return _dt.datetime(2026, 6, 1)
    monkeypatch.setattr(cvm, "datetime", FakeDT)
    got = []
    monkeypatch.setattr(cvm, "filed_panel", lambda y, d: got.append((y, d)) or [])
    out = run_main(monkeypatch, capsys)
    assert got == [(2025, "DFP")]
    assert "companies with a dated filing: 0" in out


def test_main_swallows_pit_demo_errors(monkeypatch, capsys):
    import pit_panel
    monkeypatch.setattr(cvm, "filed_panel", lambda y, d: sample_panel())

    def boom(*a, **k):
        raise RuntimeError("pit broke")
    monkeypatch.setattr(pit_panel, "as_of", boom)
    out = run_main(monkeypatch, capsys, "2023")
    assert "companies with a dated filing: 3" in out
    assert "pit_panel.as_of" not in out
