"""Offline tests for literature_scout.py (scoring core, faked scholarly APIs, report, CLI)."""
import json
import os
import sys
import types
from datetime import date

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import literature_scout as ls  # noqa: E402

FRONTIER = {"quantum": {"kw": ["quantum finance", "qubit"], "modules": []}}


# ── text helpers ──────────────────────────────────────────────────────────────
def test_normalise_strips_punctuation_keeps_hyphen_and_digits():
    assert ls.normalise("Fama-French (1992): Size & B/M!") == "fama-french  1992   size   b m "
    assert ls.normalise(None) == ""


def test_reconstruct_abstract_orders_by_position_and_repeats():
    inv = {"the": [0, 3], "cat": [1], "sat": [2], "mat": [4]}
    assert ls.reconstruct_abstract(inv) == "the cat sat the mat"
    assert ls.reconstruct_abstract({}) == "" and ls.reconstruct_abstract(None) == ""


def test_theme_hits_weights_phrases_double():
    assert ls._theme_hits("Momentum and value", ["momentum", "value"]) == 2.0
    assert ls._theme_hits("a quality factor study", ["quality factor"]) == 2.0
    assert ls._theme_hits("nothing relevant", ["momentum"]) == 0.0
    assert ls._theme_hits("MOMENTUM", ["momentum"]) == 1.0                  # case-insensitive


def test_match_themes_accepts_dict_and_list_specs():
    themes = {"a": {"kw": ["alpha"], "modules": []}, "b": ["beta", "gamma delta"], "c": ["zzz"]}
    assert ls.match_themes("alpha beta gamma delta", themes) == {"a": 1.0, "b": 3.0}


# ── score_paper ───────────────────────────────────────────────────────────────
def test_score_paper_exact_blend_single_keyword_current_year():
    p = ls.score_paper({"title": "momentum", "year": 2025, "citations": 0}, now_year=2025)
    # kw 1/6 * 0.6 + recency 1.0 * 0.2 + cites 0 = 0.3
    assert p["score"] == 0.3
    assert p["covered_themes"] == ["momentum"] and p["coverage"] == "covered"
    assert p["modules"] == ["dvm_global.py", "ml_signal_engine.py", "sector_rotation.py"]


def test_score_paper_recency_decays_linearly_over_40_years():
    base = {"title": "momentum", "citations": 0}
    s0 = ls.score_paper({**base, "year": 2025}, now_year=2025)["score"]
    s20 = ls.score_paper({**base, "year": 2005}, now_year=2025)["score"]
    s40 = ls.score_paper({**base, "year": 1985}, now_year=2025)["score"]
    s60 = ls.score_paper({**base, "year": 1965}, now_year=2025)["score"]
    assert (s0, s20, s40) == (0.3, 0.2, 0.1)
    assert s60 == s40                                       # clamped at 0 recency
    assert ls.score_paper({**base, "year": 2030}, now_year=2025)["score"] == s0   # future year: age clamped


def test_score_paper_citation_term_saturates_at_10k():
    base = {"title": "momentum", "year": 1985}
    s_hi = ls.score_paper({**base, "citations": 9999}, now_year=2025)["score"]
    s_big = ls.score_paper({**base, "citations": 10**7}, now_year=2025)["score"]
    assert s_hi == 0.3 and s_big == 0.3                     # 0.1 kw + 0 rec + 0.2 cites
    assert ls.score_paper({**base, "citations": None}, now_year=2025)["score"] == 0.1
    assert ls.score_paper({**base, "citations": 99}, now_year=2025)["score"] == pytest.approx(0.1 + 0.2 * 0.5, abs=1e-4)


def test_score_paper_keyword_term_saturates_at_six_weighted_hits():
    text = ("quality factor quality minus junk qmj profitability factor piotroski "
            "gross profitability earnings quality")
    p = ls.score_paper({"title": text, "year": 1985, "citations": 0}, now_year=2025)
    assert p["score"] == 0.6                                # kw_rel capped at 1.0


def test_score_paper_uses_abstract_and_ignores_missing_fields():
    p = ls.score_paper({"title": "A study", "abstract": "we use machine learning"}, now_year=2025)
    assert p["covered_themes"] == ["ml_pricing"]
    bare = ls.score_paper({}, now_year=2025)
    assert bare["coverage"] == "unmapped" and bare["modules"] == []
    assert bare["score"] == 0.2                              # year defaults to now -> full recency only


def test_score_paper_default_now_year_is_today(monkeypatch):
    class FakeDate(date):
        @classmethod
        def today(cls):
            return date(2030, 6, 1)
    monkeypatch.setattr(ls.dt, "date", FakeDate)
    p = ls.score_paper({"title": "momentum", "year": 2030, "citations": 0})
    assert p["score"] == 0.3


def test_score_paper_preserves_input_fields_and_does_not_mutate():
    paper = {"title": "momentum", "year": 2020, "doi": "10.1/x", "custom": 7}
    out = ls.score_paper(paper, now_year=2025)
    assert out["custom"] == 7 and out["doi"] == "10.1/x"
    assert "score" not in paper


def test_coverage_classification_with_frontier_themes(monkeypatch):
    monkeypatch.setattr(ls, "FRONTIER_THEMES", FRONTIER)
    gap = ls.score_paper({"title": "quantum finance with a qubit"}, now_year=2025)
    assert gap["coverage"] == "gap" and gap["frontier_themes"] == ["quantum"] and gap["covered_themes"] == []
    ext = ls.score_paper({"title": "momentum via quantum finance"}, now_year=2025)
    assert ext["coverage"] == "extends"
    cov = ls.score_paper({"title": "momentum"}, now_year=2025)
    assert cov["coverage"] == "covered"
    assert ls.score_paper({"title": "rocks"}, now_year=2025)["coverage"] == "unmapped"


@pytest.mark.xfail(strict=True, raises=TypeError,
                   reason="literature_scout.py:195 score_paper does int(paper.get('year', now_year)) which "
                          "raises on year=None (Crossref/OpenAlex/arXiv fetchers emit year None)")
def test_score_paper_tolerates_missing_year_none():
    p = ls.score_paper({"title": "momentum", "year": None, "citations": 0}, now_year=2025)
    assert 0.0 <= p["score"] <= 1.0


# ── dedup / rank / summary ───────────────────────────────────────────────────
def test_dedup_prefers_doi_case_insensitive_then_normalised_title():
    papers = [{"title": "A", "doi": "10.1/ABC"}, {"title": "B", "doi": "10.1/abc"},
              {"title": "Same  Title!", "doi": ""}, {"title": "same title", "doi": None},
              {"title": "", "doi": ""}, {"title": "Keep", "doi": ""}]
    out = ls.dedup(papers)
    assert [p["title"] for p in out] == ["A", "Same  Title!", "Keep"]    # keyless record dropped


def test_rank_sorted_descending_and_deduped():
    papers = [{"title": "momentum", "year": 2025, "citations": 0},
              {"title": "momentum", "year": 2025, "citations": 0},
              {"title": "momentum and machine learning and capm", "year": 2025, "citations": 100},
              {"title": "rocks", "year": 2025}]
    out = ls.rank(papers, now_year=2025)
    assert len(out) == 3
    assert [p["title"] for p in out][0].startswith("momentum and machine")
    assert [p["score"] for p in out] == sorted((p["score"] for p in out), reverse=True)


def test_coverage_summary_counts_and_zero_filled(monkeypatch):
    monkeypatch.setattr(ls, "FRONTIER_THEMES", FRONTIER)
    ranked = ls.rank([{"title": "momentum quantum finance"}, {"title": "momentum"}], now_year=2025)
    s = ls.coverage_summary(ranked)
    assert s["covered"]["momentum"] == 2 and s["covered"]["quality"] == 0
    assert set(s["covered"]) == set(ls.COVERED_THEMES)
    assert s["frontier"] == {"quantum": 1}


def test_seed_corpus_maps_to_expected_modules():
    ranked = {p["title"]: p for p in ls.rank(ls.SEED_PAPERS, now_year=2026)}
    qmj = ranked["Quality Minus Junk"]
    assert "quality_factor.py" in qmj["modules"] and qmj["coverage"] == "covered"
    assert ranked["Portfolio Selection"]["covered_themes"] == ["mean_variance"]


# ── fetchers ──────────────────────────────────────────────────────────────────
class _Resp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code, self._p, self.text = status, payload, text

    def json(self):
        return self._p


def _fake_api(monkeypatch, resp):
    api = types.ModuleType("apiclient")
    api.calls = []

    def http_get(source, url, timeout=45, retries=5):
        api.calls.append({"source": source, "url": url, "timeout": timeout, "retries": retries})
        if isinstance(resp, Exception):
            raise resp
        return resp
    api.http_get = http_get
    monkeypatch.setitem(sys.modules, "apiclient", api)
    return api


def test_require_ok_raises_on_non_200():
    r = _Resp(200)
    assert ls._require_ok(r, "x") is r
    with pytest.raises(RuntimeError, match="openalex HTTP 503"):
        ls._require_ok(_Resp(503), "openalex")
    with pytest.raises(RuntimeError, match="HTTP 429"):
        ls._require_ok(_Resp(429), "arxiv")


def test_openalex_parses_results_and_builds_url(monkeypatch):
    monkeypatch.delenv("SCOUT_MAILTO", raising=False)
    payload = {"results": [
        {"title": "T1", "abstract_inverted_index": {"hello": [0], "world": [1]},
         "publication_year": 2020, "cited_by_count": 7, "doi": "https://doi.org/10.1/x",
         "id": "https://openalex.org/W1",
         "authorships": [{"author": {"display_name": f"A{i}"}} for i in range(6)]},
        {"title": None, "abstract_inverted_index": None, "publication_year": None,
         "doi": None, "id": "W2", "authorships": None}]}
    api = _fake_api(monkeypatch, _Resp(200, payload))
    out = ls._openalex("quality factor", 120)
    call = api.calls[0]
    assert call["source"] == "openalex" and call["retries"] == 2 and call["timeout"] == 20
    assert "search=quality%20factor" in call["url"] and "per-page=50" in call["url"]   # capped at 50
    assert "mailto" not in call["url"]
    assert out[0] == {"source": "openalex", "title": "T1", "abstract": "hello world", "year": 2020,
                      "citations": 7, "doi": "10.1/x", "url": "https://openalex.org/W1",
                      "authors": "A0, A1, A2, A3"}                                    # first four authors
    assert out[1]["title"] == "" and out[1]["abstract"] == "" and out[1]["doi"] == "" \
        and out[1]["authors"] == "" and out[1]["citations"] == 0 and out[1]["year"] is None


def test_openalex_mailto_from_env_is_quoted(monkeypatch):
    monkeypatch.setenv("SCOUT_MAILTO", "me+x@example.org")
    api = _fake_api(monkeypatch, _Resp(200, {"results": []}))
    assert ls._openalex("q", 5) == []
    assert "&mailto=me%2Bx%40example.org" in api.calls[0]["url"] and "per-page=5&" in api.calls[0]["url"]


def test_openalex_non_200_raises(monkeypatch):
    _fake_api(monkeypatch, _Resp(503))
    with pytest.raises(RuntimeError):
        ls._openalex("q", 5)


def test_strip_jats():
    assert ls._strip_jats("<jats:p>Hello <b>world</b></jats:p>") == "Hello  world"
    assert ls._strip_jats(None) == "" and ls._strip_jats("plain") == "plain"


def test_crossref_parses_filters_types_and_dates(monkeypatch):
    monkeypatch.delenv("SCOUT_MAILTO", raising=False)
    payload = {"message": {"items": [
        {"type": "journal-article", "title": ["A", "Subtitle"], "abstract": "<jats:p>abs</jats:p>",
         "published": {"date-parts": [[2019, 5]]}, "is-referenced-by-count": 12, "DOI": "10.2/y",
         "author": [{"family": f"F{i}"} for i in range(5)]},
        {"type": "book-chapter", "title": ["skip me"]},
        {"title": ["no type is allowed"], "published": {"date-parts": [[None]]}},
        {"type": "posted-content", "title": [], "published": {"date-parts": [[]]}},
        {"type": "proceedings-article", "title": ["P"], "DOI": "10.3/z"}]}}
    api = _fake_api(monkeypatch, _Resp(200, payload))
    out = ls._crossref("quality", 99)
    assert "rows=50" in api.calls[0]["url"] and api.calls[0]["source"] == "crossref"
    assert len(out) == 4
    assert out[0] == {"source": "crossref", "title": "A Subtitle", "abstract": "abs", "year": 2019,
                      "citations": 12, "doi": "10.2/y", "url": "https://doi.org/10.2/y",
                      "authors": "F0, F1, F2, F3"}
    assert out[1]["title"] == "no type is allowed" and out[1]["year"] is None
    assert out[2]["title"] == "" and out[2]["year"] is None
    assert out[3]["year"] is None and out[3]["url"] == "https://doi.org/10.3/z"


def test_crossref_mailto(monkeypatch):
    monkeypatch.setenv("SCOUT_MAILTO", "a@b.co")
    api = _fake_api(monkeypatch, _Resp(200, {"message": {"items": []}}))
    ls._crossref("q", 5)
    assert api.calls[0]["url"].endswith("&mailto=a%40b.co")


ATOM = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
 <entry><id>http://arxiv.org/abs/1</id><title> Deep  Hedging </title><summary> sum </summary>
  <published>2021-03-04T00:00:00Z</published>
  <author><name>Ann</name></author><author><name>Bob</name></author></entry>
 <entry><id>http://arxiv.org/abs/2</id><title>T2</title><summary/><published>n/a</published></entry>
</feed>"""


def test_arxiv_parses_atom(monkeypatch):
    api = _fake_api(monkeypatch, _Resp(200, text=ATOM))
    out = ls._arxiv("deep hedging", 100)
    assert "all%3Adeep%20hedging" in api.calls[0]["url"] and "max_results=50" in api.calls[0]["url"]
    assert out[0] == {"source": "arxiv", "title": "Deep  Hedging", "abstract": "sum", "year": 2021,
                      "citations": 0, "doi": "", "url": "http://arxiv.org/abs/1", "authors": "Ann, Bob"}
    assert out[1]["year"] is None and out[1]["abstract"] == "" and out[1]["authors"] == ""


def test_fetch_collects_and_skips_failing_sources(monkeypatch, capsys):
    calls = []

    def ok(q, n):
        calls.append(("ok", q, n))
        return [{"title": "x"}, {"title": "y"}]

    def bad(q, n):
        raise RuntimeError("down")
    monkeypatch.setattr(ls, "_FETCHERS", {"openalex": bad, "crossref": ok, "arxiv": ok})
    got = ls.fetch("q", 7)
    assert len(got) == 4 and calls == [("ok", "q", 7)] * 2
    err = capsys.readouterr().err
    assert "openalex unavailable: down" in err and "crossref: 2 hits" in err


def test_fetch_respects_source_selection(monkeypatch):
    seen = []
    monkeypatch.setattr(ls, "_FETCHERS", {"openalex": lambda q, n: seen.append("o") or [],
                                          "crossref": lambda q, n: seen.append("c") or [],
                                          "arxiv": lambda q, n: seen.append("a") or []})
    assert ls.fetch("q", 1, sources=("arxiv", "openalex")) == [] and seen == ["a", "o"]


# ── report ────────────────────────────────────────────────────────────────────
def _ranked(monkeypatch, titles):
    monkeypatch.setattr(ls, "FRONTIER_THEMES", FRONTIER)
    papers = [{"title": t, "year": 2024, "citations": 5, "authors": "Au"} for t in titles]
    ranked = ls.rank(papers, now_year=2025)
    return ranked, ls.coverage_summary(ranked)


def test_render_report_gap_section_and_flags(monkeypatch):
    ranked, summ = _ranked(monkeypatch, ["quantum finance qubit", "momentum", "plain rocks"])
    md = ls.render_report(ranked, summ, query="my query", top=10)
    assert "Query: **my query**. 3 papers scored." in md
    assert "| **gap** | frontier |" in md                                  # gap row, 'frontier' module cell
    assert "| unmapped | — |" in md                                       # no modules, not a gap
    assert "| covered | dvm_global.py, ml_signal_engine.py |" in md     # only first two modules
    assert "| score | frontier theme(s) | paper |" in md
    assert "- `quantum` — 1 paper(s)  ⟵ opportunity" in md
    assert "- `momentum` — 1 paper(s) → dvm_global.py, ml_signal_engine.py, sector_rotation.py" in md
    assert md.endswith("\n")


def test_render_report_no_gaps_and_no_query(monkeypatch):
    monkeypatch.setattr(ls, "FRONTIER_THEMES", {})
    ranked = ls.rank([{"title": "momentum", "year": 2024}], now_year=2025)
    md = ls.render_report(ranked, ls.coverage_summary(ranked), None)
    assert "Query: **seed themes**" in md and "_None surfaced in this run._" in md
    assert "⟵ opportunity" not in md


def test_render_report_truncates_long_titles_and_honours_top(monkeypatch):
    monkeypatch.setattr(ls, "FRONTIER_THEMES", {})
    t71, t72 = "m" * 70 + "x", "m" * 71 + "x"
    ranked = ls.rank([{"title": t71 + " momentum", "year": 2024}, {"title": "momentum", "year": 2020}],
                     now_year=2025)
    md = ls.render_report(ranked, ls.coverage_summary(ranked), None, top=1)
    rows = [l for l in md.splitlines() if l.startswith("| 1 |") or l.startswith("| 2 |")]
    assert len(rows) == 1 and rows[0].startswith("| 1 |")
    assert "…" in rows[0] and ("m" * 70 + "x momentum")[:70] in rows[0]
    # boundary: exactly 71 chars is NOT truncated, 72 is
    r71 = ls.rank([{"title": t71, "year": 2024}], now_year=2025)
    r72 = ls.rank([{"title": t72, "year": 2024}], now_year=2025)
    assert "…" not in ls.render_report(r71, ls.coverage_summary(r71), None)
    assert "…" in ls.render_report(r72, ls.coverage_summary(r72), None)


# ── CLI ───────────────────────────────────────────────────────────────────────
def _argv(monkeypatch, *a):
    monkeypatch.setattr(sys, "argv", ["literature_scout.py", *a])


def test_main_offline_writes_report_and_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ls, "fetch", lambda *a, **k: pytest.fail("offline must not fetch"))
    out = tmp_path / "rep.md"
    _argv(monkeypatch, "--offline", "--out", str(out), "--top", "3")
    ls.main()
    assert "## Top relevant papers" in out.read_text()
    data = json.loads((tmp_path / "rep.json").read_text())
    assert len(data) == 3 and data[0]["score"] >= data[1]["score"] >= data[2]["score"]
    stdout = capsys.readouterr().out
    assert f"wrote {out} ({len(ls.SEED_PAPERS)} papers; 0 gaps)" in stdout
    assert stdout.count("\n  ") >= 10 and "top 10:" in stdout


def test_main_online_default_queries_and_custom_query(tmp_path, monkeypatch, capsys):
    qs = []
    monkeypatch.setattr(ls, "fetch", lambda q, n: qs.append((q, n)) or [{"title": "Extra momentum", "year": 2024}])
    _argv(monkeypatch, "--out", str(tmp_path / "a.md"))
    ls.main()
    assert [q for q, _ in qs] == ["equity factor investing", "cross-section of stock returns",
                                  "quality factor emerging markets", "machine learning asset pricing",
                                  "transaction costs factor strategy"] and {n for _, n in qs} == {20}
    assert f"scouted {len(ls.SEED_PAPERS) + 5} raw hits across 5 queries" in capsys.readouterr().err
    qs.clear()
    _argv(monkeypatch, "--query", "my topic", "--limit", "9", "--out", str(tmp_path / "b.md"))
    ls.main()
    assert qs == [("my topic", 9)]
    assert "Query: **my topic**" in (tmp_path / "b.md").read_text()


def test_main_default_out_path_is_beside_module(monkeypatch, tmp_path):
    written = {}

    class Sink:
        def __init__(self, name):
            self.name = name

        def write(self, s):
            written[self.name] = s

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def close(self):
            pass
    monkeypatch.setattr(ls, "open", lambda path, mode="r": Sink(path), raising=False)
    monkeypatch.setattr(ls.json, "dump", lambda obj, fh, **k: written.setdefault("json_fh", fh.name))
    _argv(monkeypatch, "--offline")
    ls.main()
    assert os.path.join(ls.HERE, "LITERATURE_SCOUT.md") in written
    assert written["json_fh"] == os.path.join(ls.HERE, "LITERATURE_SCOUT.json")


def test_main_gaps_mode_prints_only_gaps(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(ls, "FRONTIER_THEMES", FRONTIER)
    monkeypatch.setattr(ls, "SEED_PAPERS", [{"title": "Quantum finance qubits", "year": 2024},
                                            {"title": "momentum", "year": 2024}])
    _argv(monkeypatch, "--offline", "--gaps", "--out", str(tmp_path / "never.md"))
    ls.main()
    out = capsys.readouterr().out
    assert out.startswith("=== 1 research-gap opportunities ===")
    assert "quantum" in out and "momentum" not in out
    assert not (tmp_path / "never.md").exists()


def test_main_gaps_mode_none(monkeypatch, capsys):
    _argv(monkeypatch, "--offline", "--gaps")
    ls.main()
    assert capsys.readouterr().out.strip() == "=== 0 research-gap opportunities ==="
