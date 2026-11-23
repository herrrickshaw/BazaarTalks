"""Offline tests for watchlist_store.py (versioned append-only CRUD over sqlite)."""
import json
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import watchlist_store as ws  # noqa: E402


@pytest.fixture
def con():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ws.init(c)
    yield c
    c.close()


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch):
    ticks = iter(f"2026-01-01T00:00:{i:02d}+00:00" for i in range(60))
    monkeypatch.setattr(ws, "_now", lambda: next(ticks))


def test_now_is_utc_iso_seconds(monkeypatch):
    monkeypatch.undo()                       # use the real clock for this one
    ts = ws._now()
    assert ts.endswith("+00:00") and "." not in ts and len(ts) == 25


def test_create_read_roundtrip_and_version_one(con):
    rec = ws.create(con, "watchlist", "mom", {"tickers": ["NVDA"], "note": "x"})
    assert (rec["version"], rec["op"], rec["deleted"]) == (1, "create", False)
    assert rec["payload"] == {"tickers": ["NVDA"], "note": "x"}
    assert rec["ts"] == "2026-01-01T00:00:00+00:00"
    assert ws.read(con, "watchlist", "mom") == {"tickers": ["NVDA"], "note": "x"}


def test_create_duplicate_live_raises_with_version(con):
    ws.create(con, "w", "a", {})
    with pytest.raises(ValueError, match=r"w/a already exists \(version 1\)"):
        ws.create(con, "w", "a", {"x": 1})


def test_create_does_not_alias_payload(con):
    p = {"tickers": ["A"]}
    ws.create(con, "w", "a", p)
    p["tickers"].append("B")                 # shallow copy only; nested list is shared in-memory
    p["new"] = 1                             # but top-level keys are not
    assert "new" not in ws.read(con, "w", "a")


def test_empty_payload_reads_back_as_empty_dict_not_none(con):
    ws.create(con, "w", "a", {})
    assert ws.read(con, "w", "a") == {}


def test_read_absent_returns_none(con):
    assert ws.read(con, "w", "ghost") is None


def test_types_and_ids_are_isolated(con):
    ws.create(con, "w", "a", {"v": 1})
    ws.create(con, "other", "a", {"v": 2})
    assert ws.read(con, "w", "a") == {"v": 1}
    assert ws.read(con, "other", "a") == {"v": 2}
    assert ws.list_ids(con, "w") == [{"id": "a", "version": 1}]


def test_update_merge_and_replace(con):
    ws.create(con, "w", "a", {"x": 1, "y": 2})
    r = ws.update(con, "w", "a", {"y": 9, "z": 3})
    assert (r["version"], r["op"]) == (2, "update")
    assert r["payload"] == {"x": 1, "y": 9, "z": 3}
    r = ws.update(con, "w", "a", {"only": True}, merge=False)
    assert r["version"] == 3 and r["payload"] == {"only": True}
    assert ws.read(con, "w", "a") == {"only": True}


def test_update_missing_or_deleted_raises(con):
    with pytest.raises(KeyError, match="does not exist"):
        ws.update(con, "w", "a", {"x": 1})
    ws.create(con, "w", "a", {})
    ws.delete(con, "w", "a")
    with pytest.raises(KeyError, match="does not exist"):
        ws.update(con, "w", "a", {"x": 1})


def test_delete_is_soft_and_returns_flags(con):
    ws.create(con, "w", "a", {"x": 1})
    assert ws.delete(con, "w", "a") is True
    assert ws.read(con, "w", "a") is None
    assert ws.delete(con, "w", "a") is False         # already tombstoned
    assert ws.delete(con, "w", "ghost") is False
    hist = ws.history(con, "w", "a")
    assert [h["op"] for h in hist] == ["create", "delete"]
    assert hist[1]["deleted"] is True and hist[1]["payload"] == {"x": 1}   # tombstone keeps last payload


def test_recreate_after_delete_bumps_version(con):
    ws.create(con, "w", "a", {"x": 1})
    ws.delete(con, "w", "a")
    r = ws.create(con, "w", "a", {"x": 2})
    assert r["version"] == 3
    assert ws.read(con, "w", "a") == {"x": 2}
    assert [h["version"] for h in ws.history(con, "w", "a")] == [1, 2, 3]


def test_restore_brings_back_last_live_payload(con):
    ws.create(con, "w", "a", {"x": 1})
    ws.update(con, "w", "a", {"x": 2})
    ws.delete(con, "w", "a")
    r = ws.restore(con, "w", "a")
    assert (r["version"], r["op"], r["deleted"]) == (4, "restore", False)
    assert r["payload"] == {"x": 2}
    assert ws.read(con, "w", "a") == {"x": 2}


def test_restore_requires_a_deleted_record(con):
    with pytest.raises(KeyError, match="not deleted"):
        ws.restore(con, "w", "ghost")
    ws.create(con, "w", "a", {})
    with pytest.raises(KeyError, match="w/a is not deleted"):
        ws.restore(con, "w", "a")


def test_history_is_oldest_first_and_full_audit_trail(con):
    ws.create(con, "w", "a", {"n": 1})
    ws.update(con, "w", "a", {"n": 2})
    ws.delete(con, "w", "a")
    ws.restore(con, "w", "a")
    hist = ws.history(con, "w", "a")
    assert [(h["version"], h["op"], h["deleted"]) for h in hist] == [
        (1, "create", False), (2, "update", False), (3, "delete", True), (4, "restore", False)]
    assert ws.history(con, "w", "ghost") == []


def test_read_version_time_travel(con):
    ws.create(con, "w", "a", {"n": 1})
    ws.update(con, "w", "a", {"n": 2})
    assert ws.read_version(con, "w", "a", 1) == {"n": 1}
    assert ws.read_version(con, "w", "a", 2) == {"n": 2}
    assert ws.read_version(con, "w", "a", 3) is None
    assert ws.read_version(con, "w", "ghost", 1) is None


def test_list_ids_only_live_sorted_with_current_version(con):
    ws.create(con, "w", "b", {})
    ws.create(con, "w", "a", {})
    ws.update(con, "w", "a", {"k": 1})
    ws.create(con, "w", "c", {})
    ws.delete(con, "w", "c")
    assert ws.list_ids(con, "w") == [{"id": "a", "version": 2}, {"id": "b", "version": 1}]
    assert ws.list_ids(con, "nothing") == []


def test_add_to_list_appends_new_values_in_order_skipping_existing(con):
    ws.create(con, "w", "a", {"tickers": ["A", "B"]})
    r = ws.add_to_list(con, "w", "a", "tickers", ["B", "C", "D"])
    assert r["payload"]["tickers"] == ["A", "B", "C", "D"]
    assert r["version"] == 2


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="watchlist_store.py:183 add_to_list tests membership against the OLD list, "
                          "so duplicates inside `values` are all appended (not a set union)")
def test_add_to_list_dedupes_within_added_values(con):
    ws.create(con, "w", "a", {"tickers": ["A"]})
    r = ws.add_to_list(con, "w", "a", "tickers", ["C", "C"])
    assert r["payload"]["tickers"] == ["A", "C"]


def test_add_to_list_creates_field_when_absent(con):
    ws.create(con, "w", "a", {"note": "n"})
    r = ws.add_to_list(con, "w", "a", "tickers", ["X"])
    assert r["payload"] == {"note": "n", "tickers": ["X"]}


def test_add_to_list_on_missing_record_raises(con):
    with pytest.raises(KeyError):
        ws.add_to_list(con, "w", "ghost", "tickers", ["X"])


def test_remove_from_list(con):
    ws.create(con, "w", "a", {"tickers": ["A", "B", "C", "A"], "note": "n"})
    r = ws.remove_from_list(con, "w", "a", "tickers", ["A", "Z"])
    assert r["payload"] == {"tickers": ["B", "C"], "note": "n"}
    r = ws.remove_from_list(con, "w", "a", "absent_field", ["A"])
    assert r["payload"]["absent_field"] == []


def test_connect_creates_schema_in_file_db(tmp_path):
    path = tmp_path / "s.db"
    c = ws.connect(str(path))
    try:
        assert c.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        ws.create(c, "w", "a", {"k": 1})
    finally:
        c.close()
    c2 = ws.connect(str(path))                 # re-open: init is idempotent, data persisted
    try:
        assert ws.read(c2, "w", "a") == {"k": 1}
    finally:
        c2.close()


def test_parse_kv_rules():
    assert ws._parse_kv(None) == {}
    assert ws._parse_kv(["tickers=nvda, msft ,"]) == {"tickers": ["NVDA", "MSFT"]}
    assert ws._parse_kv(["tickers=aapl"]) == {"tickers": ["AAPL"]}       # 'tickers' always list
    assert ws._parse_kv(["tickers="]) == {"tickers": []}
    assert ws._parse_kv(["note=hello"]) == {"note": "hello"}             # scalar kept as-is
    assert ws._parse_kv(["tags=a,b"]) == {"tags": ["A", "B"]}            # comma -> upper-cased list
    assert ws._parse_kv(["flag"]) == {"flag": ""}
    assert ws._parse_kv(["k=a=b"]) == {"k": "a=b"}                       # partition on first '='


# ── CLI ───────────────────────────────────────────────────────────────────────
def _cli(monkeypatch, capsys, db, *argv):
    monkeypatch.setattr(sys, "argv", ["watchlist_store.py", *argv, "--db", str(db)])
    try:
        ws.main()
        code = 0
    except SystemExit as e:
        code = e.code
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def test_cli_full_lifecycle(tmp_path, monkeypatch, capsys):
    db = tmp_path / "cli.db"
    code, out, _ = _cli(monkeypatch, capsys, db, "create", "watchlist", "mom",
                        "--set", "tickers=nvda,msft,aapl")
    assert code == 0
    assert out.strip() == "created watchlist/mom v1: {'tickers': ['NVDA', 'MSFT', 'AAPL']}"

    code, out, _ = _cli(monkeypatch, capsys, db, "update", "watchlist", "mom",
                        "--add", "tickers=tsla", "--remove", "tickers=aapl")
    assert code == 0
    assert out.strip() == "updated watchlist/mom: {'tickers': ['NVDA', 'MSFT', 'TSLA']}"

    code, out, _ = _cli(monkeypatch, capsys, db, "read", "watchlist", "mom")
    assert json.loads(out) == {"tickers": ["NVDA", "MSFT", "TSLA"]}

    code, out, _ = _cli(monkeypatch, capsys, db, "list", "watchlist")
    assert out.split() == ["mom", "v3"]

    code, out, _ = _cli(monkeypatch, capsys, db, "version", "watchlist", "mom", "--v", "1")
    assert json.loads(out) == {"tickers": ["NVDA", "MSFT", "AAPL"]}

    code, out, _ = _cli(monkeypatch, capsys, db, "delete", "watchlist", "mom")
    assert out.strip() == "deleted (soft)"
    code, out, _ = _cli(monkeypatch, capsys, db, "delete", "watchlist", "mom")
    assert out.strip() == "nothing live to delete"
    code, out, _ = _cli(monkeypatch, capsys, db, "read", "watchlist", "mom")
    assert out.strip() == "(absent or deleted)"

    code, out, _ = _cli(monkeypatch, capsys, db, "restore", "watchlist", "mom")
    assert out.strip().startswith("restored v5:")

    code, out, _ = _cli(monkeypatch, capsys, db, "history", "watchlist", "mom")
    lines = out.strip().splitlines()
    assert [l.split()[:2] for l in lines] == [["v1", "create"], ["v2", "update"], ["v3", "update"],
                                              ["v4", "delete"], ["v5", "restore"]]
    assert "[deleted]" in lines[3] and "[deleted]" not in lines[4]


def test_cli_update_set_scalar_and_list(tmp_path, monkeypatch, capsys):
    db = tmp_path / "cli.db"
    _cli(monkeypatch, capsys, db, "create", "watchlist", "w", "--set", "note=hi")
    code, out, _ = _cli(monkeypatch, capsys, db, "update", "watchlist", "w",
                        "--set", "note=bye", "tickers=a,b")
    assert code == 0
    assert out.strip() == "updated watchlist/w: {'note': 'bye', 'tickers': ['A', 'B']}"
    code, out, _ = _cli(monkeypatch, capsys, db, "history", "watchlist", "w")
    assert len(out.strip().splitlines()) == 3        # create + one update per --set field


def test_cli_error_exit_code_and_message(tmp_path, monkeypatch, capsys):
    db = tmp_path / "cli.db"
    _cli(monkeypatch, capsys, db, "create", "watchlist", "w")
    code, out, err = _cli(monkeypatch, capsys, db, "create", "watchlist", "w")
    assert code == 1 and out == ""
    assert err.strip() == "error: watchlist/w already exists (version 1)"
    code, _, err = _cli(monkeypatch, capsys, db, "update", "watchlist", "ghost", "--set", "k=v")
    assert code == 1 and "does not exist" in err
    code, _, err = _cli(monkeypatch, capsys, db, "restore", "watchlist", "w")
    assert code == 1 and "is not deleted" in err
