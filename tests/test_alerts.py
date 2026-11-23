"""Offline tests for alerts.py (snapshot diff + SMTP email) using tmp sqlite and a fake smtplib."""
import json
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alerts  # noqa: E402


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Redirect STATE and HERE into tmp_path; clear SMTP env."""
    monkeypatch.setattr(alerts, "STATE", str(tmp_path / "state"))
    monkeypatch.setattr(alerts, "HERE", str(tmp_path))
    for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASS", "SMTP_PORT"):
        monkeypatch.delenv(k, raising=False)
    return tmp_path


def _make_db(path, rows):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE dvm_composite (market TEXT, ticker TEXT, code TEXT)")
    con.executemany("INSERT INTO dvm_composite VALUES (?,?,?)", rows)
    con.commit()
    con.close()


class FakeSMTP:
    instances = []

    def __init__(self, host, port):
        self.host, self.port = host, port
        self.calls = []
        self.exited = False
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.exited = True

    def starttls(self):
        self.calls.append("starttls")

    def login(self, u, p):
        self.calls.append(("login", u, p))

    def sendmail(self, frm, to, msg):
        self.calls.append(("sendmail", frm, to, msg))


@pytest.fixture
def smtp(monkeypatch):
    FakeSMTP.instances = []
    monkeypatch.setattr(alerts.smtplib, "SMTP", FakeSMTP)
    return FakeSMTP


# ── diff_sets / format_alert ────────────────────────────────────────────────
def test_diff_sets_sorted_and_counts():
    d = alerts.diff_sets({"b", "a", "c"}, {"c", "d", "e"})
    assert d == {"new": ["d", "e"], "dropped": ["a", "b"], "retained": ["c"],
                 "n_new": 2, "n_dropped": 2, "n_total": 3}


def test_diff_sets_accepts_lists_and_empty():
    assert alerts.diff_sets([], []) == {"new": [], "dropped": [], "retained": [],
                                        "n_new": 0, "n_dropped": 0, "n_total": 0}
    d = alerts.diff_sets(["x", "x"], ["x", "y", "y"])
    assert d["new"] == ["y"] and d["n_total"] == 2 and d["dropped"] == []


def test_format_alert_header_only_when_no_changes():
    d = alerts.diff_sets({"a"}, {"a"})
    assert alerts.format_alert(d, "lbl", "2024-01-02") == "[2024-01-02] lbl: 1 names (+0 new, -0 dropped)"


def test_format_alert_new_and_dropped_lines():
    d = alerts.diff_sets({"a"}, {"b"})
    out = alerts.format_alert(d, "lbl", "D")
    assert out.split("\n") == ["[D] lbl: 1 names (+1 new, -1 dropped)",
                               "  NEW:     b", "  DROPPED: a"]


def test_format_alert_truncates_lists_at_40_but_counts_all():
    curr = {f"n{i:03d}" for i in range(45)}
    out = alerts.format_alert(alerts.diff_sets(set(), curr), "l", "D")
    assert "(+45 new, -0 dropped)" in out
    listed = out.split("NEW:     ")[1].split(", ")
    assert len(listed) == 40 and listed[-1] == "n039"
    assert "DROPPED" not in out


# ── snapshot storage ────────────────────────────────────────────────────────
def test_save_snapshot_creates_dir_and_sorted_json(env):
    path = alerts.save_snapshot("lbl", {"b", "a"}, "2024-01-02")
    assert path == str(env / "state" / "lbl_2024-01-02.json")
    assert json.load(open(path)) == ["a", "b"]


def test_latest_prior_none_without_files(env):
    assert alerts.latest_prior("lbl", "2024-01-05") is None


def test_latest_prior_strictly_before_and_most_recent(env):
    alerts.save_snapshot("lbl", {"old"}, "2024-01-01")
    alerts.save_snapshot("lbl", {"mid"}, "2024-01-03")
    alerts.save_snapshot("lbl", {"same"}, "2024-01-05")
    alerts.save_snapshot("lbl", {"future"}, "2024-01-09")
    assert alerts.latest_prior("lbl", "2024-01-05") == {"mid"}      # same date excluded (strict <)
    assert alerts.latest_prior("lbl", "2024-01-06") == {"same"}
    assert alerts.latest_prior("lbl", "2024-01-01") is None          # nothing before the first


def test_latest_prior_ignores_other_labels(env):
    alerts.save_snapshot("other", {"o"}, "2024-01-01")
    assert alerts.latest_prior("lbl", "2024-02-01") is None


# ── ggg_members ─────────────────────────────────────────────────────────────
def test_ggg_members_filters_code_and_joins_market_ticker(tmp_path):
    db = tmp_path / "d.db"
    _make_db(db, [("US", "AAPL", "GGG"), ("US", "MSFT", "GGB"), ("IN", "TCS", "GGG"),
                  ("US", "AAPL", "GGG")])
    assert alerts.ggg_members(str(db)) == {"US:AAPL", "IN:TCS"}


def test_ggg_members_empty(tmp_path):
    db = tmp_path / "d.db"
    _make_db(db, [("US", "X", "BBG")])
    assert alerts.ggg_members(str(db)) == set()


# ── send_email ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("missing", ["SMTP_HOST", "SMTP_USER", "SMTP_PASS"])
def test_send_email_skips_when_any_credential_missing(env, smtp, monkeypatch, capsys, missing):
    for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASS"):
        monkeypatch.setenv(k, "v")
    monkeypatch.delenv(missing)
    assert alerts.send_email("a@b.c", "s", "b") is False
    assert "not set — skipping send" in capsys.readouterr().err
    assert smtp.instances == []


def test_send_email_empty_password_counts_as_unset(env, smtp, monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "h"); monkeypatch.setenv("SMTP_USER", "u")
    monkeypatch.setenv("SMTP_PASS", "")
    assert alerts.send_email("a@b.c", "s", "b") is False


def test_send_email_default_port_and_message(env, smtp, monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "mail.example"); monkeypatch.setenv("SMTP_USER", "me@x.io")
    monkeypatch.setenv("SMTP_PASS", "pw")
    assert alerts.send_email("to@y.io", "Subj", "Body text") is True
    s = smtp.instances[0]
    assert (s.host, s.port) == ("mail.example", 587)
    assert s.calls[0] == "starttls" and s.calls[1] == ("login", "me@x.io", "pw")
    _, frm, to, raw = s.calls[2]
    assert frm == "me@x.io" and to == ["to@y.io"]
    assert "Subject: Subj" in raw and "From: me@x.io" in raw and "To: to@y.io" in raw
    assert s.exited


def test_send_email_custom_port(env, smtp, monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "h"); monkeypatch.setenv("SMTP_USER", "u")
    monkeypatch.setenv("SMTP_PASS", "p"); monkeypatch.setenv("SMTP_PORT", "2525")
    alerts.send_email("t@t", "s", "b")
    assert smtp.instances[0].port == 2525


# ── main ────────────────────────────────────────────────────────────────────
def _main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["alerts.py", *argv])
    alerts.main()


def test_main_no_args_prints_help(env, monkeypatch, capsys):
    _main(monkeypatch)
    assert "--snapshot" in capsys.readouterr().out


def test_main_history_lists_snapshots(env, monkeypatch, capsys):
    alerts.save_snapshot("a", {"1", "2"}, "2024-01-01")
    alerts.save_snapshot("b", set(), "2024-01-02")
    _main(monkeypatch, "--history")
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["a_2024-01-01.json", "2", "names"]
    assert lines[1].split() == ["b_2024-01-02.json", "0", "names"]


def test_main_history_empty_prints_nothing(env, monkeypatch, capsys):
    _main(monkeypatch, "--history")
    assert capsys.readouterr().out == ""


def test_main_snapshot_missing_db_exits(env, monkeypatch):
    with pytest.raises(SystemExit) as ei:
        _main(monkeypatch, "--snapshot", "--db", str(env / "nope.db"))
    assert "run dvm_composite.py first" in str(ei.value)


def test_main_first_snapshot_baseline_only(env, monkeypatch, capsys):
    db = env / "d.db"; _make_db(db, [("US", "A", "GGG")])
    _main(monkeypatch, "--snapshot", "--db", str(db), "--date", "2024-01-02")
    cap = capsys.readouterr()
    assert "baseline stored" in cap.out
    assert "snapshot: 1 names ->" in cap.err
    assert not (env / "alert_ggg_global_2024-01-02.json").exists()
    assert json.load(open(env / "state" / "ggg_global_2024-01-02.json")) == ["US:A"]


def test_main_second_snapshot_diffs_writes_json_no_email(env, monkeypatch, capsys):
    db = env / "d.db"; _make_db(db, [("US", "A", "GGG"), ("US", "B", "GGG")])
    alerts.save_snapshot("ggg_global", {"US:A", "US:C"}, "2024-01-01")
    _main(monkeypatch, "--snapshot", "--db", str(db), "--date", "2024-01-02")
    cap = capsys.readouterr()
    assert "[2024-01-02] ggg_global: 2 names (+1 new, -1 dropped)" in cap.out
    assert "NEW:     US:B" in cap.out and "DROPPED: US:C" in cap.out
    written = json.load(open(env / "alert_ggg_global_2024-01-02.json"))
    assert written["new"] == ["US:B"] and written["dropped"] == ["US:C"]
    assert written["retained"] == ["US:A"]
    assert "email" not in cap.err


def test_main_rerun_same_date_diffs_against_previous_day_not_itself(env, monkeypatch, capsys):
    db = env / "d.db"; _make_db(db, [("US", "A", "GGG")])
    alerts.save_snapshot("ggg_global", {"US:A"}, "2024-01-01")
    for _ in range(2):
        _main(monkeypatch, "--snapshot", "--db", str(db), "--date", "2024-01-02")
    assert "(+0 new, -0 dropped)" in capsys.readouterr().out


def test_main_email_sent(env, smtp, monkeypatch, capsys):
    db = env / "d.db"; _make_db(db, [("US", "A", "GGG")])
    alerts.save_snapshot("lbl", set(), "2024-01-01")
    for k, v in (("SMTP_HOST", "h"), ("SMTP_USER", "u"), ("SMTP_PASS", "p")):
        monkeypatch.setenv(k, v)
    _main(monkeypatch, "--snapshot", "--db", str(db), "--date", "2024-01-02",
          "--label", "lbl", "--email", "me@x.io")
    assert "email sent" in capsys.readouterr().err
    raw = smtp.instances[0].calls[-1][3]
    assert "[lbl] 1 new / 0 dropped (2024-01-02)" in raw


def test_main_email_skipped_without_credentials(env, monkeypatch, capsys):
    db = env / "d.db"; _make_db(db, [("US", "A", "GGG")])
    alerts.save_snapshot("ggg_global", set(), "2024-01-01")
    _main(monkeypatch, "--snapshot", "--db", str(db), "--date", "2024-01-02", "--email", "me@x.io")
    assert "email skipped" in capsys.readouterr().err
