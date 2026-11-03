"""Characterisation of serve.create_app: every route's status/body and the SQL it sends, with a fake store."""
from __future__ import annotations

import pandas as pd
import pytest
from _golden import check

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("duckdb")
from fastapi.testclient import TestClient  # noqa: E402

import serve  # noqa: E402
import warehouse  # noqa: E402
import watchlist_store  # noqa: E402


class FakeCon:
    def __init__(self, log):
        self.log = log

    def execute(self, sql):
        self.log.append(sql)
        return self

    def df(self):
        return pd.DataFrame({"market": ["US", "IN"], "x": [1.5, float("nan")]})

    def close(self):
        self.log.append("<close>")


@pytest.fixture()
def client(monkeypatch):
    log: list = []
    monkeypatch.setattr(serve, "_connect", lambda: FakeCon(log))

    def detail(con, ticker, market, bars=60):
        log.append(("detail", ticker, market, bars))
        empty = ticker == "NONE"
        return {"ohlc": [] if empty else [{"Date": "2026-01-02", "Close": 2.0}, {"Date": "2026-01-01", "Close": 1.0}],
                "fundamentals": None, "dvm_technical": None, "dvm_composite": None}

    monkeypatch.setattr(warehouse, "ticker_detail", detail)
    store: dict = {}

    class VC:
        @staticmethod
        def connect():
            return FakeCon(log)

        @staticmethod
        def list_ids(con, kind):
            return sorted(store)

        @staticmethod
        def create(con, kind, name, payload):
            if name in store:
                raise ValueError("exists")
            store[name] = dict(payload)
            return {"created": name, **payload}

        @staticmethod
        def read(con, kind, name):
            return store.get(name)

        @staticmethod
        def add_to_list(con, kind, name, field, items):
            if name not in store:
                raise KeyError(name)
            store[name].setdefault(field, []).extend(items)

        @staticmethod
        def remove_from_list(con, kind, name, field, items):
            store[name][field] = [t for t in store[name][field] if t not in items]

        @staticmethod
        def update(con, kind, name, patch):
            store[name].update(patch)

        @staticmethod
        def delete(con, kind, name):
            return store.pop(name, None) is not None

        @staticmethod
        def history(con, kind, name):
            return [{"v": 1}]

    for fn in ("connect", "list_ids", "create", "read", "add_to_list", "remove_from_list", "update", "delete",
               "history"):
        monkeypatch.setattr(watchlist_store, fn, getattr(VC, fn))
    c = TestClient(serve.create_app())
    c.log = log
    return c


REQUESTS = [
    ("GET", "/health", None),
    ("GET", "/markets", None),
    ("GET", "/ggg?limit=3", None),
    ("GET", "/ggg?market=US&limit=3", None),
    ("GET", "/screen/dvm_dist?limit=2", None),
    ("GET", "/screen/nope", None),
    ("GET", "/filter?predicate=roe>15%20and%20de<1&limit=5", None),
    ("GET", "/filter?predicate=bogus>1", None),
    ("GET", "/filter", None),
    ("GET", "/ticker/aapl?market=us", None),
    ("GET", "/ticker/none?market=us", None),
    ("GET", "/chart/ticker/aapl?market=us&bars=5", None),
    ("GET", "/chart/ticker/none?market=us", None),
    ("GET", "/watchlists", None),
    ("POST", "/watchlists/core", {"tickers": ["aapl", "msft"], "note": "n"}),
    ("POST", "/watchlists/core", {"tickers": []}),
    ("GET", "/watchlists/core", None),
    ("GET", "/watchlists/missing", None),
    ("PUT", "/watchlists/core", {"add": ["tsla"], "remove": ["aapl"], "note": "m"}),
    ("PUT", "/watchlists/missing", {"add": ["x"]}),
    ("GET", "/watchlists/core/history", None),
    ("DELETE", "/watchlists/core", None),
    ("DELETE", "/watchlists/core", None),
]


def test_routes_match_golden(client):
    out = []
    for method, path, body in REQUESTS:
        r = client.request(method, path, json=body) if body is not None else client.request(method, path)
        ctype = r.headers.get("content-type", "")
        out.append([method, path, r.status_code, ctype.split(";")[0], r.text])
    out.append(client.log)
    check("serve_routes", out)
