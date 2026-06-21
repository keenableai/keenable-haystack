"""Offline unit tests for the Keenable Haystack components.

The HTTP transport is faked at the ``requests`` boundary, so these exercise
component wiring, serialization, endpoint selection, attribution headers, the
SSRF guard, HTTPS enforcement, error mapping, and Document/links construction
without a network.
"""

import pytest
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.components.fetchers.keenable import KeenableFetcher
from haystack_integrations.components.websearch.keenable import KeenableWebSearch
from haystack_integrations.components.websearch.keenable import _client
from haystack_integrations.components.websearch.keenable._client import (
    KeenableError,
    keenable_post,
    normalize_key,
    reject_private_fetch_target,
    resolve_base_url,
)


class _FakeResponse:
    def __init__(self, status_code=200, json_body=None, text="", raise_on_json=False):
        self.status_code = status_code
        self._json = json_body
        self.text = text
        self._raise_on_json = raise_on_json

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        if self._raise_on_json:
            raise ValueError("no json")
        return self._json


class _Recorder:
    """Records the last request issued through the faked requests module."""

    last = {}

    def __init__(self, response):
        self._response = response

    def post(self, url, json=None, headers=None, timeout=None):
        _Recorder.last = {"method": "POST", "url": url, "json": json, "headers": headers}
        return self._response

    def get(self, url, params=None, headers=None, timeout=None):
        _Recorder.last = {"method": "GET", "url": url, "params": params, "headers": headers}
        return self._response


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    monkeypatch.delenv("KEENABLE_API_URL", raising=False)


def _patch(monkeypatch, response):
    rec = _Recorder(response)
    monkeypatch.setattr(_client.requests, "post", rec.post)
    monkeypatch.setattr(_client.requests, "get", rec.get)


# --------------------------------------------------------------------------- #
# Component wiring + output sockets
# --------------------------------------------------------------------------- #


def test_websearch_output_sockets():
    sockets = KeenableWebSearch().__haystack_output__._sockets_dict
    assert set(sockets) == {"documents", "links"}


def test_fetcher_output_sockets():
    sockets = KeenableFetcher().__haystack_output__._sockets_dict
    assert set(sockets) == {"documents"}


def test_websearch_keyless_by_default():
    # No key in env, strict=False Secret -> resolves to None, no raise at init.
    ws = KeenableWebSearch()
    assert ws.api_key.resolve_value() is None


# --------------------------------------------------------------------------- #
# Serialization (Secret round-trips by env var name, not value)
# --------------------------------------------------------------------------- #


def test_websearch_to_from_dict(monkeypatch):
    monkeypatch.setenv("KEENABLE_API_KEY", "secret")
    ws = KeenableWebSearch(top_k=5, mode="realtime", site="github.com", timeout=12.0)
    data = ws.to_dict()
    init = data["init_parameters"]
    assert init["top_k"] == 5
    assert init["mode"] == "realtime"
    assert init["site"] == "github.com"
    assert init["timeout"] == 12.0
    # The raw key value must never be serialized; only the env var name is.
    assert "secret" not in str(data)
    assert init["api_key"]["env_vars"] == ["KEENABLE_API_KEY"]

    restored = KeenableWebSearch.from_dict(data)
    assert restored.top_k == 5
    assert restored.mode == "realtime"
    assert restored.api_key.resolve_value() == "secret"


def test_fetcher_to_from_dict():
    f = KeenableFetcher(raise_on_failure=True, timeout=15.0)
    data = f.to_dict()
    restored = KeenableFetcher.from_dict(data)
    assert restored.raise_on_failure is True
    assert restored.timeout == 15.0


# --------------------------------------------------------------------------- #
# resolve_base_url + SSRF + key normalization
# --------------------------------------------------------------------------- #


def test_base_url_default_https():
    assert resolve_base_url() == "https://api.keenable.ai"


def test_base_url_http_public_rejected(monkeypatch):
    monkeypatch.setenv("KEENABLE_API_URL", "http://api.keenable.ai")
    with pytest.raises(KeenableError):
        resolve_base_url()


def test_base_url_http_loopback_ok(monkeypatch):
    monkeypatch.setenv("KEENABLE_API_URL", "http://localhost:8000")
    assert resolve_base_url() == "http://localhost:8000"


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/x",
        "http://127.0.0.1/x",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/x",
        "http://metadata.google.internal/x",
        "https:///nohost",
    ],
)
def test_reject_private_fetch_target(url):
    with pytest.raises(KeenableError):
        reject_private_fetch_target(url)


def test_public_fetch_target_allowed():
    reject_private_fetch_target("https://example.com/article")
    reject_private_fetch_target("https://8.8.8.8/x")  # public numeric IP is fine


@pytest.mark.parametrize(
    "url",
    [
        "http://2130706433/secret",  # decimal form of 127.0.0.1
        "http://0x7f000001/secret",  # hex form of 127.0.0.1
        "http://0177.0.0.1/secret",  # octal-dotted form of 127.0.0.1
        "http://127.0.0.1./secret",  # trailing dot on IP literal
        "http://localhost./secret",  # trailing dot on hostname
        "http://LOCALHOST/secret",  # case
    ],
)
def test_reject_ssrf_bypass_encodings(url):
    with pytest.raises(KeenableError):
        reject_private_fetch_target(url)


def test_ipv6_public_address_allowed():
    # A globally routable IPv6 address must pass the guard.
    reject_private_fetch_target("https://[2606:4700:4700::1111]/x")


def test_normalize_key():
    assert normalize_key("  ") is None
    assert normalize_key("") is None
    assert normalize_key(None) is None
    assert normalize_key(" explicit ") == "explicit"


# --------------------------------------------------------------------------- #
# Transport: endpoint selection, attribution, errors
# --------------------------------------------------------------------------- #


def test_keyless_uses_public_path_and_attribution(monkeypatch):
    _patch(monkeypatch, _FakeResponse(json_body={"results": []}))
    keenable_post("/v1/search/public", "/v1/search", {"query": "x"}, None, 30.0)
    sent = _Recorder.last
    assert sent["url"].endswith("/v1/search/public")
    assert sent["headers"]["X-Keenable-Title"] == "Haystack"
    assert "X-API-Key" not in sent["headers"]
    assert sent["headers"]["User-Agent"].startswith("keenable-haystack/")


def test_keyed_uses_authenticated_path(monkeypatch):
    _patch(monkeypatch, _FakeResponse(json_body={"results": []}))
    keenable_post("/v1/search/public", "/v1/search", {"query": "x"}, "secret", 30.0)
    sent = _Recorder.last
    assert sent["url"].endswith("/v1/search")
    assert sent["headers"]["X-API-Key"] == "secret"


@pytest.mark.parametrize(
    ("status", "needle"),
    [(401, "authentication"), (402, "credits"), (429, "rate limit"), (500, "500")],
)
def test_error_status_mapping(monkeypatch, status, needle):
    _patch(monkeypatch, _FakeResponse(status_code=status, json_body={"message": "boom"}))
    with pytest.raises(KeenableError) as exc:
        keenable_post("/v1/search/public", "/v1/search", {"query": "x"}, None, 30.0)
    assert needle in str(exc.value).lower()


def test_non_json_raises(monkeypatch):
    _patch(monkeypatch, _FakeResponse(text="<html>", raise_on_json=True))
    with pytest.raises(KeenableError):
        keenable_post("/v1/search/public", "/v1/search", {"query": "x"}, None, 30.0)


# --------------------------------------------------------------------------- #
# KeenableWebSearch.run -> documents + links
# --------------------------------------------------------------------------- #


def test_search_returns_documents_and_links(monkeypatch):
    body = {
        "results": [
            {"title": "T1", "url": "https://e1.com", "description": "d1"},
            {"title": "T2", "url": "https://e2.com", "description": "d2"},
        ]
    }
    _patch(monkeypatch, _FakeResponse(json_body=body))
    out = KeenableWebSearch().run(query="typescript", site="github.com", mode="pro")
    docs = out["documents"]
    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)
    assert docs[0].content == "d1"
    assert docs[0].meta["url"] == "https://e1.com"
    assert out["links"] == ["https://e1.com", "https://e2.com"]
    sent = _Recorder.last["json"]
    assert sent["query"] == "typescript"
    assert sent["site"] == "github.com"
    assert sent["mode"] == "pro"


def test_search_content_falls_back_to_title(monkeypatch):
    _patch(monkeypatch, _FakeResponse(json_body={"results": [{"title": "OnlyTitle", "url": "https://e.com"}]}))
    out = KeenableWebSearch().run(query="q")
    assert out["documents"][0].content == "OnlyTitle"


def test_search_top_k_limits_client_side(monkeypatch):
    body = {"results": [{"url": f"https://e{i}.com", "description": f"d{i}"} for i in range(5)]}
    _patch(monkeypatch, _FakeResponse(json_body=body))
    out = KeenableWebSearch(top_k=2).run(query="q")
    assert len(out["documents"]) == 2
    assert len(out["links"]) == 2


def test_search_default_mode_and_site(monkeypatch):
    _patch(monkeypatch, _FakeResponse(json_body={"results": []}))
    KeenableWebSearch(mode="realtime", site="example.com").run(query="q")
    sent = _Recorder.last["json"]
    assert sent["mode"] == "realtime"
    assert sent["site"] == "example.com"


def test_search_no_max_results_in_payload(monkeypatch):
    _patch(monkeypatch, _FakeResponse(json_body={"results": []}))
    KeenableWebSearch(top_k=3).run(query="q")
    assert "max_results" not in _Recorder.last["json"]


def test_search_bad_payload_raises(monkeypatch):
    _patch(monkeypatch, _FakeResponse(json_body={"unexpected": True}))
    with pytest.raises(KeenableError):
        KeenableWebSearch().run(query="q")


def test_search_keyed_endpoint(monkeypatch):
    _patch(monkeypatch, _FakeResponse(json_body={"results": []}))
    KeenableWebSearch(api_key=Secret.from_token("secret")).run(query="q")
    assert _Recorder.last["url"].endswith("/v1/search")
    assert _Recorder.last["headers"]["X-API-Key"] == "secret"


# --------------------------------------------------------------------------- #
# KeenableFetcher.run -> documents
# --------------------------------------------------------------------------- #


def test_fetch_returns_documents(monkeypatch):
    _patch(monkeypatch, _FakeResponse(json_body={"url": "https://e.com", "title": "T", "content": "body"}))
    out = KeenableFetcher().run(urls=["https://e.com"])
    docs = out["documents"]
    assert len(docs) == 1
    assert docs[0].content == "body"
    assert docs[0].meta["title"] == "T"
    assert _Recorder.last["url"].endswith("/v1/fetch/public")


def test_fetch_keyed_path(monkeypatch):
    monkeypatch.setenv("KEENABLE_API_KEY", "secret")
    _patch(monkeypatch, _FakeResponse(json_body={"content": "x"}))
    KeenableFetcher().run(urls=["https://e.com"])
    assert _Recorder.last["url"].endswith("/v1/fetch")
    assert _Recorder.last["headers"]["X-API-Key"] == "secret"


def test_fetch_skips_bad_urls_by_default(monkeypatch):
    _patch(monkeypatch, _FakeResponse(json_body={"content": "ok"}))
    out = KeenableFetcher().run(urls=["ftp://e.com/x", "http://127.0.0.1/x", "https://good.com"])
    assert len(out["documents"]) == 1
    assert out["documents"][0].content == "ok"


@pytest.mark.parametrize("bad_url", ["ftp://e.com/x", "http://127.0.0.1/x", "not-a-url"])
def test_fetch_raises_on_failure_when_configured(bad_url):
    with pytest.raises(KeenableError):
        KeenableFetcher(raise_on_failure=True).run(urls=[bad_url])


def test_fetch_skips_non_keenable_errors_when_tolerant(monkeypatch):
    # raise_on_failure=False must keep the batch alive even if a non-KeenableError
    # escapes the transport (contract: one bad URL never fails the whole batch).
    def boom(*_a, **_k):
        raise RuntimeError("unexpected")

    f = KeenableFetcher()
    monkeypatch.setattr(f, "_fetch_one", boom)
    assert f.run(urls=["https://good.com"]) == {"documents": []}


@pytest.mark.parametrize("bad_timeout", [0, -1.0])
def test_non_positive_timeout_rejected(bad_timeout):
    with pytest.raises(ValueError):
        KeenableWebSearch(timeout=bad_timeout)
    with pytest.raises(ValueError):
        KeenableFetcher(timeout=bad_timeout)
