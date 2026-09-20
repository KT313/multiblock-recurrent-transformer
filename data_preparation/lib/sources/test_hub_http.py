# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Offline HTTP client lifecycle tests; only httpx mock transports execute requests."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from huggingface_hub import get_session
from huggingface_hub.utils import _http

from data_preparation.lib.sources import hub_files


@pytest.fixture(autouse=True)
def isolated_hub_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # Save the actual shared client without closing it, then restore it after closing this test's clients.
    monkeypatch.setattr(_http, "_GLOBAL_CLIENT", None)
    monkeypatch.setattr(_http, "_GLOBAL_CLIENT_FACTORY", _http.default_client_factory)
    monkeypatch.setattr(hub_files, "_HUB_HTTP_CONFIGURED", False)
    yield
    _http.close_session()


def test_factory_keeps_default_hub_hooks_and_redirects() -> None:
    hub_files.configure_hub_http()
    client = get_session()
    assert client.timeout == httpx.Timeout(hub_files.HUB_REQUEST_TIMEOUT)
    assert client.follow_redirects
    assert _http.hf_request_event_hook in client.event_hooks["request"]


def test_retry_recreation_keeps_timeouts_transport_auth_and_explicit_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    clients: list[httpx.Client] = []
    requests: list[httpx.Request] = []
    hooks: list[httpx.Request] = []
    failures = 2

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal failures
        requests.append(request)
        if request.url.path == "/retry" and failures:
            failures -= 1
            raise httpx.ConnectError("simulated connection failure", request=request)
        if "/paths-info/" in request.url.path:
            return httpx.Response(200, json=[{"type": "file", "path": "data/a.parquet", "size": 12, "oid": "abc"}])
        return httpx.Response(200)

    def factory() -> httpx.Client:
        client = httpx.Client(
            transport=httpx.MockTransport(handle), auth=httpx.BasicAuth("fixture", "fixture"),
            event_hooks={"request": [hooks.append]}, follow_redirects=True, timeout=None,
        )
        clients.append(client)
        return client

    monkeypatch.setattr(_http, "default_client_factory", factory)
    hub_files.configure_hub_http()
    assert _http.http_backoff("GET", "https://fixture.invalid/retry", max_retries=2, base_wait_time=0).status_code == 200
    assert len(clients) == 3 and all(client.is_closed for client in clients[:2])
    assert hub_files.paths_info("fixture/repo", ["data/a.parquet"], "commit-a", None) == {"data/a.parquet": 12}
    assert all(client.timeout == httpx.Timeout(hub_files.HUB_REQUEST_TIMEOUT) for client in clients)
    assert all(request.extensions["timeout"] == dict.fromkeys(("connect", "read", "write", "pool"), 30.0) for request in requests)
    assert all("authorization" in request.headers for request in requests)
    assert hooks == requests
    for explicit in (7.0, None):
        get_session().get("https://fixture.invalid/explicit", timeout=explicit)
        assert requests[-1].extensions["timeout"] == dict.fromkeys(("connect", "read", "write", "pool"), explicit)


def test_concurrent_setup_and_lookups_install_factory_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    registrations = 0
    original_set_factory = huggingface_hub.set_client_factory
    start = threading.Barrier(8)
    timeout_records: list[object] = []
    clients: list[httpx.Client] = []

    def handle(request: httpx.Request) -> httpx.Response:
        timeout_records.append(request.extensions["timeout"])
        return httpx.Response(200, json=[{"type": "file", "path": "f", "size": 1, "oid": "abc"}])

    def factory() -> httpx.Client:
        client = httpx.Client(transport=httpx.MockTransport(handle), timeout=None)
        clients.append(client)
        return client

    def register(factory: _http.CLIENT_FACTORY_T) -> None:
        nonlocal registrations
        registrations += 1
        original_set_factory(factory)

    def lookup(_: int) -> dict[str, int]:
        start.wait(timeout=10)
        return hub_files.paths_info("fixture/repo", ["f"], "commit-a", None)

    monkeypatch.setattr(_http, "default_client_factory", factory)
    monkeypatch.setattr(huggingface_hub, "set_client_factory", register)
    try:
        for _ in range(3):
            _http.close_session()
            with ThreadPoolExecutor(max_workers=8) as pool:
                assert list(pool.map(lookup, range(8))) == [{"f": 1}] * 8
        assert registrations == 1
        assert len(timeout_records) == 24
        assert all(record == dict.fromkeys(("connect", "read", "write", "pool"), 30.0) for record in timeout_records)
    finally:
        # The installed Hub get_session can create more than one client during simultaneous first calls.
        for client in clients:
            client.close()


def test_repeated_connection_failure_keeps_existing_retry_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[httpx.Request] = []

    def fail(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise httpx.ConnectError("simulated persistent failure", request=request)

    monkeypatch.setattr(_http, "default_client_factory", lambda: httpx.Client(transport=httpx.MockTransport(fail)))
    hub_files.configure_hub_http()
    with pytest.raises(httpx.ConnectError, match="simulated persistent failure"):
        _http.http_backoff("GET", "https://fixture.invalid/retry", max_retries=2, base_wait_time=0)
    assert len(requests) == 3
