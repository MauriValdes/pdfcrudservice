"""RED tests for the independent worker HTTP client for MS2.

The client and its exceptions are intentionally not implemented yet. These
tests describe the approved worker-to-MS2 contract using only MockTransport.
"""

import asyncio
from dataclasses import FrozenInstanceError
import sys

import httpx
import pytest

from src.worker.ms2_client import (
    ExtractionResult,
    MS2Client,
    MS2HTTPError,
    MS2InvalidResponseError,
    MS2TransportError,
)


BASE_URL = "http://ms2.internal"
PDF_BYTES = b"\x00%PDF-1.7\r\n\xff exact pdf bytes \x00"
SUCCESS_BODY = {"content": "extracted text", "page_count": 2}


def make_client(handler, *, base_url=BASE_URL, timeout_seconds=40.0, **client_kwargs):
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, **client_kwargs)
    return client, MS2Client(
        client,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
    )


@pytest.fixture(autouse=True)
async def close_injected_clients(monkeypatch):
    original_make_client = make_client
    clients = []

    def tracked_make_client(*args, **kwargs):
        client, ms2 = original_make_client(*args, **kwargs)
        clients.append(client)
        return client, ms2

    monkeypatch.setattr(sys.modules[__name__], "make_client", tracked_make_client)
    try:
        yield
    finally:
        for client in clients:
            if not client.is_closed:
                await client.aclose()


@pytest.mark.asyncio
async def test_extract_sends_one_exact_multipart_request_and_reuses_injected_client():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=SUCCESS_BODY)

    http_client, ms2 = make_client(handler, base_url=f"{BASE_URL}/")

    result = await ms2.extract(PDF_BYTES, "invoice 2026.pdf")

    assert result == ExtractionResult(content="extracted text", page_count=2)
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == f"{BASE_URL}/extract"
    assert request.headers["content-type"].startswith("multipart/form-data;")
    assert b'name="file"' in request.content
    assert b'filename="invoice 2026.pdf"' in request.content
    assert b"application/pdf" in request.content
    assert PDF_BYTES in request.content
    assert not http_client.is_closed
    await http_client.aclose()


@pytest.mark.parametrize(
    ("timeout_seconds", "expected_timeout"),
    [
        (40.0, 40.0),
        (3.25, 3.25),
    ],
)
@pytest.mark.asyncio
async def test_extract_sets_all_httpx_phase_timeouts(
    timeout_seconds, expected_timeout
):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=SUCCESS_BODY)

    http_client, ms2 = make_client(handler, timeout_seconds=timeout_seconds)

    await ms2.extract(PDF_BYTES, "document.pdf")

    timeout = requests[0].extensions["timeout"]
    assert timeout == {
        "connect": expected_timeout,
        "read": expected_timeout,
        "write": expected_timeout,
        "pool": expected_timeout,
    }
    assert len(requests) == 1
    await http_client.aclose()


@pytest.mark.asyncio
async def test_extract_returns_immutable_typed_result():
    http_client, ms2 = make_client(
        lambda request: httpx.Response(200, json=SUCCESS_BODY)
    )

    result = await ms2.extract(PDF_BYTES, "document.pdf")

    assert isinstance(result, ExtractionResult)
    assert result.content == "extracted text"
    assert result.page_count == 2
    with pytest.raises(FrozenInstanceError):
        result.content = "changed"
    await http_client.aclose()


@pytest.mark.asyncio
async def test_extract_accepts_empty_content_and_zero_pages():
    http_client, ms2 = make_client(
        lambda request: httpx.Response(
            200,
            json={"content": "", "page_count": 0},
        )
    )

    assert await ms2.extract(PDF_BYTES, "scan.pdf") == ExtractionResult("", 0)

    await http_client.aclose()


@pytest.mark.asyncio
async def test_success_response_ignores_additional_json_fields():
    http_client, ms2 = make_client(
        lambda request: httpx.Response(
            200,
            json={
                "content": "text",
                "page_count": 1,
                "future_metadata": {"version": 2},
            },
        )
    )

    assert await ms2.extract(PDF_BYTES, "document.pdf") == ExtractionResult("text", 1)

    await http_client.aclose()


@pytest.mark.parametrize(
    ("status_code", "classification"),
    [
        (429, "transient"),
        (500, "transient"),
        (502, "transient"),
        (503, "transient"),
        (504, "transient"),
        (400, "permanent_input"),
        (413, "permanent_input"),
        (422, "permanent_input"),
    ],
)
@pytest.mark.asyncio
async def test_http_errors_preserve_status_classification_and_retry_after(
    status_code, classification
):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            status_code,
            headers={"Retry-After": "17"},
            content=b'{"detail":"MS2 internal detail"}',
        )

    http_client, ms2 = make_client(handler)

    with pytest.raises(MS2HTTPError) as raised:
        await ms2.extract(PDF_BYTES, "document.pdf")

    assert raised.value.status_code == status_code
    assert raised.value.classification == classification
    assert raised.value.retry_after == "17"
    assert len(requests) == 1
    await http_client.aclose()


@pytest.mark.parametrize(
    ("body", "content_type"),
    [
        (
            b'{"detail":"INTERNAL_MS2_SECRET_73c9"}',
            "application/json",
        ),
        (b"<html>INTERNAL_MS2_SECRET_73c9</html>", "text/html"),
        (b"INTERNAL_MS2_SECRET_73c9", "text/plain"),
    ],
)
@pytest.mark.asyncio
async def test_http_error_does_not_expose_internal_body(body, content_type):
    http_client, ms2 = make_client(
        lambda request: httpx.Response(
            503,
            headers={"content-type": content_type},
            content=body,
        )
    )

    with pytest.raises(MS2HTTPError) as raised:
        await ms2.extract(PDF_BYTES, "document.pdf")

    assert raised.value.status_code == 503
    assert raised.value.classification == "transient"
    assert "INTERNAL_MS2_SECRET_73c9" not in str(raised.value)
    await http_client.aclose()


@pytest.mark.asyncio
async def test_http_error_retry_after_is_none_when_header_is_absent():
    http_client, ms2 = make_client(
        lambda request: httpx.Response(503, content=b"unavailable")
    )

    with pytest.raises(MS2HTTPError) as raised:
        await ms2.extract(PDF_BYTES, "document.pdf")

    assert raised.value.retry_after is None
    await http_client.aclose()


@pytest.mark.parametrize(
    ("status_code", "classification"),
    [
        (201, "unclassified"),
        (301, "unclassified"),
        (404, "unclassified"),
        (418, "unclassified"),
    ],
)
@pytest.mark.asyncio
async def test_other_non_200_statuses_are_unclassified_and_not_retried(
    status_code, classification
):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status_code, content=b"unexpected status")

    http_client, ms2 = make_client(handler)

    with pytest.raises(MS2HTTPError) as raised:
        await ms2.extract(PDF_BYTES, "document.pdf")

    assert raised.value.status_code == status_code
    assert raised.value.classification == classification
    assert len(requests) == 1
    await http_client.aclose()


@pytest.mark.asyncio
async def test_redirect_is_not_followed_even_if_injected_client_enables_redirects():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(302, headers={"Location": "/other"})

    http_client, ms2 = make_client(handler, follow_redirects=True)

    with pytest.raises(MS2HTTPError) as raised:
        await ms2.extract(PDF_BYTES, "document.pdf")

    assert raised.value.status_code == 302
    assert raised.value.classification == "unclassified"
    assert len(requests) == 1
    await http_client.aclose()


@pytest.mark.parametrize(
    ("error", "transport_kind"),
    [
        (httpx.TimeoutException("read timed out"), "timeout"),
        (httpx.ConnectTimeout("connect timed out"), "timeout"),
        (httpx.ReadTimeout("read timed out"), "timeout"),
        (httpx.ConnectError("connection failed"), "connection"),
        (httpx.ConnectError("DNS lookup failed"), "connection"),
        (httpx.ReadError("transport failed"), "protocol"),
        (httpx.WriteError("transport failed"), "protocol"),
        (httpx.RemoteProtocolError("remote protocol failed"), "protocol"),
    ],
)
@pytest.mark.asyncio
async def test_httpx_timeout_and_network_errors_are_structured_and_keep_cause(
    error, transport_kind
):
    def handler(request):
        raise error

    http_client, ms2 = make_client(handler)

    with pytest.raises(MS2TransportError) as raised:
        await ms2.extract(PDF_BYTES, "document.pdf")

    assert raised.value.classification == "transient"
    assert raised.value.transport_kind.value == transport_kind
    assert raised.value.__cause__ is error
    await http_client.aclose()


@pytest.mark.parametrize(
    "error",
    [
        httpx.LocalProtocolError("invalid local protocol state"),
        httpx.UnsupportedProtocol("unsupported local protocol"),
    ],
)
@pytest.mark.asyncio
async def test_local_protocol_errors_are_not_classified_as_transient(error):
    def handler(request):
        raise error

    http_client, ms2 = make_client(handler)

    with pytest.raises(type(error)) as raised:
        await ms2.extract(PDF_BYTES, "document.pdf")

    assert raised.value is error
    assert not isinstance(raised.value, MS2TransportError)
    await http_client.aclose()


@pytest.mark.asyncio
async def test_total_deadline_cancels_in_flight_request_and_is_transient():
    request_started = asyncio.Event()
    handler_cancelled = asyncio.Event()

    async def handler(request):
        request_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            handler_cancelled.set()
            raise

    http_client, ms2 = make_client(handler, timeout_seconds=0.03, timeout=5.0)

    with pytest.raises(MS2TransportError) as raised:
        await asyncio.wait_for(
            ms2.extract(PDF_BYTES, "document.pdf"),
            timeout=1.0,
        )

    assert request_started.is_set()
    assert handler_cancelled.is_set()
    assert raised.value.classification == "transient"
    assert raised.value.transport_kind.value == "timeout"
    assert isinstance(raised.value.__cause__, TimeoutError)
    await http_client.aclose()


@pytest.mark.asyncio
async def test_unexpected_programming_error_is_not_converted_to_transport_error():
    error = RuntimeError("unexpected programming defect")

    def handler(request):
        raise error

    http_client, ms2 = make_client(handler)

    with pytest.raises(RuntimeError) as raised:
        await ms2.extract(PDF_BYTES, "document.pdf")

    assert raised.value is error
    await http_client.aclose()


@pytest.mark.asyncio
async def test_timeout_error_from_handler_before_deadline_propagates_unchanged():
    error = TimeoutError("unexpected timeout from handler")

    def handler(request):
        raise error

    http_client, ms2 = make_client(handler, timeout_seconds=5.0)

    with pytest.raises(TimeoutError) as raised:
        await ms2.extract(PDF_BYTES, "document.pdf")

    assert raised.value is error
    assert not isinstance(raised.value, MS2TransportError)
    await http_client.aclose()


@pytest.mark.parametrize(
    "body",
    [
        b"{invalid json",
        b"null",
        b"[]",
        b"42",
        b'"json string"',
    ],
)
@pytest.mark.asyncio
async def test_success_response_requires_valid_json_object(body):
    http_client, ms2 = make_client(
        lambda request: httpx.Response(200, content=body)
    )

    with pytest.raises(MS2InvalidResponseError):
        await ms2.extract(PDF_BYTES, "document.pdf")

    await http_client.aclose()


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"content": "text"},
        {"page_count": 1},
    ],
)
@pytest.mark.asyncio
async def test_success_response_requires_both_fields(body):
    http_client, ms2 = make_client(lambda request: httpx.Response(200, json=body))

    with pytest.raises(MS2InvalidResponseError):
        await ms2.extract(PDF_BYTES, "document.pdf")

    await http_client.aclose()


@pytest.mark.parametrize(
    "content",
    [None, 1, False, [], {}],
)
@pytest.mark.asyncio
async def test_success_response_rejects_non_string_content(content):
    http_client, ms2 = make_client(
        lambda request: httpx.Response(
            200,
            json={"content": content, "page_count": 1},
        )
    )

    with pytest.raises(MS2InvalidResponseError):
        await ms2.extract(PDF_BYTES, "document.pdf")

    await http_client.aclose()


@pytest.mark.parametrize(
    "page_count",
    [-1, True, False, 1.5, "2", None, [], {}],
)
@pytest.mark.asyncio
async def test_success_response_rejects_invalid_page_count(page_count):
    http_client, ms2 = make_client(
        lambda request: httpx.Response(
            200,
            json={"content": "text", "page_count": page_count},
        )
    )

    with pytest.raises(MS2InvalidResponseError):
        await ms2.extract(PDF_BYTES, "document.pdf")

    await http_client.aclose()


@pytest.mark.asyncio
async def test_caller_cancellation_propagates_and_cancels_in_flight_request():
    request_started = asyncio.Event()
    request_cancelled = asyncio.Event()

    async def handler(request):
        request_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            request_cancelled.set()
            raise

    http_client, ms2 = make_client(handler)
    task = asyncio.create_task(ms2.extract(PDF_BYTES, "document.pdf"))
    try:
        await asyncio.wait_for(request_started.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert request_cancelled.is_set()
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await http_client.aclose()


@pytest.mark.parametrize(
    "timeout_seconds",
    [0, -1, float("inf"), float("-inf"), float("nan"), True, False],
)
def test_constructor_rejects_invalid_timeout(timeout_seconds):
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: None))

    with pytest.raises((TypeError, ValueError)):
        MS2Client(
            client,
            base_url=BASE_URL,
            timeout_seconds=timeout_seconds,
        )

    assert not client.is_closed
    asyncio.run(client.aclose())


@pytest.mark.parametrize(
    "base_url",
    [
        "",
        "/relative/path",
        "ftp://ms2.internal",
        "file:///tmp/ms2",
        "http://",
        "https://",
        "http:///path",
        "http://ms2.internal?mode=test",
        "http://ms2.internal/#fragment",
        "http://user:password@ms2.internal",
    ],
)
def test_constructor_rejects_invalid_or_non_http_base_url(base_url):
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: None))

    with pytest.raises((TypeError, ValueError)):
        MS2Client(client, base_url=base_url)

    assert not client.is_closed
    asyncio.run(client.aclose())


@pytest.mark.parametrize(
    "base_url",
    [
        "http://@ms2.internal",
        "http://:@ms2.internal",
        "http://ms2.internal?",
        "http://ms2.internal#",
        "http://ms2.internal?#",
    ],
)
def test_constructor_rejects_empty_url_delimiters(base_url):
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: None))

    with pytest.raises(ValueError):
        MS2Client(client, base_url=base_url)

    assert not client.is_closed
    asyncio.run(client.aclose())


@pytest.mark.asyncio
async def test_constructor_accepts_at_sign_in_valid_path_and_explicit_port():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=SUCCESS_BODY)

    http_client, ms2 = make_client(
        handler,
        base_url="https://ms2.internal:8443/tenant/@extract/",
    )

    await ms2.extract(PDF_BYTES, "document.pdf")

    assert str(requests[0].url) == (
        "https://ms2.internal:8443/tenant/@extract/extract"
    )
    await http_client.aclose()


@pytest.mark.parametrize("base_url", ["http://ms2.internal", "https://ms2.internal"])
def test_constructor_accepts_http_and_https_base_urls(base_url):
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: None))

    MS2Client(client, base_url=base_url)

    assert not client.is_closed
    asyncio.run(client.aclose())


@pytest.mark.parametrize("status_code", [429, 500, 502, 503, 504])
@pytest.mark.asyncio
async def test_transient_http_failure_still_makes_only_one_request(status_code):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status_code, content=b"temporary failure")

    http_client, ms2 = make_client(handler)

    with pytest.raises(MS2HTTPError):
        await ms2.extract(PDF_BYTES, "document.pdf")

    assert len(requests) == 1
    assert not http_client.is_closed
    await http_client.aclose()


@pytest.mark.asyncio
async def test_same_injected_client_is_reused_across_invocations_and_not_closed():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=SUCCESS_BODY)

    http_client, ms2 = make_client(handler)

    await ms2.extract(PDF_BYTES, "first.pdf")
    await ms2.extract(PDF_BYTES, "second.pdf")

    assert len(requests) == 2
    assert not http_client.is_closed
    await http_client.aclose()
