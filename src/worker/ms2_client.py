"""Reusable asynchronous HTTP client for PDF extraction through MS2."""

import asyncio
from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real
from typing import Literal
from urllib.parse import urlsplit

import httpx


ErrorClassification = Literal["transient", "permanent_input", "unclassified"]


class TransportKind(str, Enum):
    """Structured kind of remote transport failure observed by the MS2 client."""

    TIMEOUT = "timeout"
    CONNECTION = "connection"
    PROTOCOL = "protocol"


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Validated extraction result returned by MS2."""

    content: str
    page_count: int


class MS2ClientError(Exception):
    """Base class for errors raised by the MS2 HTTP client."""


class MS2HTTPError(MS2ClientError):
    """Non-200 response returned by MS2 or its internal proxy."""

    def __init__(
        self,
        status_code: int,
        classification: ErrorClassification,
        retry_after: str | None,
    ) -> None:
        self.status_code = status_code
        self.classification = classification
        self.retry_after = retry_after
        super().__init__(f"MS2 returned HTTP {status_code}")


class MS2TransportError(MS2ClientError):
    """Transient network or timeout failure while calling MS2."""

    classification: Literal["transient"] = "transient"

    def __init__(
        self,
        message: str = "MS2 request failed",
        *,
        transport_kind: TransportKind | None = None,
    ) -> None:
        if transport_kind is not None and not isinstance(
            transport_kind, TransportKind
        ):
            raise TypeError("transport_kind must be a TransportKind or None")
        self.transport_kind = transport_kind
        super().__init__(message)


class MS2InvalidResponseError(MS2ClientError):
    """MS2 returned HTTP 200 with a body outside the extraction contract."""


_TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})
_PERMANENT_INPUT_STATUSES = frozenset({400, 413, 422})


def validate_ms2_base_url(base_url: str) -> str:
    """Validate and normalize the base URL accepted by the MS2 client."""
    if not isinstance(base_url, str):
        raise TypeError("base_url must be a string")

    try:
        parsed_url = httpx.URL(base_url)
        original_parts = urlsplit(base_url)
    except httpx.InvalidURL as exc:
        raise ValueError("base_url must be an absolute HTTP or HTTPS URL") from exc
    except ValueError as exc:
        raise ValueError("base_url must be a valid absolute URL") from exc

    if (
        parsed_url.scheme not in {"http", "https"}
        or not parsed_url.host
        or parsed_url.username
        or parsed_url.password
        or "@" in original_parts.netloc
        or parsed_url.query
        or parsed_url.fragment
        or "?" in base_url.split("#", maxsplit=1)[0]
        or "#" in base_url
    ):
        raise ValueError(
            "base_url must have an HTTP or HTTPS scheme and host, "
            "without credentials, query parameters, or fragment"
        )

    return str(parsed_url).rstrip("/")


class MS2Client:
    """Send PDFs to MS2 using a caller-owned, reusable HTTPX client."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        base_url: str,
        timeout_seconds: float = 40.0,
    ) -> None:
        if isinstance(timeout_seconds, bool) or not isinstance(
            timeout_seconds, Real
        ):
            raise TypeError("timeout_seconds must be a finite positive number")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a finite positive number")

        self._client = client
        self._base_url = validate_ms2_base_url(base_url)
        self._timeout_seconds = float(timeout_seconds)

    async def extract(
        self,
        pdf_bytes: bytes,
        filename: str,
    ) -> ExtractionResult:
        """Submit one PDF and validate the successful extraction response."""
        phase_timeout = httpx.Timeout(self._timeout_seconds)
        deadline = asyncio.timeout(self._timeout_seconds)

        try:
            async with deadline:
                response = await self._client.post(
                    f"{self._base_url}/extract",
                    files={
                        "file": (
                            filename,
                            pdf_bytes,
                            "application/pdf",
                        )
                    },
                    timeout=phase_timeout,
                    follow_redirects=False,
                )
        except TimeoutError as exc:
            # Do not misclassify an unrelated TimeoutError raised by the
            # transport or a custom handler as expiration of our deadline.
            if not deadline.expired():
                raise
            raise MS2TransportError(
                "MS2 request exceeded its total deadline",
                transport_kind=TransportKind.TIMEOUT,
            ) from exc
        except httpx.TimeoutException as exc:
            raise MS2TransportError(
                "MS2 request timed out",
                transport_kind=TransportKind.TIMEOUT,
            ) from exc
        except httpx.ConnectError as exc:
            raise MS2TransportError(
                "MS2 connection failed",
                transport_kind=TransportKind.CONNECTION,
            ) from exc
        except (
            httpx.ReadError,
            httpx.WriteError,
            httpx.RemoteProtocolError,
        ) as exc:
            raise MS2TransportError(
                "MS2 remote transport or protocol failure",
                transport_kind=TransportKind.PROTOCOL,
            ) from exc

        if response.status_code != 200:
            if response.status_code in _TRANSIENT_STATUSES:
                classification: ErrorClassification = "transient"
            elif response.status_code in _PERMANENT_INPUT_STATUSES:
                classification = "permanent_input"
            else:
                classification = "unclassified"
            raise MS2HTTPError(
                status_code=response.status_code,
                classification=classification,
                retry_after=response.headers.get("Retry-After"),
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise MS2InvalidResponseError(
                "MS2 returned an invalid JSON extraction response"
            ) from exc

        if not isinstance(payload, dict):
            raise MS2InvalidResponseError(
                "MS2 extraction response must be a JSON object"
            )

        content = payload.get("content")
        page_count = payload.get("page_count")
        if not isinstance(content, str):
            raise MS2InvalidResponseError(
                "MS2 extraction response has an invalid content field"
            )
        if (
            not isinstance(page_count, int)
            or isinstance(page_count, bool)
            or page_count < 0
        ):
            raise MS2InvalidResponseError(
                "MS2 extraction response has an invalid page_count field"
            )

        return ExtractionResult(content=content, page_count=page_count)
