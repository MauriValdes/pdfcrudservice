"""Pure classification of typed errors raised by the MS2 worker client."""

from dataclasses import dataclass
from enum import Enum

from src.worker.ms2_client import (
    MS2HTTPError,
    MS2InvalidResponseError,
    MS2TransportError,
    TransportKind,
)


class DocumentAction(str, Enum):
    """Document-level treatment indicated by an MS2 error."""

    FAIL_INPUT = "fail_input"
    RETRY = "retry"
    FAIL_TECHNICAL = "fail_technical"


class CircuitBreakerSignal(str, Enum):
    """Signal an MS2 error provides to a Circuit Breaker."""

    NEUTRAL = "neutral"
    FAILURE = "failure"


@dataclass(frozen=True, slots=True)
class ErrorCategory:
    """Immutable classification data; applying the policy is a caller concern."""

    code: str
    document_action: DocumentAction
    closed_failure_eligible: bool
    half_open_signal: CircuitBreakerSignal
    alert_required: bool
    http_status: int | None


_PERMANENT_INPUT_CODES = {
    400: "ms2_invalid_input",
    413: "pdf_too_large",
    422: "pdf_unprocessable",
}

_RETRYABLE_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})
_ALERT_HTTP_STATUSES = frozenset({401, 403, 404})


def _category(
    code: str,
    document_action: DocumentAction,
    *,
    closed_failure_eligible: bool,
    alert_required: bool = False,
    http_status: int | None = None,
) -> ErrorCategory:
    """Build a category with the matching HALF-OPEN signal."""
    return ErrorCategory(
        code=code,
        document_action=document_action,
        closed_failure_eligible=closed_failure_eligible,
        half_open_signal=(
            CircuitBreakerSignal.FAILURE
            if closed_failure_eligible
            else CircuitBreakerSignal.NEUTRAL
        ),
        alert_required=alert_required,
        http_status=http_status,
    )


def classify_ms2_error(error: Exception) -> ErrorCategory | None:
    """Classify only recognized, typed MS2 client errors without side effects."""
    if isinstance(error, MS2HTTPError):
        status_code = error.status_code
        permanent_code = _PERMANENT_INPUT_CODES.get(status_code)
        if permanent_code is not None:
            return _category(
                permanent_code,
                DocumentAction.FAIL_INPUT,
                closed_failure_eligible=False,
                http_status=status_code,
            )

        if status_code in _RETRYABLE_HTTP_STATUSES:
            return _category(
                f"ms2_http_{status_code}",
                DocumentAction.RETRY,
                closed_failure_eligible=True,
                http_status=status_code,
            )

        code = (
            f"ms2_http_{status_code}"
            if status_code in _ALERT_HTTP_STATUSES
            else "ms2_http_unclassified"
        )
        return _category(
            code,
            DocumentAction.FAIL_TECHNICAL,
            closed_failure_eligible=False,
            alert_required=True,
            http_status=status_code,
        )

    if isinstance(error, MS2TransportError):
        transport_codes = {
            TransportKind.TIMEOUT: "ms2_timeout",
            TransportKind.CONNECTION: "ms2_connection_error",
            TransportKind.PROTOCOL: "ms2_protocol_error",
        }
        code = transport_codes.get(error.transport_kind)
        if code is None:
            return None
        return _category(
            code,
            DocumentAction.RETRY,
            closed_failure_eligible=True,
        )

    if isinstance(error, MS2InvalidResponseError):
        return _category(
            "ms2_invalid_response",
            DocumentAction.RETRY,
            closed_failure_eligible=True,
        )

    return None
