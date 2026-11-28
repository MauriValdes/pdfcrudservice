"""Contract tests for pure classification of typed MS2 client errors."""

from dataclasses import FrozenInstanceError

import pytest

from src.worker.ms2_client import (
    MS2ClientError,
    MS2HTTPError,
    MS2InvalidResponseError,
    MS2TransportError,
)


def _classifier_api():
    from src.worker.error_classifier import (
        CircuitBreakerSignal,
        DocumentAction,
        ErrorCategory,
        classify_ms2_error,
    )

    return CircuitBreakerSignal, DocumentAction, ErrorCategory, classify_ms2_error


@pytest.mark.parametrize(
    (
        "status_code",
        "code",
        "action_name",
        "closed_failure_eligible",
        "half_open_signal_name",
        "alert_required",
    ),
    [
        (400, "ms2_invalid_input", "FAIL_INPUT", False, "NEUTRAL", False),
        (413, "pdf_too_large", "FAIL_INPUT", False, "NEUTRAL", False),
        (422, "pdf_unprocessable", "FAIL_INPUT", False, "NEUTRAL", False),
        (429, "ms2_http_429", "RETRY", True, "FAILURE", False),
        (500, "ms2_http_500", "RETRY", True, "FAILURE", False),
        (502, "ms2_http_502", "RETRY", True, "FAILURE", False),
        (503, "ms2_http_503", "RETRY", True, "FAILURE", False),
        (504, "ms2_http_504", "RETRY", True, "FAILURE", False),
        (401, "ms2_http_401", "FAIL_TECHNICAL", False, "NEUTRAL", True),
        (403, "ms2_http_403", "FAIL_TECHNICAL", False, "NEUTRAL", True),
        (404, "ms2_http_404", "FAIL_TECHNICAL", False, "NEUTRAL", True),
        (302, "ms2_http_unclassified", "FAIL_TECHNICAL", False, "NEUTRAL", True),
        (418, "ms2_http_unclassified", "FAIL_TECHNICAL", False, "NEUTRAL", True),
    ],
)
def test_http_status_classification(
    status_code,
    code,
    action_name,
    closed_failure_eligible,
    half_open_signal_name,
    alert_required,
):
    circuit_signal, document_action, error_category, classify_ms2_error = (
        _classifier_api()
    )
    category = classify_ms2_error(
        MS2HTTPError(status_code, "unclassified", retry_after="23")
    )

    assert isinstance(category, error_category)
    assert category.code == code
    assert category.document_action is getattr(document_action, action_name)
    assert category.closed_failure_eligible is closed_failure_eligible
    assert category.half_open_signal is getattr(
        circuit_signal, half_open_signal_name
    )
    assert category.alert_required is alert_required
    assert category.http_status == status_code


@pytest.mark.parametrize(
    ("kind_name", "code"),
    [
        ("TIMEOUT", "ms2_timeout"),
        ("CONNECTION", "ms2_connection_error"),
        ("PROTOCOL", "ms2_protocol_error"),
    ],
)
def test_transport_kind_maps_to_stable_retry_category(kind_name, code):
    from src.worker.ms2_client import TransportKind
    from src.worker.error_classifier import (
        CircuitBreakerSignal,
        DocumentAction,
        classify_ms2_error,
    )

    error = MS2TransportError(
        "message deliberately irrelevant",
        transport_kind=getattr(TransportKind, kind_name),
    )
    category = classify_ms2_error(error)

    assert category is not None
    assert category.code == code
    assert category.document_action is DocumentAction.RETRY
    assert category.closed_failure_eligible is True
    assert category.half_open_signal is CircuitBreakerSignal.FAILURE
    assert category.alert_required is False
    assert category.http_status is None


def test_invalid_ms2_response_is_retryable_and_breaker_eligible():
    CircuitBreakerSignal, DocumentAction, _, classify_ms2_error = _classifier_api()
    category = classify_ms2_error(
        MS2InvalidResponseError("response body must not affect classification")
    )

    assert category is not None
    assert category.code == "ms2_invalid_response"
    assert category.document_action is DocumentAction.RETRY
    assert category.closed_failure_eligible is True
    assert category.half_open_signal is CircuitBreakerSignal.FAILURE
    assert category.alert_required is False
    assert category.http_status is None


def test_error_category_is_immutable():
    _, _, _, classify_ms2_error = _classifier_api()
    category = classify_ms2_error(
        MS2HTTPError(503, "transient", retry_after=None)
    )

    with pytest.raises(FrozenInstanceError):
        category.code = "changed"


def test_untyped_transport_error_is_not_assumed_to_be_a_known_failure():
    # Preserve the existing message-only constructor for current consumers,
    # but do not infer timeout/connection/protocol from its message.
    error = MS2TransportError("timeout")

    _, _, _, classify_ms2_error = _classifier_api()
    assert classify_ms2_error(error) is None


@pytest.mark.parametrize(
    "error",
    [
        MS2ClientError("unknown MS2 client error"),
        RuntimeError("internal worker failure"),
        ValueError("programming error"),
    ],
)
def test_nonclassified_errors_are_not_misreported_as_ms2_failures(error):
    _, _, _, classify_ms2_error = _classifier_api()
    assert classify_ms2_error(error) is None
