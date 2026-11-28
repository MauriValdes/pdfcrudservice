from copy import deepcopy
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient

from src.api.docs_api import get_document_service
from src.main import app
from src.schemas.document import DocumentResponse


PUBLIC_ERRORS = (
    "El archivo PDF no es válido.",
    "El archivo PDF supera el tamaño máximo permitido.",
    "No se pudo procesar el archivo PDF.",
    "Se agotaron los intentos de procesamiento.",
    "Se agotó el tiempo disponible para el procesamiento.",
)
GENERIC_PUBLIC_ERROR = "No se pudo procesar el documento PDF."
PUBLIC_FIELDS = {
    "id",
    "filename",
    "checksum",
    "size_bytes",
    "status",
    "content",
    "page_count",
    "error_message",
    "created_at",
    "updated_at",
}
OFFICIAL_STATES = ("uploading", "pending", "processing", "completed", "failed")


def document_data(status, **overrides):
    data = {
        "id": str(ObjectId()),
        "filename": "document.pdf",
        "checksum": "abc123",
        "size_bytes": 128,
        "status": status,
        "content": "persisted text",
        "page_count": 3,
        "error_message": None,
        "created_at": datetime(2024, 1, 2, 3, 4, 5),
        "updated_at": datetime(2024, 1, 2, 3, 5, 5, tzinfo=timezone(timedelta(hours=2))),
        "file_id": ObjectId(),
        "worker_id": "private-worker",
        "lease_until": datetime(2030, 1, 1),
        "fencing_token": "private-token",
        "attempt_count": 4,
        "next_retry_at": datetime(2030, 1, 1),
        "last_error": "private internal exception",
        "original_cleanup_status": "pending",
    }
    data.update(overrides)
    return data


@pytest.fixture
def public_get_client():
    service = AsyncMock()
    app.dependency_overrides[get_document_service] = lambda: service
    client = TestClient(app)
    yield client, service
    app.dependency_overrides.pop(get_document_service, None)


@pytest.mark.parametrize("state", OFFICIAL_STATES)
@pytest.mark.parametrize("endpoint", ("list", "single"))
def test_get_state_representation_is_consistent_and_public_only(
    public_get_client, state, endpoint
):
    client, service = public_get_client
    source = document_data(state)
    original = deepcopy(source)
    if endpoint == "list":
        service.get_all_documents.return_value = [
            DocumentResponse.for_public_read(**source)
        ]
        response = client.get("/documents")
        service.get_all_documents.assert_awaited_once()
    else:
        service.get_document_by_id.return_value = DocumentResponse.for_public_read(
            **source
        )
        response = client.get(f"/documents/{source['id']}")
        service.get_document_by_id.assert_awaited_once_with(source["id"])

    assert response.status_code == 200
    body = response.json()[0] if endpoint == "list" else response.json()
    assert set(body) == PUBLIC_FIELDS
    assert body["status"] == state
    assert body["created_at"] == "2024-01-02T03:04:05Z"
    assert body["updated_at"] == "2024-01-02T01:05:05Z"
    assert source == original

    if state in ("uploading", "pending", "processing"):
        assert (body["content"], body["page_count"], body["error_message"]) == (
            None,
            None,
            None,
        )
    elif state == "completed":
        assert (body["content"], body["page_count"], body["error_message"]) == (
            "persisted text",
            3,
            None,
        )
    else:
        assert (body["content"], body["page_count"]) == (None, None)
        assert body["error_message"] == GENERIC_PUBLIC_ERROR


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"content": "", "page_count": 0}, ("", 0)),
        ({"content": None, "page_count": None}, (None, None)),
        ({"content": "stale", "page_count": 9}, ("stale", 9)),
    ],
)
def test_completed_preserves_persisted_optional_results(overrides, expected):
    response = DocumentResponse.for_public_read(
        **document_data("completed", error_message="stale error", **overrides)
    )

    assert (response.content, response.page_count) == expected
    assert response.error_message is None


@pytest.mark.parametrize("message", PUBLIC_ERRORS)
def test_failed_allows_only_approved_public_messages(message):
    response = DocumentResponse.for_public_read(
        **document_data("failed", error_message=message)
    )

    assert response.content is None
    assert response.page_count is None
    assert response.error_message == message


@pytest.mark.parametrize(
    "message",
    [
        None,
        "unknown message",
        "MongoServerSelectionError: db.internal:27017",
        "MS2 returned 500 with private response body",
    ],
)
def test_failed_hides_unknown_or_internal_messages(message):
    response = DocumentResponse.for_public_read(
        **document_data(
            "failed",
            content="stale content",
            page_count=12,
            error_message=message,
            last_error="private traceback",
        )
    )

    assert response.content is None
    assert response.page_count is None
    assert response.error_message == GENERIC_PUBLIC_ERROR
    assert "last_error" not in response.model_dump()


def test_unknown_historical_status_is_preserved_but_results_are_hidden():
    response = DocumentResponse.for_public_read(
        **document_data("legacy-paused", error_message="legacy error")
    )

    assert response.status == "legacy-paused"
    assert (response.content, response.page_count, response.error_message) == (
        None,
        None,
        None,
    )


def test_response_normalizes_naive_and_aware_dates_to_utc():
    response = DocumentResponse.for_public_read(
        **document_data(
            "pending",
            created_at=datetime(2024, 1, 2, 3, 4, 5),
            updated_at=datetime(
                2024, 1, 2, 5, 5, 5, tzinfo=timezone(timedelta(hours=2))
            ),
        )
    )

    assert response.created_at == datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert response.updated_at == datetime(2024, 1, 2, 3, 5, 5, tzinfo=timezone.utc)


def test_response_contract_keeps_exactly_ten_public_fields():
    assert set(DocumentResponse.model_fields) == PUBLIC_FIELDS


@pytest.mark.parametrize(
    ("path", "method_name", "result"),
    [
        ("/documents", "get_all_documents", []),
        ("/documents/not-an-object-id", "get_document_by_id", None),
    ],
)
def test_get_http_statuses_remain_unchanged(public_get_client, path, method_name, result):
    client, service = public_get_client
    getattr(service, method_name).return_value = result

    response = client.get(path)

    expected_status = 200 if method_name == "get_all_documents" else 400
    assert response.status_code == expected_status
