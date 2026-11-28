"""HTTP contract tests for asynchronous document reception."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient

from src.api import docs_api
from src.api.pdf_validation import PdfValidationExecutor
from src.core.exceptions import (
    DatabaseUnavailableError,
    EmptyFileError,
    ExtractionError,
    FileTooLargeError,
    InvalidFileTypeError,
)
from src.main import app
from src.models.document import DocumentModel, DocumentStatus
from src.schemas.document import DocumentReceipt
from src.services.document_service import (
    UploadFinalizationRejectedError,
    UploadLeaseLostError,
)


PDF_BYTES = b"%PDF-1.7 test payload"
PUBLIC_503 = "Servicio temporalmente no disponible"
PUBLIC_DATABASE_503 = (
    "Servicio temporalmente no disponible: base de datos no accesible"
)
PUBLIC_500 = "Ocurrió un error inesperado durante la recepción del documento"


def make_document(status, *, created_at=None, document_id=None):
    return DocumentModel(
        id=str(document_id or ObjectId()),
        filename="stored-original.pdf",
        checksum="internal-checksum",
        size_bytes=len(PDF_BYTES),
        status=status,
        created_at=created_at or datetime(2024, 5, 6, 7, 8, 9, tzinfo=timezone.utc),
        updated_at=datetime(2024, 5, 6, 7, 9, 9, tzinfo=timezone.utc),
        file_id=ObjectId(),
        upload_owner_id="private-owner",
        upload_fencing_token="private-upload-token",
        upload_lease_until=datetime(2030, 1, 1, tzinfo=timezone.utc),
        worker_id="private-worker",
        fencing_token="private-worker-token",
        lease_until=datetime(2030, 1, 1, tzinfo=timezone.utc),
        content="private extracted content",
        page_count=9,
    )


@pytest.fixture
def api(monkeypatch):
    service = AsyncMock()
    service.submit_pdf = AsyncMock()
    service.process_pdf_extraction = AsyncMock()
    service.call_extraction_service = AsyncMock()
    service.get_document_by_id = AsyncMock()
    executor = PdfValidationExecutor()
    app.state.pdf_validation_executor = executor
    app.dependency_overrides[docs_api.get_document_service] = lambda: service
    validator = AsyncMock(return_value=PDF_BYTES)
    monkeypatch.setattr(docs_api, "validate_pdf_upload", validator)
    # Do not enter TestClient's context manager: it starts the real lifespan
    # and MongoDB supervision. This fixture owns the executor instead.
    client = TestClient(app)
    yield client, service, validator, executor

    app.dependency_overrides.pop(docs_api.get_document_service, None)
    app.state.pdf_validation_executor = None
    executor.shutdown(wait=True, cancel_futures=True)


def post(client, filename="client-name.pdf"):
    return client.post(
        "/documents/",
        files={"file": (filename, PDF_BYTES, "application/pdf")},
    )


@pytest.mark.parametrize(
    ("classification", "status_code", "document_status"),
    [
        ("accepted", 202, DocumentStatus.PENDING),
        ("existing", 200, DocumentStatus.COMPLETED),
    ],
)
def test_receipt_response_has_exact_ticket_and_location(
    api, classification, status_code, document_status
):
    client, service, validator, executor = api
    created_at = datetime(2024, 5, 6, 7, 8, 9, tzinfo=timezone.utc)
    document = make_document(document_status, created_at=created_at)
    service.receive_pdf.return_value = (document, classification)

    response = post(client)
    body = response.json()
    expected_url = f"/documents/{document.id}"

    assert response.status_code == status_code
    assert body == {
        "document_id": document.id,
        "status": document_status.value,
        "created_at": created_at.isoformat().replace("+00:00", "Z"),
        "status_url": expected_url,
    }
    assert response.headers["location"] == expected_url
    assert set(body) == {"document_id", "status", "created_at", "status_url"}
    validator.assert_awaited_once_with(
        validator.await_args.args[0],
        executor=executor,
    )
    service.receive_pdf.assert_awaited_once_with(
        file_bytes=PDF_BYTES,
        filename="client-name.pdf",
    )
    service.get_document_by_id.assert_not_awaited()
    service.submit_pdf.assert_not_awaited()
    service.process_pdf_extraction.assert_not_awaited()
    service.call_extraction_service.assert_not_awaited()


@pytest.mark.parametrize(
    "document_status",
    [
        DocumentStatus.UPLOADING,
        DocumentStatus.PENDING,
        DocumentStatus.PROCESSING,
        DocumentStatus.COMPLETED,
        DocumentStatus.FAILED,
    ],
)
def test_all_duplicate_states_return_200_and_preserve_identity_and_created_at(
    api, document_status
):
    client, service, _, _ = api
    document_id = str(ObjectId())
    original_created_at = datetime(2022, 3, 4, 5, 6, 7, tzinfo=timezone.utc)
    original = make_document(
        document_status,
        created_at=original_created_at,
        document_id=document_id,
    )
    service.receive_pdf.return_value = (original, "existing")

    response = post(client)

    assert response.status_code == 200
    assert response.json()["document_id"] == document_id
    assert response.json()["created_at"] == original_created_at.isoformat().replace(
        "+00:00", "Z"
    )
    assert response.json()["status"] == document_status.value
    assert response.headers["location"] == f"/documents/{document_id}"


def test_bson_naive_datetime_is_serialized_as_utc(api):
    client, service, _, _ = api
    naive_bson_datetime = datetime(2020, 2, 3, 4, 5, 6)
    service.receive_pdf.return_value = (
        make_document(DocumentStatus.PENDING, created_at=naive_bson_datetime),
        "accepted",
    )

    response = post(client)

    assert response.status_code == 202
    assert response.json()["created_at"] == "2020-02-03T04:05:06Z"


def test_validation_precedes_receive_and_original_filename_is_preserved(api):
    client, service, validator, _ = api
    events = []
    document = make_document(DocumentStatus.PENDING)

    async def validated_bytes(upload, **kwargs):
        events.append("validated")
        return PDF_BYTES

    validator.side_effect = validated_bytes

    async def receive_pdf(*, file_bytes, filename):
        events.append("received")
        assert file_bytes == PDF_BYTES
        assert filename == "original-name.pdf"
        return document, "accepted"

    service.receive_pdf.side_effect = receive_pdf
    response = post(client, filename="original-name.pdf")

    assert response.status_code == 202
    assert events == ["validated", "received"]


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (InvalidFileTypeError(), 400),
        (EmptyFileError(), 400),
        (FileTooLargeError("too large"), 413),
        (ExtractionError(), 422),
    ],
)
def test_validation_errors_keep_their_public_status_and_skip_service(
    api, error, status_code
):
    client, service, validator, _ = api
    validator.side_effect = error

    response = post(client)

    assert response.status_code == status_code
    service.receive_pdf.assert_not_awaited()


@pytest.mark.parametrize(
    ("error", "detail"),
    [
        (DatabaseUnavailableError(), PUBLIC_DATABASE_503),
        (UploadLeaseLostError("internal ownership detail"), PUBLIC_503),
        (
            UploadFinalizationRejectedError("internal finalization detail"),
            PUBLIC_503,
        ),
    ],
)
def test_domain_persistence_and_ownership_errors_return_503(api, error, detail):
    client, service, _, _ = api
    service.receive_pdf.side_effect = error

    response = post(client)

    assert response.status_code == 503
    assert response.json()["detail"] == detail
    assert "internal" not in response.text


@pytest.mark.parametrize("executor_state", ["missing", "closing"])
def test_unavailable_or_closing_executor_returns_503(api, executor_state):
    client, service, validator, executor = api
    if executor_state == "missing":
        app.state.pdf_validation_executor = None
    else:
        executor.begin_shutdown()

    response = post(client)

    assert response.status_code == 503
    assert response.json()["detail"] == PUBLIC_503
    validator.assert_not_awaited()
    service.receive_pdf.assert_not_awaited()


def test_executor_shutdown_race_is_503_but_other_runtime_errors_are_500(api):
    client, service, validator, executor = api
    executor.begin_shutdown()

    response = post(client)

    assert response.status_code == 503
    assert response.json()["detail"] == PUBLIC_503

    racing_executor = PdfValidationExecutor()
    app.state.pdf_validation_executor = racing_executor

    def raise_shutdown_race(*args, **kwargs):
        racing_executor.begin_shutdown()
        raise RuntimeError("PDF validation executor is shutting down")

    validator.side_effect = raise_shutdown_race
    racing_response = post(client)
    assert racing_response.status_code == 503
    assert racing_response.json()["detail"] == PUBLIC_503

    app.state.pdf_validation_executor = PdfValidationExecutor()
    validator.side_effect = RuntimeError("unexpected parser runtime failure")
    unexpected_response = post(client)

    assert unexpected_response.status_code == 500
    assert unexpected_response.json()["detail"] == PUBLIC_500
    assert "unexpected parser runtime failure" not in unexpected_response.text
    app.state.pdf_validation_executor.shutdown(wait=True, cancel_futures=True)
    racing_executor.shutdown(wait=True, cancel_futures=True)
    app.state.pdf_validation_executor = executor


def test_unexpected_service_error_is_generic_500_and_synchronous_flow_is_unused(api):
    client, service, _, _ = api
    service.receive_pdf.side_effect = RuntimeError("private database detail")

    response = post(client)

    assert response.status_code == 500
    assert response.json()["detail"] == PUBLIC_500
    assert "private database detail" not in response.text
    service.submit_pdf.assert_not_awaited()
    service.process_pdf_extraction.assert_not_awaited()
    service.call_extraction_service.assert_not_awaited()
    service.get_document_by_id.assert_not_awaited()


def test_openapi_documents_202_200_location_and_relevant_errors():
    openapi = app.openapi()
    post_operation = openapi["paths"]["/documents/"]["post"]
    responses = post_operation["responses"]

    assert "201" not in responses
    for code in ("200", "202"):
        assert responses[code]["content"]["application/json"]["schema"]["$ref"].endswith(
            "/DocumentReceipt"
        )
        assert "Location" in responses[code]["headers"]
    assert {"400", "413", "422", "500", "503"} <= set(responses)
