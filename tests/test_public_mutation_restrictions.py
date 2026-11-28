import pytest
from bson import ObjectId
from fastapi.testclient import TestClient

from src.api.docs_api import get_document_service
from src.main import app


PATCH_BODIES = [
    pytest.param({"status": "completed"}, id="status"),
    pytest.param({"content": "text", "page_count": 2}, id="result"),
    pytest.param({"error_message": "error"}, id="error-message"),
    pytest.param({"filename": "renamed.pdf"}, id="filename"),
    pytest.param({"checksum": "new-checksum", "file_id": "new-file"}, id="identity"),
    pytest.param(
        {
            "upload_owner_id": "owner",
            "upload_lease_until": "2030-01-01T00:00:00Z",
            "upload_fencing_token": "upload-token",
        },
        id="upload-ownership",
    ),
    pytest.param(
        {
            "worker_id": "worker",
            "lease_until": "2030-01-01T00:00:00Z",
            "fencing_token": "worker-token",
        },
        id="worker-ownership",
    ),
    pytest.param(
        {
            "attempt_count": 2,
            "processing_cycle_id": "cycle",
            "next_retry_at": "2030-01-01T00:00:00Z",
            "last_error": "private",
            "original_cleanup_status": "pending",
        },
        id="worker-and-cleanup-fields",
    ),
    pytest.param({"unexpected": "value"}, id="unknown-field"),
    pytest.param(None, id="empty-body"),
    pytest.param("malformed-json", id="malformed-json"),
]


@pytest.fixture
def patch_client():
    resolver_calls = []

    def unexpected_service_resolution():
        resolver_calls.append(True)
        raise AssertionError("PATCH no debe resolver DocumentService")

    previous_override = app.dependency_overrides.get(get_document_service)
    app.dependency_overrides[get_document_service] = unexpected_service_resolution
    client = TestClient(app)
    yield client, resolver_calls
    if previous_override is None:
        app.dependency_overrides.pop(get_document_service, None)
    else:
        app.dependency_overrides[get_document_service] = previous_override


@pytest.mark.parametrize("path_id", [str(ObjectId()), "invalid-id"])
@pytest.mark.parametrize("body", PATCH_BODIES)
def test_public_patch_is_not_routed_for_any_id_or_body(patch_client, path_id, body):
    client, resolver_calls = patch_client
    request_args = {}
    if isinstance(body, dict):
        request_args["json"] = body
    elif isinstance(body, str):
        request_args["content"] = body
        request_args["headers"] = {"Content-Type": "application/json"}

    response = client.patch(f"/documents/{path_id}", **request_args)

    assert response.status_code == 405
    assert resolver_calls == []


def test_patch_is_absent_from_document_openapi_operations():
    operation = app.openapi()["paths"]["/documents/{document_id}"]

    assert "patch" not in operation
