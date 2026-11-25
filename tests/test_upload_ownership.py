"""Tests unitarios para acquire_upload_ownership() (fase RED)."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from bson import ObjectId
from pymongo import ReturnDocument
from pymongo.errors import ServerSelectionTimeoutError

from src.core.exceptions import DatabaseUnavailableError
from src.models.document import DocumentModel, DocumentStatus
from src.repository.document_repository import DocumentRepository


@pytest.fixture
def repo_with_mocks():
    """Crea un DocumentRepository con una colección Motor mockeada."""
    collection = MagicMock()
    collection.find_one_and_update = AsyncMock()
    collection.find_one = AsyncMock()

    repo = DocumentRepository.__new__(DocumentRepository)
    repo.collection = collection
    return repo


def mongo_document(
    *,
    document_id=None,
    checksum="checksum-ownership-test",
    filename="reserved.pdf",
    size_bytes=1024,
    status=DocumentStatus.UPLOADING.value,
    upload_owner_id=None,
    upload_lease_until=None,
    upload_fencing_token=None,
):
    """Crea una representación MongoDB con _id de tipo ObjectId."""
    return {
        "_id": document_id or ObjectId(),
        "filename": filename,
        "checksum": checksum,
        "size_bytes": size_bytes,
        "status": status,
        "content": None,
        "page_count": None,
        "error_message": None,
        "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
        "file_id": None,
        "upload_owner_id": upload_owner_id,
        "upload_lease_until": upload_lease_until,
        "upload_fencing_token": upload_fencing_token,
    }


def get_call_argument(call, position, name):
    """Recupera un argumento admitiendo forma posicional o nombrada."""
    if len(call.args) > position:
        return call.args[position]
    return call.kwargs[name]


@pytest.mark.asyncio
async def test_acquire_upload_ownership_success(repo_with_mocks):
    """Adquiere y retorna el documento actualizado con el propietario asignado."""
    repo = repo_with_mocks
    checksum = "checksum-ownership-test"
    owner_id = "upload-process-1"
    fencing_token = "claim-token-1"
    lease_until = datetime.now(timezone.utc) + timedelta(minutes=5)
    document_id = ObjectId()
    persisted_document = mongo_document(
        document_id=document_id,
        checksum=checksum,
        upload_owner_id=owner_id,
        upload_lease_until=lease_until,
        upload_fencing_token=fencing_token,
    )
    repo.collection.find_one_and_update.return_value = persisted_document

    before = datetime.now(timezone.utc)
    result_document, acquired = await repo.acquire_upload_ownership(
        checksum=checksum,
        upload_owner_id=owner_id,
        upload_fencing_token=fencing_token,
        upload_lease_until=lease_until,
    )
    after = datetime.now(timezone.utc)

    assert acquired is True
    assert isinstance(result_document, DocumentModel)
    assert result_document.id == str(document_id)
    assert result_document.upload_owner_id == owner_id
    assert result_document.upload_lease_until == lease_until
    assert result_document.upload_fencing_token == fencing_token

    repo.collection.find_one_and_update.assert_awaited_once()
    repo.collection.find_one.assert_not_called()
    call = repo.collection.find_one_and_update.call_args
    query = get_call_argument(call, 0, "filter")
    update = get_call_argument(call, 1, "update")
    assert call.kwargs["return_document"] == ReturnDocument.AFTER
    assert call.kwargs.get("upsert", False) is False

    assert query["checksum"] == checksum
    assert query["status"] == DocumentStatus.UPLOADING.value
    assert query["file_id"] is None
    now = query["$or"][2]["upload_lease_until"]["$lt"]
    assert query["$or"] == [
        {"upload_owner_id": {"$exists": False}},
        {"upload_owner_id": None},
        {"upload_lease_until": {"$lt": now}},
    ]
    assert before <= now <= after
    assert now.tzinfo == timezone.utc
    assert now.utcoffset() == timedelta(0)

    assert set(update) == {"$set"}
    set_fields = update["$set"]
    assert set(set_fields) == {
        "upload_owner_id",
        "upload_lease_until",
        "upload_fencing_token",
        "updated_at",
    }
    assert set_fields["upload_owner_id"] == owner_id
    assert set_fields["upload_lease_until"] == lease_until
    assert set_fields["upload_fencing_token"] == fencing_token
    assert set_fields["updated_at"].tzinfo == timezone.utc
    assert set_fields["updated_at"].utcoffset() == timedelta(0)


@pytest.mark.asyncio
async def test_acquire_upload_ownership_denied_returns_existing_document(
    repo_with_mocks,
):
    """Si el claim no coincide, recupera el documento existente sin alterarlo."""
    repo = repo_with_mocks
    checksum = "checksum-ownership-test"
    existing_id = ObjectId()
    existing_owner = "active-upload-process"
    existing_lease = datetime.now(timezone.utc) + timedelta(minutes=2)
    existing_token = "active-token"
    existing_document = mongo_document(
        document_id=existing_id,
        checksum=checksum,
        filename="existing.pdf",
        size_bytes=2048,
        upload_owner_id=existing_owner,
        upload_lease_until=existing_lease,
        upload_fencing_token=existing_token,
    )
    repo.collection.find_one_and_update.return_value = None
    repo.collection.find_one.return_value = existing_document

    lease_until = datetime.now(timezone.utc) + timedelta(minutes=5)
    result_document, acquired = await repo.acquire_upload_ownership(
        checksum=checksum,
        upload_owner_id="contending-process",
        upload_fencing_token="contending-token",
        upload_lease_until=lease_until,
    )

    assert acquired is False
    assert isinstance(result_document, DocumentModel)
    assert result_document.id == str(existing_id)
    assert result_document.filename == "existing.pdf"
    assert result_document.checksum == checksum
    assert result_document.size_bytes == 2048
    assert result_document.status == DocumentStatus.UPLOADING
    assert result_document.upload_owner_id == existing_owner
    assert result_document.upload_lease_until == existing_lease
    assert result_document.upload_fencing_token == existing_token

    repo.collection.find_one_and_update.assert_awaited_once()
    repo.collection.find_one.assert_awaited_once_with({"checksum": checksum})


@pytest.mark.asyncio
async def test_acquire_upload_ownership_document_not_found(repo_with_mocks):
    """Si el claim falla y no hay documento con checksum, retorna None/False."""
    repo = repo_with_mocks
    checksum = "missing-checksum"
    repo.collection.find_one_and_update.return_value = None
    repo.collection.find_one.return_value = None

    result = await repo.acquire_upload_ownership(
        checksum=checksum,
        upload_owner_id="upload-process-1",
        upload_fencing_token="claim-token-1",
        upload_lease_until=datetime.now(timezone.utc) + timedelta(minutes=5),
    )

    assert result == (None, False)
    repo.collection.find_one_and_update.assert_awaited_once()
    repo.collection.find_one.assert_awaited_once_with({"checksum": checksum})


@pytest.mark.asyncio
async def test_acquire_upload_ownership_connectivity_error_during_claim(
    repo_with_mocks,
):
    """Traduce a DatabaseUnavailableError el fallo de conectividad del claim."""
    repo = repo_with_mocks
    repo.collection.find_one_and_update.side_effect = ServerSelectionTimeoutError(
        "Connection timed out"
    )

    with pytest.raises(DatabaseUnavailableError):
        await repo.acquire_upload_ownership(
            checksum="checksum-ownership-test",
            upload_owner_id="upload-process-1",
            upload_fencing_token="claim-token-1",
            upload_lease_until=datetime.now(timezone.utc) + timedelta(minutes=5),
        )

    repo.collection.find_one_and_update.assert_awaited_once()
    repo.collection.find_one.assert_not_called()


@pytest.mark.asyncio
async def test_acquire_upload_ownership_connectivity_error_during_recovery(
    repo_with_mocks,
):
    """Traduce a DatabaseUnavailableError el fallo al recuperar el documento."""
    repo = repo_with_mocks
    checksum = "checksum-ownership-test"
    repo.collection.find_one_and_update.return_value = None
    repo.collection.find_one.side_effect = ServerSelectionTimeoutError(
        "Connection timed out"
    )

    with pytest.raises(DatabaseUnavailableError):
        await repo.acquire_upload_ownership(
            checksum=checksum,
            upload_owner_id="upload-process-1",
            upload_fencing_token="claim-token-1",
            upload_lease_until=datetime.now(timezone.utc) + timedelta(minutes=5),
        )

    repo.collection.find_one_and_update.assert_awaited_once()
    repo.collection.find_one.assert_awaited_once_with({"checksum": checksum})


@pytest.mark.parametrize(
    "lease_until",
    [
        pytest.param(
            datetime.now(),
            id="naive-datetime",
        ),
        pytest.param(
            datetime.now(timezone(timedelta(hours=1))) + timedelta(minutes=5),
            id="non-utc-datetime",
        ),
        pytest.param(
            datetime.now(timezone.utc) - timedelta(seconds=1),
            id="expired-datetime",
        ),
    ],
)
@pytest.mark.asyncio
async def test_acquire_upload_ownership_rejects_invalid_expiration(
    repo_with_mocks,
    lease_until,
):
    """Rechaza expiraciones naive, no UTC o vencidas antes de acceder a MongoDB."""
    repo = repo_with_mocks

    with pytest.raises(ValueError):
        await repo.acquire_upload_ownership(
            checksum="checksum-ownership-test",
            upload_owner_id="upload-process-1",
            upload_fencing_token="claim-token-1",
            upload_lease_until=lease_until,
        )

    repo.collection.find_one_and_update.assert_not_called()
    repo.collection.find_one.assert_not_called()
