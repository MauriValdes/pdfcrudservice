"""Tests unitarios para renew_upload_lease() (fase RED)."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from bson import ObjectId
from pymongo.errors import ServerSelectionTimeoutError

from src.core.exceptions import DatabaseUnavailableError
from src.models.document import DocumentStatus
from src.repository.document_repository import DocumentRepository


@pytest.fixture
def repo_with_mocks():
    """Crea un DocumentRepository con una colección Motor mockeada."""
    collection = MagicMock()
    collection.update_one = AsyncMock()
    collection.find_one = AsyncMock()

    repo = DocumentRepository.__new__(DocumentRepository)
    repo.collection = collection
    return repo


def get_call_argument(call, position, name):
    """Recupera un argumento admitiendo forma posicional o nombrada."""
    if len(call.args) > position:
        return call.args[position]
    return call.kwargs[name]


def update_result(*, matched_count, modified_count):
    """Crea un resultado mock de update_one()."""
    result = Mock()
    result.matched_count = matched_count
    result.modified_count = modified_count
    return result


@pytest.mark.asyncio
async def test_renew_upload_lease_success(repo_with_mocks):
    """Verifica filtro, actualización exacta y retorno True al hacer match."""
    repo = repo_with_mocks
    document_id = str(ObjectId())
    owner_id = "upload-owner-1"
    fencing_token = "upload-token-1"
    lease_until = datetime.now(timezone.utc) + timedelta(minutes=5)
    repo.collection.update_one.return_value = update_result(
        matched_count=1,
        modified_count=1,
    )

    before = datetime.now(timezone.utc)
    renewed = await repo.renew_upload_lease(
        document_id=document_id,
        upload_owner_id=owner_id,
        upload_fencing_token=fencing_token,
        upload_lease_until=lease_until,
    )
    after = datetime.now(timezone.utc)

    assert renewed is True
    repo.collection.update_one.assert_awaited_once()
    repo.collection.find_one.assert_not_called()

    call = repo.collection.update_one.call_args
    query = get_call_argument(call, 0, "filter")
    update = get_call_argument(call, 1, "update")
    assert query == {
        "_id": ObjectId(document_id),
        "status": DocumentStatus.UPLOADING.value,
        "upload_owner_id": owner_id,
        "upload_fencing_token": fencing_token,
        "upload_lease_until": {
            "$gte": query["upload_lease_until"]["$gte"],
            "$lt": lease_until,
        },
    }
    now = query["upload_lease_until"]["$gte"]
    assert before <= now <= after
    assert now.tzinfo == timezone.utc
    assert now.utcoffset() == timedelta(0)
    assert call.kwargs.get("upsert", False) is False

    assert set(update) == {"$set"}
    assert set(update["$set"]) == {"upload_lease_until", "updated_at"}
    assert update["$set"]["upload_lease_until"] == lease_until
    updated_at = update["$set"]["updated_at"]
    assert before <= updated_at <= after
    assert updated_at.tzinfo == timezone.utc
    assert updated_at.utcoffset() == timedelta(0)


@pytest.mark.asyncio
async def test_renew_upload_lease_denied_returns_false(repo_with_mocks):
    """Retorna False cuando update_one() no encuentra un documento elegible."""
    repo = repo_with_mocks
    repo.collection.update_one.return_value = update_result(
        matched_count=0,
        modified_count=0,
    )

    renewed = await repo.renew_upload_lease(
        document_id=str(ObjectId()),
        upload_owner_id="upload-owner-1",
        upload_fencing_token="upload-token-1",
        upload_lease_until=datetime.now(timezone.utc) + timedelta(minutes=5),
    )

    assert renewed is False
    repo.collection.update_one.assert_awaited_once()
    repo.collection.find_one.assert_not_called()


@pytest.mark.asyncio
async def test_renew_upload_lease_uses_matched_count_not_modified_count(
    repo_with_mocks,
):
    """Retorna True si matched_count es uno aunque modified_count sea cero."""
    repo = repo_with_mocks
    repo.collection.update_one.return_value = update_result(
        matched_count=1,
        modified_count=0,
    )

    renewed = await repo.renew_upload_lease(
        document_id=str(ObjectId()),
        upload_owner_id="upload-owner-1",
        upload_fencing_token="upload-token-1",
        upload_lease_until=datetime.now(timezone.utc) + timedelta(minutes=5),
    )

    assert renewed is True
    repo.collection.update_one.assert_awaited_once()
    repo.collection.find_one.assert_not_called()


@pytest.mark.asyncio
async def test_renew_upload_lease_translates_connectivity_error(repo_with_mocks):
    """Traduce la indisponibilidad de MongoDB y conserva la causa original."""
    repo = repo_with_mocks
    original_error = ServerSelectionTimeoutError("Connection timed out")
    repo.collection.update_one.side_effect = original_error

    with pytest.raises(DatabaseUnavailableError) as raised:
        await repo.renew_upload_lease(
            document_id=str(ObjectId()),
            upload_owner_id="upload-owner-1",
            upload_fencing_token="upload-token-1",
            upload_lease_until=datetime.now(timezone.utc) + timedelta(minutes=5),
        )

    assert raised.value.__cause__ is original_error
    repo.collection.update_one.assert_awaited_once()
    repo.collection.find_one.assert_not_called()


@pytest.mark.parametrize(
    "lease_until",
    [
        pytest.param(datetime.now(), id="naive-datetime"),
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
async def test_renew_upload_lease_rejects_invalid_expiration(
    repo_with_mocks,
    lease_until,
):
    """Rechaza fechas naive, no UTC o vencidas antes de consultar MongoDB."""
    repo = repo_with_mocks

    with pytest.raises(ValueError):
        await repo.renew_upload_lease(
            document_id=str(ObjectId()),
            upload_owner_id="upload-owner-1",
            upload_fencing_token="upload-token-1",
            upload_lease_until=lease_until,
        )

    repo.collection.update_one.assert_not_called()
    repo.collection.find_one.assert_not_called()


@pytest.mark.asyncio
async def test_renew_upload_lease_rejects_invalid_document_id(repo_with_mocks):
    """Rechaza un ID inválido antes de acceder a MongoDB."""
    repo = repo_with_mocks

    with pytest.raises(ValueError):
        await repo.renew_upload_lease(
            document_id="not-an-object-id",
            upload_owner_id="upload-owner-1",
            upload_fencing_token="upload-token-1",
            upload_lease_until=datetime.now(timezone.utc) + timedelta(minutes=5),
        )

    repo.collection.update_one.assert_not_called()
    repo.collection.find_one.assert_not_called()
