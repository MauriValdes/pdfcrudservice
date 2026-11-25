"""Tests unitarios para finalize_upload() (fase RED)."""

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
async def test_finalize_upload_success(repo_with_mocks):
    """Verifica filtro, actualización exacta y retorno True al hacer match."""
    repo = repo_with_mocks
    document_id = str(ObjectId())
    owner_id = str(ObjectId())
    fencing_token = str(ObjectId())
    file_id = ObjectId()
    repo.collection.update_one.return_value = update_result(
        matched_count=1,
        modified_count=1,
    )

    before = datetime.now(timezone.utc)
    finalized = await repo.finalize_upload(
        document_id=document_id,
        upload_owner_id=owner_id,
        upload_fencing_token=fencing_token,
        file_id=file_id,
    )
    after = datetime.now(timezone.utc)

    assert finalized is True
    repo.collection.update_one.assert_awaited_once()
    repo.collection.find_one.assert_not_called()

    call = repo.collection.update_one.call_args
    query = get_call_argument(call, 0, "filter")
    update = get_call_argument(call, 1, "update")
    assert call.kwargs["upsert"] is False

    assert query["_id"] == ObjectId(document_id)
    assert query["status"] == DocumentStatus.UPLOADING.value
    assert query["upload_owner_id"] == owner_id
    assert query["upload_fencing_token"] == fencing_token
    now = query["upload_lease_until"]["$gte"]
    assert query["upload_lease_until"] == {"$gte": now}
    assert query["$or"] == [
        {"file_id": {"$exists": False}},
        {"file_id": None},
    ]
    assert before <= now <= after
    assert now.tzinfo == timezone.utc
    assert now.utcoffset() == timedelta(0)

    assert set(update) == {"$set"}
    assert set(update["$set"]) == {"file_id", "status", "updated_at"}
    assert update["$set"]["file_id"] is file_id
    assert isinstance(update["$set"]["file_id"], ObjectId)
    assert update["$set"]["status"] == DocumentStatus.PENDING.value
    updated_at = update["$set"]["updated_at"]
    assert before <= updated_at <= after
    assert updated_at.tzinfo == timezone.utc
    assert updated_at.utcoffset() == timedelta(0)


@pytest.mark.asyncio
async def test_finalize_upload_denied_returns_false(repo_with_mocks):
    """Retorna False cuando update_one() no encuentra una reserva elegible."""
    repo = repo_with_mocks
    repo.collection.update_one.return_value = update_result(
        matched_count=0,
        modified_count=0,
    )

    finalized = await repo.finalize_upload(
        document_id=str(ObjectId()),
        upload_owner_id=str(ObjectId()),
        upload_fencing_token=str(ObjectId()),
        file_id=ObjectId(),
    )

    assert finalized is False
    repo.collection.update_one.assert_awaited_once()
    repo.collection.find_one.assert_not_called()


@pytest.mark.asyncio
async def test_finalize_upload_uses_matched_count_not_modified_count(
    repo_with_mocks,
):
    """Retorna True si matched_count es uno aunque modified_count sea cero."""
    repo = repo_with_mocks
    repo.collection.update_one.return_value = update_result(
        matched_count=1,
        modified_count=0,
    )

    finalized = await repo.finalize_upload(
        document_id=str(ObjectId()),
        upload_owner_id=str(ObjectId()),
        upload_fencing_token=str(ObjectId()),
        file_id=ObjectId(),
    )

    assert finalized is True
    repo.collection.update_one.assert_awaited_once()
    repo.collection.find_one.assert_not_called()


@pytest.mark.asyncio
async def test_finalize_upload_translates_connectivity_error(repo_with_mocks):
    """Traduce indisponibilidad de MongoDB y conserva la causa original."""
    repo = repo_with_mocks
    original_error = ServerSelectionTimeoutError("Connection timed out")
    repo.collection.update_one.side_effect = original_error

    with pytest.raises(DatabaseUnavailableError) as raised:
        await repo.finalize_upload(
            document_id=str(ObjectId()),
            upload_owner_id=str(ObjectId()),
            upload_fencing_token=str(ObjectId()),
            file_id=ObjectId(),
        )

    assert raised.value.__cause__ is original_error
    repo.collection.update_one.assert_awaited_once()
    repo.collection.find_one.assert_not_called()


@pytest.mark.asyncio
async def test_finalize_upload_rejects_invalid_document_id(repo_with_mocks):
    """Rechaza un document_id inválido antes de acceder a MongoDB."""
    repo = repo_with_mocks

    with pytest.raises(ValueError):
        await repo.finalize_upload(
            document_id="not-an-object-id",
            upload_owner_id=str(ObjectId()),
            upload_fencing_token=str(ObjectId()),
            file_id=ObjectId(),
        )

    repo.collection.update_one.assert_not_called()
    repo.collection.find_one.assert_not_called()


@pytest.mark.parametrize(
    "file_id",
    [
        pytest.param(str(ObjectId()), id="object-id-string"),
        pytest.param(None, id="none"),
        pytest.param(123, id="integer"),
    ],
)
@pytest.mark.asyncio
async def test_finalize_upload_rejects_invalid_file_id(repo_with_mocks, file_id):
    """Rechaza file_id que no sea ObjectId antes de acceder a MongoDB."""
    repo = repo_with_mocks

    with pytest.raises(ValueError):
        await repo.finalize_upload(
            document_id=str(ObjectId()),
            upload_owner_id=str(ObjectId()),
            upload_fencing_token=str(ObjectId()),
            file_id=file_id,
        )

    repo.collection.update_one.assert_not_called()
    repo.collection.find_one.assert_not_called()
