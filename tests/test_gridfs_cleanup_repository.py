"""Pruebas unitarias RED de borrado y verificación de ausencia en GridFS."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from bson import ObjectId
from pymongo.errors import AutoReconnect


def _cleanup_repository(db, gridfs_repository):
    # Import diferido: el repositorio de limpieza aún no existe en RED.
    from src.repository.gridfs_cleanup_repository import GridFSCleanupRepository

    return GridFSCleanupRepository(
        db,
        gridfs_repository=gridfs_repository,
    )


def _mock_gridfs_cleanup_dependencies():
    files = MagicMock()
    files.find_one = AsyncMock(return_value=None)
    chunks = MagicMock()
    chunks.find_one = AsyncMock(return_value=None)
    db = MagicMock()
    db.__getitem__.side_effect = {
        "fs.files": files,
        "fs.chunks": chunks,
    }.__getitem__
    gridfs_repository = MagicMock()
    gridfs_repository.delete = AsyncMock(return_value=True)
    return db, files, chunks, gridfs_repository


@pytest.mark.parametrize("delete_result", [True, False])
@pytest.mark.asyncio
async def test_delete_and_verify_absent_checks_exact_file_and_chunks_even_if_delete_returns_false(
    delete_result,
):
    db, files, chunks, gridfs = _mock_gridfs_cleanup_dependencies()
    gridfs.delete.return_value = delete_result
    repository = _cleanup_repository(db, gridfs)
    file_id = ObjectId()

    result = await repository.delete_and_verify_absent(file_id)

    assert result is True
    gridfs.delete.assert_awaited_once_with(file_id)
    files.find_one.assert_awaited_once_with({"_id": file_id}, {"_id": 1})
    chunks.find_one.assert_awaited_once_with(
        {"files_id": file_id},
        {"_id": 1},
    )


@pytest.mark.parametrize(
    ("remaining_file", "remaining_chunk"),
    [
        ({"_id": ObjectId()}, None),
        (None, {"_id": ObjectId(), "files_id": ObjectId()}),
        ({"_id": ObjectId()}, {"_id": ObjectId()}),
    ],
)
@pytest.mark.asyncio
async def test_delete_and_verify_absent_returns_false_while_any_gridfs_record_remains(
    remaining_file, remaining_chunk
):
    db, files, chunks, gridfs = _mock_gridfs_cleanup_dependencies()
    files.find_one.return_value = remaining_file
    chunks.find_one.return_value = remaining_chunk
    repository = _cleanup_repository(db, gridfs)
    file_id = ObjectId()

    result = await repository.delete_and_verify_absent(file_id)

    assert result is False
    gridfs.delete.assert_awaited_once_with(file_id)
    files.find_one.assert_awaited_once()
    chunks.find_one.assert_awaited_once()


@pytest.mark.parametrize(
    "error",
    [
        AutoReconnect("GridFS unavailable"),
        asyncio.CancelledError(),
    ],
)
@pytest.mark.asyncio
async def test_delete_and_verify_absent_propagates_delete_errors_without_claiming_absence(
    error,
):
    db, files, chunks, gridfs = _mock_gridfs_cleanup_dependencies()
    gridfs.delete.side_effect = error
    repository = _cleanup_repository(db, gridfs)

    with pytest.raises(type(error)):
        await repository.delete_and_verify_absent(ObjectId())

    gridfs.delete.assert_awaited_once()
    files.find_one.assert_not_awaited()
    chunks.find_one.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_and_verify_absent_rejects_non_objectid_before_storage_access():
    db, files, chunks, gridfs = _mock_gridfs_cleanup_dependencies()
    repository = _cleanup_repository(db, gridfs)

    with pytest.raises(ValueError):
        await repository.delete_and_verify_absent("not-an-object-id")

    gridfs.delete.assert_not_awaited()
    files.find_one.assert_not_awaited()
    chunks.find_one.assert_not_awaited()


@pytest.mark.parametrize(
    ("collection_name", "error"),
    [
        ("fs.files", AutoReconnect("file verification unavailable")),
        ("fs.chunks", asyncio.CancelledError()),
    ],
)
@pytest.mark.asyncio
async def test_delete_and_verify_absent_propagates_verification_errors(
    collection_name, error
):
    db, files, chunks, gridfs = _mock_gridfs_cleanup_dependencies()
    verification_collection = files if collection_name == "fs.files" else chunks
    verification_collection.find_one.side_effect = error
    repository = _cleanup_repository(db, gridfs)

    with pytest.raises(type(error)) as raised:
        await repository.delete_and_verify_absent(ObjectId())

    assert raised.value is error
    gridfs.delete.assert_awaited_once()
    if collection_name == "fs.files":
        files.find_one.assert_awaited_once()
        chunks.find_one.assert_not_awaited()
    else:
        files.find_one.assert_awaited_once()
        chunks.find_one.assert_awaited_once()
