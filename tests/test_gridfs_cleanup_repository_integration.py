"""Pruebas de integración RED para limpieza GridFS con IDs propios."""

import pytest
import pytest_asyncio
from bson import ObjectId

from src.config.config import MONGO_TEST_DATABASE
from src.database.mongodb import MongoDB
from src.repository.gridfs_repository import GridFSRepository


@pytest_asyncio.fixture
async def cleanup_gridfs_context():
    manager = MongoDB.for_testing()
    owned_file_ids = []
    try:
        await manager.connect()
        if (
            manager._testing is not True
            or manager.db is None
            or manager.db.name != MONGO_TEST_DATABASE
        ):
            raise RuntimeError(
                "El fixture GridFS solo puede usar pdf_database_test."
            )
        yield manager.db, owned_file_ids
    finally:
        try:
            if manager.db is not None:
                if (
                    manager._testing is not True
                    or manager.db.name != MONGO_TEST_DATABASE
                ):
                    raise RuntimeError(
                        "Se bloqueó la limpieza fuera de pdf_database_test."
                    )
                for file_id in reversed(owned_file_ids):
                    await manager.db["fs.files"].delete_one({"_id": file_id})
                    await manager.db["fs.chunks"].delete_many(
                        {"files_id": file_id}
                    )
        finally:
            await manager.disconnect()


def _small_pdf_bytes(label):
    return (
        b"%PDF-1.4\n"
        + f"% owned-cleanup-fixture:{label}\n".encode("ascii")
        + b"1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF\n"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cleanup_repository_deletes_only_the_exact_file_and_is_repeatable(
    cleanup_gridfs_context,
):
    db, owned_file_ids = cleanup_gridfs_context
    target_id = ObjectId()
    unrelated_id = ObjectId()
    owned_file_ids.extend([target_id, unrelated_id])
    gridfs = GridFSRepository(db)
    target_bytes = _small_pdf_bytes("target")
    unrelated_bytes = _small_pdf_bytes("unrelated")
    await gridfs.save(target_id, "cleanup-target.pdf", target_bytes)
    await gridfs.save(unrelated_id, "cleanup-unrelated.pdf", unrelated_bytes)

    # Import diferido: este repositorio aún no existe durante RED.
    from src.repository.gridfs_cleanup_repository import GridFSCleanupRepository

    repository = GridFSCleanupRepository(db, gridfs_repository=gridfs)

    assert await repository.delete_and_verify_absent(target_id) is True
    assert await db["fs.files"].find_one({"_id": target_id}) is None
    assert await db["fs.chunks"].find_one({"files_id": target_id}) is None
    assert await db["fs.files"].find_one({"_id": unrelated_id}) is not None
    assert await gridfs.read(unrelated_id) == unrelated_bytes

    # Segunda ejecución: delete devuelve False, pero ambos recursos siguen ausentes.
    assert await repository.delete_and_verify_absent(target_id) is True
    assert await db["fs.files"].find_one({"_id": target_id}) is None
    assert await db["fs.chunks"].find_one({"files_id": target_id}) is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cleanup_repository_recovers_chunks_when_gridfs_metadata_is_missing(
    cleanup_gridfs_context,
):
    db, owned_file_ids = cleanup_gridfs_context
    file_id = ObjectId()
    owned_file_ids.append(file_id)
    gridfs = GridFSRepository(db)
    await gridfs.save(file_id, "cleanup-orphan-chunks.pdf", _small_pdf_bytes("orphan"))
    assert await db["fs.chunks"].find_one({"files_id": file_id}) is not None
    assert (await db["fs.files"].delete_one({"_id": file_id})).deleted_count == 1

    from src.repository.gridfs_cleanup_repository import GridFSCleanupRepository

    repository = GridFSCleanupRepository(db, gridfs_repository=gridfs)

    assert await repository.delete_and_verify_absent(file_id) is True
    assert await db["fs.files"].find_one({"_id": file_id}) is None
    assert await db["fs.chunks"].find_one({"files_id": file_id}) is None


@pytest.mark.parametrize("residual_kind", ["file", "chunk"])
@pytest.mark.integration
@pytest.mark.asyncio
async def test_cleanup_repository_reports_false_if_storage_still_contains_target(
    cleanup_gridfs_context, residual_kind
):
    db, owned_file_ids = cleanup_gridfs_context
    file_id = ObjectId()
    owned_file_ids.append(file_id)
    gridfs = GridFSRepository(db)
    await gridfs.save(file_id, "cleanup-residual.pdf", _small_pdf_bytes("residual"))
    if residual_kind == "chunk":
        assert (await db["fs.files"].delete_one({"_id": file_id})).deleted_count == 1

    class NoDelete:
        async def delete(self, requested_file_id):
            assert requested_file_id == file_id
            return False

    from src.repository.gridfs_cleanup_repository import GridFSCleanupRepository

    repository = GridFSCleanupRepository(db, gridfs_repository=NoDelete())

    assert await repository.delete_and_verify_absent(file_id) is False
    if residual_kind == "file":
        assert await db["fs.files"].find_one({"_id": file_id}) is not None
    assert await db["fs.chunks"].find_one({"files_id": file_id}) is not None
