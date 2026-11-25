
"""Pruebas de integración real contra MongoDB para GridFSRepository."""

from bson import ObjectId

import pytest
import pytest_asyncio

from src.database.mongodb import MongoDB
from src.repository.gridfs_repository import GridFSRepository
from src.repository.document_repository import DocumentRepository
from src.services.document_service import DocumentService


EXPECTED_WRITE_CONCERN = {"w": 1, "j": True}


@pytest_asyncio.fixture(scope="function")
async def _isolated_testing_db():
    """Conecta a MongoDB de testing y garantiza la desconexión."""
    testing_db = MongoDB.for_testing()

    try:
        await testing_db.connect()
        assert testing_db.db is not None
        yield testing_db
    finally:
        await testing_db.disconnect()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gridfs_save_stat_read_delete_real_mongodb(
    _isolated_testing_db,
):
    """Verifica el ciclo completo de almacenamiento en GridFS."""
    db = _isolated_testing_db.db
    gridfs_repo = GridFSRepository(db)

    test_data = b"contenido de prueba para GridFS integration test"
    filename = "integracion_test.txt"
    file_id = ObjectId()
    metadata = {"checksum": "abc123"}

    try:
        # Guardar archivo.
        result_id = await gridfs_repo.save(
            file_id, filename, test_data, metadata
        )
        assert result_id == file_id

        # Verificar información y metadata.
        info = await gridfs_repo.stat(file_id)

        assert info is not None
        assert info.filename == filename
        assert info.length == len(test_data)
        assert info.content_type is None
        assert info.metadata == metadata

        # Verificar integridad del contenido.
        recovered = await gridfs_repo.read(file_id)
        assert recovered == test_data

        # Verificar persistencia real en MongoDB.
        file_doc = await db["fs.files"].find_one(
            {"_id": file_id}
        )
        chunks_doc = await db["fs.chunks"].find_one(
            {"files_id": file_id}
        )

        assert file_doc is not None
        assert chunks_doc is not None

        # Eliminar archivo.
        deleted = await gridfs_repo.delete(file_id)
        assert deleted is True

        # Verificar que ya no existe.
        assert await gridfs_repo.stat(file_id) is None

        # Verificar eliminación idempotente.
        deleted_again = await gridfs_repo.delete(file_id)
        assert deleted_again is False

        # Verificar ausencia de registros huérfanos.
        file_doc_after = await db["fs.files"].find_one(
            {"_id": file_id}
        )
        chunks_doc_after = await db["fs.chunks"].find_one(
            {"files_id": file_id}
        )

        assert file_doc_after is None
        assert chunks_doc_after is None

    finally:
        # Limpiar exclusivamente los datos de esta prueba.
        await db["fs.files"].delete_one({"_id": file_id})
        await db["fs.chunks"].delete_many({"files_id": file_id})


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gridfs_write_concern_real_mongodb(
    _isolated_testing_db,
):
    """Verifica el Write Concern efectivo de MongoDB y GridFS."""
    testing_db = _isolated_testing_db
    db = testing_db.db

    assert (
        testing_db.client.write_concern.document
        == EXPECTED_WRITE_CONCERN
    )

    assert (
        db.write_concern.document
        == EXPECTED_WRITE_CONCERN
    )

    assert (
        db["fs.files"].write_concern.document
        == EXPECTED_WRITE_CONCERN
    )

    assert (
        db["fs.chunks"].write_concern.document
        == EXPECTED_WRITE_CONCERN
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gridfs_service_integration_real_mongodb(
    _isolated_testing_db,
):
    """Verifica la integración entre DocumentService y GridFS."""
    db = _isolated_testing_db.db

    repository = DocumentRepository(db)
    service = DocumentService(repository=repository)
    gridfs_repo = GridFSRepository(db)

    test_data = b"datos de prueba de servicio integration"
    filename = "servicio_integracion.txt"
    file_id = ObjectId()
    metadata = {"checksum": "def456"}

    try:
        # Guardar mediante DocumentService.
        stored_id = await service.save_original_pdf(
            file_id=file_id,
            filename=filename,
            data=test_data,
            metadata=metadata,
        )

        assert stored_id == file_id

        # Verificar información almacenada.
        info = await gridfs_repo.stat(file_id)

        assert info is not None
        assert info.filename == filename
        assert info.length == len(test_data)
        assert info.metadata == metadata

        # Verificar integridad del contenido.
        recovered = await gridfs_repo.read(file_id)
        assert recovered == test_data

        # Eliminar archivo.
        deleted = await gridfs_repo.delete(file_id)
        assert deleted is True

        # Verificar eliminación efectiva.
        file_doc = await db["fs.files"].find_one(
            {"_id": file_id}
        )
        assert file_doc is None

    finally:
        # Limpiar exclusivamente los datos de esta prueba.
        await db["fs.files"].delete_one({"_id": file_id})
        await db["fs.chunks"].delete_many({"files_id": file_id})
