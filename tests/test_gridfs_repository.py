"""Tests unitarios para GridFSRepository (sin MongoDB real)."""

import asyncio

import pytest
from unittest.mock import AsyncMock, Mock, patch

from bson import ObjectId

from gridfs.errors import NoFile

from src.repository.gridfs_repository import GridFSRepository, GridFSFileInfo


# ---- Fixture que crea repo con bucket y grid_in mock por defecto ----

@pytest.fixture
def repo_with_mocks():
    """Crea GridFSRepository con bucket y grid_in mock inyectados.

    Retorna: (repo, bucket, grid_in)

    Los mocks reflejan la API real de Motor 3.7.1:
    - open_upload_stream_with_id retorna GridIn síncronamente (sin await),
      por lo que usa Mock y assert_called_once_with().
    - grid_in.write/close/abort son métodos awaitable (usar AsyncMock).
    - open_download_stream es una coroutine (requiere await).
    - delete es un método async en AsyncIOMotorGridFSBucket.
    """
    repo = GridFSRepository(object())  # db dummy

    bucket = AsyncMock()
    grid_in = AsyncMock()

    # open_upload_stream_with_id retorna grid_in síncronamente.
    # Usamos Mock (no AsyncMock) porque el código cliente NO usa await
    # en esta llamada (adaptador corregido).
    bucket.open_upload_stream_with_id = Mock(return_value=grid_in)

    # open_download_stream retorna None por defecto (los tests lo sobreescriben)
    bucket.open_download_stream = AsyncMock(return_value=None)

    # delete hace nothing por defecto
    bucket.delete = AsyncMock()

    # Monkeypatch _bucket en la instancia repo
    repo._bucket = lambda: bucket

    return repo, bucket, grid_in


# ---- Tests save() ----


@pytest.mark.asyncio
async def test_save_success(repo_with_mocks):
    """1. save(): escritura y cierre exitosos.

    Verifica que write() se llama antes de close() y que abort() no se ejecuta.
    """
    repo, bucket, grid_in = repo_with_mocks
    file_id = ObjectId()
    data = b"contenido de prueba"

    result = await repo.save(file_id, "test.pdf", data)

    # open_upload_stream_with_id usa Mock (no await), por assert_called_once_with
    bucket.open_upload_stream_with_id.assert_called_once_with(
        file_id, "test.pdf", metadata=None
    )
    # write, close, abort son awaitable, por assert_awaited_once_with
    grid_in.write.assert_awaited_once_with(data)
    grid_in.close.assert_awaited_once()
    grid_in.abort.assert_not_awaited()
    assert result == file_id


@pytest.mark.asyncio
async def test_save_write_error_attempts_abort(repo_with_mocks):
    """2. save(): error durante write(), verificando que intenta abort()."""
    repo, bucket, grid_in = repo_with_mocks
    file_id = ObjectId()
    data = b"datos"

    # Hacemos que write falle
    grid_in.write = AsyncMock(side_effect=RuntimeError("write failed"))
    # close y abort por defecto son AsyncMock() sin side_effect (retornan None)

    with pytest.raises(RuntimeError, match="write failed"):
        await repo.save(file_id, "test.pdf", data)

    # abort() debería haber sido intentado
    grid_in.abort.assert_awaited_once()
    # close() no debería haber sido llamado (write falló antes)
    grid_in.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_save_close_error_attempts_abort(repo_with_mocks):
    """3. save(): error durante close(), verificando que intenta abort()."""
    repo, bucket, grid_in = repo_with_mocks
    file_id = ObjectId()
    data = b"datos"

    # write exitoso
    grid_in.write = AsyncMock(return_value=None)
    # close falla
    grid_in.close = AsyncMock(side_effect=RuntimeError("close failed"))
    grid_in.abort = AsyncMock()

    with pytest.raises(RuntimeError, match="close failed"):
        await repo.save(file_id, "test.pdf", data)

    # abort() fue intentado después del fallo de close
    grid_in.abort.assert_awaited_once()
    grid_in.write.assert_awaited_once_with(data)


@pytest.mark.asyncio
async def test_save_cancelled_error_write(repo_with_mocks):
    """4. save(): cancelación durante write()."""
    repo, bucket, grid_in = repo_with_mocks
    file_id = ObjectId()
    data = b"datos"

    grid_in.write = AsyncMock(side_effect=asyncio.CancelledError())
    grid_in.close = AsyncMock()
    grid_in.abort = AsyncMock()

    with pytest.raises(asyncio.CancelledError):
        await repo.save(file_id, "test.pdf", data)

    # abort() debería ser intentado antes de propagar la cancelación
    grid_in.abort.assert_awaited_once()


@pytest.mark.asyncio
async def test_save_cancelled_error_close(repo_with_mocks):
    """4. save(): cancelación durante close()."""
    repo, bucket, grid_in = repo_with_mocks
    file_id = ObjectId()
    data = b"datos"

    grid_in.write = AsyncMock(return_value=None)
    grid_in.close = AsyncMock(side_effect=asyncio.CancelledError())
    grid_in.abort = AsyncMock()

    with pytest.raises(asyncio.CancelledError):
        await repo.save(file_id, "test.pdf", data)

    # abort() debería ser intentado después del fallo de close
    grid_in.abort.assert_awaited_once()


@pytest.mark.asyncio
async def test_save_abort_failure_preserves_original(repo_with_mocks, caplog):
    """5. save(): fallo de abort(), verificando que se preserva la excepción original."""
    repo, bucket, grid_in = repo_with_mocks
    file_id = ObjectId()
    data = b"datos"

    grid_in.write = AsyncMock(side_effect=RuntimeError("write failed"))
    grid_in.abort = AsyncMock(side_effect=RuntimeError("abort failed"))
    grid_in.close = AsyncMock()

    # Debería lanzar RuntimeError original ("write failed"), no "abort failed"
    with pytest.raises(RuntimeError, match="write failed"):
        await repo.save(file_id, "test.pdf", data)


# ---- Tests stat() ----



@pytest.mark.asyncio
async def test_stat_existing_file(repo_with_mocks):
    """6. stat(): archivo existente."""
    from motor.motor_asyncio import AsyncIOMotorGridOut

    repo, bucket, grid_in = repo_with_mocks

    # Sobreescribimos _bucket para retornar un bucket que devuelve un GridOut mock
    grid_out = AsyncMock()
    grid_out.filename = "test.pdf"
    grid_out.length = 1024
    grid_out.metadata = None
    grid_out.upload_date = None
    grid_out.content_type = "application/pdf"

    bucket.open_download_stream = AsyncMock(return_value=grid_out)

    result = await repo.stat(ObjectId())

    assert isinstance(result, GridFSFileInfo)
    assert result.filename == "test.pdf"
    assert result.length == 1024
    assert result.content_type == "application/pdf"


@pytest.mark.asyncio
async def test_stat_nonexistent_file(repo_with_mocks):
    """6. stat(): archivo inexistente (devuelve None)."""
    repo, bucket, grid_in = repo_with_mocks

    # open_download_stream lanza NoFile
    bucket.open_download_stream = AsyncMock(side_effect=NoFile())

    result = await repo.stat(ObjectId())

    assert result is None


# ---- Tests read() ----



@pytest.mark.asyncio
async def test_read_success(repo_with_mocks):
    """7. read(): lectura exitosa."""
    repo, bucket, grid_in = repo_with_mocks

    grid_out = AsyncMock()
    expected_data = b"contenido leido"
    grid_out.read = AsyncMock(return_value=expected_data)

    bucket.open_download_stream = AsyncMock(return_value=grid_out)

    result = await repo.read(ObjectId())

    assert result == expected_data


@pytest.mark.asyncio
async def test_read_nonexistent_file(repo_with_mocks):
    """7. read(): propagación de NoFile."""
    repo, bucket, grid_in = repo_with_mocks

    bucket.open_download_stream = AsyncMock(side_effect=NoFile())

    with pytest.raises(NoFile):
        await repo.read(ObjectId())


# ---- Tests delete() ----



@pytest.mark.asyncio
async def test_delete_existing_file(repo_with_mocks):
    """8. delete(): eliminación exitosa."""
    repo, bucket, grid_in = repo_with_mocks

    # delete already mocked as AsyncMock() doing nothing
    result = await repo.delete(ObjectId())

    assert result is True
    bucket.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_nonexistent_file(repo_with_mocks):
    """8. delete(): archivo inexistente (devuelve False)."""
    repo, bucket, grid_in = repo_with_mocks

    # Hacemos que delete lance NoFile
    bucket.delete = AsyncMock(side_effect=NoFile())

    result = await repo.delete(ObjectId())

    assert result is False