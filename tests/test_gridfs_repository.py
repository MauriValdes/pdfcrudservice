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
    grid_in.chunk_size = 255 * 1024

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
async def test_save_write_error_does_not_abort_uncertain_upload(repo_with_mocks):
    """Un error genérico de write no permite asumir que no hubo efecto remoto."""
    repo, bucket, grid_in = repo_with_mocks
    file_id = ObjectId()
    data = b"datos"

    grid_in.write = AsyncMock(side_effect=RuntimeError("write failed"))

    with pytest.raises(RuntimeError, match="write failed"):
        await repo.save(file_id, "test.pdf", data)

    grid_in.abort.assert_not_awaited()
    grid_in.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_save_close_error_does_not_abort_uncertain_file(repo_with_mocks):
    """Un error de close puede ocurrir después de persistir fs.files."""
    repo, bucket, grid_in = repo_with_mocks
    file_id = ObjectId()
    data = b"datos"

    grid_in.write = AsyncMock(return_value=None)
    grid_in.close = AsyncMock(side_effect=RuntimeError("close failed"))

    with pytest.raises(RuntimeError, match="close failed"):
        await repo.save(file_id, "test.pdf", data)

    grid_in.abort.assert_not_awaited()
    grid_in.write.assert_awaited_once_with(data)


@pytest.mark.asyncio
async def test_save_independently_cancelled_write_does_not_abort(repo_with_mocks):
    """Un future de write cancelado independientemente es de resultado incierto."""
    repo, bucket, grid_in = repo_with_mocks
    write_future = asyncio.get_running_loop().create_future()
    write_started = asyncio.Event()

    def write(block):
        write_started.set()
        return write_future

    grid_in.write = Mock(side_effect=write)

    task = asyncio.create_task(repo.save(ObjectId(), "test.pdf", b"datos"))
    try:
        await _wait_for_event_or_task(write_started, task)
        write_future.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not write_future.done():
            write_future.cancel()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert write_future.cancelled()
    grid_in.abort.assert_not_awaited()
    grid_in.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_save_cancelled_error_close(repo_with_mocks):
    """No abortar después de una cancelación durante close()."""
    repo, bucket, grid_in = repo_with_mocks

    grid_in.write = AsyncMock(return_value=None)
    grid_in.close = AsyncMock(side_effect=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await repo.save(ObjectId(), "test.pdf", b"datos")

    grid_in.abort.assert_not_awaited()
    grid_in.write.assert_awaited_once_with(b"datos")


@pytest.mark.asyncio
async def test_save_abort_failure_preserves_original(repo_with_mocks, caplog):
    """Un abort fallido no reemplaza el error primario de interrupción."""
    repo, bucket, grid_in = repo_with_mocks
    callback_error = RuntimeError("callback failed")
    grid_in.chunk_size = 8
    grid_in.write = AsyncMock(return_value=None)
    grid_in.abort = AsyncMock(side_effect=RuntimeError("abort failed"))
    should_continue = Mock(side_effect=[True, True, callback_error])

    with pytest.raises(RuntimeError) as raised:
        await repo.save(
            ObjectId(),
            "test.pdf",
            b"datos",
            should_continue=should_continue,
        )

    assert raised.value is callback_error
    grid_in.abort.assert_awaited_once()
    grid_in.close.assert_not_awaited()
    assert "No se pudo abortar la carga GridFS" in caplog.text


def _completed_future(result=None):
    future = asyncio.get_running_loop().create_future()
    future.set_result(result)
    return future


async def _wait_for_event_or_task(event, task):
    """Espera al punto de control o detecta fallo previo sin bloquearse."""
    event_task = asyncio.create_task(event.wait())
    try:
        done, _ = await asyncio.wait_for(
            asyncio.wait(
                {event_task, task},
                return_when=asyncio.FIRST_COMPLETED,
            ),
            timeout=5,
        )
        if task in done:
            await task
            pytest.fail("save terminó antes del punto de control esperado")
        await event_task
    finally:
        if not event_task.done():
            event_task.cancel()
        await asyncio.gather(event_task, return_exceptions=True)


def _assert_interruption_exception(raised):
    # No importar la excepción nueva en RED: el módulo debe poder recolectarse.
    assert raised.type.__name__ == "GridFSUploadInterruptedError"


@pytest.mark.asyncio
async def test_save_writes_gridfs_sized_blocks_and_returns_after_close(repo_with_mocks):
    repo, bucket, grid_in = repo_with_mocks
    file_id = ObjectId()
    data = b"abcdefghij"
    grid_in.chunk_size = 4
    operation_order = []
    write_futures = []

    def write(block):
        operation_order.append(("write-start", block))
        future = _completed_future()
        write_futures.append(future)
        return future

    async def close():
        assert all(future.done() for future in write_futures)
        operation_order.append(("close", None))

    grid_in.write = Mock(side_effect=write)
    grid_in.close = AsyncMock(side_effect=close)

    result = await repo.save(file_id, "blocks.pdf", data, {"tag": "value"})

    bucket.open_upload_stream_with_id.assert_called_once_with(
        file_id,
        "blocks.pdf",
        metadata={"tag": "value"},
    )
    assert [entry[1] for entry in operation_order[:-1]] == [
        b"abcd",
        b"efgh",
        b"ij",
    ]
    assert operation_order[-1] == ("close", None)
    assert result == file_id
    grid_in.abort.assert_not_awaited()


@pytest.mark.asyncio
async def test_save_interruption_before_open_does_not_open_or_abort(repo_with_mocks):
    repo, bucket, grid_in = repo_with_mocks
    should_continue = Mock(return_value=False)

    with pytest.raises(Exception) as raised:
        await repo.save(
            ObjectId(),
            "interrupted.pdf",
            b"data",
            should_continue=should_continue,
        )

    _assert_interruption_exception(raised)
    should_continue.assert_called_once_with()
    bucket.open_upload_stream_with_id.assert_not_called()
    grid_in.abort.assert_not_awaited()


@pytest.mark.asyncio
async def test_save_interruption_while_write_pending_drains_then_aborts(
    repo_with_mocks,
):
    repo, bucket, grid_in = repo_with_mocks
    grid_in.chunk_size = 4
    write_started = asyncio.Event()
    write_future = asyncio.get_running_loop().create_future()
    abort_saw_write_done = []
    continuation = {"allowed": True}
    should_continue = Mock(side_effect=lambda: continuation["allowed"])

    def write(block):
        assert block == b"abcd"
        write_started.set()
        return write_future

    async def abort():
        abort_saw_write_done.append(write_future.done())

    grid_in.write = Mock(side_effect=write)
    grid_in.abort = AsyncMock(side_effect=abort)
    task = asyncio.create_task(
        repo.save(
            ObjectId(),
            "interrupted.pdf",
            b"abcdefgh",
            should_continue=should_continue,
        )
    )
    try:
        await _wait_for_event_or_task(write_started, task)
        assert not task.done()
        assert grid_in.abort.await_count == 0
        continuation["allowed"] = False
        write_future.set_result(None)
        with pytest.raises(Exception) as raised:
            await task
        _assert_interruption_exception(raised)
    finally:
        if not write_future.done():
            write_future.set_result(None)
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert grid_in.write.call_count == 1
    assert abort_saw_write_done == [True]
    grid_in.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_save_interruption_before_close_aborts_completed_writes(repo_with_mocks):
    repo, bucket, grid_in = repo_with_mocks
    grid_in.chunk_size = 8
    should_continue = Mock(side_effect=[True, True, False])
    grid_in.write = Mock(return_value=_completed_future())

    with pytest.raises(Exception) as raised:
        await repo.save(
            ObjectId(),
            "interrupted.pdf",
            b"data",
            should_continue=should_continue,
        )

    _assert_interruption_exception(raised)
    assert grid_in.write.call_count == 1
    grid_in.abort.assert_awaited_once()
    grid_in.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_save_callback_exception_before_close_aborts_and_preserves_it(
    repo_with_mocks,
):
    repo, bucket, grid_in = repo_with_mocks
    callback_error = RuntimeError("ownership check failed")
    grid_in.chunk_size = 8
    grid_in.write = Mock(return_value=_completed_future())
    should_continue = Mock(side_effect=[True, True, callback_error])

    with pytest.raises(RuntimeError) as raised:
        await repo.save(
            ObjectId(),
            "callback-error.pdf",
            b"data",
            should_continue=should_continue,
        )

    assert raised.value is callback_error
    grid_in.abort.assert_awaited_once()
    grid_in.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_save_interruption_during_close_waits_without_abort(repo_with_mocks):
    repo, bucket, grid_in = repo_with_mocks
    close_started = asyncio.Event()
    close_future = asyncio.get_running_loop().create_future()
    continuation = {"allowed": True}
    should_continue = Mock(side_effect=lambda: continuation["allowed"])
    grid_in.chunk_size = 8
    grid_in.write = Mock(return_value=_completed_future())

    def close():
        close_started.set()
        return close_future

    grid_in.close = Mock(side_effect=close)
    task = asyncio.create_task(
        repo.save(
            ObjectId(),
            "close-interrupted.pdf",
            b"data",
            should_continue=should_continue,
        )
    )
    try:
        await _wait_for_event_or_task(close_started, task)
        assert not task.done()
        grid_in.abort.assert_not_awaited()
        continuation["allowed"] = False
        close_future.set_result(None)
        with pytest.raises(Exception) as raised:
            await task
        _assert_interruption_exception(raised)
    finally:
        if not close_future.done():
            close_future.set_result(None)
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    grid_in.abort.assert_not_awaited()


@pytest.mark.asyncio
async def test_save_callback_exception_after_close_preserves_file(repo_with_mocks):
    repo, bucket, grid_in = repo_with_mocks
    callback_error = RuntimeError("lease lost after close")
    grid_in.chunk_size = 8
    grid_in.write = Mock(return_value=_completed_future())
    grid_in.close = Mock(return_value=_completed_future())
    should_continue = Mock(side_effect=[True, True, True, callback_error])

    with pytest.raises(RuntimeError) as raised:
        await repo.save(
            ObjectId(),
            "closed-file.pdf",
            b"data",
            should_continue=should_continue,
        )

    assert raised.value is callback_error
    grid_in.close.assert_called_once_with()
    grid_in.abort.assert_not_awaited()


@pytest.mark.asyncio
async def test_save_external_cancellation_during_write_drains_before_abort(
    repo_with_mocks,
):
    repo, bucket, grid_in = repo_with_mocks
    write_started = asyncio.Event()
    write_future = asyncio.get_running_loop().create_future()
    abort_saw_write_done = []
    grid_in.chunk_size = 8

    def write(block):
        write_started.set()
        return write_future

    async def abort():
        abort_saw_write_done.append(write_future.done())

    grid_in.write = Mock(side_effect=write)
    grid_in.abort = AsyncMock(side_effect=abort)
    task = asyncio.create_task(repo.save(ObjectId(), "cancel.pdf", b"data"))
    loop = asyncio.get_running_loop()
    first_cancel_delivered = asyncio.Event()
    second_cancel_delivered = asyncio.Event()
    try:
        await _wait_for_event_or_task(write_started, task)
        task.cancel()
        loop.call_soon(first_cancel_delivered.set)
        await first_cancel_delivered.wait()
        assert not task.done()
        assert not write_future.cancelled()
        grid_in.abort.assert_not_awaited()

        task.cancel()
        loop.call_soon(second_cancel_delivered.set)
        await second_cancel_delivered.wait()
        assert not task.done()
        assert not write_future.cancelled()
        grid_in.abort.assert_not_awaited()

        write_future.set_result(None)
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not write_future.done():
            write_future.set_result(None)
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert abort_saw_write_done == [True]
    grid_in.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_save_cancellation_during_abort_waits_for_cleanup(repo_with_mocks):
    repo, bucket, grid_in = repo_with_mocks
    abort_started = asyncio.Event()
    abort_future = asyncio.get_running_loop().create_future()
    callback_error = RuntimeError("ownership check failed")
    grid_in.chunk_size = 8
    grid_in.write = Mock(return_value=_completed_future())
    should_continue = Mock(side_effect=[True, True, callback_error])

    def abort():
        abort_started.set()
        return abort_future

    grid_in.abort = Mock(side_effect=abort)
    task = asyncio.create_task(
        repo.save(
            ObjectId(),
            "abort-cancel.pdf",
            b"data",
            should_continue=should_continue,
        )
    )
    try:
        await _wait_for_event_or_task(abort_started, task)
        task.cancel()
        cancellation_delivered = asyncio.Event()
        asyncio.get_running_loop().call_soon(cancellation_delivered.set)
        await cancellation_delivered.wait()
        assert not task.done()
        assert not abort_future.cancelled()

        abort_future.set_result(None)
        with pytest.raises(RuntimeError) as raised:
            await task
        assert raised.value is callback_error
    finally:
        if not abort_future.done():
            abort_future.set_result(None)
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    grid_in.abort.assert_called_once_with()


@pytest.mark.asyncio
async def test_save_cancelled_write_with_uncertain_result_is_not_aborted(
    repo_with_mocks,
):
    repo, bucket, grid_in = repo_with_mocks
    write_started = asyncio.Event()
    write_future = asyncio.get_running_loop().create_future()
    grid_in.chunk_size = 8

    def write(block):
        write_started.set()
        return write_future

    grid_in.write = Mock(side_effect=write)
    task = asyncio.create_task(repo.save(ObjectId(), "uncertain.pdf", b"data"))
    cancel_delivered = asyncio.Event()
    try:
        await _wait_for_event_or_task(write_started, task)
        task.cancel()
        asyncio.get_running_loop().call_soon(cancel_delivered.set)
        await cancel_delivered.wait()
        assert not task.done()
        write_future.set_exception(TimeoutError("write outcome unknown"))
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not write_future.done():
            write_future.set_result(None)
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    grid_in.abort.assert_not_awaited()
    grid_in.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_save_external_cancellation_during_close_drains_without_abort(
    repo_with_mocks,
):
    repo, bucket, grid_in = repo_with_mocks
    close_started = asyncio.Event()
    close_future = asyncio.get_running_loop().create_future()
    grid_in.chunk_size = 8
    grid_in.write = Mock(return_value=_completed_future())

    def close():
        close_started.set()
        return close_future

    grid_in.close = Mock(side_effect=close)
    task = asyncio.create_task(repo.save(ObjectId(), "cancel-close.pdf", b"data"))
    try:
        await _wait_for_event_or_task(close_started, task)
        task.cancel()
        cancelled_delivery = asyncio.Event()
        asyncio.get_running_loop().call_soon(cancelled_delivery.set)
        await cancelled_delivery.wait()
        assert not task.done()
        assert not close_future.cancelled()
        grid_in.abort.assert_not_awaited()
        close_future.set_result(None)
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not close_future.done():
            close_future.set_result(None)
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    grid_in.abort.assert_not_awaited()


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
