"""Pruebas unitarias RED para la coordinación de cargas con lease."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock

import pytest
from bson import ObjectId

import src.services.document_service as document_service_module
from src.core.exceptions import DatabaseUnavailableError
from src.models.document import DocumentModel, DocumentStatus
from src.repository.gridfs_repository import GridFSUploadInterruptedError
from src.services.document_service import DocumentService


class FakeClock:
    """Relojes de pared y monotónico controlados por los tests."""

    def __init__(self):
        # Fechas persistidas por BSON tienen precisión de milisegundos.
        self.utc = datetime(2035, 1, 2, 3, 4, 5, 987000, timezone.utc)
        self.monotonic = 1000.0
        self._waiters = []
        self.wait_requests = []
        self.wait_request_queue = asyncio.Queue()

    def utc_now(self):
        return self.utc

    def monotonic_now(self):
        return self.monotonic

    def advance(self, seconds):
        self.utc += timedelta(seconds=seconds)
        self.monotonic += seconds
        pending = []
        for deadline, future in self._waiters:
            if future.done():
                continue
            if deadline <= self.monotonic:
                future.set_result(False)
            else:
                pending.append((deadline, future))
        self._waiters = pending

    async def wait_for_signal(self, event, timeout):
        """Retorna True por señal o False al vencer el deadline simulado."""
        if event.is_set():
            return True
        if timeout <= 0:
            return False
        deadline = self.monotonic + max(timeout, 0)
        self.wait_requests.append((self.monotonic, timeout))
        self.wait_request_queue.put_nowait((self.monotonic, timeout))
        timer = asyncio.get_running_loop().create_future()
        waiter = (deadline, timer)
        self._waiters.append(waiter)
        event_task = asyncio.create_task(event.wait())
        try:
            done, _ = await asyncio.wait(
                {event_task, timer},
                return_when=asyncio.FIRST_COMPLETED,
            )
            return event_task in done and event_task.result()
        finally:
            if waiter in self._waiters:
                self._waiters.remove(waiter)
            if not timer.done():
                timer.cancel()
            if not event_task.done():
                event_task.cancel()
            await asyncio.wait_for(
                asyncio.gather(event_task, return_exceptions=True),
                timeout=2,
            )

    async def tick(self, seconds):
        """Avanza los relojes y entrega los deadlines ya alcanzados."""
        self.advance(seconds)
        await event_loop_checkpoint()

    async def next_wait_request(self):
        """Confirma que el coordinador ya armó su siguiente espera temporal."""
        return await asyncio.wait_for(self.wait_request_queue.get(), timeout=2)


@pytest.fixture
def fake_clock(monkeypatch):
    clock = FakeClock()
    # Estos seams se incorporarán en GREEN; raising=False permite RED sin ellos.
    monkeypatch.setattr(
        document_service_module,
        "_upload_utc_now",
        clock.utc_now,
        raising=False,
    )
    monkeypatch.setattr(
        document_service_module,
        "_upload_monotonic_now",
        clock.monotonic_now,
        raising=False,
    )
    monkeypatch.setattr(
        document_service_module,
        "_wait_for_upload_signal",
        clock.wait_for_signal,
        raising=False,
    )
    monkeypatch.setattr(
        document_service_module.settings,
        "upload_lease_seconds",
        120,
    )
    monkeypatch.setattr(
        document_service_module.settings,
        "upload_lease_renew_interval_seconds",
        30,
    )
    return clock


@pytest.mark.asyncio
async def test_fake_clock_wait_respects_deadline_and_early_signal(fake_clock):
    event = asyncio.Event()
    wait_task = asyncio.create_task(fake_clock.wait_for_signal(event, 30))
    await fake_clock.next_wait_request()

    await fake_clock.tick(29)
    assert not wait_task.done()
    event.set()
    assert await await_task(wait_task) is True

    expired = asyncio.create_task(fake_clock.wait_for_signal(event, 5))
    assert await await_task(expired) is True


@pytest.mark.asyncio
async def test_fake_clock_tick_advances_without_waiter_and_does_not_replay(fake_clock):
    start_utc = fake_clock.utc
    start_monotonic = fake_clock.monotonic

    await fake_clock.tick(7)
    assert fake_clock.utc == start_utc + timedelta(seconds=7)
    assert fake_clock.monotonic == start_monotonic + 7

    event = asyncio.Event()
    wait_task = asyncio.create_task(fake_clock.wait_for_signal(event, 3))
    await fake_clock.next_wait_request()
    await fake_clock.tick(2)
    assert not wait_task.done()
    await fake_clock.tick(1)
    assert await await_task(wait_task) is False
    assert fake_clock.monotonic == start_monotonic + 10


def owned_document(clock, *, lease_seconds=120, **overrides):
    values = {
        "id": str(ObjectId()),
        "filename": "upload.pdf",
        "checksum": "a" * 64,
        "size_bytes": 1024,
        "status": DocumentStatus.UPLOADING,
        "upload_owner_id": "upload-owner",
        "upload_fencing_token": "upload-fencing-token",
        "upload_lease_until": clock.utc + timedelta(seconds=lease_seconds),
        "file_id": None,
    }
    values.update(overrides)
    return DocumentModel(**values)


def service_with_mocks(*, writer=None, renewer=None):
    repository = Mock()
    gridfs_repository = Mock()
    gridfs_repository.save = AsyncMock(
        side_effect=writer or _successful_writer
    )
    repository._gridfs_repository = Mock(return_value=gridfs_repository)
    service = DocumentService(repository=repository)
    service.renew_upload_lease = AsyncMock(
        side_effect=renewer or _successful_renewal
    )
    return service, repository, gridfs_repository


async def _successful_writer(*, file_id, **kwargs):
    return file_id


async def _successful_renewal(document):
    return document.upload_lease_until + timedelta(seconds=120)


async def _controlled_gridfs_operation(future, started, cancellation_seen=None):
    """Emula la espera shielded del adaptador durante cancelación externa."""
    started.set()
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError as cancellation:
        if cancellation_seen is not None:
            cancellation_seen.set()
        while not future.done():
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        try:
            future.result()
        except BaseException:
            pass
        raise cancellation


async def coordinate(
    service,
    document,
    file_id=None,
    *,
    ownership_acquired=True,
    data=b"pdf-bytes",
):
    return await service.save_original_pdf_with_lease(
        document=document,
        file_id=ObjectId() if file_id is None else file_id,
        filename="upload.pdf",
        data=data,
        metadata={"checksum": document.checksum},
        ownership_acquired=ownership_acquired,
    )


async def wait_for_event_or_task(event, task):
    """Espera un evento o el fallo temprano del coordinador, con guard."""
    event_task = asyncio.create_task(event.wait())
    try:
        done, _ = await asyncio.wait_for(
            asyncio.wait(
                {event_task, task},
                return_when=asyncio.FIRST_COMPLETED,
            ),
            timeout=2,
        )
        if task in done:
            await task
            pytest.fail("El coordinador terminó antes del evento esperado")
        await event_task
    finally:
        if not event_task.done():
            event_task.cancel()
        await asyncio.wait_for(
            asyncio.gather(event_task, return_exceptions=True),
            timeout=2,
        )


async def wait_for_wait_request_or_task(clock, task):
    """Confirma el armado del timer o propaga el fallo temprano del coordinador."""
    request_task = asyncio.create_task(clock.wait_request_queue.get())
    try:
        done, _ = await asyncio.wait_for(
            asyncio.wait(
                {request_task, task},
                return_when=asyncio.FIRST_COMPLETED,
            ),
            timeout=2,
        )
        if task in done:
            await task
            pytest.fail("El coordinador terminó antes de armar su timer")
        await request_task
    finally:
        if not request_task.done():
            request_task.cancel()
        await asyncio.wait_for(
            asyncio.gather(request_task, return_exceptions=True),
            timeout=2,
        )


async def await_task(task):
    """Límite de seguridad para detectar tasks que no terminan."""
    return await asyncio.wait_for(asyncio.shield(task), timeout=2)


async def cancel_and_drain(task):
    """Cancela y recoge una task con un límite de seguridad para teardown."""
    if not task.done():
        task.cancel()
    done, _ = await asyncio.wait({task}, timeout=2)
    if not done:
        task.cancel()
        await asyncio.wait({task}, timeout=2)
        pytest.fail("La task no terminó dentro del límite de teardown.")
    if task.cancelled():
        return
    try:
        task.result()
    except Exception:
        # El cuerpo del test verifica la causa; teardown solo recupera el resultado.
        pass


async def event_loop_checkpoint():
    """Punto de cesión observable; no introduce un retardo temporal."""
    future = asyncio.get_running_loop().create_future()
    asyncio.get_running_loop().call_soon(future.set_result, None)
    await future


def assert_lease_lost(exception_info):
    # No importar la excepción nueva: RED debe poder recolectarse.
    assert exception_info.type.__name__ == "UploadLeaseLostError"


def assert_no_pending_tasks_since(before_tasks):
    pending = {
        task
        for task in asyncio.all_tasks() - before_tasks
        if not task.done()
    }
    assert pending == set()


@pytest.mark.asyncio
async def test_save_original_pdf_preserves_old_callers_and_forwards_callback():
    service, repository, gridfs_repository = service_with_mocks()
    file_id = ObjectId()
    callback = Mock(return_value=True)

    old_result = await service.save_original_pdf(
        file_id=file_id,
        filename="legacy.pdf",
        data=b"legacy",
    )
    assert old_result == file_id
    gridfs_repository.save.assert_awaited_once_with(
        file_id=file_id,
        filename="legacy.pdf",
        data=b"legacy",
        metadata=None,
    )

    gridfs_repository.save.reset_mock()
    result = await service.save_original_pdf(
        file_id=file_id,
        filename="cooperative.pdf",
        data=b"cooperative",
        metadata={"k": "v"},
        should_continue=callback,
    )

    assert result == file_id
    gridfs_repository.save.assert_awaited_once_with(
        file_id=file_id,
        filename="cooperative.pdf",
        data=b"cooperative",
        metadata={"k": "v"},
        should_continue=callback,
    )


@pytest.mark.asyncio
async def test_rejected_acquisition_is_not_confused_with_existing_owner(fake_clock):
    service, repository, gridfs_repository = service_with_mocks()
    document = owned_document(fake_clock)

    with pytest.raises(Exception) as raised:
        await coordinate(
            service,
            document,
            ownership_acquired=False,
        )

    assert_lease_lost(raised)
    service.renew_upload_lease.assert_not_awaited()
    gridfs_repository.save.assert_not_awaited()


@pytest.mark.parametrize(
    "overrides",
    [
        {"id": None},
        {"id": ""},
        {"status": DocumentStatus.PENDING},
        {"file_id": ObjectId()},
        {"upload_owner_id": None},
        {"upload_owner_id": ""},
        {"upload_fencing_token": None},
        {"upload_fencing_token": ""},
        {"upload_lease_until": None},
    ],
)
@pytest.mark.asyncio
async def test_structural_preconditions_fail_before_renewal_or_write(
    fake_clock,
    overrides,
):
    service, repository, gridfs_repository = service_with_mocks()
    document = owned_document(fake_clock, **overrides)

    with pytest.raises(ValueError):
        await coordinate(service, document)

    service.renew_upload_lease.assert_not_awaited()
    gridfs_repository.save.assert_not_awaited()


@pytest.mark.asyncio
async def test_invalid_file_id_is_rejected_before_starting_tasks(fake_clock):
    service, repository, gridfs_repository = service_with_mocks()
    document = owned_document(fake_clock)

    with pytest.raises(ValueError):
        await coordinate(service, document, file_id="not-an-object-id")

    service.renew_upload_lease.assert_not_awaited()
    gridfs_repository.save.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("lease_seconds", [-0.001, 0])
async def test_expired_initial_lease_is_lease_loss(fake_clock, lease_seconds):
    service, repository, gridfs_repository = service_with_mocks()
    document = owned_document(
        fake_clock,
        lease_seconds=lease_seconds,
    )

    with pytest.raises(Exception) as raised:
        await coordinate(service, document)

    assert_lease_lost(raised)
    service.renew_upload_lease.assert_not_awaited()
    gridfs_repository.save.assert_not_awaited()


@pytest.mark.parametrize(
    "expiry_kind",
    ["naive_bson_utc", "aware_non_utc"],
)
@pytest.mark.asyncio
async def test_expiration_normalization_and_bson_precision(
    fake_clock,
    expiry_kind,
):
    service, repository, gridfs_repository = service_with_mocks()
    expiry = fake_clock.utc + timedelta(seconds=120, microseconds=987)
    if expiry_kind == "naive_bson_utc":
        expiry = expiry.replace(tzinfo=None)
    elif expiry_kind == "aware_non_utc":
        expiry = expiry.astimezone(timezone(timedelta(hours=2)))
    document = owned_document(fake_clock, upload_lease_until=expiry)

    result = await coordinate(service, document)

    assert result.__class__ is ObjectId
    assert gridfs_repository.save.await_count == 1
    service.renew_upload_lease.assert_not_awaited()


@pytest.mark.asyncio
async def test_bson_millisecond_truncation_rejects_reached_deadline(fake_clock):
    service, repository, gridfs_repository = service_with_mocks()
    document = owned_document(
        fake_clock,
        upload_lease_until=fake_clock.utc
        + timedelta(seconds=120, microseconds=1999),
    )
    # El BSON deadline truncado es .988000; la fecha original era .988999.
    fake_clock.advance(120.001)

    with pytest.raises(Exception) as raised:
        await coordinate(service, document)

    assert_lease_lost(raised)
    gridfs_repository.save.assert_not_awaited()
    service.renew_upload_lease.assert_not_awaited()


@pytest.mark.asyncio
async def test_fast_write_finishes_without_unnecessary_renewal(fake_clock):
    service, repository, gridfs_repository = service_with_mocks()
    document = owned_document(fake_clock)
    file_id = ObjectId()

    result = await coordinate(service, document, file_id)

    assert result == file_id
    service.renew_upload_lease.assert_not_awaited()
    gridfs_repository.save.assert_awaited_once()
    gridfs_repository.save.await_args.kwargs["should_continue"]
    assert document.upload_lease_until == (
        fake_clock.utc + timedelta(seconds=120)
    )


@pytest.mark.asyncio
async def test_renewal_is_requested_immediately_when_remaining_lease_is_short(
    fake_clock,
):
    renewed_until = fake_clock.utc + timedelta(seconds=150)
    service, repository, gridfs_repository = service_with_mocks(
        renewer=AsyncMock(return_value=renewed_until)
    )
    document = owned_document(fake_clock, lease_seconds=30)
    writer_started = asyncio.Event()
    release_writer = asyncio.Event()
    renewal_started = asyncio.Event()

    async def renew(document_arg):
        renewal_started.set()
        return renewed_until

    service.renew_upload_lease.side_effect = renew

    async def writer(*, file_id, should_continue, **kwargs):
        writer_started.set()
        await release_writer.wait()
        assert should_continue()
        return file_id

    gridfs_repository.save.side_effect = writer
    task = asyncio.create_task(coordinate(service, document))
    try:
        await wait_for_event_or_task(writer_started, task)
        await wait_for_event_or_task(renewal_started, task)
        service.renew_upload_lease.assert_awaited_once_with(document)
        release_writer.set()
        await await_task(task)
    finally:
        release_writer.set()
        await cancel_and_drain(task)


@pytest.mark.asyncio
async def test_long_write_renews_repeatedly_without_overlapping_calls(fake_clock):
    service, repository, gridfs_repository = service_with_mocks()
    document = owned_document(fake_clock)
    writer_started = asyncio.Event()
    release_writer = asyncio.Event()
    renewal_started = asyncio.Event()
    renewal_release = asyncio.Queue()
    active_renewals = 0
    maximum_active = 0
    renewal_number = 0

    async def renew(document_arg):
        nonlocal active_renewals, maximum_active, renewal_number
        active_renewals += 1
        maximum_active = max(maximum_active, active_renewals)
        renewal_number += 1
        renewal_started.set()
        await renewal_release.get()
        active_renewals -= 1
        return fake_clock.utc + timedelta(seconds=120)

    async def writer(*, file_id, should_continue, **kwargs):
        writer_started.set()
        await release_writer.wait()
        assert should_continue()
        return file_id

    service.renew_upload_lease.side_effect = renew
    gridfs_repository.save.side_effect = writer
    task = asyncio.create_task(coordinate(service, document))
    try:
        await wait_for_event_or_task(writer_started, task)
        await wait_for_wait_request_or_task(fake_clock, task)
        first_wait = fake_clock.wait_requests[0]
        assert first_wait[1] == 30
        assert first_wait[0] == 1000.0
        await fake_clock.tick(30)
        await wait_for_event_or_task(renewal_started, task)
        await wait_for_wait_request_or_task(fake_clock, task)
        await fake_clock.tick(30)
        assert active_renewals == 1
        assert renewal_number == 1
        renewal_release.put_nowait(None)
        renewal_started.clear()
        await wait_for_wait_request_or_task(fake_clock, task)
        await fake_clock.tick(30)
        await wait_for_event_or_task(renewal_started, task)
        assert maximum_active == 1
        renewal_release.put_nowait(None)
        release_writer.set()
        await await_task(task)
    finally:
        release_writer.set()
        renewal_release.put_nowait(None)
        await cancel_and_drain(task)


@pytest.mark.asyncio
async def test_rejected_renewal_interrupts_active_writer(fake_clock):
    service, repository, gridfs_repository = service_with_mocks()
    document = owned_document(fake_clock)
    write_started = asyncio.Event()
    release_write = asyncio.Event()
    renewal_started = asyncio.Event()
    writes_started = 0

    async def renew(document_arg):
        renewal_started.set()
        return None

    async def writer(*, file_id, should_continue, **kwargs):
        nonlocal writes_started
        writes_started += 1
        write_started.set()
        await release_write.wait()
        if not should_continue():
            raise GridFSUploadInterruptedError()
        return file_id

    gridfs_repository.save.side_effect = writer
    service.renew_upload_lease.side_effect = renew
    before_tasks = asyncio.all_tasks()
    task = asyncio.create_task(coordinate(service, document))
    try:
        await wait_for_event_or_task(write_started, task)
        await wait_for_wait_request_or_task(fake_clock, task)
        await fake_clock.tick(30)
        await wait_for_event_or_task(renewal_started, task)
        assert not release_write.is_set()
        release_write.set()
        with pytest.raises(Exception) as raised:
            await await_task(task)
        assert_lease_lost(raised)
    finally:
        release_write.set()
        await cancel_and_drain(task)

    assert writes_started == 1
    assert service.renew_upload_lease.await_count == 1
    assert_no_pending_tasks_since(before_tasks)


@pytest.mark.parametrize(
    "renewal_error_factory",
    [
        lambda: DatabaseUnavailableError(),
        lambda: RuntimeError("renewal failed"),
    ],
)
@pytest.mark.asyncio
async def test_renewal_error_during_write_preserves_same_error(
    fake_clock,
    renewal_error_factory,
):
    error = renewal_error_factory()
    service, repository, gridfs_repository = service_with_mocks()
    document = owned_document(fake_clock)
    writer_started = asyncio.Event()
    release_writer = asyncio.Event()
    renewal_started = asyncio.Event()

    async def renew(document_arg):
        renewal_started.set()
        raise error

    async def writer(*, file_id, should_continue, **kwargs):
        writer_started.set()
        await release_writer.wait()
        if not should_continue():
            raise GridFSUploadInterruptedError()
        return file_id

    gridfs_repository.save.side_effect = writer
    service.renew_upload_lease.side_effect = renew
    task = asyncio.create_task(coordinate(service, document))
    try:
        await wait_for_event_or_task(writer_started, task)
        await wait_for_wait_request_or_task(fake_clock, task)
        await fake_clock.tick(30)
        await wait_for_event_or_task(renewal_started, task)
        assert not release_writer.is_set()
        release_writer.set()
        with pytest.raises(type(error)) as raised:
            await await_task(task)
        assert raised.value is error
    finally:
        release_writer.set()
        await cancel_and_drain(task)


@pytest.mark.asyncio
async def test_expiry_while_renewal_pending_latches_loss(fake_clock):
    fake_clock.utc += timedelta(microseconds=654)
    renewal_started = asyncio.Event()
    renewal_future = asyncio.get_running_loop().create_future()
    service, repository, gridfs_repository = service_with_mocks()
    async def renew(document_arg):
        renewal_started.set()
        return await renewal_future

    service.renew_upload_lease.side_effect = renew
    document = owned_document(fake_clock, lease_seconds=40)
    writer_started = asyncio.Event()
    release_write = asyncio.Event()
    writer_cancelled = asyncio.Event()

    async def writer(*, file_id, should_continue, **kwargs):
        writer_started.set()
        try:
            await release_write.wait()
        except asyncio.CancelledError:
            writer_cancelled.set()
            raise
        return file_id

    gridfs_repository.save.side_effect = writer
    before_tasks = asyncio.all_tasks()
    task = asyncio.create_task(coordinate(service, document))
    try:
        await wait_for_event_or_task(writer_started, task)
        await wait_for_wait_request_or_task(fake_clock, task)
        await fake_clock.tick(30)
        await wait_for_event_or_task(renewal_started, task)
        # El deadline BSON vence 0.000346 s antes del deadline calculado
        # si también se trunca el reloj UTC actual.
        await fake_clock.tick(9.9995)
        # El watchdog debe detectar el vencimiento sin una consulta del callback.
        await wait_for_event_or_task(writer_cancelled, task)
        assert not renewal_future.done()
        assert not renewal_future.cancelled()
        assert not task.done()
        renewal_future.set_result(None)
        with pytest.raises(Exception) as raised:
            await await_task(task)
        assert_lease_lost(raised)
    finally:
        release_write.set()
        if not renewal_future.done():
            renewal_future.set_result(None)
        await cancel_and_drain(task)

    assert_no_pending_tasks_since(before_tasks)


@pytest.mark.asyncio
async def test_successful_renewal_deadline_uses_full_utc_precision(fake_clock):
    fake_clock.utc += timedelta(microseconds=654)
    renewal_started = asyncio.Event()
    renewal_future = asyncio.get_running_loop().create_future()
    second_renewal_future = asyncio.get_running_loop().create_future()
    renewal_calls = 0
    service, repository, gridfs_repository = service_with_mocks()
    document = owned_document(fake_clock, lease_seconds=40)
    writer_started = asyncio.Event()
    release_writer = asyncio.Event()
    writer_cancelled = asyncio.Event()
    callback_ref = []

    async def renew(document_arg):
        nonlocal renewal_calls
        renewal_calls += 1
        renewal_started.set()
        if renewal_calls == 1:
            return await renewal_future
        return await second_renewal_future

    async def writer(*, file_id, should_continue, **kwargs):
        writer_started.set()
        callback_ref.append(should_continue)
        try:
            await release_writer.wait()
        except asyncio.CancelledError:
            writer_cancelled.set()
            raise
        return file_id

    service.renew_upload_lease.side_effect = renew
    gridfs_repository.save.side_effect = writer
    before_tasks = asyncio.all_tasks()
    task = asyncio.create_task(coordinate(service, document))
    try:
        await wait_for_event_or_task(writer_started, task)
        await wait_for_wait_request_or_task(fake_clock, task)
        await fake_clock.tick(30)
        await wait_for_event_or_task(renewal_started, task)
        renewed_expiry = fake_clock.utc + timedelta(seconds=90, microseconds=500)
        renewal_future.set_result(renewed_expiry)
        # Espera la siguiente renovación para confirmar que la primera ya fue
        # observada y que su deadline quedó anclado.
        await wait_for_wait_request_or_task(fake_clock, task)
        renewal_started.clear()
        await fake_clock.tick(30)
        await wait_for_event_or_task(renewal_started, task)
        await fake_clock.tick(60.0005)
        assert callback_ref[0]() is False
        await wait_for_event_or_task(writer_cancelled, task)
        assert not task.done()
        assert not second_renewal_future.cancelled()
        second_renewal_future.set_result(None)
        with pytest.raises(Exception) as raised:
            await await_task(task)
        assert_lease_lost(raised)
    finally:
        release_writer.set()
        if not second_renewal_future.done():
            second_renewal_future.set_result(None)
        await cancel_and_drain(task)
    assert_no_pending_tasks_since(before_tasks)


@pytest.mark.asyncio
async def test_secondary_renewal_exception_is_retrieved_after_lease_loss(fake_clock):
    renewal_started = asyncio.Event()
    renewal_future = asyncio.get_running_loop().create_future()
    service, repository, gridfs_repository = service_with_mocks()
    document = owned_document(fake_clock, lease_seconds=40)
    writer_started = asyncio.Event()
    writer_cancelled = asyncio.Event()
    loop = asyncio.get_running_loop()
    unhandled = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))

    async def renew(document_arg):
        renewal_started.set()
        return await renewal_future

    async def writer(*, file_id, **kwargs):
        writer_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            writer_cancelled.set()
            raise

    service.renew_upload_lease.side_effect = renew
    gridfs_repository.save.side_effect = writer
    before_tasks = asyncio.all_tasks()
    task = asyncio.create_task(coordinate(service, document))
    try:
        await wait_for_event_or_task(writer_started, task)
        await wait_for_wait_request_or_task(fake_clock, task)
        await fake_clock.tick(30)
        await wait_for_event_or_task(renewal_started, task)
        await fake_clock.tick(10)
        await wait_for_event_or_task(writer_cancelled, task)
        assert not task.done()
        renewal_future.set_exception(RuntimeError("secondary renewal failure"))
        with pytest.raises(Exception) as raised:
            await await_task(task)
        assert_lease_lost(raised)
        await event_loop_checkpoint()
        assert not unhandled
    finally:
        loop.set_exception_handler(previous_handler)
        if not renewal_future.done():
            renewal_future.set_result(None)
        await cancel_and_drain(task)
    assert_no_pending_tasks_since(before_tasks)


@pytest.mark.asyncio
async def test_unexpected_coordinator_error_cancels_and_drains_writer(
    fake_clock,
    monkeypatch,
):
    error = RuntimeError("unexpected coordinator failure")
    service, repository, gridfs_repository = service_with_mocks()
    document = owned_document(fake_clock)
    writer_started = asyncio.Event()
    writer_cancelled = asyncio.Event()

    async def broken_wait(event, timeout):
        await writer_started.wait()
        raise error

    async def writer(*, file_id, **kwargs):
        writer_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            writer_cancelled.set()
            raise

    monkeypatch.setattr(
        document_service_module,
        "_wait_for_upload_signal",
        broken_wait,
    )
    gridfs_repository.save.side_effect = writer
    before_tasks = asyncio.all_tasks()
    task = asyncio.create_task(coordinate(service, document))
    try:
        await asyncio.wait_for(writer_started.wait(), timeout=2)
        await asyncio.wait_for(writer_cancelled.wait(), timeout=2)
        with pytest.raises(RuntimeError) as raised:
            await await_task(task)
        assert raised.value is error
    finally:
        if not task.done():
            await cancel_and_drain(task)
    assert_no_pending_tasks_since(before_tasks)


def _raise_interrupted():
    raise GridFSUploadInterruptedError()


def signal_task_completion(event):
    """Publica que una operación controlada ya terminó, antes de seguir el test."""
    asyncio.current_task().add_done_callback(lambda completed: event.set())


@pytest.mark.asyncio
async def test_positive_renewal_observed_after_old_deadline_cannot_restore_lease(
    fake_clock,
):
    renewal_started = asyncio.Event()
    renewal_future = asyncio.get_running_loop().create_future()
    service, repository, gridfs_repository = service_with_mocks()
    async def renew(document_arg):
        renewal_started.set()
        return await renewal_future

    service.renew_upload_lease.side_effect = renew
    document = owned_document(fake_clock, lease_seconds=40)
    writer_started = asyncio.Event()
    release_writer = asyncio.Event()
    callback_results = []
    callback_ref = []
    callback_checked = asyncio.Event()

    async def writer(*, file_id, should_continue, **kwargs):
        writer_started.set()
        callback_ref.append(should_continue)
        await release_writer.wait()
        callback_results.append(should_continue())
        callback_checked.set()
        return file_id if callback_results[-1] else _raise_interrupted()

    gridfs_repository.save.side_effect = writer
    before_tasks = asyncio.all_tasks()
    task = asyncio.create_task(coordinate(service, document))
    try:
        await wait_for_event_or_task(writer_started, task)
        await wait_for_wait_request_or_task(fake_clock, task)
        await fake_clock.tick(30)
        await wait_for_event_or_task(renewal_started, task)
        await fake_clock.tick(10)
        release_writer.set()
        await wait_for_event_or_task(callback_checked, task)
        assert callback_results == [False]
        assert not renewal_future.done()
        assert not renewal_future.cancelled()
        assert not task.done()
        # La respuesta positiva llega solo después de observar la pérdida.
        renewal_future.set_result(fake_clock.utc + timedelta(seconds=180))
        with pytest.raises(Exception) as raised:
            await await_task(task)
        assert_lease_lost(raised)
        assert callback_ref[0]() is False
    finally:
        release_writer.set()
        if not renewal_future.done():
            renewal_future.set_result(None)
        await cancel_and_drain(task)

    assert callback_results == [False]
    assert renewal_future.done() and not renewal_future.cancelled()
    assert_no_pending_tasks_since(before_tasks)


@pytest.mark.parametrize(
    "invalid_expiry",
    [
        "already_expired",
        "not_extended",
        "invalid_type",
    ],
)
@pytest.mark.asyncio
async def test_invalid_renewal_deadline_does_not_extend_authorization(
    fake_clock,
    invalid_expiry,
):
    old_expiry = fake_clock.utc + timedelta(seconds=120)
    renewed_expiry = (
        fake_clock.utc - timedelta(seconds=1)
        if invalid_expiry == "already_expired"
        else (
            "not-a-datetime"
            if invalid_expiry == "invalid_type"
            else old_expiry
        )
    )
    service, repository, gridfs_repository = service_with_mocks()
    document = owned_document(fake_clock)
    writer_started = asyncio.Event()
    release_writer = asyncio.Event()
    renewal_started = asyncio.Event()

    async def renew(document_arg):
        renewal_started.set()
        return renewed_expiry

    service.renew_upload_lease.side_effect = renew

    async def writer(*, file_id, should_continue, **kwargs):
        writer_started.set()
        await release_writer.wait()
        return file_id if should_continue() else _raise_interrupted()

    gridfs_repository.save.side_effect = writer
    before_tasks = asyncio.all_tasks()
    task = asyncio.create_task(coordinate(service, document))
    try:
        await wait_for_event_or_task(writer_started, task)
        await wait_for_wait_request_or_task(fake_clock, task)
        await fake_clock.tick(30)
        await wait_for_event_or_task(renewal_started, task)
        release_writer.set()
        with pytest.raises(Exception) as raised:
            await await_task(task)
        assert_lease_lost(raised)
    finally:
        release_writer.set()
        await cancel_and_drain(task)
    assert_no_pending_tasks_since(before_tasks)


@pytest.mark.asyncio
async def test_loss_during_close_waits_and_never_reports_success(fake_clock):
    service, repository, gridfs_repository = service_with_mocks()
    document = owned_document(fake_clock)
    close_started = asyncio.Event()
    close_future = asyncio.get_running_loop().create_future()
    renewal_started = asyncio.Event()
    abort_calls = []

    async def renew(document_arg):
        renewal_started.set()
        return None

    async def writer(*, file_id, **kwargs):
        return await _controlled_gridfs_operation(
            close_future,
            close_started,
            cancellation_seen=writer_cancelled,
        )

    writer_cancelled = asyncio.Event()
    gridfs_repository.save.side_effect = writer
    gridfs_repository.delete = AsyncMock(
        side_effect=lambda *args, **kwargs: abort_calls.append(args)
    )
    service.renew_upload_lease.side_effect = renew
    before_tasks = asyncio.all_tasks()
    task = asyncio.create_task(coordinate(service, document))
    try:
        await wait_for_event_or_task(close_started, task)
        await wait_for_wait_request_or_task(fake_clock, task)
        await fake_clock.tick(30)
        await wait_for_event_or_task(renewal_started, task)
        await wait_for_event_or_task(writer_cancelled, task)
        assert not task.done()
        assert not close_future.cancelled()
        assert not close_future.done()
        # La renovación rechazada ya registró pérdida; el close protegido
        # continúa pendiente hasta que se libere su operación remota.
        close_future.set_result(ObjectId())
        with pytest.raises(Exception) as raised:
            await await_task(task)
        assert_lease_lost(raised)
    finally:
        if not close_future.done():
            close_future.set_result(ObjectId())
        await cancel_and_drain(task)

    assert abort_calls == []
    assert_no_pending_tasks_since(before_tasks)


@pytest.mark.parametrize("renewal_outcome", ["accepted", "rejected", "expired"])
@pytest.mark.asyncio
async def test_writer_finishing_with_renewal_pending_waits_and_checks_authority(
    fake_clock,
    renewal_outcome,
):
    renewal_started = asyncio.Event()
    renewal_future = asyncio.get_running_loop().create_future()
    service, repository, gridfs_repository = service_with_mocks()

    async def renew(document_arg):
        renewal_started.set()
        return await renewal_future

    service.renew_upload_lease.side_effect = renew
    document = owned_document(fake_clock, lease_seconds=40)
    writer_finished = asyncio.Event()
    release_writer = asyncio.Event()
    file_id = ObjectId()

    async def writer(*, file_id, **kwargs):
        await release_writer.wait()
        writer_finished.set()
        return file_id

    gridfs_repository.save.side_effect = writer
    before_tasks = asyncio.all_tasks()
    task = asyncio.create_task(coordinate(service, document, file_id))
    try:
        await wait_for_wait_request_or_task(fake_clock, task)
        await fake_clock.tick(30)
        await wait_for_event_or_task(renewal_started, task)
        release_writer.set()
        await wait_for_event_or_task(writer_finished, task)
        assert not task.done()

        if renewal_outcome == "accepted":
            renewal_future.set_result(fake_clock.utc + timedelta(seconds=180))
            assert await await_task(task) == file_id
        elif renewal_outcome == "rejected":
            renewal_future.set_result(None)
            with pytest.raises(Exception) as raised:
                await await_task(task)
            assert_lease_lost(raised)
        else:
            await fake_clock.tick(40)
            renewal_future.set_result(fake_clock.utc + timedelta(seconds=180))
            with pytest.raises(Exception) as raised:
                await await_task(task)
            assert_lease_lost(raised)
    finally:
        release_writer.set()
        if not renewal_future.done():
            renewal_future.set_result(None)
        await cancel_and_drain(task)
    assert_no_pending_tasks_since(before_tasks)


@pytest.mark.asyncio
async def test_writer_error_drains_inflight_renewal_and_preserves_writer_error(
    fake_clock,
):
    writer_error = RuntimeError("gridfs write failed")
    renewal_started = asyncio.Event()
    writer_finished = asyncio.Event()
    renewal_future = asyncio.get_running_loop().create_future()
    service, repository, gridfs_repository = service_with_mocks()

    async def renew(document_arg):
        renewal_started.set()
        return await renewal_future

    service.renew_upload_lease.side_effect = renew
    document = owned_document(fake_clock)
    writer_started = asyncio.Event()

    async def writer(**kwargs):
        writer_started.set()
        await renewal_started.wait()
        signal_task_completion(writer_finished)
        raise writer_error

    gridfs_repository.save.side_effect = writer
    before_tasks = asyncio.all_tasks()
    task = asyncio.create_task(coordinate(service, document))
    try:
        await wait_for_event_or_task(writer_started, task)
        await wait_for_wait_request_or_task(fake_clock, task)
        await fake_clock.tick(30)
        await wait_for_event_or_task(renewal_started, task)
        await wait_for_event_or_task(writer_finished, task)
        await event_loop_checkpoint()
        assert not task.done()
        assert not renewal_future.cancelled()
        renewal_future.set_result(fake_clock.utc + timedelta(seconds=180))
        with pytest.raises(RuntimeError) as raised:
            await await_task(task)
        assert raised.value is writer_error
    finally:
        if not renewal_future.done():
            renewal_future.set_result(None)
        await cancel_and_drain(task)
    assert_no_pending_tasks_since(before_tasks)


@pytest.mark.asyncio
async def test_cancellation_during_drain_does_not_replace_registered_writer_error(
    fake_clock,
):
    writer_error = RuntimeError("writer failed first")
    renewal_started = asyncio.Event()
    writer_started = asyncio.Event()
    writer_finished = asyncio.Event()
    renewal_future = asyncio.get_running_loop().create_future()
    service, repository, gridfs_repository = service_with_mocks()

    async def renew(document_arg):
        renewal_started.set()
        return await renewal_future

    async def writer(**kwargs):
        writer_started.set()
        await renewal_started.wait()
        signal_task_completion(writer_finished)
        raise writer_error

    service.renew_upload_lease.side_effect = renew
    gridfs_repository.save.side_effect = writer
    before_tasks = asyncio.all_tasks()
    task = asyncio.create_task(coordinate(service, owned_document(fake_clock)))
    try:
        await wait_for_event_or_task(writer_started, task)
        await wait_for_wait_request_or_task(fake_clock, task)
        await fake_clock.tick(30)
        await wait_for_event_or_task(renewal_started, task)
        await wait_for_event_or_task(writer_finished, task)
        await event_loop_checkpoint()
        assert not task.done()

        # El writer ya falló mientras el coordinador espera drenar el renew.
        task.cancel()
        assert not task.done()
        assert not renewal_future.cancelled()
        renewal_future.set_result(fake_clock.utc + timedelta(seconds=180))

        with pytest.raises(RuntimeError) as raised:
            await await_task(task)
        assert raised.value is writer_error
    finally:
        if not renewal_future.done():
            renewal_future.set_result(None)
        await cancel_and_drain(task)
    assert_no_pending_tasks_since(before_tasks)


@pytest.mark.parametrize("during", ["write", "renewal"])
@pytest.mark.asyncio
async def test_external_and_repeated_cancellation_drains_started_work(
    fake_clock,
    during,
):
    writer_started = asyncio.Event()
    writer_cancelled = asyncio.Event()
    renewal_started = asyncio.Event()
    operation_future = asyncio.get_running_loop().create_future()
    service, repository, gridfs_repository = service_with_mocks()
    document = owned_document(fake_clock)

    async def writer(*, file_id, **kwargs):
        return await _controlled_gridfs_operation(
            operation_future,
            writer_started,
            writer_cancelled,
        )

    async def renewer(document_arg):
        renewal_started.set()
        return await operation_future

    gridfs_repository.save.side_effect = writer
    if during == "renewal":
        service.renew_upload_lease.side_effect = renewer
        document = owned_document(fake_clock, lease_seconds=30)

    before_tasks = asyncio.all_tasks()
    task = asyncio.create_task(coordinate(service, document))
    try:
        await wait_for_event_or_task(writer_started, task)
        if during == "renewal":
            await wait_for_event_or_task(renewal_started, task)
        task.cancel()
        await wait_for_event_or_task(writer_cancelled, task)
        assert not task.done()
        assert not operation_future.cancelled()

        task.cancel()
        assert not task.done()

        operation_future.set_result(
            fake_clock.utc + timedelta(seconds=180)
            if during == "renewal"
            else ObjectId()
        )
        with pytest.raises(asyncio.CancelledError):
            await await_task(task)
    finally:
        if not operation_future.done():
            operation_future.set_result(None)
        await cancel_and_drain(task)
    assert_no_pending_tasks_since(before_tasks)


@pytest.mark.parametrize("first_failure", ["writer", "renewer"])
@pytest.mark.asyncio
async def test_first_terminal_failure_wins_when_tasks_fail_concurrently(
    fake_clock,
    first_failure,
):
    writer_error = RuntimeError("writer primary")
    renew_error = DatabaseUnavailableError()
    writer_started = asyncio.Event()
    renewal_started = asyncio.Event()
    writer_finished = asyncio.Event()
    renewal_finished = asyncio.Event()
    release_writer = asyncio.Event()
    release_renewal = asyncio.Event()
    renewal_future = asyncio.get_running_loop().create_future()
    service, repository, gridfs_repository = service_with_mocks()
    document = owned_document(fake_clock)

    async def writer(**kwargs):
        writer_started.set()
        if first_failure == "renewer":
            await release_writer.wait()
            if not kwargs["should_continue"]():
                raise GridFSUploadInterruptedError()
            raise writer_error
        else:
            await renewal_started.wait()
        signal_task_completion(writer_finished)
        raise writer_error

    async def renewer(document_arg):
        renewal_started.set()
        if first_failure == "writer":
            signal_task_completion(renewal_finished)
            return await renewal_future
        await release_renewal.wait()
        signal_task_completion(renewal_finished)
        raise renew_error

    gridfs_repository.save.side_effect = writer
    service.renew_upload_lease.side_effect = renewer
    task = asyncio.create_task(coordinate(service, document))
    try:
        await wait_for_event_or_task(writer_started, task)
        await wait_for_wait_request_or_task(fake_clock, task)
        await fake_clock.tick(30)
        await wait_for_event_or_task(renewal_started, task)
        if first_failure == "writer":
            await wait_for_event_or_task(writer_finished, task)
            await event_loop_checkpoint()
            assert not task.done()
            assert not renewal_future.cancelled()
            renewal_future.set_exception(renew_error)
            expected = writer_error
        else:
            release_renewal.set()
            await wait_for_event_or_task(renewal_finished, task)
            assert not task.done()
            release_writer.set()
            expected = renew_error

        with pytest.raises(type(expected)) as raised:
            await await_task(task)
        assert raised.value is expected
    finally:
        release_writer.set()
        release_renewal.set()
        if not renewal_future.done():
            renewal_future.set_result(None)
        if not task.done():
            await cancel_and_drain(task)


@pytest.mark.asyncio
async def test_deadline_is_checked_again_before_success_return(
    fake_clock,
    monkeypatch,
):
    service, repository, gridfs_repository = service_with_mocks()
    document = owned_document(fake_clock, lease_seconds=1)
    writer_finished = asyncio.Event()
    # Initial lease is longer than the interval, avoiding immediate renewal.
    monkeypatch.setattr(
        document_service_module.settings,
        "upload_lease_renew_interval_seconds",
        0.5,
    )

    async def writer(*, file_id, **kwargs):
        fake_clock.advance(1)
        writer_finished.set()
        return file_id

    gridfs_repository.save.side_effect = writer
    task = asyncio.create_task(coordinate(service, document))
    try:
        await wait_for_event_or_task(writer_finished, task)
        with pytest.raises(Exception) as raised:
            await await_task(task)
        assert_lease_lost(raised)
        service.renew_upload_lease.assert_not_awaited()
    finally:
        await cancel_and_drain(task)


@pytest.mark.parametrize("writer_fails", [False, True])
@pytest.mark.asyncio
async def test_coordinator_leaves_no_owned_tasks_pending(fake_clock, writer_fails):
    error = RuntimeError("writer failure")
    service, repository, gridfs_repository = service_with_mocks()
    document = owned_document(fake_clock)
    writer_started = asyncio.Event()
    release_writer = asyncio.Event()

    async def writer(*, file_id, **kwargs):
        writer_started.set()
        await release_writer.wait()
        if writer_fails:
            raise error
        return file_id

    gridfs_repository.save.side_effect = writer
    before_tasks = set(asyncio.all_tasks())
    task = asyncio.create_task(coordinate(service, document))
    try:
        await wait_for_event_or_task(writer_started, task)
        release_writer.set()
        if writer_fails:
            with pytest.raises(RuntimeError) as raised:
                await await_task(task)
            assert raised.value is error
        else:
            assert await await_task(task)
    finally:
        release_writer.set()
        await cancel_and_drain(task)

    current_tasks = asyncio.all_tasks()
    assert {
        pending
        for pending in current_tasks - before_tasks
        if not pending.done()
    } == set()
