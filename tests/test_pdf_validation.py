"""Focused tests for bounded and structural PDF upload validation."""

import asyncio
import io
import threading
from unittest.mock import AsyncMock, patch

import pymupdf
import pytest
from fastapi import FastAPI
from starlette.datastructures import Headers, UploadFile

from src.config.config import settings
from src.core.exceptions import (
    EmptyFileError,
    ExtractionError,
    FileTooLargeError,
    InvalidFileTypeError,
)
from src.api.pdf_validation import PdfValidationExecutor, validate_pdf_upload


def make_pdf() -> bytes:
    document = pymupdf.open()
    document.new_page()
    try:
        return document.tobytes()
    finally:
        document.close()


class RecordingUploadFile(UploadFile):
    def __init__(self, data: bytes, content_type="application/pdf"):
        super().__init__(
            filename="upload.pdf",
            file=io.BytesIO(data),
            headers=Headers({"content-type": content_type}),
        )
        self.read_sizes = []

    async def read(self, size=-1):
        self.read_sizes.append(size)
        return await super().read(size)


@pytest.fixture
async def pdf_executor():
    executor = PdfValidationExecutor()
    yield executor
    await executor.aclose()


async def event_loop_checkpoint():
    future = asyncio.get_running_loop().create_future()
    asyncio.get_running_loop().call_soon(future.set_result, None)
    await future


def signal_after_submissions(executor, expected, loop):
    submitted = []
    ready = asyncio.Event()
    original_submit = executor.submit

    def tracked_submit(fn, /, *args, **kwargs):
        future = original_submit(fn, *args, **kwargs)
        submitted.append(future)
        if len(submitted) == expected:
            loop.call_soon_threadsafe(ready.set)
        return future

    executor.submit = tracked_submit
    return submitted, ready


@pytest.mark.asyncio
async def test_valid_real_pdf_returns_original_bytes_without_text_extraction(
    pdf_executor,
):
    data = make_pdf()
    upload = RecordingUploadFile(data)

    result = await validate_pdf_upload(upload, executor=pdf_executor)

    assert result == data
    assert result is not None
    assert upload.read_sizes == [settings.max_file_size_mb * 1024 * 1024 + 1]


@pytest.mark.parametrize(
    ("content_type", "data", "exception"),
    [
        ("application/octet-stream", b"not pdf", InvalidFileTypeError),
        ("application/pdf", b"", EmptyFileError),
    ],
)
@pytest.mark.asyncio
async def test_rejects_invalid_mime_and_empty_file(
    content_type, data, exception, pdf_executor
):
    with pytest.raises(exception) as raised:
        await validate_pdf_upload(
            RecordingUploadFile(data, content_type), executor=pdf_executor
        )
    assert raised.value.detail


@pytest.mark.asyncio
async def test_file_at_configured_size_limit_is_accepted(monkeypatch, pdf_executor):
    monkeypatch.setattr(settings, "max_file_size_mb", 1)
    pdf = make_pdf()
    data = pdf + b" " * (1024 * 1024 - len(pdf))

    assert len(data) == 1024 * 1024
    assert await validate_pdf_upload(
        RecordingUploadFile(data), executor=pdf_executor
    ) == data


@pytest.mark.asyncio
async def test_file_over_limit_is_rejected_from_bounded_read(
    monkeypatch, pdf_executor
):
    monkeypatch.setattr(settings, "max_file_size_mb", 1)
    upload = RecordingUploadFile(b"x" * (1024 * 1024 + 100))

    with pytest.raises(FileTooLargeError):
        await validate_pdf_upload(upload, executor=pdf_executor)

    assert upload.read_sizes == [1024 * 1024 + 1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    [
        b"not a pdf",
        b"%PDF-1.7\nthis is not a structurally valid PDF",
    ],
)
async def test_signature_or_structure_failure_is_422(data, pdf_executor):
    with pytest.raises(ExtractionError) as raised:
        await validate_pdf_upload(RecordingUploadFile(data), executor=pdf_executor)
    assert "PyMuPDF" not in raised.value.detail


@pytest.mark.asyncio
async def test_unexpected_validation_error_is_not_reclassified_as_invalid_pdf(
    pdf_executor,
):
    data = make_pdf()
    upload = RecordingUploadFile(data)
    error = RuntimeError("unexpected infrastructure failure")

    with patch(
        "src.api.pdf_validation._validate_pdf_structure",
        side_effect=error,
    ):
        with pytest.raises(RuntimeError) as raised:
            await validate_pdf_upload(upload, executor=pdf_executor)

    assert raised.value is error


@pytest.mark.asyncio
async def test_structural_validation_runs_outside_event_loop_thread(pdf_executor):
    data = make_pdf()
    loop_thread = threading.get_ident()
    validation_threads = []

    def record_thread(pdf_bytes):
        validation_threads.append(threading.get_ident())
        return None

    with patch(
        "src.api.pdf_validation._validate_pdf_structure",
        side_effect=record_thread,
    ):
        assert await validate_pdf_upload(
            RecordingUploadFile(data), executor=pdf_executor
        ) == data

    assert len(validation_threads) == 1
    assert validation_threads[0] != loop_thread


@pytest.mark.asyncio
async def test_document_is_closed_even_when_structural_validation_fails(
    monkeypatch, pdf_executor
):
    data = make_pdf()
    real_open = pymupdf.open
    opened = []

    class TrackedDocument:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.closed = False

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def close(self):
            self.closed = True
            self.wrapped.close()

    def tracked_open(*args, **kwargs):
        doc = TrackedDocument(real_open(*args, **kwargs))
        opened.append(doc)
        return doc

    monkeypatch.setattr(pymupdf, "open", tracked_open)
    monkeypatch.setattr(
        "src.api.pdf_validation._inspect_pdf_document",
        lambda doc: (_ for _ in ()).throw(pymupdf.FileDataError("bad PDF")),
    )

    with pytest.raises(ExtractionError):
        await validate_pdf_upload(
            RecordingUploadFile(data), executor=pdf_executor
        )

    assert len(opened) == 1
    assert opened[0].closed is True


@pytest.mark.asyncio
async def test_validation_does_not_extract_text_or_call_ms2(pdf_executor):
    data = make_pdf()
    with patch.object(pymupdf.Page, "get_text", side_effect=AssertionError):
        with patch(
            "src.services.document_service.DocumentService.call_extraction_service",
            side_effect=AssertionError,
        ):
            assert await validate_pdf_upload(
                RecordingUploadFile(data), executor=pdf_executor
            ) == data


@pytest.mark.asyncio
async def test_concurrent_validations_are_serial_on_one_dedicated_thread(
    pdf_executor,
):
    loop = asyncio.get_running_loop()
    all_started = asyncio.Event()
    release = threading.Event()
    active = 0
    maximum_active = 0
    threads = []
    lock = threading.Lock()
    submitted, all_submitted = signal_after_submissions(
        pdf_executor, expected=4, loop=loop
    )

    def controlled_inspection(_data):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
            threads.append(threading.get_ident())
            if len(threads) == 1:
                loop.call_soon_threadsafe(all_started.set)
        if not release.wait(timeout=3):
            raise TimeoutError("test did not release the inspection")
        with lock:
            active -= 1

    data = make_pdf()
    with patch(
        "src.api.pdf_validation._validate_pdf_structure",
        side_effect=controlled_inspection,
    ):
        tasks = [
            asyncio.create_task(
                validate_pdf_upload(
                    RecordingUploadFile(data), executor=pdf_executor
                )
            )
            for _ in range(4)
        ]
        await asyncio.wait_for(all_submitted.wait(), timeout=2)
        await asyncio.wait_for(all_started.wait(), timeout=2)
        assert len(submitted) == 4
        assert len(threads) == 1
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)

    assert maximum_active == 1
    assert len(set(threads)) == 1


@pytest.mark.asyncio
async def test_cancelled_active_inspection_continues_and_blocks_the_next(
    pdf_executor,
):
    loop = asyncio.get_running_loop()
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    release_first = threading.Event()
    calls = []
    _, second_submitted = signal_after_submissions(
        pdf_executor, expected=2, loop=loop
    )

    def controlled_inspection(data):
        calls.append(data)
        if len(calls) == 1:
            loop.call_soon_threadsafe(first_started.set)
            if not release_first.wait(timeout=3):
                raise TimeoutError("test did not release the first inspection")
        else:
            loop.call_soon_threadsafe(second_started.set)

    data = make_pdf()
    with patch(
        "src.api.pdf_validation._validate_pdf_structure",
        side_effect=controlled_inspection,
    ):
        first = asyncio.create_task(
            validate_pdf_upload(RecordingUploadFile(data), executor=pdf_executor)
        )
        await asyncio.wait_for(first_started.wait(), timeout=2)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        second = asyncio.create_task(
            validate_pdf_upload(RecordingUploadFile(data), executor=pdf_executor)
        )
        await asyncio.wait_for(second_submitted.wait(), timeout=2)
        assert not second_started.is_set()
        release_first.set()
        await asyncio.wait_for(second_started.wait(), timeout=2)
        assert await asyncio.wait_for(second, timeout=2) == data


@pytest.mark.asyncio
async def test_cancelled_queued_inspection_is_discarded(pdf_executor):
    loop = asyncio.get_running_loop()
    first_started = asyncio.Event()
    release_first = threading.Event()
    calls = []
    _, second_submitted = signal_after_submissions(
        pdf_executor, expected=2, loop=loop
    )

    def controlled_inspection(data):
        calls.append(data)
        if len(calls) == 1:
            loop.call_soon_threadsafe(first_started.set)
            if not release_first.wait(timeout=3):
                raise TimeoutError("test did not release the active inspection")

    data = make_pdf()
    with patch(
        "src.api.pdf_validation._validate_pdf_structure",
        side_effect=controlled_inspection,
    ):
        first = asyncio.create_task(
            validate_pdf_upload(RecordingUploadFile(data), executor=pdf_executor)
        )
        await asyncio.wait_for(first_started.wait(), timeout=2)
        queued = asyncio.create_task(
            validate_pdf_upload(RecordingUploadFile(data), executor=pdf_executor)
        )
        await asyncio.wait_for(second_submitted.wait(), timeout=2)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        release_first.set()
        assert await asyncio.wait_for(first, timeout=2) == data
        await event_loop_checkpoint()

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_executor_shutdown_is_nonblocking_and_rejects_new_submissions(
    pdf_executor,
):
    loop = asyncio.get_running_loop()
    inspection_started = asyncio.Event()
    release = threading.Event()

    def blocked_inspection(_data):
        loop.call_soon_threadsafe(inspection_started.set)
        if not release.wait(timeout=3):
            raise TimeoutError("test did not release the inspection")

    data = make_pdf()
    with patch(
        "src.api.pdf_validation._validate_pdf_structure",
        side_effect=blocked_inspection,
    ):
        operation = asyncio.create_task(
            validate_pdf_upload(RecordingUploadFile(data), executor=pdf_executor)
        )
        await asyncio.wait_for(inspection_started.wait(), timeout=2)
        closing = asyncio.create_task(pdf_executor.aclose())
        await event_loop_checkpoint()
        assert not closing.done()
        assert pdf_executor.closed
        with pytest.raises(RuntimeError):
            await validate_pdf_upload(
                RecordingUploadFile(data), executor=pdf_executor
            )
        release.set()
        await asyncio.wait_for(asyncio.gather(operation, closing), timeout=5)


@pytest.mark.asyncio
async def test_cancelled_shutdown_waiter_still_waits_for_executor_release(pdf_executor):
    loop = asyncio.get_running_loop()
    inspection_started = asyncio.Event()
    release = threading.Event()

    def blocked_inspection(_data):
        loop.call_soon_threadsafe(inspection_started.set)
        if not release.wait(timeout=3):
            raise TimeoutError("test did not release the inspection")

    data = make_pdf()
    with patch(
        "src.api.pdf_validation._validate_pdf_structure",
        side_effect=blocked_inspection,
    ):
        operation = asyncio.create_task(
            validate_pdf_upload(RecordingUploadFile(data), executor=pdf_executor)
        )
        await asyncio.wait_for(inspection_started.wait(), timeout=2)
        closing = asyncio.create_task(pdf_executor.aclose())
        await event_loop_checkpoint()
        closing.cancel()
        await event_loop_checkpoint()
        assert not closing.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(closing, timeout=5)
        assert operation.done()
    assert pdf_executor.closed


@pytest.mark.asyncio
async def test_each_lifespan_cycle_creates_and_closes_a_fresh_pdf_executor(
    monkeypatch,
):
    import src.main as main
    from src.database.mongodb import db_connection

    async def suspend_connect():
        await asyncio.Future()

    monkeypatch.setattr(main, "retry_connect", suspend_connect)
    monkeypatch.setattr(db_connection, "stop_supervision", AsyncMock())
    monkeypatch.setattr(db_connection, "disconnect", AsyncMock())
    application = FastAPI()
    first_executor = None

    async with main.lifespan(application):
        first_executor = application.state.pdf_validation_executor
    assert first_executor.closed

    async with main.lifespan(application):
        second_executor = application.state.pdf_validation_executor
        assert second_executor is not first_executor
    assert second_executor.closed

    third_executor = None
    with pytest.raises(RuntimeError, match="lifespan failure"):
        async with main.lifespan(application):
            third_executor = application.state.pdf_validation_executor
            raise RuntimeError("lifespan failure")
    assert third_executor is not second_executor
    assert third_executor.closed
