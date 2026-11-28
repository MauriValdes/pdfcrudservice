"""Bounded validation of PDF uploads before asynchronous reception."""

import asyncio
import threading
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Any, Callable

import pymupdf
from fastapi import UploadFile

from src.config.config import settings
from src.core.exceptions import (
    EmptyFileError,
    ExtractionError,
    FileTooLargeError,
    InvalidFileTypeError,
)


class PdfValidationExecutor(Executor):
    """Serial, lifecycle-managed executor for all PyMuPDF work in this process."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="pymupdf",
        )
        self._lock = threading.Lock()
        self._closed = False

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def submit(
        self,
        fn: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ):
        with self._lock:
            if self._closed:
                raise RuntimeError("PDF validation executor is shutting down")
            return self._executor.submit(fn, *args, **kwargs)

    def begin_shutdown(self) -> None:
        """Reject new work before asynchronous shutdown waiting begins."""
        with self._lock:
            self._closed = True

    def shutdown(
        self,
        wait: bool = True,
        *,
        cancel_futures: bool = False,
    ) -> None:
        self.begin_shutdown()
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    async def aclose(self) -> None:
        """Cancel queued work and await active work without blocking the loop."""
        self.begin_shutdown()
        # This thread only waits for the PyMuPDF worker to finish; it does not
        # perform PyMuPDF operations and therefore may use asyncio's default pool.
        shutdown_task = asyncio.create_task(
            asyncio.to_thread(
                self.shutdown,
                wait=True,
                cancel_futures=True,
            )
        )
        cancellation = None
        while not shutdown_task.done():
            try:
                await asyncio.shield(shutdown_task)
            except asyncio.CancelledError as error:
                # Keep waiting for the real executor shutdown before allowing
                # lifespan cancellation to escape.
                if cancellation is None:
                    cancellation = error
        shutdown_task.result()
        if cancellation is not None:
            raise cancellation


def _inspect_pdf_document(document: pymupdf.Document) -> None:
    """Force a lightweight structural read without extracting page content."""
    if not document.is_pdf:
        raise pymupdf.FileDataError("Input is not a PDF document")

    # Encrypted documents open without a password but are not authenticated.
    # Accept them here rather than imposing a password policy at upload time.
    if not document.needs_pass:
        _ = document.page_count


def _validate_pdf_structure(file_bytes: bytes) -> None:
    """Open and inspect a PDF with PyMuPDF, always closing a successful open."""
    document = None
    try:
        try:
            document = pymupdf.open(stream=file_bytes, filetype="pdf")
            _inspect_pdf_document(document)
        except pymupdf.FileDataError:
            raise ExtractionError() from None
    finally:
        if document is not None:
            document.close()


async def validate_pdf_upload(
    upload: UploadFile,
    *,
    executor: PdfValidationExecutor,
) -> bytes:
    """Validate an uploaded PDF and return its exact original bytes.

    The upload is read once, with a maximum of the configured limit plus one
    byte. PyMuPDF runs serially in the dedicated executor. Cancelling a request
    cannot stop an inspection that has already started; it continues to occupy
    the sole worker, and pending requests remain queued until it finishes.
    """
    if upload.content_type != "application/pdf":
        raise InvalidFileTypeError()

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    file_bytes = await upload.read(max_bytes + 1)

    if not file_bytes:
        raise EmptyFileError()
    if len(file_bytes) > max_bytes:
        raise FileTooLargeError(
            f"El archivo supera el tamaño máximo de {settings.max_file_size_mb} MB"
        )
    if not file_bytes.startswith(b"%PDF-"):
        raise ExtractionError()

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(executor, _validate_pdf_structure, file_bytes)
    return file_bytes
