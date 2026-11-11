"""Tests unitarios de la integración HTTP síncrona MS3 -> MS2.

Cobertura:
- call_extraction_service(): comunicación HTTP con MS2 vía httpx.MockTransport
  (sin levantar MS2 ni realizar llamadas HTTP reales).
- process_pdf_extraction(): orquestación, estados COMPLETED/FAILED y persistencia.
"""
import httpx
import pytest
from unittest.mock import AsyncMock

import src.services.document_service as ds
from src.services.document_service import DocumentService
from src.schemas.document import DocumentUpdate


VALID_URL = "http://localhost:8002/extract"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def build_service(persist_error=None):
    """Crea un DocumentService con repositorio mockeado.

    Args:
        persist_error: excepción que lanzará el repositorio al intentar
            persistir una actualización (None por defecto).
    Returns:
        (service, repository)
    """
    repository = AsyncMock()
    if persist_error is not None:
        repository.update_document.side_effect = persist_error
    service = DocumentService(repository=repository)
    return service, repository


def install_mock_transport(monkeypatch, handler, captured_kwargs=None):
    """Reemplaza httpx.AsyncClient por una fábrica que inyecta MockTransport.

    No se realizan llamadas HTTP reales: el transport mockeado invoca
    `handler(request)` localmente.
    """
    real_client = httpx.AsyncClient  # clase real, capturada antes del parche

    def client_factory(*args, **kwargs):
        if captured_kwargs is not None:
            captured_kwargs.append(kwargs)
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(ds.httpx, "AsyncClient", client_factory)


def make_request_recorder(captured_requests=None, status_code=200, json_body=None,
                          content_text=None):
    """Crea un handler de MockTransport que registra la petición y responde."""
    def handler(request: httpx.Request) -> httpx.Response:
        if captured_requests is not None:
            captured_requests.append(request)
        if json_body is not None:
            return httpx.Response(status_code, json=json_body)
        if content_text is not None:
            return httpx.Response(status_code, content=content_text.encode())
        return httpx.Response(status_code, json={})
    return handler


# ------------------------------------------------------------------
# call_extraction_service: respuestas exitosas
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_extraction_success_returns_content_and_page_count(monkeypatch):
    """HTTP 200 con content y page_count válidos retorna el dict parseado."""
    service, _ = build_service()
    requests = []
    handler = make_request_recorder(requests, json_body={
        "content": "Hola mundo", "page_count": 3,
    })
    install_mock_transport(monkeypatch, handler)

    result = await service.call_extraction_service(b"%PDF-1.4", "documento.pdf")

    assert result == {"content": "Hola mundo", "page_count": 3}


@pytest.mark.asyncio
async def test_call_extraction_allows_empty_content(monkeypatch):
    """content="" es válido (PDFs escaneados sin texto)."""
    service, _ = build_service()
    handler = make_request_recorder(json_body={"content": "", "page_count": 0})
    install_mock_transport(monkeypatch, handler)

    result = await service.call_extraction_service(b"%PDF-1.4", "scan.pdf")

    assert result["content"] == ""
    assert result["page_count"] == 0


@pytest.mark.asyncio
async def test_call_extraction_uses_extract_endpoint(monkeypatch):
    """La petición se envía al path /extract del servicio MS2."""
    service, _ = build_service()
    requests = []
    handler = make_request_recorder(requests, json_body={
        "content": "x", "page_count": 1,
    })
    install_mock_transport(monkeypatch, handler)

    await service.call_extraction_service(b"%PDF-1.4", "doc.pdf")

    request = requests[0]
    assert request.url.path == "/extract"


@pytest.mark.asyncio
async def test_call_extraction_sends_multipart_file_field(monkeypatch):
    """El cuerpo es multipart/form-data con el campo 'file'."""
    service, _ = build_service()
    requests = []
    handler = make_request_recorder(requests, json_body={
        "content": "x", "page_count": 1,
    })
    install_mock_transport(monkeypatch, handler)

    await service.call_extraction_service(b"%PDF-1.4 contenidos", "manual.pdf")

    request = requests[0]
    assert request.headers["content-type"].startswith("multipart/form-data")
    body = request.content
    assert b'name="file"' in body


@pytest.mark.asyncio
async def test_call_extraction_sends_correct_filename(monkeypatch):
    """El nombre del archivo se envía en la parte multipart."""
    service, _ = build_service()
    requests = []
    handler = make_request_recorder(requests, json_body={
        "content": "x", "page_count": 1,
    })
    install_mock_transport(monkeypatch, handler)

    await service.call_extraction_service(b"%PDF-1.4", "mi-informe-2026.pdf")

    request = requests[0]
    assert b'filename="mi-informe-2026.pdf"' in request.content


@pytest.mark.asyncio
async def test_call_extraction_uses_configured_base_url(monkeypatch):
    """La petición usa la URL base configurada + /extract."""
    service, _ = build_service()
    requests = []
    handler = make_request_recorder(requests, json_body={
        "content": "x", "page_count": 1,
    })
    install_mock_transport(monkeypatch, handler)

    await service.call_extraction_service(b"%PDF-1.4", "doc.pdf")

    request = requests[0]
    assert request.url.host == "localhost"
    assert request.url.port == 8002
    assert str(request.url) == VALID_URL


@pytest.mark.asyncio
async def test_call_extraction_respects_configured_timeout(monkeypatch):
    """El timeout configurado se transfiere al cliente HTTP."""
    service, _ = build_service()
    captured = []
    handler = make_request_recorder(json_body={"content": "x", "page_count": 1})
    install_mock_transport(monkeypatch, handler, captured)

    await service.call_extraction_service(b"%PDF-1.4", "doc.pdf")

    assert captured[0]["timeout"] == service.extract_service_timeout


# ------------------------------------------------------------------
# call_extraction_service: errores HTTP
# ------------------------------------------------------------------

@pytest.mark.parametrize("status_code,detail", [
    (400, "Formato de archivo inválido"),
    (413, "Archivo demasiado grande"),
    (422, "Campo file ausente"),
])
@pytest.mark.asyncio
async def test_call_extraction_http_client_errors_raise_valueerror_with_message(
    monkeypatch, status_code, detail
):
    """Los errores 4xx generan ValueError conservando el mensaje útil de MS2."""
    service, _ = build_service()
    handler = make_request_recorder(status_code=status_code, json_body={"detail": detail})
    install_mock_transport(monkeypatch, handler)

    with pytest.raises(ValueError) as exc_info:
        await service.call_extraction_service(b"%PDF-1.4", "doc.pdf")

    assert f"HTTP {status_code}" in str(exc_info.value)
    assert detail in str(exc_info.value)


@pytest.mark.parametrize("status_code", [500, 503])
@pytest.mark.asyncio
async def test_call_extraction_http_server_errors_raise_valueerror(
    monkeypatch, status_code
):
    """Los errores 5xx generan ValueError con mensaje genérico, sin exponer
    información interna enviada por MS2."""
    service, _ = build_service()
    handler = make_request_recorder(
        status_code=status_code,
        json_body={
            "detail": (
                f"INTERNAL_SECRET_DATABASE_HOST=host-{status_code} "
                "Traceback: AttributeError en módulo 'pymupdf_layout'"
            )
        },
    )
    install_mock_transport(monkeypatch, handler)

    with pytest.raises(ValueError) as exc_info:
        await service.call_extraction_service(b"%PDF-1.4", "doc.pdf")

    message = str(exc_info.value)
    # Excepción controlada que identifica el código HTTP recibido
    assert f"HTTP {status_code}" in message
    # Mensaje genérico incluido
    assert "no disponible" in message
    # Nada de la información interna de MS2 filtra al mensaje
    assert "INTERNAL_SECRET_DATABASE_HOST" not in message
    assert "Traceback" not in message
    assert "pymupdf_layout" not in message
    # Nunca viaja como excepción httpx, solo como ValueError controlado
    assert not isinstance(exc_info.value, httpx.HTTPError)


@pytest.mark.asyncio
async def test_call_extraction_timeout_propagates(monkeypatch):
    """Un timeout se propaga como httpx.TimeoutException."""
    service, _ = build_service()

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("Simulated timeout")

    install_mock_transport(monkeypatch, handler)

    with pytest.raises(httpx.TimeoutException):
        await service.call_extraction_service(b"%PDF-1.4", "doc.pdf")


@pytest.mark.asyncio
async def test_call_extraction_connect_error_propagates(monkeypatch):
    """Un fallo de conexión se propaga como httpx.ConnectError."""
    service, _ = build_service()

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("MS2 inalcanzable")

    install_mock_transport(monkeypatch, handler)

    with pytest.raises(httpx.ConnectError):
        await service.call_extraction_service(b"%PDF-1.4", "doc.pdf")


# ------------------------------------------------------------------
# call_extraction_service: respuestas inválidas
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_extraction_malformed_json_raises_valueerror(monkeypatch):
    service, _ = build_service()
    install_mock_transport(monkeypatch, make_request_recorder(
        content_text="{no es json válido",
    ))

    with pytest.raises(ValueError, match="no JSON"):
        await service.call_extraction_service(b"%PDF-1.4", "doc.pdf")


@pytest.mark.asyncio
async def test_call_extraction_empty_json_raises_valueerror(monkeypatch):
    """JSON vacío {}: content y page_count ausentes."""
    service, _ = build_service()
    install_mock_transport(monkeypatch, make_request_recorder(json_body={}))

    with pytest.raises(ValueError, match="content"):
        await service.call_extraction_service(b"%PDF-1.4", "doc.pdf")


@pytest.mark.asyncio
async def test_call_extraction_missing_content_raises_valueerror(monkeypatch):
    service, _ = build_service()
    install_mock_transport(monkeypatch, make_request_recorder(
        json_body={"page_count": 2},
    ))

    with pytest.raises(ValueError, match="content"):
        await service.call_extraction_service(b"%PDF-1.4", "doc.pdf")


@pytest.mark.asyncio
async def test_call_extraction_missing_page_count_raises_valueerror(monkeypatch):
    service, _ = build_service()
    install_mock_transport(monkeypatch, make_request_recorder(
        json_body={"content": "texto"},
    ))

    with pytest.raises(ValueError, match="page_count"):
        await service.call_extraction_service(b"%PDF-1.4", "doc.pdf")


@pytest.mark.asyncio
async def test_call_extraction_content_wrong_type_raises_valueerror(monkeypatch):
    service, _ = build_service()
    install_mock_transport(monkeypatch, make_request_recorder(
        json_body={"content": 12345, "page_count": 2},
    ))

    with pytest.raises(ValueError, match="content"):
        await service.call_extraction_service(b"%PDF-1.4", "doc.pdf")


@pytest.mark.asyncio
async def test_call_extraction_page_count_string_raises_valueerror(monkeypatch):
    service, _ = build_service()
    install_mock_transport(monkeypatch, make_request_recorder(
        json_body={"content": "texto", "page_count": "tres"},
    ))

    with pytest.raises(ValueError, match="page_count"):
        await service.call_extraction_service(b"%PDF-1.4", "doc.pdf")


@pytest.mark.parametrize("bool_value", [True, False])
@pytest.mark.asyncio
async def test_call_extraction_page_count_boolean_rejected(
    monkeypatch, bool_value
):
    """page_count=True y page_count=False NO son enteros válidos."""
    service, _ = build_service()
    install_mock_transport(monkeypatch, make_request_recorder(
        json_body={"content": "texto", "page_count": bool_value},
    ))

    with pytest.raises(ValueError, match="page_count"):
        await service.call_extraction_service(b"%PDF-1.4", "doc.pdf")


@pytest.mark.asyncio
async def test_call_extraction_page_count_float_rejected(monkeypatch):
    """page_count decimal (3.5) se rechaza."""
    service, _ = build_service()
    install_mock_transport(monkeypatch, make_request_recorder(
        json_body={"content": "texto", "page_count": 3.5},
    ))

    with pytest.raises(ValueError, match="page_count"):
        await service.call_extraction_service(b"%PDF-1.4", "doc.pdf")


@pytest.mark.asyncio
async def test_call_extraction_page_count_negative_rejected(monkeypatch):
    service, _ = build_service()
    install_mock_transport(monkeypatch, make_request_recorder(
        json_body={"content": "texto", "page_count": -1},
    ))

    with pytest.raises(ValueError, match="page_count"):
        await service.call_extraction_service(b"%PDF-1.4", "doc.pdf")


# ------------------------------------------------------------------
# process_pdf_extraction: orquestación y persistencia
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_process_extraction_success_updates_completed():
    """Extracción exitosa: se persiste COMPLETED con content y page_count."""
    service, repository = build_service()
    service.call_extraction_service = AsyncMock(
        return_value={"content": "Texto extraído", "page_count": 4}
    )

    await service.process_pdf_extraction("doc-1", b"%PDF-1.4", "doc.pdf")

    repository.update_document.assert_awaited_once_with(
        "doc-1",
        DocumentUpdate(status="completed", content="Texto extraído", page_count=4),
    )


@pytest.mark.asyncio
async def test_process_extraction_empty_content_still_completed():
    """Contenido vacío: COMPLETED y content='' (los escaneados son válidos)."""
    service, repository = build_service()
    service.call_extraction_service = AsyncMock(
        return_value={"content": "", "page_count": 0}
    )

    await service.process_pdf_extraction("doc-1", b"%PDF-1.4", "doc.pdf")

    repository.update_document.assert_awaited_once_with(
        "doc-1",
        DocumentUpdate(status="completed", content="", page_count=0),
    )


@pytest.mark.asyncio
async def test_process_extraction_http_error_marks_failed_and_reraises():
    """Error HTTP (ValueError de validación): FAILED con mensaje y re-lanzamiento."""
    service, repository = build_service()
    service.call_extraction_service = AsyncMock(
        side_effect=ValueError("MS2 retornó error HTTP 500: boom")
    )

    with pytest.raises(ValueError):
        await service.process_pdf_extraction("doc-1", b"%PDF-1.4", "doc.pdf")

    repository.update_document.assert_awaited_once_with(
        "doc-1",
        DocumentUpdate(status="failed", error_message="MS2 retornó error HTTP 500: boom"),
    )


@pytest.mark.asyncio
async def test_process_extraction_timeout_marks_failed_and_reraises():
    """Timeout: FAILED con mensaje y la excepción original se re-lanza."""
    service, repository = build_service()
    service.call_extraction_service = AsyncMock(
        side_effect=httpx.TimeoutException("Simulated timeout")
    )

    with pytest.raises(httpx.TimeoutException):
        await service.process_pdf_extraction("doc-1", b"%PDF-1.4", "doc.pdf")

    repository.update_document.assert_awaited_once_with(
        "doc-1",
        DocumentUpdate(status="failed", error_message="Simulated timeout"),
    )


@pytest.mark.asyncio
async def test_process_extraction_connect_error_marks_failed_and_reraises():
    """Error de conexión: FAILED con mensaje y la excepción original se re-lanza."""
    service, repository = build_service()
    service.call_extraction_service = AsyncMock(
        side_effect=httpx.ConnectError("MS2 inalcanzable")
    )

    with pytest.raises(httpx.ConnectError):
        await service.process_pdf_extraction("doc-1", b"%PDF-1.4", "doc.pdf")

    repository.update_document.assert_awaited_once_with(
        "doc-1",
        DocumentUpdate(status="failed", error_message="MS2 inalcanzable"),
    )


@pytest.mark.asyncio
async def test_process_extraction_invalid_response_marks_failed():
    """Respuesta inválida de MS2 (ValueError de validación): FAILED."""
    service, repository = build_service()
    service.call_extraction_service = AsyncMock(
        side_effect=ValueError("El campo 'page_count' debe ser entero >= 0, recibido: True")
    )

    with pytest.raises(ValueError):
        await service.process_pdf_extraction("doc-1", b"%PDF-1.4", "doc.pdf")

    repository.update_document.assert_awaited_once_with(
        "doc-1",
        DocumentUpdate(
            status="failed",
            error_message="El campo 'page_count' debe ser entero >= 0, recibido: True",
        ),
    )


@pytest.mark.asyncio
async def test_process_extraction_persist_completed_failure_propagates():
    """Fallo al persistir COMPLETED: la excepción MongoDB se propaga sin ocultar."""
    service, repository = build_service(persist_error=Exception("Mongo down"))
    service.call_extraction_service = AsyncMock(
        return_value={"content": "Texto", "page_count": 2}
    )

    with pytest.raises(Exception, match="Mongo down"):
        await service.process_pdf_extraction("doc-1", b"%PDF-1.4", "doc.pdf")

    # El intento de persistencia se realizó con los datos de extracción
    repository.update_document.assert_awaited_once_with(
        "doc-1",
        DocumentUpdate(status="completed", content="Texto", page_count=2),
    )


@pytest.mark.asyncio
async def test_process_extraction_persist_failed_failure_propagates_original_error():
    """Fallo al persistir FAILED: se propaga la excepción de persistencia
    conservando el error original de extracción como contexto."""
    service, repository = build_service(persist_error=Exception("Mongo down"))
    service.call_extraction_service = AsyncMock(
        side_effect=ValueError("MS2 retornó error HTTP 503")
    )

    with pytest.raises(Exception, match="Mongo down") as exc_info:
        await service.process_pdf_extraction("doc-1", b"%PDF-1.4", "doc.pdf")

    # La excepción de persistencia se propaga; el ValueError original
    # queda preservado como contextual (no se oculta ni se descarta).
    assert isinstance(exc_info.value.__context__, ValueError)
    assert "MS2 retornó error HTTP 503" in str(exc_info.value.__context__)
    # Se intentó registrar el estado FAILED antes de propagar
    repository.update_document.assert_awaited_once()
