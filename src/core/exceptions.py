"""Excepciones de dominio y manejo centralizado de errores.

SRP: este módulo concentra TODA la traducción de errores de negocio a
respuestas HTTP. Las capas de API y servicio simplemente levantan
excepciones de dominio (AppError y subclases); FastAPI las convierte a
JSON en un único lugar (app_error_handler), sin try/except repetidos
en cada endpoint.
"""
import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Error de aplicación con traducción HTTP explícita.

    Las subclases definen su status_code y un detalle por defecto;
    el handler centralizado hace el resto.
    """

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail: str = "Error interno del servidor"

    def __init__(self, detail: str | None = None):
        if detail is not None:
            self.detail = detail
        super().__init__(self.detail)


class InvalidFileTypeError(AppError):
    """El archivo recibido no es un PDF (MIME o extensión inválidos)."""

    status_code = status.HTTP_400_BAD_REQUEST
    detail = "El archivo debe ser un PDF"


class EmptyFileError(AppError):
    """El archivo recibido no contiene bytes."""

    status_code = status.HTTP_400_BAD_REQUEST
    detail = "El archivo está vacío"


class FileTooLargeError(AppError):
    """El archivo supera el tamaño máximo permitido por configuración."""

    status_code = status.HTTP_413_CONTENT_TOO_LARGE


class ExtractionError(AppError):
    """El PDF no pudo procesarse (firma inválida, archivo corrupto, etc.)."""

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    detail = "No se pudo extraer el texto del PDF (archivo inválido o corrupto)"


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Traduce cualquier AppError (y subclases) a una respuesta JSON uniforme."""
    logger.warning(
        "%s: %s %s -> %s",
        type(exc).__name__,
        request.method,
        request.url.path,
        exc.detail,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Registra los handlers centralizados en la aplicación.

    Starlette resuelve el handler recorriendo el MRO de la excepción,
    por lo que registrar la clase base AppError cubre todas las subclases.
    """
    app.add_exception_handler(AppError, app_error_handler)
