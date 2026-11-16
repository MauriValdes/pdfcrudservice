"""Configuración centralizada de logging del servicio (Factor 11 — 12-Factor App).

Este módulo es la única fuente de configuración de logging de la aplicación.
Se invoca una sola vez, al importar el módulo de entrada de la aplicación:

    from src.core.logging_config import configure_logging
    configure_logging("ms3-crud")

Características:

- Solo biblioteca estándar de Python (logging, logging.config, json, ...).
- Salida estructurada en JSON, una línea por registro.
- Los registros de nivel INFO y DEBUG van a stdout; WARNING y superiores a
  stderr. Nunca se escribe a archivos (requisito 12-Factor).
- El nivel se resuelve desde la variable de entorno LOG_LEVEL (o desde el
  argumento explícito `level`, que tiene precedencia).
- Un nivel inválido NUNCA provoca una excepción: se degrada a INFO y se
  emite un warning en JSON por stderr.
- No se registran credenciales: las URI con usuario/clave se redactan.

Advertencia: configure_logging() reemplaza la configuración global de logging
mediante logging.config.dictConfig(), lo que CIERRA cualquier handler
registrado previamente en el proceso (incluidos los de integraciones de APM,
OpenTelemetry o Sentry). Debe invocarse antes de registrar esos handlers.

Alcance de la protección de datos sensibles
--------------------------------------------
La redacción implementada en este módulo es una BARRERA ADICIONAL de primer
nivel, NO una garantía absoluta. Oculta:

- Credenciales embebidas en una URI (mongodb://usuario:clave@host:27017).
- Valores cuyas claves son sensibles, incluidas las variantes compuestas
  habituales ("db_password", "api_key_v2", "x-auth-token"), tanto en primer
  nivel como anidadas en diccionarios y listas.
- Los mismos patrones dentro de diccionarios y listas anidados.
- Los mensajes interpolados con %s (incluidos los diccionarios usados como
  argumento), los tracebacks completos y el stack info.
- Los valores no textuales (bytes u objetos con __str__) cuya
  representación textual exponga una URI con credenciales.

NO puede ocultar, y por lo tanto no debe considerarse que lo haga:

- Tokens opacos sin forma de URI (por ejemplo "Bearer eyJhbGciOi...").
- Credenciales concatenadas o transformadas que ya no coinciden con el
  patrón de URI.
- El contenido de los documentos (texto extraído de un PDF), que puede
  contener datos personales.
- Secretos escritos fuera de este módulo (un print, un logging.basicConfig
  posterior o un escritor a archivo).

La defensa real es la disciplina de no registrar información sensible; este
módulo es la red de seguridad que la hace menos probable.
"""

from __future__ import annotations

import json
import logging
import logging.config
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any, Mapping

# ---------------------------------------------------------------------------
# Constantes de entorno y valores por defecto
# ---------------------------------------------------------------------------

ENV_LOG_LEVEL = "LOG_LEVEL"
ENV_SERVICE_NAME = "SERVICE_NAME"

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_SERVICE_NAME = "unknown-service"

# Conjunto canónico de niveles. Cualquier otro valor es inválido y se
# degrada a DEFAULT_LOG_LEVEL (incluidos TRACE, WARN, FATAL y NOTSET).
VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

# Límite exclusivo del handler de stdout: los registros con nivel >= WARNING
# van exclusivamente a stderr, de modo que nunca se emiten duplicados.
STDOUT_HANDLER_MAX_LEVEL = logging.WARNING

REDACTED = "***"

# ---------------------------------------------------------------------------
# Política de datos sensibles
# ---------------------------------------------------------------------------

SENSITIVE_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "database_url",
        "mongo_uri",
        "mongo_url",
        "passwd",
        "password",
        "private_key",
        "proxy-authorization",
        "pwd",
        "refresh_token",
        "secret",
        "set-cookie",
        "token",
        "x-api-key",
    }
)

# URI con credenciales embebidas: mongodb://usuario:clave@host:puerto/...
_URI_CREDENTIALS_RE = re.compile(
    r"(?P<scheme>[a-zA-Z][\w+.-]*://)(?P<credentials>[^/@\s]+)@"
)

# Normalización de claves para detectar variantes de nombres sensibles.
# Se eliminan los separadores ("db_password" -> "dbpassword") y se pasan a
# minúsculas, de modo que la comparación sea estable y predecible.
_KEY_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")

# Tokens sensibles buscados dentro de una clave normalizada. Cubre las
# variantes habituales que no aparecen literalmente en SENSITIVE_KEYS
# ("db_password", "api_key_v2", "x-auth-token", "refreshToken", ...).
# Se listan de forma exhaustiva y simple: no se intenta detectar secretos
# por heurística, solo por nombre de clave.
SENSITIVE_KEY_TOKENS = (
    "password",
    "passwd",
    "pwd",
    "secret",
    "token",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "privatekey",
)

_SENSITIVE_KEYS_NORMALIZED = frozenset(
    _KEY_NORMALIZE_RE.sub("", str(key).lower()) for key in SENSITIVE_KEYS
)


def is_sensitive_key(key: Any) -> bool:
    """Indica si una clave debe tratarse como sensible.

    Criterio simple y predecible, basado únicamente en el nombre de la clave:

    1. La clave normalizada (minúsculas y sin separadores) coincide con
       alguna de las claves catalogadas en SENSITIVE_KEYS.
    2. La clave normalizada contiene alguno de los tokens de
       SENSITIVE_KEY_TOKENS, lo que cubre variantes habituales como
       "db_password", "api_key_v2" o "x-auth-token".
    """
    normalized = _KEY_NORMALIZE_RE.sub("", str(key).lower())
    if normalized in _SENSITIVE_KEYS_NORMALIZED:
        return True
    return any(token in normalized for token in SENSITIVE_KEY_TOKENS)


def redact_uri(uri: str | None) -> str:
    """Oculta las credenciales embebidas en una URI.

    Conserva el esquema y el host para que el log siga siendo útil para
    diagnosticar, y no solo el texto del mensaje (aplicable también a
    tracebacks completos y a stack info).

    >>> redact_uri("mongodb://usuario:clave@db:27017/?authSource=admin")
    'mongodb://***:***@db:27017/?authSource=admin'
    """
    if not uri:
        return ""
    return _URI_CREDENTIALS_RE.sub(rf"\g<scheme>{REDACTED}:{REDACTED}@", str(uri))


def redact_value(key: Any, value: Any) -> Any:
    """Redacta un valor de forma recursiva.

    Reglas aplicadas en orden:

    1. Si la clave es sensible (ver is_sensitive_key), todo el valor se
       reemplaza por "***".
    2. Si el valor es texto, se ocultan las credenciales de una URI.
    3. Si el valor es un diccionario, se aplica la misma regla a cada
       entrada, conservando la estructura del contexto.
    4. Si el valor es una lista, tupla o conjunto, se aplica la misma regla
       a cada elemento, conservando la estructura.
    5. Cualquier otro valor (bytes u objetos con __str__) se convierte a
       texto y se redacta: su representación textual podría exponer una
       URI con credenciales.

    La estructura del contexto se conserva: solo se sustituyen los valores
    sensibles, nunca el contenedor que los contiene.
    """
    if is_sensitive_key(key):
        return REDACTED
    if isinstance(value, str):
        return redact_uri(value)
    if isinstance(value, Mapping):
        return {str(k): redact_value(k, v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [redact_value(key, item) for item in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return redact_uri(bytes(value).decode("utf-8", errors="replace"))
    if value is None or isinstance(value, (bool, int, float)):
        return value
    # Objeto de cualquier otro tipo: se serializa a texto y se redacta,
    # porque su __str__ puede exponer una URI con credenciales.
    return redact_uri(str(value))


def redact_mapping(mapping: Mapping[str, Any] | None) -> dict[str, Any]:
    """Devuelve una copia del mapping con los valores sensibles redactados."""
    if not mapping:
        return {}
    return {str(key): redact_value(str(key), value) for key, value in mapping.items()}


def _jsonable(value: Any) -> Any:
    """Convierte un valor a algo representable en JSON sin romper el log."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


def _redact_record_args(args: Any) -> Any:
    """Redacta los argumentos de interpolación de un LogRecord.

    logging admite dos formas de argumentos: una tupla posicional
    ("%s", valor) o un diccionario ("%(nombre)s", {"nombre": valor}).
    En ambos casos se aplica la redacción recursiva antes de interpolar,
    para que un diccionario sensible no termine representado como texto
    sin revisar sus claves.
    """
    if not args:
        return args
    if isinstance(args, Mapping):
        return redact_value("", args)
    if isinstance(args, tuple):
        return tuple(redact_value("", arg) for arg in args)
    return args


def _format_record_message(record: logging.LogRecord) -> str:
    """Construye el mensaje de un registro con los argumentos ya redactados.

    Replica el comportamiento de logging.LogRecord.getMessage() (str del
    formato aplicado a los argumentos), pero interpola los argumentos ya
    redactados. Los mensajes sin argumentos no se modifican.
    """
    msg = record.msg
    args = record.args
    if not args:
        return redact_uri(str(msg))
    try:
        return redact_uri(str(msg) % _redact_record_args(args))
    except Exception:  # pragma: no cover - defensivo
        return redact_uri(str(msg))


# ---------------------------------------------------------------------------
# Formateadores
# ---------------------------------------------------------------------------


class JsonFormatter(logging.Formatter):
    """Emite un objeto JSON por línea y por registro.

    Campos obligatorios: timestamp, level, service, logger y message.
    Se agregan los campos contextuales pasados vía extra={...} y, solo
    cuando corresponde, los campos exception y stack.
    """

    # Atributos internos de logging.LogRecord que nunca deben emitirse.
    # Se calculan dinámicamente para tolerar las diferencias entre las
    # versiones de Python soportadas (3.12 en Docker, 3.14 en desarrollo).
    RESERVED = frozenset(
        set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)
        | {"message", "asctime", "taskName"}
    )

    # Nombres reservados en el JSON: un valor de extra con estos nombres
    # no puede sobrescribir el campo real; se conserva en el objeto "extra".
    RESERVED_JSON_FIELDS = frozenset(
        {"timestamp", "level", "service", "logger", "message", "exception", "stack"}
    )

    def __init__(
        self,
        service_name: str | None = None,
        *,
        ensure_ascii: bool = False,
    ) -> None:
        super().__init__()
        self._service_name = service_name
        self._ensure_ascii = ensure_ascii

    @property
    def service_name(self) -> str:
        """Nombre del servicio: explícito > variable de entorno > por defecto."""
        return (
            self._service_name
            or os.getenv(ENV_SERVICE_NAME)
            or DEFAULT_SERVICE_NAME
        )

    def format(self, record: logging.LogRecord) -> str:
        # El mensaje se construye con los argumentos ya redactados, de modo
        # que un diccionario sensible no quede expuesto en su forma textual.
        message = _format_record_message(record)

        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            )
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "service": self.service_name,
            "logger": record.name,
            "message": message,
        }
        mandatory = dict(payload)

        collisions: dict[str, Any] = {}
        for key, value in record.__dict__.items():
            if key in self.RESERVED or key.startswith("_"):
                continue
            if key in self.RESERVED_JSON_FIELDS:
                # Colisión: el campo obligatorio gana; el valor colidente se
                # conserva (redactado) dentro del objeto "extra".
                collisions[key] = _jsonable(redact_value(key, value))
                continue
            payload[key] = _jsonable(redact_value(key, value))

        if record.exc_info:
            # formatException incluye la cadena completa de excepciones
            # (__cause__ / __context__), no solo str(exc).
            payload["exception"] = redact_uri(self.formatException(record.exc_info))
        if record.stack_info:
            payload["stack"] = redact_uri(self.formatStack(record.stack_info))

        # Los campos obligatorios prevalecen ante cualquier colisión.
        payload.update(mandatory)
        if collisions:
            payload["extra"] = collisions

        return json.dumps(payload, ensure_ascii=self._ensure_ascii, default=str)


# ---------------------------------------------------------------------------
# Handlers y filtros
# ---------------------------------------------------------------------------


class DynamicStreamHandler(logging.StreamHandler):
    """Handler que resuelve el stream en cada emisión.

    La forma habitual de configurarlo ("ext://sys.stdout") fija el objeto
    stream en el momento de aplicar la configuración: si la aplicación
    reemplaza sys.stdout después (por ejemplo un supervisor o la captura de
    salida de los tests), los logs seguirían yendo al stream original.

    Resolviendo el stream dinámicamente, el handler sigue al stream vigente
    en cada momento, sin perder la separación entre stdout y stderr.
    """

    def __init__(self, stream_name: str = "stdout") -> None:
        super().__init__()
        self._stream_name = stream_name

    @property
    def stream(self):
        return getattr(sys, self._stream_name)

    @stream.setter
    def stream(self, value) -> None:
        # logging.StreamHandler.__init__ asigna self.stream = stream; la
        # propiedad resuelve el stream de forma dinámica, por lo que la
        # asignación se ignora intencionadamente.
        pass


class MaxLevelFilter(logging.Filter):
    """Deja pasar únicamente los registros con nivel estrictamente menor.

    Se aplica al handler de stdout para que un WARNING no se emita dos
    veces (una por stdout y otra por stderr).
    """

    def __init__(self, max_level: int = STDOUT_HANDLER_MAX_LEVEL) -> None:
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < self.max_level


class RedactionFilter(logging.Filter):
    """Redacta claves sensibles de primer nivel en los campos extra.

    Complementa a la redacción recursiva de JsonFormatter: al actuar sobre
    el registro, el valor queda redactado para cualquier handler, incluso
    los que agreguen terceros.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        for key in list(record.__dict__):
            if is_sensitive_key(key):
                record.__dict__[key] = REDACTED
        return True


# ---------------------------------------------------------------------------
# Resolución del nivel
# ---------------------------------------------------------------------------


def _resolve_log_level_info(
    value: str | int | None = None,
) -> tuple[int, bool, str | None]:
    """Resuelve el nivel y devuelve (nivel, es_valido, valor_original).

    Precedencia: argumento explícito > variable de entorno LOG_LEVEL > INFO.

    Nunca lanza excepción: un valor inválido degrada a INFO y se informa
    mediante el booleano devuelto, para que el warning pueda emitirse una
    vez configurados los handlers.
    """
    raw = value if value is not None else os.getenv(ENV_LOG_LEVEL)

    if raw is None:
        return logging.INFO, True, None

    if isinstance(raw, bool):
        return logging.INFO, False, str(raw)

    if isinstance(raw, int):
        return raw, True, None

    text = str(raw).strip()
    if not text:
        return logging.INFO, True, None

    normalized = text.upper()
    if normalized in VALID_LOG_LEVELS:
        return logging.getLevelName(normalized), True, text

    return logging.INFO, False, text


def resolve_log_level(value: str | int | None = None) -> int:
    """Resuelve el nivel de logging a partir de un valor o de LOG_LEVEL.

    Devuelve el nivel numérico efectivo. Un valor inválido degrada a INFO
    sin lanzar excepción.
    """
    level, _is_valid, _raw = _resolve_log_level_info(value)
    return level


# ---------------------------------------------------------------------------
# Construcción y aplicación de la configuración
# ---------------------------------------------------------------------------


def build_logging_config(
    service_name: str,
    *,
    level: str | int | None = None,
) -> dict[str, Any]:
    """Construye el diccionario de configuración para logging.config.dictConfig.

    Se expone separadamente de configure_logging() para poder inspeccionarla
    y verificarla en los tests sin mutar el estado global de logging.
    """
    resolved_level, _is_valid, _raw = _resolve_log_level_info(level)
    level_name = logging.getLevelName(resolved_level)

    return {
        "version": 1,
        # Imprescindible: con True (valor por defecto de dictConfig) se
        # silenciarían los loggers ya existentes en la aplicación.
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": JsonFormatter,
                "service_name": service_name,
            }
        },
        "filters": {
            "stdout_only": {
                "()": MaxLevelFilter,
                "max_level": STDOUT_HANDLER_MAX_LEVEL,
            },
            "redaction": {"()": RedactionFilter},
        },
        "handlers": {
            "stdout": {
                "()": DynamicStreamHandler,
                "stream_name": "stdout",
                "formatter": "default",
                "filters": ["stdout_only", "redaction"],
                "level": "NOTSET",
            },
            "stderr": {
                "()": DynamicStreamHandler,
                "stream_name": "stderr",
                "formatter": "default",
                "filters": ["redaction"],
                "level": "WARNING",
            },
        },
        "root": {"level": level_name, "handlers": ["stdout", "stderr"]},
        "loggers": {
            # Se vacían los handlers propios de uvicorn y se fuerza la
            # propagación al root para que todo pase por el formateador
            # JSON, sin duplicados. El nivel se fija explícitamente porque
            # el nivel del root no filtra los registros propagados desde un
            # logger hijo.
            "uvicorn": {"level": level_name, "handlers": [], "propagate": True},
            "uvicorn.error": {"level": level_name, "handlers": [], "propagate": True},
            "uvicorn.access": {"level": level_name, "handlers": [], "propagate": True},
        },
    }


def _force_utf8_streams() -> None:
    """Evita UnicodeEncodeError en consolas no UTF-8 (Windows cp1252).

    Sin esto, un log con acentos o emojis puede romper el proceso cuando
    stdout está redirigido a un archivo fuera de Docker.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, ValueError, OSError):  # pragma: no cover
            pass


def configure_logging(
    service_name: str,
    *,
    level: str | int | None = None,
) -> None:
    """Configura el logging de la aplicación. Idempotente.

    Args:
        service_name: Identificador del servicio (por ejemplo "ms3-crud").
            Se incluye en cada registro para poder identificar el origen.
        level: Nivel explícito. Si es None se toma de LOG_LEVEL. Tiene
            precedencia sobre la variable de entorno.

    Notes:
        - Los registros previos a esta llamada conservan su configuración.
        - Es seguro llamarla más de una vez: dictConfig reemplaza los
          handlers del root en lugar de acumularlos.
        - EFECTO GLOBAL: logging.config.dictConfig() no solo reemplaza los
          handlers del root logger, sino que además CIERRA todos los
          handlers registrados previamente en el proceso (los que
          logging.shutdown() alcanza a través de logging._handlerList),
          incluidos los pertenecientes a loggers de terceros y los que
          hubiera agregado un APM, OpenTelemetry, Sentry u otra integración
          similar. Por eso la inicialización del logging centralizado
          debe realizarse ANTES de registrar cualquier handler de terceros:
          configure_logging() no debe invocarse después de configurar
          esas integraciones, o sus handlers quedarían cerrados y
          dejarían de recibir eventos.
    """
    resolved_level, is_valid, raw = _resolve_log_level_info(level)

    _force_utf8_streams()

    config = build_logging_config(service_name, level=resolved_level)
    logging.config.dictConfig(config)

    if not is_valid:
        # Se emite DESPUÉS de configurar los handlers para que el warning
        # salga en JSON y por stderr, igual que el resto de los registros.
        logging.getLogger(__name__).warning(
            "LOG_LEVEL invalido %r; se usara %s", raw, DEFAULT_LOG_LEVEL
        )
