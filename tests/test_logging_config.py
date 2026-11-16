"""Tests unitarios del módulo central de logging (src/core/logging_config.py).

Verifican:
- Resolución de niveles válidos e inválidos (y el warning asociado).
- Configuración del root logger y de sus dos handlers.
- Separación entre stdout y stderr, sin duplicaciones.
- Formato JSON, campos obligatorios y timestamp UTC.
- Identificación del servicio y del logger.
- Propagación de atributos extra y exclusión de atributos internos.
- Excepciones, tracebacks (incluida la cadena de excepciones) y stack info.
- Interpolación de argumentos con %s.
- Redacción de URI, claves sensibles, diccionarios y listas anidados.
- Colisiones entre extra y campos obligatorios.
- Loggers creados antes de la configuración y reconfiguración repetida.
- Comportamiento de los loggers de uvicorn.
- Handler dinámico ante reasignación de stdout.

El estado global de logging se aísla y se restaura en cada test mediante la
fixture `isolate_logging_state`, de modo que la suite existente del servicio
no se ve afectada.
"""

from __future__ import annotations

import json
import logging
import sys

import pytest

from src.core.logging_config import (
    DEFAULT_LOG_LEVEL,
    REDACTED,
    SENSITIVE_KEYS,
    DynamicStreamHandler,
    JsonFormatter,
    MaxLevelFilter,
    build_logging_config,
    configure_logging,
    is_sensitive_key,
    redact_mapping,
    redact_uri,
    redact_value,
    resolve_log_level,
)

SERVICE_NAME = "ms3-crud"


# ---------------------------------------------------------------------------
# Aislamiento del estado global de logging
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolate_logging_state():
    """Guarda y restaura el estado global de logging tras cada test.

    configure_logging() reemplaza los handlers del root logger y de los
    loggers de uvicorn; sin este aislamiento esos cambios se filtrarían al
    resto de la suite.
    """
    state = capture_logging_state()
    try:
        yield
    finally:
        restore_logging_state(state)


def capture_logging_state() -> dict:
    """Captura el estado global de logging que debe restaurarse."""
    root = logging.getLogger()
    watched = ("uvicorn", "uvicorn.error", "uvicorn.access")

    streams = []
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        streams.append(
            (
                stream,
                getattr(stream, "encoding", None),
                getattr(stream, "errors", None),
            )
        )

    return {
        "root_handlers": list(root.handlers),
        "root_level": root.level,
        "root_filters": list(root.filters),
        "loggers": {
            name: (
                list(logging.getLogger(name).handlers),
                logging.getLogger(name).level,
                logging.getLogger(name).propagate,
                logging.getLogger(name).disabled,
            )
            for name in watched
        },
        "streams": streams,
    }


def _reopen_handler(handler) -> bool:
    """Reabre un handler que dictConfig() cerró, si su stream sigue usable.

    logging no expone una API pública para reabrir un handler, por lo que se
    restauran los dos estados internos que close() modifica: la marca
    "_closed" y el registro en la lista global de cleanup.

    Solo se reabre cuando el stream del handler sigue abierto: en un
    FileHandler, close() cierra además el archivo, y reabrirlo produciría
    errores de escritura. En ese caso el handler se deja cerrado y se
    desvincula del root.

    Es necesario porque pytest crea sus handlers de captura una vez por
    sesión y los reutiliza: si quedaran cerrados, pytest los volvería a
    adjuntar al root en la fase siguiente, ya cerrados.
    """
    if not getattr(handler, "_closed", False):
        return False

    stream = getattr(handler, "stream", None)
    if stream is None or getattr(stream, "closed", False):
        return False

    handler._closed = False
    logging._addHandlerRef(handler)
    return True


def restore_logging_state(state: dict) -> None:
    """Restaura el estado global de logging capturado previamente.

    Los handlers que dictConfig() cerró se reabren cuando su stream sigue
    siendo utilizable, y no se vuelven a adjuntar al root los que no se
    pueden reabrir (un handler cerrado provocaría "--- Logging error ---" al
    emitir sobre él). Tampoco se altera el stream vigente (que durante los
    tests pertenece a la captura de pytest): solo se restaura la
    codificación de los objetos de stream capturados.
    """
    root = logging.getLogger()

    for handler in root.handlers[:]:
        # Se detachment los handlers que no estaban antes y también los que
        # dictConfig() ya cerró: un handler cerrado en el root provocaría
        # "--- Logging error ---" al emitir sobre él.
        if handler not in state["root_handlers"] or getattr(handler, "_closed", False):
            root.removeHandler(handler)

    for handler in state["root_handlers"]:
        if handler in root.handlers:
            continue
        if getattr(handler, "_closed", False):
            _reopen_handler(handler)
        if not getattr(handler, "_closed", False):
            root.addHandler(handler)

    root.setLevel(state["root_level"])
    root.filters = state["root_filters"]

    for name, (handlers, level, propagate, disabled) in state["loggers"].items():
        logger = logging.getLogger(name)
        for handler in logger.handlers[:]:
            if handler not in handlers or getattr(handler, "_closed", False):
                logger.removeHandler(handler)
        for handler in handlers:
            if handler in logger.handlers:
                continue
            if getattr(handler, "_closed", False):
                _reopen_handler(handler)
            if not getattr(handler, "_closed", False):
                logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = propagate
        logger.disabled = disabled

    for stream, encoding, errors in state["streams"]:
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            if encoding is not None:
                reconfigure(encoding=encoding)
            if errors is not None:
                reconfigure(errors=errors)
        except (AttributeError, ValueError, OSError):  # pragma: no cover
            pass


@pytest.fixture
def clean_env(monkeypatch):
    """Elimina LOG_LEVEL y SERVICE_NAME del entorno."""
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("SERVICE_NAME", raising=False)
    return monkeypatch


def emitted(captured) -> list[dict]:
    """Convierte las líneas capturadas en objetos JSON."""
    return [
        json.loads(line)
        for line in captured.strip().splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# 1. Resolución de niveles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("DEBUG", logging.DEBUG),
        ("debug", logging.DEBUG),
        ("Debug", logging.DEBUG),
        ("INFO", logging.INFO),
        ("info", logging.INFO),
        ("WARNING", logging.WARNING),
        ("warning", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("error", logging.ERROR),
        ("CRITICAL", logging.CRITICAL),
        ("critical", logging.CRITICAL),
    ],
)
def test_niveles_validos_se_resuelven(clean_env, value, expected):
    clean_env.setenv("LOG_LEVEL", value)
    assert resolve_log_level() == expected


@pytest.mark.parametrize(
    "value",
    ["TRACE", "trace", "WARN", "warn", "FATAL", "fatal", "NOTSET", "notset",
     "verbose", "", "   "],
)
def test_niveles_invalidos_degradan_a_info_sin_excepcion(clean_env, value):
    clean_env.setenv("LOG_LEVEL", value)
    assert resolve_log_level() == logging.INFO


def test_nivel_ausente_usa_info(clean_env):
    assert resolve_log_level() == logging.INFO


def test_argumento_explicito_tiene_precedencia_sobre_el_entorno(clean_env):
    clean_env.setenv("LOG_LEVEL", "DEBUG")
    assert resolve_log_level("ERROR") == logging.ERROR


def test_entorno_tiene_precedencia_sobre_el_default(clean_env):
    clean_env.setenv("LOG_LEVEL", "WARNING")
    assert resolve_log_level() == logging.WARNING


def test_nivel_invalido_no_propaga_excepcion(clean_env):
    clean_env.setenv("LOG_LEVEL", "TRACE")
    try:
        resolve_log_level()
    except Exception as exc:  # pragma: no cover - no debe ocurrir
        pytest.fail(f"resolve_log_level no debe lanzar: {exc}")
    assert resolve_log_level() == logging.getLevelName(DEFAULT_LOG_LEVEL)


def test_warning_por_nivel_invalido_sale_en_json_por_stderr(clean_env, capsys):
    clean_env.setenv("LOG_LEVEL", "TRACE")
    configure_logging(SERVICE_NAME)

    captured = capsys.readouterr()
    assert captured.out == ""

    records = emitted(captured.err)
    assert len(records) == 1
    assert records[0]["level"] == "WARNING"
    assert records[0]["service"] == SERVICE_NAME
    assert "TRACE" in records[0]["message"]
    assert records[0]["logger"] == "src.core.logging_config"


def test_nivel_invalido_iguala_el_nivel_efectivo_a_info(clean_env):
    clean_env.setenv("LOG_LEVEL", "FATAL")
    configure_logging(SERVICE_NAME)
    assert logging.getLogger().level == logging.INFO


def test_warning_por_nivel_invalido_explicito(clean_env, capsys):
    configure_logging(SERVICE_NAME, level="WARN")
    captured = capsys.readouterr()
    records = emitted(captured.err)
    assert len(records) == 1
    assert records[0]["level"] == "WARNING"
    assert "WARN" in records[0]["message"]


def test_nivel_valido_no_emite_warning(clean_env, capsys):
    configure_logging(SERVICE_NAME, level="DEBUG")
    captured = capsys.readouterr()
    assert captured.err == ""


# ---------------------------------------------------------------------------
# 2. Configuración del root y de sus handlers
# ---------------------------------------------------------------------------


def test_root_tiene_exactamente_dos_handlers(clean_env):
    configure_logging(SERVICE_NAME)
    assert len(logging.getLogger().handlers) == 2


def test_handlers_usan_streams_distintos_y_dinamicos(clean_env):
    configure_logging(SERVICE_NAME)
    handlers = logging.getLogger().handlers

    assert all(isinstance(h, DynamicStreamHandler) for h in handlers)
    assert {h._stream_name for h in handlers} == {"stdout", "stderr"}


def test_no_se_crean_file_handlers(clean_env):
    configure_logging(SERVICE_NAME)
    for handler in logging.getLogger().handlers:
        assert not isinstance(handler, logging.FileHandler)
        assert not isinstance(handler, logging.handlers.WatchedFileHandler)


def test_nivel_del_root_refleja_log_level(clean_env):
    clean_env.setenv("LOG_LEVEL", "DEBUG")
    configure_logging(SERVICE_NAME)
    assert logging.getLogger().level == logging.DEBUG


def test_configuracion_desactiva_existing_loggers(clean_env):
    config = build_logging_config(SERVICE_NAME, level="INFO")
    assert config["disable_existing_loggers"] is False


def test_reconfiguracion_repetida_no_duplica_handlers(clean_env):
    configure_logging(SERVICE_NAME)
    configure_logging(SERVICE_NAME)
    configure_logging(SERVICE_NAME)
    assert len(logging.getLogger().handlers) == 2


def test_reconfiguracion_repetida_no_duplica_registros(clean_env, capsys):
    for _ in range(3):
        configure_logging(SERVICE_NAME)
    logging.getLogger("src.api.docs_api").info("mensaje unico")
    captured = capsys.readouterr()
    records = emitted(captured.out)
    assert len(records) == 1
    assert records[0]["message"] == "mensaje unico"


# ---------------------------------------------------------------------------
# 3. Separación stdout / stderr y ausencia de duplicados
# ---------------------------------------------------------------------------


def test_info_va_solo_a_stdout(clean_env, capsys):
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.api.docs_api").info("mensaje info")
    captured = capsys.readouterr()

    assert "mensaje info" in captured.out
    assert "mensaje info" not in captured.err
    assert len(emitted(captured.out)) == 1


def test_debug_va_solo_a_stdout(clean_env, capsys):
    configure_logging(SERVICE_NAME, level="DEBUG")
    logging.getLogger("src.api.docs_api").debug("mensaje debug")
    captured = capsys.readouterr()

    assert "mensaje debug" in captured.out
    assert "mensaje debug" not in captured.err


def test_warning_va_solo_a_stderr(clean_env, capsys):
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.api.docs_api").warning("mensaje warning")
    captured = capsys.readouterr()

    assert "mensaje warning" in captured.err
    assert "mensaje warning" not in captured.out
    assert len(emitted(captured.err)) == 1


def test_error_va_solo_a_stderr(clean_env, capsys):
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.api.docs_api").error("mensaje error")
    captured = capsys.readouterr()

    assert "mensaje error" in captured.err
    assert "mensaje error" not in captured.out


def test_critical_va_solo_a_stderr(clean_env, capsys):
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.api.docs_api").critical("mensaje critical")
    captured = capsys.readouterr()

    assert "mensaje critical" in captured.err
    assert "mensaje critical" not in captured.out


def test_no_hay_duplicados_entre_streams(clean_env, capsys):
    configure_logging(SERVICE_NAME)
    logger = logging.getLogger("src.api.docs_api")
    logger.info("info")
    logger.warning("warning")
    logger.error("error")
    captured = capsys.readouterr()

    assert captured.out.count("info") == 1
    assert captured.err.count("warning") == 1
    assert captured.err.count("error") == 1
    assert "warning" not in captured.out
    assert "error" not in captured.out


# ---------------------------------------------------------------------------
# 4. Formato JSON y campos obligatorios
# ---------------------------------------------------------------------------


def test_salida_es_json_por_linea(clean_env, capsys):
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.api.docs_api").info("primero")
    logging.getLogger("src.api.docs_api").info("segundo")
    captured = capsys.readouterr()

    lines = [line for line in captured.out.splitlines() if line.strip()]
    assert len(lines) == 2
    for line in lines:
        json.loads(line)


def test_campos_obligatorios_presentes(clean_env, capsys):
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.api.docs_api").info("mensaje")
    record = emitted(capsys.readouterr().out)[0]

    for field in ("timestamp", "level", "service", "logger", "message"):
        assert field in record
    assert record["level"] == "INFO"
    assert record["message"] == "mensaje"


def test_timestamp_utc_iso8601_terminado_en_z(clean_env, capsys):
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.api.docs_api").info("mensaje")
    record = emitted(capsys.readouterr().out)[0]

    timestamp = record["timestamp"]
    assert timestamp.endswith("Z")
    assert "T" in timestamp
    # ISO-8601 parseable.
    from datetime import datetime

    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 0


def test_nombre_del_servicio(clean_env, capsys):
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.api.docs_api").info("mensaje")
    record = emitted(capsys.readouterr().out)[0]
    assert record["service"] == SERVICE_NAME


def test_nombre_del_logger_es_jerarquico(clean_env, capsys):
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.services.document_service").info("mensaje")
    record = emitted(capsys.readouterr().out)[0]
    assert record["logger"] == "src.services.document_service"


def test_el_nombre_del_servicio_explicito_gana_sobre_el_entorno(clean_env, capsys):
    """El nombre pasado a configure_logging() es el que se registra."""
    clean_env.setenv("SERVICE_NAME", "ms3-desde-entorno")
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.api.docs_api").info("mensaje")
    record = emitted(capsys.readouterr().out)[0]

    assert record["service"] == SERVICE_NAME
    assert record["service"] != "ms3-desde-entorno"


def test_el_formateador_usa_el_entorno_cuando_no_hay_nombre_explicito(clean_env):
    """Sin nombre explícito, el formateador toma SERVICE_NAME del entorno."""
    clean_env.setenv("SERVICE_NAME", "ms3-desde-entorno")
    assert JsonFormatter().service_name == "ms3-desde-entorno"


def test_el_formateador_usa_el_valor_por_defecto_sin_entorno(clean_env):
    """Sin nombre explícito ni variable de entorno, se usa el valor por defecto."""
    assert JsonFormatter().service_name == "unknown-service"


def test_no_se_exponen_atributos_internos_de_logrecord(clean_env, capsys):
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.api.docs_api").info("mensaje")
    record = emitted(capsys.readouterr().out)[0]

    for internal in ("args", "pathname", "filename", "module", "lineno",
                     "funcName", "created", "msecs", "relativeCreated",
                     "thread", "threadName", "process", "processName",
                     "exc_info", "exc_text", "stack_info", "taskName",
                     "levelname", "levelno", "name", "msg"):
        assert internal not in record


# ---------------------------------------------------------------------------
# 5. Atributos extra, excepciones y argumentos interpolados
# ---------------------------------------------------------------------------


def test_atributos_extra_se_propagan(clean_env, capsys):
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.api.docs_api").info(
        "documento creado",
        extra={"document_id": "665f1a", "size_bytes": 1234},
    )
    record = emitted(capsys.readouterr().out)[0]

    assert record["document_id"] == "665f1a"
    assert record["size_bytes"] == 1234


def test_argumentos_porcentuales_son_interpolados(clean_env, capsys):
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.services.extraction_service").info(
        "Extracción síncrona OK: '%s' -> %d página(s)", "doc.pdf", 3
    )
    record = emitted(capsys.readouterr().out)[0]

    assert "doc.pdf" in record["message"]
    assert "3 página(s)" in record["message"]


def test_excepcion_se_registra_con_traceback(clean_env, capsys):
    configure_logging(SERVICE_NAME)
    try:
        raise ValueError("boom de prueba")
    except ValueError:
        logging.getLogger("src.core.exceptions").error("fallo al procesar", exc_info=True)

    record = emitted(capsys.readouterr().err)[0]
    assert "exception" in record
    assert "Traceback" in record["exception"]
    assert "ValueError" in record["exception"]
    assert "boom de prueba" in record["exception"]


def test_excepcion_con_causa_explicita_incluye_la_cadena_completa(clean_env, capsys):
    configure_logging(SERVICE_NAME)
    try:
        try:
            raise ValueError("causa original")
        except ValueError as causa:
            raise RuntimeError("error envolvente") from causa
    except RuntimeError:
        logging.getLogger("src.core.exceptions").error("fallo encadenado", exc_info=True)

    exception = emitted(capsys.readouterr().err)[0]["exception"]
    assert "RuntimeError" in exception
    assert "error envolvente" in exception
    # La cadena de excepciones se representa completa en el traceback.
    assert "ValueError" in exception
    assert "causa original" in exception
    assert "The above exception was the direct cause" in exception


def test_excepcion_con_contexto_implicito_incluye_la_cadena_completa(clean_env, capsys):
    configure_logging(SERVICE_NAME)
    try:
        try:
            raise ValueError("causa original")
        except ValueError:
            # Sin 'from': encadenamiento implícito (__context__).
            raise RuntimeError("error envolvente")
    except RuntimeError:
        logging.getLogger("src.core.exceptions").error("fallo encadenado", exc_info=True)

    exception = emitted(capsys.readouterr().err)[0]["exception"]
    assert "RuntimeError" in exception
    assert "error envolvente" in exception
    assert "ValueError" in exception
    assert "causa original" in exception
    assert "During handling of the above exception" in exception


def test_sin_excepcion_no_hay_campo_exception(clean_env, capsys):
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.api.docs_api").info("mensaje normal")
    record = emitted(capsys.readouterr().out)[0]
    assert "exception" not in record


def test_stack_info_se_registra(clean_env, capsys):
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.api.docs_api").error("con stack", stack_info=True)
    record = emitted(capsys.readouterr().err)[0]

    assert "stack" in record
    assert "Stack (most recent call last)" in record["stack"]


# ---------------------------------------------------------------------------
# 6. Redacción de información sensible
# ---------------------------------------------------------------------------


def test_redact_uri_oculta_credenciales_y_conserva_el_host():
    redacted = redact_uri("mongodb://usuario:clave@db:27017/?authSource=admin")
    assert "usuario" not in redacted
    assert "clave" not in redacted
    assert "db:27017" in redacted
    assert "mongodb://" in redacted


def test_redact_uri_sin_credenciales_no_altera_el_texto():
    assert redact_uri("http://localhost:8000/health") == "http://localhost:8000/health"
    assert redact_uri("") == ""
    assert redact_uri(None) == ""


def test_redact_value_por_clave_sensible():
    assert redact_value("password", "hunter2") == REDACTED
    assert redact_value("Authorization", "Bearer abc") == REDACTED


def test_redact_value_recursivo_en_diccionarios_anidados():
    result = redact_value("ctx", {"password": "hunter2", "user": "mauri"})
    assert result["password"] == REDACTED
    assert result["user"] == "mauri"


def test_redact_value_recursivo_en_listas_de_diccionarios():
    result = redact_value("items", [{"token": "abc"}, {"token": "def"}])
    assert result[0]["token"] == REDACTED
    assert result[1]["token"] == REDACTED


def test_redact_value_recursivo_en_listas_de_texto():
    result = redact_value("uris", ["mongodb://u:p@db:27017", "sin credenciales"])
    assert "u:p" not in result[0]
    assert "db:27017" in result[0]
    assert result[1] == "sin credenciales"


# --- Claves sensibles compuestas ---


def test_las_claves_catalogadas_siguen_detectandose_como_sensibles():
    """Cada clave del catálogo debe ser reconocida por la detección."""
    for key in SENSITIVE_KEYS:
        assert is_sensitive_key(key), f"{key} debería detectarse como sensible"


@pytest.mark.parametrize(
    "key",
    [
        "db_password",
        "api_key_v2",
        "x-auth-token",
        "DB_PASSWORD",
        "Api_Key_V2",
        "X-Auth-Token",
        "refresh_token",
        "client_secret",
        "user_pwd",
        "credentials",
        "private_key",
    ],
)
def test_claves_sensibles_compuestas_se_redactan(clean_env, capsys, key):
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.api.docs_api").info("contexto", extra={key: "CLAVEDEVALORES"})
    record = emitted(capsys.readouterr().out)[0]

    assert "CLAVEDEVALORES" not in json.dumps(record)
    assert record[key] == REDACTED


@pytest.mark.parametrize(
    "key",
    [
        "document_id",
        "page_count",
        "status",
        "file_name",
        "content",
        "checksum",
        "created_at",
        "size_bytes",
        "error_message",
        "monkey",
    ],
)
def test_claves_no_sensibles_no_se_redactan(clean_env, capsys, key):
    """La detección por token no debe redactar claves innocuous."""
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.api.docs_api").info("contexto", extra={key: "valor-visible"})
    record = emitted(capsys.readouterr().out)[0]

    assert record[key] == "valor-visible"


def test_falso_positivo_conocido_de_la_deteccion_por_token(clean_env, capsys):
    """'tokenizer' contiene el token 'token': falso positivo conocido y aceptado.

    La detección es sencilla y basada en tokens de clave; se documenta este
    caso para que el comportamiento no se considere un error oculto.
    """
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.api.docs_api").info("contexto", extra={"tokenizer": "valor"})
    record = emitted(capsys.readouterr().out)[0]

    assert record["tokenizer"] == REDACTED


def test_redact_mapping():
    result = redact_mapping({"password": "x", "document_id": "abc"})
    assert result["password"] == REDACTED
    assert result["document_id"] == "abc"


def test_clave_sensible_en_extra_queda_redactada(clean_env, capsys):
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.api.docs_api").info(
        "contexto", extra={"password": "hunter2", "document_id": "abc"}
    )
    record = emitted(capsys.readouterr().out)[0]

    assert record["password"] == REDACTED
    assert record["document_id"] == "abc"


def test_credencial_en_argumentos_interpolados_queda_redactada(clean_env, capsys):
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.database.mongodb").error(
        "Error al conectar con MongoDB: %s",
        "mongodb://mauri:CLAVEDEVALORES@db:27017/?authSource=admin",
    )
    record = emitted(capsys.readouterr().err)[0]

    assert "CLAVEDEVALORES" not in record["message"]
    assert "mauri" not in record["message"]
    assert "db:27017" in record["message"]


def test_credencial_en_el_traceback_queda_redactada(clean_env, capsys):
    configure_logging(SERVICE_NAME)
    try:
        raise RuntimeError(
            "fallo conectando a mongodb://mauri:CLAVEDEVALORES@db:27017"
        )
    except RuntimeError:
        logging.getLogger("src.database.mongodb").error(
            "Error al conectar con MongoDB", exc_info=True
        )
    record = emitted(capsys.readouterr().err)[0]

    assert "CLAVEDEVALORES" not in record["exception"]
    assert "mauri" not in record["exception"]
    assert "db:27017" in record["exception"]


# --- I-2: valores no textuales (objetos con __str__ y bytes) ---


def test_objeto_con_str_sensible_se_redacta(clean_env, capsys):
    """Un objeto cuyo __str__ devuelve una URI con credenciales no debe filtrarla."""

    class CredencialesMongo:
        def __str__(self):
            return "mongodb://usuario:CLAVEDEVALORES@db:27017/?authSource=admin"

    configure_logging(SERVICE_NAME)
    logging.getLogger("src.database.mongodb").error(
        "Error al conectar con MongoDB", extra={"conn": CredencialesMongo()}
    )
    record = emitted(capsys.readouterr().err)[0]

    assert "CLAVEDEVALORES" not in json.dumps(record)
    assert "usuario" not in json.dumps(record)
    assert "db:27017" in record["conn"]


def test_bytes_con_uri_sensible_se_redacta(clean_env, capsys):
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.database.mongodb").error(
        "bytes", extra={"raw": b"mongodb://u:CLAVEDEVALORES@db:27017"}
    )
    record = emitted(capsys.readouterr().err)[0]

    assert "CLAVEDEVALORES" not in json.dumps(record)
    assert "db:27017" in record["raw"]


def test_objeto_sensible_anidado_en_extra_se_redacta(clean_env, capsys):
    class Credenciales:
        def __str__(self):
            return "mongodb://usuario:CLAVEDEVALORES@db:27017"

    configure_logging(SERVICE_NAME)
    logging.getLogger("src.api.docs_api").info(
        "contexto", extra={"ctx": {"conn": Credenciales(), "db": "pdf_database_dev"}}
    )
    record = emitted(capsys.readouterr().out)[0]

    assert "CLAVEDEVALORES" not in json.dumps(record)
    assert record["ctx"]["db"] == "pdf_database_dev"


def test_valores_no_sensibles_se_conservan_sin_alterar(clean_env, capsys):
    """La redacción no debe alterar los valores que no son sensibles."""
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.api.docs_api").info(
        "contexto",
        extra={"document_id": "665f1a", "size_bytes": 1234, "estado": "completado"},
    )
    record = emitted(capsys.readouterr().out)[0]

    assert record["document_id"] == "665f1a"
    assert record["size_bytes"] == 1234
    assert record["estado"] == "completado"


# --- I-3: argumentos de interpolación ---


def test_diccionario_como_argumento_se_redacta(clean_env, capsys):
    """Un dict pasado como argumento no debe quedar en claro."""
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.database.mongodb").error(
        "contexto: %s", {"password": "hunter2", "user": "mauri"}
    )
    record = emitted(capsys.readouterr().err)[0]

    assert "hunter2" not in record["message"]
    # La estructura del contexto se conserva: la clave sigue visible.
    assert "password" in record["message"]
    assert "mauri" in record["message"]


def test_diccionario_anidado_como_argumento_se_redacta(clean_env, capsys):
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.api.docs_api").error(
        "contexto: %s", {"ctx": {"api_key": "CLAVEAPI", "ok": 1}}
    )
    record = emitted(capsys.readouterr().err)[0]

    assert "CLAVEAPI" not in record["message"]
    assert "ok" in record["message"]


def test_argumento_con_uri_sensible_se_redacta(clean_env, capsys):
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.database.mongodb").error(
        "conectando a %s con usuario %s",
        "mongodb://u:CLAVEDEVALORES@db:27017",
        "mauri",
    )
    record = emitted(capsys.readouterr().err)[0]

    assert "CLAVEDEVALORES" not in record["message"]
    assert "db:27017" in record["message"]
    assert "mauri" in record["message"]


def test_argumentos_multiples_se_conservan(clean_env, capsys):
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.services.document_service").info(
        "Extracción OK: '%s' -> %d página(s)", "doc.pdf", 3
    )
    record = emitted(capsys.readouterr().out)[0]

    assert "doc.pdf" in record["message"]
    assert "3 página(s)" in record["message"]


def test_formato_nombrado_con_diccionario_se_conserva(clean_env, capsys):
    """logging admite %(nombre)s con un diccionario: debe seguir funcionando."""
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.api.docs_api").info("doc %(id)s", {"id": "665f1a"})
    record = emitted(capsys.readouterr().out)[0]

    assert record["message"] == "doc 665f1a"


def test_mensaje_sin_argumentos_no_se_modifica(clean_env, capsys):
    """Los mensajes sin argumentos no deben modificarse (incluido un % literal)."""
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.api.docs_api").info("Conexión con MongoDB cerrada (100%)")
    record = emitted(capsys.readouterr().out)[0]

    assert record["message"] == "Conexión con MongoDB cerrada (100%)"


def test_diccionario_anidado_en_extra_queda_redactado(clean_env, capsys):
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.api.docs_api").info(
        "contexto",
        extra={"ctx": {"password": "hunter2", "db": {"mongo_url": "mongodb://u:p@db:27017"}}},
    )
    record = emitted(capsys.readouterr().out)[0]

    # La clave sensible se sustituye por su valor enmascarado, conservando
    # la estructura del contexto (la clave sigue presente).
    assert record["ctx"]["password"] == REDACTED
    assert record["ctx"]["db"]["mongo_url"] == REDACTED
    assert "u:p@" not in json.dumps(record["ctx"])


def test_lista_de_diccionarios_en_extra_queda_redactada(clean_env, capsys):
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.api.docs_api").info(
        "contexto", extra={"items": [{"token": "abc"}, {"ok": 1}]}
    )
    record = emitted(capsys.readouterr().out)[0]

    assert record["items"][0]["token"] == REDACTED
    assert record["items"][1]["ok"] == 1


# ---------------------------------------------------------------------------
# 7. Colisiones entre extra y campos obligatorios
# ---------------------------------------------------------------------------


def test_extra_no_puede_sobrescribir_el_service(clean_env, capsys):
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.api.docs_api").info("mensaje", extra={"service": "OTRO"})
    record = emitted(capsys.readouterr().out)[0]

    assert record["service"] == SERVICE_NAME


def test_extra_no_puede_sobrescribir_timestamp_level_ni_logger(clean_env, capsys):
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.api.docs_api").info(
        "mensaje real",
        extra={
            "timestamp": "1999-01-01T00:00:00.000Z",
            "level": "FAKE",
            "logger": "fake.logger",
        },
    )
    record = emitted(capsys.readouterr().out)[0]

    assert record["timestamp"].endswith("Z")
    assert record["timestamp"] != "1999-01-01T00:00:00.000Z"
    assert record["level"] == "INFO"
    assert record["logger"] == "src.api.docs_api"
    assert record["message"] == "mensaje real"


def test_logging_rechaza_extra_con_message(clean_env):
    """'message' y 'asctime' están reservados por logging.

    Es la propia biblioteca logging la que impide pasar un 'message' en
    extra (KeyError en makeRecord), por lo que nunca llega al formateador.
    """
    configure_logging(SERVICE_NAME)
    with pytest.raises(KeyError):
        logging.getLogger("src.api.docs_api").info("mensaje", extra={"message": "otro"})


def test_valores_colisionantes_se_conservan_en_el_objeto_extra(clean_env, capsys):
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.api.docs_api").info(
        "mensaje", extra={"service": "OTRO", "level": "FAKE"}
    )
    record = emitted(capsys.readouterr().out)[0]

    assert record["service"] == SERVICE_NAME
    assert record["level"] == "INFO"
    assert record["extra"]["service"] == "OTRO"
    assert record["extra"]["level"] == "FAKE"


def test_no_se_crea_el_objeto_extra_sin_colisiones(clean_env, capsys):
    configure_logging(SERVICE_NAME)
    logging.getLogger("src.api.docs_api").info("mensaje", extra={"document_id": "abc"})
    record = emitted(capsys.readouterr().out)[0]

    assert "extra" not in record
    assert record["document_id"] == "abc"


# ---------------------------------------------------------------------------
# 8. Loggers existentes y reasignación de stdout
# ---------------------------------------------------------------------------


def test_logger_creado_antes_de_la_configuracion_sigue_emitiendo(clean_env, capsys):
    logger = logging.getLogger("src.api.docs_api")
    configure_logging(SERVICE_NAME)

    assert logger.disabled is False
    logger.info("despues de configurar")
    captured = capsys.readouterr()
    records = emitted(captured.out)

    assert len(records) == 1
    assert records[0]["message"] == "despues de configurar"
    assert records[0]["logger"] == "src.api.docs_api"


def test_sin_configurar_no_se_emiten_registros_info(clean_env, capsys):
    """Sin configure_logging() el root no tiene handlers: un INFO no sale.

    Es el comportamiento nativo de logging (el handler de último recurso
    solo cubre WARNING en adelante) y es justamente el problema que este
    módulo viene a corregir.
    """
    logging.getLogger("src.api.docs_api").info("sin configurar")
    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err == ""


def test_loggers_con_getlogger_name_existenes_siguen_funcionando(clean_env, capsys):
    configure_logging(SERVICE_NAME)
    for name in (
        "src.api.docs_api",
        "src.core.exceptions",
        "src.database.mongodb",
        "src.services.document_service",
        "src.repository.document_repository",
    ):
        logging.getLogger(name).info("mensaje desde %s", name)
    records = emitted(capsys.readouterr().out)

    assert len(records) == 5
    for record in records:
        assert record["logger"].startswith("src.")


def test_handler_dinamico_sigue_a_stdout_reasignado(clean_env, monkeypatch):
    import io

    configure_logging(SERVICE_NAME)
    buffer = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buffer)

    logging.getLogger("src.api.docs_api").info("mensaje redirigido")

    assert "mensaje redirigido" in buffer.getvalue()


def test_handler_dinamico_sigue_a_stderr_reasignado(clean_env, monkeypatch):
    import io

    configure_logging(SERVICE_NAME)
    buffer = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buffer)

    logging.getLogger("src.api.docs_api").error("error redirigido")

    assert "error redirigido" in buffer.getvalue()


# ---------------------------------------------------------------------------
# 9. Loggers de uvicorn
# ---------------------------------------------------------------------------


def test_loggers_de_uvicorn_no_mantienen_handlers_propios(clean_env):
    configure_logging(SERVICE_NAME)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        assert logging.getLogger(name).handlers == []


def test_loggers_de_uvicorn_propagan_al_root(clean_env):
    configure_logging(SERVICE_NAME)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        assert logging.getLogger(name).propagate is True


def test_loggers_de_uvicorn_heredan_el_nivel_configurado(clean_env):
    configure_logging(SERVICE_NAME, level="WARNING")
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        assert logging.getLogger(name).level == logging.WARNING


def test_access_log_de_uvicorn_se_emite_una_sola_vez_por_stdout(clean_env, capsys):
    configure_logging(SERVICE_NAME)
    logging.getLogger("uvicorn.access").info(
        '%s - "%s %s HTTP/%s" %d', "127.0.0.1", "GET", "/documents", "1.1", 200
    )
    captured = capsys.readouterr()

    records = emitted(captured.out)
    assert len(records) == 1
    assert records[0]["logger"] == "uvicorn.access"
    assert "GET /documents" in records[0]["message"]
    assert captured.err == ""


def test_nivel_warning_silencia_el_access_log(clean_env, capsys):
    configure_logging(SERVICE_NAME, level="WARNING")
    logging.getLogger("uvicorn.access").info(
        '%s - "%s %s HTTP/%s" %d', "127.0.0.1", "GET", "/documents", "1.1", 200
    )
    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err == ""


def test_nivel_warning_mantiene_los_warning_del_access_log(clean_env, capsys):
    configure_logging(SERVICE_NAME, level="WARNING")
    logging.getLogger("uvicorn.access").warning(
        '%s - "%s %s HTTP/%s" %d', "127.0.0.1", "GET", "/documents", "1.1", 500
    )
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "GET /documents" in captured.err


# ---------------------------------------------------------------------------
# 10. Componentes públicos del módulo
# ---------------------------------------------------------------------------


def test_max_level_filter_filtra_por_nivel_estricto():
    filt = MaxLevelFilter(max_level=logging.WARNING)

    class FakeRecord:
        def __init__(self, levelno):
            self.levelno = levelno

    assert filt.filter(FakeRecord(logging.INFO)) is True
    assert filt.filter(FakeRecord(logging.DEBUG)) is True
    assert filt.filter(FakeRecord(logging.WARNING)) is False
    assert filt.filter(FakeRecord(logging.ERROR)) is False


def test_configuracion_sin_nivel_usa_info(clean_env):
    """Sin LOG_LEVEL ni argumento explícito, el nivel efectivo es INFO."""
    configure_logging(SERVICE_NAME)
    assert logging.getLogger().level == logging.INFO


# --- Aislamiento del estado global de logging ---


def test_la_restauracion_no_rea_agrega_handlers_cerrados(clean_env):
    """dictConfig() cierra los handlers previos: no deben volver al root."""
    state = capture_logging_state()
    propio = logging.NullHandler()
    logging.getLogger().addHandler(propio)

    configure_logging(SERVICE_NAME)
    assert getattr(propio, "_closed", False), "dictConfig debería cerrar el handler"

    restore_logging_state(state)

    root = logging.getLogger()
    assert propio not in root.handlers
    assert not [h for h in root.handlers if getattr(h, "_closed", False)]


def test_la_restauracion_no_deja_handlers_del_modulo(clean_env):
    """Tras restaurar, el root no debe conservar handlers del módulo."""
    state = capture_logging_state()
    configure_logging(SERVICE_NAME)
    assert any(isinstance(h, DynamicStreamHandler) for h in logging.getLogger().handlers)

    restore_logging_state(state)

    root = logging.getLogger()
    assert not [h for h in root.handlers if isinstance(h, DynamicStreamHandler)]


def test_la_restauracion_devuelve_el_nivel_del_root(clean_env):
    state = capture_logging_state()
    nivel_original = logging.getLogger().level

    configure_logging(SERVICE_NAME, level="DEBUG")
    assert logging.getLogger().level == logging.DEBUG

    restore_logging_state(state)
    assert logging.getLogger().level == nivel_original


def test_la_restauracion_revierte_la_codificacion_de_los_streams(clean_env):
    """_force_utf8_streams() cambia el encoding: la restauración lo revierte."""

    class FakeStream:
        def __init__(self):
            self.encoding = "cp1252"
            self.errors = "strict"
            self.reconfigurations = []

        def reconfigure(self, **kwargs):
            self.reconfigurations.append(kwargs)
            for key, value in kwargs.items():
                setattr(self, key, value)

    fake = FakeStream()
    state = capture_logging_state()
    state["streams"] = [(fake, "cp1252", "strict")]

    configure_logging(SERVICE_NAME)
    restore_logging_state(state)

    assert fake.reconfigurations, "debería haber reconfigurado el stream capturado"
    assert fake.encoding == "cp1252"
    assert fake.errors == "strict"


def test_la_restauracion_no_altera_el_stream_vigente_de_pytest(clean_env):
    """La restauración no debe tocar el stream que pytest tiene capturado."""
    state = capture_logging_state()
    stream_vigente = sys.stdout
    encoding_vigente = stream_vigente.encoding

    configure_logging(SERVICE_NAME)
    restore_logging_state(state)

    assert sys.stdout is stream_vigente
    assert sys.stdout.encoding == encoding_vigente
