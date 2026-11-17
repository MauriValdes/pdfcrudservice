"""Tests de la integración del logging centralizado en el arranque de MS3.

Verifican que:

- La configuración expone el nivel de logging (LOG_LEVEL) y que un valor
  inválido no impide el arranque.
- El arranque configura el logging centralizado usando el nivel de Settings.
- Se utiliza el nombre de servicio "ms3-crud".
- No quedan los mensajes de texto plano que antes emitían los print().
- El arranque no genera errores y los registros mantienen el formato JSON.
"""

import asyncio
import importlib
import json
import logging
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import src.main
from src.config.config import Settings, settings

# Helpers de aislamiento ya probados en test_logging_config.py: restauran
# handlers, nivel, filtros, loggers de uvicorn y la codificación de los
# streams. Reutilizarlos evita duplicar la lógica y mantiene el estado
# global de logging aislado entre tests.
from tests.test_logging_config import capture_logging_state, restore_logging_state

SERVICE_NAME = "ms3-crud"

# Textos exactos de los tres print() que existían antes de la integración.
MENSAJES_DE_PRINT_ANTIGUOS = (
    "¡Hola desde main.py!",
    "Iniciando conexión con MongoDB... 📡",
    "Cerrando conexiones... 🔌",
)


# ---------------------------------------------------------------------------
# Aislamiento del estado global de logging
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def restore_global_logging_state():
    """Restaura el estado global de logging tras cada test.

    Recargar src.main vuelve a ejecutar configure_logging(), que reemplaza
    los handlers del root y cierra los handlers registrados en el proceso
    (los de pytest entre ellos).
    """
    state = capture_logging_state()
    try:
        yield
    finally:
        restore_logging_state(state)


def recaptured(captured) -> list[dict]:
    """Convierte las líneas capturadas en objetos JSON."""
    return [
        json.loads(line)
        for line in captured.splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# Configuración del nivel
# ---------------------------------------------------------------------------


def test_settings_expone_el_nivel_de_logging_por_defecto(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    assert Settings().log_level == "INFO"


def test_settings_lee_el_nivel_de_logging_del_entorno(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    assert Settings().log_level == "DEBUG"


def test_settings_normaliza_el_nivel_del_entorno(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "  warning  ")
    assert Settings().log_level == "WARNING"


def test_un_nivel_invalido_no_impide_la_arranque(monkeypatch):
    """Un LOG_LEVEL inválido no puede abortar el servicio (decisión C2)."""
    monkeypatch.setenv("LOG_LEVEL", "TRACE")
    assert Settings().log_level == "TRACE"


# ---------------------------------------------------------------------------
# Integración en el arranque
# ---------------------------------------------------------------------------


def test_el_arranque_configura_el_logging_con_el_nivel_de_settings(monkeypatch):
    """El nivel efectivo proviene de Settings, no de otra fuente."""
    monkeypatch.setattr(settings, "log_level", "DEBUG")
    importlib.reload(src.main)

    assert logging.getLogger().level == logging.DEBUG


def test_el_arranque_usa_el_nombre_de_servicio_ms3_crud(monkeypatch, capsys):
    monkeypatch.setattr(settings, "log_level", "INFO")
    importlib.reload(src.main)

    captured = capsys.readouterr()
    records = recaptured(captured.out)
    assert records, "el arranque debería emitir al menos un registro"
    assert {record["service"] for record in records} == {SERVICE_NAME}


def test_el_arranque_no_emite_texto_plano(monkeypatch, capsys):
    """Toda la salida del arranque debe ser JSON: nada de mensajes sueltos."""
    monkeypatch.setattr(settings, "log_level", "INFO")
    importlib.reload(src.main)

    captured = capsys.readouterr()
    lineas = [line for line in captured.out.splitlines() if line.strip()]
    assert lineas, "el arranque debería emitir al menos un registro"

    for linea in lineas:
        json.loads(linea)  # falla si alguna línea no es JSON


def test_el_arranque_no_emite_los_mensajes_de_print_antiguos(monkeypatch, capsys):
    monkeypatch.setattr(settings, "log_level", "INFO")
    importlib.reload(src.main)

    captured = capsys.readouterr()
    salida = captured.out + captured.err
    for mensaje in MENSAJES_DE_PRINT_ANTIGUOS:
        assert mensaje not in salida, f"quedó el mensaje de print: {mensaje!r}"


def test_el_arranque_no_genera_errores(monkeypatch, capsys):
    monkeypatch.setattr(settings, "log_level", "INFO")
    importlib.reload(src.main)

    captured = capsys.readouterr()
    registros = recaptured(captured.out) + recaptured(captured.err)
    errores = [r for r in registros if r["level"] in ("ERROR", "CRITICAL")]
    assert not errores, f"el arranque no debería registrar errores: {errores}"


def test_el_arranque_emite_un_registro_de_inicio_del_servicio(monkeypatch, capsys):
    monkeypatch.setattr(settings, "log_level", "INFO")
    importlib.reload(src.main)

    captured = capsys.readouterr()
    mensajes = [r["message"] for r in recaptured(captured.out)]
    assert any("pdfcrudservice" in mensaje for mensaje in mensajes)


def test_el_lifespan_registra_el_inicio_y_el_cierre(monkeypatch, capsys):
    """Los dos print() del lifespan quedan sustituidos por registros JSON."""
    from src.database.mongodb import db_connection

    # El test es auto-contenido: no depende de que otro test haya dejado el
    # logging configurado. El reload registra el arranque y deja operativos
    # los handlers del módulo antes de ejercitar el lifespan.
    monkeypatch.setattr(settings, "log_level", "INFO")
    importlib.reload(src.main)
    capsys.readouterr()

    # Los mocks anteriores (connect/close) eran vestigiales: el lifespan
    # real invoca la función de módulo retry_connect(), stop_supervision()
    # y disconnect(). Se mockea exactamente lo que el lifespan usa, sin
    # intentar ninguna conexión a MongoDB (antes el retry_connect real
    # creaba un cliente Motor e intentaba un ping a localhost:27017).
    async def retry_connect_suspendido():
        # Permanece suspendida hasta que el apagado del lifespan la
        # cancele: misma forma que la tarea real (tarea pendiente al
        # apagar -> cancelada), sin conexión real.
        await asyncio.Event().wait()

    monkeypatch.setattr(src.main, "retry_connect", retry_connect_suspendido)
    monkeypatch.setattr(db_connection, "stop_supervision", AsyncMock())
    monkeypatch.setattr(db_connection, "disconnect", AsyncMock())

    with TestClient(src.main.app):
        pass

    captured = capsys.readouterr()
    mensajes = [r["message"] for r in recaptured(captured.out)]

    assert any("Iniciando conexión con MongoDB" in m for m in mensajes)
    assert any("Cerrando conexiones" in m for m in mensajes)


def test_los_registros_de_ms3_mantienen_el_formato_json(monkeypatch, capsys):
    """Un logger del servicio sigue el formato JSON centralizado."""
    monkeypatch.setattr(settings, "log_level", "INFO")
    importlib.reload(src.main)

    logging.getLogger("src.database.mongodb").info(
        "Conectado a MongoDB en la base de datos: %s", "pdf_database_dev"
    )

    captured = capsys.readouterr()
    registros = recaptured(captured.out)
    registro = registros[-1]

    assert registro["level"] == "INFO"
    assert registro["service"] == SERVICE_NAME
    assert registro["logger"] == "src.database.mongodb"
    assert registro["message"].endswith("pdf_database_dev")
    assert registro["timestamp"].endswith("Z")
