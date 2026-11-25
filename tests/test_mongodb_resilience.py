"""Tests de resiliencia MongoDB (Factor 4): reintentos, backoff y supervisión.

P1 - núcleo de reintentos de src/database/mongodb.py:

1. Secuencia exacta del backoff: 1, 2, 4, 8, 16, 30, 30.
2. retry_connect() reintenta ante fallos de conexión aplicando el backoff
   correspondiente, con cancelación controlada al finalizar.
3. El contador _retry_attempt se reinicia a cero después de una conexión
   completamente exitosa (ping + índices) y el backoff vuelve a 1 segundo.
4. Tras una caída detectada por el supervisor hay exactamente UNA espera de
   backoff antes de intentar reconectar (sin esperas duplicadas).

P2 - supervisión continua (_supervise_loop / start_supervision /
stop_supervision):

5. Pings periódicos: espera inicial de 5 s y luego cada SUPERVISION_INTERVAL
   (30 s); los pings exitosos mantienen el estado y el supervisor no crea
   clientes ni reinicializa índices.
6. Detección de caída: ping fallido marca _connected/_indexes_created en
   False, el supervisor termina por sí mismo (sin excepción ni cancelación)
   y preserva las referencias client/db sin cerrar el cliente.
7. Cancelación del supervisor: CancelledError propagada, estado marcado como
   desconectado y referencias conservadas para disconnect().
8. stop_supervision(): cancela y espera la tarea activa, limpia la referencia
   de una tarea ya terminada y es un no-op seguro si no hay tarea.

P3 - recuperación completa y endpoints de readiness:

9. Ciclo completo de recuperación: caída -> UNA espera de backoff ->
   reconexión exitosa con recreación del índice, indexes_ready True y
   contador de reintentos reiniciado.
10. /ready y /health: 503 con detalle completo en estado desconectado y 200
    con cuerpo completo tras una recuperación simulada (sin lifespan).
11. Arranque sin MongoDB con el lifespan real: /health 200, /ready 503 y
    apagado graceful que cancela retry_connect y limpia el singleton.

Nota sobre TestClient: en la prueba 10 se usa sin `with` (el lifespan no se
ejecuta) y en la 11 dentro de `with` (el lifespan corre en el loop del
portal, en OTRO hilo); por eso nunca se hace `await` directo de los Events
del reloj falso desde el loop del test, sino sondeo acotado con `is_set()`.

P4 - apagado graceful:

12. Ciclo de vida completo CON supervisión activa: el lifespan conecta,
    arranca el supervisor y al cerrar el TestClient cancela retry_connect,
    detiene la supervisión y disconnect() cierra el cliente.
13. Cancelación durante el backoff post-caída: CancelledError propagada,
    estado desconectado y recursos client/db conservados para disconnect()
    (más la fase literal sin recursos previos).
14. Cancelación durante la reconexión, dentro de connect() (ping bloqueado):
    la cancelación atraviesa connect() sin limpiar recursos.
"""

import asyncio
import contextlib

from unittest.mock import AsyncMock, MagicMock

import pytest
from pymongo.errors import ServerSelectionTimeoutError

import src.database.mongodb as mongodb
import src.main as src_main
from src.database.mongodb import _apply_backoff, db_connection, retry_connect
from fastapi.testclient import TestClient
from src.main import app


# ---------------------------------------------------------------------------
# Infraestructura: reloj falso y aislamiento del singleton db_connection
# ---------------------------------------------------------------------------


class FakeClock:
    """Reemplazo de asyncio.sleep: registra los delays sin esperar tiempo real.

    Si `stop_after` está configurado, al alcanzarse esa cantidad de delays la
    tarea llamadora queda congelada en un gate (Event) hasta que el test la
    cancele o la libere: esto permite aserciones deterministas sobre el
    estado exacto del flujo y evita bucles infinitos.
    """

    def __init__(self, real_sleep):
        self.real_sleep = real_sleep
        self.delays: list = []
        self.stop_after = None
        self.reached_stop = asyncio.Event()
        self.gate = asyncio.Event()

    async def __call__(self, delay=None, result=None):
        self.delays.append(delay)
        if self.stop_after is not None and len(self.delays) >= self.stop_after:
            # Congelamiento one-shot: solo la primera vez que se alcanza
            # stop_after; los sleeps posteriores fluyen normalmente.
            self.stop_after = None
            self.reached_stop.set()
            await self.gate.wait()
        # Ceder el control al event loop sin esperar tiempo real
        await self.real_sleep(0)

    async def wait_until(self, predicate, timeout_s: float = 2.0) -> None:
        """Espera activa acotada usando tiempo real (ticks de 10 ms)."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while not predicate():
            if loop.time() > deadline:
                raise AssertionError(
                    "Condición no alcanzada dentro del timeout de prueba"
                )
            await self.real_sleep(0.01)


@pytest.fixture
async def fake_clock(monkeypatch):
    """Parchea asyncio.sleep con el reloj falso (restaurado al finalizar)."""
    clock = FakeClock(asyncio.sleep)
    monkeypatch.setattr(asyncio, "sleep", clock)
    return clock


@pytest.fixture
async def clean_db_connection():
    """Aísla el singleton db_connection y restaura su estado tras cada test.

    Además detiene la supervisión y barre cualquier tarea pendiente creada
    por el test, para no dejar tareas huérfanas ni warnings de tareas sin
    esperar.
    """
    state_names = (
        "client",
        "db",
        "_connected",
        "_indexes_created",
        "_retry_attempt",
        "_supervision_shutdown",
        "_supervision_task",
    )
    saved = {name: getattr(db_connection, name) for name in state_names}

    # Estado limpio de partida
    db_connection.client = None
    db_connection.db = None
    db_connection._connected = False
    db_connection._indexes_created = False
    db_connection._retry_attempt = 0
    db_connection._supervision_shutdown = False
    db_connection._supervision_task = None

    # Registrar las tareas preexistentes al comenzar la prueba (excluyendo la
    # tarea actual): solo las tareas NUEVAS creadas durante la prueba podrán
    # ser canceladas en la limpieza, nunca las ajenas al test.
    preexisting_tasks = {
        t for t in asyncio.all_tasks() if t is not asyncio.current_task()
    }

    try:
        yield db_connection
    finally:
        try:
            # Detener supervisión si el test dejó una tarea activa
            supervision = db_connection._supervision_task
            if supervision is not None and not supervision.done():
                supervision.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await supervision
            # Barrido de limpieza: SOLO tareas creadas durante esta prueba.
            # Las tareas preexistentes (registradas al comenzar la prueba)
            # nunca se cancelan; la tarea actual siempre queda excluida.
            current = asyncio.current_task()
            new_pending = [
                t
                for t in asyncio.all_tasks()
                if t is not current and t not in preexisting_tasks
            ]
            for pending_task in new_pending:
                pending_task.cancel()
            if new_pending:
                await asyncio.gather(*new_pending, return_exceptions=True)
        finally:
            # Restaurar el estado original del singleton, incluso si la
            # limpieza anterior lanzó un error
            for name, value in saved.items():
                setattr(db_connection, name, value)


def _make_mongo_mocks():
    """Cliente Motor mockeado con ping controlado desde el test.

    `state["mongo_disponible"]` permite simular la caída y la recuperación
    de MongoDB sin conexión real. La creación del índice es exitosa.
    """
    state = {"mongo_disponible": False}

    def ping_side_effect(command, *args, **kwargs):
        if state["mongo_disponible"]:
            return {"ok": 1}
        raise ServerSelectionTimeoutError("MongoDB simulado no disponible")

    mock_collection = MagicMock()
    mock_collection.create_index = AsyncMock(return_value="checksum_unique")
    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    mock_client = MagicMock()
    mock_client.admin.command = AsyncMock(side_effect=ping_side_effect)
    mock_client.__getitem__.return_value = mock_db

    factory = MagicMock(return_value=mock_client)
    return state, factory, mock_client


async def test_cada_conexion_recrea_cliente_con_write_concern(
    clean_db_connection, monkeypatch
):
    """Cada connect(), incluida una reconexión, recrea el cliente con
    w=1 y journal=True."""
    state, factory, _ = _make_mongo_mocks()
    monkeypatch.setattr(mongodb, "AsyncIOMotorClient", factory)
    state["mongo_disponible"] = True

    await db_connection.connect()
    await db_connection.connect()

    assert factory.call_count == 2
    for call in factory.call_args_list:
        _, kwargs = call
        assert kwargs == {"w": 1, "journal": True}


# ---------------------------------------------------------------------------
# 1. Secuencia del backoff
# ---------------------------------------------------------------------------


async def test_secuencia_backoff_exacta(clean_db_connection, fake_clock):
    """_apply_backoff produce exactamente 1, 2, 4, 8, 16, 30, 30 segundos
    y no reinicia el contador por límite de intentos."""
    assert db_connection._retry_attempt == 0

    for _ in range(7):
        await _apply_backoff()

    assert fake_clock.delays == [1, 2, 4, 8, 16, 30, 30], (
        f"secuencia de backoff incorrecta: {fake_clock.delays}"
    )
    assert db_connection._retry_attempt == 7


# ---------------------------------------------------------------------------
# 2. Reintentos ante fallos
# ---------------------------------------------------------------------------


async def test_retry_connect_reintenta_con_backoff_tras_fallos(
    clean_db_connection, fake_clock, monkeypatch
):
    """Fallos consecutivos de conexión disparan reintentos con backoff
    1, 2, 4; la cancelación controlada termina la tarea sin huérfanas."""
    state, factory, mock_client = _make_mongo_mocks()
    monkeypatch.setattr(mongodb, "AsyncIOMotorClient", factory)

    # Congelar la tarea dentro del tercer backoff para aserciones exactas
    fake_clock.stop_after = 3
    task = asyncio.create_task(retry_connect())
    await fake_clock.reached_stop.wait()

    assert factory.call_count == 3, "retry_connect debe reintentar la conexión"
    assert fake_clock.delays == [1, 2, 4], (
        f"backoff aplicado incorrecto: {fake_clock.delays}"
    )
    assert db_connection._retry_attempt == 3

    # Cancelación controlada: la tarea está congelada en el gate del backoff
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert task.cancelled()


# ---------------------------------------------------------------------------
# 3. Reinicio del contador tras recuperación completa
# ---------------------------------------------------------------------------


async def test_contador_reiniciado_tras_recuperacion_completa(
    clean_db_connection, fake_clock, monkeypatch
):
    """Dos fallos (backoff 1, 2) -> conexión completamente exitosa (ping +
    índice) reinicia _retry_attempt a 0 -> nuevo fallo -> backoff en 1s."""
    state, factory, mock_client = _make_mongo_mocks()
    monkeypatch.setattr(mongodb, "AsyncIOMotorClient", factory)

    # Fase 1: dos fallos consecutivos mediante retry_connect
    fake_clock.stop_after = 2
    task = asyncio.create_task(retry_connect())
    await fake_clock.reached_stop.wait()
    assert fake_clock.delays == [1, 2]
    assert db_connection._retry_attempt == 2

    # Cancelación controlada de la tarea (congelada en el 2do backoff)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert task.cancelled()

    # Fase 2: conexión completamente exitosa (ping + índice) -> contador a 0
    state["mongo_disponible"] = True
    await db_connection.connect()
    assert db_connection.connected is True
    assert db_connection._indexes_created is True
    assert db_connection._retry_attempt == 0

    # Fase 3: nuevo fallo; el siguiente backoff comienza en 1 segundo
    state["mongo_disponible"] = False
    with pytest.raises(ServerSelectionTimeoutError):
        await db_connection.connect()
    assert db_connection._retry_attempt == 0  # el fallo no reinicia ni incrementa
    await _apply_backoff()
    assert fake_clock.delays[-1] == 1
    assert db_connection._retry_attempt == 1


# ---------------------------------------------------------------------------
# 4. Una sola espera de backoff tras una caída
# ---------------------------------------------------------------------------


async def test_una_sola_espera_de_backoff_tras_caida(
    clean_db_connection, fake_clock, monkeypatch
):
    """Conexión exitosa -> caída detectada por el supervisor -> exactamente
    UNA espera de backoff (1s) antes del intento de reconexión."""
    state, factory, mock_client = _make_mongo_mocks()
    monkeypatch.setattr(mongodb, "AsyncIOMotorClient", factory)
    state["mongo_disponible"] = True  # partida con MongoDB disponible

    # Contar las esperas de backoff distinguiéndolas del polling interno:
    # registrar solo los delays consumidos por cada llamada a _apply_backoff.
    # Se congela la tarea tras la 1ra espera (pre-reconexión) y tras la 2da.
    real_backoff = mongodb._apply_backoff
    backoff_waits: list = []
    first_backoff_reached = asyncio.Event()
    second_backoff_reached = asyncio.Event()
    first_gate = asyncio.Event()
    second_gate = asyncio.Event()

    async def counting_backoff():
        antes = len(fake_clock.delays)
        await real_backoff()
        backoff_waits.extend(fake_clock.delays[antes:])
        if len(backoff_waits) == 1:
            first_backoff_reached.set()
            await first_gate.wait()
        elif len(backoff_waits) >= 2:
            second_backoff_reached.set()
            await second_gate.wait()

    monkeypatch.setattr(mongodb, "_apply_backoff", counting_backoff)

    task = asyncio.create_task(retry_connect())

    # Conexión exitosa + supervisión activa
    await fake_clock.wait_until(
        lambda: db_connection.connected
        and db_connection._supervision_task is not None
        and not db_connection._supervision_task.done()
    )

    # Simular la caída: el próximo ping del supervisor fallará
    state["mongo_disponible"] = False

    # El supervisor marca la caída y termina; retry_connect hace UNA espera
    await fake_clock.wait_until(lambda: not db_connection.connected)
    await first_backoff_reached.wait()

    assert backoff_waits == [1], (
        "la espera tras la caída debe ser de 1s (contador reiniciado); "
        f"esperas registradas: {backoff_waits}"
    )
    # Antes de reconectar: exactamente UNA espera y referencias preservadas
    # (el supervisor y la caída no descartan el cliente)
    assert db_connection.client is mock_client
    assert db_connection.db is not None
    assert db_connection._supervision_task is None

    # Liberar: retry_connect reconecta (falla) y entra al segundo backoff
    first_gate.set()
    await second_backoff_reached.wait()

    assert backoff_waits == [1, 2], (
        "entre la caída y la reconexión debe haber exactamente UNA espera"
    )
    assert factory.call_count == 2, (
        "debe reconectar exactamente una vez tras la caída"
    )

    # Cancelación controlada con la tarea congelada en el segundo backoff
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert task.cancelled()
    assert db_connection._supervision_task is None


# ---------------------------------------------------------------------------
# 5 (P2). Pings periódicos respetando SUPERVISION_INTERVAL
# ---------------------------------------------------------------------------


async def test_pings_periodicos_respetan_intervalo(
    clean_db_connection, fake_clock, monkeypatch
):
    """El supervisor espera 5 s, luego hace un ping cada SUPERVISION_INTERVAL
    (30 s). Con tres pings exitosos los delays son [5, 30, 30, 30]; mantiene
    el estado de conexión y no crea clientes ni reinicializa índices."""
    state, factory, mock_client = _make_mongo_mocks()
    monkeypatch.setattr(mongodb, "AsyncIOMotorClient", factory)
    state["mongo_disponible"] = True

    await db_connection.connect()
    assert db_connection.connected is True
    assert db_connection._indexes_created is True

    # Contar SOLO los pings del supervisor (el ping de connect() no cuenta) y
    # congelar la tarea dentro del 3er ping para aserciones exactas.
    real_ping = mock_client.admin.command
    supervisor_pings = 0
    tercer_ping_alcanzado = asyncio.Event()
    gate = asyncio.Event()

    async def counting_ping(command, *args, **kwargs):
        nonlocal supervisor_pings
        supervisor_pings += 1
        resultado = await real_ping(command, *args, **kwargs)
        if supervisor_pings == 3:
            tercer_ping_alcanzado.set()
            await gate.wait()
        return resultado

    mock_client.admin.command = AsyncMock(side_effect=counting_ping)

    await db_connection.start_supervision()
    task = db_connection._supervision_task
    await tercer_ping_alcanzado.wait()

    # Esperas exactas: 5 inicial + 30 por cada ciclo de supervisión
    assert fake_clock.delays == [5, 30, 30, 30], (
        f"el supervisor no respeta SUPERVISION_INTERVAL: {fake_clock.delays}"
    )
    assert supervisor_pings == 3
    # Los pings exitosos mantienen el estado
    assert db_connection.connected is True
    assert db_connection._indexes_created is True
    # El supervisor no crea clientes ni reinicializa índices
    assert factory.call_count == 1
    assert db_connection.db.__getitem__.return_value.create_index.call_count == 1
    # La tarea sigue activa mientras está congelada en el gate
    assert task.done() is False
    assert task.cancelled() is False

    # Cierre controlado: cancelar mientras está congelado dentro del ping
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert task.cancelled()


# ---------------------------------------------------------------------------
# 6 (P2). Detección de caída por parte del supervisor
# ---------------------------------------------------------------------------


async def test_caida_detectada_por_supervisor(
    clean_db_connection, fake_clock, monkeypatch
):
    """Con MongoDB disponible el supervisor hace pings exitosos; al simular
    la caída, el ping falla, marca el estado como caído y la tarea termina
    POR SÍ MISMA (sin excepción ni cancelación), sin crear otro cliente, sin
    cerrar el existente y preservando client/db."""
    state, factory, mock_client = _make_mongo_mocks()
    monkeypatch.setattr(mongodb, "AsyncIOMotorClient", factory)
    state["mongo_disponible"] = True

    await db_connection.connect()
    collection = db_connection.db.__getitem__.return_value
    index_calls_iniciales = collection.create_index.call_count

    await db_connection.start_supervision()
    task = db_connection._supervision_task

    # Primer ping exitoso del supervisor (el de connect() ya consumió 1)
    await fake_clock.wait_until(
        lambda: mock_client.admin.command.call_count >= 2
    )
    assert db_connection.connected is True

    # Simular la caída: el próximo ping del supervisor fallará
    state["mongo_disponible"] = False

    # El supervisor termina por sí mismo tras el ping fallido
    await fake_clock.wait_until(lambda: task.done())

    assert task.cancelled() is False, "la tarea no debió cancelarse"
    assert task.exception() is None, "la tarea no debió terminar con error"
    assert db_connection.connected is False
    assert db_connection._indexes_created is False
    # No reconecta por su cuenta: no crea otro cliente ni reinicializa índices
    assert factory.call_count == 1
    assert collection.create_index.call_count == index_calls_iniciales
    # No cierra ni descarta el cliente: client/db se conservan
    assert mock_client.close.call_count == 0
    assert db_connection.client is mock_client
    assert db_connection.db is not None


# ---------------------------------------------------------------------------
# 7 (P2). Cancelación del supervisor
# ---------------------------------------------------------------------------


async def test_cancelacion_del_supervisor(
    clean_db_connection, fake_clock, monkeypatch
):
    """Cancelar la tarea del supervisor propaga CancelledError, marca el
    estado como desconectado y conserva client/db sin cerrar el cliente."""
    state, factory, mock_client = _make_mongo_mocks()
    monkeypatch.setattr(mongodb, "AsyncIOMotorClient", factory)
    state["mongo_disponible"] = True

    await db_connection.connect()

    # Congelar al supervisor dentro del bucle (tras el primer ping exitoso):
    # delays [5, 30, 30]; el congelamiento ocurre en el 2do sleep del bucle
    fake_clock.stop_after = 3
    await db_connection.start_supervision()
    task = db_connection._supervision_task
    await fake_clock.reached_stop.wait()

    assert task.done() is False
    assert db_connection.connected is True  # primer ping exitoso

    # Cancelación explícita mientras está congelado en un sleep del bucle
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert task.cancelled() is True, "CancelledError debió propagarse"
    assert db_connection.connected is False
    assert db_connection._indexes_created is False
    # Referencias conservadas para que disconnect() cierre los recursos
    assert db_connection.client is mock_client
    assert db_connection.db is not None
    assert mock_client.close.call_count == 0


# ---------------------------------------------------------------------------
# 8 (P2). stop_supervision en tres estados: activo / terminado / inexistente
# ---------------------------------------------------------------------------


async def test_stop_supervision_en_tres_estados(
    clean_db_connection, fake_clock, monkeypatch
):
    """stop_supervision() cancela y espera la tarea activa, limpia la
    referencia de una tarea ya terminada por caída y es un no-op seguro si
    no existe tarea; nunca deja excepciones sin recuperar."""
    state, factory, mock_client = _make_mongo_mocks()
    monkeypatch.setattr(mongodb, "AsyncIOMotorClient", factory)
    state["mongo_disponible"] = True

    await db_connection.connect()

    # --- a) Supervisor activo: cancela y espera la tarea ---
    await db_connection.start_supervision()
    task_a = db_connection._supervision_task
    assert task_a is not None and not task_a.done()

    await db_connection.stop_supervision()

    assert task_a.cancelled() is True, "la tarea activa debió ser cancelada"
    assert db_connection._supervision_task is None
    # Si CancelledError escapara de stop_supervision, el test fallaría aquí

    # --- b) Supervisor ya terminado (por caída): limpia la referencia ---
    state["mongo_disponible"] = False
    await db_connection.start_supervision()
    task_b = db_connection._supervision_task
    await fake_clock.wait_until(lambda: task_b.done())
    assert task_b.cancelled() is False
    assert task_b.exception() is None

    await db_connection.stop_supervision()

    assert db_connection._supervision_task is None

    # --- c) Sin supervisor: no-op seguro ---
    await db_connection.stop_supervision()

    assert db_connection._supervision_task is None


# ---------------------------------------------------------------------------
# 9 (P3). Ciclo completo de recuperación: caída -> backoff -> reconexión
# ---------------------------------------------------------------------------


async def test_ciclo_completo_de_recuperacion(
    clean_db_connection, fake_clock, monkeypatch
):
    """Conexión exitosa -> caída detectada -> UNA espera de backoff (congelada)
    -> MongoDB vuelve -> reconexión completa con recreate_index,
    indexes_ready True y contador de reintentos reiniciado."""
    state, factory, mock_client = _make_mongo_mocks()
    monkeypatch.setattr(mongodb, "AsyncIOMotorClient", factory)
    state["mongo_disponible"] = True

    # Contar las esperas de backoff y congelar la tarea en la primera
    # (patrón de P1 test 4, pero aquí se libera hacia una reconexión EXITOSA)
    real_backoff = mongodb._apply_backoff
    backoff_waits: list = []
    primera_espera_alcanzada = asyncio.Event()
    gate = asyncio.Event()

    async def counting_backoff():
        antes = len(fake_clock.delays)
        await real_backoff()
        backoff_waits.extend(fake_clock.delays[antes:])
        if len(backoff_waits) == 1:
            primera_espera_alcanzada.set()
            await gate.wait()

    monkeypatch.setattr(mongodb, "_apply_backoff", counting_backoff)

    task = asyncio.create_task(retry_connect())

    # 1) Conexión inicial exitosa + supervisión activa
    await fake_clock.wait_until(
        lambda: db_connection.connected
        and db_connection._supervision_task is not None
        and not db_connection._supervision_task.done()
    )
    collection = db_connection.db.__getitem__.return_value
    index_calls_iniciales = collection.create_index.call_count
    factory_calls_iniciales = factory.call_count
    assert index_calls_iniciales == 1
    assert factory_calls_iniciales == 1

    # 2) Caída de MongoDB: el supervisor la detecta y termina por sí mismo
    state["mongo_disponible"] = False
    await fake_clock.wait_until(lambda: not db_connection.connected)

    # 3) retry_connect queda congelada en la ÚNICA espera de backoff
    await primera_espera_alcanzada.wait()
    assert backoff_waits == [1], (
        f"debe haber exactamente UNA espera antes de recuperar: {backoff_waits}"
    )
    assert db_connection._supervision_task is None

    # 4) MongoDB vuelve: liberar la espera -> reconexión completa
    state["mongo_disponible"] = True
    gate.set()
    await fake_clock.wait_until(
        lambda: db_connection.connected
        and db_connection._supervision_task is not None
        and not db_connection._supervision_task.done()
    )

    # 5) Recuperación completa verificada
    assert factory.call_count == factory_calls_iniciales + 1, (
        "retry_connect debe crear una NUEVA conexión tras la caída"
    )
    assert collection.create_index.call_count == index_calls_iniciales + 1, (
        "el índice obligatorio debe recrearse en la reconexión"
    )
    assert db_connection._indexes_created is True
    assert db_connection.indexes_ready is True
    assert db_connection._retry_attempt == 0, (
        "el contador de reintentos debe reiniciarse tras la recuperación"
    )
    assert len(backoff_waits) == 1, "no debe haber esperas de backoff duplicadas"

    # Limpieza: cancelación controlada de la tarea de conexión
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert task.cancelled()


# ---------------------------------------------------------------------------
# 10 (P3). /ready 503 -> 200 y /health 200 (sin lifespan)
# ---------------------------------------------------------------------------


def test_ready_y_health_503_a_200(monkeypatch):
    """Sin lifespan: en estado desconectado /ready es 503 (con detalle
    completo) y /health es 200 (con cuerpo completo); tras simular una
    recuperación completa /ready pasa a 200. El estado del singleton se
    restaura automáticamente vía monkeypatch."""
    # Estado desconectado (recuperación pendiente)
    monkeypatch.setattr(db_connection, "db", None)
    monkeypatch.setattr(db_connection, "_connected", False)
    monkeypatch.setattr(db_connection, "_indexes_created", False)

    # Sin `with`: el lifespan NO se ejecuta, no hay tareas de conexión
    tc = TestClient(app)

    ready = tc.get("/ready")
    assert ready.status_code == 503
    assert ready.json()["detail"] == {
        "database": "mongodb_unavailable",
        "reason": "Índices no listos o MongoDB no disponible",
    }

    health = tc.get("/health")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "service": "ms3-crud",
        "timestamp": "live",
    }

    # Estado de recuperación completa simulada
    monkeypatch.setattr(db_connection, "db", MagicMock())
    monkeypatch.setattr(db_connection, "_connected", True)
    monkeypatch.setattr(db_connection, "_indexes_created", True)

    ready = tc.get("/ready")
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready", "database": "mongodb_connected"}


# ---------------------------------------------------------------------------
# 11 (P3). Arranque sin MongoDB con el lifespan real
# ---------------------------------------------------------------------------


async def test_arranque_sin_mongodb_health_y_ready(
    clean_db_connection, fake_clock, monkeypatch
):
    """El lifespan arranca con MongoDB caído: la tarea retry_connect queda
    congelada en su primer backoff; /health es 200 y /ready es 503; al cerrar
    el TestClient el apagado graceful cancela la tarea y limpia el estado."""
    state, factory, mock_client = _make_mongo_mocks()
    # CRÍTICO: parchear el cliente Motor ANTES de iniciar el contexto, para
    # que el lifespan nunca intente una conexión real (ping siempre falla).
    monkeypatch.setattr(mongodb, "AsyncIOMotorClient", factory)

    # Registrar la tarea creada por el lifespan para verificar su cancelación
    connection_tasks: list = []
    real_retry_connect = src_main.retry_connect

    async def recording_retry_connect():
        connection_tasks.append(asyncio.current_task())
        await real_retry_connect()

    monkeypatch.setattr(src_main, "retry_connect", recording_retry_connect)

    # Congelar la conexión en su PRIMER backoff (delay 1): sin spin de CPU
    fake_clock.stop_after = 1

    with TestClient(app) as tc:
        # La tarea quedó congelada en el primer backoff; el Event fue
        # señalado desde el loop del portal (otro hilo), por lo que se sondea
        # con is_set() (thread-safe) y NO con await directo del Event.
        await fake_clock.wait_until(lambda: fake_clock.reached_stop.is_set())
        assert fake_clock.delays == [1]
        assert len(connection_tasks) == 1

        # Liveness independiente de MongoDB
        health = tc.get("/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "ok",
            "service": "ms3-crud",
            "timestamp": "live",
        }

        # Readiness: MongoDB no disponible
        ready = tc.get("/ready")
        assert ready.status_code == 503
        assert ready.json()["detail"] == {
            "database": "mongodb_unavailable",
            "reason": "Índices no listos o MongoDB no disponible",
        }
        assert db_connection.connected is False
        assert db_connection.client is None

    # Tras el apagado graceful del lifespan:
    connection_task = connection_tasks[0]
    assert connection_task.done(), "la tarea de conexión debió finalizar"
    assert connection_task.cancelled(), (
        "la tarea de conexión debió ser cancelada por el lifespan"
    )
    # disconnect() ejecutado: no hay cliente ni supervisión residual
    assert db_connection.client is None
    assert db_connection.db is None
    assert db_connection._supervision_task is None
    # Sin tareas pendientes en el loop del test (all_tasks excluye las done)
    assert [
        t for t in asyncio.all_tasks() if t is not asyncio.current_task()
    ] == []


# ---------------------------------------------------------------------------
# 12 (P4). Ciclo de vida completo CON supervisión activa
# ---------------------------------------------------------------------------


async def test_ciclo_de_vida_completo_con_supervision_activa(
    clean_db_connection, fake_clock, monkeypatch
):
    """Con MongoDB disponible, el lifespan conecta y arranca la supervisión;
    al cerrar el TestClient cancela retry_connect(), detiene la supervisión
    y disconnect() cierra el cliente y limpia todo el estado."""
    state, factory, mock_client = _make_mongo_mocks()
    monkeypatch.setattr(mongodb, "AsyncIOMotorClient", factory)
    state["mongo_disponible"] = True

    # Congelar al supervisor dentro de su primer ping (el 2do ping total:
    # el 1ro es el de connect()). Sistema totalmente congelado: cero spin.
    real_ping = mock_client.admin.command
    total_pings = 0
    primer_ping_supervisor = asyncio.Event()
    gate_supervisor = asyncio.Event()

    async def ping_controlado(command, *args, **kwargs):
        nonlocal total_pings
        total_pings += 1
        resultado = await real_ping(command, *args, **kwargs)
        if total_pings == 2:
            primer_ping_supervisor.set()
            await gate_supervisor.wait()
        return resultado

    mock_client.admin.command = AsyncMock(side_effect=ping_controlado)

    # Capturar la tarea de conexión creada por el lifespan (corre en el loop
    # del portal, invisible para asyncio.all_tasks() del loop del test)
    connection_tasks: list = []
    real_retry_connect = src_main.retry_connect

    async def recording_retry_connect():
        connection_tasks.append(asyncio.current_task())
        await real_retry_connect()

    monkeypatch.setattr(src_main, "retry_connect", recording_retry_connect)

    with TestClient(app):
        # El Event fue señalado desde el loop del portal (otro hilo):
        # sondeo con is_set(), nunca await directo.
        await fake_clock.wait_until(lambda: primer_ping_supervisor.is_set())
        assert len(connection_tasks) == 1
        connection_task = connection_tasks[0]
        assert not connection_task.done()
        supervision_task = db_connection._supervision_task
        assert supervision_task is not None and not supervision_task.done()
        assert db_connection.connected is True
        assert db_connection.client is mock_client
        assert mock_client.close.call_count == 0

    # Tras el apagado graceful del lifespan:
    assert connection_task.cancelled(), (
        "la tarea de conexión debió ser cancelada por el lifespan"
    )
    assert db_connection._supervision_task is None, (
        "la supervisión debió detenerse y limpiarse"
    )
    assert mock_client.close.call_count == 1, (
        "disconnect() debió cerrar el cliente (close() es síncrono)"
    )
    assert db_connection.client is None
    assert db_connection.db is None
    assert db_connection.connected is False
    assert db_connection._indexes_created is False
    # Sin tareas pendientes nuevas en el loop del test
    assert [
        t for t in asyncio.all_tasks() if t is not asyncio.current_task()
    ] == []


# ---------------------------------------------------------------------------
# 13 (P4). Cancelación durante el backoff
# ---------------------------------------------------------------------------


async def test_cancelacion_durante_backoff_conserva_recursos(
    clean_db_connection, fake_clock, monkeypatch
):
    """Fase 1 (literal): Mongo caído desde el inicio; la cancelación en el
    primer backoff no deja recursos (connect() fallido ya los limpió).
    Fase 2: conexión previa exitosa -> caída -> cancelación en el backoff
    post-caída ANTES de reconectar: CancelledError propagada y recursos
    client/db conservados para disconnect()."""
    state, factory, mock_client = _make_mongo_mocks()
    monkeypatch.setattr(mongodb, "AsyncIOMotorClient", factory)

    real_backoff = mongodb._apply_backoff
    backoff_waits: list = []
    backoff_alcanzado = asyncio.Event()
    gate = asyncio.Event()
    congelar_en = {"espera": 1}

    async def counting_backoff():
        antes = len(fake_clock.delays)
        await real_backoff()
        backoff_waits.extend(fake_clock.delays[antes:])
        if len(backoff_waits) >= congelar_en["espera"]:
            backoff_alcanzado.set()
            await gate.wait()

    monkeypatch.setattr(mongodb, "_apply_backoff", counting_backoff)

    # --- Fase 1: Mongo caído desde el inicio ---
    tarea_fase1 = asyncio.create_task(retry_connect())
    await backoff_alcanzado.wait()

    assert backoff_waits == [1]
    assert tarea_fase1.done() is False
    assert db_connection.client is None, (
        "connect() fallido limpió los recursos: no hay nada que conservar"
    )
    assert db_connection.db is None
    assert db_connection._supervision_task is None

    tarea_fase1.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await tarea_fase1
    assert tarea_fase1.cancelled()
    assert db_connection._supervision_task is None

    # --- Fase 2: conexión exitosa previa, cancelación en el backoff
    # post-caída (la 2da espera registrada) ---
    backoff_alcanzado.clear()
    congelar_en["espera"] = 2
    state["mongo_disponible"] = True
    tarea_fase2 = asyncio.create_task(retry_connect())
    await fake_clock.wait_until(
        lambda: db_connection.connected
        and db_connection._supervision_task is not None
        and not db_connection._supervision_task.done()
    )

    state["mongo_disponible"] = False
    await fake_clock.wait_until(lambda: not db_connection.connected)
    await backoff_alcanzado.wait()

    # La 2da espera es de 1s: connect() exitoso reinició el contador
    assert backoff_waits == [1, 1]
    assert tarea_fase2.done() is False
    assert db_connection._supervision_task is None
    assert db_connection.client is mock_client
    assert db_connection.db is not None

    # Línea base de cierres: el close() de la fase 1 es la limpieza propia
    # de connect() ante ping fallido (no obra de ninguna cancelación)
    cerrados_antes_de_cancelar = mock_client.close.call_count

    # Cancelación congelada en el backoff, antes de reconectar
    tarea_fase2.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await tarea_fase2

    assert tarea_fase2.cancelled(), "CancelledError debió propagarse"
    assert db_connection._connected is False
    assert db_connection._indexes_created is False
    # Conservación: la cancelación NO descarta client/db (disconnect()
    # los cerrará). El único close() registrado (fase 1) es la limpieza
    # propia de connect() ante ping fallido, no obra de la cancelación.
    assert db_connection.client is mock_client
    assert db_connection.db is not None
    assert db_connection._supervision_task is None
    assert mock_client.close.call_count == cerrados_antes_de_cancelar, (
        "la cancelación no debe cerrar el cliente: es trabajo de disconnect()"
    )


# ---------------------------------------------------------------------------
# 14 (P4). Cancelación durante la reconexión (dentro de connect())
# ---------------------------------------------------------------------------


async def test_cancelacion_durante_reconexion_dentro_de_connect(
    clean_db_connection, fake_clock, monkeypatch
):
    """Conexión exitosa -> caída -> la reconexión queda bloqueada dentro de
    connect() (ping suspendido en un Event) -> cancelar ahí: CancelledError
    atraviesa connect() SIN limpiar recursos, la supervisión queda limpia y
    client/db se conservan para disconnect()."""
    state, factory, mock_client = _make_mongo_mocks()
    monkeypatch.setattr(mongodb, "AsyncIOMotorClient", factory)
    state["mongo_disponible"] = True
    state["bloquear_siguiente_ping"] = False

    # Ping controlado: bloquea SOLO el ping de la reconexión
    ping_original = mock_client.admin.command
    ping_bloqueado = asyncio.Event()
    bloqueo = asyncio.Event()

    async def ping_controlado(command, *args, **kwargs):
        if state["bloquear_siguiente_ping"]:
            ping_bloqueado.set()
            await bloqueo.wait()
        return await ping_original(command, *args, **kwargs)

    mock_client.admin.command = AsyncMock(side_effect=ping_controlado)

    # El wrapper del backoff activa el bloqueo del PRÓXIMO ping (el de la
    # reconexión): lo ejecuta la propia tarea de retry_connect antes de
    # llamar a connect(), sin carreras y sin afectar al supervisor (que ya
    # terminó por la caída cuando el flag se activa).
    real_backoff = mongodb._apply_backoff
    backoff_waits: list = []

    async def counting_backoff():
        antes = len(fake_clock.delays)
        await real_backoff()
        backoff_waits.extend(fake_clock.delays[antes:])
        state["bloquear_siguiente_ping"] = True

    monkeypatch.setattr(mongodb, "_apply_backoff", counting_backoff)

    task = asyncio.create_task(retry_connect())

    # 1) Conexión inicial exitosa + supervisión activa
    await fake_clock.wait_until(
        lambda: db_connection.connected
        and db_connection._supervision_task is not None
        and not db_connection._supervision_task.done()
    )
    collection = db_connection.db.__getitem__.return_value
    assert collection.create_index.call_count == 1

    # 2) Caída: el supervisor termina por sí mismo
    state["mongo_disponible"] = False
    await fake_clock.wait_until(lambda: not db_connection.connected)
    assert db_connection._supervision_task is None
    assert db_connection.client is mock_client
    assert db_connection.db is not None

    # 3) La reconexión queda bloqueada dentro de connect() (ping suspendido)
    await ping_bloqueado.wait()  # mismo loop: await directo seguro
    assert task.done() is False
    assert factory.call_count == 2, "el intento de reconexión creó un cliente"
    assert collection.create_index.call_count == 1, (
        "la reconexión quedó a mitad de camino: sin índice recreado"
    )

    # 4) Cancelación dentro de connect(): el ping está en el Event.
    # Línea base de cierres: connect() ya cerró al "cliente anterior" al
    # entrar a reconectar (lógica propia de connect(), no de la cancelación)
    cerrados_antes_de_cancelar = mock_client.close.call_count
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert task.cancelled(), "CancelledError debió propagarse desde connect()"
    # La supervisión quedó limpia y los recursos conservados: ni connect()
    # (except Exception no captura CancelledError) ni retry_connect
    # descartan client/db al cancelarse; el cierre es de disconnect().
    assert db_connection._supervision_task is None
    assert db_connection.client is mock_client
    assert db_connection.db is not None
    assert db_connection._connected is False
    assert db_connection._indexes_created is False
    assert mock_client.close.call_count == cerrados_antes_de_cancelar, (
        "la cancelación no debe cerrar el cliente: es trabajo de disconnect()"
    )
