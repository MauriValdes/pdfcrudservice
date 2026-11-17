import asyncio
import logging

from motor.motor_asyncio import AsyncIOMotorClient

from src.config.config import settings

logger = logging.getLogger(__name__)

# Intervalo de supervisión en segundos (configurable)
SUPERVISION_INTERVAL = 30


class MongoDB:
    """Gestiona la conexión a MongoDB con reintentos y recuperación automática.

    Sigue el principio 4 de Twelve-Factor App: los backing services se tratan
    como recursos adjuntos. La aplicación puede iniciar aunque MongoDB no esté
    disponible y se recupera automáticamente cuando vuelve.

    Características:
    - Conexión perezosa: FastAPI inicia aunque MongoDB esté caído.
    - Reintentos indefinidos con backoff exponencial.
    - Supervisión continua mediante ping para detectar desconexiones.
    - Recuperación automática cuando MongoDB vuelve a estar disponible.
    - Índices creados idempotentemente; reintentados tras recuperación.
    """

    def __init__(self):
        self.client = None
        self.db = None
        self._connected = False
        # Flag para hacer seguimiento si los índices obligatorios se crearon exitosamente
        self._indexes_created = False
        # Contador de reintentos para backoff exponencial (único, gestionado por retry_connect)
        self._retry_attempt = 0
        # Bandera para indicar si el supervisor debe terminar por caída detectada
        self._supervision_shutdown = False
        # Tarea de supervisión en background (creada por start_supervision)
        self._supervision_task = None

    @property
    def connected(self) -> bool:
        """Indica si la conexión actual está activa."""
        return self._connected

    @property
    def indexes_ready(self) -> bool:
        """Indica si los índices obligatorios están inicializados sobre la conexión actual.

        Returns True solo si hay una conexión activa Y los índices obligatorios
        ('checksum_unique') se crearon exitosamente. Un bucle de supervisión
        continuo intenta re-crear los índices si fallaron inicialmente.

        Esta propiedad es síncrona para evitar coroutines sin await en contexto
        de tests y endpoints síncronos.
        """
        if not self._connected or self.db is None:
            return False
        return self._indexes_created

    async def connect(self, database_name: str = None):
        """Intenta conectar a MongoDB y dejar los índices obligatorios listos.

        Este método puede lanzarse varias veces desde retry_connect().
        Si el ping o la creación del índice fallan, limpia el estado y los
        recursos y PROPAGA la excepción: la conexión sólo se considera
        exitosa cuando ping e índices terminaron correctamente.

        Args:
            database_name: Nombre de la base de datos. Si no se provee, usa
                la configuración por defecto.

        Raises:
            Exception: si el ping falla o si no se pudo crear el índice
                obligatorio 'checksum_unique'.
        """
        # Crear cliente MongoDB (único, no acumular instancias)
        if self.client is not None:
            try:
                close_result = self.client.close()
                # Soporte para mocks asíncronos cuyo close() devuelve coroutine
                if asyncio.iscoroutine(close_result):
                    await close_result
                logger.info("Cliente MongoDB anterior cerrado antes de reconectar")
            except Exception:
                logger.warning("Error al cerrar cliente MongoDB anterior")
            self.client = None
            self.db = None

        self.client = AsyncIOMotorClient(settings.mongo_url)

        # Selección dinámica de la base de datos
        if database_name:
            self.db = self.client[database_name]
        else:
            self.db = self.client[settings.mongo_db_dev]

        # Verificar conexión con ping; si falla, limpiar y propagar
        try:
            await self.client.admin.command('ping')
            logger.info("Ping a MongoDB exitoso")
        except Exception as e:
            logger.warning(f"Ping a MongoDB falló: {e}. No se establece conexión.")
            self._connected = False
            self._indexes_created = False
            try:
                close_result = self.client.close()  # close() es síncrono, no usar await
                if asyncio.iscoroutine(close_result):
                    await close_result
            except Exception:
                pass
            self.client = None
            self.db = None
            logger.error("No se pudo establecer conexión con MongoDB - ping falló")
            raise

        # Ping exitoso: recién ahora la conexión está establecida
        self._connected = True
        logger.info(f"Conectado a MongoDB en la base de datos: {self.db.name}")

        # Verificar o crear índice único sobre 'checksum' de forma idempotente.
        # La conexión NO está completa hasta que el índice exista.
        try:
            await self._ensure_unique_index_on_checksum()
        except Exception as e:
            logger.error(
                f"Fallo la inicialización del índice obligatorio: "
                f"la conexión no se considera completa: {e}"
            )
            self._connected = False
            self._indexes_created = False
            try:
                close_result = self.client.close()  # close() es síncrono, no usar await
                if asyncio.iscoroutine(close_result):
                    await close_result
            except Exception:
                pass
            self.client = None
            self.db = None
            raise

        # Recuperación completa (ping + índices): recién aquí se reinicia el
        # contador de reintentos que gestiona retry_connect.
        self._retry_attempt = 0

    async def _ensure_unique_index_on_checksum(self):
        """Crea un índice único sobre el campo 'checksum' si no existe.

        La operación es idempotente: si el índice ya existe con la misma configuración,
        MongoDB no lanza error. Si el índice se crea exitosamente, actualiza el
        estado interno para reflejar que los índices obligatorios están disponibles.

        Si falla la creación, marca _indexes_created = False, registra el error
        y PROPAGA la excepción para que el caller (connect / retry_connect)
        sepa que la recuperación no está completa.

        Raises:
            Exception: si no hay base de datos seleccionada o si la creación
                del índice falla.
        """
        if self.db is None:
            self._indexes_created = False
            logger.error("No hay base de datos seleccionada; no se puede crear el índice 'checksum_unique'.")
            raise RuntimeError("No hay base de datos seleccionada para crear el índice 'checksum_unique'.")
        try:
            await self.db[settings.mongo_collection].create_index(
                [("checksum", 1)],
                name="checksum_unique",
                unique=True,
            )
            logger.info("Índice único 'checksum_unique' creado o ya existente.")
            # Confirmar que los índices obligatorios están disponibles
            self._indexes_created = True
        except Exception as e:
            logger.error(f"Error al crear índice único en 'checksum': {e}")
            # Controlar que el índice falló creación
            self._indexes_created = False
            raise

    async def disconnect(self):
        """Cierra la conexión con MongoDB de forma gracefull."""
        if self.client:
            self.client.close()
            logger.info("Conexión con MongoDB cerrada")
        self._connected = False
        self.client = None
        self.db = None
        self._indexes_created = False

    # Método alias por compatibilidad con código existente que usa .close()
    async def close(self):
        await self.disconnect()

    # -----------------------------------------------------------------
    # Supervisión continua de la conexión
    # -----------------------------------------------------------------

    async def _supervise_loop(self):
        """Bucle de supervisión continua.

        Se ejecuta en background después de conectarse exitosamente.
        Esta es la única tarea de supervisión. Su trabajo es:
        - Cada SUPERVISION_INTERVAL segundos hacer ping a MongoDB.
        - Si el ping falla, marcar _connected = False y _indexes_created = False,
          y FINALIZAR su ciclo sin intentar reconectar.
        - Si el ping tiene éxito, actualizar _connected = True.
        - Propagar correctamente las cancelaciones para apagado graceful.
        - NUNCA crea ni cierra clientes MongoDB.
        - NUNCA intenta reconectar - eso lo hace retry_connect exclusivamentE.

        La finalización del supervisor permite que retry_connect() detecte la
        caída y asuma la reconexión con backoff correspondiente.
        """
        logger.info("Iniciando bucle de supervisión continua de MongoDB")
        # Primera verificación: esperar un poco para que el sistema estabilice
        await asyncio.sleep(5)

        while True:
            try:
                # Ping cada SUPERVISION_INTERVAL segundos
                await asyncio.sleep(SUPERVISION_INTERVAL)
                # Verificar si alguien solicitó apagado antes de hacer ping
                if self._supervision_shutdown:
                    logger.info("Supervisión cancelada: se solicitó apagado antes de ping")
                    self._connected = False
                    self._indexes_created = False
                    # NO se descarta el cliente: disconnect() es responsable
                    # del cierre efectivo de los recursos.
                    return
                await self.client.admin.command('ping')
                # Si llegamos aquí, el ping fue exitoso
                if not self._connected:
                    logger.info("MongoDB recuperado después de una caída previa")
                self._connected = True
                # El supervisor se limita al ping periódico y a la detección
                # de caída de MongoDB.
                # La creación/verificación de índices es responsabilidad
                # exclusiva de connect(), que la ejecuta antes de considerar
                # exitosa una conexión o reconexión.
            except asyncio.CancelledError:
                # Se recibió señal de cancelación durante el apagado
                logger.info("Supervisión cancelada durante apagado")
                self._connected = False
                self._indexes_created = False
                # NO se descarta el cliente: se conservan las referencias
                # client/db para que disconnect() cierre los recursos.
                raise
            except Exception as e:
                # El ping falló: MongoDB podría haber caído
                logger.warning(f"Ping a MongoDB falló: {e}. Marcando conexión como caída.")
                # Marcar conexión como caída e índices no disponibles
                self._connected = False
                self._indexes_created = False
                # No se reconecta, no se aplica backoff y NO se cierra ni
                # descarta el cliente: disconnect() es responsable del cierre.
                # Finalizar inmediatamente para que retry_connect tome el
                # control, independientemente de _supervision_shutdown.
                return

    async def start_supervision(self):
        """Inicia la tarea de supervisión en background.

        Crea y guarda una única tarea de supervisión. Si ya existe, la reutiliza.
        Debe llamarse después de una conexión exitosa a MongoDB.
        """
        # Evitar crear múltiples tareas
        if self._supervision_task is not None and not self._supervision_task.done():
            logger.warning("Ya existe una tarea de supervisión activa; se omite crear otra.")
            return

        self._supervision_task = asyncio.create_task(self._supervise_loop())
        logger.info("Tarea de supervisión MongoDB iniciada en background")

    async def stop_supervision(self):
        """Detiene la tarea de supervisión de forma gracefull.

        Cancela la tarea si sigue activa, espera su finalización y limpia
        la referencia incluso si la tarea ya había terminado por sí sola.
        """
        task = self._supervision_task
        # Limpiar la referencia siempre, incluso si la tarea ya terminó
        self._supervision_task = None
        if task is None:
            return
        if not task.done():
            task.cancel()
        try:
            # Esperar la finalización (también recupera el resultado/excepción
            # de una tarea ya terminada para evitar warnings de "never retrieved")
            await task
        except asyncio.CancelledError:
            logger.info("Tarea de supervisión MongoDB cancelada durante apagado")
        except Exception as e:
            logger.warning(f"La tarea de supervisión terminó con error: {e}")
        logger.info("Tarea de supervisión detenida")


# Función auxiliar para solicitar apagado ordenado del supervisor
async def request_supervision_shutdown():
    """Solicita al supervisor que termine su ciclo de forma ordenada.

    Esta función es llamada por retry_connect cuando necesita reconectar.
    El supervisor finalizará su ciclo actual y regresará, permitiendo
    que retry_connect aplique backoff y reconecte.
    """
    db_connection._supervision_shutdown = True


# Instancia global del gestor de MongoDB
db_connection = MongoDB()


# -----------------------------------------------------------------
# Función standalone para conexión con reintentos (usada en main.py lifespan)
# -----------------------------------------------------------------

async def retry_connect():
    """Intenta reconectar a MongoDB aplicando backoff exponencial de forma indefinida.

    Esta función es el ÚNICO responsable de conectar y reconectar MongoDB.

    Arquitectura:
    1. retry_connect llama a db_connection.connect()
    2. Si connect() tiene éxito: inicia UNA tarea de supervisión
    3. retry_connect espera a que la supervisión termine (por caída o cancelación)
    4. Cuando el supervisor termina, retry_connect aplica backoff y vuelve a conectar
    5. Si connect() falla directamente: aplica backoff y vuelve a intentar

    Secuencia de backoff: 1, 2, 4, 8, 16, 30, 30, 30... segundos.
    El contador _retry_attempt se reinicia después de cada recuperación exitosa.

    Esta función NUNCA termina definitivamente después de una conexión exitosa.
    Espera a que ocurra una caída real o una cancelación antes de reconectar.
    """
    logger.info("Tarea retry_connect iniciada: único responsable de conexión MongoDB")

    while True:
        try:
            # Paso 1: Intentar conectar
            await db_connection.connect()

            # Paso 2: Verificar conexión
            if not db_connection.connected:
                # connect() falló o no estableció conexión; aplicar backoff
                logger.warning("No fue posible establecer conexión; aplicando backoff exponencial")
                await _apply_backoff()
                continue

            # Paso 3: Conexión exitosa - iniciar supervisión UNA sola vez
            # Cancelar supervisión anterior si existe
            await db_connection.stop_supervision()
            db_connection._supervision_task = None
            db_connection._supervision_shutdown = False

            # Iniciar supervisión continua
            await db_connection.start_supervision()
            logger.info("Supervisión MongoDB iniciada, esperando eventos...")

            # Paso 4: Esperar a que ocurra una caída real o cancelación
            # Este bucle espera a que el supervisor detecte una caída o reciba cancelación
            try:
                # Esperamos indefinidamente; el supervisor nos notificará
                # cambiando _connected a False cuando detecte una caída
                while db_connection.connected:
                    # Pausa corta para no consumir CPU
                    await asyncio.sleep(1)

                # Si salimos del while, significa que connected se puso False
                # (ya sea por caída detectada por supervisor o cancelación)
                logger.info("retry_connect: conexión caída detectada, terminando supervisión")

                # Detener supervisión antes de reconectar
                await db_connection.stop_supervision()
                db_connection._supervision_task = None

            except asyncio.CancelledError:
                # Se recibió señal de cancelación durante el apagado
                logger.info("retry_connect: cancelado durante apagado")
                # Detener supervisión si está corriendo
                try:
                    await db_connection.stop_supervision()
                except Exception:
                    pass
                db_connection._supervision_task = None
                # Marcar estado como caído, pero NO descartar client/db:
                # se conservan las referencias para que disconnect() cierre
                # los recursos durante el apagado.
                db_connection._connected = False
                db_connection._indexes_created = False
                raise

            # Si llegamos aquí, la conexión cayó y ya no hay supervisor corriendo
            # Aplicar backoff y volver a intentar conexión (ÚNICA espera por este fallo)
            await _apply_backoff()

        except asyncio.CancelledError:
            # Se recibió señal de cancelación durante el apagado
            logger.info("retry_connect: cancelado durante apagado - propagando")
            # Detener supervisión si sigue activa (no-op si ya terminó)
            try:
                await db_connection.stop_supervision()
            except Exception:
                pass
            # Marcar estado como caído, pero NO descartar client/db:
            # se conservan las referencias para que disconnect() cierre
            # los recursos durante el apagado.
            db_connection._connected = False
            db_connection._indexes_created = False
            raise

        except Exception as e:
            logger.error(f"Error en tarea retry_connect: {e}")
            # Fallo inesperado: exactamente UNA espera de backoff por este fallo
            await _apply_backoff()


async def _apply_backoff():
    """Aplica la secuencia de backoff exponencial: 1, 2, 4, 8, 16, 30, 30, 30... segundos.

    Este es el único lugar que gestiona el contador de reintentos.
    El contador se reinicia después de una recuperación exitosa.

    Lógica:
    - Intento 1: 1 segundo
    - Intento 2: 2 segundos
    - Intento 3: 4 segundos
    - Intento 4: 8 segundos
    - Intento 5: 16 segundos
    - Intento 6 y posteriores: 30 segundos (máximo)
    - Después de una recuperación exitosa, _retry_attempt se reinicia en connect()
    """
    # Calcular backoff basándose en el contador actual
    # El contador se reinicia a 0 después de cada conexión exitosa en connect()
    attempt_num = db_connection._retry_attempt + 1

    # Secuencia: 1, 2, 4, 8, 16, 30, 30, 30...
    backoff_map = {1: 1, 2: 2, 3: 4, 4: 8, 5: 16}
    backoff = backoff_map.get(attempt_num, 30)  # 6+ intentos = 30s

    # Incrementar contador para próximo fallo.
    # SIN reinicio por límite de intentos: la secuencia continúa indefinidamente
    # como 1, 2, 4, 8, 16, 30, 30, 30... El contador SOLO se reinicia en
    # connect(), tras una recuperación completa (ping + índices).
    db_connection._retry_attempt += 1

    logger.info(f"Reintentando conexión en {backoff}s (intento {attempt_num})...")
    await asyncio.sleep(backoff)
