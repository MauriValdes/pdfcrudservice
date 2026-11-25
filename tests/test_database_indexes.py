"""Pruebas unitarias del contrato de inicialización de índices requeridos."""

from unittest.mock import AsyncMock, MagicMock, call

import pytest

from src.database.indexes import ensure_required_indexes


EXPECTED_INDEX_CALLS = [
    call(
        [("checksum", 1)],
        name="checksum_unique",
        unique=True,
    ),
    call(
        [("status", 1), ("next_retry_at", 1), ("created_at", 1)],
        name="pending_jobs_idx",
    ),
    call(
        [("status", 1), ("lease_until", 1)],
        name="expired_leases_idx",
    ),
]


@pytest.mark.asyncio
async def test_solicita_los_tres_indices_en_la_coleccion_recibida():
    collection = MagicMock()
    collection.create_index = AsyncMock()

    await ensure_required_indexes(collection)

    # La igualdad exacta fija orden, nombres, claves y opciones: el checksum
    # es único; los índices de procesamiento omiten unique (false por defecto)
    # y no agregan TTL, sparse ni filtros parciales.
    assert collection.create_index.await_args_list == EXPECTED_INDEX_CALLS


@pytest.mark.asyncio
async def test_segunda_inicializacion_repite_las_mismas_definiciones():
    """Los mocks verifican estabilidad de solicitudes, no idempotencia real."""
    collection = MagicMock()
    collection.create_index = AsyncMock()

    await ensure_required_indexes(collection)
    await ensure_required_indexes(collection)

    assert collection.create_index.await_args_list == (
        EXPECTED_INDEX_CALLS + EXPECTED_INDEX_CALLS
    )


@pytest.mark.parametrize("failing_call", [1, 2, 3])
@pytest.mark.asyncio
async def test_propaga_fallos_de_cualquiera_de_las_creaciones_sin_borrar(
    failing_call,
):
    error = RuntimeError(f"fallo en creación {failing_call}")
    collection = MagicMock()
    collection.create_index = AsyncMock(
        side_effect=[None] * (failing_call - 1) + [error]
    )

    with pytest.raises(RuntimeError, match=f"fallo en creación {failing_call}"):
        await ensure_required_indexes(collection)

    assert collection.create_index.await_count == failing_call
    collection.drop_index.assert_not_called()
    collection.drop_indexes.assert_not_called()
    collection.drop.assert_not_called()
    collection.delete_many.assert_not_called()
    collection.delete_one.assert_not_called()
