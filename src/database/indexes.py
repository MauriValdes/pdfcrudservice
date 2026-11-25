"""Inicialización común de índices MongoDB."""

from pymongo import ASCENDING


async def ensure_required_indexes(collection) -> None:
    """Garantiza los índices requeridos sobre la colección recibida."""
    await collection.create_index(
        [("checksum", ASCENDING)],
        name="checksum_unique",
        unique=True,
    )
    await collection.create_index(
        [
            ("status", ASCENDING),
            ("next_retry_at", ASCENDING),
            ("created_at", ASCENDING),
        ],
        name="pending_jobs_idx",
    )
    await collection.create_index(
        [
            ("status", ASCENDING),
            ("lease_until", ASCENDING),
        ],
        name="expired_leases_idx",
    )
