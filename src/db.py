"""PostgreSQL connection pool management using asyncpg."""

import logging

import asyncpg

from src.config import Settings

log = logging.getLogger(__name__)


async def create_db_pool(settings: Settings) -> asyncpg.Pool:
    """Creates a PostgreSQL connection pool.

    Note:
        `min_size=0` ensures that no connections are opened upon initialization.
        This allows the application to start even if the database is unavailable,
        letting the readiness probe report the degradation gracefully.

    Args:
        settings: Application settings instance containing the database URL.

    Returns:
        asyncpg.Pool: Configured asynchronous database connection pool.
    """
    pool = await asyncpg.create_pool(
        settings.database_url,
        min_size=0,
        max_size=5,
        command_timeout=5,
    )

    log.info("PostgreSQL connection pool established: %s", settings.database_url.rsplit("@", 1)[-1])
    return pool


async def close_db_pool(pool: asyncpg.Pool) -> None:
    """Closes the connection pool and terminates all active connections.

    Args:
        pool: Active asyncpg connection pool instance to be closed.
    """
    await pool.close()

    log.info("PostgreSQL connection pool closed.")
