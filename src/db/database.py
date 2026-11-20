"""PostgreSQL pool owned by one process."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

import asyncpg

from src.config import Settings

log = logging.getLogger(__name__)


class Database:
    """Open asyncpg pool."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    async def declare_table(
        self,
        initialize: Callable[[asyncpg.Pool], Awaitable[None]],
    ) -> None:
        """Run a schema function against this pool."""
        await initialize(self.pool)

    async def close(self) -> None:
        """Close the pool and its connections."""
        await self.pool.close()
        log.info("PostgreSQL connection pool closed.")


async def create_database(settings: Settings) -> Database:
    """Open a pool and return it as a Database.

    With min_size=0 the first connection waits until a query.
    """
    pool = await asyncpg.create_pool(
        settings.database_url,
        min_size=0,
        max_size=5,
        command_timeout=5,
    )
    log.info(
        "PostgreSQL connection pool established: %s",
        settings.database_url.rsplit("@", 1)[-1],
    )
    return Database(pool)
