"""Postgres storage for finished inference tasks."""

import asyncpg

from src.schemas import TaskResult


async def save_result(result: TaskResult, pool: asyncpg.Pool) -> None:
    """Atomically save the first result using the unique task_id."""
    await pool.execute(
        """
        INSERT INTO task_results (task_id, payload)
        VALUES ($1, $2::jsonb)
        ON CONFLICT (task_id) DO NOTHING
        """,
        result.task_id,
        result.model_dump_json(),
    )


async def initialize_results(pool: asyncpg.Pool) -> None:
    """Create the results table for the initial pipeline."""
    await pool.execute(
        """
        CREATE TABLE IF NOT EXISTS task_results (
            task_id uuid PRIMARY KEY,
            payload jsonb NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
