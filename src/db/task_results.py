"""Postgres repository for finished inference tasks."""

import asyncio
from uuid import UUID

import asyncpg

from src.schemas import InferenceResult, TaskResult


def _task_result(row: asyncpg.Record) -> TaskResult:
    """Build a task result from one task_results row."""
    return TaskResult(
        task_id=row["task_id"],
        prediction=InferenceResult(
            label=row["label"],
            confidence=row["confidence"],
        ),
        model_ref=row["model_ref"],
        model_version=row["model_version"],
        processing_seconds=row["processing_seconds"],
    )


async def save_result(result: TaskResult, pool: asyncpg.Pool) -> None:
    """Atomically save the first result using the unique task_id."""
    await pool.execute(
        """
        INSERT INTO task_results (
            task_id, label, confidence, model_ref, model_version, processing_seconds
        )
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (task_id) DO NOTHING
        """,
        result.task_id,
        result.prediction.label,
        result.prediction.confidence,
        result.model_ref,
        result.model_version,
        result.processing_seconds,
    )


async def initialize_results(pool: asyncpg.Pool) -> None:
    """Create the results table for the initial pipeline."""
    await pool.execute(
        """
        CREATE TABLE IF NOT EXISTS task_results (
            task_id uuid PRIMARY KEY,
            label text NOT NULL,
            confidence double precision NOT NULL,
            model_ref text NOT NULL,
            model_version text NOT NULL,
            processing_seconds double precision NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )


async def fetch_task_result(task_id: UUID, pool: asyncpg.Pool) -> TaskResult | None:
    """Load a persisted task result, if the worker has written one."""
    row = await pool.fetchrow(
        """
        SELECT task_id, label, confidence, model_ref, model_version, processing_seconds
        FROM task_results
        WHERE task_id = $1
        """,
        task_id,
    )
    if row is None:
        return None
    return _task_result(row)


async def fetch_task_results(
    pool: asyncpg.Pool,
    task_ids: list[UUID],
) -> dict[UUID, TaskResult]:
    """Load persisted results for the given task identifiers."""
    if not task_ids:
        return {}
    rows = await pool.fetch(
        """
        SELECT task_id, label, confidence, model_ref, model_version, processing_seconds
        FROM task_results
        WHERE task_id = ANY($1::uuid[])
        """,
        task_ids,
    )
    return {row["task_id"]: _task_result(row) for row in rows}


async def _count_task_results(pool: asyncpg.Pool, task_ids: list[UUID]) -> int:
    """Return how many of the given tasks already have a row."""
    if not task_ids:
        return 0
    count = await pool.fetchval(
        "SELECT count(*) FROM task_results WHERE task_id = ANY($1::uuid[])",
        task_ids,
    )
    return int(count)


async def wait_for_task_results(
    pool: asyncpg.Pool,
    task_ids: list[UUID],
    *,
    poll_interval_seconds: float = 0.05,
    timeout_seconds: float = 600.0,
    deadline: float | None = None,
) -> dict[UUID, TaskResult]:
    """Wait until every task_id has a row, then load them once."""
    loop = asyncio.get_running_loop()
    if deadline is None:
        deadline = loop.time() + timeout_seconds
    expected = len(set(task_ids))
    while loop.time() < deadline:
        if await _count_task_results(pool, task_ids) == expected:
            break
        remaining = deadline - loop.time()
        if remaining <= 0:
            break
        await asyncio.sleep(min(poll_interval_seconds, remaining))
    if await _count_task_results(pool, task_ids) != expected:
        raise TimeoutError(f"Evaluation timed out after {timeout_seconds:g} seconds")
    return await fetch_task_results(pool, task_ids)
