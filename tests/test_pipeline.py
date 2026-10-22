"""Test the initial pipeline without external services."""

import asyncio
import json
import threading
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from faststream.rabbit import TestRabbitBroker
from pydantic import ValidationError

from src.config import Settings
from src.inference_worker.consumer import (
    recover_failed_delivery,
    register_handler,
    run_inference_worker,
)
from src.inference_worker.processing import process_task
from src.messaging.broker import create_broker, task_queue
from src.schemas import InferenceResult, InferenceTask, TaskResult
from src.scripts.batch_publisher import (
    publish_manifest,
    read_tasks,
    to_inference_task,
)
from src.services.inference import load_model
from src.tasks.results import save_result


def _manifest_line(task: InferenceTask) -> str:
    """Serialize a task the way prepare_batch writes fixture records."""
    return json.dumps({"id": str(task.task_id), "path": task.image_path})


@pytest.fixture(autouse=True)
def _queue_below_soft_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid passive declare against TestRabbitBroker during unit tests."""
    monkeypatch.setattr("src.messaging.publishing.queue_depth", AsyncMock(return_value=0))


@pytest.fixture(autouse=True)
def _fake_worker_health(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not bind a real health port while unit-testing the worker."""
    server = Mock()
    server.serve = AsyncMock()
    server.should_exit = False
    monkeypatch.setattr(
        "src.inference_worker.consumer.create_health_server",
        lambda *_args, **_kwargs: server,
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(
        rabbitmq_url="amqp://guest:guest@127.0.0.1:59998/",
        inference_min_seconds=0,
        task_retry_delay_seconds=0.01,
    )


@pytest.fixture
def task() -> InferenceTask:
    return InferenceTask(task_id=uuid4(), image_path="leaf.jpg")


@pytest.fixture
def result(task: InferenceTask) -> TaskResult:
    return TaskResult(
        task_id=task.task_id,
        prediction=InferenceResult(label="healthy", confidence=0.9, model="plant@champion"),
        model_version="7",
        processing_seconds=0.1,
    )


@pytest.mark.parametrize("path", ["../leaf.jpg", "/leaf.jpg", "C:/leaf.jpg", "a\\b", "."])
def test_task_rejects_paths_outside_portable_storage(path: str) -> None:
    with pytest.raises(ValidationError):
        InferenceTask(task_id=uuid4(), image_path=path)


def test_task_round_trips_through_json(task: InferenceTask) -> None:
    assert InferenceTask.model_validate_json(task.model_dump_json()) == task


def test_batch_publisher_adapts_prepare_batch_record(task: InferenceTask) -> None:
    adapted = to_inference_task({"id": str(task.task_id), "path": task.image_path})
    assert adapted == task


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_settings_reject_invalid_inference_duration(value: float) -> None:
    with pytest.raises(ValidationError):
        Settings(inference_min_seconds=value)


def test_manifest_yields_first_task_before_reading_invalid_next_line(
    tmp_path: Path, task: InferenceTask
) -> None:
    manifest = tmp_path / "tasks.jsonl"
    manifest.write_text(_manifest_line(task) + "\nnot json", encoding="utf-8")
    tasks = read_tasks(manifest)
    assert next(tasks) == task
    with pytest.raises(json.JSONDecodeError):
        next(tasks)


def test_manifest_rejects_oversized_line(tmp_path: Path) -> None:
    manifest = tmp_path / "tasks.jsonl"
    manifest.write_text("x" * 4097, encoding="utf-8")
    with pytest.raises(ValueError, match="4096"):
        next(read_tasks(manifest))


async def test_batch_publisher_delivers_manifest_to_worker_and_saves_result(
    tmp_path: Path, settings: Settings, task: InferenceTask, result: TaskResult
) -> None:
    manifest = tmp_path / "tasks.jsonl"
    manifest.write_text(_manifest_line(task) + "\n", encoding="utf-8")
    execute = AsyncMock(return_value=result)
    save = AsyncMock()
    broker = create_broker(settings)
    register_handler(broker, settings, execute, save)
    async with TestRabbitBroker(broker):
        await publish_manifest(manifest, broker, settings)
    execute.assert_awaited_once_with(task)
    save.assert_awaited_once_with(result)


async def test_handler_waits_for_inference_before_saving(
    settings: Settings, task: InferenceTask, result: TaskResult
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def execute(message: InferenceTask) -> TaskResult:
        assert message == task
        started.set()
        await release.wait()
        return result

    save = AsyncMock()
    broker = create_broker(settings)
    register_handler(broker, settings, execute, save)
    async with TestRabbitBroker(broker):
        publication = asyncio.create_task(broker.publish(task, queue=task_queue(settings)))
        await asyncio.wait_for(started.wait(), timeout=2)
        save.assert_not_awaited()
        release.set()
        await publication
    save.assert_awaited_once_with(result)


async def test_handler_does_not_save_when_inference_fails(
    settings: Settings, task: InferenceTask
) -> None:
    settings.task_max_delivery_attempts = 1
    execute = AsyncMock(side_effect=ValueError("invalid image"))
    save = AsyncMock()
    broker = create_broker(settings)
    register_handler(broker, settings, execute, save)
    async with TestRabbitBroker(broker):
        await broker.publish(task, queue=task_queue(settings))
    save.assert_not_awaited()


async def test_batch_publisher_stops_at_failed_publication_without_reading_next_task(
    tmp_path: Path, settings: Settings, task: InferenceTask
) -> None:
    manifest = tmp_path / "tasks.jsonl"
    manifest.write_text(_manifest_line(task) + "\ninvalid", encoding="utf-8")
    broker = Mock()
    broker.publish = AsyncMock(side_effect=TimeoutError("confirm timeout"))
    with pytest.raises(TimeoutError, match="confirm timeout"):
        await publish_manifest(manifest, broker, settings)
    assert broker.publish.await_count == 1
    assert broker.publish.call_args.kwargs["persist"] is True
    assert broker.publish.call_args.kwargs["message_id"] == str(task.task_id)


@pytest.mark.parametrize(
    ("inference_seconds", "expected_delay", "total_seconds"), [(0.5, 1.5, 2), (3, 0, 3)]
)
def test_processing_uses_model_and_adds_only_remaining_delay(
    tmp_path: Path,
    settings: Settings,
    task: InferenceTask,
    result: TaskResult,
    monkeypatch: pytest.MonkeyPatch,
    inference_seconds: float,
    expected_delay: float,
    total_seconds: float,
) -> None:
    settings.task_storage_dir = tmp_path
    settings.inference_min_seconds = 2
    (tmp_path / "leaf.jpg").write_bytes(b"image")
    model = Mock()
    predict = Mock(return_value=result.prediction)
    sleep = Mock()
    monkeypatch.setattr("src.inference_worker.processing.predict", predict)
    monkeypatch.setattr("src.inference_worker.processing.time.sleep", sleep)
    monkeypatch.setattr(
        "src.inference_worker.processing.time.perf_counter",
        Mock(side_effect=[0, inference_seconds, total_seconds]),
    )
    actual = process_task(task, settings, model, ["healthy"], "7")
    predict.assert_called_once_with(
        model, ["healthy"], b"image", f"{settings.mlflow_model_name}@{settings.mlflow_model_alias}"
    )
    sleep.assert_called_once_with(expected_delay)
    assert actual.processing_seconds == total_seconds
    assert actual.model_version == "7"


async def test_save_result_uses_atomic_unique_key_insert(result: TaskResult) -> None:
    pool = Mock()
    pool.execute = AsyncMock()
    await save_result(result, pool)
    query, task_id, payload = pool.execute.call_args.args
    assert "ON CONFLICT (task_id) DO NOTHING" in query
    assert task_id == result.task_id
    assert TaskResult.model_validate_json(payload) == result


async def test_worker_loads_model_once_before_consuming_and_closes_pool(
    settings: Settings, task: InferenceTask, result: TaskResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = Mock()
    load = Mock(return_value=(model, ["healthy"], "7"))
    pool = Mock()
    pool.execute = AsyncMock()
    close = AsyncMock()
    monkeypatch.setattr("src.inference_worker.consumer.load_model", load)
    monkeypatch.setattr(
        "src.inference_worker.consumer.create_db_pool", AsyncMock(return_value=pool)
    )
    monkeypatch.setattr("src.inference_worker.consumer.close_db_pool", close)
    main_thread = threading.get_ident()

    def predict_task(
        message: InferenceTask,
        config: Settings,
        loaded_model: object,
        labels: list[str],
        version: str,
    ) -> TaskResult:
        assert threading.get_ident() != main_thread
        assert loaded_model is model
        assert message == task
        assert version == "7"
        return result

    monkeypatch.setattr("src.inference_worker.consumer.process_task", predict_task)

    async def run(app: object) -> None:
        load.assert_called_once_with(settings)
        assert pool.execute.await_count == 1
        async with TestRabbitBroker(app.broker):
            await app.broker.publish(task, queue=task_queue(settings))
            await app.broker.publish(task, queue=task_queue(settings))

    monkeypatch.setattr("src.inference_worker.consumer.FastStream.run", run)
    await run_inference_worker(settings)
    load.assert_called_once_with(settings)
    assert pool.execute.await_count == 3
    close.assert_awaited_once_with(pool)


async def test_handler_acknowledges_only_after_result_is_saved(
    settings: Settings,
    task: InferenceTask,
    result: TaskResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ack = AsyncMock()
    monkeypatch.setattr("faststream.rabbit.message.RabbitMessage.ack", ack)

    async def save(value: TaskResult) -> None:
        assert value == result
        ack.assert_not_awaited()

    broker = create_broker(settings)
    register_handler(broker, settings, AsyncMock(return_value=result), save)
    async with TestRabbitBroker(broker):
        await broker.publish(task, queue=task_queue(settings))
    ack.assert_awaited_once()


async def test_handler_rejects_to_dlq_when_attempts_exhausted(
    settings: Settings,
    task: InferenceTask,
    result: TaskResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings.task_max_delivery_attempts = 1
    ack = AsyncMock()
    reject = AsyncMock()
    monkeypatch.setattr("faststream.rabbit.message.RabbitMessage.ack", ack)
    monkeypatch.setattr("faststream.rabbit.message.RabbitMessage.reject", reject)
    broker = create_broker(settings)
    register_handler(
        broker,
        settings,
        AsyncMock(return_value=result),
        AsyncMock(side_effect=ConnectionError("database unavailable")),
    )
    async with TestRabbitBroker(broker):
        await broker.publish(task, queue=task_queue(settings))
    ack.assert_not_awaited()
    reject.assert_awaited_once_with(requeue=False)


async def test_recover_failed_delivery_republishes_with_incremented_attempt(
    settings: Settings, task: InferenceTask
) -> None:
    settings.task_max_delivery_attempts = 3
    broker = Mock()
    broker.publish = AsyncMock()
    message = Mock()
    message.ack = AsyncMock()
    message.reject = AsyncMock()
    await recover_failed_delivery(broker, settings, task, message, attempt=0)
    message.reject.assert_not_awaited()
    message.ack.assert_awaited_once()
    broker.publish.assert_awaited_once()
    assert broker.publish.await_args.kwargs["headers"]["x-delivery-attempt"] == 1


async def test_recover_failed_delivery_rejects_to_dlq_when_exhausted(
    settings: Settings, task: InferenceTask
) -> None:
    settings.task_max_delivery_attempts = 3
    broker = Mock()
    broker.publish = AsyncMock()
    message = Mock()
    message.ack = AsyncMock()
    message.reject = AsyncMock()
    await recover_failed_delivery(broker, settings, task, message, attempt=2)
    broker.publish.assert_not_awaited()
    message.ack.assert_not_awaited()
    message.reject.assert_awaited_once_with(requeue=False)


async def test_worker_does_not_start_when_model_load_fails(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "src.inference_worker.consumer.load_model",
        Mock(side_effect=RuntimeError("model unavailable")),
    )
    run = AsyncMock()
    monkeypatch.setattr("src.inference_worker.consumer.FastStream.run", run)
    with pytest.raises(RuntimeError, match="model unavailable"):
        await run_inference_worker(settings)
    run.assert_not_awaited()


def test_model_loader_resolves_configured_alias_before_loading_version(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.mlflow_model_name = "test-plant-model"
    settings.mlflow_model_alias = "candidate"
    client = Mock()
    client.get_model_version_by_alias.return_value = Mock(version="42")
    model = Mock()
    load = Mock(return_value=model)
    monkeypatch.setattr("src.services.inference.probe_mlflow_tracking", Mock())
    monkeypatch.setattr("src.services.inference.mlflow.set_tracking_uri", Mock())
    monkeypatch.setattr("src.services.inference.MlflowClient", Mock(return_value=client))
    monkeypatch.setattr("src.services.inference.mlflow.pytorch.load_model", load)
    loaded_model, labels, version = load_model(settings)
    client.get_model_version_by_alias.assert_called_once_with("test-plant-model", "candidate")
    load.assert_called_once_with("models:/test-plant-model/42")
    model.eval.assert_called_once_with()
    assert loaded_model is model
    assert version == "42"
    assert labels
