# AgroDiagnosis Core Service

FastAPI microservice providing application health monitoring, liveness/readiness probes, and core API entrypoints.

Also watch:

- [AgroDiagnosis MLflow Platform](deploy/mlflow/README.md) for the MLflow Tracking Server used by training and
  inference;
- [AgroDiagnosis Training Service](training/README.md) for the training service used to train and register new models.

## Quick Start

### 1. Requirements & Installation

Install dependencies and copy the example environment file:

```bash
uv sync
cp .env.example .env
```

or

```bash
make setup
```

### 2. Docker Deployment

Spin up the service using Docker Compose:

```bash
docker compose up -d --build
```

or

```bash
make up
```

### 3. Extra commands

Read [`Makefile`](Makefile) for additional commands.

### 4. API Endpoints

| Method | Endpoint        | Description                                        |
|--------|-----------------|----------------------------------------------------|
| GET    | /               | Application greeting and basic metadata            |
| GET    | /healthz        | Quick liveness probe for orchestration systems     |
| GET    | /api/v1/version | Current version                                    |
| GET    | /api/v1/health  | Comprehensive readiness report across dependencies |
| POST   | /evaluate         | Evaluate a labelled ZIP dataset and log results to MLflow |
| POST   | /api/v1/tasks     | Accept an image into the inference queue (202 / 503) |
| GET    | /api/v1/tasks/{id}| Fetch a finished task result (404 while processing) |

Swagger UI is available at `/docs`.

For `POST /evaluate`, upload a ZIP archive in the multipart `file` field through
`/docs`. Supported image formats: JPG, JPEG and PNG. Labels are the directory names:

```text
Apple/Apple_scab/image1.jpg
Cherry/Powdery_mildew/image2.png
```

A single enclosing directory such as `test/` is also supported. Class names must
match `src/class_names.json`.

## Inference pipeline (backpressure lab)

```text
prepare_batch → batch_publisher ─┐
POST /api/v1/tasks ──────────────┼─► RabbitMQ (bounded) ─► inference-worker ─► Postgres
                                 │
soft limit (batch only)          hard limit (reject-publish → HTTP 503)
```

Code layout (pipeline vs HTTP vs worker):

| Folder | Role |
|---|---|
| `src/messaging/` | Queue definition, publish, soft/hard backpressure |
| `src/tasks/` | HTTP upload + `task_results` in Postgres |
| `src/inference_worker/` | FastStream consumer (compose `inference-worker`), thread pool, `/healthz` |
| `src/scripts/batch_publisher.py` | Batch producer CLI |
| `src/api/tasks.py` | Routes → `src/tasks/http.py` |

### Run locally

```bash
docker compose up -d rabbitmq postgres inference-worker
uv run uvicorn src.app:app --reload
```

Full stack in Docker (app + worker + RabbitMQ + Postgres):

```bash
docker compose up -d --build
```

Worker liveness while inference runs (proves the event loop is not blocked):

```bash
curl -s http://127.0.0.1:8001/healthz
```

Batch load (waits at soft watermark; does not steal the HTTP headroom):

```bash
uv run python task-data/scripts/prepare_batch.py training/data/plant-diseases-processed/val --name demo --limit 100
uv run python -m src.scripts.batch_publisher task-data/demo/tasks.jsonl
```

### Design notes for the defence

| Topic | Choice | Why |
|---|---|---|
| `prefetch_count` | `= WORKER_EXECUTOR_WORKERS` (default 1) | Never pull more unacked messages than the thread pool can run; larger prefetch would hide backpressure and inflate memory |
| Inference off the loop | `ThreadPoolExecutor` | `torch` / `time.sleep` must not block FastStream; `/healthz` on `:8001` stays responsive under load |
| Batch vs HTTP | soft limit for batch, hard for everyone | Overnight jobs wait below soft watermark; interactive `POST /tasks` may use the remaining band until hard `reject-publish` → **503 + Retry-After** |
| `Retry-After` | `ceil(INFERENCE_MIN_SECONDS)` | Approximate time for one busy worker to free a slot |
| Ack policy | manual: ack after DB write | Ack-before-save can lose results on crash; nack/requeue forever creates poison loops |
| Retries → DLQ | `TASK_MAX_DELIVERY_ATTEMPTS` then `reject(requeue=False)` | Bounded retries with header `x-delivery-attempt`, then dead-letter queue |
| Idempotency | `task_results.task_id` PK + `ON CONFLICT DO NOTHING` | Safe under at-least-once redelivery |
| Graceful stop | Compose `stop_grace_period: 60s` + SIGTERM | Worker finishes the in-flight message before exit |

### Experiment results (§7)

Re-run locally (RabbitMQ must be up; workers optional for the steady rows):

```bash
docker compose up -d rabbitmq
uv run python -m src.scripts.measure_backpressure --tasks 80 --soft 20 --hard 30 --steady --work-seconds 0.5
make test-integration   # Testcontainers: hard reject-publish on a real broker
```

Measured on 2026-10-22 (publisher process RSS; no inference-worker required for the first three rows; steady rows use slow stand-in consumers with the same soft/hard publish path as `batch_publisher`):

| Scenario | Published | Elapsed (s) | Max depth | Peak RSS (MiB) | Rate (tasks/s) |
|---|---:|---:|---:|---:|---:|
| unbounded (no limit) | 80 | 0.47 | 80 | 49.6 | 171.71 |
| hard reject-publish (max=30) | 30 | 0.17 | 30 | 49.6 | 173.98 |
| soft backoff (soft=20, hard=30) | 20 | 1.23 | 20 | 49.6 | 16.25 |
| steady soft=20 consumers=1 work=0.5s | 40 | 9.75 | 20 | 49.8 | 4.10 |
| steady soft=20 consumers=2 work=0.5s | 40 | 4.72 | 20 | 49.9 | 8.47 |

**Conclusions**

- Without a bound, depth grows with the burst (80/80) while publisher RSS stays flat — the source streams JSONL and does not buffer the batch in RAM.
- With `x-max-length` + `reject-publish`, depth never exceeds the hard limit; further publishes fail immediately (HTTP maps this to **503 + Retry-After**).
- Soft watermark keeps batch depth ≤ soft even when hard headroom remains; interactive `POST /api/v1/tasks` can still use the band between soft and hard.
- In steady state, source rate tracks consumer capacity: doubling stand-in consumers roughly doubled publish rate (4.10 → 8.47 tasks/s) with **no publisher code change**, and max depth stayed at the soft limit.
- Soft signal = AMQP passive declare (not Management API); hard signal = publisher confirm reject. Management UI (`15672`) is for observation only.
- The same coupling holds for real `inference-worker` when `INFERENCE_MIN_SECONDS` sets per-task duration.

### Defence demo checklist

1. Publish a large batch; watch queue depth in RabbitMQ Management stay ≤ soft for batch, ≤ hard overall.
2. Flood `POST /api/v1/tasks` until **503** with `Retry-After`; confirm no pending rows for rejected uploads.
3. Start a second worker (separate health port):  
   `docker compose run -d --name inference-worker-2 -e WORKER_HEALTH_PORT=8002 -p 127.0.0.1:8002:8002 inference-worker`
4. `curl` worker `/healthz` during a slow inference (`INFERENCE_MIN_SECONDS=5`).
5. `docker compose kill -s SIGTERM` a busy worker; in-flight task should complete or be redelivered, not silently drop.
