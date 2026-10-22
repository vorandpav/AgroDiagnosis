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
