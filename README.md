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
| POST   | /process        | Plant-disease inference (multipart image `file`)   |

Swagger UI is available at `/docs`.
