# AgroDiagnosis Core Service

FastAPI microservice providing application health monitoring, liveness/readiness probes, and core API entrypoints.

## Quick Start

### 1. Requirements & Installation

Ensure `uv` is installed on your system.

```bash
uv sync
cp .env.example .env
```

### 2. Local Execution

Run the development server with hot-reload:

```bash
make run
```

### 3. Docker Deployment

Spin up the service using Docker Compose:

```bash
make up
```

Useful Make targets:

```bash
make mlflow-up          # Tracking Server (deploy/mlflow)
make train-sync         # uv sync training extras (TORCH_EXTRA=cu124|cpu)
make train              # one Hydra job
make train-grid         # Hydra -m 3×2 grid
make promote            # best f1_macro -> Model Registry @champion
make process IMAGE=...  # smoke POST /process
```

### 4. API Endpoints

| Method | Endpoint        | Description                                        |
|--------|-----------------|----------------------------------------------------|
| GET    | /               | Application greeting and basic metadata            |
| GET    | /healthz        | Quick liveness probe for orchestration systems     |
| GET    | /api/v1/version | Current version                                    |
| GET    | /api/v1/health  | Comprehensive readiness report across dependencies |
| POST   | /process        | Plant-disease inference (multipart image `file`)   |

Swagger UI is available at `/docs`.
