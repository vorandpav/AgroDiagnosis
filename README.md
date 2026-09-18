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

### 4. API Endpoints

| Method | Endpoint        | Description                                        |
|--------|-----------------|----------------------------------------------------|
| GET    | /               | Application greeting and basic metadata            |
| GET    | /healthz        | Quick liveness probe for orchestration systems     |
| GET    | /api/v1/version | Current version                                    |
| GET    | /api/v1/health  | Comprehensive readiness report across dependencies |

Swagger UI is available at `/docs`.
