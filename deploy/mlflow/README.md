# AgroDiagnosis MLflow Platform

Docker Compose stack for the MLflow Tracking Server used by training and inference:
dedicated Backend Store (Postgres) and Artifact Store (local volume).

## Quick Start

### 1. Environment

From the repository root:

```bash
cp deploy/mlflow/.env.example deploy/mlflow/.env
```

### 2. Docker Deployment

Build the thin MLflow image and start the stack:

```bash
docker compose -f deploy/mlflow/docker-compose.yml --env-file deploy/mlflow/.env up -d --build
```

### 3. MLflow Tracking UI

Tracking UI: http://localhost:5000

