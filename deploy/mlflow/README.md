# AgroDiagnosis MLflow Platform

Docker Compose stack for the MLflow Tracking Server used by training and inference:
dedicated Backend Store (Postgres) and Artifact Store (local volume).

## Quick Start

### 1. Environment

Copy the example environment file:

```bash
cp deploy/mlflow/.env.example deploy/mlflow/.env
```

or

```bash
make mlflow-env
```

### 2. Docker Deployment

Build the thin MLflow image and start the stack:

```bash
docker compose -f deploy/mlflow/docker-compose.yml --env-file deploy/mlflow/.env up -d --build
```

or

```bash
make mlflow-up
```

### 3. Extra commands

Read [`Makefile`](Makefile) for additional commands.

### 4. MLflow Tracking UI

Tracking UI: http://localhost:5000

