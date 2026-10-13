.PHONY: install lint fmt test run up down logs psq \
	mlflow-up mlflow-down mlflow-logs \
	train-sync data-prepare train train-grid promote process

TORCH_EXTRA ?= cu124
MLFLOW_COMPOSE := docker compose -f deploy/mlflow/docker-compose.yml --env-file deploy/mlflow/.env
IMAGE ?= training/data/plant-diseases-processed/test/Potato/Early_blight/PotatoEarlyBlight1.JPG

# Install dependencies from lockfile
install:
	uv sync

# Run code linters and formatting checks
lint:
	uv run ruff check .
	uv run ruff format --check .

# Auto-fix linting errors and format code
fmt:
	uv run ruff check . --fix
	uv run ruff format .

# Run test suite with coverage report
test:
	uv run pytest --cov=src --cov-report=term-missing tests/

# Run local development server
run:
	uv run uvicorn src.app:app --reload --port 8000

# Spin up Docker containers (app + postgres)
up:
	docker compose up -d --build

# Stop and remove Docker containers
down:
	docker compose down

# Stream application container logs
logs:
	docker compose logs -f app

# Open psql terminal inside running Postgres container
psq:
	docker compose exec postgres psql -U postgres -d agrodiagnosis

# MLflow Tracking stack (separate compose)
mlflow-up:
	$(MLFLOW_COMPOSE) up -d --build

mlflow-down:
	$(MLFLOW_COMPOSE) down

mlflow-logs:
	$(MLFLOW_COMPOSE) logs -f mlflow

# Training package (override: make train TORCH_EXTRA=cpu)
train-sync:
	cd training && uv sync --extra $(TORCH_EXTRA)

data-prepare:
	cd training && uv run --extra $(TORCH_EXTRA) python scripts/download_data.py && uv run --extra $(TORCH_EXTRA) python scripts/prepare_dataset.py

train:
	cd training && uv run --extra $(TORCH_EXTRA) python -u train.py

train-grid:
	cd training && uv run --extra $(TORCH_EXTRA) python -u train.py -m \
		data=200per_class,800per_class \
		model=baseline,dropout,label_smooth

promote:
	cd training && uv run --extra $(TORCH_EXTRA) python scripts/promote_champion.py

# Smoke POST /process (override: make process IMAGE=path/to.jpg)
process:
	curl -sS -F "file=@$(IMAGE)" http://localhost:8000/process
