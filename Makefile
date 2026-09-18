.PHONY: install lint fmt test run up down logs psq

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
