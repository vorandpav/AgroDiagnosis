# AgroDiagnosis Training

EDA, Hydra-driven training runs, and MLflow experiment workflow for plant disease classification.
Separate from the FastAPI product service: own `pyproject.toml`, lockfile, and venv.

## Quick Start

### 1. Requirements & Installation

Ensure `uv` is installed. From this directory pick **one** PyTorch build (they conflict):

```bash
# laptop / CI / no NVIDIA
uv sync --extra cpu

# Windows/Linux + NVIDIA (CUDA 12.4 wheels)
uv sync --extra cu124

cp .env.example .env
```

From repo root: `make train-sync` (default `TORCH_EXTRA=cu124`, override with `TORCH_EXTRA=cpu`).

### 2. Start MLflow

Tracking Server must be up before EDA or training - see [`deploy/mlflow/README.md`](../deploy/mlflow/README.md)
or `make mlflow-up` from the repo root.

### 3. Prepare Dataset

See [`data/README.md`](data/README.md) for instructions to download and prepare the dataset.
Or from repo root: `make data-prepare`.

### 4. EDA

See [`notebooks/eda.ipynb`](notebooks/eda.ipynb) for exploratory data analysis.

### 5. Train (Hydra)

One composed job (defaults from `conf/config.yaml`: `data=200per_class`, `model=baseline`):

```bash
uv run --extra cu124 python -u train.py
# or: make train
```

Override a single run:

```bash
uv run --extra cu124 python -u train.py data=800per_class model=label_smooth
uv run --extra cu124 python -u train.py training.num_workers=0   # if Windows workers hang
```

Full 3×2 grid via Hydra multirun (`-m`):

```bash
uv run --extra cu124 python -u train.py -m \
  data=200per_class,800per_class \
  model=baseline,dropout,label_smooth
# or: make train-grid
```

Each job logs params/metrics/artifacts to MLflow. Champion is **not** set here.

### 6. Promote @champion

After the grid (or any set of runs), promote the best **val** `f1_macro`:

```bash
uv run --extra cu124 python scripts/promote_champion.py
# or: make promote
```

Uses the run tag `model_uri` written by `train.py`. Test probe metrics stay on the run
(`test_f1_macro`, `confusion_matrix_test.png`).

Configs live under `conf/` (`data/`, `model/`). Machine-local paths stay in `.env`.
