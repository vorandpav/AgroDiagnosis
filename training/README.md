# AgroDiagnosis Training

EDA, Hydra-driven training, and MLflow experiment workflow for plant-disease
classification. Separate package: own `pyproject.toml`, lockfile, and venv.

## Quick Start

### 1. Requirements & Installation

Install dependencies and copy the example environment file:

```bash
cd training
uv sync --extra cu124   # or: --extra cpu
cp .env.example .env
```

or

```bash
make train-sync TORCH_EXTRA=cu124   # or TORCH_EXTRA=cpu
make train-env
```

### 2. Start MLflow

Tracking Server must be up before EDA or training — see
[`deploy/mlflow/README.md`](../deploy/mlflow/README.md).

### 3. Prepare Dataset

Download the Kaggle archive and build the processed tree
(`data/plant-diseases-processed/`). Details:
[`data/README.md`](data/README.md).

```bash
make data-prepare
```

### 4. Train

One Hydra job (defaults: `data=200per_class`, `model=baseline`):

```bash
cd training && uv run --extra cu124 python -u train.py
```

or

```bash
make train
```

Full 3×2 grid:

```bash
make train-grid
```

### 5. Promote @champion

Promote the best validation `f1_macro` to the Model Registry:

```bash
make promote
```

### 6. Extra commands

Read [`Makefile`](../Makefile) for additional commands (`TORCH_EXTRA`,
Hydra overrides, EDA notebook under [`notebooks/eda.ipynb`](notebooks/eda.ipynb)).
