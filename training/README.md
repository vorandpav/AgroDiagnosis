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

### 2. Start MLflow

Tracking Server must be up before EDA or training - see [`deploy/mlflow/README.md`](../deploy/mlflow/README.md).

### 3. Prepare Dataset

See [`data/README.md`](data/README.md) for instructions to download and prepare the dataset.

### 4. EDA

See [`notebooks/eda.ipynb`](notebooks/eda.ipynb) for exploratory data analysis.

### 5. Train & Register (Hydra)

```bash
uv run --extra cu124 python -u train.py   # or --extra cpu on laptops without NVIDIA
```

Reads `DATA_DIR` (processed tree). **3×2 grid**: MobileNetV3-Small
`baseline` / `dropout` / `label_smooth` × `200per_class` / `800per_class`
(train/val caps per class). Champion by **val** `f1_macro`; test probe metrics
logged separately (`test_f1_macro`, `confusion_matrix_test.png`).

Useful overrides:

```bash
uv run --extra cu124 python -u train.py data_to_run=[200per_class] models_to_run=[baseline]
uv run --extra cu124 python -u train.py training.num_workers=0   # if Windows workers hang
uv run --extra cu124 python train.py --cfg job
uv run --extra cu124 python scripts/reset_mlflow.py   # wipe training experiment + registry model
```

Configs live under `conf/` (`data/`, `model/`). Machine-local paths stay in `.env`.

