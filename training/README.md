# AgroDiagnosis Training

EDA, training runs, and MLflow experiment workflow for plant disease classification.

## Quick Start

### 1. Requirements & Installation

From this directory:

```bash
uv sync
```

### 2. Start MLflow

Tracking Server must be up before EDA or training - see [`deploy/mlflow/README.md`](../deploy/mlflow/README.md).

### 3. Download Dataset

Download the New Plant Diseases Dataset - see [`data/README.md`](data/README.md).

### 4. EDA

Open `notebooks/eda.ipynb`, run all cells.
Artifacts and dataset lineage go to experiment `AgroDiagnosisEDA`.
