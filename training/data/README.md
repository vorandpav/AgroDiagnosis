# Data

Local copy of the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).
This directory is gitignored except for this README.

## Raw layout (Kaggle)

After `uv run python scripts/download_data.py` (from `training/`):

## Processed layout (training)

```bash
uv run python scripts/prepare_dataset.py
```

Builds `data/plant-diseases-processed/`:

```text
train/<Plant>/<Condition>/*.jpg   # from Kaggle train/
val/<Plant>/<Condition>/*.jpg     # from Kaggle valid/
test/<Plant>/<Condition>/*.jpg    # probe, names parsed via regex + prefix map
```
