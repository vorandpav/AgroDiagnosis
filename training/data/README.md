# Data

Local copy of the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).
This directory is gitignored except for this README.

## Layout

After `uv run python scripts/download_data.py` (from `training/`):

```text
data/plant-diseases/
└── .../train/<class>/*.jpg
└── .../valid/<class>/*.jpg
```

Exact nesting depends on the Kaggle archive; open the tree once and point EDA / `train.py` at the real `train` and `valid` folders.

## Manual download

If `kagglehub` fails: download the ZIP from Kaggle, unpack into `plant-diseases/`, keep class subfolders under `train/` and `valid/`.
