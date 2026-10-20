# Batch Pipeline Data

This directory serves as the shared storage (`/data`) for pipeline containers. All generated data files within this directory are excluded from version control.

## Generating a Batch

You can automatically generate a batch of tasks from any existing directory of images:

```bash
uv run python -m src.prepare_batch training/data/plant-diseases-processed/val --name demo --limit 100

```

**Options:**

* `--limit 100`: Copies up to 100 images. Omit to include all supported images.
* `--name demo`: Specifies the output directory name. Omit to auto-generate a unique UUID-based name.

*Example `tasks.jsonl` entry:*

```json
{"task_id": "edb72890-2d93-4f89-83b4-269aa9346214", "image_path": "demo/images/edb72890-2d93-4f89-83b4-269aa9346214.jpg"}

```

All paths in the manifest are relative to this root directory and strictly use forward slashes (`/`).
