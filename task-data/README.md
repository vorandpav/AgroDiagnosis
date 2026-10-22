# Batch Pipeline Data

This directory serves as the shared storage (`/data`) for pipeline containers. All generated data files within this directory are excluded from version control.

## Generating a Batch

You can automatically generate a batch of tasks from any existing directory of images:

```bash
uv run python task-data/scripts/prepare_batch.py training/data/plant-diseases-processed/val --name demo --limit 100
```

**Options:**

* `--limit 100`: Copies up to 100 images. Omit to include all supported images.
* `--name demo`: Specifies the output directory name. Omit to auto-generate a unique UUID-based name.

*Example `tasks.jsonl` entry (fixture format, not the queue message):*

```json
{"id": "edb72890-2d93-4f89-83b4-269aa9346214", "path": "demo/images/edb72890-2d93-4f89-83b4-269aa9346214.jpg"}
```

All paths in the manifest are relative to this root directory and strictly use forward slashes (`/`).

## Publishing a Batch

Run the publisher locally against RabbitMQ (manual CLI, not a Compose service). It adapts fixture lines into `InferenceTask` messages:

```bash
uv run python -m src.scripts.batch_publisher task-data/demo/tasks.jsonl
```

The long-running consumer is `inference-worker`:

```bash
docker compose up -d inference-worker
# or locally:
uv run python -m src.inference_worker
```
