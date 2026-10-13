"""Reproducible RNG setup for training runs."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, Torch and force deterministic CuDNN algorithms."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Make hash-based set/dict iteration stable across processes.
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker from Torch's per-worker initial seed."""
    _ = worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
