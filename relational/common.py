"""Small shared helpers: seeding, LR schedule, checkpoints, logging, time budget."""
import json
import math
import os
import random
import time

import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_at(step: int, base_lr: float, warmup: int, total: int, min_ratio: float = 0.1, schedule: str = "cosine",
          decay_start: int = 0) -> float:
    """Linear warmup, then either constant base_lr (schedule="constant", the teammate recipe) or cosine decay to
    min_ratio * base_lr by step `total`. With decay_start > 0 the cosine LR stays at base_lr until decay_start and
    decays from there to `total` (e.g. a final decaying epoch after constant-LR epochs, with no jump at its start)."""
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if schedule == "constant":
        return base_lr
    if decay_start > 0:
        if step < decay_start:
            return base_lr
        progress = (step - decay_start) / max(1, total - decay_start)
    else:
        progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


class TimeBudget:
    """Lets a training loop stop cleanly before the Slurm wall-clock limit."""

    def __init__(self, minutes=None):
        self.deadline = None if not minutes else time.time() + 60.0 * minutes

    def exhausted(self) -> bool:
        return self.deadline is not None and time.time() >= self.deadline


def save_ckpt(path: str, **state):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic, so a killed job never leaves a half-written file


def log_jsonl(path: str, record: dict):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def count_params(module: torch.nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def infinite(loader):
    while True:
        for batch in loader:
            yield batch
