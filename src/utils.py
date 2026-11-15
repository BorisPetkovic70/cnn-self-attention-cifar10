"""Seeding, device selection, checkpointing and small training helpers."""

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int) -> None:
    """Seed the Python, NumPy and PyTorch RNGs (CPU and all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    """Return the CUDA device if available, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Return the number of (trainable) parameters in ``model``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    **extra,
) -> None:
    """Save model (and optional optimizer) state plus any extra values, e.g. ``epoch``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": model.state_dict(), **extra}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Load state saved by ``save_checkpoint`` into ``model`` (and ``optimizer``).

    Returns the full checkpoint dict so extra values such as ``epoch`` can be read.
    """
    state = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state


class AverageMeter:
    """Running average weighted by the number of samples in each update."""

    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)
