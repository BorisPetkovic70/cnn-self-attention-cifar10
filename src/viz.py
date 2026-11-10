"""Plots for training curves, confusion matrices, attention maps and run comparisons.

Each function draws one matplotlib figure and saves it to ``save_path`` when given.
"""

from __future__ import annotations

import numpy as np
from torch import Tensor, nn


def plot_history(
    histories: dict[str, dict], metric: str = "acc", save_path: str | None = None
) -> None:
    """Overlay train and val curves of several runs.

    Args:
        histories: Run label -> ``Trainer.fit()`` history.
        metric: ``"acc"`` or ``"loss"``.
        save_path: File to save the figure to.
    """
    raise NotImplementedError


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    normalize: bool = True,
    save_path: str | None = None,
) -> None:
    """Draw an annotated confusion-matrix heatmap, rows normalized when ``normalize``."""
    raise NotImplementedError


def plot_attention_map(
    model: nn.Module,
    image: Tensor,
    stage: str = "after_stage1",
    save_path: str | None = None,
) -> None:
    """Show where positions of a normalized image (3, 32, 32) attend at ``stage``."""
    raise NotImplementedError


def plot_comparison_bars(results_csv: str, save_path: str | None = None) -> None:
    """Compare runs from ``results.csv`` as a bar chart of test accuracy."""
    raise NotImplementedError
