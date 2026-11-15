"""Plots for training curves, confusion matrices, attention maps and run comparisons.

Each function draws one matplotlib figure and saves it to ``save_path`` when given.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from torch import Tensor, nn

# Fixed categorical order: run i always gets color i, so a run keeps its color across figures
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def _finish(fig: Figure, save_path: str | None) -> None:
    """Tighten the layout and save the figure when ``save_path`` is given."""
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_history(
    histories: dict[str, dict], metric: str = "acc", save_path: str | None = None
) -> None:
    """Overlay train and val curves of several runs.

    Args:
        histories: Run label -> ``Trainer.fit()`` history.
        metric: ``"acc"`` or ``"loss"``.
        save_path: File to save the figure to.
    """
    if metric not in ("acc", "loss"):
        raise ValueError(f"metric must be 'acc' or 'loss', got {metric!r}")
    if len(histories) > len(SERIES_COLORS):
        raise ValueError(f"At most {len(SERIES_COLORS)} runs per plot, got {len(histories)}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for color, (label, history) in zip(SERIES_COLORS, histories.items()):
        epochs = range(1, len(history[f"val_{metric}"]) + 1)
        # Solid = validation, dashed = train, same color per run: the gap between them shows overfitting
        ax.plot(epochs, history[f"val_{metric}"], color=color, linewidth=2, label=f"{label} (val)")
        ax.plot(epochs, history[f"train_{metric}"], color=color, linewidth=1.5, linestyle="--",
                label=f"{label} (train)")

    name = "accuracy" if metric == "acc" else "loss"
    ax.set_xlabel("Epoch")
    ax.set_ylabel(name.capitalize())
    if metric == "acc":
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))  # history stores fractions in [0, 1]
    ax.set_title(f"Training curves: {name}")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    _finish(fig, save_path)


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
