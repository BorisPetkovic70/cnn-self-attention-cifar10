"""Plots for training curves, confusion matrices, attention maps and run comparisons.

Each function draws one matplotlib figure and saves it to ``save_path`` when given.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from torch import Tensor, nn

# Fixed categorical order: run i always gets color i, so a run keeps its color across figures
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

# Sequential single-hue ramp (light = few, dark = many) for heatmaps
CM_CMAP = LinearSegmentedColormap.from_list(
    "blues", ["#f7fbff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


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
    n = len(class_names)
    if cm.shape != (n, n):
        raise ValueError(f"cm has shape {cm.shape}, expected ({n}, {n}) for {n} class names")

    if normalize:
        totals = cm.sum(axis=1, keepdims=True)
        # Row i = how images of true class i were predicted, as fractions summing to 1
        values = np.divide(cm, totals, out=np.zeros(cm.shape), where=totals > 0)
    else:
        values = cm

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(values, cmap=CM_CMAP, vmin=0, vmax=1 if normalize else None)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04,
                 format=PercentFormatter(xmax=1.0) if normalize else None)

    threshold = values.max() / 2  # dark cells get white text, light cells dark text
    for i in range(n):
        for j in range(n):
            text = f"{values[i, j]:.0%}" if normalize else f"{values[i, j]:d}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if values[i, j] > threshold else "#1a1a19")

    ax.set_xticks(range(n), class_names, rotation=45, ha="right")
    ax.set_yticks(range(n), class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix" + (" (row-normalized)" if normalize else ""))
    _finish(fig, save_path)


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
