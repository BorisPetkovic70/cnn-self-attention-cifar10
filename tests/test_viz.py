"""Smoke tests: each plot runs on small fake inputs and writes its file."""

import matplotlib

matplotlib.use("Agg")  # no window; must run before pyplot is imported

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.viz import plot_confusion_matrix, plot_history


def _fake_history(n_epochs: int = 3) -> dict[str, list[float]]:
    return {
        "train_loss": [2.0, 1.5, 1.0][:n_epochs],
        "train_acc": [0.3, 0.5, 0.7][:n_epochs],
        "val_loss": [2.1, 1.7, 1.4][:n_epochs],
        "val_acc": [0.28, 0.45, 0.6][:n_epochs],
    }


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("metric", ["acc", "loss"])
def test_plot_history_saves_file(tmp_path, metric: str) -> None:
    path = tmp_path / "figures" / f"history_{metric}.png"  # parent folder does not exist yet
    plot_history({"baseline": _fake_history(), "hybrid": _fake_history()}, metric, str(path))
    assert path.stat().st_size > 0


def test_plot_history_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError):
        plot_history({"baseline": _fake_history()}, metric="f1")


@pytest.mark.parametrize("normalize", [True, False])
def test_plot_confusion_matrix_saves_file(tmp_path, normalize: bool) -> None:
    cm = np.array([[5, 1, 0], [0, 6, 0], [2, 0, 0]])  # last class never predicted correctly
    path = tmp_path / f"cm_{normalize}.png"
    plot_confusion_matrix(cm, ["cat", "dog", "frog"], normalize, str(path))
    assert path.stat().st_size > 0


def test_plot_confusion_matrix_rejects_wrong_class_count() -> None:
    with pytest.raises(ValueError):
        plot_confusion_matrix(np.zeros((3, 3), dtype=int), ["cat", "dog"])
