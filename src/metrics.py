"""Evaluation metrics computed from model predictions."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def predict_all(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model in eval mode over ``loader``.

    Returns:
        ``(y_true, y_pred)``, both of shape (N,) and dtype int64.
    """
    raise NotImplementedError


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correct predictions, in [0, 1]."""
    raise NotImplementedError


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 10
) -> np.ndarray:
    """Count matrix of shape (n_classes, n_classes), indexed ``cm[true, pred]``."""
    raise NotImplementedError


def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    """Accuracy of each true class from a confusion matrix, shape (n_classes,)."""
    raise NotImplementedError
