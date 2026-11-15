"""Evaluation metrics computed from model predictions."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def predict_all(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model in eval mode over ``loader``.

    Returns:
        ``(y_true, y_pred)``, both of shape (N,) and dtype int64.
    """
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        logits = model(images.to(device, non_blocking=True))  # (B, num_classes)
        y_pred.append(logits.argmax(dim=1).cpu())             # (B,) predicted class
        y_true.append(labels)
    return (
        torch.cat(y_true).numpy().astype(np.int64),
        torch.cat(y_pred).numpy().astype(np.int64),
    )


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correct predictions, in [0, 1]."""
    return float((y_true == y_pred).mean())


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 10
) -> np.ndarray:
    """Count matrix of shape (n_classes, n_classes), indexed ``cm[true, pred]``."""
    # Each (true, pred) pair maps to one flat index true * n + pred; bincount counts them all at once
    flat = y_true * n_classes + y_pred
    return np.bincount(flat, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    """Accuracy of each true class from a confusion matrix, shape (n_classes,)."""
    # Diagonal = correct per true class, row sum = all images of that class
    totals = cm.sum(axis=1)
    return np.divide(cm.diagonal(), totals, out=np.zeros(len(cm)), where=totals > 0)
