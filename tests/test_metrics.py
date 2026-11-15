"""Metric tests on small hand-checked examples."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.metrics import accuracy, confusion_matrix, per_class_accuracy, predict_all

# 4 samples, 3 classes; the third sample (true 2) is predicted as 1
Y_TRUE = np.array([0, 1, 2, 2])
Y_PRED = np.array([0, 1, 1, 2])


def test_accuracy() -> None:
    assert accuracy(Y_TRUE, Y_PRED) == 0.75
    assert accuracy(Y_TRUE, Y_TRUE) == 1.0


def test_confusion_matrix_is_indexed_true_pred() -> None:
    cm = confusion_matrix(Y_TRUE, Y_PRED, n_classes=3)
    expected = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 1],  # true 2: once predicted as 1, once as 2
    ])
    np.testing.assert_array_equal(cm, expected)
    assert cm.sum() == len(Y_TRUE)


def test_per_class_accuracy() -> None:
    cm = confusion_matrix(Y_TRUE, Y_PRED, n_classes=3)
    np.testing.assert_allclose(per_class_accuracy(cm), [1.0, 1.0, 0.5])


def test_per_class_accuracy_empty_class_is_zero() -> None:
    cm = confusion_matrix(np.array([0, 0]), np.array([0, 1]), n_classes=3)  # class 2 never appears
    np.testing.assert_allclose(per_class_accuracy(cm), [0.5, 0.0, 0.0])


class _FixedLogits(nn.Module):
    """Predicts class ``x[:, 0]`` by putting a high logit on it."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.one_hot(x[:, 0].long(), num_classes=10).float()


def test_predict_all_collects_every_batch() -> None:
    labels = torch.tensor([3, 1, 4, 1, 5, 9, 2])
    loader = DataLoader(TensorDataset(labels.float().unsqueeze(1), labels), batch_size=3)
    y_true, y_pred = predict_all(_FixedLogits(), loader, torch.device("cpu"))

    assert y_true.shape == y_pred.shape == (7,)
    assert y_true.dtype == y_pred.dtype == np.int64
    np.testing.assert_array_equal(y_true, labels.numpy())
    np.testing.assert_array_equal(y_pred, labels.numpy())
