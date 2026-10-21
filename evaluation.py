"""Honest final-only test-set reporting helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .core import BinaryMLP, binary_cross_entropy


def evaluate(model: BinaryMLP, x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    probabilities = model.predict_proba(x).ravel()
    predicted = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, predicted, average="binary", zero_division=0)
    return {
        "loss": binary_cross_entropy(np.asarray(y).reshape(-1, 1), probabilities.reshape(-1, 1)),
        "accuracy": float(np.mean(predicted == y)),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]),
        "probabilities": probabilities, "predicted": predicted,
    }
