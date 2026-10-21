"""NumPy neural-network primitives used by Neural Network X-Ray.

The model deliberately implements its own forward pass, backpropagation, and
parameter updates.  Third-party libraries are used only by the notebook for
data preparation and reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Iterable

import numpy as np


EPSILON = 1e-12


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def activation(name: str) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    """Return an activation and its derivative from pre/post-activation values."""
    if name == "relu":
        return lambda z: np.maximum(0.0, z), lambda z, a: (z > 0.0).astype(float)
    if name == "tanh":
        return np.tanh, lambda z, a: 1.0 - a**2
    if name == "sigmoid":
        return sigmoid, lambda z, a: a * (1.0 - a)
    raise ValueError("activation must be one of: relu, tanh, sigmoid")


def binary_cross_entropy(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Mean binary cross entropy; inputs are column vectors."""
    p = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


@dataclass
class TrainingHistory:
    epochs: list[int]
    train_loss: list[float]
    validation_loss: list[float]
    train_accuracy: list[float]
    validation_accuracy: list[float]
    gradient_norm: list[float]
    seconds: float = 0.0


class BinaryMLP:
    """A fully connected binary classifier trained with mini-batch SGD.

    ``hidden_layers=(8, 8)`` creates 2 -> 8 -> 8 -> 1 for 2D data.  The final
    layer is always a sigmoid unit and BCE's sigmoid/BCE simplification is used
    for its gradient: dZ = (prediction - target) / batch_size.
    """

    def __init__(
        self,
        hidden_layers: Iterable[int] = (8, 8),
        hidden_activation: str = "tanh",
        learning_rate: float = 0.1,
        l2: float = 0.0,
        seed: int = 42,
    ) -> None:
        widths = tuple(int(w) for w in hidden_layers)
        if not widths or any(w < 1 for w in widths):
            raise ValueError("hidden_layers must contain one or more positive widths")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        activation(hidden_activation)  # validate early
        self.hidden_layers = widths
        self.hidden_activation = hidden_activation
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.seed = int(seed)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self.initial_weights: list[np.ndarray] = []
        self.initial_biases: list[np.ndarray] = []

    @property
    def parameter_count(self) -> int:
        return int(sum(w.size + b.size for w, b in zip(self.weights, self.biases)))

    def initialize(self, input_dim: int) -> None:
        rng = np.random.default_rng(self.seed)
        layer_sizes = (int(input_dim), *self.hidden_layers, 1)
        self.weights, self.biases = [], []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # He for ReLU, Xavier for tanh/sigmoid and the sigmoid output.
            scale = np.sqrt(2.0 / fan_in) if self.hidden_activation == "relu" and fan_out != 1 else np.sqrt(1.0 / fan_in)
            self.weights.append(rng.normal(0.0, scale, size=(fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out), dtype=float))
        self.initial_weights = [w.copy() for w in self.weights]
        self.initial_biases = [b.copy() for b in self.biases]

    def _ensure_initialized(self, input_dim: int) -> None:
        if not self.weights:
            self.initialize(input_dim)
        if self.weights[0].shape[0] != input_dim:
            raise ValueError("input dimension differs from the initialized model")

    def forward(self, x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        self._ensure_initialized(x.shape[1])
        act, _ = activation(self.hidden_activation)
        zs: list[np.ndarray] = []
        activations = [x]
        current = x
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = current @ weight + bias
            current = sigmoid(z) if index == len(self.weights) - 1 else act(z)
            zs.append(z)
            activations.append(current)
        return zs, activations

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.forward(np.asarray(x, dtype=float))[1][-1]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= 0.5).astype(int).ravel()

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        data_loss = binary_cross_entropy(y, self.predict_proba(x))
        penalty = 0.5 * self.l2 * sum(float(np.sum(w * w)) for w in self.weights)
        return data_loss + penalty

    def gradients(self, x: np.ndarray, y: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        zs, acts = self.forward(np.asarray(x, dtype=float))
        _, derivative = activation(self.hidden_activation)
        batch_size = x.shape[0]
        grad_w = [np.zeros_like(w) for w in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]
        dz = (acts[-1] - y) / batch_size
        for layer in range(len(self.weights) - 1, -1, -1):
            grad_w[layer] = acts[layer].T @ dz + self.l2 * self.weights[layer]
            grad_b[layer] = np.sum(dz, axis=0, keepdims=True)
            if layer:
                dz = (dz @ self.weights[layer].T) * derivative(zs[layer - 1], acts[layer])
        return grad_w, grad_b

    def _metrics(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        probabilities = self.predict_proba(x)
        y_col = np.asarray(y).reshape(-1, 1)
        return self.loss(x, y_col), float(np.mean((probabilities >= 0.5) == y_col))

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_validation: np.ndarray,
        y_validation: np.ndarray,
        epochs: int = 1000,
        batch_size: int | None = None,
        record_every: int = 10,
    ) -> TrainingHistory:
        if epochs < 1 or record_every < 1:
            raise ValueError("epochs and record_every must be positive")
        x_train, y_train = np.asarray(x_train, dtype=float), np.asarray(y_train).reshape(-1, 1)
        x_validation, y_validation = np.asarray(x_validation, dtype=float), np.asarray(y_validation).reshape(-1, 1)
        if len(x_train) != len(y_train) or len(x_validation) != len(y_validation):
            raise ValueError("features and labels must have matching sample counts")
        self.initialize(x_train.shape[1])
        rng = np.random.default_rng(self.seed + 1)
        batch_size = len(x_train) if batch_size in (None, 0) else min(int(batch_size), len(x_train))
        history = TrainingHistory([], [], [], [], [], [])
        start = perf_counter()
        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(x_train))
            gradient_sq = 0.0
            for first in range(0, len(x_train), batch_size):
                indices = order[first:first + batch_size]
                grad_w, grad_b = self.gradients(x_train[indices], y_train[indices])
                gradient_sq += sum(float(np.sum(g * g)) for g in (*grad_w, *grad_b))
                for layer in range(len(self.weights)):
                    self.weights[layer] -= self.learning_rate * grad_w[layer]
                    self.biases[layer] -= self.learning_rate * grad_b[layer]
            if epoch == 1 or epoch % record_every == 0 or epoch == epochs:
                train_loss, train_acc = self._metrics(x_train, y_train)
                val_loss, val_acc = self._metrics(x_validation, y_validation)
                history.epochs.append(epoch)
                history.train_loss.append(train_loss)
                history.validation_loss.append(val_loss)
                history.train_accuracy.append(train_acc)
                history.validation_accuracy.append(val_acc)
                history.gradient_norm.append(float(np.sqrt(gradient_sq)))
        history.seconds = perf_counter() - start
        return history


def gradient_check(seed: int = 7, epsilon: float = 1e-5, tolerance: float = 2e-5) -> dict[str, float | bool]:
    """Compare all analytical gradients with centered finite differences.

    It deliberately uses a tiny network and fixed data so it is suitable for a
    notebook verification cell and automated tests, not ordinary training.
    """
    x = np.array([[-0.7, 0.3], [0.2, -0.5], [0.8, 0.4]], dtype=float)
    y = np.array([0, 1, 1], dtype=float)
    model = BinaryMLP((2,), "tanh", learning_rate=0.1, seed=seed)
    model.initialize(2)
    grad_w, grad_b = model.gradients(x, y)
    analytic = np.concatenate([g.ravel() for g in (*grad_w, *grad_b)])
    numeric: list[float] = []
    params = [*model.weights, *model.biases]
    for parameter in params:
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + epsilon
            plus = model.loss(x, y)
            parameter[index] = original - epsilon
            minus = model.loss(x, y)
            parameter[index] = original
            numeric.append((plus - minus) / (2.0 * epsilon))
    numeric_arr = np.asarray(numeric)
    denominator = np.maximum(1e-10, np.abs(analytic) + np.abs(numeric_arr))
    relative_error = float(np.max(np.abs(analytic - numeric_arr) / denominator))
    return {"relative_error": relative_error, "tolerance": tolerance, "passed": relative_error < tolerance}
